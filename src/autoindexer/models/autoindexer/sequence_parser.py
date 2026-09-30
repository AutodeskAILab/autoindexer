from typing import Optional, List, Dict

import numpy as np
import torch
from torch import Tensor

from autoindexer.models.autoindexer.type_utils import MarkerMap, IterPositions


class AutoIndexerIdParser:
    def __init__(
        self,
        marker_map: MarkerMap,
        batch_size: int,
        device: torch.device = None,
        parse_output: bool = True,
        editable_start: Optional[Tensor] = None,
        eos_in_vocab: bool = False,
    ):
        self.marker_map = marker_map
        self.device = device
        self.batch_size = batch_size
        # When true, eos is a real vocabulary token (see `AutoIndexerModelBase.eos_in_vocab`)
        # and must survive into `parsed_sequences` instead of being trimmed off.
        self.eos_in_vocab = eos_in_vocab
        # Per-row token count locked out of the index head's cursor (see `get_parsed_positions`'s `index_attn_mask`)
        self.editable_start: Tensor = (
            editable_start if editable_start is not None else torch.zeros((batch_size,), dtype=torch.long, device=device)
        )

        self.position_ids: Tensor = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        self.abs_pos_mask: Tensor = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        self.attn_mask: Tensor = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)

        self.curr_pos: Tensor = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.curr_end_pos: Tensor = torch.full((batch_size,), -1, dtype=torch.long, device=device)  # Current last position
        self.num_delete_tokens: Tensor = torch.zeros((batch_size,), dtype=torch.long, device=device)

        self._batch_idx = torch.arange(batch_size, dtype=torch.long, device=device)
        self.active_edit_marker_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        self.parse_output = parse_output
        self.sequences = None
        self.parse_indices = [None] * batch_size
        self.parsed_sequences = [None] * batch_size
        self.reached_eos: Tensor = torch.zeros(batch_size, dtype=torch.bool, device=device)

    @classmethod
    def parse_sequence(cls, marker_map: MarkerMap, sequence: Tensor | np.ndarray, cursor_indices: Tensor | np.ndarray, device: torch.device = None) -> List[Tensor | np.ndarray]:
        parser = cls(marker_map, batch_size=len(sequence), device=device)
        return parser(sequence, cursor_indices)["parsed_sequences"]

    def get_parsed_positions(self) -> IterPositions:
        step_so_far = self.position_ids.shape[-1] - 1
        is_edit_token = self.active_edit_marker_idx == step_so_far
        key_pos_ids = self.position_ids.clone()
        key_pos_ids = (key_pos_ids - self.curr_pos.unsqueeze(-1)) * self.attn_mask
        index_attn_mask = self.attn_mask & (self.position_ids >= self.editable_start.unsqueeze(-1))
        return IterPositions(
            key_pos_ids=key_pos_ids * self.attn_mask,
            attn_mask=self.attn_mask.clone(),
            is_edit_token=is_edit_token,
            index_attn_mask=index_attn_mask,
        )

    def __call__(self, sequences: Tensor | np.ndarray, cursor_indices: Tensor, compute_matrix: bool = False, finish_parsing: bool = True) -> Dict[str, Optional[Tensor | np.ndarray]]:
        """
        :param sequences: (B, L)
        :param cursor_indices: (B, L, 2) where each entry is either [-100, -100] for non-edit markers, or [edit_start_idx, edit_end_idx] for edit markers
        :return:
        """
        if len(sequences.shape) == 1 and cursor_indices.ndim == 2:
            sequences = sequences[None, :]
            cursor_indices = cursor_indices[None, ...]
        assert len(sequences.shape) == 2, f"Expected sequence to have shape (B, L), but got {sequences.shape}"
        assert len(cursor_indices.shape) == 3 and cursor_indices.shape[-1] == 2, f"Expected cursor indices to have shape (B, L, 2), but got {cursor_indices.shape}"
        batch_size, seq_len = sequences.shape
        if compute_matrix:
            position_matrix = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=self.device)
            attention_mask = torch.eye(seq_len, dtype=torch.bool, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            position_matrix, attention_mask = None, None

        for i in range(sequences.shape[-1]):
            self.update(sequences[:, i], cursor_indices[:, i, :])
            if compute_matrix:
                parsed_positions = self.get_parsed_positions()
                k_len = parsed_positions.key_pos_ids.shape[1]
                position_matrix[:, i, :k_len] = parsed_positions.key_pos_ids
                attention_mask[:, i, :k_len] = parsed_positions.attn_mask

        if finish_parsing:
            self.resolve_parsing(sequences.dtype)

        return {"position_matrix": position_matrix, "attention_mask": attention_mask, "parsed_sequences": self.parsed_sequences if self.parse_output else None}

    def close_prev_edits(self, close_previous_edit):
        if close_previous_edit.any():
            cursor_pos = self.position_ids[self._batch_idx, self.active_edit_marker_idx].unsqueeze(-1)
            abs_pos_mask_upon_edit_close = self.abs_pos_mask & close_previous_edit.unsqueeze(-1)
            tokens_to_delete = abs_pos_mask_upon_edit_close & (self.position_ids <= cursor_pos)
            self.position_ids -= abs_pos_mask_upon_edit_close * (self.num_delete_tokens.unsqueeze(-1) + 1)
            self.position_ids = torch.where(tokens_to_delete, -1, self.position_ids)
            self.attn_mask &= ~tokens_to_delete
            self.curr_end_pos -= self.num_delete_tokens * close_previous_edit
            self.num_delete_tokens[close_previous_edit] = 0
            self.curr_pos = torch.where(close_previous_edit, self.curr_end_pos, self.curr_pos)

            self.abs_pos_mask[close_previous_edit, :] = 0
            # Close previous edit
            self.active_edit_marker_idx[close_previous_edit] = -1

    def update(self, tokens: Tensor | np.ndarray, indices: Tensor):
        """
        :param tokens: (B,)
        :param indices: (B, 2)
        :param iterative: bool
        :return:
        """
        is_edit_token = tokens == self.marker_map["edit"]
        is_eos_token = tokens == self.marker_map["eos"]
        is_ret_token = tokens == self.marker_map["return_to_end"]
        if isinstance(tokens, np.ndarray):
            is_edit_token = torch.from_numpy(is_edit_token)
            is_eos_token = torch.from_numpy(is_eos_token)
            is_ret_token = torch.from_numpy(is_ret_token)
        is_normal_token = ~(is_edit_token | is_eos_token | is_ret_token)
        is_mid_edit = self.active_edit_marker_idx != -1
        curr_step = self.position_ids.shape[-1]

        seq_init = (self.curr_end_pos == -1)
        if seq_init.any() and (curr_step != 0):
            # RET keys are masked on append; clearing the last visible key here would leave none.
            clear_prev = seq_init & ~is_ret_token
            if clear_prev.any():
                self.position_ids[clear_prev, -1] = -1
                self.attn_mask[clear_prev, -1] = 0

        prev_was_edit_token = (self.active_edit_marker_idx == curr_step - 1) & (curr_step != 0)
        if prev_was_edit_token.any():
            edit_end_idx = indices[:, 1]
            edit_end_pos = self.position_ids[self._batch_idx, edit_end_idx]
            edit_end_pos = torch.where(edit_end_idx == curr_step - 1, self.curr_end_pos + 1, edit_end_pos)
            # Correcting for the subtraction of 1
            edit_start_pos = self.curr_pos + 1
            num_delete_tokens = edit_end_pos - edit_start_pos
            num_delete_tokens = num_delete_tokens[prev_was_edit_token]
            assert (num_delete_tokens >= 0).all(), f"Expected edit_end_pos {edit_end_pos} >= edit_start_pos {edit_start_pos}"
            self.num_delete_tokens[prev_was_edit_token] = num_delete_tokens
            self.position_ids += (self.position_ids >= edit_end_pos.unsqueeze(-1)) & prev_was_edit_token.unsqueeze(-1)
            self.position_ids[prev_was_edit_token, -1] = edit_end_pos[prev_was_edit_token]

        close_previous_edit = (~is_normal_token) & is_mid_edit
        self.close_prev_edits(close_previous_edit)

        if self.parse_output:
            for i in is_edit_token.nonzero().squeeze(dim=-1).tolist():
                if not self.reached_eos[i]:
                    self.parse_indices[i] = (self.position_ids[i].clone(), self.curr_end_pos[i].item())

            if isinstance(tokens, np.ndarray):
                self.sequences = np.concatenate([self.sequences, tokens[:, None]], axis=-1) if self.sequences is not None else tokens[:, None]
            else:
                self.sequences = torch.cat([self.sequences, tokens[:, None]], dim=-1) if self.sequences is not None else tokens[:, None]

        self.curr_pos[is_normal_token & (~seq_init)] += 1
        # eos gets the same "append after the last real token" position as an edit token --
        # a placeholder discarded by `resolve_parsing`'s trim unless `eos_in_vocab`, in which
        # case it must be distinct from every already-assigned real-token position.
        new_pos = self.curr_pos * is_normal_token + (self.curr_end_pos + 1) * (is_edit_token | is_eos_token) + (-1) * is_ret_token

        self.position_ids = torch.cat([self.position_ids + self.abs_pos_mask, new_pos.unsqueeze(-1)], dim=-1)
        self.abs_pos_mask = torch.cat([self.abs_pos_mask, is_edit_token.unsqueeze(-1)], dim=-1)
        self.attn_mask = torch.cat([self.attn_mask, ~is_ret_token.unsqueeze(-1)], dim=-1)
        self.curr_end_pos += is_normal_token

        if is_edit_token.any():
            edit_start_idx = indices[:, 0]
            edit_start_pos = self.position_ids[self._batch_idx, edit_start_idx]
            after_start_pos = self.position_ids >= edit_start_pos.unsqueeze(-1)
            self.curr_pos[is_edit_token] = edit_start_pos[is_edit_token] - 1
            self.abs_pos_mask[is_edit_token] = after_start_pos[is_edit_token]
            self.active_edit_marker_idx[is_edit_token] = curr_step

        if is_eos_token.any():
            if self.parse_output:
                eos_batch_indices = is_eos_token.nonzero(as_tuple=False).squeeze(-1).tolist()
                self.resolve_parsing(dtype=tokens.dtype, batch_indices=eos_batch_indices, last_is_eos=True)
                self.reached_eos[is_eos_token] = True

            self.abs_pos_mask[is_eos_token, :] = 0
            self.attn_mask[is_eos_token, :curr_step] = 0
            self.position_ids[is_eos_token, :curr_step] = -1
            self.position_ids[is_eos_token, curr_step] = 0
            self.curr_end_pos[is_eos_token] = -1
            self.curr_pos[is_eos_token] = 0
            self.active_edit_marker_idx[is_eos_token] = -1

        # Dealing with edge cases where the edits removed all tokens from the context
        no_visible = self.attn_mask.sum(dim=-1) == 0
        if no_visible.any():
            self.attn_mask[no_visible, -1] = True

    def resolve_parsing(self, dtype, batch_indices: Optional[List[int]] = None, last_is_eos: bool = False):
        if not self.parse_output:
            return

        if batch_indices is None:
            batch_indices = range(len(self.parse_indices))
        for i in batch_indices:
            if not self.reached_eos[i]:
                if self.active_edit_marker_idx[i] == -1:
                    trim_eos = last_is_eos and not self.eos_in_vocab
                    position_ids = self.position_ids[i, :-1 if trim_eos else None]
                    self.parse_indices[i] = (position_ids.clone(), self.curr_end_pos[i].item())

                position_indices, max_idx = self.parse_indices[i]
                live = position_indices[position_indices >= 0]
                end_idx = max(int(max_idx), int(live.max()) if len(live) else -1)
                outputs = torch.zeros((end_idx + 2,), dtype=dtype, device=self.device) \
                    if isinstance(dtype, torch.dtype) else np.zeros((end_idx + 2,), dtype=dtype)
                outputs[position_indices] = self.sequences[i, :len(position_indices)]
                self.parsed_sequences[i] = outputs[:-1]
