import random
from dataclasses import dataclass

import numpy as np
from typing import TypedDict, Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from autoindexer.models.autoindexer.type_utils import PositionBlock, MarkerMap, RotaryEmbeddingFunc, IterPositions, \
    BlockType, CursorIndices


class PerturbKwargs(TypedDict):
    min_len: Optional[int]
    mean_num_edits: Optional[int]
    mean_insert: Optional[int]
    mean_delete: Optional[int]
    mean_extend: Optional[int]
    inplace_edits: Optional[bool]
    sort_edits: Optional[bool]
    edit_type_ratio: Optional[list]
    edit_lag_prob: Optional[float]
    mean_edit_lag: Optional[int]


@dataclass
class Operation:
    num_insert: int = 0
    num_delete: int = 0
    start_rel_pos: int = 0
    end_rel_pos: int = 0
    num_extend: int = 0


# (num_insert, num_delete, start_rel_pos, end_rel_pos, num_extend)
OperationTuple = Tuple[int, int, int, int, int]


def to_operations(op_tuples: List[OperationTuple]):
    return [
        Operation(num_insert=num_insert, num_delete=num_delete, start_rel_pos=start_rel_pos, end_rel_pos=end_rel_pos, num_extend=num_extend)
        for num_insert, num_delete, start_rel_pos, end_rel_pos, num_extend in op_tuples
    ]


@dataclass
class Perturbation:
    token_indices: List[int]
    num_tokens_pre_edit: int
    num_tokens_post_edit: int
    edit_token_idx: Optional[int]
    start_cursor_idx: Optional[int]
    end_cursor_idx: Optional[int]
    return_token_idx: Optional[int]
    curr_seq_idx: int
    operation: Operation

    @property
    def num_tokens_pre_cursor(self):
        return self.operation.start_rel_pos + self.operation.num_insert


def delta_operation(op: Operation) -> int:
    """Returns the increase in token length after the edit operation"""
    return op.num_insert - op.num_delete + op.num_extend


def is_valid_operation(op: Operation, curr_len: int) -> bool:
    return op.start_rel_pos <= op.end_rel_pos and op.start_rel_pos < curr_len and op.end_rel_pos <= curr_len


def operations_are_valid_order(operations: List[Operation]) -> bool:
    """Check whether ``operations`` can be replayed in this order without violating
    the invariant (enforced by ``perturb_labels_per_sequence``) that every edit's
    ``end_rel_pos`` must be within the bounds of the sequence built up so far, i.e.
    ``end_rel_pos <= len(token_indices)`` at the time the edit is applied.
    """
    if not operations:
        return True
    first, *rest = operations
    if first.num_insert != 0 or first.num_delete != 0:
        return False
    curr_len = first.num_extend
    for op in rest:
        # start_rel_pos indexes directly into token_indices (no bounds special-case,
        # unlike end_rel_pos which may equal len(token_indices) for an append), so it
        # must be strictly within bounds.
        if not is_valid_operation(op, curr_len):
            return False
        curr_len += delta_operation(op)
    return True


def sort_operation_order(op: Operation) -> tuple:
    """
    Sort order
    1. Edits with a non-negative net delta (additive or neutral) are scheduled before subtractive edits.
    2. Out of the edits that are additive or neutral, start with the smallest end_rel_pos first because the context grows
    3. Out of the edits that are subtractive, start with the largest end_rel_pos first because the context shrinks
    """
    delta = delta_operation(op)
    sign = 1 if delta >= 0 else -1
    return - sign, sign * op.end_rel_pos, sign * op.start_rel_pos


def sort_operations(operations: List[Operation]) -> Tuple[List[Operation], bool]:
    """Reorder edit operations in ascending order of position (``end_rel_pos``, then
    ``start_rel_pos``) whenever doing so keeps every operation valid, i.e. the running
    sequence has already grown long enough for the edit to be applied
    (``end_rel_pos <= len(token_indices)``) by the time it is its turn.

    ``operations`` is expected to already be in a valid chronological order (as
    returned by the default, unsorted generation), with the very first operation
    being a pure extend that seeds the sequence. That first operation is always kept
    in place, since insert/delete operations cannot happen at the start of the
    sequence.

    Whenever a fully ascending order isn't achievable without breaking validity, this
    falls back to the original (guaranteed-valid) order for the affected operations.

    Returns the operations and a boolean for whether they were sorted or not (True if sorted).
    """
    sorted_ops = sorted(operations, key=sort_operation_order)

    if operations_are_valid_order(sorted_ops):
        return sorted_ops, True
    else:
        # Contributions of some edits (e.g. net deletions) can be negative, so the greedy
        # walk above isn't guaranteed to produce a valid order in all cases. Fall back to
        # the original chronological order (known valid) rather than risk an invalid one.
        return operations, False


def sample_cursor_operations(
    seq_len: int,
    min_len: int = 5,
    mean_num_edits: int = 10,
    mean_insert: int = 5,
    mean_delete: int = 3,
    mean_extend: int = 7,
    inplace_edits: bool = False,
    sort_edits: bool = False,
    edit_type_ratio: list = None,
    edit_lag_prob: float = 0.0,
    mean_edit_lag: int = 3,
    editable_start: int = 0,
):
    """
    :param seq_len: Length of the original sequence
    :param min_len: Minimum length of the sequence after perturbations
    :param mean_num_edits: Number of edit operations
    :param mean_insert: Mean number of insertions per operation
    :param mean_delete: Mean number of deletions per operation
    :param mean_extend: Mean number of appends per operation
    :param inplace_edits: Only edit operations with substitutions (num_insert = num_delete)
    :param sort_edits: If True, reorder edits in ascending order of position (from the
        beginning of the sequence to the end) whenever that ordering remains valid.
    :param edit_type_ratio: Normalized ``(pure_insert, substitution, pure_deletion)``
        weights. Pure insertions/deletions force one side of the Poisson draw to zero;
        without that mass, ``e^-mean`` is the only source of pure edits and they are sampled
        roughly never at typical means. Ignored when ``inplace_edits``.
    :param edit_lag_prob: Fraction of edits placed a short distance behind the write head --
        ``1 + Poisson(mean_edit_lag)`` tokens -- instead of at a uniformly random earlier
        position. Uniform placement (the default) means the *lag* between writing a corrupted
        span and emitting the edit marker is spread over the whole sequence, so however well the
        model detects the corruption its per-step ``P(edit marker)`` cannot exceed ~1/E[lag].
        Mixing in short-lag edits teaches *when* to edit; keeping the uniform branch preserves
        long-range addressing.
    :param mean_edit_lag: Mean lag used by the ``edit_lag_prob`` branch.
    :param editable_start: Positions ``< editable_start`` (e.g. everything up to and
        including the current chat turn's prompt, for a multi-turn SFT example -- see
        ``AutoIndexerModelBase.forward``'s ``editable_start`` argument) are never sampled as
        an edit's ``start_rel_pos``/``end_rel_pos``, and the seed operation (below) always
        covers at least this many tokens -- so no edit ever targets, or cursors into, locked
        context. 0 (the default) reproduces the original unconstrained behavior.
    :return: A list of tuples (num_insert, num_delete, start_rel_pos, end_rel_pos, num_extend)
    """
    if inplace_edits:
        pure_insert_prob, _, pure_deletion_prob = (0.0, 1.0, 0.0)
        assert mean_insert == mean_delete, "Inplace expects the same number of inserts and deletes."
    else:
        pure_insert_prob, substitution_prob, pure_deletion_prob = edit_type_ratio or (1/3, 1/3, 1/3)
        assert pure_insert_prob > 0 and substitution_prob > 0 and pure_deletion_prob > 0, (
            "When inplace_edits is false, edit_type_ratio must have positive pure_insert, substitution, and pure_deletion mass"
        )
    operations = []
    num_edits = int(np.random.poisson(mean_num_edits))
    for _ in range(num_edits):
        sampled_extend = np.random.poisson(mean_extend)
        drop = None
        if not inplace_edits:
            r = random.random()
            if r < pure_insert_prob:
                drop = "insert"
            elif r < pure_insert_prob + pure_deletion_prob:
                drop = "delete"
        sampled_delete, sampled_insert = 0, 0
        while sampled_insert == 0 and sampled_delete == 0:
            sampled_insert = 0 if drop == "insert" else int(np.random.poisson(mean_insert))
            if inplace_edits:
                sampled_delete = sampled_insert
            else:
                sampled_delete = 0 if drop == "delete" else int(np.random.poisson(mean_delete))
        num_tokens_pre_insert = seq_len - sampled_extend - sampled_insert
        # `editable_start + 1` (rather than bare `editable_start`) since `randint`'s upper
        # bound below is `num_tokens_pre_insert - 1`, which needs to stay `>= editable_start`.
        if num_tokens_pre_insert <= max(min_len, 1, editable_start + 1):
            break
        # The replayed sequence is `num_tokens_pre_insert + sampled_delete` long when this edit
        # is applied and the edit deletes `[start, start + delete)`, so the lag between the end
        # of the corrupted span and the edit marker is `num_tokens_pre_insert - start`.
        if random.random() < edit_lag_prob:
            lag = 1 + int(np.random.poisson(mean_edit_lag))
            sampled_start_pos = max(editable_start, num_tokens_pre_insert - lag)
        else:
            sampled_start_pos = random.randint(editable_start, num_tokens_pre_insert - 1)
        sampled_end_pos = sampled_start_pos + sampled_delete
        operations.append(Operation(num_insert=sampled_insert, num_delete=sampled_delete, num_extend=sampled_extend, start_rel_pos=sampled_start_pos, end_rel_pos=sampled_end_pos))
        seq_len = num_tokens_pre_insert + sampled_delete
    operations.append(Operation(num_extend=seq_len))

    token_count = 0
    for op in operations:
        if op.num_insert > 0 or op.num_delete > 0:
            token_count += op.num_insert + 2
        token_count += op.num_extend

    operations = operations[::-1]
    if sort_edits:
        operations, _ = sort_operations(operations)

    return operations, token_count


def perturb_labels_per_sequence(operations: List[Operation]) -> List[Perturbation]:
    # token_indices I[j]: Each element I[j] indicates the index of target tokens Z[I[j]] at which target label Y[j] will be mapped to.
    token_indices = []
    perturbations = []

    curr_seq_idx = 0
    for op in operations:
        edit_token_idx, return_token_idx = None, None
        start_cursor_idx, end_cursor_idx = None, None
        num_tokens_pre_edit = len(token_indices)
        if op.num_insert > 0 or op.num_delete > 0:
            assert curr_seq_idx > 0, "Insert/Delete operations cannot be performed at the beginning of the sequence."
            assert op.start_rel_pos <= op.end_rel_pos, f"Should be start_rel_pos <= end_rel_pos but got {op.start_rel_pos} and {op.end_rel_pos}"
            # token abs idx to start pointer (abs)
            edit_token_idx, return_token_idx = curr_seq_idx, curr_seq_idx + op.num_insert + 1
            start_cursor_idx = token_indices[op.start_rel_pos]
            # If there is no deletion then end cursor should point to the edit marker
            end_cursor_idx = edit_token_idx if len(token_indices) == op.end_rel_pos else token_indices[op.end_rel_pos]

            token_indices = token_indices[: op.start_rel_pos] + list(range(curr_seq_idx + 1, curr_seq_idx + 1 + op.num_insert)) + token_indices[op.end_rel_pos :]

            curr_seq_idx += op.num_insert + 2
        num_tokens_post_edit = len(token_indices)
        token_indices += list(range(curr_seq_idx, curr_seq_idx + op.num_extend))
        curr_seq_idx += op.num_extend

        perturbation = Perturbation(
            token_indices=list(token_indices),
            num_tokens_pre_edit=num_tokens_pre_edit,
            num_tokens_post_edit=num_tokens_post_edit,
            edit_token_idx=edit_token_idx,
            start_cursor_idx=start_cursor_idx,
            end_cursor_idx=end_cursor_idx,
            return_token_idx=return_token_idx,
            curr_seq_idx=curr_seq_idx,
            operation=op,
        )
        perturbations.append(perturbation)

    return perturbations


class PositionBlockList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.query_pos_ids: Optional[Tensor] = None
        self.concat_key_pos_ids: Optional[Tensor] = None
        self.concat_abs_pos_masks: Optional[Tensor] = None
        self.concat_attn_masks: Optional[Tensor] = None
        self.boundary_indices = []
        self._position_matrix: Optional[Tensor] = None
        self._mask_matrix: Optional[Tensor] = None

        self._query_rotary_pos_emb: Optional[Tuple[Tensor, Tensor]] = None
        self._keys_rotary_pos_emb: Optional[Tuple[Tensor, Tensor]] = None

        self._triton_launch_cache = None
        self._triton_launch_cache_key = None

        self._cutedsl_meta = None
        self._cutedsl_meta_key = None

    def get_cutedsl_meta(self, batch_size: int, seq_len: int, device):
        """Return the (cached) CuteDSL R/N-path metadata for this position layout.

        Built once and memoized on the shared ``PositionBlockList``, so every
        attention layer in a forward reuses it instead of rebuilding the identical
        packing/inverse-index metadata per layer. Mirrors ``get_triton_launch_cache``."""
        cache_key = (batch_size, seq_len, str(device))
        if self._cutedsl_meta_key != cache_key:
            from autoindexer.models.autoindexer.attention.cutedsl.meta import build_cutedsl_meta
            from autoindexer.models.autoindexer._profiling import timed

            with timed("build_cutedsl_meta"):
                self._cutedsl_meta = build_cutedsl_meta(self, batch_size, seq_len, device)
            self._cutedsl_meta_key = cache_key
        return self._cutedsl_meta

    def get_triton_launch_cache(self, seq_len: int, head_dim: int):
        """Return cached Triton tile schedules and segment metadata for this position layout."""
        if not self.boundary_indices:
            return None

        device = self.device
        cache_key = (seq_len, head_dim, str(device))
        if self._triton_launch_cache_key != cache_key or self._triton_launch_cache is None:
            from autoindexer.models.autoindexer.attention.kernels.triton_rope_attn import build_multi_segment_launch_cache

            self._triton_launch_cache = build_multi_segment_launch_cache(
                self.boundary_indices,
                seq_len,
                head_dim,
                device,
            )
            self._triton_launch_cache_key = cache_key
        return self._triton_launch_cache

    def get_rotary_pos_emb(self, rotary_emb: RotaryEmbeddingFunc, x: Tensor, seq_len: int) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        if self._query_rotary_pos_emb is None:
            padded_query_pos_ids = F.pad(self.query_pos_ids, (0, seq_len - len(self.query_pos_ids)))
            self._query_rotary_pos_emb = rotary_emb(x, padded_query_pos_ids.unsqueeze(0))
        if self._keys_rotary_pos_emb is None:
            self._keys_rotary_pos_emb = rotary_emb(x, self.concat_key_pos_ids.unsqueeze(0))
        return self._query_rotary_pos_emb, self._keys_rotary_pos_emb

    def concat(self):
        self._triton_launch_cache = None
        self._triton_launch_cache_key = None
        self._cutedsl_meta = None
        self._cutedsl_meta_key = None
        self.query_pos_ids = torch.cat([block.query_pos_ids for block in self], dim=0)
        self.concat_key_pos_ids = torch.cat([block.key_pos_ids for block in self], dim=0)
        self.concat_attn_masks = torch.cat([block.attn_mask for block in self], dim=0)
        self.concat_abs_pos_masks = torch.zeros_like(self.concat_attn_masks)
        key_cum_idx = 0
        for block in self:
            end_idx = len(block.key_pos_ids)
            start_idx = end_idx - len(block.query_pos_ids)
            self.boundary_indices.append((start_idx, end_idx, key_cum_idx, key_cum_idx + end_idx))
            if block.abs_pos_mask is not None:
                self.concat_abs_pos_masks[key_cum_idx : key_cum_idx + end_idx] = block.abs_pos_mask
            key_cum_idx += end_idx

    @property
    def device(self):
        if len(self) == 0:
            return None
        return self[0].key_pos_ids.device

    def get_matrices(self, seq_len: int):
        if self._position_matrix is not None and self._mask_matrix is not None:
            assert self._position_matrix.shape == (seq_len, seq_len), f"Position matrix has shape {self._position_matrix.shape}, expected {(seq_len, seq_len)}."
            assert self._mask_matrix.shape == (seq_len, seq_len), f"Mask matrix has shape {self._mask_matrix.shape}, expected {(seq_len, seq_len)}."
        else:
            device = self.device
            position_matrix = torch.zeros((seq_len, seq_len), dtype=torch.long, device=device)
            mask_matrix = torch.eye(seq_len, dtype=torch.bool, device=device)

            for block in self:
                end_idx = len(block.key_pos_ids)
                start_idx = end_idx - len(block.query_pos_ids)
                if block.abs_pos_mask is None:
                    positions = block.key_pos_ids - block.query_pos_ids[:, None]
                else:
                    positions = block.key_pos_ids - block.query_pos_ids[:, None] * (~block.abs_pos_mask)
                position_matrix[start_idx:end_idx, :end_idx] = positions
                mask_matrix[start_idx:end_idx, :end_idx] = block.attn_mask

            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))

            self._mask_matrix = mask_matrix & causal_mask
            self._position_matrix = position_matrix * self._mask_matrix

        return self._position_matrix, self._mask_matrix


ParsedPositions = IterPositions | PositionBlockList


def get_parsed_positions_from_perturbations(perturbations: List[Perturbation], device: torch.device = None) -> PositionBlockList:
    """
    Returns the ParsedPositions corresponding to the given perturbations.
        - seg_indices:
            - A 1D tensor of shape (num_segments + 1,) where num_segments is the number of segments (= num_edits + num_extend) in the perturbed sequence.
            The first element is 0 at training time, and the last element is (prefix + total length of the perturbed sequence).
            The i-th segment corresponds to the tokens in the range [seg_indices[i], seg_indices[i+1]) in the perturbed sequence.
        - whole_query_pos_ids:
            - A 1D tensor of shape (prefix + total length of the perturbed sequence,) containing the position ids for the query tokens in the perturbed sequence.
        - whole_key_pos_ids
            - A 1D tensor of shape (sum(seg_indices),) containing the position ids for the key tokens in the perturbed sequence.
        - whole_abs_pos_masks:
            - A 1D boolean tensor of shape (sum(seg_indices),)
    """
    token_count = perturbations[-1].curr_seq_idx
    pos_ids = torch.zeros((token_count,), dtype=torch.long, device=device)
    abs_pos_mask = torch.zeros((token_count,), dtype=torch.bool, device=device)
    attn_mask = torch.ones((token_count,), dtype=torch.bool, device=device)

    parsed_positions = PositionBlockList()
    start_idx = 0

    def push_positions(end_idx, block_type: BlockType):
        nonlocal start_idx
        key_pos_ids = pos_ids[:end_idx].clone()
        if block_type == BlockType.EXTEND:
            assert attn_mask[:end_idx].sum() > 1, "Edit token should not be at the start of the sequence."
            key_pos_ids[-1] = key_pos_ids[-2] + 1
        key_pos_ids = torch.where(attn_mask[:end_idx], key_pos_ids, -1)
        parsed_positions.append(
            PositionBlock(
                query_pos_ids=pos_ids[start_idx:end_idx].clone(),
                key_pos_ids=key_pos_ids,
                abs_pos_mask=abs_pos_mask[:end_idx].clone() if block_type == BlockType.EDIT else None,
                attn_mask=attn_mask[:end_idx].clone(),
                start_idx=start_idx,
                end_idx=end_idx,
                block_type=block_type,
            )
        )
        start_idx = end_idx

    for pt in perturbations:
        op = pt.operation
        if pt.edit_token_idx is not None:
            pos_ids[pt.edit_token_idx] = op.start_rel_pos - 1
            abs_pos_mask[: pt.edit_token_idx] = pos_ids[: pt.edit_token_idx] >= op.start_rel_pos
            abs_pos_mask[pt.edit_token_idx] = True
            abs_pos_mask &= attn_mask
            push_positions(pt.edit_token_idx + 1, block_type=BlockType.EXTEND)

            pos_ids[abs_pos_mask] += 1 - op.start_rel_pos
            pos_ids[abs_pos_mask & (pos_ids > op.num_delete)] += 1
            pos_ids[pt.edit_token_idx] = op.num_delete + 1
            if op.num_insert > 0:
                pos_ids[start_idx : pt.return_token_idx] = torch.arange(op.start_rel_pos, op.start_rel_pos + op.num_insert, device=device)
                push_positions(pt.return_token_idx, block_type=BlockType.EDIT)
            pos_ids[abs_pos_mask] += op.start_rel_pos + op.num_insert - op.num_delete - 2

            pos_ids[pt.return_token_idx] = pt.num_tokens_post_edit - 1
            delete_tokens = (pos_ids <= pos_ids[pt.edit_token_idx]) & abs_pos_mask
            attn_mask = attn_mask & (~delete_tokens)
            attn_mask[pt.return_token_idx] = False
            abs_pos_mask.zero_()

        extend_start_pos = pos_ids[start_idx].item() + 1 if start_idx > 0 else 0
        extend_start_idx = start_idx + 1 if start_idx > 0 else 0
        pos_ids[extend_start_idx : extend_start_idx + op.num_extend] = torch.arange(extend_start_pos, extend_start_pos + op.num_extend, device=device)

    push_positions(token_count, block_type=BlockType.COMPLETE)

    parsed_positions.concat()
    return parsed_positions


def batch_perturb_labels(
    marker_map: MarkerMap,
    labels: torch.Tensor,
    max_length: int,
    ignore_index: int = -100,
    operations: List[Operation] = None,
    return_indices_as_dict: bool = False,
    append_eos: bool = False,
    editable_start: int = 0,
    **perturb_kwargs,
) -> Tuple[Tensor, CursorIndices | Tensor, List[Perturbation]]:
    """
    :param editable_start: Token count to lock out of editing -- e.g. everything up to and
        including the current chat turn's prompt, for multi-turn SFT (see
        ``AutoIndexerModelBase.forward``, which also feeds the same value to
        ``index_head`` to hard-mask its cursor). A single value rather than one per row
        since ``operations`` (below) is one edit script shared across the whole batch --
        ``forward`` pools per-row boundaries into one via ``.max()``, the most conservative
        shared choice. 0 (the default) reproduces the original unconstrained behavior.
    """
    batch_size, label_len = labels.shape
    device = labels.device

    perturbed_tokens = torch.full((batch_size, max_length), fill_value=ignore_index, dtype=labels.dtype, device=device)
    cursor_indices = {} if return_indices_as_dict else torch.zeros((batch_size, max_length, 2), dtype=torch.long, device=device)

    labels_slices = []
    max_seq_len = 0

    for i in range(batch_size):
        eos_positions = torch.nonzero(labels[i] == marker_map["eos"]).view(-1).tolist() if marker_map["eos"] is not None else []
        has_eos = len(eos_positions) > 0
        label_i_len = label_len if not has_eos else eos_positions[0]
        emit_eos = has_eos or append_eos
        labels_slices.append((i, label_i_len, emit_eos))
        max_seq_len = max(max_seq_len, label_i_len)

    if operations is None:
        while True:
            operations, token_count = sample_cursor_operations(max_seq_len, editable_start=editable_start, **perturb_kwargs)
            if token_count + 1 <= max_length:
                break
            print(f"Resampling operations as token count: {token_count} exceeded max_length: {max_length}")

    perturbations = perturb_labels_per_sequence(operations)

    for pt in perturbations:
        if pt.edit_token_idx is not None:
            if return_indices_as_dict:
                cursor_indices[pt.edit_token_idx] = (pt.start_cursor_idx, pt.end_cursor_idx)
            else:
                cursor_indices[:, pt.edit_token_idx, 0] = pt.start_cursor_idx
                cursor_indices[:, pt.edit_token_idx + 1, 1] = pt.end_cursor_idx

    for i, label_i_len, emit_eos in labels_slices:
        labels_i = labels[i, :label_i_len]
        applied_labels = False
        for pt in reversed(perturbations):
            op = pt.operation
            excess_tokens = len(pt.token_indices) - label_i_len
            if applied_labels:
                if pt.edit_token_idx is not None:
                    perturbed_tokens[i, pt.edit_token_idx] = marker_map["edit"]
                if pt.return_token_idx is not None:
                    perturbed_tokens[i, pt.return_token_idx] = marker_map["return_to_end"]
            else:
                if excess_tokens > op.num_insert + op.num_extend or (excess_tokens == op.num_insert + op.num_extend and op.num_delete == 0):
                    continue
                applied_labels = True
                if pt.edit_token_idx is not None:
                    perturbed_tokens[i, pt.edit_token_idx] = marker_map["edit"]
                if 0 <= excess_tokens <= op.num_extend:
                    if pt.return_token_idx is not None:
                        perturbed_tokens[i, pt.return_token_idx] = marker_map["return_to_end"]
                    perturbed_tokens[i, pt.token_indices[:label_i_len]] = labels_i
                    if emit_eos:
                        perturbed_tokens[i, pt.curr_seq_idx - excess_tokens] = marker_map["eos"]
                else:
                    trimmed_indices = pt.token_indices[: -op.num_extend] if op.num_extend > 0 else pt.token_indices
                    trimmed_indices = trimmed_indices[: op.start_rel_pos + op.num_insert + op.num_extend - excess_tokens] + trimmed_indices[op.start_rel_pos + op.num_insert :]
                    perturbed_tokens[i, trimmed_indices] = labels_i
                    perturbed_tokens[i, pt.curr_seq_idx - excess_tokens - 1] = marker_map["return_to_end"]
                    if emit_eos:
                        perturbed_tokens[i, pt.curr_seq_idx - excess_tokens] = marker_map["eos"]

    return perturbed_tokens, cursor_indices, perturbations


def transplant_corrupted_spans(
    tokens: Tensor,
    perturbations: List[Perturbation],
    donor: Tensor,
    ignore_index: int,
    transplant_prob: float,
) -> Tensor:
    """Overwrite a `transplant_prob` fraction of each edit's corrupted (to-be-deleted) span with
    a same-length contiguous run lifted from `donor`, instead of the i.i.d. uniform-random tokens
    `AutoIndexerModelBase.sanitize_labels` fills every span with by default.

    Every real edit's corrupted span sits at `[pt.start_cursor_idx, pt.start_cursor_idx +
    pt.operation.num_delete)` in the raw stream -- exactly where `batch_perturb_labels` leaves it
    at `ignore_index` (the eval harness's `EditCase.draft` writes its own scripted corruption at
    that same offset). Uniform-random filler is trivially anomalous on token statistics alone,
    letting the marker head shortcut detection to "did something statistically weird just happen"
    instead of learning to notice fluent-but-wrong content in context -- a shortcut pure
    insertions get none of, since their "corrupted" span is empty (a content gap, not injected
    noise). Splicing in a real span from a different document (mirroring the eval harness's own
    `transplant_corruption`) forces the same in-context-coherence judgment substitutions/deletions
    would otherwise get to skip.

    `donor` rows are matched to `tokens` rows by index, wrapping if `donor` has fewer rows (e.g.
    it comes from a differently-sized previous training step). Any `ignore_index` sampled from a
    padded `donor` tail falls back to `tokens`' own filler at that position, so `input_ids` never
    sees `ignore_index` leak through.
    """
    if transplant_prob <= 0 or donor is None or donor.numel() == 0:
        return tokens
    batch_size, seq_len = tokens.shape
    donor = donor[torch.arange(batch_size, device=tokens.device) % donor.shape[0]].to(tokens.device)
    donor_len = donor.shape[1]
    out = tokens.clone()
    for pt in perturbations:
        k = pt.operation.num_delete
        s = pt.start_cursor_idx
        if not k or s is None or s + k > seq_len or k > donor_len:
            continue
        use_transplant = torch.rand(batch_size, device=tokens.device) < transplant_prob
        if not bool(use_transplant.any()):
            continue
        offsets = torch.randint(0, donor_len - k + 1, (batch_size,), device=tokens.device)
        span_idx = offsets[:, None] + torch.arange(k, device=tokens.device)[None, :]
        donor_span = donor.gather(1, span_idx)
        current = out[:, s : s + k]
        donor_span = torch.where(donor_span == ignore_index, current, donor_span)
        out[:, s : s + k] = torch.where(use_transplant[:, None], donor_span, current)
    return out
