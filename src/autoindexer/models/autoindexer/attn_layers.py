from typing import Optional, Tuple, Dict

import torch
from torch import Tensor, nn as nn
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from autoindexer.models.autoindexer.attention.utils import apply_cos_sin
from autoindexer.models.autoindexer.config import AutoIndexerConfig
from autoindexer.models.autoindexer.perturb_labels import ParsedPositions, PositionBlockList
from autoindexer.models.autoindexer.type_utils import CursorIndices, RotaryEmbeddingFunc, IterPositions, PositionBlock, \
    BlockType


def compute_attn_weights_iter(rotary_emb: RotaryEmbeddingFunc, query: Tensor, keys: Tensor, parsed_positions: IterPositions, scaling: Optional[float] = None) -> Tensor:
    batch_size, num_heads, q_len, head_dim = query.shape
    kv_len = keys.shape[-2]
    assert isinstance(parsed_positions, IterPositions)
    # This should be true at inference time when we are processing one token at a time
    assert q_len == 1
    key_with_pos = apply_cos_sin(keys, *parsed_positions.get_rotary_pos_emb(rotary_emb, keys))
    attn_weights = query @ key_with_pos.transpose(-1, -2)  # (B, H, I, J)
    attn_weights.masked_fill_(~parsed_positions.attn_mask.view(-1, 1, 1, kv_len), float("-inf"))
    if scaling is None:
        scaling = head_dim**-0.5
    attn_weights = attn_weights * scaling
    return attn_weights


def compute_end_index_mask(parsed_positions: ParsedPositions, max_span: Optional[int] = None) -> Tensor:
    """Keys the end-index head is allowed to point at: strictly past the start cursor."""
    allowed = parsed_positions.key_pos_ids > 0
    if max_span is not None:
        allowed = allowed & (parsed_positions.key_pos_ids <= max_span)
    if parsed_positions.is_edit_token is not None:
        allowed[~parsed_positions.is_edit_token] = True
    return allowed  # (batch_size, k_len) at inference, (k_len,) per block at training


def compute_index_losses(
        rotary_emb: RotaryEmbeddingFunc,
        query: Tensor,
        keys: Tensor,
        parsed_positions: PositionBlockList,
        cursor_indices: CursorIndices,
        scaling: Optional[float] = None,
        return_weights: bool = False,
        editable_start: int = 0,
) -> Tuple[Tuple[Tensor, Tensor], Dict[int, Tuple[Tensor, Tensor]]]:
    batch_size, num_heads, q_len, head_dim = query.shape
    assert isinstance(parsed_positions, list)
    assert num_heads == 2
    kv_len = keys.shape[2]
    assert q_len == kv_len

    batch_size, num_heads, q_len, _ = query.shape
    k_len = keys.shape[-2]
    assert q_len == k_len

    start_index_loss = torch.tensor(0, dtype=query.dtype, device=query.device)
    end_index_loss = torch.tensor(0, dtype=query.dtype, device=query.device)
    num_edits = 0
    weights = {}
    for block in parsed_positions:
        block: PositionBlock
        if block.block_type == BlockType.EXTEND:
            num_edits += 1
            edit_token_idx = block.end_idx - 1
            start_index_query = query[:, 0, edit_token_idx - 1, :][:, None, None, :]
            end_index_query = query[:, 1, edit_token_idx, :][:, None, None, :]

            # `block.key_pos_ids` is absolute (pre-rotary-offset) here, so `editable_start` (also
            # an absolute token count) compares directly against it to hard-block locked context.
            cursor_start_positions = IterPositions(
                key_pos_ids=block.key_pos_ids[:-1] - block.query_pos_ids[-2],
                attn_mask=block.attn_mask[:-1] & (block.key_pos_ids[:-1] >= editable_start),
            )
            cursor_end_positions = IterPositions(
                key_pos_ids=block.key_pos_ids - block.query_pos_ids[-1],
                attn_mask=block.attn_mask & (block.key_pos_ids >= editable_start),
            )

            # (B, 1, 1, k_len)
            start_attn_weights = compute_attn_weights_iter(
                rotary_emb, start_index_query, keys[:, [0], :edit_token_idx, :], cursor_start_positions, scaling
            )
            end_attn_weights = compute_attn_weights_iter(
                rotary_emb, end_index_query, keys[:, [1], :edit_token_idx + 1, :], cursor_end_positions, scaling
            )
            end_attn_weights = end_attn_weights.masked_fill(~compute_end_index_mask(cursor_end_positions), float("-inf"))

            start_cursor_idx, end_cursor_idx = cursor_indices[edit_token_idx]
            start_index_loss += - torch.log_softmax(start_attn_weights, dim=-1)[..., start_cursor_idx].mean()
            end_index_loss += - torch.log_softmax(end_attn_weights, dim=-1)[..., end_cursor_idx].mean()

            if return_weights:
                weights[edit_token_idx] = (start_attn_weights, end_attn_weights)

    if num_edits > 0:
        start_index_loss /= num_edits
        end_index_loss /= num_edits
    else:
        # This sample had no Extend blocks, so q_proj/k_proj got no gradient this step. Under
        # DeepSpeed ZeRO that desyncs the collective gradient reduction across ranks, so keep
        # them in the graph with a true zero contribution instead of skipping the backward.
        zero = (query.sum() + keys.sum()) * 0.0
        start_index_loss = start_index_loss + zero
        end_index_loss = end_index_loss + zero

    return (start_index_loss, end_index_loss), weights


class IndexKeysCache:
    def __init__(self):
        self.keys = None

    def update(self, key_states: torch.Tensor) -> Tensor:
        self.keys = key_states if self.keys is None else torch.cat([self.keys, key_states], dim=-2)
        return self.keys


class AutoIndexerIndexHead(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AutoIndexerConfig):
        super().__init__()
        num_attention_heads: int = 2
        self.head_dim = getattr(config, "head_dim", config.hidden_size // num_attention_heads)
        self.q_proj = nn.Linear(config.hidden_size, num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, num_attention_heads * self.head_dim, bias=config.attention_bias)
        index_head_scaling = getattr(config, "index_head_scaling", 1.0)
        self.attn_scaling = self.head_dim**-0.5 * index_head_scaling

    def forward(
        self,
        hidden_states: Tensor,
        parsed_positions: ParsedPositions,
        cursor_indices: Optional[CursorIndices] = None,
        attention_mask: Optional[Tensor] = None,
        index_keys_cache: Optional[IndexKeysCache] = None,
        rotary_emb: RotaryEmbeddingFunc = None,
        return_weights: bool = False,
        editable_start: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[Tensor | Dict[int, Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]], Optional[IndexKeysCache]]:
        for unused_key in ("position_embeddings", "cache_position"):
            if unused_key in kwargs:
                kwargs.pop(unused_key)

        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if index_keys_cache is not None:
            key_states = index_keys_cache.update(key_states)

        # TODO: handle attention mask

        if isinstance(parsed_positions, IterPositions):
            # `index_attn_mask` (set by `AutoIndexerIdParser`, e.g. to the prompt length during
            # generation) additionally locks out positions outside the current response.
            cursor_positions = parsed_positions
            if parsed_positions.index_attn_mask is not None:
                cursor_positions = IterPositions(
                    key_pos_ids=parsed_positions.key_pos_ids,
                    attn_mask=parsed_positions.index_attn_mask,
                    is_edit_token=parsed_positions.is_edit_token,
                )
            attn_weights = compute_attn_weights_iter(rotary_emb, query_states, key_states, cursor_positions, self.attn_scaling)
            end_allowed = compute_end_index_mask(cursor_positions)[:, None, :]
            attn_weights[:, 1] = attn_weights[:, 1].masked_fill(~end_allowed, float("-inf"))
            index_losses = None
        else:
            assert cursor_indices is not None
            index_losses, attn_weights = compute_index_losses(
                rotary_emb,
                query_states,
                key_states,
                parsed_positions,
                cursor_indices,
                scaling=self.attn_scaling,
                return_weights=return_weights,
                editable_start=editable_start,
            )

        return attn_weights, index_losses, index_keys_cache  # (batch_size, 2, q_len, k_len)
