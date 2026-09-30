from typing import Optional, Tuple

import torch
from torch import Tensor, nn as nn

from autoindexer.models.autoindexer.attention.utils import apply_cos_sin
from autoindexer.models.autoindexer.perturb_labels import ParsedPositions, PositionBlockList
from autoindexer.models.autoindexer.type_utils import RotaryEmbeddingFunc


def compute_attn_weights_over_segments_ref(
    query,
    query_with_pos,
    keys,
    concat_key_rotary_pos_emb,
    concat_abs_pos_mask,
    concat_attn_mask,
    boundary_indices,
):
    """
    Compute attention logits segment-by-segment.

    :param query: (batch_size, num_heads, q_len, head_dim) unrotated query states.
    :param query_with_pos: (batch_size, num_heads, q_len, head_dim) query states with RoPE applied.
    :param keys: (batch_size, num_heads, kv_len, head_dim) unrotated key states.
    :param concat_key_rotary_pos_emb: tuple of cos and sin of shape (batch_size, sum_kv_len, half_dim) for the rotary positions for each key token.
    :param concat_abs_pos_mask: (sum_kv_len,) boolean mask; True selects absolute (unrotated) query-key dot products.
    :param concat_attn_mask: (sum_kv_len,) boolean mask; False entries are masked to -inf.
    :param boundary_indices: list of (start_idx, end_idx, key_start_idx, key_end_idx) tuples,
        each defining one segment. Per segment, with i_seg = end_idx - start_idx and
        j_seg = key_end_idx - key_start_idx:
        - ``_query``, ``_query_with_pos``: (batch_size, num_heads, i_seg, head_dim)
        - ``_key_with_pos_tr``: (batch_size, num_heads, head_dim, j_seg)
        - ``_attn_weights``: (batch_size, num_heads, i_seg, j_seg), written to
          ``attn_weights[..., start_idx:end_idx, :end_idx]``.
    :return: (batch_size, num_heads, q_len, kv_len) attention logits; unowned cells use diagonal
        zeros and ``-inf`` elsewhere.
    """
    batch_size, num_heads, q_len, _ = query.shape
    k_len = keys.shape[-2]
    assert q_len == k_len
    attn_weights = torch.full((batch_size, num_heads, q_len, k_len), float("-inf"), dtype=query.dtype, device=query.device)
    attn_weights.diagonal(dim1=-2, dim2=-1).fill_(0)
    assert boundary_indices, "No boundary_indices provided."

    for start_idx, end_idx, key_start_idx, key_end_idx in boundary_indices:
        _query = query[..., start_idx:end_idx, :]
        _query_with_pos = query_with_pos[..., start_idx:end_idx, :]
        concat_key_cos, concat_key_sin = concat_key_rotary_pos_emb
        _key_with_pos_tr = apply_cos_sin(
            keys[..., :end_idx, :],
            concat_key_cos[:, key_start_idx:key_end_idx, :],
            concat_key_sin[:, key_start_idx:key_end_idx, :],
        ).transpose(-1, -2)

        _abs_pos_mask = concat_abs_pos_mask[key_start_idx:key_end_idx]
        _attn_weights = _query_with_pos @ _key_with_pos_tr  # (B, I, J)
        abs_attn_dot_prod_diff = (_query - _query_with_pos) @ _key_with_pos_tr  # (B, I, J)
        _attn_weights += abs_attn_dot_prod_diff * _abs_pos_mask

        _attn_weights.masked_fill_(~concat_attn_mask[key_start_idx:key_end_idx], float("-inf"))
        attn_weights[..., start_idx:end_idx, :end_idx] = _attn_weights

    return attn_weights


def compute_attn_weights_over_segments_ref_wrapper(
    rotary_emb: RotaryEmbeddingFunc,
    query: Tensor,
    keys: Tensor,
    position_blocks: PositionBlockList,
) -> Tensor:
    """
    :param rotary_emb: RotaryEmbeddingFunc used to rotate query and key states.
    :param query: (batch_size, num_heads, q_len, head_dim) unrotated query states.
    :param keys: (batch_size, num_heads, kv_len, head_dim) unrotated key states.
    :param position_blocks: list of ``PositionBlock`` instances, one per sequence segment.
    :return: (batch_size, num_heads, q_len, kv_len) attention logits after segment aggregation.
    """
    batch_size, num_heads, q_len, head_dim = query.shape
    kv_len = keys.shape[-2]
    assert q_len == kv_len
    query_rotary_pos_emb, concat_key_rotary_pos_emb = position_blocks.get_rotary_pos_emb(rotary_emb, query, q_len)
    query_with_pos = apply_cos_sin(query, *query_rotary_pos_emb)
    return compute_attn_weights_over_segments_ref(
        query,
        query_with_pos,
        keys,
        concat_key_rotary_pos_emb,
        position_blocks.concat_abs_pos_masks,
        position_blocks.concat_attn_masks,
        position_blocks.boundary_indices,
    )


def compute_attn_weights_ref(rotary_emb: RotaryEmbeddingFunc, query: Tensor, keys: Tensor, parsed_positions: ParsedPositions, scaling: Optional[float] = None) -> Tensor:
    batch_size, num_heads, q_len, head_dim = query.shape
    assert isinstance(parsed_positions, list)
    kv_len = keys.shape[2]
    assert q_len == kv_len

    device = query.device

    attn_weights = compute_attn_weights_over_segments_ref_wrapper(rotary_emb, query, keys, parsed_positions)

    causal_mask = torch.tril(torch.ones((q_len, kv_len), dtype=torch.bool, device=device), diagonal=kv_len - q_len)
    attn_weights.masked_fill_(~causal_mask, float("-inf"))

    if scaling is None:
        scaling = head_dim**-0.5

    attn_weights = attn_weights * scaling
    return attn_weights


def chain_of_edits_attn_ref(
    rotary_emb: RotaryEmbeddingFunc,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    parsed_positions: PositionBlockList,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scaling: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    attn_weights = compute_attn_weights_ref(rotary_emb, query, keys, parsed_positions, scaling)

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights.masked_fill_(attention_mask.logical_not(), float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # `dropout_p` is already 0.0 when not training (see the caller, `attn_layers.py`'s
    # `attention_interface`), so `nn.functional.dropout`'s default `training=True` is a no-op
    # in that case and there is no need for a separate `training` flag here.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
    attn_output = torch.matmul(attn_weights, values)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
