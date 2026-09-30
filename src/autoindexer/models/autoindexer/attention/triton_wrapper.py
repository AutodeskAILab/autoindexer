from typing import Optional, Tuple

from torch import Tensor

from autoindexer.models.autoindexer.type_utils import RotaryEmbeddingFunc
from autoindexer.models.autoindexer.perturb_labels import (
    PositionBlockList,
)

try:
    # Fused triton kernels for compute_attn_weights_over_segments and the fully-fused
    # softmax+dropout+attn@V attention output. Importing triton itself does not require a
    # GPU; kernel launches do, so we additionally gate on query.is_cuda at call time (see
    # compute_attn_weights_over_segments / compute_fused_attn_output).
    from autoindexer.models.autoindexer.attention.kernels.triton_rope_attn import (
        compute_attn_weights_over_segments_triton,
        compute_fused_attn_output_over_segments_triton,
    )

    _HAS_TRITON = True
except ImportError:
    compute_attn_weights_over_segments_triton = None
    compute_fused_attn_output_over_segments_triton = None
    _HAS_TRITON = False


def compute_attn_weights_over_segments_triton_wrapper(
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
    launch_cache = position_blocks.get_triton_launch_cache(q_len, head_dim)
    return compute_attn_weights_over_segments_triton(
        query,
        query_rotary_pos_emb,
        keys,
        concat_key_rotary_pos_emb,
        position_blocks.concat_abs_pos_masks,
        position_blocks.concat_attn_masks,
        position_blocks.boundary_indices,
        launch_cache,
    )


def fused_chain_of_edits_attn_triton(
    rotary_emb: RotaryEmbeddingFunc,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    parsed_positions: PositionBlockList,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scaling: Optional[float] = None,
) -> Tuple[Tensor, None]:
    """Triton-only counterpart to ``compute_attn_weights`` that fuses causal masking, the
    external ``attention_mask``, softmax, dropout, and the ``attn_weights @ value`` matmul
    into a single kernel, so the (q_len, kv_len) attention-weights matrix is never
    materialized. Only used by ``autoindexer_attention_forward`` when Triton + CUDA are available;
    callers must guard for that themselves (see ``_HAS_TRITON``).

    The caller (``attn_layers.py``'s ``attention_interface``) already resolves ``dropout_p`` to
    ``0.0`` when not training, so there is no separate ``training`` flag to thread through here.

    ``compute_fused_attn_output_over_segments_triton`` writes its output directly in
    ``(batch, seq, heads, head_dim)`` layout (see that function's docstring), matching what
    ``chain_of_edits_attn_ref`` produces via its own ``.transpose(1, 2).contiguous()``, so no
    separate transpose is needed here.
    """
    batch_size, num_heads, q_len, head_dim = query.shape
    assert isinstance(parsed_positions, list)
    kv_len = keys.shape[2]
    assert q_len == kv_len

    query_rotary_pos_emb, concat_key_rotary_pos_emb = parsed_positions.get_rotary_pos_emb(rotary_emb, query, q_len)
    launch_cache = parsed_positions.get_triton_launch_cache(q_len, head_dim)
    return compute_fused_attn_output_over_segments_triton(
        query,
        query_rotary_pos_emb,
        keys,
        values,
        concat_key_rotary_pos_emb,
        parsed_positions.concat_abs_pos_masks,
        parsed_positions.concat_attn_masks,
        parsed_positions.boundary_indices,
        attention_mask=attention_mask,
        dropout_p=dropout_p,
        scaling=scaling,
        launch_cache=launch_cache,
    ), None


