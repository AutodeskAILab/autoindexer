from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from autoindexer.models.autoindexer.attention.utils import apply_cos_sin
from autoindexer.models.autoindexer.type_utils import RotaryEmbeddingFunc


def plain_causal_attn_sdpa(
    rotary_emb: RotaryEmbeddingFunc,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs
) -> Tuple[Tensor, None]:
    """Plain-causal counterpart to ``chain_of_edits_attn_ref``, backed by
    ``torch.nn.functional.scaled_dot_product_attention`` instead of the chain-of-edits kernels.

    Used only for ``_plain_causal_positions`` forwards -- a single, un-perturbed ``COMPLETE``
    block with count-up positions and no edits, where the chain-of-edits machinery buys nothing.
    ``AutoIndexerModelBase._prefill_prompt`` needs this specifically because that forward batches
    left-padded prompts of *different* real lengths (see its docstring), and the custom
    eager/Triton/CuteDSL kernels' structural ``attn_mask`` is a single vector shared by the whole
    batch -- it has no per-row stride, so it cannot express a per-row real/padding boundary.
    ``scaled_dot_product_attention``'s own ``attn_mask`` natively broadcasts per row, so it can.

    RoPE is applied explicitly here: the backbone's own ``rotary_emb`` submodule is swapped for
    an identity (see ``AutoIndexerModelBase.__init__``) so the chain-of-edits kernels can apply
    their own position-dependent rotation internally, which means ``query``/``keys`` arrive here
    unrotated. ``past_key_values.update(...)`` (inside ``Qwen3Attention``/``LlamaAttention.
    forward()``) caches ``keys``/``values`` *before* this function ever runs, so -- exactly like
    every other AutoIndexer attention backend -- the KV cache still stores unrotated keys; the
    rotation computed here is local to this one forward's attention weights and is never
    persisted.
    """
    batch_size, num_heads, q_len, head_dim = query.shape
    kv_len = keys.shape[2]
    assert q_len == kv_len, "plain_causal_attn_sdpa assumes a from-scratch forward (no KV cache yet)"

    device = query.device
    # Shared across the whole batch (`unsqueeze(0)` -> batch dim of 1, broadcasting against the
    # real batch size in `apply_cos_sin`), matching every other position-id usage in this
    # architecture -- see `PositionBlockList.get_rotary_pos_emb`.
    position_ids = torch.arange(q_len, device=device).unsqueeze(0)
    cos, sin = rotary_emb(query, position_ids)
    query = apply_cos_sin(query, cos, sin)
    keys = apply_cos_sin(keys, cos, sin)

    if attention_mask is None:
        attn_output = F.scaled_dot_product_attention(
            query, keys, values, dropout_p=dropout_p, scale=scaling, is_causal=True
        )
    else:
        # `attention_mask` carries only per-key padding validity here, shaped `(batch, 1, 1,
        # kv_len)` (see `_prefill_prompt`, which builds it that way precisely so HF's
        # mask-construction utilities -- which drop anything not already 4D for an unregistered
        # `attn_implementation` -- pass it through untouched). Causality is layered on top of it
        # here, since this path always is a single causal block.
        causal = torch.tril(torch.ones((q_len, kv_len), dtype=torch.bool, device=device))
        combined_mask = causal & attention_mask.to(torch.bool)
        attn_output = F.scaled_dot_product_attention(
            query, keys, values, attn_mask=combined_mask, dropout_p=dropout_p, scale=scaling, is_causal=False
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
