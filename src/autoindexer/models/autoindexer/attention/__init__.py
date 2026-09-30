from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv

from autoindexer.models.autoindexer.attention.cutedsl.wrapper import _HAS_CUTEDSL, fused_chain_of_edits_attn_cutedsl
from autoindexer.models.autoindexer.attention.prefill_sdpa import plain_causal_attn_sdpa
from autoindexer.models.autoindexer.attention.reference import chain_of_edits_attn_ref
from autoindexer.models.autoindexer.attention.triton_wrapper import _HAS_TRITON, fused_chain_of_edits_attn_triton
from autoindexer.models.autoindexer.perturb_labels import PositionBlockList
from autoindexer.models.autoindexer.type_utils import IterPositions, RotaryEmbeddingFunc

_registered = False


def _wrap_chain_of_edits_attn_backend(backend_fn: Callable) -> Callable:
    def attention_interface(
        module: nn.Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        rotary_emb: RotaryEmbeddingFunc = None,
        parsed_positions: PositionBlockList = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Deferred import avoids a circular dependency: `attn_layers.py` imports from this package's submodules
        from autoindexer.models.autoindexer.attn_layers import compute_attn_weights_iter

        # Grouped-query attention: `Qwen3Attention`/`LlamaAttention` leave `key`/`value` un-expanded
        # and expect the registered interface to expand them, like `eager_attention_forward` does.
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

        if isinstance(parsed_positions, IterPositions):
            attn_weights = compute_attn_weights_iter(rotary_emb, query, key, parsed_positions, scaling)

            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attn_weights.masked_fill_(attention_mask.logical_not(), float("-inf"))
                else:
                    attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=False)
            attn_output = torch.matmul(attn_weights, value)
            return attn_output, attn_weights

        kwargs = {}
        if parsed_positions is not None:
            kwargs.update(parsed_positions=parsed_positions)

        return backend_fn(
            rotary_emb,
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout_p=dropout,
            scaling=scaling,
            **kwargs
        )

    return attention_interface


def register_autoindexer_attention_kernels() -> None:
    """Register AutoIndexer's chain-of-edits attention backends into HuggingFace's `ALL_ATTENTION_FUNCTIONS`."""
    global _registered
    if _registered:
        return

    hf_key_to_backend_fn: Dict[str, Callable] = {
        "autoindexer_eager": chain_of_edits_attn_ref,
        # Not a configurable checkpoint `attn_implementation` -- `_prefill_prompt` swaps to this
        # backend internally for its one plain-causal, no-edits forward pass.
        "autoindexer_prefill_sdpa": plain_causal_attn_sdpa,
    }
    if _HAS_TRITON:
        hf_key_to_backend_fn["autoindexer_triton"] = fused_chain_of_edits_attn_triton
    if _HAS_CUTEDSL:
        hf_key_to_backend_fn["autoindexer_cutedsl"] = fused_chain_of_edits_attn_cutedsl

    for hf_key, backend_fn in hf_key_to_backend_fn.items():
        ALL_ATTENTION_FUNCTIONS.register(hf_key, _wrap_chain_of_edits_attn_backend(backend_fn))

    _registered = True


def resolve_available_attn_implementation(requested: str) -> str:
    """Downgrade `requested` to the best available backend if its own isn't installed (CuteDSL/Triton)."""
    if requested == "autoindexer_cutedsl" and not _HAS_CUTEDSL:
        requested = "autoindexer_triton"
    if requested == "autoindexer_triton" and not _HAS_TRITON:
        return "autoindexer_eager"
    return requested


__all__ = ["register_autoindexer_attention_kernels", "resolve_available_attn_implementation"]
