"""
CuteDSL counterpart to ``attention/triton_wrapper.py`` / ``attention/reference.py``.
Exposes ``fused_chain_of_edits_attn_cutedsl`` with the signature the
``autoindexer_attention_forward`` dispatch expects, backed by the FA2x2
CuteDSL pipeline (fast_rope + prep-pack + varlen attention), R-path /
N-path decomposed and log-sum-exp merged. Backward is supported end-to-end.

``theta`` is read from ``rotary_emb.config.rope_theta`` and passed to the fast_rope / prep
entry points; ``head_dim`` is taken from the query tensor at call time. Those kernels
compile lazily on first use and cache the
result via flash_attn.cute.cache_utils.get_jit_cache — in-memory by default, or
disk-backed with FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 — keyed by
(head_dim, theta, direction, dtype). Nothing compiles at import.

Correctness caveats to validate against ``chain_of_edits_attn_ref``:
  * intra-segment causality — the reference applies a global causal ``tril`` on
    top of the segment routing; this path uses ``causal=False`` (segment routing
    only). Reconcile before trusting a run.
  * bf16 only; full-rotary head_dim assumed (matches Qwen3 default).
  * external ``attention_mask`` and attention ``dropout`` are unsupported.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from autoindexer.models.autoindexer.type_utils import RotaryEmbeddingFunc

try:
    # points FA4 imports to the newly installed one, not a pre-installed copy on the image
    import cutlass
    import cutlass.cute as cute
    import sysconfig
    import flash_attn

    _fa4_purelib_path = sysconfig.get_paths()["purelib"] + "/flash_attn"
    if _fa4_purelib_path not in flash_attn.__path__:
        flash_attn.__path__.insert(0, _fa4_purelib_path)

    from flash_attn.cute.utils import scalar_to_ssa

    from autoindexer.models.autoindexer.attention.cutedsl.fast_rope.autograd_key import fast_rope_key
    from autoindexer.models.autoindexer.attention.cutedsl.prep_multi_head.autograd import prep_pack
    from flash_attn.cute.interface import flash_attn_varlen_func

    @cute.jit
    def causal_mask_mod(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
        # prep_pack reorders K/V, so packed row/col position no longer implies sequence order -- causal has to compare real token indices instead.
        q_tok = aux_tensors[0]
        k_tok = aux_tensors[1]
        m_frag = cute.make_rmem_tensor(1, cutlass.Int32); m_frag.store(q_idx + seqlen_info.offset_q)
        n_frag = cute.make_rmem_tensor(1, cutlass.Int32); n_frag.store(kv_idx + seqlen_info.offset_k)
        m = cutlass.min(m_frag[0], q_tok.shape[0] - 1)
        n = cutlass.min(n_frag[0], k_tok.shape[0] - 1)
        q_frag = cute.make_rmem_tensor(1, cutlass.Int32); q_frag[0] = q_tok[m]
        k_frag = cute.make_rmem_tensor(1, cutlass.Int32); k_frag[0] = k_tok[n]
        return (q_frag.load() - k_frag.load()) >= scalar_to_ssa(0, cutlass.Int32)

    _HAS_CUTEDSL = True
except (ImportError, AttributeError):
    # AttributeError too: a stale pre-installed `cutlass` (see the purelib redirect above,
    # which only covers `flash_attn`) can shadow the pip-installed one and raise e.g.
    # `module 'cutlass.cute.core' has no attribute 'ThrMma'` instead of failing to import.
    fast_rope_key = None
    prep_pack = None
    flash_attn_varlen_func = None
    causal_mask_mod = None
    _HAS_CUTEDSL = False

# flash_attn's cute varlen kernels only support Hopper+ (sm 9.x/10.x/11.x/12.x). On older
# GPUs (e.g. Ampere, sm 8.x) the import above succeeds -- `_HAS_CUTEDSL` would report True --
# but the kernel asserts at launch instead of failing to import, so gate on compute capability
# too instead of just import success.
if _HAS_CUTEDSL and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9):
    _HAS_CUTEDSL = False

def _cutedsl_fwd(q_nope_f, k_f, v_f, meta, dk, softmax_scale, B, seq_len, theta):
    """All tensors flattened to (B*seq_len, H, dk) bf16. Returns (B, seq_len, H, dk)."""
    device = q_nope_f.device
    H = q_nope_f.shape[1]
    R, N = meta["R"], meta["N"]

    # query RoPE by REAL position ids (explicit-position kernel), single call
    # over the tiled positions — correct across sample boundaries.
    q_rope_f = fast_rope_key(q_nope_f, meta["query_pos"], theta)

    q_R = q_rope_f[R["q_idx"]]
    k_R, v_R = prep_pack(k_f, v_f, R["flat"], R["key_pos"], R["inv_idx"], theta)
    out_R, lse_R = flash_attn_varlen_func(
        q_R, k_R, v_R,
        cu_seqlens_q=R["cu_q"], cu_seqlens_k=R["cu_k"], max_seqlen_q=R["max_q"], max_seqlen_k=R["max_k"],
        softmax_scale=softmax_scale, causal=False, return_lse=True,
        mask_mod=causal_mask_mod, aux_tensors=[R["q_tok"], R["k_tok"]],
    )
    lse_R = lse_R.transpose(0, 1)  # (H, T) -> (T, H)

    out_flat = torch.zeros((B * seq_len, H, dk), dtype=torch.bfloat16, device=device)
    out_flat[R["q_idx"]] = out_R

    if N is not None:
        q_N = q_nope_f[N["q_idx"]]
        k_N, v_N = prep_pack(k_f, v_f, N["flat"], N["key_pos"], N["inv_idx"], theta)
        out_N, lse_N = flash_attn_varlen_func(
            q_N, k_N, v_N,
            cu_seqlens_q=N["cu_q"], cu_seqlens_k=N["cu_k"], max_seqlen_q=N["max_q"], max_seqlen_k=N["max_k"],
            softmax_scale=softmax_scale, causal=False, return_lse=True,
            mask_mod=causal_mask_mod, aux_tensors=[N["q_tok"], N["k_tok"]],
        )
        lse_N = lse_N.transpose(0, 1)

        # position of each N row within the R pack. Every N row must also be an
        # R row (every segment has >=1 R key). Sentinel-fill + assert so a
        # violated invariant fails loudly instead of reading uninitialized memory.
        r_pos = torch.full((B * seq_len,), -1, dtype=torch.long, device=device)
        r_pos[R["q_idx"]] = torch.arange(R["q_idx"].shape[0], device=device)
        r_pos_for_N = r_pos[N["q_idx"]]
        assert (r_pos_for_N >= 0).all(), "cutedsl: an N-path row has no R-path row (a segment lacks an R key)"

        lse_sub = torch.logaddexp(lse_R[r_pos_for_N], lse_N)
        w_N = torch.exp(lse_N - lse_sub).to(out_flat.dtype)
        out_flat[N["q_idx"]] = torch.lerp(out_R[r_pos_for_N], out_N, w_N[:, :, None])

    return out_flat.reshape(B, seq_len, H, dk)


def fused_chain_of_edits_attn_cutedsl(
    rotary_emb: RotaryEmbeddingFunc,
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    parsed_positions,
    attention_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scaling: Optional[float] = None,
) -> Tuple[Tensor, None]:
    """Returns ``(attn_output, None)`` with ``attn_output`` shaped
    (batch, seq_len, num_heads, head_dim) — the layout the caller reshapes via
    ``.view(batch, seq_len, -1)``.

    Only used by ``autoindexer_attention_forward`` when CuteDSL + CUDA are available;
    callers must guard for that themselves (see ``_HAS_CUTEDSL``).
    """
    assert isinstance(parsed_positions, list), "cutedsl path expects a PositionBlockList"
    # NOTE: `attention_mask` here is HF Qwen3Model's auto-injected standard causal
    # mask. Causality is now enforced inside the kernel by causal_mask_mod (key_tok
    # <= query_tok), so this redundant external mask is intentionally not applied.
    # Padding is handled structurally by the segmentation, not this mask.
    # The caller (`attn_layers.py`'s `attention_interface`) already resolves `dropout_p` to 0.0
    # when not training, so a nonzero `dropout_p` here always means dropout was requested.
    if dropout_p:
        raise NotImplementedError("cutedsl path has no attention dropout")

    B, H, q_len, dk = query.shape
    assert keys.shape[2] == q_len, "cutedsl path assumes self-attention (q_len == kv_len)"

    softmax_scale = scaling if scaling is not None else dk ** -0.5
    device = query.device
    theta = float(rotary_emb.config.rope_theta)

    # Built once per forward and cached on parsed_positions (shared across all
    # attention layers), so later layers reuse it instead of rebuilding.
    meta = parsed_positions.get_cutedsl_meta(B, q_len, device)

    def _flat(x):
        return x.permute(0, 2, 1, 3).reshape(B * q_len, H, dk).contiguous().to(torch.bfloat16)

    out = _cutedsl_fwd(_flat(query), _flat(keys), _flat(values), meta, dk, softmax_scale, B, q_len, theta)
    # kernels are bf16-only; cast back to the model's dtype (e.g. fp32) so o_proj matches.
    return out.to(query.dtype).contiguous(), None


if not _HAS_CUTEDSL:
    fused_chain_of_edits_attn_cutedsl = None
