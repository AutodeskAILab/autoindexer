"""
torch.autograd.Function gluing the multi-head prep-kernel forward/backward
together — same spirit as fast_rope/autograd.py: kernel build/compile logic
stays in fwd_fa2x2_prep.py, this module only holds the autograd wiring.

`inv_idx` is the deliberate exception to "forward only takes what forward
needs": forward's own gather doesn't touch it, but it's required as an input
anyway so ctx.save_for_backward can carry it to backward without the caller
managing that wiring by hand. It's dataloader-side metadata — a pure
function of src_idx, built once per sample via
utils/fa2_path_meta.build_inverse_index, same tier as src_idx/rope_idx
themselves — so it must be built together with src_idx and passed in
alongside it; if it ever goes stale relative to src_idx, backward produces
silently wrong gradients, not an error.
"""

import torch

from .fwd_fa2x2_prep import fwd_prep, bwd_prep


class PrepPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, src_idx, rope_idx, inv_idx, theta):
        k, v = k.contiguous(), v.contiguous()
        num_out, H, dk = src_idx.shape[0], k.shape[1], k.shape[2]

        k_out = torch.empty((num_out, H, dk), dtype=torch.bfloat16, device=k.device)
        v_out = torch.empty((num_out, H, dk), dtype=torch.bfloat16, device=k.device)
        fwd_prep(k, v, src_idx, rope_idx, k_out, v_out, theta)

        ctx.save_for_backward(rope_idx, inv_idx)
        ctx.num_src = k.shape[0]
        ctx.dk = dk
        ctx.theta = theta
        return k_out, v_out

    @staticmethod
    def backward(ctx, grad_k_out, grad_v_out):
        rope_idx, inv_idx = ctx.saved_tensors
        grad_k_out = grad_k_out.contiguous()
        grad_v_out = grad_v_out.contiguous()
        H = grad_k_out.shape[1]

        grad_k = torch.empty((ctx.num_src, H, ctx.dk), dtype=torch.bfloat16, device=grad_k_out.device)
        grad_v = torch.empty((ctx.num_src, H, ctx.dk), dtype=torch.bfloat16, device=grad_k_out.device)
        bwd_prep(grad_k_out, grad_v_out, rope_idx, inv_idx, grad_k, grad_v, ctx.theta)

        # No gradient for src_idx / rope_idx / inv_idx / theta.
        return grad_k, grad_v, None, None, None, None


def prep_pack(k, v, src_idx, rope_idx, inv_idx, theta):
    """
    Gather + RoPE-pack K, gather-pack V, with autograd support.

    k, v: (num_src, H, DK) bfloat16
    src_idx, rope_idx: (num_out,) int32 — dataloader-precomputed, see
        fa2_path_meta.build_fa2_path_meta_batched.
    inv_idx: (num_src, max_mult) int32 — dataloader-precomputed inverse of
        src_idx, see utils/fa2_path_meta.build_inverse_index. Only used by
        backward, but required here so it survives to backward via ctx.

    theta is the RoPE base; the prep kernels are compiled+cached on first use
    (see fwd_prep / bwd_prep), keyed on head_dim and theta.

    Returns (k_out, v_out), each (num_out, H, dk) bfloat16.
    """
    return PrepPackFunction.apply(k, v, src_idx, rope_idx, inv_idx, theta)
