"""
torch.autograd.Function gluing fast_rope_key's forward/backward CuteDSL
kernels together. `pos` is an explicit, non-differentiable per-row position
array (not implied by row index) that must be saved for backward, since
backward needs the SAME positions forward used to correctly invert each
row's rotation.
"""

import torch

from .kernel_key import rope_key_run


class FastRopeKeyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pos, theta):
        x = x.contiguous()
        pos = pos.contiguous()
        out = torch.empty_like(x)
        rope_key_run(x, pos, out, theta, 1.0)
        ctx.save_for_backward(pos)
        ctx.theta = theta
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pos, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        rope_key_run(grad_output, pos, grad_input, ctx.theta, -1.0)
        return grad_input, None, None


def fast_rope_key(x, pos, theta):
    """Apply RoPE to x: (seq_len, H, DK), using an explicit per-row position
    array pos: (seq_len,) int32 — NOT assumed to equal each row's own index.
    Autograd-tracked w.r.t. x; pos is a non-differentiable index array. The
    kernel is compiled+cached on first use (see rope_key_run), keyed on theta."""
    return FastRopeKeyFunction.apply(x, pos.to(torch.int32), theta)
