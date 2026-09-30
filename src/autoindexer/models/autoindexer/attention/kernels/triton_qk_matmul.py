"""
Triton forward and backward kernels for the attention-style matmul: Q @ K^T.

Q: (batch_size, q_len, head_dim)
K: (batch_size, kv_len, head_dim)
Output: (batch_size, q_len, kv_len)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_matmul_fwd_kernel(
    Q_ptr,
    K_ptr,
    Out_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_ob,
    stride_oq,
    stride_ok,
    q_len,
    kv_len,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_offset = pid_q * BLOCK_Q
    kv_offset = pid_kv * BLOCK_KV

    q_range = q_offset + tl.arange(0, BLOCK_Q)
    kv_range = kv_offset + tl.arange(0, BLOCK_KV)

    acc = tl.zeros((BLOCK_Q, BLOCK_KV), dtype=tl.float32)

    for d_start in range(0, head_dim, BLOCK_D):
        d_range = d_start + tl.arange(0, BLOCK_D)

        q_ptrs = Q_ptr + pid_b * stride_qb + q_range[:, None] * stride_qq + d_range[None, :] * stride_qd
        q_mask = (q_range[:, None] < q_len) & (d_range[None, :] < head_dim)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        k_ptrs = K_ptr + pid_b * stride_kb + kv_range[:, None] * stride_kk + d_range[None, :] * stride_kd
        k_mask = (kv_range[:, None] < kv_len) & (d_range[None, :] < head_dim)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc += tl.dot(q_block, tl.trans(k_block), input_precision="ieee")

    out_ptrs = Out_ptr + pid_b * stride_ob + q_range[:, None] * stride_oq + kv_range[None, :] * stride_ok
    out_mask = (q_range[:, None] < q_len) & (kv_range[None, :] < kv_len)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _qk_matmul_bwd_dq_kernel(
    dOut_ptr,
    K_ptr,
    dQ_ptr,
    stride_dob,
    stride_doq,
    stride_dok,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    q_len,
    kv_len,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """dQ = dOut @ K"""
    pid_q = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_offset = pid_q * BLOCK_Q
    d_offset = pid_d * BLOCK_D

    q_range = q_offset + tl.arange(0, BLOCK_Q)
    d_range = d_offset + tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for kv_start in range(0, kv_len, BLOCK_KV):
        kv_range = kv_start + tl.arange(0, BLOCK_KV)

        do_ptrs = dOut_ptr + pid_b * stride_dob + q_range[:, None] * stride_doq + kv_range[None, :] * stride_dok
        do_mask = (q_range[:, None] < q_len) & (kv_range[None, :] < kv_len)
        do_block = tl.load(do_ptrs, mask=do_mask, other=0.0)

        k_ptrs = K_ptr + pid_b * stride_kb + kv_range[:, None] * stride_kk + d_range[None, :] * stride_kd
        k_mask = (kv_range[:, None] < kv_len) & (d_range[None, :] < head_dim)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc += tl.dot(do_block, k_block, input_precision="ieee")

    dq_ptrs = dQ_ptr + pid_b * stride_dqb + q_range[:, None] * stride_dqq + d_range[None, :] * stride_dqd
    dq_mask = (q_range[:, None] < q_len) & (d_range[None, :] < head_dim)
    tl.store(dq_ptrs, acc.to(dQ_ptr.dtype.element_ty), mask=dq_mask)


@triton.jit
def _qk_matmul_bwd_dk_kernel(
    dOut_ptr,
    Q_ptr,
    dK_ptr,
    stride_dob,
    stride_doq,
    stride_dok,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    q_len,
    kv_len,
    head_dim: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """dK = dOut^T @ Q"""
    pid_kv = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    kv_offset = pid_kv * BLOCK_KV
    d_offset = pid_d * BLOCK_D

    kv_range = kv_offset + tl.arange(0, BLOCK_KV)
    d_range = d_offset + tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_KV, BLOCK_D), dtype=tl.float32)

    for q_start in range(0, q_len, BLOCK_Q):
        q_range = q_start + tl.arange(0, BLOCK_Q)

        do_ptrs = dOut_ptr + pid_b * stride_dob + q_range[:, None] * stride_doq + kv_range[None, :] * stride_dok
        do_mask = (q_range[:, None] < q_len) & (kv_range[None, :] < kv_len)
        do_block = tl.load(do_ptrs, mask=do_mask, other=0.0)

        q_ptrs = Q_ptr + pid_b * stride_qb + q_range[:, None] * stride_qq + d_range[None, :] * stride_qd
        q_mask = (q_range[:, None] < q_len) & (d_range[None, :] < head_dim)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        acc += tl.dot(tl.trans(do_block), q_block, input_precision="ieee")

    dk_ptrs = dK_ptr + pid_b * stride_dkb + kv_range[:, None] * stride_dkk + d_range[None, :] * stride_dkd
    dk_mask = (kv_range[:, None] < kv_len) & (d_range[None, :] < head_dim)
    tl.store(dk_ptrs, acc.to(dK_ptr.dtype.element_ty), mask=dk_mask)


class QKMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        batch_size, q_len, head_dim = q.shape
        kv_len = k.shape[1]

        q = q.contiguous()
        k = k.contiguous()

        out = torch.empty(
            (batch_size, q_len, kv_len),
            device=q.device,
            dtype=q.dtype,
        )

        BLOCK_Q = min(64, triton.next_power_of_2(q_len))
        BLOCK_KV = min(64, triton.next_power_of_2(kv_len))
        BLOCK_D = min(64, triton.next_power_of_2(head_dim))

        grid = (
            triton.cdiv(q_len, BLOCK_Q),
            triton.cdiv(kv_len, BLOCK_KV),
            batch_size,
        )

        _qk_matmul_fwd_kernel[grid](
            q,
            k,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            q_len,
            kv_len,
            head_dim,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q, k)
        ctx.shapes = (batch_size, q_len, kv_len, head_dim)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k = ctx.saved_tensors
        batch_size, q_len, kv_len, head_dim = ctx.shapes

        grad_output = grad_output.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)

        BLOCK_Q = min(64, triton.next_power_of_2(q_len))
        BLOCK_KV = min(64, triton.next_power_of_2(kv_len))
        BLOCK_D = min(64, triton.next_power_of_2(head_dim))

        grid_dq = (
            triton.cdiv(q_len, BLOCK_Q),
            triton.cdiv(head_dim, BLOCK_D),
            batch_size,
        )
        _qk_matmul_bwd_dq_kernel[grid_dq](
            grad_output,
            k,
            dq,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            q_len,
            kv_len,
            head_dim,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            BLOCK_D=BLOCK_D,
        )

        grid_dk = (
            triton.cdiv(kv_len, BLOCK_KV),
            triton.cdiv(head_dim, BLOCK_D),
            batch_size,
        )
        _qk_matmul_bwd_dk_kernel[grid_dk](
            grad_output,
            q,
            dk,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            q_len,
            kv_len,
            head_dim,
            BLOCK_KV=BLOCK_KV,
            BLOCK_Q=BLOCK_Q,
            BLOCK_D=BLOCK_D,
        )

        return dq, dk


def triton_qk_matmul(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return QKMatmul.apply(q, k)


def test_forward_backward():
    torch.manual_seed(42)

    for batch_size, q_len, kv_len, head_dim in [
        (2, 37, 51, 64),
        (4, 128, 128, 128),
        (1, 32, 32, 1024),
    ]:
        print(f"\n--- B={batch_size}, Q={q_len}, KV={kv_len}, D={head_dim} ---")

        q = torch.randn(batch_size, q_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(batch_size, kv_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)

        # --- reference (PyTorch) ---
        ref_out = q @ k.transpose(-1, -2)
        ref_out.sum().backward()
        ref_dq, ref_dk = q.grad.clone(), k.grad.clone()

        q.grad, k.grad = None, None

        # --- triton ---
        tri_out = triton_qk_matmul(q, k)
        tri_out.sum().backward()
        tri_dq, tri_dk = q.grad.clone(), k.grad.clone()

        # --- compare ---
        print("  Forward:")
        print(f"    max |diff| = {(tri_out - ref_out).abs().max().item():.2e}")
        assert torch.allclose(tri_out, ref_out, atol=1e-3, rtol=1e-3), "Forward mismatch!"
        print("    PASSED")

        print("  Backward dQ:")
        print(f"    max |diff| = {(tri_dq - ref_dq).abs().max().item():.2e}")
        assert torch.allclose(tri_dq, ref_dq, atol=1e-3, rtol=1e-3), "dQ mismatch!"
        print("    PASSED")

        print("  Backward dK:")
        print(f"    max |diff| = {(tri_dk - ref_dk).abs().max().item():.2e}")
        assert torch.allclose(tri_dk, ref_dk, atol=1e-3, rtol=1e-3), "dK mismatch!"
        print("    PASSED")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_forward_backward()

    batch_size, q_len, kv_len, head_dim = 32, 32, 32, 1024

    q = torch.randn(batch_size, q_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch_size, kv_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)

    output = triton_qk_matmul(q, k)

    q = torch.randn(batch_size, q_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch_size, kv_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)

    import time

    start_time = time.time()
    output1 = triton_qk_matmul(q, k)
    print("Triton time:", time.time() - start_time)
    start_time = time.time()
    output2 = q @ k.transpose(-1, -2)
    print("Native time:", time.time() - start_time)
    print(f"  max |diff|  = {(output1 - output2).abs().max().item():.2e}")
    assert torch.allclose(output1, output2, atol=1e-4, rtol=1e-4)
