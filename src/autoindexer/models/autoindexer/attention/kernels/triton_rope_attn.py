"""
Triton forward and backward kernels for fused segment RoPE attention.

Key/query RoPE ``key_cos``/``key_sin`` and ``query_cos``/``query_sin`` are precomputed in
``compute_attn_weights_over_segments``. Query RoPE is applied inline in the forward and
backward dK kernels from ``query_rotary_pos_emb``.
``compute_attn_weights_over_segments_triton`` delegates to ``MultiSegmentRoPEAttn`` /
``multi_segment_rope_attn`` so the same fused forward and backward path is used for
training and inference.

Per segment:
    attn = Q_rope @ K_rope^T + (Q - Q_rope) @ K_rope^T * abs_pos_mask
with attn_mask (False -> -inf). Key RoPE is applied inline in Triton.

Backward: tiled dQ/dQ_rope and dK kernels. Key RoPE backward is folded into the dK
kernel via decomposed atomic scatters (no concat ``dK_rope`` staging buffer). Host-side
schedule construction only; reuse ``MultiSegmentLaunchCache`` /
``PositionBlockList.get_triton_launch_cache`` to avoid rebuilding bounds, owner maps,
and tile schedules every layer.

Tests/benchmarks that check these kernels against the eager reference implementations in
``attention/reference.py`` live under ``attention/tests/`` (see
``test_triton_rope_attn.py``), rather than alongside the kernels themselves.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# tl.dot requires M, N, K >= 16 on typical GPU backends.
_MIN_DOT = 16


def head_dim_is_power_of_2(head_dim: int) -> bool:
    return head_dim > 0 and (head_dim & (head_dim - 1)) == 0


def _block_sizes_fwd(q_len: int, kv_len: int, head_dim: int):
    block_q = max(_MIN_DOT, min(32, triton.next_power_of_2(max(q_len, 1))))
    block_kv = max(_MIN_DOT, min(32, triton.next_power_of_2(max(kv_len, 1))))
    block_d = max(_MIN_DOT, min(32, triton.next_power_of_2(head_dim)))
    return block_q, block_kv, block_d


def _block_sizes_bwd(q_len: int, kv_len: int, head_dim: int):
    block_q = max(_MIN_DOT, min(16, triton.next_power_of_2(max(q_len, 1))))
    block_kv = max(_MIN_DOT, min(16, triton.next_power_of_2(max(kv_len, 1))))
    block_d = max(_MIN_DOT, min(32, triton.next_power_of_2(head_dim)))
    return block_q, block_kv, block_d


def _fused_block_sizes(q_len: int, kv_len: int, head_dim: int):
    """Block sizes for the fully-fused attention kernels.

    Unlike ``_block_sizes_fwd``/``_block_sizes_bwd`` (which chunk head_dim into BLOCK_D-sized
    pieces, so only a (block, BLOCK_D) slice of Q/K is ever resident at once), the fused
    kernels load/RoPE-rotate/matmul Q, K, and V over the *entire* head_dim per tile (see
    ``_multi_segment_fused_attn_fwd_kernel``). A (BLOCK_Q or BLOCK_KV, head_dim) tile's
    shared-memory footprint therefore scales directly with head_dim, so BLOCK_Q/BLOCK_KV are
    shrunk as head_dim grows to keep that footprint roughly on par with the old (block=32) x
    (BLOCK_D=32) tiling and avoid ``OutOfResources: shared memory`` for larger head dims (e.g.
    128/256).
    """
    tile_budget = 32 * 32  # matches the pre-fusion (block=32) x (block_d=32) tile area
    max_side = max(_MIN_DOT, triton.next_power_of_2(max(tile_budget // head_dim, 1)))
    block_q = max(_MIN_DOT, min(32, max_side, triton.next_power_of_2(max(q_len, 1))))
    block_kv = max(_MIN_DOT, min(32, max_side, triton.next_power_of_2(max(kv_len, 1))))
    return block_q, block_kv


def _prepare_masks(abs_pos_mask: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        abs_pos_mask.to(torch.float32).contiguous(),
        attn_mask.to(torch.float32).contiguous(),
    )


def _prepare_rotary_emb(
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    emb_cos, emb_sin = rotary_pos_emb
    return emb_cos.squeeze(0).contiguous(), emb_sin.squeeze(0).contiguous()


def _apply_rope_backward(d_rope: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    """Backprop through Llama RoPE; rope_cos/rope_sin: (seq, D)."""
    rope_cos = rope_cos.unsqueeze(0)
    rope_sin = rope_sin.unsqueeze(0)
    half = d_rope.shape[-1] // 2
    d_feats = d_rope * rope_cos
    d_feats[..., :half] = d_feats[..., :half] + d_rope[..., half:] * rope_sin[..., half:]
    d_feats[..., half:] = d_feats[..., half:] - d_rope[..., :half] * rope_sin[..., half:]
    return d_feats


def _build_segment_owner_map(boundary_indices, seq_len: int, device: torch.device) -> torch.Tensor:
    owner = torch.full((seq_len, seq_len), -1, dtype=torch.int32, device=device)
    for seg_id, (start_idx, end_idx, _key_start_idx, _key_end_idx) in enumerate(boundary_indices):
        owner[start_idx:end_idx, :end_idx] = seg_id
    return owner


def _build_segment_id_map(boundary_indices, seq_len: int, device: torch.device) -> torch.Tensor:
    """Compact ``(seq_len,)`` counterpart to ``_build_segment_owner_map`` for the fully-fused
    kernels (``_multi_segment_fused_attn_fwd_kernel`` / ``_multi_segment_fused_attn_bwd_*``),
    which -- per those kernels' docstrings -- only ever need the owning segment of each query
    row's *diagonal* cell (i.e. ``owner[q, q]``), not the full ``(seq_len, seq_len)`` owner
    matrix that the non-fused kernels require. Avoiding that O(seq_len^2) allocation for the
    fused-only path is a real memory reduction, not just a cosmetic one.
    """
    segment_id = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    for seg_id, (start_idx, end_idx, _key_start_idx, _key_end_idx) in enumerate(boundary_indices):
        segment_id[start_idx:end_idx] = seg_id
    return segment_id


def _prepare_bounds_tensor(boundary_indices, device: torch.device) -> torch.Tensor:
    return torch.tensor(boundary_indices, dtype=torch.int32, device=device)


def _build_tile_schedule(
    boundary_indices,
    block_q: int,
    block_kv: int,
    device: torch.device,
) -> torch.Tensor:
    schedule: list[list[int]] = []
    for seg_id, (start_idx, end_idx, _key_start_idx, _key_end_idx) in enumerate(boundary_indices):
        q_seg = end_idx - start_idx
        kv_seg = end_idx
        for pid_q in range(triton.cdiv(q_seg, block_q)):
            for pid_kv in range(triton.cdiv(kv_seg, block_kv)):
                schedule.append([seg_id, pid_q, pid_kv])
    if not schedule:
        return torch.empty((0, 3), dtype=torch.int32, device=device)
    return torch.tensor(schedule, dtype=torch.int32, device=device)


def _build_q_tile_schedule(
    boundary_indices,
    block_q: int,
    device: torch.device,
) -> torch.Tensor:
    """Schedule for the fused attention forward kernel: one entry per (segment, query tile).

    Unlike ``_build_tile_schedule``, there is no ``pid_kv`` dimension: the fused forward
    kernel loops over the full causal key range internally (online-softmax), since it must
    reduce over keys sequentially rather than write independent output cells.
    """
    schedule: list[list[int]] = []
    for seg_id, (start_idx, end_idx, _key_start_idx, _key_end_idx) in enumerate(boundary_indices):
        q_seg = end_idx - start_idx
        for pid_q in range(triton.cdiv(q_seg, block_q)):
            schedule.append([seg_id, pid_q])
    if not schedule:
        return torch.empty((0, 2), dtype=torch.int32, device=device)
    return torch.tensor(schedule, dtype=torch.int32, device=device)


@triton.jit
def _multi_segment_attn_fwd_kernel(
    Q_ptr,
    Q_cos_ptr,
    Q_sin_ptr,
    K_ptr,
    K_cos_ptr,
    K_sin_ptr,
    Abs_mask_ptr,
    Attn_mask_ptr,
    Out_ptr,
    Owner_ptr,
    Bounds_ptr,
    Schedule_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_qcos_q,
    stride_qcos_d,
    stride_qsin_q,
    stride_qsin_d,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_kcos_k,
    stride_kcos_d,
    stride_ksin_k,
    stride_ksin_d,
    stride_owner_q,
    stride_owner_k,
    stride_ob,
    stride_oq,
    stride_ok,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_bh = tl.program_id(1)

    seg_id = tl.load(Schedule_ptr + tile_id * 3 + 0)
    pid_q = tl.load(Schedule_ptr + tile_id * 3 + 1)
    pid_kv = tl.load(Schedule_ptr + tile_id * 3 + 2)

    start_idx = tl.load(Bounds_ptr + seg_id * 4 + 0)
    end_idx = tl.load(Bounds_ptr + seg_id * 4 + 1)
    key_start_idx = tl.load(Bounds_ptr + seg_id * 4 + 2)

    q_seg = end_idx - start_idx
    kv_seg = end_idx

    q_offset = pid_q * BLOCK_Q
    kv_offset = pid_kv * BLOCK_KV

    q_range = q_offset + tl.arange(0, BLOCK_Q)
    kv_range = kv_offset + tl.arange(0, BLOCK_KV)

    q_global = start_idx + q_range
    key_concat = key_start_idx + kv_range

    owner_ptrs = Owner_ptr + q_global[:, None] * stride_owner_q + kv_range[None, :] * stride_owner_k
    owner_mask = (q_range[:, None] < q_seg) & (kv_range[None, :] < kv_seg)
    owner = tl.load(owner_ptrs, mask=owner_mask, other=-1)
    owned = owner == seg_id

    abs_mask = tl.load(Abs_mask_ptr + key_concat, mask=kv_range < kv_seg, other=0.0)
    attn_mask = tl.load(Attn_mask_ptr + key_concat, mask=kv_range < kv_seg, other=0.0)

    acc = tl.zeros((BLOCK_Q, BLOCK_KV), dtype=tl.float32)
    half = head_dim // 2

    for d_start in range(0, head_dim, BLOCK_D):
        d_range = d_start + tl.arange(0, BLOCK_D)
        partner_d = d_range + tl.where(d_range < half, half, -half)
        rot_sign = tl.where(d_range < half, -1.0, 1.0)

        q_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + d_range[None, :] * stride_qd
        q_mask = (q_range[:, None] < q_seg) & (d_range[None, :] < head_dim)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        q_partner_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + partner_d[None, :] * stride_qd
        q_partner_mask = q_mask & (partner_d[None, :] >= 0) & (partner_d[None, :] < head_dim)
        q_partner = tl.load(q_partner_ptrs, mask=q_partner_mask, other=0.0)

        q_cos_ptrs = Q_cos_ptr + q_global[:, None] * stride_qcos_q + d_range[None, :] * stride_qcos_d
        q_sin_ptrs = Q_sin_ptr + q_global[:, None] * stride_qsin_q + d_range[None, :] * stride_qsin_d
        q_cos_block = tl.load(q_cos_ptrs, mask=q_mask, other=0.0)
        q_sin_block = tl.load(q_sin_ptrs, mask=q_mask, other=0.0)
        q_rope_block = q_block * q_cos_block + rot_sign[None, :] * q_partner * q_sin_block

        k_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + d_range[None, :] * stride_kd
        k_mask = (kv_range[:, None] < kv_seg) & (d_range[None, :] < head_dim)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        k_partner_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + partner_d[None, :] * stride_kd
        k_partner_mask = k_mask & (partner_d[None, :] >= 0) & (partner_d[None, :] < head_dim)
        k_partner = tl.load(k_partner_ptrs, mask=k_partner_mask, other=0.0)

        k_cos_ptrs = K_cos_ptr + key_concat[:, None] * stride_kcos_k + d_range[None, :] * stride_kcos_d
        k_sin_ptrs = K_sin_ptr + key_concat[:, None] * stride_ksin_k + d_range[None, :] * stride_ksin_d
        k_cos_block = tl.load(k_cos_ptrs, mask=k_mask, other=0.0)
        k_sin_block = tl.load(k_sin_ptrs, mask=k_mask, other=0.0)

        k_rope_block = k_block * k_cos_block + rot_sign[None, :] * k_partner * k_sin_block
        k_rope_t = tl.trans(k_rope_block)
        q_diff = q_block - q_rope_block

        acc += tl.dot(q_rope_block, k_rope_t, input_precision="ieee")
        acc += tl.dot(q_diff, k_rope_t, input_precision="ieee") * abs_mask[None, :]

    out = tl.where(attn_mask[None, :] > 0, acc, -float("inf"))
    out_ptrs = Out_ptr + pid_bh * stride_ob + q_global[:, None] * stride_oq + kv_range[None, :] * stride_ok
    out_mask = owner_mask & owned
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _multi_segment_attn_bwd_dq_kernel(
    dOut_ptr,
    K_ptr,
    K_cos_ptr,
    K_sin_ptr,
    Abs_mask_ptr,
    Attn_mask_ptr,
    Owner_ptr,
    Bounds_ptr,
    Schedule_ptr,
    dQ_ptr,
    dQ_rope_ptr,
    stride_dob,
    stride_doq,
    stride_dok,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_kcos_k,
    stride_kcos_d,
    stride_ksin_k,
    stride_ksin_d,
    stride_owner_q,
    stride_owner_k,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_dqrb,
    stride_dqrq,
    stride_dqrd,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_bh = tl.program_id(1)

    seg_id = tl.load(Schedule_ptr + tile_id * 3 + 0)
    pid_q = tl.load(Schedule_ptr + tile_id * 3 + 1)
    pid_kv = tl.load(Schedule_ptr + tile_id * 3 + 2)

    start_idx = tl.load(Bounds_ptr + seg_id * 4 + 0)
    end_idx = tl.load(Bounds_ptr + seg_id * 4 + 1)
    key_start_idx = tl.load(Bounds_ptr + seg_id * 4 + 2)

    q_seg = end_idx - start_idx
    kv_seg = end_idx

    q_offset = pid_q * BLOCK_Q
    kv_offset = pid_kv * BLOCK_KV

    q_range = q_offset + tl.arange(0, BLOCK_Q)
    kv_range = kv_offset + tl.arange(0, BLOCK_KV)

    q_global = start_idx + q_range
    key_concat = key_start_idx + kv_range

    owner_ptrs = Owner_ptr + q_global[:, None] * stride_owner_q + kv_range[None, :] * stride_owner_k
    owner_mask = (q_range[:, None] < q_seg) & (kv_range[None, :] < kv_seg)
    owner = tl.load(owner_ptrs, mask=owner_mask, other=-1)
    owned = owner == seg_id

    do_ptrs = dOut_ptr + pid_bh * stride_dob + q_global[:, None] * stride_doq + kv_range[None, :] * stride_dok
    do_block = tl.load(do_ptrs, mask=owner_mask, other=0.0)

    attn_mask = tl.load(Attn_mask_ptr + key_concat, mask=kv_range < kv_seg, other=0.0)
    do_block = tl.where((attn_mask[None, :] > 0) & owned, do_block, 0.0)

    abs_mask = tl.load(Abs_mask_ptr + key_concat, mask=kv_range < kv_seg, other=0.0)
    half = head_dim // 2
    do_abs = do_block * abs_mask[None, :]
    do_q_rope = do_block * (1.0 - abs_mask[None, :])

    for d_start in range(0, head_dim, BLOCK_D):
        d_range = d_start + tl.arange(0, BLOCK_D)
        partner_d = d_range + tl.where(d_range < half, half, -half)
        rot_sign = tl.where(d_range < half, -1.0, 1.0)

        acc_q = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
        acc_q_rope = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        k_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + d_range[None, :] * stride_kd
        k_mask = (kv_range[:, None] < kv_seg) & (d_range[None, :] < head_dim)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        k_partner_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + partner_d[None, :] * stride_kd
        k_partner_mask = k_mask & (partner_d[None, :] >= 0) & (partner_d[None, :] < head_dim)
        k_partner = tl.load(k_partner_ptrs, mask=k_partner_mask, other=0.0)

        k_cos_ptrs = K_cos_ptr + key_concat[:, None] * stride_kcos_k + d_range[None, :] * stride_kcos_d
        k_sin_ptrs = K_sin_ptr + key_concat[:, None] * stride_ksin_k + d_range[None, :] * stride_ksin_d
        k_cos_block = tl.load(k_cos_ptrs, mask=k_mask, other=0.0)
        k_sin_block = tl.load(k_sin_ptrs, mask=k_mask, other=0.0)

        k_rope_block = k_block * k_cos_block + rot_sign[None, :] * k_partner * k_sin_block

        acc_q += tl.dot(do_abs.to(k_rope_block.dtype), k_rope_block, input_precision="ieee")
        acc_q_rope += tl.dot(do_q_rope.to(k_rope_block.dtype), k_rope_block, input_precision="ieee")

        dq_mask = (q_range[:, None] < q_seg) & (d_range[None, :] < head_dim)
        dq_ptrs = dQ_ptr + pid_bh * stride_dqb + q_global[:, None] * stride_dqq + d_range[None, :] * stride_dqd
        dqr_ptrs = dQ_rope_ptr + pid_bh * stride_dqrb + q_global[:, None] * stride_dqrq + d_range[None, :] * stride_dqrd
        tl.atomic_add(dq_ptrs, acc_q, mask=dq_mask)
        tl.atomic_add(dqr_ptrs, acc_q_rope, mask=dq_mask)


@triton.jit
def _multi_segment_attn_bwd_dk_kernel(
    dOut_ptr,
    Q_ptr,
    Q_cos_ptr,
    Q_sin_ptr,
    K_cos_ptr,
    K_sin_ptr,
    Abs_mask_ptr,
    Attn_mask_ptr,
    Owner_ptr,
    Bounds_ptr,
    Schedule_ptr,
    dK_ptr,
    stride_dob,
    stride_doq,
    stride_dok,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_qcos_q,
    stride_qcos_d,
    stride_qsin_q,
    stride_qsin_d,
    stride_kcos_k,
    stride_kcos_d,
    stride_ksin_k,
    stride_ksin_d,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_owner_q,
    stride_owner_k,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_bh = tl.program_id(1)

    seg_id = tl.load(Schedule_ptr + tile_id * 3 + 0)
    pid_q = tl.load(Schedule_ptr + tile_id * 3 + 1)
    pid_kv = tl.load(Schedule_ptr + tile_id * 3 + 2)

    start_idx = tl.load(Bounds_ptr + seg_id * 4 + 0)
    end_idx = tl.load(Bounds_ptr + seg_id * 4 + 1)
    key_start_idx = tl.load(Bounds_ptr + seg_id * 4 + 2)

    q_seg = end_idx - start_idx
    kv_seg = end_idx

    q_offset = pid_q * BLOCK_Q
    kv_offset = pid_kv * BLOCK_KV

    q_range = q_offset + tl.arange(0, BLOCK_Q)
    kv_range = kv_offset + tl.arange(0, BLOCK_KV)

    q_global = start_idx + q_range
    key_concat = key_start_idx + kv_range

    owner_ptrs = Owner_ptr + q_global[:, None] * stride_owner_q + kv_range[None, :] * stride_owner_k
    owner_mask = (q_range[:, None] < q_seg) & (kv_range[None, :] < kv_seg)
    owner = tl.load(owner_ptrs, mask=owner_mask, other=-1)
    owned = owner == seg_id

    do_ptrs = dOut_ptr + pid_bh * stride_dob + q_global[:, None] * stride_doq + kv_range[None, :] * stride_dok
    do_block = tl.load(do_ptrs, mask=owner_mask, other=0.0)

    attn_mask = tl.load(Attn_mask_ptr + key_concat, mask=kv_range < kv_seg, other=0.0)
    do_block = tl.where((attn_mask[None, :] > 0) & owned, do_block, 0.0)

    abs_mask = tl.load(Abs_mask_ptr + key_concat, mask=kv_range < kv_seg, other=0.0)
    half = head_dim // 2
    abs_kv = abs_mask[:, None]
    one_minus_abs_kv = 1.0 - abs_kv
    do_t = tl.trans(do_block)

    for d_start in range(0, head_dim, BLOCK_D):
        d_range = d_start + tl.arange(0, BLOCK_D)
        partner_d = d_range + tl.where(d_range < half, half, -half)
        rot_sign = tl.where(d_range < half, -1.0, 1.0)

        q_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + d_range[None, :] * stride_qd
        q_mask = (q_range[:, None] < q_seg) & (d_range[None, :] < head_dim)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        q_partner_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + partner_d[None, :] * stride_qd
        q_partner_mask = q_mask & (partner_d[None, :] >= 0) & (partner_d[None, :] < head_dim)
        q_partner = tl.load(q_partner_ptrs, mask=q_partner_mask, other=0.0)

        q_cos_ptrs = Q_cos_ptr + q_global[:, None] * stride_qcos_q + d_range[None, :] * stride_qcos_d
        q_sin_ptrs = Q_sin_ptr + q_global[:, None] * stride_qsin_q + d_range[None, :] * stride_qsin_d
        q_cos_block = tl.load(q_cos_ptrs, mask=q_mask, other=0.0)
        q_sin_block = tl.load(q_sin_ptrs, mask=q_mask, other=0.0)
        q_rope_block = q_block * q_cos_block + rot_sign[None, :] * q_partner * q_sin_block

        acc_dk_rope = tl.dot((do_t * abs_kv).to(q_block.dtype), q_block, input_precision="ieee")
        acc_dk_rope += tl.dot((do_t * one_minus_abs_kv).to(q_rope_block.dtype), q_rope_block, input_precision="ieee")

        kv_d_mask = (kv_range[:, None] < kv_seg) & (d_range[None, :] < head_dim)
        k_cos_ptrs = K_cos_ptr + key_concat[:, None] * stride_kcos_k + d_range[None, :] * stride_kcos_d
        k_sin_ptrs = K_sin_ptr + key_concat[:, None] * stride_ksin_k + d_range[None, :] * stride_ksin_d
        k_cos_block = tl.load(k_cos_ptrs, mask=kv_d_mask, other=0.0)
        k_sin_block = tl.load(k_sin_ptrs, mask=kv_d_mask, other=0.0)

        dk_ptrs = dK_ptr + pid_bh * stride_dkb + kv_range[:, None] * stride_dkk + d_range[None, :] * stride_dkd
        tl.atomic_add(dk_ptrs, acc_dk_rope * k_cos_block, mask=kv_d_mask)

        partner_mask = kv_d_mask & (partner_d[None, :] >= 0) & (partner_d[None, :] < head_dim)
        dk_partner_ptrs = dK_ptr + pid_bh * stride_dkb + kv_range[:, None] * stride_dkk + partner_d[None, :] * stride_dkd
        lower_mask = partner_mask & (d_range[None, :] < half)
        upper_mask = partner_mask & (d_range[None, :] >= half)
        tl.atomic_add(dk_partner_ptrs, -acc_dk_rope * k_sin_block, mask=lower_mask)
        tl.atomic_add(dk_partner_ptrs, acc_dk_rope * k_sin_block, mask=upper_mask)


@triton.jit
def _multi_segment_fused_attn_fwd_kernel(
    Q_ptr,
    Q_cos_ptr,
    Q_sin_ptr,
    K_ptr,
    K_cos_ptr,
    K_sin_ptr,
    V_ptr,
    Abs_mask_ptr,
    Attn_mask_ptr,
    ExtMask_ptr,
    SegmentId_ptr,
    Bounds_ptr,
    Schedule_ptr,
    Out_ptr,
    Lse_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_qcos_q,
    stride_qcos_d,
    stride_qsin_q,
    stride_qsin_d,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_kcos_k,
    stride_kcos_d,
    stride_ksin_k,
    stride_ksin_d,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_segid_q,
    stride_extb,
    stride_exth,
    stride_extq,
    stride_extk,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_od,
    stride_lseb,
    stride_lseq,
    seq_len,
    num_heads,
    seed,
    dropout_p,
    scaling,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_EXT_MASK: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
):
    """Fused forward: softmax(mask(Q_rope @ K_rope^T)) @ V via online-softmax, without ever
    materializing the full (q_len, kv_len) attention matrix.

    Assumes each query row is owned by exactly one segment across its *entire* causal key
    range (true for ``boundary_indices`` built by ``PositionBlockList.concat()``, which is
    what ``autoindexer_attention_forward`` always uses -- query ranges partition the sequence without
    overlap). This is checked defensively via ``SegmentId_ptr`` on the diagonal cell (a
    compact ``(seq_len,)`` array rather than the full ``(seq_len, seq_len)`` owner matrix the
    non-fused kernels need -- see ``_build_segment_id_map``), but unlike
    ``_multi_segment_attn_fwd_kernel`` it does not support a query row whose causal range is
    split across multiple owning segments.

    Q's RoPE terms depend only on the query tile, not on the KV tile being scanned below, so
    they are loaded/rotated exactly once here (over the full head_dim, like V/Out already are)
    and reused for every KV tile. An earlier version of this kernel BLOCK_D-chunked and
    recomputed both Q's and K's RoPE terms from scratch on every KV tile, in an attempt to keep
    the peak per-tile size down for large head_dim; in practice this multiplied Q's load/rotate
    cost by ``kv_seg / BLOCK_KV`` for a register-pressure benefit that did not pay for itself at
    this kernel's head_dim/block sizes, making it a net slowdown -- so both Q and K are now
    loaded/rotated over the full head_dim in one shot per tile (``BLOCK_D`` is unused, kept only
    for call-site compatibility with the non-fused kernels' launch helpers).

    The KV loop is additionally capped at the causal frontier of the query tile (any KV block
    strictly beyond the largest query position in this tile is entirely masked out anyway), so
    such tiles are skipped without ever loading/rotating K or V for them.
    """
    tile_id = tl.program_id(0)
    pid_bh = tl.program_id(1)

    seg_id = tl.load(Schedule_ptr + tile_id * 2 + 0)
    pid_q = tl.load(Schedule_ptr + tile_id * 2 + 1)

    start_idx = tl.load(Bounds_ptr + seg_id * 4 + 0)
    end_idx = tl.load(Bounds_ptr + seg_id * 4 + 1)
    key_start_idx = tl.load(Bounds_ptr + seg_id * 4 + 2)

    q_seg = end_idx - start_idx
    kv_seg = end_idx

    q_offset = pid_q * BLOCK_Q
    q_range = q_offset + tl.arange(0, BLOCK_Q)
    q_global = start_idx + q_range
    q_valid = q_range < q_seg

    segid_ptrs = SegmentId_ptr + q_global * stride_segid_q
    segid = tl.load(segid_ptrs, mask=q_valid, other=-1)
    row_owned = q_valid & (segid == seg_id)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    half = head_dim // 2
    d_full = tl.arange(0, head_dim)
    partner_d_full = d_full + tl.where(d_full < half, half, -half)
    rot_sign_full = tl.where(d_full < half, -1.0, 1.0)

    q_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + d_full[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    q_partner_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + partner_d_full[None, :] * stride_qd
    q_partner = tl.load(q_partner_ptrs, mask=q_valid[:, None], other=0.0)

    q_cos_ptrs = Q_cos_ptr + q_global[:, None] * stride_qcos_q + d_full[None, :] * stride_qcos_d
    q_sin_ptrs = Q_sin_ptr + q_global[:, None] * stride_qsin_q + d_full[None, :] * stride_qsin_d
    q_cos_block = tl.load(q_cos_ptrs, mask=q_valid[:, None], other=0.0)
    q_sin_block = tl.load(q_sin_ptrs, mask=q_valid[:, None], other=0.0)

    q_rope = q_block * q_cos_block + rot_sign_full[None, :] * q_partner * q_sin_block
    q_diff = q_block - q_rope

    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    # Any KV tile starting past the largest query position owned by this Q tile is entirely
    # non-causal (fully masked below), so it can be skipped outright.
    q_max = start_idx + q_offset + BLOCK_Q - 1
    kv_seg_causal = tl.minimum(kv_seg, q_max + 1)

    for kv_start in range(0, kv_seg_causal, BLOCK_KV):
        kv_range = kv_start + tl.arange(0, BLOCK_KV)
        kv_valid = kv_range < kv_seg
        key_concat = key_start_idx + kv_range

        abs_mask_col = tl.load(Abs_mask_ptr + key_concat, mask=kv_valid, other=0.0)

        k_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + d_full[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=kv_valid[:, None], other=0.0)

        k_partner_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + partner_d_full[None, :] * stride_kd
        k_partner = tl.load(k_partner_ptrs, mask=kv_valid[:, None], other=0.0)

        k_cos_ptrs = K_cos_ptr + key_concat[:, None] * stride_kcos_k + d_full[None, :] * stride_kcos_d
        k_sin_ptrs = K_sin_ptr + key_concat[:, None] * stride_ksin_k + d_full[None, :] * stride_ksin_d
        k_cos_block = tl.load(k_cos_ptrs, mask=kv_valid[:, None], other=0.0)
        k_sin_block = tl.load(k_sin_ptrs, mask=kv_valid[:, None], other=0.0)

        k_rope = k_block * k_cos_block + rot_sign_full[None, :] * k_partner * k_sin_block
        k_rope_t = tl.trans(k_rope)

        s = tl.dot(q_rope, k_rope_t, input_precision="ieee")
        s += tl.dot(q_diff, k_rope_t, input_precision="ieee") * abs_mask_col[None, :]
        s = s * scaling

        attn_mask_col = tl.load(Attn_mask_ptr + key_concat, mask=kv_valid, other=0.0)
        causal_valid = kv_range[None, :] <= q_global[:, None]
        valid = kv_valid[None, :] & causal_valid & (attn_mask_col[None, :] > 0)

        if HAS_EXT_MASK:
            ext_ptrs = ExtMask_ptr + batch_idx * stride_extb + head_idx * stride_exth + q_global[:, None] * stride_extq + kv_range[None, :] * stride_extk
            ext_bias = tl.load(ext_ptrs, mask=valid, other=0.0)
            s = s + ext_bias

        s_masked = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s_masked, axis=1))
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(valid, tl.exp(s_masked - m_new[:, None]), 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + pid_bh * stride_vb + kv_range[:, None] * stride_vk + d_full[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)

        if USE_DROPOUT:
            offsets = (pid_bh * seq_len + q_global)[:, None] * seq_len + kv_range[None, :]
            keep = tl.rand(seed, offsets) >= dropout_p
            p_dropped = tl.where(keep, p * (1.0 / (1.0 - dropout_p)), 0.0)
            acc += tl.dot(p_dropped.to(v_block.dtype), v_block, input_precision="ieee")
        else:
            acc += tl.dot(p.to(v_block.dtype), v_block, input_precision="ieee")

        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = m_i + tl.log(l_safe)

    # Written directly into a (batch, seq_len, num_heads, head_dim)-laid-out buffer (via
    # separate batch/head strides) instead of the (batch*heads, seq_len, head_dim) layout Q/K/V
    # use, so callers get the ``(batch, seq, heads, head_dim)`` shape they need without a
    # separate transpose().contiguous() pass over the output afterward.
    out_ptrs = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh + q_global[:, None] * stride_oq + d_full[None, :] * stride_od
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=(q_valid & row_owned)[:, None])

    lse_ptrs = Lse_ptr + pid_bh * stride_lseb + q_global * stride_lseq
    tl.store(lse_ptrs, lse, mask=q_valid & row_owned)


@triton.jit
def _multi_segment_fused_attn_bwd_kernel(
    dOut_ptr,
    Q_ptr,
    Q_cos_ptr,
    Q_sin_ptr,
    K_ptr,
    K_cos_ptr,
    K_sin_ptr,
    V_ptr,
    Lse_ptr,
    D_ptr,
    Abs_mask_ptr,
    Attn_mask_ptr,
    ExtMask_ptr,
    SegmentId_ptr,
    Bounds_ptr,
    Schedule_ptr,
    dQ_ptr,
    dQ_rope_ptr,
    dK_ptr,
    dV_ptr,
    stride_dob,
    stride_doh,
    stride_doq,
    stride_dod,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_qcos_q,
    stride_qcos_d,
    stride_qsin_q,
    stride_qsin_d,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_kcos_k,
    stride_kcos_d,
    stride_ksin_k,
    stride_ksin_d,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_lseb,
    stride_lseq,
    stride_db,
    stride_dh,
    stride_dq,
    stride_segid_q,
    stride_extb,
    stride_exth,
    stride_extq,
    stride_extk,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_dqrb,
    stride_dqrq,
    stride_dqrd,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    seq_len,
    num_heads,
    seed,
    dropout_p,
    scaling,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_EXT_MASK: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
):
    """Backward for ``_multi_segment_fused_attn_fwd_kernel``.

    Recomputes the (unmaterialized) attention logits/probabilities tile-by-tile from saved
    ``Lse`` (row log-sum-exp, computed without dropout) and ``D`` (row sum of ``dOut * Out``,
    precomputed on the host -- cheap, O(seq_len) work), then propagates gradients into
    dQ/dQ_rope, dK, and dV (all scattered atomically, by query or key position) using the same
    tile schedule as ``_multi_segment_attn_bwd_dq_kernel`` / ``_multi_segment_attn_bwd_dk_kernel``.

    A single kernel invocation recomputes RoPE and S_ij/P_ij once per (q, kv) tile and reuses
    them for all of dQ/dQ_rope/dK/dV. An earlier version of this kernel split dQ/dQ_rope/dK and
    dV into two separate kernels (trading redundant RoPE/S_ij recompute for fewer simultaneous
    atomic-scatter destinations and lower per-kernel register pressure); in practice the
    redundant recompute cost more than the split saved, making it a net slowdown, so both
    gradients are computed together here again.

    Ownership uses the same diagonal-only ``SegmentId_ptr`` check as the fused forward kernel
    (see ``_build_segment_id_map``), so this must only be used with the same
    ``boundary_indices`` / assumptions (see the forward kernel's docstring).
    """
    tile_id = tl.program_id(0)
    pid_bh = tl.program_id(1)

    seg_id = tl.load(Schedule_ptr + tile_id * 3 + 0)
    pid_q = tl.load(Schedule_ptr + tile_id * 3 + 1)
    pid_kv = tl.load(Schedule_ptr + tile_id * 3 + 2)

    start_idx = tl.load(Bounds_ptr + seg_id * 4 + 0)
    end_idx = tl.load(Bounds_ptr + seg_id * 4 + 1)
    key_start_idx = tl.load(Bounds_ptr + seg_id * 4 + 2)

    q_seg = end_idx - start_idx
    kv_seg = end_idx

    q_offset = pid_q * BLOCK_Q
    kv_offset = pid_kv * BLOCK_KV

    q_range = q_offset + tl.arange(0, BLOCK_Q)
    kv_range = kv_offset + tl.arange(0, BLOCK_KV)

    q_global = start_idx + q_range
    key_concat = key_start_idx + kv_range

    q_valid = q_range < q_seg
    kv_valid = kv_range < kv_seg

    segid_ptrs = SegmentId_ptr + q_global * stride_segid_q
    segid = tl.load(segid_ptrs, mask=q_valid, other=-1)
    row_owned = q_valid & (segid == seg_id)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    half = head_dim // 2
    d_full = tl.arange(0, head_dim)
    partner_d_full = d_full + tl.where(d_full < half, half, -half)
    rot_sign_full = tl.where(d_full < half, -1.0, 1.0)

    abs_mask_col = tl.load(Abs_mask_ptr + key_concat, mask=kv_valid, other=0.0)
    attn_mask_col = tl.load(Attn_mask_ptr + key_concat, mask=kv_valid, other=0.0)
    causal_valid = kv_range[None, :] <= q_global[:, None]
    valid = row_owned[:, None] & kv_valid[None, :] & causal_valid & (attn_mask_col[None, :] > 0)

    # RoPE for Q and K over the full head_dim, computed once and reused both for recomputing
    # S_ij below and for the dQ/dK gradient formulas further down (see docstring).
    q_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + d_full[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    q_partner_ptrs = Q_ptr + pid_bh * stride_qb + q_global[:, None] * stride_qq + partner_d_full[None, :] * stride_qd
    q_partner = tl.load(q_partner_ptrs, mask=q_valid[:, None], other=0.0)

    q_cos_ptrs = Q_cos_ptr + q_global[:, None] * stride_qcos_q + d_full[None, :] * stride_qcos_d
    q_sin_ptrs = Q_sin_ptr + q_global[:, None] * stride_qsin_q + d_full[None, :] * stride_qsin_d
    q_cos_block = tl.load(q_cos_ptrs, mask=q_valid[:, None], other=0.0)
    q_sin_block = tl.load(q_sin_ptrs, mask=q_valid[:, None], other=0.0)
    q_rope_block = q_block * q_cos_block + rot_sign_full[None, :] * q_partner * q_sin_block
    q_diff = q_block - q_rope_block

    k_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + d_full[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=kv_valid[:, None], other=0.0)

    k_partner_ptrs = K_ptr + pid_bh * stride_kb + kv_range[:, None] * stride_kk + partner_d_full[None, :] * stride_kd
    k_partner = tl.load(k_partner_ptrs, mask=kv_valid[:, None], other=0.0)

    k_cos_ptrs = K_cos_ptr + key_concat[:, None] * stride_kcos_k + d_full[None, :] * stride_kcos_d
    k_sin_ptrs = K_sin_ptr + key_concat[:, None] * stride_ksin_k + d_full[None, :] * stride_ksin_d
    k_cos_block = tl.load(k_cos_ptrs, mask=kv_valid[:, None], other=0.0)
    k_sin_block = tl.load(k_sin_ptrs, mask=kv_valid[:, None], other=0.0)
    k_rope_block = k_block * k_cos_block + rot_sign_full[None, :] * k_partner * k_sin_block
    k_rope_t = tl.trans(k_rope_block)

    s = tl.dot(q_rope_block, k_rope_t, input_precision="ieee")
    s += tl.dot(q_diff, k_rope_t, input_precision="ieee") * abs_mask_col[None, :]
    s = s * scaling
    if HAS_EXT_MASK:
        ext_ptrs = ExtMask_ptr + batch_idx * stride_extb + head_idx * stride_exth + q_global[:, None] * stride_extq + kv_range[None, :] * stride_extk
        ext_bias = tl.load(ext_ptrs, mask=valid, other=0.0)
        s = s + ext_bias

    lse = tl.load(Lse_ptr + pid_bh * stride_lseb + q_global * stride_lseq, mask=q_valid, other=0.0)
    p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)

    # --- dS_ij via softmax(+dropout) backward ---
    do_ptrs = dOut_ptr + batch_idx * stride_dob + head_idx * stride_doh + q_global[:, None] * stride_doq + d_full[None, :] * stride_dod
    do_block = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0)

    v_ptrs = V_ptr + pid_bh * stride_vb + kv_range[:, None] * stride_vk + d_full[None, :] * stride_vd
    v_block = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)

    dot_do_v = tl.dot(do_block, tl.trans(v_block), input_precision="ieee")

    d_i = tl.load(D_ptr + batch_idx * stride_db + head_idx * stride_dh + q_global * stride_dq, mask=q_valid, other=0.0)

    if USE_DROPOUT:
        offsets = (pid_bh * seq_len + q_global)[:, None] * seq_len + kv_range[None, :]
        keep = tl.rand(seed, offsets) >= dropout_p
        drop_scale = 1.0 / (1.0 - dropout_p)
        dp_softmax = tl.where(valid & keep, dot_do_v * drop_scale, 0.0)
        p_used = tl.where(valid & keep, p * drop_scale, 0.0)
    else:
        dp_softmax = tl.where(valid, dot_do_v, 0.0)
        p_used = p

    d_raw = p * (dp_softmax - d_i[:, None]) * scaling
    d_raw = tl.where(valid, d_raw, 0.0)

    # --- dV: scatter by key position ---
    dv_block = tl.dot(tl.trans(p_used).to(do_block.dtype), do_block, input_precision="ieee")
    dv_ptrs = dV_ptr + pid_bh * stride_dvb + kv_range[:, None] * stride_dvk + d_full[None, :] * stride_dvd
    tl.atomic_add(dv_ptrs, dv_block, mask=kv_valid[:, None])

    # --- dQ/dQ_rope and dK via the same RoPE-decomposed linear formulas used by the
    # non-fused backward kernels, with ``d_raw`` (grad wrt the raw, pre-scaling dot product)
    # playing the role of their ``dOut``. Reuses q_block/q_rope_block/k_rope_block/
    # k_cos_block/k_sin_block computed above instead of recomputing them a second time. ---
    abs_kv = abs_mask_col[:, None]
    one_minus_abs_kv = 1.0 - abs_kv
    d_raw_t = tl.trans(d_raw)
    do_abs = d_raw * abs_mask_col[None, :]
    do_q_rope = d_raw * (1.0 - abs_mask_col[None, :])

    # dQ / dQ_rope (mirrors _multi_segment_attn_bwd_dq_kernel, with d_raw as "dOut").
    # do_abs/do_q_rope are fp32 (derived from d_raw); cast to k_rope_block's dtype first.
    acc_q = tl.dot(do_abs.to(k_rope_block.dtype), k_rope_block, input_precision="ieee")
    acc_q_rope = tl.dot(do_q_rope.to(k_rope_block.dtype), k_rope_block, input_precision="ieee")

    dq_mask = q_valid[:, None]
    dq_ptrs = dQ_ptr + pid_bh * stride_dqb + q_global[:, None] * stride_dqq + d_full[None, :] * stride_dqd
    dqr_ptrs = dQ_rope_ptr + pid_bh * stride_dqrb + q_global[:, None] * stride_dqrq + d_full[None, :] * stride_dqrd
    tl.atomic_add(dq_ptrs, acc_q, mask=dq_mask)
    tl.atomic_add(dqr_ptrs, acc_q_rope, mask=dq_mask)

    # dK (mirrors _multi_segment_attn_bwd_dk_kernel, with d_raw as "dOut"). Same
    acc_dk_rope = tl.dot((d_raw_t * abs_kv).to(q_block.dtype), q_block, input_precision="ieee")
    acc_dk_rope += tl.dot((d_raw_t * one_minus_abs_kv).to(q_rope_block.dtype), q_rope_block, input_precision="ieee")

    kv_d_mask = kv_valid[:, None]
    dk_ptrs = dK_ptr + pid_bh * stride_dkb + kv_range[:, None] * stride_dkk + d_full[None, :] * stride_dkd
    tl.atomic_add(dk_ptrs, acc_dk_rope * k_cos_block, mask=kv_d_mask)

    dk_partner_ptrs = dK_ptr + pid_bh * stride_dkb + kv_range[:, None] * stride_dkk + partner_d_full[None, :] * stride_dkd
    lower_mask = kv_d_mask & (d_full[None, :] < half)
    upper_mask = kv_d_mask & (d_full[None, :] >= half)
    tl.atomic_add(dk_partner_ptrs, -acc_dk_rope * k_sin_block, mask=lower_mask)
    tl.atomic_add(dk_partner_ptrs, acc_dk_rope * k_sin_block, mask=upper_mask)


def _prepare_multi_segment_launch(
    boundary_indices,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    *,
    backward: bool = False,
):
    if not boundary_indices:
        return None

    max_q_seg = max(end_idx - start_idx for start_idx, end_idx, _, _ in boundary_indices)
    max_kv_seg = max(end_idx for _, end_idx, _, _ in boundary_indices)
    block_sizes = _block_sizes_bwd if backward else _block_sizes_fwd
    block_q, block_kv, block_d = block_sizes(max_q_seg, max_kv_seg, head_dim)

    bounds = _prepare_bounds_tensor(boundary_indices, device)
    owner = _build_segment_owner_map(boundary_indices, seq_len, device)
    schedule = _build_tile_schedule(boundary_indices, block_q, block_kv, device)
    return bounds, owner, schedule, block_q, block_kv, block_d


@dataclass
class MultiSegmentLaunchCache:
    """Precomputed Triton launch metadata for a fixed ``boundary_indices`` / shape."""

    bounds: torch.Tensor
    owner: torch.Tensor
    fwd_schedule: torch.Tensor
    bwd_schedule: torch.Tensor
    fwd_block_q: int
    fwd_block_kv: int
    fwd_block_d: int
    bwd_block_q: int
    bwd_block_kv: int
    bwd_block_d: int
    seq_len: int
    head_dim: int
    # Fused (softmax + dropout + attn @ V) forward schedule: one tile per (segment, query
    # tile), since the fused kernel reduces over keys internally via online-softmax instead
    # of writing independent (q, kv) output cells. The fused backward kernel (dQ/dQ_rope/dK/dV
    # together) reuses ``bwd_schedule``/``bwd_block_*`` above -- it still decomposes into
    # independent (q, kv) tiles that atomically scatter into dQ/dK/dV, same as the non-fused
    # backward kernels.
    fused_fwd_schedule: torch.Tensor = None
    fused_fwd_block_q: int = None
    fused_fwd_block_kv: int = None
    fused_fwd_block_d: int = None
    # Compact (seq_len,) counterpart to ``owner`` (see ``_build_segment_id_map``) used by the
    # fully-fused kernels, which only ever need the owning segment of each query row's diagonal
    # cell rather than the full (seq_len, seq_len) owner matrix the non-fused kernels need.
    fused_segment_id: torch.Tensor = None

    @property
    def owned(self) -> torch.Tensor:
        return self.owner >= 0


def build_multi_segment_launch_cache(
    boundary_indices,
    seq_len: int,
    head_dim: int,
    device: torch.device,
) -> MultiSegmentLaunchCache:
    """Build bounds, owner map, and tile schedules once for repeated kernel launches."""
    bounds, owner, fwd_schedule, fwd_block_q, fwd_block_kv, fwd_block_d = _prepare_multi_segment_launch(
        boundary_indices,
        seq_len,
        head_dim,
        device,
        backward=False,
    )
    _, _, bwd_schedule, bwd_block_q, bwd_block_kv, bwd_block_d = _prepare_multi_segment_launch(
        boundary_indices,
        seq_len,
        head_dim,
        device,
        backward=True,
    )
    fused_fwd_block_q, fused_fwd_block_kv = _fused_block_sizes(seq_len, seq_len, head_dim)
    fused_fwd_block_d = max(_MIN_DOT, min(32, triton.next_power_of_2(head_dim)))
    fused_fwd_schedule = _build_q_tile_schedule(boundary_indices, fused_fwd_block_q, device)
    fused_segment_id = _build_segment_id_map(boundary_indices, seq_len, device)
    return MultiSegmentLaunchCache(
        bounds=bounds,
        owner=owner,
        fwd_schedule=fwd_schedule,
        bwd_schedule=bwd_schedule,
        fwd_block_q=fwd_block_q,
        fwd_block_kv=fwd_block_kv,
        fwd_block_d=fwd_block_d,
        bwd_block_q=bwd_block_q,
        bwd_block_kv=bwd_block_kv,
        bwd_block_d=bwd_block_d,
        seq_len=seq_len,
        head_dim=head_dim,
        fused_fwd_schedule=fused_fwd_schedule,
        fused_fwd_block_q=fused_fwd_block_q,
        fused_fwd_block_kv=fused_fwd_block_kv,
        fused_fwd_block_d=fused_fwd_block_d,
        fused_segment_id=fused_segment_id,
    )


def _launch_multi_segment_attn_fwd(
    query: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    keys: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    abs_pos_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    attn_weights: torch.Tensor,
    owner: torch.Tensor,
    bounds: torch.Tensor,
    schedule: torch.Tensor,
    block_q: int,
    block_kv: int,
    block_d: int,
) -> None:
    batch_heads = query.shape[0]
    head_dim = query.shape[2]
    num_tiles = schedule.shape[0]
    if num_tiles == 0:
        return

    grid = (num_tiles, batch_heads)
    _multi_segment_attn_fwd_kernel[grid](
        query,
        query_cos,
        query_sin,
        keys,
        key_cos,
        key_sin,
        abs_pos_mask,
        attn_mask,
        attn_weights,
        owner,
        bounds,
        schedule,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_cos.stride(0),
        query_cos.stride(1),
        query_sin.stride(0),
        query_sin.stride(1),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        key_cos.stride(0),
        key_cos.stride(1),
        key_sin.stride(0),
        key_sin.stride(1),
        owner.stride(0),
        owner.stride(1),
        attn_weights.stride(0),
        attn_weights.stride(1),
        attn_weights.stride(2),
        head_dim,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        BLOCK_D=block_d,
    )


def _launch_multi_segment_attn_bwd(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    keys: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    abs_pos_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    owner: torch.Tensor,
    bounds: torch.Tensor,
    schedule: torch.Tensor,
    block_q: int,
    block_kv: int,
    block_d: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_heads, seq_len, head_dim = query.shape

    dq = torch.zeros(batch_heads, seq_len, head_dim, device=query.device, dtype=torch.float32)
    dq_rope = torch.zeros(batch_heads, seq_len, head_dim, device=query.device, dtype=torch.float32)
    dk = torch.zeros(batch_heads, seq_len, head_dim, device=query.device, dtype=torch.float32)

    num_tiles = schedule.shape[0]
    if num_tiles == 0:
        return dq, dq_rope, dk

    grid = (num_tiles, batch_heads)
    _multi_segment_attn_bwd_dq_kernel[grid](
        grad_output,
        keys,
        key_cos,
        key_sin,
        abs_pos_mask,
        attn_mask,
        owner,
        bounds,
        schedule,
        dq,
        dq_rope,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        key_cos.stride(0),
        key_cos.stride(1),
        key_sin.stride(0),
        key_sin.stride(1),
        owner.stride(0),
        owner.stride(1),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq_rope.stride(0),
        dq_rope.stride(1),
        dq_rope.stride(2),
        head_dim,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        BLOCK_D=block_d,
    )

    _multi_segment_attn_bwd_dk_kernel[grid](
        grad_output,
        query,
        query_cos,
        query_sin,
        key_cos,
        key_sin,
        abs_pos_mask,
        attn_mask,
        owner,
        bounds,
        schedule,
        dk,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_cos.stride(0),
        query_cos.stride(1),
        query_sin.stride(0),
        query_sin.stride(1),
        key_cos.stride(0),
        key_cos.stride(1),
        key_sin.stride(0),
        key_sin.stride(1),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        owner.stride(0),
        owner.stride(1),
        head_dim,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        BLOCK_D=block_d,
    )

    return dq, dq_rope, dk


def _reshape_batch_heads(
    query: torch.Tensor,
    keys: torch.Tensor,
    batch_heads: int,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_bh = query.reshape(batch_heads, seq_len, head_dim)
    keys_bh = keys.reshape(batch_heads, seq_len, head_dim)
    if not query_bh.is_contiguous():
        query_bh = query_bh.contiguous()
    if not keys_bh.is_contiguous():
        keys_bh = keys_bh.contiguous()
    return query_bh, keys_bh


class MultiSegmentRoPEAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        query_cos: torch.Tensor,
        query_sin: torch.Tensor,
        keys: torch.Tensor,
        key_cos: torch.Tensor,
        key_sin: torch.Tensor,
        abs_pos_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        boundary_indices,
        launch_cache: MultiSegmentLaunchCache | None,
    ) -> torch.Tensor:
        assert boundary_indices, "No boundary_indices provided."

        seq_len = query.shape[1]
        head_dim = query.shape[2]
        device = query.device
        dtype = query.dtype
        batch_heads = query.shape[0]

        query = query.contiguous()
        query_cos = query_cos.contiguous()
        query_sin = query_sin.contiguous()
        keys = keys.contiguous()
        key_cos = key_cos.contiguous()
        key_sin = key_sin.contiguous()
        abs_pos_mask, attn_mask = _prepare_masks(abs_pos_mask, attn_mask)

        if launch_cache is not None:
            assert launch_cache.seq_len == seq_len and launch_cache.head_dim == head_dim
            bounds = launch_cache.bounds
            owner = launch_cache.owner
            schedule = launch_cache.fwd_schedule
            block_q, block_kv, block_d = (
                launch_cache.fwd_block_q,
                launch_cache.fwd_block_kv,
                launch_cache.fwd_block_d,
            )
        else:
            bounds, owner, schedule, block_q, block_kv, block_d = _prepare_multi_segment_launch(
                boundary_indices,
                seq_len,
                head_dim,
                device,
                backward=False,
            )

        out = torch.full((batch_heads, seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        out.diagonal(dim1=-2, dim2=-1).fill_(0)

        _launch_multi_segment_attn_fwd(
            query,
            query_cos,
            query_sin,
            keys,
            key_cos,
            key_sin,
            abs_pos_mask,
            attn_mask,
            out,
            owner,
            bounds,
            schedule,
            block_q,
            block_kv,
            block_d,
        )

        ctx.save_for_backward(query, query_cos, query_sin, keys, key_cos, key_sin, abs_pos_mask, attn_mask, bounds, owner)
        ctx.launch_cache = launch_cache
        ctx.boundary_indices = boundary_indices
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, query_cos, query_sin, keys, key_cos, key_sin, abs_pos_mask, attn_mask, bounds, owner = ctx.saved_tensors
        seq_len = query.shape[1]
        head_dim = query.shape[2]
        launch_cache = ctx.launch_cache

        if launch_cache is not None:
            assert launch_cache.seq_len == seq_len and launch_cache.head_dim == head_dim
            schedule = launch_cache.bwd_schedule
            block_q, block_kv, block_d = (
                launch_cache.bwd_block_q,
                launch_cache.bwd_block_kv,
                launch_cache.bwd_block_d,
            )
        else:
            _, _, schedule, block_q, block_kv, block_d = _prepare_multi_segment_launch(
                ctx.boundary_indices,
                seq_len,
                head_dim,
                query.device,
                backward=True,
            )

        grad_output = grad_output.contiguous()

        dq, dq_rope, dk = _launch_multi_segment_attn_bwd(
            grad_output,
            query,
            query_cos,
            query_sin,
            keys,
            key_cos,
            key_sin,
            abs_pos_mask,
            attn_mask,
            owner,
            bounds,
            schedule,
            block_q,
            block_kv,
            block_d,
        )
        dq = dq + _apply_rope_backward(dq_rope, query_cos, query_sin)

        return (
            dq.to(query.dtype),
            None,
            None,
            dk.to(keys.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def multi_segment_rope_attn(
    query: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    keys: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    abs_pos_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    boundary_indices,
    launch_cache: MultiSegmentLaunchCache | None = None,
) -> torch.Tensor:
    """Fused multi-segment RoPE attention over a full (seq_len, seq_len) matrix."""
    return MultiSegmentRoPEAttn.apply(
        query,
        query_cos,
        query_sin,
        keys,
        key_cos,
        key_sin,
        abs_pos_mask,
        attn_mask,
        boundary_indices,
        launch_cache,
    )


def compute_attn_weights_over_segments_triton(
    query: torch.Tensor,
    query_rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    keys: torch.Tensor,
    concat_key_rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    concat_abs_pos_mask: torch.Tensor,
    concat_attn_mask: torch.Tensor,
    boundary_indices,
    launch_cache: MultiSegmentLaunchCache | None = None,
) -> torch.Tensor:
    """Compute segment attention logits via ``MultiSegmentRoPEAttn`` (supports autograd)."""
    batch_size, num_heads, seq_len, head_dim = query.shape
    batch_heads = batch_size * num_heads

    assert boundary_indices, "No boundary_indices provided."

    query_cos, query_sin = _prepare_rotary_emb(query_rotary_pos_emb)
    concat_key_cos, concat_key_sin = _prepare_rotary_emb(concat_key_rotary_pos_emb)
    abs_pos_mask, attn_mask = _prepare_masks(concat_abs_pos_mask, concat_attn_mask)

    query_bh, keys_bh = _reshape_batch_heads(
        query,
        keys,
        batch_heads,
        seq_len,
        head_dim,
    )

    return multi_segment_rope_attn(
        query_bh,
        query_cos,
        query_sin,
        keys_bh,
        concat_key_cos,
        concat_key_sin,
        abs_pos_mask,
        attn_mask,
        boundary_indices,
        launch_cache,
    ).view(batch_size, num_heads, seq_len, seq_len)


def _prepare_ext_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Normalize an optional HF-style attention mask into an additive float32 bias.

    Does not change the mask's shape/broadcast layout (e.g. ``(batch, 1, q_len, kv_len)``),
    so it is not the N-by-N-per-head allocation the fusion is meant to avoid -- it is at most
    the size of whatever mask the caller already passed in.
    """
    if attention_mask is None:
        return None
    if attention_mask.dtype == torch.bool:
        bias = torch.zeros(attention_mask.shape, dtype=torch.float32, device=attention_mask.device)
        bias.masked_fill_(~attention_mask, float("-inf"))
        return bias
    return attention_mask.to(torch.float32)


def _ext_mask_strides(ext_mask: torch.Tensor | None) -> tuple[int, int, int, int]:
    """Strides for an optional ``(batch, heads_or_1, q_len_or_1, kv_len)`` additive mask.

    Broadcast dims (size 1, e.g. a pure padding mask shaped ``(batch, 1, 1, kv_len)``) get
    stride 0 so the kernel can index them directly, without the mask ever being
    expanded/materialized to the full ``(batch, num_heads, q_len, kv_len)`` shape.
    """
    if ext_mask is None:
        return 0, 0, 0, 0
    assert ext_mask.ndim == 4
    return tuple(ext_mask.stride(dim) if ext_mask.shape[dim] != 1 else 0 for dim in range(4))


def _launch_multi_segment_fused_attn_fwd(
    query: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    keys: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    values: torch.Tensor,
    abs_pos_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    ext_mask: torch.Tensor | None,
    num_heads: int,
    segment_id: torch.Tensor,
    bounds: torch.Tensor,
    schedule: torch.Tensor,
    block_q: int,
    block_kv: int,
    block_d: int,
    seed: int,
    dropout_p: float,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_heads, seq_len, head_dim = query.shape
    batch_size = batch_heads // num_heads
    # Written directly in (batch, seq_len, num_heads, head_dim) layout (rather than
    # (batch*heads, seq_len, head_dim), like Q/K/V) so that callers get their desired
    # (batch, seq, heads, head_dim) output without a separate transpose().contiguous() pass.
    out = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=query.device, dtype=query.dtype)
    lse = torch.full((batch_heads, seq_len), float("-inf"), device=query.device, dtype=torch.float32)

    num_tiles = schedule.shape[0]
    if num_tiles == 0:
        return out, lse

    has_ext_mask = ext_mask is not None
    ext_stride_b, ext_stride_h, ext_stride_q, ext_stride_k = _ext_mask_strides(ext_mask)
    ext_ptr = ext_mask if has_ext_mask else query

    grid = (num_tiles, batch_heads)
    _multi_segment_fused_attn_fwd_kernel[grid](
        query,
        query_cos,
        query_sin,
        keys,
        key_cos,
        key_sin,
        values,
        abs_pos_mask,
        attn_mask,
        ext_ptr,
        segment_id,
        bounds,
        schedule,
        out,
        lse,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_cos.stride(0),
        query_cos.stride(1),
        query_sin.stride(0),
        query_sin.stride(1),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        key_cos.stride(0),
        key_cos.stride(1),
        key_sin.stride(0),
        key_sin.stride(1),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        segment_id.stride(0),
        ext_stride_b,
        ext_stride_h,
        ext_stride_q,
        ext_stride_k,
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        seq_len,
        num_heads,
        seed,
        dropout_p,
        scaling,
        head_dim,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        BLOCK_D=block_d,
        HAS_EXT_MASK=has_ext_mask,
        USE_DROPOUT=dropout_p > 0.0,
        # Q/K/V are staged over the full head_dim per tile here (unlike the non-fused
        # kernels' BLOCK_D-chunked tiles), so cap pipelining depth to keep the software
        # pipeliner's double/triple-buffered shared-memory usage in budget (see
        # ``_fused_block_sizes``, which already shrinks BLOCK_Q/BLOCK_KV for the same reason).
        num_stages=2,
    )
    return out, lse


def _launch_multi_segment_fused_attn_bwd(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    keys: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    values: torch.Tensor,
    lse: torch.Tensor,
    d_row: torch.Tensor,
    abs_pos_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    ext_mask: torch.Tensor | None,
    num_heads: int,
    segment_id: torch.Tensor,
    bounds: torch.Tensor,
    schedule: torch.Tensor,
    block_q: int,
    block_kv: int,
    block_d: int,
    seed: int,
    dropout_p: float,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launches ``_multi_segment_fused_attn_bwd_kernel`` (dQ/dQ_rope/dK/dV together).
    ``grad_output`` and ``d_row`` are addressed in (batch, seq_len, num_heads[, head_dim])
    layout (matching the fused forward's output layout), while Q/K/V and the
    dq/dq_rope/dk/dv accumulators stay in (batch*heads, seq_len, head_dim) layout, same as
    the forward pass.
    """
    batch_heads, seq_len, head_dim = query.shape
    batch_size = batch_heads // num_heads

    dq = torch.zeros(batch_heads, seq_len, head_dim, device=query.device, dtype=torch.float32)
    dq_rope = torch.zeros(batch_heads, seq_len, head_dim, device=query.device, dtype=torch.float32)
    dk = torch.zeros(batch_heads, seq_len, head_dim, device=query.device, dtype=torch.float32)
    dv = torch.zeros(batch_heads, seq_len, head_dim, device=query.device, dtype=torch.float32)

    num_tiles = schedule.shape[0]
    if num_tiles == 0:
        return dq, dq_rope, dk, dv

    has_ext_mask = ext_mask is not None
    ext_stride_b, ext_stride_h, ext_stride_q, ext_stride_k = _ext_mask_strides(ext_mask)
    ext_ptr = ext_mask if has_ext_mask else query

    assert grad_output.shape == (batch_size, seq_len, num_heads, head_dim)
    assert d_row.shape == (batch_size, seq_len, num_heads)

    grid = (num_tiles, batch_heads)
    _multi_segment_fused_attn_bwd_kernel[grid](
        grad_output,
        query,
        query_cos,
        query_sin,
        keys,
        key_cos,
        key_sin,
        values,
        lse,
        d_row,
        abs_pos_mask,
        attn_mask,
        ext_ptr,
        segment_id,
        bounds,
        schedule,
        dq,
        dq_rope,
        dk,
        dv,
        grad_output.stride(0),
        grad_output.stride(2),
        grad_output.stride(1),
        grad_output.stride(3),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_cos.stride(0),
        query_cos.stride(1),
        query_sin.stride(0),
        query_sin.stride(1),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        key_cos.stride(0),
        key_cos.stride(1),
        key_sin.stride(0),
        key_sin.stride(1),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        lse.stride(0),
        lse.stride(1),
        d_row.stride(0),
        d_row.stride(2),
        d_row.stride(1),
        segment_id.stride(0),
        ext_stride_b,
        ext_stride_h,
        ext_stride_q,
        ext_stride_k,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq_rope.stride(0),
        dq_rope.stride(1),
        dq_rope.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        seq_len,
        num_heads,
        seed,
        dropout_p,
        scaling,
        head_dim,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        BLOCK_D=block_d,
        HAS_EXT_MASK=has_ext_mask,
        USE_DROPOUT=dropout_p > 0.0,
        # See the matching comment in _launch_multi_segment_fused_attn_fwd.
        num_stages=2,
    )
    return dq, dq_rope, dk, dv


class FusedMultiSegmentRoPEAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        query_cos: torch.Tensor,
        query_sin: torch.Tensor,
        keys: torch.Tensor,
        key_cos: torch.Tensor,
        key_sin: torch.Tensor,
        values: torch.Tensor,
        abs_pos_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        ext_mask: torch.Tensor | None,
        num_heads: int,
        boundary_indices,
        dropout_p: float,
        scaling: float,
        launch_cache: MultiSegmentLaunchCache | None,
    ) -> torch.Tensor:
        assert boundary_indices, "No boundary_indices provided."
        assert head_dim_is_power_of_2(query.shape[2]), (
            f"fused_multi_segment_rope_attn requires a power-of-2 head_dim (it loads/stores V "
            f"and the output over the full head_dim in one tile), got {query.shape[2]}"
        )

        # `dropout_p` is already 0.0 when not training (see callers, e.g. `attn_layers.py`'s
        # `attention_interface`), so there is no separate `training` flag to resolve here --
        # unlike `nn.functional.dropout(p=dropout_p, training=training)`, which needs one.

        seq_len = query.shape[1]
        head_dim = query.shape[2]
        device = query.device

        query = query.contiguous()
        query_cos = query_cos.contiguous()
        query_sin = query_sin.contiguous()
        keys = keys.contiguous()
        key_cos = key_cos.contiguous()
        key_sin = key_sin.contiguous()
        values = values.contiguous()
        abs_pos_mask, attn_mask = _prepare_masks(abs_pos_mask, attn_mask)

        if launch_cache is not None:
            assert launch_cache.seq_len == seq_len and launch_cache.head_dim == head_dim
            bounds = launch_cache.bounds
            segment_id = launch_cache.fused_segment_id
            schedule = launch_cache.fused_fwd_schedule
            block_q, block_kv, block_d = (
                launch_cache.fused_fwd_block_q,
                launch_cache.fused_fwd_block_kv,
                launch_cache.fused_fwd_block_d,
            )
        else:
            # Deliberately avoids ``_prepare_multi_segment_launch`` here: that helper builds
            # the full (seq_len, seq_len) owner matrix, which this fused path never needs (see
            # ``_build_segment_id_map``).
            bounds = _prepare_bounds_tensor(boundary_indices, device)
            segment_id = _build_segment_id_map(boundary_indices, seq_len, device)
            block_q, block_kv = _fused_block_sizes(seq_len, seq_len, head_dim)
            block_d = max(_MIN_DOT, min(32, triton.next_power_of_2(head_dim)))
            schedule = _build_q_tile_schedule(boundary_indices, block_q, device)

        # A fresh seed per forward call keeps dropout i.i.d. across steps while staying
        # reproducible within this call's backward (which recomputes the same mask via the
        # same seed + per-element offset).
        seed = int(torch.randint(0, 2**31 - 1, (1,), device=device).item()) if dropout_p > 0.0 else 0

        out, lse = _launch_multi_segment_fused_attn_fwd(
            query,
            query_cos,
            query_sin,
            keys,
            key_cos,
            key_sin,
            values,
            abs_pos_mask,
            attn_mask,
            ext_mask,
            num_heads,
            segment_id,
            bounds,
            schedule,
            block_q,
            block_kv,
            block_d,
            seed,
            dropout_p,
            scaling,
        )

        ctx.save_for_backward(query, query_cos, query_sin, keys, key_cos, key_sin, values, abs_pos_mask, attn_mask, bounds, segment_id, out, lse)
        # ext_mask is not a differentiable input (no gradient is computed for it, matching
        # how the eager path treats attention_mask), so it is stashed as a plain ctx
        # attribute rather than via save_for_backward (which expects real tensors, not None).
        ctx.ext_mask = ext_mask
        ctx.launch_cache = launch_cache
        ctx.boundary_indices = boundary_indices
        ctx.num_heads = num_heads
        ctx.dropout_p = dropout_p
        ctx.scaling = scaling
        ctx.seed = seed
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            query,
            query_cos,
            query_sin,
            keys,
            key_cos,
            key_sin,
            values,
            abs_pos_mask,
            attn_mask,
            bounds,
            segment_id,
            out,
            lse,
        ) = ctx.saved_tensors
        ext_mask = ctx.ext_mask
        seq_len = query.shape[1]
        head_dim = query.shape[2]
        launch_cache = ctx.launch_cache

        if launch_cache is not None:
            assert launch_cache.seq_len == seq_len and launch_cache.head_dim == head_dim
            schedule = launch_cache.bwd_schedule
            block_q, block_kv, block_d = (
                launch_cache.bwd_block_q,
                launch_cache.bwd_block_kv,
                launch_cache.bwd_block_d,
            )
        else:
            _, _, schedule, block_q, block_kv, block_d = _prepare_multi_segment_launch(
                ctx.boundary_indices,
                seq_len,
                head_dim,
                query.device,
                backward=True,
            )

        grad_output = grad_output.contiguous()
        # grad_output/out are (batch, seq_len, num_heads, head_dim) (see the fused forward
        # kernel's Out_ptr addressing), so this reduction naturally comes out as (batch,
        # seq_len, num_heads) -- matching layout, no permute/contiguous needed.
        d_row = (grad_output.to(torch.float32) * out.to(torch.float32)).sum(dim=-1)

        dq, dq_rope, dk, dv = _launch_multi_segment_fused_attn_bwd(
            grad_output,
            query,
            query_cos,
            query_sin,
            keys,
            key_cos,
            key_sin,
            values,
            lse,
            d_row,
            abs_pos_mask,
            attn_mask,
            ext_mask,
            ctx.num_heads,
            segment_id,
            bounds,
            schedule,
            block_q,
            block_kv,
            block_d,
            ctx.seed,
            ctx.dropout_p,
            ctx.scaling,
        )
        dq = dq + _apply_rope_backward(dq_rope, query_cos, query_sin)

        return (
            dq.to(query.dtype),
            None,
            None,
            dk.to(keys.dtype),
            None,
            None,
            dv.to(values.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_multi_segment_rope_attn(
    query: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    keys: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    values: torch.Tensor,
    abs_pos_mask: torch.Tensor,
    attn_mask: torch.Tensor,
    ext_mask: torch.Tensor | None,
    num_heads: int,
    boundary_indices,
    dropout_p: float = 0.0,
    scaling: float | None = None,
    launch_cache: MultiSegmentLaunchCache | None = None,
) -> torch.Tensor:
    """Fused multi-segment RoPE attention: softmax(mask(Q@K^T))@V without materializing the
    full (seq_len, seq_len) attention matrix.

    Callers are expected to already resolve `dropout_p` to `0.0` when not training (e.g. as
    `nn.functional.dropout(p=dropout_p, training=training)` would), since this function has no
    separate `training` flag of its own.
    """
    head_dim = query.shape[-1]
    if scaling is None:
        scaling = head_dim**-0.5
    return FusedMultiSegmentRoPEAttn.apply(
        query,
        query_cos,
        query_sin,
        keys,
        key_cos,
        key_sin,
        values,
        abs_pos_mask,
        attn_mask,
        ext_mask,
        num_heads,
        boundary_indices,
        dropout_p,
        scaling,
        launch_cache,
    )


def compute_fused_attn_output_over_segments_triton(
    query: torch.Tensor,
    query_rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    keys: torch.Tensor,
    values: torch.Tensor,
    concat_key_rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    concat_abs_pos_mask: torch.Tensor,
    concat_attn_mask: torch.Tensor,
    boundary_indices,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scaling: float | None = None,
    launch_cache: MultiSegmentLaunchCache | None = None,
) -> torch.Tensor:
    """Compute the fully fused segment-RoPE attention output (batch, seq, heads, head_dim),
    handling causal masking, the external ``attention_mask``, and dropout internally so the
    (seq_len, seq_len) attention-weights matrix is never materialized.

    The output is produced directly in ``(batch, seq, heads, head_dim)`` layout by the Triton
    kernel itself (see ``_multi_segment_fused_attn_fwd_kernel``'s ``Out_ptr`` addressing),
    rather than in the ``(batch, heads, seq, head_dim)`` layout Q/K/V use -- so, unlike a plain
    ``(batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)`` transpose, callers don't
    need a separate ``.transpose(1, 2).contiguous()`` pass over the output afterward.

    The caller is expected to already resolve ``dropout_p`` to ``0.0`` when not training (matching
    the eager ``chain_of_edits_attn_ref`` path, which relies on the same convention rather than a
    separate ``training`` flag).
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    batch_heads = batch_size * num_heads

    assert boundary_indices, "No boundary_indices provided."

    query_cos, query_sin = _prepare_rotary_emb(query_rotary_pos_emb)
    concat_key_cos, concat_key_sin = _prepare_rotary_emb(concat_key_rotary_pos_emb)
    abs_pos_mask, attn_mask = _prepare_masks(concat_abs_pos_mask, concat_attn_mask)
    ext_mask = _prepare_ext_mask(attention_mask)

    query_bh, keys_bh = _reshape_batch_heads(query, keys, batch_heads, seq_len, head_dim)
    values_bh = values.reshape(batch_heads, seq_len, head_dim)
    if not values_bh.is_contiguous():
        values_bh = values_bh.contiguous()

    return fused_multi_segment_rope_attn(
        query_bh,
        query_cos,
        query_sin,
        keys_bh,
        concat_key_cos,
        concat_key_sin,
        values_bh,
        abs_pos_mask,
        attn_mask,
        ext_mask,
        num_heads,
        boundary_indices,
        dropout_p,
        scaling,
        launch_cache,
    )

