"""
Dataloader-side FA2x2 path metadata builder.

Copied out of exp/fa2x2_v2.py (a self-contained piece with no dependency on
the rest of that file) since it's genuinely reusable dataloader/preprocessing
logic — same category as genSegIdx.py / genSegIdx_clip.py in this directory.

build_fa2_path_meta_batched pre-computes, from each sample's boundary_indices /
route masks, everything needed to run one FlashAttention-2 varlen call over
either the R-path or the N-path: which query/key rows belong to that path, each
key's RoPE position, and the cu_seqlens/max_seqlen varlen bookkeeping.
"""

import torch


def _vec_pack(boundary_indices, abs_mask, attn_mask, route_value, device):
    """Vectorized, LOOP-FREE core for one sample's R/N packing.

    The old per-segment Python loop launched a fistful of GPU kernels per
    segment plus a data-dependent `torch.where` (which forces a device sync)
    every iteration — ~0.5 ms/segment/path, so ~1000 segments cost ~0.5 s. This
    is trivial integer indexing; do it as a few whole-tensor ops.

    Relies on the concat key ranges being a contiguous partition
    (ke_i == ks_{i+1}, ke-ks == qe — as the original loop already required, since
    it indexes route[:qe] with attn_mask[ks:ke]). Returns sample-LOCAL
    (q_idx, k_idx, rope_idx, q_lens, k_lens) for the kept segments (segments with
    >= 1 key on this route), or None if there are none. Bit-identical to the old
    loop (verified in debug/prof_build_meta.py).
    """
    n_seg = len(boundary_indices)
    b = torch.tensor(boundary_indices, dtype=torch.long, device=device)   # (n_seg, 4)
    qs_a, qe_a, ks_a, ke_a = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    # route over the WHOLE concat in one shot: attn ? (abs ? 2 : 1) : 0
    route = torch.where(attn_mask, torch.where(abs_mask, 2, 1), 0)
    g = (route == route_value).nonzero(as_tuple=True)[0]        # selected concat positions (sorted)
    seg = torch.searchsorted(ke_a.contiguous(), g, right=True)  # segment each key falls in
    k_idx = g - ks_a[seg]                                       # local key index (== physical, 0-based)
    rope_idx = g                                                # ks + local == global concat position
    k_lens = torch.bincount(seg, minlength=n_seg)

    has = k_lens > 0
    if not bool(has.any()):
        return None
    q_lens = (qe_a - qs_a)[has]
    k_lens = k_lens[has]

    # concatenate arange(qs, qe) over kept segments, loop-free
    total_q = int(q_lens.sum())
    seg_start = torch.cumsum(q_lens, 0) - q_lens
    within = torch.arange(total_q, device=device) - torch.repeat_interleave(seg_start, q_lens)
    q_idx = torch.repeat_interleave(qs_a[has], q_lens) + within

    return q_idx, k_idx.to(torch.long), rope_idx.to(torch.long), q_lens, k_lens


def _cu_from_lens(lens, device, repeat=1):
    lens = lens.to(torch.int32).repeat(repeat)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=device),
                      torch.cumsum(lens, dim=0).to(torch.int32)])


def build_fa2_path_meta_batched(boundary_indices_list, abs_mask_list, attn_mask_list, route_value, seq_len_max, device):
    """
    Batched, heterogeneous-length FA2x2 path-metadata builder.

    Each sample b has its OWN boundary_indices / abs_mask / attn_mask (its
    own real seq_len, its own segmentation — samples are NOT assumed to
    share a routing pattern). Physical storage is (B, seq_len_max, H, DK),
    padded to the batch's longest sample — so sample b's real rows live at
    global offset b*seq_len_max in the flattened (B*seq_len_max, H, DK) view
    every kernel in this pipeline operates on.

    k_idx / q_idx get that global offset baked in (they're gather/scatter
    indices into the flattened physical storage). rope_idx does NOT — RoPE
    position resets at the start of every sample.

    cu_q / cu_k concatenate every segment of every sample in order (sample 0's
    segments, then sample 1's, ...), since different samples have different
    segmentations. FA2 sees sum(len(boundary_indices) for each sample)
    independent "sequences" total.

    Returns a dict of packed indices + cu_seqlens/max_seqlen bookkeeping, or
    None if no sample has any row on this route.
    """
    # One vectorized _vec_pack per sample (B iterations, not B*n_seg): the inner
    # per-segment GPU-op loop is gone. Global offset (b*seq_len_max) is added to
    # the sample-local q_idx/k_idx; rope_idx stays sample-local (RoPE resets).
    q_idx_parts, k_idx_parts, rope_idx_parts = [], [], []
    q_lens_parts, k_lens_parts = [], []
    for b, (boundary_indices, abs_mask, attn_mask) in enumerate(
        zip(boundary_indices_list, abs_mask_list, attn_mask_list)
    ):
        packed = _vec_pack(boundary_indices, abs_mask, attn_mask, route_value, device)
        if packed is None:
            continue
        q_idx, k_idx, rope_idx, q_lens, k_lens = packed
        offset = b * seq_len_max
        q_idx_parts.append(q_idx + offset)
        k_idx_parts.append(k_idx + offset)
        rope_idx_parts.append(rope_idx)
        q_lens_parts.append(q_lens)
        k_lens_parts.append(k_lens)

    if not q_idx_parts:
        return None

    q_lens = torch.cat(q_lens_parts)
    k_lens = torch.cat(k_lens_parts)
    return {
        "q_idx":    torch.cat(q_idx_parts,    dim=0),
        "k_idx":    torch.cat(k_idx_parts,    dim=0),
        "rope_idx": torch.cat(rope_idx_parts, dim=0),
        "cu_q":     _cu_from_lens(q_lens, device),
        "cu_k":     _cu_from_lens(k_lens, device),
        "max_q":    int(q_lens.max()),
        "max_k":    int(k_lens.max()),
    }


def build_inverse_index(src_idx, num_src, device):
    """
    Dense inverse of a gather index, for a scatter-add backward without
    atomics: for each physical row p in [0, num_src), the set of output rows
    i with src_idx[i] == p (the prep kernel's k_idx / route_code_*_flat is
    exactly this kind of gather index — the same physical key row can be
    read by multiple packed output rows, once per segment that reuses it).

    A backward kernel can then assign ONE warp per physical row, walk that
    row's entries, and accumulate in registers — no atomics, no race,
    because no two warps ever write the same output row.

    This is derived purely from src_idx (no dependence on any layer's
    runtime values), so — like src_idx itself — it belongs with the rest of
    the dataloader-side metadata: computed once per sample, reused by every
    layer's backward, not recomputed per call.

    Returns
    -------
    inv_idx : (num_src, max_mult) int32
        inv_idx[p, m] is the m-th output row that read physical row p, or -1
        for unused slots. max_mult = the largest number of output rows that
        share any single physical row (0 if src_idx is empty).
    """
    num_out = src_idx.shape[0]
    src_idx = src_idx.to(torch.long)

    counts = torch.bincount(src_idx, minlength=num_src)
    max_mult = int(counts.max().item()) if num_out > 0 else 0

    inv_idx = torch.full((num_src, max(max_mult, 1)), -1, dtype=torch.int32, device=device)
    if num_out == 0:
        return inv_idx

    # Stable sort groups equal src_idx values together, preserving original
    # (output-row) order within each group — so position-within-group is a
    # ready-made "slot" index for the dense layout above.
    order = torch.argsort(src_idx, stable=True)
    sorted_src = src_idx[order]

    group_start = torch.zeros(num_src, dtype=torch.long, device=device)
    group_start[1:] = torch.cumsum(counts, dim=0)[:-1]

    slot = torch.arange(num_out, device=device) - group_start[sorted_src]
    inv_idx[sorted_src, slot] = order.to(torch.int32)
    return inv_idx
