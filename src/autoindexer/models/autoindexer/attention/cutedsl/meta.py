"""
Adapter: AutoIndexer ``PositionBlockList`` -> the structural metadata the
FA2x2 CuteDSL forward consumes.

The two codebases already share the same segmentation model — SegAttn's
``gen_boundary_indices`` returns exactly the fields ``PositionBlockList`` exposes
(``boundary_indices``, ``concat_abs_pos_masks``, ``concat_attn_masks``). So the
route metadata is a pure transform of those fields via
``build_fa2_path_meta_batched`` (route 1 = R-path = rotated query, route 2 =
N-path = unrotated/absolute query, route 0 = masked).

One deliberate deviation from SegAttn's own ``build_meta``: SegAttn derives each
key's RoPE position as ``ks + local_k_idx`` (the flat concatenated key index),
because in its data model position == index. AutoIndexer keys carry *real* edit-
chain position ids in ``concat_key_pos_ids`` (insertions/deletions make them
non-contiguous), so we override the RoPE position with
``concat_key_pos_ids[concat_slot]`` — the same values the eager reference feeds
its key rotary embedding.

AutoIndexer shares ONE segmentation across the whole batch, so every sample gets
the same ``boundary_indices``; we replicate B times and treat seq_len_max ==
seq_len (no padding) for the batched builder.
"""

import torch

from autoindexer.models.autoindexer.attention.cutedsl.fa2_path_meta import (
    build_fa2_path_meta_batched,
    build_inverse_index,
    _cu_from_lens,
)


def _build_path(boundary_indices, abs_mask, attn_mask, key_pos_ids, route_value, B, seq_len, device):
    # Build ONE sample's pack, then offset-tile it to B. The batch shares a single
    # segmentation (batch_perturb_labels samples the edit chain once per batch), so
    # build_fa2_path_meta_batched([bi]*B) would recompute identical work B times;
    # this does it once and tiles. Bit-identical to the loop — regression-guarded
    # by tests/test_meta_builder.py.
    one = build_fa2_path_meta_batched(
        [boundary_indices], [abs_mask], [attn_mask], route_value, seq_len, device,
    )
    if one is None:
        return None
    offs = (torch.arange(B, device=device) * seq_len).unsqueeze(1)      # (B, 1)
    q_idx = (one["q_idx"].unsqueeze(0) + offs).reshape(-1)              # per-sample global offset baked in
    k_idx = (one["k_idx"].unsqueeze(0) + offs).reshape(-1)
    rope_idx = one["rope_idx"].repeat(B)                               # sample-local (concat slot), repeated
    # rope_idx is the concat-slot index; map it to the REAL key position id.
    real_key_pos = key_pos_ids[rope_idx].to(torch.int32)
    return {
        "q_idx": q_idx,
        "flat": k_idx.to(torch.int32),
        "key_pos": real_key_pos,
        "inv_idx": build_inverse_index(k_idx, B * seq_len, device),
        "cu_q": _cu_from_lens(one["cu_q"].diff(), device, repeat=B),
        "cu_k": _cu_from_lens(one["cu_k"].diff(), device, repeat=B),
        "max_q": one["max_q"], "max_k": one["max_k"],
        # 1-D aux arrays for the causal mask_mod: token index per packed
        # query row / key col. keep iff k_tok <= q_tok (causal, index space).
        "q_tok": q_idx.to(torch.int32),
        "k_tok": k_idx.to(torch.int32),
    }


def build_cutedsl_meta(parsed_positions, B, seq_len, device):
    """Return the dict of tensors ``cutedsl_fwd`` expects.

    ``parsed_positions`` is a ``PositionBlockList`` (its ``concat()`` has run).
    Raises if the segmentation has no R-path (rotated) keys while it has real
    content — every non-empty segment must have >=1 R key, and a genuine
    violation of that is treated as a hard error rather than silently degraded.
    """
    bi = parsed_positions.boundary_indices
    abs_mask = parsed_positions.concat_abs_pos_masks
    attn_mask = parsed_positions.concat_attn_masks
    key_pos_ids = parsed_positions.concat_key_pos_ids

    if attn_mask.numel() == 0:
        raise Exception("Nothing to attend to. Make sure that the sequence length is non-zero (i.e. EOS side by side)")

    meta_R = _build_path(bi, abs_mask, attn_mask, key_pos_ids, 1, B, seq_len, device)
    meta_N = _build_path(bi, abs_mask, attn_mask, key_pos_ids, 2, B, seq_len, device)
    if meta_R is None:
        raise ValueError(
            "cutedsl: segmentation has no R-path (rotated) keys — every segment "
            "must have >=1 R key; got a degenerate/empty segmentation."
        )

    # query RoPE positions (real ids), padded to the physical seq_len then tiled
    # over the batch (shared segmentation => identical per sample). Rows beyond
    # the block structure get position 0 and are never gathered into a path.
    q_pos = parsed_positions.query_pos_ids
    assert q_pos.numel() <= seq_len, f"cutedsl: query_pos_ids ({q_pos.numel()}) longer than seq_len ({seq_len})"
    if q_pos.numel() < seq_len:
        q_pos = torch.nn.functional.pad(q_pos, (0, seq_len - q_pos.numel()))
    query_pos_tiled = q_pos.to(torch.int32).repeat(B)

    return {"R": meta_R, "N": meta_N, "query_pos": query_pos_tiled}
