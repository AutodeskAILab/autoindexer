"""
CuteDSL RoPE kernel — one warp per (row, head), each lane owning DK//32
contiguous elements with the rotate-half partner fetched via an intra-warp
XOR (butterfly) shuffle, so each row is read/written to global memory once.
The rotation position for row i is read from an explicit `pos` tensor rather
than being implied by i's own index.

Needed because inside a multi-segment attention scheme, the SAME physical
key row can be rotated at DIFFERENT absolute positions depending on which
segment is attending to it (key_start_idx + local offset, not simply "this
row's own index") — see prep_multi_head/fwd_fa2x2_prep.py's PrepPack kernel,
which reads `rope_idx[row_idx]` the same way; this kernel is that same idea,
minus the gather (src_idx), since here the row to rotate is already the row
we want — only the position needs to be arbitrary.

Forward and backward are the same kernel, compiled twice with a different
`direction` constant: RoPE is an orthogonal rotation, so its VJP is the
inverse rotation, i.e. the forward formula with `s` negated (direction=-1.0),
and nothing from the forward pass needs to be saved.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import hashlib
import math

from flash_attn.cute.cache_utils import get_jit_cache

# Folded into the compile key so a source edit invalidates the on-disk cache.
_SRC_FP = hashlib.sha256(open(__file__, "rb").read()).hexdigest()


class RopeKey():
    """
    x / out: (seq_len, H, DK). pos: (seq_len,) int32 — explicit rotation
    position per row (NOT assumed to equal the row's own index). One warp
    handles one (row, head) pair; each lane owns `elems_per_lane` (= DK//32)
    contiguous elements of that row.

    Requires DK to be a multiple of the warp size (32). No batch (B) support.
    """

    def __init__(self, dk, theta, num_warps=16, rows_per_warp=8, direction=1.0, dtype=cutlass.BFloat16):
        assert dk % 32 == 0, "RopeKey requires DK to be a multiple of the warp size (32)"

        self.dk = dk
        self.elems_per_lane = dk // 32
        self.num_warps = num_warps
        self.rows_per_warp = rows_per_warp
        self.rows_per_block = num_warps * rows_per_warp
        self.num_threads = num_warps * 32
        self.log_theta = math.log(theta)
        self.direction = direction
        self.dtype = dtype

    @cute.jit
    def __call__(self, x, pos, out, stream):
        seq_len, num_heads, _ = x.shape
        rows_per_block: cutlass.Constexpr = self.rows_per_block

        num_blocks = (seq_len + rows_per_block - 1) // rows_per_block

        self.kernel(x, pos, out).launch(
            grid=[num_blocks, num_heads, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x, pos, out):
        seq_len, _, _ = x.shape

        elems_per_lane: cutlass.Constexpr = self.elems_per_lane
        num_warps: cutlass.Constexpr = self.num_warps
        rows_per_warp: cutlass.Constexpr = self.rows_per_warp
        rows_per_block: cutlass.Constexpr = self.rows_per_block
        dk: cutlass.Constexpr = self.dk
        log_theta: cutlass.Constexpr = self.log_theta
        direction: cutlass.Constexpr = self.direction
        dtype: cutlass.Constexpr = self.dtype

        bidx, bidy, _ = cute.arch.block_idx()
        head_idx = bidy
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        # Round-robin: `rows_per_warp` rows are processed by this warp, one per round.
        for round_idx in range(rows_per_warp):
            row_idx = bidx * rows_per_block + round_idx * num_warps + warp_idx

            if row_idx < seq_len:
                row_pos = pos[row_idx]

                x_row = x[row_idx, head_idx, None]
                x_row_z = cute.zipped_divide(x_row, (elems_per_lane,))
                x_frag = x_row_z[(None, (lane_idx,))].load()

                out_frag = self._rope(x_frag, lane_idx, row_pos, elems_per_lane, dk, log_theta, direction)

                out_row = out[row_idx, head_idx, None]
                out_row_z = cute.zipped_divide(out_row, (elems_per_lane,))
                out_row_z[(None, (lane_idx,))].store(out_frag.to(dtype))

    @staticmethod
    @cute.jit
    def _rope(
        x_frag,
        lane_idx,
        pos,
        elems_per_lane,
        dk,
        log_theta,
        direction,
    ):
        """RoPE (or its transpose, for direction=-1.0) on a full DK-wide row
        split across a warp; `pos` is passed in explicitly by the caller
        rather than being the row's own index."""
        wsize: cutlass.Constexpr = cute.arch.WARP_SIZE
        lanes_per_half: cutlass.Constexpr = wsize // 2
        half_dim: cutlass.Constexpr = dk // 2

        lane_in_half = lane_idx % lanes_per_half
        freq_base = lane_in_half * elems_per_lane

        sign = cutlass.Float32(-1.0)
        if lane_idx >= lanes_per_half:
            sign = cutlass.Float32(1.0)
        sign = sign * cutlass.Float32(direction)

        pos_f = cutlass.Float32(pos)

        out_regs = cute.make_rmem_tensor(
            cute.make_layout((elems_per_lane,), stride=(1,)),
            cutlass.Float32,
        )

        for i in range(elems_per_lane):
            partner_val = cute.arch.shuffle_sync_bfly(x_frag[i], offset=lanes_per_half)

            freq_idx = freq_base + i

            freq = cute.math.exp(
                -cutlass.Float32(freq_idx) * cutlass.Float32(log_theta) / cutlass.Float32(half_dim)
            )
            angle = pos_f * freq

            c = cute.math.cos(angle)
            s = cute.math.sin(angle)

            x = cutlass.Float32(x_frag[i])
            y = cutlass.Float32(partner_val)

            out_regs[i] = x * c + sign * y * s

        return out_regs.load()


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_rope_key(dk, theta, num_warps=16, rows_per_warp=8, direction=1.0, dtype=cute.BFloat16):
    SEQ_LEN   = cute.sym_int()
    NUM_HEADS = cute.sym_int()

    x   = _fake(dtype, (SEQ_LEN, NUM_HEADS, dk), (2, 1, 0), 16)
    pos = _fake(cute.Int32, (SEQ_LEN,), (0,), 4)
    out = _fake(dtype, (SEQ_LEN, NUM_HEADS, dk), (2, 1, 0), 16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    _rope = RopeKey(dk, theta, num_warps, rows_per_warp, direction, dtype)
    _compiled = cute.compile(
        _rope, x, pos, out, stream,
        options="--enable-tvm-ffi",
    )

    return _rope, _compiled


def rope_key_run(x, pos, out, theta, direction):
    """Launch RopeKey on x/pos -> out, compiling (and caching) the kernel keyed
    by (head_dim, theta, direction, dtype) on first use. Cache is in-memory by
    default, disk-backed with FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1."""
    compile_key = (_SRC_FP, x.shape[-1], float(theta), float(direction), x.dtype)
    if compile_key not in rope_key_run.compile_cache:
        _, compiled = compile_rope_key(x.shape[-1], theta, direction=direction)
        rope_key_run.compile_cache[compile_key] = compiled
    rope_key_run.compile_cache[compile_key](x, pos, out)


rope_key_run.compile_cache = get_jit_cache("rope_key")
