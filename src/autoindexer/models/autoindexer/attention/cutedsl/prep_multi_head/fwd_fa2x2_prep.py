import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import hashlib
import math
import torch

from flash_attn.cute.cache_utils import get_jit_cache

# Folded into the compile key so a source edit invalidates the on-disk cache.
_SRC_FP = hashlib.sha256(open(__file__, "rb").read()).hexdigest()


class PrepPack():
    """
    FA2x2 prep-stage kernel — multi-head variant of
    prep_cutedsl/single_head/fwd_fa2x2_prep.py.

    Routing (src_idx / rope_idx) is per-token, not per-head — every head
    attends over the same key columns at the same RoPE positions — so the
    only change from the single-head kernel is a second grid dimension over
    H: grid=[num_blocks, num_heads, 1], with block_idx().y selecting the head.
    The per-row warp-shuffle RoPE math is byte-for-byte identical to the
    single-head version.

    k / v:           (num_src, H, DK)
    k_out / v_out:    (num_out, H, DK)
    src_idx / rope_idx: (num_out,) — shared across all H heads.

    Requires DK to be a multiple of the warp size (32). No batch (B) support
    yet — B=1 is assumed throughout, same as the single-head kernel.
    """

    def __init__(self, dk, theta, num_warps=16, rows_per_warp=8):
        assert dk % 32 == 0, "PrepPack requires DK to be a multiple of the warp size (32)"

        self.dk = dk
        self.elems_per_lane = dk // 32
        self.num_warps = num_warps
        self.rows_per_warp = rows_per_warp
        self.rows_per_block = num_warps * rows_per_warp
        self.num_threads = num_warps * 32
        self.log_theta = math.log(theta)

    @cute.jit
    def __call__(
        self,
        k,
        v,
        src_idx,
        rope_idx,
        k_out,
        v_out,
        stream,
    ):
        num_out, num_heads, _ = k_out.shape
        rows_per_block: cutlass.Constexpr = self.rows_per_block

        num_blocks = (num_out + rows_per_block - 1) // rows_per_block

        self.kernel(k, v, src_idx, rope_idx, k_out, v_out).launch(
            grid=[num_blocks, num_heads, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        k,
        v,
        src_idx,
        rope_idx,
        k_out,
        v_out,
    ):
        num_out, _, _ = k_out.shape

        elems_per_lane: cutlass.Constexpr = self.elems_per_lane
        num_warps: cutlass.Constexpr = self.num_warps
        rows_per_warp: cutlass.Constexpr = self.rows_per_warp
        rows_per_block: cutlass.Constexpr = self.rows_per_block
        dk: cutlass.Constexpr = self.dk
        log_theta: cutlass.Constexpr = self.log_theta

        bidx, bidy, _ = cute.arch.block_idx()
        head_idx = bidy
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        # Round-robin: `rows_per_warp` rows are processed by this warp, one per round.
        for round_idx in range(rows_per_warp):
            row_idx = bidx * rows_per_block + round_idx * num_warps + warp_idx

            if row_idx < num_out:
                src = src_idx[row_idx]

                k_row = k[src, head_idx, None]
                v_row = v[src, head_idx, None]

                k_row_z = cute.zipped_divide(k_row, (elems_per_lane,))
                v_row_z = cute.zipped_divide(v_row, (elems_per_lane,))

                # 1 warp holds a full row of DK elements — 1 load per lane.
                k_frag = k_row_z[(None, (lane_idx,))].load()
                v_frag = v_row_z[(None, (lane_idx,))].load()

                k_pos = rope_idx[row_idx]
                k_frag_roped = self._rope(k_frag, lane_idx, k_pos, elems_per_lane, dk, log_theta)

                k_out_row = k_out[row_idx, head_idx, None]
                v_out_row = v_out[row_idx, head_idx, None]

                k_out_z = cute.zipped_divide(k_out_row, (elems_per_lane,))
                v_out_z = cute.zipped_divide(v_out_row, (elems_per_lane,))

                k_out_z[(None, (lane_idx,))].store(k_frag_roped.to(cutlass.BFloat16))
                v_out_z[(None, (lane_idx,))].store(v_frag)

    @staticmethod
    @cute.jit
    def _rope(
        k_frag,
        lane_idx,
        key_pos,
        elems_per_lane,
        dk,
        log_theta,
    ):
        """Identical to the single-head kernel's _rope — RoPE math never touches
        the head dimension, only which (row, head) slice it's applied to."""
        wsize: cutlass.Constexpr = cute.arch.WARP_SIZE
        lanes_per_half: cutlass.Constexpr = wsize // 2
        half_dim: cutlass.Constexpr = dk // 2

        lane_in_half = lane_idx % lanes_per_half
        freq_base = lane_in_half * elems_per_lane

        # lo half (lane_idx < lanes_per_half): out = x*c - y*s
        # hi half (lane_idx >= lanes_per_half): out = x*c + y*s
        sign = cutlass.Float32(-1.0)
        if lane_idx >= lanes_per_half:
            sign = cutlass.Float32(1.0)

        key_pos_f = cutlass.Float32(key_pos)

        out_regs = cute.make_rmem_tensor(
            cute.make_layout((elems_per_lane,), stride=(1,)),
            cutlass.Float32,
        )

        for i in range(elems_per_lane):
            # Partner (other) half fetched via intra-warp XOR shuffle — no 2nd global load.
            partner_val = cute.arch.shuffle_sync_bfly(k_frag[i], offset=lanes_per_half)

            freq_idx = freq_base + i

            freq = cute.math.exp(
                -cutlass.Float32(freq_idx) * cutlass.Float32(log_theta) / cutlass.Float32(half_dim)
            )
            angle = key_pos_f * freq

            c = cute.math.cos(angle)
            s = cute.math.sin(angle)

            x = cutlass.Float32(k_frag[i])
            y = cutlass.Float32(partner_val)

            out_regs[i] = x * c + sign * y * s

        return out_regs.load()


class PrepPackBackward():
    """
    Backward for PrepPack's forward gather (+ RoPE on K). Forward can read
    the same physical row into more than one output row (a physical key row
    reused by several segments at different RoPE positions), so backward is
    a scatter-ADD, not a scatter-assign.

    Rather than atomics, this takes a dataloader-precomputed dense inverse
    index (inv_idx: (num_src, max_mult) int32, -1 = empty slot — see
    utils/fa2_path_meta.build_inverse_index) so each physical row's gradient
    can be summed entirely within ONE warp: no two warps ever write the same
    output row, so the final write needs no atomic.

    One warp handles one physical row `p`. It walks inv_idx[p, :]; for every
    valid entry `i`, it gathers grad_k_out[i] / grad_v_out[i], applies the
    inverse RoPE rotation to the K gradient (same _rope primitive as the
    forward kernel, with direction=-1 — the transpose of an orthogonal
    rotation, same trick as fast_rope's backward), and accumulates in
    registers. After the loop, ONE store writes grad_k[p] / grad_v[p] — rows
    with no contributions correctly get all-zero gradient.

    grad_k_out / grad_v_out: (num_out, H, DK)
    rope_idx: (num_out,) — same array the forward kernel used
    inv_idx:  (num_src, max_mult) — see above
    grad_k / grad_v: (num_src, H, DK)
    """

    def __init__(self, dk, theta, num_warps=16, rows_per_warp=8):
        assert dk % 32 == 0, "PrepPackBackward requires DK to be a multiple of the warp size (32)"

        self.dk = dk
        self.elems_per_lane = dk // 32
        self.num_warps = num_warps
        self.rows_per_warp = rows_per_warp
        self.rows_per_block = num_warps * rows_per_warp
        self.num_threads = num_warps * 32
        self.log_theta = math.log(theta)

    @cute.jit
    def __call__(
        self,
        grad_k_out,
        grad_v_out,
        rope_idx,
        inv_idx,
        grad_k,
        grad_v,
        stream,
    ):
        num_src, num_heads, _ = grad_k.shape
        rows_per_block: cutlass.Constexpr = self.rows_per_block

        num_blocks = (num_src + rows_per_block - 1) // rows_per_block

        self.kernel(grad_k_out, grad_v_out, rope_idx, inv_idx, grad_k, grad_v).launch(
            grid=[num_blocks, num_heads, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        grad_k_out,
        grad_v_out,
        rope_idx,
        inv_idx,
        grad_k,
        grad_v,
    ):
        num_src, _, _ = grad_k.shape
        _, max_mult = inv_idx.shape

        elems_per_lane: cutlass.Constexpr = self.elems_per_lane
        num_warps: cutlass.Constexpr = self.num_warps
        rows_per_warp: cutlass.Constexpr = self.rows_per_warp
        rows_per_block: cutlass.Constexpr = self.rows_per_block
        dk: cutlass.Constexpr = self.dk
        log_theta: cutlass.Constexpr = self.log_theta

        bidx, bidy, _ = cute.arch.block_idx()
        head_idx = bidy
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        for round_idx in range(rows_per_warp):
            row_idx = bidx * rows_per_block + round_idx * num_warps + warp_idx

            if row_idx < num_src:
                acc_k = cute.make_rmem_tensor(
                    cute.make_layout((elems_per_lane,), stride=(1,)), cutlass.Float32,
                )
                acc_v = cute.make_rmem_tensor(
                    cute.make_layout((elems_per_lane,), stride=(1,)), cutlass.Float32,
                )
                for t in range(elems_per_lane):
                    acc_k[t] = cutlass.Float32(0.0)
                    acc_v[t] = cutlass.Float32(0.0)

                for m in range(max_mult):
                    out_idx = inv_idx[row_idx, m]

                    if out_idx >= 0:
                        gko_row = grad_k_out[out_idx, head_idx, None]
                        gvo_row = grad_v_out[out_idx, head_idx, None]

                        gko_z = cute.zipped_divide(gko_row, (elems_per_lane,))
                        gvo_z = cute.zipped_divide(gvo_row, (elems_per_lane,))

                        gko_frag = gko_z[(None, (lane_idx,))].load()
                        gvo_frag = gvo_z[(None, (lane_idx,))].load()

                        pos = rope_idx[out_idx]
                        # direction=-1.0: inverse rotation, transpose of the forward RoPE.
                        gk_contrib = self._rope(gko_frag, lane_idx, pos, elems_per_lane, dk, log_theta, -1.0)

                        for t in range(elems_per_lane):
                            acc_k[t] += gk_contrib[t]
                            acc_v[t] += cutlass.Float32(gvo_frag[t])

                grad_k_row = grad_k[row_idx, head_idx, None]
                grad_v_row = grad_v[row_idx, head_idx, None]

                grad_k_z = cute.zipped_divide(grad_k_row, (elems_per_lane,))
                grad_v_z = cute.zipped_divide(grad_v_row, (elems_per_lane,))

                grad_k_z[(None, (lane_idx,))].store(acc_k.load().to(cutlass.BFloat16))
                grad_v_z[(None, (lane_idx,))].store(acc_v.load().to(cutlass.BFloat16))

    @staticmethod
    @cute.jit
    def _rope(
        k_frag,
        lane_idx,
        key_pos,
        elems_per_lane,
        dk,
        log_theta,
        direction,
    ):
        """Same primitive as PrepPack._rope, generalized with a `direction`
        sign flip (see fast_rope/kernel.py's module docstring for why this is
        exactly the transpose rotation, not separate backward math)."""
        wsize: cutlass.Constexpr = cute.arch.WARP_SIZE
        lanes_per_half: cutlass.Constexpr = wsize // 2
        half_dim: cutlass.Constexpr = dk // 2

        lane_in_half = lane_idx % lanes_per_half
        freq_base = lane_in_half * elems_per_lane

        sign = cutlass.Float32(-1.0)
        if lane_idx >= lanes_per_half:
            sign = cutlass.Float32(1.0)
        sign = sign * cutlass.Float32(direction)

        key_pos_f = cutlass.Float32(key_pos)

        out_regs = cute.make_rmem_tensor(
            cute.make_layout((elems_per_lane,), stride=(1,)),
            cutlass.Float32,
        )

        for i in range(elems_per_lane):
            partner_val = cute.arch.shuffle_sync_bfly(k_frag[i], offset=lanes_per_half)

            freq_idx = freq_base + i

            freq = cute.math.exp(
                -cutlass.Float32(freq_idx) * cutlass.Float32(log_theta) / cutlass.Float32(half_dim)
            )
            angle = key_pos_f * freq

            c = cute.math.cos(angle)
            s = cute.math.sin(angle)

            x = cutlass.Float32(k_frag[i])
            y = cutlass.Float32(partner_val)

            out_regs[i] = x * c + sign * y * s

        return out_regs.load()


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_prep_pack(dk, theta, num_warps=16, rows_per_warp=8):
    NUM_SRC   = cute.sym_int()
    NUM_OUT   = cute.sym_int()
    NUM_HEADS = cute.sym_int()   # dynamic — one compiled kernel works for any H

    k        = _fake(cute.BFloat16, (NUM_SRC, NUM_HEADS, dk), (2, 1, 0), 16)
    v        = _fake(cute.BFloat16, (NUM_SRC, NUM_HEADS, dk), (2, 1, 0), 16)
    src_idx  = _fake(cute.Int32,    (NUM_OUT,),               (0,),     4)
    rope_idx = _fake(cute.Int32,    (NUM_OUT,),               (0,),     4)
    k_out    = _fake(cute.BFloat16, (NUM_OUT, NUM_HEADS, dk), (2, 1, 0), 16)
    v_out    = _fake(cute.BFloat16, (NUM_OUT, NUM_HEADS, dk), (2, 1, 0), 16)
    stream   = make_fake_stream(use_tvm_ffi_env_stream=True)

    _prep = PrepPack(dk, theta, num_warps, rows_per_warp)
    _compiled = cute.compile(
        _prep, k, v, src_idx, rope_idx, k_out, v_out, stream,
        options="--enable-tvm-ffi",
    )

    return _prep, _compiled


def fwd_prep(k, v, src_idx, rope_idx, k_out, v_out, theta):
    """Launch the prep-pack forward, compiling+caching keyed by (head_dim, theta,
    dtype) on first use (in-memory, or disk with FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1)."""
    compile_key = (_SRC_FP, k.shape[-1], float(theta), k.dtype)
    if compile_key not in fwd_prep.compile_cache:
        _, compiled = compile_prep_pack(k.shape[-1], theta)
        fwd_prep.compile_cache[compile_key] = compiled
    fwd_prep.compile_cache[compile_key](k, v, src_idx, rope_idx, k_out, v_out)


fwd_prep.compile_cache = get_jit_cache("prep_fwd")


def compile_prep_pack_backward(dk, theta, num_warps=16, rows_per_warp=8):
    NUM_SRC   = cute.sym_int()
    NUM_OUT   = cute.sym_int()
    NUM_HEADS = cute.sym_int()
    MAX_MULT  = cute.sym_int()

    grad_k_out = _fake(cute.BFloat16, (NUM_OUT, NUM_HEADS, dk), (2, 1, 0), 16)
    grad_v_out = _fake(cute.BFloat16, (NUM_OUT, NUM_HEADS, dk), (2, 1, 0), 16)
    rope_idx   = _fake(cute.Int32,    (NUM_OUT,),               (0,),     4)
    inv_idx    = _fake(cute.Int32,    (NUM_SRC, MAX_MULT),      (1, 0),   4)
    grad_k     = _fake(cute.BFloat16, (NUM_SRC, NUM_HEADS, dk), (2, 1, 0), 16)
    grad_v     = _fake(cute.BFloat16, (NUM_SRC, NUM_HEADS, dk), (2, 1, 0), 16)
    stream     = make_fake_stream(use_tvm_ffi_env_stream=True)

    _bwd = PrepPackBackward(dk, theta, num_warps, rows_per_warp)
    _compiled = cute.compile(
        _bwd, grad_k_out, grad_v_out, rope_idx, inv_idx, grad_k, grad_v, stream,
        options="--enable-tvm-ffi",
    )

    return _bwd, _compiled


def bwd_prep(grad_k_out, grad_v_out, rope_idx, inv_idx, grad_k, grad_v, theta):
    """Launch the prep-pack backward, compiling+caching keyed by (head_dim, theta,
    dtype) on first use (in-memory, or disk with FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1)."""
    compile_key = (_SRC_FP, grad_k_out.shape[-1], float(theta), grad_k_out.dtype)
    if compile_key not in bwd_prep.compile_cache:
        _, compiled = compile_prep_pack_backward(grad_k_out.shape[-1], theta)
        bwd_prep.compile_cache[compile_key] = compiled
    bwd_prep.compile_cache[compile_key](grad_k_out, grad_v_out, rope_idx, inv_idx, grad_k, grad_v)


bwd_prep.compile_cache = get_jit_cache("prep_bwd")


