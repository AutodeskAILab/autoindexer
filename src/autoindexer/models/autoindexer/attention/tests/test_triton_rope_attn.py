"""Numerically compare the Triton fused segment-RoPE attention kernels
(``attention/kernels/triton_rope_attn.py``, reached through the ``triton_wrapper`` entry
points ``autoindexer_attention_forward`` actually calls) against the eager reference implementations
in ``attention/reference.py`` (``compute_attn_weights_ref`` / ``chain_of_edits_attn_ref``).

Inputs are built from the same synthetic ``OperationCase`` pool
(``generate_sequences.OPERATION_CASES`` / ``get_parsed_positions_from_case``) used by the
rest of the model's test suite, rather than the hand-rolled ``boundary_indices`` and
duplicated segment-attention reference math that used to live alongside the kernels.

Requires CUDA (Triton kernel launches need a GPU); skipped otherwise.
"""

import pytest
import torch

from autoindexer.models.autoindexer.attention.reference import chain_of_edits_attn_ref, compute_attn_weights_ref
from autoindexer.models.autoindexer.attention.tests.attn_test_utils import assert_finite_allclose, build_attention_case
from autoindexer.models.autoindexer.attention.triton_wrapper import (
    _HAS_TRITON,
    compute_attn_weights_over_segments_triton_wrapper,
    fused_chain_of_edits_attn_triton,
)
from autoindexer.models.autoindexer.tests.generate_sequences import OPERATION_CASES, OperationCase

pytestmark = pytest.mark.skipif(not (_HAS_TRITON and torch.cuda.is_available()), reason="Triton attention kernels require CUDA")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else None


@pytest.fixture(autouse=True)
def _disable_tf32():
    """TF32 matmuls lose enough precision to blow past the tolerances below; the Triton
    kernels themselves already request ``input_precision="ieee"``, so match that on the
    eager reference side too."""
    if not torch.cuda.is_available():
        yield
        return
    prev_matmul, prev_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def _attn_weights_triton(rotary_emb, query, keys, parsed_positions, scaling=None):
    """Triton counterpart of ``compute_attn_weights_ref``: the causal mask + scaling
    ``compute_attn_weights_ref`` applies on top of the (shared) segment-attention step,
    applied here on top of the Triton-computed segment logits instead, so the two are
    directly comparable."""
    q_len, kv_len, head_dim = query.shape[-2], keys.shape[-2], query.shape[-1]
    attn_weights = compute_attn_weights_over_segments_triton_wrapper(rotary_emb, query, keys, parsed_positions)
    causal_mask = torch.tril(torch.ones((q_len, kv_len), dtype=torch.bool, device=query.device), diagonal=kv_len - q_len)
    attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
    scaling = head_dim**-0.5 if scaling is None else scaling
    return attn_weights * scaling


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_attn_weights_forward_matches_reference(operation_case: OperationCase):
    rotary_emb, query, keys, _, parsed_positions = build_attention_case(operation_case, DEVICE)
    ref = compute_attn_weights_ref(rotary_emb, query, keys, parsed_positions)
    tri = _attn_weights_triton(rotary_emb, query, keys, parsed_positions)
    assert_finite_allclose("attn_weights", tri, ref)


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_attn_weights_backward_matches_reference(operation_case: OperationCase):
    rotary_emb, query, keys, _, parsed_positions = build_attention_case(operation_case, DEVICE)

    q_ref, k_ref = query.clone().requires_grad_(True), keys.clone().requires_grad_(True)
    compute_attn_weights_ref(rotary_emb, q_ref, k_ref, parsed_positions).sum().backward()

    q_tri, k_tri = query.clone().requires_grad_(True), keys.clone().requires_grad_(True)
    _attn_weights_triton(rotary_emb, q_tri, k_tri, parsed_positions).sum().backward()

    assert_finite_allclose("dQ", q_tri.grad, q_ref.grad)
    assert_finite_allclose("dK", k_tri.grad, k_ref.grad)


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_fused_attn_output_forward_matches_reference(operation_case: OperationCase):
    rotary_emb, query, keys, values, parsed_positions = build_attention_case(operation_case, DEVICE)
    ref_out, _ = chain_of_edits_attn_ref(rotary_emb, query, keys, values, parsed_positions)
    tri_out, _ = fused_chain_of_edits_attn_triton(rotary_emb, query, keys, values, parsed_positions)
    assert_finite_allclose("fused attn output", tri_out, ref_out)


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_fused_attn_output_backward_matches_reference(operation_case: OperationCase):
    rotary_emb, query, keys, values, parsed_positions = build_attention_case(operation_case, DEVICE)

    def _grads(attn_fn):
        q, k, v = (t.clone().requires_grad_(True) for t in (query, keys, values))
        out, _ = attn_fn(rotary_emb, q, k, v, parsed_positions)
        out.sum().backward()
        return q.grad, k.grad, v.grad

    dq_ref, dk_ref, dv_ref = _grads(chain_of_edits_attn_ref)
    dq_tri, dk_tri, dv_tri = _grads(fused_chain_of_edits_attn_triton)

    assert_finite_allclose("dQ", dq_tri, dq_ref)
    assert_finite_allclose("dK", dk_tri, dk_ref)
    assert_finite_allclose("dV", dv_tri, dv_ref)


def test_fused_attn_output_ext_mask_and_dropout_smoke():
    """Not an exact-match test (Triton's dropout RNG differs from PyTorch's): just checks
    that the external-attention-mask and dropout code paths run, keep finite gradients, and
    that dropout actually perturbs the output relative to the no-dropout case."""
    rotary_emb, query, keys, values, parsed_positions = build_attention_case(OPERATION_CASES[0], DEVICE, seed=6)
    query, keys, values = (t.clone().requires_grad_(True) for t in (query, keys, values))
    batch_size, seq_len = query.shape[0], query.shape[-2]

    # Additive-bool style mask, broadcast over heads; blocks looking at the very last key.
    ext_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=DEVICE)
    ext_mask[:, :, :, -1] = False

    def _run(dropout_p):
        out, _ = fused_chain_of_edits_attn_triton(
            rotary_emb, query, keys, values, parsed_positions, attention_mask=ext_mask, dropout_p=dropout_p
        )
        return out

    out_no_dropout = _run(0.0)
    out_dropout = _run(0.5)
    assert out_no_dropout.shape == out_dropout.shape
    assert torch.isfinite(out_no_dropout).all()
    assert torch.isfinite(out_dropout).all()
    assert not torch.allclose(out_no_dropout, out_dropout), "dropout had no effect"

    out_dropout.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert keys.grad is not None and torch.isfinite(keys.grad).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()
