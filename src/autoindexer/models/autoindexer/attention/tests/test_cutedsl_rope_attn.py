"""Numerically compare the CuteDSL fused segment-RoPE attention kernel
(``attention/cutedsl/wrapper.py`` → ``fused_chain_of_edits_attn_cutedsl``) against
the eager reference ``chain_of_edits_attn_ref`` in ``attention/reference.py``.

Inputs are built from the same synthetic ``OperationCase`` pool
(``generate_sequences.OPERATION_CASES`` / ``get_parsed_positions_from_case``) used by
``test_triton_rope_attn.py`` and the rest of the model's test suite.

Requires CUDA and cutedsl deps (nvidia-cutlass-dsl, quack-kernels, …); skipped
otherwise. Only the fully-fused forward/backward path is covered here — unlike the
Triton suite there is no separate attn-weights kernel, and external attention masks /
dropout are unsupported on the cutedsl path.
"""

import pytest
import torch

from autoindexer.models.autoindexer.attention.cutedsl.wrapper import _HAS_CUTEDSL, fused_chain_of_edits_attn_cutedsl
from autoindexer.models.autoindexer.attention.reference import chain_of_edits_attn_ref
from autoindexer.models.autoindexer.attention.tests.attn_test_utils import assert_finite_allclose, build_attention_case
from autoindexer.models.autoindexer.tests.generate_sequences import OPERATION_CASES, OperationCase


def _cutedsl_runnable() -> bool:
    # `_HAS_CUTEDSL` already gates on compute capability (Hopper+), not just import success.
    return _HAS_CUTEDSL and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not _cutedsl_runnable(),
    reason="CuteDSL attention kernels require cutedsl deps, CUDA, and compute capability >= 9.0 (Hopper+)",
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else None

# Looser than the Triton test: cutedsl runs bf16 attention internally and casts back.
_ATOL = 2e-2
_RTOL = 2e-2


@pytest.fixture(autouse=True)
def _disable_tf32():
    """TF32 matmuls lose enough precision to blow past the tolerances below."""
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


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_fused_attn_output_forward_matches_reference(operation_case: OperationCase):
    rotary_emb, query, keys, values, parsed_positions = build_attention_case(operation_case, DEVICE)
    ref_out, _ = chain_of_edits_attn_ref(rotary_emb, query, keys, values, parsed_positions)
    cute_out, _ = fused_chain_of_edits_attn_cutedsl(rotary_emb, query, keys, values, parsed_positions)
    assert_finite_allclose("fused attn output", cute_out, ref_out, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_fused_attn_output_backward_matches_reference(operation_case: OperationCase):
    rotary_emb, query, keys, values, parsed_positions = build_attention_case(operation_case, DEVICE)

    def _grads(attn_fn):
        q, k, v = (t.clone().requires_grad_(True) for t in (query, keys, values))
        out, _ = attn_fn(rotary_emb, q, k, v, parsed_positions)
        out.sum().backward()
        return q.grad, k.grad, v.grad

    dq_ref, dk_ref, dv_ref = _grads(chain_of_edits_attn_ref)
    dq_cute, dk_cute, dv_cute = _grads(fused_chain_of_edits_attn_cutedsl)

    assert_finite_allclose("dQ", dq_cute, dq_ref, atol=_ATOL, rtol=_RTOL)
    assert_finite_allclose("dK", dk_cute, dk_ref, atol=_ATOL, rtol=_RTOL)
    assert_finite_allclose("dV", dv_cute, dv_ref, atol=_ATOL, rtol=_RTOL)
