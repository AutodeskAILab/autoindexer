"""Shared fixtures for the Triton fused segment-RoPE attention test suite.

Builds Q/K/V tensors and real ``PositionBlockList`` objects (via
``generate_sequences.get_parsed_positions_from_case``) from the shared pool of synthetic
``OperationCase``s used across the model's test suite, so the Triton kernels in
``attention/kernels/triton_rope_attn.py`` can be checked against the eager reference
implementations in ``attention/reference.py`` instead of hand-rolled duplicates of the same
segment-attention math.
"""

from __future__ import annotations

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from autoindexer.models.autoindexer.config import AutoIndexerConfig
from autoindexer.models.autoindexer.perturb_labels import PositionBlockList
from autoindexer.models.autoindexer.tests.generate_sequences import OperationCase, get_parsed_positions_from_case

DEFAULT_HEAD_DIM = 64


def build_attention_case(
    operation_case: OperationCase,
    device: torch.device,
    batch_size: int = 2,
    num_heads: int = 2,
    head_dim: int = DEFAULT_HEAD_DIM,
    seed: int = 0,
) -> tuple[LlamaRotaryEmbedding, torch.Tensor, torch.Tensor, torch.Tensor, PositionBlockList]:
    """Build ``(rotary_emb, query, keys, values, parsed_positions)`` for one ``operation_case``.

    Q/K/V are random ``(batch_size, num_heads, seq_len, head_dim)`` tensors, where ``seq_len``
    is however many tokens ``operation_case`` produces once run through
    ``get_parsed_positions_from_case`` (the same machinery ``generate_sequences`` uses
    elsewhere in the test suite).
    """
    torch.manual_seed(seed)
    parsed_positions = get_parsed_positions_from_case(operation_case, device)
    seq_len = len(parsed_positions.query_pos_ids)

    config = AutoIndexerConfig(num_heads=num_heads, head_dim=head_dim)
    rotary_emb = LlamaRotaryEmbedding(config, device=device)

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    return rotary_emb, query, keys, values, parsed_positions


def assert_finite_allclose(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    """Compare two tensors that may hold ``-inf`` at (the same) masked positions: finite
    entries must be numerically close, and masked (``-inf``) entries must match exactly."""
    assert actual.shape == expected.shape, f"{name} shape mismatch: {actual.shape} vs {expected.shape}"
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    finite = torch.isfinite(expected_flat)
    if finite.any():
        max_diff = (actual_flat[finite] - expected_flat[finite]).abs().max().item()
        assert torch.allclose(actual_flat[finite], expected_flat[finite], atol=atol, rtol=rtol), f"{name} mismatch: max |diff| = {max_diff:.2e}"
    masked = ~finite
    if masked.any():
        assert torch.all(torch.isinf(actual_flat[masked]) & (actual_flat[masked] < 0)), f"{name} masked positions should be -inf"
