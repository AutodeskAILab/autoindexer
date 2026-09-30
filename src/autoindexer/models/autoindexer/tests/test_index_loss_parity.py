"""Parity tests for index-head loss vs a reference full-attention implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from autoindexer.models.autoindexer.config import AutoIndexerConfig
from autoindexer.models.autoindexer.attention.reference import compute_attn_weights_ref
from autoindexer.models.autoindexer.attn_layers import compute_end_index_mask, compute_attn_weights_iter, AutoIndexerIndexHead
from autoindexer.models.autoindexer.perturb_labels import (
    PositionBlockList,
    get_parsed_positions_from_perturbations,
)
from autoindexer.models.autoindexer.tests.generate_sequences import (
    DEFAULT_MARKER_MAP as MARKER_MAP,
    OPERATION_CASES,
    OperationCase,
    build_synthetic_sequence,
)
from autoindexer.models.autoindexer.type_utils import BlockType, CursorIndices, IterPositions, MarkerMap, PositionBlock


def _ref_get_edit_end_index_mask(parsed_positions: PositionBlockList, seq_len: int) -> torch.Tensor:
    """Reference `get_edit_end_index_mask`: the full (seq_len, seq_len) boolean mask over the edit_end row."""
    device = parsed_positions.device
    edit_end_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    for block in parsed_positions:
        if block.abs_pos_mask is None and block.block_type == BlockType.EXTEND:
            end_idx = len(block.key_pos_ids)
            edit_end_mask[end_idx - 1, :end_idx] = block.key_pos_ids > block.query_pos_ids[-1]
    return edit_end_mask


def _ref_calc_index_loss(
    weights: torch.Tensor, token_sequence: torch.Tensor, cursor_indices: torch.Tensor, marker_map: MarkerMap, ignore_index: int = -100
) -> torch.Tensor:
    """Reference ``calc_index_loss``: cross-entropy over the full (batch, 2, q_len, k_len) weight matrix."""
    is_edit_marker = token_sequence == marker_map["edit"]
    target_indices = cursor_indices[:, 1:].clone()
    # Edit start index can only be predicted at the token before the edit marker position.
    target_indices[:, :, 0].masked_fill_(~is_edit_marker[:, 1:], ignore_index)
    # Edit end index can only be predicted at the edit token position.
    target_indices[:, :, 1].masked_fill_(~is_edit_marker[:, :-1], ignore_index)
    # weights: (batch_size, 2, q_len, k_len) -> (batch_size, k_len, q_len, 2) i.e. (N, C, d1, d2)
    return F.cross_entropy(weights.permute(0, 3, 2, 1), target_indices, ignore_index=ignore_index, reduction="mean")


def _ref_index_head_forward(
    index_head: AutoIndexerIndexHead, hidden_states: torch.Tensor, parsed_positions: PositionBlockList, rotary_emb
) -> torch.Tensor:
    """Reference full-sequence index-head forward pass: builds the whole weight matrix rather than the sparse per-edit logits."""
    batch_size, seq_len, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_len, -1, index_head.head_dim)
    query_states = index_head.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = index_head.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    attn_weights = compute_attn_weights_ref(rotary_emb, query_states, key_states, parsed_positions)
    edit_end_mask = _ref_get_edit_end_index_mask(parsed_positions, key_states.shape[-2])
    # (q_len, k_len) -> (1, q_len, k_len), broadcasts over batch
    attn_weights[:, 1] = attn_weights[:, 1].masked_fill(~edit_end_mask.unsqueeze(0), float("-inf"))
    return attn_weights


def _index_loss(
    index_head: AutoIndexerIndexHead, hidden_states: torch.Tensor, parsed_positions: PositionBlockList, cursor_indices: CursorIndices, rotary_emb
) -> torch.Tensor:
    """The index loss as actually computed by `AutoIndexerIndexHead`: sparse per-edit log_softmax losses."""
    _, (start_loss, end_loss), _ = index_head(
        hidden_states,
        parsed_positions=parsed_positions,
        cursor_indices=cursor_indices,
        rotary_emb=rotary_emb,
    )
    return (start_loss + end_loss) / 2


def _truncate_parsed_positions(parsed_positions: PositionBlockList, seq_len: int) -> PositionBlockList:
    """Trim position blocks to the decoder-input length (``perturbed_ids[:, :-1]``)."""
    truncated = PositionBlockList()
    for block in parsed_positions:
        if block.start_idx >= seq_len:
            continue
        end_idx = min(block.end_idx, seq_len)
        query_len = end_idx - block.start_idx
        if query_len <= 0:
            continue
        truncated.append(
            PositionBlock(
                query_pos_ids=block.query_pos_ids[:query_len].clone(),
                key_pos_ids=block.key_pos_ids[:end_idx].clone(),
                abs_pos_mask=block.abs_pos_mask[:end_idx].clone() if block.abs_pos_mask is not None else None,
                attn_mask=block.attn_mask[:end_idx].clone(),
                start_idx=block.start_idx,
                end_idx=end_idx,
                block_type=block.block_type,
            )
        )
    truncated.concat()
    return truncated


def _build_case(operation_case: OperationCase, batch_size: int, seed: int, hidden_size: int = 64, train: bool = False):
    """Shared setup for all parity tests: an index head + rotary embedding, a synthetic perturbed sequence, and its parsed positions."""
    torch.manual_seed(seed)
    device = torch.device("cpu")

    config = AutoIndexerConfig(hidden_size=hidden_size, head_dim=hidden_size, max_position_embeddings=1000, inplace_edits=True, mean_insert=2)
    index_head = AutoIndexerIndexHead(config).to(device)
    index_head.train(train)
    rotary_emb = LlamaRotaryEmbedding(config, device)

    tokens, cursor_tensor, cursor_dict, perturbations = build_synthetic_sequence(
        operation_case, MARKER_MAP, batch_size, device
    )
    token_count = tokens.shape[1]
    seq_len = token_count - 1
    parsed_positions = _truncate_parsed_positions(get_parsed_positions_from_perturbations(perturbations, device=device), seq_len)

    return index_head, rotary_emb, tokens, cursor_tensor, cursor_dict, seq_len, parsed_positions


def _edit_target_logits(index_head: AutoIndexerIndexHead, query_states, key_states, rotary_emb, block: PositionBlock):
    """Recompute the sparse edit_start/edit_end logit rows exactly as ``compute_index_losses`` does for one edit block."""
    edit_token_idx = block.end_idx - 1
    start_q = query_states[:, 0:1, edit_token_idx - 1 : edit_token_idx, :]
    end_q = query_states[:, 1:2, edit_token_idx : edit_token_idx + 1, :]

    cursor_start_positions = IterPositions(
        key_pos_ids=block.key_pos_ids[:-1] - block.query_pos_ids[-2],
        attn_mask=block.attn_mask[:-1],
    )
    cursor_end_positions = IterPositions(
        key_pos_ids=block.key_pos_ids - block.query_pos_ids[-1],
        attn_mask=block.attn_mask,
    )

    start_logits = compute_attn_weights_iter(rotary_emb, start_q, key_states[:, [0], :edit_token_idx, :], cursor_start_positions).squeeze((1, 2))
    end_logits = compute_attn_weights_iter(rotary_emb, end_q, key_states[:, [1], : edit_token_idx + 1, :], cursor_end_positions).squeeze((1, 2))
    end_logits = end_logits.masked_fill(~compute_end_index_mask(cursor_end_positions), float("-inf"))
    return edit_token_idx, start_logits, end_logits


def _collect_supervised_logits(attn_weights: torch.Tensor, cursor_indices: CursorIndices):
    """Extract the logit rows the reference ``calc_index_loss`` supervises, from its full weight matrix."""
    start_logits = {}
    end_logits = {}
    for edit_token_idx in cursor_indices:
        start_q = edit_token_idx - 1
        start_logits[edit_token_idx] = attn_weights[:, 0, start_q, : start_q + 1]
        end_logits[edit_token_idx] = attn_weights[:, 1, edit_token_idx, : edit_token_idx + 1]
    return start_logits, end_logits


def _per_edit_loss_from_logits(start_logits, end_logits, cursor_indices: CursorIndices):
    """Replicate the actual normalization (mean per edit, then average start/end) on top of the reference logits."""
    start_total = torch.zeros(())
    end_total = torch.zeros(())
    for edit_token_idx, (start_idx, end_idx) in cursor_indices.items():
        start_total = start_total - torch.log_softmax(start_logits[edit_token_idx], dim=-1)[..., start_idx].mean()
        end_total = end_total - torch.log_softmax(end_logits[edit_token_idx], dim=-1)[..., end_idx].mean()
    num_edits = len(cursor_indices)
    return ((start_total / num_edits) + (end_total / num_edits)) / 2


def _run_parity_case(operation_case: OperationCase, seed: int = 0, batch_size: int = 2):
    index_head, rotary_emb, tokens, cursor_tensor, cursor_dict, seq_len, parsed_positions = _build_case(operation_case, batch_size, seed)
    index_head.eval()
    hidden_states = torch.randn(batch_size, seq_len, index_head.q_proj.in_features, requires_grad=True)

    ref_weights = _ref_index_head_forward(index_head, hidden_states, parsed_positions, rotary_emb)
    ref_loss = _ref_calc_index_loss(ref_weights, tokens, cursor_tensor, MARKER_MAP)
    loss = _index_loss(index_head, hidden_states, parsed_positions, cursor_dict, rotary_emb)

    start_ref, end_ref = _collect_supervised_logits(ref_weights, cursor_dict)

    query_states = index_head.q_proj(hidden_states).view(batch_size, seq_len, 2, index_head.head_dim).transpose(1, 2)
    key_states = index_head.k_proj(hidden_states).view(batch_size, seq_len, 2, index_head.head_dim).transpose(1, 2)

    max_start_target_err = 0.0
    max_end_target_err = 0.0
    for block in parsed_positions:
        if block.block_type != BlockType.EXTEND:
            continue
        edit_token_idx, start_logits, end_logits = _edit_target_logits(index_head, query_states, key_states, rotary_emb, block)
        start_idx, end_idx = cursor_dict[edit_token_idx]

        # Off-target logits can legitimately differ by an additive per-row constant between the
        # two implementations (log_softmax/cross_entropy are shift-invariant); only the target matches.
        start_target_err = (start_ref[edit_token_idx][..., start_idx] - start_logits[..., start_idx]).abs().max().item()
        end_target_err = (end_ref[edit_token_idx][..., end_idx] - end_logits[..., end_idx]).abs().max().item()
        max_start_target_err = max(max_start_target_err, start_target_err)
        max_end_target_err = max(max_end_target_err, end_target_err)

    return {
        "ref_loss": ref_loss.item(),
        "loss": loss.item(),
        "loss_from_ref_logits": _per_edit_loss_from_logits(start_ref, end_ref, cursor_dict).item(),
        "max_start_target_err": max_start_target_err,
        "max_end_target_err": max_end_target_err,
        "num_edits": len(cursor_dict),
    }


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_index_loss_matches_reference_implementation(operation_case):
    result = _run_parity_case(operation_case)
    assert result["max_start_target_err"] < 1e-4, f"start target logit diverges: {result['max_start_target_err']:.3e}"
    assert result["max_end_target_err"] < 1e-4, f"end target logit diverges: {result['max_end_target_err']:.3e}"
    assert abs(result["ref_loss"] - result["loss"]) < 1e-4, (
        f"loss mismatch: ref={result['ref_loss']:.6f} actual={result['loss']:.6f}"
    )


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_reference_cross_entropy_equals_per_edit_log_softmax(operation_case):
    """Sanity check: the reference F.cross_entropy over the full weight matrix must equal an explicit per-edit mean log_softmax."""
    result = _run_parity_case(operation_case)
    assert abs(result["ref_loss"] - result["loss_from_ref_logits"]) < 1e-5


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_index_loss_gradients_match_reference_implementation(operation_case):
    """Gradients into the index-head projections must match between loss implementations."""
    index_head, rotary_emb, tokens, cursor_tensor, cursor_dict, seq_len, parsed_positions = _build_case(operation_case, batch_size=2, seed=7, train=True)

    def grads_for(loss_fn) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.randn(2, seq_len, index_head.q_proj.in_features, requires_grad=True)
        index_head.zero_grad()
        loss_fn(hidden_states).backward()
        return index_head.q_proj.weight.grad.clone(), index_head.k_proj.weight.grad.clone()

    torch.manual_seed(0)
    ref_q, ref_k = grads_for(lambda hs: _ref_calc_index_loss(_ref_index_head_forward(index_head, hs, parsed_positions, rotary_emb), tokens, cursor_tensor, MARKER_MAP))
    torch.manual_seed(0)
    q, k = grads_for(lambda hs: _index_loss(index_head, hs, parsed_positions, cursor_dict, rotary_emb))

    assert torch.allclose(ref_q, q, atol=1e-4, rtol=1e-4), f"q_proj grad max err={(ref_q - q).abs().max():.3e}"
    assert torch.allclose(ref_k, k, atol=1e-4, rtol=1e-4), f"k_proj grad max err={(ref_k - k).abs().max():.3e}"


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_index_loss_gradients_flow(operation_case):
    """The index loss must backprop into the index-head projections."""
    index_head, rotary_emb, _, _, cursor_dict, seq_len, parsed_positions = _build_case(
        operation_case, batch_size=1, seed=1, hidden_size=32, train=True
    )
    hidden_states = torch.randn(1, seq_len, index_head.q_proj.in_features, requires_grad=True)

    loss = _index_loss(index_head, hidden_states, parsed_positions, cursor_dict, rotary_emb)
    loss.backward()

    assert index_head.q_proj.weight.grad is not None
    assert index_head.q_proj.weight.grad.abs().sum() > 0


def _print_parity_summary():
    print("Index loss parity: reference vs actual\n")
    for operation_case in OPERATION_CASES:
        result = _run_parity_case(operation_case)
        print(f"ops={operation_case.operations} label_len={operation_case.label_len}")
        print(f"  edits={result['num_edits']}")
        print(f"  ref_loss={result['ref_loss']:.6f}")
        print(f"  loss={result['loss']:.6f}")
        print(f"  loss_from_ref_logits={result['loss_from_ref_logits']:.6f}")
        print(f"  max_start_target_err={result['max_start_target_err']:.3e}")
        print(f"  max_end_target_err={result['max_end_target_err']:.3e}")
        match = abs(result["ref_loss"] - result["loss"]) < 1e-4
        print(f"  MATCH={match}\n")


if __name__ == "__main__":
    _print_parity_summary()
