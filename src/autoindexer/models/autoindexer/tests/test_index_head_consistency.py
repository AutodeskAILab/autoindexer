"""Numerically compare the index head's edit_start / edit_end logits between the training path and the inference path."""

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from autoindexer.models.autoindexer.config import AutoIndexerConfig
from autoindexer.models.autoindexer.attn_layers import compute_end_index_mask, compute_attn_weights_iter, IndexKeysCache, AutoIndexerIndexHead
from autoindexer.models.autoindexer.type_utils import BlockType, IterPositions
from autoindexer.models.autoindexer.perturb_labels import get_parsed_positions_from_perturbations
from autoindexer.models.autoindexer.sequence_parser import AutoIndexerIdParser
from autoindexer.models.autoindexer.tests.generate_sequences import DEFAULT_MARKER_MAP, OPERATION_CASES, OperationCase, build_synthetic_sequence


def _training_edit_logits(index_head, hidden_states, parsed_positions, rotary_emb):
    """Reconstruct per-edit logits the same way ``compute_index_losses`` does."""
    batch_size, seq_len, _ = hidden_states.shape
    query_states = index_head.q_proj(hidden_states).view(batch_size, seq_len, 2, index_head.head_dim).transpose(1, 2)
    key_states = index_head.k_proj(hidden_states).view(batch_size, seq_len, 2, index_head.head_dim).transpose(1, 2)

    start_logits = {}
    end_logits = {}
    for block in parsed_positions:
        if block.block_type != BlockType.EXTEND:
            continue
        edit_token_idx = block.end_idx - 1
        cursor_start = IterPositions(
            key_pos_ids=block.key_pos_ids[:-1] - block.query_pos_ids[-2],
            attn_mask=block.attn_mask[:-1],
        )
        cursor_end = IterPositions(
            key_pos_ids=block.key_pos_ids - block.query_pos_ids[-1],
            attn_mask=block.attn_mask,
        )
        start_q = query_states[:, 0:1, edit_token_idx - 1 : edit_token_idx, :]
        end_q = query_states[:, 1:2, edit_token_idx : edit_token_idx + 1, :]
        start_w = compute_attn_weights_iter(rotary_emb, start_q, key_states[:, [0], :edit_token_idx, :], cursor_start).squeeze((1, 2))
        end_w = compute_attn_weights_iter(rotary_emb, end_q, key_states[:, [1], : edit_token_idx + 1, :], cursor_end).squeeze((1, 2))
        end_w = end_w.masked_fill(~compute_end_index_mask(cursor_end), float("-inf"))
        start_logits[edit_token_idx] = start_w
        end_logits[edit_token_idx] = end_w
    return start_logits, end_logits


def _run(operation_case: OperationCase, verbose: bool = False):
    torch.manual_seed(0)
    device = torch.device("cpu")  # forces the non-triton reference path for the training side
    batch_size = 2

    config = AutoIndexerConfig(
        hidden_size=64,
        head_dim=64,
        num_attention_heads=4,
        max_position_embeddings=1000,
        inplace_edits=True,
        mean_insert=2,
    )
    marker_map = DEFAULT_MARKER_MAP

    index_head = AutoIndexerIndexHead(config).to(device).eval()
    rotary_emb = LlamaRotaryEmbedding(config, device)

    tokens, cursor_indices, cursor_dict, perturbations = build_synthetic_sequence(operation_case, marker_map, batch_size, device)
    token_count = tokens.shape[1]

    hidden_states = torch.randn(batch_size, token_count, config.hidden_size, device=device)

    # ---- Training path: sparse per-edit logits (same as compute_index_losses) ----
    parsed_train = get_parsed_positions_from_perturbations(perturbations, device=device)
    with torch.no_grad():
        w_train_start, w_train_end = _training_edit_logits(index_head, hidden_states, parsed_train, rotary_emb)

    # ---- Inference path: step-by-step with the parser + key cache ----
    parser = AutoIndexerIdParser(marker_map, batch_size, device=device)
    cache = IndexKeysCache()
    w_infer_per_step = []
    with torch.no_grad():
        for i in range(token_count):
            parser.update(tokens[:, i], cursor_indices[:, i, :])
            pp = parser.get_parsed_positions()
            w_i, _, _ = index_head(hidden_states[:, i : i + 1], parsed_positions=pp, index_keys_cache=cache, rotary_emb=rotary_emb)  # (B, 2, 1, i+1)
            w_infer_per_step.append(w_i)

    # ---- Compare at the supervised query rows ----
    max_start_err = 0.0
    max_end_err = 0.0
    all_allowed_match = True
    for edit_token_idx, (start_idx, end_idx) in cursor_dict.items():
        k = edit_token_idx + 1  # number of visible keys at the edit token step

        # edit_start (head 0) is predicted by the token *before* the edit marker.
        start_q = edit_token_idx - 1
        train_start = w_train_start[edit_token_idx]
        infer_start = w_infer_per_step[start_q][:, 0, 0, :]
        start_err = _masked_abs_diff(train_start, infer_start)
        max_start_err = max(max_start_err, start_err)

        # edit_end (head 1) is predicted at the edit marker itself.
        train_end = w_train_end[edit_token_idx]
        infer_end = w_infer_per_step[edit_token_idx][:, 1, 0, :]
        end_err = _masked_abs_diff(train_end, infer_end)
        max_end_err = max(max_end_err, end_err)

        # Argmax / allowed-set comparison for edit_end (the symptom the user reported).
        train_allowed = torch.isfinite(train_end)
        infer_allowed = torch.isfinite(infer_end)
        allowed_match = torch.equal(train_allowed, infer_allowed)
        all_allowed_match = all_allowed_match and allowed_match

        if verbose:
            # Diagnostics: relative offsets the edit_end head must choose among, and the relative offset of the true target
            parser_dbg = AutoIndexerIdParser(marker_map, batch_size, device=device)
            for j in range(edit_token_idx + 1):
                parser_dbg.update(tokens[:, j], cursor_indices[:, j, :])
            rel_pos = parser_dbg.get_parsed_positions().key_pos_ids[0]
            allowed_offsets = rel_pos[infer_allowed[0]].tolist()
            target_offset = rel_pos[end_idx].item()
            print(
                f"edit@{edit_token_idx} start={start_idx} end={end_idx} | "
                f"start_logit_err={start_err:.3e} end_logit_err={end_err:.3e} "
                f"end_allowed_sets_match={allowed_match}"
            )
            print(f"    edit_end target relative offset = {target_offset}; allowed offsets = {sorted(allowed_offsets)}")

    if verbose:
        print(f"\nMax edit_start logit error: {max_start_err:.3e}")
        print(f"Max edit_end   logit error: {max_end_err:.3e}")

    return max_start_err, max_end_err, all_allowed_match


@pytest.mark.parametrize("operation_case", OPERATION_CASES)
def test_index_head_train_inference_consistency(operation_case):
    """The index head must produce identical edit_start/edit_end logits whether run in the training (full-sequence) path or the inference (step-by-step) path."""
    max_start_err, max_end_err, all_allowed_match = _run(operation_case, verbose=False)
    assert all_allowed_match, "edit_end candidate (allowed) sets differ between training and inference"
    assert max_start_err < 1e-4, f"edit_start logits diverge between train/inference: {max_start_err:.3e}"
    assert max_end_err < 1e-4, f"edit_end logits diverge between train/inference: {max_end_err:.3e}"


def main():
    for operation_case in OPERATION_CASES:
        print(f"\n=== operations={operation_case.operations} label_len={operation_case.label_len} ===")
        max_start_err, max_end_err, all_allowed_match = _run(operation_case, verbose=True)
        if max_start_err < 1e-4 and max_end_err < 1e-4 and all_allowed_match:
            print("RESULT: training and inference index-head logits MATCH (no implementation inconsistency).")
        else:
            print("RESULT: MISMATCH between training and inference index-head logits -> bug.")


def _masked_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max abs diff over positions finite in both (treat -inf masks as equal)."""
    both_finite = torch.isfinite(a) & torch.isfinite(b)
    mask_mismatch = torch.isfinite(a) ^ torch.isfinite(b)
    if mask_mismatch.any():
        return float("inf")
    if both_finite.any():
        return (a[both_finite] - b[both_finite]).abs().max().item()
    return 0.0


if __name__ == "__main__":
    main()
