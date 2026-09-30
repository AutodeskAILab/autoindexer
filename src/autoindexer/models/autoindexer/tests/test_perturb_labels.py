import json
import os
import time
from dataclasses import asdict
from typing import List

import torch

import sys

sys.path.append("")

from autoindexer.models.autoindexer.tests.test_sequence_parser import markersets_to_cursor_indices

from autoindexer.models.autoindexer.perturb_labels import (
    batch_perturb_labels,
    perturb_labels_per_sequence,
    get_parsed_positions_from_perturbations,
    sample_cursor_operations,
    transplant_corrupted_spans,
    Perturbation,
    to_operations,
    Operation,
)
from autoindexer.models.autoindexer.sequence_parser import AutoIndexerIdParser
from autoindexer.models.autoindexer.tests.generate_sequences import DEFAULT_MARKER_MAP
from autoindexer.models.autoindexer.type_utils import MarkerSet


POS_MATRIX_LABEL_FILE_NAME = os.path.join(os.path.dirname(__file__), "labels", "position_matrix.json")
PARSED_POS_LABEL_FILE_NAME = os.path.join(os.path.dirname(__file__), "labels", "parsed_positions.json")
BATCH_PERTURB_LABEL_FILE_NAME = os.path.join(os.path.dirname(__file__), "labels", "batch_perturb.json")


def perturbations_to_markersets(perturbations: List[Perturbation]) -> List[MarkerSet]:
    markersets = []
    for pt in perturbations:
        if pt.edit_token_idx is not None:
            markersets.append((pt.start_cursor_idx, pt.end_cursor_idx))
    return markersets


def test_perturb_labels():
    operations = to_operations([(0, 0, 0, 0, 5), (9, 0, 2, 2, 0), (4, 2, 0, 2, 4), (5, 1, 19, 20, 0), (5, 3, 6, 9, 0)])

    perturbations = perturb_labels_per_sequence(operations)
    target_results = [
        Perturbation([0, 1, 2, 3, 4], 0, 0, None, None, None, None, 5, Operation(0, 0, 0, 0, 5)),
        Perturbation([0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2, 3, 4], 5, 14, 5, 2, 2, 15, 16, Operation(9, 0, 2, 2, 0)),
        Perturbation([17, 18, 19, 20, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2, 3, 4, 22, 23, 24, 25], 14, 16, 16, 0, 6, 21, 26, Operation(4, 2, 0, 2, 4)),
        Perturbation([17, 18, 19, 20, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2, 3, 4, 22, 23, 24, 27, 28, 29, 30, 31], 20, 24, 26, 25, 26, 32, 33, Operation(5, 1, 19, 20, 0)),
        Perturbation([17, 18, 19, 20, 6, 7, 34, 35, 36, 37, 38, 11, 12, 13, 14, 2, 3, 4, 22, 23, 24, 27, 28, 29, 30, 31], 24, 26, 33, 8, 11, 39, 40, Operation(5, 3, 6, 9, 0)),
    ]

    for i, (perturbation, target_result) in enumerate(zip(perturbations, target_results)):
        assert perturbation == target_result, f"Unexpected output: {perturbation}"


def test_random_sample_perturbation():
    target_str = "abcdefghijklmnopqrstuvwxyz"

    start_time = time.time()
    operations, token_count = sample_cursor_operations(len(target_str), 5, 4, 5, 3, 5)
    print(f"Time taken: {time.time() - start_time:.4f} seconds")
    print(operations)

    start_time = time.time()
    print(perturb_labels_per_sequence(operations))
    print(f"Time taken: {time.time() - start_time:.4f} seconds")


def test_batch_perturb_labels():
    target_str = "abcdefghijklmnopqrstuvwxyz"
    target_labels = [ord(c) for c in target_str]  # Convert to ASCII values for simplicity
    batch_target_labels = [
        target_labels + [500] + target_labels[:2],
        target_labels[:-2] + [500] + target_labels[:4],
        target_labels[:-4] + [500] + target_labels[:6],
        target_labels[:-6] + [500] + target_labels[:8],
        target_labels[:-8] + [500] + target_labels[:10],
        target_labels[:-10] + [500] + target_labels[:12],
        target_labels[:-12] + [500] + target_labels[:14],
        target_labels[:-14] + [500] + target_labels[:16],
        target_labels[:-16] + [500] + target_labels[:18],
        target_labels[:-18] + [500] + target_labels[:20],
        target_labels[:-20] + [500] + target_labels[:22],
        target_labels[:-22] + [500] + target_labels[:24],
    ]
    batch_target_labels = torch.tensor(batch_target_labels, dtype=torch.int32)
    marker_map = DEFAULT_MARKER_MAP

    operations = to_operations([(0, 0, 0, 0, 5), (9, 0, 2, 2, 0), (4, 2, 0, 2, 4), (5, 1, 19, 20, 0), (5, 3, 6, 9, 0)])

    target_sequences, cursor_indices, perturbations = batch_perturb_labels(marker_map, batch_target_labels, max_length=60, operations=operations)

    with open(BATCH_PERTURB_LABEL_FILE_NAME, "r") as f:
        expected = json.load(f)
    assert target_sequences.tolist() == expected["expected_sequence"], f"Unexpected sequence: {target_sequences[0].tolist()}"

    markersets = perturbations_to_markersets(perturbations)
    print(markersets)

    assert markersets == [(2, 2), (0, 6), (25, 26), (8, 11)]

    parsed_sequences = AutoIndexerIdParser.parse_sequence(marker_map, target_sequences, cursor_indices)

    output_strs = ["".join([chr(c) for c in seq.tolist()]) for seq in parsed_sequences]
    target_strs = [target_str[: 26 - 2 * i] for i in range(12)]
    print("Parsed outputs: ", output_strs)
    assert output_strs == target_strs, f"Unexpected output: {output_strs}"

    parsed_positions = get_parsed_positions_from_perturbations(perturbations)

    token_count = perturbations[-1].curr_seq_idx
    position_matrix, mask_matrix = parsed_positions.get_matrices(token_count)

    with open(POS_MATRIX_LABEL_FILE_NAME, "r") as f:
        expected = json.load(f)
        expected_position_matrix = torch.tensor(expected["position_matrix"])[0, :token_count, :token_count]
        expected_attention_mask = torch.tensor(expected["attention_mask"])[0, :token_count, :token_count]

    assert torch.equal(position_matrix, expected_position_matrix), f"Position matrix {position_matrix} do not match expected output."
    assert torch.equal(mask_matrix, expected_attention_mask), f"Attention mask {mask_matrix} do not match expected output."

    parsed_positions = [
        {k: v.long().tolist() if isinstance(v, torch.Tensor) else v for k, v in asdict(positions).items() if k not in ("is_edit_token", "block_type")} for positions in parsed_positions
    ]
    print("Parsed position:", parsed_positions)

    with open(PARSED_POS_LABEL_FILE_NAME, "r") as f:
        expected_positions = json.load(f)["expected_positions"]

    print("Expected position:", expected_positions)

    assert parsed_positions == expected_positions, f"Parsed positions {parsed_positions} do not match expected output."


def test_batch_perturb_labels_random():
    batch_size = 16
    marker_map = DEFAULT_MARKER_MAP
    target_labels = torch.arange(100, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    for i in range(10):
        # target_labels has no explicit eos token, but is fixed-size (always exactly 100 tokens)
        target_sequences, cursor_indices, perturbations = batch_perturb_labels(
            marker_map, target_labels, max_length=300, min_len=0, mean_insert=3, mean_delete=3, mean_extend=3, append_eos=True
        )
        print(i)

        parsed_outputs = AutoIndexerIdParser.parse_sequence(marker_map, target_sequences, cursor_indices)
        parsed_outputs = torch.stack(parsed_outputs)
        assert torch.all(parsed_outputs == target_labels), f"Parsed output {parsed_outputs.tolist()} does not match target labels {target_labels.tolist()}"


def _one_deletion_perturbation() -> Perturbation:
    """A single edit with a real deletion span, for the `transplant_corrupted_spans` tests below."""
    operations = to_operations([(0, 0, 0, 0, 5), (0, 3, 2, 5, 3)])
    perturbations = perturb_labels_per_sequence(operations)
    deletion = next(pt for pt in perturbations if pt.operation.num_delete > 0)
    assert deletion.start_cursor_idx is not None
    return perturbations, deletion


def test_transplant_corrupted_spans_splices_donor_content():
    perturbations, deletion = _one_deletion_perturbation()
    seq_len = perturbations[-1].curr_seq_idx
    batch_size = 4
    s, k = deletion.start_cursor_idx, deletion.operation.num_delete

    # Sentinel-valued tokens/donor so the two content pools never overlap by chance.
    tokens = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    donor = 10_000 + torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)

    torch.manual_seed(0)
    out = transplant_corrupted_spans(tokens, perturbations, donor, ignore_index=-100, transplant_prob=1.0)

    assert out.shape == tokens.shape
    # Everything outside the deleted span is untouched.
    outside = torch.ones(seq_len, dtype=torch.bool)
    outside[s : s + k] = False
    assert torch.equal(out[:, outside], tokens[:, outside])
    # The span itself came entirely from `donor`'s sentinel range, and is contiguous within a
    # single donor row (not, e.g., scattered token-by-token).
    span = out[:, s : s + k]
    assert bool((span >= 10_000).all())
    for row in range(batch_size):
        row_vals = span[row].tolist()
        donor_row = donor[row].tolist()
        # `row_vals` must appear as a contiguous run inside `donor_row`.
        assert any(donor_row[o : o + k] == row_vals for o in range(len(donor_row) - k + 1))


def test_transplant_corrupted_spans_noop_at_zero_prob():
    perturbations, _ = _one_deletion_perturbation()
    seq_len = perturbations[-1].curr_seq_idx
    tokens = torch.arange(3 * seq_len).reshape(3, seq_len)
    donor = 10_000 + torch.arange(3 * seq_len).reshape(3, seq_len)

    out = transplant_corrupted_spans(tokens, perturbations, donor, ignore_index=-100, transplant_prob=0.0)
    assert torch.equal(out, tokens)

    # Same when there's no donor to splice from at all.
    out_no_donor = transplant_corrupted_spans(tokens, perturbations, donor=None, ignore_index=-100, transplant_prob=1.0)
    assert torch.equal(out_no_donor, tokens)


def test_transplant_corrupted_spans_falls_back_on_ignore_index_and_wraps_donor_rows():
    perturbations, deletion = _one_deletion_perturbation()
    seq_len = perturbations[-1].curr_seq_idx
    s, k = deletion.start_cursor_idx, deletion.operation.num_delete
    batch_size = 4

    tokens = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    # A single-row donor (fewer rows than `tokens`) that is *entirely* `ignore_index` inside the deletable window
    donor = torch.full((1, seq_len), -100, dtype=torch.long)

    torch.manual_seed(0)
    out = transplant_corrupted_spans(tokens, perturbations, donor, ignore_index=-100, transplant_prob=1.0)

    # Falls back to the pre-transplant filler rather than leaking -100 into the stream.
    assert torch.equal(out[:, s : s + k], tokens[:, s : s + k])
    assert not bool((out == -100).any())


if __name__ == "__main__":
    test_random_sample_perturbation()
    test_perturb_labels()
    test_batch_perturb_labels()
    test_batch_perturb_labels_random()
    test_transplant_corrupted_spans_splices_donor_content()
    test_transplant_corrupted_spans_noop_at_zero_prob()
    test_transplant_corrupted_spans_falls_back_on_ignore_index_and_wraps_donor_rows()