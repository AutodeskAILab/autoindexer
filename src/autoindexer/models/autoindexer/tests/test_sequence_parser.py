import json
import os
import time

import numpy as np
import torch
from torch import Tensor

import sys
sys.path.append("")

from autoindexer.models.autoindexer.sequence_parser import MarkerMap, AutoIndexerIdParser
from autoindexer.models.autoindexer.type_utils import MarkerSet


LABEL_FILE_NAME = os.path.join(os.path.dirname(__file__), "labels", "position_matrix.json")

def markersets_to_cursor_indices(marker_map: MarkerMap, tokens: np.ndarray | Tensor, markersets: list[MarkerSet]) -> Tensor | np.ndarray:
    batch_size, seq_len = tokens.shape
    device = tokens.device if isinstance(tokens, Tensor) else None
    cursor_indices = torch.zeros((batch_size, seq_len, 2), dtype=torch.long, device=device)
    if isinstance(tokens, Tensor):
        np_tokens = tokens.cpu().data.numpy()
    else:
        np_tokens = tokens
    marker_indices = np.nonzero(np_tokens == marker_map["edit"])

    edit_indices_gen = {batch_idx: iter(markersets) for batch_idx in range(batch_size)}
    for batch_idx, marker_idx in zip(*marker_indices):
        edit_start_idx, edit_end_idx = next(edit_indices_gen[batch_idx])
        assert marker_idx + 1 < seq_len, "Edit token shouldn't appear at the start of the sequence."
        cursor_indices[batch_idx, marker_idx, 0] = edit_start_idx
        cursor_indices[batch_idx, marker_idx + 1, 1] = edit_end_idx
    return cursor_indices


def test_parse_sequence():
    marker_map = MarkerMap(edit="<", return_to_end=">", eos=".")
    labels = "abcdefghijklmnopqrstuvwxyz"
    tokens = "^&pqr<efl@$lmno><abcd>stu%<vwxyz><ghijk>.^&pqr<efl@$lmno><abcd>stu%<vwxyz><ghijk>."
    tokens = np.array(list(tokens))[None, ...]

    markersets = [
        (2, 2),
        (0, 6),
        (25, 26),
        (8, 11),
        (43, 43),
        (41, 47),
        (66, 67),
        (49, 52),
    ]
    cursor_indices = markersets_to_cursor_indices(marker_map, tokens, markersets)

    print(tokens, cursor_indices)

    seq_parser = AutoIndexerIdParser(marker_map, batch_size=1)

    parsed_outputs = seq_parser(tokens, cursor_indices, compute_matrix=True)

    output_strs = ["".join(output.tolist()) for output in parsed_outputs["parsed_sequences"]]
    assert output_strs == [labels], f"Unexpected output: {output_strs}"
    print("Parsed output: ", output_strs)

    print(parsed_outputs["position_matrix"].tolist())
    print(parsed_outputs["attention_mask"].long().tolist())

    with open(LABEL_FILE_NAME, "r") as f:
        expected_positions = json.load(f)

    assert parsed_outputs["position_matrix"].tolist() == expected_positions["position_matrix"]
    assert parsed_outputs["attention_mask"].tolist() == expected_positions["attention_mask"]


def test_finished_sequence_forces_eos_instead_of_ret():
    """Regression: after EOS, sampling RET leaves zero visible keys and NaN attention."""
    marker_map = MarkerMap(edit=100, return_to_end=101, eos=102)
    eos_token_id = marker_map["eos"]
    parser = AutoIndexerIdParser(marker_map, batch_size=2, device="cpu")
    prefix = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    cursor_indices = np.zeros((2, 20, 2), dtype=np.int64)
    parser(prefix, cursor_indices[:, :5])

    is_finished = np.zeros(2, dtype=bool)
    for token in (eos_token_id, marker_map["return_to_end"], marker_map["return_to_end"]):
        new_tokens = np.array([token, 7], dtype=np.int64)
        sampled = np.where(is_finished, eos_token_id, new_tokens)
        is_finished = is_finished | (sampled == eos_token_id)
        parser.update(sampled, np.zeros((2, 2), dtype=np.int64))
        attn_sums = parser.get_parsed_positions().attn_mask.sum(dim=1)
        assert (attn_sums > 0).all(), f"zero visible keys after tokens={sampled.tolist()}"


def test_edit_edit_ret_keeps_visible_keys():
    """Regression: nested edits closed by RET must not zero out all attention keys."""
    marker_map = MarkerMap(edit=2, return_to_end=3, eos=4)
    parser = AutoIndexerIdParser(marker_map, batch_size=1, device="cpu")
    parser(torch.tensor([[0]]), torch.zeros(1, 20, 2, dtype=torch.long))
    for tok, idx in ((2, [0, 0]), (2, [0, 1]), (3, [0, 2])):
        parser.update(torch.tensor([tok]), torch.tensor([idx]))
        attn_sum = parser.get_parsed_positions().attn_mask.sum().item()
        assert attn_sum > 0, f"zero visible keys after token={tok}"


if __name__ == "__main__":
    marker_map = MarkerMap(edit="<", return_to_end=">", eos=".")
    labels = "abcdefghijklmnopqrstuvwxyz"
    tokens = "^&pqr<efl?$lmno><abcd>stu%<vwxyz><ghijk>.^&pqr<efl@$lmno><abcd>stu%<vwxyz><ghijk>."
    tokens = np.array(list(tokens))[None, ...]

    markersets = [
        (2, 2),
        (0, 6),
        (25, 26),
        (8, 11),
        (43, 43),
        (41, 47),
        (66, 67),
        (49, 52),
    ]

    cursor_indices = markersets_to_cursor_indices(marker_map, tokens, markersets)

    print(tokens, cursor_indices)

    seq_parser = AutoIndexerIdParser(marker_map, batch_size=1, parse_output=True)

    start_time = time.time()

    parsed_outputs = seq_parser(tokens, cursor_indices, compute_matrix=True)

    end_time = time.time()
    print(f"Parsing took {end_time - start_time:.4f} seconds")

    output_strs = ["".join(output.tolist()) for output in parsed_outputs["parsed_sequences"]]
    assert output_strs == [labels], f"Unexpected output: {output_strs}"
    print("Parsed output: ", output_strs)

    print(parsed_outputs["position_matrix"].tolist())
    print(parsed_outputs["attention_mask"].long().tolist())

    with open(LABEL_FILE_NAME, "r") as f:
        expected_positions = json.load(f)

    assert parsed_outputs["position_matrix"].tolist() == expected_positions["position_matrix"]
    assert parsed_outputs["attention_mask"].tolist() == expected_positions["attention_mask"]


def test_get_parsed_positions_default_editable_start_leaves_index_attn_mask_unrestricted():
    """No `editable_start` passed in (every existing pretraining config) must keep the index head's view identical to the backbone's."""
    marker_map = MarkerMap(edit="<", return_to_end=">", eos=".")
    parser = AutoIndexerIdParser(marker_map, batch_size=1)
    for token in "abcdef":
        parser.update(np.array([token]), np.array([[-100, -100]]))

    parsed = parser.get_parsed_positions()
    assert torch.equal(parsed.index_attn_mask, parsed.attn_mask)


def test_get_parsed_positions_editable_start_locks_index_head_but_not_attn_mask():
    """`editable_start` must narrow `index_attn_mask` to the current turn while leaving the backbone's own `attn_mask` untouched."""
    marker_map = MarkerMap(edit="<", return_to_end=">", eos=".")
    editable_start = torch.tensor([3])
    parser = AutoIndexerIdParser(marker_map, batch_size=1, editable_start=editable_start)
    for token in "abcdef":  # 6 plain tokens at positions 0..5
        parser.update(np.array([token]), np.array([[-100, -100]]))

    parsed = parser.get_parsed_positions()
    assert parsed.attn_mask.tolist() == [[True] * 6]
    assert parsed.index_attn_mask.tolist() == [[False, False, False, True, True, True]]
