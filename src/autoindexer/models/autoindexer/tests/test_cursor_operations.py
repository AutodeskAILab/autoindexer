from autoindexer.models.autoindexer.perturb_labels import perturb_labels_per_sequence, sample_cursor_operations, batch_perturb_labels
from autoindexer.models.autoindexer.type_utils import MarkerMap
import torch


def test_cursor_operations():
    for _ in range(5):
        seq_len = 100
        min_len = 10
        mean_num_edits = 10
        max_insert = 10
        max_delete = 5
        max_extend = 15
        operations, token_count = sample_cursor_operations(seq_len, min_len, mean_num_edits, max_insert, max_delete, max_extend)

        curr_seq_len = 0
        for i, op in enumerate(operations):
            if i == 0:
                assert op.num_extend >= min_len, f"First operation must ensure min_len, got num_extend {op.num_extend}"
                assert op.num_insert == 0 and op.num_delete == 0, "First operation cannot have insertions or deletions"
                assert op.start_rel_pos == 0 and op.end_rel_pos == 0, "First operation must have start and end positions at 0"
            else:
                assert 0 <= op.start_rel_pos <= curr_seq_len, f"start_rel_pos {op.start_rel_pos} out of bounds for current seq_len {curr_seq_len}"
                assert op.start_rel_pos <= op.end_rel_pos <= curr_seq_len, f"end_rel_pos {op.end_rel_pos} out of bounds for current seq_len {curr_seq_len}"
            new_seq_len = curr_seq_len + op.num_insert - op.num_delete + op.num_extend
            curr_seq_len = new_seq_len

        assert curr_seq_len == seq_len, f"Final sequence length {curr_seq_len} does not match original {seq_len}"


def test_cursor_operations_sort_edits():
    configs = [
        dict(seq_len=100, min_len=10, mean_num_edits=10, mean_insert=10, mean_delete=5, mean_extend=15),
        dict(seq_len=50, min_len=5, mean_num_edits=5, mean_insert=5, mean_delete=3, mean_extend=7),
        dict(seq_len=200, min_len=20, mean_num_edits=20, mean_insert=15, mean_delete=10, mean_extend=5),
        # Delete-heavy configuration, where individual edits can shrink the sequence.
        dict(seq_len=30, min_len=5, mean_num_edits=8, mean_insert=2, mean_delete=6, mean_extend=1),
        # Inplace edits (num_insert == num_delete for every operation).
        dict(seq_len=100, min_len=5, mean_num_edits=6, mean_insert=4, mean_delete=4, mean_extend=3, inplace_edits=True),
        # Small sequence with many edits, likely to trigger the min_len early-stop.
        dict(seq_len=15, min_len=3, mean_num_edits=15, mean_insert=3, mean_delete=2, mean_extend=1),
        # Adversarial delete-heavy configuration that previously required frequent fallback.
        dict(seq_len=20, min_len=1, mean_num_edits=30, mean_insert=1, mean_delete=8, mean_extend=1),
        # Training-like inplace configuration.
        dict(seq_len=100, min_len=15, mean_num_edits=10, mean_insert=2, mean_delete=2, mean_extend=0, inplace_edits=True),
    ]

    for config in configs:
        for _ in range(50):
            operations, token_count = sample_cursor_operations(sort_edits=True, **config)

            # The crucial invariant: every edit must have enough sequence length to operate on
            # (end_rel_pos <= len(token_indices)) at the time it is applied.
            perturbations = perturb_labels_per_sequence(operations)
            assert len(perturbations) == len(operations)

            curr_seq_len = 0
            for i, op in enumerate(operations):
                if i == 0:
                    assert op.num_insert == 0 and op.num_delete == 0, "First operation cannot have insertions or deletions"
                    assert op.start_rel_pos == 0 and op.end_rel_pos == 0, "First operation must have start and end positions at 0"
                else:
                    # start_rel_pos must be strictly within bounds (it indexes directly into token_indices)
                    assert 0 <= op.start_rel_pos < curr_seq_len, f"start_rel_pos {op.start_rel_pos} out of bounds for current seq_len {curr_seq_len}"
                    assert op.start_rel_pos <= op.end_rel_pos <= curr_seq_len, f"end_rel_pos {op.end_rel_pos} out of bounds for current seq_len {curr_seq_len}"
                curr_seq_len += op.num_insert - op.num_delete + op.num_extend

            assert curr_seq_len == config["seq_len"], f"Final sequence length {curr_seq_len} does not match original {config['seq_len']}"


def test_sample_cursor_operations_respects_editable_start():
    """No edit's start/end may reference a position before `editable_start`."""
    editable_start = 40
    for _ in range(200):
        operations, _ = sample_cursor_operations(
            seq_len=100, min_len=5, mean_num_edits=10, mean_insert=5, mean_delete=3, mean_extend=7,
            edit_type_ratio=(0.15, 0.7, 0.15), editable_start=editable_start,
        )
        # The seed (first-applied) operation's extend must itself cover the whole locked region
        assert operations[0].num_extend >= editable_start
        for op in operations[1:]:
            assert op.start_rel_pos >= editable_start
            assert op.end_rel_pos >= editable_start


def test_batch_perturb_labels_respects_editable_start():
    marker_map = MarkerMap(edit=300, return_to_end=400, eos=500)
    editable_start = 50
    labels = torch.arange(120).unsqueeze(0)
    for _ in range(200):
        _, cursor_indices, _ = batch_perturb_labels(
            marker_map,
            labels,
            max_length=400,
            return_indices_as_dict=True,
            editable_start=editable_start,
            min_len=5,
            mean_num_edits=10,
            mean_insert=5,
            mean_delete=3,
            mean_extend=7,
            edit_type_ratio=(0.15, 0.7, 0.15),
        )
        for start_idx, end_idx in cursor_indices.values():
            assert start_idx >= editable_start
            assert end_idx >= editable_start


def test_editable_start_zero_is_unconstrained():
    """`editable_start=0` (the default) must reproduce the original, unconstrained sampling"""
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)
    baseline, _ = sample_cursor_operations(seq_len=100, min_len=10, mean_num_edits=10, mean_insert=10, mean_delete=5, mean_extend=15)

    random.seed(0)
    np.random.seed(0)
    with_default, _ = sample_cursor_operations(
        seq_len=100, min_len=10, mean_num_edits=10, mean_insert=10, mean_delete=5, mean_extend=15, editable_start=0
    )
    assert baseline == with_default
