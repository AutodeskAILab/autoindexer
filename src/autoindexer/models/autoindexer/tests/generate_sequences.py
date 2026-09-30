"""Shared helper for building synthetic marker-annotated token sequences, used across the
index-head / index-loss test suites (``test_index_loss_parity.py``,
``test_index_head_consistency.py``, ...).

Rather than hand-assembling token buffers from a list of ``Perturbation``s (which each
call site used to duplicate), this drives the sequences through ``batch_perturb_labels``
-- the same machinery used at training time -- for a fixed, deterministic set of edit
``Operation``s.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch

from autoindexer.models.autoindexer.perturb_labels import (
    Perturbation,
    batch_perturb_labels,
    to_operations,
    OperationTuple, perturb_labels_per_sequence, get_parsed_positions_from_perturbations,
)
from autoindexer.models.autoindexer.type_utils import CursorIndices, MarkerMap

DEFAULT_MARKER_MAP: MarkerMap = MarkerMap(edit=300, return_to_end=400, eos=500)

OPERATION_TUPLES_FILE = os.path.join(os.path.dirname(__file__), "cases", "operation_tuples.json")


@dataclass(frozen=True)
class OperationCase:
    operations: List[OperationTuple]
    label_len: int


def load_operation_cases(path: str = OPERATION_TUPLES_FILE) -> List[OperationCase]:
    """Load the pool of diverse edit-``Operation`` tuple sets used to parametrize the
    index-head / index-loss tests, generated (see ``sample_cursor_operations``) rather
    than hand-picked, for broader coverage of insert/delete/extend shapes."""
    with open(path, "r") as f:
        operation_sets = json.load(f)
    return [
        OperationCase(
            operations=[tuple(op) for op in entry["operations"]],
            label_len=entry["label_len"],
        )
        for entry in operation_sets
    ]


OPERATION_CASES: List[OperationCase] = load_operation_cases()


def get_parsed_positions_from_case(operation_case: OperationCase, device):
    operations = to_operations(operation_case.operations)
    perturbations = perturb_labels_per_sequence(operations)
    return get_parsed_positions_from_perturbations(perturbations, device)


def build_synthetic_sequence(
    operation_case: OperationCase,
    marker_map: MarkerMap = DEFAULT_MARKER_MAP,
    batch_size: int = 2,
    device: torch.device = None,
    max_length: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor, CursorIndices, List[Perturbation]]:
    """Build a batch of marker-annotated token sequences (+ cursor indices, in both tensor
    and dict form) for a fixed, deterministic set of edit ``operations``.

    Returns ``(tokens, cursor_indices, cursor_indices_dict, perturbations)``.
    """
    operations = to_operations(operation_case.operations)
    target_labels = torch.arange(operation_case.label_len, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

    tokens, cursor_indices, perturbations = batch_perturb_labels(marker_map, target_labels, max_length=max_length, operations=operations)

    cursor_indices_dict: CursorIndices = {
        pt.edit_token_idx: (pt.start_cursor_idx, pt.end_cursor_idx)
        for pt in perturbations
        if pt.edit_token_idx is not None
    }

    return tokens, cursor_indices, cursor_indices_dict, perturbations
