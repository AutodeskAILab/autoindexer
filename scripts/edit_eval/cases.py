"""Single-edit evaluation cases: a clean token window plus one scripted repair operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from autoindexer.models.autoindexer.perturb_labels import (
    Operation,
    Perturbation,
    batch_perturb_labels,
    perturb_labels_per_sequence,
)
from autoindexer.models.autoindexer.type_utils import CursorIndices, MarkerMap

OP_TYPES = ("insert", "delete", "substitute")
# A fourth evaluation category: an uncorrupted draft, where the correct behaviour is to emit no edit at all.
NO_EDIT = "none"
ALL_CATEGORIES = OP_TYPES + (NO_EDIT,)


@dataclass
class EditCase:
    """One clean window plus the single edit that repairs its corrupted draft."""

    clean: List[int]
    op_type: str
    start: int
    num_insert: int
    num_delete: int
    lag: int
    corruption: List[int]
    source: str = ""
    doc_id: int = -1
    window_offset: int = 0

    @property
    def draft_len(self) -> int:
        return self.start + self.num_delete + self.lag

    @property
    def tail_len(self) -> int:
        return len(self.clean) - (self.start + self.num_insert + self.lag)

    @property
    def gold_span(self) -> Tuple[int, int]:
        """Resolved-position span the edit deletes, `[start, start + num_delete)`."""
        return self.start, self.start + self.num_delete

    @property
    def gold_insert(self) -> List[int]:
        return self.clean[self.start : self.start + self.num_insert]

    @property
    def restored_len(self) -> int:
        """Length of the clean prefix a correct repair reconstructs."""
        return self.start + self.num_insert + self.lag

    def draft(self) -> List[int]:
        return self.clean[: self.start] + list(self.corruption) + self.clean[self.start + self.num_insert : self.restored_len]

    def to_dict(self) -> Dict:
        return {
            "op_type": self.op_type,
            "start": self.start,
            "num_insert": self.num_insert,
            "num_delete": self.num_delete,
            "lag": self.lag,
            "window_len": len(self.clean),
            "draft_len": self.draft_len,
            "source": self.source,
            "doc_id": self.doc_id,
            "window_offset": self.window_offset,
        }


@dataclass
class BuiltStream:
    """A case rendered into the raw marker-annotated stream the model consumes."""

    tokens: torch.Tensor  # (1, T)
    cursor_indices: torch.Tensor  # (1, T, 2)
    cursor_dict: CursorIndices
    perturbations: List[Perturbation]
    edit_idx: Optional[int] = None
    return_idx: Optional[int] = None
    start_cursor: Optional[int] = None
    end_cursor: Optional[int] = None

    @property
    def length(self) -> int:
        return self.tokens.shape[1]


def build_operations(case: EditCase) -> List[Operation]:
    return [
        Operation(num_extend=case.draft_len),
        Operation(
            num_insert=case.num_insert,
            num_delete=case.num_delete,
            start_rel_pos=case.start,
            end_rel_pos=case.start + case.num_delete,
            num_extend=case.tail_len,
        ),
    ]


def case_is_feasible(case: EditCase, min_prefix: int = 1, min_tail: int = 0) -> bool:
    """Whether the case's operation chain satisfies `perturb_labels_per_sequence`'s invariants."""
    if case.op_type == NO_EDIT:
        return (
            case.num_insert == case.num_delete == 0
            and case.start >= min_prefix
            and case.tail_len >= min_tail
            and case.restored_len <= len(case.clean)
        )
    return (
        case.op_type in OP_TYPES
        and case.start >= min_prefix
        and case.num_insert + case.num_delete > 0
        and case.num_delete + case.lag >= 1
        and case.tail_len >= min_tail
        and case.restored_len <= len(case.clean)
    )


def build_stream(case: EditCase, marker_map: MarkerMap, device: torch.device = None) -> BuiltStream:
    """Render `case` into its scripted raw stream, with the corrupted span filled in."""
    assert case_is_feasible(case), f"Infeasible case: {case.to_dict()}"
    assert len(case.corruption) == case.num_delete

    operations = build_operations(case)
    token_count = perturb_labels_per_sequence(operations)[-1].curr_seq_idx
    labels = torch.tensor([case.clean], dtype=torch.long, device=device)
    tokens, cursor_indices, perturbations = batch_perturb_labels(
        marker_map, labels, max_length=token_count + 1, operations=operations
    )
    tokens, cursor_indices = tokens[:, :token_count], cursor_indices[:, :token_count]

    edit_pt = perturbations[1]
    if case.num_delete:
        corruption = torch.tensor(case.corruption, dtype=tokens.dtype, device=tokens.device)
        tokens[0, edit_pt.start_cursor_idx : edit_pt.start_cursor_idx + case.num_delete] = corruption
    assert (tokens >= 0).all(), "Unfilled ignore_index positions left in the scripted stream"

    return BuiltStream(
        tokens=tokens,
        cursor_indices=cursor_indices,
        cursor_dict={edit_pt.edit_token_idx: (edit_pt.start_cursor_idx, edit_pt.end_cursor_idx)},
        perturbations=perturbations,
        edit_idx=edit_pt.edit_token_idx,
        return_idx=edit_pt.return_token_idx,
        start_cursor=edit_pt.start_cursor_idx,
        end_cursor=edit_pt.end_cursor_idx,
    )


def build_clean_stream(clean: Sequence[int], marker_map: MarkerMap, device: torch.device = None) -> BuiltStream:
    """The same window with no edit at all -- the negative control for edit-detection metrics."""
    operations = [Operation(num_extend=len(clean))]
    labels = torch.tensor([list(clean)], dtype=torch.long, device=device)
    tokens, cursor_indices, perturbations = batch_perturb_labels(
        marker_map, labels, max_length=len(clean) + 1, operations=operations
    )
    return BuiltStream(
        tokens=tokens[:, : len(clean)],
        cursor_indices=cursor_indices[:, : len(clean)],
        cursor_dict={},
        perturbations=perturbations,
    )


def poisson_positive(rng: np.random.Generator, mean: float, max_tries: int = 100) -> int:
    """Poisson(mean) conditioned on `k >= 1`."""
    for _ in range(max_tries):
        k = int(rng.poisson(mean))
        if k > 0:
            return k
    return 1


def random_corruption(rng: np.random.Generator, n: int, vocab_ids: np.ndarray) -> List[int]:
    """Uniform random tokens -- what `sanitize_labels` puts in deleted spans during training."""
    return [] if n == 0 else rng.choice(vocab_ids, size=n, replace=True).tolist()


def transplant_corruption(rng: np.random.Generator, n: int, donor: Sequence[int]) -> List[int]:
    """A contiguous span lifted from another document -- fluent text in the wrong place."""
    if n == 0:
        return []
    if len(donor) <= n:
        return list(donor) + random_corruption(rng, n - len(donor), np.asarray(donor))
    offset = int(rng.integers(0, len(donor) - n))
    return list(donor[offset : offset + n])


def sample_case(
    rng: np.random.Generator,
    clean: Sequence[int],
    op_type: str,
    mean_insert: float,
    mean_delete: float,
    corruption_fn,
    lag: Optional[int] = None,
    min_prefix: int = 8,
    min_tail: int = 8,
    max_tries: int = 32,
) -> Optional[EditCase]:
    """Sample one case over `clean`, or None if no feasible (start, lag) exists after retries."""
    length = len(clean)
    if op_type == NO_EDIT:
        # Nothing to corrupt
        for _ in range(max_tries):
            max_lag = length - min_prefix - min_tail
            if max_lag < 1:
                return None
            this_lag = int(rng.integers(1, max_lag + 1)) if lag is None else lag
            max_start = length - this_lag - min_tail
            if max_start < min_prefix:
                continue
            case = EditCase(
                clean=list(clean), op_type=NO_EDIT, start=int(rng.integers(min_prefix, max_start + 1)),
                num_insert=0, num_delete=0, lag=this_lag, corruption=[],
            )
            if case_is_feasible(case, min_prefix=min_prefix, min_tail=min_tail):
                return case
        return None

    for _ in range(max_tries):
        k = poisson_positive(rng, mean_insert if op_type != "delete" else mean_delete)
        num_insert = 0 if op_type == "delete" else k
        num_delete = 0 if op_type == "insert" else k

        min_lag = 1 if num_delete == 0 else 0
        max_lag = length - min_prefix - num_insert - min_tail
        if max_lag < min_lag:
            continue
        this_lag = int(rng.integers(min_lag, max_lag + 1)) if lag is None else lag
        max_start = length - num_insert - this_lag - min_tail
        if not (min_lag <= this_lag <= max_lag) or max_start < min_prefix:
            continue

        start = int(rng.integers(min_prefix, max_start + 1))
        case = EditCase(
            clean=list(clean),
            op_type=op_type,
            start=start,
            num_insert=num_insert,
            num_delete=num_delete,
            lag=this_lag,
            corruption=corruption_fn(rng, num_delete),
        )
        if case_is_feasible(case, min_prefix=min_prefix, min_tail=min_tail):
            return case
    return None
