"""Free-generation measurements: hand the model a corrupted draft and see if it repairs it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import GenerationConfig

from autoindexer.models.autoindexer.sequence_parser import AutoIndexerIdParser

from .cases import EditCase


@dataclass
class EditRecord:
    """One edit the model issued, in resolved-sequence coordinates at the time it was made."""

    step: int
    start: int
    end: Optional[int] = None
    inserted: List[int] = field(default_factory=list)
    closed: bool = False

    @property
    def num_delete(self) -> Optional[int]:
        return None if self.end is None else self.end - self.start


def replay_edit_stream(marker_map, tokens: Sequence[int], indices: Sequence[Sequence[int]], device=None, prompt_len: int = 0):
    """Re-parse a generated raw stream, recording each edit's resolved span and insertion."""
    parser = AutoIndexerIdParser(marker_map, batch_size=1, device=device)
    records: List[EditRecord] = []
    open_record: Optional[EditRecord] = None
    appended = 0

    for step, (token, index) in enumerate(zip(tokens, indices)):
        token = int(token)
        if open_record is not None and open_record.end is None:
            end_idx = int(index[1])
            open_record.end = int(
                parser.curr_end_pos[0] + 1 if end_idx == step - 1 else parser.position_ids[0, end_idx]
            )

        parser.update(
            torch.tensor([token], dtype=torch.long, device=device),
            torch.tensor([list(index)], dtype=torch.long, device=device),
        )

        if token == marker_map["edit"]:
            open_record = EditRecord(step=step, start=int(parser.curr_pos[0]) + 1)
            records.append(open_record)
        elif token in (marker_map["return_to_end"], marker_map["eos"]):
            if open_record is not None:
                open_record.closed = token == marker_map["return_to_end"]
            open_record = None
        elif open_record is not None:
            open_record.inserted.append(token)
        elif step >= prompt_len:
            appended += 1

    parser.resolve_parsing(torch.long)
    resolved = parser.parsed_sequences[0]
    return records, [int(t) for t in resolved.tolist()], appended


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (x != y))
        prev = curr
    return prev[-1]


def _build_generation_config(
    model, max_new_tokens: int,
    do_sample: bool, temperature: float, top_p: float, top_k: int,
    marker_calibration_weight: float, greedy_mid_edit: bool, max_delete_span: Optional[int],
    calibration_bias: float, exclude_prompt_from_edits: bool, marker_top_p: float = 1.0,
    marker_temperature: float = 1.0,
) -> GenerationConfig:
    """Shared by `rollout`/`batch_rollout`."""
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        return_dict_in_generate=True,
        marker_calibration_weight=marker_calibration_weight,
        greedy_mid_edit=greedy_mid_edit,
        max_delete_span=max_delete_span,
        calibration_bias=calibration_bias,
        exclude_prompt_from_edits=exclude_prompt_from_edits,
        marker_top_p=marker_top_p,
        marker_temperature=marker_temperature,
    )


def _truncate_at_eos(tokens: List[int], indices: List[List[int]], eos: int, prompt_len: int):
    for i in range(prompt_len, len(tokens)):
        if tokens[i] == eos:
            return tokens[: i + 1], indices[: i + 1]
    return tokens, indices


@torch.no_grad()
def rollout(
    model, draft: Sequence[int], device, max_new_tokens: int,
    do_sample: bool, temperature: float, top_p: float, top_k: int = 0,
    marker_calibration_weight: float = 0.0, greedy_mid_edit: bool = False, max_delete_span: int = None,
    calibration_bias: float = 0.0, exclude_prompt_from_edits: bool = False, marker_top_p: float = 1.0,
    marker_temperature: float = 1.0,
):
    """Generate a continuation of `draft` and return `(raw_tokens, sampled_indices)`."""
    input_ids = torch.tensor([list(draft)], dtype=torch.long, device=device)
    generation_config = _build_generation_config(
        model, max_new_tokens, do_sample, temperature, top_p, top_k,
        marker_calibration_weight, greedy_mid_edit, max_delete_span, calibration_bias,
        exclude_prompt_from_edits, marker_top_p, marker_temperature,
    )
    output = model.generate(input_ids, generation_config=generation_config)

    tokens = output.sampled_tokens[0].tolist()
    indices = output.sampled_indices[0].tolist()
    return _truncate_at_eos(tokens, indices, model.marker_map["eos"], len(draft))


@torch.no_grad()
def batch_rollout(
    model, drafts: Sequence[Sequence[int]], device, max_new_tokens: int,
    do_sample: bool, temperature: float, top_p: float, top_k: int = 0,
    marker_calibration_weight: float = 0.0, greedy_mid_edit: bool = False, max_delete_span: int = None,
    calibration_bias: float = 0.0, exclude_prompt_from_edits: bool = False, marker_top_p: float = 1.0,
    marker_temperature: float = 1.0,
) -> List[Tuple[List[int], List[List[int]]]]:
    """`rollout`, batched over several `drafts` (of possibly different lengths) in one `generate()` call."""
    pad_id = int(model.non_special_token_ids[0])
    lens = [len(draft) for draft in drafts]
    max_len = max(lens)
    batch_size = len(drafts)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    pad_lens = [max_len - length for length in lens]
    for i, (draft, pad_len) in enumerate(zip(drafts, pad_lens)):
        input_ids[i, pad_len:] = torch.tensor(list(draft), dtype=torch.long, device=device)
        attention_mask[i, pad_len:] = True

    generation_config = _build_generation_config(
        model, max_new_tokens, do_sample, temperature, top_p, top_k,
        marker_calibration_weight, greedy_mid_edit, max_delete_span, calibration_bias,
        exclude_prompt_from_edits, marker_top_p, marker_temperature,
    )
    output = model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config)

    eos = model.marker_map["eos"]
    results = []
    for i, (draft, pad_len) in enumerate(zip(drafts, pad_lens)):
        # Slice off this row's own left-pad prefix, then rebase its cursor indices to it.
        tokens = output.sampled_tokens[i, pad_len:].tolist()
        indices = (output.sampled_indices[i, pad_len:] - pad_len).tolist()
        results.append(_truncate_at_eos(tokens, indices, eos, len(draft)))
    return results


def classify(case: EditCase, records: List[EditRecord], dist_before: int, dist_after: int) -> str:
    """Coarse outcome label for the qualitative breakdown."""
    if not records:
        return "no_edit"
    if dist_after == 0:
        return "repaired"
    first = records[0]
    gold_start, gold_end = case.gold_span
    if first.start != gold_start:
        return "wrong_position"
    if first.num_delete != case.num_delete:
        return "over_delete" if (first.num_delete or 0) > case.num_delete else "under_delete"
    if dist_after < dist_before:
        return "partial_repair"
    return "wrong_content" if dist_after == dist_before else "made_worse"


def measure_rollout(case: EditCase, marker_map, tokens: Sequence[int], indices: Sequence[Sequence[int]], device=None) -> Dict:
    """Localization / repair metrics for one generated rollout."""
    draft, gold_start, gold_end = case.draft(), *case.gold_span
    records, resolved, appended = replay_edit_stream(marker_map, tokens, indices, device=device, prompt_len=len(draft))
    reference = case.clean[: case.restored_len]

    # Drop the free continuation the model wrote past the draft's end before scoring the repair.
    dist_before = levenshtein(draft, reference)
    dist_after = levenshtein(resolved[: max(0, len(resolved) - appended)], reference)

    out = {
        "num_edits": len(records),
        "num_new_tokens": len(tokens) - len(draft),
        "dist_before": dist_before,
        "dist_after": dist_after,
        "repair_gain": (dist_before - dist_after) / dist_before if dist_before else float("nan"),
        "repair_exact": dist_after == 0,
        "resolved_len": len(resolved),
    }

    if records:
        first = records[0]
        out.update(
            {
                "first_edit_delay": first.step - len(draft),
                "gen_start": first.start,
                "gen_end": first.end,
                "gen_num_delete": first.num_delete,
                "gen_num_insert": len(first.inserted),
                "gen_start_err": first.start - gold_start,
                "gen_start_correct": first.start == gold_start,
                "gen_span_correct": first.start == gold_start and first.end == gold_end,
                "gen_insert_correct": first.inserted == case.gold_insert,
                "gen_span_iou": _span_iou((first.start, first.end or first.start), (gold_start, gold_end)),
            }
        )
    out["outcome"] = classify(case, records, dist_before, dist_after)
    return out


def measure_clean_rollout(case: EditCase, marker_map, tokens: Sequence[int], indices: Sequence[Sequence[int]], device=None) -> Dict:
    """Metrics for a rollout on a draft with nothing wrong with it -- `no_edit` is success here."""
    draft = case.draft()
    records, resolved, appended = replay_edit_stream(marker_map, tokens, indices, device=device, prompt_len=len(draft))
    reference = case.clean[: case.restored_len]
    dist_after = levenshtein(resolved[: max(0, len(resolved) - appended)], reference)

    out = {
        "num_edits": len(records),
        "num_new_tokens": len(tokens) - len(draft),
        "dist_before": 0,
        "dist_after": dist_after,
        "repair_exact": dist_after == 0,
        "resolved_len": len(resolved),
    }
    if records:
        first = records[0]
        out.update({
            "first_edit_delay": first.step - len(draft),
            "gen_start": first.start,
            "gen_end": first.end,
            "gen_num_delete": first.num_delete,
            "gen_num_insert": len(first.inserted),
        })
        out["outcome"] = "spurious_harmless" if dist_after == 0 else "spurious_damaging"
    else:
        out["outcome"] = "no_edit"
    return out


def _span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """IoU of two half-open spans; empty-vs-empty counts as 1.0 iff they start at the same place."""
    if a[0] >= a[1] and b[0] >= b[1]:
        return 1.0 if a[0] == b[0] else 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0
