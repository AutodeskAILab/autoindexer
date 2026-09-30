"""Teacher-forced measurements over a case's scripted edit stream."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from autoindexer.models.autoindexer.type_utils import MarkerType
from autoindexer.models.autoindexer.perturb_labels import get_parsed_positions_from_perturbations

from .cases import BuiltStream, EditCase

TOP_KS = (1, 5, 10)
NEAR_MISS_RADII = (0, 1, 2, 5)


@torch.no_grad()
def run_stream(model, built: BuiltStream, device: torch.device, want_index: bool = True):
    """Forward the scripted stream; return `(logprobs, marker_logits, index_weights, parsed_positions)`."""
    tokens = built.tokens.to(device)
    parsed_positions = get_parsed_positions_from_perturbations(built.perturbations, device=device)
    token_logits, hidden_states, _ = model.decode(input_ids=tokens, parsed_positions=parsed_positions, return_dict=True)
    logprobs = torch.log_softmax(token_logits.float(), dim=-1)[0]
    marker_logits = model.marker_head(hidden_states)[0]

    index_weights = None
    if want_index and built.edit_idx is not None:
        index_weights, _, _ = model.index_head(
            hidden_states,
            parsed_positions=parsed_positions,
            cursor_indices=built.cursor_dict,
            rotary_emb=model.rotary_emb,
            return_weights=True,
        )
    return logprobs, marker_logits, index_weights, parsed_positions


def _prefix_marker_states(tokens: torch.Tensor, marker_map, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(is_mid_edit, num_visible)` after each token, aligned with `logprobs[t]` / `marker_logits[t]`."""
    toks = tokens[0].tolist()
    is_mid: List[bool] = []
    visible: List[int] = []
    mid = False
    vis = 0
    for tok in toks:
        if tok == marker_map["edit"]:
            mid = True
        elif tok in (marker_map["return_to_end"], marker_map["eos"]):
            mid = False
        elif not mid:
            vis += 1
        is_mid.append(mid)
        visible.append(vis)
    return (
        torch.tensor(is_mid, device=device, dtype=torch.bool),
        torch.tensor(visible, device=device, dtype=torch.long),
    )


def _marker_softmax(model, marker_logits: torch.Tensor, is_mid: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Softmax over `MarkerType` after masking illegal classes to `-inf` (`marker_logit_mask`)."""
    illegal = model.marker_logit_mask(is_mid, visible)
    return torch.softmax(marker_logits.masked_fill(illegal, float("-inf")), dim=-1)


def edit_marker_trace(model, marker_logits: torch.Tensor, tokens: torch.Tensor, marker_map, upto: int) -> torch.Tensor:
    """P(emit the edit marker) at each prefix `0..upto - 1`, from `marker_head`."""
    is_mid, visible = _prefix_marker_states(tokens, marker_map, marker_logits.device)
    rows = marker_logits[:upto].float()
    probs = _marker_softmax(model, rows, is_mid[:upto], visible[:upto])
    return probs[:, MarkerType.EDIT]


def marker_prob_at(model, marker_logits: torch.Tensor, tokens: torch.Tensor, marker_map, idx: int, marker_class: int) -> float:
    is_mid, visible = _prefix_marker_states(tokens, marker_map, marker_logits.device)
    probs = _marker_softmax(model, marker_logits[idx : idx + 1].float(), is_mid[idx : idx + 1], visible[idx : idx + 1])
    return float(probs[0, marker_class])


def _marker_decision(
    model,
    marker_logits_row: torch.Tensor,
    is_mid_row: torch.Tensor,
    visible_row: torch.Tensor,
    edit_logit_bonus: float = 0.0,
) -> int:
    """Argmax class of the masked marker softmax -- the greedy proxy for the firing decision."""
    illegal = model.marker_logit_mask(is_mid_row, visible_row)
    logits = marker_logits_row.float().masked_fill(illegal, float("-inf")).clone()
    logits[..., MarkerType.EDIT] = logits[..., MarkerType.EDIT] + edit_logit_bonus
    return int(torch.argmax(logits, dim=-1))


def _distribution_metrics(logprobs: torch.Tensor, target: int, prefix: str) -> Dict:
    """Accuracy/rank/NLL/near-miss mass of a 1-D log-distribution against `target`."""
    valid = torch.isfinite(logprobs)
    order = torch.argsort(logprobs, descending=True)
    rank = int((order == target).nonzero()[0]) if bool(valid[target]) else -1
    argmax = int(order[0])
    probs = logprobs.exp()
    out = {
        f"{prefix}_argmax": argmax,
        f"{prefix}_target": target,
        f"{prefix}_err": argmax - target,
        f"{prefix}_nll": -float(logprobs[target]),
        f"{prefix}_rank": rank,
        f"{prefix}_num_valid": int(valid.sum()),
        f"{prefix}_entropy": float(-(probs * logprobs.nan_to_num(neginf=0.0)).sum()),
    }
    for k in TOP_KS:
        out[f"{prefix}_top{k}"] = bool(0 <= rank < k)
    for r in NEAR_MISS_RADII:
        lo, hi = max(0, target - r), min(len(logprobs) - 1, target + r)
        out[f"{prefix}_mass_within{r}"] = float(probs[lo : hi + 1].sum())
    return out


def measure_no_edit_case(model, case: EditCase, clean_built: BuiltStream, device: torch.device) -> Dict:
    """Detection metrics for a case with nothing to fix -- there is no edit to teacher-force."""
    logprobs, marker_logits, _, _ = run_stream(model, clean_built, device, want_index=False)
    trace = edit_marker_trace(model, marker_logits, clean_built.tokens, model.marker_map, case.restored_len)
    out: Dict = dict(case.to_dict())
    out["p_edit_trigger"] = float(trace[-1])
    out["p_edit_max"] = float(trace.max())
    out["p_edit_mean"] = float(trace.mean())
    # No scripted edit here
    is_mid, visible = _prefix_marker_states(clean_built.tokens, model.marker_map, marker_logits.device)
    trigger_idx = case.restored_len - 1
    argmax_class = _marker_decision(model, marker_logits[trigger_idx], is_mid[trigger_idx], visible[trigger_idx])
    out["would_trigger"] = argmax_class == MarkerType.EDIT
    out["marker_logit_margin"] = float("nan")
    out["p_edit_trace"] = [round(float(p), 6) for p in trace]
    return out


def measure_case(
    model,
    case: EditCase,
    built: BuiltStream,
    clean_built: BuiltStream,
    device: torch.device,
    calibration_weight: float = 1.0,
    calibration_bias: float = 0.0,
) -> Dict:
    """All teacher-forced metrics for one case, plus its clean-window control."""
    marker_map = model.marker_map
    edit_idx, return_idx = built.edit_idx, built.return_idx
    logprobs, marker_logits, index_weights, _ = run_stream(model, built, device)
    clean_logprobs, clean_marker_logits, _, _ = run_stream(model, clean_built, device, want_index=False)
    is_mid, visible = _prefix_marker_states(built.tokens, marker_map, marker_logits.device)

    out: Dict = dict(case.to_dict())

    # --- localization (computed first: detection's calibration term below needs `start_lp`) -----
    start_w, end_w = index_weights[edit_idx]
    start_lp = torch.log_softmax(start_w.float().view(-1), dim=-1)
    end_lp = torch.log_softmax(end_w.float().view(-1), dim=-1)
    out.update(_distribution_metrics(start_lp, built.start_cursor, "start"))
    out.update(_distribution_metrics(end_lp, built.end_cursor, "end"))

    # --- detection -------------------------------------------------------------------------
    corrupt_trace = edit_marker_trace(model, marker_logits, built.tokens, marker_map, edit_idx)
    clean_trace = edit_marker_trace(model, clean_marker_logits, clean_built.tokens, marker_map, clean_logprobs.shape[0])
    out["p_edit_trigger"] = float(corrupt_trace[-1])
    # Match the control in *resolved* coordinates: the draft's last token sits at resolved
    # position `restored_len - 1`, not `edit_idx - 1` (pure deletions carry extra draft tokens).
    out["p_edit_trigger_clean"] = float(clean_trace[case.restored_len - 1])
    out["p_edit_max_pre_corruption"] = float(corrupt_trace[: case.start].max()) if case.start > 0 else 0.0
    out["p_edit_max_post_corruption"] = float(corrupt_trace[case.start :].max())
    out["p_edit_clean_max"] = float(clean_trace.max())
    out["detect_pairwise_win"] = out["p_edit_trigger"] > out["p_edit_trigger_clean"]

    # `_sample` biases the EDIT logit, before the marker decision, by `calibration_weight *
    # log P(sampled start) + calibration_bias`; in expectation that's `-calibration_weight`
    # times this position's start-index entropy.
    start_entropy = float(-(start_lp.exp() * start_lp.nan_to_num(neginf=0.0)).sum())
    edit_logit_bonus = -calibration_weight * start_entropy + calibration_bias
    trigger_idx = edit_idx - 1
    trigger_logits = marker_logits[trigger_idx].float()
    argmax_class = _marker_decision(model, trigger_logits, is_mid[trigger_idx], visible[trigger_idx], edit_logit_bonus)
    out["would_trigger"] = argmax_class == MarkerType.EDIT
    out["marker_logit_margin"] = float((trigger_logits[MarkerType.EDIT] + edit_logit_bonus) - trigger_logits[MarkerType.NONE])
    out["p_edit_trace"] = [round(float(p), 6) for p in corrupt_trace]
    out["p_edit_trace_clean"] = [round(float(p), 6) for p in clean_trace]

    # Chance baselines: both heads now score their legal keys with nothing but content
    out["start_uniform_nll"] = math.log(max(1, out["start_num_valid"]))
    out["end_uniform_nll"] = math.log(max(1, out["end_num_valid"]))

    # In a single-edit draft, raw stream index == resolved position
    end_argmax = out["end_argmax"]
    out["pred_end_pos"] = case.draft_len if end_argmax == edit_idx else end_argmax
    out["pred_num_delete"] = out["pred_end_pos"] - case.start
    out["delete_count_correct"] = out["pred_num_delete"] == case.num_delete
    out["span_correct"] = bool(out["start_top1"]) and out["delete_count_correct"]

    # --- content ---------------------------------------------------------------------------
    if case.num_insert:
        gold = torch.tensor(case.gold_insert, device=logprobs.device)
        rows = logprobs[edit_idx : edit_idx + case.num_insert]
        out["insert_nll"] = float(-rows.gather(1, gold[:, None]).mean())
        out["insert_acc"] = float((rows.argmax(-1) == gold).float().mean())
        out["insert_first_correct"] = bool(rows[0].argmax() == gold[0])
        out["insert_greedy_exact"] = bool((rows.argmax(-1) == gold).all())
        # Same tokens, same clean context, no edit detour: the ceiling this insertion is against.
        ref_rows = clean_logprobs[case.start - 1 : case.start - 1 + case.num_insert]
        out["insert_nll_clean_ref"] = float(-ref_rows.gather(1, gold[:, None]).mean())
    else:
        # A pure deletion has no insertion to score; None (not 0/True) so aggregates skip it.
        out["insert_nll"] = out["insert_acc"] = out["insert_nll_clean_ref"] = float("nan")
        out["insert_first_correct"] = out["insert_greedy_exact"] = None

    out["return_prob"] = marker_prob_at(model, marker_logits, built.tokens, marker_map, return_idx - 1, MarkerType.RETURN)
    # No calibration term applies to RETURN -- `_sample` only biases the EDIT class.
    return_argmax = _marker_decision(model, marker_logits[return_idx - 1], is_mid[return_idx - 1], visible[return_idx - 1])
    out["return_correct"] = return_argmax == MarkerType.RETURN

    tail_start = return_idx  # first row predicting a post-return token
    if built.length - 1 > tail_start:
        tail_gold = built.tokens[0, tail_start + 1 :].to(logprobs.device)
        tail_rows = logprobs[tail_start : built.length - 1]
        out["tail_nll"] = float(-tail_rows.gather(1, tail_gold[:, None]).mean())
        ref_lo = case.restored_len - 1
        ref_rows = clean_logprobs[ref_lo : ref_lo + tail_gold.numel()]
        out["tail_nll_clean_ref"] = float(-ref_rows.gather(1, tail_gold[: ref_rows.shape[0], None]).mean())
    else:
        out["tail_nll"] = out["tail_nll_clean_ref"] = float("nan")

    return out
