"""Compile edit-eval run directories into paper-format tables, JSON stats, and a gallery."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CATEGORIES = ["insert", "delete", "substitute", "none"]
OP_COLUMNS = ["insert", "delete", "substitute"]

PAPER_FREE_GEN_ROWS: List[Tuple[str, Optional[str], Optional[str], Sequence[str]]] = [
    ("Repaired exactly", "✓", "✓", ["repaired"]),
    ("No edit needed (true negative)", None, None, ["no_edit_correct"]),
    ("No edit emitted (false negative)", None, None, ["no_edit"]),
    ("Correct span, improved", "✓", "✓", ["partial_repair"]),
    ("Correct span, neutral", "✓", "✓", ["wrong_content"]),
    ("Correct span, worse", "✓", "✓", ["made_worse"]),
    ("Correct position, deleted too few", "✓", "✗", ["under_delete"]),
    ("Correct position, deleted too many", "✓", "✗", ["over_delete"]),
    ("Incorrect position", "✗", None, ["wrong_position"]),
]


def load_records(run_dir: Path) -> List[Dict]:
    with open(run_dir / "cases.jsonl") as f:
        return [json.loads(line) for line in f]


def load_rollouts_by_category(gen_dirs: Dict[str, Path]) -> Dict[str, List[Dict]]:
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for category, run_dir in gen_dirs.items():
        for record in load_records(run_dir):
            for roll in record.get("rollouts", []):
                by_category[category].append({**roll, "op_type": record["op_type"]})
    return by_category


def _mean(values: Sequence) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(clean)) if clean else None


def _median(values: Sequence) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(clean)) if clean else None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        return None if isinstance(val, float) and math.isnan(val) else val
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _format_prob(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _format_elevation(corrupt: Optional[float], clean: Optional[float]) -> str:
    if corrupt is None or clean is None or clean <= 0:
        return "—"
    ratio = corrupt / clean
    return f"~{int(round(ratio))}×"


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def _format_nats(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "—"
    return f"{x:+.2f}"


def compute_teacher_forced_stats(tf_dir: Path, tf_none_dir: Optional[Path] = None) -> Dict:
    records = load_records(tf_dir)
    by_op = {op: [r for r in records if r["op_type"] == op] for op in OP_COLUMNS}
    stats: Dict = {"by_operation": {}, "num_windows": len(records) // len(OP_COLUMNS) if OP_COLUMNS else 0}

    for op in OP_COLUMNS:
        bucket = by_op[op]
        nll_gain = None
        if op != "delete":
            gains = [
                r["insert_nll"] - r["insert_nll_clean_ref"]
                for r in bucket
                if not math.isnan(r.get("insert_nll", float("nan")))
                and not math.isnan(r.get("insert_nll_clean_ref", float("nan")))
            ]
            nll_gain = _mean(gains)

        corrupt = _mean([r["p_edit_trigger"] for r in bucket])
        clean = _mean([r["p_edit_trigger_clean"] for r in bucket])
        stats["by_operation"][op] = {
            "n": len(bucket),
            "p_edit_corrupt": corrupt,
            "p_edit_clean": clean,
            "elevation_ratio": (corrupt / clean) if corrupt is not None and clean and clean > 0 else None,
            "median_marker_logit_margin": _median([r["marker_logit_margin"] for r in bucket]),
            "would_trigger_rate": _mean([r["would_trigger"] for r in bucket]),
            "start_top1": _mean([r["start_top1"] for r in bucket]),
            "span_correct": _mean([r["span_correct"] for r in bucket]),
            "replacement_nll_gain_vs_causal": nll_gain,
            "detect_pairwise_win_rate": _mean([r["detect_pairwise_win"] for r in bucket]),
        }

    if tf_none_dir is not None:
        none_records = load_records(tf_none_dir)
        stats["none_draft_end"] = {
            "n": len(none_records),
            "p_edit_mean": _mean([r["p_edit_trigger"] for r in none_records]),
            "p_edit_max": _mean([r.get("p_edit_max", r["p_edit_trigger"]) for r in none_records]),
            # A "would_trigger" on a no-op draft is a false positive
            "false_positive_rate": _mean([r["would_trigger"] for r in none_records]),
        }

        # Precision/recall of the trigger decision
        tp = sum(1 for op in OP_COLUMNS for r in by_op[op] if r["would_trigger"])
        fn = sum(1 for op in OP_COLUMNS for r in by_op[op] if not r["would_trigger"])
        fp = sum(1 for r in none_records if r["would_trigger"])
        tn = sum(1 for r in none_records if not r["would_trigger"])
        stats["trigger_precision_recall"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": (tp / (tp + fp)) if (tp + fp) > 0 else None,
            "recall": (tp / (tp + fn)) if (tp + fn) > 0 else None,
        }

    return stats


def outcome_counts(rollouts: Sequence[Dict], category: str) -> Counter:
    counts = Counter(roll["outcome"] for roll in rollouts)
    if category == "none":
        counts["no_edit_correct"] = counts.pop("no_edit", 0)
    return counts


def compute_free_gen_stats(gen_dirs: Dict[str, Path]) -> Dict:
    by_category = load_rollouts_by_category(gen_dirs)
    stats: Dict = {"by_category": {}, "per": 100}

    for category in CATEGORIES:
        rollouts = by_category.get(category, [])
        counts = outcome_counts(rollouts, category)
        total = len(rollouts)
        improved = sum(1 for r in rollouts if r["dist_after"] < r["dist_before"])
        worsened = sum(1 for r in rollouts if r["dist_after"] > r["dist_before"])

        if category == "none":
            correct_emission = counts.get("no_edit_correct", 0)
        else:
            correct_emission = total - counts.get("no_edit", 0)

        stats["by_category"][category] = {
            "n_observed": total,
            "outcome_counts": dict(counts),
            "correct_edit_emission": correct_emission,
            "net_improvement": improved - worsened,
            "dist_improved": improved,
            "dist_worsened": worsened,
        }

    return stats


def _scaled_count(count: int, total: int, per: int = 100) -> int:
    if total == 0:
        return 0
    return int(round(count * per / total))


def format_teacher_forced_table(stats: Dict) -> str:
    ops = OP_COLUMNS
    rows = [
        (
            "$P(\\textsc{edit})$ at trigger (corrupted / clean)",
            [
                f"{_format_prob(stats['by_operation'][op]['p_edit_corrupt'])} / "
                f"{_format_prob(stats['by_operation'][op]['p_edit_clean'])}"
                for op in ops
            ],
        ),
        (
            "Elevation over clean control",
            [_format_elevation(stats["by_operation"][op]["p_edit_corrupt"], stats["by_operation"][op]["p_edit_clean"]) for op in ops],
        ),
        (
            "Median edit logit margin at trigger",
            [
                _format_nats(stats["by_operation"][op]["median_marker_logit_margin"])
                for op in ops
            ],
        ),
        (
            "Would trigger (calibrated argmax)",
            [_format_pct(stats["by_operation"][op]["would_trigger_rate"]) for op in ops],
        ),
        (
            "Start cursor exactly right",
            [_format_pct(stats["by_operation"][op]["start_top1"]) for op in ops],
        ),
        (
            "Both cursors exactly right",
            [_format_pct(stats["by_operation"][op]["span_correct"]) for op in ops],
        ),
        (
            "Replacement NLL gain vs. causal (nats)",
            [_format_nats(stats["by_operation"][op]["replacement_nll_gain_vs_causal"]) for op in ops],
        ),
    ]

    header = ["Measurement"] + [op.capitalize() for op in ops]
    widths = [max(len(header[0]), *(len(r[0]) for r in rows))] + [max(len(h), 12) for h in header[1:]]

    def fmt(cells: Sequence[str]) -> str:
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    lines = [
        "# Teacher-forced edit execution",
        "",
        "Oracle-placed markers; detection compares $P(\\textsc{edit})$ from `marker_head` at the "
        "scripted trigger position against the same position in an uncorrupted control. "
        "'Would trigger' replays `_sample`'s actual masked-argmax marker decision, including its "
        "start-index calibration term, rather than a fixed probability threshold.",
        "",
        fmt(header),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for label, cells in rows:
        lines.append(fmt([label, *cells]))

    pr = stats.get("trigger_precision_recall")
    if pr is not None:
        lines += [
            "",
            "**Trigger precision/recall** (insert/delete/substitute pooled as the positive class, "
            "the no-op control as the negative class): "
            f"precision {_format_pct(pr['precision'])}, recall {_format_pct(pr['recall'])} "
            f"(TP={pr['tp']}, FP={pr['fp']}, FN={pr['fn']}, TN={pr['tn']}).",
        ]
    return "\n".join(lines) + "\n"


def _paper_cell(category: str, counts: Counter, keys: Sequence[str], total: int, per: int) -> str:
    if category == "none" and keys == ["no_edit"]:
        return "·"
    n = sum(counts.get(k, 0) for k in keys)
    if n == 0:
        return "·"
    return str(_scaled_count(n, total, per))


def format_free_gen_table(stats: Dict, per: int = 100) -> str:
    by_cat = stats["by_category"]
    totals = {c: by_cat[c]["n_observed"] for c in CATEGORIES if c in by_cat}
    counts = {c: Counter(by_cat[c]["outcome_counts"]) for c in CATEGORIES if c in by_cat}

    header = ["Outcome", "Start cursor", "End cursor", "Insert", "Delete", "Substitute", "No modification"]
    col_keys = ["insert", "delete", "substitute", "none"]
    widths = [28, 13, 11] + [max(16, len(str(per))) for _ in col_keys]

    def fmt(cells: Sequence[str]) -> str:
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    lines = [
        "# Free-generation per-outcome counts",
        "",
        f"100 examples per category (scaled to {per} when sample sizes differ)",
        "",
        fmt(header),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]

    for label, start_mark, end_mark, keys in PAPER_FREE_GEN_ROWS:
        start_col = start_mark if start_mark is not None else "—"
        end_col = end_mark if end_mark is not None else "—"
        lines.append(
            fmt(
                [label, start_col, end_col]
                + [_paper_cell(c, counts[c], keys, totals[c], per) for c in col_keys]
            )
        )

    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    emission_row = ["Correct edit emission", "—", "—"]
    for c in col_keys:
        val = by_cat[c]["correct_edit_emission"]
        emission_row.append(str(_scaled_count(val, totals[c], per)) if totals[c] else "·")
    lines.append(fmt(emission_row))

    net_row = ["Net improvement", "—", "—"]
    for c in col_keys:
        if c == "none":
            net_row.append("·")
        else:
            net = by_cat[c]["net_improvement"]
            net_row.append(f"{net:+d}" if totals[c] else "·")
    lines.append(fmt(net_row))
    return "\n".join(lines) + "\n"


def resolve_gen_dirs(eval_dir: Path, skip_noop: bool = False) -> Dict[str, Path]:
    gen_dirs = {
        "insert": eval_dir / "gen_insert",
        "delete": eval_dir / "gen_delete",
        "substitute": eval_dir / "gen_substitute",
    }
    if not skip_noop:
        gen_dirs["none"] = eval_dir / "gen_none"
    missing = [p for p in gen_dirs.values() if not (p / "cases.jsonl").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cases.jsonl in: " + ", ".join(str(p) for p in missing)
        )
    return gen_dirs


def resolve_run_layout(
    eval_dir: Path, genonly: bool = False, skip_noop: bool = False
) -> Tuple[Path, Optional[Path], Dict[str, Path]]:
    gen_dirs = resolve_gen_dirs(eval_dir, skip_noop=skip_noop)
    if genonly:
        return eval_dir / "tf", None, gen_dirs

    tf_dir = eval_dir / "tf"
    tf_none_dir = eval_dir / "tf_none"
    missing = [p for p in [tf_dir] if not (p / "cases.jsonl").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cases.jsonl in: " + ", ".join(str(p) for p in missing)
        )
    return tf_dir, tf_none_dir if (tf_none_dir / "cases.jsonl").exists() else None, gen_dirs


def compile_eval_dir(
    eval_dir: Path,
    per: int = 100,
    gallery_per_outcome: Optional[int] = None,
    genonly: bool = False,
    skip_noop: bool = False,
) -> Dict:
    eval_dir = eval_dir.resolve()
    tf_dir, tf_none_dir, gen_dirs = resolve_run_layout(eval_dir, genonly=genonly, skip_noop=skip_noop)

    fg_stats = compute_free_gen_stats(gen_dirs)
    fg_md = format_free_gen_table(fg_stats, per=per)
    (eval_dir / "FREE_GENERATION.md").write_text(fg_md)

    stats: Dict = {
        "eval_dir": str(eval_dir),
        "free_generation": fg_stats,
        "tables": {"free_generation_markdown": fg_md},
    }

    if not genonly:
        tf_stats = compute_teacher_forced_stats(tf_dir, tf_none_dir)
        tf_md = format_teacher_forced_table(tf_stats)
        (eval_dir / "TEACHER_FORCED.md").write_text(tf_md)
        stats["teacher_forced"] = tf_stats
        stats["tables"]["teacher_forced_markdown"] = tf_md

    with open(eval_dir / "stats.json", "w") as f:
        json.dump(_json_safe(stats), f, indent=2)

    if not genonly:
        if __package__ in (None, ""):
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from edit_eval.examples import build_gallery
        else:
            from .examples import build_gallery

        per_outcome = gallery_per_outcome if gallery_per_outcome is not None else 9999
        build_gallery(list(gen_dirs.values()), eval_dir / "GALLERY.md", per_outcome=per_outcome)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", help="Run root containing tf/, gen_*/ subdirs")
    parser.add_argument("--per", type=int, default=100, help="Normalise free-gen counts to this sample size")
    parser.add_argument(
        "--gallery-per-outcome",
        type=int,
        default=None,
        help="Max examples per outcome in GALLERY.md (default: all)",
    )
    parser.add_argument(
        "--genonly",
        action="store_true",
        help="Only compile free-generation outputs (skip teacher-forced and gallery)",
    )
    parser.add_argument(
        "--skip-noop",
        action="store_true",
        help="Run directory has no tf_none / gen_none phases (produced by `run.py --skip-noop`); "
        "compile without requiring them",
    )
    args = parser.parse_args()

    stats = compile_eval_dir(
        Path(args.eval_dir),
        per=args.per,
        gallery_per_outcome=args.gallery_per_outcome,
        genonly=args.genonly,
        skip_noop=args.skip_noop,
    )
    eval_dir = Path(args.eval_dir)
    if not args.genonly:
        print(f"Wrote {eval_dir / 'TEACHER_FORCED.md'}")
        print(f"Wrote {eval_dir / 'GALLERY.md'}")
    print(f"Wrote {eval_dir / 'FREE_GENERATION.md'}")
    print(f"Wrote {eval_dir / 'stats.json'}")
    print()
    if not args.genonly:
        print(stats["tables"]["teacher_forced_markdown"])
        print()
    print(stats["tables"]["free_generation_markdown"])


if __name__ == "__main__":
    main()
