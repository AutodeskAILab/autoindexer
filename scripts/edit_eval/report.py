"""Aggregate `cases.jsonl` from `run.py` into summary tables (and optional plots)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# (column header, per-record accessor). NaNs are dropped per column, so op types that can't
# populate a metric (e.g. `insert_*` for pure deletions) just show as empty.
TEACHER_FORCED_COLUMNS = [
    ("n", None),
    ("k (mean)", lambda r: max(r["num_insert"], r["num_delete"])),
    ("lag (mean)", lambda r: r["lag"]),
    ("start top1", lambda r: r["start_top1"]),
    ("start top5", lambda r: r["start_top5"]),
    ("start |err|", lambda r: abs(r["start_err"])),
    ("start NLL", lambda r: r["start_nll"]),
    ("start NLL (unif)", lambda r: r["start_uniform_nll"]),
    # Position 0 is a classic attention sink; a head that falls back to it is not localizing.
    ("start→pos0", lambda r: r["start_argmax"] == 0),
    ("end top1", lambda r: r["end_top1"]),
    ("end NLL", lambda r: r["end_nll"]),
    ("end NLL (unif)", lambda r: r["end_uniform_nll"]),
    ("Δdelete", lambda r: r["pred_num_delete"] - r["num_delete"]),
    ("delete-count acc", lambda r: r["delete_count_correct"]),
    ("span acc", lambda r: r["span_correct"]),
    ("insert acc", lambda r: r["insert_acc"]),
    ("insert NLL", lambda r: r["insert_nll"]),
    ("insert NLL (ref)", lambda r: r["insert_nll_clean_ref"]),
    ("return prob", lambda r: r["return_prob"]),
    ("tail NLL", lambda r: r["tail_nll"]),
    ("tail NLL (ref)", lambda r: r["tail_nll_clean_ref"]),
    ("logit margin", lambda r: r["marker_logit_margin"]),
    ("would trigger", lambda r: r["would_trigger"]),
    ("P(edit) corrupt", lambda r: r["p_edit_trigger"]),
    ("P(edit) clean", lambda r: r["p_edit_trigger_clean"]),
    ("detect win-rate", lambda r: r["detect_pairwise_win"]),
]

# The narrow view: one column per pipeline stage (detect -> localize -> size -> content).
HEADLINE_COLUMNS = [
    ("n", None),
    ("k", lambda r: max(r["num_insert"], r["num_delete"])),
    ("P(edit) x clean", lambda r: r["p_edit_trigger"] / max(r["p_edit_trigger_clean"], 1e-9)),
    ("detect win", lambda r: r["detect_pairwise_win"]),
    ("start top1", lambda r: r["start_top1"]),
    ("start→pos0", lambda r: r["start_argmax"] == 0),
    ("len acc", lambda r: r["delete_count_correct"]),
    ("Δlen", lambda r: r["pred_num_delete"] - r["num_delete"]),
    ("span acc", lambda r: r["span_correct"]),
    ("insert acc", lambda r: r["insert_acc"]),
    ("NLL vs causal", lambda r: r["insert_nll"] - r["insert_nll_clean_ref"]),
    ("P(return)", lambda r: r["return_prob"]),
]

FREE_GEN_COLUMNS = [
    ("n", None),
    ("edit rate", lambda r: r["num_edits"] > 0),
    ("edits/rollout", lambda r: r["num_edits"]),
    ("start correct", lambda r: r.get("gen_start_correct")),
    ("span correct", lambda r: r.get("gen_span_correct")),
    ("span IoU", lambda r: r.get("gen_span_iou")),
    ("insert correct", lambda r: r.get("gen_insert_correct")),
    ("repair exact", lambda r: r["repair_exact"]),
    ("repair gain", lambda r: r["repair_gain"]),
    ("dist before", lambda r: r["dist_before"]),
    ("dist after", lambda r: r["dist_after"]),
    ("edit delay", lambda r: r.get("first_edit_delay")),
]


def load_records(run_dir: Path) -> List[Dict]:
    with open(run_dir / "cases.jsonl") as f:
        return [json.loads(line) for line in f]


def flatten_rollouts(records: Sequence[Dict]) -> List[Dict]:
    return [{**{k: v for k, v in r.items() if k != "rollouts"}, **roll} for r in records for roll in r.get("rollouts", [])]


def _mean(values: Sequence) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(clean)) if clean else None


def _stderr(values: Sequence) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.std(clean) / math.sqrt(len(clean))) if len(clean) > 1 else None


def summarize(records: Sequence[Dict], columns, group_by: Callable[[Dict], str], with_stderr: bool = False) -> str:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        groups[group_by(record)].append(record)

    headers = [name for name, _ in columns]
    rows = []
    for key in sorted(groups):
        bucket = groups[key]
        cells = []
        for name, accessor in columns:
            if accessor is None:
                cells.append(str(len(bucket)))
                continue
            mean = _mean([accessor(r) for r in bucket])
            if mean is None:
                cells.append("")
            elif with_stderr and (se := _stderr([accessor(r) for r in bucket])):
                cells.append(f"{mean:.3f}±{se:.3f}")
            else:
                cells.append(f"{mean:.3f}")
        rows.append([key] + cells)

    widths = [max(len(str(x)) for x in col) for col in zip(*([[""] + headers] + rows))]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    lines = [fmt([""] + headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt(row) for row in rows]
    return "\n".join(lines)


def lag_bucket(record: Dict, edges=(1, 5, 15, 40)) -> str:
    lag = record["lag"]
    for i, edge in enumerate(edges):
        if lag < edge:
            return f"{i}:lag<{edge}"
    return f"{len(edges)}:lag>={edges[-1]}"


def k_bucket(record: Dict, edges=(3, 7, 12, 20)) -> str:
    k = max(record["num_insert"], record["num_delete"])
    for i, edge in enumerate(edges):
        if k < edge:
            return f"{i}:k<{edge}"
    return f"{len(edges)}:k>={edges[-1]}"


def outcome_table(rollouts: Sequence[Dict]) -> str:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for roll in rollouts:
        counts[roll["op_type"]][roll["outcome"]] += 1
    outcomes = sorted({o for row in counts.values() for o in row})
    widths = [max(len("op_type"), *(len(o) for o in counts))] + [max(len(o), 6) for o in outcomes]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    lines = [fmt(["op_type"] + outcomes), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for op in sorted(counts):
        total = sum(counts[op].values())
        lines.append(fmt([op] + [f"{counts[op][o] / total:.2f}" for o in outcomes]))
    return "\n".join(lines)


def detection_auc(records: Sequence[Dict]) -> float:
    """Paired AUROC: how often P(edit) at the trigger beats the same point on the clean window."""
    wins = [
        1.0 if r["p_edit_trigger"] > r["p_edit_trigger_clean"] else 0.5 if r["p_edit_trigger"] == r["p_edit_trigger_clean"] else 0.0
        for r in records
    ]
    return float(np.mean(wins)) if wins else float("nan")


def make_plots(records: Sequence[Dict], rollouts: Sequence[Dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    for op in sorted({r["op_type"] for r in records}):
        bucket = [r for r in records if r["op_type"] == op]
        errs = np.array([r["start_err"] for r in bucket])
        axes[0].hist(np.clip(errs, -20, 20), bins=41, range=(-20, 20), histtype="step", label=op, density=True)
        lags = np.array([r["lag"] for r in bucket])
        hits = np.array([float(r["start_top1"]) for r in bucket])
        edges = np.quantile(lags, np.linspace(0, 1, 7))
        centers, rates = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (lags >= lo) & (lags <= hi)
            if mask.sum() > 3:
                centers.append(lags[mask].mean())
                rates.append(hits[mask].mean())
        axes[1].plot(centers, rates, marker="o", label=op)
        deltas = np.array([r["pred_num_delete"] - r["num_delete"] for r in bucket])
        axes[2].hist(np.clip(deltas, -20, 20), bins=41, range=(-20, 20), histtype="step", label=op, density=True)

    axes[0].set(xlabel="start-cursor error (tokens)", ylabel="density", title="Localization error")
    axes[1].set(xlabel="lag (tokens between corruption and edit)", ylabel="start top-1", title="Localization vs. lag")
    axes[2].set(xlabel="predicted − true num_delete", ylabel="density", title="Span-length error")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "localization.png", dpi=130)

    traces = [r for r in records if "p_edit_trace" in r]
    if traces:
        fig, ax = plt.subplots(figsize=(7, 4))
        max_lag = max(r["lag"] for r in traces)
        grid = np.arange(0, min(max_lag, 60) + 1)
        for op in sorted({r["op_type"] for r in traces}):
            curves = []
            for r in (t for t in traces if t["op_type"] == op):
                trace = np.array(r["p_edit_trace"])
                after = trace[r["start"] :]
                if len(after) > len(grid):
                    curves.append(after[: len(grid)])
            if curves:
                ax.plot(grid, np.mean(curves, axis=0), label=f"{op} (corrupted)")
        clean = [np.array(r["p_edit_trace_clean"])[r["start"] :][: len(grid)] for r in traces]
        clean = [c for c in clean if len(c) == len(grid)]
        if clean:
            ax.plot(grid, np.mean(clean, axis=0), "k--", label="clean control")
        ax.set(xlabel="tokens since corruption", ylabel="P(emit edit marker)", yscale="log", title="Edit detection")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "detection.png", dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--compare", nargs="*", default=(), help="Additional run dirs to table side by side")
    parser.add_argument("--headline", action="store_true", help="Narrow table of the key per-stage metrics")
    args = parser.parse_args()

    if args.compare:
        pooled = []
        for run in [args.run_dir, *args.compare]:
            pooled += [{**r, "_run": Path(run).name} for r in load_records(Path(run))]
        columns = HEADLINE_COLUMNS if args.headline else TEACHER_FORCED_COLUMNS
        print(summarize(pooled, columns, lambda r: f"{r['_run']}/{r['op_type']}", with_stderr=not args.headline))
        return

    run_dir = Path(args.run_dir)
    records = load_records(run_dir)
    rollouts = flatten_rollouts(records)

    sections = [
        f"# AutoIndexer edit evaluation -- `{run_dir}`\n",
        f"{len(records)} cases; {len(rollouts)} rollouts.\n",
        "## Teacher-forced, headline\n",
        summarize(records, HEADLINE_COLUMNS, lambda r: r["op_type"], with_stderr=True),
        "\n## Teacher-forced, by operation (full)\n",
        summarize(records, TEACHER_FORCED_COLUMNS, lambda r: r["op_type"], with_stderr=True),
        "\n## Teacher-forced, by edit size\n",
        summarize(records, TEACHER_FORCED_COLUMNS, lambda r: f"{r['op_type']}/{k_bucket(r)}"),
        "\n## Teacher-forced, by lag\n",
        summarize(records, TEACHER_FORCED_COLUMNS, lambda r: f"{r['op_type']}/{lag_bucket(r)}"),
        "\n## Teacher-forced, by source\n",
        summarize(records, TEACHER_FORCED_COLUMNS, lambda r: r["source"]),
        f"\nPaired edit-detection win-rate (corrupted vs. clean): {detection_auc(records):.3f}\n",
    ]
    if rollouts:
        control = [r["clean_control_num_edits"] for r in records if "clean_control_num_edits" in r]
        sections += [
            "## Free generation, by operation\n",
            summarize(rollouts, FREE_GEN_COLUMNS, lambda r: r["op_type"], with_stderr=True),
            "\n## Free generation, outcome distribution\n",
            outcome_table(rollouts),
            "\n## Free generation, by lag\n",
            summarize(rollouts, FREE_GEN_COLUMNS, lambda r: f"{r['op_type']}/{lag_bucket(r)}"),
        ]
        if control:
            sections.append(f"\nSpurious-edit rate on uncorrupted drafts: {np.mean([c > 0 for c in control]):.3f}\n")

    report = "\n".join(sections)
    (run_dir / "report.md").write_text(report)
    print(report)

    if args.plots:
        make_plots(records, rollouts, run_dir)
        print(f"\nPlots written to {run_dir}")


if __name__ == "__main__":
    main()
