"""Free-generation outcomes as counts per category, normalised to a fixed sample size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from edit_eval.compile_results import (
        CATEGORIES,
        compute_free_gen_stats,
        format_free_gen_table,
    )
else:
    from .compile_results import CATEGORIES, compute_free_gen_stats, format_free_gen_table

LABELS = {
    "insert": "pure insertion",
    "delete": "pure deletion",
    "substitute": "substitution",
    "none": "no modification",
}

# Legacy row layout (verdict column) kept for exploratory runs.
ROWS = [
    ("repaired exactly", ["repaired"], "success"),
    ("no edit needed, none made", ["no_edit_correct"], "success"),
    ("partial repair (improved, not exact)", ["partial_repair"], "partial"),
    ("right span, wrong replacement", ["wrong_content"], "partial"),
    ("missed it (no edit emitted)", ["no_edit"], "failure"),
    ("wrong position", ["wrong_position"], "failure"),
    ("right position, deleted too few", ["under_delete"], "failure"),
    ("right position, deleted too many", ["over_delete"], "failure"),
    ("edit made the text worse", ["made_worse"], "failure"),
    ("spurious edit, text unharmed", ["spurious_harmless"], "failure"),
    ("spurious edit, text damaged", ["spurious_damaging"], "failure"),
]
VERDICT_ORDER = ["success", "partial", "failure"]


def _legacy_table(stats: dict, per: int, raw: bool) -> str:
    from collections import Counter

    by_cat = stats["by_category"]
    present = [c for c in CATEGORIES if c in by_cat]
    counts = {c: Counter(by_cat[c]["outcome_counts"]) for c in present}
    totals = {c: by_cat[c]["n_observed"] for c in present}

    def cell(category: str, keys) -> str:
        n = sum(counts[category].get(k, 0) for k in keys)
        if raw:
            return str(n) if n else "·"
        scaled = n * per / totals[category] if totals[category] else 0
        return f"{scaled:.0f}" if n else "·"

    header = ["outcome", "verdict"] + [LABELS.get(c, c) for c in present]
    widths = [max(len(r[0]) for r in ROWS) + 2, 8] + [max(len(h), 15) for h in header[2:]]

    def line(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    unit = "observed" if raw else f"per {per}"
    lines = [f"Free-generation outcomes ({unit}); each column sums to the sample size.\n"]
    lines.append(line(header))
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    for verdict in VERDICT_ORDER:
        for label, keys, row_verdict in ROWS:
            if row_verdict != verdict:
                continue
            if not any(counts[c].get(k) for c in present for k in keys):
                continue
            lines.append(line([label, verdict] + [cell(c, keys) for c in present]))

    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for verdict in VERDICT_ORDER:
        keys = [k for _, ks, v in ROWS if v == verdict for k in ks]
        lines.append(line([f"TOTAL {verdict}", verdict] + [cell(c, keys) for c in present]))
    lines.append(line(["samples observed", ""] + [str(totals[c]) for c in present]))
    return "\n".join(lines)


def _gen_dirs_from_args(run_dirs: list[Path]) -> dict[str, Path]:
    by_name = {}
    for d in run_dirs:
        name = d.name
        if name.startswith("gen_"):
            by_name[name.removeprefix("gen_")] = d
        else:
            by_name[name] = d
    return by_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="+")
    parser.add_argument("--per", type=int, default=100, help="Normalise each column to this many samples")
    parser.add_argument("--raw", action="store_true", help="Print observed counts instead of normalised")
    parser.add_argument("--paper", action="store_true", help="Paper-format table (matches FREE_GENERATION.md)")
    parser.add_argument("--json", action="store_true", help="Write stats JSON to stdout or --out-json path")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dir]
    gen_dirs = _gen_dirs_from_args(run_dirs)
    if not gen_dirs:
        raise SystemExit("No run dirs given")

    stats = compute_free_gen_stats(gen_dirs)
    if args.paper:
        report = format_free_gen_table(stats, per=args.per)
    else:
        report = _legacy_table(stats, per=args.per, raw=args.raw)

    if args.json:
        payload = json.dumps(stats, indent=2)
        if args.out_json:
            Path(args.out_json).write_text(payload)
        else:
            print(payload)
    print(report)


if __name__ == "__main__":
    main()
