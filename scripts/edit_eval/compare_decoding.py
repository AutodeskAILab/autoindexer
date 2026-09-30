"""Compare free-generation decoding settings: trigger rate, repair quality, false positives."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from edit_eval.free_gen import replay_edit_stream
else:
    from .free_gen import replay_edit_stream


def wilson(k: int, n: int, z: float = 1.96) -> str:
    """Wilson score interval -- the normal approximation is unusable at these rates."""
    if n == 0:
        return "—"
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f}% [{100 * (centre - half):.1f}–{100 * (centre + half):.1f}]"


def collect(dirs: Sequence[str]) -> Dict:
    """Pool rollouts across run dirs, re-parsing each emitted edit for its `closed` flag."""
    marker_map = None
    rollouts, edits, controls = 0, [], []
    for d in dirs:
        config = json.loads((Path(d) / "config.json").read_text())
        if marker_map is None:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(config["checkpoint_dir"])
            marker_map = {
                "edit": model_config.vocab_size,
                "return_to_end": model_config.vocab_size + 1,
                "eos": model_config.eos_token_id,
            }
        for line in open(Path(d) / "cases.jsonl"):
            record = json.loads(line)
            if "clean_control_num_edits" in record:
                controls.append(record["clean_control_num_edits"])
            for roll in record.get("rollouts", []):
                rollouts += 1
                if roll["num_edits"]:
                    first = replay_edit_stream(marker_map, roll["tokens"], roll["indices"])[0][0] if "tokens" in roll else None
                    edits.append((record, roll, first))
    return {"rollouts": rollouts, "edits": edits, "controls": controls}


ROWS = [
    ("trigger rate", lambda s: wilson(len(s["edits"]), s["rollouts"])),
    ("  insert", lambda s: f"{sum(1 for r, _, _ in s['edits'] if r['op_type'] == 'insert')}"),
    ("  delete", lambda s: f"{sum(1 for r, _, _ in s['edits'] if r['op_type'] == 'delete')}"),
    ("  substitute", lambda s: f"{sum(1 for r, _, _ in s['edits'] if r['op_type'] == 'substitute')}"),
    ("spurious on clean drafts", lambda s: f"{sum(1 for c in s['controls'] if c > 0)}/{len(s['controls'])}"),
    ("edits emitted", lambda s: str(len(s["edits"]))),
    ("  finished in budget", lambda s: _frac(s, lambda r, x, f: f is not None and f.closed)),
    ("  start cursor correct", lambda s: _frac(s, lambda r, x, f: x.get("gen_start_correct"))),
    ("  full span correct", lambda s: _frac(s, lambda r, x, f: x.get("gen_span_correct"))),
    ("  exact repair", lambda s: _frac(s, lambda r, x, f: x["repair_exact"])),
    ("  improved the text", lambda s: _frac(s, lambda r, x, f: x["dist_after"] < x["dist_before"])),
    ("  made it worse", lambda s: _frac(s, lambda r, x, f: x["dist_after"] > x["dist_before"])),
    ("net repairs per 100 rollouts", lambda s: _net(s)),
]


def _frac(state: Dict, predicate) -> str:
    edits = state["edits"]
    if not edits:
        return "—"
    hits = sum(1 for r, x, f in edits if predicate(r, x, f))
    return f"{hits}/{len(edits)} ({100 * hits / len(edits):.0f}%)"


def _net(state: Dict) -> str:
    """Improved minus worsened, per 100 rollouts -- the net effect of the decoding setting."""
    if not state["rollouts"]:
        return "—"
    better = sum(1 for _, x, _ in state["edits"] if x["dist_after"] < x["dist_before"])
    worse = sum(1 for _, x, _ in state["edits"] if x["dist_after"] > x["dist_before"])
    return f"{100 * (better - worse) / state['rollouts']:+.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", help="label=dir[,dir...]")
    parser.add_argument("--outcomes", action="store_true", help="Also print the outcome histogram")
    args = parser.parse_args()

    states, labels = {}, []
    for group in args.groups:
        label, _, dirs = group.rpartition("=")
        labels.append(label)
        states[label] = collect(dirs.split(","))

    width = max(len(r[0]) for r in ROWS) + 2
    cols = [max(len(l), 22) for l in labels]
    print("| " + "metric".ljust(width) + " | " + " | ".join(l.ljust(c) for l, c in zip(labels, cols)) + " |")
    print("|" + "-" * (width + 2) + "|" + "|".join("-" * (c + 2) for c in cols) + "|")
    for name, fn in ROWS:
        cells = [fn(states[l]).ljust(c) for l, c in zip(labels, cols)]
        print("| " + name.ljust(width) + " | " + " | ".join(cells) + " |")

    if args.outcomes:
        print()
        for label in labels:
            counts = Counter(x["outcome"] for _, x, _ in states[label]["edits"])
            print(f"{label}: {dict(counts.most_common())}")


if __name__ == "__main__":
    main()
