"""Render qualitative examples from a `run.py --save-streams` output directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from transformers import AutoConfig, AutoTokenizer

import autoindexer.models.autoindexer  # noqa: F401  -- registers the Auto* classes

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from edit_eval.free_gen import replay_edit_stream
else:
    from .free_gen import replay_edit_stream

MARKER_NAMES = {"edit": "EDIT", "return_to_end": "RET", "eos": "EOS"}


def render_stream(tokenizer, marker_map: Dict[str, int], tokens: Sequence[int], indices: Sequence[Sequence[int]]) -> str:
    """Decode a raw stream, showing each edit marker with the cursor pair it sampled."""
    names = {token_id: name for name, token_id in marker_map.items()}
    parts: List[str] = []
    pending: List[int] = []
    for i, token in enumerate(tokens):
        name = names.get(token)
        if name is None:
            pending.append(token)
            continue
        if pending:
            parts.append(tokenizer.decode(pending))
            pending = []
        if name == "edit":
            end = indices[i + 1][1] if i + 1 < len(indices) else "?"
            parts.append(f"⟦EDIT start={indices[i][0]} end={end}⟧")
        else:
            parts.append(f"⟦{MARKER_NAMES[name]}⟧")
    if pending:
        parts.append(tokenizer.decode(pending))
    return "".join(parts)


def render_case(tokenizer, marker_map, record: Dict, rollout: Dict, origin: str = "") -> str:
    clean = record["clean_tokens"]
    start, num_insert, num_delete = record["start"], record["num_insert"], record["num_delete"]
    restored = record["start"] + num_insert + record["lag"]
    draft = clean[:start] + record["corruption"] + clean[start + num_insert : restored]

    lines = [
        f"### {record['op_type']} · {rollout['outcome']} · `{record['source']}`"
        + (f" · {origin}" if origin else ""),
        "",
        f"- `k={num_insert}` inserted, `d={num_delete}` deleted, at position {start}, "
        f"{record['lag']} tokens before the edit fires",
        f"- gold span `[{start}, {start + num_delete})`, model span "
        f"`[{rollout.get('gen_start')}, {rollout.get('gen_end')})`; "
        f"Levenshtein to gold {rollout['dist_before']} → {rollout['dist_after']}",
        "",
        "**1. Draft it was given** (corrupted region between the guillemets):",
        "```",
        tokenizer.decode(draft[:start]) + " «««" + tokenizer.decode(record["corruption"]) + "»»» " + tokenizer.decode(draft[start + num_delete :]),
        "```",
    ]
    if "tokens" in rollout:
        # `prompt_len` lets the replay separate the repaired prefix from the free continuation
        _, resolved, appended = replay_edit_stream(
            marker_map, rollout["tokens"], rollout["indices"], prompt_len=len(draft)
        )
        repaired = resolved[: max(0, len(resolved) - appended)]
        lines += [
            "**2. What it emitted** (continuation only; markers inline):",
            "```",
            render_stream(tokenizer, marker_map, rollout["tokens"][len(draft) :], rollout["indices"][len(draft) :]),
            "```",
            "**3. Resolved text after applying the edit:**",
            "```",
            tokenizer.decode(repaired),
            "```",
        ]
    lines += [
        "**4. Gold** (what a perfect repair produces):",
        "```",
        tokenizer.decode(clean[:restored]),
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_gallery(
    run_dirs: Sequence[Path],
    out_path: Path,
    *,
    per_outcome: int = 3,
    only: Optional[Sequence[str]] = None,
    tokenizer_path: Optional[str] = None,
) -> str:
    run_dirs = [Path(d) for d in run_dirs]
    run_dir = run_dirs[0]
    config = json.loads((run_dir / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or config["checkpoint_dir"], trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(config["checkpoint_dir"])
    vocab_size = model_config.vocab_size
    marker_map = {"edit": vocab_size, "return_to_end": vocab_size + 1, "eos": model_config.eos_token_id}

    buckets: Dict[str, List] = {}
    saw_streams = False
    for d in run_dirs:
        for line in open(d / "cases.jsonl"):
            record = json.loads(line)
            if "clean_tokens" not in record:
                continue
            saw_streams = True
            for rollout in record.get("rollouts", []):
                key = f"{record['op_type']}/{rollout['outcome']}"
                buckets.setdefault(key, []).append((record, rollout, d.name))
    if not saw_streams:
        raise SystemExit("No saved streams -- rerun `run.py` with --save-streams.")
    if only:
        buckets = {k: v for k, v in buckets.items() if k in set(only)}

    title = run_dir.parent.name if run_dir.name.startswith("gen_") else run_dir.name
    if len(run_dirs) > 1:
        title = f"{title} (free generation)"
    sections = [
        f"# Edit evaluation gallery — {title}\n",
        "Each case shows the corrupted draft, emitted stream, resolved text, gold repair, "
        "and Levenshtein distance to gold.\n",
    ]
    for key in sorted(buckets):
        sections.append(f"## {key} ({len(buckets[key])} cases)\n")
        for record, rollout, origin in buckets[key][:per_outcome]:
            sections.append(render_case(tokenizer, marker_map, record, rollout, origin))

    report = "\n".join(sections)
    out_path.write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="+", help="One or more run dirs; pooled if several")
    parser.add_argument("--per-outcome", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Include every case per outcome bucket")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these `op/outcome` keys")
    parser.add_argument("--out", default=None, help="Defaults to <first run dir>/examples.md")
    parser.add_argument("--tokenizer", default=None, help="Defaults to the run's checkpoint dir")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dir]
    out_path = Path(args.out or run_dirs[0] / "examples.md")
    per_outcome = 999999 if args.all else args.per_outcome
    report = build_gallery(
        run_dirs,
        out_path,
        per_outcome=per_outcome,
        only=args.only,
        tokenizer_path=args.tokenizer,
    )
    print(report)


if __name__ == "__main__":
    main()
