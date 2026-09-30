"""Run the AutoIndexer edit-capability evaluation."""

from __future__ import annotations

from tqdm import tqdm
import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import autoindexer.models.autoindexer  # noqa: F401  -- registers the Auto* classes

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from edit_eval import cases as case_lib
    from edit_eval import data as data_lib
    from edit_eval import free_gen, teacher_forced
else:
    from . import cases as case_lib
    from . import data as data_lib
    from . import free_gen, teacher_forced

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATAMIX = REPO_ROOT / "configs" / "cpt_datamix" / "_datamix.yaml"


def parse_config_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_config_overrides(config, pairs: Sequence[str]) -> Dict[str, Any]:
    """Set checkpoint config fields after load; keys may use a `config.` prefix or dotted paths."""
    applied: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--config expects KEY=VALUE, got: {pair!r}")
        key, value_raw = pair.split("=", 1)
        key = key.strip().removeprefix("config.")
        if not key:
            raise ValueError(f"--config expects KEY=VALUE, got: {pair!r}")

        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if isinstance(target, dict):
                target = target[part]
            else:
                target = getattr(target, part)

        parsed = parse_config_value(value_raw)
        final_key = parts[-1]
        if isinstance(target, dict):
            target[final_key] = parsed
        else:
            setattr(target, final_key, parsed)
        applied[key] = parsed
    return applied


def resolve_checkpoint(checkpoint: str, cache_dir: str) -> Path:
    if not checkpoint.startswith("s3://"):
        return Path(checkpoint)
    local_dir = Path(cache_dir) / checkpoint.rstrip("/").rsplit("/", 2)[-2] / "final_checkpoint"
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", checkpoint.rstrip("/") + "/", str(local_dir), "--only-show-errors"], check=True
    )
    return local_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint", required=True, help="Local checkpoint dir or s3:// URI")
    model.add_argument("--checkpoint-cache-dir", default=str(Path.home() / "checkpoints"))
    model.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    model.add_argument("--dtype", default="bfloat16")
    model.add_argument("--attn-implementation", default="autoindexer_eager")
    model.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a loaded checkpoint config field (repeatable; optional `config.` prefix; "
        "e.g. index_head_scaling=0.5)",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--datamix-config", default=str(DEFAULT_DATAMIX))
    data.add_argument("--local-data-dir", default=None, help="Mirror dir for datamix sources with a local_dir")
    data.add_argument("--num-val-samples-per-source", type=int, default=None)
    data.add_argument("--sources", nargs="*", default=None, help="Restrict to these datamix source names")
    data.add_argument("--window-len", type=int, default=100)
    data.add_argument("--num-windows", type=int, default=128)
    data.add_argument("--window-cache-dir", default=None)
    data.add_argument("--window-pool", type=int, default=512, help="Windows to tokenize/cache; --num-windows slices this")

    perturb = parser.add_argument_group("perturbation")
    perturb.add_argument("--op-types", nargs="+", default=list(case_lib.OP_TYPES), choices=case_lib.ALL_CATEGORIES)
    perturb.add_argument("--mean-insert", type=float, default=10.0)
    perturb.add_argument("--mean-delete", type=float, default=10.0)
    perturb.add_argument("--lag", type=int, default=None, help="Fixed edit lag; default samples it uniformly")
    perturb.add_argument("--corruption", default="random", choices=("random", "transplant"))
    perturb.add_argument("--min-prefix", type=int, default=8)
    perturb.add_argument("--min-tail", type=int, default=8)
    perturb.add_argument("--repeats", type=int, default=1, help="Cases per (window, op_type)")
    perturb.add_argument(
        "--skip-noop",
        action="store_true",
        help="Skip the no-op / clean-trajectory (`none`) category: drops it from --op-types, and "
        "with --suite skips the tf_none phase and the gen_none free-gen phase entirely",
    )

    gen = parser.add_argument_group("free generation")
    gen.add_argument("--free-gen", action="store_true", help="Also run the free-running rollout mode")
    gen.add_argument("--gen-max-new-tokens", type=int, default=64)
    gen.add_argument("--gen-rollouts", type=int, default=1)
    gen.add_argument("--gen-greedy", action="store_true")
    gen.add_argument("--gen-temperature", type=float, default=0.7)
    gen.add_argument("--gen-top-p", type=float, default=0.8)
    gen.add_argument("--gen-top-k", type=int, default=50)
    gen.add_argument(
        "--gen-marker-calibration-weight",
        type=float,
        default=1.0,
        help="Scale P(EDIT) in logit space by the sampled start index's probability relative to "
        "chance, before marker sampling. A `generation_config` field, not a trained one -- see "
        "`AutoIndexerModelBase._sample`. Also fed to the teacher-forced `would_trigger`/"
        "`marker_logit_margin` measurement so it matches the same firing decision.",
    )
    gen.add_argument(
        "--gen-calibration-bias",
        type=float,
        default=0.0,
        help="Bias term to reduce the frequency that the edit token is fired. Also fed to the "
        "teacher-forced measurement, like `--gen-marker-calibration-weight`.",
    )
    gen.add_argument(
        "--gen-marker-top-p",
        type=float,
        default=1.0,
        help="Nucleus filtering for the marker head's own softmax, independent of "
        "--gen-top-p (which only shapes the vocabulary text): markers whose cumulative "
        "probability mass falls below `1 - p` are excluded before sampling. 1.0 (default) "
        "disables it. A `generation_config` field, not a trained one -- see "
        "`AutoIndexerModelBase._sample`.",
    )
    gen.add_argument(
        "--gen-marker-temperature",
        type=float,
        default=1.0,
        help="Temperature rescaling for the marker head's own softmax, independent of "
        "--gen-temperature (which only shapes the vocabulary text), applied before "
        "--gen-marker-top-p's nucleus filtering. 1.0 (default) disables it. A "
        "`generation_config` field, not a trained one -- see `AutoIndexerModelBase._sample`.",
    )
    gen.add_argument(
        "--gen-stochastic-mid-edit",
        action="store_true",
        help="Sample content tokens stochastically (not greedily) while an edit is open, using the "
        "configured temperature/top-p/top-k.",
    )
    gen.add_argument(
        "--gen-max-delete-span",
        type=int,
        default=None,
        help="Cap the end cursor to at most this many tokens past the sampled start, bounding "
        "how much a single edit can delete -- a safety valve against rare, severe mislocalized "
        "end-cursor 'blast radius' failures, not a change to ordinary correctly-sized deletions. "
        "See `AutoIndexerModelBase._sample`.",
    )
    gen.add_argument(
        "--gen-exclude-prompt-from-edits",
        action="store_true",
        help="Lock the cursor out of the draft, matching `AutoIndexerModelBase._sample`'s own "
        "default for chat generation (a response must not edit its prompt). Off by default here "
        "because free-gen's `draft` *is* the document to repair in place -- the cursor needs to "
        "be able to point into it. See `generation_config.exclude_prompt_from_edits`.",
    )
    gen.add_argument("--gen-clean-control", action="store_true", help="Also roll out uncorrupted drafts")
    gen.add_argument(
        "--gen-batch-size", type=int, default=1,
        help="Free-generation rollouts per `generate()` call (left-padded batching, see "
        "`free_gen.batch_rollout`). 1 (default) reproduces the original one-case-at-a-time "
        "behavior; >1 trades a bit of write-streaming granularity (records are now flushed to "
        "disk a batch at a time, not one case at a time) for GPU utilization during --genonly.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-traces", action="store_true", help="Keep the per-position P(edit) traces")
    parser.add_argument("--save-streams", action="store_true", help="Keep raw token streams, for scripts/edit_eval/examples.py")

    suite = parser.add_argument_group("suite (run_evals.sh)")
    suite.add_argument("--suite", action="store_true", help="Run the full eval suite in one process (model loaded once)")
    suite.add_argument("--tf-num-windows", type=int, default=500)
    suite.add_argument("--free-gen-num-windows", type=int, default=100)
    suite.add_argument("--genonly", action="store_true", help="With --suite: skip teacher-forced phases")
    return parser


def load_model(args):
    checkpoint_dir = resolve_checkpoint(args.checkpoint, args.checkpoint_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True
    # `marker_calibration_weight`/`greedy_mid_edit`/`max_delete_span`/`exclude_prompt_from_edits`/
    config_overrides = apply_config_overrides(model.config, args.config)
    return model, tokenizer, checkpoint_dir, config_overrides


def build_cases(args, windows: List[Dict], vocab_ids: np.ndarray, rng: np.random.Generator) -> List[case_lib.EditCase]:
    donors = [w["tokens"] for w in windows]
    op_types = [op for op in args.op_types if not (args.skip_noop and op == case_lib.NO_EDIT)]
    built: List[case_lib.EditCase] = []
    for i, window in enumerate(windows):
        for op_type in op_types:
            for _ in range(args.repeats):
                if args.corruption == "random":
                    corruption_fn = partial(case_lib.random_corruption, vocab_ids=vocab_ids)
                else:
                    corruption_fn = partial(case_lib.transplant_corruption, donor=donors[(i + 1) % len(donors)])
                case = case_lib.sample_case(
                    rng,
                    window["tokens"],
                    op_type,
                    mean_insert=args.mean_insert,
                    mean_delete=args.mean_delete,
                    corruption_fn=lambda r, n, fn=corruption_fn: fn(r, n),
                    lag=args.lag,
                    min_prefix=args.min_prefix,
                    min_tail=args.min_tail,
                )
                if case is None:
                    continue
                case.source, case.doc_id, case.window_offset = window["source"], window["doc_id"], window["window_offset"]
                built.append(case)
    return built


def load_windows(args, tokenizer, pool_size: int) -> List[Dict]:
    key = f"{args.window_len}_{pool_size}_{args.seed}"
    return data_lib.load_or_build_windows(
        args.window_cache_dir,
        key,
        lambda: data_lib.tokenize_windows(
            data_lib.load_val_texts(
                args.datamix_config,
                num_val_samples_per_source=args.num_val_samples_per_source,
                local_data_dir=args.local_data_dir,
                sources=args.sources,
            ),
            tokenizer,
            args.window_len,
            pool_size,
            np.random.default_rng(args.seed),
        ),
    )


def run_phase(
    args,
    model,
    tokenizer,
    checkpoint_dir: Path,
    config_overrides: Dict[str, Any],
    all_windows: List[Dict],
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    vocab_ids = model.non_special_token_ids.cpu().numpy()

    windows = all_windows[: args.num_windows]
    print(f"{len(windows)} clean windows of {args.window_len} tokens", flush=True)

    eval_cases = build_cases(args, windows, vocab_ids, rng)
    print(f"{len(eval_cases)} eval cases across {args.op_types}", flush=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                **vars(args),
                "checkpoint_dir": str(checkpoint_dir),
                "config_overrides": config_overrides,
                "checkpoint_perturb": model.config.perturb,
                "marker_loss_weight": getattr(model.config, "marker_loss_weight", 1.0),
                "marker_focal_gamma": getattr(model.config, "marker_focal_gamma", 1.0),
            },
            f,
            indent=2,
        )

    records_path = out_dir / "cases.jsonl"
    start_time = time.time()
    # Free generation is batched `args.gen_batch_size` cases at a time (see `free_gen.batch_rollout`).
    gen_batch_size = max(1, args.gen_batch_size)
    num_written = 0
    with open(records_path, "w") as f:
        for chunk_start in tqdm(range(0, len(eval_cases), gen_batch_size)):
            chunk = list(enumerate(eval_cases[chunk_start : chunk_start + gen_batch_size], start=chunk_start))
            chunk_records = []
            for i, case in chunk:
                no_edit_case = case.op_type == case_lib.NO_EDIT
                clean_built = case_lib.build_clean_stream(case.clean, model.marker_map, device=device)
                if no_edit_case:
                    record = teacher_forced.measure_no_edit_case(model, case, clean_built, device)
                else:
                    built = case_lib.build_stream(case, model.marker_map, device=device)
                    record = teacher_forced.measure_case(
                        model, case, built, clean_built, device,
                        calibration_weight=args.gen_marker_calibration_weight,
                        calibration_bias=args.gen_calibration_bias,
                    )
                if not args.save_traces:
                    record.pop("p_edit_trace", None)
                    record.pop("p_edit_trace_clean", None)

                if args.save_streams:
                    record["clean_tokens"] = case.clean
                    record["corruption"] = case.corruption
                chunk_records.append((i, case, no_edit_case, record))

            if args.free_gen:
                for _, _, _, record in chunk_records:
                    record["rollouts"] = []
                drafts = [case.draft() for _, case, _, _ in chunk_records]
                for _ in range(args.gen_rollouts):
                    batch_results = free_gen.batch_rollout(
                        model, drafts, device, args.gen_max_new_tokens,
                        do_sample=not args.gen_greedy, temperature=args.gen_temperature, top_p=args.gen_top_p,
                        top_k=args.gen_top_k,
                        marker_calibration_weight=args.gen_marker_calibration_weight,
                        greedy_mid_edit=not args.gen_stochastic_mid_edit, max_delete_span=args.gen_max_delete_span,
                        calibration_bias=args.gen_calibration_bias,
                        exclude_prompt_from_edits=args.gen_exclude_prompt_from_edits,
                        marker_top_p=args.gen_marker_top_p,
                        marker_temperature=args.gen_marker_temperature,
                    )
                    for (_, case, no_edit_case, record), (tokens, indices) in zip(chunk_records, batch_results):
                        measure = free_gen.measure_clean_rollout if no_edit_case else free_gen.measure_rollout
                        rollout_record = measure(case, model.marker_map, tokens, indices, device=device)
                        if args.save_streams:
                            rollout_record["tokens"] = tokens
                            rollout_record["indices"] = indices
                        record["rollouts"].append(rollout_record)

                if args.gen_clean_control:
                    control_entries = [(case, record) for _, case, no_edit_case, record in chunk_records if not no_edit_case]
                    if control_entries:
                        clean_drafts = [case.clean[: case.restored_len] for case, _ in control_entries]
                        control_results = free_gen.batch_rollout(
                            model, clean_drafts, device, args.gen_max_new_tokens,
                            do_sample=not args.gen_greedy, temperature=args.gen_temperature, top_p=args.gen_top_p,
                            top_k=args.gen_top_k,
                            marker_calibration_weight=args.gen_marker_calibration_weight,
                            greedy_mid_edit=not args.gen_stochastic_mid_edit, max_delete_span=args.gen_max_delete_span,
                            calibration_bias=args.gen_calibration_bias,
                            exclude_prompt_from_edits=args.gen_exclude_prompt_from_edits,
                            marker_top_p=args.gen_marker_top_p,
                        )
                        for (case, record), (tokens, indices) in zip(control_entries, control_results):
                            control_records, _, _ = free_gen.replay_edit_stream(model.marker_map, tokens, indices, device=device)
                            record["clean_control_num_edits"] = len(control_records)

            for i, case, no_edit_case, record in chunk_records:
                record["case_index"] = i
                f.write(json.dumps(record) + "\n")
            f.flush()
            prev_written, num_written = num_written, num_written + len(chunk_records)
            # Same "every 20 cases" cadence as the original per-case loop
            if num_written // 20 > prev_written // 20 or num_written == len(eval_cases):
                rate = num_written / (time.time() - start_time)
                print(f"  {num_written}/{len(eval_cases)} cases ({rate:.2f}/s)", flush=True)

    print(f"Wrote {records_path}", flush=True)


def _phase_args(base_args, out_subdir: str, **overrides):
    args = argparse.Namespace(**vars(base_args))
    args.out_dir = str(Path(base_args.out_dir) / out_subdir)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_suite(args) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint (once)...", flush=True)
    model, tokenizer, checkpoint_dir, config_overrides = load_model(args)
    torch.manual_seed(args.seed)

    pool_size = max(args.window_pool, args.tf_num_windows, args.free_gen_num_windows)
    all_windows = load_windows(args, tokenizer, pool_size)
    print(f"Cached {len(all_windows)} windows (pool size {pool_size})", flush=True)
    if args.skip_noop:
        print("--skip-noop: omitting tf_none and gen_none phases", flush=True)

    phases = []
    if not args.genonly:
        phases.append(
            (
                f"Teacher-forced ({args.tf_num_windows} windows × insert/delete/substitute)",
                _phase_args(
                    args,
                    "tf",
                    num_windows=args.tf_num_windows,
                    op_types=list(case_lib.OP_TYPES),
                    save_traces=True,
                    save_streams=False,
                    free_gen=False,
                ),
            )
        )
        if not args.skip_noop:
            phases.append(
                (
                    f"Teacher-forced detection on uncorrupted drafts ({args.tf_num_windows} windows)",
                    _phase_args(
                        args,
                        "tf_none",
                        num_windows=args.tf_num_windows,
                        op_types=[case_lib.NO_EDIT],
                        save_traces=True,
                        save_streams=False,
                        free_gen=False,
                    ),
                )
            )

    gen_ops = case_lib.OP_TYPES if args.skip_noop else case_lib.OP_TYPES + (case_lib.NO_EDIT,)
    for op in gen_ops:
        phases.append(
            (
                f"Free generation — {op} ({args.free_gen_num_windows} examples)",
                _phase_args(
                    args,
                    f"gen_{op}",
                    num_windows=args.free_gen_num_windows,
                    op_types=[op],
                    save_traces=False,
                    save_streams=True,
                    free_gen=True,
                ),
            )
        )

    for label, phase_args in phases:
        print(f"\n=== {label} ===", flush=True)
        run_phase(phase_args, model, tokenizer, checkpoint_dir, config_overrides, all_windows)


def main() -> None:
    args = build_parser().parse_args()
    if args.suite:
        run_suite(args)
        return

    torch.manual_seed(args.seed)
    model, tokenizer, checkpoint_dir, config_overrides = load_model(args)
    pool_size = max(args.window_pool, args.num_windows)
    all_windows = load_windows(args, tokenizer, pool_size)
    run_phase(args, model, tokenizer, checkpoint_dir, config_overrides, all_windows)


if __name__ == "__main__":
    main()
