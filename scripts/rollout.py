"""Run a single AutoIndexer rollout from a trained checkpoint."""

import argparse
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Registers AutoIndexer's config/model/tokenizer classes with the `Auto*` factories
import autoindexer.models.autoindexer  # noqa: F401


def resolve_checkpoint(checkpoint: str, cache_dir: str) -> Path:
    if not checkpoint.startswith("s3://"):
        return Path(checkpoint)

    local_dir = Path(cache_dir) / checkpoint.rstrip("/").rsplit("/", 2)[-2] / "final_checkpoint"
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {checkpoint} -> {local_dir}")
    subprocess.run(["aws", "s3", "sync", checkpoint.rstrip("/") + "/", str(local_dir), "--only-show-errors"], check=True)
    return local_dir


def render_raw_stream(tokenizer, marker_map: dict, tokens: Sequence[int], indices: Sequence[Tuple[int, int]]) -> str:
    """Decode a rollout's raw token stream, rendering edit/return/eos markers inline."""
    marker_names = {token_id: name for name, token_id in marker_map.items()}
    parts: List[str] = []
    pending: List[int] = []
    for i, token in enumerate(tokens):
        name = marker_names.get(token)
        if name is None:
            pending.append(token)
            continue
        if pending:
            parts.append(tokenizer.decode(pending))
            pending = []
        if name == "edit":
            end = indices[i + 1][1] if i + 1 < len(indices) else "?"
            parts.append(f"⟦edit start={indices[i][0]} end={end}⟧")
        elif name == "return_to_end":
            parts.append("⟦return_to_end⟧")
        else:
            parts.append("⟦eos⟧")
    if pending:
        parts.append(tokenizer.decode(pending))
    return "".join(parts)


def truncate_at_eos(tokens: Sequence[int], eos_id: int, prompt_len: int) -> List[int]:
    """Trim the trailing zero padding `_sample` leaves in `total_sampled_tokens` once a row finishes."""
    tokens = list(tokens)
    for i in range(prompt_len, len(tokens)):
        if tokens[i] == eos_id:
            return tokens[: i + 1]
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint dir or s3:// URI")
    parser.add_argument("--prompt", default="hello, what is your name?")
    parser.add_argument("--cache-dir", default=str(Path.home() / "checkpoints"), help="Where s3:// checkpoints are synced to")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50, help="0 disables top-k (marker tokens are exempt from it either way)")
    parser.add_argument("--greedy", action="store_true", help="Argmax instead of sampling")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    # The fused cutedsl/triton backends are only needed for long prefills
    parser.add_argument("--attn-implementation", default="autoindexer_eager")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint_dir = resolve_checkpoint(args.checkpoint, args.cache_dir)
    print(f"Loading tokenizer from {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    print(f"Loading model from {checkpoint_dir} (dtype={args.dtype}, attn={args.attn_implementation})")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )

    print(f"\nPrompt ({prompt_len} tokens, raw text -- this is a base-model checkpoint, no chat template): {args.prompt!r}")
    for rollout in range(args.num_rollouts):
        with torch.no_grad():
            output = model.generate(input_ids, generation_config=generation_config)

        sampled_tokens = truncate_at_eos(output.sampled_tokens[0].tolist(), model.marker_map["eos"], prompt_len)
        sampled_indices = output.sampled_indices[0].tolist()
        raw_stream = render_raw_stream(tokenizer, model.marker_map, sampled_tokens, sampled_indices)
        resolved = tokenizer.decode(output.sequences[0], skip_special_tokens=False)

        header = f"=== rollout {rollout + 1}/{args.num_rollouts} " if args.num_rollouts > 1 else "=== rollout "
        print(f"\n{header}({len(sampled_tokens) - prompt_len} new tokens) ===")
        print("--- raw generation stream (prompt + markers) ---")
        print(raw_stream)
        print("--- resolved sequence (edits applied) ---")
        print(resolved)


if __name__ == "__main__":
    main()
