"""Clean token windows drawn from the datamix's validation split."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf

from autoindexer.domains.hf_datasets import load_dataset_mix


def load_val_texts(
    datamix_config: str,
    num_val_samples_per_source: Optional[int] = None,
    local_data_dir: Optional[str] = None,
    sources: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    config = OmegaConf.load(datamix_config)
    source_list = OmegaConf.to_container(config.data.sources, resolve=True)
    if sources:
        wanted = set(sources)
        source_list = [s for s in source_list if s.get("name", s["dataset_name"]) in wanted]
        assert source_list, f"No datamix source matched {sorted(wanted)}"

    dataset = load_dataset_mix(
        sources=source_list,
        num_val_samples_per_source=num_val_samples_per_source or config.data.num_val_samples_per_source,
        seed=seed if seed is not None else config.data.seed,
        local_data_dir=local_data_dir,
    )
    return [{"text": row["text"], "source": row["source"]} for row in dataset["validation"]]


def tokenize_windows(
    texts: Sequence[Dict],
    tokenizer,
    window_len: int,
    num_windows: int,
    rng: np.random.Generator,
    min_doc_tokens: Optional[int] = None,
) -> List[Dict]:
    """One random `window_len`-token window per document, until `num_windows` are collected."""
    min_doc_tokens = min_doc_tokens or window_len
    order = rng.permutation(len(texts))
    windows: List[Dict] = []
    for doc_id in order:
        if len(windows) >= num_windows:
            break
        row = texts[int(doc_id)]
        ids = tokenizer(row["text"], add_special_tokens=False).input_ids
        if len(ids) < min_doc_tokens:
            continue
        offset = int(rng.integers(0, len(ids) - window_len + 1))
        windows.append(
            {
                "tokens": ids[offset : offset + window_len],
                "source": row["source"],
                "doc_id": int(doc_id),
                "window_offset": offset,
            }
        )
    return windows


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"windows_{key}.json")


def load_or_build_windows(cache_dir: Optional[str], key: str, build_fn) -> List[Dict]:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = cache_path(cache_dir, key)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    windows = build_fn()
    if cache_dir:
        with open(cache_path(cache_dir, key), "w") as f:
            json.dump(windows, f)
    return windows
