"""Lightweight, opt-in timing guards."""

import contextlib
import os
import time

import torch

_PROFILE = os.environ.get("AUTOINDEXER_PROFILE", "0") == "1"


@contextlib.contextmanager
def timed(label: str, sync: bool = True):
    if not _PROFILE:
        yield
        return
    _sync = sync and torch.cuda.is_available()
    if _sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if _sync:
            torch.cuda.synchronize()
        print(f"[PROFILE] {label}: {(time.perf_counter() - t0) * 1e3:.3f} ms", flush=True)
