"""Attention-backend numerical-parity harness (cutedsl vs eager)."""
import copy
import os
import random
import statistics
from datetime import datetime

import comet_ml
import numpy as np
import torch
import transformers
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from torch.utils.data import DataLoader

BACKEND = "cutedsl"          # backend compared against the eager reference
REF = "eager"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "cpt_datamix")
CONFIG_NAME = "autoindexer_8B"
BATCH_SIZE = 8
N_STEPS = 200
SEED = 42
DEVICE = "cuda"


def set_backend(model: torch.nn.Module, backend: str) -> str:
    """Set `config._attn_implementation` to `autoindexer_{backend}`."""
    attn_implementation = f"autoindexer_{backend}"
    model.config._attn_implementation = attn_implementation
    return attn_implementation


def make_optimizer(model, trainer_cfg) -> torch.optim.Optimizer:
    """Mirror `transformers.Trainer.create_optimizer` (AdamW + grouped weight decay), minus the trainer-dependent LR scheduler"""
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": trainer_cfg.get("weight_decay", 0.0)},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=trainer_cfg.get("learning_rate", 3e-4),
        betas=(trainer_cfg.get("adam_beta1", 0.9), trainer_cfg.get("adam_beta2", 0.999)),
    )


def train_step(lm, optimizer, batch, seed) -> dict:
    """One forward/backward/optimizer step under a fixed RNG seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer.zero_grad()
    outputs = lm(**batch)
    outputs.loss.backward()
    optimizer.step()
    return {k: v.item() for k, v in outputs.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1}


def main():
    transformers.set_seed(SEED)

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=CONFIG_NAME)

    # One model, then a deep copy -> identical initial weights, only the attention backend differs
    lm_ref = instantiate(cfg.model).to(DEVICE).train()
    lm_test = copy.deepcopy(lm_ref).to(DEVICE).train()
    print(f"attn_implementation set: {REF}={set_backend(lm_ref, REF)} "
          f"{BACKEND}={set_backend(lm_test, BACKEND)}")

    trainer_cfg = cfg.get("trainer", {})
    opt_ref = make_optimizer(lm_ref, trainer_cfg)
    opt_test = make_optimizer(lm_test, trainer_cfg)

    dataset = instantiate(cfg.data)
    collate_fn = instantiate(cfg.data_collator)
    loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{BACKEND}_vs_{REF}"
    experiment = comet_ml.Experiment(project_name="autoindexer-attn-backend-comparison")
    experiment.set_name(exp_name)

    diffs = []
    step = 0
    for batch in loader:
        if step >= N_STEPS:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        seed = SEED + step  # same seed for both -> identical perturbation this step

        m_ref = train_step(lm_ref, opt_ref, batch, seed)
        m_test = train_step(lm_test, opt_test, batch, seed)

        for key, val in m_ref.items():
            experiment.log_metric(f"comparison/{key}_{REF}", val, step=step)
        for key, val in m_test.items():
            experiment.log_metric(f"comparison/{key}_{BACKEND}", val, step=step)

        diff = abs(m_ref["loss"] - m_test["loss"])
        diffs.append(diff)
        experiment.log_metric("comparison/loss_abs_diff", diff, step=step)
        print(f"step {step:04d}  {REF}={m_ref['loss']:.6f}  "
              f"{BACKEND}={m_test['loss']:.6f}  |diff|={diff:.3e}", flush=True)
        step += 1

    if diffs:
        print(f"\n=== {step} steps: {BACKEND} vs {REF} ===")
        print(f"mean |loss diff| = {statistics.mean(diffs):.3e}")
        print(f"max  |loss diff| = {max(diffs):.3e}")
        print(f"final {REF}={m_ref['loss']:.6f}  final {BACKEND}={m_test['loss']:.6f}")
    experiment.end()


if __name__ == "__main__":
    main()
