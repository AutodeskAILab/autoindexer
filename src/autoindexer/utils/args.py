from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from transformers import TrainingArguments


@dataclass
class AutoIndexerTrainingArguments(TrainingArguments):
    """`transformers.TrainingArguments` extended with the extra knobs that
    used to live on the PyTorch Lightning `trainer:` / `model:` config
    blocks (monitoring callbacks, sample generation logging).
    """

    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": (
                "Defaults to False (overriding HF's True default): AutoIndexer's `data_collator`s "
                "(e.g. `AutoIndexerSFTCollator`) may need non-standard batch columns beyond "
                "directly rather than pre-tokenized model kwargs, so `Trainer`'s automatic column "
                "pruning -- driven off `model.forward`'s signature -- would otherwise strip those "
                "columns before the collator ever sees them."
            )
        },
    )
    num_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "How many training samples to run, as an alternative to `num_train_epochs` "
                "(None keeps the epoch-based schedule). `build_trainer` converts it into the "
                "`max_steps` that `Trainer` understands, dividing by the effective batch size "
                "(`per_device_train_batch_size * gradient_accumulation_steps * world_size`), so "
                "the training budget stays fixed when the batch size or GPU count changes."
            )
        },
    )
    num_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "How many samples each evaluation pass scores (None evaluates the whole split). "
                "`Trainer` has no native equivalent -- `eval_steps` is the *interval between* "
                "evaluations, not their length -- so `build_trainer` implements this by sampling "
                "that many rows out of the eval dataset, via a shuffle seeded with `data_seed` "
                "(falling back to `seed`)."
            )
        },
    )
    monitoring_options: List[str] = field(
        default_factory=list,
        metadata={"help": "Which extra monitoring callbacks to enable, e.g. ['gradients', 'throughput', 'token_accuracy', 'perplexity']."},
    )
    monitoring_log_interval: int = field(
        default=1_000,
        metadata={"help": "Logging interval (in steps) for the extra monitoring callbacks."},
    )
    log_sample_freq: int = field(
        default=0,
        metadata={"help": "Log generated samples every N evaluations (0 disables sample logging)."},
    )
    image_shape: Optional[Tuple[int, ...]] = field(
        default=None,
        metadata={"help": "(C, H, W) shape used to reshape generated sequences back into images for logging."},
    )
    sample_log_prompt_len: int = field(
        default=1,
        metadata={"help": "Number of leading tokens fed to `model.generate` as the prompt for sample logging."},
    )
    generation_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "kwargs used to build a transformers.GenerationConfig for sample logging."},
    )
    merge_lora_at_end: bool = field(
        default=True,
        metadata={
            "help": (
                "At end of training, save a directly-loadable checkpoint under "
                "`<output_dir>/final_checkpoint`. For LoRA runs, merges the adapter into the "
                "base model first; for full fine-tuning runs, saves the trained model directly."
            )
        },
    )
