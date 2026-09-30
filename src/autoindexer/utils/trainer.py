import copy
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import DatasetDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import Trainer, TrainerCallback, default_data_collator

from autoindexer.utils.args import AutoIndexerTrainingArguments
from autoindexer.models.autoindexer._profiling import _PROFILE
from autoindexer.models.wrappers.hf_module import merge_lora_checkpoint
from autoindexer.utils.generation_logging import SampleGenerationCallback
from autoindexer.utils.training_monitors import MonitoringManager, TimingCallback, compute_token_metrics_factory

if os.environ.get("TMPDIR"):
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)


def build_trainer(hydra_config: DictConfig, exp_name: Optional[str] = None) -> Tuple[Trainer, "DictConfig", DatasetDict]:
    """Instantiate ``hydra_config`` and build a fully-configured `transformers.Trainer`, without running it."""
    print("=== Instantiated ===")
    # `data_collator.tokenizer_name` interpolates `${model.model_name}`, so the `model:` node has
    # to stay in the config -- only its `_target_` is dropped, deferring construction without
    # breaking that reference.
    deferred_config = copy.deepcopy(hydra_config)
    model_node = hydra_config.get("model")
    if model_node is not None:
        with open_dict(deferred_config):
            deferred_config.model.pop("_target_", None)

    config = instantiate(deferred_config)
    print(config)
    print("====================")

    exp_name = exp_name or (datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + config.experiment_name)

    os.environ.setdefault("COMET_PROJECT_NAME", config.project_name)

    comet_workspace = config.get("comet_workspace")
    if comet_workspace:
        os.environ.setdefault("COMET_WORKSPACE", str(comet_workspace))

    log_save_dir = config.save_dir
    output_dir = os.path.join(log_save_dir, exp_name)

    dataset = config.data
    if not isinstance(dataset, dict):
        dataset = {"train": dataset}
    train_dataset = dataset["train"]
    val_dataset = dataset.get("validation")
    data_collator = config.get("data_collator") or default_data_collator

    trainer_kwargs = OmegaConf.to_container(config.trainer, resolve=True) if "trainer" in config else {}
    if config.mode != "train":
        trainer_kwargs["logging_steps"] = 1

    trainer_kwargs.setdefault("eval_strategy", "epoch" if val_dataset is not None else "no")
    trainer_kwargs.setdefault("save_strategy", "epoch")
    trainer_kwargs.setdefault("report_to", ["comet_ml", "tensorboard"])
    trainer_kwargs.setdefault("logging_dir", output_dir)
    trainer_kwargs.setdefault("run_name", exp_name)
    # `Trainer.__init__` auto-detects which batch keys are "labels" via `transformers.utils.find_labels(model.__class__)`
    trainer_kwargs.setdefault("label_names", ["labels"])

    compile_mode = os.environ.get("AUTOINDEXER_TORCH_COMPILE_MODE")
    if compile_mode:
        # Compiling through `TrainingArguments` (accelerate applies it to the wrapped model
        trainer_kwargs.setdefault("torch_compile", True)
        if compile_mode != "default":
            trainer_kwargs.setdefault("torch_compile_mode", compile_mode)
        print(f"[compile] enabling TrainingArguments torch_compile (mode={compile_mode!r})", flush=True)

    training_args = AutoIndexerTrainingArguments(output_dir=output_dir, **trainer_kwargs)

    if training_args.num_train_samples is not None:
        # `world_size` is only resolved once `TrainingArguments` has set up the distributed state
        samples_per_step = (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        )
        training_args.max_steps = math.ceil(training_args.num_train_samples / samples_per_step)
        print(f"Training for {training_args.max_steps:,} steps ({training_args.num_train_samples:,} samples at {samples_per_step:,}/step)")

    if training_args.num_eval_samples is not None and val_dataset is not None:
        if training_args.num_eval_samples < len(val_dataset):
            # Shuffle before slicing so the cap samples the whole split rather than its leading rows
            eval_seed = training_args.data_seed if training_args.data_seed is not None else training_args.seed
            print(f"Sampling eval dataset down from {len(val_dataset):,} to {training_args.num_eval_samples:,} rows (shuffle seed {eval_seed})")
            val_dataset = val_dataset.shuffle(seed=eval_seed).select(range(training_args.num_eval_samples))

    # Only now is it safe to build the model: `training_args` above has installed the DeepSpeed config weakref
    model = instantiate(model_node, _convert_="all") if model_node is not None else None

    # For LoRA runs, `configure_autoindexer_trainer` needs a `build_hf_model`-shaped kwargs dict to rebuild a fresh
    lora_config = None
    lora_base_model_kwargs = None
    if model_node is not None:
        model_kwargs = OmegaConf.to_container(model_node, resolve=True)
        model_kwargs.pop("_target_", None)
        lora_config = model_kwargs.get("lora")
        if lora_config is not None:
            lora_base_model_kwargs = {**model_kwargs, "lora": None, "gradient_checkpointing": False}

    callbacks = []
    if _PROFILE:
        callbacks.append(TimingCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=getattr(data_collator, "tokenizer", None),
        callbacks=callbacks,
    )
    configure_autoindexer_trainer(
        trainer,
        training_args,
        lora_config=lora_config,
        lora_base_model_kwargs=lora_base_model_kwargs,
    )
    return trainer, config, dataset


def configure_autoindexer_trainer(
    trainer: Trainer,
    args: AutoIndexerTrainingArguments,
    lora_config: Optional[dict] = None,
    lora_base_model_kwargs: Optional[dict] = None,
) -> Trainer:
    """Attach all AutoIndexer-specific behavior onto a *standard* `transformers.Trainer` instance via composition."""
    _drop_empty_optimizer_param_groups(trainer)
    modules_to_save = (lora_config or {}).get("modules_to_save")
    if modules_to_save:
        exclude_modules_to_save_from_weight_decay(trainer, modules_to_save)
    _attach_extra_metric_logging(trainer)

    manager = MonitoringManager(
        trainer=trainer,
        log_interval=args.monitoring_log_interval,
        options=args.monitoring_options,
    )
    for callback in manager.get_callbacks():
        trainer.add_callback(callback)

    if manager.wants_token_metrics() and trainer.compute_metrics is None:
        preprocess_logits_for_metrics, compute_metrics = compute_token_metrics_factory()
        trainer.compute_metrics = compute_metrics
        trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics

    if args.log_sample_freq > 0 and args.generation_config:
        trainer.add_callback(
            SampleGenerationCallback(
                trainer=trainer,
                tokenizer=getattr(trainer, "processing_class", None),
                generation_config=args.generation_config,
                image_shape=args.image_shape,
                log_sample_freq=args.log_sample_freq,
                prompt_len=args.sample_log_prompt_len,
            )
        )

    if args.merge_lora_at_end:
        if lora_base_model_kwargs is not None:
            trainer.add_callback(_MergeLoraOnTrainEnd(trainer, lora_base_model_kwargs))
        else:
            trainer.add_callback(_SaveFinalCheckpointOnTrainEnd(trainer))

    return trainer


class _SaveFinalCheckpointOnTrainEnd(TrainerCallback):
    """Saves the trained model under `<output_dir>/final_checkpoint` once training finishes."""

    def __init__(self, trainer: Trainer):
        super().__init__()
        self.trainer = trainer

    def on_train_end(self, args, state, control, **kwargs):
        final_dir = os.path.join(args.output_dir, "final_checkpoint")
        # `save_model` is collective under DeepSpeed (every rank must join the gather).
        self.trainer.save_model(final_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if not state.is_world_process_zero:
            return
        tokenizer = getattr(self.trainer, "processing_class", None)
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(final_dir)
            except Exception as e:
                print(f"[final_checkpoint] Tokenizer save failed ({e!r}); model weights are still at {final_dir!r}.")


class _MergeLoraOnTrainEnd(TrainerCallback):
    """Merges the LoRA adapter into a fresh base model and saves it under `<output_dir>/final_checkpoint` once training finishes."""

    def __init__(self, trainer: Trainer, base_model_kwargs: dict):
        super().__init__()
        self.trainer = trainer
        self.base_model_kwargs = base_model_kwargs

    def on_train_end(self, args, state, control, **kwargs):
        # `save_model` is collective under DeepSpeed (every rank must join the gather), so call it unconditionally
        adapter_dir = os.path.join(args.output_dir, "checkpoint-final-adapter")
        self.trainer.save_model(adapter_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if not state.is_world_process_zero:
            return
        merged_dir = os.path.join(args.output_dir, "final_checkpoint")
        try:
            merge_lora_checkpoint(adapter_dir, merged_dir, **self.base_model_kwargs)
        except Exception as e:
            # The adapter is already safely saved above
            print(f"[merge_lora] Merge failed ({e!r}); trained adapter is still available at {adapter_dir!r}.")
            return


def _drop_empty_optimizer_param_groups(trainer: Trainer) -> None:
    """Wrap `trainer.create_optimizer` to drop weight-decay groups with zero trainable params."""
    original_create_optimizer = trainer.create_optimizer

    def create_optimizer():
        optimizer = original_create_optimizer()
        optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
        return optimizer

    trainer.create_optimizer = create_optimizer


def _param_belongs_to_module(param_name: str, module_path: str) -> bool:
    prefix = module_path if module_path.endswith(".") else f"{module_path}."
    return param_name == module_path or param_name.startswith(prefix) or f".{prefix}" in param_name


def exclude_modules_to_save_from_weight_decay(trainer: Trainer, modules_to_save: list[str]) -> None:
    """Move `lora.modules_to_save` parameters out of the weight-decayed optimizer group."""
    original_create_optimizer = trainer.create_optimizer

    def create_optimizer():
        optimizer = original_create_optimizer()
        saved_params = {
            id(param)
            for name, param in trainer.model.named_parameters()
            if param.requires_grad and any(_param_belongs_to_module(name, module_path) for module_path in modules_to_save)
        }
        if not saved_params:
            return optimizer
        no_decay = next((g for g in optimizer.param_groups if not g.get("weight_decay")), None)
        for group in optimizer.param_groups:
            if not group.get("weight_decay"):
                continue
            moved = [p for p in group["params"] if id(p) in saved_params]
            if not moved:
                continue
            group["params"] = [p for p in group["params"] if id(p) not in saved_params]
            if no_decay is not None:
                no_decay["params"].extend(moved)
            else:
                optimizer.add_param_group({**group, "params": moved, "weight_decay": 0.0})
        return optimizer

    trainer.create_optimizer = create_optimizer


def _attach_extra_metric_logging(trainer: Trainer) -> None:
    """Log every scalar the model returns (e.g. `token_loss`, `index_loss`) alongside `loss`, without overriding `Trainer.compute_loss`."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    trainer._last_batch = None

    def _forward_hook(module, args, kwargs, output):
        if not module.training:
            return
        trainer._last_batch = kwargs
        if not isinstance(output, dict):
            return
        for key, value in output.items():
            if key == "loss" or not isinstance(value, torch.Tensor) or value.numel() != 1:
                continue
            sums[key] += value.detach().item()
            counts[key] += 1

    trainer.model.register_forward_hook(_forward_hook, with_kwargs=True)

    original_log = trainer.log

    def _patched_log(logs: Dict[str, float], *log_args, **log_kwargs) -> None:
        if "loss" in logs and counts:
            for key, count in counts.items():
                if count > 0:
                    logs[key] = round(sums[key] / count, 4)
            sums.clear()
            counts.clear()
        original_log(logs, *log_args, **log_kwargs)

    trainer.log = _patched_log
