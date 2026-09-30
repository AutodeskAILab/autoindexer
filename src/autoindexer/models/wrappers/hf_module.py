import os
import tempfile
from functools import lru_cache
from typing import Optional, Type

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from autoindexer.models import get_hf_model

# Metadata fields on a pretrained checkpoint's config that describe *that* checkpoint/model class rather than architecture hyperparameters
_PRETRAINED_CONFIG_META_FIELDS = ("model_type", "architectures", "transformers_version", "auto_map", "torch_dtype", "dtype")


@lru_cache(maxsize=None)
def _resolve_pretrained_checkpoint(pretrained_model_name_or_path: str) -> str:
    """Sync an `s3://...` checkpoint to a local directory and return that local path"""
    if not pretrained_model_name_or_path.startswith("s3://"):
        return pretrained_model_name_or_path

    from autoindexer.utils.aws import aws_s3_sync

    local_dir = os.path.join(tempfile.gettempdir(), "autoindexer_s3_checkpoints", pretrained_model_name_or_path[len("s3://"):])
    os.makedirs(local_dir, exist_ok=True)
    aws_s3_sync(pretrained_model_name_or_path, local_dir)
    return local_dir


def _build_pretrained_backbone_config(
    config_class: Type[PretrainedConfig],
    pretrained_model_name_or_path: str,
    model_config: Optional[dict],
) -> PretrainedConfig:
    """Build the pretrained backbone's config, with AutoIndexer-specific fields overlaid."""
    pretrained_config_dict = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True).to_dict()
    for field in _PRETRAINED_CONFIG_META_FIELDS:
        pretrained_config_dict.pop(field, None)
    pretrained_config_dict.update(model_config or {})
    return config_class(**pretrained_config_dict)


def _copy_into(dest: torch.nn.Parameter, source: torch.Tensor) -> None:
    """`dest[:source.shape].copy_(source)`, gathering both sides first if either is a ZeRO-3-partitioned parameter"""
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if not is_deepspeed_zero3_enabled():
        dest[tuple(slice(0, size) for size in source.shape)].copy_(source)
        return

    import deepspeed

    with deepspeed.zero.GatheredParameters([dest, source], modifier_rank=0):
        dest[tuple(slice(0, size) for size in source.shape)].copy_(source)


def _load_pretrained_backbone_weights(model: PreTrainedModel, pretrained_model_name_or_path: str, torch_dtype: torch.dtype) -> None:
    """Copy a pretrained causal-LM checkpoint's transformer backbone and LM head weights into `model`."""

    # device_map="cpu" + low_cpu_mem_usage=True load the checkpoint once on CPU instead of materializing a full extra copy
    pretrained = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, dtype=torch_dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    pretrained_backbone = pretrained.get_decoder()

    backbone_params = dict(model.model.named_parameters())
    with torch.no_grad():
        for name, param in pretrained_backbone.named_parameters():
            dest = backbone_params.get(name)
            if dest is None:
                print(f"Skipping pretrained backbone param {name!r}: no matching parameter on the model backbone.")
                continue
            _copy_into(dest, param)

        pretrained_lm_head = pretrained.get_output_embeddings()
        if pretrained_lm_head is not None:
            _copy_into(model.lm_head.weight, pretrained_lm_head.weight)

    del pretrained


def _apply_lora(model: PreTrainedModel, lora_config: dict) -> PreTrainedModel:
    """Wrap `model` with a LoRA adapter via `peft`."""
    import peft
    from omegaconf import OmegaConf

    # Guard against OmegaConf containers
    if OmegaConf.is_config(lora_config):
        lora_config = OmegaConf.to_container(lora_config, resolve=True)

    model = peft.get_peft_model(model, peft.LoraConfig(**lora_config))
    # Frozen base model params otherwise block gradient flow through checkpointed layers.
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def build_hf_model(
    model_name: str,
    model_config: Optional[dict] = None,
    gradient_checkpointing: bool = True,
    dtype: str = "float32",
    lora: Optional[dict] = None,
    pretrained: bool = False,
    pretrained_model_name_or_path: Optional[str] = None,
) -> PreTrainedModel:
    """Build a HuggingFace `PreTrainedModel` for AutoIndexer/Llama/Qwen3/etc."""
    if model_config is not None and model_config.get("attn_implementation"):
        from autoindexer.models import register_attn_implementation

        register_attn_implementation(model_config["attn_implementation"])

    torch_dtype = getattr(torch, dtype)

    model_class, config_class = get_hf_model(model_name)
    if model_class is None:
        model_name = _resolve_pretrained_checkpoint(model_name)
        if pretrained:
            # device_map="cpu" + low_cpu_mem_usage=True avoid materializing a full extra copy of the checkpoint per rank
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, dtype=torch_dtype, device_map="cpu", low_cpu_mem_usage=True
            )
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)
    else:
        if pretrained:
            if pretrained_model_name_or_path is None:
                raise ValueError(
                    f"pretrained=True has no effect for model_name={model_name!r} on its own: it resolves to "
                    "a from-scratch architecture shorthand via get_hf_model, not a checkpoint with weights to "
                    "load. Also set pretrained_model_name_or_path to a real hub/local checkpoint (e.g. "
                    "'Qwen/Qwen3-8B-Base') to initialize this architecture's backbone from it."
                )
            pretrained_model_name_or_path = _resolve_pretrained_checkpoint(pretrained_model_name_or_path)
            config = _build_pretrained_backbone_config(config_class, pretrained_model_name_or_path, model_config)
            model = model_class(config)
            _load_pretrained_backbone_weights(model, pretrained_model_name_or_path, torch_dtype)
            model = model.to(torch_dtype)
        else:
            if pretrained_model_name_or_path is not None:
                raise ValueError("pretrained_model_name_or_path was set but pretrained=False; set pretrained=True to use it.")
            config = config_class(**(model_config or {}))
            model = model_class(config).to(torch_dtype)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if lora is not None:
        model = _apply_lora(model, lora)

    # Under `deepspeed.zero.Init()` each rank only holds its shard, so `numel()` reports a fraction of the real size
    total_params = sum(getattr(p, "ds_numel", p.numel()) for p in model.parameters())
    trainable_params = sum(getattr(p, "ds_numel", p.numel()) for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def merge_lora_checkpoint(adapter_checkpoint_dir: str, output_dir: str, **build_hf_model_kwargs) -> None:
    """Merge a LoRA adapter checkpoint into a fresh copy of its base model and save the result"""
    import peft

    base_model = build_hf_model(**{**build_hf_model_kwargs, "lora": None})
    merged = peft.PeftModel.from_pretrained(base_model, adapter_checkpoint_dir).merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)

    # `save_pretrained` above only writes weights/config
    tokenizer_source = build_hf_model_kwargs.get("pretrained_model_name_or_path") or build_hf_model_kwargs.get("model_name")
    if tokenizer_source:
        from transformers import AutoTokenizer

        tokenizer_source = _resolve_pretrained_checkpoint(tokenizer_source)
        AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True).save_pretrained(output_dir)

    print(f"Merged LoRA checkpoint saved to {output_dir}")
