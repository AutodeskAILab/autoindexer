import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.utils import ModelOutput

_GENERATION_LOG_SKIP_KEYS = frozenset(
    {"sequences", "past_key_values", "scores", "logits", "attentions", "hidden_states"}
)


class SampleGenerationCallback(TrainerCallback):
    """Generates and logs sample sequences/images to Comet every `log_sample_freq` evaluations."""

    def __init__(
        self,
        trainer,
        tokenizer=None,
        generation_config: Optional[Dict[str, Any]] = None,
        image_shape: Optional[Tuple[int, ...]] = None,
        log_sample_freq: int = 0,
        max_samples: int = 5,
        prompt_len: int = 1,
    ):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.image_shape = image_shape
        self.log_sample_freq = log_sample_freq
        self.max_samples = max_samples
        self.prompt_len = prompt_len
        self._eval_count = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.log_sample_freq <= 0 or not self.generation_config:
            return

        self._eval_count += 1
        if self._eval_count % self.log_sample_freq != 0 or not state.is_world_process_zero:
            return

        try:
            import comet_ml

            experiment = comet_ml.get_running_experiment()
        except ImportError:
            experiment = None
        if experiment is None:
            return

        batch = self._get_eval_batch()
        if batch is None:
            return

        gen_output = self._generate(batch)
        if gen_output is not None:
            self._log_samples(experiment, gen_output, state)

    def _get_eval_batch(self):
        eval_dataset = self.trainer.eval_dataset
        if eval_dataset is None:
            return None
        dataloader = self.trainer.get_eval_dataloader(eval_dataset)
        try:
            return next(iter(dataloader))
        except StopIteration:
            return None

    def _generate(self, batch: Dict[str, torch.Tensor]):
        model = self.trainer.model
        if "input_ids" not in batch:
            return None

        x = batch["input_ids"].to(model.device)
        n_samples = min(self.max_samples, x.size(0))
        prompts = x[:n_samples, : self.prompt_len]

        generation_config = self.generation_config
        if not isinstance(generation_config, GenerationConfig):
            config_dict = dict(generation_config)
            if self.tokenizer:
                config_dict.setdefault("pad_token_id", self.tokenizer.pad_token_id)
                config_dict.setdefault("eos_token_id", self.tokenizer.eos_token_id)
                config_dict.setdefault("bos_token_id", self.tokenizer.bos_token_id)
            config_dict["return_dict_in_generate"] = True
            generation_config = GenerationConfig(**config_dict)
            self.generation_config = generation_config

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return model.generate(prompts, generation_config=generation_config)
        except Exception as e:  # pragma: no cover - defensive, generation is best-effort logging
            logging.info(f"SampleGenerationCallback: generation failed: {e}")
            return None
        finally:
            model.train(was_training)

    def _log_samples(self, experiment, gen_output, state: TrainerState) -> None:
        if isinstance(gen_output, ModelOutput):
            for key, value in gen_output.items():
                if key in _GENERATION_LOG_SKIP_KEYS or not isinstance(value, torch.Tensor):
                    continue
                self._log_tensor(experiment, key, value, state)
            gen_seq = gen_output.sequences
        else:
            gen_seq = gen_output

        gen_seq = gen_seq.cpu().numpy() if isinstance(gen_seq, torch.Tensor) else gen_seq

        if self.image_shape:
            seq_len = int(np.prod(self.image_shape))
            for i, seq in enumerate(gen_seq):
                seq = torch.as_tensor(seq)
                if len(seq) > seq_len:
                    seq = seq[:seq_len]
                elif len(seq) < seq_len:
                    seq = F.pad(seq, (0, seq_len - len(seq)))
                image = seq.reshape(*self.image_shape).float().cpu().numpy()
                experiment.log_image(image, name=f"step {state.global_step:07} generated image {i:02}", image_channels="first")

        for i, seq in enumerate(gen_seq):
            if self.tokenizer:
                text = self.tokenizer.decode(seq, skip_special_tokens=False)
            else:
                seq_values = seq.tolist() if isinstance(seq, (torch.Tensor, np.ndarray)) else list(seq)
                text = f"sequence: {seq_values}"
            experiment.log_text(
                text=text,
                metadata={"field": "sequence", "epoch": state.epoch, "sample_idx": i},
                step=state.global_step,
            )

    def _log_tensor(self, experiment, field: str, tensor: torch.Tensor, state: TrainerState) -> None:
        tensor = tensor.detach().cpu()
        if tensor.ndim <= 1:
            experiment.log_text(
                text=f"{field}: {tensor.tolist()}",
                metadata={"field": field, "kind": "tensor", "epoch": state.epoch},
                step=state.global_step,
            )
            return
        for i, row in enumerate(tensor):
            experiment.log_text(
                text=f"{field}: {row.tolist()}",
                metadata={"field": field, "kind": "tensor", "epoch": state.epoch, "sample_idx": i},
                step=state.global_step,
            )
