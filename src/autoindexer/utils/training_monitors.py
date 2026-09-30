import logging
import math
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

try:
    import pynvml
except ImportError:
    pynvml = None


logging.getLogger("transformers").setLevel(logging.INFO)


class GradientParameterMonitor(TrainerCallback):
    """Monitors gradient and parameter norms during training."""

    def __init__(self, trainer, log_interval: int = 100):
        super().__init__()
        self.trainer = trainer
        self.log_interval = log_interval

    @staticmethod
    def _norms(tensor: torch.Tensor) -> Dict[str, float]:
        if tensor.numel() == 0:
            # Under DeepSpeed ZeRO-3, `param`/`param.grad` is this rank's local partition shard, not the full parameter
            return {}
        return {
            "l1": torch.norm(tensor, p=1).item(),
            "l2": torch.norm(tensor).item(),
            "linf": torch.max(torch.abs(tensor)).item(),
        }

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.log_interval <= 0 or state.global_step == 0 or state.global_step % self.log_interval != 0:
            return

        model = kwargs.get("model", self.trainer.model)
        metrics: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                for norm_type, value in self._norms(param.grad.detach()).items():
                    metrics[f"grad_norms/{name}/{norm_type}"] = value
            for norm_type, value in self._norms(param.detach()).items():
                metrics[f"param_norms/{name}/{norm_type}"] = value

        if metrics and state.is_world_process_zero:
            self.trainer.log(metrics)


class ThroughputCallback(TrainerCallback):
    """Logs training throughput (tokens/second), using the trainer's last training batch (`trainer._last_batch`)."""

    def __init__(self, trainer, log_interval: int = 100):
        super().__init__()
        self.trainer = trainer
        self.log_interval = log_interval
        self.last_logged_step = 0
        self.last_logged_time: Optional[float] = None
        self.tokens_since_last_log = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.last_logged_time = time.time()
        self.last_logged_step = 0
        self.tokens_since_last_log = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        batch = self.trainer._last_batch
        if not batch or "input_ids" not in batch:
            return

        input_ids = batch["input_ids"]
        self.tokens_since_last_log += input_ids.numel()

        if state.global_step - self.last_logged_step < self.log_interval:
            return

        current_time = time.time()

        if self.last_logged_time is None:
            self.last_logged_time = current_time
            return

        elapsed_time = current_time - self.last_logged_time
        total_tokens = self.tokens_since_last_log

        if dist.is_available() and dist.is_initialized():
            tokens_tensor = torch.tensor(total_tokens, device=input_ids.device)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
            total_tokens = tokens_tensor.item()

        if state.is_world_process_zero and elapsed_time > 0:
            world_size = max(args.world_size, 1)
            tokens_per_second = total_tokens / elapsed_time
            self.trainer.log(
                {
                    "throughput/total_tokens_per_second": tokens_per_second,
                    "throughput/tokens_per_second_per_gpu_avg": tokens_per_second / world_size,
                }
            )

        self.last_logged_step = state.global_step
        self.last_logged_time = current_time
        self.tokens_since_last_log = 0


@dataclass
class GPUStats:
    memory_used: int
    memory_total: int
    utilization: int


class AsyncGPUMonitor:
    """Asynchronous GPU monitoring to avoid blocking the training loop."""

    def __init__(self, device_count: int = 1, polling_interval: float = 0.1):
        self.queue = Queue()
        self.should_stop = False
        self.device_count = device_count
        self.latest_stats: Dict[int, GPUStats] = {}
        self.polling_interval = polling_interval
        self.initialization_successful = False
        self._start_monitor_thread()

    def _start_monitor_thread(self):
        def monitor_loop():
            try:
                pynvml.nvmlInit()
                self.initialization_successful = True
                logging.info("NVML initialized successfully")
            except Exception as e:
                logging.info(f"Failed to initialize pynvml: {e}")
                return

            try:
                actual_device_count = pynvml.nvmlDeviceGetCount()
                self.device_count = min(self.device_count, actual_device_count)
                logging.info(f"Monitoring {self.device_count} GPU devices")
            except pynvml.NVMLError as e:
                logging.info(f"Error getting device count: {e}")
                self.device_count = 1

            gpu_handles = {}
            for i in range(self.device_count):
                try:
                    gpu_handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
                except pynvml.NVMLError as e:
                    logging.info(f"Failed to get handle for GPU {i}: {e}")
                    continue

            if not gpu_handles:
                logging.info("No valid GPU handles found, exiting monitor thread")
                return

            while not self.should_stop:
                for device_id, gpu_handle in gpu_handles.items():
                    try:
                        memory = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                        self.latest_stats[device_id] = GPUStats(
                            memory_used=memory.used // 1024 // 1024,
                            memory_total=memory.total // 1024 // 1024,
                            utilization=utilization.gpu,
                        )
                    except pynvml.NVMLError as e:
                        logging.info(f"Error getting stats for GPU {device_id}: {e}")

                time.sleep(self.polling_interval)

            try:
                pynvml.nvmlShutdown()
                logging.info("NVML shutdown successfully")
            except Exception as e:
                logging.info(f"Error shutting down pynvml: {e}")

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.should_stop = True
        if hasattr(self, "monitor_thread") and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

    def get_stats(self) -> Dict[int, GPUStats]:
        return self.latest_stats

    def is_initialized(self) -> bool:
        return self.initialization_successful


class GPUMonitorCallback(TrainerCallback):
    """Trainer callback that logs GPU stats from an async NVML poller."""

    def __init__(self, trainer, log_interval: int = 100, polling_interval: float = 0.1):
        super().__init__()
        self.trainer = trainer
        self.log_interval = log_interval
        self.monitor: Optional[AsyncGPUMonitor] = None
        self.polling_interval = polling_interval
        self.enabled = pynvml is not None and torch.cuda.is_available()
        if not self.enabled:
            logging.info("GPUMonitorCallback disabled: pynvml not available or CUDA not available")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled:
            return
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        try:
            self.monitor = AsyncGPUMonitor(device_count=device_count, polling_interval=self.polling_interval)
            time.sleep(0.5)
            if not self.monitor.is_initialized():
                logging.info("GPU monitoring initialization failed, disabling callback")
                self.enabled = False
        except Exception as e:
            logging.info(f"Failed to initialize GPU monitor: {e}")
            self.enabled = False

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled or not self.monitor or self.log_interval <= 0 or state.global_step % self.log_interval != 0:
            return

        stats_dict = self.monitor.get_stats()
        if not stats_dict:
            return

        metrics = {}
        for device_id, stats in stats_dict.items():
            prefix = f"gpu/device_{device_id}"
            metrics[f"{prefix}/memory_used_mib"] = stats.memory_used
            metrics[f"{prefix}/memory_utilization_pct"] = (stats.memory_used / max(stats.memory_total, 1)) * 100
            metrics[f"{prefix}/gpu_utilization_pct"] = stats.utilization

        try:
            self.trainer.log(metrics)
        except Exception as e:
            logging.info(f"Error logging GPU metrics: {e}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.monitor is not None:
            logging.info("Stopping GPU monitor thread")
            self.monitor.stop()


class TimingCallback(TrainerCallback):
    """Per-training-step timing (opt-in via AUTOINDEXER_PROFILE=1). CUDA-synced
    at each hook so the numbers reflect device time."""

    def _now(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._t_start = self._now()

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._t_pre_opt = self._now()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        end = self._now()
        fwd_bwd = (getattr(self, "_t_pre_opt", end) - self._t_start) * 1e3
        opt = (end - getattr(self, "_t_pre_opt", end)) * 1e3
        print(f"[PROFILE] step {state.global_step}: fwd+bwd={fwd_bwd:.1f}ms opt={opt:.1f}ms", flush=True)


def compute_token_metrics_factory(ignore_index: int = -100):
    """Build a `(preprocess_logits_for_metrics, compute_metrics)` pair that
    reproduce `TokenAccuracyCallback` + `PerplexityCallback` using the
    native `Trainer(compute_metrics=...)` hook instead of per-batch PL
    callbacks.
    """

    def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
        # Reduce full [batch, seq, vocab] logits down to per-token predicted ids + per-token NLL before accumulation
        if labels is None:
            # `labels` is only ever None here if `Trainer.label_names` doesn't include "labels",
            # which should already be fixed at the source; raise a clear, actionable error instead.
            raise ValueError(
                "preprocess_logits_for_metrics received labels=None. This means "
                "Trainer.label_names doesn't include 'labels' -- check that "
                "AutoIndexerTrainingArguments(label_names=[...]) is set (build_trainer() "
                "sets this by default; a custom Trainer subclass or explicit override may "
                "have cleared it)."
            )
        if isinstance(logits, tuple):
            logits = logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        shifted_labels = torch.full_like(labels, fill_value=ignore_index)
        shifted_labels[:, :-1] = labels[:, 1:]
        mask = shifted_labels != ignore_index
        safe_labels = shifted_labels.clamp(min=0)
        token_nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_nll = torch.where(mask, token_nll, torch.zeros_like(token_nll))
        predictions = logits.argmax(dim=-1).to(torch.int32)
        # Keep integer predictions separate from float auxiliaries: stacking
        # would force a float dtype and lose precision for large token ids.
        aux = torch.stack([token_nll, mask.to(torch.float32)], dim=-1)
        return predictions, aux

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, aux = eval_pred.predictions
        labels = eval_pred.label_ids
        token_nll = aux[..., 0]
        mask = aux[..., 1].astype(bool)

        shifted_labels = labels.copy()
        shifted_labels[:, :-1] = labels[:, 1:]
        shifted_labels[:, -1] = ignore_index

        total_tokens = mask.sum()
        if total_tokens == 0:
            return {}

        correct = ((predictions == shifted_labels) & mask).sum()
        avg_loss = token_nll[mask].sum() / total_tokens
        try:
            perplexity = float(math.exp(avg_loss))
        except OverflowError:
            print(f"Perplexity overflow with avg_loss of {avg_loss}. Setting perplexity to 1000.")
            perplexity = 1000

        return {
            "token_accuracy": float(100.0 * correct / total_tokens),
            "perplexity": perplexity,
        }

    return preprocess_logits_for_metrics, compute_metrics


class MonitoringManager:
    """
    Builds the set of `TrainerCallback`s (and, for token-level metrics, a
    `compute_metrics`/`preprocess_logits_for_metrics` pair) requested via
    `AutoIndexerTrainingArguments.monitoring_options`.
    """

    def __init__(self, trainer, log_interval: int = 1_000, verbose: bool = True, options: Optional[List[str]] = None):
        self.trainer = trainer
        self.log_interval = log_interval
        self.verbose = verbose
        self.options = options or []
        if self.verbose:
            logging.info(f"Initializing MonitoringManager with log_interval={log_interval}")

    def wants_token_metrics(self) -> bool:
        return "token_accuracy" in self.options or "perplexity" in self.options

    def get_callbacks(self) -> List[TrainerCallback]:
        """Callbacks for monitoring training performance (gradients,
        throughput, GPU usage). Token-accuracy/perplexity are handled
        separately via `compute_token_metrics_factory` since they're
        computed over the whole eval set through `compute_metrics`."""
        callbacks: List[TrainerCallback] = []

        if "gradients" in self.options:
            callbacks.append(GradientParameterMonitor(self.trainer, log_interval=self.log_interval))

        if "throughput" in self.options:
            callbacks.append(ThroughputCallback(self.trainer, log_interval=self.log_interval))

        if pynvml is not None and torch.cuda.is_available():
            callbacks.append(GPUMonitorCallback(self.trainer, log_interval=self.log_interval))
            if self.verbose:
                logging.info("Added GPUMonitorCallback (GPU monitoring available)")
        elif self.verbose:
            logging.info("GPUMonitorCallback not added (pynvml not available or CUDA not available)")

        if self.verbose:
            names = [cb.__class__.__name__ for cb in callbacks]
            logging.info(f"Monitoring callbacks: {', '.join(names)}")

        return callbacks
