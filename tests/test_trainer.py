# Regression tests for Trainer eval metrics (`label_names`, `labels=None` guards).

import os

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

from autoindexer.utils.trainer import build_trainer
from autoindexer.utils.training_monitors import compute_token_metrics_factory


def test_preprocess_logits_for_metrics_raises_clear_error_when_labels_none():
    """Preprocess logits for metrics raises clear error when labels none"""
    preprocess_logits_for_metrics, _ = compute_token_metrics_factory()
    logits = torch.randn(2, 4, 10)

    with pytest.raises(ValueError, match="label_names"):
        preprocess_logits_for_metrics(logits, None)


def test_preprocess_logits_for_metrics_computes_predictions_and_aux():
    """With real labels (the normal path -- this is what every batch this
    repo's Collator produces looks like), preprocessing should succeed and
    return shapes/dtypes compute_metrics expects."""
    preprocess_logits_for_metrics, _ = compute_token_metrics_factory()
    batch, seq_len, vocab = 2, 4, 10
    logits = torch.randn(batch, seq_len, vocab)
    labels = torch.randint(0, vocab, (batch, seq_len))

    predictions, aux = preprocess_logits_for_metrics(logits, labels)

    assert predictions.shape == (batch, seq_len)
    assert predictions.dtype == torch.int32
    assert aux.shape == (batch, seq_len, 2)  # [token_nll, mask] stacked


def test_preprocess_logits_for_metrics_handles_tuple_logits():
    """Some model forward passes return `(logits, ...)` tuples; the first
    element must be unwrapped the same way whether or not labels are present."""
    preprocess_logits_for_metrics, _ = compute_token_metrics_factory()
    logits = torch.randn(1, 3, 5)
    labels = torch.randint(0, 5, (1, 3))

    predictions, aux = preprocess_logits_for_metrics((logits, "unused_second_element"), labels)

    assert predictions.shape == (1, 3)
    assert aux.shape == (1, 3, 2)


def test_compute_metrics_token_accuracy_and_perplexity_end_to_end():
    """End-to-end (still no GPU/model needed): feed compute_metrics exactly"""
    preprocess_logits_for_metrics, compute_metrics = compute_token_metrics_factory()
    batch, seq_len, vocab = 1, 4, 5
    labels = torch.tensor([[1, 2, 3, 4]])
    # Construct logits whose argmax exactly reproduces the shifted labels ([2, 3, 4, ignored]) at every valid position
    shifted = torch.tensor([2, 3, 4, 0])
    logits = torch.zeros(batch, seq_len, vocab)
    for t, tok in enumerate(shifted[:-1]):
        logits[0, t, tok] = 2.0
        logits[0, t] += torch.randn(vocab) * 0.01  # tiny noise so NLL != 0

    predictions, aux = preprocess_logits_for_metrics(logits, labels)
    eval_pred = EvalPrediction(
        predictions=(predictions.numpy(), aux.numpy()),
        label_ids=labels.numpy(),
    )
    metrics = compute_metrics(eval_pred)

    assert metrics["token_accuracy"] == pytest.approx(100.0)
    assert metrics["perplexity"] > 1.0
    assert np.isfinite(metrics["perplexity"])


def _stub_trainer_module(monkeypatch, trainer_module):
    class _StubTrainer:
        def __init__(self, *args, **kwargs):
            self.captured_kwargs = kwargs

        def add_callback(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(trainer_module, "Trainer", _StubTrainer)
    monkeypatch.setattr(trainer_module, "configure_autoindexer_trainer", lambda trainer, args, **kwargs: trainer)
    return _StubTrainer


def test_build_trainer_defaults_label_names_to_labels_list(monkeypatch):
    """The actual production fix: `build_trainer` must set"""
    import autoindexer.utils.trainer as trainer_module
    from omegaconf import OmegaConf

    captured_kwargs = {}

    class _StubTrainer:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            self.model = kwargs.get("model")
            self.args = kwargs.get("args")

        def add_callback(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(trainer_module, "Trainer", _StubTrainer)
    monkeypatch.setattr(trainer_module, "configure_autoindexer_trainer", lambda trainer, args, **kwargs: trainer)

    hydra_config = OmegaConf.create(
        {
            "mode": "train",
            "save_dir": "/tmp/test_build_trainer_label_names",
            "project_name": "test",
            "experiment_name": "test",
            "data": {"train": []},
            "trainer": {
                "per_device_train_batch_size": 1,
                "num_train_epochs": 1,
                "monitoring_options": [],
            },
            "model": None,
        }
    )

    build_trainer(hydra_config)

    assert captured_kwargs["args"].label_names == ["labels"]


def test_build_trainer_sets_comet_workspace_from_config(monkeypatch):
    """Worker-side Comet reads `COMET_WORKSPACE` from the environment; `build_trainer`"""
    import autoindexer.utils.trainer as trainer_module
    from omegaconf import OmegaConf

    monkeypatch.delenv("COMET_WORKSPACE", raising=False)
    monkeypatch.delenv("COMET_PROJECT_NAME", raising=False)

    _stub_trainer_module(monkeypatch, trainer_module)

    hydra_config = OmegaConf.create(
        {
            "mode": "train",
            "save_dir": "/tmp/test_build_trainer_comet_workspace",
            "project_name": "cpt_datamix-train",
            "experiment_name": "test",
            "data": {"train": []},
            "trainer": {
                "per_device_train_batch_size": 1,
                "num_train_epochs": 1,
                "monitoring_options": [],
            },
            "model": None,
            "comet_workspace": "autoindexer",
        }
    )

    build_trainer(hydra_config)

    assert os.environ["COMET_WORKSPACE"] == "autoindexer"
    assert os.environ["COMET_PROJECT_NAME"] == "cpt_datamix-train"


def test_build_trainer_does_not_set_comet_workspace_when_absent(monkeypatch):
    """If a config has no `comet_workspace`, COMET_WORKSPACE should stay unset."""
    import autoindexer.utils.trainer as trainer_module
    from omegaconf import OmegaConf

    monkeypatch.delenv("COMET_WORKSPACE", raising=False)

    _stub_trainer_module(monkeypatch, trainer_module)

    hydra_config = OmegaConf.create(
        {
            "mode": "train",
            "save_dir": "/tmp/test_build_trainer_no_comet_workspace",
            "project_name": "test",
            "experiment_name": "test",
            "data": {"train": []},
            "trainer": {
                "per_device_train_batch_size": 1,
                "num_train_epochs": 1,
                "monitoring_options": [],
            },
            "model": None,
        }
    )

    build_trainer(hydra_config)

    assert "COMET_WORKSPACE" not in os.environ


class _RaisingOnCompile:
    """Stands in for the `torch` module so any `torch.compile` call fails loudly."""

    def compile(self, *_args, **_kwargs):
        raise AssertionError("build_trainer must not wrap the model with torch.compile itself")

    def __getattr__(self, name):
        return getattr(torch, name)


def test_build_trainer_compiles_via_training_args_not_by_wrapping_the_model(monkeypatch):
    """Regression test for a 31 GB-per-checkpoint bug: `build_trainer` used to call"""
    import autoindexer.utils.trainer as trainer_module
    from omegaconf import OmegaConf

    monkeypatch.setenv("AUTOINDEXER_TORCH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setattr(trainer_module, "torch", _RaisingOnCompile())

    captured_kwargs = {}

    class _StubTrainer:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

        def add_callback(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(trainer_module, "Trainer", _StubTrainer)
    monkeypatch.setattr(trainer_module, "configure_autoindexer_trainer", lambda trainer, args, **kwargs: trainer)

    hydra_config = OmegaConf.create(
        {
            "mode": "train",
            "save_dir": "/tmp/test_build_trainer_torch_compile",
            "project_name": "test",
            "experiment_name": "test",
            "data": {"train": []},
            "trainer": {
                "per_device_train_batch_size": 1,
                "num_train_epochs": 1,
                "monitoring_options": [],
            },
            "model": None,
        }
    )

    build_trainer(hydra_config)

    assert captured_kwargs["args"].torch_compile is True
    assert captured_kwargs["args"].torch_compile_mode == "reduce-overhead"


def test_verify_s3_upload_passes_when_all_local_files_present_remotely(tmp_path):
    """Post-sync listing catches files that never landed remotely."""
    from autoindexer.utils.aws import verify_s3_upload
    import fsspec

    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "weights.bin").write_bytes(b"weights")

    s3_uri = "memory://verify-upload-ok/final_checkpoint"
    fs, root = fsspec.core.url_to_fs(s3_uri)
    fs.mkdirs(root, exist_ok=True)
    fs.pipe_file(root + "/config.json", b"{}")
    fs.pipe_file(root + "/sub/weights.bin", b"weights")

    verify_s3_upload(str(tmp_path), s3_uri)  # must not raise


def test_verify_s3_upload_raises_when_a_file_is_missing_remotely(tmp_path):
    """A sync that drops a file must surface as a loud failure."""
    from autoindexer.utils.aws import verify_s3_upload
    import fsspec

    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "weights.bin").write_bytes(b"weights")

    s3_uri = "memory://verify-upload-missing/final_checkpoint"
    fs, root = fsspec.core.url_to_fs(s3_uri)
    fs.mkdirs(root, exist_ok=True)
    fs.pipe_file(root + "/config.json", b"{}")  # weights.bin deliberately not uploaded

    with pytest.raises(RuntimeError, match="weights.bin"):
        verify_s3_upload(str(tmp_path), s3_uri)


def test_configure_autoindexer_trainer_registers_final_checkpoint_callback_for_non_lora():
    from autoindexer.utils.args import AutoIndexerTrainingArguments
    from autoindexer.utils.trainer import _SaveFinalCheckpointOnTrainEnd, configure_autoindexer_trainer

    class _StubTrainer:
        def __init__(self):
            self.model = torch.nn.Module()
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def create_optimizer(self):
            raise NotImplementedError("not exercised: configure_autoindexer_trainer only wraps this")

        def log(self, logs, *args, **kwargs):
            raise NotImplementedError("not exercised: configure_autoindexer_trainer only wraps this")

    trainer = _StubTrainer()
    args = AutoIndexerTrainingArguments(output_dir="/tmp/test-final-checkpoint", monitoring_options=[])

    configure_autoindexer_trainer(trainer, args)

    # `monitoring_options=[]` only suppresses the opt-in gradient/throughput callbacks --
    # GPUMonitorCallback still gets added whenever CUDA is available, so assert on the
    # final-checkpoint callback's presence rather than the exact callback count.
    final_checkpoint_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, _SaveFinalCheckpointOnTrainEnd)]
    assert len(final_checkpoint_callbacks) == 1


def test_save_final_checkpoint_on_train_end_writes_model_and_tokenizer(tmp_path):
    from autoindexer.utils.trainer import _SaveFinalCheckpointOnTrainEnd
    from transformers import TrainerState, TrainingArguments

    saved_dirs = []

    class _StubTrainer:
        processing_class = type(
            "Tokenizer",
            (),
            {"save_pretrained": staticmethod(lambda output_dir: saved_dirs.append(("tokenizer", output_dir)))},
        )()

        def save_model(self, output_dir):
            saved_dirs.append(("model", output_dir))
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                f.write("{}")

    callback = _SaveFinalCheckpointOnTrainEnd(_StubTrainer())
    args = TrainingArguments(output_dir=str(tmp_path))
    state = TrainerState(is_world_process_zero=True)

    callback.on_train_end(args, state, control=None)

    final_dir = str(tmp_path / "final_checkpoint")
    assert ("model", final_dir) in saved_dirs
    assert ("tokenizer", final_dir) in saved_dirs
    assert os.path.isfile(os.path.join(final_dir, "config.json"))
