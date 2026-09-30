# Description: Unit tests for autoindexer.domains.hf_datasets

import os
from unittest.mock import MagicMock

import datasets
import pytest
import torch

from autoindexer.domains import hf_datasets


class _FakeChatTokenizer:
    """Renders chat turns deterministically without a real tokenizer or network access."""

    pad_token = "<pad>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        rendered = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return rendered + "<assistant>" if add_generation_prompt else rendered

    def __call__(self, text, truncation=True, max_length=None, **kwargs):
        # One token per word, so tests can assert on lengths without caring about a real vocabulary
        return {"input_ids": list(range(len(text.split())))}


class _FakeBatchTokenizer(_FakeChatTokenizer):
    """`_FakeChatTokenizer`, plus the batch/padding call shape `Collator` needs."""

    def __call__(self, text, truncation=True, max_length=None, padding=None, return_tensors=None, **kwargs):
        texts = text if isinstance(text, list) else [text]
        token_ids = [list(range(1, len(t.split()) + 1))[:max_length] for t in texts]

        if padding != "max_length":
            return {"input_ids": token_ids if isinstance(text, list) else token_ids[0]}

        input_ids = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in token_ids]
        attention_mask = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in token_ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_load_source_stream_streams_from_hub_by_default(monkeypatch):
    load_dataset = MagicMock(return_value="hub_stream")
    monkeypatch.setattr(hf_datasets.datasets, "load_dataset", load_dataset)

    source = {"dataset_name": "bigcode/starcoderdata", "data_dir": "python", "local_dir": "starcoderdata/python"}
    result = hf_datasets._load_source_stream(source, local_data_dir=None)

    assert result == "hub_stream"
    load_dataset.assert_called_once_with("bigcode/starcoderdata", None, data_dir="python", split="train", streaming=True)


def test_load_source_stream_ignores_local_data_dir_without_a_local_dir(monkeypatch):
    load_dataset = MagicMock(return_value="hub_stream")
    monkeypatch.setattr(hf_datasets.datasets, "load_dataset", load_dataset)

    source = {"dataset_name": "mlfoundations/dclm-baseline-1.0"}
    result = hf_datasets._load_source_stream(source, local_data_dir="/tmp/ray_data")

    assert result == "hub_stream"
    load_dataset.assert_called_once_with(
        "mlfoundations/dclm-baseline-1.0", None, data_dir=None, split="train", streaming=True
    )


def test_load_source_stream_reads_local_mirror_when_configured(monkeypatch, tmp_path):
    load_dataset = MagicMock(return_value="local_stream")
    monkeypatch.setattr(hf_datasets.datasets, "load_dataset", load_dataset)

    # The mirror branch is gated on the shard glob actually matching, so the shard has to exist
    # on disk -- pointing at a path with no shards exercises the Hub-fallback branch instead.
    shard_dir = tmp_path / "starcoderdata" / "python"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shard_00000.parquet").touch()

    source = {"dataset_name": "bigcode/starcoderdata", "data_dir": "python", "local_dir": "starcoderdata/python"}
    result = hf_datasets._load_source_stream(source, local_data_dir=str(tmp_path))

    assert result == "local_stream"
    load_dataset.assert_called_once_with(
        "parquet",
        data_files=os.path.join(str(tmp_path), "starcoderdata/python", "**", "*.parquet"),
        split="train",
        streaming=True,
    )


def test_load_dataset_mix_threads_local_data_dir_through_each_source(monkeypatch):
    calls = []

    def fake_load_source_stream(source, local_data_dir):
        calls.append((source["name"], local_data_dir))
        rows = [{"text": f"{source['name']}-{i}"} for i in range(5)]
        return datasets.Dataset.from_list(rows).to_iterable_dataset()

    monkeypatch.setattr(hf_datasets, "_load_source_stream", fake_load_source_stream)

    sources = [
        {"name": "mirrored", "dataset_name": "org/mirrored", "local_dir": "mirrored"},
        {"name": "hub-only", "dataset_name": "org/hub-only"},
    ]
    result = hf_datasets.load_dataset_mix(sources, num_val_samples_per_source=1, local_data_dir="/tmp/ray_data")

    # Every source is handed the same `local_data_dir`
    assert calls == [("mirrored", "/tmp/ray_data"), ("hub-only", "/tmp/ray_data")]
    assert set(result.keys()) == {"train", "validation"}


def test_alpaca_prompt_and_response_appends_input_only_when_present():
    with_input = {"instruction": "Translate", "input": "Bonjour", "output": "Hello"}
    assert hf_datasets._alpaca_prompt_and_response(with_input) == ("Translate\n\nBonjour", "Hello")

    without_input = {"instruction": "Say hi", "input": "", "output": "Hi there"}
    assert hf_datasets._alpaca_prompt_and_response(without_input) == ("Say hi", "Hi there")


def test_alpaca_prompt_and_response_honors_custom_column_names():
    # e.g. Glaive-code-assistant's question/answer, or Magicoder's problem/solution.
    example = {"question": "2+2?", "answer": "4"}
    prompt, response = hf_datasets._alpaca_prompt_and_response(
        example, instruction_column="question", input_column=None, output_column="answer"
    )
    assert (prompt, response) == ("2+2?", "4")


def test_messages_prompt_and_response_strips_and_remaps_final_turn():
    # ShareGPT-shaped: "from"/"value" keys, "human"/"gpt" roles.
    example = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}
    prompt_messages, response = hf_datasets._messages_prompt_and_response(
        example,
        messages_column="conversations",
        role_key="from",
        content_key="value",
        role_map={"human": "user", "gpt": "assistant"},
    )
    assert prompt_messages == [{"role": "user", "content": "hi"}]
    assert response == "hello"


def test_messages_prompt_and_response_rejects_non_assistant_final_turn():
    example = {"messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(AssertionError):
        hf_datasets._messages_prompt_and_response(example)


def test_render_instruction_example_alpaca_text_starts_with_prompt():
    example = {"instruction": "Say hi", "input": "", "output": "hi there"}
    rendered = hf_datasets._render_instruction_example(example, {"format": "alpaca"}, _FakeChatTokenizer())

    assert rendered["prompt"] == "<user>Say hi</user><assistant>"
    assert rendered["text"] == "<user>Say hi</user><assistant>hi there</assistant>"
    assert rendered["text"].startswith(rendered["prompt"])


def test_render_instruction_example_messages_text_starts_with_prompt():
    example = {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]}
    rendered = hf_datasets._render_instruction_example(example, {"format": "messages"}, _FakeChatTokenizer())

    assert rendered["text"].startswith(rendered["prompt"])
    assert rendered["text"] == "<user>2+2?</user><assistant>4</assistant>"


def test_render_instruction_example_rejects_unknown_format():
    with pytest.raises(ValueError, match="format"):
        hf_datasets._render_instruction_example({}, {"format": "sql"}, _FakeChatTokenizer())


def test_render_instruction_example_defaults_to_no_thinking_and_threads_it_through():
    # Must match eval time's `enable_thinking=False` (see `_render_instruction_example`'s docstring)
    tokenizer = MagicMock(wraps=_FakeChatTokenizer())
    example = {"instruction": "Say hi", "input": "", "output": "hi there"}

    hf_datasets._render_instruction_example(example, {"format": "alpaca"}, tokenizer)
    hf_datasets._render_instruction_example(example, {"format": "alpaca"}, tokenizer, enable_thinking=True)

    calls = tokenizer.apply_chat_template.call_args_list
    assert [call.kwargs["enable_thinking"] for call in calls] == [False, False, True, True]


def test_load_instruction_dataset_mix_renders_and_tags_each_source(monkeypatch):
    monkeypatch.setattr(hf_datasets, "_build_tokenizer", lambda tokenizer_name, pad_token_id=None: _FakeChatTokenizer())

    def fake_load_source_stream(source, local_data_dir):
        rows = [{"instruction": f"{source['name']}-instruction", "input": "", "output": "response"}]
        return datasets.Dataset.from_list(rows).to_iterable_dataset()

    monkeypatch.setattr(hf_datasets, "_load_source_stream", fake_load_source_stream)

    sources = [
        {"name": "alpaca", "dataset_name": "org/alpaca", "format": "alpaca"},
        {"name": "codealpaca", "dataset_name": "org/codealpaca", "format": "alpaca"},
    ]
    result = hf_datasets.load_instruction_dataset_mix(sources, tokenizer_name="fake", num_val_samples_per_source=1)

    assert set(result.keys()) == {"train", "validation"}
    for row in result["validation"]:
        assert row["text"].startswith(row["prompt"])
        assert row["source"] in {"alpaca", "codealpaca"}


def test_load_instruction_dataset_mix_threads_enable_thinking_through(monkeypatch):
    tokenizer = MagicMock(wraps=_FakeChatTokenizer())
    monkeypatch.setattr(hf_datasets, "_build_tokenizer", lambda tokenizer_name, pad_token_id=None: tokenizer)

    def fake_load_source_stream(source, local_data_dir):
        rows = [{"instruction": "hi", "input": "", "output": "hi there"}]
        return datasets.Dataset.from_list(rows).to_iterable_dataset()

    monkeypatch.setattr(hf_datasets, "_load_source_stream", fake_load_source_stream)

    sources = [{"name": "alpaca", "dataset_name": "org/alpaca", "format": "alpaca"}]
    result = hf_datasets.load_instruction_dataset_mix(
        sources, tokenizer_name="fake", num_val_samples_per_source=1, enable_thinking=True
    )
    list(result["train"])  # force the lazy stream to actually render a row

    calls = tokenizer.apply_chat_template.call_args_list
    assert calls and all(call.kwargs["enable_thinking"] is True for call in calls)


def test_sft_collator_masks_prompt_and_padding_out_of_the_loss():
    collator = hf_datasets.SFTCollator.__new__(hf_datasets.SFTCollator)
    collator.tokenizer = _FakeBatchTokenizer()
    collator.max_length = 8
    collator.text_column = "text"
    collator.is_pretokenized = False
    collator.prompt_column = "prompt"

    # 3-word prompt + 1-word response, padded out to max_length=8.
    examples = [{"prompt": "say hi to", "text": "say hi to you"}]
    batch = collator(examples)

    assert batch["labels"][0].tolist() == [-100, -100, -100, 4, -100, -100, -100, -100]


def test_autoindexer_sft_collator_leaves_labels_untouched_and_reports_editable_start():
    collator = hf_datasets.AutoIndexerSFTCollator.__new__(hf_datasets.AutoIndexerSFTCollator)
    collator.tokenizer = _FakeBatchTokenizer()
    collator.max_length = 8
    collator.text_column = "text"
    collator.is_pretokenized = False
    collator.prompt_column = "prompt"

    examples = [{"prompt": "say hi to", "text": "say hi to you"}]
    batch = collator(examples)

    # Unlike SFTCollator, labels carry the real tokens throughout
    assert batch["labels"][0].tolist() == batch["input_ids"][0].tolist()
    assert batch["editable_start"].tolist() == [3]
