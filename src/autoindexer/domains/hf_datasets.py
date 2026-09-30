import glob
import logging
import os
from typing import Optional, Dict, List, Union

import datasets
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)
import torch

from autoindexer.models.wrappers.hf_module import _resolve_pretrained_checkpoint

_logger = logging.getLogger(__name__)


def _build_tokenizer(tokenizer_name: str, pad_token_id: Optional[int] = None) -> PreTrainedTokenizer:
    """Loads `tokenizer_name`, filling in `pad_token_id` if it ships no pad token."""
    tokenizer = AutoTokenizer.from_pretrained(_resolve_pretrained_checkpoint(tokenizer_name), trust_remote_code=True)
    if not tokenizer.pad_token:
        assert pad_token_id is not None, "tokenizer has no pad token; pad_token_id must be provided"
        tokenizer.pad_token_id = pad_token_id
    return tokenizer


class Collator:
    """`transformers.Trainer`-compatible collator that tokenizes (or pads already-tokenized) text examples."""

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        text_column: str = "text",
        pad_token_id: Optional[int] = None,
        is_pretokenized: bool = False,
        input_ids_column: str = "input_ids",
        attention_mask_column: str = "attention_mask",
    ):
        self.tokenizer: PreTrainedTokenizer = _build_tokenizer(tokenizer_name, pad_token_id)
        self.max_length = max_length
        self.text_column = text_column
        self.is_pretokenized = is_pretokenized
        self.input_ids_column = input_ids_column
        self.attention_mask_column = attention_mask_column

    def pad_sequence(self, sequences: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        # Find max length in the batch
        max_len = min(max(len(sequence) for sequence in sequences), self.max_length)

        # Create output tensor
        out_dims = (len(sequences), max_len)
        out_tensor = sequences[0].new_full(out_dims, padding_value)

        # Copy data
        for i, sequence in enumerate(sequences):
            length = min(len(sequence), max_len)
            out_tensor[i, :length] = sequence[:length]

        return out_tensor

    def process_pretokenized(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Convert input_ids to tensors if they're lists
        input_ids = [
            (torch.tensor(ex[self.input_ids_column]) if isinstance(ex[self.input_ids_column], list) else ex[self.input_ids_column])
            for ex in examples
        ]

        # Pad sequences
        input_ids = self.pad_sequence(input_ids, self.tokenizer.pad_token_id)

        # Create attention mask if not provided
        if self.attention_mask_column in examples[0]:
            attention_mask = [
                (
                    torch.tensor(ex[self.attention_mask_column])
                    if isinstance(ex[self.attention_mask_column], list)
                    else ex[self.attention_mask_column]
                )
                for ex in examples
            ]
            attention_mask = self.pad_sequence(attention_mask, 0)
        else:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # For language modeling, labels are the same as inputs
        labels = batch["input_ids"].clone()
        # -100 is the default ignore_index for Pytorch's cross entropy loss
        labels[~attention_mask.bool()] = -100
        batch["labels"] = labels

        return batch

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        if self.is_pretokenized:
            return self.process_pretokenized(examples)

        # Extract text from the batch
        texts = [example[self.text_column] for example in examples]

        # Tokenize
        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # For language modeling, labels are the same as inputs
        batch["labels"] = batch["input_ids"].clone()

        return batch


class SFTCollator(Collator):
    """Collator that masks `labels` to the response tokens only (via `prompt_column`)."""

    def __init__(self, *args, prompt_column: str = "prompt", **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_column = prompt_column

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        labels = batch["labels"]
        for row, prompt_length in enumerate(_prompt_token_lengths(self.tokenizer, examples, self.prompt_column, self.max_length)):
            labels[row, :prompt_length] = -100
        # The base `Collator` never masks padding out of `labels`
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class AutoIndexerSFTCollator(Collator):
    """Like `Collator`, but adds per-row `editable_start` instead of masking `labels`."""

    def __init__(self, *args, prompt_column: str = "prompt", **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_column = prompt_column

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        batch["editable_start"] = torch.tensor(_prompt_token_lengths(self.tokenizer, examples, self.prompt_column, self.max_length))
        return batch


def _prompt_token_lengths(tokenizer: PreTrainedTokenizer, examples: List[Dict[str, str]], prompt_column: str, max_length: int) -> List[int]:
    """Per-example token count of `example[prompt_column]` -- the prompt/response boundary the SFT collators mask on."""
    return [
        len(tokenizer(example[prompt_column], truncation=True, max_length=max_length)["input_ids"]) for example in examples
    ]


def _alpaca_prompt_and_response(
    example: Dict,
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
) -> tuple[str, str]:
    """Builds a prompt (instruction, plus input if present) and a response from Alpaca-shaped columns."""
    instruction = example[instruction_column]
    extra_input = example.get(input_column) if input_column else None
    prompt = f"{instruction}\n\n{extra_input}" if extra_input else instruction
    return prompt, example[output_column]


def _messages_prompt_and_response(
    example: Dict,
    messages_column: str = "messages",
    role_key: str = "role",
    content_key: str = "content",
    role_map: Optional[Dict[str, str]] = None,
) -> tuple[List[Dict], str]:
    """Normalizes a chat-turn list into `role`/`content` dicts (UltraChat/OpenHermes/ShareGPT-style)."""
    role_map = role_map or {}
    turns = example[messages_column]
    messages = [{"role": role_map.get(turn[role_key], turn[role_key]), "content": turn[content_key]} for turn in turns]
    assert messages and messages[-1]["role"] == "assistant", "last chat turn must be the assistant's response"
    return messages[:-1], messages[-1]["content"]


def _fewshot_pack_prompt_and_response(
    batch: Dict[str, List],
    instruction_column: str,
    output_column: str,
    template: str = "Problem:\n{q}\n\nSolution:\n{a}",
) -> tuple[str, str]:
    """Pack prior rows as few-shot exemplars before the last row's question."""
    questions, answers = batch[instruction_column], batch[output_column]
    exemplars = "\n\n".join(template.format(q=q, a=a) for q, a in zip(questions[:-1], answers[:-1]))
    prompt_text = f"{exemplars}\n\nProblem:\n{questions[-1]}\n\nSolution:\n"
    return prompt_text, answers[-1]


def _render_fewshot_pack_batch(
    batch: Dict[str, List], source: Dict, tokenizer: PreTrainedTokenizer, enable_thinking: bool, source_name: str
) -> Dict[str, List[str]]:
    """Batched (many-rows-in, one-row-out) counterpart of `_render_instruction_example` for `format: "fewshot_pack"`."""
    prompt_text, response = _fewshot_pack_prompt_and_response(
        batch,
        instruction_column=source.get("instruction_column", "instruction"),
        output_column=source.get("output_column", "output"),
    )
    prompt_messages = [{"role": "user", "content": prompt_text}]
    full_messages = prompt_messages + [{"role": "assistant", "content": response}]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking
    )
    return {"text": [text], "prompt": [prompt], "source": [source_name]}


def _render_instruction_example(
    example: Dict, source: Dict, tokenizer: PreTrainedTokenizer, enable_thinking: bool = False
) -> Dict[str, str]:
    """Renders one row of an instruction-format source into a full `text` and a `prompt`-only prefix."""
    fmt = source.get("format", "alpaca")
    if fmt == "replay":
        # Raw pretraining-style continuation (no instruction shape, no chat template): an empty `prompt` masks nothing
        return {"prompt": "", "text": example[source.get("text_column", "text")]}
    if fmt == "alpaca":
        prompt_text, response = _alpaca_prompt_and_response(
            example,
            instruction_column=source.get("instruction_column", "instruction"),
            input_column=source.get("input_column", "input"),
            output_column=source.get("output_column", "output"),
        )
        prompt_messages = [{"role": "user", "content": prompt_text}]
    elif fmt == "messages":
        prompt_messages, response = _messages_prompt_and_response(
            example,
            messages_column=source.get("messages_column", "messages"),
            role_key=source.get("role_key", "role"),
            content_key=source.get("content_key", "content"),
            role_map=source.get("role_map"),
        )
    else:
        raise ValueError(f"Unknown instruction source format {fmt!r}")

    full_messages = prompt_messages + [{"role": "assistant", "content": response}]
    # `add_generation_prompt=True` renders exactly the prefix that precedes the assistant's content in `full_messages`'s own
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking
    )
    return {"prompt": prompt, "text": text}


def load_hf_dataset(
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "validation",
    val_split_size: float = 0.1,  # only used if val_split is not in the dataset
) -> datasets.DatasetDict:
    """Loads `dataset_name` and returns its `datasets.DatasetDict` as-is"""
    dataset = datasets.load_dataset(dataset_name, dataset_config)

    if val_split not in dataset and train_split in dataset:
        split_dataset = dataset[train_split].train_test_split(
            test_size=val_split_size,
            shuffle=True,
            seed=42,  # For reproducibility
        )
        dataset = datasets.DatasetDict(
            {
                **{split: data for split, data in dataset.items() if split != train_split},
                train_split: split_dataset["train"],
                val_split: split_dataset["test"],
            }
        )

    return dataset


def _load_source_stream(source: Dict, local_data_dir: Optional[str]) -> "datasets.IterableDataset":
    """Loads a single mix source as a streaming `IterableDataset`."""
    local_dir = source.get("local_dir")
    if local_data_dir and local_dir:
        # Fall back to Hub streaming if the local mirror has no shards yet, rather than
        # letting `datasets.load_dataset("parquet", ...)` raise on an empty glob.
        glob_pattern = os.path.join(local_data_dir, local_dir, "**", "*.parquet")
        if glob.glob(glob_pattern, recursive=True):
            return datasets.load_dataset(
                "parquet",
                data_files=glob_pattern,
                split=source.get("split", "train"),
                streaming=True,
            )
        _logger.warning(
            "No mirrored parquet shards found under %s for source %r -- falling back to streaming "
            "%r from the HF Hub. Sync a mirror with scripts/mirror_hf_dataset_to_s3.py and set "
            "`data.local_data_dir` if you meant to read from local disk.",
            glob_pattern,
            source.get("name", source["dataset_name"]),
            source["dataset_name"],
        )
    return datasets.load_dataset(
        source["dataset_name"],
        source.get("dataset_config"),
        data_dir=source.get("data_dir"),
        split=source.get("split", "train"),
        streaming=True,
    )


def load_dataset_mix(
    sources: List[Dict],
    num_val_samples_per_source: int = 200,
    seed: int = 42,
    stopping_strategy: str = "all_exhausted",
    local_data_dir: Optional[str] = None,
) -> Dict[str, "datasets.Dataset"]:
    """Streams and interleaves several HuggingFace datasets into a single pretraining mix."""
    weights = [float(source.get("weight", 1.0)) for source in sources]
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    schema = datasets.Features({"text": datasets.Value("string"), "source": datasets.Value("string")})

    train_streams = []
    val_rows = []
    for source in sources:
        stream = _load_source_stream(source, local_data_dir)

        text_column = source.get("text_column", "text")
        if text_column != "text":
            stream = stream.rename_column(text_column, "text")

        # `remove_columns` has to be computed from the pre-tag schema: once tagged below,
        # `stream.column_names` can no longer infer it (the schema becomes unknown to `datasets`).
        source_name = source.get("name", source["dataset_name"])
        extra_columns = [column for column in stream.column_names if column != "text"]
        stream = stream.map(
            # `source_name=source_name` binds the *current* loop iteration's value as a default
            # arg -- without it every closure would share the same (final) `source_name`, since
            # `map` on an `IterableDataset` evaluates lazily, at iteration time.
            lambda example, source_name=source_name: {"text": example["text"], "source": source_name},
            remove_columns=extra_columns,
            features=schema,
        )

        val_rows.extend(stream.take(num_val_samples_per_source))
        train_streams.append(stream.skip(num_val_samples_per_source))

    train_dataset = datasets.interleave_datasets(
        train_streams,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    val_dataset = datasets.Dataset.from_list(val_rows, features=schema).shuffle(seed=seed)

    # Deliberately `datasets.DatasetDict` rather than a plain `{}` -- Hydra's `instantiate()`
    # would otherwise recursively convert a plain dict into a `DictConfig`, breaking downstream
    # `isinstance(dataset, dict)` checks.
    return datasets.DatasetDict({"train": train_dataset, "validation": val_dataset})


def load_instruction_dataset_mix(
    sources: List[Dict],
    tokenizer_name: str,
    num_val_samples_per_source: int = 200,
    seed: int = 42,
    stopping_strategy: str = "all_exhausted",
    local_data_dir: Optional[str] = None,
    enable_thinking: bool = False,
) -> Dict[str, "datasets.Dataset"]:
    """Streams and interleaves several instruction/chat-formatted HF datasets into a single SFT mix."""
    tokenizer = _build_tokenizer(tokenizer_name)
    weights = [float(source.get("weight", 1.0)) for source in sources]
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    schema = datasets.Features(
        {
            "text": datasets.Value("string"),
            "prompt": datasets.Value("string"),
            "source": datasets.Value("string"),
        }
    )

    train_streams = []
    val_rows = []
    for source in sources:
        stream = _load_source_stream(source, local_data_dir)
        source_name = source.get("name", source["dataset_name"])
        original_columns = stream.column_names  # unavailable once `.map()` below is applied, see load_dataset_mix
        if source.get("format") == "fewshot_pack":
            # Pack `num_shots + 1` consecutive rows into one few-shot example.
            stream = stream.map(
                lambda batch, source=source, source_name=source_name: _render_fewshot_pack_batch(
                    batch, source, tokenizer, enable_thinking, source_name
                ),
                batched=True,
                batch_size=source.get("num_shots", 4) + 1,
                drop_last_batch=True,
                remove_columns=original_columns,
                features=schema,
            )
        else:
            stream = stream.map(
                # `source=source, source_name=source_name` bind the *current* loop iteration's values
                lambda example, source=source, source_name=source_name: {
                    **_render_instruction_example(example, source, tokenizer, enable_thinking),
                    "source": source_name,
                },
                remove_columns=original_columns,
                features=schema,
            )

        val_rows.extend(stream.take(num_val_samples_per_source))
        train_streams.append(stream.skip(num_val_samples_per_source))

    train_dataset = datasets.interleave_datasets(
        train_streams,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    val_dataset = datasets.Dataset.from_list(val_rows, features=schema).shuffle(seed=seed)

    return datasets.DatasetDict({"train": train_dataset, "validation": val_dataset})
