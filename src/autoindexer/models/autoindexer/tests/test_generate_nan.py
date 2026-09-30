"""Regression test for a `_sample` bug that crashed generation with a CUDA NaN-probability error."""

import torch
from transformers import GenerationConfig

from autoindexer.models.autoindexer.autoindexer import AutoIndexerLlamaModel
from autoindexer.models.autoindexer.config import AutoIndexerLlamaConfig


def test_generate_from_single_token_prompt_does_not_nan():
    torch.manual_seed(0)
    config = AutoIndexerLlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8,
        vocab_size=50, max_position_embeddings=1024, max_seq_length=800,
        attn_implementation="autoindexer_eager",
        eos_token_id=49, bos_token_id=None, pad_token_id=49,
    )
    model = AutoIndexerLlamaModel(config).eval()

    # A 1-token prompt locks every row out of the index head from the very first decode step
    # (`editable_start == cur_len`), producing the all-`-inf` row that used to crash `_sample`.
    input_ids = torch.tensor([[3]])
    generation_config = GenerationConfig(
        do_sample=True, max_length=8, pad_token_id=49, eos_token_id=49,
        temperature=0.7, top_k=50, top_p=0.8,
    )

    # Must not raise (previously: `ValueError`/CUDA assert from a NaN `Categorical` sample).
    model.generate(input_ids, generation_config=generation_config)
