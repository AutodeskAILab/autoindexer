"""`editable_start` locks the index head's cursor out of everything before a given token count."""

import random

import numpy as np
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from autoindexer.models.autoindexer.attn_layers import AutoIndexerIndexHead, IndexKeysCache
from autoindexer.models.autoindexer.autoindexer import AutoIndexerLlamaModel
from autoindexer.models.autoindexer.config import AutoIndexerConfig, AutoIndexerLlamaConfig
from autoindexer.models.autoindexer.perturb_labels import batch_perturb_labels, get_parsed_positions_from_perturbations
from autoindexer.models.autoindexer.sequence_parser import AutoIndexerIdParser
from autoindexer.models.autoindexer.tests.generate_sequences import DEFAULT_MARKER_MAP as MARKER_MAP


def _build_index_head(device):
    config = AutoIndexerConfig(hidden_size=64, head_dim=64, num_attention_heads=4, max_position_embeddings=1000)
    return AutoIndexerIndexHead(config).to(device).eval(), LlamaRotaryEmbedding(config, device)


def test_training_loss_masks_out_locked_positions():
    torch.manual_seed(0)
    device = torch.device("cpu")
    editable_start = 40
    labels = torch.arange(120, device=device).unsqueeze(0)

    _, cursor_indices, perturbations = batch_perturb_labels(
        MARKER_MAP,
        labels,
        max_length=400,
        return_indices_as_dict=True,
        editable_start=editable_start,
        min_len=5,
        mean_num_edits=10,
        mean_insert=5,
        mean_delete=3,
        mean_extend=7,
        edit_type_ratio=(0.15, 0.7, 0.15),
    )
    parsed_positions = get_parsed_positions_from_perturbations(perturbations, device=device)
    token_count = parsed_positions.concat_key_pos_ids.shape[0] + 1  # +1: the query row the block list doesn't itself store a key for

    index_head, rotary_emb = _build_index_head(device)
    hidden_states = torch.randn(1, token_count - 1, index_head.q_proj.in_features, device=device)

    attn_weights, (start_loss, end_loss), _ = index_head(
        hidden_states, parsed_positions=parsed_positions, cursor_indices=cursor_indices, rotary_emb=rotary_emb, editable_start=editable_start, return_weights=True,
    )
    assert torch.isfinite(start_loss) and torch.isfinite(end_loss)

    for edit_token_idx, (start_w, end_w) in attn_weights.items():
        # Every *legal* (finite) key must be at or past editable_start
        num_start_keys = start_w.shape[-1]
        num_end_keys = end_w.shape[-1]
        legal_start = torch.isfinite(start_w).squeeze()
        legal_end = torch.isfinite(end_w).squeeze()
        # keys are positions [0, num_keys) in this block's own (absolute, pre-edit-shift) coordinate frame.
        assert not legal_start[:editable_start].any() or num_start_keys <= editable_start
        assert not legal_end[:editable_start].any() or num_end_keys <= editable_start


def test_inference_index_head_masks_out_prompt_positions():
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch_size = 1
    prompt_len = 5
    editable_start = torch.full((batch_size,), prompt_len, dtype=torch.long, device=device)

    index_head, rotary_emb = _build_index_head(device)
    parser = AutoIndexerIdParser(MARKER_MAP, batch_size, device=device, editable_start=editable_start)
    cache = IndexKeysCache()

    # Replay a plain-token prompt (5 tokens), then one more "generated" token after it,
    # step-by-step through the parser + key cache, as the index head sees at inference.
    tokens = torch.arange(10, 16, device=device).unsqueeze(0)  # 6 plain, non-marker tokens
    hidden_states = torch.randn(batch_size, tokens.shape[1], index_head.q_proj.in_features, device=device)

    attn_weights = None
    for i in range(tokens.shape[1]):
        parser.update(tokens[:, i], torch.zeros(batch_size, 2, dtype=torch.long, device=device))
        parsed_positions = parser.get_parsed_positions()
        attn_weights, _, cache = index_head(
            hidden_states[:, i : i + 1], parsed_positions=parsed_positions, index_keys_cache=cache, rotary_emb=rotary_emb,
        )  # (batch_size, 2, 1, k_len=i+1)

    assert not torch.isfinite(attn_weights[:, :, :, :prompt_len]).any()
    assert torch.isfinite(attn_weights[:, :, :, prompt_len:]).any()


def test_forward_accepts_editable_start_and_trains_the_index_head():
    """End-to-end check that passing `editable_start` doesn't break `forward` and still yields a nonzero, finite index-head loss."""
    torch.manual_seed(0)
    config = AutoIndexerLlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        vocab_size=50,
        max_position_embeddings=1024,
        max_seq_length=800,
        mean_num_edits=10,
        mean_insert=3,
        mean_delete=2,
        mean_extend=5,
        perturb_prob=1.0,
        min_len=5,
        attn_implementation="autoindexer_eager",
        eos_token_id=49,
        bos_token_id=None,
        pad_token_id=None,
    )
    model = AutoIndexerLlamaModel(config).eval()

    # Keep labels below `eos_token_id` so none of them are spuriously read as an early EOS
    # by `batch_perturb_labels`, which would starve the edit sampler of room.
    labels = torch.randint(0, 40, (2, 300))
    editable_start = torch.tensor([50, 100])

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    out = model(labels=labels, editable_start=editable_start)

    assert torch.isfinite(out.loss)
    assert out.start_index_loss.item() > 0
    assert out.end_index_loss.item() > 0

    # `editable_start=None` (every existing pretraining config) must still work unchanged.
    out_unrestricted = model(labels=labels)
    assert torch.isfinite(out_unrestricted.loss)
