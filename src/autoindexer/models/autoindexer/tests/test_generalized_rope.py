import json
import os
from typing import List, Tuple

import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import sys
sys.path.append("")

from autoindexer.models.autoindexer.config import AutoIndexerConfig
from autoindexer.models.autoindexer.attention.reference import compute_attn_weights_ref
from autoindexer.models.autoindexer.perturb_labels import get_parsed_positions_from_perturbations, perturb_labels_per_sequence, to_operations
from autoindexer.models.autoindexer.type_utils import PositionBlock
from autoindexer.models.utils.tensors import print_matrix


LABEL_FILE_NAME = os.path.join(os.path.dirname(__file__), "labels", "position_matrix.json")


def get_position_matrix_from_parsed_positions(parsed_positions: List[PositionBlock], token_count: int, device: torch.device = None) -> Tuple[Tensor, Tensor]:
    # Testing rotary embedding and attention weight computation
    num_heads = 1
    head_dim = 2  # Set this to 2 so that we can directly recover the position ids from the attention weights for testing purposes
    query = torch.zeros((1, num_heads, token_count, head_dim), device=device)  # (batch_size, num_heads, seq_len, head_dim)
    query[..., 0] = 1
    keys = torch.zeros((1, num_heads, token_count, head_dim), device=device)  # (batch_size, num_heads, seq_len, head_dim)
    keys[..., 0] = 1
    keys_sign = torch.ones((1, num_heads, token_count, head_dim), device=device)  # (batch_size, num_heads, seq_len, head_dim)
    keys_sign[..., 0] = 0
    config = AutoIndexerConfig(num_heads=num_heads, head_dim=head_dim)
    rotary_emb = LlamaRotaryEmbedding(config, device=device)
    rotary_emb.inv_freq /= 100  # scale down inv_freq for testing to make the cos values within a reasonable range for arccos
    attn_weights = compute_attn_weights_ref(rotary_emb, query, keys, parsed_positions, scaling=1.0)
    attn_weights_sign = -torch.sign(compute_attn_weights_ref(rotary_emb, query, keys_sign, parsed_positions, scaling=1.0) * 100)
    attention_mask = attn_weights != float("-inf")
    # Get arccos of weights to recover position ids. Clamp into acos's [-1, 1] domain first:
    # tiny floating-point errors can otherwise push it slightly out of range and yield NaN.
    recovered_pos_ids = torch.round(attn_weights_sign * torch.acos(attn_weights.clamp(-1.0, 1.0)) / rotary_emb.inv_freq.item()).long()
    # Masked cells hold -inf, which clamps to -1 and would decode to acos(-1)=pi (e.g. 314 here).
    # They carry no position, so force them to 0.
    recovered_pos_ids = torch.where(attention_mask, recovered_pos_ids, torch.zeros_like(recovered_pos_ids))
    return recovered_pos_ids.squeeze(0).squeeze(0), attention_mask.squeeze(0).squeeze(0)


def test_generalized_rope():
    operations = to_operations([(0, 0, 0, 0, 5), (9, 0, 2, 2, 0), (4, 2, 0, 2, 4), (5, 1, 19, 20, 0), (5, 3, 6, 9, 0)])
    perturbations = perturb_labels_per_sequence(operations)
    token_count = perturbations[-1].curr_seq_idx
    device = torch.device("cuda") if torch.cuda.is_available() else None
    parsed_positions = get_parsed_positions_from_perturbations(perturbations, device)
    position_matrix, attention_mask = get_position_matrix_from_parsed_positions(parsed_positions, token_count, device)

    position_matrix = position_matrix.tolist()
    attention_mask = attention_mask.long().tolist()

    with open(os.path.join(os.path.dirname(__file__), "labels", "position_matrix.json"), "r") as f:
        expected = json.load(f)
        expected_position_matrix = torch.tensor(expected["position_matrix"])[0, :token_count, :token_count].tolist()
        expected_attention_mask = torch.tensor(expected["attention_mask"])[0, :token_count, :token_count].tolist()

    assert position_matrix == expected_position_matrix, f"Position matrix {position_matrix} do not match expected output."
    assert attention_mask == expected_attention_mask, f"Attention mask {attention_mask} do not match expected output."


if __name__ == "__main__":
    operations = to_operations([(0, 0, 0, 0, 5), (9, 0, 2, 2, 0), (4, 2, 0, 2, 4), (5, 1, 19, 20, 0), (5, 3, 6, 9, 0)])
    perturbations = perturb_labels_per_sequence(operations)
    token_count = perturbations[-1].curr_seq_idx
    device = torch.device("cuda") if torch.cuda.is_available() else None
    parsed_positions = get_parsed_positions_from_perturbations(perturbations, device)
    position_matrix, attention_mask = get_position_matrix_from_parsed_positions(parsed_positions, token_count, device)

    position_matrix = position_matrix.tolist()
    attention_mask = attention_mask.long().tolist()

    print_matrix(position_matrix)
    print_matrix(attention_mask)

    with open(LABEL_FILE_NAME, "r") as f:
        expected = json.load(f)
        expected_position_matrix = torch.tensor(expected["position_matrix"])[0, :token_count, :token_count].tolist()
        expected_attention_mask = torch.tensor(expected["attention_mask"])[0, :token_count, :token_count].tolist()

    assert position_matrix == expected_position_matrix, f"Position matrix {position_matrix} do not match expected output."
    assert attention_mask == expected_attention_mask, f"Attention mask {attention_mask} do not match expected output."
