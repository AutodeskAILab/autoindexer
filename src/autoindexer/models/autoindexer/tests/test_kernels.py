import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import sys
sys.path.append("")

from autoindexer.models.autoindexer.config import AutoIndexerConfig
from autoindexer.models.autoindexer.attention.reference import compute_attn_weights_ref
from autoindexer.models.autoindexer.perturb_labels import sample_cursor_operations, perturb_labels_per_sequence, \
    get_parsed_positions_from_perturbations

if __name__ == "__main__":
    operations, token_count = sample_cursor_operations(100, 5, 4, 5, 3, 5)
    perturbations = perturb_labels_per_sequence(operations)
    token_count = perturbations[-1].curr_seq_idx
    device = torch.device("cuda") if torch.cuda.is_available() else None
    parsed_positions = get_parsed_positions_from_perturbations(perturbations, device)

    # Testing rotary embedding and attention weight computation
    batch_size = 16
    num_heads = 4
    head_dim = 32
    query = torch.randn((batch_size, num_heads, token_count, head_dim), device=device)
    keys = torch.zeros((batch_size, num_heads, token_count, head_dim), device=device)
    values = torch.zeros((batch_size, num_heads, token_count, head_dim), device=device)
    config = AutoIndexerConfig(num_heads=num_heads, head_dim=head_dim)
    rotary_emb = LlamaRotaryEmbedding(config, device=device)

    attn_weights = compute_attn_weights_ref(rotary_emb, query, keys, parsed_positions, scaling=1.0)
    print(attn_weights)