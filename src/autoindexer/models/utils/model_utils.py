import torch.nn as nn


class IdentityRotaryEmbedding(nn.Module):
    """Drop-in replacement for a backbone model's own ``rotary_emb`` submodule (e.g.
    ``Qwen3RotaryEmbedding``/``LlamaRotaryEmbedding``) that returns an identity rotation
    (``cos=1``, ``sin=0``) instead of a real one, so that any attention module which
    unconditionally pre-applies ``apply_rotary_pos_emb(query, key, cos, sin)`` using
    ``position_embeddings`` computed from it effectively leaves ``query``/``key`` unrotated.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids):
        shape = (*position_ids.shape, self.head_dim)
        cos = x.new_ones(shape)
        sin = x.new_zeros(shape)
        return cos, sin
