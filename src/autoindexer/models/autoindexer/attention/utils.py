from torch import Tensor
from transformers.models.llama.modeling_llama import rotate_half


def apply_cos_sin(x: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim=1):
    return (x * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(x) * sin.unsqueeze(unsqueeze_dim))
