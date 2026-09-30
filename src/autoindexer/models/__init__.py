def get_hf_model(model_name):
    if model_name == "autoindexer":
        from autoindexer.models.autoindexer import AutoIndexerModel, AutoIndexerConfig

        return AutoIndexerModel, AutoIndexerConfig
    elif model_name == "autoindexer_qwen3":
        from autoindexer.models.autoindexer import AutoIndexerQwen3Model, AutoIndexerQwen3Config

        return AutoIndexerQwen3Model, AutoIndexerQwen3Config
    elif model_name == "autoindexer_llama":
        from autoindexer.models.autoindexer import AutoIndexerLlamaModel, AutoIndexerLlamaConfig

        return AutoIndexerLlamaModel, AutoIndexerLlamaConfig
    elif model_name == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        return LlamaForCausalLM, LlamaConfig
    elif model_name == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        return Qwen3ForCausalLM, Qwen3Config
    else:
        return None, None


def register_attn_implementation(attn_impl_name: str) -> None:
    if attn_impl_name and str(attn_impl_name).startswith("autoindexer_"):
        from autoindexer.models.autoindexer.attention import register_autoindexer_attention_kernels

        register_autoindexer_attention_kernels()
