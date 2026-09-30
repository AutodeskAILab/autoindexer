from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, Qwen2TokenizerFast

from autoindexer.models.autoindexer.autoindexer import *
from autoindexer.models.autoindexer.config import *

# Register with transformers' Auto* factories so any consumer that has imported this package can
# resolve an AutoIndexer checkpoint's `model_type` via `AutoConfig`/`AutoModelForCausalLM.from_pretrained`.
for _config_cls, _model_cls, _tokenizer_cls in (
    (AutoIndexerQwen3Config, AutoIndexerQwen3Model, Qwen2TokenizerFast),
    (AutoIndexerLlamaConfig, AutoIndexerLlamaModel, LlamaTokenizerFast),
):
    AutoConfig.register(_config_cls.model_type, _config_cls)
    AutoModelForCausalLM.register(_config_cls, _model_cls)
    # `AutoIndexerQwen3Model`/`AutoIndexerLlamaModel` are the complete model (no separate
    # bare-backbone class), so the same class also registers under the generic `AutoModel`.
    AutoModel.register(_config_cls, _model_cls)
    # `merge_lora_checkpoint` doesn't save tokenizer files alongside the merged model, so
    # `AutoTokenizer.from_pretrained` needs the tokenizer class registered against the config type.
    AutoTokenizer.register(_config_cls, fast_tokenizer_class=_tokenizer_cls)
del _config_cls, _model_cls, _tokenizer_cls
