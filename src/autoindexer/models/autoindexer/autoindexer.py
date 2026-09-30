
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple, List, Type

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from transformers import (
    PreTrainedModel,
    DynamicCache,
    GenerationMixin,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    LlamaModel,
    Qwen3Model,
    PretrainedConfig,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateNonBeamOutput, GenerateDecoderOnlyOutput
from transformers.loss.loss_utils import fixed_cross_entropy
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, logging

from autoindexer.models.autoindexer.config import AutoIndexerConfig, AutoIndexerLlamaConfig, AutoIndexerQwen3Config
from autoindexer.models.autoindexer.attn_layers import (
    IndexKeysCache,
    AutoIndexerIndexHead,
    compute_end_index_mask,
)
from autoindexer.models.autoindexer.attention import (
    register_autoindexer_attention_kernels,
    resolve_available_attn_implementation,
)
from autoindexer.models.autoindexer.sequence_parser import AutoIndexerIdParser
from autoindexer.models.autoindexer.perturb_labels import (
    batch_perturb_labels,
    get_parsed_positions_from_perturbations,
    transplant_corrupted_spans,
    Operation,
    ParsedPositions,
    PositionBlockList,
)
from autoindexer.models.autoindexer.type_utils import MarkerMap, MarkerType, PositionBlock, BlockType
from autoindexer.models.autoindexer._profiling import timed
from autoindexer.models.utils.model_utils import IdentityRotaryEmbedding

logger = logging.get_logger(__name__)


@dataclass
class AutoIndexerOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when
            `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    token_loss: Optional[torch.FloatTensor] = None
    marker_loss: Optional[torch.FloatTensor] = None
    index_loss: Optional[torch.FloatTensor] = None
    start_index_loss: Optional[torch.FloatTensor] = None
    end_index_loss: Optional[torch.FloatTensor] = None
    index_entropy_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    marker_logits: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor, ...]] = None
    sequences: List[Tensor] = None
    sampled_tokens: Optional[torch.LongTensor] = None
    sampled_indices: Optional[torch.LongTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class AutoIndexerGenerateOutput(GenerateDecoderOnlyOutput):
    sampled_tokens: Optional[torch.LongTensor] = None
    sampled_indices: Optional[torch.LongTensor] = None


class AutoIndexerPreTrainedModel(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def get_non_special_token_ids(config) -> List[int]:
    special_token_ids = {config.eos_token_id, config.bos_token_id, config.pad_token_id}
    if config.special_token_ids is not None:
        special_token_ids.update(config.special_token_ids)
    special_token_ids.discard(None)

    all_ids = set(range(config.vocab_size))
    valid_ids = all_ids - special_token_ids
    return sorted(valid_ids)


class AutoIndexerModelBase(AutoIndexerPreTrainedModel, GenerationMixin):
    backbone_model: Type[PreTrainedModel]
    rotary_emb_class: Type[nn.Module]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: AutoIndexerConfig):
        register_autoindexer_attention_kernels()
        # Select an appropriate `autoindexer_*` attention implementation depending on Triton/CuteDSL availability.
        requested = str(config._attn_implementation)
        available = resolve_available_attn_implementation(requested)
        if available != requested:
            slowdown = (
                ", which computes the same attention but materializes the full attention "
                "weights, so it is slower and uses more memory"
                if available == "autoindexer_eager"
                else ""
            )
            logger.warning(
                f"attn_implementation={requested!r} is not available in this environment "
                f"(its optional kernel dependency is not installed, or the installed version "
                f"doesn't support this machine's GPU); falling back to {available!r}{slowdown}."
            )
            config._attn_implementation = available

        super().__init__(config)
        # `_check_and_adjust_attn_implementation` (in `PreTrainedModel.__init__` above) silently falls back to a generic
        # `sdpa` implementation whenever nothing usable was requested
        if not str(self.config._attn_implementation).startswith("autoindexer_"):
            raise ValueError(
                f"AutoIndexer requires an autoindexer_* attn_implementation, got "
                f"{self.config._attn_implementation!r}. Pass attn_implementation="
                "'autoindexer_eager' (always available) or 'autoindexer_triton'/"
                "'autoindexer_cutedsl' if those kernels are installed."
            )

        self.model = self.backbone_model(config)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.model.rotary_emb = IdentityRotaryEmbedding(head_dim)

        vocab_size = config.vocab_size
        self.marker_map: MarkerMap = {
            "edit": vocab_size,
            "return_to_end": vocab_size + 1,
            "eos": config.eos_token_id if config.eos_token_id is not None else vocab_size + 2,
        }
        self.eos_in_vocab = config.eos_token_id is not None
        self.ignore_index = -100
        self.register_buffer("non_special_token_ids", torch.tensor(get_non_special_token_ids(config)), persistent=False)
        # Previous training step's clean `labels`
        self._prev_clean_labels: Optional[torch.Tensor] = None

        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        # Predicts, at every position, whether the *next* stream token is an ordinary token, the edit marker, the return marker, or eos
        self.marker_head = nn.Linear(config.hidden_size, len(MarkerType))
        # The edit marker is the only marker that is ever attended to, so it is the only one that needs an input representation.
        self.edit_embedding = nn.Embedding(1, config.hidden_size)
        if not self.eos_in_vocab:
            self.eos_embedding = nn.Embedding(1, config.hidden_size)

        # `self.device` resolves to `meta` when constructed under `accelerate.init_empty_weights()`
        rotary_emb_device = self.device if self.device.type != "meta" else None
        self.rotary_emb = self.rotary_emb_class(config, rotary_emb_device)

        self.index_head = AutoIndexerIndexHead(config)

        self._supports_flash_attn = True
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def embed_stream(self, input_ids: Tensor) -> Tensor:
        """Embed a raw stream whose marker ids sit just past the end of the vocabulary."""
        is_edit = input_ids == self.marker_map["edit"]
        is_return = input_ids == self.marker_map["return_to_end"]
        # `eos` only needs a placeholder/learnable embedding here when it isn't a real vocabulary id
        is_eos = (not self.eos_in_vocab) and (input_ids == self.marker_map["eos"])
        is_marker = is_edit | is_return | is_eos if not self.eos_in_vocab else is_edit | is_return
        # Markers are out of range for the embedding table, so look up a placeholder for them and overwrite afterwards
        embeds = self.model.embed_tokens(torch.where(is_marker, 0, input_ids))
        embeds = torch.where(is_edit.unsqueeze(-1), self.edit_embedding.weight[0].to(embeds.dtype), embeds)
        if not self.eos_in_vocab:
            embeds = torch.where(is_eos.unsqueeze(-1), self.eos_embedding.weight[0].to(embeds.dtype), embeds)
        return torch.where(is_return.unsqueeze(-1), torch.zeros_like(embeds), embeds)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        editable_start: Optional[torch.LongTensor] = None,
        perturb_operations: Optional[List[Operation]] = None,
        corruption_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, AutoIndexerOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is None:
            # No perturbation/loss machinery to run without `labels` -- likelihood-based eval
            # harnesses (e.g. dllm/lm-eval's `AREvalHarness.loglikelihood`) call `model(input_ids)`
            # expecting a plain causal-LM forward, so score `input_ids` verbatim instead of
            # crashing on `labels.device`/discarding the caller's tokens.
            return self._score(
                input_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        with timed("batch_perturb_labels"):
            # With probability `config.perturb_prob` train on the edit-perturbed sequence (the usual AutoIndexer objective)
            perturb_kwargs = self.config.perturb
            operations = perturb_operations
            if operations is not None:
                assert labels.shape[0] == 1, (
                    "perturb_operations (explicit, diff-derived edits) requires batch_size == 1: "
                    "the chain-of-edits attention kernel only supports one shared perturbation "
                    f"per batch, got batch_size={labels.shape[0]}."
                )
            elif random.random() >= self.config.perturb_prob:
                perturb_kwargs = {**perturb_kwargs, "mean_num_edits": 0}
            max_editable_start = int(editable_start.max().item()) if editable_start is not None else 0
            perturbed_ids, cursor_indices, perturbations = batch_perturb_labels(
                self.marker_map,
                labels,
                max_length=self.config.max_seq_length,
                ignore_index=self.ignore_index,
                return_indices_as_dict=True,
                editable_start=max_editable_start,
                operations=operations,
                **perturb_kwargs,
            )
        with timed("get_parsed_positions"):
            parsed_positions = get_parsed_positions_from_perturbations(perturbations, device=labels.device)

        # Replace tokens which will be deleted with random normal tokens
        transplant_prob = getattr(self.config, "transplant_prob", 0.0)
        input_ids = self.sanitize_labels(
            perturbed_ids[:, :-1],
            perturbations=perturbations,
            donor=self._prev_clean_labels if transplant_prob > 0 else None,
            corruption_ids=corruption_ids[:, : perturbed_ids.shape[1] - 1] if corruption_ids is not None else None,
        )
        if transplant_prob > 0:
            # Detached, real (pre-perturbation) tokens for *next* step's transplants
            self._prev_clean_labels = labels.detach()
        target_ids = perturbed_ids[:, 1:]
        token_logits, hidden_states, _ = self.decode(
            input_ids=input_ids,
            parsed_positions=parsed_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # EDIT/RETURN are never real vocabulary entries
        marker_targets = self.marker_targets(target_ids)
        non_vocab_marker = (marker_targets == MarkerType.EDIT) | (marker_targets == MarkerType.RETURN)
        if not self.eos_in_vocab:
            non_vocab_marker = non_vocab_marker | (marker_targets == MarkerType.EOS)
        # Prompt tokens are conditioning, not supervision: `target_ids[:, i]` predicts sequence position `i + 1`
        if editable_start is not None:
            positions = torch.arange(target_ids.shape[1], device=target_ids.device).unsqueeze(0)
            prompt_mask = positions < (editable_start.to(target_ids.device) - 1).unsqueeze(1)
        else:
            prompt_mask = torch.zeros_like(non_vocab_marker)
        token_targets = torch.where(non_vocab_marker | prompt_mask, self.ignore_index, target_ids)
        token_loss = self.calc_token_loss(token_logits, token_targets, num_items_in_batch=num_items_in_batch)
        marker_logits = self.marker_head(hidden_states)
        marker_loss = self.calc_marker_loss(marker_logits, marker_targets, num_items_in_batch=num_items_in_batch)

        ratio = getattr(self.config, "index_grad_ratio", 0.0)
        if ratio > 0:
            index_head_hidden_states = ratio * hidden_states + (1 - ratio) * hidden_states.detach()
        else:
            index_head_hidden_states = hidden_states.detach()
        # Distinct from `marker_calibration_weight` (a generation-only knob read off `generation_config` in `_sample`)
        entropy_weight = getattr(self.config, "index_entropy_weight", 0.0)
        index_weights, (start_index_loss, end_index_loss), _ = self.index_head(
            index_head_hidden_states,
            parsed_positions=parsed_positions,
            cursor_indices=cursor_indices,
            rotary_emb=self.rotary_emb,
            return_weights=entropy_weight > 0,
            editable_start=max_editable_start,
        )  # (batch_size, 2, q_len, k_len) where q_len = k_len = seq_len - 1

        out_sequences = None

        start_weight = getattr(self.config, "start_index_weight", 1.0)
        end_weight = getattr(self.config, "end_index_weight", 1.0)
        weighted_index_loss = start_weight * start_index_loss + end_weight * end_index_loss

        if entropy_weight > 0:
            index_entropy_loss = self.calc_index_entropy_loss(index_weights, reference=weighted_index_loss)
            weighted_index_loss = weighted_index_loss + entropy_weight * index_entropy_loss
        else:
            index_entropy_loss = None

        if num_items_in_batch is not None:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            weighted_index_loss = weighted_index_loss / world_size

        marker_weight = getattr(self.config, "marker_loss_weight", 1.0)
        total_loss = token_loss + marker_weight * marker_loss + weighted_index_loss

        return AutoIndexerOutput(
            loss=total_loss,
            token_loss=token_loss,
            marker_loss=marker_loss,
            start_index_loss=start_index_loss,
            end_index_loss=end_index_loss,
            index_entropy_loss=index_entropy_loss,
            sequences=out_sequences,
        )

    def calc_token_loss(self, token_logits: torch.Tensor, labels: torch.Tensor, num_items_in_batch: Optional[torch.Tensor] = None):
        token_logits = token_logits.transpose(-1, -2)  # "b l c -> b c l"
        return fixed_cross_entropy(token_logits, labels, num_items_in_batch, ignore_index=self.ignore_index)

    def marker_targets(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Map next-token targets to `MarkerType` classes."""
        targets = torch.full_like(target_ids, MarkerType.NONE)
        targets[target_ids == self.marker_map["edit"]] = MarkerType.EDIT
        targets[target_ids == self.marker_map["return_to_end"]] = MarkerType.RETURN
        targets[target_ids == self.marker_map["eos"]] = MarkerType.EOS
        return torch.where(target_ids == self.ignore_index, self.ignore_index, targets)

    def calc_marker_loss(self, marker_logits: torch.Tensor, marker_targets: torch.Tensor, num_items_in_batch: Optional[torch.Tensor] = None):
        """Focal loss (Lin et al., 2017) over `MarkerType` (`NONE`/`EDIT`/`RETURN`/`EOS`): scales
        each token's cross-entropy by `(1 - p_target) ** config.marker_focal_gamma`, so
        already-confident predictions -- mostly the vastly-more-common `MarkerType.NONE` -- barely
        contribute, letting the rare marker classes drive most of the gradient without an explicit
        class-frequency weight. `config.marker_loss_weight` (applied by the caller, `forward`) can
        scale the result further, the same way `start_index_weight`/`end_index_weight` do.
        """
        gamma = getattr(self.config, "marker_focal_gamma", 1.0)
        valid = marker_targets != self.ignore_index
        target = marker_targets.clamp(min=0)
        log_pt = F.log_softmax(marker_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = -((1.0 - log_pt.exp()) ** gamma) * log_pt
        loss = torch.where(valid, loss, torch.zeros_like(loss)).sum()
        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        else:
            loss = loss / valid.sum().clamp(min=1)
        return loss

    @staticmethod
    def _normalized_start_entropy(start_attn_weights: torch.Tensor) -> torch.Tensor:
        """Normalized entropy of the start-index distribution."""
        k_len = start_attn_weights.shape[-1]
        if k_len <= 1:
            return start_attn_weights.new_zeros(start_attn_weights.shape[:-1])
        probs = torch.softmax(start_attn_weights.float(), dim=-1)
        entropy = torch.special.entr(probs.clamp(min=torch.finfo(probs.dtype).tiny)).sum(dim=-1)
        return (entropy / math.log(k_len)).clamp(0.0, 1.0)

    def calc_index_entropy_loss(self, index_weights: Dict[int, Tuple[torch.Tensor, torch.Tensor]], reference: torch.Tensor) -> torch.Tensor:
        """Mean normalized start-index entropy at edit positions."""
        if not index_weights:
            return reference.new_zeros(())
        terms = [
            self._normalized_start_entropy(start_attn_weights).mean()
            for start_attn_weights, _end_attn_weights in index_weights.values()
        ]
        return torch.stack(terms).mean()

    @staticmethod
    def sample_indices(index_weights: Tensor, do_sample: bool = True) -> Tensor:
        """Draws one unmasked `(start, end)` index per row; `do_sample=False` takes the argmax."""
        if do_sample:
            return Categorical(logits=index_weights).sample()  # (batch_size, [q_len,] 2)
        return torch.argmax(index_weights, dim=-1)

    def mask_sampled_indices(self, raw_indices, new_sampled_tokens, prev_sampled_tokens):
        """Masks an already-drawn `sample_indices(...)` result: start unless this step is `edit`, end unless the previous step was."""
        raw_indices[..., 0].masked_fill_(new_sampled_tokens != self.marker_map["edit"], 0)
        raw_indices[..., 1].masked_fill_(prev_sampled_tokens != self.marker_map["edit"], 0)
        return raw_indices

    def sanitize_labels(
        self,
        labels: torch.Tensor,
        perturbations: Optional[List] = None,
        donor: Optional[torch.Tensor] = None,
        corruption_ids: Optional[torch.Tensor] = None,
    ):
        """Replace corrupted-span `ignore_index` labels (random or transplant filler)."""
        replacement = self.non_special_token_ids[torch.randint(0, len(self.non_special_token_ids), labels.shape, device=labels.device)]
        if corruption_ids is not None:
            replacement = torch.where(corruption_ids != self.ignore_index, corruption_ids, replacement)
        out = torch.where(labels != self.ignore_index, labels, replacement)
        transplant_prob = getattr(self.config, "transplant_prob", 0.0)
        if transplant_prob > 0 and perturbations is not None and donor is not None:
            out = transplant_corrupted_spans(out, perturbations, donor, self.ignore_index, transplant_prob)
        return out

    def decode(
        self,
        input_ids: Tensor,
        parsed_positions: ParsedPositions,
        past_key_values: Optional[DynamicCache] = None,
        index_keys_cache: Optional[IndexKeysCache] = None,
        is_inference: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        outputs = self.model(
            inputs_embeds=self.embed_stream(input_ids),
            parsed_positions=parsed_positions,
            rotary_emb=self.rotary_emb,
            use_cache=is_inference,
            past_key_values=past_key_values,
            index_keys_cache=index_keys_cache,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs["last_hidden_state"]

        past_key_values = outputs.past_key_values

        token_logits = self.lm_head(hidden_states)
        del outputs

        return token_logits, hidden_states, past_key_values

    @staticmethod
    def _plain_causal_positions(seq_len: int, device: torch.device) -> PositionBlockList:
        """Single `COMPLETE` block with ordinary causal attention."""
        pos_ids = torch.arange(seq_len, device=device)
        parsed_positions = PositionBlockList(
            [
                PositionBlock(
                    query_pos_ids=pos_ids,
                    key_pos_ids=pos_ids,
                    attn_mask=torch.ones(seq_len, dtype=torch.bool, device=device),
                    start_idx=0,
                    end_idx=seq_len,
                    block_type=BlockType.COMPLETE,
                )
            ]
        )
        parsed_positions.concat()
        return parsed_positions

    def _score(
        self,
        input_ids: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> AutoIndexerOutput:
        """Likelihood eval forward without edit perturbation."""
        parsed_positions = self._plain_causal_positions(input_ids.shape[1], input_ids.device)
        token_logits, hidden_states, _ = self.decode(
            input_ids=input_ids,
            parsed_positions=parsed_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        return AutoIndexerOutput(logits=token_logits, marker_logits=self.marker_head(hidden_states))

    @staticmethod
    def sample_from_logits(logits, do_sample: bool = True):
        """Sample or argmax from per-token logits."""
        if do_sample:
            return Categorical(logits=logits).sample()
        return torch.argmax(logits, dim=-1)

    def marker_logit_mask(self, is_mid_edit: Tensor, num_visible_tokens: Tensor) -> Tensor:
        """Boolean mask shaped like `marker_head`'s output, `True` on classes illegal at this step (see `_sample`)."""
        illegal = torch.zeros((*is_mid_edit.shape, len(MarkerType)), dtype=torch.bool, device=is_mid_edit.device)
        illegal[..., MarkerType.EDIT] = is_mid_edit | (num_visible_tokens == 0)
        illegal[..., MarkerType.RETURN] = ~is_mid_edit
        illegal[..., MarkerType.EOS] = is_mid_edit
        return illegal

    def _prefill_prompt(
        self,
        parser: AutoIndexerIdParser,
        input_ids: Tensor,
        total_sampled_indices: Tensor,
        past_key_values: Optional[DynamicCache],
        index_keys_cache: IndexKeysCache,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Optional[DynamicCache], ParsedPositions]:
        """Prefill KV and index-key cache for the prompt prefix."""
        batch_size, cur_len = input_ids.shape[:2]
        device = input_ids.device

        prefix_len = cur_len - 1
        for i in range(prefix_len):
            parser.update(input_ids[:, i], total_sampled_indices[:, i, :])
        if prefix_len > 0:
            # Already 4D so HF's mask-construction utilities pass it through untouched instead of dropping it
            sdpa_attention_mask = attention_mask[:, None, None, :prefix_len] if attention_mask is not None else None
            prev_attn_implementation = self.config._attn_implementation
            self.config._attn_implementation = "autoindexer_prefill_sdpa"
            try:
                _, prefill_hidden_states, past_key_values = self.decode(
                    input_ids=input_ids[:, :prefix_len],
                    parsed_positions=None,
                    past_key_values=past_key_values,
                    index_keys_cache=index_keys_cache,
                    is_inference=True,
                    attention_mask=sdpa_attention_mask,
                    return_dict=True,
                )
            finally:
                self.config._attn_implementation = prev_attn_implementation
            # Only `index_keys_cache` needs updating here
            hidden_shape = (batch_size, prefix_len, -1, self.index_head.head_dim)
            key_states = self.index_head.k_proj(prefill_hidden_states).view(hidden_shape).transpose(1, 2)
            index_keys_cache.update(key_states)

        parser.update(input_ids[:, cur_len - 1], total_sampled_indices[:, cur_len - 1, :])
        if attention_mask is not None:
            # `parser.attn_mask` was built assuming every replayed column is a real, visible token (see `AutoIndexerIdParser.update`)
            parser.attn_mask &= attention_mask[:, : parser.attn_mask.shape[-1]].bool()
        parsed_positions = parser.get_parsed_positions()
        return past_key_values, parsed_positions

    @staticmethod
    def _normalize_attention_mask(attention_mask: Tensor) -> Tensor:
        """Normalize `generate(..., attention_mask=...)` to a bool `(batch, seq_len)` mask."""
        if attention_mask.dim() not in (2, 4):
            raise ValueError(
                "attention_mask must be 2D (batch_size, seq_len) or 4D (batch_size, 1, "
                f"q_len, kv_len); got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            mask = attention_mask
        elif attention_mask.is_floating_point():
            # Additive-bias convention: attend wherever the bias isn't the large-negative/-inf sentinel used to zero a position out
            mask = attention_mask > torch.finfo(attention_mask.dtype).min
        else:
            # Integer 0/1 mask (e.g. straight off a dataset's `attention_mask` column).
            mask = attention_mask != 0
        if mask.dim() == 4:
            mask = mask.any(dim=1).any(dim=1)  # (batch, 1, q_len, kv_len) -> (batch, kv_len)
        return mask

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        AutoIndexer's counterpart to [`GenerationMixin._sample`]. The control flow deliberately mirrors the base
        implementation step-by-step (init the `scores`/`logits`/`attentions`/`hidden_states` tuples, track
        `unfinished_sequences`, run `logits_processor` on the raw next-token logits, sample or argmax a token,
        force finished rows to a fixed token, append to the running sequence, evaluate `stopping_criteria`, repeat)
        so the two can be compared directly. What differs is AutoIndexer-specific:

        - There is no KV-cache-aware `prepare_inputs_for_generation`/`cache_position` step; the custom
          `autoindexer_attention_forward`/index-head machinery tracks position information itself via
          `parsed_positions` and `index_keys_cache`, updated by `AutoIndexerIdParser` after every sampled token.
        - "Finished" rows are forced to the `eos` marker token (instead of `pad_token_id`) so the parser can
          correctly close out those sequences.
        - The `edit`/`return_to_end`/`eos` markers are never sampled from `lm_head`: a separate `marker_head`
          was trained with a plain cross-entropy over `MarkerType`, so at each step a marker is sampled
          directly from that same softmax (`sample_from_logits`, same `do_sample`/argmax choice as the
          vocabulary token), after masking illegal classes to `-inf` (`marker_logit_mask`) -- e.g.
          `EDIT`/`RETURN` alternation, `EOS` mid-edit. When `eos` is also a real vocabulary id
          (`eos_in_vocab`), its `lm_head` logit is masked out before `logits_processor` runs, so
          `marker_head` is the sole source of the EOS decision either way. A step that samples anything but
          `MarkerType.NONE` emits that marker instead of the sampled vocabulary token. The sampling
          parameters therefore only shape the text.
        - A second head (`index_head`) is sampled at every step to select the `(edit_start, edit_end)` positions
          associated with `edit` marker tokens.
        - When `generation_config.marker_calibration_weight > 0`, the index is sampled *before* the marker
          decision (not after, unlike the base loop) so the `EDIT` logit can be biased, in logit space, by the
          sampled start index's own probability -- see the comment at the call site. `config.index_entropy_weight`
          (a genuine *training*-time knob, unlike this one) is the unrelated counterpart; see `config.py`.
        - When `generation_config.greedy_mid_edit` is set, the vocabulary token is sampled greedily (argmax) on
          rows currently inside an open edit (`is_mid_edit`), regardless of `do_sample`/temperature/top-p/top-k --
          those still govern ordinary (not-mid-edit) text and the marker decision itself. Once the marker head
          and index head have committed to *where* to edit, the inserted/replacement content is the one place
          per-token sampling noise has no localization signal to correct it against, so it is the cheapest thing
          to de-risk without retraining -- see the free-gen `made_worse`/`wrong_content` outcomes this targets.
        - When `generation_config.max_delete_span` is set, the end-index head's candidates are additionally
          masked to within that many tokens of the sampled start (`compute_end_index_mask`'s `max_span`), capping
          how much a single edit can delete. This only guards against a mislocalized end cursor wiping out a
          large, otherwise-untouched tail -- the rare but severe "blast radius" failures free-gen eval turned up
          (e.g. a scripted 7-token deletion resolving as 382) -- it does not make ordinary, correctly-sized
          deletions any different. `None` (the default) reproduces the unbounded behavior training saw.
        - `generation_config.exclude_prompt_from_edits` (default `True`) sets the cursor's `editable_start`
          (`AutoIndexerIdParser`/`IterPositions.index_attn_mask`) to `cur_len`, i.e. the prompt length, so the
          cursor can only ever point at freshly generated tokens, never back into the prompt -- correct for
          instruction-tuned chat generation, where the prompt is a system/user turn the response must not edit.
          Set to `False` when the "prompt" is itself the document to repair in place (e.g. `scripts/edit_eval`'s
          free-running rollout), so the cursor may point anywhere in it, matching `forward`'s training-time
          default of `editable_start=0`.
        - `generation_config.editable_start` (default `None`) overrides `exclude_prompt_from_edits` with an
          exact cursor start instead of its all-or-nothing `0`/`cur_len` choice -- e.g. code repair, where a
          "prompt" (function signature/docstring) must stay locked but a "draft" appended after it (the buggy
          solution body the cursor should repair) must remain editable, so the caller passes the token length
          of the prompt alone. An `int` broadcasts to every row; a `(batch_size,)` `torch.Tensor` sets it per
          row (e.g. differing prompt lengths under left-padded batching, in absolute columns of the padded
          input -- see `scripts/edit_eval`'s `batch_rollout`).
        - `generation_config.marker_top_p` (default `1.0`, i.e. disabled) applies nucleus filtering to the
          marker head's own softmax, independently of the vocabulary head's `top_p` -- since `marker_head`'s
          logits never pass through `logits_processor` (see above), a caller wanting to suppress low-probability
          marker classes (e.g. a stray `EDIT` sampled off the calibration-biased tail) sets this directly rather
          than via `generation_config.top_p`, which only shapes the text.
        - `generation_config.marker_temperature` (default `1.0`, i.e. disabled) rescales the marker head's own
          logits before `marker_top_p`'s nucleus filtering, independently of the vocabulary head's
          `temperature` -- same rationale as `marker_top_p`, and applied first to match
          `GenerationMixin._get_logits_warper`'s own temperature-before-top_p ordering for the vocabulary head.
        `marker_calibration_weight`/`greedy_mid_edit`/`max_delete_span`/`exclude_prompt_from_edits`/
        `editable_start`/`marker_top_p`/`marker_temperature` are all read off `generation_config` (with
        `getattr(..., default)`, since none are real `GenerationConfig` fields), not `self.config` -- they're
        decoding strategies, not properties of the trained checkpoint, so a caller sets them per `generate()`
        call (e.g. `model.generate(..., max_delete_span=20)`) exactly like `temperature`/`top_p`.
        """
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample

        if output_attentions or output_hidden_states:
            logger.warning_once(
                "AutoIndexerModelBase._sample does not currently plumb `output_attentions`/`output_hidden_states` "
                "through the custom `decode()` path used for step-by-step inference; these will be returned empty."
            )

        # init attention / hidden states / scores tuples -- mirrors `GenerationMixin._sample`
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        attentions = () if (return_dict_in_generate and output_attentions) else None
        hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        device = input_ids.device
        dtype = input_ids.dtype

        # Batching prompts of different lengths (see `scripts/edit_eval/free_gen.batch_rollout`) needs left-padding
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = self._normalize_attention_mask(attention_mask)

        # `generation_config.max_length` bounds the *final* sequence length (prefix + generated)
        max_length = generation_config.max_length or self.config.max_seq_length
        # Decoding strategies, not trained-checkpoint properties
        calibration_weight = getattr(generation_config, "marker_calibration_weight", 1.0)
        # Orthogonal to `calibration_weight`: forces the vocabulary token sample to argmax while `is_mid_edit` is set
        greedy_mid_edit = getattr(generation_config, "greedy_mid_edit", True)
        # Safety valve: caps how many tokens the end cursor may delete
        max_delete_span = getattr(generation_config, "max_delete_span", self.config.perturb["mean_delete"] * 5)
        calibration_bias = getattr(generation_config, "calibration_bias", 0.0)
        if max_delete_span is not None:
            max_delete_span = max(0, int(max_delete_span))
        # Whether the cursor may point back into the prompt at all
        exclude_prompt_from_edits = getattr(generation_config, "exclude_prompt_from_edits", True)
        # Nucleus filtering for the marker head's own softmax
        marker_top_p = getattr(generation_config, "marker_top_p", 1.0)
        # Temperature rescaling for the marker head's own softmax
        marker_temperature = getattr(generation_config, "marker_temperature", 1.0)
        # Explicit override of the cursor's `editable_start`
        editable_start_override = getattr(generation_config, "editable_start", None)

        # keep track of which sequences are already finished -- mirrors `GenerationMixin._sample`
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        total_sampled_tokens = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        total_sampled_tokens[:, :cur_len] = input_ids
        total_sampled_indices = torch.zeros((batch_size, max_length, 2), dtype=torch.long, device=device)

        # `exclude_prompt_from_edits=True` (the default) locks the cursor out of the prompt
        if editable_start_override is not None:
            if isinstance(editable_start_override, torch.Tensor):
                editable_start = editable_start_override.to(device=device, dtype=torch.long)
            else:
                editable_start = torch.full((batch_size,), int(editable_start_override), dtype=torch.long, device=device)
        else:
            editable_start = torch.full(
                (batch_size,), cur_len if exclude_prompt_from_edits else 0, dtype=torch.long, device=device
            )
        parser = AutoIndexerIdParser(self.marker_map, batch_size, device=device, editable_start=editable_start, eos_in_vocab=self.eos_in_vocab)
        
        prev_sampled_tokens = input_ids[:, -1]
        is_mid_edit = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        past_key_values = None
        index_keys_cache = IndexKeysCache()

        past_key_values, parsed_positions = self._prefill_prompt(
            parser, input_ids, total_sampled_indices, past_key_values, index_keys_cache, attention_mask=attention_mask
        )
        input_ids = input_ids[:, -1:]

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=device) and cur_len < max_length:
            # forward pass to get next token logits/index-weights
            token_logits, step_hidden_states, past_key_values = self.decode(
                input_ids=input_ids,
                parsed_positions=parsed_positions,
                past_key_values=past_key_values,
                index_keys_cache=index_keys_cache,
                is_inference=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            index_weights, _, index_keys_cache = self.index_head(
                step_hidden_states,
                parsed_positions=parsed_positions,
                index_keys_cache=index_keys_cache,
                rotary_emb=self.rotary_emb,
            )  # (batch_size, 2, q_len, k_len)

            # Copy is needed to avoid keeping a hanging ref to token_logits
            next_token_logits = token_logits[:, -1, :].to(copy=True, dtype=torch.float32, device=device)
            index_weights = index_weights[:, :, -1]  # (B, 2, k_len)

            # EOS is now decided exclusively by `marker_head` (below), even when it's a real vocabulary id (`eos_in_vocab`)
            if self.eos_in_vocab:
                next_token_logits[..., self.marker_map["eos"]] = float("-inf")

            # Finished rows may carry stale logits
            next_token_logits[unfinished_sequences == 0] = 0
            if max_delete_span is not None:
                # `index_head` already masked `key_pos_ids <= 0` (end at/before start)
                end_allowed = compute_end_index_mask(parsed_positions, max_span=max_delete_span + 1)
                index_weights[:, 1] = index_weights[:, 1].masked_fill(~end_allowed, float("-inf"))
            index_all_masked = torch.isneginf(index_weights).all(dim=-1)
            index_weights[(unfinished_sequences == 0).unsqueeze(-1) | index_all_masked] = 0

            # Sampled before the marker decision (not after, unlike the base loop) and reused below via `mask_sampled_indices`
            raw_sampled_indices = self.sample_indices(index_weights, do_sample=do_sample)  # (B, 2)

            # pre-process distribution -- mirrors `GenerationMixin._sample`
            next_token_scores = logits_processor(total_sampled_tokens[:, :cur_len], next_token_logits)

            # AutoIndexer-specific: the marker head decides whether this step is a marker at all
            index_attn_mask = parsed_positions.index_attn_mask if parsed_positions.index_attn_mask is not None else parsed_positions.attn_mask
            num_visible_tokens = index_attn_mask.sum(dim=-1)
            step_marker_logits = self.marker_head(step_hidden_states[:, -1]).float()
            step_marker_logits = step_marker_logits.masked_fill(
                self.marker_logit_mask(is_mid_edit, num_visible_tokens), float("-inf")
            )
            if calibration_weight > 0:
                start_logprobs = F.log_softmax(index_weights[:, 0, :].float(), dim=-1)
                sampled_start_logprob = start_logprobs.gather(-1, raw_sampled_indices[:, 0:1]).squeeze(-1)  # (B,)
                step_marker_logits[:, MarkerType.EDIT] = step_marker_logits[:, MarkerType.EDIT] + calibration_weight * sampled_start_logprob + calibration_bias
            if do_sample and marker_temperature != 1.0:
                step_marker_logits = TemperatureLogitsWarper(temperature=marker_temperature)(None, step_marker_logits)
            if do_sample and marker_top_p < 1.0:
                step_marker_logits = TopPLogitsWarper(top_p=marker_top_p)(None, step_marker_logits)
            sampled_marker = self.sample_from_logits(step_marker_logits, do_sample=do_sample)

            # Store scores, attentions and hidden_states when required -- mirrors `GenerationMixin._sample`
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    attentions += (None,)
                if output_hidden_states:
                    hidden_states += (step_hidden_states,)

            # token selection
            new_tokens = self.sample_from_logits(next_token_scores, do_sample=do_sample)
            if greedy_mid_edit and do_sample:
                # `is_mid_edit` here is still last step's value (updated below, after this token is chosen)
                new_tokens = torch.where(is_mid_edit, next_token_scores.argmax(dim=-1), new_tokens)
            marker_ids = torch.where(
                sampled_marker == MarkerType.RETURN,
                torch.full_like(new_tokens, self.marker_map["return_to_end"]),
                torch.full_like(new_tokens, self.marker_map["edit"]),
            )
            marker_ids = torch.where(sampled_marker == MarkerType.EOS, torch.full_like(new_tokens, self.marker_map["eos"]), marker_ids)
            new_tokens = torch.where(sampled_marker != MarkerType.NONE, marker_ids, new_tokens)

            # finished sequences should have their next token be the eos marker -- mirrors the pad-token forcing
            # in `GenerationMixin._sample`, but uses the marker-based eos token so `AutoIndexerIdParser` can close
            # out those rows correctly
            sampled_tokens = new_tokens * unfinished_sequences + self.marker_map["eos"] * (1 - unfinished_sequences)

            is_edit = sampled_tokens == self.marker_map["edit"]
            is_return = sampled_tokens == self.marker_map["return_to_end"]
            is_eos = sampled_tokens == self.marker_map["eos"]
            is_mid_edit = (is_mid_edit & ~(is_return | is_eos)) | is_edit

            # update generated ids, length for next step -- mirrors `GenerationMixin._sample`
            total_sampled_tokens[:, cur_len] = sampled_tokens
            if streamer is not None:
                streamer.put(sampled_tokens.cpu())

            # AutoIndexer-specific: mask the (edit_start, edit_end) index pair already drawn above (reused, not
            # re-sampled -- see the module note near `raw_sampled_indices`) and feed the token/index pair through
            # the parser to keep `parsed_positions` (and the reconstructed sequence) up to date.
            sampled_indices = self.mask_sampled_indices(raw_sampled_indices, sampled_tokens, prev_sampled_tokens)
            parser.update(sampled_tokens, sampled_indices)
            parsed_positions = parser.get_parsed_positions()
            total_sampled_indices[:, cur_len] = sampled_indices

            prev_sampled_tokens = sampled_tokens
            input_ids = sampled_tokens.unsqueeze(1)
            cur_len += 1

            # mirrors `GenerationMixin._sample`, additionally combining in the model's own marker-based eos signal
            unfinished_sequences = unfinished_sequences & ~is_eos & ~stopping_criteria(total_sampled_tokens[:, :cur_len], scores)
            this_peer_finished = unfinished_sequences.max() == 0

        parser.resolve_parsing(dtype)

        if streamer is not None:
            streamer.end()

        sequences = parser.parsed_sequences

        if return_dict_in_generate:
            return AutoIndexerGenerateOutput(
                sequences=sequences,
                scores=scores,
                logits=raw_logits,
                attentions=attentions,
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                sampled_tokens=total_sampled_tokens,
                sampled_indices=total_sampled_indices,
            )
        else:
            return sequences


class AutoIndexerQwen3Model(AutoIndexerModelBase):
    backbone_model = Qwen3Model
    rotary_emb_class = Qwen3RotaryEmbedding
    config_class = AutoIndexerQwen3Config


class AutoIndexerLlamaModel(AutoIndexerModelBase):
    backbone_model = LlamaModel
    rotary_emb_class = LlamaRotaryEmbedding
    config_class = AutoIndexerLlamaConfig


AutoIndexerModel = AutoIndexerQwen3Model