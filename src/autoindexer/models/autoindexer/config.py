from typing import Optional

from transformers import LlamaConfig, Qwen3Config

# AutoIndexer's chain-of-edits attention backends
DEFAULT_ATTN_IMPLEMENTATION = "autoindexer_cutedsl"


def _normalize_edit_type_ratio(ratio) -> tuple[float, float, float]:
    ratio = tuple(ratio)
    if len(ratio) != 3:
        raise ValueError(f"edit_type_ratio must have 3 elements (pure_insert, substitution, pure_deletion), got {len(ratio)}")
    total = sum(ratio)
    assert total > 0
    return tuple(x / total for x in ratio)


def _set_autoindexer_fields(
    self,
    max_seq_length: int,
    min_len: int,
    mean_num_edits: int,
    mean_insert: int,
    mean_delete: int,
    mean_extend: int,
    inplace_edits: bool,
    sort_edits: bool,
    append_eos: bool,
    edit_type_ratio,
    edit_lag_prob: float,
    mean_edit_lag: int,
    special_token_ids: list,
    index_grad_ratio: float,
    index_head_scaling: float,
    start_index_weight: float,
    end_index_weight: float,
    perturb_prob: float,
    marker_loss_weight: float,
    marker_init_from_base: bool,
    index_entropy_weight: float = 0.0,
    marker_focal_gamma: float = 1.0,
    transplant_prob: float = 0.0,
    saved_perturb: dict = None,
) -> None:
    # `save_pretrained` persists the perturbation hyperparameters only inside the nested `perturb` dict
    saved_perturb = saved_perturb or {}
    min_len = saved_perturb.get("min_len", min_len)
    mean_num_edits = saved_perturb.get("mean_num_edits", mean_num_edits)
    mean_insert = saved_perturb.get("mean_insert", mean_insert)
    mean_delete = saved_perturb.get("mean_delete", mean_delete)
    mean_extend = saved_perturb.get("mean_extend", mean_extend)
    inplace_edits = saved_perturb.get("inplace_edits", inplace_edits)
    sort_edits = saved_perturb.get("sort_edits", sort_edits)
    append_eos = saved_perturb.get("append_eos", append_eos)
    edit_lag_prob = saved_perturb.get("edit_lag_prob", edit_lag_prob)
    mean_edit_lag = saved_perturb.get("mean_edit_lag", mean_edit_lag)
    if edit_type_ratio is not None:
        edit_type_ratio = _normalize_edit_type_ratio(edit_type_ratio)
    elif "edit_type_ratio" in saved_perturb:
        edit_type_ratio = _normalize_edit_type_ratio(saved_perturb["edit_type_ratio"])
    else:
        edit_type_ratio = (0, 1, 0) if inplace_edits else (1/3, 1/3, 1/3)

    self.max_seq_length = max_seq_length
    self.inplace_edits = inplace_edits
    self.sort_edits = sort_edits
    self.append_eos = append_eos
    # (pure_insert, substitution, pure_deletion) weights for edit sampling -- see `sample_cursor_operations`
    self.edit_type_ratio = edit_type_ratio
    # Fraction of forward passes that apply edit perturbation (and index-head/edit-token losses)
    self.perturb_prob = perturb_prob
    if inplace_edits:
        assert mean_insert == mean_delete, "When inplace_edits (substitution edits only) is true, mean_insert and mean_delete must be equal"
        pure_insert, _, pure_deletion = edit_type_ratio
        assert pure_insert == 0 and pure_deletion == 0, (
            "When inplace_edits is true, edit_type_ratio must have zero pure_insert and pure_deletion mass"
        )
    self.special_token_ids = special_token_ids
    self.index_grad_ratio = index_grad_ratio  # Default to 0.0 to fully detach gradients from propagating
    self.index_head_scaling = index_head_scaling  # Multiplier on the default 1/sqrt(head_dim) index-head logit scaling
    self.start_index_weight = start_index_weight  # Relative weight of start_index_loss within index_loss
    self.end_index_weight = end_index_weight  # Relative weight of end_index_loss within index_loss
    # Flat multiplier on the marker head's focal loss (`AutoIndexerModelBase.calc_marker_loss`).
    self.marker_loss_weight = marker_loss_weight
    # Focal loss (Lin et al., 2017) exponent for the marker head's cross-entropy
    self.marker_focal_gamma = marker_focal_gamma
    # Training-time: minimizes the start-index head's normalized entropy at real edits
    self.index_entropy_weight = index_entropy_weight
    # Fraction of corrupted spans filled from `_prev_clean_labels` instead of random tokens.
    self.transplant_prob = transplant_prob
    # Seed the edit embedding from the pretrained vocabulary's own row statistics rather than `initializer_range` noise
    self.marker_init_from_base = marker_init_from_base
    self.perturb = {
        "min_len": min_len,
        "mean_num_edits": mean_num_edits,
        "mean_insert": mean_insert,
        "mean_delete": mean_delete,
        "mean_extend": mean_extend,
        "inplace_edits": inplace_edits,
        "sort_edits": sort_edits,
        # Poisson(mean_insert)/Poisson(mean_delete) put only ~e^-mean mass on zero
        "edit_type_ratio": list(edit_type_ratio),
        # Probability an edit fires soon after its corruption (vs uniform later placement).
        "edit_lag_prob": edit_lag_prob,
        "mean_edit_lag": mean_edit_lag,
        # If True, an EOS marker is appended at the end of every label even when no EOS token is explicitly present
        "append_eos": append_eos,
    }


class AutoIndexerQwen3Config(Qwen3Config):
    model_type = "autoindexer_qwen3"

    def __init__(
        self,
        max_seq_length: int = 2048,
        min_len: int = 15,
        mean_num_edits: int = 10,
        mean_insert: int = 2,
        mean_delete: int = 2,
        mean_extend: int = 0,
        inplace_edits: bool = False,
        sort_edits: bool = False,
        append_eos: bool = False,
        edit_type_ratio=None,
        edit_lag_prob: float = 0.0,
        mean_edit_lag: int = 3,
        special_token_ids: list = None,
        index_grad_ratio: float = 0.0,
        index_head_scaling: float = 1.0,
        start_index_weight: float = 1.0,
        end_index_weight: float = 1.0,
        perturb_prob: float = 1.0,
        marker_loss_weight: float = 1.0,
        marker_init_from_base: bool = True,
        index_entropy_weight: float = 0.0,
        marker_focal_gamma: float = 1.0,
        transplant_prob: float = 0.0,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Ensure RoPE supports max_seq_length.
        kwargs["max_position_embeddings"] = max(kwargs.get("max_position_embeddings", max_seq_length), max_seq_length)
        # `from_dict` injects `attn_implementation=None`; prefer explicit or persisted backend.
        attn_implementation = (
            kwargs.get("attn_implementation") or kwargs.get("autoindexer_attn_implementation") or DEFAULT_ATTN_IMPLEMENTATION
        )
        kwargs["attn_implementation"] = attn_implementation
        super().__init__(eos_token_id=eos_token_id, **kwargs)
        _set_autoindexer_fields(
            self,
            max_seq_length,
            min_len,
            mean_num_edits,
            mean_insert,
            mean_delete,
            mean_extend,
            inplace_edits,
            sort_edits,
            append_eos,
            edit_type_ratio,
            edit_lag_prob,
            mean_edit_lag,
            special_token_ids,
            index_grad_ratio,
            index_head_scaling,
            start_index_weight,
            end_index_weight,
            perturb_prob,
            marker_loss_weight,
            marker_init_from_base,
            index_entropy_weight=index_entropy_weight,
            marker_focal_gamma=marker_focal_gamma,
            transplant_prob=transplant_prob,
            saved_perturb=kwargs.get("perturb"),
        )
        # Persist backend: `attn_implementation` is stripped from saved configs.
        self.autoindexer_attn_implementation = attn_implementation


class AutoIndexerLlamaConfig(LlamaConfig):
    model_type = "autoindexer_llama"

    def __init__(
        self,
        max_seq_length: int = 2048,
        min_len: int = 15,
        mean_num_edits: int = 10,
        mean_insert: int = 2,
        mean_delete: int = 2,
        mean_extend: int = 0,
        inplace_edits: bool = False,
        sort_edits: bool = False,
        append_eos: bool = False,
        edit_type_ratio=None,
        edit_lag_prob: float = 0.0,
        mean_edit_lag: int = 3,
        special_token_ids: list = None,
        index_grad_ratio: float = 0.0,
        index_head_scaling: float = 1.0,
        start_index_weight: float = 1.0,
        end_index_weight: float = 1.0,
        perturb_prob: float = 1.0,
        marker_loss_weight: float = 1.0,
        marker_init_from_base: bool = True,
        index_entropy_weight: float = 0.0,
        marker_focal_gamma: float = 1.0,
        transplant_prob: float = 0.0,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Ensure RoPE supports max_seq_length.
        kwargs["max_position_embeddings"] = max(kwargs.get("max_position_embeddings", max_seq_length), max_seq_length)
        # `from_dict` injects `attn_implementation=None`; prefer explicit or persisted backend.
        attn_implementation = (
            kwargs.get("attn_implementation") or kwargs.get("autoindexer_attn_implementation") or DEFAULT_ATTN_IMPLEMENTATION
        )
        kwargs["attn_implementation"] = attn_implementation
        super().__init__(eos_token_id=eos_token_id, **kwargs)
        _set_autoindexer_fields(
            self,
            max_seq_length,
            min_len,
            mean_num_edits,
            mean_insert,
            mean_delete,
            mean_extend,
            inplace_edits,
            sort_edits,
            append_eos,
            edit_type_ratio,
            edit_lag_prob,
            mean_edit_lag,
            special_token_ids,
            index_grad_ratio,
            index_head_scaling,
            start_index_weight,
            end_index_weight,
            perturb_prob,
            marker_loss_weight,
            marker_init_from_base,
            index_entropy_weight=index_entropy_weight,
            marker_focal_gamma=marker_focal_gamma,
            transplant_prob=transplant_prob,
            saved_perturb=kwargs.get("perturb"),
        )
        # See the matching comment in `AutoIndexerQwen3Config.__init__`.
        self.autoindexer_attn_implementation = attn_implementation


class AutoIndexerConfig(AutoIndexerQwen3Config):
    model_type = "autoindexer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


__all__ = ["AutoIndexerConfig", "AutoIndexerQwen3Config", "AutoIndexerLlamaConfig"]
