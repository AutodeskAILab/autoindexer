"""Inference-only generation-time safety valves in `AutoIndexerModelBase._sample`."""

from typing import List, Optional

import torch
from transformers import GenerationConfig

from autoindexer.models.autoindexer.config import AutoIndexerQwen3Config
from autoindexer.models.autoindexer.autoindexer import AutoIndexerQwen3Model

TINY_MODEL_KWARGS = dict(
    hidden_size=32,
    head_dim=16,
    num_attention_heads=2,
    num_key_value_heads=2,
    num_hidden_layers=1,
    intermediate_size=64,
    vocab_size=200,
    max_position_embeddings=1000,
    max_seq_length=400,
    mean_num_edits=3,
    mean_insert=3,
    mean_delete=3,
    mean_extend=10,
    min_len=5,
    edit_type_ratio=[1, 1, 1],
    perturb_prob=1.0,
    attn_implementation="autoindexer_eager",
    eos_token_id=None,
)


def _make_model(seed: int = 123) -> AutoIndexerQwen3Model:
    torch.manual_seed(seed)
    config = AutoIndexerQwen3Config(**TINY_MODEL_KWARGS)
    config.use_cache = True
    model = AutoIndexerQwen3Model(config)
    model.eval()
    return model


def _generate(model, prompt, seed: int, max_new_tokens: int = 120, output_scores: bool = False, **generation_overrides):
    """`**generation_overrides` reaches `_sample` via `generation_config`, e.g. `max_delete_span=5`."""
    torch.manual_seed(seed)
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        eos_token_id=model.marker_map["eos"],
        pad_token_id=model.marker_map["eos"],
        return_dict_in_generate=True,
        output_scores=output_scores,
        **generation_overrides,
    )
    return model.generate(prompt, generation_config=gen_config)


def _resolved_num_deletes(model, tokens: List[int], indices: List[List[int]]) -> List[int]:
    """Reconstruct each edit's resolved deletion length -- mirrors `free_gen.py`'s `replay_edit_stream`."""
    from autoindexer.models.autoindexer.sequence_parser import AutoIndexerIdParser

    marker_map = model.marker_map
    parser = AutoIndexerIdParser(marker_map, batch_size=1, device=None)
    num_deletes = []
    pending_end: Optional[int] = None
    for step, (token, index) in enumerate(zip(tokens, indices)):
        token = int(token)
        if pending_end is None and token == marker_map["edit"]:
            pending_end = "open"
        elif pending_end == "open" and token not in (marker_map["edit"], marker_map["return_to_end"], marker_map["eos"]):
            # First token after the edit marker carries the end index
            end_idx = int(index[1])
            resolved_end = int(parser.curr_end_pos[0] + 1) if end_idx == step - 1 else int(parser.position_ids[0, end_idx])
            resolved_start = int(parser.curr_pos[0]) + 1
            num_deletes.append(resolved_end - resolved_start)
            pending_end = "closed"  # only read once, matches replay_edit_stream's open_record.end
        parser.update(
            torch.tensor([token], dtype=torch.long),
            torch.tensor([list(index)], dtype=torch.long),
        )
        if token in (marker_map["return_to_end"], marker_map["eos"]):
            pending_end = None
    return num_deletes


def test_generation_config_knobs_do_not_affect_model_config():
    """Sanity check for the "generation config, not training config" contract itself."""
    model = _make_model()
    assert not hasattr(model.config, "marker_calibration_weight")
    assert not hasattr(model.config, "greedy_mid_edit")
    assert not hasattr(model.config, "max_delete_span")
    assert not hasattr(model.config, "calibration_bias")
    assert not hasattr(model.config, "exclude_prompt_from_edits")


def _count_edit_markers(model, prompt, seeds, **generation_overrides) -> int:
    marker_map = model.marker_map
    total = 0
    for seed in seeds:
        out = _generate(model, prompt, seed, max_new_tokens=80, **generation_overrides)
        tokens = out.sampled_tokens[0].tolist()
        total += sum(1 for t in tokens if t == marker_map["edit"])
    return total


def _matches_across_seeds(model, prompt, seeds, left_overrides, right_overrides) -> int:
    """Count seeds where two generation runs produce bit-identical raw token streams."""
    matches = 0
    for seed in seeds:
        left = _generate(model, prompt, seed, max_new_tokens=80, **left_overrides)
        right = _generate(model, prompt, seed, max_new_tokens=80, **right_overrides)
        if torch.equal(left.sampled_tokens, right.sampled_tokens):
            matches += 1
    return matches


def test_calibration_bias_suppresses_calibration_boost():
    prompt = torch.randint(0, 199, (1, 30))
    model = _make_model()
    seeds = range(60)
    counts = {
        d: _count_edit_markers(model, prompt, seeds, marker_calibration_weight=1.0, calibration_bias=d)
        for d in (0.0, -0.5, -1.0, 0.5, 1.0)
    }
    assert counts[1.0] >= counts[0.5] >= counts[0.0] >= counts[-0.5] >= counts[-1.0], counts


def test_max_delete_span_none_is_unbounded():
    """`max_delete_span=None` must leave deletion length unbounded."""
    prompt = torch.randint(0, 199, (1, 30))
    model = _make_model()
    deletes = []
    for seed in range(80):
        out = _generate(model, prompt, seed, max_delete_span=None, exclude_prompt_from_edits=False)
        deletes += _resolved_num_deletes(model, out.sampled_tokens[0].tolist(), out.sampled_indices[0].tolist())
    assert deletes, "test didn't exercise any edits at all"
    assert max(deletes) > 5, "expected at least one large, uncapped deletion to make the other assertions meaningful"


def test_max_delete_span_caps_exactly():
    """`max_delete_span=N` must cap every generated deletion at exactly `N`."""
    prompt = torch.randint(0, 199, (1, 30))
    model = _make_model()
    for cap in (0, 1, 2, 3, 5):
        deletes = []
        for seed in range(300):
            out = _generate(model, prompt, seed, max_delete_span=cap, exclude_prompt_from_edits=False)
            deletes += _resolved_num_deletes(model, out.sampled_tokens[0].tolist(), out.sampled_indices[0].tolist())
        assert deletes, f"cap={cap}: test didn't exercise any edits at all"
        assert min(deletes) >= 0, f"cap={cap}: got a negative deletion length {min(deletes)}"
        assert max(deletes) == cap, f"cap={cap}: expected max deletion exactly {cap}, got {max(deletes)}"


def test_greedy_mid_edit_forces_argmax_only_inside_open_edits():
    prompt = torch.randint(0, 199, (1, 20))
    model = _make_model()
    marker_map = model.marker_map
    special = {marker_map["edit"], marker_map["return_to_end"], marker_map["eos"]}

    matches = mismatches = non_mid_edit_seen = 0
    for seed in range(20):
        out = _generate(model, prompt, seed, max_new_tokens=60, output_scores=True, greedy_mid_edit=True)
        tokens = out.sampled_tokens[0].tolist()
        scores = out.scores
        prompt_len = prompt.shape[1]

        is_mid = False
        for step in range(len(scores)):
            tok = tokens[prompt_len + step]
            if is_mid and tok not in special:
                argmax_tok = int(scores[step][0].argmax())
                if tok == argmax_tok:
                    matches += 1
                else:
                    mismatches += 1
            elif not is_mid and tok not in special:
                non_mid_edit_seen += 1
            if tok == marker_map["edit"]:
                is_mid = True
            elif tok in (marker_map["return_to_end"], marker_map["eos"]):
                is_mid = False

    assert mismatches == 0, f"{mismatches} mid-edit content tokens were not argmax"
    assert matches > 0, "test didn't exercise any mid-edit content tokens at all"
    assert non_mid_edit_seen > 0, "test didn't exercise any ordinary (non-mid-edit) content tokens at all"


def test_greedy_mid_edit_defaults_on():
    """`greedy_mid_edit` defaults to `True`: mid-edit content tokens must already be argmax by default."""
    prompt = torch.randint(0, 199, (1, 20))
    model = _make_model()
    marker_map = model.marker_map
    special = {marker_map["edit"], marker_map["return_to_end"], marker_map["eos"]}

    matches = mismatches = 0
    for seed in range(20):
        out = _generate(model, prompt, seed, max_new_tokens=60, output_scores=True)  # greedy_mid_edit omitted -> default
        tokens = out.sampled_tokens[0].tolist()
        scores = out.scores
        prompt_len = prompt.shape[1]

        is_mid = False
        for step in range(len(scores)):
            tok = tokens[prompt_len + step]
            if is_mid and tok not in special:
                if tok == int(scores[step][0].argmax()):
                    matches += 1
                else:
                    mismatches += 1
            if tok == marker_map["edit"]:
                is_mid = True
            elif tok in (marker_map["return_to_end"], marker_map["eos"]):
                is_mid = False

    assert mismatches == 0, f"{mismatches} mid-edit content tokens were not argmax with greedy_mid_edit left at its default"
    assert matches > 0, "test didn't exercise any mid-edit content tokens at all"


def test_exclude_prompt_from_edits_defaults_on():
    """`exclude_prompt_from_edits` defaults to `True`: every edit's start cursor must land at or past the prompt by default."""
    prompt = torch.randint(0, 199, (1, 30))
    prompt_len = prompt.shape[1]
    model = _make_model()
    marker_map = model.marker_map
    edits_seen = 0
    for seed in range(40):
        out = _generate(model, prompt, seed, max_new_tokens=60)  # exclude_prompt_from_edits omitted -> default
        tokens = out.sampled_tokens[0].tolist()
        indices = out.sampled_indices[0].tolist()
        for step, tok in enumerate(tokens):
            if tok == marker_map["edit"]:
                edits_seen += 1
                assert indices[step][0] >= prompt_len, (
                    f"seed={seed}, step={step}: edit start cursor {indices[step][0]} pointed "
                    f"into the locked prompt (len {prompt_len}) with exclude_prompt_from_edits "
                    "left at its default"
                )
    assert edits_seen > 0, "test didn't exercise any edits at all"


def test_exclude_prompt_from_edits_false_allows_cursor_into_prompt():
    prompt = torch.randint(0, 199, (1, 30))
    prompt_len = prompt.shape[1]
    model = _make_model()
    marker_map = model.marker_map
    starts_in_prompt = 0
    for seed in range(40):
        out = _generate(model, prompt, seed, max_new_tokens=60, exclude_prompt_from_edits=False)
        tokens = out.sampled_tokens[0].tolist()
        indices = out.sampled_indices[0].tolist()
        for step, tok in enumerate(tokens):
            if tok == marker_map["edit"] and indices[step][0] < prompt_len:
                starts_in_prompt += 1
    assert starts_in_prompt > 0, "expected at least one edit to start inside the prompt with exclude_prompt_from_edits=False"


if __name__ == "__main__":
    test_generation_config_knobs_do_not_affect_model_config()
    test_calibration_bias_suppresses_calibration_boost()
    test_max_delete_span_none_is_unbounded()
    test_max_delete_span_caps_exactly()
    test_greedy_mid_edit_forces_argmax_only_inside_open_edits()
    test_greedy_mid_edit_defaults_on()
    test_exclude_prompt_from_edits_defaults_on()
    test_exclude_prompt_from_edits_false_allows_cursor_into_prompt()
