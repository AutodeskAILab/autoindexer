"""Left-padded batched generation must be indistinguishable from running each row alone."""

import torch
from transformers import GenerationConfig

from autoindexer.models.autoindexer.config import AutoIndexerQwen3Config
from autoindexer.models.autoindexer.autoindexer import AutoIndexerQwen3Model
from autoindexer.models.autoindexer.tests.test_generation_guards import TINY_MODEL_KWARGS, _make_model


def _make_multilayer_model(seed: int = 123, num_hidden_layers: int = 3) -> AutoIndexerQwen3Model:
    """`_make_model`'s single decoder layer hides padding contamination in prefill's attention output; more layers propagate it into every later step's logits."""
    torch.manual_seed(seed)
    config = AutoIndexerQwen3Config(**{**TINY_MODEL_KWARGS, "num_hidden_layers": num_hidden_layers})
    config.use_cache = True
    model = AutoIndexerQwen3Model(config)
    model.eval()
    return model


def _greedy_config(model, max_new_tokens=100):
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.marker_map["eos"],
        pad_token_id=model.marker_map["eos"],
        return_dict_in_generate=True,
    )


def test_no_attention_mask_matches_explicit_all_ones():
    """Passing an all-`True` mask (the shape `batch_rollout` always builds) must be a no-op vs. the no-mask code path."""
    model = _make_model()
    prompt = torch.randint(0, 199, (1, 40))

    out_default = model.generate(prompt, generation_config=_greedy_config(model))
    out_explicit = model.generate(
        prompt, attention_mask=torch.ones_like(prompt, dtype=torch.bool), generation_config=_greedy_config(model)
    )
    assert torch.equal(out_default.sampled_tokens, out_explicit.sampled_tokens)
    assert torch.equal(out_default.sampled_indices, out_explicit.sampled_indices)


def _check_same_content_different_padding(model, draft, pad_len_a, pad_len_b, fill_a, fill_b):
    """Two rows sharing one real draft, left-padded to different amounts with different filler, must produce identical output once padding is sliced off."""
    max_len = len(draft) + max(pad_len_a, pad_len_b)
    rows = []
    pad_lens = (pad_len_a, pad_len_b)
    fills = (fill_a, fill_b)
    for pad_len, fill in zip(pad_lens, fills):
        lead = max_len - len(draft) - pad_len
        rows.append([fill] * (lead + pad_len) + draft if lead + pad_len else list(draft))
    # Every row must be exactly `max_len` long and marked real only over its own draft.
    batch = torch.zeros((2, max_len), dtype=torch.long)
    attention_mask = torch.zeros((2, max_len), dtype=torch.bool)
    for i, (pad_len, fill) in enumerate(zip(pad_lens, fills)):
        real_start = max_len - len(draft)
        batch[i, :real_start] = fill
        batch[i, real_start:] = torch.tensor(draft, dtype=torch.long)
        attention_mask[i, real_start:] = True

    out = model.generate(batch, attention_mask=attention_mask, generation_config=_greedy_config(model))
    real_start = max_len - len(draft)
    tokens0, tokens1 = out.sampled_tokens[0, real_start:], out.sampled_tokens[1, real_start:]
    indices0, indices1 = out.sampled_indices[0, real_start:], out.sampled_indices[1, real_start:]
    assert torch.equal(tokens0, tokens1), "same real draft, different padding -> tokens diverged"
    assert torch.equal(indices0, indices1), "same real draft, different padding -> indices diverged"

    solo = model.generate(torch.tensor([draft], dtype=torch.long), generation_config=_greedy_config(model))
    assert torch.equal(tokens0, solo.sampled_tokens[0]), "padded row doesn't match an unpadded solo run"


def test_left_padding_is_invisible_to_generation():
    model = _make_model()
    torch.manual_seed(11)
    draft = torch.randint(0, 199, (37,)).tolist()
    _check_same_content_different_padding(model, draft, pad_len_a=15, pad_len_b=15, fill_a=3, fill_b=198)


def test_left_padding_amount_does_not_matter_either():
    """Same real draft, but the two rows aren't even padded to the same length as each other in isolation -- still must be invisible."""
    model = _make_model()
    torch.manual_seed(23)
    draft = torch.randint(0, 199, (20,)).tolist()
    _check_same_content_different_padding(model, draft, pad_len_a=5, pad_len_b=31, fill_a=42, fill_b=0)


def test_left_padding_invisible_across_seeds_and_lengths():
    """Generalizes `test_left_padding_is_invisible_to_generation` across several random model/draft combinations."""
    for model_seed in (1, 2, 3, 4, 5):
        model = _make_model(seed=model_seed)
        for draft_len in (9, 24, 50):
            torch.manual_seed(model_seed * 1000 + draft_len)
            draft = torch.randint(0, 199, (draft_len,)).tolist()
            _check_same_content_different_padding(model, draft, pad_len_a=6, pad_len_b=13, fill_a=1, fill_b=150)


def test_multilayer_left_padding_is_invisible_to_generation():
    """Every test above uses `_make_model`'s single-decoder-layer config, which can't actually reach the primary token logits -- this multi-layer model can."""
    model = _make_multilayer_model()
    torch.manual_seed(11)
    draft = torch.randint(0, 199, (37,)).tolist()
    _check_same_content_different_padding(model, draft, pad_len_a=5, pad_len_b=31, fill_a=42, fill_b=0)


if __name__ == "__main__":
    test_no_attention_mask_matches_explicit_all_ones()
    test_left_padding_is_invisible_to_generation()
    test_left_padding_amount_does_not_matter_either()
    test_left_padding_invisible_across_seeds_and_lengths()
    test_multilayer_left_padding_is_invisible_to_generation()
    test_left_padding_invisible_across_seeds_and_lengths()
