# Evaluating AutoIndexer's edit capability

An experiment that measures how well a trained AutoIndexer checkpoint performs **insertions**,
**deletions**, and **substitutions** on held-out text: whether it finds the right place to edit,
whether it deletes the right amount, and whether it writes the right replacement.

```bash
# Full pipeline: teacher-forced + free-gen + paper tables + gallery + stats.json
bash scripts/edit_eval/run_evals.sh <checkpoint-run-name>

# Teacher-forced only (fast diagnostic)
PYTHONPATH=src python scripts/edit_eval/run.py \
    --checkpoint s3://<bucket>/<run>/final_checkpoint \
    --local-data-dir ~/data/cpt_datamix \
    --num-windows 256 --out-dir runs/edit_eval/tf

# Recompile tables from an existing run directory
PYTHONPATH=src:scripts python scripts/edit_eval/compile_results.py runs/edit_eval/<checkpoint-run-name>

PYTHONPATH=src python scripts/edit_eval/report.py runs/edit_eval/tf --plots
```

## 1. What the training objective actually asks for

Worth being precise about, because it determines what a fair evaluation looks like.

`batch_perturb_labels` takes a clean document and emits a **raw token stream** whose replay
(`AutoIndexerIdParser`) reconstructs that document exactly. The stream is a *noisy draft
interleaved with repairs*: the model appends tokens, then emits an `edit` marker, and the index
head picks two cursors — a **start** (where in the resolved sequence the edit begins) and an
**end** (how far it deletes) — after which the model writes the replacement tokens, emits
`return_to_end`, and resumes appending at the end.

Two details drive the whole design:

- **Deleted spans are uniform-random tokens.** Positions that a later edit removes are never
  assigned a label, so `AutoIndexerModelBase.forward`'s `sanitize_labels` fills them with random
  non-special token ids in the input and `-100` in the target. The model is therefore trained to
  spot *garbage* and replace it, not to spot subtly-wrong-but-fluent text. In-distribution
  corruption is random tokens; anything else is a generalization test (`--corruption transplant`).
- **The three operations are not symmetric.** A substitution or deletion leaves a visible garbage
  cue in the draft. A **pure insertion leaves no cue at all** — the draft is simply missing `k`
  tokens — so the model has to notice an *omission*. Expect insertion to be the hard case, and
  the metrics below to separate that from localization failure.
- **Pure insertions and deletions need `pure_edit_prob` to appear in training at all.**
  `sample_cursor_operations` draws `num_insert` and `num_delete` independently from
  `Poisson(mean_insert)` / `Poisson(mean_delete)` and only rejects the both-zero case, so
  `P(num_delete = 0) = e^-mean` — 4.5e-5 at mean 10, 6.7e-3 at mean 5. Without
  `pure_edit_prob > 0` essentially every trained edit is a substitution. **Check the
  checkpoint's `perturb.pure_edit_prob` (`report.py` records it in `config.json`): at 0, the
  insert and delete cells are generalization tests rather than in-distribution capability
  tests, and should be read as such.**

## 2. Case construction

Each case is a clean 100-token window `X` from the validation split plus **one** edit, expressed
as the operation chain

```
[Operation(num_extend=draft_len),
 Operation(num_insert=k, num_delete=d, start_rel_pos=s, end_rel_pos=s+d, num_extend=tail)]
```

fed to the real `batch_perturb_labels`. That yields the draft the model sees:

```
draft = X[:s] ++ corruption(d tokens) ++ X[s+k : s+k+lag]
```

| operation    | `num_insert` | `num_delete` | draft relative to `X`      |
|--------------|--------------|--------------|----------------------------|
| insert       | `k ~ Pois(mean_insert)` | 0 | `k` tokens **missing** at `s` |
| delete       | 0 | `d ~ Pois(mean_delete)` | `d` **extra** garbage tokens at `s` |
| substitute   | `k` | `k` | `X[s:s+k]` **replaced** by `k` garbage tokens |

Magnitudes are Poisson, conditioned on `>= 1` (mirroring `sample_cursor_operations`'s resample
loop), and set with `--mean-insert` / `--mean-delete` (default 10). Match these to the
checkpoint's own `perturb` block — `report.py` records it in `config.json` — or the edit-size
sweep will read as a failure when it is really an off-distribution probe. Substitution matches
the `inplace_edits` regime (`num_insert == num_delete`).

**`lag`** — the number of clean tokens the draft carries *past* the corruption before the edit
marker fires — is a first-class variable, sampled uniformly by default and pinnable with `--lag`.
It matters because training samples `start_rel_pos` uniformly over the sequence built so far, so
the model must localize an edit at any distance; short windows only exercise the short-lag end of
that distribution.

Two structural constraints, both inherent to the format rather than to this harness:

- An edit's start cursor must point at an existing draft token, so **pure insertion needs
  `lag >= 1`** — appending at the very end of the draft isn't an edit, it's an extend.
- Windows are 100 tokens against a 2048-token training `max_length`, so absolute numbers here
  describe the short-context end of the training distribution. Sweep `--window-len` to check.

## 3. Measurements

### Mode A — teacher-forced (one forward per case)

The scripted stream is scored in a single `decode` call through the same path
`forward` uses, so these numbers are directly comparable to training losses. Each case is paired
with a **clean control**: the identical window with no edit at all.

**Detection** — does the model *want* to edit?
`P(emit edit marker)` from the auxiliary `marker_head` at every draft prefix, using the same
`legal_marker_class` / `marker_probs` path as free generation (markers are not vocabulary tokens).
Reported at the scripted trigger point, as a max before/after the corruption, and against the
clean control at the matched position → a **paired win-rate** (chance = 0.5). Without the clean
control, a model that edits constantly would look like a good detector. Also reports
`marker_logit_margin` and `would_trigger`, which replay the real firing decision `_sample` makes
at generation time: an argmax over `marker_head`'s masked softmax (`marker_logit_mask`), with the
EDIT logit biased by the calibration term described below — not a comparison against a fixed
probability threshold.

**Localization** — the index head's two distributions at the edit marker:

| metric | meaning |
|---|---|
| `start_top1` / `top5` / `rank` | is the start cursor on the first corrupted token? |
| `start_err` | signed error in tokens (the error *distribution* matters more than the mean) |
| `start_nll` vs. `start_uniform_nll` | learned signal over a uniform-over-draft baseline |
| `end_top1`, `pred_num_delete` | does it delete the right *amount*? |
| `end_nll` vs. `end_uniform_nll` | learned signal over uniform-over-legal-keys |

Both heads score their candidates on content alone. The end head is *masked* to keys strictly
past the start cursor (`compute_end_index_mask`) — an end cursor at or before the start would
mean a negative deletion length — but carries no length prior: a `Poisson(mean_delete)`
log-prior used to be added on top and was removed, since it capped accuracy away from
`k ≈ mean_delete` and made pure insertion (gold `num_delete = 0`) cost ~7.4 nats of adverse
prior.

**Content** — given oracle cursors:
per-token NLL/accuracy of the inserted tokens against `X[s:s+k]`; `P(return marker)` from
`marker_head` at the correct stopping point (length control — and for pure deletion this *is* the
content metric, since a correct deletion emits `return_to_end` immediately); and post-edit tail
NLL, i.e. whether the model resumes coherently. Each is reported against `*_clean_ref` — the NLL
of the *same* tokens under a plain causal forward of the clean window — so the cost of the edit
detour is separated from the intrinsic difficulty of the text.

**Marker triggering is separate from text sampling.** `marker_head` decides whether each step
emits a marker — sampled (or argmax'd, matching `do_sample`) directly from its own masked softmax
over `MarkerType`, entirely outside the vocabulary distribution; temperature / top-p / top-k shape
only the ordinary text tokens. The edit rate is set by `--gen-marker-calibration-weight` /
`--gen-calibration-bias` (`marker_calibration_weight`/`calibration_bias` on `generation_config`),
which bias the EDIT logit by the sampled start index's log-probability before the marker decision
— there is no fixed probability threshold to tune. Text-quality decoding knobs no longer need
special-case exemptions.

### Mode B — free generation (`--free-gen`)

The model gets only the corrupted draft and generates freely. `replay_edit_stream` re-parses the
sampled stream into resolved-coordinate `(start, end, inserted)` records (verified to round-trip
exactly on scripted streams).

`AutoIndexerModelBase._sample` locks the cursor out of the prompt by default
(`generation_config.exclude_prompt_from_edits`, default `True`) — correct for instruction-tuned
chat generation, where the prompt is a turn the response must not edit, but wrong here: the
`draft` passed to `generate()` *is* the document to repair in place. `run.py` passes
`exclude_prompt_from_edits=False` to `free_gen.rollout`/`batch_rollout` by default for exactly
this reason; pass `--gen-exclude-prompt-from-edits` to restore the locked behavior (e.g. to sanity
check it against a chat-style checkpoint).

- `edit rate` within the budget, and `first_edit_delay`
- first edit's `start_correct` / `span_correct` / `span_iou` / `insert_correct`
- `repair_exact`, and `repair_gain = (dist_before - dist_after) / dist_before` using Levenshtein
  against `X[:restored_len]` — did the edit actually *improve* the sequence?
- **spurious-edit rate** on uncorrupted drafts (`--gen-clean-control`) — the precision control
- an outcome taxonomy: `no_edit`, `repaired`, `partial_repair`, `wrong_position`,
  `over_delete`, `under_delete`, `wrong_content`, `made_worse`

**Tune `--gen-marker-calibration-weight`/`--gen-calibration-bias` when you care about the trigger
rate.** Because markers bypass the vocab softmax, `top_p` and `top_k` no longer affect whether an
edit fires — only these two do (recorded in `config.json` alongside the checkpoint's training
config).

Also give the budget room: `--gen-max-new-tokens 32` truncates repairs that fire late, which
shows up as `wrong_content` on edits that were actually on track.

Mode B answers the real question; Mode A explains *which stage* fails when it does.

## 4. Sweeps

Run `run.py` once per cell and point `report.py` at each output dir. Grouped breakdowns by
operation, edit size, lag, and datamix source come for free in every report.

| axis | flag | question |
|---|---|---|
| operation | `--op-types` | insert vs. delete vs. substitute |
| skip no-op | `--skip-noop` | drop the `none` / clean-trajectory category (drafts with nothing to edit); with `--suite`, skips the `tf_none` and `gen_none` phases entirely |
| edit size | `--mean-insert` / `--mean-delete` (2, 5, 10, 20) | does it generalize off the trained Poisson mean? |
| lag | `--lag` (1, 5, 15, 40) | how far back can it reach? |
| window length | `--window-len` (100, 256, 512) | short-context penalty vs. the 2048 training length |
| corruption | `--corruption transplant` | garbage (in-distribution) vs. fluent-but-wrong text |
| domain | grouped by `source` | prose vs. code vs. math |

## 5. Caveats

- **Validation-split provenance.** `load_dataset_mix` holds out the first
  `num_val_samples_per_source` rows of each *streamed* source, so any prefix of the val split is
  also held out — but only if the source ordering matches the training run's. With
  `--local-data-dir`, ordering comes from `glob.glob` over the mirror, which is filesystem-order
  and not guaranteed stable across machines. Syncing the same S3 mirror the run used is the
  closest available reproduction; treat "unseen" as high-confidence, not proven. A memorized
  window would inflate the insertion-content metrics specifically.
- **`--attn-implementation autoindexer_eager`** is the default here: it materializes full
  attention weights (slower, more memory) but runs anywhere and is the reference implementation.
  Use `autoindexer_cutedsl` to match training exactly on supported hardware.
- Free generation runs at batch size 1 — AutoIndexer's prefill has no padding path, so prompts of
  differing lengths can't be batched.
