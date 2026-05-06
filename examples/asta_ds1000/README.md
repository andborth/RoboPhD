# DS-1000 (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's DS-1000 task (Code & Execution category, Standard tools tier). 100-sample validation split for training, 900-sample test split for the leaderboard metric.

For the current state of the art and the bar we're aiming at, see the live leaderboard: https://huggingface.co/spaces/allenai/asta-bench-leaderboard (DS-1000 tab). We deliberately don't quote specific score targets in `background.md` to avoid anchoring evolution on a number; this README has the engineer-facing context.

## Setup

### 1. Install the Python dependencies

```bash
# From the repo root
pip install -r requirements.txt
pip install -r examples/asta_ds1000/requirements.txt
```

### 2. Install Docker

Docker is **required**. The `python_session` tool runs Python inside a Docker container — this is what AstaBench's leaderboard runtime uses, so we keep parity. macOS options:

- **Docker Desktop** — https://www.docker.com/products/docker-desktop. GUI-based, free for personal use, easiest path.
- **colima** — `brew install colima docker docker-compose` then `colima start`.
- **OrbStack** — https://orbstack.dev. Fast on Apple Silicon, free for personal use.

Verify:
```bash
docker info  # should print daemon info without error
```

On the **first** evaluator run, AstaBench's image is pulled (~2–2.5 GB; one-time, ~30s–2min depending on connection). If you also run `examples/asta_discoverybench`, the image is shared.

### 3. Credentials

DS-1000 evolution may pick any of six solver models, paired by family into a cheap/fast tier and a stronger/slower tier:

- OpenAI: GPT-5.4 Mini / GPT-5.4 (full)
- Anthropic: Claude Haiku 4.5 / Claude Sonnet 4.6
- Google: Gemini 3.1 Flash Lite Preview / Gemini 3 Flash Preview

All three provider keys must be set before running, even if your seed only uses one — evolution can produce an agent that uses any of the six at any iteration, and a 401 mid-run is the worst time to discover a missing key.

```bash
# OpenAI — seed model (gpt-5.4-mini); evolution may also pick gpt-5.4 (full)
export OPENAI_API_KEY="sk-..."

# Anthropic — Haiku 4.5 / Sonnet 4.6. Prefer ANTHROPIC_API_KEY_FOR_ROBOPHD
# so your Claude Code CLI sessions (which read ANTHROPIC_API_KEY) keep using
# their normal subscription credentials. Either env var works.
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."

# Google — Gemini 3.1 Flash Lite Preview / Flash Image Preview
export GOOGLE_API_KEY="..."
```

The Anthropic provider validates its key at `model_registry` import time (asymmetric vs OpenAI/Google, which validate lazily on first `.generate()`), so importing the registry hard-fails if the Anthropic key is missing.

No `HF_ACCESS_TOKEN` needed — DS-1000 uses the public `xlangai/DS-1000` HuggingFace dataset.

Verify the dataset half (run from the repo root):
```bash
python -c "import sys; sys.path.insert(0, 'examples/asta_ds1000'); \
           from evaluator import load_ds1000; \
           print(len(load_ds1000('validation')), 'validation samples')"
# expect: 100 validation samples
```

The example's `load_ds1000` helper sets `INSPECT_EVAL_MODEL` defensively before hitting the AstaBench wrapper — the upstream `inspect_evals.ds1000.ds1000()` task constructor calls `get_model()` to pick a system message, which would otherwise raise without an env var or `--model` flag.

## Dataset

The 100 / 900 validation/test split is fixed and cached in `astabench/evals/inspect_eval_wrappers/ds1000_splits.json` (a one-time random shuffle with `seed=0` of the canonical `inspect_evals.ds1000` dataset). Don't expect the validation set to be a representative sub-sample of the test set — it's a uniform random pick.

| Phase | Train pool | Test set | Iter | Examples/iter | Total evals |
| --- | --- | --- | --- | --- | --- |
| **experiment** | full 100-sample `ds1000_validation` | 90-sample fixed sub-sample (~10%) of `ds1000_test` | 15 | 20 | 300 |
| **final** | same 100 | all **900** `ds1000_test` (leaderboard metric) | 15 | 20 | 300 |

### Sampling

Test sub-sampling for the experiment phase is **deterministic** — driven by a hardcoded `SPLIT_SEED = 42`, independent of `--random-seed`. So `--phase experiment` always tests against the same 90 samples regardless of what other flags you pass.

`--random-seed` (default `None` → fresh seed each run) only controls **RoboPhD-internal** RNG: which examples get drawn from the train pool each iteration, ELO matchup pairing, etc.

## Running

```bash
# Default: phase=experiment, full 100-sample validation train, 90-sample held-out test.
python examples/asta_ds1000/main.py --eval-test-set

# Final: train on the same 100, evaluate against all 900 ds1000_test.
python examples/asta_ds1000/main.py --phase final --eval-test-set

# Re-evaluate a prior run's best agent on the experiment-phase test set:
python examples/asta_ds1000/main.py --eval-only --resume <prior-run-dir>

# Re-evaluate against the full ds1000_test (writes to test_results_final.json):
python examples/asta_ds1000/main.py --eval-only --resume <prior-run-dir> --phase final
```

Default models: six handles in `model_registry.py`, paired by family into a mini/standard tier (the seed picks GPT-5.4 Mini; evolution may pick any of the six per call).

```bash
# Tighten the cost-penalty endpoints (ablation):
python examples/asta_ds1000/main.py --cost-threshold 0.005 --cost-saturation 0.5

# Smaller per-iteration sample for cheaper iteration:
python examples/asta_ds1000/main.py --examples-per-iteration 5

# Other engines (RoboPhD is the focus; GEPA / Autoresearch should work
# conceptually since the evaluator and seed are engine-agnostic, but
# weren't tuned for this task):
python examples/asta_ds1000/main.py --engine gepa
```

## Scoring details

The canonical scorer is `inspect_evals/ds1000/ds1000.py:86-114` (`ds1000_scorer()`). Per-sample it returns:

- `"C"` — the appended code makes `result` match the reference under all hidden test inputs (and any pattern checks for the problem); we map to `1.0`.
- `"I"` — failure for any reason (test exec error, mismatch, pattern check fail, sandbox timeout); we map to `0.0`.

**No partial credit.** The leaderboard's 0–1 numbers come from `accuracy()` averaging across the test set.

The scorer reads `state.metadata["code_context"]` to construct the test program — see "Architectural notes" below for why this matters.

## Architectural notes

### Metadata scrubbing

The canonical DS-1000 Sample carries three keys in `state.metadata`:

- `code_context` — the test harness AND the reference implementation. The scorer needs this. The agent must not see it (it's literally the answer).
- `perturbation_type` — `"Origin"` (a memorizable Stack Overflow original) vs `"Surface"`/`"Semantic"` (perturbed variants). Leaks training-set membership; an agent could route to a memorized cache when it sees `"Origin"`.
- `library` — fair signal (it's also obvious from the prompt).

Our `evaluator.py` wraps the candidate's solver in a higher-order function (`_wrap_with_metadata_scrub`) that pops `code_context` and `perturbation_type` from `state.metadata` before delegating, then restores them in a `finally` block so the scorer (which runs after the solver) sees the canonical metadata.

This is deliberate — we enforce the no-leakage invariant in code rather than rely on `background.md` warnings, because warning the evolution AI off a key inadvertently advertises that interesting content lives there. See the `feedback_enforce_dont_describe` memory for the principle.

### Cost-penalty math

During training the per-example score is:

```
score = 100 · raw_score − cost_penalty
cost_penalty = clip((agent_spend − $0.01) / ($1.00 − $0.01), 0, 1)
```

`raw_score` is the binary 0/1 from the canonical scorer. The penalty's `[0, 1]` range is two orders of magnitude smaller than the score axis, so it acts as a tiebreaker between correctness-tied agents — it never reorders agents whose correctness differs. Worked examples at leaderboard reference points:

| Per-eval spend | Cost penalty |
| --- | --- |
| $0.001 (cheap seed) | 0.000 (free zone) |
| $0.02 (leaderboard floor) | 0.010 |
| $0.04 (median competitive) | 0.030 |
| $0.10 | 0.091 |
| $0.25 (leaderboard top) | 0.242 |
| $1.00 | 1.000 (saturated) |

Per-problem `result.json` carries `raw_score` (the 0/1 outcome before any penalty) and `cost_penalty` (the [0,1] value subtracted) so failure analysis can distinguish "wrong answer" from "correct but expensive" without back-deriving the formula.

### Cost-penalty asymmetry

The training evaluator applies the formula above. The test evaluator (derived via `with_overrides(apply_cost_penalty=False)` in `main.py`) does not — test scores are raw 0/1 in `[0, 1]` for leaderboard parity. The leaderboard test number is therefore raw accuracy regardless of training-time spend.

### Model registry

Six pre-resolved Inspect-AI Model handles live in `model_registry.py` (outside the candidate's `file_mapping`, which only contains `agent.py`), paired by family:

- OpenAI: `GPT_5_4_MINI`, `GPT_5_4`
- Anthropic: `CLAUDE_HAIKU_4_5`, `CLAUDE_SONNET_4_6`
- Google: `GEMINI_3_1_FLASH_LITE_PREVIEW`, `GEMINI_3_FLASH_PREVIEW`

Pairing same-family handles by tier (mini/standard) lets evolution choose between cheap-and-fast or stronger-and-slower per call. Evolved agents `from model_registry import` whichever they want and call `.generate()`; the underlying model strings are not part of the evolvable artifact, so evolution can't substitute an arbitrary provider/model. All three provider keys are required at startup — an agent produced at iteration 4 might use Claude Sonnet, and a 401 mid-run is a worse failure mode than a startup error.

Provider-prefix translation: Inspect-AI requires `google/...` to route Google models, but litellm prices them under `gemini/...`. `evaluator.py:_estimate_cost` normalizes at the cost-pricing boundary (translates `google/` → `gemini/`) so cost tracking works for the Gemini handles. Without this, Gemini calls would silently price as $0 — the once-per-process "model priced at $0 despite tokens" warning would fire as the symptom.

### Subprocess isolation

Every evaluation runs in its own Python subprocess (`_eval_worker.py`). Inspect-AI's `inspect.eval()` raises if two calls are in flight in the same process — subprocess isolation gives us real parallelism across RoboPhD's worker threads. Each subprocess pays ~7s of cold imports; at ~30–60s/eval that's ~10–20% overhead, acceptable for the parallelism gain.

## Cost notes

DS-1000 is **much cheaper per evaluation than DiscoveryBench**:

- No judge LLM. Programmatic scoring only.
- Single LLM call per problem in the seed agent (evolution may add more).
- Sandbox runs the test program (no GPU; CPU torch/tensorflow only).

Rough budget for a full training run + experiment-phase test:

- 300 training evals × ~$0.005/eval = ~$1.50 (well in the cost-penalty free zone)
- 90-sample experiment-phase test × ~$0.005 = ~$0.45
- Total: ~$2 for a complete experiment-phase run

A `--phase final` test sweep (900 samples) costs ~$5 on top.

## Files

- `main.py` — `optimize_anything()` entry point; `--phase {experiment,final}` swaps the test set
- `evaluator.py` — `Ds1000Evaluator`; runs `inspect.eval()` on a 1-sample dataset per evaluation, attaches Docker sandbox + `python_session`, scrubs leak-prone metadata
- `_eval_worker.py` — subprocess shim for parallel inspect.eval() calls
- `seeds/baseline/agent.py` — minimal `@solver` factory exported as `make_solver`. One-shot LLM call, wraps response in `<code>` tags. Scores low by design.
- `objective.md` — what evolution should optimize
- `background.md` — task spec for the evolution AI (audience-clean: no leaderboard scores, no scorer internals, no enforce-warnings)
- `requirements.txt` — astabench + inspect_evals

## Submission to the AstaBench leaderboard

Once `--phase final --eval-only --resume <best-dir>` produces a leaderboard number you're happy with:

1. Locate the `.eval` log files inside the run dir's evaluator scratch space (the temp `ds1000_eval_*` dir created by the test evaluator).
2. Bundle them into a `tar.gz` along with an `eval_config.json` describing the agent.
3. Upload via the [AstaBench leaderboard HF Space](https://huggingface.co/spaces/allenai/asta-bench-leaderboard) and fill in the metadata (agent name, openness, tool-usage category).

See the AstaBench README at https://github.com/allenai/asta-bench for the canonical submission flow.
