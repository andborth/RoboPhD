# DS-1000 (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's DS-1000 task (Code & Execution category, Standard tools tier). 100-sample validation split for training, 900-sample test split for the leaderboard metric.

For the current state of the art and the bar we're aiming at, see the live leaderboard: https://allenai-asta-bench-leaderboard.hf.space/code-execution. We deliberately don't quote specific score targets in `background.md` to avoid anchoring evolution on a number; this README has the engineer-facing context.

## Leaderboard submissions

Two RoboPhD-evolved agents are live on the AstaBench DS-1000 leaderboard. Both sit on the Pareto frontier:

| Submission | Frontier position | Accuracy | Cost / problem | In-repo snapshot |
|---|---|---|---|---|
| v0_0_1_soft_cap_0_16 | **#1 accuracy** on the leaderboard | 86.2% | $0.13 | [example_runs/robophd/asta_ds1000/v0_0_1_soft_cap_0_16/](../../example_runs/robophd/asta_ds1000/v0_0_1_soft_cap_0_16/) |
| v0_0_2_soft_cap_0_08 | Lowest-cost submission above 80% accuracy | 80.9% | $0.01 | [example_runs/robophd/asta_ds1000/v0_0_2_soft_cap_0_08/](../../example_runs/robophd/asta_ds1000/v0_0_2_soft_cap_0_08/) |

Each snapshot README documents the architecture, lineage, model usage, and submission resilience wrapper of its agent in detail.

## Setup

### 1. Install the Python dependencies

```bash
# From the repo root
pip install -r requirements.txt
pip install -r examples/asta_ds1000/requirements.txt

# Preparing a leaderboard submission? Also upgrade litellm (separate
# step — see the comment in examples/asta_ds1000/requirements.txt):
pip install litellm==1.88.1
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

On the **first** evaluator run, AstaBench's image is pulled (~2–2.5 GB; one-time, ~30s–2min depending on connection).

#### Docker prerequisites for reliable submission runs

The sandbox builds `FROM python:3.11-bookworm` (`build: .` in AstaBench's `util/sandbox/sandbox_compose.yaml`), so **every per-sample sandbox build resolves that tag against Docker Hub**. Over a 900-sample eval that's ~900 Hub auth contacts; transient `auth.docker.io … 404` blips occasionally fail a sandbox build mid-run (observed: 2 brief windows, 4/900 samples, in run -032). The two-tier seed-fallback wrapper catches these, so they rarely cost more than a problem or two — but to minimize them:

- **`docker login`** (free Docker Hub account) — the strongest lever: authenticated pulls get higher rate limits and stabler tokens, which is what reduces the intermittent auth 404s. This is per-machine secret state, so it can't be committed — do it once per machine.
- **Keep the base image cached.** `scripts/asta_ds1000_submit.py` pre-pulls it automatically (best-effort) before evaluating, but a cold cache still has to fetch once. To warm manually: `docker pull python:3.11-bookworm`.
- **Do NOT `docker system prune -af`.** It wipes the base image + build cache, forcing fresh Hub fetches on the next run (and re-bloating afterward). To relieve Docker storage, prune *scoped*: `docker container prune` and `docker image prune` (dangling only).

**OrbStack startup wedge.** Separately, OrbStack (2.2.1) has intermittently failed to start its Docker daemon (containerd `boltdb open` timeout → crash-loop). `orb start` / the GUI report "already running" and do nothing; the fix is a force-restart:
```bash
killall OrbStack; sleep 3; open -a OrbStack   # then wait ~10s and re-check `docker ps`
```
If it recurs, updating OrbStack past 2.2.1 is the likely durable fix (cause unconfirmed; disk/Time-Machine ruled out).

### 3. Credentials

DS-1000 evolution may pick any of nine solver models, grouped by family into cheap/fast, standard, and strong/slow tiers (see "Model registry" below):

- OpenAI: GPT-5.4 Mini / GPT-5.4 (full) / GPT-5.5
- Anthropic: Claude Haiku 4.5 / Claude Sonnet 4.6 / Claude Opus 4.8
- Google: Gemini 3.1 Flash Lite / Gemini 3.5 Flash / Gemini 3.1 Pro Preview

All three provider keys must be set before running, even if your seed only uses one — evolution can produce an agent that uses any of the nine at any iteration, and a 401 mid-run is the worst time to discover a missing key.

```bash
# OpenAI — seed model (gpt-5.4-mini); evolution may also pick gpt-5.4 (full)
export OPENAI_API_KEY="sk-..."

# Anthropic — Haiku 4.5 / Sonnet 4.6. Prefer ANTHROPIC_API_KEY_FOR_ROBOPHD
# so your Claude Code CLI sessions (which read ANTHROPIC_API_KEY) keep using
# their normal subscription credentials. Either env var works.
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."

# Google — Gemini 3.1 Flash Lite / Gemini 3 Flash Preview
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

| Phase | Train pool | Test set | Examples/iter | Evaluation budget |
| --- | --- | --- | --- | --- |
| **final** (default) | full 100-sample `ds1000_validation` | all **900** `ds1000_test` (leaderboard metric) | 20 | 750 (the binding limit) |
| **experiment** | same 100 | 90-sample fixed sub-sample (~10%) of `ds1000_test` | 20 | 750 (the binding limit) |

The run is bound by the **evaluation budget** (750 fresh evals — the same budget-bound regime as the standard RoboPhD tasks), not by an iteration count; the iteration cap (999) is loose and won't normally be reached. 750 is sized to the 100-sample train pool, past which extra budget mostly re-samples the same problems.

### Sampling

Test sub-sampling for the experiment phase is **deterministic** — driven by a hardcoded `SPLIT_SEED = 42`, independent of `--random-seed`. So `--phase experiment` always tests against the same 90 samples regardless of what other flags you pass.

`--random-seed` (default `None` → fresh seed each run) only controls **RoboPhD-internal** RNG: which examples get drawn from the train pool each iteration, Elo matchup pairing, etc.

## Running

```bash
# Default: phase=final — train on the 100-sample validation pool, evaluate
# against all 900 ds1000_test (the leaderboard metric).
python examples/asta_ds1000/main.py --eval-test-set

# Experiment: cheaper 90-sample held-out test (~10% of ds1000_test) for faster iteration.
python examples/asta_ds1000/main.py --phase experiment --eval-test-set

# Re-evaluate a prior run's best agent on the full ds1000_test (writes test_results_final.json):
python examples/asta_ds1000/main.py --eval-only --resume <prior-run-dir>

# Re-evaluate against the cheaper experiment-phase test set:
python examples/asta_ds1000/main.py --eval-only --resume <prior-run-dir> --phase experiment
```

Default models: nine handles in `model_registry.py`, grouped by family into mini / standard / stronger tiers (the seed picks GPT-5.4 Mini; evolution may pick any of the nine per call). All nine handles are always available; the cost-penalty disciplines overuse of the strong tier — see "Model registry" below for the cost shape.

```bash
# Tighten the cost-penalty endpoints (ablation): cheaper free zone +
# half-cent dollars-per-error makes each extra cent equal two errors.
python examples/asta_ds1000/main.py --cost-threshold 0.005 --cost-per-error 0.005

# Recover pure-tiebreaker semantics (cost sorts within accuracy ties
# but never overrides a one-problem accuracy gap):
python examples/asta_ds1000/main.py --cost-per-error 10

# Give the stronger tier more cost headroom (~5-10x default-tier price):
python examples/asta_ds1000/main.py --cost-threshold 0.08

# Smaller per-iteration sample for cheaper iteration:
python examples/asta_ds1000/main.py --examples-per-iteration 5

# Opt into Deep-Focus Round 2 (each new agent re-evaluated on a fresh
# batch within the same iteration). Doubles per-agent training
# exposure and switches objective.md to a Round-2-aware framing that
# nudges the agent away from overfitting to the visible batch:
python examples/asta_ds1000/main.py --new-agent-test-rounds 1

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

The iteration-aggregate score during training is:

```
errors_equivalent = max(0, mean_cost − $0.05) / $0.01
score = 100 · mean_accuracy − errors_equivalent · (100 / n)
```

where `mean_cost` is the batch's mean agent spend and `n` is the iteration batch size. One error-equivalent of penalty costs exactly one wrong answer of raw score, so the penalty lives in the agent's own currency (errors), not dollars. The free-zone width ($0.05) gives typical "cheap" leaderboard entries (~$0.02/problem) headroom to stay fully inside the free zone. Above the threshold, the penalty is **unbounded** — a catastrophically expensive agent can score well negative, which is intentional.

The two knobs are independently tunable. `--cost-threshold` widens the free zone; `--cost-per-error` chooses the regime:

| `--cost-per-error` | At iter7 mean cost ($0.13, $0.08 excess) | Crosses 1 error of penalty at | Behavior |
| --- | --- | --- | --- |
| $10 (≈ legacy default) | 0.04 pts | mean $10.05 | Pure tiebreaker — sorts ties, never overrides |
| $1 | 0.40 pts | mean $1.05 | Pure tiebreaker (slightly stronger sort) |
| $0.01 (default) | 40 pts | mean $0.06 | Active pull — penalty trades off against accuracy |

The per-iteration `aggregate_explanation` (in `evaluation.json`) carries the resolved excess and error-count so failure analysis can read "correct but expensive" off the page without back-deriving the formula.

### Cost-penalty asymmetry

The training evaluator applies the formula above. The test evaluator (derived via `with_overrides(apply_cost_penalty=False)` in `main.py`) does not — test scores are raw 0/1 in `[0, 1]` for leaderboard parity. The leaderboard test number is therefore raw accuracy regardless of training-time spend.

### Model registry

Nine pre-resolved Inspect-AI Model handles live in `model_registry.py` (outside the candidate's `file_mapping`, which only contains `agent.py`), grouped by family into three tiers:

- OpenAI: `GPT_5_4_MINI`, `GPT_5_4`, `GPT_5_5` ($5.00 / $30.00 per M tokens for the strong tier)
- Anthropic: `CLAUDE_HAIKU_4_5`, `CLAUDE_SONNET_4_6`, `CLAUDE_OPUS_4_8` ($5.00 / $25.00 per M tokens for the strong tier. Claude Fable 5 is intentionally excluded to keep runs comparable with the existing experimental campaign — no evolved agent ever selected it as a solver during its brief availability)
- Google: `GEMINI_3_1_FLASH_LITE`, `GEMINI_3_5_FLASH`, `GEMINI_3_1_PRO_PREVIEW` ($2.00 / $12.00 per M tokens for the strong tier)

Evolved agents `from model_registry import` whichever handles they want and call `.generate()`. The model strings live outside the evolvable artifact (`agent.py`), so evolution can't substitute an arbitrary provider/model. All three provider keys are required at startup — see "Credentials" above.

Strong-tier handles cost ~5–40× the cheap tier, so a single naive call can blow past the default $0.05 cost threshold and rack up many error-equivalents at the default `--cost-per-error 0.01`. That's the point: evolution must buy extra correctness with each expensive call. Raise `--cost-threshold` (e.g. `0.08`) for a wider free zone, or `--cost-per-error` to make the penalty a tiebreaker rather than an active pull. All three Gemini handles ship pinned to `reasoning_effort="low"` (the provider can't disable thinking), with `"high"` as the only opt-up.

Provider-prefix translation: Inspect-AI requires `google/...` to route Google models, but litellm prices them under `gemini/...`. `evaluator.py:_estimate_cost` normalizes at the cost-pricing boundary (translates `google/` → `gemini/`) so cost tracking works for the Gemini handles. Without this, Gemini calls would silently price as $0 — the once-per-process "model priced at $0 despite tokens" warning would fire as the symptom.

Billing basis: internal costs are priced from litellm's **bundled** price snapshot — the same table the official `astabench score` bills (agenteval runs under `LITELLM_LOCAL_MODEL_COST_MAP=True`) — not the live remote map, and Gemini reasoning ("thinking") tokens are folded into completion billing per agenteval's rule. The bundled snapshot can lag providers' current list prices (litellm 1.88.1 bills `gemini-3.1-flash-lite` at $0.45/$2.70 per M vs Google's current $0.25/$1.50); we deliberately track the leaderboard's numbers for comparability with other submitted systems, and move only when Ai2 moves (their bundled-map refresh would retroactively reprice historic runs — their call). The `background.md` model-menu table therefore advertises the *billed* rates, and `test_background_md_prices_match_litellm_registry` pins menu-vs-billed consistency on the same bundled-first lookup the evaluator uses. Full postmortem of the divergence this replaced: `example_runs/robophd/asta_ds1000/v0_0_6_soft_cap_0_003_fable/README.md`.

### Subprocess isolation

Every evaluation runs in its own Python subprocess (`_eval_worker.py`). Inspect-AI's `inspect.eval()` raises if two calls are in flight in the same process — subprocess isolation gives us real parallelism across RoboPhD's worker threads. Each subprocess pays ~7s of cold imports; at ~30–60s/eval that's ~10–20% overhead, acceptable for the parallelism gain.

`--max-workers` controls the parallel width. Resolution order: (1) the CLI flag if set, (2) on `--resume`, the value stored in the resumed run's `checkpoint.json` (so resume preserves the original run's settings), (3) the framework default of 10. Lower it (e.g. 4–6) if you see ENOSPC errors — each parallel eval takes an overlay snapshot of the inspect-ai sandbox's heavy ML base layer (torch/tensorflow/grpc), and wide parallelism can exhaust disk.

## Cost notes

Rough budget for a full training run + the default final-phase test:

- 750 training evals (the evaluation budget) × ~$0.005/eval = ~$3.75 (well in the cost-penalty free zone)
- 900-sample final-phase test × ~$0.005 = ~$4.50
- Total: ~$8 for a complete default (final-phase) run

Using `--phase experiment` swaps in the cheaper 90-sample test (~$0.45 instead of ~$4.50).

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
