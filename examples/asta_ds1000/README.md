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

```bash
# OpenAI: powers the solver model (gpt-5.4-mini default). DS-1000 has
# no judge LLM, so this is the only credential needed.
export OPENAI_API_KEY="sk-..."
```

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

Default model: `openai/gpt-5.4-mini`. Default per-example agent cost cap: `$0.06` (score multiplied by 0.9 if breached at training time).

```bash
# Override model:
python examples/asta_ds1000/main.py --model openai/gpt-5

# Override cost cap:
python examples/asta_ds1000/main.py --cost-budget 0.02

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

### Cost-penalty asymmetry

The training evaluator applies `score *= 0.9` if agent spend exceeds `cost_budget`. The test evaluator (derived via `with_overrides(apply_cost_penalty=False)` in `main.py`) does not. The leaderboard test number is therefore raw 0/1 accuracy regardless of breach.

### Subprocess isolation

Every evaluation runs in its own Python subprocess (`_eval_worker.py`). Inspect-AI's `inspect.eval()` raises if two calls are in flight in the same process — subprocess isolation gives us real parallelism across RoboPhD's worker threads. Each subprocess pays ~7s of cold imports; at ~30–60s/eval that's ~10–20% overhead, acceptable for the parallelism gain.

## Cost notes

DS-1000 is **much cheaper per evaluation than DiscoveryBench**:

- No judge LLM. Programmatic scoring only.
- Single LLM call per problem in the seed agent (evolution may add more).
- Sandbox runs the test program (no GPU; CPU torch/tensorflow only).

Rough budget for a full training run + experiment-phase test:

- 300 training evals × ~$0.005/eval = ~$1.50 (well under the $0.06 cap)
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
