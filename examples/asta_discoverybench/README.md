# DiscoveryBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's DiscoveryBench task (Data Analysis category, Standard tools tier). Real validation = 25 samples, real test = 239 samples. The DiscoveryBench paper ships an additional public **synth/** split (903 samples total; **703 are scoreable** — `synth/train`=550 and `synth/dev`=153; `synth/test`=200 has gold withheld as an upstream held-out competition set) which we use to distribution-pad the small real pool during training.

For the current state of the art and the bar we're aiming at, see the live leaderboard: https://huggingface.co/spaces/allenai/asta-bench-leaderboard (DiscoveryBench tab). Reference numbers there are in flux as agents are added; we deliberately don't quote specific HMS targets here or in `background.md` to avoid anchoring evolution on a number.

## Setup

This example has two non-trivial setup requirements: **Docker** and a few credentials. (Compare with `protein_go` which needs DIAMOND + ~2.5 GB of data, and `cant_be_late` which needs trace downloads.) Once these are set, no further data downloads are needed.

### 1. Install the Python dependencies

```bash
# From the repo root
pip install -r requirements.txt
pip install -r examples/asta_discoverybench/requirements.txt
```

### 2. Install Docker

Docker is **required**. The `python_session` tool runs Python inside a Docker container — this is what AstaBench's leaderboard runtime uses, so we keep parity. macOS options:

- **Docker Desktop** — https://www.docker.com/products/docker-desktop. GUI-based, free for personal use, easiest path.
- **colima** — `brew install colima docker docker-compose` then `colima start`. Lightweight, CLI-only.
- **OrbStack** — https://orbstack.dev. Fast on Apple Silicon, free for personal use.

Verify:
```bash
docker info  # should print daemon info without error
```

On the **first** evaluator run, AstaBench's image is pulled (~2–2.5 GB; one-time, ~30s–2min depending on connection).

### 3. Credentials

```bash
# Gated allenai/asta-bench HF dataset (real/ split metadata).
# Same token as the asta_paper_finder example.
export HF_ACCESS_TOKEN="hf_..."
export HF_TOKEN="hf_..."

# All three solver-provider keys are required at evaluator startup,
# even for seed-only smoke tests. Evolution can produce an agent that
# uses any of the three models at any iteration; failing loudly at
# startup beats discovering a missing key as a 401 mid-run.

# OpenAI: powers gpt-5.4-mini (one of three solver models) and the
# scorer's gpt-4o-2024-08-06 judge.
export OPENAI_API_KEY="sk-..."

# Anthropic: powers claude-haiku-4-5-20251001.
export ANTHROPIC_API_KEY="sk-ant-..."

# Google AI Studio: powers gemini-3.1-flash-lite-preview.
export GOOGLE_API_KEY="..."
```

`ASTA_TOOL_KEY` is **not** required for DiscoveryBench (no Asta MCP tools).

Verify the dataset half:
```bash
python -c "from astabench.evals.discoverybench.task import load_discoverybench_hf; \
           print(len(load_discoverybench_hf('validation')), 'validation samples')"
# expect: 25 validation samples
```

The synth split is fetched from the public `allenai/discoverybench` GitHub repo on first use (`load_synth.py` does a shallow git clone into `~/.cache/robophd/discoverybench_synth/`).

## Dataset

Single training configuration; `--phase` only varies the test set:

| Phase | Train pool | Test set | Iter | Examples/iter | Total evals |
| --- | --- | --- | --- | --- | --- |
| **experiment** | 175 synth + all 25 real/val = **200** | 24-sample fixed sub-sample (~10%) of real/test | 15 | 20 | 300 |
| **final** | same 200 | all **239** real/test (leaderboard metric) | 15 | 20 | 300 |
| **synth-holdout** | same 200 | **375** synth/train samples not in the train pool (550 − `--num-synth-train`) | 15 | 20 | 300 |

`--num-synth-train N` (default 175) overrides the synth count for ablations: `--num-synth-train 0` gives a real-only train pool (25 examples), `--num-synth-train 525` is closer to the full synth/train (550 scoreable).

### Sampling

Pool composition is **deterministic** — driven by a hardcoded `SPLIT_SEED = 42`, independent of `--random-seed`. Two independent `random.Random(SPLIT_SEED)` instances:

- `synth_rng` — picks the synth subset out of `synth/train`'s 550.
- `test_rng` — picks the experiment-phase 24-sample test subset out of `real/test`'s 239.

Independent RNGs mean changing `--num-synth-train` doesn't perturb the test sample selection (and vice versa). So `--phase experiment` always tests against the same 24 samples regardless of what other flags you pass — including `--random-seed`.

### `--random-seed`

`--random-seed` (default `None` → fresh seed each run, logged on startup as `🎲 Random seed: <N>` and persisted in the experiment dir's checkpoint) only controls **RoboPhD-internal** RNG: which examples get drawn from the train pool each iteration, ELO matchup pairing, evolution strategy random choices, etc. It does **not** affect the train/test pool composition.

### A note on synth/test

`synth/test` (200 samples) is upstream's held-out competition set with `true_hypothesis` removed — it can't be scored locally. `load_synth("test")` raises rather than returning empty. Use `synth/dev` (153 scoreable) if you want a held-out synth signal.

## Running

```bash
# Default: phase=experiment, 175 synth + 25 real train pool, 24-sample held-out test.
python examples/asta_discoverybench/main.py --eval-test-set

# Final: train on the same 200, evaluate against all 239 real/test.
python examples/asta_discoverybench/main.py --phase final --eval-test-set

# Re-evaluate a prior run's best agent on the experiment-phase test set:
python examples/asta_discoverybench/main.py --eval-only --resume <prior-run-dir>

# Re-evaluate against the full real/test (writes to test_results_final.json,
# distinct from test_results_experiment.json so they don't clobber):
python examples/asta_discoverybench/main.py --eval-only --resume <prior-run-dir> --phase final

# Synth in-distribution sanity check before committing to a real/test sweep:
python examples/asta_discoverybench/main.py --eval-only --resume <prior-run-dir> --phase synth-holdout
```

The seed agent calls `GPT_5_4_MINI` from `model_registry.py`. Evolved agents may pick from any of the three handles documented in `background.md` (`GPT_5_4_MINI`, `CLAUDE_HAIKU_4_5`, `GEMINI_3_1_FLASH_LITE_PREVIEW`). Default per-example agent cost cap: `$0.10` (score multiplied by 0.9 if breached; judge cost excluded).

```bash
# Override cost cap:
python examples/asta_discoverybench/main.py --cost-budget 0.20

# Synth-padding ablation (real-only train pool):
python examples/asta_discoverybench/main.py --num-synth-train 0

# Other engines:
python examples/asta_discoverybench/main.py --engine gepa
```

## Cost notes

- Per-example **agent** budget is $0.10. The agent's spend is capped only at training time; see "When the cap fires" below.
- Per-example **judge** cost is ~$0.015–0.020 (5 fixed gpt-4o-2024-08-06 calls per sample). This is evaluator overhead, **not** counted against the agent's cap.
- **Reports separate agent and judge cost.** `eval_cost` (in `result.json`, the "Eval" column of `cost_report.md` and `interim_report.md`) is **agent-only**. Judge spend goes into `other_cost` (the "Other" column, only shown when non-zero). The headline `Total` column sums all buckets — `Eval + Evo + Meta + Other` — so the run-level cost is honest, but evolution and meta-evolution see only `eval_cost` and aren't biased by the fixed judge overhead. The per-problem breakdown is in `agent_cost_usd`, `judge_cost_usd`, `cost_breached`, `cost_penalty_applied`.

### When the cap fires

The cap is a **training-time soft penalty**, not a test-time score modifier:

- **Training (RoboPhD ELO competition).** When `agent_cost_usd > $0.10`, the score for that ELO match is multiplied by 0.9. This nudges evolution toward cheaper agents — the soft penalty makes expensive runs lose head-to-head matches more often, even when their HMS would otherwise be slightly higher. Per-problem records show `cost_breached: true` and `cost_penalty_applied: true`.
- **Test (`--eval-only`, `--eval-agent`, `--eval-test-set`).** The reported HMS is **raw** — no penalty, regardless of breach. The agent's cost is recorded so it can be placed at its true point on the Pareto cost-vs-score curve, but cost does not modify the score. Per-problem records show `cost_breached: <whatever happened>` and `cost_penalty_applied: false` always.

The intent: evolution is guided by the soft penalty toward better cost discipline, but the headline number we report (and the leaderboard data point) is the raw HMS at whatever cost the evolved artifact actually incurs. Two separate evaluator instances inside `main.py` enforce the asymmetry: one with `apply_cost_penalty=True` for training, one with `apply_cost_penalty=False` for all test paths.
- A full real/test sweep (239 samples) costs ~$7 just in judge tokens.
- A 300-eval training run + final test sweep ≈ $15–$25 total ($6 judge across the 300 evals + $7 final test + agent's own LLM spend at GPT-5.4 Mini rates).
- Wall-clock: 4–10s sandbox warm-start per sample + agent execution + 5 judge calls ≈ ~50s/sample observed.

## Files

- `main.py` — `optimize_anything()` entry point; `--phase {experiment,final}` swaps the test set, `--num-synth-train N` controls synth padding
- `evaluator.py` — `DiscoveryBenchEvaluator`; runs `inspect.eval()` on a 1-sample dataset per evaluation, attaches Docker sandbox + `python_session`, splits cost into agent vs judge
- `model_registry.py` — pre-resolved `Model` handles (`GPT_5_4_MINI`, `CLAUDE_HAIKU_4_5`, `GEMINI_3_1_FLASH_LITE_PREVIEW`) imported by the seed and any evolved agent. Lives outside the candidate's `file_mapping` so evolution can use the handles but can't substitute different model strings.
- `load_synth.py` — fetches DiscoveryBench's public synth split from GitHub on first use, normalizes the column-metadata path to match real
- `seeds/baseline/agent.py` — minimal `@solver` factory exported as `make_solver`. Demonstrates file copy, stateful Python, Inspect-tracked LLM, JSON output. Scores near zero by design.
- `objective.md` — what evolution should optimize
- `background.md` — task spec, output schema, sandbox idioms, scoring breakdown, cost cap
- `requirements.txt` — astabench (which bundles inspect_ai)

## Status

### Done

- [x] Scaffold matches existing examples (main.py, evaluator.py, load_synth.py, seed, objective, background, requirements)
- [x] Real loader works via `astabench.evals.discoverybench.task.load_discoverybench_hf`
- [x] Synth loader (`load_synth.py`) shallow-clones the public repo on first use; scoreable counts: 550 train / 153 dev / 0 test (synth/test is upstream's held-out competition set with no gold; load_synth("test") returns 0 samples)
- [x] Single-configuration CLI: `--phase {experiment,final}` + `--num-synth-train N` (default 175)
- [x] Cost-cap with judge filtered out

### Open

- [x] End-to-end smoke test (3 samples / 1 iteration; mean HMS 0.07; ~50s/sample wall-clock; total $0.04)
- [x] **Real parallelism via subprocess isolation.** Each evaluation runs in its own Python subprocess (`_eval_worker.py`), bypassing Inspect-AI's `eval_async` process-global singleton lock. `--max-workers 12` is the default; verified 2× speedup on a 3-sample iteration vs the previous serialized lock. Each subprocess pays ~7s of cold imports (inspect-ai + astabench + torch); at ~80s/eval that's ~9% overhead, acceptable for the parallelism gain.
- [ ] **Solver-import allowlist (AST scan).** Currently absent (same gap as `asta_paper_finder`); evolution could in principle import outside the allowed set documented in `background.md`'s API surfaces section.
- [ ] **Cost-cap penalty observation.** Verify in a real evolved-agent run that an over-budget agent triggers `cost_breached: True` and `score *= 0.9`.
- [ ] **HMS variance characterization.** The judge has no temperature controls and `num_retries=1`. Worth a 2× replicate over 5 samples to measure run-to-run variance before trusting individual numbers.
