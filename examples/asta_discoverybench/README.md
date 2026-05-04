# DiscoveryBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's DiscoveryBench task (Data Analysis category, Standard tools tier). Real validation = 25 samples, real test = 239 samples. The DiscoveryBench paper ships an additional public **synth/** split (903 samples total; **703 are scoreable** — `synth/train`=550 and `synth/dev`=153; `synth/test`=200 has gold withheld as an upstream held-out competition set) which we use for distribution-padding in some training regimes.

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

# OpenAI: powers both the solver model (gpt-5-mini default) and the
# scorer's gpt-4o-2024-08-06 judge. Must be set.
export OPENAI_API_KEY="sk-..."
```

`ASTA_TOOL_KEY` is **not** required for DiscoveryBench (no Asta MCP tools).

Verify the dataset half:
```bash
python -c "from astabench.evals.discoverybench.task import load_discoverybench_hf; \
           print(len(load_discoverybench_hf('validation')), 'validation samples')"
# expect: 25 validation samples
```

The synth split is fetched from the public `allenai/discoverybench` GitHub repo on first use (`load_synth.py` does a shallow git clone into `~/.cache/robophd/discoverybench_synth/`).

## Training regimes

Three regimes are wired in (`--regime {1,2,3}`). They differ in what the agent gets to train on, how much budget evolution gets, and what test set we report against. Pick based on what question you're trying to answer.

| Regime | Train pool | Test set | Iter | Examples/iter | Eval budget | Per-ex reuse |
| --- | --- | --- | --- | --- | --- | --- |
| **1** synth-only | synth/train (550) | synth/dev (153) + real/val (25) + real/test (239) | ~19 | 20 | 1500 | 0.54× |
| **2A** mixed, experiment | 85 synth + 15 real = 100 | 10 held-out real | ~19 | 10 | 750 | 1.9× |
| **2B** mixed, final | 85 synth + all 25 real = 110 | real/test (239) | ~19 | 10 | 750 | 1.7× |
| **3A** real-only, experiment | 15 of real/val | 10 held-out real | 15 | 3 | iter-bounded | 3.0× |
| **3B** real-only, final | all 25 real/val | real/test (239) | 15 | 3 | iter-bounded | 1.8× |

### When to use which

**Regime 1** — answer "does evolution learn anything in-distribution from a large synth pool, and does it transfer at all to real?" Tests broadly: in-distribution synth/dev plus the cross-distribution real splits. Cheapest per-eval, but scores against synth aren't directly comparable to the leaderboard.

**Regime 2** — the main "real" run. The 85:15 synth-to-real ratio gives evolution distributional padding (so a 25-real-sample pool doesn't memorize) while keeping the gold real-distribution exposure stable. Phase A uses 15 random real for training and the other 10 as held-out (cheap experimentation, 100 example pool); Phase B uses all 25 real for training and reports against the leaderboard-comparable real/test (239 samples).

**Regime 3** — the "no synth at all, just see what 25 examples can do" baseline. Phase A is for sanity-checking the loop on tiny pools; Phase B is the leaderboard-comparable real-only number. Reuse is high (3.0× / 1.8×) so overfit risk is real, but it isolates the real-distribution learning signal.

### Sampling and `--random-seed`

Two independent RNGs both seeded from `--random-seed` (default 0) drive the random sampling:

| RNG | Used in | Draws |
| --- | --- | --- |
| `real_rng` | Regime 2A, Regime 3A | 15-train / 10-held-out split of real/validation |
| `synth_rng` | Regime 2A, Regime 2B | 85-sample subset of synth/train |

Decoupling the two RNGs means each draw is a pure function of the seed alone, so the following invariants all hold simultaneously:

| Invariant | At seed=0 | At any other seed |
| --- | --- | --- |
| Two runs of the same regime+phase → same draws | ✓ | ✓ |
| **Regime 2A held-out 10 == Regime 3A held-out 10** | ✓ | ✓ |
| **Regime 2A 85-synth subset == Regime 2B 85-synth subset** | ✓ | ✓ |

The first invariant means determinism within a regime. The second means **evolved-agent comparisons across regime 2A and regime 3A are meaningful** — you can run the same agent through both regimes' `--eval-test-set` paths and compare scores on the same 10 held-out samples. The third means the 2A → 2B "experiment then final" workflow sees a stable synth pool.

#### Rotating seeds for replication

Passing a different `--random-seed` rotates **both** the real-split (in 2A/3A) and the synth subset (in 2). This is the right way to replicate or stress-test results across multiple seeds:

```bash
# Three seeds, each rotates everything:
python examples/asta_discoverybench/main.py --regime 3 --phase experiment --random-seed 0
python examples/asta_discoverybench/main.py --regime 3 --phase experiment --random-seed 1
python examples/asta_discoverybench/main.py --regime 3 --phase experiment --random-seed 2
```

To re-evaluate a specific candidate against the same 10 held-out samples a prior run used, pass the same `--random-seed` along with `--eval-agent <name> --eval-only --resume <prior-run-dir>`.

### A note on synth/test

`synth/test` (200 samples) is upstream's held-out competition set with `true_hypothesis` removed — it can't be scored locally. `load_synth("test")` raises rather than returning empty. Use `synth/dev` (153 scoreable) when you want a held-out synth signal.

## Running

Three regimes (`--regime`):

```bash
# Regime 1 — synth-only, 1500 evals, 20 examples/iter.
# Test broadly: synth/test + real/train + real/test.
python examples/asta_discoverybench/main.py --regime 1 --eval-test-set

# Regime 2 — mixed (85% synth, 15% real), 750 evals, 10 examples/iter.
# Phase A: experimentation against held-out real subset.
python examples/asta_discoverybench/main.py --regime 2 --phase experiment
# Phase B: final, evaluating on real/test.
python examples/asta_discoverybench/main.py --regime 2 --phase final --eval-test-set

# Regime 3 — real-only, 15 fixed iterations × 3 examples/iter.
python examples/asta_discoverybench/main.py --regime 3 --phase experiment
python examples/asta_discoverybench/main.py --regime 3 --phase final --eval-test-set
```

Default model: `openai/gpt-5-mini`. Default per-example agent cost cap: `$0.10` (score multiplied by 0.9 if breached; judge cost excluded).

```bash
# Override model:
python examples/asta_discoverybench/main.py --regime 2 --model openai/gpt-5

# Override cost cap:
python examples/asta_discoverybench/main.py --regime 2 --cost-budget 0.20

# Other engines:
python examples/asta_discoverybench/main.py --regime 2 --engine gepa
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
- Regime 2 phase B at 750 evals + real/test sweep ≈ $30–$60 total ($25 judge across the run + $7 final test + agent's own LLM spend at GPT-5 Mini rates).
- Wall-clock: 4–10s sandbox warm-start per sample + agent execution + 5 judge calls ≈ ~50s/sample observed.

## Files

- `main.py` — `optimize_anything()` entry point with `--regime` selection
- `evaluator.py` — `DiscoveryBenchEvaluator`; runs `inspect.eval()` on a 1-sample dataset per evaluation, attaches Docker sandbox + `python_session`, splits cost into agent vs judge
- `load_synth.py` — fetches DiscoveryBench's public synth split from GitHub on first use, normalizes the column-metadata path to match real
- `seeds/baseline/agent.py` — minimal `@solver` factory exported as `make_solver`. Demonstrates file copy, stateful Python, Inspect-tracked LLM, JSON output. Scores near zero by design.
- `objective.md` — what evolution should optimize
- `background.md` — task spec, output schema, sandbox idioms, scoring breakdown, cost cap, Standard Tools constraint
- `requirements.txt` — astabench (which bundles inspect_ai)

## Status

### Done

- [x] Scaffold matches existing examples (main.py, evaluator.py, load_synth.py, seed, objective, background, requirements)
- [x] Real loader works via `astabench.evals.discoverybench.task.load_discoverybench_hf`
- [x] Synth loader (`load_synth.py`) shallow-clones the public repo on first use; scoreable counts: 550 train / 153 dev / 0 test (synth/test is upstream's held-out competition set with no gold; load_synth("test") returns 0 samples)
- [x] Three-regime CLI per the design doc
- [x] Cost-cap with judge filtered out

### Open

- [x] End-to-end smoke test (Regime 3A, 3 samples / 1 iteration; mean HMS 0.07; ~50s/sample wall-clock; total $0.04)
- [x] **Real parallelism via subprocess isolation.** Each evaluation runs in its own Python subprocess (`_eval_worker.py`), bypassing Inspect-AI's `eval_async` process-global singleton lock. `--max-workers 8` is the new default; verified 2× speedup on a 3-sample iteration vs the previous serialized lock. Each subprocess pays ~7s of cold imports (inspect-ai + astabench + torch); at ~80s/eval that's ~9% overhead, acceptable for the parallelism gain.
- [ ] **Standard Tools allowlist (AST scan).** Currently absent (same gap as `asta_paper_finder`); evolution could in principle import outside the allowed set.
- [ ] **Cost-cap penalty observation.** Verify in a real evolved-agent run that an over-budget agent triggers `cost_breached: True` and `score *= 0.9`.
- [ ] **HMS variance characterization.** The judge has no temperature controls and `num_retries=1`. Worth a 2× replicate over 5 samples to measure run-to-run variance before trusting individual numbers.
