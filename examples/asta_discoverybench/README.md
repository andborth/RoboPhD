# DiscoveryBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's DiscoveryBench task (Data Analysis category, Standard tools tier). Real validation = 25 samples, real test = 239 samples. The DiscoveryBench paper ships an additional public **synth/** split (903 samples total; **703 are scoreable** — `synth/train`=550 and `synth/dev`=153; `synth/test`=200 has gold withheld as an upstream held-out competition set) which we use for distribution-padding in some training regimes.

DiscoveryBench paper baselines: ReAct + GPT-4o = 15.4% HMS; Reflexion (oracle) + GPT-4o = 24.5% HMS. The Standard-tier ceiling per paper is in the low-to-mid 20s; we target ~20–22% HMS on real/test.

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

- Per-example judge cost is fixed at ~$0.029 (5 fixed gpt-4o-2024-08-06 calls per sample). This is evaluator overhead, **not** counted against the agent's $0.10 cap.
- A full real/test sweep (239 samples) costs ~$7 just in judge tokens.
- Regime 2 phase B at 750 evals + real/test sweep ≈ $30–$60 total ($25 judge across the run + $7 final test + agent's own LLM spend at GPT-5 Mini rates).
- Wall-clock dominates: 4–10s sandbox warm-start per sample + agent execution time. A 750-eval run with `--max-workers 4` is ~30–90 minutes wall-clock.

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

- [ ] **End-to-end smoke test with Docker installed.** Currently blocked on Docker setup; the evaluator pre-flights `docker info` and fails fast with a clear message until that's done.
- [ ] **Verify `python_session` calling convention** — first real sample run will confirm whether `await py(code=...)` returns a string and behaves statefully across calls within a sample.
- [ ] **Standard Tools allowlist (AST scan).** Currently absent (same gap as `asta_paper_finder`); evolution could in principle import outside the allowed set.
- [ ] **Cost-cap penalty observation.** Verify in a real run that an artificially-inflated agent triggers `cost_breached: True` and `score *= 0.9`.
- [ ] **HMS variance characterization.** The judge has no temperature controls and `num_retries=1`. Worth a 2× replicate over 5 samples to measure run-to-run variance before trusting individual numbers.
