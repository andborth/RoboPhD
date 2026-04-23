# Text2SQL Example

Evolve agents that generate SQL from natural language questions on the [BIRD benchmark](https://bird-bench.github.io/).

This is a **two-artifact** domain: evolution controls both `agent.py` (SQL generation with `llm()` and `test_sql()` callables) and `analyze_db.py` (database schema analysis, runs as a subprocess).

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the BIRD dataset

```bash
bash benchmark_resources/download_bird.sh
```

This downloads the BIRD benchmark data (~2 GB) to `benchmark_resources/datasets/`.

### 3. Set API key

```bash
# Text2SQL's default eval model is haiku-4.5 (called via litellm inside evaluator.py).
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."
```

Evolution itself uses the Claude Code CLI (Claude Max auth) and does not need this key.
`evaluator.py` passes the resolved key directly to `litellm.completion(api_key=...)` rather than
exporting it, so it stays out of the process environment and does not affect any Claude Code CLI
subprocesses spawned by the evolution loop.

## Quick Start

```bash
# Smoke test (3 problems, 1 iteration)
python examples/text2sql/main.py \
  --num-iterations 1 --examples-per-iteration 3 --evaluation-budget 10

# Full run (default: 1500 evaluation budget, ~21 iterations)
python examples/text2sql/main.py

# With test-set evaluation on BIRD dev after optimization
python examples/text2sql/main.py --eval-test-set
```

## Resume and Extend

```bash
# Resume from checkpoint
python examples/text2sql/main.py \
  --resume ../robophd_runs/robophd/text2sql_20260408_120000

# Extend by 5 more iterations
python examples/text2sql/main.py \
  --resume ../robophd_runs/robophd/text2sql_20260408_120000 \
  --extend 5

# Evaluate best agent from a prior run on dev set
python examples/text2sql/main.py \
  --eval-only --resume ../robophd_runs/robophd/text2sql_20260408_120000
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--eval-model` | `haiku-4.5` | LLM for the agent's `llm()` callable |
| `--dataset` | `train-filtered` | Training dataset (`train-filtered`, `train`, `dev`) |
| `--cost-budget` | `0.10` | Per-problem cost budget ($) |
| `--max-test-sql-calls` | `5` | Max `test_sql()` calls per problem |
| `--evaluation-budget` | `1500` | Total evaluator calls (the real run limiter) |
| `--examples-per-iteration` | `20` | Problems sampled per iteration |
| `--max-workers` | `8` | Parallel evaluation workers |

## How It Works

Each evaluation has two phases:

1. **Phase 1 — Schema Analysis**: `analyze_db.py` runs as a subprocess against the SQLite database. Output is cached per `(code_hash, database)` pair so identical analysis code isn't re-run.

2. **Phase 2 — SQL Generation**: `agent.py`'s `solve()` function receives the analysis output, the question, and two callables:
   - `llm(prompt)` — call the eval LLM (default: haiku-4.5)
   - `test_sql(sql)` — execute SQL against the database (max 5 calls)

   The agent returns the final SQL string for scoring.

**Scoring**: `set(predicted_results) == set(ground_truth_results)` (BIRD methodology). Correct answers within cost budget score 1.0; over-budget correct answers score 0.9.
