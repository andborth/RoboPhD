# GEPA Integration Documentation

Documentation for the GEPA (`optimize_anything()`) integration with RoboPhD domains.

## Overview

[GEPA](https://github.com/gepa-ai/gepa) (Generalized Evolutionary Prompt Optimization with AI) is an external optimization framework that uses reflective text evolution with Pareto selection. RoboPhD provides adapter layers that let GEPA optimize agents on RoboPhD's domains.

GEPA is **not a domain** — it's an alternative optimization interface. The domains remain codegen and text2sql. The longer-term vision is for both domains to be runnable under GEPA's `optimize_anything()` interface.

**Current status**: Codegen adapter implemented. Text2SQL adapter is future work.

## Architecture

Both systems optimize the same thing — text that guides an LLM — but use different representations:

| System | Candidate Format | Evolution Method |
|--------|-----------------|------------------|
| GEPA | `dict[str, str]` of named text components | Reflective text evolution with Pareto selection |
| RoboPhD | Agent directory (files per task `file_mapping`) | Multi-strategy evolution with ELO ranking |

The adapter layer translates between these via a **file mapping**:

```python
CODEGEN_FILE_MAPPING = {
    "eval_instructions": "eval_instructions.md",
    "tool_code": "tools/problem_analyzer.py",
}
```

```
              extract_candidate              materialize_candidate
    agent dir ──────────────► dict ──────────────► agent dir
         │                     │                      │
         ▼                     ▼                      ▼
    RoboPhD evolution     GEPA optimize_anything  RoboPhD evolution
```

## Budget Economics

GEPA pays for two things out of one budget (`max_metric_calls`):

1. **Learning signal** — A small minibatch (2–3 examples) for reflection. This is where GEPA's evolutionary intelligence lives.
2. **Ranking** — Every accepted candidate is swept across the entire validation set to maintain the Pareto frontier. These calls produce **zero learning signal**.

Each mutation cycle costs: minibatch (~3) + val sweep (val_size). With `--val-ratio 0.05` (~39 val examples), each cycle ≈ 42 calls. The budget check happens **between rounds** — a round completes fully before GEPA checks whether it's exceeded the budget.

**Keep val small to maximize exploration.** With binary scoring (pass/fail), large valsets burn budget on ranking without proportionally better signal. See `local_docs/robophd_vs_gepa_evaluation_economics.md` for the full analysis.

## Usage

### Codegen

```bash
# Smoke test (~5 mutation cycles)
python scripts/run_gepa.py --task codegen \
    --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
    --engine-config '{"evaluation_budget": 200, "val_ratio": 0.05}'

# Full run with Opus reflection
python scripts/run_gepa.py --task codegen \
    --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
    --engine-config '{"evaluation_budget": 600, "val_ratio": 0.05, "reflection_model": "opus-4.6"}'

# Sequential execution (cleaner stack traces for debugging)
python scripts/run_gepa.py --task codegen \
    --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
    --engine-config '{"evaluation_budget": 200, "max_workers": 1}'

# With held-out test set evaluation
python scripts/run_gepa.py --task codegen \
    --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
    --engine-config '{"evaluation_budget": 600, "val_ratio": 0.05}' \
    --eval-test-set
```

### Cross-Benchmarking

Side-by-side comparison harness running GEPA and RoboPhD with matched evaluation budgets:

```bash
python scripts/cross_benchmark.py \
    --seed-agent RoboPhD/codegen_agents/naive_critic \
    --evaluation-budget 200
```

## Components

### `RoboPhD/adapters/gepa_codegen.py`

Core adapter with four pieces:

- **`materialize_candidate()`** — Writes candidate dict entries to files per `file_mapping`.
- **`extract_candidate()`** — Reads an agent directory back into a candidate dict.
- **`build_codegen_dataset()`** — Builds lightweight example dicts from codegen cache. Each example is just `{"question_id": "abc314_c"}` — no test data included, preventing leakage to GEPA's reflection LM.
- **`RoboPhDCodeGenEvaluator`** — GEPA-compatible evaluator wrapping `CriticEvaluator.evaluate_problem()`. Thread-safe for concurrent val sweeps via `threading.Lock`.

### `scripts/run_gepa.py`

Generic entry point for running GEPA on any registered task. Extracts seed candidate, builds train/val split, runs `optimize_anything()`, materializes best candidate back to an agent directory.

### `scripts/cross_benchmark.py`

Matched-budget comparison harness for GEPA vs RoboPhD.

### `RoboPhD/domains/external/domain.py`

`ExternalEvaluatorDomain` — wraps any `evaluator(candidate, example) -> (score, diagnostics)` function as a RoboPhD `DomainInterface`, letting RoboPhD's evolution loop consume external benchmarks.

## Thread Safety

`RoboPhDCodeGenEvaluator.__call__()` is called from up to 8 threads concurrently (via GEPA's `ThreadPoolExecutor` during val sweeps). A `threading.Lock` guards shared mutable state (`_eval_count`, `_ensure_agent_materialized`, `_get_evaluator`). The expensive `evaluate_problem()` call runs outside the lock since it uses per-problem output dirs.

**Assumption**: All concurrent callers pass the same candidate (same-candidate val sweeps). Cross-candidate parallelism would require per-candidate agent directories.

## Dependencies

```
pip install gepa cloudpickle
```

Both are included in `requirements.txt`. Also requires: codegen cache (`../robophd_runs/codegen_cache/`), HuggingFace dataset, Claude Code CLI.

## Future Work

- **Text2SQL adapter**: Add `RoboPhDText2SQLEvaluator` following the same pattern.
- **Unified interface**: Have both codegen and text2sql runnable under `optimize_anything()`.
- **`evaluation_budget` in researcher.py**: Count fresh evaluations across iterations for truly matched budget comparisons.
