# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

- **Recommended CLI Tools**: `jq` and `tree` (install via your package manager if not present)

## Project Overview

RoboPhD evolves AI agents to improve task performance without human intervention. The primary interface is the `optimize_anything()` API, with self-contained examples for each domain. Three optimization engines are available: RoboPhD (ELO-based evolution), GEPA (Pareto reflective), and Autoresearch (greedy hill-climbing).

**Active domains:**
- **ARC-AGI-1**: Abstract reasoning agents (Gemini via OpenRouter)
- **Can't Be Late**: Cloud scheduling strategies, no LLM calls (NSDI'24)
- **DocFinQA**: Retrieval + QA for long financial documents (GPT-4.1-mini)
- **Text2SQL**: SQL generation with `llm()` + `test_sql()` callables (BIRD benchmark)
- **Sudoku**: Pure-Python solver optimization, no LLM calls

**Paper**: [RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets](https://arxiv.org/abs/2604.04347)

## Domains

| Domain | Benchmark | Agent Files | Example |
|--------|-----------|-------------|---------|
| ARC-AGI-1 | ARC-AGI (HuggingFace) | `agent.py` | `examples/arc_agi_1/` |
| Can't Be Late | AWS spot traces (NSDI'24) | `agent.py` | `examples/cant_be_late/` |
| DocFinQA | DocFinQA (ACL 2024) | `agent.py` | `examples/docfinqa/` |
| Text2SQL | BIRD | `agent.py` + `analyze_db.py` | `examples/text2sql/` |
| Sudoku | sapientinc/sudoku-extreme | `agent.py` | `examples/sudoku/` |

Each example is self-contained: `main.py` (entry point), `evaluator.py` (scoring), `objective.md` + `background.md` (evolution context), and `seeds/baseline/` (seed agent).

## Key Commands

### Environment Setup
```bash
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"
pip install -r requirements.txt

# For GEPA and ARC-AGI (adds gepa, dspy, datasets, cloudpickle)
pip install -r requirements-gepa.txt

# For ARC-AGI: OpenRouter API key (routes to Gemini)
export OPENROUTER_API_KEY="sk-or-..."

# For DocFinQA: OpenAI API key (gpt-4.1-mini + embeddings)
export OPENAI_API_KEY="sk-..."

# Install Claude Code CLI (required for evolution)
# See: https://docs.anthropic.com/en/docs/claude-code
```

### Running Evolution

Each example is a standalone script calling `optimize_anything()`. Runs stop early when the `evaluation_budget` is exhausted (default 1500 evaluations).

```bash
# ARC-AGI evolution
python examples/arc_agi_1/main.py

# Can't Be Late evolution (download traces first)
bash examples/cant_be_late/download_traces.sh
python examples/cant_be_late/main.py

# DocFinQA evolution
python examples/docfinqa/main.py

# Text2SQL evolution (download BIRD dataset first)
bash benchmark_resources/download_bird.sh
python examples/text2sql/main.py

# Sudoku evolution
python examples/sudoku/main.py

# Quick smoke test (any domain)
python examples/arc_agi_1/main.py --num-iterations 2 --evaluation-budget 60
```

### Engine Selection

All examples support three optimization engines via `--engine`:

```bash
# RoboPhD ELO competition (default)
python examples/docfinqa/main.py

# GEPA Pareto-based reflective evolution
python examples/docfinqa/main.py --engine gepa

# Autoresearch single-session greedy hill-climbing
python examples/docfinqa/main.py --engine autoresearch
```

### Resume and Extend
```bash
# Resume from checkpoint
python examples/arc_agi_1/main.py \
  --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016

# Extend completed run with additional iterations
python examples/arc_agi_1/main.py \
  --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016 \
  --extend 5
```

### Test-Set Evaluation
```bash
# Evaluate best agent from a run on the held-out test set
python examples/arc_agi_1/main.py --eval-test-set  # after optimization

# Evaluate a prior run without re-optimizing
python examples/arc_agi_1/main.py \
  --eval-only --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016
```

## System Architecture

The `optimize_anything()` API supports three engines, selected by config type:

- **`RoboPhDConfig`** (default): Multi-agent ELO competition with Deep Focus refinement
- **`GEPAConfig`**: Pareto-based reflective text evolution
- **`AutoresearchConfig`**: Single Claude Code session with greedy experimentation

### How Evolution Works (RoboPhD Engine)

```
    ┌─────────────────────────────────────────────────────────────┐
    │                      ITERATION CYCLE                        │
    │                                                             │
    │  ┌──────────────────┐         ┌────────────────────┐        │
    │  │  EVOLUTION AI    │ Creates │  AGENT ARTIFACTS    │        │
    │  │  (Claude Code    │────────▶│  (per file_mapping) │        │
    │  │   CLI session)   │         └────────┬───────────┘        │
    │  └──────────────────┘                  │                    │
    │           ▲                             ▼                    │
    │           │                    ┌────────────────────┐        │
    │   Performance                  │  EVALUATOR FN      │        │
    │   data from                    │  Black-box scoring │        │
    │   prior iterations             │  (candidate,example)│       │
    │           │                    │   → (score, diag)  │        │
    │           │                    └────────┬───────────┘        │
    │           │                             │                    │
    │           │                             ▼                    │
    │  ┌────────┴─────────┐         ┌────────────────────┐        │
    │  │  AGENT RANKINGS  │◀────────│  ELO COMPETITION   │        │
    │  │  Top agents      │         │  Head-to-head on   │        │
    │  │  inform next     │         │  sampled problems   │        │
    │  │  evolution round │         └────────────────────┘        │
    │  └──────────────────┘                                       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

## Key Classes and Files

### API
- **`RoboPhD/api.py`**: `optimize_anything()`, `eval_candidate()`, `eval_run()`, config dataclasses
- **`RoboPhD/engines/`**: GEPA and Autoresearch engine wrappers

### Core
- **`RoboPhD/researcher.py`**: ParallelAgentResearcher — ELO evolution loop
- **`RoboPhD/evolution.py`**: Evolution strategy selector
- **`RoboPhD/deep_focus_evolution_manager.py`**: Multi-round evolution with testing
- **`RoboPhD/meta_evolution_manager.py`**: Meta-evolution for strategy improvement
- **`RoboPhD/domains/external/domain.py`**: Bridges evaluator functions into the evolution loop

### Utilities
- **`RoboPhD/eval_utils.py`**: `exec_with_stdout_capture`, `run_parallel_eval`, `retry_on_rate_limit`
- **`RoboPhD/candidate_utils.py`**: `extract_candidate` / `materialize_candidate`
- **`RoboPhD/runner_utils.py`**: `find_best_agent`, `to_litellm_model`, `CostTrackingLM`
- **`RoboPhD/config_manager.py`**: Delta-based configuration management
- **`RoboPhD/config.py`**: Model mappings, API keys

## Evolution System

### Evolution Strategies
- `RoboPhD/evolution_strategies/` — `cross_pollination`, `data_focus`, `refinement`, `use_your_judgment` (default)

### Advanced Configuration (via `--engine-config`)

```bash
# Evolution schedule
python examples/arc_agi_1/main.py --engine-config '{
  "evolution_strategy": "data_focus",
  "config_schedule": {"3": {"evolution_strategy": "none"}}
}'

# Deep Focus tuning
python examples/text2sql/main.py --engine-config '{
  "new_agent_test_rounds": 2,
  "evolution_model": "opus-4.7"
}'

# Weighted random strategies
python examples/cant_be_late/main.py --engine-config '{
  "use_weighted_random": true,
  "weighted_random_configs": [
    [{"evolution_strategy": "data_focus"}, 50],
    [{"evolution_strategy": "refinement"}, 30],
    [{"evolution_strategy": "none"}, 20]
  ]
}'
```

### ELO System
- **K-factor**: 32, **Initial ELO**: 1500
- **Tie Handling**: 0.5 points each, random winner selection
- **Clone detection**: -200 ELO penalty for identical predictions

## Development Tips

- **Quick Test**: `python examples/cant_be_late/main.py --num-iterations 2 --evaluation-budget 60`
- **Check Progress**: Review `checkpoint.json` and `final_report.md` in the experiment dir
- **Debug Evaluation**: Check `iteration_XXX/agent_YYY/problems/` in the experiment dir
- **Evolution Output**: Check `evolution_output/iteration_XXX/` for Claude's reasoning
- **Run Outputs**: All runs land in `../robophd_runs/` (`robophd/` for ELO, `gepa/` for GEPA, `autoresearch/` for Autoresearch)
- **Cleanup**: `python scripts/cleanup_runs.py` to find and remove short/experimental runs

## Troubleshooting

### Memory (OOM) Errors
- **Symptom**: Process killed with "zsh: killed"
- **Solution**: Use `--max-workers 4` or reduce to 2

### Evolution Failures
- **Claude CLI not found**: Ensure Claude Code CLI is installed
- **Context too long**: Use `--engine-config '{"examples_per_iteration": 3}'`
- **Session errors**: Check Claude CLI authentication with `claude --version`

### Domain-Specific Issues
- **ARC-AGI**: Requires `requirements-gepa.txt` (dspy, datasets) and `OPENROUTER_API_KEY`. Default solver: Gemini 3.1 Flash Lite via OpenRouter.
- **Can't Be Late**: Requires trace data download via `bash examples/cant_be_late/download_traces.sh`. No LLM calls — pure algorithmic optimization.
- **DocFinQA**: Requires OpenAI API key for `gpt-4.1-mini` and `text-embedding-3-small`. Dataset from HuggingFace.
- **Text2SQL**: Requires BIRD dataset via `bash benchmark_resources/download_bird.sh`. Default eval model: `haiku-4.5`.
- **Sudoku**: Requires HuggingFace `datasets` package. No API keys needed for the solver.

## License

MIT License - see LICENSE file for details.
