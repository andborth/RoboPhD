# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

- **Recommended CLI Tools**: `jq` and `tree` (install via your package manager if not present)
- **GEPA integration**: See [docs/claude/gepa.md](docs/claude/gepa.md)
- **Text2SQL domain**: See [docs/claude/text2sql.md](docs/claude/text2sql.md)

## Project Overview

RoboPhD is a multi-domain evolution system that implements a three-level AI hierarchy where AI agents conduct autonomous research to improve other AI agents. The primary interface is the `optimize_anything()` API, with self-contained examples for each domain.

**Active domains:**
- **ARC-AGI-1**: Abstract reasoning agents (Gemini via OpenRouter)
- **Can't Be Late**: Cloud scheduling strategies, no LLM calls (NSDI'24)
- **DocFinQA**: Retrieval + QA for long financial documents (GPT-4.1-mini)
- **Text2SQL**: SQL generation with `llm()` + `test_sql()` callables (BIRD benchmark)

Additional domains (CodeGen, AIME, CodeCritic) are available in the task registry but not actively maintained.

**Paper**: [RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets](https://arxiv.org/abs/2604.04347)

## Domains

| Domain | Benchmark | Agent Files | Example |
|--------|-----------|-------------|---------|
| ARC-AGI-1 | ARC-AGI (HuggingFace) | `agent.py` | `examples/arc_agi_1/` |
| Can't Be Late | AWS spot traces (NSDI'24) | `agent.py` | `examples/cant_be_late/` |
| DocFinQA | DocFinQA (ACL 2024) | `agent.py` | `examples/docfinqa/` |
| Text2SQL | BIRD | `agent.py` + `analyze_db.py` | `examples/text2sql/` |

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

Each example is a standalone script calling `optimize_anything()`. Runs stop early when the `evaluation_budget` is exhausted (default 1500 evaluations), so `--num-iterations 999` typically completes around iteration 21.

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

# Quick smoke test (any domain)
python examples/arc_agi_1/main.py --num-iterations 2 --evaluation-budget 60
```

### Resume and Extend
```bash
# Resume from checkpoint (auto-continues from last completed iteration)
python examples/arc_agi_1/main.py \
  --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016

# Restart from specific iteration with modifications
python examples/arc_agi_1/main.py \
  --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016 \
  --from-iteration 5

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

### Alternative Engines (GEPA and Autoresearch)

GEPA and Autoresearch are alternative optimization engines that use the task registry (`RoboPhD/tasks/`). They will be ported to `optimize_anything()` engine configs in the future.

```bash
# GEPA: Pareto-based reflective text evolution
python scripts/run_gepa.py --task arc_agi_1 \
  --engine-config '{"evaluation_budget": 300}'

python scripts/run_gepa.py --task cant_be_late_stdout \
  --engine-config '{"evaluation_budget": 1500, "val_size": 200}' \
  --eval-test-set

python scripts/run_gepa.py --task text2sql_integrated \
  --engine-config '{"evaluation_budget": 1500, "val_size": 200}' \
  --eval-test-set

# Autoresearch: single continuous Claude Code session with greedy keep/discard
python scripts/run_autoresearch.py --task arc_agi_1 \
  --engine-config '{"evaluation_budget": 300}'

python scripts/run_autoresearch.py --task cant_be_late_stdout \
  --engine-config '{"evaluation_budget": 1500}'

# Sequential (easier debugging, no ThreadPoolExecutor)
python scripts/run_gepa.py --task cant_be_late_stdout \
  --engine-config '{"evaluation_budget": 200, "max_workers": 1}'
```

**Note**: GEPA and Autoresearch use the old task names (`cant_be_late_stdout`, `text2sql_integrated`) from the task registry, not the simplified names used by the examples.

**Evaluation budget**: `evaluation_budget` (default 1500) caps total `(agent, example)` evaluations per run. All engines track evaluations and stop early when the budget is exhausted.

**GEPA budget math**: Each mutation cycle costs ~minibatch (3) + val sweep (val_size). With `--val-ratio 0.05` (~39 val examples), each cycle ≈ 42 calls. Keep val small to maximize exploration within the budget.

### Legacy Entry Point (`run_robophd.py`)

The original `scripts/run_robophd.py` still works and is required for meta-evolution, evolution schedules, and weighted random strategies. It uses the task registry.

```bash
python scripts/run_robophd.py --task arc_agi_1 --num-iterations 30
python scripts/run_robophd.py --task cant_be_late_stdout --list-params
```

## Three-Level AI Architecture

### Level 1: Development Layer
Claude Code writes and maintains the entire research system through natural language interaction.

### Level 2: Research Layer
RoboPhD agents conduct autonomous prompt/agent engineering research:
- **Parallel Agent Researcher**: Tests self-contained agents with embedded instructions
- **Evolution Strategies**: Dynamically loaded from `RoboPhD/evolution_strategies/`
- **Checkpoint System**: Full state preservation for fault tolerance
- **Evolution Schedule**: Fine-grained per-iteration control of evolution strategies

### Level 3: Execution Layer
Evolved text artifacts guide task execution with discovered optimizations.

## System Architecture

Three optimization engines share a common evaluation infrastructure:

- **`optimize_anything()`**: Primary API — multi-agent ELO competition (used by `examples/`)
- **`run_gepa.py`**: GEPA's reflective text evolution with Pareto selection
- **`run_autoresearch.py`**: Single Claude Code session with greedy experimentation

### How Evolution Works

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

1. **Evolution AI** (Claude Code CLI session) — Receives performance data from prior iterations, creates new agent artifacts
2. **Evaluator function** — Black-box `(candidate, example) -> (score, diagnostics)` that scores each agent on sampled problems. Each domain provides its own evaluator.
3. **ELO ranking** — Agents compete head-to-head on shared problem samples; top agents inform the next evolution round

## Agent Representation

Agents are directories containing text files. Each example defines which files constitute an agent via `seed_candidate` keys:

| Domain | Agent Files | Seed Agent |
|--------|-------------|------------|
| ARC-AGI-1 | `agent.py` | `examples/arc_agi_1/seeds/baseline/` |
| Can't Be Late | `agent.py` | `examples/cant_be_late/seeds/baseline/` |
| DocFinQA | `agent.py` | `examples/docfinqa/seeds/baseline/` |
| Text2SQL | `agent.py` + `analyze_db.py` | `examples/text2sql/seeds/baseline/` |

Conversion between agent directories and flat candidate dicts is handled by `candidate_utils.py` (`extract_candidate`, `materialize_candidate`).

## Evolution System

### Evolution Strategies
- `RoboPhD/evolution_strategies/` — `cross_pollination`, `data_focus`, `refinement`, `use_your_judgment` (used by all domains)

**Note**: Meta-evolution can generate additional strategies beyond these built-in options.

**Current defaults** (see `config_manager.py`):
- `evolution_strategy`: `use_your_judgment`
- `new_agent_test_rounds`: `1`
- `random_agent_wins_ties`: `True` (randomly selects one winner from tied agents)
- `include_evolution_rankings`: `False`

**Agent selection**: Prioritizes pending winners, new agents, then untested agents. Remaining slots filled randomly from top ELO > 1500 agents (falling back to lower ELO if needed).

**Selection strategies** (skip evolution):
- `challenger`: Skip evolution, test under-tested agents (fewest tests first)
- `greedy`: Skip evolution, use deterministic top-k ELO selection
- `none`: Skip evolution, use randomized ELO-based agent selection

### Evolution Schedule Control

Evolution strategies can be controlled per-iteration using `--engine-config`:

```bash
python examples/arc_agi_1/main.py \
  --engine-config '{
    "evolution_strategy": "data_focus",
    "config_schedule": {
      "3": {"evolution_strategy": "none"},
      "5": {"evolution_strategy": "refinement"},
      "7": {"evolution_strategy": "challenger"}
    }
  }'
```

### Weighted Random Evolution
```bash
python examples/arc_agi_1/main.py \
  --engine-config '{
    "use_weighted_random": true,
    "weighted_random_configs": [
      [{"evolution_strategy": "data_focus"}, 50],
      [{"evolution_strategy": "refinement"}, 30],
      [{"evolution_strategy": "none"}, 20]
    ]
  }'
```

### Deep Focus Evolution
Deep Focus uses multiple rounds of refinement within a single evolution session:

```bash
python examples/text2sql/main.py \
  --engine-config '{
    "new_agent_test_rounds": 2,
    "new_agent_test_round_offset": -2,
    "evolution_model": "opus-4.6"
  }'
```

- `"new_agent_test_rounds": 0`: Planning + implementation only
- `"new_agent_test_rounds": 1`: Adds testing against 1 prior iteration [DEFAULT]
- `"new_agent_test_rounds": 2`: Adds testing against 2 prior iterations
- `"new_agent_test_round_offset": -2`: Starting offset from current iteration [DEFAULT]. At iteration 8, tests against iterations 6 and 5.

### Meta-Evolution
Meta-evolution evolves the evolution strategies themselves (experimental — uses `run_robophd.py`):

```bash
python scripts/run_robophd.py --task cant_be_late_stdout --num-iterations 20 \
  --engine-config configs/robophd_engine/meta_evolution_starts_at_5.json
```

## Key Classes and Files

### Entry Points
- **`examples/*/main.py`**: Primary — self-contained evolution runners using `optimize_anything()`
- **`scripts/run_gepa.py`**: GEPA optimization via task registry
- **`scripts/run_autoresearch.py`**: Autoresearch optimization via task registry
- **`scripts/run_robophd.py`**: Legacy ELO evolution runner (still needed for meta-evolution)

### API
- **`RoboPhD/api.py`**: `optimize_anything()`, `eval_candidate()`, `eval_run()`, `RoboPhDConfig`
- **`RoboPhD/eval_utils.py`**: Shared helpers — `retry_on_rate_limit`, `exec_with_stdout_capture`, `force_exit_if_threads_leaked`, `run_parallel_eval`

### Tasks and Adapters
- **`tasks/base.py`**: `TaskDefinition` dataclass (name, evaluator_factory, dataset_builder, file_mapping, objective)
- **`tasks/__init__.py`**: Task registry — `get_task(name)`, `list_tasks()` (used by GEPA, Autoresearch, run_robophd.py)
- **`adapters/candidate_utils.py`**: `extract_candidate` / `materialize_candidate` — convert between agent dirs and flat dicts
- Vendored files (`*_unmodified*`): exact copies from upstream, do not modify

### Core
- **`researcher.py`**: Evolution loop orchestrator (called by `optimize_anything()`)
- **`evolution.py`**: Evolution strategy selector and orchestration
- **`deep_focus_evolution_manager.py`**: Multi-round evolution with testing
- **`meta_evolution_manager.py`**: Meta-evolution for strategy improvement
- **`domains/external/domain.py`**: `ExternalEvaluatorDomain` — bridges evaluator functions into the evolution loop

### Config
- **`config.py`**: Model mappings and fallbacks
- **`config_manager.py`**: Delta-based configuration management

## Critical Implementation Details

### ELO System
- **Tie Handling**: Agents with equal accuracy exchange 0.5 points each
- **Ranking Display**: Tied agents show same rank (e.g., #1, #1, #3)
- **K-factor**: 32 for moderate rating changes
- **Initial ELO**: 1500 for new agents

### Model Configuration
- **API Models**: opus-4.6 ($5/$25/MTok), sonnet-4.5 ($3/$15/MTok), haiku-4.5 ($1/$5/MTok)
- **Timeouts**: 3600s (60 minutes) default for evolution
- **Eval Timeout**: `eval_timeout` (300s default, 600s for ARC-AGI) — per-evaluation timeout on `future.result()` in all ThreadPoolExecutor eval loops. Timed-out evals score 0 with `"error": "timeout"` in result.json. The hung thread keeps burning CPU until process exit (Python limitation); leaked thread count is tracked and warned.
- **API Key**: Set via `ANTHROPIC_API_KEY_FOR_ROBOPHD` environment variable

## Development Tips

- **Quick Test**: `python examples/cant_be_late/main.py --num-iterations 2 --evaluation-budget 60`
- **Check Progress**: Review `checkpoint.json` and `final_report.md` in the experiment dir
- **Debug Evaluation**: Check `iteration_XXX/agent_YYY/problems/` and `evaluation.json` in the experiment dir
- **Evolution Output**: Check `evolution_output/iteration_XXX/` in the experiment dir for Claude's reasoning
- **Run Outputs**: All runs land in `../robophd_runs/` (`robophd/` for ELO, `gepa/` for GEPA, `autoresearch/` for Autoresearch). Results JSON files in `../robophd_runs/results/`.
- **Cleanup**: `python scripts/cleanup_runs.py` to find and remove short/experimental runs
- **Config Files**: Save common configs to JSON files and use `--engine-config path/to/config.json`

## Troubleshooting

### Memory (OOM) Errors
- **Symptom**: Process killed with "zsh: killed"
- **Solution**: Use `--max-workers 4` or reduce to 2

### Evolution Failures
- **Claude CLI not found**: Ensure Claude Code CLI is installed
- **Context too long**: Use `--examples-per-iteration 3`
- **Session errors**: Check Claude CLI authentication with `claude --version`

### Domain-Specific Issues
- **ARC-AGI**: Requires `requirements-gepa.txt` (dspy, datasets) and `OPENROUTER_API_KEY`. Default solver: `gemini-2.5-flash-lite` via OpenRouter.
- **Can't Be Late**: Requires trace data download via `bash examples/cant_be_late/download_traces.sh`. No LLM calls — pure algorithmic optimization via subprocess simulation.
- **DocFinQA**: Requires OpenAI API key for `gpt-4.1-mini` (reasoning) and `text-embedding-3-small` (retrieval). Dataset loaded from HuggingFace (`kensho/DocFinQA`). Per-question cost budget of $0.10 enforced.
- **Text2SQL**: Requires BIRD dataset via `bash benchmark_resources/download_bird.sh`. Default eval model: `haiku-4.5`. Per-question cost budget of $0.10 enforced.

## License

MIT License - see LICENSE file for details.
