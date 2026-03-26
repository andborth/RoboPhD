# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

- **Available CLI Tools**: `jq` and `tree` are installed and available
- **GEPA integration**: See [docs/claude/gepa.md](docs/claude/gepa.md)
- **Text2SQL domain**: See [docs/claude/text2sql.md](docs/claude/text2sql.md)

## Project Overview

RoboPhD is a multi-domain evolution system that implements a three-level AI hierarchy where AI agents conduct autonomous research to improve other AI agents.

**Active domains** (current focus):
- **ARC-AGI-1**: Evolving abstract reasoning agents with rich diagnostics and stdout capture (Gemini via OpenRouter)
- **Can't Be Late (stdout)**: Evolving cloud scheduling strategies with stdout capture, standalone Agent class (NSDI'24)
- **DocFinQA**: Evolving retrieval + QA agents for long-document financial questions
- **Text2SQL Integrated**: Agent-controlled SQL generation with `llm()` + `test_sql()` callables (BIRD)

**Older task variants** (available, not primary focus):
- **ARC-AGI**, **Can't Be Late**, **Text2SQL**, **Text2SQL (stdout)** — earlier versions with less agent control or fewer diagnostics

Additional domains (CodeGen, AIME, CodeCritic) are available in the task registry but not actively maintained.

New domains are added via the task registry (`RoboPhD/tasks/`) — implement a `TaskDefinition` with an evaluator function, dataset builder, and file mapping. See `RoboPhD/adapters/README.md` for a guide.

**Paper**: [RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution](https://arxiv.org/abs/2601.01126)
(Text2SQL achieved **73.67% accuracy** on BIRD benchmark test set.)

## Domains

| Domain | Benchmark | Agent Artifacts (`file_mapping`) |
|--------|-----------|-------------------------------|
| ARC-AGI-1 | ARC-AGI (HuggingFace) | `agent.py` |
| Can't Be Late (stdout) | AWS spot traces (NSDI'24) | `agent.py` (standalone Agent class) |
| DocFinQA | DocFinQA (ACL 2024) | `agent.py` |
| Text2SQL Integrated | BIRD | `agent.py` + `analyze_db.py` |
| Text2SQL | BIRD | `eval_instructions.md` + `tools/analyze_db.py` + `verify_prompt.md` |

## Key Commands

### Environment Setup
```bash
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"
pip install -r requirements.txt

# For GEPA and ARC-AGI (adds gepa, dspy, datasets, cloudpickle)
pip install -r requirements-gepa.txt

# For ARC-AGI: OpenRouter API key (routes to Gemini)
export OPENROUTER_API_KEY="sk-or-..."

# Install Claude Code CLI (required for evolution)
# See: https://docs.anthropic.com/en/docs/claude-code
```

### Running Evolution (`run_robophd.py`)

Runs stop early when the `evaluation_budget` is exhausted (default 1500 evaluations), so `--num-iterations 30` typically completes around iteration 21.

```bash
# ARC-AGI evolution
python scripts/run_robophd.py --task arc_agi_1 --num-iterations 30

# Can't Be Late evolution (download traces first)
bash scripts/download_cant_be_late_traces.sh
python scripts/run_robophd.py --task cant_be_late_stdout --num-iterations 30

# DocFinQA evolution
python scripts/run_robophd.py --task docfinqa --num-iterations 30

# Text2SQL evolution
python scripts/run_robophd.py --task text2sql_integrated --num-iterations 30

# Quick test
python scripts/run_robophd.py --task cant_be_late_stdout --num-iterations 2 \
  --engine-config '{"examples_per_iteration": 3}'

# List all valid parameters for a task
python scripts/run_robophd.py --task cant_be_late_stdout --list-params
```

### Running GEPA (`run_gepa.py`)
```bash
# Can't Be Late (stdout) via GEPA
python scripts/run_gepa.py --task cant_be_late_stdout \
  --engine-config '{"evaluation_budget": 1500, "val_size": 200}' \
  --eval-test-set

# ARC-AGI-1 via GEPA (pre-split: train=200, val=200 matching GEPA exactly)
python scripts/run_gepa.py --task arc_agi_1 \
  --engine-config '{"evaluation_budget": 300}'

# Text2SQL Integrated via GEPA
python scripts/run_gepa.py --task text2sql_integrated \
  --engine-config '{"evaluation_budget": 1500, "val_size": 200}' \
  --eval-test-set

# DocFinQA via GEPA
python scripts/run_gepa.py --task docfinqa \
  --engine-config '{"evaluation_budget": 1500, "val_size": 200}' \
  --eval-test-set

# Sequential (easier debugging, no ThreadPoolExecutor)
python scripts/run_gepa.py --task cant_be_late_stdout \
  --engine-config '{"evaluation_budget": 200, "max_workers": 1}'
```

**Evaluation budget**: `evaluation_budget` (default 1500) caps the total number of `(agent, example)` evaluations per run. Both engines track evaluations and stop early when the budget is exhausted — so `--num-iterations 30` acts as an upper bound, not a target. A typical RoboPhD run with default settings completes ~21 iterations before hitting the budget.

**GEPA budget math**: Each mutation cycle costs ~minibatch (3) + val sweep (val_size). With `--val-ratio 0.05` (~39 val examples), each cycle ≈ 42 calls. Keep val small to maximize exploration within the budget.

### Task Config Persistence
Task-level parameters (e.g. `split`, `solver_model`, `cost_budget`) are persisted in `checkpoint.json` under `task_config` and automatically restored on resume. This ensures evaluator/dataset construction uses identical settings across resume boundaries. Engine-config overrides on resume are merged on top.

### Resume and Extend
```bash
# Resume from checkpoint (auto-continues from last completed iteration)
python scripts/run_robophd.py \
  --resume ../robophd_runs/robophd/cant_be_late_20260313_230325

# Restart from specific iteration with modifications
python scripts/run_robophd.py \
  --resume ../robophd_runs/robophd/cant_be_late_20260313_230325 \
  --from-iteration 5 \
  --engine-config '{"examples_per_iteration": 10}'

# Extend completed run with additional iterations
python scripts/run_robophd.py \
  --resume ../robophd_runs/robophd/cant_be_late_20260313_230325 \
  --extend 5 \
  --engine-config '{"evolution_strategy": "challenger"}'
```

### Test-Set Evaluation (`eval_test_set.py`)
```bash
# Auto-select best agent by ELO from a run
python scripts/eval_test_set.py --task arc_agi_1 \
  --run-dir ../robophd_runs/robophd/arc_agi_1_20260322_183016

# Specify agent directly
python scripts/eval_test_set.py --task text2sql_integrated \
  --agent-dir RoboPhD/text2sql_integrated_agents/baseline

# King-of-the-hill: test the last-round winner instead of ELO leader
python scripts/eval_test_set.py --task cant_be_late_stdout \
  --run-dir ../robophd_runs/robophd/cant_be_late_stdout_20260322_183016 \
  --last-winner

# With repeats and config overrides
python scripts/eval_test_set.py --task arc_agi_1 --run-dir ... \
  --test-repeats 3
```

## Three-Level AI Architecture

### Level 1: Development Layer
Claude Code writes and maintains the entire research system through natural language interaction.

### Level 2: Research Layer
RoboPhD agents conduct autonomous prompt/agent engineering research:
- **Parallel Agent Researcher**: Tests self-contained agents with embedded instructions
- **Evolution Strategies**: Dynamically loaded from `RoboPhD/evolution_strategies/` (generic) or domain-specific directories
- **Checkpoint System**: Full state preservation for fault tolerance
- **Evolution Schedule**: Fine-grained per-iteration control of evolution strategies

### Level 3: Execution Layer
Evolved text artifacts guide task execution with discovered optimizations.

## System Architecture

Two optimization engines share a common task registry:

- **`run_robophd.py`**: Multi-agent ELO competition with evolution strategies
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

1. **Evolution AI** (Claude Code CLI session) — Receives performance data from prior iterations, creates new agent artifacts according to the task's `file_mapping`
2. **Evaluator function** — Black-box `(candidate, example) -> (score, diagnostics)` that scores each agent on sampled problems. Each domain provides its own evaluator.
3. **ELO ranking** — Agents compete head-to-head on shared problem samples; top agents inform the next evolution round

## Agent Representation

Agents are directories containing text files declared by the task's `file_mapping`. Each task defines which files constitute an agent:

| Domain | `file_mapping` | Seed Agent |
|--------|---------------|------------|
| ARC-AGI | `{"agent_code": "agent.py"}` | `RoboPhD/arcagi_agents/baseline/` |
| Can't Be Late | `{"agent_code": "agent.py"}` | `RoboPhD/cant_be_late_agents/baseline/` |
| DocFinQA | `{"agent_code": "agent.py"}` | `RoboPhD/docfinqa_agents/baseline/` |
| Text2SQL | `{"eval_instructions": "eval_instructions.md", "database_analysis_code": "tools/analyze_db.py", "verify_prompt": "verify_prompt.md"}` | `RoboPhD/text2sql_agents/naive/` |

Conversion between agent directories and flat candidate dicts is handled by `candidate_utils.py` (`extract_candidate`, `materialize_candidate`).

## Evolution System

### Evolution Strategies
Evolution strategies are organized by domain:

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

Evolution strategies can be controlled per-iteration using the `config_schedule` parameter:

```bash
python scripts/run_robophd.py --task cant_be_late --num-iterations 10 \
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
Configure weighted random strategy selection:

```bash
python scripts/run_robophd.py --task cant_be_late --num-iterations 10 \
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
Deep Focus is an advanced evolution mode that uses multiple rounds of refinement:

```bash
python scripts/run_robophd.py --task text2sql --num-iterations 10 \
  --engine-config '{
    "new_agent_test_rounds": 2,
    "new_agent_test_round_offset": -2,
    "evolution_model": "opus-4.6"
  }'
```

- `"new_agent_test_rounds": 0`: Planning + implementation only
- `"new_agent_test_rounds": 1`: Adds testing against 1 prior iteration [DEFAULT]
- `"new_agent_test_rounds": 2`: Adds testing against 2 prior iterations
- `"new_agent_test_round_offset": -2`: Starting offset from current iteration [DEFAULT]. At iteration 8, tests against iterations 6 and 5. Use `-1` for legacy behavior (tests 7 and 6). Iterations < 1 are skipped.

### Meta-Evolution
Meta-evolution allows evolving the evolution strategies themselves (experimental — currently under study):

```bash
python scripts/run_robophd.py --task cant_be_late --num-iterations 20 \
  --engine-config configs/robophd_engine/meta_evolution_starts_at_5.json
```

Available meta-evolution strategies:
- `train_a_winner`: Evolve strategies that produce winning agents

## Key Classes and Files

### Entry Points
- **`scripts/run_robophd.py`**: Multi-agent ELO evolution runner
- **`scripts/run_gepa.py`**: GEPA optimization runner
- **`scripts/run_autoresearch.py`**: Autoresearch single-session optimizer
- **`scripts/eval_test_set.py`**: Standalone test-set evaluation for any agent

### Tasks and Adapters
- **`tasks/base.py`**: `TaskDefinition` dataclass (name, evaluator_factory, dataset_builder, file_mapping, objective)
- **`tasks/__init__.py`**: Task registry — `get_task(name)`, `list_tasks()`
- Each task has a definition in `tasks/` and an evaluator in `adapters/`. See `adapters/README.md` for how to add new tasks.
- **`adapters/candidate_utils.py`**: `extract_candidate` / `materialize_candidate` — convert between agent dirs and flat dicts
- Vendored files (`*_unmodified*`): exact copies from upstream, do not modify (except targeted `--silent` changes in `cant_be_late_utils_unmodified/`)

### Core
- **`researcher.py`**: Evolution loop orchestrator (called by `run_robophd.py`)
- **`evolution.py`**: Evolution strategy selector and orchestration
- **`deep_focus_evolution_manager.py`**: Multi-round evolution with testing
- **`meta_evolution_manager.py`**: Meta-evolution for strategy improvement
- **`domains/external/domain.py`**: `ExternalEvaluatorDomain` — bridges evaluator functions into RoboPhD's evolution loop

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
- **Eval Timeout**: `eval_timeout` (300s default, 600s for ARC-AGI) — per-evaluation timeout on `future.result()` in all ThreadPoolExecutor eval loops (domain.py, run_gepa.py, eval_test_set.py). Timed-out evals score 0 with `"error": "timeout"` in result.json. The hung thread keeps burning CPU until process exit (Python limitation); `domain.py` tracks leaked thread count across iterations and warns at each iteration start.
- **API Key**: Set via `ANTHROPIC_API_KEY_FOR_ROBOPHD` environment variable

## Development Tips

- **Quick Test**: `python scripts/run_robophd.py --task cant_be_late --num-iterations 2 --engine-config '{"examples_per_iteration": 3}'`
- **List Parameters**: `python scripts/run_robophd.py --task cant_be_late --list-params`
- **Check Progress**: Review `checkpoint.json` and `final_report.md` in the experiment dir
- **Debug Evaluation**: Check `iteration_XXX/agent_YYY/problems/` and `evaluation.json` in the experiment dir
- **Evolution Output**: Check `evolution_output/iteration_XXX/` in the experiment dir for Claude's reasoning
- **Run Outputs**: All runs land in `../robophd_runs/` (`robophd/` for ELO, `gepa/` for GEPA). Results JSON files and run symlinks in `../robophd_runs/results/`. Agent tests in `../robophd_runs/agent_tests/`.
- **Cleanup**: `python scripts/cleanup_runs.py` to find and remove short/experimental runs
- **Config Files**: Save common configs to JSON files and use `--engine-config path/to/config.json`

## Troubleshooting

### Memory (OOM) Errors
- **Symptom**: Process killed with "zsh: killed"
- **Solution**: Use `--engine-config '{"max_concurrent": 4}'` or reduce to 2

### Evolution Failures
- **Claude CLI not found**: Ensure Claude Code CLI is installed
- **Context too long**: Use `--engine-config '{"examples_per_iteration": 3}'`
- **Session errors**: Check Claude CLI authentication with `claude --version`

### Domain-Specific Issues
- **ARC-AGI**: Requires `requirements-gepa.txt` (dspy, datasets) and `OPENROUTER_API_KEY`. Default solver: `gemini-3.1-flash-lite-preview` via OpenRouter. Cost tracking uses `resp.usage.cost` from OpenRouter (litellm's pricing DB doesn't cover these models).
- **Can't Be Late**: Requires trace data download via `bash scripts/download_cant_be_late_traces.sh`. Requires `configargparse`, `colorama`, `pyyaml` (simulator dependencies). No LLM calls — pure algorithmic optimization via subprocess simulation.
- **DocFinQA**: Requires OpenAI API key for `gpt-4.1-mini` (reasoning) and `text-embedding-3-small` (retrieval). Dataset loaded from HuggingFace (`kensho/DocFinQA`). Per-question cost budget of $0.10 enforced.
- **Text2SQL**: See [docs/claude/text2sql.md](docs/claude/text2sql.md) for database-related issues.

## License

MIT License - see LICENSE file for details.
