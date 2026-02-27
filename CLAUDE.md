# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

- **Available CLI Tools**: `jq` and `tree` are installed and available
- **CodeGen domain**: See [docs/claude/codegen.md](docs/claude/codegen.md)
- **GEPA integration**: See [docs/claude/gepa.md](docs/claude/gepa.md)
- **Text2SQL domain (legacy, pending migration)**: See [docs/claude/text2sql.md](docs/claude/text2sql.md)

## Project Overview

RoboPhD is a multi-domain evolution system that implements a three-level AI hierarchy where AI agents conduct autonomous research to improve other AI agents.

**Active domains**:
- **CodeGen**: Evolving critic agents for code review on LiveCodeBench
- **AIME**: Evolving math reasoning prompts on AIME 2022-2024

New domains are added via the task registry (`RoboPhD/tasks/`) — implement a `TaskDefinition` with an evaluator function, dataset builder, and file mapping.

**Paper**: [RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution](https://arxiv.org/abs/2601.01126)
(Text2SQL achieved **73.67% accuracy** on BIRD benchmark test set. Not yet migrated to the current architecture.)

## Domains

| Domain | Benchmark | Agent Artifacts (`file_mapping`) |
|--------|-----------|-------------------------------|
| CodeGen | LiveCodeBench | `eval_instructions.md` + `tools/problem_analyzer.py` |
| AIME | AIME 2022-2024 | `system_prompt.md` |

## Key Commands

### Environment Setup
```bash
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"
pip install -r requirements.txt

# Install Claude Code CLI (required for evolution)
# See: https://docs.anthropic.com/en/docs/claude-code
```

### Running Evolution (`run_robophd.py`)
```bash
# CodeGen evolution (10 iterations)
python scripts/run_robophd.py --task codegen --num-iterations 10

# AIME evolution
python scripts/run_robophd.py --task aime --num-iterations 10

# Quick test
python scripts/run_robophd.py --task codegen --num-iterations 2 \
  --engine-config '{"examples_per_iteration": 3}'

# List all valid parameters for a task
python scripts/run_robophd.py --task codegen --list-params
```

### Running GEPA (`run_gepa.py`)
```bash
# Smoke test (~5 mutation cycles)
python scripts/run_gepa.py --task codegen \
  --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
  --engine-config '{"evaluation_budget": 200, "val_ratio": 0.05}'

# Full run
python scripts/run_gepa.py --task codegen \
  --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
  --engine-config '{"evaluation_budget": 600, "val_ratio": 0.05, "reflection_model": "opus-4.6"}'

# AIME via GEPA
python scripts/run_gepa.py --task aime \
  --engine-config '{"evaluation_budget": 200, "val_ratio": 0.2}'

# Sequential (easier debugging, no ThreadPoolExecutor)
python scripts/run_gepa.py --task codegen \
  --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic"}' \
  --engine-config '{"evaluation_budget": 200, "max_workers": 1}'
```

**Budget math**: Each mutation cycle costs ~minibatch (3) + val sweep (val_size). With `--val-ratio 0.05` (~39 val examples), each cycle ≈ 42 calls. Keep val small to maximize exploration within the budget.

### Resume and Extend
```bash
# Resume from checkpoint (auto-continues from last completed iteration)
python scripts/run_robophd.py --task codegen \
  --resume evolution/robophd_20251031_043607

# Restart from specific iteration with modifications
python scripts/run_robophd.py --task codegen \
  --resume evolution/robophd_20251031_043607 \
  --from-iteration 5 \
  --engine-config '{"examples_per_iteration": 10, "eval_model": "sonnet-4.5"}'

# Extend completed run with additional iterations
python scripts/run_robophd.py --task codegen \
  --resume evolution/robophd_20251031_043607 \
  --extend 5 \
  --engine-config '{"evolution_strategy": "challenger"}'
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
| CodeGen | `{"eval_instructions": "eval_instructions.md", "tool_code": "tools/problem_analyzer.py"}` | `RoboPhD/codegen_agents/naive_critic/` |
| AIME | `{"system_prompt": "system_prompt.md"}` | `RoboPhD/aime_agents/baseline/` |

Conversion between agent directories and flat candidate dicts is handled by `candidate_utils.py` (`extract_candidate`, `materialize_candidate`).

## Evolution System

### Evolution Strategies
Evolution strategies are organized by domain:

- `RoboPhD/evolution_strategies/` — **Generic** (used by domains like AIME): `cross_pollination`, `refinement`
- `RoboPhD/evolution_strategies_codegen/` — **CodeGen**: `cross_pollination_tool_only`, `refinement_tool_only`

**Note**: Meta-evolution can generate additional strategies beyond these built-in options.

**Agent selection**: Prioritizes pending winners, new agents, then untested agents. Remaining slots filled randomly from top ELO > 1500 agents (falling back to lower ELO if needed).

**Selection strategies** (skip evolution):
- `challenger`: Skip evolution, test under-tested agents (fewest tests first)
- `greedy`: Skip evolution, use deterministic top-k ELO selection
- `none`: Skip evolution, use randomized ELO-based agent selection

### Evolution Schedule Control

Evolution strategies can be controlled per-iteration using the `config_schedule` parameter:

```bash
python scripts/run_robophd.py --task codegen --num-iterations 10 \
  --engine-config '{
    "evolution_strategy": "cross_pollination_tool_only",
    "config_schedule": {
      "3": {"evolution_strategy": "none"},
      "5": {"evolution_strategy": "refinement_tool_only"},
      "7": {"evolution_strategy": "challenger"}
    }
  }'
```

### Weighted Random Evolution
Configure weighted random strategy selection:

```bash
python scripts/run_robophd.py --task codegen --num-iterations 10 \
  --engine-config '{
    "use_weighted_random": true,
    "weighted_random_configs": [
      [{"evolution_strategy": "cross_pollination_tool_only"}, 50],
      [{"evolution_strategy": "refinement_tool_only"}, 30],
      [{"evolution_strategy": "none"}, 20]
    ]
  }'
```

### Deep Focus Evolution
Deep Focus is an advanced evolution mode that uses multiple rounds of refinement:

```bash
python scripts/run_robophd.py --task codegen --num-iterations 10 \
  --engine-config '{
    "new_agent_test_rounds": 2,
    "evolution_model": "opus-4.6",
    "eval_model": "haiku-4.5"
  }'
```

- `"new_agent_test_rounds": 0`: Planning + implementation only
- `"new_agent_test_rounds": 1`: Adds testing against 1 prior iteration [DEFAULT]
- `"new_agent_test_rounds": 2`: Adds testing against 2 prior iterations

### Meta-Evolution
Meta-evolution allows evolving the evolution strategies themselves:

```bash
python scripts/run_robophd.py --task codegen --num-iterations 20 \
  --engine-config '{
    "meta_evolution_strategy": "train_a_winner",
    "config_schedule": {
      "10": {"meta_evolution_strategy": "train_a_winner"},
      "11": {"meta_evolution_strategy": null}
    }
  }'
```

Available meta-evolution strategies:
- `train_a_winner`: Evolve strategies that produce winning agents
- `train_a_new_champion`: Evolve strategies for consistent top performance

## Key Classes and Files

### Entry Points
- **`scripts/run_robophd.py`**: Multi-agent ELO evolution runner
- **`scripts/run_gepa.py`**: GEPA optimization runner

### Task Registry
- **`tasks/base.py`**: `TaskDefinition` dataclass (name, evaluator_factory, dataset_builder, file_mapping, objective)
- **`tasks/codegen.py`**: CodeGen task — LiveCodeBench critic evolution
- **`tasks/aime.py`**: AIME task — math reasoning prompt evolution

### Core
- **`researcher.py`**: Evolution loop orchestrator (called by `run_robophd.py`)
- **`evolution.py`**: Evolution strategy selector and orchestration
- **`deep_focus_evolution_manager.py`**: Multi-round evolution with testing
- **`meta_evolution_manager.py`**: Meta-evolution for strategy improvement

### Domains
- **`domains/external/domain.py`**: `ExternalEvaluatorDomain` — bridges evaluator functions into RoboPhD's evolution loop

### Adapters
- **`adapters/candidate_utils.py`**: `extract_candidate` / `materialize_candidate` — convert between agent dirs and flat dicts
- **`adapters/gepa_codegen.py`**: GEPA adapter for CodeGen evaluator
- **`adapters/gepa_aime.py`**: GEPA adapter for AIME evaluator

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
- **Timeouts**: 1800s (30 minutes) default for evolution
- **CodeGen Timeouts**: `codegen_timeout` (1200s) for solution generation, `critic_timeout` (600s) for critic/revision/acceptance
- **API Key**: Set via `ANTHROPIC_API_KEY_FOR_ROBOPHD` environment variable

## Development Tips

- **Quick Test**: `python scripts/run_robophd.py --task codegen --num-iterations 2 --engine-config '{"examples_per_iteration": 1}'`
- **List Parameters**: `python scripts/run_robophd.py --task codegen --list-params`
- **Check Progress**: Review `checkpoint.json` and `final_report.md`
- **Debug Evaluation**: Check `iteration_XXX/agent_YYY/problems/` and `evaluation.json`
- **Evolution Output**: Check `evolution_output/iteration_XXX/` for Claude's reasoning
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
- **CodeGen**: See [docs/claude/codegen.md](docs/claude/codegen.md) for test execution issues

## License

MIT License - see LICENSE file for details.
