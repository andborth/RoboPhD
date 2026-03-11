# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

- **Available CLI Tools**: `jq` and `tree` are installed and available
- **CodeGen domain**: See [docs/claude/codegen.md](docs/claude/codegen.md)
- **GEPA integration**: See [docs/claude/gepa.md](docs/claude/gepa.md)
- **Text2SQL domain**: See [docs/claude/text2sql.md](docs/claude/text2sql.md)

## Project Overview

RoboPhD is a multi-domain evolution system that implements a three-level AI hierarchy where AI agents conduct autonomous research to improve other AI agents.

**Active domains**:
- **CodeGen**: Evolving critic agents for code review on LiveCodeBench
- **AIME**: Evolving math reasoning prompts on AIME 2022-2024
- **Text2SQL**: Evolving database analysis agents for BIRD benchmark SQL generation
- **ARC-AGI**: Evolving abstract reasoning agents (Gemini via OpenRouter)
- **CodeCritic**: CodeGen variant with fresh-session revision (no code generation)

New domains are added via the task registry (`RoboPhD/tasks/`) — implement a `TaskDefinition` with an evaluator function, dataset builder, and file mapping.

**Paper**: [RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution](https://arxiv.org/abs/2601.01126)
(Text2SQL achieved **73.67% accuracy** on BIRD benchmark test set.)

## Domains

| Domain | Benchmark | Agent Artifacts (`file_mapping`) |
|--------|-----------|-------------------------------|
| CodeGen | LiveCodeBench | `eval_instructions.md` + `tools/problem_analyzer.py` |
| AIME | AIME 2022-2024 | `system_prompt.md` |
| Text2SQL | BIRD | `eval_instructions.md` + `tools/analyze_db.py` + `verify_prompt.md` |
| ARC-AGI | ARC-AGI (HuggingFace) | `agent.py` |
| CodeCritic | LiveCodeBench | `eval_instructions.md` + `tools/problem_analyzer.py` |

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
```bash
# CodeGen evolution (10 iterations)
python scripts/run_robophd.py --task codegen --num-iterations 10

# AIME evolution
python scripts/run_robophd.py --task aime --num-iterations 10

# Text2SQL evolution
python scripts/run_robophd.py --task text2sql --num-iterations 10

# ARC-AGI evolution
python scripts/run_robophd.py --task arc_agi --num-iterations 10

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

# ARC-AGI via GEPA (pre-split: train=200, val=200 matching GEPA exactly)
python scripts/run_gepa.py --task arc_agi \
  --engine-config '{"evaluation_budget": 300}'

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
  --resume ../robophd_runs/robophd/codegen_20251031_043607

# Restart from specific iteration with modifications
python scripts/run_robophd.py --task codegen \
  --resume ../robophd_runs/robophd/codegen_20251031_043607 \
  --from-iteration 5 \
  --engine-config '{"examples_per_iteration": 10, "eval_model": "sonnet-4.5"}'

# Extend completed run with additional iterations
python scripts/run_robophd.py --task codegen \
  --resume ../robophd_runs/robophd/codegen_20251031_043607 \
  --extend 5 \
  --engine-config '{"evolution_strategy": "challenger"}'
```

### Test-Set Evaluation (`eval_test_set.py`)
```bash
# Auto-select best agent by ELO from a run
python scripts/eval_test_set.py --task arc_agi \
  --run-dir ../robophd_runs/robophd/arc_agi_20260306_104927

# Specify agent directly
python scripts/eval_test_set.py --task aime \
  --agent-dir ../robophd_runs/robophd/aime_20260227_180324/agents/iter4_verified_enumerator

# With repeats and config overrides
python scripts/eval_test_set.py --task aime --run-dir ... \
  --test-repeats 5 --task-config '{"solver_model": "gpt-4.1"}'
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
| Text2SQL | `{"eval_instructions": "eval_instructions.md", "database_analysis_code": "tools/analyze_db.py", "verify_prompt": "verify_prompt.md"}` | `RoboPhD/text2sql_agents/naive/` |
| ARC-AGI | `{"agent_code": "agent.py"}` | `RoboPhD/arcagi_agents/baseline/` |
| CodeCritic | `{"eval_instructions": "eval_instructions.md", "tool_code": "tools/problem_analyzer.py"}` | `RoboPhD/codegen_agents/naive_critic/` |

Conversion between agent directories and flat candidate dicts is handled by `candidate_utils.py` (`extract_candidate`, `materialize_candidate`).

## Evolution System

### Evolution Strategies
Evolution strategies are organized by domain:

- `RoboPhD/evolution_strategies/` — `cross_pollination`, `refinement`, `use_your_judgment` (used by all domains)

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
    "evolution_strategy": "use_your_judgment",
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
python scripts/run_robophd.py --task codegen --num-iterations 10 \
  --engine-config '{
    "use_weighted_random": true,
    "weighted_random_configs": [
      [{"evolution_strategy": "use_your_judgment"}, 50],
      [{"evolution_strategy": "refinement"}, 30],
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
    "new_agent_test_round_offset": -2,
    "evolution_model": "opus-4.6",
    "eval_model": "haiku-4.5"
  }'
```

- `"new_agent_test_rounds": 0`: Planning + implementation only
- `"new_agent_test_rounds": 1`: Adds testing against 1 prior iteration
- `"new_agent_test_rounds": 2`: Adds testing against 2 prior iterations [DEFAULT]
- `"new_agent_test_round_offset": -2`: Starting offset from current iteration [DEFAULT]. At iteration 8, tests against iterations 6 and 5. Use `-1` for legacy behavior (tests 7 and 6). Iterations < 1 are skipped.

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

## Key Classes and Files

### Entry Points
- **`scripts/run_robophd.py`**: Multi-agent ELO evolution runner
- **`scripts/run_gepa.py`**: GEPA optimization runner
- **`scripts/eval_test_set.py`**: Standalone test-set evaluation for any agent

### Task Registry
- **`tasks/base.py`**: `TaskDefinition` dataclass (name, evaluator_factory, dataset_builder, file_mapping, objective)
- **`tasks/codegen.py`**: CodeGen task — LiveCodeBench critic evolution
- **`tasks/aime.py`**: AIME task — math reasoning prompt evolution
- **`tasks/text2sql.py`**: Text2SQL task — BIRD benchmark SQL generation
- **`tasks/arc_agi.py`**: ARC-AGI task — abstract reasoning agent evolution

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
- **`adapters/gepa_arc_agi.py`**: ARC-AGI evaluator, TrackedLLM (with cost fix), dataset splits
- **`adapters/arc_agi_utils_unmodified.py`**: Vendored GEPA utils (exact copy, do not modify)

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
- **Eval Timeout**: `eval_timeout` (300s default) — per-evaluation timeout on `future.result()` in all ThreadPoolExecutor eval loops (domain.py, run_gepa.py, eval_test_set.py). Timed-out evals score 0 with `"error": "timeout"` in result.json. The hung thread keeps burning CPU until process exit (Python limitation); `domain.py` tracks leaked thread count across iterations and warns at each iteration start.
- **CodeGen Timeouts**: `codegen_timeout` (1200s) for solution generation, `critic_timeout` (600s) for critic/revision/acceptance
- **API Key**: Set via `ANTHROPIC_API_KEY_FOR_ROBOPHD` environment variable

## Development Tips

- **Quick Test**: `python scripts/run_robophd.py --task codegen --num-iterations 2 --engine-config '{"examples_per_iteration": 1}'`
- **List Parameters**: `python scripts/run_robophd.py --task codegen --list-params`
- **Check Progress**: Review `checkpoint.json` and `final_report.md` in the experiment dir
- **Debug Evaluation**: Check `iteration_XXX/agent_YYY/problems/` and `evaluation.json` in the experiment dir
- **Evolution Output**: Check `evolution_output/iteration_XXX/` in the experiment dir for Claude's reasoning
- **Run Outputs**: All runs land in `../robophd_runs/` (`robophd/` for ELO, `gepa/` for GEPA)
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
- **CodeGen**: See [docs/claude/codegen.md](docs/claude/codegen.md) for test execution issues
- **Text2SQL**: See [docs/claude/text2sql.md](docs/claude/text2sql.md) for database-related issues
- **ARC-AGI**: Requires `requirements-gepa.txt` (dspy, datasets) and `OPENROUTER_API_KEY`. Default solver: `gemini-3.1-flash-lite-preview` via OpenRouter. Cost tracking uses `resp.usage.cost` from OpenRouter (litellm's pricing DB doesn't cover these models).

## License

MIT License - see LICENSE file for details.
