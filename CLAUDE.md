# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

- **Available CLI Tools**: `jq` and `tree` are installed and available
- **Text2SQL domain**: See [docs/claude/text2sql.md](docs/claude/text2sql.md)
- **CodeGen domain**: See [docs/claude/codegen.md](docs/claude/codegen.md)
- **GEPA integration**: See [docs/claude/gepa.md](docs/claude/gepa.md)

## Project Overview

RoboPhD is a multi-domain evolution system that implements a three-level AI hierarchy where AI agents conduct autonomous research to improve other AI agents.

**Text2SQL**: Achieving **73.67% accuracy** on BIRD benchmark test set.
**Paper**: [RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution](https://arxiv.org/abs/2601.01126)

**CodeGen**: Evolving critic agents for code review on LiveCodeBench.

## Domains

| Domain | Benchmark | Phase 1 Input | Phase 2 Output |
|--------|-----------|---------------|----------------|
| Text2SQL | BIRD | Database schema | SQL query |
| CodeGen | LiveCodeBench | Problem context | Revised code |

## Key Commands

### Environment Setup
```bash
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"
pip install -r requirements.txt

# Install Claude Code CLI (required for evolution)
# See: https://docs.anthropic.com/en/docs/claude-code
```

### GEPA Optimization (CodeGen)
```bash
# Smoke test (~5 mutation cycles)
python scripts/run_gepa_codegen.py \
  --seed-agent RoboPhD/codegen_agents/naive_critic \
  --max-metric-calls 200 --val-ratio 0.05

# Full run
python scripts/run_gepa_codegen.py \
  --seed-agent RoboPhD/codegen_agents/naive_critic \
  --max-metric-calls 600 --val-ratio 0.05 \
  --reflection-model opus-4.6

# Sequential (easier debugging, no ThreadPoolExecutor)
python scripts/run_gepa_codegen.py \
  --seed-agent RoboPhD/codegen_agents/naive_critic \
  --max-metric-calls 200 --max-workers 1
```

**Budget math**: Each mutation cycle costs ~minibatch (3) + val sweep (val_size). With `--val-ratio 0.05` (~39 val examples), each cycle ≈ 42 calls. Keep val small to maximize exploration within the budget.

### Resume and Extend
```bash
# Resume from checkpoint (auto-continues from last completed iteration)
python RoboPhD/researcher.py --resume evolution/robophd_20251031_043607

# Restart from specific iteration with modifications
python RoboPhD/researcher.py \
  --resume evolution/robophd_20251031_043607 \
  --from-iteration 5 \
  --modify-config '{"contexts_per_iteration": 10, "eval_model": "sonnet-4.5"}'

# Extend completed run with additional iterations
python RoboPhD/researcher.py \
  --resume evolution/robophd_20251031_043607 \
  --extend 5 \
  --modify-config '{"evolution_strategy": "challenger"}'
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
Evolved prompts/agents guide task execution with discovered optimizations.

## System Architecture: The Three AI Calls

The system orchestrates three distinct AI model calls per iteration:

```
                            ITERATION CYCLE
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  ┌──────────────────┐         ┌────────────────────┐        │
    │  │   EVOLUTION AI   │ Creates │   3-ARTIFACT       │        │
    │  │   (Opus-4.6)     │────────▶│   AGENT PACKAGE    │        │
    │  └──────────────────┘         └────────────────────┘        │
    │           ▲                            │                    │
    │           │                            ▼                    │
    │           │                   ┌────────────────────┐        │
    │           │                   │ • agent.md         │        │
    │   Analyzes│                   │ • eval_instructions.md      │
    │   Results │                   │ • tools/           │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            ▼                    │
    │  ┌───────────────────┐       ┌────────────────────┐         │
    │  │   EVALUATION      │       │  DOMAIN ANALYSIS   │         │
    │  │   RESULTS         │       │  (Tool-only or LLM)│         │
    │  │ • Successes       │       └────────────────────┘         │
    │  │ • Failures        │                │                     │
    │  │ • Error Patterns  │                │ Analyzes            │
    │  └────────┬──────────┘                │ Context             │
    │           │                           ▼                     │
    │           │                   ┌────────────────────┐        │
    │           │                   │ DOMAIN-SPECIFIC    │        │
    │           │                   │ ANALYSIS           │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            │ Combined with      │
    │           │                            │ eval_instructions  │
    │           │                            ▼                    │
    │           │                   ┌────────────────────┐        │
    │           │                   │  EVAL AI           │        │
    │           │                   │  (Haiku or Sonnet) │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            │ Generates          │
    │           │                            ▼                    │
    │           │                   ┌────────────────────┐        │
    │           │                   │   OUTPUT           │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            │ Evaluated          │
    │           │                            ▼                    │
    │           └────────────────────────────┘                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### The Three AI Calls Explained:

1. **Evolution AI (Opus-4.6)** - Runs once per iteration
   - **Input**: Performance results from previous iteration, error analysis, agent rankings
   - **Output**: New 3-artifact agent package (agent.md, eval_instructions.md, tools/)
   - **Purpose**: Learn from failures and evolve better agents

2. **Domain Analysis AI** - Runs once per context (database/problem)
   - **What it needs to do**: Analyze domain input and produce comprehensive documentation
   - **How it's accomplished**: Three strategy patterns:
     - **Tool-only**: Deterministic Python scripts do all analysis ($0.00, fast, consistent)
     - **Agent-centric**: Agent uses natural language reasoning
     - **Hybrid approach**: Agent provides high-level analysis while tools handle specifics
   - **Input**: agent.md instructions + domain context (NO questions)
   - **Output**: Domain-specific analysis

3. **Eval AI (usually Haiku-4.5 or Sonnet-4.5)** - Runs once per problem
   - **Input**: Domain analysis + eval_instructions.md + question
   - **Output**: Solution (SQL query, revised code, etc.)
   - **Purpose**: Generate accurate output based on the provided context

### Key Insights:
- **Evolution learns from mistakes** - each iteration analyzes what went wrong and evolves
- **Separation of concerns** - domain analysis is separate from solution generation
- **Strategy diversity**: Successful agents use different mixes of agent reasoning vs tool automation

## Unified Agent Architecture

### Agent Structure
- **Source Location**: `RoboPhD/agents/` (default) or custom via config `agents_directory`
- **Runtime Location**: `<experiment_dir>/agents/` (agents copied to each experiment workspace)
- **Structure**: Self-contained directories with three artifacts
- **Three-Artifact Format**:
  - `agent.md`: Domain analysis agent with model configuration
  - `eval_instructions.md`: Direct solution generation instructions for eval model
  - `tools/`: Optional Python/shell analysis scripts

### Tool-Only Execution Mode

Agents can use **tool-only execution mode** where a Python/shell script generates complete analysis output, bypassing the AI agent entirely.

**Benefits**: Speed (1-5s vs 30-60s), cost ($0.00 for Phase 1), consistency (deterministic output), debuggability (clear code vs AI reasoning).

**YAML Frontmatter Configuration** (tool-only mode):
```yaml
---
name: hybrid-comprehensive-analyzer
description: Cross-pollinated tool combining techniques from top agents
execution_mode: tool_only
tool_command: python tools/comprehensive_analyzer.py
tool_output_file: tool_output/analysis.txt
---
```

| Field | Description |
|-------|-------------|
| `name` | Short identifier for the agent |
| `description` | One-line summary of the agent's approach |
| `execution_mode` | Set to `tool_only` to bypass LLM agent |
| `tool_command` | Command to execute (runs from agent directory) |
| `tool_output_file` | Path where tool writes output (relative to agent directory) |

**Execution**: System runs tool command (300s timeout), verifies output (exit 0, file exists, ≥200 bytes), copies to `output/agent_output.txt`. Falls back to normal agent execution on any failure.

## Evolution System

### Evolution Strategies
Evolution strategies are loaded from `RoboPhD/evolution_strategies/`:

**Tool-only variants** (deterministic, no LLM in analysis):
- `cross_pollination_tool_only`: Cross-pollination with emphasis on combining tool-only patterns
- `refinement_tool_only`: Refinement with emphasis on tool-only execution
- `research_driven_tool_only`: Research-driven with emphasis on implementing insights as tool-only

**Neutral variants** (allow LLM in analysis):
- `cross_pollination_neutral`: Cross-pollination presenting multiple approaches with equal weight
- `refinement_neutral`: Refinement presenting multiple approaches with equal weight
- `research_driven_neutral`: Research-driven presenting multiple approaches with equal weight

**Note**: Meta-evolution can generate additional strategies beyond these built-in options.

**Agent selection**: Prioritizes pending winners, new agents, then untested agents. Remaining slots filled randomly from top ELO > 1500 agents (falling back to lower ELO if needed).

**Selection strategies** (skip evolution):
- `challenger`: Skip evolution, test under-tested agents (fewest tests first)
- `greedy`: Skip evolution, use deterministic top-k ELO selection
- `none`: Skip evolution, use randomized ELO-based agent selection

### Evolution Schedule Control

Evolution strategies can be controlled per-iteration using the `config_schedule` parameter:

```bash
python RoboPhD/researcher.py \
  --num-iterations 10 \
  --config '{
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
python RoboPhD/researcher.py \
  --num-iterations 10 \
  --config '{
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
python RoboPhD/researcher.py \
  --num-iterations 10 \
  --config '{
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
python RoboPhD/researcher.py \
  --num-iterations 20 \
  --config '{
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

### Core Research Components
- **`researcher.py`**: Main entry point and experiment orchestrator
- **`evolution.py`**: Evolution strategy selector and orchestration
- **`deep_focus_evolution_manager.py`**: Multi-round evolution with testing
- **`meta_evolution_manager.py`**: Meta-evolution for strategy improvement
- **`agent_orchestrator.py`**: Phase 1 analysis orchestration

### Common Utilities
- **`core.py`**: Contains `SQLGenerator`, `Evaluator`, `DatabaseManager`
- **`config.py`**: Model mappings and fallbacks
- **`config_manager.py`**: Delta-based configuration management
- **`utilities/cached_sql_executor.py`**: SQL execution with caching

## Critical Implementation Details

### ELO System
- **Tie Handling**: Agents with equal accuracy exchange 0.5 points each
- **Ranking Display**: Tied agents show same rank (e.g., #1, #1, #3)
- **K-factor**: 32 for moderate rating changes
- **Initial ELO**: 1500 for new agents

### Model Configuration
- **API Models**: opus-4.6 ($5/$25/MTok), sonnet-4.5 ($3/$15/MTok), haiku-4.5 ($1/$5/MTok)
- **Timeouts**: 1800s (30 minutes) default for phase1, phase2, sql, and evolution
- **CodeGen Timeouts**: `codegen_call_timeout` (1200s) for solution generation, `critic_call_timeout` (600s) for critic/revision/acceptance
- **API Key**: Set via `ANTHROPIC_API_KEY_FOR_ROBOPHD` environment variable

## Development Tips

- **Quick Test**: Use `--config '{"contexts_per_iteration": 1, "problems_per_context": 5}'`
- **Check Progress**: Review `checkpoint.json` and `final_report.md`
- **Debug Phase 1**: Check `iteration_XXX/agent_YYY/CONTEXT_NAME/` for agent outputs
- **Evolution Output**: Check `evolution_output/iteration_XXX/` for Claude's reasoning
- **Config Files**: Save common configs to JSON files and use `--config path/to/config.json`

## Troubleshooting

### Memory (OOM) Errors
- **Symptom**: Process killed with "zsh: killed"
- **Solution**: Use `--config '{"max_concurrent": 4}'` or reduce to 2

### Evolution Failures
- **Claude CLI not found**: Ensure Claude Code CLI is installed
- **Context too long**: Use `--config '{"problems_per_context": 10, "contexts_per_iteration": 3}'`
- **Session errors**: Check Claude CLI authentication with `claude --version`

### Domain-Specific Issues
- **Text2SQL**: See [docs/claude/text2sql.md](docs/claude/text2sql.md) for database-related issues
- **CodeGen**: See [docs/claude/codegen.md](docs/claude/codegen.md) for test execution issues

## License

MIT License - see LICENSE file for details.
