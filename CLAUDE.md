# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text-to-SQL research system achieving **73.67% accuracy** on BIRD benchmark test set using RoboPhD architecture. The system implements a three-level AI hierarchy where AI agents conduct autonomous research to improve other AI agents.

**Paper**: [RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution](https://arxiv.org/abs/XXXX.XXXXX)

## Dataset Overview

| Dataset | Total Questions | Usable Questions | Databases | Notes |
|---------|----------------|------------------|-----------|-------|
| train | 9,428 | ~9,300 | 62 | Original BIRD training set; only retail_world blacklisted |
| train-filtered | 6,601 | 6,601 (100%) | 69 | BIRD23 curated subset; all databases working **DEFAULT** |
| train-no-evidence | 6,601 | 6,601 (100%) | 69 | Same as train-filtered with all evidence fields cleared |
| dev | 1,534 | 1,534 | 11 | Development set; all databases working |
| dev-no-evidence | 1,534 | 1,534 | 11 | Same as dev with all evidence fields cleared |

**train-filtered breakdown:**
- 6,601 fully usable questions (100%) across all 69 databases
- Represents 70.0% of original 9,428 train questions with improved quality
- All previously problematic databases fixed via proper extraction

## Key Commands

### Environment Setup
```bash
export ANTHROPIC_API_KEY="your_key"
pip install -r requirements.txt

# Install Claude Code CLI (required for evolution)
# See: https://docs.anthropic.com/en/docs/claude-code
```

### Ground Truth Pre-Computation (Recommended)
Pre-compute ground truth results to prevent "database is locked" errors during research runs.

```bash
# Pre-compute for train-filtered (default) or dev dataset
python RoboPhD/tools/precompute_ground_truth.py
python RoboPhD/tools/precompute_ground_truth.py --dataset dev

# Use --max-concurrent 2 if hitting file descriptor limits
# Use --timeout 600 for slow databases (default: 300s)
```

**Caching behavior:**
- Results cached up to 2500 rows per query
- Run after deleting cache/ or switching datasets

### Basic Usage
```bash
# Run with defaults (train-filtered dataset, 6,601 questions, 100% usable)
python RoboPhD/researcher.py --num-iterations 10

# Use a pre-configured config file
python RoboPhD/researcher.py --num-iterations 10 --config configs/primary_production.json

# Quick test with custom config
python RoboPhD/researcher.py \
  --num-iterations 2 \
  --config '{"databases_per_iteration": 3, "questions_per_database": 10}'
```

### Configuration via --config
All parameters can be configured via `--config` (JSON string or file path):

```bash
# Use different dataset and models
python RoboPhD/researcher.py \
  --num-iterations 10 \
  --config '{"dataset": "train", "eval_model": "sonnet-4.5", "analysis_model": "opus-4.5"}'

# Load config from file
python RoboPhD/researcher.py --num-iterations 10 --config configs/primary_production.json
```

**Note**: Both `"questions-per-database"` (CLI-style) and `"questions_per_database"` (Python-style) work - hyphens are automatically converted to underscores.

### Dev Set Evaluation Mode
```bash
# Evaluate on dev set (with evidence)
python RoboPhD/researcher.py \
  --dev-eval \
  --config '{"initial_agents": ["opus_best"], "eval_model": "haiku-4.5"}'

# Evaluate on dev-no-evidence set (evidence fields cleared)
python RoboPhD/researcher.py \
  --dev-no-evidence-eval \
  --config '{"initial_agents": ["opus_best"], "eval_model": "haiku-4.5"}'
```

### Resume and Extend
```bash
# Resume from checkpoint (auto-continues from last completed iteration)
python RoboPhD/researcher.py --resume output/robophd_20251031_043607

# Restart from specific iteration with modifications
python RoboPhD/researcher.py \
  --resume output/robophd_20251031_043607 \
  --from-iteration 5 \
  --modify-config '{"databases_per_iteration": 10, "eval_model": "sonnet-4.5"}'

# Extend completed run with additional iterations
python RoboPhD/researcher.py \
  --resume output/robophd_20251031_043607 \
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
Evolved prompts/agents guide SQL generation with discovered optimizations.

## System Architecture: The Three AI Calls

The system orchestrates three distinct AI model calls to achieve accurate SQL generation:

```
                            ITERATION CYCLE
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  ┌──────────────────┐         ┌────────────────────┐        │
    │  │   EVOLUTION AI   │ Creates │   3-ARTIFACT       │        │
    │  │   (Opus-4.5)     │────────▶│   AGENT PACKAGE    │        │
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
    │  │   EVALUATION      │       │  DB ANALYSIS AI    │         │
    │  │   RESULTS         │       │  (Tool-only or LLM)│         │
    │  │ • Successes       │       └────────────────────┘         │
    │  │ • Failures        │                │                     │
    │  │ • Error Patterns  │                │ Analyzes            │
    │  └────────┬──────────┘                │ Database            │
    │           │                           ▼                     │
    │           │                   ┌────────────────────┐        │
    │           │                   │ DATABASE-SPECIFIC  │        │
    │           │                   │ ANALYSIS           │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            │ Combined with      │
    │           │                            │ eval_instructions  │
    │           │                            ▼                    │
    │           │                   ┌────────────────────┐        │
    │           │                   │  SQL EVAL AI       │        │
    │           │                   │  (Haiku or Sonnet) │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            │ Generates          │
    │           │                            ▼                    │
    │           │                   ┌────────────────────┐        │
    │           │                   │   SQL QUERIES      │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            │ Evaluated          │
    │           │                            ▼                    │
    │           └────────────────────────────┘                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### The Three AI Calls Explained:

1. **Evolution AI (Opus-4.5)** - Runs once per iteration
   - **Input**: Performance results from previous iteration, error analysis, agent rankings
   - **Output**: New 3-artifact agent package (agent.md, eval_instructions.md, tools/)
   - **Purpose**: Learn from failures and evolve better agents

2. **Database Analysis AI** - Runs once per database
   - **What it needs to do**: Analyze database schema and produce comprehensive documentation that enables accurate SQL generation without direct database access
   - **How it's accomplished**: Three distinct strategy patterns have emerged:
     - **Tool-only**: Deterministic Python scripts do all analysis ($0.00, fast, consistent)
     - **Agent-centric**: Agent uses natural language reasoning to analyze schema
     - **Hybrid approach**: Agent provides high-level analysis while tools handle specific technical details
   - **Input**: agent.md instructions + database schema (NO questions)
   - **Output**: Database-specific analysis describing tables, relationships, patterns

3. **SQL Evaluation AI (Haiku-4.5 or Sonnet-4.5)** - Runs once per question
   - **Input**: Database analysis + eval_instructions.md + question + evidence
   - **Output**: SQL query
   - **Purpose**: Generate accurate SQL based on the provided context

### Key Insights:
- **The eval model never sees the database** - it relies entirely on the analysis agent's output
- **Evolution learns from mistakes** - each iteration analyzes what went wrong and evolves
- **Separation of concerns** - database analysis is separate from SQL generation instructions
- **Strategy diversity**: Successful agents use different mixes of agent reasoning vs tool automation
- **Iterative improvement** - the system gets better with each evolution cycle

## Unified Agent Architecture

### Agent Structure
- **Source Location**: `RoboPhD/agents/` (default) or custom via config `agents_directory`
- **Runtime Location**: `<experiment_dir>/agents/` (agents copied to each experiment workspace)
- **Structure**: Self-contained directories with three artifacts
- **Three-Artifact Format**:
  - `agent.md`: Database analysis agent with model configuration
  - `eval_instructions.md`: Direct SQL generation instructions for eval model
  - `tools/`: Optional Python/shell analysis scripts

### Included Agents

| Agent | Description | Dev Accuracy |
|-------|-------------|--------------|
| `naive` | Baseline agent | 57-69% |
| `opus_best` | Best Opus-4.5 evolved agent | 71.3% |
| `sonnet_best` | Best Sonnet-4.5 evolved agent | 69.2% |
| `haiku_best` | Best Haiku-4.5 evolved agent | 66.1% |

### Tool-Only Execution Mode

Agents can use **tool-only execution mode** where a Python/shell script generates complete database analysis output, bypassing the AI agent entirely.

**Benefits**: Speed (1-5s vs 30-60s), cost ($0.00 for Phase 1), consistency (deterministic output), debuggability (clear code vs AI reasoning).

**YAML Frontmatter Configuration**:
```yaml
---
execution_mode: tool_only
tool_command: python tools/comprehensive_analyzer.py
tool_output_file: tool_output/schema_analysis.txt
---
```

**Execution**: System runs tool command (300s timeout), verifies output (exit 0, file exists, ≥200 bytes), copies to `output/agent_output.txt`. Falls back to normal agent execution on any failure.

## Evolution System

### Evolution Strategies
Evolution strategies are loaded from `RoboPhD/evolution_strategies/`:

**Tool-only variants** (deterministic, no LLM in DB analysis - recommended):
- `cross_pollination_tool_only`: Cross-pollination with emphasis on combining tool-only patterns
- `refinement_tool_only`: Refinement with emphasis on tool-only execution
- `research_driven_tool_only`: Research-driven with emphasis on implementing insights as tool-only

**Neutral variants** (allow LLM in DB analysis):
- `cross_pollination_neutral`: Cross-pollination presenting multiple approaches with equal weight
- `refinement_neutral`: Refinement presenting multiple approaches with equal weight
- `research_driven_neutral`: Research-driven presenting multiple approaches with equal weight

**Selection strategies**:
- `challenger`: Skip evolution, test under-tested high-ELO agents
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
    "evolution_model": "opus-4.5",
    "eval_model": "haiku-4.5"
  }'
```

- `"new_agent_test_rounds": 0`: Planning + implementation only
- `"new_agent_test_rounds": 1`: Adds testing against 1 prior iteration
- `"new_agent_test_rounds": 2`: Adds testing against 2 prior iterations [DEFAULT]

### Meta-Evolution
Meta-evolution allows evolving the evolution strategies themselves:

```bash
python RoboPhD/researcher.py \
  --num-iterations 20 \
  --config '{
    "meta_evolution_strategy": "train_a_new_winner",
    "config_schedule": {
      "10": {"meta_evolution_strategy": "train_a_new_winner"},
      "11": {"meta_evolution_strategy": null}
    }
  }'
```

Available meta-evolution strategies:
- `train_a_new_winner`: Evolve strategies to beat current best agent
- `train_a_new_champion`: Evolve strategies for consistent top performance

## Key Classes and Files

### Core Research Components
- **`researcher.py`**: Main entry point and experiment orchestrator
- **`evolution.py`**: Evolution strategy selector and orchestration
- **`deep_focus_evolution_manager.py`**: Multi-round evolution with testing
- **`meta_evolution_manager.py`**: Meta-evolution for strategy improvement
- **`agent_orchestrator.py`**: Phase 1 database analysis orchestration

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

### BIRD Evaluation Methodology

**CRITICAL**: Accuracy is based on comparing query RESULTS, not SQL syntax.

```python
# Set comparison - row order ignored, duplicates removed
set(predicted_results) == set(ground_truth_results)
```

**What this means**:
- Different SQL queries can be equally correct if they produce the same result set
- Row order is completely ignored
- Duplicates are ignored
- Column order must match

### Model Configuration
- **API Models**: opus-4.5 ($15/$75/MTok), sonnet-4.5 ($3/$15), haiku-4.5 ($1/$5/MTok)
- **Timeouts**: 1800s (30 minutes) default for phase1, phase2, sql, and evolution
- **API Key**: Set via `ANTHROPIC_API_KEY` environment variable

## Development Tips

- **Quick Test**: Use `--config '{"databases_per_iteration": 1, "questions_per_database": 5}'`
- **Check Progress**: Review `checkpoint.json` and `final_report.md`
- **Debug Phase 1**: Check `iteration_XXX/agent_YYY/DB_NAME/` for agent outputs
- **Evolution Output**: Check `evolution_output/iteration_XXX/` for Claude's reasoning
- **Config Files**: Save common configs to JSON files and use `--config path/to/config.json`

## Troubleshooting

### Memory (OOM) Errors
- **Symptom**: Process killed with "zsh: killed"
- **Solution**: Use `--config '{"max_concurrent_dbs": 4}'` or reduce to 2

### Evolution Failures
- **Claude CLI not found**: Ensure Claude Code CLI is installed
- **Context too long**: Use `--config '{"questions_per_database": 10, "databases_per_iteration": 3}'`
- **Session errors**: Check Claude CLI authentication with `claude --version`

### Database Locks
- **Symptom**: "Database is locked" errors
- **Solution**: Run `python RoboPhD/tools/precompute_ground_truth.py` before research runs

## License

MIT License - see LICENSE file for details.
