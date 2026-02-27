# Text2SQL Domain Documentation

Domain-specific documentation for the BIRD benchmark Text2SQL task.

## Architecture (Legacy)

> **Note**: This architecture predates the `file_mapping` system used by CodeGen and AIME. Text2SQL will be migrated to `ExternalEvaluatorDomain` + `TaskDefinition`. See [Migration Notes](#migration-notes) below.

### The Three AI Calls

The Text2SQL system orchestrates three distinct AI model calls per iteration:

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
    │           │                   │   SQL OUTPUT       │        │
    │           │                   └────────┬───────────┘        │
    │           │                            │                    │
    │           │                            │ Evaluated          │
    │           │                            ▼                    │
    │           └────────────────────────────┘                    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

1. **Evolution AI (Opus-4.6)** — Runs once per iteration
   - **Input**: Performance results from previous iteration, error analysis, agent rankings
   - **Output**: New 3-artifact agent package (agent.md, eval_instructions.md, tools/)
   - **Purpose**: Learn from failures and evolve better agents

2. **Domain Analysis AI** — Runs once per database context
   - **What it does**: Analyze database schema and produce comprehensive documentation
   - **How**: Three strategy patterns:
     - **Tool-only**: Deterministic Python scripts do all analysis ($0.00, fast, consistent)
     - **Agent-centric**: Agent uses natural language reasoning
     - **Hybrid approach**: Agent provides high-level analysis while tools handle specifics
   - **Input**: agent.md instructions + database context (NO questions)
   - **Output**: Domain-specific schema analysis
   - **Orchestrator**: `AgentOrchestrator` (`agent_orchestrator.py`)

3. **Eval AI (usually Haiku-4.5 or Sonnet-4.5)** — Runs once per problem
   - **Input**: Domain analysis + eval_instructions.md + question
   - **Output**: SQL query
   - **Purpose**: Generate accurate SQL based on the provided context

### Three-Artifact Agent Format

Text2SQL agents are self-contained directories with three artifacts:
- `agent.md`: Domain analysis agent with model configuration and YAML frontmatter
- `eval_instructions.md`: Direct SQL generation instructions for eval model
- `tools/`: Optional Python/shell analysis scripts

### Tool-Only Execution Mode

Agents can use **tool-only execution mode** where a Python/shell script generates complete analysis output, bypassing the AI agent entirely.

**Benefits**: Speed (1-5s vs 30-60s), cost ($0.00 for Phase 1), consistency (deterministic output), debuggability (clear code vs AI reasoning).

**YAML Frontmatter Configuration** (in `agent.md`):
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

### Key Legacy Components
- **`agent_orchestrator.py`**: Phase 1 analysis orchestration
- **`core.py`**: `SQLGenerator`, `Evaluator`, `DatabaseManager`
- **`utilities/cached_sql_executor.py`**: SQL execution with caching

### Migration Notes

To migrate Text2SQL to the current architecture:

1. **Create `tasks/text2sql.py`** with a `TaskDefinition` containing:
   - `evaluator_factory`: Wrap SQL generation + evaluation in a `(candidate, example) -> (score, diagnostics)` function
   - `dataset_builder`: Load BIRD questions as flat example dicts
   - `file_mapping`: Likely `{"agent_instructions": "agent.md", "eval_instructions": "eval_instructions.md", "tool_code": "tools/..."}` or similar
2. **Wrap `AgentOrchestrator`** functionality inside the evaluator function
3. **Register the task** so `run_robophd.py --task text2sql` and `run_gepa.py --task text2sql` work
4. **Text2SQL evolution strategies** (`evolution_strategies_text2sql/`) would continue to work via the `evolution_strategies_directory` config

---

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

## Ground Truth Pre-Computation

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

## Basic Usage

```bash
# Run with defaults (train-filtered dataset, 6,601 questions, 100% usable)
python RoboPhD/researcher.py --num-iterations 10

# Use a pre-configured config file
python RoboPhD/researcher.py --num-iterations 10 --config configs/primary_production.json

# Quick test with custom config
python RoboPhD/researcher.py \
  --num-iterations 2 \
  --config '{"examples_per_iteration": 3, "problems_per_context": 10}'
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

**Note**: Both `"problems-per-context"` (CLI-style) and `"problems_per_context"` (Python-style) work - hyphens are automatically converted to underscores.

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

## BIRD Evaluation Methodology

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

## Included Agents

| Agent | Description | Dev Accuracy |
|-------|-------------|--------------|
| `naive` | Baseline agent | 57-69% |
| `opus_best` | Best Opus-4.5 evolved agent | 71.3% |
| `sonnet_best` | Best Sonnet-4.5 evolved agent | 69.2% |
| `haiku_best` | Best Haiku-4.5 evolved agent | 66.1% |

## Phase 1: Database Analysis

In Text2SQL, Phase 1 analyzes a database schema:
- **Input**: Database file (SQLite)
- **Output**: Comprehensive schema documentation (system_prompt.txt)
- **Goal**: Enable accurate SQL generation without direct database access

The Database Analysis AI (or tool-only script) examines:
- Table structures and relationships
- Column types and constraints
- Foreign key relationships
- Data patterns and distributions

## Troubleshooting

### Database Locks
- **Symptom**: "Database is locked" errors
- **Solution**: Run `python RoboPhD/tools/precompute_ground_truth.py` before research runs

### Slow Databases
- **Symptom**: Timeouts on specific databases
- **Solution**: Use `--timeout 600` for slow databases (default: 300s)

### File Descriptor Limits
- **Symptom**: "Too many open files" errors
- **Solution**: Use `--max-concurrent 2` when pre-computing ground truth
