# Text2SQL Domain Documentation

Domain-specific documentation for the BIRD benchmark Text2SQL task.

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
