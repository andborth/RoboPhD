---
name: record-result
description: Record a test-set evaluation result to the results JSON for COLM 2026
allowed-tools: [Read, Edit, Bash, Grep]
disable-model-invocation: true
argument-hint: <path-to-run-or-agent-dir> [--baseline]
---

# Record Test Result

Record a completed test-set evaluation to the results JSON file for tracking.

## Input

`$ARGUMENTS` = `<path> [--baseline]`

- First positional arg: path to a run directory or agent directory
- `--baseline`: optional flag indicating this is a seed/baseline agent (no evolution cost)

## Workflow

### Step 1: Parse arguments

Extract the path and `--baseline` flag from `$ARGUMENTS`.

### Step 2: Infer run dir vs agent dir

- If path contains `checkpoint.json` or `optimization_summary.json` → it's a **run directory**
- Otherwise (contains agent artifact files like `system_prompt.md`, `eval_instructions.md`, or is inside an `agents/` parent) → it's an **agent directory**

### Step 3: Infer task name

Extract from the directory name: `aime_20260227_...` → task=`aime`, `codegen_20260301_...` → task=`codegen`.

For agent dirs, walk up to find the run dir name (e.g., `.../aime_20260227_180324/agents/baseline` → `aime`).

The task name is always the prefix before the first `_` followed by a date pattern (`YYYYMMDD`).

### Step 4: Detect engine type

- If run dir contains `checkpoint.json` → **RoboPhD** engine
- If run dir contains `optimization_summary.json` → **GEPA** engine
- If agent dir only (no parent run dir with either file) → engine is "N/A" (standalone agent eval)

### Step 5: Read test results

Look for `test_results.json` in the run dir or agent dir.

Extract: `test_accuracy`, `test_correct`, `test_total`.

### Step 6: Determine the agent

- **RoboPhD + run dir**: Read `checkpoint.json`, find best agent by ELO from `performance_records`, resolve `agent_pool[name].package_dir` relative to run dir
- **GEPA + run dir**: Read `best_agent/` directory
- **agent dir**: Use directly, agent name = directory name
- **`--baseline`**: Agent is the seed/unevolved prompt

### Step 7: Read the agent's artifacts

Use the task's `file_mapping` to know which files to read:
- AIME: `system_prompt.md`
- CodeGen: `eval_instructions.md` + `tools/problem_analyzer.py`

Read the agent's main artifact(s) to understand the approach.

### Step 8: Extract evolution cost (skip if `--baseline`)

| Field | GEPA | RoboPhD |
|-------|------|---------|
| eval cost | `optimization_summary.json` → `cost.eval_cost_usd` | `checkpoint.json` → `sum(iteration_claude_costs[].eval_cost)` |
| evolution/reflection cost | `optimization_summary.json` → `cost.reflection_cost_usd` | `checkpoint.json` → `sum(iteration_claude_costs[].evolution_cost)` |
| total cost | `optimization_summary.json` → `cost.total_cost_usd` | Sum of the above two |

### Step 9: Compute inference cost per problem

- **Primary**: Read `test_eval_cost_usd` from `test_results.json`, divide by `test_total`
- **Fallback** (older runs without `test_eval_cost_usd`): Estimate from the run's `eval_cost / total_evaluations`. **Warn the user** that this is a training-eval estimate and may not match test-eval cost (e.g., different models or timeouts). Mark the value as `"inference_cost_per_problem_approximate": true` in the JSON entry.
- For baselines without either: ask the user to provide the value or skip it

Round to 3 decimal places.

### Step 10: Extract run config

- **GEPA**: From `optimization_summary.json` → evaluation_budget, val_size, train_size, max_workers, seed, reflection_model, seed_agent
- **RoboPhD**: From `checkpoint.json` → `config_manager.iteration_configs["1"]` for examples_per_iteration, evolution_model, etc. Plus `num_iterations`, `last_completed_iteration`, engine_config filename if present

### Step 11: Extract results metadata

- **GEPA**: candidates_explored, total_evaluations, best_val_score from optimization_summary
- **RoboPhD**: best_agent name, ELO, val accuracy from checkpoint performance_records; total_evaluations from sum of `iteration_fresh_evals[]`

### Step 12: Write the approach description

Read all artifacts in the agent directory (not just the file_mapping files — examine everything). Write a 2-4 sentence summary of the agent's approach and what makes it distinctive.

**Ask the user to review/edit the approach description before saving.**

### Step 13: Generate the entry and update the results file

Results file: `../robophd_runs/results/{task}.json`

**If `--baseline`**: Add/update entry in the `baselines` section (keyed by `seed_prompt` or similar identifier).

Baseline entry format:
```json
{
  "test_accuracy": 39.33,
  "test_correct": 59,
  "test_total": 150,
  "result_file": "robophd/aime_20260227_180324/baseline_test_results.json",
  "inference_cost_per_problem": 0.006,
  "approach": "...",
  "notes": "..."
}
```

**If evolved agent**: Append to the `runs` array with auto-generated ID.

ID format: `{engine_lower}-{task}-{NNN}` where NNN is the next sequential number for that engine-task combo (e.g., `gepa-aime-003`, `robophd-codegen-001`).

Evolved run entry format:
```json
{
  "id": "gepa-aime-003",
  "engine": "GEPA",
  "date": "2026-02-27",
  "run_dir": "gepa/aime_20260227_181536",
  "config": { ... },
  "results": {
    "test_accuracy": 48.0,
    "test_correct": 72,
    "test_total": 150,
    ...engine-specific fields...
  },
  "evolution_cost": {
    ...engine-specific cost fields...
  },
  "inference_cost_per_problem": 0.008,
  "approach": "...",
  "notes": "..."
}
```

For `run_dir`, store as relative path from `../robophd_runs/` (e.g., `gepa/aime_20260227_181536` not the absolute path).

For `notes`, ask the user if they want to add any notes.

### Step 14: Show the user the new entry

Print the full JSON entry that was added. Confirm it was written to the results file.

## Key data sources by engine

| Field | GEPA | RoboPhD |
|-------|------|---------|
| test results | `test_results.json` | `test_results.json` |
| eval cost | `optimization_summary.json → cost.eval_cost_usd` | `checkpoint.json → sum(iteration_claude_costs[].eval_cost)` |
| evolution cost | `optimization_summary.json → cost.reflection_cost_usd` | `checkpoint.json → sum(iteration_claude_costs[].evolution_cost)` |
| total evaluations | `optimization_summary.json → total_evaluations` | `checkpoint.json → sum(iteration_fresh_evals[])` |
| config | `optimization_summary.json` | `checkpoint.json → config_manager.iteration_configs` |
| best agent | `best_agent/` directory | `checkpoint.json → max(performance_records, key=elo)` |
| agent artifacts | `best_agent/{file_mapping files}` | `agents/{name}/{file_mapping files}` |

## Baseline vs evolved differences

| Aspect | Baseline (`--baseline`) | Evolved |
|--------|------------------------|---------|
| Section in JSON | `baselines.seed_prompt` | `runs[]` |
| evolution_cost | Omitted | Required |
| config | Omitted | Required |
| result_file | Path to test_results.json | Omitted (redundant with run_dir) |
| ID generation | Fixed key | Auto-increment `{engine}-{task}-{NNN}` |
