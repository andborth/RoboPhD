# Adding a New Task

This guide explains how to add a new benchmark/domain to RoboPhD.

## Overview

A task consists of:
1. **Evaluator** (`adapters/`) — scores agent outputs on individual problems
2. **Task definition** (`tasks/`) — wires everything together
3. **Seed agent** (`{task}_agents/baseline/`) — starting point for evolution

## Step 1: Create the Evaluator

Create `adapters/my_task.py` with a callable evaluator class:

```python
class MyTaskEvaluator:
    def __init__(self, model="gpt-4.1-mini", cost_budget=0.10):
        self.model = model
        self.cost_budget = cost_budget

    def __call__(self, candidate: dict, example: dict, *, problem_dir=None) -> tuple[float, dict]:
        """
        Args:
            candidate: Dict mapping file_mapping keys to file contents (strings).
            example: One item from the dataset (e.g. {"question_id": "q1", "question": "..."}).
            problem_dir: Optional Path where diagnostic files should be written.

        Returns:
            (score, diagnostics) where score is a float and diagnostics is a dict.
            Dict values that are strings are written as individual files in problem_dir.
        """
        agent_code = candidate["agent_code"]  # matches file_mapping key

        # Execute the agent, score it, return diagnostics
        score = ...
        diagnostics = {
            "question.md": f"# Question\n{example['question']}",
            "predicted.md": f"# Agent Output\n{predicted}",
        }
        return score, diagnostics
```

**Key requirements:**
- Must be **thread-safe** — RoboPhD calls it concurrently from a `ThreadPoolExecutor`
- `candidate` keys match the task's `file_mapping` keys (not filenames)
- String values in the diagnostics dict are written as files in `problem_dir`
- Return `(0.0, {"error": "..."})` on failures

Also export the constants the task definition needs:

```python
FILE_MAPPING = {"agent_code": "agent.py"}

OBJECTIVE = "Generate correct answers for MyTask benchmark problems."

BACKGROUND = """You are evolving an agent that solves MyTask problems.
...detailed domain documentation for evolution AI..."""
```

## Step 2: Create the Task Definition

Create `tasks/my_task.py`:

```python
from typing import Any, Dict, List
from .base import TaskDefinition

def _evaluator_factory(config: Dict[str, Any]):
    from RoboPhD.adapters.my_task import MyTaskEvaluator
    return MyTaskEvaluator(
        model=config.get("model", "gpt-4.1-mini"),
        cost_budget=config.get("cost_budget", 0.10),
    )

def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    from RoboPhD.adapters.my_task import load_dataset
    split = config.get("split", "train")
    return load_dataset(split)

def make_my_task() -> TaskDefinition:
    from RoboPhD.adapters.my_task import FILE_MAPPING, BACKGROUND, OBJECTIVE
    return TaskDefinition(
        name="my_task",
        description="Short description for CLI help",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=FILE_MAPPING,
        default_seed_agent="RoboPhD/my_task_agents/baseline",
        objective=OBJECTIVE,
        background=BACKGROUND,
        diagnostic_files={
            "question.md": "Problem statement",
            "predicted.md": "Agent output",
        },
        config_defaults={
            "split": "train",
            "model": "gpt-4.1-mini",
            "evaluation_budget": 1500,
        },
        test_overrides={"split": "test"},
    )
```

## Step 3: Register the Task

Add to `tasks/__init__.py` in `_ensure_builtins()`:

```python
from .my_task import make_my_task
register_task(make_my_task())
```

Wrap in `try/except ImportError` if the task has optional dependencies.

## Step 4: Create the Seed Agent

Create `RoboPhD/my_task_agents/baseline/` with files matching your `file_mapping`:

```
RoboPhD/my_task_agents/baseline/
└── agent.py          # if file_mapping = {"agent_code": "agent.py"}
```

The seed agent should be minimal but functional — evolution will improve it.

## Step 5: Write BACKGROUND and OBJECTIVE

These are critical — they're what the evolution AI sees:

- **OBJECTIVE**: 1-2 sentences. What "better" means (e.g. "higher accuracy", "lower cost").
- **BACKGROUND**: Multi-paragraph domain documentation. Explain the problem, available APIs/callables, scoring rules, and constraints. Written as `CLAUDE.md` into evolution working directories.

## TaskDefinition Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | CLI identifier (e.g. `"my_task"`) |
| `evaluator_factory` | Yes | `(config) -> evaluator_fn` |
| `dataset_builder` | Yes | `(config) -> [examples]` |
| `file_mapping` | Yes | Candidate key -> agent file path |
| `default_seed_agent` | Yes | Path to seed agent directory |
| `objective` | No | Optimization goal for evolution AI |
| `background` | No | Domain docs for evolution AI |
| `diagnostic_files` | No | `{filename: description}` — tells evolution AI what's available |
| `config_defaults` | No | Lowest-priority config defaults |
| `test_overrides` | No | Config overrides for test-set evaluation |
| `gepa_datasets_builder` | No | `(config) -> (train, val)` for GEPA pre-split |

## Patterns from Existing Tasks

### Simplest: AIME (prompt-only evolution)
- Single text artifact: `system_prompt.md`
- No agent code, no callables
- Evaluator just sends the prompt to a solver model
- Good starting template for new tasks

### Algorithmic: Can't Be Late (no LLM at inference)
- Agent is a Python class with `reset()` and `step()` methods
- Evaluated via subprocess simulation (no LLM calls)
- Standalone Agent class receives state as explicit parameters (no framework inheritance)
- Zero inference cost — evolution is the only cost

### LLM + Callables: Text2SQL Integrated
- Agent receives `llm()` and `test_sql()` callables at runtime
- Full agent control over strategy (generate-test-refine loops, multi-candidate selection)
- Cost budget enforcement ($0.10/problem)
- Rich diagnostics: agent trace, stdout capture, SQL comparison

### LLM + Embeddings: DocFinQA
- Agent receives `llm()` and `embed()` callables
- Retrieval + QA pipeline over long documents
- Cost budget enforcement ($0.10/problem)
- Binary scoring (exact match)

### Multi-Artifact: Text2SQL (original)
- Three text files evolved together: `eval_instructions.md` + `tools/analyze_db.py` + `verify_prompt.md`
- Model for multi-artifact prompt evolution without agent code
- Each file has a distinct role in a fixed pipeline

## Verify It Works

```bash
# List tasks — yours should appear
python scripts/run_robophd.py --task my_task --list-params

# Quick evolution test
python scripts/run_robophd.py --task my_task --num-iterations 2 \
  --engine-config '{"examples_per_iteration": 3}'

# Test-set evaluation
python scripts/eval_test_set.py --task my_task \
  --agent-dir RoboPhD/my_task_agents/baseline
```
