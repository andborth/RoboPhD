"""
Text2SQL task definition: evolve database analysis agents for BIRD benchmark.

Tool-only Phase 1 (analyze_db.py) + LLM Phase 2 (SQL generation) +
evolvable verification loop.
"""

from pathlib import Path
from typing import Any, Dict, List

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build a Text2SQLEvaluator from merged config."""
    from RoboPhD.adapters.gepa_text2sql import Text2SQLEvaluator

    work_dir = config.get("work_dir")
    if work_dir is None:
        work_dir = Path(config.get("output_dir", "gepa_runs/work")) / "work"

    return Text2SQLEvaluator(
        eval_model=config.get("eval_model", "haiku-4.5"),
        dataset=config.get("dataset", "train-filtered"),
        use_evidence=config.get("use_evidence", True),
        verification_retries=config.get("verification_retries", 2),
        temperature_strategy=config.get("temperature_strategy", "fixed"),
        output_dir=config.get("output_dir"),
        work_dir=work_dir,
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build Text2SQL dataset from BIRD benchmark files."""
    from RoboPhD.adapters.gepa_text2sql import build_text2sql_dataset
    return build_text2sql_dataset(config)


_TEXT2SQL_BACKGROUND = """\
## BIRD Benchmark Text2SQL

The Text2SQL domain generates SQL queries from natural language questions
against the BIRD benchmark (BIg Bench for LaRge-scale Database Grounded
Text-to-SQL Evaluation).

### Architecture: Tool + LLM + Verification

```
Phase 1 (Tool): analyze_db.py examines database.sqlite
  -> Produces schema analysis text (cached per code+database)

Phase 2 (LLM): eval_instructions.md + analysis + question
  -> Generates initial SQL query

Verification Loop (up to k retries):
  Execute SQL -> Summarize results -> verify_prompt -> CORRECT or new SQL

Scoring: set(predicted_results) == set(ground_truth_results)
```

### What Evolution Controls

The evolved agent consists of three files:

1. **`eval_instructions.md`** — System prompt for the eval LLM.
   Guides SQL generation strategy, dialect handling, evidence usage.

2. **`tools/analyze_db.py`** — Database analysis script.
   Reads `database.sqlite` from its working directory, performs schema
   analysis, and writes findings to `tool_output/analysis.txt`.
   The eval LLM uses this analysis alongside eval_instructions to
   generate SQL. Common techniques: DDL extraction, sample data,
   foreign key mapping, column statistics.

3. **`verify_prompt.md`** — Verification prompt template.
   Used in the verification loop to assess whether generated SQL
   correctly answers the question. Can be evolved to improve
   self-correction behavior.

### Scoring

- **Correct**: `set(predicted_results) == set(ground_truth_results)`
- Row order is ignored, duplicates are removed (BIRD methodology)
- Score: 1.0 if match, 0.0 otherwise

### Dataset: BIRD Benchmark

- **train-filtered** (default): 6,601 questions across 69 databases
- **dev**: 1,534 questions across 11 databases
- All databases are SQLite
"""


def make_text2sql_task() -> TaskDefinition:
    """Create the Text2SQL task definition."""
    from RoboPhD.adapters.gepa_text2sql import TEXT2SQL_FILE_MAPPING

    return TaskDefinition(
        name="text2sql",
        description="Evolve database analysis agents for BIRD benchmark SQL generation",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=TEXT2SQL_FILE_MAPPING,
        default_seed_agent="RoboPhD/text2sql_agents/naive",
        objective=(
            "Optimize the database analysis tool and SQL generation instructions "
            "to maximize accuracy on BIRD benchmark questions. "
            "The analyze_db.py script examines database schemas, "
            "eval_instructions.md guides the LLM's SQL generation, "
            "and verify_prompt.md controls the self-correction verification loop."
        ),
        background=_TEXT2SQL_BACKGROUND,
        diagnostic_files={
            "question.md": "Question text and evidence",
            "analysis.md": "Database analysis from tool",
            "predicted_sql.md": "Generated SQL query",
            "expected_sql.md": "Ground truth SQL query",
            "predicted_results.md": "Result set from predicted SQL",
            "ground_truth_results.md": "Result set from ground truth SQL",
            "result_comparison.md": "Truncated/deduplicated flags and match status",
            "verification_attempts.md": "Per-attempt SQL, results, and verifier decisions",
            "error.md": "SQL execution error (if any)",
        },
        config_defaults={
            "dataset": "train-filtered",
            "eval_model": "haiku-4.5",
            "use_evidence": True,
            "verification_retries": 2,
            "temperature_strategy": "fixed",
        },
    )
