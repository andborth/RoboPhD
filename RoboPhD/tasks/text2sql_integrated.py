"""
Text2SQL integrated task: agent.py controls SQL generation with llm() and test_sql().

Unlike text2sql (rigid pipeline: analyze_db → eval_instructions → verify_prompt),
this gives the agent full control via callables. The agent can design its own
strategy: generate-test-refine loops, multi-candidate selection, etc.
"""

from typing import Any, Dict, List

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build a Text2SQLIntegratedEvaluator from merged config."""
    from RoboPhD.adapters.text2sql_integrated import Text2SQLIntegratedEvaluator
    from pathlib import Path

    work_dir = config.get("work_dir")
    if work_dir is None:
        work_dir = Path(config.get("output_dir", "gepa_runs/work")) / "work"

    return Text2SQLIntegratedEvaluator(
        eval_model=config.get("eval_model", "haiku-4.5"),
        dataset=config.get("dataset", "train-filtered"),
        work_dir=work_dir,
        cost_budget=config.get("cost_budget", 0.10),
        over_budget_penalty=config.get("over_budget_penalty", 0.9),
        max_test_sql_calls=config.get("max_test_sql_calls", 5),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build BIRD dataset."""
    from RoboPhD.adapters.gepa_text2sql import build_text2sql_dataset
    return build_text2sql_dataset(config)


def make_text2sql_integrated_task() -> TaskDefinition:
    """Create the Text2SQL integrated task."""
    from RoboPhD.adapters.text2sql_integrated import (
        TEXT2SQL_INTEGRATED_FILE_MAPPING,
        BACKGROUND,
        OBJECTIVE,
    )

    return TaskDefinition(
        name="text2sql_integrated",
        description="Evolve Text2SQL agents with full control via llm() and test_sql() callables",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=TEXT2SQL_INTEGRATED_FILE_MAPPING,
        default_seed_agent="RoboPhD/text2sql_integrated_agents/baseline",
        objective=OBJECTIVE,
        background=BACKGROUND,
        diagnostic_files={
            "question.md": "Question text and evidence",
            "analysis.md": "Database analysis from tool",
            "predicted_sql.md": "Final SQL returned by agent",
            "expected_sql.md": "Ground truth SQL",
            "predicted_results.md": "Result set from predicted SQL",
            "ground_truth_results.md": "Result set from ground truth SQL",
            "result_comparison.md": "Match status and cost",
            "agent_trace.md": "LLM calls and test_sql calls with results",
            "agent_stdout": "Captured print() output from agent",
            "tool_stdout": "Captured print() output from analyze_db.py",
            "error.md": "Agent execution error (if any)",
        },
        config_defaults={
            "dataset": "train-filtered",
            "eval_model": "haiku-4.5",
            "cost_budget": 0.10,
            "over_budget_penalty": 0.9,
            "max_test_sql_calls": 5,
            "evaluation_budget": 1500,
        },
        test_overrides={"dataset": "dev", "max_test_workers": 4},
    )
