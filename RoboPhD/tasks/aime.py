"""
AIME task definition: optimize system prompts for math competition problems.

Targets gpt-4.1-mini by default, matching the GEPA blog post setup.
"""

from typing import Any, Dict, List

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build an AIMEEvaluator from merged config."""
    from RoboPhD.adapters.gepa_aime import AIMEEvaluator

    work_dir = config.get("work_dir")
    if work_dir is None:
        from pathlib import Path
        work_dir = Path(config.get("output_dir", "gepa_runs/work")) / "work"

    return AIMEEvaluator(
        solver_model=config.get("solver_model", "gpt-4.1-mini"),
        work_dir=work_dir,
        debug_log_probability=config.get("debug_log_probability", 0.0),
        debug_log_dir=config.get("debug_log_dir"),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build AIME dataset from HuggingFace."""
    from RoboPhD.adapters.gepa_aime import build_aime_dataset

    split = config.get("aime_split", "train")
    return build_aime_dataset(split=split)


def make_aime_task() -> TaskDefinition:
    """Create the AIME task definition."""
    from RoboPhD.adapters.gepa_aime import AIME_FILE_MAPPING

    return TaskDefinition(
        name="aime",
        description="Optimize system prompts for AIME math competition (gpt-4.1-mini)",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=AIME_FILE_MAPPING,
        default_seed_agent="RoboPhD/aime_agents/baseline",
        diagnostic_files={
            "problem.md": "Problem statement",
            "response.md": "Model's full response",
            "reference_solution.md": "Reference solution (when available)",
            "system_prompt.md": "System prompt sent to the solver",
        },
        config_defaults={
            "solver_model": "gpt-4.1-mini",
            "aime_split": "train",
            "evaluation_budget": 600,
        },
        test_overrides={"aime_split": "test", "test_repeats": 5},
    )
