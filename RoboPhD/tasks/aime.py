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
        objective=(
            "Optimize the system prompt to maximize accuracy on AIME math competition problems. "
            "The prompt should guide the model to break down problems systematically, show clear "
            "reasoning, and provide the final answer as a single integer on its own line."
        ),
        config_defaults={
            "solver_model": "gpt-4.1-mini",
            "aime_split": "train",
        },
    )
