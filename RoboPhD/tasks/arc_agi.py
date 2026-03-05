"""
ARC-AGI task definition: evolve agent code for abstract reasoning.

Targets Gemini 3 Flash by default, matching the GEPA examples/arc_agi setup.
"""

from typing import Any, Dict, List

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build an ArcAGIEvaluator from merged config."""
    from RoboPhD.adapters.gepa_arc_agi import ArcAGIEvaluator

    work_dir = config.get("work_dir")
    if work_dir is None:
        from pathlib import Path
        work_dir = Path(config.get("output_dir", "gepa_runs/work")) / "work"

    return ArcAGIEvaluator(
        solver_model=config.get("solver_model", "openrouter/google/gemini-3-flash-preview"),
        work_dir=work_dir,
        max_llm_calls=config.get("max_llm_calls", 10),
        reasoning_effort=config.get("reasoning_effort", "high"),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build ARC-AGI dataset from HuggingFace."""
    from RoboPhD.adapters.gepa_arc_agi import build_arc_agi_dataset

    split = config.get("arc_agi_split", "train")
    return build_arc_agi_dataset(split=split)


def make_arc_agi_task() -> TaskDefinition:
    """Create the ARC-AGI task definition."""
    from RoboPhD.adapters.gepa_arc_agi import ARC_AGI_FILE_MAPPING, BACKGROUND, OBJECTIVE

    return TaskDefinition(
        name="arc_agi",
        description="Evolve ARC-AGI solving agents (Gemini 3 Flash)",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=ARC_AGI_FILE_MAPPING,
        default_seed_agent="RoboPhD/arcagi_agents/baseline",
        objective=OBJECTIVE,
        background=BACKGROUND,
        diagnostic_files={
            "grid_comparison.md": "Per-example grid comparison results",
            "error.md": "Agent execution error (if any)",
        },
        config_defaults={
            "solver_model": "openrouter/google/gemini-3-flash-preview",
            "arc_agi_split": "train",
            "evaluation_budget": 3000,
        },
        test_overrides={"arc_agi_split": "test"},
    )
