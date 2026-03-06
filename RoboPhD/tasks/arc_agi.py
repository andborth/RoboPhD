"""
ARC-AGI task definition: evolve agent code for abstract reasoning.

Targets Gemini 3.1 Flash Lite by default (cheaper than GEPA's Gemini 3 Flash).
"""

from typing import Any, Dict, List, Tuple

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build an ArcAGIEvaluator from merged config."""
    from RoboPhD.adapters.gepa_arc_agi import ArcAGIEvaluator

    work_dir = config.get("work_dir")
    if work_dir is None:
        from pathlib import Path
        work_dir = Path(config.get("output_dir", "gepa_runs/work")) / "work"

    from RoboPhD.adapters.gepa_arc_agi import DEFAULT_SOLVER_MODEL

    return ArcAGIEvaluator(
        solver_model=config.get("solver_model", DEFAULT_SOLVER_MODEL),
        work_dir=work_dir,
        max_llm_calls=config.get("max_llm_calls", 10),
        reasoning_effort=config.get("reasoning_effort", "high"),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build ARC-AGI dataset.

    For run_robophd (split="train"): returns train+val concatenated (400 problems).
    For test (split="test"): returns HF evaluation (400 problems).
    """
    split = config.get("arc_agi_split", "train")
    if split == "test":
        from RoboPhD.adapters.gepa_arc_agi import load_arc_test
        return load_arc_test()
    from RoboPhD.adapters.gepa_arc_agi import load_arc_train_val
    train, val = load_arc_train_val()
    return train + val


def _gepa_datasets_builder(config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Pre-split datasets for GEPA: train=200, val=200 matching GEPA exactly."""
    from RoboPhD.adapters.gepa_arc_agi import load_arc_train_val
    return load_arc_train_val()


def make_arc_agi_task() -> TaskDefinition:
    """Create the ARC-AGI task definition."""
    from RoboPhD.adapters.gepa_arc_agi import (
        ARC_AGI_FILE_MAPPING, BACKGROUND, OBJECTIVE, DEFAULT_SOLVER_MODEL,
    )

    return TaskDefinition(
        name="arc_agi",
        description="Evolve ARC-AGI solving agents (Gemini 3.1 Flash Lite, $0.25 cap)",
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
            "solver_model": DEFAULT_SOLVER_MODEL,
            "arc_agi_split": "train",
            "evaluation_budget": 1500,
        },
        test_overrides={"arc_agi_split": "test"},
        gepa_datasets_builder=_gepa_datasets_builder,
    )
