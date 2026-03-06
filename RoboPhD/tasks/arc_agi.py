"""
ARC-AGI task definition: evolve agent code for abstract reasoning.

Targets Gemini 3 Flash by default, matching the GEPA examples/arc_agi setup.
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

    return ArcAGIEvaluator(
        solver_model=config.get("solver_model", "openrouter/google/gemini-3-flash-preview"),
        work_dir=work_dir,
        max_llm_calls=config.get("max_llm_calls", 10),
        reasoning_effort=config.get("reasoning_effort", "high"),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build ARC-AGI dataset.

    For run_robophd (split="train"): returns train+val concatenated (400 problems).
    For test (split="test"): returns HF evaluation (400 problems).
    """
    from RoboPhD.adapters.gepa_arc_agi import load_arc_splits

    split = config.get("arc_agi_split", "train")
    train, val, test = load_arc_splits()
    if split == "test":
        return test
    return train + val


def _gepa_datasets_builder(config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Pre-split datasets for GEPA: train=200, val=200 matching GEPA exactly."""
    from RoboPhD.adapters.gepa_arc_agi import load_arc_splits

    train, val, _ = load_arc_splits()
    return train, val


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
        gepa_datasets_builder=_gepa_datasets_builder,
    )
