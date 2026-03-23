"""
ARC-AGI-1 task definition with rich diagnostics and stdout capture.

Separate from arc_agi to avoid conflating results — arc_agi_1 provides
richer feedback to the evolution AI (formatted grids, LLM traces, agent
stdout, cost visibility).
"""

from typing import Any, Dict, List, Tuple

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build an ArcAGI1Evaluator from merged config."""
    from RoboPhD.adapters.arc_agi_1 import ArcAGI1Evaluator, DEFAULT_SOLVER_MODEL
    from pathlib import Path

    work_dir = config.get("work_dir")
    if work_dir is None:
        work_dir = Path(config.get("output_dir", "gepa_runs/work")) / "work"

    return ArcAGI1Evaluator(
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
    """Pre-split datasets for GEPA: train=200, val=200."""
    from RoboPhD.adapters.gepa_arc_agi import load_arc_train_val
    train, val = load_arc_train_val()
    train_size = config.get("train_size")
    val_size = config.get("val_size")
    if val_size is not None:
        train = train + val[val_size:]
        val = val[:val_size]
    if train_size is not None:
        train = train[:train_size]
    return train, val


def make_arc_agi_1_task() -> TaskDefinition:
    """Create the ARC-AGI-1 task definition with rich diagnostics."""
    from RoboPhD.adapters.arc_agi_1 import (
        ARC_AGI_FILE_MAPPING, BACKGROUND, OBJECTIVE, DEFAULT_SOLVER_MODEL,
    )

    return TaskDefinition(
        name="arc_agi_1",
        description="Evolve ARC-AGI-1 solving agents with rich diagnostics and stdout capture",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=ARC_AGI_FILE_MAPPING,
        default_seed_agent="RoboPhD/arcagi1_agents/baseline",
        objective=OBJECTIVE,
        background=BACKGROUND,
        diagnostic_files={
            "problem.md": "The ARC-AGI problem: training I/O grids and test input/output",
            "result.md": "Score, cost, predictions vs gold grids",
            "agent_trace.md": "LLM call trajectory (prompts + responses + costs)",
            "grid_comparison.md": "Per-example PASS/FAIL with cell-level feedback",
            "agent_stdout": "Captured print() output from the agent",
            "error.md": "Agent execution error (if any)",
        },
        config_defaults={
            "solver_model": DEFAULT_SOLVER_MODEL,
            "arc_agi_split": "train",
            "evaluation_budget": 1500,
            "eval_timeout": 600,
            "train_size": None,
        },
        test_overrides={"arc_agi_split": "test"},
        gepa_datasets_builder=_gepa_datasets_builder,
    )
