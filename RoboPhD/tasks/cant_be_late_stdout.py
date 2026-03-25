"""
Can't Be Late task with stdout capture from agent subprocess.

Separate from cant_be_late to avoid conflating results — cant_be_late_stdout
captures print() output from the agent's _step() method, which may make
evolution more effective by letting it design its own diagnostics.
"""

from typing import Any, Dict, List, Tuple

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build a CantBeLateStdoutEvaluator from merged config."""
    from RoboPhD.adapters.cant_be_late_stdout import CantBeLateStdoutEvaluator

    return CantBeLateStdoutEvaluator(
        simulation_timeout=config.get("simulation_timeout", 300),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build Can't Be Late dataset."""
    from RoboPhD.adapters.cant_be_late import load_cant_be_late_dataset

    split = config.get("cant_be_late_split", "train")
    dataset_root = config.get("dataset_root")
    max_traces = config.get("max_traces_per_split")

    ds = load_cant_be_late_dataset(
        dataset_root=dataset_root,
        max_traces_per_split=max_traces,
    )

    if split == "test":
        return ds["test"]
    return ds["train"] + ds["val"]


def _gepa_datasets_builder(config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Pre-split datasets for GEPA."""
    from RoboPhD.adapters.cant_be_late import load_cant_be_late_dataset

    dataset_root = config.get("dataset_root")
    max_traces = config.get("max_traces_per_split")

    ds = load_cant_be_late_dataset(
        dataset_root=dataset_root,
        max_traces_per_split=max_traces,
    )

    train, val = ds["train"], ds["val"]
    train_size = config.get("train_size")
    val_size = config.get("val_size")
    if val_size is not None:
        train = train + val[val_size:]
        val = val[:val_size]
    if train_size is not None:
        train = train[:train_size]
    return train, val


def make_cant_be_late_stdout_task() -> TaskDefinition:
    """Create the Can't Be Late task with stdout capture."""
    from RoboPhD.adapters.cant_be_late_stdout import (
        CANT_BE_LATE_FILE_MAPPING,
        BACKGROUND,
        OBJECTIVE,
    )

    return TaskDefinition(
        name="cant_be_late_stdout",
        description="Evolve cloud scheduling strategies with stdout capture from agent subprocess",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=CANT_BE_LATE_FILE_MAPPING,
        default_seed_agent="RoboPhD/cant_be_late_stdout_agents/baseline",
        objective=OBJECTIVE,
        background=BACKGROUND,
        diagnostic_files={
            "summary.md": "Simulation result: cost, timeline, spot availability",
            "error.md": "Strategy syntax/simulation error (if any)",
            "agent_stdout": "Captured print() output from the agent's _step() method",
        },
        config_defaults={
            "cant_be_late_split": "train",
            "simulation_timeout": 300,
            "dataset_root": None,
            "max_traces_per_split": None,
            "train_size": None,
            "evaluation_budget": 1500,
        },
        test_overrides={"cant_be_late_split": "test"},
        gepa_datasets_builder=_gepa_datasets_builder,
    )
