"""
Can't Be Late task definition: evolve cloud scheduling strategies.

Targets the "Can't Be Late" problem (NSDI'24) where a strategy decides
when to use SPOT (cheap, preemptible) vs ON_DEMAND (expensive, reliable)
instances to complete a task before a deadline at minimum cost.

Scores are continuous real-valued: -cost in dollars (higher = cheaper = better).
"""

from typing import Any, Dict, List, Tuple

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build a CantBeLateEvaluator from merged config."""
    from RoboPhD.adapters.cant_be_late import CantBeLateEvaluator

    return CantBeLateEvaluator(
        simulation_timeout=config.get("simulation_timeout", 300),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build Can't Be Late dataset.

    For run_robophd (split="train"): returns train+val concatenated (~2,000 problems).
    For test (split="test"): returns test split (~1,080 problems).
    """
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
    """Pre-split datasets for GEPA: train=1,000, val=1,000.

    Override with train_size/val_size in task-config for testing:
        --task-config '{"train_size": 20, "val_size": 20}'
    """
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
        # Move unused val examples to train
        train = train + val[val_size:]
        val = val[:val_size]
    if train_size is not None:
        train = train[:train_size]
    return train, val


def make_cant_be_late_task() -> TaskDefinition:
    """Create the Can't Be Late task definition."""
    from RoboPhD.adapters.cant_be_late import (
        CANT_BE_LATE_FILE_MAPPING,
        BACKGROUND,
        OBJECTIVE,
    )

    return TaskDefinition(
        name="cant_be_late",
        description="Evolve cloud scheduling strategies for spot vs on-demand instance selection",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=CANT_BE_LATE_FILE_MAPPING,
        default_seed_agent="RoboPhD/cant_be_late_agents/baseline",
        objective=OBJECTIVE,
        background=BACKGROUND,
        diagnostic_files={
            "summary.md": "Simulation result: cost, timeline, spot availability",
            "error.md": "Strategy syntax/simulation error (if any)",
        },
        config_defaults={
            "cant_be_late_split": "train",
            "simulation_timeout": 300,
            "dataset_root": None,       # None = use default path
            "max_traces_per_split": None,  # None = use all traces
            "train_size": None,         # Override GEPA train split size
            "val_size": None,           # Override GEPA val split size
        },
        test_overrides={"cant_be_late_split": "test"},
        gepa_datasets_builder=_gepa_datasets_builder,
    )
