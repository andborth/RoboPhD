"""
DocFinQA task definition: evolve agent code for long-document financial QA.

Targets gpt-4.1-mini for reasoning and text-embedding-3-small for retrieval.
"""

from typing import Any, Dict, List, Tuple

from .base import TaskDefinition


def _evaluator_factory(config: Dict[str, Any]):
    """Build a DocFinQAEvaluator from merged config."""
    from RoboPhD.adapters.gepa_docfinqa import DocFinQAEvaluator

    return DocFinQAEvaluator(
        model=config.get("model", "gpt-4.1-mini"),
        embed_model=config.get("embed_model", "text-embedding-3-small"),
        cost_budget=config.get("cost_budget", 0.10),
        over_budget_penalty=config.get("over_budget_penalty", 0.9),
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build DocFinQA dataset.

    For run_robophd (split="train"): returns train+val concatenated (6,515 problems).
    For test (split="test"): returns test split (922 problems).
    """
    from RoboPhD.adapters.gepa_docfinqa import load_docfinqa

    split = config.get("split", "train")
    if split == "test":
        return load_docfinqa("test")
    return load_docfinqa("train") + load_docfinqa("validation")


def _gepa_datasets_builder(config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Pre-split datasets for GEPA.

    Default: train=5735, val=780 (uses HF validation split as val).
    Override val_size in task-config for testing:
        --task-config '{"val_size": 20}'
    """
    from RoboPhD.adapters.gepa_docfinqa import load_docfinqa

    train = load_docfinqa("train")
    val = load_docfinqa("validation")

    val_size = config.get("val_size")
    if val_size is not None:
        # Move unused val examples to train
        train = train + val[val_size:]
        val = val[:val_size]

    return train, val


def make_docfinqa_task() -> TaskDefinition:
    """Create the DocFinQA task definition."""
    from RoboPhD.adapters.gepa_docfinqa import FILE_MAPPING, BACKGROUND, OBJECTIVE

    return TaskDefinition(
        name="docfinqa",
        description="Evolve financial QA agents over long SEC filings (DocFinQA benchmark)",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=FILE_MAPPING,
        default_seed_agent="RoboPhD/docfinqa_agents/baseline",
        objective=OBJECTIVE,
        background=BACKGROUND,
        diagnostic_files={},
        config_defaults={
            "split": "train",
            "model": "gpt-4.1-mini",
            "embed_model": "text-embedding-3-small",
            "cost_budget": 0.10,
            "over_budget_penalty": 0.9,
            "evaluation_budget": 1500,
            "val_size": None,
        },
        test_overrides={"split": "test"},
        gepa_datasets_builder=_gepa_datasets_builder,
    )
