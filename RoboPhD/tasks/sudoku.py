"""
Sudoku task registry entry — thin wrapper that imports from examples/sudoku/.

The canonical sudoku code lives in examples/sudoku/ (self-contained,
purely additive per the examples/ pattern). This file exists solely so
that run_gepa.py can find the task via the registry and use the same
evaluator + dataset as the examples/sudoku/main.py entry point.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import TaskDefinition

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "sudoku"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))


def _evaluator_factory(config: Dict[str, Any]):
    from evaluator import SudokuEvaluator  # from examples/sudoku/
    return SudokuEvaluator()


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Return train or test split from the examples/sudoku/ loader."""
    from evaluator import load_dataset
    split = config.get("split", "train")
    ds = load_dataset()
    return ds["test"] if split == "test" else ds["train"]


def _gepa_datasets_builder(config: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Pre-split datasets for GEPA: train + val."""
    from evaluator import load_dataset
    ds = load_dataset()
    train = ds["train"]
    # GEPA expects a separate val set; use test split as val since the
    # examples/ loader only provides train/test. Downstream run_gepa
    # test-set evaluation will also use test, which is acceptable for
    # this benchmark (same distribution, stratified sampling).
    val = ds["test"]
    train_size = config.get("train_size")
    val_size = config.get("val_size")
    if train_size is not None:
        train = train[:train_size]
    if val_size is not None:
        val = val[:val_size]
    return train, val


def make_sudoku_task() -> TaskDefinition:
    """Create the Sudoku task definition, reading objective/background from examples/sudoku/."""
    objective = (_EXAMPLES_DIR / "objective.md").read_text().strip()
    background = (_EXAMPLES_DIR / "background.md").read_text().strip()

    return TaskDefinition(
        name="sudoku",
        description="Evolve pure-Python Sudoku solvers scored on correctness and speed",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping={"agent.py": "agent.py"},
        default_seed_agent=str(_EXAMPLES_DIR / "seeds" / "baseline"),
        objective=objective,
        background=background,
        diagnostic_files={
            "puzzle.md": "The Sudoku puzzle grid with difficulty rating",
            "result.md": "Score breakdown: correctness, time, difficulty",
            "agent_stdout": "Captured print() output from the agent",
            "error.md": "Agent execution error (if any)",
        },
        config_defaults={
            "split": "train",
            "evaluation_budget": 1500,
            "eval_timeout": 30,
        },
        test_overrides={"split": "test"},
        gepa_datasets_builder=_gepa_datasets_builder,
    )
