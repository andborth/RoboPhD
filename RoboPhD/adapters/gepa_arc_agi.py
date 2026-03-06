"""
GEPA adapter for ARC-AGI abstract reasoning benchmark.

Evolves agent code (a Python file with a solve() function) that solves
ARC-AGI tasks using LLM calls. Uses the vendored GEPA utils
(arc_agi_utils_unmodified.py) for dataset loading and evaluation logic.

Candidate representation:
    {"agent_code": "<python source with solve() function>"}

Usage:
    from RoboPhD.adapters.gepa_arc_agi import (
        ArcAGIEvaluator,
        load_arc_train_val,
        load_arc_test,
        ARC_AGI_FILE_MAPPING,
        BACKGROUND,
        OBJECTIVE,
    )
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_vendored():
    """Lazy-import vendored utils to avoid requiring dspy/datasets at import time."""
    from RoboPhD.adapters import arc_agi_utils_unmodified as _utils
    return _utils

# ---------------------------------------------------------------------------
# File mapping: candidate dict keys -> agent directory paths
# ---------------------------------------------------------------------------

ARC_AGI_FILE_MAPPING = {
    "agent_code": "agent.py",
}

# BACKGROUND and OBJECTIVE come from the vendored utils (single source of truth).
# Imported eagerly — this module requires dspy/datasets at import time.
from RoboPhD.adapters.arc_agi_utils_unmodified import BACKGROUND, OBJECTIVE


# ---------------------------------------------------------------------------
# Dataset loading (wraps vendored load_arc_dataset)
# ---------------------------------------------------------------------------

_EXAMPLE_KEYS = ("problem_id", "train_in", "train_out", "test_in", "test_out")


def _example_to_dict(ex) -> Dict[str, Any]:
    """Convert a dspy.Example to a plain dict with explicit keys."""
    return {k: ex[k] for k in _EXAMPLE_KEYS}


def load_arc_train_val(seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """Load ARC-AGI train and val splits (200 + 200 from HF training).

    Calls the vendored load_arc_dataset() which loads all HF splits internally.
    """
    _utils = _load_vendored()
    train_dspy, val_dspy, _test_dspy = _utils.load_arc_dataset(seed)
    return [_example_to_dict(e) for e in train_dspy], [_example_to_dict(e) for e in val_dspy]


def load_arc_test(seed: int = 0) -> List[Dict]:
    """Load ARC-AGI test split (400 from HF evaluation)."""
    _utils = _load_vendored()
    _train_dspy, _val_dspy, test_dspy = _utils.load_arc_dataset(seed)
    return [_example_to_dict(e) for e in test_dspy]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ArcAGIEvaluator:
    """
    GEPA-compatible evaluator for ARC-AGI problems.

    Executes agent code (a Python file with solve()) on each problem,
    scores via grid comparison, and applies a cost penalty.

    Implements: evaluator(candidate, example) -> (score, diagnostics)
    """

    def __init__(
        self,
        solver_model: str = "openrouter/google/gemini-3-flash-preview",
        work_dir: Optional[Path] = None,
        max_llm_calls: int = 10,
        reasoning_effort: Optional[str] = "high",
    ):
        self.solver_model = solver_model
        self.work_dir = Path(work_dir) if work_dir else Path("gepa_arc_agi_work")
        self.max_llm_calls = max_llm_calls
        self.reasoning_effort = reasoning_effort

        self._eval_count = 0
        self._total_eval_cost = 0.0
        self._last_logged_count = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a candidate agent on a single ARC-AGI problem.

        Args:
            candidate: {"agent_code": "<python source>"}.
            example: dict with problem_id, train_in, train_out, test_in, test_out.
            problem_dir: Optional path for writing per-problem artifacts.

        Returns:
            (score, diagnostics) where score includes cost penalty.
        """
        agent_code = candidate.get("agent_code", "")
        problem_id = example["problem_id"]

        _utils = _load_vendored()
        result = _utils.run_agent(
            agent_code=agent_code,
            train_in=example["train_in"],
            train_out=example["train_out"],
            test_in=example["test_in"],
            test_out=example.get("test_out"),
            model_id=self.solver_model,
            max_llm_calls=self.max_llm_calls,
            reasoning_effort=self.reasoning_effort,
        )

        llms = result["llms"]
        cost = llms.total_cost

        # Score with cost penalty (matches GEPA)
        score = max(0, result["test_score"] - 0.1 * (cost > 1.0))

        # Track stats
        with self._lock:
            self._eval_count += 1
            self._total_eval_cost += cost
            count = self._eval_count
            total_cost = self._total_eval_cost
            milestone = count // 50 * 50
            should_log = milestone > 0 and milestone > self._last_logged_count
            if should_log:
                self._last_logged_count = milestone
        if should_log:
            logger.info(f"ARC-AGI evaluator: {milestone} evaluations completed (${total_cost:.2f} spent)")

        # Build diagnostics (GEPA side_info keys)
        diagnostics: Dict[str, Any] = {
            "score": score,
            "problem_id": problem_id,
            "agent_code": agent_code,
            "training_score": result["training_score"],
            "test_score": result["test_score"],
            "cost": cost,
            "error": result["error"],
            "train_examples": result["train_examples"],
            "test_examples": result["test_examples"],
            **llms.get_traces(),
        }

        # Formatted string diagnostics for RoboPhD's file-based evolution context
        if result["error"]:
            diagnostics["error.md"] = f"# Agent Error\n\n```\n{result['error']}\n```"

        # Grid comparison summary
        grid_lines = []
        for i, ex in enumerate(result["test_examples"]):
            status = "PASS" if ex["correct"] else "FAIL"
            grid_lines.append(f"## Test {i}: {status}\n\n{ex['feedback']}")
        for i, ex in enumerate(result["train_examples"]):
            status = "PASS" if ex["correct"] else "FAIL"
            grid_lines.append(f"## Train {i}: {status}\n\n{ex['feedback']}")
        if grid_lines:
            diagnostics["grid_comparison.md"] = "\n\n".join(grid_lines)

        # Write result.json if problem_dir provided (RoboPhD path)
        if problem_dir is not None:
            problem_dir = Path(problem_dir)
            problem_dir.mkdir(parents=True, exist_ok=True)
            result_entry = {
                "problem_id": problem_id,
                "score": score,
                "test_score": result["test_score"],
                "training_score": result["training_score"],
                "cost": cost,
                "error": result["error"],
            }
            with open(problem_dir / "result.json", "w") as f:
                json.dump(result_entry, f, indent=2)

        return score, diagnostics

    @property
    def total_evaluations(self) -> int:
        return self._eval_count

    @property
    def total_eval_cost(self) -> float:
        return self._total_eval_cost
