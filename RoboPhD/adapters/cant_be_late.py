"""
GEPA adapter for Can't Be Late cloud scheduling benchmark.

Evolves a Python scheduling strategy (a Strategy subclass with a _step method)
that decides when to use SPOT, ON_DEMAND, or NONE to minimize cost while
completing before a deadline.

Candidate representation:
    {"agent_code": "<python source with Strategy subclass>"}

Usage:
    from RoboPhD.adapters.cant_be_late import (
        CantBeLateEvaluator,
        load_cant_be_late_dataset,
        CANT_BE_LATE_FILE_MAPPING,
        BACKGROUND,
        OBJECTIVE,
    )
"""

import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from RoboPhD.adapters.cant_be_late_utils_unmodified.simulation import (
    FAILED_SCORE,
    run_simulation,
    simulation_failure_info,
    simulation_success_info,
)
from RoboPhD.adapters.cant_be_late_utils_unmodified.dataset import load_trace_dataset
from RoboPhD.adapters.cant_be_late_constants_unmodified import (
    OPTIMIZATION_OBJECTIVE,
    OPTIMIZATION_BACKGROUND,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File mapping: candidate dict keys -> agent directory paths
# ---------------------------------------------------------------------------

CANT_BE_LATE_FILE_MAPPING = {
    "agent_code": "agent.py",
}

# ---------------------------------------------------------------------------
# Objective and background
# ---------------------------------------------------------------------------

OBJECTIVE = OPTIMIZATION_OBJECTIVE

BACKGROUND = OPTIMIZATION_BACKGROUND + """

RoboPhD-specific notes:
- The agent file is `agent.py` containing a Strategy subclass
- The strategy's `_step()` method is called at each simulation tick
- Available state: `self.env`, `self.task_duration`, `self.task_done_time`,
  `self.deadline`, `self.restart_overhead`, plus args `last_cluster_type` and `has_spot`
- Score = -cost (negative dollars). Higher is better. Failed simulations score -100,000.
- Simulations run as subprocesses — no LLM calls, pure algorithmic optimization."""

# ---------------------------------------------------------------------------
# Default dataset root
# ---------------------------------------------------------------------------

_DEFAULT_DATASET_ROOT = str(
    Path(__file__).resolve().parent.parent / "data" / "cant_be_late" / "traces"
)


# ---------------------------------------------------------------------------
# Thread-safe program file management
# ---------------------------------------------------------------------------

def _check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Check syntax and required structure. Returns (ok, error_message)."""
    try:
        compile(code, "<strategy>", "exec")
        if "class" not in code or "Strategy" not in code:
            return False, "No Strategy class found"
        if "_step" not in code:
            return False, "No _step method found"
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, str(e)


def _write_program(code: str) -> Tuple[str, str]:
    """Write code to a temp file. Returns (program_path, tmpdir). Caller must clean up."""
    tmpdir = tempfile.mkdtemp(prefix="cant_be_late_eval_")
    path = os.path.join(tmpdir, "strategy.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path, tmpdir


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cant_be_late_dataset(
    dataset_root: Optional[str] = None,
    seed: int = 0,
    max_traces_per_split: Optional[int] = None,
) -> Dict[str, list]:
    """Load Can't Be Late trace dataset with train/val/test splits."""
    root = dataset_root or _DEFAULT_DATASET_ROOT
    return load_trace_dataset(
        dataset_root=root,
        seed=seed,
        max_traces_per_split=max_traces_per_split,
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CantBeLateEvaluator:
    """
    Evaluator for Can't Be Late cloud scheduling problem.

    Runs the sky_spot simulator as a subprocess with the candidate strategy,
    scores by -cost (higher = cheaper = better).

    Thread-safe: each call writes a fresh temp file (no shared program cache).

    Implements: evaluator(candidate, example) -> (score, diagnostics)
    """

    def __init__(self, simulation_timeout: int = 300):
        self.simulation_timeout = simulation_timeout
        self._eval_count = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        agent_code = candidate.get("agent_code", "")

        # Syntax check (no temp file needed)
        ok, syntax_error = _check_syntax(agent_code)
        if not ok:
            score = FAILED_SCORE
            diagnostics: Dict[str, Any] = {
                "scores": {"cost": FAILED_SCORE},
                "Input": {
                    "trace_file": os.path.basename(example.get("trace_file", "?")),
                    "config": example.get("config", {}),
                },
                "Error": syntax_error or "Syntax validation failed",
                "error.md": f"# Syntax/Structure Error\n\n```\n{syntax_error}\n```",
            }
            self._write_result(problem_dir, example, score, error="syntax")
            self._bump_count()
            return score, diagnostics

        # Write to temp file and run simulation
        program_path, tmpdir = _write_program(agent_code)
        try:
            success, cost, error, details = run_simulation(
                program_path,
                example["trace_file"],
                example["config"],
                timeout=self.simulation_timeout,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if not success:
            score = FAILED_SCORE
            diagnostics = simulation_failure_info(error, example)
            diagnostics["error.md"] = f"# Simulation Error\n\n```\n{error}\n```"
            self._write_result(problem_dir, example, score, error="simulation")
            self._bump_count()
            return score, diagnostics

        score = -cost
        diagnostics = simulation_success_info(score, example, details)
        self._write_result(problem_dir, example, score, cost=cost)
        self._bump_count()
        return score, diagnostics

    def _write_result(self, problem_dir, example, score, cost=None, error=None):
        """Write result.json if problem_dir provided."""
        if problem_dir is None:
            return
        problem_dir = Path(problem_dir)
        problem_dir.mkdir(parents=True, exist_ok=True)
        result_entry = {
            "trace_file": os.path.basename(example.get("trace_file", "")),
            "config": example.get("config", {}),
            "score": score,
            "cost": cost if cost is not None else (-score if score != FAILED_SCORE else None),
            "error": error,
        }
        with open(problem_dir / "result.json", "w") as f:
            json.dump(result_entry, f, indent=2)

    def _bump_count(self):
        with self._lock:
            self._eval_count += 1
            count = self._eval_count
        if count % 100 == 0:
            logger.info(f"Can't Be Late evaluator: {count} evaluations completed")
