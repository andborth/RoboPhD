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

BACKGROUND = OPTIMIZATION_BACKGROUND

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
        compile(code, "<agent>", "exec")
        if "class" not in code or "Agent" not in code:
            return False, "No Agent class found"
        if "def step" not in code:
            return False, "No step method found"
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, str(e)


# Driver template: wraps the standalone Agent class in a Strategy subclass.
# The agent code is written to _agent.py; this driver imports and wraps it.
_DRIVER_TEMPLATE = """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
from _agent import Agent

class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'

    def __init__(self, args):
        super().__init__(args)

    def reset(self, env, task):
        super().reset(env, task)
        self._agent = Agent()
        self._agent.reset()

    def _step(self, last_cluster_type, has_spot):
        return self._agent.step(
            last_cluster_type, has_spot,
            self.env.elapsed_seconds, self.env.gap_seconds,
            self.restart_overhead, self.task_duration,
            self.deadline, self.task_done_time,
        )

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
"""


def _write_program(code: str) -> Tuple[str, str]:
    """Write agent code + driver to temp dir. Returns (strategy_path, tmpdir).

    The agent code (standalone Agent class) is written to _agent.py.
    A driver (Strategy subclass wrapping the Agent) is written to strategy.py.
    The simulator loads strategy.py which imports _agent.py.
    """
    tmpdir = tempfile.mkdtemp(prefix="cant_be_late_eval_")
    # Write agent code
    agent_path = os.path.join(tmpdir, "_agent.py")
    with open(agent_path, "w", encoding="utf-8") as f:
        f.write(code)
    # Write driver
    driver_path = os.path.join(tmpdir, "strategy.py")
    with open(driver_path, "w", encoding="utf-8") as f:
        f.write(_DRIVER_TEMPLATE)
    return driver_path, tmpdir


def _full_spot_availability(trace_file: str) -> Tuple[str, float]:
    """Summarise spot availability from trace file without segment cap.

    Same logic as vendored _extract_spot_availability but without the
    10-segment truncation, so the evolution AI sees the full pattern.

    Returns (availability_string, gap_hours) to avoid re-reading the file.
    """
    if not trace_file or not os.path.exists(trace_file):
        return "N/A", 0.0
    try:
        with open(trace_file) as f:
            trace = json.load(f)
        gap_hours = trace.get("metadata", {}).get("gap_seconds", 0) / 3600.0
        data = trace.get("data", [])
        if not data or gap_hours <= 0:
            return "N/A", gap_hours
        parts: list = []
        current_state = None
        start_tick = 0
        for i, val in enumerate(data):
            has_spot = (val == 0)
            if current_state is None:
                current_state, start_tick = has_spot, i
            elif current_state != has_spot:
                label = "S" if current_state else "X"
                parts.append(f"{start_tick * gap_hours:.1f}-{(i - 1) * gap_hours:.1f}:{label}")
                current_state, start_tick = has_spot, i
        if current_state is not None:
            label = "S" if current_state else "X"
            parts.append(f"{start_tick * gap_hours:.1f}-{len(data) * gap_hours:.1f}:{label}")
        return " | ".join(parts), gap_hours
    except Exception:
        return "N/A", 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _make_problem_id(example: Dict[str, Any]) -> str:
    """Build a human-readable problem_id from trace path and config.

    E.g. "us-west-2a_k80_1__281__d48_dl92_oh0.40"
    """
    trace_path = Path(example["trace_file"])
    trace_id = trace_path.stem  # "281"
    # Path: .../real/<env>/traces/random_start/<id>.json
    env = trace_path.parent.parent.parent.name  # "us-west-2a_k80_1"
    cfg = example["config"]
    return f"{env}__{trace_id}__d{cfg['duration']}_dl{cfg['deadline']}_oh{cfg['overhead']:.2f}"


def load_cant_be_late_dataset(
    dataset_root: Optional[str] = None,
    seed: int = 0,
    max_traces_per_split: Optional[int] = None,
) -> Dict[str, list]:
    """Load Can't Be Late trace dataset with train/val/test splits."""
    root = dataset_root or _DEFAULT_DATASET_ROOT
    ds = load_trace_dataset(
        dataset_root=root,
        seed=seed,
        max_traces_per_split=max_traces_per_split,
    )
    # Add human-readable problem_id to each example
    for split in ds.values():
        for example in split:
            example["problem_id"] = _make_problem_id(example)
    return ds


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
        self.total_eval_cost = 0.0  # No LLM calls — pure subprocess simulation
        self.total_evaluations = 0
        self._lock = threading.Lock()
        self._last_details_local = threading.local()  # Thread-safe details for subclass

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

        self._last_details_local.details = details  # Expose for subclass stdout capture

        if not success:
            score = FAILED_SCORE
            diagnostics = simulation_failure_info(error, example)
            diagnostics["error.md"] = f"# Simulation Error\n\n```\n{error}\n```"
            self._bump_count()
            return score, diagnostics

        score = -cost
        diagnostics = simulation_success_info(score, example, details)

        # Build readable summary for evolution AI (string values get written as files)
        # Use details directly for untruncated timeline and spot availability
        # (diagnostics contain truncated versions from vendored code)
        config = example.get("config", {})
        summary_lines = [
            f"# Simulation Result",
            f"",
            f"- **Trace**: {os.path.basename(example.get('trace_file', '?'))}",
            f"- **Duration**: {config.get('duration', '?')}h, **Deadline**: {config.get('deadline', '?')}h, **Overhead**: {config.get('overhead', '?')}h",
            f"- **Cost**: ${cost:.2f} (score: {score:.2f})",
        ]
        # Full timeline from details (vendored code truncates to 12 events)
        timeline_events = details.get("timeline_events", [])
        if timeline_events:
            timeline_str = " | ".join(timeline_events)
            summary_lines.append(f"- **Timeline**: {timeline_str}")
        output = diagnostics.get("Output", {})
        if output.get("segments"):
            summary_lines.append(f"- **Segments**: {output['segments']}")
        # Full spot availability (vendored code truncates to 10 segments)
        spot_avail, _ = _full_spot_availability(example.get("trace_file", ""))
        if spot_avail and spot_avail != "N/A":
            summary_lines.append(f"- **Spot availability**: {spot_avail}")
        diagnostics["summary.md"] = "\n".join(summary_lines)

        self._bump_count()
        return score, diagnostics

    def _bump_count(self):
        with self._lock:
            self.total_evaluations += 1
            count = self.total_evaluations
        if count % 100 == 0:
            logger.info(f"Can't Be Late evaluator: {count} evaluations completed")
