"""
ARC-AGI-1 adapter with rich diagnostics and stdout capture.

Based on gepa_arc_agi.py but with enhanced diagnostics: formatted visual grids,
LLM call traces, agent stdout capture, and cost visibility. All diagnostics are
returned as strings for exact parity between GEPA and RoboPhD/Autoresearch.

Reuses TrackedLLM (cost fix) and dataset loading from gepa_arc_agi.py.
Uses vendored arc_agi_utils_unmodified.py for evaluation logic.
"""

import io
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from RoboPhD.adapters.gepa_arc_agi import (
    TrackedLLM,
    ARC_AGI_FILE_MAPPING,
    DEFAULT_SOLVER_MODEL,
    load_arc_train_val,
    load_arc_test,
)

logger = logging.getLogger(__name__)


def _load_vendored():
    from RoboPhD.adapters import arc_agi_utils_unmodified as _utils
    return _utils


# ---------------------------------------------------------------------------
# BACKGROUND and OBJECTIVE
# ---------------------------------------------------------------------------

OBJECTIVE = "Build an ARC-AGI agent program that maximizes a test score."

BACKGROUND = """You are optimizing an ARC-AGI solving agent.

ARC-AGI task format:
- Each task has training examples (input/output pairs) and test inputs
- The (multi) agent(s) must infer the transformation pattern from training examples
- Competition allows maximum of 2 parallel output attempts per test input (pass if either matches)
- You can also use up to 10 LLM calls to solve the problem.
- Freely explore diverse strategies like multi agent systems, ensembles, voting, etc.

LLM cost:
- You are allowed to build an agent system with up to 10 LLM calls and total of $0.20~0.25 LLM cost per problem.

Per-question cost budget: $0.25 is enforced. Correct answers within budget score 1.0. \
Correct answers that exceed the budget are penalized to 0.9 (a 10% reduction). \
Incorrect answers score 0.0 regardless of cost.

The agent receives:
- train_in, train_out: Training examples (list of 2D grids)
- test_in: Test inputs (no ground truth given to agent)
- llm: Callable for LLM queries with token/call tracking

The agent must return:
{
    "train": [grid, ...],           # 1 prediction per train example
    "test": [[grid, grid], ...],    # up to 2 attempts per test example
}

We evaluate on both training (training_score) and test (test_score with 2 attempts).

Diagnostics: Any print() output from the agent is captured and included \
in evaluation diagnostics as agent_stdout. Use print() to log any \
information you think would be helpful for you to see in improving the \
agent in later rounds of testing and refinement."""


# ---------------------------------------------------------------------------
# Grid formatting
# ---------------------------------------------------------------------------

def _format_grid(grid) -> str:
    """Render a 2D grid as space-separated rows."""
    if not grid or not isinstance(grid, list):
        return str(grid)
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _grid_shape(grid) -> str:
    """Return 'RxC' shape string for a grid."""
    if not grid or not isinstance(grid, list):
        return "?"
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 and isinstance(grid[0], list) else 0
    return f"{rows}x{cols}"


# ---------------------------------------------------------------------------
# Agent execution with stdout capture
# ---------------------------------------------------------------------------

def run_agent(agent_code, train_in, train_out, test_in, test_out, model_id, max_llm_calls, reasoning_effort=None):
    """Run agent with stdout capture and vendored evaluation logic."""
    _utils = _load_vendored()

    llms = TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls, reasoning_effort=reasoning_effort)

    # Capture print() from agent code
    buf = io.StringIO()

    def _captured_print(*args, **kwargs):
        kwargs.setdefault("file", buf)
        print(*args, **kwargs)

    try:
        namespace = {"print": _captured_print}
        exec(agent_code, namespace)
        result = namespace["solve"](train_in, train_out, test_in, llms)
        train_preds = result.get("train", [])
        test_preds = result.get("test", [])
    except Exception as e:
        return {
            "training_score": 0.0, "test_score": 0.0, "error": str(e),
            "train_examples": [], "test_examples": [], "llms": llms,
            "agent_stdout": buf.getvalue(),
        }

    training_score, train_results = _utils.evaluate_predictions(train_preds, train_out)
    if test_out:
        test_score, test_results = _utils.evaluate_test(test_preds, test_out)
    else:
        test_score, test_results = 0.0, []

    train_examples = []
    for i, (inp, gold, res) in enumerate(zip(train_in, train_out, train_results)):
        pred = train_preds[i] if i < len(train_preds) else None
        train_examples.append({"input": inp, "gold": gold, "prediction": pred, "correct": res["correct"], "feedback": res["feedback"]})

    test_examples = []
    for i, res in enumerate(test_results):
        inp = test_in[i] if i < len(test_in) else None
        gold = test_out[i] if test_out and i < len(test_out) else None
        pred = test_preds[i] if i < len(test_preds) else None
        test_examples.append({"input": inp, "gold": gold, "prediction": pred, "correct": res["correct"], "feedback": res["feedback"]})

    return {
        "training_score": training_score, "test_score": test_score, "error": None,
        "train_examples": train_examples, "test_examples": test_examples, "llms": llms,
        "agent_stdout": buf.getvalue(),
    }


# ---------------------------------------------------------------------------
# Diagnostic formatting
# ---------------------------------------------------------------------------

def _format_problem(problem_id, train_in, train_out, test_in, test_out) -> str:
    """Format the ARC-AGI problem as readable markdown with visual grids."""
    lines = [f"# ARC-AGI Problem: {problem_id}", ""]

    for i, (inp, out) in enumerate(zip(train_in, train_out)):
        lines.append(f"## Train Example {i}")
        lines.append(f"### Input ({_grid_shape(inp)})")
        lines.append(_format_grid(inp))
        lines.append(f"### Output ({_grid_shape(out)})")
        lines.append(_format_grid(out))
        lines.append("")

    for i, inp in enumerate(test_in):
        lines.append(f"## Test Input {i} ({_grid_shape(inp)})")
        lines.append(_format_grid(inp))
        lines.append("")

    if test_out:
        for i, out in enumerate(test_out):
            lines.append(f"## Test Expected Output {i} ({_grid_shape(out)})")
            lines.append(_format_grid(out))
            lines.append("")

    return "\n".join(lines)


def _format_result(problem_id, score, result, cost, cost_budget=0.25) -> str:
    """Format result summary with predictions vs gold as visual grids."""
    over_budget = cost > cost_budget
    penalty = " **COST PENALTY**" if over_budget and result["test_score"] > 0 else ""
    lines = [
        f"# Result: {problem_id}",
        f"",
        f"Score: {score:.1f} (test_score={result['test_score']:.1f}, "
        f"cost=${cost:.3f}{penalty})",
        f"Training: {result['training_score']:.0%} correct",
        "",
    ]

    if result["error"]:
        lines.append(f"**Error**: {result['error']}")
        return "\n".join(lines)

    for i, ex in enumerate(result["test_examples"]):
        status = "PASS" if ex["correct"] else "FAIL"
        lines.append(f"## Test {i}: {status}")
        if ex.get("prediction"):
            lines.append(f"### Prediction ({_grid_shape(ex['prediction'])})")
            lines.append(_format_grid(ex["prediction"]))
        else:
            lines.append("### Prediction\n(none)")
        if ex.get("gold"):
            lines.append(f"### Expected ({_grid_shape(ex['gold'])})")
            lines.append(_format_grid(ex["gold"]))
        lines.append(f"### Feedback")
        lines.append(ex.get("feedback", ""))
        lines.append("")

    for i, ex in enumerate(result["train_examples"]):
        status = "PASS" if ex["correct"] else "FAIL"
        lines.append(f"## Train {i}: {status}")
        if not ex["correct"] and ex.get("prediction"):
            lines.append(f"### Prediction ({_grid_shape(ex['prediction'])})")
            lines.append(_format_grid(ex["prediction"]))
            lines.append(f"### Feedback")
            lines.append(ex.get("feedback", ""))
        lines.append("")

    return "\n".join(lines)


def _format_trace(llms) -> str:
    """Format LLM call trajectory as readable markdown."""
    calls = llms.calls
    max_calls = getattr(llms, 'max_llm_calls', '?')
    lines = [
        f"# Agent LLM Trace",
        f"",
        f"Calls: {len(calls)}/{max_calls}, "
        f"Cost: ${llms.total_cost:.3f} (budget: $0.25)",
        "",
    ]

    for i, call in enumerate(calls):
        cost = call.get("cost", 0.0)
        duration = call.get("duration", 0.0)
        lines.append(f"## Call {i+1} (${cost:.4f}, {duration:.1f}s)")

        prompt = call.get("prompt", "")
        if len(prompt) > 500:
            lines.append(f"### Prompt (first 500 of {len(prompt)} chars)")
            lines.append(prompt[:500] + "...")
        else:
            lines.append(f"### Prompt")
            lines.append(prompt)

        response = call.get("response", "")
        lines.append(f"### Response ({len(response)} chars)")
        lines.append(response)

        if call.get("reasoning"):
            reasoning = call["reasoning"]
            if len(reasoning) > 300:
                lines.append(f"### Reasoning (first 300 of {len(reasoning)} chars)")
                lines.append(reasoning[:300] + "...")
            else:
                lines.append(f"### Reasoning")
                lines.append(reasoning)

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ArcAGI1Evaluator:
    """
    ARC-AGI-1 evaluator with rich string diagnostics and stdout capture.

    Returns only string diagnostics (no structured lists/dicts) for exact
    parity between GEPA and RoboPhD. Numeric scores flow via result.json.
    """

    def __init__(
        self,
        solver_model: str = DEFAULT_SOLVER_MODEL,
        work_dir: Optional[Path] = None,
        max_llm_calls: int = 10,
        reasoning_effort: Optional[str] = "high",
    ):
        self.solver_model = solver_model
        self.work_dir = Path(work_dir) if work_dir else Path("arc_agi_1_work")
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
        agent_code = candidate.get("agent_code", "")
        problem_id = example["problem_id"]

        result = run_agent(
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
        score = max(0, result["test_score"] - 0.1 * (cost > 0.25))

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

        # Build diagnostics — strings for evolution context, plus cost_usd for tracking
        diagnostics: Dict[str, Any] = {
            "cost_usd": cost,  # Used by ExternalEvaluatorDomain for cost tracking
        }

        # Problem grids (train I/O + test I/O)
        diagnostics["problem.md"] = _format_problem(
            problem_id, example["train_in"], example["train_out"],
            example["test_in"], example.get("test_out"),
        )

        # Result summary with predictions vs gold
        diagnostics["result.md"] = _format_result(problem_id, score, result, cost)

        # LLM call trajectory
        diagnostics["agent_trace.md"] = _format_trace(llms)

        # Grid comparison (cell-level feedback from vendored compare_grid)
        grid_lines = []
        for i, ex in enumerate(result["test_examples"]):
            status = "PASS" if ex["correct"] else "FAIL"
            grid_lines.append(f"## Test {i}: {status}\n\n{ex['feedback']}")
        for i, ex in enumerate(result["train_examples"]):
            status = "PASS" if ex["correct"] else "FAIL"
            grid_lines.append(f"## Train {i}: {status}\n\n{ex['feedback']}")
        if grid_lines:
            diagnostics["grid_comparison.md"] = "\n\n".join(grid_lines)

        # Agent stdout
        agent_stdout = result.get("agent_stdout", "")
        if agent_stdout.strip():
            diagnostics["agent_stdout"] = agent_stdout

        # Error
        if result["error"]:
            diagnostics["error.md"] = f"# Agent Error\n\n```\n{result['error']}\n```"

        # Write result.json if problem_dir provided
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
