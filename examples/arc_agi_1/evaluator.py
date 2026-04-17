"""
ARC-AGI-1 evaluator with rich diagnostics and stdout capture.

Self-contained evaluator for the ARC-AGI-1 example. Merges functionality from:
- RoboPhD/adapters/arc_agi_1.py (diagnostics, stdout capture)
- RoboPhD/adapters/gepa_arc_agi.py (TrackedLLM, dataset loading)
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm
from RoboPhD.eval_utils import retry_on_rate_limit, exec_with_stdout_capture

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SOLVER_MODEL = "openrouter/google/gemini-3.1-flash-lite-preview"



# ---------------------------------------------------------------------------
# Vendored utils (lazy import — requires dspy/datasets at import time)
# We use only three functions from utils/arc_agi_eval.py:
#   - load_arc_dataset: loads ARC-AGI from HuggingFace
#   - evaluate_predictions: scores train predictions
#   - evaluate_test: scores test predictions (2-attempt format)
# ---------------------------------------------------------------------------

_vendored = None


def _get_vendored():
    global _vendored
    if _vendored is None:
        import sys
        utils_dir = str(Path(__file__).resolve().parent)
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        from utils import arc_agi_eval
        _vendored = arc_agi_eval
    return _vendored


# ---------------------------------------------------------------------------
# TrackedLLM — self-contained LLM callable with cost tracking
# ---------------------------------------------------------------------------

class TrackedLLM:
    """LLM callable with call/cost tracking for ARC-AGI agents.

    Falls back to resp.usage.cost when litellm's pricing DB doesn't
    cover the model (common with OpenRouter).
    """

    def __init__(self, model_id: str, max_llm_calls: int = 10, reasoning_effort: Optional[str] = None):
        self.model_id = model_id
        self.max_llm_calls = max_llm_calls
        self.reasoning_effort = reasoning_effort
        self.calls: list = []

    @property
    def total_cost(self) -> float:
        return sum(c.get("cost", 0.0) for c in self.calls)

    def __call__(self, prompt: str, temperature: float = 1.0) -> str:
        if len(self.calls) >= self.max_llm_calls:
            raise RuntimeError(f"LLM budget exhausted ({self.max_llm_calls} calls)")

        start = time.time()
        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "timeout": 300,
            "num_retries": 0,  # don't amplify timeout wall time via litellm internal retries
        }
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}

        resp = retry_on_rate_limit(lambda: litellm.completion(**kwargs))
        duration = time.time() - start
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        # Cost: try litellm's DB first, fall back to provider-reported cost
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = getattr(resp.usage, "cost", None) or 0.0

        call_data = {
            "prompt": prompt,
            "response": content,
            "cost": cost,
            "duration": duration,
        }
        if reasoning:
            call_data["reasoning"] = reasoning
        self.calls.append(call_data)
        return content



# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_EXAMPLE_KEYS = ("problem_id", "train_in", "train_out", "test_in", "test_out")


def _example_to_dict(ex) -> Dict[str, Any]:
    """Convert a dspy.Example to a plain dict with explicit keys."""
    return {k: ex[k] for k in _EXAMPLE_KEYS}


def load_arc_train_val(seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """Load ARC-AGI train and val splits (200 + 200 from HF training)."""
    _v = _get_vendored()
    train_dspy, val_dspy, _test_dspy = _v.load_arc_dataset(seed)
    return [_example_to_dict(e) for e in train_dspy], [_example_to_dict(e) for e in val_dspy]


def load_arc_test(seed: int = 0) -> List[Dict]:
    """Load ARC-AGI test split (400 from HF evaluation)."""
    _v = _get_vendored()
    _train_dspy, _val_dspy, test_dspy = _v.load_arc_dataset(seed)
    return [_example_to_dict(e) for e in test_dspy]


# ---------------------------------------------------------------------------
# Agent execution with stdout capture
# ---------------------------------------------------------------------------

def run_agent(agent_code, train_in, train_out, test_in, test_out, model_id, max_llm_calls, reasoning_effort=None):
    """Run agent with stdout capture and vendored evaluation logic."""
    _v = _get_vendored()
    llms = TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls, reasoning_effort=reasoning_effort)

    try:
        namespace, agent_stdout = exec_with_stdout_capture(
            agent_code,
            then=lambda ns: ns["solve"](train_in, train_out, test_in, llms),
        )
        result = namespace["_result"]
        train_preds = result.get("train", [])
        test_preds = result.get("test", [])
    except Exception as e:
        return {
            "training_score": 0.0, "test_score": 0.0, "error": str(e),
            "train_examples": [], "test_examples": [], "llms": llms,
            "agent_stdout": getattr(e, "stdout", ""),
        }

    training_score, train_results = _v.evaluate_predictions(train_preds, train_out)
    if test_out:
        test_score, test_results = _v.evaluate_test(test_preds, test_out)
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
        "agent_stdout": agent_stdout,
    }


# ---------------------------------------------------------------------------
# Diagnostic formatting
# ---------------------------------------------------------------------------

def _format_grid(grid) -> str:
    if not grid or not isinstance(grid, list):
        return str(grid)
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _grid_shape(grid) -> str:
    if not grid or not isinstance(grid, list):
        return "?"
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 and isinstance(grid[0], list) else 0
    return f"{rows}x{cols}"


def _format_problem(problem_id, train_in, train_out, test_in, test_out) -> str:
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


def _format_trace(llms, cost_budget: float = 0.10) -> str:
    calls = llms.calls
    max_calls = getattr(llms, 'max_llm_calls', '?')
    lines = [
        f"# Agent LLM Trace",
        f"",
        f"Calls: {len(calls)}/{max_calls}, "
        f"Cost: ${llms.total_cost:.3f} (budget: ${cost_budget:.2f})",
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
    """ARC-AGI-1 evaluator with rich string diagnostics and stdout capture."""

    def __init__(
        self,
        solver_model: str = DEFAULT_SOLVER_MODEL,
        work_dir: Optional[Path] = None,
        max_llm_calls: int = 10,
        reasoning_effort: Optional[str] = None,
        cost_budget: float = 0.10,
    ):
        self.solver_model = solver_model
        self.work_dir = Path(work_dir) if work_dir else Path("arc_agi_1_work")
        self.max_llm_calls = max_llm_calls
        self.reasoning_effort = reasoning_effort
        self.cost_budget = cost_budget
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
        agent_code = candidate.get("agent.py", "")
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
        score = max(0, result["test_score"] - 0.1 * (cost > self.cost_budget))

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

        diagnostics: Dict[str, Any] = {"cost_usd": cost}
        diagnostics["problem.md"] = _format_problem(
            problem_id, example["train_in"], example["train_out"],
            example["test_in"], example.get("test_out"),
        )
        diagnostics["result.md"] = _format_result(problem_id, score, result, cost, self.cost_budget)
        diagnostics["agent_trace.md"] = _format_trace(llms, self.cost_budget)

        grid_lines = []
        for i, ex in enumerate(result["test_examples"]):
            status = "PASS" if ex["correct"] else "FAIL"
            grid_lines.append(f"## Test {i}: {status}\n\n{ex['feedback']}")
        for i, ex in enumerate(result["train_examples"]):
            status = "PASS" if ex["correct"] else "FAIL"
            grid_lines.append(f"## Train {i}: {status}\n\n{ex['feedback']}")
        if grid_lines:
            diagnostics["grid_comparison.md"] = "\n\n".join(grid_lines)

        agent_stdout = result.get("agent_stdout", "")
        if agent_stdout.strip():
            diagnostics["agent_stdout"] = agent_stdout

        if result["error"]:
            diagnostics["error.md"] = f"# Agent Error\n\n```\n{result['error']}\n```"

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
