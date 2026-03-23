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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# Default solver model — single source of truth (also in arc_agi.py config_defaults)
DEFAULT_SOLVER_MODEL = "openrouter/google/gemini-3.1-flash-lite-preview"


def _load_vendored():
    """Lazy-import vendored utils to avoid requiring dspy/datasets at import time."""
    from RoboPhD.adapters import arc_agi_utils_unmodified as _utils
    return _utils


# ---------------------------------------------------------------------------
# TrackedLLM subclass with cost fix
# ---------------------------------------------------------------------------
# The vendored TrackedLLM uses litellm.completion_cost() which fails for
# OpenRouter models not in litellm's pricing DB. This subclass overrides
# __call__ to fall back to resp.usage.cost which OpenRouter provides directly.


class TrackedLLM:
    """TrackedLLM with fixed cost tracking for OpenRouter models.

    Wraps the vendored TrackedLLM, overriding only __call__ to fix the cost
    calculation. All other behavior (get_traces, total_cost, etc.) is inherited.
    """

    def __init__(self, model_id: str, max_llm_calls: int = 20, reasoning_effort: str = "high"):
        _utils = _load_vendored()
        self._inner = _utils.TrackedLLM(
            model_id=model_id,
            max_llm_calls=max_llm_calls,
            reasoning_effort=reasoning_effort,
        )

    def __call__(self, prompt: str, temperature: float = 1.0) -> str:
        if len(self._inner.calls) >= self._inner.max_llm_calls:
            raise RuntimeError(f"LLM budget exhausted ({self._inner.max_llm_calls} calls)")

        start = time.time()
        kwargs: dict = {
            "model": self._inner.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if self._inner.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self._inner.reasoning_effort}}

        resp = litellm.completion(**kwargs)
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
        self._inner.calls.append(call_data)
        return content

    # Delegate everything else to the inner vendored TrackedLLM
    @property
    def total_cost(self):
        return self._inner.total_cost

    @property
    def max_llm_calls(self):
        return self._inner.max_llm_calls

    @property
    def calls(self):
        return self._inner.calls

    def get_traces(self):
        return self._inner.get_traces()


def run_agent(agent_code, train_in, train_out, test_in, test_out, model_id, max_llm_calls, reasoning_effort=None):
    """Run agent using our TrackedLLM (with cost fix), vendored evaluation logic."""
    _utils = _load_vendored()

    llms = TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls, reasoning_effort=reasoning_effort)
    # Delegate to vendored run_agent's logic but with our LLM
    # We can't call _utils.run_agent directly since it constructs its own TrackedLLM
    try:
        namespace = {}
        exec(agent_code, namespace)
        result = namespace["solve"](train_in, train_out, test_in, llms)
        train_preds = result.get("train", [])
        test_preds = result.get("test", [])
    except Exception as e:
        return {
            "training_score": 0.0, "test_score": 0.0, "error": str(e),
            "train_examples": [], "test_examples": [], "llms": llms,
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
    }

# ---------------------------------------------------------------------------
# File mapping: candidate dict keys -> agent directory paths
# ---------------------------------------------------------------------------

ARC_AGI_FILE_MAPPING = {
    "agent_code": "agent.py",
}

# BACKGROUND and OBJECTIVE come from the vendored utils (single source of truth).
# Imported eagerly — this module requires dspy/datasets at import time.
from RoboPhD.adapters.arc_agi_utils_unmodified import OBJECTIVE
from RoboPhD.adapters.arc_agi_utils_unmodified import BACKGROUND as _UPSTREAM_BACKGROUND

BACKGROUND = _UPSTREAM_BACKGROUND.replace(
    "$0.8~1.0 LLM cost per problem",
    "$0.20~0.25 LLM cost per problem",
)
assert BACKGROUND != _UPSTREAM_BACKGROUND, (
    "BACKGROUND cost override failed — upstream text may have changed"
)


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
        solver_model: str = DEFAULT_SOLVER_MODEL,
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

        # Build diagnostics (GEPA side_info keys)
        diagnostics: Dict[str, Any] = {
            "score": score,
            "problem_id": problem_id,
            "agent_code": agent_code,
            "training_score": result["training_score"],
            "test_score": result["test_score"],
            "cost": cost,
            "cost_usd": cost,
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
