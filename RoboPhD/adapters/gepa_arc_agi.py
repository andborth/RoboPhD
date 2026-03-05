"""
GEPA adapter for ARC-AGI abstract reasoning benchmark.

Evolves agent code (a Python file with a solve() function) that solves
ARC-AGI tasks using LLM calls. Mirrors the GEPA examples/arc_agi setup
for direct comparison.

Utility functions (TrackedLLM, compare_grid, evaluate_predictions,
evaluate_test, run_agent, load_arc_dataset) are vendored from
gepa-ai/gepa@831cb70 examples/arc_agi/utils.py for provenance.

Candidate representation:
    {"agent_code": "<python source with solve() function>"}

Usage:
    from RoboPhD.adapters.gepa_arc_agi import (
        ArcAGIEvaluator,
        build_arc_agi_dataset,
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

# ---------------------------------------------------------------------------
# File mapping: candidate dict keys -> agent directory paths
# ---------------------------------------------------------------------------

ARC_AGI_FILE_MAPPING = {
    "agent_code": "agent.py",
}

# ---------------------------------------------------------------------------
# Prompts (from gepa-ai/gepa@831cb70 examples/arc_agi/utils.py)
# ---------------------------------------------------------------------------

BACKGROUND = """You are optimizing an ARC-AGI solving agent.

ARC-AGI task format:
- Each task has training examples (input/output pairs) and test inputs
- The (multi) agent(s) must infer the transformation pattern from training examples
- Competition allows maximum of 2 parallel output attempts per test input (pass if either matches)
- You can also use up to 10 LLM calls to solve the problem.
- Freely explore diverse strategies like multi agent systems, ensembles, voting, etc.

LLM cost:
- You are allowed to build an agent system with up to 10 LLM calls and total of $0.8~1.0 LLM cost per problem.

The agent receives:
- train_in, train_out: Training examples (list of 2D grids)
- test_in: Test inputs (no ground truth given to agent)
- llm: Callable for LLM queries with token/call tracking

The agent must return:
{
    "train": [grid, ...],           # 1 prediction per train example
    "test": [[grid, grid], ...],    # up to 2 attempts per test example
}

We evaluate on both training (training_score) and test (test_score with 2 attempts)."""

OBJECTIVE = """Build an ARC-AGI agent program that maximizes a test score."""

# ---------------------------------------------------------------------------
# Vendored utilities from gepa-ai/gepa@831cb70 examples/arc_agi/utils.py
# ---------------------------------------------------------------------------
# The functions below are copied verbatim from GEPA's published ARC-AGI
# example so that RoboPhD uses the exact same evaluation logic.

import random
import time
from dataclasses import dataclass, field

import litellm
from litellm import completion as litellm_completion

litellm.suppress_debug_info = True


@dataclass
class TrackedLLM:
    """Simple LLM wrapper that tracks calls and costs."""

    model_id: str = "openrouter/google/gemini-3-flash-preview"
    max_llm_calls: int = 20
    reasoning_effort: str = "high"
    calls: list = field(default_factory=list)

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
        }
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}

        resp = litellm_completion(**kwargs)
        duration = time.time() - start
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0

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

    def get_traces(self) -> dict:
        calls_to_include = self.calls
        sampled = False
        if len(self.calls) > 10:
            calls_to_include = random.sample(self.calls, 10)
            sampled = True

        trajectory = []
        for c in calls_to_include:
            entry = {
                "prompt": c["prompt"],
                "response": c["response"],
                "cost": c.get("cost", 0.0),
            }
            if c.get("reasoning"):
                entry["reasoning"] = c["reasoning"]
            trajectory.append(entry)

        result = {
            "llm_calls": len(self.calls),
            "llm_budget": self.max_llm_calls,
            "total_cost": self.total_cost,
            "trajectory": trajectory,
        }
        if sampled:
            result["trajectory_note"] = f"Randomly sampled 10 of {len(self.calls)} LLM calls"
        return result


def compare_grid(pred, gold) -> tuple:
    if not isinstance(pred, list):
        return (False, f"Must be List[List[int]], found {type(pred).__name__}. Correct: {gold}.")
    if len(pred) == 0:
        return False, f"Must have at least one row. Correct: {gold}."
    if not isinstance(pred[0], list):
        return False, f"Must be 2D list. Row 0 is {type(pred[0]).__name__}. Correct: {gold}."
    n, m = len(pred), len(pred[0])
    if m == 0:
        return False, f"Must have at least one column. Correct: {gold}."
    for i in range(n):
        if not isinstance(pred[i], list):
            return False, f"Row {i} must be list, found {type(pred[i]).__name__}. Correct: {gold}."
        if len(pred[i]) != m:
            return False, f"Staggered: row 0 has {m} cols, row {i} has {len(pred[i])}. Correct: {gold}."
        for j in range(m):
            if not isinstance(pred[i][j], (int, float)):
                return False, f"({i},{j}) must be int, found {type(pred[i][j]).__name__}. Correct: {gold}."
    gold_shape = (len(gold), len(gold[0]))
    if (n, m) != gold_shape:
        return False, f"Shape {(n, m)} != expected {gold_shape}. Correct: {gold}."
    wrong = [(i, j) for i in range(len(gold)) for j in range(len(gold[0])) if int(pred[i][j]) != gold[i][j]]
    if not wrong:
        return True, "Correct!"
    if len(wrong) < 10:
        return False, f"Wrong at {wrong}. Correct: {gold}."
    return False, f"Wrong at {len(wrong)} positions. Correct: {gold}."


def evaluate_predictions(preds, golds):
    if not preds:
        return 0.0, [{"idx": i, "correct": False, "feedback": "No prediction"} for i in range(len(golds))]
    results = []
    for i in range(len(golds)):
        if i < len(preds) and preds[i] is not None:
            correct, feedback = compare_grid(preds[i], golds[i])
        else:
            correct, feedback = False, "No prediction"
        results.append({"idx": i, "correct": correct, "feedback": feedback})
    score = sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    return score, results


def evaluate_test(test_preds, test_out):
    if not test_preds:
        return 0.0, [{"idx": i, "correct": False, "feedback": "No prediction"} for i in range(len(test_out))]
    normalized = [a[:2] if isinstance(a, list) else [a] for a in test_preds]
    attempt1 = [attempts[0] if attempts else None for attempts in normalized]
    attempt2 = [attempts[1] if len(attempts) > 1 else None for attempts in normalized]
    _, results1 = evaluate_predictions(attempt1, test_out)
    _, results2 = evaluate_predictions(attempt2, test_out)
    results = []
    for i in range(len(test_out)):
        r1, r2 = results1[i], results2[i]
        correct = r1["correct"] or r2["correct"]
        feedback = r1["feedback"] if r1["correct"] else (r2["feedback"] if r2["correct"] else r1["feedback"])
        results.append({"idx": i, "correct": correct, "feedback": feedback})
    all_correct = all(r["correct"] for r in results)
    return (1.0 if all_correct else 0.0), results


def run_agent(agent_code, train_in, train_out, test_in, test_out, model_id, max_llm_calls, reasoning_effort=None):
    llms = TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls, reasoning_effort=reasoning_effort)
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

    training_score, train_results = evaluate_predictions(train_preds, train_out)
    if test_out:
        test_score, test_results = evaluate_test(test_preds, test_out)
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
# Dataset builder
# ---------------------------------------------------------------------------

def load_arc_dataset(seed: int = 0):
    """Load ARC-AGI dataset from HuggingFace.

    Returns (train_set, val_set, test_set) as lists of dspy.Example.
    Matches GEPA's split: train=3200, val=200 (from HF training), test=HF evaluation.
    """
    import dspy
    from datasets import load_dataset

    ds = load_dataset("dataartist/arc-agi")

    def make_example(ex):
        return dspy.Example(
            problem_id=ex["id"],
            train_in=[t["input"] for t in ex["train"]],
            train_out=[t["output"] for t in ex["train"]],
            test_in=[t["input"] for t in ex["test"]],
            test_out=[t["output"] for t in ex["test"]],
        ).with_inputs("problem_id", "train_in", "train_out", "test_in", "test_out")

    trainset = [make_example(ex) for ex in ds["training"]]
    testset = [make_example(ex) for ex in ds["evaluation"]]

    random.Random(seed).shuffle(trainset)

    val_set = trainset[-200:]
    train_set = trainset[:-200]
    test_set = testset

    logger.info(f"ARC-AGI dataset: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    return train_set, val_set, test_set


def build_arc_agi_dataset(split: str = "train", seed: int = 0) -> List[Dict[str, Any]]:
    """Build a flat list of ARC-AGI example dicts for RoboPhD/GEPA.

    Wraps load_arc_dataset() and converts dspy.Example -> plain dicts.

    Args:
        split: "train" (3200 problems), "val" (200 problems), or "test" (HF evaluation set).
        seed: Random seed for train/val split (default 0, matching GEPA).

    Returns:
        List of dicts with keys: problem_id, train_in, train_out, test_in, test_out.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown ARC-AGI split: {split!r}. Use 'train', 'val', or 'test'.")

    train_set, val_set, test_set = load_arc_dataset(seed=seed)

    if split == "train":
        examples = train_set
    elif split == "val":
        examples = val_set
    else:
        examples = test_set

    result = []
    for ex in examples:
        result.append({
            "problem_id": ex.problem_id,
            "train_in": ex.train_in,
            "train_out": ex.train_out,
            "test_in": ex.test_in,
            "test_out": ex.test_out,
        })

    logger.info(f"ARC-AGI {split} set: {len(result)} problems")
    return result


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
