"""
GEPA adapter for DocFinQA long-document financial question answering.

Evolves agent code (a Python file with an answer() function) that answers
numerical questions over full SEC 10-K filings (~123K words each). The agent
receives the document, a question, an llm() callable, and an embed() callable,
and returns a short Python program whose last line assigns the result to `answer`.

Candidate representation:
    {"agent_code": "<python source with answer() function>"}

Usage:
    from RoboPhD.adapters.gepa_docfinqa import (
        DocFinQAEvaluator,
        load_docfinqa,
        FILE_MAPPING,
        BACKGROUND,
        OBJECTIVE,
    )
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILE_MAPPING = {"agent_code": "agent.py"}

OBJECTIVE = (
    "Evolve a Python function that answers numerical questions over long "
    "financial documents (SEC 10-K filings, averaging 123K words). The function "
    "receives the full document as markdown text, a question, an llm() callable, "
    "and an embed() callable. It must return a Python program string whose last "
    "line assigns the result to `answer`."
)

BACKGROUND = (
    "The document is a complete SEC filing in clean markdown with tables preserved. "
    "Documents average 123K words (~250 pages); the relevant information is typically "
    "in a single section or table. Questions require numerical reasoning: ratios, "
    "differences, percentages, averages, and multi-step arithmetic.\n\n"
    "Available tools:\n"
    "  llm(prompt) -> str : Call a language model. Expensive (~$0.003-0.01 per call).\n"
    "  embed(text) -> list[float] : Embed text for similarity search. Cheap (~$0.0001 per call).\n\n"
    "A per-question cost budget of $0.05 is enforced. Correct answers within budget "
    "score 1.0. Correct answers that exceed the budget are penalized to 0.9 (a 10% "
    "reduction). Incorrect answers score 0.0 regardless of cost. The program output "
    "is executed via exec(); if it raises an exception the answer is counted as incorrect."
)


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class CostTracker:
    """Lightweight per-evaluation cost tracker."""

    def __init__(self, budget: float = 0.05):
        self.budget = budget
        self.llm_cost = 0.0
        self.embed_cost = 0.0
        self.llm_calls = 0
        self.embed_calls = 0

    @property
    def total(self) -> float:
        return self.llm_cost + self.embed_cost


def make_tracked_llm(model: str, tracker: CostTracker):
    """Return an llm(prompt) -> str callable with cost tracking."""

    def llm(prompt: str) -> str:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        tracker.llm_cost += cost
        tracker.llm_calls += 1
        return resp.choices[0].message.content or ""

    return llm


def make_tracked_embed(model: str, tracker: CostTracker):
    """Return an embed(text) -> list[float] callable with cost tracking."""

    def embed(text: str) -> list:
        resp = litellm.embedding(model=model, input=[text])
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        tracker.embed_cost += cost
        tracker.embed_calls += 1
        return resp.data[0]["embedding"]

    return embed


# ---------------------------------------------------------------------------
# Agent execution helpers
# ---------------------------------------------------------------------------

def run_agent(agent_code: str, document: str, question: str, llm, embed) -> str:
    """Execute the candidate's answer() function, return the program string."""
    namespace = {}
    exec(agent_code, namespace)
    answer_fn = namespace["answer"]
    return answer_fn(document, question, llm, embed)


def _extract_program(text: str) -> str:
    """Strip markdown code fences if present."""
    if text is None:
        return ""
    text = text.strip()
    # Match ```python ... ``` or ``` ... ```
    m = re.match(r'^```(?:python)?\s*\n(.*?)```\s*$', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def check_numeric_answer(predicted, expected, rel_tol: float = 0.01) -> bool:
    """Check if predicted matches expected within relative tolerance."""
    try:
        p, e = float(predicted), float(expected)
        return abs(p - e) / max(abs(e), 1e-9) < rel_tol
    except (TypeError, ValueError):
        return str(predicted).strip() == str(expected).strip()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_SPLIT_FILES = {"train": "train.json", "validation": "dev.json", "test": "test.json"}


def load_docfinqa(split: str = "train") -> List[Dict[str, Any]]:
    """Load DocFinQA dataset from HuggingFace.

    Downloads via huggingface_hub and loads the JSON directly (bypassing
    pyarrow, which can't handle the large SEC filing text fields).

    Args:
        split: "train" (5735), "validation" (780), or "test" (922).

    Returns:
        List of example dicts with id, document, question, answer, program.
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unknown split: {split!r}. Use one of {tuple(_SPLIT_FILES)}.")

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="kensho/DocFinQA",
        filename=_SPLIT_FILES[split],
        repo_type="dataset",
    )

    try:
        with open(path) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse DocFinQA {split} split from {path}. "
            f"Cache may be corrupt — try deleting the file and re-running."
        ) from e

    examples = []
    for idx, row in enumerate(raw):
        examples.append({
            "id": f"docfinqa_{split}_{idx}",
            "document": row["Context"],
            "question": row["Question"],
            "answer": row["Answer"],
            "program": row["Program"],
        })
    logger.info(f"DocFinQA {split} set: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DocFinQAEvaluator:
    """
    Evaluator for DocFinQA financial question answering.

    Executes agent code (a Python file with answer()) on each problem.
    The agent receives the full document, question, llm(), and embed()
    callables, and returns a Python program string that computes the answer.

    Implements: evaluator(candidate, example) -> (score, diagnostics)
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        embed_model: str = "text-embedding-3-small",
        cost_budget: float = 0.05,
        over_budget_penalty: float = 0.9,
    ):
        self.model = model
        self.embed_model = embed_model
        self.cost_budget = cost_budget
        self.over_budget_penalty = over_budget_penalty

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
        Evaluate a candidate agent on a single DocFinQA problem.

        Args:
            candidate: {"agent_code": "<python source>"}.
            example: dict with id, document, question, answer, program.
            problem_dir: Optional path for writing per-problem artifacts.

        Returns:
            (score, diagnostics).
        """
        agent_code = candidate.get("agent_code", "")
        question = example["question"]
        expected = example["answer"]

        # Create per-evaluation cost tracker and callables
        tracker = CostTracker(budget=self.cost_budget)
        llm = make_tracked_llm(self.model, tracker)
        embed = make_tracked_embed(self.embed_model, tracker)

        # Run the agent
        try:
            program_str = run_agent(agent_code, example["document"], question, llm, embed)
        except Exception as e:
            return 0.0, {
                "error": f"Agent crashed: {e}",
                "question": question,
                "expected": expected,
                "reference_program": example["program"],
            }

        # Strip code fences if present
        program_str = _extract_program(program_str)

        # Warn on unexpectedly large programs
        if program_str and len(program_str) > 2000:
            logger.warning(
                f"Generated program is unexpectedly large ({len(program_str)} chars). "
                f"Expected short arithmetic programs (~200 chars). First 500 chars:\n"
                f"{program_str[:500]}"
            )

        # Execute the generated program
        exec_error = None
        predicted = None
        try:
            namespace = {}
            exec(program_str, namespace)
            predicted = namespace.get("answer")
        except Exception as e:
            exec_error = str(e)

        correct = check_numeric_answer(predicted, expected)
        over_budget = tracker.total > self.cost_budget
        score = (1.0 if correct else 0.0) * (self.over_budget_penalty if over_budget else 1.0)

        # Track stats
        with self._lock:
            self._eval_count += 1
            self._total_eval_cost += tracker.total
            count = self._eval_count
            total_cost = self._total_eval_cost
            milestone = count // 50 * 50
            should_log = milestone > 0 and milestone > self._last_logged_count
            if should_log:
                self._last_logged_count = milestone
        # should_log is a local bool captured under the lock — safe to read unlocked
        if should_log:
            logger.info(
                f"DocFinQA evaluator: {milestone} evaluations completed "
                f"(${total_cost:.2f} spent)"
            )

        diagnostics: Dict[str, Any] = {
            "score": score,
            "correct": correct,
            "predicted": predicted,
            "expected": expected,
            "reference_program": example["program"],
            "program": program_str,
            "exec_error": exec_error,
            "cost_llm": f"${tracker.llm_cost:.4f}",
            "cost_embed": f"${tracker.embed_cost:.4f}",
            "cost_total": f"${tracker.total:.4f}",
            "over_budget": over_budget,
            "num_llm_calls": tracker.llm_calls,
            "num_embed_calls": tracker.embed_calls,
            "doc_words": len(example["document"].split()),
            "question": question,
        }

        # Write result.json if problem_dir provided
        if problem_dir is not None:
            problem_dir = Path(problem_dir)
            problem_dir.mkdir(parents=True, exist_ok=True)
            result_entry = {
                "id": example["id"],
                "score": score,
                "correct": correct,
                "predicted": predicted,
                "expected": expected,
                "cost": tracker.total,
                "exec_error": exec_error,
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
