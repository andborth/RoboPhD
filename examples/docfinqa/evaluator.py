"""
DocFinQA evaluator for long-document financial question answering.

Self-contained evaluator for the DocFinQA example. Evolves agent code
(a Python file with an answer() function) that answers numerical questions
over full SEC 10-K filings (~123K words each).

No vendored code — uses litellm for LLM/embedding calls and
huggingface_hub for dataset loading.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

from RoboPhD.eval_utils import retry_on_rate_limit, exec_with_stdout_capture

litellm.suppress_debug_info = True
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class CostTracker:
    """Lightweight per-evaluation cost tracker."""

    def __init__(self, budget: float = 0.10):
        self.budget = budget
        self.llm_cost = 0.0
        self.embed_cost = 0.0
        self.llm_calls = 0
        self.embed_calls = 0
        self.llm_prompts = []
        self.llm_responses = []

    @property
    def total(self) -> float:
        return self.llm_cost + self.embed_cost


def make_tracked_llm(model: str, tracker: CostTracker):
    """Return an llm(prompt) -> str callable with cost tracking and rate limit retry."""

    def llm(prompt: str, temperature: float = 0.0) -> str:
        resp = retry_on_rate_limit(
            lambda: litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=300,
                num_retries=0,
            )
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        response_text = resp.choices[0].message.content or ""
        tracker.llm_cost += cost
        tracker.llm_calls += 1
        tracker.llm_prompts.append(prompt)
        tracker.llm_responses.append(response_text)
        return response_text

    return llm


def make_tracked_embed(model: str, tracker: CostTracker):
    """Return an embed(text) -> list[float] callable with cost tracking and rate limit retry."""

    def embed(text: str) -> list:
        resp = retry_on_rate_limit(
            lambda: litellm.embedding(model=model, input=[text], timeout=300, num_retries=0)
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        tracker.embed_cost += cost
        tracker.embed_calls += 1
        return resp.data[0]["embedding"]

    return embed


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

def run_agent(agent_code: str, document: str, question: str, llm, embed) -> tuple:
    """Execute the candidate's answer() function.

    Returns (program_str, captured_stdout).
    """
    namespace, stdout = exec_with_stdout_capture(
        agent_code,
        then=lambda ns: ns["answer"](document, question, llm, embed),
    )
    return namespace["_result"], stdout


def _extract_program(text: str) -> str:
    """Strip markdown code fences if present."""
    if text is None:
        return ""
    text = text.strip()
    m = re.match(r'^```(?:python)?\s*\n(.*?)```\s*$', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _prompt_preview(prompt: str, head: int = 500, tail: int = 500) -> str:
    """Truncate a prompt to head + tail with ellipsis."""
    if len(prompt) <= head + tail + 50:
        return prompt
    return prompt[:head] + f"\n\n... [{len(prompt) - head - tail} chars omitted] ...\n\n" + prompt[-tail:]


def _parse_number(val) -> float:
    """Parse a number, stripping common suffixes like % $ , and whitespace."""
    s = str(val).strip().rstrip("%").replace(",", "").replace("$", "").strip()
    return float(s)


def check_numeric_answer(predicted, expected, rel_tol: float = 0.01) -> bool:
    """Check if predicted matches expected within relative tolerance."""
    try:
        p, e = _parse_number(predicted), _parse_number(expected)
        return abs(p - e) / max(abs(e), 1e-9) < rel_tol
    except (TypeError, ValueError):
        return str(predicted).strip() == str(expected).strip()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_SPLIT_FILES = {"train": "train.json", "validation": "dev.json", "test": "test.json"}


def load_docfinqa(split: str = "train") -> List[Dict[str, Any]]:
    """Load DocFinQA dataset from HuggingFace.

    Args:
        split: "train" (5735), "validation" (780), or "test" (922).
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
    """Evaluator for DocFinQA financial question answering.

    Executes agent code (a Python file with answer()) on each problem.
    The agent receives the full document, question, llm(), and embed()
    callables, and returns a Python program string that computes the answer.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        cost_budget: float = 0.10,
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
        agent_code = candidate.get("agent.py", "")
        question = example["question"]
        expected = example["answer"]

        tracker = CostTracker(budget=self.cost_budget)
        llm = make_tracked_llm(self.model, tracker)
        embed = make_tracked_embed(self.embed_model, tracker)

        # Run the agent
        try:
            program_str, agent_stdout = run_agent(agent_code, example["document"], question, llm, embed)
        except Exception as e:
            agent_stdout = getattr(e, "stdout", "")
            diagnostics = {
                "error.md": f"Agent crashed: {e}",
                "question": question,
                "expected": expected,
                "reference_program": example["program"],
            }
            if agent_stdout.strip():
                diagnostics["agent_stdout"] = agent_stdout
            with self._lock:
                self._eval_count += 1
            return 0.0, diagnostics

        program_str = _extract_program(program_str)

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
            "cost_usd": tracker.total,
            "cost_llm": f"${tracker.llm_cost:.4f}",
            "cost_embed": f"${tracker.embed_cost:.4f}",
            "cost_total": f"${tracker.total:.4f}",
            "over_budget": over_budget,
            "num_llm_calls": tracker.llm_calls,
            "num_embed_calls": tracker.embed_calls,
            "doc_words": len(example["document"].split()),
            "question": question,
        }

        # LLM prompt/response previews
        for i, (p, r) in enumerate(zip(tracker.llm_prompts, tracker.llm_responses)):
            suffix = f"_{i+1}"
            diagnostics[f"llm_prompt{suffix}.md"] = _prompt_preview(p)
            diagnostics[f"llm_response{suffix}.md"] = r

        if agent_stdout.strip():
            diagnostics["agent_stdout"] = agent_stdout

        if exec_error:
            diagnostics["error.md"] = f"# Execution Error\n\n```\n{exec_error}\n```"

        return score, diagnostics

    @property
    def total_evaluations(self) -> int:
        return self._eval_count

    @property
    def total_eval_cost(self) -> float:
        return self._total_eval_cost
