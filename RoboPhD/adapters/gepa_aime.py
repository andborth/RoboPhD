"""
GEPA adapter for AIME math competition prompt optimization.

Optimizes a system prompt for an LLM (default: gpt-4.1-mini) on AIME problems.
Pure prompt optimization — no tool code, no subprocess.

Candidate representation:
    {"system_prompt": "Solve the math problem carefully..."}

Usage:
    from RoboPhD.adapters.gepa_aime import (
        AIMEEvaluator,
        build_aime_dataset,
        AIME_FILE_MAPPING,
    )
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from RoboPhD.adapters.debug_logging import maybe_debug_log

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File mappings: candidate dict keys -> agent directory paths
# ---------------------------------------------------------------------------

AIME_FILE_MAPPING = {
    "system_prompt": "system_prompt.md",
}


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_aime_answer(response: str) -> Optional[int]:
    """
    Extract the final integer answer from an LLM response.

    AIME answers are integers 0-999. We look for the last standalone integer
    in the response, preferring answers on their own line (as instructed by
    the system prompt).
    """
    if not response:
        return None

    # First try: look for a final line that is just a number
    lines = response.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if re.fullmatch(r'\d+', line):
            return int(line)

    # Fallback: last standalone integer in the response
    matches = re.findall(r'\b(\d+)\b', response)
    if matches:
        return int(matches[-1])

    return None


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_aime_dataset(split: str = "train") -> List[Dict[str, Any]]:
    """
    Build a flat list of AIME example dicts from HuggingFace.

    Args:
        split: "train" (AIME 2022-2024, 90 problems) or "test" (AIME 2025, 30 problems).

    Returns:
        List of example dicts with problem_id, problem, answer, and optional solution.
    """
    from datasets import load_dataset

    if split not in ("train", "test"):
        raise ValueError(f"Unknown AIME split: {split!r}. Use 'train' or 'test'.")

    if split == "test":
        ds = load_dataset("MathArena/aime_2025")["train"]
        examples = []
        for row in ds:
            examples.append({
                "problem_id": f"aime_2025_{row['problem_idx']}",
                "problem": row["problem"],
                "answer": str(row["answer"]),
            })
        logger.info(f"AIME test set: {len(examples)} problems")
        return examples

    # Default: training set (AIME 2022-2024)
    ds = load_dataset("AI-MO/aimo-validation-aime")["train"]
    examples = []
    for row in ds:
        examples.append({
            "problem_id": f"aime_train_{row['id']}",
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": row.get("solution", ""),
        })
    logger.info(f"AIME training set: {len(examples)} problems")
    return examples


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class AIMEEvaluator:
    """
    GEPA-compatible evaluator for AIME math problems.

    Sends the candidate system prompt + problem to an LLM via litellm,
    parses the answer, and compares to ground truth.

    Implements: evaluator(candidate, example) -> (score, diagnostics)
    """

    def __init__(
        self,
        solver_model: str = "gpt-4.1-mini",
        work_dir: Optional[Path] = None,
        debug_log_probability: float = 0.0,
        debug_log_dir: Optional[Path] = None,
    ):
        self.solver_model = solver_model
        self.work_dir = Path(work_dir) if work_dir else Path("gepa_aime_work")
        self.debug_log_probability = debug_log_probability
        self.debug_log_dir = Path(debug_log_dir) if debug_log_dir else None

        self._eval_count = 0
        self._total_eval_cost = 0.0
        self._lock = threading.Lock()

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a candidate system prompt on a single AIME problem.

        Args:
            candidate: {"system_prompt": "..."}.
            example: {"problem_id": "...", "problem": "...", "answer": "42"}.
            problem_dir: Optional path for writing per-problem artifacts.

        Returns:
            (score, diagnostics) where score is 1.0 if correct, 0.0 otherwise.
        """
        import litellm

        system_prompt = candidate.get("system_prompt", "")
        problem_text = example["problem"]
        expected_answer = str(example["answer"]).lstrip("0") or "0"

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]

        try:
            completion = litellm.completion(model=self.solver_model, messages=messages)
            response = completion.choices[0].message.content or ""
            cost = litellm.completion_cost(completion_response=completion)
        except Exception as e:
            logger.error(f"LLM call failed for {example['problem_id']}: {e}")
            return 0.0, {"error": str(e), "problem_id": example["problem_id"]}

        # Track cost
        with self._lock:
            self._eval_count += 1
            self._total_eval_cost += cost
            count = self._eval_count
            total_cost = self._total_eval_cost
        if count % 50 == 0:
            logger.info(f"AIME evaluator: {count} evaluations completed (${total_cost:.2f} spent)")

        # Debug logging
        maybe_debug_log(
            debug_log_probability=self.debug_log_probability,
            debug_log_dir=self.debug_log_dir,
            call_type="generation",
            model=self.solver_model,
            messages=messages,
            response_text=response,
            metadata={
                "cost_usd": cost,
                "problem_id": example["problem_id"],
            },
        )

        # Parse and score
        predicted = parse_aime_answer(response)
        predicted_str = str(predicted).lstrip("0") or "0" if predicted is not None else None
        score = 1.0 if predicted_str == expected_answer else 0.0

        # Diagnostics (ASI for GEPA reflection)
        diagnostics = {
            "problem_id": example["problem_id"],
            "score": score,
            "correct": score >= 0.5,
            "cost_usd": cost,
            # String diagnostics — keys are filenames written by ExternalEvaluatorDomain
            "problem.md": problem_text,
            "system_prompt.md": system_prompt,
            "response.md": response,
            "predicted_answer": predicted,
            "expected_answer": example["answer"],
        }

        if "solution" in example and example["solution"]:
            diagnostics["reference_solution.md"] = example["solution"]

        # Write result.json if problem_dir provided (RoboPhD path)
        # String diagnostics are written as files by ExternalEvaluatorDomain
        if problem_dir is not None:
            problem_dir = Path(problem_dir)
            problem_dir.mkdir(parents=True, exist_ok=True)
            result_entry = {
                "problem_id": example["problem_id"],
                "correct": score >= 0.5,
                "score": score,
                "predicted_answer": predicted,
                "expected_answer": example["answer"],
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
