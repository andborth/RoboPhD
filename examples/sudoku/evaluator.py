"""
Sudoku evaluator with forbidden-module enforcement.

Self-contained evaluator for the Sudoku example. Evolves a pure-Python
solver scored on correctness and speed. No LLM calls — pure algorithmic
optimization.
"""

import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from RoboPhD.eval_utils import exec_with_stdout_capture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES = ["ctypes", "subprocess", "cffi", "os.system", "os.popen"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    train_size: int = 200,
    test_size: int = 200,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load stratified Sudoku puzzles from HuggingFace.

    Returns dict with 'train' and 'test' keys.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def _load_split(split_name):
        path = hf_hub_download("sapientinc/sudoku-extreme", f"{split_name}.csv", repo_type="dataset")
        # answer column is 81-digit string — force str to avoid pyarrow int overflow
        df = pd.read_csv(path, dtype={"answer": str, "question": str})
        return df.to_dict("records")

    def _stratified_sample(records, size, rng):
        easy = [ex for ex in records if ex["rating"] <= 5]
        medium = [ex for ex in records if 5 < ex["rating"] <= 10]
        hard = [ex for ex in records if 10 < ex["rating"] <= 15]
        extreme = [ex for ex in records if ex["rating"] > 15]

        quarter = size // 4
        remainder = size - 4 * quarter

        def sample(pool, n):
            return rng.sample(pool, min(n, len(pool)))

        examples = (
            sample(easy, quarter)
            + sample(medium, quarter)
            + sample(hard, quarter)
            + sample(extreme, quarter + remainder)
        )
        rng.shuffle(examples)
        return examples

    def _to_dicts(examples, split, start_idx=0):
        result = []
        for i, ex in enumerate(examples):
            result.append({
                "id": f"sudoku_{split}_{i + start_idx:04d}",
                "puzzle": ex["question"],
                "solution": ex["answer"],
                "rating": ex["rating"],
                "difficulty": (
                    "easy" if ex["rating"] <= 5
                    else "medium" if ex["rating"] <= 10
                    else "hard" if ex["rating"] <= 15
                    else "extreme"
                ),
            })
        return result

    ds_train = _load_split("train")
    ds_test = _load_split("test")

    train_examples = _stratified_sample(ds_train, train_size, random.Random(seed))
    test_examples = _stratified_sample(ds_test, test_size, random.Random(seed + 2))

    return {
        "train": _to_dicts(train_examples, "train"),
        "test": _to_dicts(test_examples, "test"),
    }


# ---------------------------------------------------------------------------
# Sudoku validation (for diagnostics)
# ---------------------------------------------------------------------------

def _validate_sudoku(solution_str: str) -> List[str]:
    """Check if a solution string is a valid completed Sudoku grid."""
    errors = []
    if len(solution_str) != 81:
        errors.append(f"Length {len(solution_str)}, expected 81")
        return errors
    if not all(c in "123456789" for c in solution_str):
        bad = [c for c in solution_str if c not in "123456789"]
        errors.append(f"Invalid characters: {set(bad)}")
        return errors

    grid = [int(c) for c in solution_str]
    for r in range(9):
        row = grid[r * 9:(r + 1) * 9]
        if len(set(row)) != 9:
            errors.append(f"Row {r} has duplicates: {row}")
    for c in range(9):
        col = [grid[r * 9 + c] for r in range(9)]
        if len(set(col)) != 9:
            errors.append(f"Col {c} has duplicates: {col}")
    for br in range(3):
        for bc in range(3):
            box = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    box.append(grid[r * 9 + c])
            if len(set(box)) != 9:
                errors.append(f"Box ({br},{bc}) has duplicates: {box}")
    return errors


def _format_grid(puzzle_str: str) -> str:
    """Format an 81-char puzzle string as a readable 9x9 grid."""
    lines = []
    for r in range(9):
        row = puzzle_str[r * 9:(r + 1) * 9]
        cells = " ".join(row[i] if row[i] != "." else "." for i in range(9))
        if r in (3, 6):
            lines.append("------+-------+------")
        line = f"{cells[:5]} | {cells[6:11]} | {cells[12:]}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SudokuEvaluator:
    """Evaluate a Sudoku solver agent.

    Executes agent.py in an isolated namespace, calls solve(puzzle),
    measures correctness + speed.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.total_eval_cost = 0.0
        self.total_evaluations = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        with self._lock:
            self.total_evaluations += 1

        agent_code = candidate["agent.py"]
        puzzle = example["puzzle"]
        expected = example["solution"]
        rating = example.get("rating", "?")
        difficulty = example.get("difficulty", "?")

        puzzle_diag = f"# Puzzle (rating={rating}, {difficulty})\n```\n{_format_grid(puzzle)}\n```"

        # Check for forbidden modules
        for mod in FORBIDDEN_MODULES:
            if mod in agent_code:
                return 0.0, {
                    "error.md": f"Agent uses forbidden module: `{mod}`. Only pure Python is allowed.",
                    "puzzle.md": puzzle_diag,
                }

        # Execute agent and call solve() with stdout capture

        def _call_solve(ns):
            if "solve" not in ns:
                raise RuntimeError("Agent does not define a `solve()` function.")
            start = time.process_time()
            result = ns["solve"](puzzle)
            ns["_elapsed"] = time.process_time() - start
            return result

        try:
            namespace, agent_stdout = exec_with_stdout_capture(
                agent_code, then=_call_solve,
            )
        except Exception as e:
            stdout = getattr(e, "stdout", "").strip()
            diag = {
                "error.md": f"Agent failed:\n```\n{type(e).__name__}: {e}\n```",
                "puzzle.md": puzzle_diag,
            }
            if stdout:
                diag["agent_stdout"] = stdout
            return 0.0, diag

        result = namespace.get("_result", "")
        elapsed = namespace.get("_elapsed", 0.0)
        agent_stdout = agent_stdout.strip()

        if not isinstance(result, str):
            result = str(result) if result is not None else ""

        correct = (result == expected)

        # Compute score (correctness gate + time score)
        if correct:
            time_score = max(0.0, 1.0 - elapsed * 100)
            score = time_score
        else:
            score = 0.0
            time_score = 0.0

        validation_errors = _validate_sudoku(result) if len(result) == 81 else ["Invalid length"]

        diagnostics = {
            "score": score,
            "correct": correct,
            "elapsed_seconds": round(elapsed, 6),
            "time_score": round(time_score, 4),
            "rating": rating,
            "difficulty": difficulty,
            "puzzle.md": puzzle_diag,
            "result.md": (
                f"# Result\n"
                f"- **Correct**: {correct}\n"
                f"- **Score**: {score:.4f} (time={time_score:.4f})\n"
                f"- **Solve time**: {elapsed*1000:.2f}ms\n"
                f"- **Difficulty**: {difficulty} (rating={rating})\n"
                + (f"\n## Validation Errors\n" + "\n".join(f"- {e}" for e in validation_errors) if validation_errors and not correct else "")
            ),
        }
        if agent_stdout:
            diagnostics["agent_stdout"] = agent_stdout

        return score, diagnostics
