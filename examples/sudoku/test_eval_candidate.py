#!/usr/bin/env python3
"""
Test-set evaluation with per-puzzle median aggregation.

Used by both RoboPhD (via examples/sudoku/main.py --eval-test-set) and GEPA
(via the wrapper in this file) to score a best candidate on the held-out
test set under identical conditions:

    - 10 repeats per puzzle
    - per-puzzle median across repeats (kills timing outliers)
    - mean of medians as headline score

Can be called as:
    1. A library function: ``test_eval(evaluator, candidate)``
    2. A CLI on a GEPA or RoboPhD run directory:
       ``python test_eval_candidate.py <run_dir>``
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from RoboPhD import eval_candidate, RoboPhDEvalConfig

logger = logging.getLogger(__name__)


def mean_of_medians(per_example_scores, num_unique):
    """Per-puzzle median across repeats, then mean across puzzles.

    The framework duplicates the dataset (``dataset * test_repeats``), so
    puzzle i's repeated scores are at indices i, i+num_unique, i+2*num_unique,
    ... — equivalent to ``scores[i::num_unique]``.
    """
    medians = [
        statistics.median(per_example_scores[i::num_unique])
        for i in range(num_unique)
    ]
    return sum(medians) / len(medians), medians


def test_eval(evaluator, candidate, test_repeats: int = 10) -> Tuple[float, Dict[str, Any]]:
    """Run the test-set evaluation and return (median_score, full_results_dict)."""
    from evaluator import load_dataset
    test_data = load_dataset()["test"]
    num_unique = len(test_data)

    logger.info(f"Test evaluation: {num_unique} problems × {test_repeats} repeats")
    eval_result = eval_candidate(
        evaluator=evaluator,
        dataset=test_data,
        candidate=candidate,
        config=RoboPhDEvalConfig(test_repeats=test_repeats),
    )

    score, _ = mean_of_medians(eval_result.per_example_scores, num_unique)
    logger.info(f"Test score: {score:.4f} (mean of per-puzzle medians, {test_repeats} repeats)")

    results = {
        "mean_test_score": score,
        "total_test_score": eval_result.total_score,
        "total_test_problems": num_unique,
        "test_repeats": test_repeats,
        "aggregation": "per_puzzle_median_then_mean",
    }
    return score, results


def _load_candidate_from_run_dir(run_dir: Path) -> Dict[str, str]:
    """Extract the best candidate from a GEPA or RoboPhD run directory."""
    # GEPA writes best_candidate.json at the top level
    gepa_best = run_dir / "best_candidate.json"
    if gepa_best.exists():
        with open(gepa_best) as f:
            return json.load(f)

    # GEPA also materializes a best_agent/ directory
    gepa_agent_dir = run_dir / "best_agent"
    if gepa_agent_dir.exists():
        return {"agent.py": (gepa_agent_dir / "agent.py").read_text()}

    # RoboPhD: read checkpoint.json, find highest ELO agent, load from agents/<name>/
    checkpoint = run_dir / "checkpoint.json"
    if checkpoint.exists():
        with open(checkpoint) as f:
            cp = json.load(f)
        records = cp.get("performance_records", {})
        if records:
            best_name = max(records.items(), key=lambda kv: kv[1]["elo"])[0]
            agent_dir = run_dir / "agents" / best_name
            if agent_dir.exists():
                return {"agent.py": (agent_dir / "agent.py").read_text()}

    raise FileNotFoundError(
        f"Could not find a best candidate in {run_dir}. "
        f"Expected best_candidate.json, best_agent/, or checkpoint.json + agents/"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test-eval a candidate with mean-of-medians aggregation"
    )
    parser.add_argument("run_dir", type=Path, help="GEPA or RoboPhD run directory")
    parser.add_argument("--test-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write test_results.json (default: <run_dir>/test_results.json)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from evaluator import SudokuEvaluator
    evaluator = SudokuEvaluator()
    candidate = _load_candidate_from_run_dir(args.run_dir)

    median_score, results = test_eval(evaluator, candidate, test_repeats=args.test_repeats)

    output_path = args.output or (args.run_dir / "test_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to {output_path}")


if __name__ == "__main__":
    main()
