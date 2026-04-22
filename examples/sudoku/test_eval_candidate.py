#!/usr/bin/env python3
"""
Test-set evaluation with per-puzzle median aggregation.

Used by both RoboPhD (via examples/sudoku/main.py --eval-test-set) and GEPA
(via the wrapper in this file) to score a best candidate on the held-out
test set under identical conditions:

    - 9 repeats per puzzle (odd so the per-puzzle median is an exact sample)
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
from typing import Any, Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from RoboPhD import eval_candidate, RoboPhDEvalConfig

logger = logging.getLogger(__name__)


# The sudoku score is max(0, 1 - elapsed * 100) — any solve past 10ms scores 0.
# Per-solve timeout is set to 100× that cutoff: generous enough that no correct
# solve which would score >0 ever hits it, tight enough to bound wall-clock on
# slow agents at ~5.5 hours for a full 2000 × 10-repeat run.
_PER_SOLVE_TIMEOUT_SECONDS = 1


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


def test_eval(
    evaluator,
    candidate,
    test_repeats: int = 9,
    agent_name: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Run the test-set evaluation and return (median_score, full_results_dict).

    The returned results dict always includes these fields (in addition to
    the score totals): ``test_repeats``, ``aggregation``, ``test_eval_cost_usd``,
    and ``evaluated_agent`` (when ``agent_name`` is provided). This is the
    canonical shape consumers of sudoku test_results.json should expect.
    """
    from evaluator import load_dataset
    test_data = load_dataset()["test"]
    num_unique = len(test_data)

    logger.info(f"Test evaluation: {num_unique} problems × {test_repeats} repeats")
    eval_result = eval_candidate(
        evaluator=evaluator,
        dataset=test_data,
        candidate=candidate,
        config=RoboPhDEvalConfig(
            test_repeats=test_repeats,
            # max_workers=1 is required for correct scoring: solves are timed via
            # time.process_time(), which on Darwin/Linux returns process-wide CPU
            # time (sum across all threads). Parallel workers contend on CPU and
            # inflate each other's measured elapsed, which drags max(0, 1 - elapsed*100)
            # toward 0. Serial execution gives each solve an uncontaminated measurement.
            max_workers=1,
            eval_timeout=_PER_SOLVE_TIMEOUT_SECONDS,
        ),
    )

    score, _ = mean_of_medians(eval_result.per_example_scores, num_unique)
    logger.info(f"Test score: {score:.4f} (mean of per-puzzle medians, {test_repeats} repeats)")

    results = {
        "mean_test_score": score,
        "total_test_score": eval_result.total_score,
        "total_test_problems": num_unique,
        "test_repeats": test_repeats,
        "aggregation": "per_puzzle_median_then_mean",
        "test_eval_cost_usd": evaluator.total_eval_cost,
    }
    if agent_name is not None:
        results["evaluated_agent"] = agent_name
    return score, results


def _load_candidate_from_run_dir(
    run_dir: Path, agent_name: Optional[str] = None
) -> Tuple[Dict[str, str], str]:
    """Extract a candidate from a GEPA or RoboPhD run directory.

    Returns (candidate_dict, resolved_agent_name). When agent_name is None,
    selects the best-ELO agent (RoboPhD) or best_agent/ (GEPA). When
    agent_name is provided, it must be a key in the RoboPhD checkpoint's
    agent_pool — GEPA runs don't carry a named pool, so agent_name is
    RoboPhD-only.
    """
    # Named-agent lookup is RoboPhD-specific (GEPA has no agent_pool).
    if agent_name is not None:
        return _load_named_agent(run_dir, agent_name)

    # Unnamed: prefer GEPA artifacts if present, otherwise fall through
    # to RoboPhD best-ELO resolution.
    gepa_best = run_dir / "best_candidate.json"
    if gepa_best.exists():
        with open(gepa_best) as f:
            return json.load(f), "best_candidate"
    gepa_agent_dir = run_dir / "best_agent"
    if gepa_agent_dir.exists():
        return {"agent.py": (gepa_agent_dir / "agent.py").read_text()}, "best_agent"

    checkpoint = run_dir / "checkpoint.json"
    if checkpoint.exists():
        with open(checkpoint) as f:
            cp = json.load(f)
        records = cp.get("performance_records", {})
        if records:
            best_name = max(records.items(), key=lambda kv: kv[1]["elo"])[0]
            agent_dir = run_dir / "agents" / best_name
            if agent_dir.exists():
                return {"agent.py": (agent_dir / "agent.py").read_text()}, best_name

    raise FileNotFoundError(
        f"Could not find a best candidate in {run_dir}. "
        f"Expected best_candidate.json, best_agent/, or checkpoint.json + agents/"
    )


def _load_named_agent(run_dir: Path, agent_name: str) -> Tuple[Dict[str, str], str]:
    """Load a specifically-named agent from a RoboPhD run's agent_pool.

    Thin wrapper over the shared RoboPhD.runner_utils.find_named_agent
    that adds the sudoku-specific step of reading the agent source from
    disk and packaging it as a candidate dict. Preserves this module's
    existing (candidate, name) return shape so _load_candidate_from_run_dir
    and its callers don't need to change.
    """
    from RoboPhD.runner_utils import find_named_agent

    name, agent_dir = find_named_agent(run_dir, agent_name)
    return {"agent.py": (agent_dir / "agent.py").read_text()}, name


def main():
    parser = argparse.ArgumentParser(
        description="Test-eval a candidate with mean-of-medians aggregation"
    )
    parser.add_argument("run_dir", type=Path, help="GEPA or RoboPhD run directory")
    parser.add_argument("--test-repeats", type=int, default=9)
    parser.add_argument("--eval-agent", type=str, default=None,
                        help="Name of agent to evaluate from the run's agent_pool. "
                             "RoboPhD runs only. Defaults to the best-ELO agent.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write test_results.json (default: "
                             "<run_dir>/test_results.json, or test_results_<NAME>.json "
                             "when --eval-agent is set)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from evaluator import SudokuEvaluator
    evaluator = SudokuEvaluator()
    candidate, agent_name = _load_candidate_from_run_dir(args.run_dir, agent_name=args.eval_agent)
    logger.info(f"Evaluating agent: {agent_name}")

    median_score, results = test_eval(
        evaluator, candidate, test_repeats=args.test_repeats, agent_name=agent_name
    )

    if args.output:
        output_path = args.output
    elif args.eval_agent:
        output_path = args.run_dir / f"test_results_{args.eval_agent}.json"
    else:
        output_path = args.run_dir / "test_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to {output_path}")


if __name__ == "__main__":
    main()
