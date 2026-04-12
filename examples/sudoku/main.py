#!/usr/bin/env python3
"""
Evolve Sudoku solvers using RoboPhD's optimize_anything() API.

Pure algorithmic optimization — no LLM calls needed for the solver.
Only Claude Code is needed for evolution.

Usage:
    # Quick smoke test
    python examples/sudoku/main.py --evaluation-budget 60 --num-iterations 2

    # Full run
    python examples/sudoku/main.py

    # With test-set evaluation after optimization
    python examples/sudoku/main.py --eval-test-set
"""

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Add project root and example dir to path
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from RoboPhD import optimize_anything, eval_candidate, eval_run, RoboPhDConfig, RoboPhDEvalConfig
from test_eval_candidate import mean_of_medians


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve pure-Python Sudoku solvers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999, help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500, help="Max evaluator calls across all iterations")
    parser.add_argument("--examples-per-iteration", type=int, default=20, help="Problems sampled per iteration")

    # Evolution
    parser.add_argument("--evolution-strategy", default="use_your_judgment", help="Evolution strategy")
    parser.add_argument("--evolution-model", default="opus-4.6", help="Model for evolution AI")

    # Infrastructure
    parser.add_argument("--runs-dir", default="../robophd_runs", help="Root directory for experiment output")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--engine-config", type=str, default=None, help="JSON string with extra engine overrides")

    # Test evaluation
    parser.add_argument("--eval-test-set", action="store_true", help="Run test-set evaluation after optimization")

    # Resume / extend
    parser.add_argument("--eval-only", action="store_true", help="Skip optimization; evaluate best agent from --resume dir on test set")
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None, help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None, help="Restart from a specific iteration")

    return parser.parse_args()


def main():
    args = parse_args()

    from evaluator import SudokuEvaluator, load_dataset

    evaluator = SudokuEvaluator()

    # --eval-only: skip optimization, evaluate best agent from a prior run
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        ds = load_dataset()
        test_data = ds["test"]
        eval_result = eval_run(
            evaluator=evaluator, dataset=test_data, experiment_dir=args.resume,
            config=RoboPhDEvalConfig(test_repeats=10),
        )
        score, _ = mean_of_medians(eval_result.per_example_scores, len(test_data))
        logger.info(
            f"Test score: {score:.3f} (mean of per-puzzle medians, {len(test_data)} puzzles)"
        )
        test_path = Path(args.resume) / "test_results.json"
        with open(test_path, "w") as f:
            json.dump({
                "mean_test_score": score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": len(test_data),
                "test_repeats": 10,
                "aggregation": "per_puzzle_median_then_mean",
                "test_eval_cost_usd": evaluator.total_eval_cost,
            }, f, indent=2)
        logger.info(f"Test results saved to {test_path}")
        return

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    ds = load_dataset()
    dataset = ds["train"]
    logger.info(f"Dataset: {len(dataset)} training problems")

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Build engine overrides
    engine_overrides = {}
    if args.engine_config:
        engine_overrides = json.loads(args.engine_config)

    cfg = RoboPhDConfig(
        num_iterations=args.num_iterations,
        evaluation_budget=args.evaluation_budget,
        examples_per_iteration=args.examples_per_iteration,
        evolution_strategy=args.evolution_strategy,
        evolution_model=args.evolution_model,
        # Solvers are scored on CPU time via time.process_time(), which measures
        # the entire process. max_workers=1 ensures only one solver runs at a
        # time so the measurement is accurate.
        max_workers=1,
        parent_experiments_dir=args.runs_dir,
        random_seed=args.random_seed,
        engine_overrides=engine_overrides or None,
    )

    if args.resume:
        cfg.experiment_dir = args.resume
    if args.extend:
        cfg.extend_iterations = args.extend
    if args.from_iteration:
        cfg.from_iteration = args.from_iteration

    result = optimize_anything(
        evaluator=evaluator,
        dataset=dataset,
        seed_candidate=seed,
        objective=objective,
        background=background,
        config=cfg,
        task_name="sudoku",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: ELO {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = ds["test"]
            logger.info(f"Test evaluation: {len(test_data)} problems (mean of per-puzzle medians, 10 repeats)")
            eval_result = eval_candidate(
                evaluator=evaluator,
                dataset=test_data,
                candidate=result.best_candidate,
                config=RoboPhDEvalConfig(test_repeats=10),
            )
            score, _ = mean_of_medians(eval_result.per_example_scores, len(test_data))
            logger.info(f"Test score: {score:.3f} ({len(test_data)} puzzles)")

            test_results = {
                "mean_test_score": score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": len(test_data),
                "test_repeats": 10,
                "aggregation": "per_puzzle_median_then_mean",
            }
            test_path = result.experiment_dir / "test_results.json"
            with open(test_path, "w") as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Test results saved to {test_path}")


if __name__ == "__main__":
    main()
