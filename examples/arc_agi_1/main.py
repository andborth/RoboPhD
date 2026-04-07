#!/usr/bin/env python3
"""
Evolve ARC-AGI-1 solving agents using RoboPhD's optimize_anything() API.

Usage:
    # Quick smoke test
    python main.py --evaluation-budget 60 --num-iterations 2

    # Full run
    python main.py

    # With test evaluation
    python main.py --eval-test-set

    # Resume a prior run
    python main.py --resume ../../robophd_runs/robophd/optimize_anything_20260401_120000
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

from RoboPhD import optimize_anything, eval_candidate, RoboPhDConfig, RoboPhDEvalConfig
from RoboPhD.adapters.candidate_utils import extract_candidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evolve ARC-AGI-1 agents")

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999)
    parser.add_argument("--evaluation-budget", type=int, default=1500)
    parser.add_argument("--examples-per-iteration", type=int, default=20)

    # Evolution
    parser.add_argument("--evolution-strategy", default="use_your_judgment")
    parser.add_argument("--evolution-model", default="opus-4.6")

    # Task-specific
    parser.add_argument("--paper-config", action="store_true",
                        help="Use paper settings: Gemini 3.1 Flash Lite + $0.25 cost budget "
                             "(default: Gemini 2.5 Flash Lite + $0.10 budget)")
    parser.add_argument("--max-llm-calls", type=int, default=10)

    # Infrastructure
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--runs-dir", default="../robophd_runs")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--engine-config", type=str, default=None,
                        help="JSON string with extra engine overrides")

    # Test evaluation
    parser.add_argument("--eval-test-set", action="store_true")
    parser.add_argument("--test-repeats", type=int, default=1)

    # Resume / extend
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None,
                        help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None,
                        help="Restart from a specific iteration")

    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy import evaluator (requires dspy/datasets)
    from evaluator import (
        ArcAGI1Evaluator, FILE_MAPPING, DEFAULT_SOLVER_MODEL,
        load_arc_train_val, load_arc_test,
    )

    # Select config tier
    if args.paper_config:
        solver_model = "openrouter/google/gemini-3.1-flash-lite-preview"
        cost_budget = 0.25
        logger.info("Using paper config: Gemini 3.1 Flash Lite, $0.25 budget")
    else:
        solver_model = DEFAULT_SOLVER_MODEL
        cost_budget = 0.10
        logger.info(f"Using default config: {solver_model}, ${cost_budget:.2f} budget")

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()
    background = background.replace("{cost_budget}", f"{cost_budget:.2f}")

    evaluator = ArcAGI1Evaluator(
        solver_model=solver_model,
        max_llm_calls=args.max_llm_calls,
        cost_budget=cost_budget,
    )

    train, val = load_arc_train_val()
    dataset = train + val
    logger.info(f"Dataset: {len(dataset)} problems ({len(train)} train + {len(val)} val)")

    seed = extract_candidate(HERE / "seeds" / "baseline", FILE_MAPPING)

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
        max_workers=args.max_workers,
        parent_experiments_dir=args.runs_dir,
        random_seed=args.random_seed,
        eval_timeout=600,
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
        task_name="arc_agi_1",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: ELO {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = load_arc_test()
            logger.info(f"Test evaluation: {len(test_data)} problems x {args.test_repeats} repeats")
            eval_result = eval_candidate(
                evaluator=evaluator,
                dataset=test_data,
                candidate=result.best_candidate,
                config=RoboPhDEvalConfig(
                    test_repeats=args.test_repeats,
                    eval_timeout=600,
                ),
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")

            # Save test results
            test_results = {
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
            }
            test_path = result.experiment_dir / "test_results.json"
            with open(test_path, "w") as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Test results saved to {test_path}")


if __name__ == "__main__":
    main()
