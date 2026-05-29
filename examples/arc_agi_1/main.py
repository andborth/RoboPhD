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

from RoboPhD import optimize_anything, eval_candidate, eval_run, RoboPhDConfig, GEPAConfig, AutoresearchConfig, RoboPhDEvalConfig

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
        description="Evolve ARC-AGI-1 agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999, help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500, help="Max evaluator calls across all iterations")

    # Engine
    parser.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd", help="Optimization engine")

    # Task-specific
    parser.add_argument("--paper-config", action="store_true",
                        help="Paper settings: high reasoning + $0.25 budget + 10 LLM calls "
                             "(default: medium reasoning + $0.10 + 20 calls)")

    # Infrastructure
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel eval workers (None = Python default)")
    parser.add_argument("--runs-dir", default="../robophd_runs", help="Root directory for experiment output (default: %(default)s)")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--engine-config", type=str, default=None, help="JSON overrides (e.g. evolution_strategy, evolution_model, examples_per_iteration)")
    parser.add_argument("--meta-evolution-strategy", default=None,
                        help="Meta-evolution strategy (e.g. train_a_winner); default off. "
                             "Cadence and first iteration default to (3, 4); override via --engine-config.")

    # Test evaluation
    parser.add_argument("--eval-test-set", action="store_true", help="Run test-set evaluation after optimization")
    parser.add_argument("--eval-only", action="store_true", help="Skip optimization; evaluate best agent from --resume dir on test set")
    # Resume / extend
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None, help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None, help="Restart from a specific iteration")

    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy import evaluator (requires dspy/datasets)
    from evaluator import (
        ArcAGI1Evaluator, DEFAULT_SOLVER_MODEL,
        load_arc_train_val, load_arc_test,
    )

    # Select config tier.
    # High reasoning scores 3.3x better on the seed (29% vs 8.7%) but costs
    # 6.8x more ($0.027 vs $0.004/problem) and runs 7.5x slower. The default
    # tier trades seed accuracy for cheaper evals and gives evolution 20 LLM
    # calls to build multi-step strategies.
    if args.paper_config:
        solver_model = DEFAULT_SOLVER_MODEL
        cost_budget = 0.25
        reasoning_effort = "high"
        max_llm_calls = 10
    else:
        solver_model = DEFAULT_SOLVER_MODEL
        cost_budget = 0.10
        reasoning_effort = "medium"
        max_llm_calls = 20

    logger.info(f"Solver config: model={solver_model}, budget=${cost_budget:.2f}, "
                f"reasoning={reasoning_effort}, max_calls={max_llm_calls}")

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()
    background = background.replace("{cost_budget}", f"{cost_budget:.2f}")
    background = background.replace("{max_llm_calls}", str(max_llm_calls))

    evaluator = ArcAGI1Evaluator(
        solver_model=solver_model,
        max_llm_calls=max_llm_calls,
        cost_budget=cost_budget,
        reasoning_effort=reasoning_effort,
    )

    train, val = load_arc_train_val()
    logger.info(f"Dataset: {len(train)} train + {len(val)} val")

    # --eval-only: skip optimization, evaluate best agent from a prior run
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        test_data = load_arc_test()
        eval_result = eval_run(
            evaluator=evaluator, dataset=test_data, experiment_dir=args.resume,
            config=RoboPhDEvalConfig(eval_timeout=600),
        )
        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")
        test_path = Path(args.resume) / "test_results.json"
        with open(test_path, "w") as f:
            json.dump({
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": evaluator.total_eval_cost,
            }, f, indent=2)
        logger.info(f"Test results saved to {test_path}")
        return

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Build config based on engine choice
    if args.engine in ("gepa", "autoresearch"):
        _robophd_flags = {"--num-iterations", "--engine-config", "--resume", "--extend", "--from-iteration", "--meta-evolution-strategy"}
        passed = {f for f in _robophd_flags if any(a == f or a.startswith(f + "=") for a in sys.argv)}
        if passed:
            logger.warning(f"Flags ignored by {args.engine} engine: {', '.join(sorted(passed))}")

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=args.max_workers,
            eval_timeout=600,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=args.max_workers,
            eval_timeout=600,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    else:
        dataset = train + val
        engine_overrides = {}
        if args.engine_config:
            engine_overrides = json.loads(args.engine_config)
        cfg = RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            max_workers=args.max_workers,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
            eval_timeout=600,
            meta_evolution_strategy=args.meta_evolution_strategy,
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
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = load_arc_test()
            logger.info(f"Test evaluation: {len(test_data)} problems")
            eval_result = eval_candidate(
                evaluator=evaluator,
                dataset=test_data,
                candidate=result.best_candidate,
                config=RoboPhDEvalConfig(eval_timeout=600),
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")

            # Save test results
            test_results = {
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": evaluator.total_eval_cost,
            }
            test_path = result.experiment_dir / "test_results.json"
            with open(test_path, "w") as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Test results saved to {test_path}")


if __name__ == "__main__":
    from RoboPhD.eval_utils import force_exit_if_threads_leaked
    try:
        main()
    finally:
        force_exit_if_threads_leaked()
