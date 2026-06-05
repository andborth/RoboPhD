#!/usr/bin/env python3
"""
Evolve Can't Be Late scheduling agents using RoboPhD's optimize_anything() API.

Pure algorithmic optimization — no LLM calls, no API keys needed for the solver.
Requires trace data: run download_traces.sh first.

Usage:
    # Quick smoke test
    python examples/cant_be_late/main.py --evaluation-budget 60 --num-iterations 2

    # Full run
    python examples/cant_be_late/main.py
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

# Default parallel eval workers when --max-workers is omitted. None = the
# framework default (min(32, cpu+4)). Applied as a task default, not argparse's
# (which stays None so resume can tell "user passed it" from "use the default").
DEFAULT_MAX_WORKERS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve Can't Be Late scheduling agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999, help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500, help="Max evaluator calls across all iterations")

    # Engine
    parser.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd", help="Optimization engine")

    # Task-specific
    parser.add_argument("--simulation-timeout", type=int, default=300, help="Timeout per simulation (seconds)")
    parser.add_argument("--dataset-root", default=None, help="Override trace data location")

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

    from evaluator import CantBeLateEvaluator, load_dataset

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    evaluator = CantBeLateEvaluator(
        simulation_timeout=args.simulation_timeout,
    )

    ds = load_dataset(dataset_root=args.dataset_root)
    train, val = ds["train"], ds["val"]
    logger.info(f"Dataset: {len(train)} train + {len(val)} val")

    # --eval-only: skip optimization, evaluate best agent from a prior run
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        test_data = ds["test"]
        eval_result = eval_run(evaluator=evaluator, dataset=test_data, experiment_dir=args.resume)
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

    # Resolve --max-workers for the gepa/autoresearch engines (which take it
    # directly, not via ConfigManager). Order: explicit flag > value recovered
    # from a resumed checkpoint > task default. The RoboPhD engine instead packs
    # it into engine_overrides below so it applies as a config delta on resume.
    from RoboPhD.runner_utils import read_checkpoint_max_workers
    if args.max_workers is not None:
        effective_max_workers = args.max_workers
    elif args.resume:
        cp_mw = read_checkpoint_max_workers(args.resume)
        effective_max_workers = cp_mw if cp_mw is not None else DEFAULT_MAX_WORKERS
        if cp_mw is not None:
            logger.info(f"Resume: using max_workers={cp_mw} from checkpoint.json "
                        f"(pass --max-workers N to override)")
    else:
        effective_max_workers = DEFAULT_MAX_WORKERS

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
            max_workers=effective_max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=effective_max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    else:
        dataset = train + val
        engine_overrides = {}
        if args.engine_config:
            engine_overrides = json.loads(args.engine_config)
        # Route max_workers through engine_overrides (not RoboPhDConfig's
        # dedicated field, which only applies on a fresh run): explicit flag
        # always packs (takes effect on a training --resume too); with no flag,
        # pack the task default ONLY on a fresh run — never on resume, where it
        # would clobber the original value. (DANGER caller-invariant, api.py.)
        if args.max_workers is not None:
            engine_overrides["max_workers"] = args.max_workers
        elif not args.resume and DEFAULT_MAX_WORKERS is not None and "max_workers" not in engine_overrides:
            engine_overrides["max_workers"] = DEFAULT_MAX_WORKERS
        cfg = RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
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
        task_name="cant_be_late",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = ds["test"]
            logger.info(f"Test evaluation: {len(test_data)} problems")
            # Evaluator is shared with optimize_anything() above, so snapshot its
            # running total to record test-only cost (not evolution + test).
            cost_before = evaluator.total_eval_cost
            eval_result = eval_candidate(
                evaluator=evaluator,
                dataset=test_data,
                candidate=result.best_candidate,
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")

            test_results = {
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": evaluator.total_eval_cost - cost_before,
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
