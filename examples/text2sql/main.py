#!/usr/bin/env python3
"""
Evolve Text2SQL agents using RoboPhD's optimize_anything() API.

Evolves two coordinated artifacts (agent.py + analyze_db.py) for SQL
generation from natural language questions on the BIRD benchmark.

Usage:
    # Quick smoke test
    python examples/text2sql/main.py --num-iterations 2

    # Full run
    python examples/text2sql/main.py

    # With test evaluation
    python examples/text2sql/main.py --eval-test-set
"""

import argparse
import json
import logging
import os
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

# Default parallel eval workers when --max-workers is omitted. Applied as a task
# default, not argparse's (which stays None so resume can tell "user passed it"
# from "use the default").
DEFAULT_MAX_WORKERS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve Text2SQL agents (BIRD benchmark)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999, help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500, help="Max evaluator calls across all iterations")

    # Engine
    parser.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd", help="Optimization engine")

    # Task-specific
    parser.add_argument("--eval-model", default="haiku-4.5", help="LLM for agent's llm() callable")
    parser.add_argument("--dataset", default="train-filtered", choices=["train-filtered", "train", "dev"], help="Training dataset")
    parser.add_argument("--cost-budget", type=float, default=0.10, help="Per-problem cost budget ($)")
    parser.add_argument("--max-test-sql-calls", type=int, default=5, help="Max test_sql() calls per problem")

    # Infrastructure
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Parallel eval workers (default: 10)"
                             "%(default).0s")  # suppress argparse's auto "(default: None)"
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

    from evaluator import Text2SQLIntegratedEvaluator, load_text2sql_dataset
    from RoboPhD.config import resolve_model_name

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    evaluator = Text2SQLIntegratedEvaluator(
        eval_model=resolve_model_name(args.eval_model),
        dataset=args.dataset,
        cost_budget=args.cost_budget,
        max_test_sql_calls=args.max_test_sql_calls,
    )

    dataset = load_text2sql_dataset(args.dataset)
    logger.info(f"Dataset: {len(dataset)} problems")

    # Resolve --max-workers for the eval path + gepa/autoresearch (which take it
    # directly, not via ConfigManager). Order: explicit flag > value recovered
    # from a resumed checkpoint > task default. The RoboPhD engine packs it into
    # engine_overrides below so it applies as a config delta on resume.
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

    # --eval-only: skip optimization, evaluate best agent from a prior run
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        test_data = load_text2sql_dataset("dev")
        # Rebuild evaluator with dev db_root (dev databases differ from train)
        test_evaluator = Text2SQLIntegratedEvaluator(
            eval_model=resolve_model_name(args.eval_model),
            dataset="dev",
            cost_budget=args.cost_budget,
            max_test_sql_calls=args.max_test_sql_calls,
        )
        eval_result = eval_run(evaluator=test_evaluator, dataset=test_data, experiment_dir=args.resume,
                               config=RoboPhDEvalConfig(max_workers=effective_max_workers))
        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")
        test_path = Path(args.resume) / "test_results.json"
        with open(test_path, "w") as f:
            json.dump({
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": test_evaluator.total_eval_cost,
            }, f, indent=2)
        logger.info(f"Test results saved to {test_path}")
        return

    seed = {
        "agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text(),
        "analyze_db.py": (HERE / "seeds" / "baseline" / "analyze_db.py").read_text(),
    }

    # Build config based on engine choice
    # Text2SQL has no separate val split — GEPA/Autoresearch auto-split from train
    if args.engine in ("gepa", "autoresearch"):
        _robophd_flags = {"--num-iterations", "--engine-config", "--resume", "--extend", "--from-iteration", "--meta-evolution-strategy"}
        passed = {f for f in _robophd_flags if any(a == f or a.startswith(f + "=") for a in sys.argv)}
        if passed:
            logger.warning(f"Flags ignored by {args.engine} engine: {', '.join(sorted(passed))}")

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=args.evaluation_budget,
            max_workers=effective_max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            max_workers=effective_max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
    else:
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

    # Read carve-out: BIRD databases live under benchmark_resources/datasets/
    # (or $BIRD_DATA_DIR if relocated, e.g. to an external drive), and per-problem
    # eval workspaces symlink the .sqlite in by ABSOLUTE resolved path. The
    # evolution sandbox does a realpath check, so we must whitelist wherever the
    # data actually lives — otherwise evolution-time introspection (cat
    # database.sqlite, sqlite3 .schema, smoke-tests against a copy of a problem's
    # DB) is denied. Read scope only; writes are still cwd-only.
    benchmark_resources = HERE.parent.parent / "benchmark_resources"
    extra_read_paths = [str(benchmark_resources.resolve())]
    bird_data_dir = os.environ.get("BIRD_DATA_DIR")
    if bird_data_dir:
        # Resolved so the realpath of the per-problem database.sqlite symlink
        # (which points into BIRD_DATA_DIR) falls inside the read scope.
        extra_read_paths.append(str(Path(bird_data_dir).resolve()))

    result = optimize_anything(
        evaluator=evaluator,
        dataset=dataset,
        seed_candidate=seed,
        objective=objective,
        background=background,
        config=cfg,
        task_name="text2sql",
        extra_read_paths=extra_read_paths,
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = load_text2sql_dataset("dev")
            # Rebuild evaluator with dev db_root (dev databases differ from train)
            test_evaluator = Text2SQLIntegratedEvaluator(
                eval_model=resolve_model_name(args.eval_model),
                dataset="dev",
                cost_budget=args.cost_budget,
                max_test_sql_calls=args.max_test_sql_calls,
            )
            logger.info(f"Test evaluation: {len(test_data)} problems")
            eval_result = eval_candidate(
                evaluator=test_evaluator,
                dataset=test_data,
                candidate=result.best_candidate,
                config=RoboPhDEvalConfig(max_workers=effective_max_workers),
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")

            test_results = {
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": test_evaluator.total_eval_cost,
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
