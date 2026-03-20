#!/usr/bin/env python3
"""
Generic GEPA runner: optimize any registered task with optimize_anything().

Generic GEPA runner for any registered task.

Usage:
    # Smoke test
    python scripts/run_gepa.py --task codegen \
        --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic", "evaluation_budget": 10}'

    # Full run with engine tuning
    python scripts/run_gepa.py --task codegen \
        --task-config task.json \
        --engine-config '{"val_ratio": 0.05, "reflection_model": "opus-4.6"}'

    # With test-set evaluation
    python scripts/run_gepa.py --task codegen \
        --task-config task.json --eval-test-set

Config merge order: task defaults -> --task-config -> --engine-config
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.config import API_KEY_ENV_VAR
from RoboPhD.adapters.candidate_utils import extract_candidate, materialize_candidate
from RoboPhD.adapters.runner_utils import to_litellm_model, CostTrackingLM, parse_config_arg, print_task_params, split_train_val
from RoboPhD.eval_utils import run_parallel_eval
from RoboPhD.tasks import get_task, list_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# GEPA engine defaults — single source of truth for both _list_params and main().
# Each entry: key -> (default, description)
_GEPA_DEFAULTS = {
    "evaluation_budget": (100, "Max evaluator calls (maps to max_metric_calls)"),
    "max_workers": (None, "Thread pool size (None = Python default: min(32, cpu_count+4))"),
    "seed": (0, "Random seed for reproducibility (matches GEPA default)"),
    "val_ratio": (None, "Fraction of dataset held out for validation (overrides val_size if set)"),
    "val_size": (200, "Validation set size (default 200)"),
    "reflection_model": ("opus-4.6", "Model for GEPA reflection (mutation proposals)"),
    "test_repeats": (1, "Number of test set repetitions (task test_overrides may increase)"),
    "max_test_workers": (None, "Test eval thread pool size (default: max_workers // 2)"),
    "debug_log_probability": (0.1, "Probability (0-1.0) of logging LLM calls for debugging"),
    "only_log_reflection": (True, "If true, debug logging only applies to reflection model"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GEPA optimize_anything() on a registered task"
    )

    parser.add_argument(
        "--task",
        required=True,
        help=f"Task to optimize. Available: {', '.join(list_tasks())}",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List all valid config parameters for the task and engine, then exit",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Task config: JSON file path or inline JSON string (shared with run_robophd.py)",
    )
    parser.add_argument(
        "--engine-config",
        default=None,
        help="GEPA engine config: JSON file path or inline JSON string",
    )
    parser.add_argument(
        "--eval-test-set",
        action="store_true",
        help="Evaluate best candidate on the held-out test set after optimization",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (overrides --runs-dir; default: <runs-dir>/gepa/<task>_<timestamp>)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Root directory for experiment outputs (default: ../robophd_runs relative to repo root)",
    )

    args = parser.parse_args()
    if args.runs_dir is None:
        args.runs_dir = Path("../robophd_runs")
    return args


def _list_params(task):
    """Print all valid parameters for run_gepa.py and exit."""
    print("=" * 70)
    print("VALID CONFIGURATION PARAMETERS — run_gepa.py")
    print("=" * 70)
    print("\nConfig merge order: task defaults -> --task-config -> --engine-config")

    # Task params
    print_task_params(task)

    print("GEPA engine parameters (--engine-config):")
    for k, (default, desc) in _GEPA_DEFAULTS.items():
        print(f"  - {k}: {default!r}  — {desc}")
    print()

    print("CLI-only arguments (not in config):")
    print("  --eval-test-set        Evaluate best candidate on held-out test set")
    print("  --output-dir PATH      Output directory (overrides --runs-dir)")
    print("  --runs-dir PATH        Root directory for outputs (default: ../robophd_runs)")
    print()

    print("Example:")
    print("  python scripts/run_gepa.py --task codegen \\")
    print('    --task-config \'{"seed_agent": "RoboPhD/codegen_agents/naive_critic", "evaluation_budget": 200}\' \\')
    print('    --engine-config \'{"val_ratio": 0.05, "reflection_model": "opus-4.6"}\'')


def main():
    args = parse_args()

    # --- Handle --list-params ---
    if args.list_params:
        task = get_task(args.task)
        _list_params(task)
        sys.exit(0)

    # Resolve API key for reflection model (litellm).
    # Do NOT set ANTHROPIC_API_KEY in the environment — that would cause
    # Claude Code CLI subprocesses (evaluator) to bill through the API
    # account instead of the Claude Max plan.
    api_key = os.environ.get(API_KEY_ENV_VAR) or os.environ.get("ANTHROPIC_API_KEY")

    # --- 1. Load task and merge config ---
    task = get_task(args.task)
    task_config = parse_config_arg(args.task_config)
    engine_config = parse_config_arg(args.engine_config)
    config = {**task.config_defaults, **task_config, **engine_config}

    logger.info(f"Task: {task.name} — {task.description}")

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.runs_dir / "gepa" / f"{task.name}_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(args.output_dir)

    logger.info(f"Output directory: {args.output_dir}")

    # --- 2. Extract seed candidate ---
    seed_agent = Path(config.get("seed_agent", task.default_seed_agent))
    logger.info(f"Extracting seed candidate from {seed_agent}")
    seed_candidate = extract_candidate(seed_agent, task.file_mapping)
    logger.info(
        f"Seed candidate keys: {list(seed_candidate.keys())}, "
        f"sizes: { {k: len(v) for k, v in seed_candidate.items()} }"
    )

    with open(args.output_dir / "seed_candidate.json", "w") as f:
        json.dump(seed_candidate, f, indent=2)

    # --- 3. Build datasets ---
    seed = config.get("seed", _GEPA_DEFAULTS["seed"][0])

    val_ratio = config.get("val_ratio", _GEPA_DEFAULTS["val_ratio"][0])
    val_size = config.get("val_size", _GEPA_DEFAULTS["val_size"][0])

    if task.gepa_datasets_builder is not None:
        # Task provides pre-split datasets (e.g. ARC-AGI matching GEPA's canonical splits)
        # Inject resolved val_size so the builder can resize the pre-split
        if "val_size" not in config and val_size is not None:
            config["val_size"] = val_size
        trainset, valset = task.gepa_datasets_builder(config)
        logger.info(f"Dataset (pre-split): {len(trainset)} train, {len(valset)} val")
    else:
        dataset = task.dataset_builder(config)
        logger.info(f"Dataset: {len(dataset)} examples")

        if not dataset:
            logger.error("Empty dataset — check cache directory and task configuration.")
            sys.exit(1)

        user_provided_keys = set(task_config) | set(engine_config)
        trainset, valset = split_train_val(
            dataset, val_size=val_size, val_ratio=val_ratio,
            user_provided_keys=user_provided_keys, seed=seed, logger=logger,
        )

    # --- 4. Create evaluator ---
    config["work_dir"] = str(args.output_dir / "work")
    evaluator = task.evaluator_factory(config)

    # --- 5. Run GEPA optimization ---
    try:
        from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig
    except ImportError:
        logger.error(
            "GEPA not installed. Install with: pip install gepa\n"
            "See: https://github.com/gepa-ai/gepa"
        )
        sys.exit(1)

    evaluation_budget = config.get("evaluation_budget", _GEPA_DEFAULTS["evaluation_budget"][0])
    max_workers = config.get("max_workers", _GEPA_DEFAULTS["max_workers"][0])

    config_kwargs = {
        "max_metric_calls": evaluation_budget,
        "seed": seed,
        "run_dir": str(args.output_dir / "gepa_run"),
        "parallel": max_workers is None or max_workers > 1,
        "max_workers": max_workers,
        "track_best_outputs": True,
    }

    engine_cfg = EngineConfig(**config_kwargs)
    gepa_config = GEPAConfig(engine=engine_cfg)

    reflection_model = config.get("reflection_model", _GEPA_DEFAULTS["reflection_model"][0])
    debug_log_prob = config.get("debug_log_probability", _GEPA_DEFAULTS["debug_log_probability"][0])
    only_log_reflection = config.get("only_log_reflection", _GEPA_DEFAULTS["only_log_reflection"][0])

    from gepa.optimize_anything import ReflectionConfig
    litellm_model = to_litellm_model(reflection_model)
    reflection_lm = CostTrackingLM(
        litellm_model,
        api_key=api_key,
        debug_log_probability=debug_log_prob,
        debug_log_dir=args.output_dir / "debug_logs" / "reflection",
    )
    gepa_config.reflection = ReflectionConfig(reflection_lm=reflection_lm)

    # Configure evaluator debug logging
    if not only_log_reflection:
        config["debug_log_dir"] = str(args.output_dir / "debug_logs" / "eval")
    else:
        config["debug_log_probability"] = 0.0

    logger.info(
        f"Starting GEPA optimization with evaluation_budget={evaluation_budget}, "
        f"reflection_model={reflection_model}"
    )

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=task.objective,
        background=task.background,
        config=gepa_config,
    )

    # --- 6. Save results ---
    best_candidate = result.best_candidate
    val_scores = getattr(result, "val_aggregate_scores", None) or []
    candidates = getattr(result, "candidates", None) or []

    best_val = max(val_scores) if val_scores else 0.0
    reflection_cost = reflection_lm.total_cost if reflection_lm else 0.0
    eval_cost = getattr(evaluator, "total_eval_cost", 0.0)
    total_cost = eval_cost + reflection_cost
    logger.info(f"Optimization complete. Best validation score: {best_val:.3f}")
    logger.info(
        f"Cost: ${total_cost:.2f} total "
        f"(eval: ${eval_cost:.2f}, reflection: ${reflection_cost:.2f})"
    )

    with open(args.output_dir / "best_candidate.json", "w") as f:
        json.dump(best_candidate, f, indent=2)

    best_agent_dir = args.output_dir / "best_agent"
    materialize_candidate(best_candidate, best_agent_dir, task.file_mapping, name="gepa_best")
    logger.info(f"Best agent materialized at: {best_agent_dir}")

    all_candidates_dir = args.output_dir / "all_candidates"
    for idx, cand in enumerate(candidates):
        cand_dir = all_candidates_dir / f"candidate_{idx}"
        score = val_scores[idx] if idx < len(val_scores) else None
        name = f"candidate_{idx}"
        if score is not None:
            name += f"_val{score:.3f}"
        materialize_candidate(cand, cand_dir, task.file_mapping, name=name)
    if candidates:
        logger.info(f"All {len(candidates)} candidates materialized at: {all_candidates_dir}")

    summary = {
        "task": task.name,
        "objective": task.objective,
        "background": (task.background[:500] + "...") if len(task.background) > 500 else task.background,
        "background_sha256": hashlib.sha256(task.background.encode()).hexdigest()[:16] if task.background else None,
        "seed_agent": str(seed_agent),
        "evaluation_budget": evaluation_budget,
        "val_ratio": val_ratio,
        "val_size": val_size if val_ratio is None else None,
        "seed": seed,
        "max_workers": max_workers,
        "reflection_model": reflection_model,
        "total_evaluations": evaluator.total_evaluations,  # required evaluator attribute
        "train_size": len(trainset),
        "val_size": len(valset),
        "num_candidates_explored": len(candidates),
        "best_val_score": best_val,
        "val_scores": val_scores,
        "config": {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, type(None)))},
        "cost": {
            "eval_cost_usd": eval_cost,
            "reflection_cost_usd": reflection_cost,
            "reflection_calls": reflection_lm.call_count if reflection_lm else 0,
            "reflection_input_tokens": reflection_lm.total_input_tokens if reflection_lm else 0,
            "reflection_output_tokens": reflection_lm.total_output_tokens if reflection_lm else 0,
            "total_cost_usd": total_cost,
        },
    }
    with open(args.output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- 7. Optional test-set evaluation ---
    if args.eval_test_set:
        logger.info("Evaluating best candidate on test set...")
        test_config = {**config, **task.test_overrides}  # test_overrides last: must override training defaults
        test_examples = task.dataset_builder(test_config)
        test_repeats = test_config.get("test_repeats", 1)
        test_examples = test_examples * test_repeats
        logger.info(f"Test set: {len(test_examples)} problems ({len(test_examples) // test_repeats} unique × {test_repeats})")

        test_config["work_dir"] = str(args.output_dir / "test_work")
        test_evaluator = task.evaluator_factory(test_config)

        test_workers = test_config.get("max_test_workers") or max_workers or max(1, min(32, (os.cpu_count() or 4) + 4) // 2)
        logger.info(f"Test evaluation: {len(test_examples)} problems, {test_workers} workers")
        eval_timeout = test_config.get("eval_timeout", 300)

        result = run_parallel_eval(
            test_evaluator, best_candidate, test_examples,
            max_workers=test_workers, eval_timeout=eval_timeout,
        )

        test_eval_cost = getattr(test_evaluator, 'total_eval_cost', 0.0)
        test_results = {**result["test_results"], "test_eval_cost_usd": test_eval_cost}
        with open(args.output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    logger.info(f"Done. Results saved to {args.output_dir}")

    # Force-exit if any non-daemon threads are still alive — Python's atexit
    # handler blocks on t.join() for hung threads, hanging the process.
    # This catches eval timeout leaks, litellm/httpx connection pool threads, etc.
    alive = [t for t in threading.enumerate()
             if t is not threading.main_thread() and t.is_alive() and not t.daemon]
    if alive:
        names = ", ".join(t.name for t in alive)
        logger.info(f"Force-exiting ({len(alive)} non-daemon thread(s) still running: {names})")
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
