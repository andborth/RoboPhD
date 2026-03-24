#!/usr/bin/env python3
"""
Generic RoboPhD runner: evolve agents for any registered task.

Wires a task definition into researcher.py via ExternalEvaluatorDomain.

Usage:
    # Fresh run
    python scripts/run_robophd.py --task codegen \
        --num-iterations 5 \
        --task-config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic", "evaluation_budget": 50}' \
        --engine-config '{"examples_per_iteration": 3}'

    # Resume
    python scripts/run_robophd.py --task codegen \
        --resume ../robophd_runs/robophd/codegen_20260225_120000 \
        --task-config task.json --engine-config robophd.json

    # Extend
    python scripts/run_robophd.py --task codegen \
        --resume ../robophd_runs/robophd/codegen_20260225_120000 --extend 5 \
        --task-config task.json --engine-config robophd.json

Config merge order: task defaults -> --task-config -> --engine-config
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.config_manager import ConfigManager, ConfigSource, IMMUTABLE_PARAMS
from RoboPhD.adapters.runner_utils import parse_config_arg, print_task_params, fmt_val
from RoboPhD.tasks import get_task, list_tasks

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RoboPhD evolution on a registered task"
    )

    parser.add_argument(
        "--task",
        default=None,
        help=f"Task to optimize. Available: {', '.join(list_tasks())}. Inferred from checkpoint on --resume.",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List all valid config parameters for the task and engine, then exit",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Task config: JSON file path or inline JSON string (shared with run_gepa.py)",
    )
    parser.add_argument(
        "--engine-config",
        default=None,
        help="RoboPhD engine config: JSON file path or inline JSON string",
    )

    parser.add_argument(
        "--eval-test-set",
        action="store_true",
        help="Evaluate best agent on held-out test set after evolution",
    )

    # Session management (same as researcher.py CLI)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--resume", default=None, help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None, help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None, help="Restart from specific iteration")
    parser.add_argument("--random-seed", type=int, default=None)
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


# Keys consumed by evaluator/dataset factories only — never enter ConfigManager.
# Accepted by all tasks regardless of domain.
_UNIVERSAL_TASK_KEYS = {
    "seed_agent", "seed", "output_dir", "work_dir", "runs_dir",
    "val_ratio", "reflection_model", "eval_timeout",
}


def split_config(full_config: dict, task: "TaskDefinition") -> tuple[dict, dict]:
    """
    Split merged config into (researcher_config, task_config).

    researcher_config: keys that ConfigManager validates (engine parameters).
    task_config: keys consumed by evaluator/dataset factories only.

    Routing priority (first match wins):
      1. task_only_keys   → task_config (+ researcher_config if key also exists
         in ConfigManager defaults, e.g. evaluation_budget)
      2. ConfigManager defaults → researcher_config only

    Valid keys: union of both sets. Unknown keys raise SystemExit.
    """
    defaults = ConfigManager().get_defaults()
    task_only_keys = _UNIVERSAL_TASK_KEYS | set(task.config_defaults)
    valid_keys = set(defaults) | task_only_keys

    researcher_config = {}
    task_config = {}

    for key, value in full_config.items():
        if key not in valid_keys:
            raise SystemExit(
                f"Unknown config key: {key!r}\n"
                f"Use --list-params to see valid parameters."
            )
        if key in task_only_keys:
            task_config[key] = value
            if key in defaults:
                researcher_config[key] = value
        elif key in defaults:
            researcher_config[key] = value

    return researcher_config, task_config




def _infer_task_from_resume(resume_path: str) -> str:
    """Infer task name from checkpoint config or directory name."""
    import json
    experiment_dir = Path(resume_path)
    available = list_tasks()

    # Try checkpoint's meta_evolution_domain (set by run_robophd.py on new runs)
    checkpoint_file = experiment_dir / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            configs = checkpoint.get("config_manager", {}).get("iteration_configs", {})
            if configs:
                first_config = next(iter(configs.values()))
                task_name = first_config.get("meta_evolution_domain")
                if task_name and task_name in available:
                    return task_name
        except (json.JSONDecodeError, IOError):
            pass

    # Fall back to directory name pattern: <task>_<timestamp>
    # Sort by length descending so "arc_agi_1" matches before "arc_agi"
    dir_name = experiment_dir.name
    for task_name in sorted(available, key=len, reverse=True):
        if dir_name.startswith(f"{task_name}_"):
            return task_name

    print(
        f"Error: Cannot infer task from {resume_path}. "
        f"Use --task to specify (available: {', '.join(available)})"
    )
    sys.exit(1)


def _list_params(task):
    """Print all valid parameters for run_robophd.py and exit."""
    print("=" * 70)
    print("VALID CONFIGURATION PARAMETERS — run_robophd.py")
    print("=" * 70)
    print("\nConfig merge order: task defaults -> --task-config -> --engine-config")

    # Task params
    print_task_params(task)

    # Task-only keys (accepted in --config, not forwarded to engine)
    relevant = sorted(_UNIVERSAL_TASK_KEYS | set(task.config_defaults))
    print("Task-only keys (--task-config, consumed by evaluator/dataset factories):")
    for k in relevant:
        default = task.config_defaults.get(k)
        suffix = f" (default: {fmt_val(default)})" if default is not None else ""
        print(f"  - {k}{suffix}")
    print()

    # Engine params from ConfigManager
    defaults = ConfigManager().get_defaults()
    categories = {
        "Sampling": ["examples_per_iteration", "agents_per_iteration"],
        "Evolution": ["evolution_strategy", "oldest_agent_wins_ties", "random_agent_wins_ties", "include_evolution_rankings"],
        "Evolution Schedule": ["config_schedule", "weighted_random_configs", "use_weighted_random"],
        "Meta-Evolution": ["meta_evolution_strategy", "meta_evolution_model", "dollar_budget"],
        "Deep Focus": ["new_agent_test_rounds", "new_agent_test_round_offset"],
        "Stopping Conditions": ["evaluation_budget"],
        "Performance": ["max_workers"],
        "Timeouts": ["evolution_timeout"],
        "Caching": ["eval_result_cache"],
        "Other": ["debug_log_probability"],
    }

    print("Engine parameters (--engine-config, forwarded to ConfigManager):")
    for category, params in categories.items():
        print(f"  {category}:")
        for p in params:
            if p in defaults:
                print(f"    - {p}: {fmt_val(defaults[p])}")
    print(f"\n  See also: python RoboPhD/researcher.py --list-config-parameters")
    print()

    print("CLI-only arguments (not in config):")
    print("  --num-iterations N     Number of evolution iterations (default: 5)")
    print("  --resume PATH          Resume from experiment directory")
    print("  --extend N             Add N more iterations to a resumed run")
    print("  --from-iteration N     Restart from specific iteration")
    print("  --random-seed N        Random seed")
    print("  --runs-dir PATH        Root directory for outputs (default: ../robophd_runs)")
    print()

    print("Example:")
    print("  python scripts/run_robophd.py --task aime \\")
    print("    --num-iterations 5 \\")
    print('    --engine-config \'{"examples_per_iteration": 12, "evolution_strategy": "refinement"}\'')


def main():
    args = parse_args()

    # --- Handle --list-params ---
    if args.list_params:
        if not args.task:
            print("Error: --list-params requires --task")
            sys.exit(1)
        task = get_task(args.task)
        _list_params(task)
        sys.exit(0)

    # --- Infer task on --resume if not provided ---
    if not args.task:
        if not args.resume:
            print(f"Error: --task is required (available: {', '.join(list_tasks())})")
            sys.exit(1)
        args.task = _infer_task_from_resume(args.resume)

    # --- 1. Load task and merge config ---
    task = get_task(args.task)
    task_config = parse_config_arg(args.task_config)
    engine_config = parse_config_arg(args.engine_config)
    full_config = {**task.config_defaults, **task_config, **engine_config}
    full_config["runs_directory"] = str(args.runs_dir)  # for evaluator/dataset factories (e.g., codegen cache dir)

    logger.info(f"Task: {task.name} — {task.description}")

    # --- 2. Build evaluator and dataset ---
    evaluator = task.evaluator_factory(full_config)
    dataset = task.dataset_builder(full_config)
    logger.info(f"Dataset: {len(dataset)} examples")

    if not dataset:
        logger.error("Empty dataset — check cache directory and task configuration.")
        sys.exit(1)

    # Build runtime_config (non-serializable, passed to ExternalEvaluatorDomain)
    # Include eval_timeout from the merged config so task defaults reach the domain.
    runtime_config = {
        "evaluator_fn": evaluator,
        "dataset": dataset,
        "file_mapping": task.file_mapping,
        "task_objective": task.objective,
        "task_description": task.description,
        "task_background": task.background,
        "task_name": task.name,
        "diagnostic_files": task.diagnostic_files,
        "runs_dir": str(args.runs_dir),
        "eval_timeout": full_config.get("eval_timeout"),
    }

    # --- 3. Split config for ConfigManager vs task-only ---
    # Seed researcher_config from task defaults so task-level settings
    # (eval_model, max_workers, evaluation_budget) reach ConfigManager.
    # User-provided keys override task defaults.
    defaults = ConfigManager().get_defaults()
    seeded_config = {k: v for k, v in task.config_defaults.items() if k in defaults}
    user_config = {**task_config, **engine_config}
    user_config["runs_directory"] = str(args.runs_dir)  # CLI arg always wins
    seeded_config.update(user_config)
    researcher_config, _ = split_config(seeded_config, task)

    # Force external domain (ExternalEvaluatorDomain wraps the task's evaluator)
    researcher_config["domain"] = "external"
    # Meta-evolution needs the real domain name for its domain header prompts
    researcher_config["meta_evolution_domain"] = task.name

    # Seed agent: set agents_directory so load_initial_agents can find it
    seed_agent = Path(full_config.get("seed_agent", task.default_seed_agent))
    if not seed_agent.exists():
        print(f"Error: Seed agent not found: {seed_agent}")
        sys.exit(1)
    researcher_config["agents_directory"] = str(seed_agent.parent)
    researcher_config["initial_agents"] = [seed_agent.name]

    # --- 4. Fresh vs Resume ---
    from RoboPhD.researcher import ParallelAgentResearcher

    if args.resume:
        # Resume from checkpoint
        experiment_dir = Path(args.resume)
        if not experiment_dir.exists():
            print(f"Error: Experiment directory not found: {experiment_dir}")
            sys.exit(1)

        checkpoint = ParallelAgentResearcher.load_checkpoint(experiment_dir)

        if "config_manager" not in checkpoint:
            print("Error: Checkpoint missing ConfigManager data")
            sys.exit(1)

        config_manager = ConfigManager.from_checkpoint(checkpoint["config_manager"])
        last_completed = checkpoint["last_completed_iteration"]
        checkpoint_num_iterations = checkpoint.get("num_iterations", last_completed)

        if args.from_iteration:
            resume_from = args.from_iteration
            config_manager.clear_from_iteration(resume_from)
            logger.info(f"Restarting from iteration {resume_from}")
        else:
            resume_from = last_completed + 1
            logger.info(f"Auto-resuming from iteration {resume_from}")

        if args.extend:
            num_iterations = checkpoint_num_iterations + args.extend
            checkpoint["num_iterations"] = num_iterations
            logger.info(f"Extending by {args.extend} iterations (to {num_iterations} total)")
        else:
            num_iterations = checkpoint_num_iterations

        # Apply CLI overrides (--engine-config) as a delta on the resume iteration
        if engine_config:
            cli_delta, task_only = split_config(engine_config, task)
            if task_only:
                raise SystemExit(
                    f"Task-only keys not supported in --engine-config on resume: {list(task_only)}\n"
                    f"Use --task-config for task-level overrides."
                )
            immutable = [k for k in cli_delta if k in IMMUTABLE_PARAMS]
            if immutable:
                raise SystemExit(
                    f"Immutable keys cannot be changed on resume: {immutable}\n"
                    f"These are fixed at run creation: {', '.join(IMMUTABLE_PARAMS)}"
                )
            if cli_delta:
                config_manager.apply_delta(
                    iteration=resume_from,
                    delta=cli_delta,
                    source=ConfigSource.CLI,
                    rationale=f"CLI --engine-config override on resume: {cli_delta}",
                )
                logger.info(f"Applied CLI config overrides at iteration {resume_from}: {cli_delta}")

        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=num_iterations,
            resume_mode=True,
            resume_from_iteration=resume_from,
            resume_checkpoint=checkpoint,
            resume_experiment_dir=experiment_dir,
            runtime_config=runtime_config,
        )

        completed = researcher.run()

    else:
        # Fresh start
        config_manager = ConfigManager()
        config_manager.set_initial_config(researcher_config, ConfigSource.CLI)

        num_iterations = args.num_iterations
        random_seed = args.random_seed if args.random_seed is not None else random.randint(0, 10000)

        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=num_iterations,
            random_seed=random_seed,
            runtime_config=runtime_config,
        )

        # load_initial_agents will find the seed in agents_directory and copy it
        completed = researcher.run(initial_agents=researcher_config["initial_agents"])

    logger.info("Done.")

    # --- Optional test-set evaluation ---
    if args.eval_test_set:
        if not completed:
            logger.info("Skipping test-set evaluation — run ended early due to failure")
        else:
            from RoboPhD.adapters.runner_utils import find_best_agent, run_test_eval
            from RoboPhD.adapters.candidate_utils import extract_candidate

            experiment_dir = researcher.experiment_dir
            logger.info("Evaluating best agent on test set...")
            agent_name, agent_dir = find_best_agent(experiment_dir)
            candidate = extract_candidate(agent_dir, task.file_mapping)

            run_test_eval(
                candidate, task, full_config, experiment_dir,
                max_workers=full_config.get("max_workers"),
                metadata={"agent_name": agent_name, "agent_dir": str(agent_dir), "task": task.name},
                logger=logger,
            )

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
