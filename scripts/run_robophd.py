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
import logging
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.config_manager import ConfigManager, ConfigSource
from RoboPhD.adapters.runner_utils import parse_config_arg, print_task_params, fmt_val
from RoboPhD.tasks import get_task, list_tasks

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RoboPhD evolution on a registered task"
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
        help="Task config: JSON file path or inline JSON string (shared with run_gepa.py)",
    )
    parser.add_argument(
        "--engine-config",
        default=None,
        help="RoboPhD engine config: JSON file path or inline JSON string",
    )

    # Session management (same as researcher.py CLI)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--resume", default=None, help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None, help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None, help="Restart from specific iteration")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("../robophd_runs"),
        help="Root directory for experiment outputs (default: ../robophd_runs)",
    )

    return parser.parse_args()


# Keys consumed by evaluator/dataset factories only — never enter ConfigManager.
_TASK_ONLY_KEYS = {
    "coder_model", "coder_model_tag", "critic_model", "cache_dir",
    "seed_agent", "codegen_split", "runs_dir", "work_dir",
    "codegen_timeout", "critic_timeout", "output_dir", "seed",
    "val_ratio", "reflection_model",
}

# Shared keys that need translation for ConfigManager.
_SHARED_KEY_MAP = {
    "max_workers": "max_concurrent",
}


def split_config(full_config: dict) -> tuple[dict, dict]:
    """
    Split merged config into (researcher_config, task_config).

    researcher_config: keys that ConfigManager validates (engine + shared operational).
    task_config: keys consumed by evaluator/dataset factories only.
    """
    defaults = ConfigManager().get_defaults()
    valid_keys = set(defaults) | _TASK_ONLY_KEYS | set(_SHARED_KEY_MAP)

    researcher_config = {}
    task_config = {}

    for key, value in full_config.items():
        if key not in valid_keys:
            raise SystemExit(
                f"Unknown config key: {key!r}\n"
                f"Use --list-params to see valid parameters."
            )
        if key in _TASK_ONLY_KEYS:
            task_config[key] = value
        elif key in _SHARED_KEY_MAP:
            # Translate to RoboPhD equivalent
            researcher_config[_SHARED_KEY_MAP[key]] = value
            task_config[key] = value  # also keep for evaluator factory
        elif key in defaults:
            researcher_config[key] = value

    return researcher_config, task_config




def _list_params(task):
    """Print all valid parameters for run_robophd.py and exit."""
    print("=" * 70)
    print("VALID CONFIGURATION PARAMETERS — run_robophd.py")
    print("=" * 70)
    print("\nConfig merge order: task defaults -> --task-config -> --engine-config")

    # Task params
    print_task_params(task)

    # Task-only keys (accepted in --config, not forwarded to engine)
    print("Task-only keys (--task-config, consumed by evaluator/dataset factories):")
    for k in sorted(_TASK_ONLY_KEYS):
        default = task.config_defaults.get(k)
        suffix = f" (default: {fmt_val(default)})" if default is not None else ""
        print(f"  - {k}{suffix}")
    print()

    # Shared keys
    print("Shared keys (accepted in --task-config, translated for engine):")
    for src, dst in sorted(_SHARED_KEY_MAP.items()):
        print(f"  - {src}  ->  {dst}")
    print()

    # Engine params from ConfigManager
    defaults = ConfigManager().get_defaults()
    categories = {
        "Sampling": ["examples_per_iteration", "agents_per_iteration"],
        "Evolution": ["evolution_strategy"],
        "Evolution Schedule": ["config_schedule", "weighted_random_configs", "use_weighted_random"],
        "Meta-Evolution": ["meta_evolution_strategy", "meta_evolution_model", "dollar_budget"],
        "Deep Focus": ["new_agent_test_rounds"],
        "Stopping Conditions": ["evaluation_budget"],
        "Performance": ["max_concurrent"],
        "Timeouts": ["phase1_timeout", "phase2_timeout", "evolution_timeout"],
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
    print("  --num-iterations N     Number of evolution iterations (default: 10)")
    print("  --resume PATH          Resume from experiment directory")
    print("  --extend N             Add N more iterations to a resumed run")
    print("  --from-iteration N     Restart from specific iteration")
    print("  --random-seed N        Random seed")
    print("  --runs-dir PATH        Root directory for outputs (default: ../robophd_runs)")
    print()

    print("Example:")
    print("  python scripts/run_robophd.py --task codegen \\")
    print("    --num-iterations 5 \\")
    print('    --task-config \'{"seed_agent": "RoboPhD/codegen_agents/naive_critic", "evaluation_budget": 50}\' \\')
    print('    --engine-config \'{"examples_per_iteration": 12, "evolution_strategy": "refinement_tool_only"}\'')


def main():
    args = parse_args()

    # --- Handle --list-params ---
    if args.list_params:
        task = get_task(args.task)
        _list_params(task)
        sys.exit(0)

    # --- 1. Load task and merge config ---
    task = get_task(args.task)
    task_config = parse_config_arg(args.task_config)
    engine_config = parse_config_arg(args.engine_config)
    full_config = {**task.config_defaults, **task_config, **engine_config}
    full_config["runs_directory"] = str(args.runs_dir)

    logger.info(f"Task: {task.name} — {task.description}")

    # --- 2. Build evaluator and dataset ---
    evaluator = task.evaluator_factory(full_config)
    dataset = task.dataset_builder(full_config)
    logger.info(f"Dataset: {len(dataset)} examples")

    if not dataset:
        logger.error("Empty dataset — check cache directory and task configuration.")
        sys.exit(1)

    # Build runtime_config (non-serializable, passed to ExternalEvaluatorDomain)
    runtime_config = {
        "evaluator_fn": evaluator,
        "dataset": dataset,
        "file_mapping": task.file_mapping,
        "task_objective": task.objective,
        "task_description": task.description,
        "task_name": task.name,
    }

    # --- 3. Split config for ConfigManager vs task-only ---
    # Only validate user-provided keys (task defaults are always valid)
    user_config = {**task_config, **engine_config}
    researcher_config, _ = split_config(user_config)

    # Force external domain (ExternalEvaluatorDomain wraps the task's evaluator)
    researcher_config["domain"] = "external"
    # Meta-evolution needs the real domain name for its domain header prompts
    researcher_config["meta_evolution_domain"] = task.name
    # Runs directory (experiment_dir root)
    researcher_config["runs_directory"] = str(args.runs_dir)

    # Seed agent: set agents_directory so load_initial_agents can find it
    seed_agent = Path(full_config.get("seed_agent", task.default_seed_agent))
    if not seed_agent.exists():
        print(f"Error: Seed agent not found: {seed_agent}")
        sys.exit(1)
    researcher_config["agents_directory"] = str(seed_agent.parent)
    researcher_config["initial_agents"] = [seed_agent.name]

    # Auto-discover all strategies (override ConfigManager default of just one)
    researcher_config["initial_strategies"] = None

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

        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=num_iterations,
            resume_mode=True,
            resume_from_iteration=resume_from,
            resume_checkpoint=checkpoint,
            resume_experiment_dir=experiment_dir,
            runtime_config=runtime_config,
        )

        researcher.run()

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
        researcher.run(initial_agents=researcher_config["initial_agents"])

    logger.info("Done.")


if __name__ == "__main__":
    main()
