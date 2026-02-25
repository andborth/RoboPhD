#!/usr/bin/env python3
"""
Generic RoboPhD runner: evolve agents for any registered task.

Wires a task definition into researcher.py via ExternalEvaluatorDomain.

Usage:
    # Fresh run
    python scripts/run_robophd.py --task codegen \
        --num-iterations 5 \
        --config '{"seed_agent": "RoboPhD/codegen_agents/naive_critic", "evaluation_budget": 50}' \
        --engine-config '{"contexts_per_iteration": 3}'

    # Resume
    python scripts/run_robophd.py --task codegen \
        --resume evolution/robophd_20260225_120000 \
        --config task.json --engine-config robophd.json

    # Extend
    python scripts/run_robophd.py --task codegen \
        --resume evolution/robophd_20260225_120000 --extend 5 \
        --config task.json --engine-config robophd.json

Config merge order: task defaults -> --config -> --engine-config
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.config import API_KEY_ENV_VAR
from RoboPhD.config_manager import ConfigManager, ConfigSource
from RoboPhD.adapters.runner_utils import parse_config_arg
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
        "--config",
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

    researcher_config = {}
    task_config = {}

    for key, value in full_config.items():
        if key in _TASK_ONLY_KEYS:
            task_config[key] = value
        elif key in _SHARED_KEY_MAP:
            # Translate to RoboPhD equivalent
            researcher_config[_SHARED_KEY_MAP[key]] = value
            task_config[key] = value  # also keep for evaluator factory
        elif key in defaults:
            researcher_config[key] = value
        else:
            # Unknown key — put in task_config (factory might use it)
            task_config[key] = value

    return researcher_config, task_config


def resolve_api_key() -> str:
    """Resolve API key from environment."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    return api_key



def main():
    args = parse_args()

    # --- 1. Load task and merge config ---
    task = get_task(args.task)
    task_config = parse_config_arg(args.config)
    engine_config = parse_config_arg(args.engine_config)
    full_config = {**task.config_defaults, **task_config, **engine_config}

    logger.info(f"Task: {task.name} — {task.description}")

    # Resolve API key
    api_key = resolve_api_key()
    if not api_key:
        print(f"Error: API key required. Set {API_KEY_ENV_VAR} environment variable.")
        sys.exit(1)

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
    }

    # --- 3. Split config for ConfigManager vs task-only ---
    researcher_config, _ = split_config(full_config)

    # Force external domain
    researcher_config["domain"] = "external"

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

        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=num_iterations,
            resume_mode=True,
            resume_from_iteration=resume_from,
            resume_checkpoint=checkpoint,
            resume_experiment_dir=experiment_dir,
            api_key=api_key,
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
            api_key=api_key,
            runtime_config=runtime_config,
        )

        # load_initial_agents will find the seed in agents_directory and copy it
        researcher.run(initial_agents=researcher_config["initial_agents"])

    logger.info("Done.")


if __name__ == "__main__":
    main()
