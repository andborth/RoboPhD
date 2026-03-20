#!/usr/bin/env python3
"""
Standalone test-set evaluation for RoboPhD agents.

Evaluates the best agent from a RoboPhD run (or a directly specified agent)
on the test set using the same protocol as run_gepa.py --eval-test-set.

Usage:
    # Auto-select best agent by ELO from a RoboPhD run
    python scripts/eval_test_set.py --task aime \
        --run-dir ../robophd_runs/robophd/aime_20260227_180324

    # Specify agent directly
    python scripts/eval_test_set.py --task aime \
        --agent-dir ../robophd_runs/robophd/aime_20260227_180324/agents/iter4_verified_enumerator

    # Override defaults
    python scripts/eval_test_set.py --task aime --run-dir ... \
        --test-repeats 3 --task-config '{"solver_model": "gpt-4.1"}'
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.adapters.candidate_utils import extract_candidate
from RoboPhD.adapters.runner_utils import parse_config_arg, find_best_agent, run_test_eval
from RoboPhD.tasks import get_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a RoboPhD agent on the test set"
    )
    parser.add_argument("--task", required=True, help="Task name (aime, codegen)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=Path,
        help="Path to RoboPhD run dir (auto-selects best agent by ELO)",
    )
    group.add_argument(
        "--agent-dir",
        type=Path,
        help="Direct path to agent directory",
    )

    parser.add_argument(
        "--test-repeats", type=int, default=1, help="Number of test repetitions (default: 1)",
    )
    parser.add_argument(
        "--task-config", type=str, default=None, help="JSON string or file for task config overrides"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output path for test_results.json"
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel workers for evaluation (default: cpu_count // 2)",
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

    if args.test_repeats < 1:
        parser.error("--test-repeats must be at least 1")

    # Resolve agent
    if args.run_dir:
        agent_name, agent_dir = find_best_agent(args.run_dir)
        default_output = args.run_dir / "test_results.json"
    else:
        agent_dir = args.agent_dir
        agent_name = agent_dir.name
        results_base = args.runs_dir / "results" / "agent_tests" / args.task / agent_name
        default_output = results_base / "test_results.json"
        if not agent_dir.exists():
            logger.error(f"Agent directory not found: {agent_dir}")
            sys.exit(1)

    output_path = args.output or default_output

    # Load task and merge config
    task = get_task(args.task)
    task_config = parse_config_arg(args.task_config)
    config = {**task.config_defaults, **task_config}

    logger.info(f"Task: {task.name}")
    logger.info(f"Agent: {agent_name} ({agent_dir})")

    # Extract candidate from agent directory
    candidate = extract_candidate(agent_dir, task.file_mapping)

    # Apply test repeats to config so run_test_eval picks them up
    if args.test_repeats > 1:
        config["test_repeats"] = args.test_repeats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_test_eval(
        candidate, task, config, output_path.parent,
        max_workers=args.max_workers,
        metadata={"agent_name": agent_name, "agent_dir": str(agent_dir), "task": task.name},
        logger=logger,
    )

    logger.info(f"Results saved to {output_path}")

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
