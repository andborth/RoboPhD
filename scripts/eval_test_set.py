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
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.adapters.candidate_utils import extract_candidate
from RoboPhD.adapters.runner_utils import parse_config_arg
from RoboPhD.tasks import get_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def find_best_agent(run_dir: Path) -> tuple[str, Path]:
    """Find the best agent by ELO from a checkpoint.json.

    Returns (agent_name, agent_dir).
    """
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint.json found in {run_dir}")

    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    perf_records = ckpt.get("performance_records", {})
    agent_pool = ckpt.get("agent_pool", {})

    if not perf_records:
        raise ValueError(f"No performance records in {checkpoint_path}")

    # Find agent with highest ELO
    best_id = max(perf_records, key=lambda k: perf_records[k]["elo"])
    best_perf = perf_records[best_id]

    logger.info(
        f"Best agent: {best_id} "
        f"(ELO: {best_perf['elo']:.0f}, "
        f"accuracy: {best_perf['mean_accuracy']:.1f}%, "
        f"tests: {best_perf['test_count']})"
    )

    # Resolve package_dir (stored as relative path from experiment dir)
    agent_info = agent_pool.get(best_id)
    if not agent_info or "package_dir" not in agent_info:
        raise ValueError(f"Agent {best_id} not found in agent_pool or missing package_dir")

    agent_dir = run_dir / agent_info["package_dir"]
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    return best_id, agent_dir


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
        "--test-repeats", type=int, default=5, help="Number of test repetitions (default: 5)"
    )
    parser.add_argument(
        "--task-config", type=str, default=None, help="JSON string or file for task config overrides"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output path for test_results.json"
    )
    args = parser.parse_args()

    # Resolve agent
    if args.run_dir:
        agent_name, agent_dir = find_best_agent(args.run_dir)
        default_output = args.run_dir / "test_results.json"
    else:
        agent_dir = args.agent_dir
        agent_name = agent_dir.name
        default_output = agent_dir / "test_results.json"
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

    # Build test dataset (matching run_gepa.py protocol)
    test_config = {**config, "codegen_split": "test", "aime_split": "test"}
    test_examples = task.dataset_builder(test_config)
    test_repeats = args.test_repeats
    test_examples = test_examples * test_repeats
    logger.info(
        f"Test set: {len(test_examples)} problems "
        f"({len(test_examples) // test_repeats} unique × {test_repeats})"
    )

    # Create evaluator
    output_dir = output_path.parent
    test_config["work_dir"] = str(output_dir / "test_work")
    test_evaluator = task.evaluator_factory(test_config)

    # Sequential eval loop (matching run_gepa.py)
    scores = []
    for i, example in enumerate(test_examples):
        score, diag = test_evaluator(candidate, example)
        scores.append(score)
        if (i + 1) % 10 == 0:
            logger.info(
                f"Test progress: {i+1}/{len(test_examples)}, "
                f"running accuracy: {sum(scores)/len(scores)*100:.1f}%"
            )

    test_accuracy = sum(scores) / len(scores) * 100 if scores else 0.0
    logger.info(f"Test set accuracy: {test_accuracy:.1f}% ({sum(scores):.0f}/{len(scores)})")

    # Write results
    test_results = {
        "test_accuracy": test_accuracy,
        "test_total": len(scores),
        "test_correct": sum(scores),
        "agent_name": agent_name,
        "agent_dir": str(agent_dir),
        "task": task.name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
