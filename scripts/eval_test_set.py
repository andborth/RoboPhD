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
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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
        f"score: {best_perf['mean_score']:.3f}, "
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
    args = parser.parse_args()

    if args.test_repeats < 1:
        parser.error("--test-repeats must be at least 1")

    # Resolve agent
    if args.run_dir:
        agent_name, agent_dir = find_best_agent(args.run_dir)
        default_output = args.run_dir / "test_results.json"
    else:
        agent_dir = args.agent_dir
        agent_name = agent_dir.name
        results_base = Path(__file__).resolve().parent.parent / "robophd_runs" / "results" / "agent_tests" / args.task / agent_name
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

    # Build test dataset (matching run_gepa.py protocol)
    test_config = {**config, **task.test_overrides}  # test_overrides last: must override training defaults
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

    # Parallel eval loop
    test_workers = args.max_workers or test_config.get("max_test_workers") or max(1, min(32, (os.cpu_count() or 4) + 4) // 2)
    logger.info(f"Test evaluation: {len(test_examples)} problems, {test_workers} workers")

    eval_timeout = test_config.get("eval_timeout", 300)
    timed_out = False
    timed_out_idxs: set[int] = set()

    score_map: dict[int, float] = {}
    idx_to_example = {i: ex for i, ex in enumerate(test_examples)}
    executor = ThreadPoolExecutor(max_workers=test_workers)
    try:
        future_to_idx = {
            executor.submit(test_evaluator, candidate, ex): i
            for i, ex in enumerate(test_examples)
        }
        remaining = set(future_to_idx.keys())

        while remaining:
            done, not_done = wait(
                remaining, timeout=eval_timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                still_remaining = set()
                for future in not_done:
                    idx = future_to_idx[future]
                    if future.cancel():
                        if idx in timed_out_idxs:
                            score_map[idx] = 0.0
                        else:
                            new_future = executor.submit(
                                test_evaluator, candidate, idx_to_example[idx]
                            )
                            future_to_idx[new_future] = idx
                            still_remaining.add(new_future)
                    else:
                        logger.warning(
                            f"EVAL TIMEOUT: example {idx} exceeded {eval_timeout}s — "
                            f"scored 0, thread leaked (will burn CPU until process exit)"
                        )
                        score_map[idx] = 0.0
                        timed_out_idxs.add(idx)
                        timed_out = True
                remaining = still_remaining
                continue

            for future in done:
                idx = future_to_idx[future]
                score, diag = future.result()
                score_map[idx] = score
            remaining = not_done

            if len(score_map) % 10 == 0:
                logger.info(
                    f"Test progress: {len(score_map)}/{len(test_examples)}, "
                    f"running accuracy: {sum(score_map.values())/len(score_map)*100:.1f}%"
                )
    finally:
        if timed_out:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)
    scores = [score_map[i] for i in range(len(test_examples))]

    test_accuracy = sum(scores) / len(scores) * 100 if scores else 0.0
    logger.info(f"Test set accuracy: {test_accuracy:.1f}% ({sum(scores):.0f}/{len(scores)})")

    # Write results
    test_eval_cost = getattr(test_evaluator, 'total_eval_cost', 0.0)
    test_results = {
        "test_accuracy": test_accuracy,
        "test_total": len(scores),
        "test_correct": sum(scores),
        "test_eval_cost_usd": test_eval_cost,
        "agent_name": agent_name,
        "agent_dir": str(agent_dir),
        "task": task.name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)

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
