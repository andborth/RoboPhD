#!/usr/bin/env python3
"""
Cross-benchmark comparison: GEPA vs RoboPhD on matched evaluation budgets.

Runs both optimizers on the same problem set with the same number of fresh
evaluations, then evaluates both winners on a held-out test set.

Usage:
    # Compare with 200 evaluations each
    python scripts/cross_benchmark.py \
        --seed-agent RoboPhD/codegen_agents/naive_critic \
        --evaluation-budget 200 \
        --coder-model haiku-4.5 \
        --critic-model haiku-4.5

    # Run only GEPA (skip RoboPhD)
    python scripts/cross_benchmark.py \
        --seed-agent RoboPhD/codegen_agents/naive_critic \
        --evaluation-budget 200 \
        --skip-robophd

    # Run only RoboPhD (skip GEPA)
    python scripts/cross_benchmark.py \
        --seed-agent RoboPhD/codegen_agents/naive_critic \
        --evaluation-budget 200 \
        --skip-gepa

Dependencies:
    pip install gepa  (for GEPA runs)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RoboPhD.adapters.gepa_codegen import (
    CODEGEN_FILE_MAPPING,
    RoboPhDCodeGenEvaluator,
    build_codegen_dataset,
    extract_candidate,
    materialize_candidate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-benchmark comparison: GEPA vs RoboPhD"
    )

    parser.add_argument(
        "--seed-agent",
        type=Path,
        required=True,
        help="Path to seed agent directory (used by both GEPA and RoboPhD)",
    )
    parser.add_argument(
        "--evaluation-budget",
        type=int,
        required=True,
        help="Number of fresh evaluations for each optimizer",
    )

    # Models
    parser.add_argument("--coder-model", default="haiku-4.5")
    parser.add_argument("--critic-model", default="haiku-4.5")

    # Data
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("../robophd_runs")
    )
    parser.add_argument("--seed", type=int, default=42)

    # Flags
    parser.add_argument(
        "--skip-gepa", action="store_true", help="Skip GEPA run"
    )
    parser.add_argument(
        "--skip-robophd", action="store_true", help="Skip RoboPhD run"
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip test-set evaluation of winners",
    )

    # RoboPhD config
    parser.add_argument(
        "--robophd-config",
        type=str,
        default=None,
        help="JSON config string or path for RoboPhD (merged with defaults)",
    )

    # Output
    parser.add_argument("--output-dir", type=Path, default=None)

    return parser.parse_args()


def resolve_cache_dir(args) -> Path:
    if args.cache_dir:
        return args.cache_dir
    cache_model_name = args.coder_model.replace("/", "--")
    return args.runs_dir / "codegen_cache" / f"{cache_model_name}_v6"


def run_gepa(args, cache_dir: Path, output_dir: Path) -> dict:
    """Run GEPA optimization and return summary."""
    logger.info("=" * 60)
    logger.info("Running GEPA optimization")
    logger.info("=" * 60)

    start_time = time.time()

    # Build command
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_gepa_codegen.py"),
        "--seed-agent", str(args.seed_agent),
        "--coder-model", args.coder_model,
        "--critic-model", args.critic_model,
        "--cache-dir", str(cache_dir),
        "--max-metric-calls", str(args.evaluation_budget),
        "--seed", str(args.seed),
        "--output-dir", str(output_dir),
    ]

    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    summary = {"elapsed_seconds": elapsed, "returncode": result.returncode}

    # Load optimization summary if available
    summary_file = output_dir / "optimization_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary.update(json.load(f))

    return summary


def run_robophd(args, cache_dir: Path, output_dir: Path) -> dict:
    """Run RoboPhD evolution and return summary."""
    logger.info("=" * 60)
    logger.info("Running RoboPhD evolution")
    logger.info("=" * 60)

    start_time = time.time()

    # Calculate iterations from budget
    # Default: 50 problems per iteration, 3 agents per iteration
    # Fresh evals per iteration ≈ contexts_per_iteration * agents_per_iteration
    contexts_per_iter = 50
    agents_per_iter = 3
    evals_per_iter = contexts_per_iter * agents_per_iter
    num_iterations = max(1, args.evaluation_budget // evals_per_iter)

    # Build RoboPhD config
    # Point agents_directory at the seed agent's parent so RoboPhD starts
    # from the same seed as GEPA (matched comparison).
    seed_agent_dir = args.seed_agent.resolve()
    robophd_config = {
        "domain": "codegen",
        "coder_model": args.coder_model,
        "critic_model": args.critic_model,
        "codegen_split": "evolution",
        "contexts_per_iteration": contexts_per_iter,
        "agents_per_iteration": agents_per_iter,
        "evaluation_budget": args.evaluation_budget,
        "random_seed": args.seed,
        "runs_directory": str(args.runs_dir),
        "agents_directory": str(seed_agent_dir.parent),
        "initial_agents": [seed_agent_dir.name],
    }

    # Merge user config
    if args.robophd_config:
        if Path(args.robophd_config).exists():
            with open(args.robophd_config) as f:
                user_config = json.load(f)
        else:
            user_config = json.loads(args.robophd_config)
        robophd_config.update(user_config)

    config_str = json.dumps(robophd_config)

    cmd = [
        sys.executable,
        str(project_root / "RoboPhD" / "researcher.py"),
        "--num-iterations", str(num_iterations),
        "--config", config_str,
    ]

    logger.info(f"Running RoboPhD with {num_iterations} iterations")
    logger.info(f"Config: {config_str}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    summary = {
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
        "num_iterations": num_iterations,
        "evaluation_budget": args.evaluation_budget,
    }

    return summary


def evaluate_on_test_set(
    agent_dir: Path,
    label: str,
    cache_dir: Path,
    coder_model: str,
    critic_model: str,
    work_dir: Path,
) -> dict:
    """Evaluate an agent on the held-out test set."""
    logger.info(f"Evaluating {label} on test set...")

    test_examples = build_codegen_dataset(cache_dir, split="test")
    logger.info(f"Test set: {len(test_examples)} problems")

    candidate = extract_candidate(agent_dir, CODEGEN_FILE_MAPPING)

    evaluator = RoboPhDCodeGenEvaluator(
        coder_model=coder_model,
        critic_model=critic_model,
        cache_dir=cache_dir,
        work_dir=work_dir,
    )

    scores = []
    v1_scores = []
    improvements = 0
    regressions = 0

    for i, example in enumerate(test_examples):
        score, diag = evaluator(candidate, example)
        scores.append(score)
        v1_passed = diag.get("v1_passed", False)
        v1_scores.append(1.0 if v1_passed else 0.0)
        if diag.get("improved"):
            improvements += 1
        if diag.get("regressed"):
            regressions += 1

        if (i + 1) % 20 == 0:
            logger.info(
                f"  {label} test progress: {i+1}/{len(test_examples)}, "
                f"v2 accuracy: {sum(scores)/len(scores)*100:.1f}%"
            )

    test_results = {
        "label": label,
        "v2_accuracy": sum(scores) / len(scores) * 100 if scores else 0.0,
        "v1_accuracy": sum(v1_scores) / len(v1_scores) * 100 if v1_scores else 0.0,
        "total": len(scores),
        "v2_correct": int(sum(scores)),
        "v1_correct": int(sum(v1_scores)),
        "improvements": improvements,
        "regressions": regressions,
        "improve_rate": improvements / len(scores) * 100 if scores else 0.0,
        "regress_rate": regressions / len(scores) * 100 if scores else 0.0,
    }

    logger.info(
        f"  {label} test: v2={test_results['v2_accuracy']:.1f}%, "
        f"v1={test_results['v1_accuracy']:.1f}%, "
        f"improve={test_results['improve_rate']:.1f}%, "
        f"regress={test_results['regress_rate']:.1f}%"
    )

    return test_results


def print_comparison(gepa_results: dict, robophd_results: dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("CROSS-BENCHMARK COMPARISON: GEPA vs RoboPhD")
    print("=" * 70)

    metrics = [
        ("V2 Accuracy (%)", "v2_accuracy"),
        ("V1 Accuracy (%)", "v1_accuracy"),
        ("Improvements", "improvements"),
        ("Regressions", "regressions"),
        ("Improve Rate (%)", "improve_rate"),
        ("Regress Rate (%)", "regress_rate"),
        ("Total Problems", "total"),
    ]

    print(f"\n{'Metric':<25} {'GEPA':>12} {'RoboPhD':>12} {'Δ':>10}")
    print("-" * 60)

    for label, key in metrics:
        g = gepa_results.get(key, 0)
        r = robophd_results.get(key, 0)
        delta = g - r if isinstance(g, (int, float)) else ""
        if isinstance(delta, float):
            delta_str = f"{delta:+.1f}"
        elif isinstance(delta, int):
            delta_str = f"{delta:+d}"
        else:
            delta_str = ""

        if isinstance(g, float):
            print(f"{label:<25} {g:>12.1f} {r:>12.1f} {delta_str:>10}")
        else:
            print(f"{label:<25} {g:>12} {r:>12} {delta_str:>10}")

    print("=" * 70)


def main():
    args = parse_args()

    if args.skip_gepa and args.skip_robophd:
        logger.error("Cannot skip both GEPA and RoboPhD")
        sys.exit(1)

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = Path("cross_benchmark_runs") / f"comparison_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = resolve_cache_dir(args)
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        sys.exit(1)

    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Cache: {cache_dir}")
    logger.info(f"Evaluation budget: {args.evaluation_budget}")

    results = {"timestamp": timestamp, "evaluation_budget": args.evaluation_budget}

    # --- Run optimizers ---
    gepa_summary = None
    robophd_summary = None

    if not args.skip_gepa:
        gepa_dir = args.output_dir / "gepa"
        gepa_summary = run_gepa(args, cache_dir, gepa_dir)
        results["gepa_optimization"] = gepa_summary

    if not args.skip_robophd:
        robophd_dir = args.output_dir / "robophd"
        robophd_summary = run_robophd(args, cache_dir, robophd_dir)
        results["robophd_optimization"] = robophd_summary

    # --- Test-set evaluation ---
    if not args.skip_test_eval:
        # GEPA best agent
        if not args.skip_gepa:
            gepa_best_agent = args.output_dir / "gepa" / "best_agent"
            if gepa_best_agent.exists():
                results["gepa_test"] = evaluate_on_test_set(
                    gepa_best_agent,
                    "GEPA",
                    cache_dir,
                    args.coder_model,
                    args.critic_model,
                    args.output_dir / "gepa_test_work",
                )
            else:
                logger.warning("GEPA best agent not found, skipping test eval")

        # RoboPhD best agent — find from the experiment directory
        if not args.skip_robophd:
            # Look for the most recent evolution run
            robophd_best = _find_robophd_best_agent(args.output_dir / "robophd")
            if robophd_best:
                results["robophd_test"] = evaluate_on_test_set(
                    robophd_best,
                    "RoboPhD",
                    cache_dir,
                    args.coder_model,
                    args.critic_model,
                    args.output_dir / "robophd_test_work",
                )
            else:
                logger.warning("RoboPhD best agent not found, skipping test eval")

        # Print comparison
        if "gepa_test" in results and "robophd_test" in results:
            print_comparison(results["gepa_test"], results["robophd_test"])

    # Save combined results
    with open(args.output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {args.output_dir / 'comparison_results.json'}")


def _find_robophd_best_agent(robophd_dir: Path) -> Path | None:
    """
    Find the best agent from a RoboPhD run.

    Looks for the agent with highest ELO in the checkpoint.
    """
    # Find evolution directory
    evolution_dirs = list(robophd_dir.glob("evolution/robophd_*"))
    if not evolution_dirs:
        # Try direct experiment dir structure
        evolution_dirs = list(robophd_dir.glob("robophd_*"))
    if not evolution_dirs:
        return None

    exp_dir = sorted(evolution_dirs)[-1]  # Most recent

    # Load checkpoint to find best agent
    checkpoint_path = exp_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    # Find agent with highest ELO from performance_records
    performance_records = checkpoint.get("performance_records", {})
    if not performance_records:
        return None

    best_agent_id = max(
        performance_records,
        key=lambda a: performance_records[a].get("elo", 1500),
    )

    # Find agent package directory
    agents_dir = exp_dir / "agents"
    agent_dir = agents_dir / best_agent_id
    if agent_dir.exists():
        return agent_dir

    # Try alternate naming
    for d in agents_dir.iterdir() if agents_dir.exists() else []:
        if best_agent_id in d.name:
            return d

    return None


if __name__ == "__main__":
    main()
