#!/usr/bin/env python3
"""
Run GEPA's optimize_anything() on RoboPhD's codegen-critic domain.

Extracts a seed candidate from a RoboPhD agent directory, builds train/val
datasets from the codegen cache, and runs GEPA optimization.

Usage:
    # Smoke test (10 evaluations)
    python scripts/run_gepa_codegen.py \
        --seed-agent RoboPhD/codegen_agents/naive_critic \
        --max-metric-calls 10

    # Full run
    python scripts/run_gepa_codegen.py \
        --seed-agent RoboPhD/codegen_agents/naive_critic \
        --max-metric-calls 600 \
        --coder-model haiku-4.5 \
        --critic-model haiku-4.5

    # With test-set evaluation of best candidate
    python scripts/run_gepa_codegen.py \
        --seed-agent RoboPhD/codegen_agents/naive_critic \
        --max-metric-calls 600 \
        --eval-test-set

Dependencies:
    pip install gepa
"""

import argparse
import json
import logging
import sys
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
        description="Run GEPA optimize_anything() on RoboPhD codegen-critic"
    )

    # Seed agent
    parser.add_argument(
        "--seed-agent",
        type=Path,
        required=True,
        help="Path to RoboPhD agent directory to use as seed candidate",
    )

    # Models
    parser.add_argument("--coder-model", default="haiku-4.5", help="Coder model (default: haiku-4.5)")
    parser.add_argument("--critic-model", default="haiku-4.5", help="Critic model (default: haiku-4.5)")

    # Cache / data
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Codegen cache directory (auto-derived from coder-model if not set)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("../robophd_runs"),
        help="Runs directory (default: ../robophd_runs)",
    )

    # GEPA parameters
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
        help="GEPA optimization budget (checked between rounds, actual evaluator calls will be higher)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of evolution set to hold out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--reflection-model",
        default=None,
        help="LLM for GEPA reflection/mutation (default: GEPA's built-in default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Evaluation
    parser.add_argument(
        "--eval-test-set",
        action="store_true",
        help="Evaluate best candidate on the held-out test set after optimization",
    )

    # Parallelism
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max parallel evaluator threads for GEPA val sweeps (default: 8)",
    )

    # Timeouts
    parser.add_argument("--codegen-timeout", type=int, default=1200)
    parser.add_argument("--critic-timeout", type=int, default=600)

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: gepa_runs/<timestamp>)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("gepa_runs") / f"codegen_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {args.output_dir}")

    # --- 1. Extract seed candidate ---
    logger.info(f"Extracting seed candidate from {args.seed_agent}")
    seed_candidate = extract_candidate(args.seed_agent, CODEGEN_FILE_MAPPING)
    logger.info(
        f"Seed candidate keys: {list(seed_candidate.keys())}, "
        f"sizes: { {k: len(v) for k, v in seed_candidate.items()} }"
    )

    # Save seed for reference
    with open(args.output_dir / "seed_candidate.json", "w") as f:
        json.dump(seed_candidate, f, indent=2)

    # --- 2. Resolve cache directory ---
    if args.cache_dir is not None:
        cache_dir = args.cache_dir
    else:
        cache_model_name = args.coder_model.replace("/", "--")
        cache_dir = args.runs_dir / "codegen_cache" / f"{cache_model_name}_v6"

    if not cache_dir.exists():
        logger.error(f"Cache directory does not exist: {cache_dir}")
        logger.error("Run codegen preprocessing first to populate the cache.")
        sys.exit(1)

    logger.info(f"Using cache directory: {cache_dir}")

    # --- 3. Build datasets ---
    import random
    rng = random.Random(args.seed)

    all_evolution = build_codegen_dataset(cache_dir, split="evolution")
    logger.info(f"Evolution set: {len(all_evolution)} problems")

    # Train/val split (GEPA optimizes on train, validates on held-out val)
    shuffled = list(all_evolution)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - args.val_ratio)))
    trainset = shuffled[:split_idx]
    valset = shuffled[split_idx:]
    logger.info(f"Training set: {len(trainset)}, Validation set: {len(valset)}")

    # --- 4. Create evaluator ---
    evaluator = RoboPhDCodeGenEvaluator(
        coder_model=args.coder_model,
        critic_model=args.critic_model,
        cache_dir=cache_dir,
        work_dir=args.output_dir / "work",
        codegen_timeout=args.codegen_timeout,
        critic_timeout=args.critic_timeout,
    )

    # --- 5. Run GEPA optimization ---
    try:
        from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig
    except ImportError:
        logger.error(
            "GEPA not installed. Install with: pip install gepa\n"
            "See: https://github.com/gepa-ai/gepa"
        )
        sys.exit(1)

    config_kwargs = {
        "max_metric_calls": args.max_metric_calls,
        "seed": args.seed,
        "run_dir": str(args.output_dir / "gepa_run"),
        "parallel": args.max_workers > 1,
        "max_workers": args.max_workers,
    }

    engine_config = EngineConfig(**config_kwargs)
    gepa_config = GEPAConfig(engine=engine_config)

    if args.reflection_model:
        from gepa.optimize_anything import ReflectionConfig
        gepa_config.reflection = ReflectionConfig(reflection_lm=args.reflection_model)

    logger.info(f"Starting GEPA optimization with max_metric_calls={args.max_metric_calls}")

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=(
            "Optimize the critic agent to accurately identify incorrect code solutions "
            "and provide actionable feedback that helps the coder fix bugs. "
            "The eval_instructions guide the critic's verdict (CORRECT/INCORRECT) and feedback. "
            "The tool_code performs static analysis before the critic reviews the code."
        ),
        config=gepa_config,
    )

    # --- 6. Save results ---
    best_candidate = result.best_candidate
    val_scores = getattr(result, "val_aggregate_scores", None)
    if val_scores is None:
        logger.warning("GEPA result missing val_aggregate_scores")
        val_scores = []
    candidates = getattr(result, "candidates", None)
    if candidates is None:
        logger.warning("GEPA result missing candidates")
        candidates = []

    best_val = max(val_scores) if val_scores else 0.0
    logger.info(f"Optimization complete. Best validation score: {best_val:.3f}")

    # Save best candidate
    with open(args.output_dir / "best_candidate.json", "w") as f:
        json.dump(best_candidate, f, indent=2)

    # Materialize best candidate as agent directory
    best_agent_dir = args.output_dir / "best_agent"
    materialize_candidate(best_candidate, best_agent_dir, CODEGEN_FILE_MAPPING, name="gepa_best")
    logger.info(f"Best agent materialized at: {best_agent_dir}")

    # Save optimization summary
    summary = {
        "seed_agent": str(args.seed_agent),
        "coder_model": args.coder_model,
        "critic_model": args.critic_model,
        "max_metric_calls": args.max_metric_calls,
        "total_evaluations": evaluator.total_evaluations,
        "train_size": len(trainset),
        "val_size": len(valset) if valset else 0,
        "num_candidates_explored": len(candidates),
        "best_val_score": best_val,
        "val_scores": val_scores,
    }
    with open(args.output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- 7. Optional test-set evaluation ---
    if args.eval_test_set:
        logger.info("Evaluating best candidate on test set...")
        test_examples = build_codegen_dataset(cache_dir, split="test")
        logger.info(f"Test set: {len(test_examples)} problems")

        test_evaluator = RoboPhDCodeGenEvaluator(
            coder_model=args.coder_model,
            critic_model=args.critic_model,
            cache_dir=cache_dir,
            work_dir=args.output_dir / "test_work",
            codegen_timeout=args.codegen_timeout,
            critic_timeout=args.critic_timeout,
        )

        scores = []
        for i, example in enumerate(test_examples):
            score, diag = test_evaluator(best_candidate, example)
            scores.append(score)
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Test progress: {i+1}/{len(test_examples)}, "
                    f"running accuracy: {sum(scores)/len(scores)*100:.1f}%"
                )

        test_accuracy = sum(scores) / len(scores) * 100 if scores else 0.0
        logger.info(f"Test set accuracy: {test_accuracy:.1f}% ({sum(scores):.0f}/{len(scores)})")

        test_results = {
            "test_accuracy": test_accuracy,
            "test_total": len(scores),
            "test_correct": sum(scores),
        }
        with open(args.output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    logger.info(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
