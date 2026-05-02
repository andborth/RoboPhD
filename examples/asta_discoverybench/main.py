#!/usr/bin/env python3
"""
Evolve DiscoveryBench agents (AstaBench, Standard tools tier) using
RoboPhD's optimize_anything() API.

Targets the AstaBench Data-Analysis category leaderboard. Real validation
= 25 samples, real test = 239. Synth (public) provides an additional 903
samples (550 train / 153 dev / 200 test) for distribution-padding the
small real/train.

Three training regimes (--regime):
  1: Synth-only   — train on synth/train (550); test on synth/dev (153) +
                    real/train (25) + real/test (239). 1500 evals,
                    20 examples/iter. synth/test is upstream's held-out
                    competition set with no gold, so we use synth/dev as
                    the local synth-side test signal instead.
  2: Mixed        — 85% synth + 15% real per iteration. Two phases.
                    --phase experiment: 85 synth + 15 real, 750 evals.
                    --phase final:      85 synth + 25 real, 750 evals.
                    Test on real/test for final.
  3: Real-only    — train on real/train. Two phases.
                    --phase experiment: 15 real, 15 iters @ 3 examples/iter.
                    --phase final:      25 real, 15 iters @ 3 examples/iter.
                    Test on real/test for final.

Credentials required:
    HF_ACCESS_TOKEN  — gated allenai/asta-bench dataset (real/ split metadata)
    OPENAI_API_KEY   — solver model (gpt-5-mini default) + judge (gpt-4o-2024-08-06)

Setup:
    Docker daemon must be running. See README.md.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from RoboPhD import (
    optimize_anything,
    eval_candidate,
    eval_run,
    RoboPhDConfig,
    GEPAConfig,
    AutoresearchConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
for noisy in ("LiteLLM", "litellm", "httpx", "openai._base_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime configuration
# ---------------------------------------------------------------------------

def _regime_dataset(regime: int, phase: str | None, seed: int):
    """Build (train_pool, test_pool, examples_per_iter, evaluation_budget,
    num_iterations) for the named regime.

    train_pool is what RoboPhD samples from each iteration. test_pool is
    the held-out evaluation set surfaced via --eval-test-set.
    """
    from evaluator import load_real, load_synth

    rng = random.Random(seed)

    if regime == 1:
        # Synth-only. Train on synth/train (550), held-out test on synth/dev
        # (153) + real/validation (25) + real/test (239).
        # synth/test is upstream's held-out competition set with no gold,
        # so it can't be scored locally and we don't include it.
        train = load_synth("train")
        test = load_synth("dev") + load_real("validation") + load_real("test")
        return train, test, 20, 1500, 999  # iters bounded by budget

    if regime == 2:
        synth_train = load_synth("train")
        real_train = load_real("validation")  # AstaBench's "validation" = paper's "train"

        # 85 fixed random synth (seeded) — same selection across both phases of
        # regime 2, so the synth subset is stable when comparing phase A→B.
        synth_subset = rng.sample(synth_train, 85)

        if phase == "experiment":
            # 85 synth + 15 real (subset); test on the other 10 real.
            real_subset_idx = rng.sample(range(len(real_train)), 15)
            real_subset = [real_train[i] for i in real_subset_idx]
            real_held_out = [real_train[i] for i in range(len(real_train)) if i not in real_subset_idx]
            train = synth_subset + real_subset
            test = real_held_out
        else:  # final
            train = synth_subset + real_train
            test = load_real("test")
        return train, test, 10, 750, 999

    if regime == 3:
        real_train = load_real("validation")
        if phase == "experiment":
            real_subset_idx = rng.sample(range(len(real_train)), 15)
            train = [real_train[i] for i in real_subset_idx]
            test = [real_train[i] for i in range(len(real_train)) if i not in real_subset_idx]
        else:  # final
            train = real_train
            test = load_real("test")
        # Iteration-bounded: examples/iter=3, num_iterations=15 → 180 evals.
        # Set evaluation_budget high enough not to bind.
        return train, test, 3, 999_999, 15

    raise ValueError(f"unknown regime: {regime}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evolve DiscoveryBench agents on AstaBench (Standard tools, Docker sandbox)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--regime", type=int, choices=[1, 2, 3], default=2,
                   help="Training regime (see module docstring)")
    p.add_argument("--phase", choices=["experiment", "final"], default="experiment",
                   help="Phase within regimes 2 and 3 (ignored for regime 1)")

    p.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd")
    p.add_argument("--num-iterations", type=int, default=None,
                   help="Override regime's iteration cap")
    p.add_argument("--evaluation-budget", type=int, default=None,
                   help="Override regime's evaluation budget")

    p.add_argument("--model", default="openai/gpt-5-mini",
                   help="Inspect model string for the candidate solver's LLM calls")
    p.add_argument("--cost-budget", type=float, default=0.10,
                   help="Per-example agent cost cap; score *= 0.9 if breached")

    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel eval workers")
    p.add_argument("--runs-dir", default="../robophd_runs")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--engine-config", type=str, default=None)
    p.add_argument("--meta-evolution-strategy", default=None)

    p.add_argument("--eval-test-set", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--extend", type=int, default=None)
    p.add_argument("--from-iteration", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    from evaluator import DiscoveryBenchEvaluator

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    evaluator = DiscoveryBenchEvaluator(
        model=args.model,
        cost_budget=args.cost_budget,
    )

    train, test, examples_per_iter, regime_budget, regime_iterations = (
        _regime_dataset(args.regime, args.phase, args.random_seed)
    )
    logger.info(
        f"Regime {args.regime}/{args.phase}: train={len(train)} test={len(test)} "
        f"examples/iter={examples_per_iter} budget={regime_budget} iters={regime_iterations}"
    )

    # RoboPhD's ExternalEvaluatorDomain JSON-serializes each example to
    # compute a stable id (SHA256 of the dict). Inspect's Sample is a
    # pydantic model and isn't directly JSON-serializable, so flatten to
    # plain dicts at the boundary; the evaluator reconstructs Sample.
    train = [s.model_dump() for s in train]
    test = [s.model_dump() for s in test]

    # --eval-only: skip optimization, evaluate the best agent from --resume on test.
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        eval_result = eval_run(evaluator=evaluator, dataset=test, experiment_dir=args.resume)
        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} samples)")
        test_path = Path(args.resume) / "test_results.json"
        with open(test_path, "w") as f:
            json.dump({
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": evaluator.total_eval_cost,
            }, f, indent=2)
        logger.info(f"Test results saved to {test_path}")
        return

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    num_iterations = args.num_iterations if args.num_iterations is not None else regime_iterations
    evaluation_budget = args.evaluation_budget if args.evaluation_budget is not None else regime_budget

    engine_overrides: dict = {"examples_per_iteration": examples_per_iter}
    if args.engine_config:
        engine_overrides.update(json.loads(args.engine_config))

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=evaluation_budget,
            val_dataset=test,
            max_workers=args.max_workers,
            seed=args.random_seed,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=evaluation_budget,
            val_dataset=test,
            max_workers=args.max_workers,
            seed=args.random_seed,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    else:
        dataset = train
        cfg = RoboPhDConfig(
            num_iterations=num_iterations,
            evaluation_budget=evaluation_budget,
            max_workers=args.max_workers,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
            meta_evolution_strategy=args.meta_evolution_strategy,
            engine_overrides=engine_overrides,
        )
        if args.resume:
            cfg.experiment_dir = args.resume
        if args.extend:
            cfg.extend_iterations = args.extend
        if args.from_iteration:
            cfg.from_iteration = args.from_iteration

    result = optimize_anything(
        evaluator=evaluator,
        dataset=dataset,
        seed_candidate=seed,
        objective=objective,
        background=background,
        config=cfg,
        task_name=f"asta_discoverybench_r{args.regime}_{args.phase or 'all'}",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: ELO {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            logger.info(f"Test evaluation: {len(test)} samples")
            eval_result = eval_candidate(
                evaluator=evaluator,
                dataset=test,
                candidate=result.best_candidate,
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} samples)")
            test_results = {
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": evaluator.total_eval_cost,
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
