#!/usr/bin/env python3
"""
Evolve DS-1000 agents (AstaBench, Standard tools tier) using
RoboPhD's optimize_anything() API.

Targets the AstaBench Code & Execution category leaderboard. DS-1000
ships a deterministic 100 / 900 validation/test split (cached in
astabench/evals/inspect_eval_wrappers/ds1000_splits.json). We use the
100-sample validation pool for training and the 900-sample test pool
for the leaderboard metric.

Test set depends on --phase: experiment → 90 fixed samples (~10%) of
ds1000_test, final → all 900. Both subsets are drawn with a fixed
SPLIT_SEED (42), independent of --random-seed.

Credentials required:
    OPENAI_API_KEY  — solver model (gpt-5.4-mini default). DS-1000 has
                      no judge LLM, so cost is agent-only.

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
# Dataset configuration
# ---------------------------------------------------------------------------

# Fixed seed for test-pool sub-sampling. Independent of --random-seed;
# --random-seed only affects RoboPhD's per-iteration draws.
SPLIT_SEED = 42


def _build_dataset(phase: str):
    """Build (train_pool, test_pool, examples_per_iter, evaluation_budget,
    num_iterations).

    Train pool = the full 100-sample ds1000_validation split.

    Test pool depends on phase:
      - experiment: 90 samples (~10%) drawn with SPLIT_SEED from the
        900-sample ds1000_test. Cheap enough to run on every
        --eval-test-set check while remaining a meaningful held-out set.
      - final: all 900 ds1000_test samples (the leaderboard metric).
    """
    from evaluator import load_ds1000

    test_rng = random.Random(SPLIT_SEED)

    train = load_ds1000("validation")  # 100
    test_full = load_ds1000("test")    # 900

    if phase == "experiment":
        test = test_rng.sample(test_full, 90)
    else:  # final
        test = test_full

    # Iteration-bounded: examples/iter=20, num_iterations=15 → 300 evals.
    # Set evaluation_budget high enough not to bind.
    return train, test, 20, 999_999, 15


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_test_results(
    eval_result,
    evaluator,
    output_dir: Path,
    agent_name: str,
    phase: str | None,
    summary_filename: str,
):
    """Write a summary JSON plus a sibling .per_problem.json for a test eval.

    DS-1000 has no judge LLM, so cost is agent-only — simpler than
    DiscoveryBench's agent/judge split.
    """
    per_problem = []
    total_agent_cost = 0.0
    diagnostics_list = eval_result.per_example_diagnostics or []
    scores_list = eval_result.per_example_scores or []
    for i, diag in enumerate(diagnostics_list):
        diag = diag or {}
        score = scores_list[i] if i < len(scores_list) else None
        agent_c = diag.get("agent_cost_usd") or 0.0
        total_agent_cost += agent_c
        err = diag.get("error")
        per_problem.append({
            "sample_id": diag.get("sample_id"),
            "score": score,
            "raw_score": diag.get("raw_score"),
            "cost_penalty": diag.get("cost_penalty"),
            "agent_cost_usd": agent_c,
            "library": diag.get("library"),
            "error": (err[:500] if err else None),
        })

    summary_path = output_dir / summary_filename
    with open(summary_path, "w") as f:
        json.dump({
            "agent": agent_name,
            "phase": phase,
            "mean_test_score": eval_result.mean_score,
            "total_test_score": eval_result.total_score,
            "total_test_problems": eval_result.num_examples,
            "test_eval_cost_usd": evaluator.total_eval_cost,
            "test_eval_agent_cost_usd": total_agent_cost,
        }, f, indent=2)

    per_problem_path = summary_path.with_suffix(".per_problem.json")
    with open(per_problem_path, "w") as f:
        json.dump(per_problem, f, indent=2)

    return summary_path, per_problem_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Evolve DS-1000 agents on AstaBench (Standard tools, Docker sandbox)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=["experiment", "final"], default="experiment",
                   help="experiment: 90-sample held-out test (~10%% of ds1000_test). "
                        "final: all 900 ds1000_test samples (leaderboard metric).")

    p.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd")
    p.add_argument("--num-iterations", type=int, default=None,
                   help="Override the default iteration cap (15)")
    p.add_argument("--examples-per-iteration", type=int, default=None,
                   help="Override the default per-iteration sample size (20)")
    p.add_argument("--evaluation-budget", type=int, default=None,
                   help="Override the default evaluation budget (iter-bounded)")

    p.add_argument("--cost-threshold", type=float, default=None,
                   help="Per-example agent spend below this is in the free zone "
                        "(no penalty). Default $0.01.")
    p.add_argument("--cost-saturation", type=float, default=None,
                   help="Per-example agent spend at this level (or above) "
                        "incurs the maximum cost penalty of 1.0. Default $1.00. "
                        "The penalty ramps linearly between threshold and "
                        "saturation. Test-path scores are raw 0/1 regardless.")

    p.add_argument("--max-workers", type=int, default=12,
                   help="Parallel eval workers. Each evaluation runs in its "
                        "own subprocess to bypass inspect.eval's process-global "
                        "singleton lock, so this is real parallelism. 12 fits "
                        "20 examples/iteration in ~2 waves on M-series Macs.")
    p.add_argument("--runs-dir", default="../robophd_runs")
    p.add_argument("--random-seed", type=int, default=None,
                   help="Seed for RoboPhD's per-iteration draws and other "
                        "internal RNG. Default None resolves to a fresh seed "
                        "each run. The train/test pool composition is "
                        "independent of this flag (driven by SPLIT_SEED).")
    p.add_argument("--engine-config", type=str, default=None)
    p.add_argument("--meta-evolution-strategy", default=None)

    p.add_argument("--eval-test-set", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-agent", type=str, default=None,
                   help="Name of a specific agent from the --resume run's agent_pool to "
                        "evaluate. Requires --eval-only. Defaults to the best-ELO agent. "
                        "Output file is suffixed with the agent name.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--extend", type=int, default=None)
    p.add_argument("--from-iteration", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    from evaluator import Ds1000Evaluator

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    # Per-example timeout: must match the value passed to RoboPhDConfig
    # below. The evaluator derives a slightly-shorter subprocess_timeout
    # internally so subprocesses get killed BEFORE RoboPhD's reaper
    # would leak the thread.
    EVAL_TIMEOUT = 600

    # Two evaluator instances. Training applies the bounded cost penalty
    # (a tiebreaker between correctness-tied agents); test paths report
    # raw 0/1 so evolved agents land at their true point on the Pareto
    # cost-vs-score curve.
    from evaluator import MIN_COST_THRESHOLD, COST_PENALTY_SATURATION
    evaluator = Ds1000Evaluator(
        eval_timeout=EVAL_TIMEOUT,
        apply_cost_penalty=True,  # training: penalty fires
        min_cost_threshold=(
            args.cost_threshold if args.cost_threshold is not None
            else MIN_COST_THRESHOLD
        ),
        cost_penalty_saturation=(
            args.cost_saturation if args.cost_saturation is not None
            else COST_PENALTY_SATURATION
        ),
    )
    test_evaluator = evaluator.with_overrides(apply_cost_penalty=False)

    train, test, examples_per_iter, default_budget, default_iterations = (
        _build_dataset(args.phase)
    )
    if args.examples_per_iteration is not None:
        examples_per_iter = args.examples_per_iteration
    logger.info(
        f"Phase {args.phase}: train={len(train)} test={len(test)} "
        f"examples/iter={examples_per_iter} budget={default_budget} iters={default_iterations}"
    )

    # RoboPhD's ExternalEvaluatorDomain JSON-serializes each example to
    # compute a stable id (SHA256 of the dict). Inspect's Sample is a
    # pydantic model; flatten to plain dicts at the boundary; the
    # evaluator reconstructs Sample.
    train = [s.model_dump() for s in train]
    test = [s.model_dump() for s in test]

    if args.eval_agent and not args.eval_only:
        raise SystemExit("--eval-agent requires --eval-only")

    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")

        if args.eval_agent:
            from RoboPhD.runner_utils import find_named_agent
            try:
                _, agent_dir = find_named_agent(Path(args.resume), args.eval_agent)
            except FileNotFoundError as e:
                raise SystemExit(str(e))
            candidate = {"agent.py": (agent_dir / "agent.py").read_text()}
            logger.info(f"Evaluating named agent: {args.eval_agent} from {agent_dir}")
            eval_result = eval_candidate(evaluator=test_evaluator, dataset=test, candidate=candidate)
        else:
            eval_result = eval_run(evaluator=test_evaluator, dataset=test, experiment_dir=args.resume)

        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} samples)")
        results_filename = (
            f"test_results_{args.phase}_{args.eval_agent}.json"
            if args.eval_agent
            else f"test_results_{args.phase}.json"
        )
        summary_path, per_problem_path = _write_test_results(
            eval_result=eval_result,
            evaluator=test_evaluator,
            output_dir=Path(args.resume),
            agent_name=args.eval_agent or "best",
            phase=args.phase,
            summary_filename=results_filename,
        )
        logger.info(f"Test summary:    {summary_path}")
        logger.info(f"Test per-problem: {per_problem_path}")
        return

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    num_iterations = args.num_iterations if args.num_iterations is not None else default_iterations
    evaluation_budget = args.evaluation_budget if args.evaluation_budget is not None else default_budget

    engine_overrides: dict = {"examples_per_iteration": examples_per_iter}
    if args.engine_config:
        engine_overrides.update(json.loads(args.engine_config))

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=evaluation_budget,
            val_dataset=test,
            max_workers=args.max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
            eval_timeout=EVAL_TIMEOUT,
        )
        dataset = train
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=evaluation_budget,
            val_dataset=test,
            max_workers=args.max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
            eval_timeout=EVAL_TIMEOUT,
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
            eval_timeout=EVAL_TIMEOUT,
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
        task_name="asta_ds1000",
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
                evaluator=test_evaluator,
                dataset=test,
                candidate=result.best_candidate,
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} samples)")
            summary_path, per_problem_path = _write_test_results(
                eval_result=eval_result,
                evaluator=test_evaluator,
                output_dir=result.experiment_dir,
                agent_name="best",
                phase=args.phase,
                summary_filename=f"test_results_{args.phase}.json",
            )
            logger.info(f"Test summary:    {summary_path}")
            logger.info(f"Test per-problem: {per_problem_path}")


if __name__ == "__main__":
    from RoboPhD.eval_utils import force_exit_if_threads_leaked
    try:
        main()
    finally:
        force_exit_if_threads_leaked()
