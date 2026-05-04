#!/usr/bin/env python3
"""
Evolve DiscoveryBench agents (AstaBench, Standard tools tier) using
RoboPhD's optimize_anything() API.

Targets the AstaBench Data-Analysis category leaderboard. Real validation
= 25 samples, real test = 239. Synth (public) provides an additional 703
scoreable samples (550 train / 153 dev) for distribution-padding the
small real pool.

Train pool: --num-synth-train (default 175) synth samples + all 25
real/validation. Test set depends on --phase: experiment → 24 fixed
samples (~10%) of real/test, final → all 239 of real/test. Both pool
and the experiment test draws use a fixed split seed (42), independent
of --random-seed; --random-seed (default None → fresh per run) only
affects RoboPhD's per-iteration draws.

Credentials required:
    HF_ACCESS_TOKEN  — gated allenai/asta-bench dataset (real/ split metadata)
    OPENAI_API_KEY   — solver model (gpt-5.4-mini default) + judge (gpt-4o-2024-08-06)

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

# Fixed seed for train/test pool construction. Independent of
# --random-seed; --random-seed only affects RoboPhD per-iteration draws.
SPLIT_SEED = 42


def _build_dataset(phase: str, num_synth_train: int):
    """Build (train_pool, test_pool, examples_per_iter, evaluation_budget,
    num_iterations).

    Train pool = `num_synth_train` synth samples (drawn with SPLIT_SEED
    from synth/train's 550 scoreable) + all 25 real/validation.

    Test pool depends on phase:
      - experiment: 24 samples (~10%) drawn with SPLIT_SEED from
        real/test's 239. Cheap enough to run as part of every
        --eval-test-set check while remaining a meaningful held-out set.
      - final: all 239 real/test samples (the leaderboard metric).

    Two independent random.Random(SPLIT_SEED) instances (synth_rng,
    test_rng) keep the synth draw and the test draw isolated, so
    changing --num-synth-train doesn't perturb the test composition.
    """
    from evaluator import load_real, load_synth

    synth_rng = random.Random(SPLIT_SEED)
    test_rng = random.Random(SPLIT_SEED)

    synth_train = load_synth("train")            # 550 scoreable
    real_train = load_real("validation")         # 25
    real_test = load_real("test")                # 239

    if not (0 <= num_synth_train <= len(synth_train)):
        raise SystemExit(
            f"--num-synth-train={num_synth_train} out of range; "
            f"synth/train has {len(synth_train)} scoreable samples"
        )

    synth_subset = synth_rng.sample(synth_train, num_synth_train)
    train = synth_subset + real_train

    if phase == "experiment":
        test = test_rng.sample(real_test, 24)
    else:  # final
        test = real_test

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

    Walks `eval_result.per_example_diagnostics` to extract per-problem
    fields (sample_id, score, agent_cost_usd, judge_cost_usd, ...) so
    the agent-vs-judge cost split isn't lost when we summarize. Without
    this, the inspect.eval logs in the temp dir are the only source of
    truth for the cost breakdown — and they get reaped on process exit.
    """
    per_problem = []
    total_agent_cost = 0.0
    total_judge_cost = 0.0
    diagnostics_list = eval_result.per_example_diagnostics or []
    scores_list = eval_result.per_example_scores or []
    for i, diag in enumerate(diagnostics_list):
        diag = diag or {}
        score = scores_list[i] if i < len(scores_list) else None
        agent_c = diag.get("agent_cost_usd") or 0.0
        judge_c = diag.get("judge_cost_usd") or 0.0
        total_agent_cost += agent_c
        total_judge_cost += judge_c
        err = diag.get("error")
        per_problem.append({
            "sample_id": diag.get("sample_id"),
            "score": score,
            "agent_cost_usd": agent_c,
            "judge_cost_usd": judge_c,
            "total_cost_usd": agent_c + judge_c,
            "cost_breached": diag.get("cost_breached"),
            "cost_penalty_applied": diag.get("cost_penalty_applied"),
            "split": diag.get("split"),
            "context_score": diag.get("context_score"),
            "var_f1": diag.get("var_f1"),
            "rel_score": diag.get("rel_score"),
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
            "test_eval_judge_cost_usd": total_judge_cost,
        }, f, indent=2)

    per_problem_path = summary_path.with_suffix(".per_problem.json")
    with open(per_problem_path, "w") as f:
        json.dump(per_problem, f, indent=2)

    return summary_path, per_problem_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Evolve DiscoveryBench agents on AstaBench (Standard tools, Docker sandbox)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=["experiment", "final"], default="experiment",
                   help="experiment: 24-sample held-out test (~10%% of real/test). "
                        "final: all 239 real/test samples (leaderboard metric).")
    p.add_argument("--num-synth-train", type=int, default=175,
                   help="Number of synth/train samples to mix into the train pool. "
                        "real/validation's 25 samples are always included.")

    p.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd")
    p.add_argument("--num-iterations", type=int, default=None,
                   help="Override the default iteration cap (15)")
    p.add_argument("--evaluation-budget", type=int, default=None,
                   help="Override the default evaluation budget (iter-bounded)")

    p.add_argument("--model", default="openai/gpt-5.4-mini",
                   help="Inspect model string for the candidate solver's LLM calls")
    p.add_argument("--cost-budget", type=float, default=0.10,
                   help="Per-example AGENT cost cap. During training (RoboPhD "
                        "ELO), score *= 0.9 if agent spend exceeds the cap; at "
                        "test time the score is raw HMS regardless of breach. "
                        "Judge-LLM cost (~$0.015-0.020/sample, 5 fixed gpt-4o "
                        "calls) is tracked separately as `other_cost` — it's "
                        "kept out of the optimization signal and excluded from "
                        "the cap. `eval_cost` in reports is agent-only and "
                        "matches the cap-relevant number.")

    p.add_argument("--max-workers", type=int, default=12,
                   help="Parallel eval workers. Each evaluation runs in its "
                        "own subprocess to bypass inspect.eval's process-global "
                        "singleton lock, so this is real parallelism. 12 fits "
                        "20 examples/iteration in ~2 waves on M-series Macs; "
                        "tune up to ~20 if your OpenAI tier supports the "
                        "resulting RPS.")
    p.add_argument("--runs-dir", default="../robophd_runs")
    p.add_argument("--random-seed", type=int, default=None,
                   help="Seed for RoboPhD's per-iteration draws and other "
                        "internal RNG. Default None resolves to a fresh seed "
                        "each run (logged and persisted in checkpoint). The "
                        "train/test pool composition is independent of this "
                        "flag (driven by a fixed SPLIT_SEED).")
    p.add_argument("--engine-config", type=str, default=None)
    p.add_argument("--meta-evolution-strategy", default=None)

    p.add_argument("--eval-test-set", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-agent", type=str, default=None,
                   help="Name of a specific agent from the --resume run's agent_pool to "
                        "evaluate (e.g. 'seed_rzvfojwy' to baseline the seed, or "
                        "'iter7_robust_pronounced_v1'). Requires --eval-only. Defaults to "
                        "the best-ELO agent. Output file is suffixed with the agent name "
                        "so results don't overwrite the default best-ELO results.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--extend", type=int, default=None)
    p.add_argument("--from-iteration", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    from evaluator import DiscoveryBenchEvaluator

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    # Per-example timeout: must match the value passed to RoboPhDConfig
    # below. The evaluator derives a slightly-shorter subprocess_timeout
    # internally so subprocesses get killed BEFORE RoboPhD's reaper would
    # leak the thread (see evaluator.py for the reasoning).
    EVAL_TIMEOUT = 600

    # Two evaluator instances. Training uses the cost penalty (a soft
    # signal nudging evolution toward cheaper agents); test paths report
    # raw HMS so evolved agents land at their true point on the Pareto
    # cost-vs-score curve. The test instance is derived via with_overrides
    # so any future constructor field added to DiscoveryBenchEvaluator
    # automatically propagates from training to test config.
    evaluator = DiscoveryBenchEvaluator(
        model=args.model,
        cost_budget=args.cost_budget,
        eval_timeout=EVAL_TIMEOUT,
        apply_cost_penalty=True,  # training: penalty fires
    )
    test_evaluator = evaluator.with_overrides(apply_cost_penalty=False)

    train, test, examples_per_iter, default_budget, default_iterations = (
        _build_dataset(args.phase, args.num_synth_train)
    )
    logger.info(
        f"Phase {args.phase}: train={len(train)} test={len(test)} "
        f"examples/iter={examples_per_iter} budget={default_budget} iters={default_iterations}"
    )

    # RoboPhD's ExternalEvaluatorDomain JSON-serializes each example to
    # compute a stable id (SHA256 of the dict). Inspect's Sample is a
    # pydantic model and isn't directly JSON-serializable, so flatten to
    # plain dicts at the boundary; the evaluator reconstructs Sample.
    train = [s.model_dump() for s in train]
    test = [s.model_dump() for s in test]

    if args.eval_agent and not args.eval_only:
        raise SystemExit("--eval-agent requires --eval-only")

    # --eval-only: skip optimization, evaluate an agent from --resume on test.
    # By default uses the best-ELO agent (via eval_run); --eval-agent overrides
    # to a specific named agent (via find_named_agent + eval_candidate). The
    # test set composition is fixed (driven by SPLIT_SEED, not --random-seed),
    # so --eval-only against the same --phase always sees the same samples.
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
        # Phase-distinct filenames so back-to-back experiment/final eval-only
        # runs against the same --resume dir don't clobber each other; agent
        # name is also suffixed when --eval-agent is set.
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
        task_name="asta_discoverybench",
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
