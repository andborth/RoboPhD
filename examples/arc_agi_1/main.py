#!/usr/bin/env python3
"""
Evolve ARC-AGI-1 solving agents using RoboPhD's optimize_anything() API.

Usage:
    # Quick smoke test
    python main.py --evaluation-budget 60 --num-iterations 2

    # Full run
    python main.py

    # With test evaluation
    python main.py --eval-test-set

    # Resume a prior run
    python main.py --resume ../../robophd_runs/robophd/optimize_anything_20260401_120000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Add project root and example dir to path
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from RoboPhD import optimize_anything, eval_candidate, eval_run, RoboPhDConfig, GEPAConfig, AutoresearchConfig, RoboPhDEvalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Diagnostic: if this process is ever terminated by SIGALRM (seen as
# "zsh: alarm" when two eval jobs thrash swap and a native socket/DNS
# deadline blows), dump a full traceback first, then let the default
# action proceed (chain=True). This names the exact stalled frame.
import faulthandler
import signal
faulthandler.enable()
faulthandler.register(signal.SIGALRM, chain=True)

# Per-evaluation timeout (seconds), single-sourced: the domain enforces this
# as its cooperative future-wait, and the evaluator derives its per-child
# subprocess timeout from the same value so the two stay coupled.
EVAL_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Per-run training-pool sidecar (--num-train), locked across --resume.
#
# The framework's checkpoint persists framework-level knobs but not this
# task-specific one, so a sidecar alongside checkpoint.json carries it across
# --resume. Mirrors the pattern in examples/asta_ds1000/main.py. The sidecar is
# written once the experiment dir is known: immediately on --resume, and after
# optimize_anything() returns for a fresh run (whose dir is auto-created with a
# timestamp inside that call). A fresh run interrupted before it returns has no
# sidecar; resuming it requires re-passing --num-train. We never silently fall
# back to the default, which would mix pool sizes within a single run.
# ---------------------------------------------------------------------------
ARC_AGI_1_RUNTIME_CONFIG_FILENAME = "arc_agi_1_runtime_config.json"


def _arc_runtime_path(experiment_dir) -> Path:
    return Path(experiment_dir) / ARC_AGI_1_RUNTIME_CONFIG_FILENAME


def _read_arc_runtime_config(resume_dir) -> dict:
    """Read the per-run sidecar; {} if absent or unreadable."""
    p = _arc_runtime_path(resume_dir)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_arc_runtime_config(experiment_dir, values: dict) -> None:
    """Persist the per-run sidecar; best-effort (warn, don't abort, on failure)."""
    p = _arc_runtime_path(experiment_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(values, indent=2, sort_keys=True))
    except OSError as e:
        logger.warning(f"Could not write {p.name}: {e}")


def _resolve_num_train(cli_value, stored_value, on_resume: bool, default: int = 400) -> int:
    """Resolve --num-train; immutable for the lifetime of a run.

    Fresh run: CLI value wins, else the default (400 = the full pool).

    On --resume the value persists from the per-run sidecar; re-passing the same
    value is allowed, a different value is rejected. If the sidecar is absent
    (the run predates this flag, or was interrupted before the sidecar was
    written), --num-train must be re-supplied to re-establish it — we never
    silently fall back to the default, which would mix pool sizes within a run.
    """
    if not on_resume:
        return cli_value if cli_value is not None else default
    if stored_value is not None:
        if cli_value is not None and cli_value != stored_value:
            raise SystemExit(
                f"--num-train cannot be changed on --resume; it is locked for the "
                f"run (stored value: {stored_value}). Start a new run to use a "
                f"different value."
            )
        return stored_value
    if cli_value is not None:
        logger.info(
            f"Resume: bootstrapping num_train={cli_value} into "
            f"{ARC_AGI_1_RUNTIME_CONFIG_FILENAME} (no prior stored value)."
        )
        return cli_value
    raise SystemExit(
        f"--resume failed: no stored num_train in {ARC_AGI_1_RUNTIME_CONFIG_FILENAME} "
        f"(this run predates --num-train, or was interrupted before the sidecar was "
        f"written). Re-pass --num-train <N> to set it, or start a new run."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve ARC-AGI-1 agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999, help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500, help="Max evaluator calls across all iterations")
    parser.add_argument("--num-train", type=int, default=None,
                        help="Training-pool size: first N of the seeded shuffle (default 400 = all). "
                             "ELO trains on all N; GEPA/Autoresearch split it N/2 train + N/2 val "
                             "(so train+val == ELO's train). Even, in [2,400]. LOCKED for the run: "
                             "it persists across --resume (re-pass the same value or omit it; "
                             "a different value is rejected).")

    # Engine
    parser.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd", help="Optimization engine")

    # Task-specific
    parser.add_argument("--paper-config", action="store_true",
                        help="Paper settings: high reasoning + $0.25 budget + 10 LLM calls "
                             "(default: medium reasoning + $0.10 + 20 calls)")

    # Infrastructure
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel eval workers (None = Python default)")
    parser.add_argument("--no-agent-isolation", dest="agent_isolation", action="store_false",
                        help="Disable running each agent eval in a memory-capped subprocess "
                             "(isolation is on by default; it contains memory-bomb / signal-misuse agents)")
    parser.add_argument("--agent-memory-gb", type=float, default=4.0,
                        help="Per-agent memory ceiling in GB when isolation is on (default: %(default)s); "
                             "note max_workers x ceiling must fit in RAM")
    parser.add_argument("--runs-dir", default="../robophd_runs", help="Root directory for experiment output (default: %(default)s)")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--engine-config", type=str, default=None, help="JSON overrides (e.g. evolution_strategy, evolution_model, examples_per_iteration)")
    parser.add_argument("--meta-evolution-strategy", default=None,
                        help="Meta-evolution strategy (e.g. train_a_winner); default off. "
                             "Cadence and first iteration default to (3, 4); override via --engine-config.")

    # Test evaluation
    parser.add_argument("--eval-test-set", action="store_true", help="Run test-set evaluation after optimization")
    parser.add_argument("--eval-only", action="store_true", help="Skip optimization; evaluate best agent from --resume dir on test set")
    # Resume / extend
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None, help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None, help="Restart from a specific iteration")

    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy import evaluator (requires dspy/datasets)
    from evaluator import (
        ArcAGI1Evaluator, DEFAULT_SOLVER_MODEL,
        load_arc_train_val, load_arc_test,
    )

    # Select config tier.
    # High reasoning scores 3.3x better on the seed (29% vs 8.7%) but costs
    # 6.8x more ($0.027 vs $0.004/problem) and runs 7.5x slower. The default
    # tier trades seed accuracy for cheaper evals and gives evolution 20 LLM
    # calls to build multi-step strategies.
    if args.paper_config:
        solver_model = DEFAULT_SOLVER_MODEL
        cost_budget = 0.25
        reasoning_effort = "high"
        max_llm_calls = 10
    else:
        solver_model = DEFAULT_SOLVER_MODEL
        cost_budget = 0.10
        reasoning_effort = "medium"
        max_llm_calls = 20

    logger.info(f"Solver config: model={solver_model}, budget=${cost_budget:.2f}, "
                f"reasoning={reasoning_effort}, max_calls={max_llm_calls}")

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()
    background = background.replace("{cost_budget}", f"{cost_budget:.2f}")
    background = background.replace("{max_llm_calls}", str(max_llm_calls))

    evaluator = ArcAGI1Evaluator(
        solver_model=solver_model,
        max_llm_calls=max_llm_calls,
        cost_budget=cost_budget,
        reasoning_effort=reasoning_effort,
        eval_timeout=EVAL_TIMEOUT,
        agent_subprocess_isolation=args.agent_isolation,
        agent_memory_limit_gb=args.agent_memory_gb,
    )

    # --eval-only: skip optimization, evaluate best agent from a prior run
    # (test set is always the full 400; --num-train does not apply here).
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        test_data = load_arc_test()
        eval_result = eval_run(
            evaluator=evaluator, dataset=test_data, experiment_dir=args.resume,
            config=RoboPhDEvalConfig(eval_timeout=EVAL_TIMEOUT),
        )
        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")
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

    # Resolve the training-pool size. --num-train is LOCKED for a run: it is
    # persisted to a per-run sidecar and re-applied on --resume. Resume is
    # RoboPhD-only; gepa/autoresearch warn-and-ignore --resume, so the lock only
    # engages there.
    on_resume = bool(args.resume) and args.engine == "robophd"
    stored_num_train = _read_arc_runtime_config(args.resume).get("num_train") if on_resume else None
    num_train = _resolve_num_train(args.num_train, stored_num_train, on_resume)
    if num_train % 2 != 0 or not (2 <= num_train <= 400):
        raise SystemExit("--num-train must be an even integer in [2, 400] "
                         "(the pool is 400 records, split half train / half val).")
    # On resume the experiment dir is known now, so persist immediately — this
    # makes a bootstrapped or re-interrupted run keep its pool size without
    # re-passing the flag. (Fresh runs are written after optimize_anything,
    # once their auto-created dir exists.)
    if on_resume:
        _write_arc_runtime_config(args.resume, {"num_train": num_train})
    train, val = load_arc_train_val(num_train=num_train)
    logger.info(f"Dataset: {len(train)} train + {len(val)} val  (num_train={num_train})")

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Build config based on engine choice
    if args.engine in ("gepa", "autoresearch"):
        _robophd_flags = {"--num-iterations", "--engine-config", "--resume", "--extend", "--from-iteration", "--meta-evolution-strategy"}
        passed = {f for f in _robophd_flags if any(a == f or a.startswith(f + "=") for a in sys.argv)}
        if passed:
            logger.warning(f"Flags ignored by {args.engine} engine: {', '.join(sorted(passed))}")

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=args.max_workers,
            eval_timeout=EVAL_TIMEOUT,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=args.max_workers,
            eval_timeout=EVAL_TIMEOUT,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    else:
        dataset = train + val
        engine_overrides = {}
        if args.engine_config:
            engine_overrides = json.loads(args.engine_config)
        cfg = RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            max_workers=args.max_workers,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
            eval_timeout=EVAL_TIMEOUT,
            meta_evolution_strategy=args.meta_evolution_strategy,
            engine_overrides=engine_overrides or None,
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
        task_name="arc_agi_1",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    # Persist the training-pool size so a future --resume recovers and locks it.
    _write_arc_runtime_config(result.experiment_dir, {"num_train": num_train})

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = load_arc_test()
            logger.info(f"Test evaluation: {len(test_data)} problems")
            # Evaluator is shared with optimize_anything() above, so snapshot its
            # running total to record test-only cost (not evolution + test).
            cost_before = evaluator.total_eval_cost
            eval_result = eval_candidate(
                evaluator=evaluator,
                dataset=test_data,
                candidate=result.best_candidate,
                config=RoboPhDEvalConfig(eval_timeout=EVAL_TIMEOUT),
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")

            # Save test results
            test_results = {
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": evaluator.total_eval_cost - cost_before,
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
