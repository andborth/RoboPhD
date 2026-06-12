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
from RoboPhD.runner_utils import read_task_config_extras, resolve_run_immutable

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
# Persisted in checkpoint.json's task_config (via RoboPhDConfig's
# task_config_extras, rewritten by the framework every iteration), so the
# value survives any interruption that leaves a resumable checkpoint.
# Mirrors the pattern in examples/asta_ds1000/main.py. Runs from before
# this migration persisted the value in a sidecar at the experiment-dir
# root — kept as a read-only fallback. We never silently fall back to the
# default, which would mix pool sizes within a single run.
# ---------------------------------------------------------------------------
ARC_TASK_CONFIG_KEY = "arc_agi_1_runtime"
LEGACY_SIDECAR_FILENAME = "arc_agi_1_runtime_config.json"

# Default parallel eval workers when --max-workers is omitted. Applied as a
# task default (not argparse's default, which stays None so resume can tell
# "user passed it" from "use the default"). read_checkpoint_max_workers lives
# in runner_utils so every example shares one implementation.
DEFAULT_MAX_WORKERS = 10


def _read_arc_runtime_config(resume_dir) -> dict:
    """ARC binding of runner_utils.read_task_config_extras."""
    return read_task_config_extras(
        resume_dir, ARC_TASK_CONFIG_KEY, LEGACY_SIDECAR_FILENAME,
    )


class _SkipNoneDefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Like ArgumentDefaultsHelpFormatter but don't append "(default: None)" for
    None-sentinel flags (where None means 'resolve the real default later'), which
    would contradict the default stated inline in the help text. Keep in sync with
    examples/asta_ds1000/main.py."""

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve ARC-AGI-1 agents",
        formatter_class=_SkipNoneDefaultsFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999, help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500, help="Max evaluator calls across all iterations")
    parser.add_argument("--num-train", type=int, default=None,
                        help="Training-pool size: first N of the seeded shuffle (default 400 = all). "
                             "ELO trains on all N; GEPA/Autoresearch split it N/2 train + N/2 val "
                             "(so train+val == ELO's train). Even, in [2,400].")

    # Engine
    parser.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd", help="Optimization engine")

    # Task-specific
    parser.add_argument("--paper-config", action="store_true",
                        help="Paper settings: high reasoning + $0.25 budget + 10 LLM calls "
                             "(default: medium reasoning + $0.10 + 20 calls)")

    # Infrastructure
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel eval workers (default: 10)")
    # BooleanOptionalAction exposes both --agent-isolation and
    # --no-agent-isolation; with default=True the help's "(default: True)" reads
    # against the positive name, where True == on (a store_false
    # --no-agent-isolation would render "(default: True)" on the *disable* flag,
    # which reads backwards).
    parser.add_argument("--agent-isolation", action=argparse.BooleanOptionalAction, default=True,
                        help="Run each agent eval in a memory-capped subprocess "
                             "(contains memory-bomb / signal-misuse agents)")
    parser.add_argument("--agent-memory-gb", type=float, default=4.0,
                        help="Per-agent memory ceiling in GB when isolation is on (default: %(default)s); "
                             "note max_workers x ceiling must fit in RAM")
    parser.add_argument("--runs-dir", default="../robophd_runs", help="Root directory for experiment output (default: %(default)s)")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--engine-config", type=str, default=None, help="JSON overrides (e.g. evolution_strategy, evolution_model, examples_per_iteration, new_agent_test_rounds)")
    parser.add_argument("--meta-evolution-strategy", default=None,
                        help="Meta-evolution strategy (e.g. train_a_winner); default off. "
                             "Cadence and first iteration default to (3, 4); override via --engine-config.")

    # Test evaluation
    parser.add_argument("--eval-test-set", action="store_true", help="Run test-set evaluation after optimization")
    parser.add_argument("--eval-only", action="store_true", help="Skip optimization; evaluate best agent from --resume dir on test set")
    parser.add_argument("--eval-agent", type=str, default=None,
                        help="With --eval-only: evaluate this named agent from the --resume dir's "
                             "agent_pool instead of the best-Elo agent. Results are written to "
                             "test_results_<agent>.json so they don't clobber the default.")
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

    if args.eval_agent and not args.eval_only:
        raise SystemExit("--eval-agent requires --eval-only")

    # Resolve --max-workers for the EVAL paths (--eval-only / --eval-test-set),
    # which take a RoboPhDEvalConfig directly and don't go through ConfigManager.
    # Order: explicit CLI flag wins; else on --resume recover the value the
    # original run used from its checkpoint; else the task default.
    #
    # The TRAINING engine is handled differently below: max_workers is packed
    # into engine_overrides only when the user passed the flag, so it applies as
    # a config delta that genuinely takes effect on a training --resume. When
    # the flag is omitted on resume, nothing is packed and ConfigManager's
    # delta-inheritance carries the original run's value forward (honored when
    # passed, persisted when not). See the DANGER caller-invariant in api.py.
    from RoboPhD.runner_utils import read_checkpoint_max_workers
    if args.max_workers is not None:
        effective_max_workers = args.max_workers
    elif args.resume:
        cp_mw = read_checkpoint_max_workers(args.resume)
        effective_max_workers = cp_mw if cp_mw is not None else DEFAULT_MAX_WORKERS
        if cp_mw is not None:
            logger.info(
                f"Resume: using max_workers={cp_mw} from checkpoint.json "
                f"(pass --max-workers N to override)"
            )
    else:
        effective_max_workers = DEFAULT_MAX_WORKERS

    # --eval-only: skip optimization, evaluate an agent from a prior run on the
    # test set (always the full 400; --num-train does not apply here). Defaults
    # to the best-Elo agent; --eval-agent targets a specific agent_pool entry.
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        test_data = load_arc_test()
        eval_cfg = RoboPhDEvalConfig(eval_timeout=EVAL_TIMEOUT, max_workers=effective_max_workers)
        if args.eval_agent:
            from RoboPhD.runner_utils import find_named_agent
            try:
                _, agent_dir = find_named_agent(Path(args.resume), args.eval_agent)
            except FileNotFoundError as e:
                raise SystemExit(str(e))
            logger.info(f"Evaluating named agent: {args.eval_agent} from {agent_dir}")
            candidate = {"agent.py": (agent_dir / "agent.py").read_text()}
            eval_result = eval_candidate(
                evaluator=evaluator, dataset=test_data, candidate=candidate, config=eval_cfg,
            )
            results_filename = f"test_results_{args.eval_agent}.json"
        else:
            eval_result = eval_run(
                evaluator=evaluator, dataset=test_data, experiment_dir=args.resume,
                config=eval_cfg,
            )
            results_filename = "test_results.json"
        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} problems)")
        test_path = Path(args.resume) / results_filename
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
    # persisted in the checkpoint's task_config and re-applied on --resume.
    # Resume is RoboPhD-only; gepa/autoresearch warn-and-ignore --resume, so
    # the lock only engages there.
    on_resume = bool(args.resume) and args.engine == "robophd"
    stored_num_train = _read_arc_runtime_config(args.resume).get("num_train") if on_resume else None
    num_train = resolve_run_immutable(
        args.num_train, stored_num_train, 400, "num-train", on_resume=on_resume,
    )
    if num_train % 2 != 0 or not (2 <= num_train <= 400):
        raise SystemExit("--num-train must be an even integer in [2, 400] "
                         "(the pool is 400 records, split half train / half val).")
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
            max_workers=effective_max_workers,
            eval_timeout=EVAL_TIMEOUT,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
        dataset = train
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=effective_max_workers,
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
        # Route max_workers through engine_overrides (not the dedicated
        # RoboPhDConfig.max_workers field, which only applies on a fresh run).
        # Explicit flag always packs (so it applies as a config delta that takes
        # effect even on a training --resume). With no flag, pack the task
        # default ONLY on a fresh run — never on resume, where packing would
        # clobber the original run's value; ConfigManager's delta-inheritance
        # carries it forward instead. (DANGER caller-invariant, api.py.)
        if args.max_workers is not None:
            engine_overrides["max_workers"] = args.max_workers
        elif not args.resume and DEFAULT_MAX_WORKERS is not None and "max_workers" not in engine_overrides:
            engine_overrides["max_workers"] = DEFAULT_MAX_WORKERS
        cfg = RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
            eval_timeout=EVAL_TIMEOUT,
            meta_evolution_strategy=args.meta_evolution_strategy,
            engine_overrides=engine_overrides or None,
            # Persisted into checkpoint.json's task_config every iteration
            # so the pool size survives mid-run interruption; on resume
            # this is the stored value re-resolved (idempotent) or the
            # one-time bootstrap value for pre-migration runs.
            task_config_extras={ARC_TASK_CONFIG_KEY: {"num_train": num_train}},
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
                config=RoboPhDEvalConfig(eval_timeout=EVAL_TIMEOUT, max_workers=effective_max_workers),
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
