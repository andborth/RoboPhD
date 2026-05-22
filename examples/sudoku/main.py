#!/usr/bin/env python3
"""
Evolve Sudoku solvers using RoboPhD's optimize_anything() API.

Pure algorithmic optimization — no LLM calls needed for the solver.
Only Claude Code is needed for evolution.

Usage:
    # Quick smoke test
    python examples/sudoku/main.py --evaluation-budget 60 --num-iterations 2

    # Full run
    python examples/sudoku/main.py

    # With test-set evaluation after optimization
    python examples/sudoku/main.py --eval-test-set
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Add project root and example dir to path
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from RoboPhD import optimize_anything, RoboPhDConfig, GEPAConfig, AutoresearchConfig


class WarmupMedianEvaluator:
    """Evolution-time wrapper: W warmup solves (discarded) + N timed solves → median.

    Each wrapper call on (candidate, example):
      1. Runs `warmup` untimed solves. Primes CPU state for this specific
         puzzle-solver combination — branch predictor learns the puzzle's
         constraint tree, shared lookup tables hit L1/L2 cache, CPU clock
         boosts, object pools allocate.
      2. Runs `timed_repeats` (odd, default 9) timed solves, collecting each
         solve's (score, diagnostics).
      3. Returns the median score and the diagnostics of the solve whose
         score equals the median. An odd repeat count guarantees the median
         is one of the measured scores, so the representative solve's
         diagnostics are internally consistent with the returned score.

    Rationale: sudoku's score is `max(0, 1 - elapsed * 100)` with a hard 10ms
    cutoff. A cold-state solve can easily push past 10ms and score 0 even for
    a healthy agent, because CPU branch predictor, caches, and frequency
    scaling all need a few iterations to adapt. Without priming, whichever
    agent runs second in an iteration inherits a warm CPU state and gets a
    systematic score advantage over the agent that runs first.

    This wrapper eliminates that by priming each (agent, puzzle) pair with a
    few untimed warmup solves before the timed measurements. The median
    across timed repeats further absorbs any residual per-repeat variance.

    Only used during evolution — test evaluation uses a separate aggregation
    path (``test_eval_candidate.test_eval``) that does its own repeats-and-
    median logic via an interleaved dispatch over the full test set.
    """

    def __init__(self, base, warmup: int = 2, timed_repeats: int = 9):
        if timed_repeats % 2 == 0:
            raise ValueError(
                "timed_repeats must be odd so the median is an exact element of the sample"
            )
        self._base = base
        self._warmup = warmup
        self._timed = timed_repeats

    def __call__(self, candidate, example, *, problem_dir=None):
        # Warmup — discard scores.
        for _ in range(self._warmup):
            self._base(candidate, example, problem_dir=problem_dir)

        # Timed solves — keep all (score, diag) pairs.
        results = []
        for _ in range(self._timed):
            score, diag = self._base(candidate, example, problem_dir=problem_dir)
            results.append((score, diag if isinstance(diag, dict) else {}))

        scores = [s for s, _ in results]
        # statistics.median on an odd-length list returns sorted(scores)[N//2]
        # by definition — no arithmetic — so the returned value is bit-identical
        # to one of the input scores and == comparison is safe.
        median_score = statistics.median(scores)
        representative_diag = next(d for s, d in results if s == median_score)

        # Sum costs across all 9 timed calls. Correct for sudoku because every
        # solve has cost_usd=0 (no LLM). If this wrapper is ever reused on a
        # task with cost-bearing solves, summing here 9× the real per-call cost
        # — switch to taking the representative's cost only, or divide by N.
        total_cost = 0.0
        for _, d in results:
            try:
                total_cost += float(d.get("cost_usd", 0.0))
            except (TypeError, ValueError):
                pass

        return median_score, {**representative_diag, "cost_usd": total_cost}

    def __getattr__(self, name):
        # Delegate stateful attrs (total_eval_cost, total_evaluations, _lock) to base
        return getattr(self._base, name)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve pure-Python Sudoku solvers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999, help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500, help="Max evaluator calls across all iterations")

    # Engine
    parser.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd", help="Optimization engine")

    # Infrastructure
    parser.add_argument("--runs-dir", default="../robophd_runs", help="Root directory for experiment output (default: %(default)s)")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--engine-config", type=str, default=None, help="JSON overrides (e.g. evolution_strategy, evolution_model, examples_per_iteration)")
    parser.add_argument("--meta-evolution-strategy", default=None,
                        help="Meta-evolution strategy (e.g. train_a_winner); default off. "
                             "Cadence and first iteration default to (3, 4); override via --engine-config.")

    # Test evaluation
    parser.add_argument("--eval-test-set", action="store_true", help="Run test-set evaluation after optimization")

    # Resume / extend
    parser.add_argument("--eval-only", action="store_true", help="Skip optimization; evaluate best agent from --resume dir on test set")
    parser.add_argument("--eval-agent", type=str, default=None,
                        help="Name of agent in the run's agent_pool to evaluate (requires --eval-only). "
                             "Defaults to the best-Elo agent.")
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None, help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None, help="Restart from a specific iteration")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval_agent and not args.eval_only:
        raise SystemExit("--eval-agent requires --eval-only")

    from evaluator import SudokuEvaluator, load_dataset
    from test_eval_candidate import test_eval, _load_candidate_from_run_dir

    base_evaluator = SudokuEvaluator()
    # Evolution wraps the base evaluator for warmup + per-puzzle median aggregation.
    # Test-eval paths below pass the unwrapped base_evaluator — test_eval already
    # handles its own repeats and median; wrapping would double-nest.
    evolution_evaluator = WarmupMedianEvaluator(base_evaluator)

    # --eval-only: skip optimization, evaluate an agent from a prior run on the test set.
    # Delegates to test_eval_candidate.test_eval() for canonical sudoku test-eval behavior:
    # 9 repeats per puzzle, per-puzzle median aggregation (mean of medians),
    # max_workers=1, and a 1s per-solve timeout (see _PER_SOLVE_TIMEOUT_SECONDS).
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        run_dir = Path(args.resume)
        candidate, agent_name = _load_candidate_from_run_dir(run_dir, agent_name=args.eval_agent)
        logger.info(f"Evaluating agent: {agent_name}")

        _, results = test_eval(base_evaluator, candidate, agent_name=agent_name)

        if args.eval_agent:
            test_path = run_dir / f"test_results_{args.eval_agent}.json"
        else:
            test_path = run_dir / "test_results.json"
        with open(test_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to {test_path}")
        return

    evaluator = evolution_evaluator  # evolution path uses the wrapper

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    ds = load_dataset()
    dataset = ds["train"]
    logger.info(f"Dataset: {len(dataset)} training problems")

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Build config based on engine choice
    # Sudoku has no separate val split — GEPA/Autoresearch auto-split from train.
    # max_workers=1 for all engines: solvers are scored on CPU time via
    # time.process_time(), which measures the entire process.
    if args.engine in ("gepa", "autoresearch"):
        _robophd_flags = {"--num-iterations", "--engine-config", "--resume", "--extend", "--from-iteration", "--meta-evolution-strategy"}
        passed = {f for f in _robophd_flags if any(a == f or a.startswith(f + "=") for a in sys.argv)}
        if passed:
            logger.warning(f"Flags ignored by {args.engine} engine: {', '.join(sorted(passed))}")

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=args.evaluation_budget,
            max_workers=1,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            max_workers=1,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
    else:
        engine_overrides = {}
        if args.engine_config:
            engine_overrides = json.loads(args.engine_config)
        cfg = RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            max_workers=1,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
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
        task_name="sudoku",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            # Delegate to canonical sudoku test-eval: 9 repeats, per-puzzle median,
            # max_workers=1, 1s per-solve timeout. Use the unwrapped base evaluator —
            # test_eval does its own repeats and median; wrapping would double-nest.
            _, test_results = test_eval(
                base_evaluator, result.best_candidate, agent_name="best_candidate"
            )

            test_path = result.experiment_dir / "test_results.json"
            with open(test_path, "w") as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Test results saved to {test_path}")


if __name__ == "__main__":
    main()
