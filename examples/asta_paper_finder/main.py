#!/usr/bin/env python3
"""
Evolve PaperFindingBench agents (AstaBench, Standard tools tier) using
RoboPhD's optimize_anything() API.

Targets the AstaBench Literature Understanding leaderboard's PaperFindingBench
subtask. Validation = 66 samples, test = 267 samples (held out).

Credentials required:
    HF_ACCESS_TOKEN           — gated allenai/asta-bench dataset
    ASTA_TOOL_KEY             — Asta MCP corpus tools (the leaderboard's
                                Standard kit). If unset, the evaluator falls
                                back to public-Semantic-Scholar search and
                                logs `tool_source=search` in diagnostics.
    OPENAI_API_KEY (or other) — whichever provider backs --model

Usage:
    # Quick smoke test (small budget, validation only)
    python examples/asta_paper_finder/main.py --num-iterations 2 --evaluation-budget 20

    # Full run
    python examples/asta_paper_finder/main.py

    # With held-out test evaluation
    python examples/asta_paper_finder/main.py --eval-test-set
"""

import argparse
import json
import logging
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Evolve PaperFindingBench agents on AstaBench (Standard tools)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-iterations", type=int, default=999)
    p.add_argument("--evaluation-budget", type=int, default=1500)
    p.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd")

    p.add_argument("--model", default="openai/gpt-4o-mini",
                   help="Inspect model string for the candidate solver's LLM calls")
    p.add_argument("--tool-source", choices=["mcp", "search", "auto"], default="auto",
                   help="Tool kit: 'mcp' (Asta MCP, Standard tier), 'search' (public S2 fallback), or 'auto' (mcp if ASTA_TOOL_KEY set)")

    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel eval workers (default 4; Asta MCP rate limit ~4 req/sec)")
    p.add_argument("--runs-dir", default="../robophd_runs")
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--engine-config", type=str, default=None)
    p.add_argument("--meta-evolution-strategy", default=None)

    p.add_argument("--eval-test-set", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-agent", type=str, default=None,
                   help="Name of a specific agent from the --resume run's agent_pool to "
                        "evaluate (e.g. the seed name to baseline, or any iter agent name). "
                        "Requires --eval-only. Defaults to the best-ELO agent. Output file "
                        "is suffixed with the agent name so results don't overwrite the "
                        "default best-ELO results.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--extend", type=int, default=None)
    p.add_argument("--from-iteration", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    from evaluator import PaperFinderEvaluator, load_paper_finder

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    tool_source = None if args.tool_source == "auto" else args.tool_source
    evaluator = PaperFinderEvaluator(model=args.model, tool_source=tool_source)
    logger.info(f"Evaluator tool_source={evaluator.tool_source}")

    val = load_paper_finder("validation")
    logger.info(f"Validation set: {len(val)} samples")

    # RoboPhD's ExternalEvaluatorDomain JSON-serializes each example to
    # compute a stable id; Inspect's Sample is a pydantic model and isn't
    # directly JSON-serializable. Flatten to plain dicts at the boundary;
    # the evaluator reconstructs Sample.
    val = [s.model_dump() for s in val]

    if args.eval_agent and not args.eval_only:
        raise SystemExit("--eval-agent requires --eval-only")

    # --eval-only: skip optimization, evaluate an agent from --resume on test.
    # By default uses the best-ELO agent (via eval_run); --eval-agent overrides
    # to a specific named agent (via find_named_agent + eval_candidate).
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        test_data = [s.model_dump() for s in load_paper_finder("test")]
        logger.info(f"Test set: {len(test_data)} samples")

        if args.eval_agent:
            from RoboPhD.runner_utils import find_named_agent
            try:
                _, agent_dir = find_named_agent(Path(args.resume), args.eval_agent)
            except FileNotFoundError as e:
                raise SystemExit(str(e))
            candidate = {"agent.py": (agent_dir / "agent.py").read_text()}
            logger.info(f"Evaluating named agent: {args.eval_agent} from {agent_dir}")
            eval_result = eval_candidate(evaluator=evaluator, dataset=test_data, candidate=candidate)
        else:
            eval_result = eval_run(evaluator=evaluator, dataset=test_data, experiment_dir=args.resume)

        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} samples)")
        results_filename = (
            f"test_results_{args.eval_agent}.json" if args.eval_agent else "test_results.json"
        )
        test_path = Path(args.resume) / results_filename
        with open(test_path, "w") as f:
            json.dump({
                "agent": args.eval_agent or "best",
                "mean_test_score": eval_result.mean_score,
                "total_test_score": eval_result.total_score,
                "total_test_problems": eval_result.num_examples,
                "test_eval_cost_usd": evaluator.total_eval_cost,
            }, f, indent=2)
        logger.info(f"Test results saved to {test_path}")
        return

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Per-example timeout: RoboPhD's default is 300s. PaperFinder's
    # semantic-query scoring can fan out into many judge LLM calls
    # (one per predicted paper × relevance criterion), and our
    # inspect.eval lock serializes everything. 600s matches arc_agi_1
    # and DiscoveryBench. Worth revisiting once we have real run data
    # against the MCP path.
    EVAL_TIMEOUT = 600

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=args.max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
            eval_timeout=EVAL_TIMEOUT,
        )
        dataset = val
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            val_dataset=val,
            max_workers=args.max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
            eval_timeout=EVAL_TIMEOUT,
        )
        dataset = val
    else:
        dataset = val
        engine_overrides = json.loads(args.engine_config) if args.engine_config else {}
        cfg = RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            max_workers=args.max_workers,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
            meta_evolution_strategy=args.meta_evolution_strategy,
            engine_overrides=engine_overrides or None,
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
        task_name="asta_paper_finder",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: ELO {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = [s.model_dump() for s in load_paper_finder("test")]
            logger.info(f"Test evaluation: {len(test_data)} samples")
            eval_result = eval_candidate(
                evaluator=evaluator,
                dataset=test_data,
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
