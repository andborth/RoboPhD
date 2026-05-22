#!/usr/bin/env python3
"""
Evolve protein GO-term prediction agents (Molecular Function sub-ontology) using RoboPhD's optimize_anything() API.

Predicts Gene Ontology Molecular Function terms for protein sequences using
BLAST against SwissProt, the GO ontology, sequence features, and an LLM.

Test-set evaluation reports two numbers:
  1. Mean per-protein Fmax (the same signal used during evolution).
  2. Canonical CAFA Fmax (computed by CAFA-evaluator over the full test set,
     directly comparable to published methods).

Requires:
  - DIAMOND binary on PATH (conda install -c bioconda diamond)
  - Data files set up via setup.sh (SwissProt, GO ontology, splits)
  - OpenAI API key (for gpt-4.1-mini and text-embedding-3-small)

Usage:
    # Quick smoke test
    python examples/protein_go/main.py --num-iterations 2

    # Full run
    python examples/protein_go/main.py

    # With test evaluation (per-protein + CAFA Fmax)
    python examples/protein_go/main.py --eval-test-set
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

HERE = Path(__file__).resolve().parent

# Add project root and example dir to path
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from RoboPhD import (
    optimize_anything, eval_candidate, eval_run,
    RoboPhDConfig, GEPAConfig, AutoresearchConfig, RoboPhDEvalConfig,
)
from RoboPhD.runner_utils import find_named_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve protein GO-term prediction agents (Molecular Function sub-ontology)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Budget & scale
    parser.add_argument("--num-iterations", type=int, default=999,
                        help="Max iterations (evaluation budget is the real limit)")
    parser.add_argument("--evaluation-budget", type=int, default=1500,
                        help="Max evaluator calls across all iterations")

    # Engine
    parser.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"],
                        default="robophd", help="Optimization engine")

    # Task-specific
    parser.add_argument("--model", default="openrouter/google/gemini-3.1-flash-lite",
                        help="LLM for the agent's llm() callable. Default matches arc_agi_1's sibling tier.")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort for OpenRouter-routed reasoning-capable models "
                             "(passed as extra_body={\"reasoning\":{\"effort\": ...}}). "
                             "Use 'none' to suppress the extra_body entirely — required when "
                             "overriding --model to an Anthropic/OpenAI model that doesn't "
                             "accept the reasoning extra_body.")
    parser.add_argument("--embed-model", default="text-embedding-3-small",
                        help="Model for embeddings")
    parser.add_argument("--cost-budget", type=float, default=0.10,
                        help="Per-protein cost budget ($)")

    # Infrastructure
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel eval workers (8 to balance DIAMOND I/O and LLM calls)")
    parser.add_argument("--runs-dir", default="../robophd_runs",
                        help="Root directory for experiment output (default: %(default)s)")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--engine-config", type=str, default=None,
                        help="JSON overrides (e.g. evolution_strategy, evolution_model, examples_per_iteration)")
    parser.add_argument("--meta-evolution-strategy", default=None,
                        help="Meta-evolution strategy (e.g. train_a_winner); default off. "
                             "Cadence and first iteration default to (3, 4); override via --engine-config.")

    # Test evaluation
    parser.add_argument("--eval-test-set", action="store_true",
                        help="Run test-set evaluation (ProteInfer clustered split) after optimization")
    parser.add_argument("--eval-price149", action="store_true",
                        help="Run Price-149 evaluation (homology-resistant held-out set) "
                             "after optimization. Same scoring and $0.10 budget as --eval-test-set.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip optimization; evaluate best agent from --resume dir on the "
                             "splits requested via --eval-test-set / --eval-price149")
    parser.add_argument("--eval-agent", type=str, default=None,
                        help="Name of a specific agent from the --resume run's agent_pool to "
                             "evaluate (e.g. 'seed_407zcoan' to baseline the seed). Requires "
                             "--eval-only. Defaults to the best-Elo agent. Output file is "
                             "suffixed with the agent name so results don't overwrite the "
                             "default 'test_results.json' / 'price149_results.json'.")
    parser.add_argument("--skip-cafa-fmax", action="store_true",
                        help="Skip the batch CAFA Fmax computation during test eval (per-protein only)")

    # Resume / extend
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to experiment directory to resume")
    parser.add_argument("--extend", type=int, default=None,
                        help="Add N more iterations to a resumed run")
    parser.add_argument("--from-iteration", type=int, default=None,
                        help="Restart from a specific iteration")

    return parser.parse_args()


# Ordered list of (split_name, args_flag_attr, label, results_filename, cafa_subdir).
# Used by both the --eval-only branch and the post-optimization eval branch so the
# ProteInfer clustered test and Price-149 paths stay identical.
_EVAL_SPLITS = [
    ("test",     "eval_test_set",  "Test (ProteInfer clustered)", "test_results.json",     "cafa_eval"),
    ("price149", "eval_price149",  "Price-149",                   "price149_results.json", "cafa_eval_price149"),
]


def _write_split_report(
    eval_result, data, evaluator, args, output_dir: Path,
    *, split: str, label: str, results_filename: str, cafa_subdir: str,
) -> dict:
    """Log + write per-protein Fmax and (optionally) canonical CAFA Fmax for one split.
    Accepts an already-computed EvalResult so callers can use either
    eval_candidate() or eval_run().
    """
    logger.info(f"{label} per-protein Fmax (mean): {eval_result.mean_score:.3f} "
                f"({eval_result.num_examples} proteins)")

    results = {
        "split": split,
        "mean_per_protein_fmax": eval_result.mean_score,
        "total_per_protein_score": eval_result.total_score,
        "total_proteins": eval_result.num_examples,
        "eval_cost_usd": evaluator.total_cost,
        # Disclose the solver under which this result was produced so
        # cross-run comparisons can account for solver-model changes
        # (matches the sudoku 'aggregation' marker pattern).
        "solver_model": evaluator.model,
        "reasoning_effort": evaluator.reasoning_effort,
    }

    if not args.skip_cafa_fmax:
        cafa_result = _run_cafa_fmax(
            eval_result=eval_result,
            test_data=data,
            out_dir=output_dir / cafa_subdir,
        )
        if cafa_result is not None:
            results.update({
                "cafa_fmax": cafa_result["fmax"],
                "cafa_best_threshold": cafa_result["best_threshold"],
                "cafa_precision_at_best": cafa_result["precision_at_best"],
                "cafa_recall_at_best": cafa_result["recall_at_best"],
                "cafa_coverage": cafa_result["coverage"],
                "cafa_eval_dir": str(output_dir / cafa_subdir),
            })

    out_path = output_dir / results_filename
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"{label} results saved to {out_path}")
    return results


def _run_cafa_fmax(eval_result, test_data, out_dir: Path):
    """Aggregate per-protein predictions from eval diagnostics and run cafaeval.

    Returns the dict from compute_cafa_fmax_batch, or None if no predictions
    were recovered (e.g. all agents crashed).
    """
    from evaluator import compute_cafa_fmax_batch

    # eval_result.per_example_diagnostics is ordered to match `test_data`, so
    # we recover each protein's accession by index rather than storing it
    # redundantly in every diagnostic dict (the per-problem directory is
    # already named by accession).
    if len(eval_result.per_example_diagnostics) != len(test_data):
        logger.warning(
            "Diagnostics/test_data length mismatch (%d vs %d); skipping CAFA Fmax.",
            len(eval_result.per_example_diagnostics), len(test_data),
        )
        return None

    predictions_by_protein = {}
    for ex, diag in zip(test_data, eval_result.per_example_diagnostics):
        if not isinstance(diag, dict):
            continue
        preds = diag.get("predictions")
        if not (isinstance(preds, dict) and preds):
            continue
        accession = ex.get("id", "")
        if not accession:
            # Skip entries without an accession — would otherwise collide under
            # the empty-string dict key and silently clobber each other.
            continue
        predictions_by_protein[accession] = preds

    if not predictions_by_protein:
        logger.warning("No per-protein predictions found in diagnostics; skipping CAFA Fmax.")
        return None

    ground_truth_by_protein = {ex["id"]: ex["go_terms_mfo"] for ex in test_data}
    cafa_result = compute_cafa_fmax_batch(
        predictions_by_protein=predictions_by_protein,
        ground_truth_by_protein=ground_truth_by_protein,
        out_dir=out_dir,
    )
    logger.info(f"CAFA Fmax (canonical): {cafa_result['fmax']:.3f} "
                f"@ tau={cafa_result['best_threshold']:.2f} "
                f"(precision={cafa_result['precision_at_best']:.3f}, "
                f"recall={cafa_result['recall_at_best']:.3f}, "
                f"coverage={cafa_result['coverage']:.3f})")
    return cafa_result


def main():
    args = parse_args()

    from evaluator import ProteinGOEvaluator, load_protein_go

    objective = (HERE / "objective.md").read_text().strip()
    background = (HERE / "background.md").read_text().strip()

    # Map argparse's "none" sentinel to None so make_tracked_llm skips the
    # reasoning extra_body entirely for non-reasoning-capable solvers.
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    evaluator = ProteinGOEvaluator(
        model=args.model,
        embed_model=args.embed_model,
        cost_budget=args.cost_budget,
        reasoning_effort=reasoning_effort,
    )

    val = load_protein_go("validation")
    # Evolution pool = ProteInfer dev split (val.jsonl). ProteInfer train is
    # the BLAST reference corpus (populated into the DIAMOND DB by setup.sh);
    # sampling train proteins as evolution examples would self-hit the DB and
    # trivialize the task. Dev is UniRef50-disjoint from train, so evolution
    # signal lands in the same homology regime as the test set.
    logger.info(f"Evolution pool: {len(val)} proteins (ProteInfer dev split); "
                f"BLAST DB is the ProteInfer-train subset of SwissProt.")

    if args.eval_agent and not args.eval_only:
        raise SystemExit("--eval-agent requires --eval-only")

    # --eval-only: skip optimization, evaluate an agent from a prior run
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval-only requires --resume <experiment_dir>")
        requested = [s for s in _EVAL_SPLITS if getattr(args, s[1])]
        if not requested:
            # Unadorned --eval-only defaults to the test split (ProteInfer clustered).
            requested = [_EVAL_SPLITS[0]]
        resume_dir = Path(args.resume)

        # Resolve the named agent up front so we fail fast if the name is wrong,
        # before spending time on dataset loading or inference. CLI layer
        # translates the library's FileNotFoundError into SystemExit so the
        # user sees a clean error message with the available-names list.
        # When --eval-agent is not set, eval_run handles engine-agnostic
        # resolution (best_candidate.json / best_agent/ for GEPA/autoresearch;
        # checkpoint + agent_pool for RoboPhD).
        override_candidate: Optional[Dict[str, str]] = None
        if args.eval_agent:
            try:
                _, agent_dir = find_named_agent(resume_dir, args.eval_agent)
            except FileNotFoundError as e:
                raise SystemExit(str(e))
            override_candidate = {"agent.py": (agent_dir / "agent.py").read_text()}
            logger.info(f"Evaluating named agent: {args.eval_agent}")

        eval_cfg = RoboPhDEvalConfig(max_workers=args.max_workers)

        for split, _attr, label, results_filename, cafa_subdir in requested:
            data = load_protein_go(split)
            logger.info(f"{label} evaluation: {len(data)} proteins")
            if override_candidate is not None:
                eval_result = eval_candidate(
                    evaluator=evaluator, dataset=data, candidate=override_candidate,
                    config=eval_cfg,
                )
            else:
                eval_result = eval_run(
                    evaluator=evaluator, dataset=data, experiment_dir=args.resume,
                    config=eval_cfg,
                )
            # Suffix output filenames with the agent name so named-agent
            # runs don't overwrite the default best-Elo results on disk.
            if args.eval_agent:
                stem = Path(results_filename).stem  # 'test_results' etc.
                results_filename = f"{stem}_{args.eval_agent}.json"
                cafa_subdir = f"{cafa_subdir}_{args.eval_agent}"
            _write_split_report(
                eval_result=eval_result, data=data, evaluator=evaluator,
                args=args, output_dir=resume_dir,
                split=split, label=label,
                results_filename=results_filename, cafa_subdir=cafa_subdir,
            )
        return

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Build config based on engine choice
    if args.engine in ("gepa", "autoresearch"):
        _robophd_flags = {"--num-iterations", "--engine-config", "--resume", "--extend", "--from-iteration", "--meta-evolution-strategy"}
        passed = {f for f in _robophd_flags if any(a == f or a.startswith(f + "=") for a in sys.argv)}
        if passed:
            logger.warning(f"Flags ignored by {args.engine} engine: {', '.join(sorted(passed))}")

    # All three engines evolve on `val` (the ProteInfer dev split).
    # For GEPA / Autoresearch: leaving val_dataset=None triggers build_val_split
    # (RoboPhD/engines/__init__.py) to shuffle `dataset` and slice [:val_size]
    # as the selection set and [val_size:] as the minibatch / feedback pool.
    # With default val_size=100 and len(val)=2000 that's a 100/1900 split, both
    # drawn from ProteInfer dev and therefore homology-resistant to the
    # train-only BLAST DB.
    dataset = val

    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=args.evaluation_budget,
            max_workers=args.max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
        )
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            max_workers=args.max_workers,
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
            max_workers=args.max_workers,
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
        task_name="protein_go",
    )

    logger.info(f"Optimization complete: {result.num_iterations_completed} iterations, "
                f"{result.total_evaluations} evaluations")
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    requested = [s for s in _EVAL_SPLITS if getattr(args, s[1])]
    if requested:
        if not result.completed_normally:
            logger.info("Skipping post-optimization evaluation -- run ended early due to failure")
        else:
            experiment_dir = Path(result.experiment_dir)
            for split, _attr, label, results_filename, cafa_subdir in requested:
                data = load_protein_go(split)
                logger.info(f"{label} evaluation: {len(data)} proteins")
                eval_result = eval_candidate(
                    evaluator=evaluator,
                    dataset=data,
                    candidate=result.best_candidate,
                    config=RoboPhDEvalConfig(max_workers=args.max_workers),
                )
                _write_split_report(
                    eval_result=eval_result, data=data, evaluator=evaluator,
                    args=args, output_dir=experiment_dir,
                    split=split, label=label,
                    results_filename=results_filename, cafa_subdir=cafa_subdir,
                )


if __name__ == "__main__":
    from RoboPhD.eval_utils import force_exit_if_threads_leaked
    try:
        main()
    finally:
        force_exit_if_threads_leaked()
