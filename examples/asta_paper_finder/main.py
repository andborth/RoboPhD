#!/usr/bin/env python3
"""
Evolve PaperFindingBench agents (AstaBench, Standard tools tier) using
RoboPhD's optimize_anything() API.

Targets the AstaBench Literature Understanding leaderboard's PaperFindingBench
subtask. Validation = 66 samples (training pool), test = 267 samples (held out).

Credentials required:
    HF_ACCESS_TOKEN — gated allenai/asta-bench dataset
    ASTA_TOOL_KEY   — Asta MCP corpus tools (the leaderboard's Standard
                      kit; 10 req/s per endpoint). If unset, the evaluator
                      falls back to public-Semantic-Scholar search and
                      logs `tool_source=search` in diagnostics.
    OPENAI_API_KEY, ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_FOR_ROBOPHD),
    GOOGLE_API_KEY  — evolution may pick any of nine solver models across
                      three providers (see model_registry.py). OPENAI is
                      doubly required: the benchmark's GPT-4o relevance
                      judge scores every semantic query.

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
import os
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
    RoboPhDEvalConfig,
    GEPAConfig,
    AutoresearchConfig,
)
from RoboPhD.runner_utils import (
    apply_engine_config,
    read_task_config_extras,
    resolve_run_immutable,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
for noisy in ("LiteLLM", "litellm", "httpx", "openai._base_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Framework default for parallel eval workers. Single source of truth in
# code: referenced from both --max-workers' resolution logic AND its
# argparse help text (via f-string). The Asta MCP rate limit is 10 req/s
# per endpoint on the default key; each worker's agent typically issues
# a few tool calls per query, so 8 workers sits comfortably under it.
DEFAULT_MAX_WORKERS = 8

DEFAULT_NUM_ITERATIONS = 999
# Default evaluation budget (max fresh evaluator calls across all
# iterations) — the binding limit for a run. Sized for ~20 iterations:
# the first completed run burned ~31 fresh evals per iteration (3
# agents × 14 examples nominal, discounted by the (agent, example)
# cache), so 600 ≈ 19-20 iterations. Originally 500 (asta_ds1000's 750
# scaled by the 66/100 training-pool ratio), which yielded 16.
DEFAULT_EVALUATION_BUDGET = 600

# Per-example timeout: must match the value passed to RoboPhDConfig /
# RoboPhDEvalConfig below. The evaluator derives a slightly-shorter
# subprocess_timeout internally (eval_timeout - 30s) so subprocesses get
# SIGKILLed BEFORE RoboPhD's reaper would leak the thread.
#
# 30-minute cap, matching asta_ds1000 and for the same reason: wall
# clock is not a leaderboard criterion, so the timeout is a runaway
# backstop, never a design constraint. The tool-heavy strategies the
# docs encourage are legitimately slow — per-paper snippet_search
# evidence loops run seconds-to-minutes per call, and the 429 retry
# wrapper can burn ~6 minutes of backoff under sustained throttling —
# and ds1000's postmortem showed evolution misattributes timeout zeros
# as reasoning regressions. Raised from 600s (which no eval in the
# first completed run exceeded, max observed ~227s, but which the
# newly-documented evidence-gathering patterns could plausibly hit).
EVAL_TIMEOUT = 1800

# Key under task_config in checkpoint.json that holds PaperFinder's
# task-specific runtime values (cost_threshold / cost_per_error /
# tool_source). Persisted by the framework every iteration via
# RoboPhDConfig's task_config_extras, so the values survive any mid-run
# interruption that leaves a resumable checkpoint.
PFB_TASK_CONFIG_KEY = "paper_finder_runtime"

# All three knobs resolve through resolve_run_immutable independently,
# so a fully-missing store needs all of them supplied on the same
# invocation.
_ALL_FLAGS_NOTE = (
    "NOTE: when no values are stored, --cost-threshold, --cost-per-error "
    "and --tool-source must all be supplied on the same resume "
    "invocation — each is checked independently. "
)


def _read_pfb_runtime_config(resume_dir: Path) -> dict:
    """PaperFinder binding of runner_utils.read_task_config_extras.

    No legacy sidecar: this example adopted task_config_extras before
    its first real run, so there are no historical sidecar-only runs.
    """
    return read_task_config_extras(resume_dir, PFB_TASK_CONFIG_KEY)


def _enforce_immutable_on_resume(
    cli_value, stored_value, default_value, name: str, *, on_resume: bool, fmt=str,
):
    """PaperFinder binding of runner_utils.resolve_run_immutable — adds
    the all-flags-together note to the missing-value error."""
    return resolve_run_immutable(
        cli_value, stored_value, default_value, name,
        on_resume=on_resume, fmt=fmt, missing_note=_ALL_FLAGS_NOTE,
    )


def _resume_enforces_task_knobs(engine: str, resume: bool, eval_only: bool) -> bool:
    """Validate --resume usage and report whether the run-immutable task
    knobs (cost-threshold / cost-per-error / tool-source) must be
    enforced on this run.

    The cost knobs are training-only — the test evaluator runs with
    apply_cost_penalty=False, so --eval-only never reads them. Enforcing
    their stored-value immutability on an eval-only resume would be wrong,
    and impossible for GEPA/Autoresearch, which never persist
    paper_finder_runtime (only RoboPhDConfig carries task_config_extras).

    GEPA/Autoresearch additionally only support --resume in --eval-only mode
    (there is no training-resume path for them); reject the misuse with a
    clear message instead of letting it fall through to a confusing error.
    """
    if engine in ("gepa", "autoresearch") and resume and not eval_only:
        raise SystemExit(
            "--resume for the gepa/autoresearch engines is only supported "
            "with --eval-only (training-resume is not available for these "
            "engines). Re-run with --eval-only, or start a fresh run."
        )
    return resume and not eval_only


def _read_checkpoint_max_workers(resume_dir: Path) -> int | None:
    """Read max_workers from a resumed run's checkpoint.json, or None if
    absent / unparseable.

    Walks `config_manager.iteration_configs` to the highest iteration that
    has an explicit `max_workers` value (skipping iterations with null,
    which represent "framework default"). Returns the last user-explicit
    value or None.

    Used by --resume + --eval-only (and resume-then-train) so the eval
    phase honors whatever max_workers the original training run used,
    matching the user expectation that resume preserves settings.
    """
    cp_path = resume_dir / "checkpoint.json"
    if not cp_path.is_file():
        return None
    try:
        cp = json.loads(cp_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    iter_configs = (cp.get("config_manager") or {}).get("iteration_configs") or {}
    if not iter_configs:
        return None
    # Iteration keys are stringified ints; walk in descending order so we
    # pick the most-recent user-explicit value.
    best: int | None = None
    for k in sorted(iter_configs.keys(), key=lambda s: int(s) if s.isdigit() else -1, reverse=True):
        val = (iter_configs[k] or {}).get("max_workers")
        if val is not None:
            best = int(val)
            break
    return best


def _write_test_results(
    eval_result,
    evaluator,
    output_dir: Path,
    agent_name: str,
    summary_filename: str,
):
    """Write a summary JSON plus a sibling .per_problem.json for a test eval.

    Costs are split agent-vs-judge: `agent_cost_usd` is the candidate's
    own registry-handle spend; `other_cost_usd` is the benchmark's GPT-4o
    relevance judge (semantic queries only), outside agent control.
    """
    # Head+tail truncation so the tail (where tracebacks carry the real
    # failure line) survives.
    from evaluator import _head_tail_truncate

    def _trunc(s):
        return _head_tail_truncate(s, head=100, tail=400) if s else None

    per_problem = []
    total_agent_cost = 0.0
    total_judge_cost = 0.0
    diagnostics_list = eval_result.per_example_diagnostics or []
    scores_list = eval_result.per_example_scores or []
    for i, diag in enumerate(diagnostics_list):
        diag = diag or {}
        score = scores_list[i] if i < len(scores_list) else None
        agent_c = diag.get("agent_cost_usd") or 0.0
        judge_c = diag.get("other_cost_usd") or 0.0
        total_agent_cost += agent_c
        total_judge_cost += judge_c
        # Coalesce both error keys: the evaluator writes "error.md" (the
        # framework's failure-detection key), but framework-level test-path
        # timeouts (RoboPhD/eval_utils.py) still emit bare "error".
        err = diag.get("error.md") or diag.get("error")
        per_problem.append({
            "sample_id": diag.get("sample_id"),
            "score": score,
            "score_type": diag.get("score_type"),
            "agent_cost_usd": agent_c,
            "other_cost_usd": judge_c,
            "eval_wall_clock_seconds": diag.get("eval_wall_clock_seconds"),
            "error": _trunc(err),
        })

    summary_path = output_dir / summary_filename
    n_problems = eval_result.num_examples
    mean_agent_cost = (total_agent_cost / n_problems) if n_problems else 0.0
    with open(summary_path, "w") as f:
        json.dump({
            "agent": agent_name,
            "tool_source": evaluator.tool_source,
            "mean_test_score": eval_result.mean_score,
            "total_test_score": eval_result.total_score,
            "total_test_problems": n_problems,
            "test_eval_cost_usd": evaluator.total_eval_cost,
            "test_eval_agent_cost_usd": total_agent_cost,
            "test_eval_judge_cost_usd": total_judge_cost,
            "mean_test_agent_cost_usd": mean_agent_cost,
            # Empty for the default test path (apply_cost_penalty=False,
            # aggregator returns mean_raw with no annotation). Populated
            # if a future test mode opts into a non-default aggregator.
            "aggregate_explanation": getattr(eval_result, "aggregate_explanation", ""),
        }, f, indent=2)

    per_problem_path = summary_path.with_suffix(".per_problem.json")
    with open(per_problem_path, "w") as f:
        json.dump(per_problem, f, indent=2)

    return summary_path, per_problem_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evolve PaperFindingBench agents on AstaBench (Standard tools)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-iterations", type=int, default=DEFAULT_NUM_ITERATIONS,
                   help="Iteration cap, deliberately loose — runs are bound by "
                        "--evaluation-budget, not this.")
    p.add_argument("--evaluation-budget", type=int, default=DEFAULT_EVALUATION_BUDGET,
                   help="Max fresh evaluator calls across the run (the binding "
                        "limit; ~30 per iteration after cache effects).")
    p.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd")

    p.add_argument("--tool-source", choices=["mcp", "search", "auto"], default=None,
                   help="Tool kit: 'mcp' (Asta MCP, Standard tier), 'search' "
                        "(public S2 fallback, explicit dev opt-in), or 'auto' "
                        "(mcp; hard error if ASTA_TOOL_KEY is unset — never a "
                        "silent fallback). Resolved to a concrete value at run "
                        "start and locked for the lifetime of the run "
                        "(immutable on --resume). Default: auto."
                        "%(default).0s")

    p.add_argument("--cost-threshold", type=float, default=None,
                   help="Mean agent cost across an iteration's batch below "
                        "this is in the free zone (no penalty). Judge cost "
                        "never counts. Default $0.10."
                        "%(default).0s")
    p.add_argument("--cost-per-error", type=float, default=None,
                   help="Dollars of mean batch spend (over --cost-threshold) "
                        "that equals one fully-wrong query of penalty. "
                        "Default $0.02. See README 'Cost-penalty math'."
                        "%(default).0s")

    p.add_argument("--max-workers", type=int, default=None,
                   help=f"Parallel eval workers (default: {DEFAULT_MAX_WORKERS}; "
                        f"on --resume, the checkpoint's value). Each evaluation "
                        f"runs in its own subprocess to bypass inspect.eval's "
                        f"process-global singleton lock. The Asta MCP rate "
                        f"limit (10 req/s per endpoint) is the ceiling to "
                        f"watch when raising this."
                        "%(default).0s")  # suppress argparse's auto "(default: None)"
    p.add_argument("--runs-dir", default="../robophd_runs",
                   help="Root directory for experiment output (default: %(default)s)")
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--engine-config", type=str, default=None)
    p.add_argument("--meta-evolution-strategy", default=None)

    p.add_argument("--eval-test-set", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-agent", type=str, default=None,
                   help="Name of a specific agent from the --resume run's agent_pool to "
                        "evaluate (e.g. the seed name to baseline, or any iter agent name). "
                        "Requires --eval-only. Defaults to the best-Elo agent. Output file "
                        "is suffixed with the agent name so results don't overwrite the "
                        "default best-Elo results.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--extend", type=int, default=None)
    p.add_argument("--from-iteration", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    from evaluator import (
        COST_PER_ERROR,
        MIN_COST_THRESHOLD,
        PaperFinderEvaluator,
        _fmt_cost,
        load_paper_finder,
    )

    # Resolve the run-immutable task knobs. These aren't known to the
    # framework's ConfigManager, so without task_config_extras they'd
    # silently revert to defaults on resume — a quiet scoring-function
    # (or tool-tier!) shift mid-run. They are immutable across a run:
    # passing a disagreeing CLI flag on --resume is a hard error.
    on_resume = _resume_enforces_task_knobs(
        args.engine, bool(args.resume), args.eval_only
    )
    checkpoint_pfb = (
        _read_pfb_runtime_config(Path(args.resume)) if args.resume else {}
    )
    cost_threshold = _enforce_immutable_on_resume(
        cli_value=args.cost_threshold,
        stored_value=checkpoint_pfb.get("cost_threshold"),
        default_value=MIN_COST_THRESHOLD,
        name="cost-threshold",
        on_resume=on_resume,
        fmt=_fmt_cost,
    )
    cost_per_error = _enforce_immutable_on_resume(
        cli_value=args.cost_per_error,
        stored_value=checkpoint_pfb.get("cost_per_error"),
        default_value=COST_PER_ERROR,
        name="cost-per-error",
        on_resume=on_resume,
        fmt=_fmt_cost,
    )

    # tool_source: resolve "auto" (or absent) to a CONCRETE value before
    # persisting, so a resume on a machine without ASTA_TOOL_KEY can't
    # silently flip a Standard-tier run down to the search fallback.
    # Auto resolves ONLY to mcp — with no key it's a hard startup error,
    # never a silent fallback (a warning was tried first and got missed;
    # run asta_paper_finder_20260710_081139 burned its budget on
    # unauthenticated-S2 429s). --tool-source search remains the explicit
    # dev escape hatch.
    cli_tool_source = args.tool_source if args.tool_source in ("mcp", "search") else None

    def _auto_tool_source() -> str:
        if os.environ.get("ASTA_TOOL_KEY"):
            return "mcp"
        raise SystemExit(
            "ASTA_TOOL_KEY is not set. The AstaBench Standard tier requires "
            "the Asta MCP corpus tools; export ASTA_TOOL_KEY in the shell "
            "that launches the run. For key-less dev against public "
            "Semantic Scholar (scores will NOT match the leaderboard), opt "
            "in explicitly with --tool-source search."
        )

    if on_resume:
        resolved_tool_source = _enforce_immutable_on_resume(
            cli_value=cli_tool_source,
            stored_value=checkpoint_pfb.get("tool_source"),
            # Only consulted if the checkpoint predates tool_source
            # persistence AND no CLI flag was passed — resolve_run_immutable
            # errors in that case before reading the default, but keep the
            # auto rule here for coherence.
            default_value=None,
            name="tool-source",
            on_resume=True,
        )
    else:
        # Fresh run: CLI wins, else auto (mcp-or-error). --eval-only: CLI
        # wins, else the resumed run's stored value (evaluate with the
        # tier the run was trained on), else auto.
        resolved_tool_source = (
            cli_tool_source or checkpoint_pfb.get("tool_source") or _auto_tool_source()
        )
    if resolved_tool_source == "search" and not os.environ.get("ASTA_TOOL_KEY"):
        logger.warning(
            "tool_source='search' with no ASTA_TOOL_KEY set: requests hit "
            "Semantic Scholar unauthenticated and will throttle hard under "
            "parallel workers. Get a free personal S2 API key "
            "(semanticscholar.org/product/api) and export it as ASTA_TOOL_KEY."
        )

    resolved_runtime = {
        "cost_threshold": cost_threshold,
        "cost_per_error": cost_per_error,
        "tool_source": resolved_tool_source,
    }

    def _build_cost_penalty_table(threshold: float, cpe: float) -> str:
        """Generate the cost-penalty table substituted into background.md.

        Each row spans 1× cpe of mean cost; the breakeven fully-wrong-
        query count grows by 1 per row. We stop at 3+ (the "decisive"
        tier) and emit a trailing pattern-continuation row so the agent
        doesn't have to extrapolate from numerics alone.
        """
        rows = [
            "| Mean agent cost | Effect on score |",
            "|---|---|",
            f"| ≤ {_fmt_cost(threshold)} | No effect on score — two free-zone "
            f"agents with the same raw mean F1 score identically, "
            f"regardless of their actual spend |",
            f"| {_fmt_cost(threshold)}–{_fmt_cost(threshold + cpe)} | "
            f"Tiebreaker — lose tied F1 to a cheaper agent; "
            f"**need 1+ more fully-correct query** to win |",
            f"| {_fmt_cost(threshold + cpe)}–{_fmt_cost(threshold + 2*cpe)} | "
            f"**Need 2+ more fully-correct queries** than a free-zone agent to win |",
            f"| {_fmt_cost(threshold + 2*cpe)}–{_fmt_cost(threshold + 3*cpe)} | "
            f"**Need 3+ more fully-correct queries** than a free-zone agent to win "
            f"(in practice, a decisive penalty) |",
            f"| … | Each additional {_fmt_cost(cpe)} of mean spend adds 1 "
            f"to the breakeven count |",
        ]
        return "\n".join(rows)

    cost_penalty_table = _build_cost_penalty_table(cost_threshold, cost_per_error)

    def _interpolate(text: str) -> str:
        return (
            text
            .replace("${COST_PENALTY_TABLE}", cost_penalty_table)
            .replace("${COST_THRESHOLD}", _fmt_cost(cost_threshold))
            .replace("${COST_PER_ERROR}", _fmt_cost(cost_per_error))
            # True per-query budget the agent experiences: EVAL_TIMEOUT
            # minus the 30s reaper buffer, floored to whole minutes.
            # Floored & buffer-aware so the doc never over-promises the
            # wall-clock the agent actually gets.
            .replace("${EVAL_TIMEOUT_MIN}", str((EVAL_TIMEOUT - 30) // 60))
        )

    objective = _interpolate((HERE / "objective.md").read_text().strip())
    background = _interpolate((HERE / "background.md").read_text().strip())

    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Two evaluator instances. Training applies the mean-cost penalty;
    # test paths report raw mean F1 so evolved agents land at their true
    # point on the Pareto cost-vs-score curve.
    evaluator = PaperFinderEvaluator(
        eval_timeout=EVAL_TIMEOUT,
        tool_source=resolved_tool_source,
        apply_cost_penalty=True,  # training: penalty fires
        min_cost_threshold=cost_threshold,
        cost_per_error=cost_per_error,
    )
    test_evaluator = evaluator.with_overrides(apply_cost_penalty=False)
    logger.info(f"Evaluator tool_source={evaluator.tool_source}")

    # Resolve --max-workers once, applied to BOTH the training engine
    # config and the test eval config so --eval-only --resume honors the
    # resumed run's setting rather than silently spinning up
    # ThreadPoolExecutor's default.
    #
    # Order: explicit CLI flag wins. On --resume with no flag, recover
    # the value the original run used (matches the user expectation that
    # resume preserves settings). Otherwise fall back to DEFAULT_MAX_WORKERS.
    if args.max_workers is not None:
        effective_max_workers = args.max_workers
    elif args.resume:
        cp_max_workers = _read_checkpoint_max_workers(Path(args.resume))
        effective_max_workers = cp_max_workers if cp_max_workers is not None else DEFAULT_MAX_WORKERS
        if cp_max_workers is not None:
            logger.info(
                f"Resume: using max_workers={cp_max_workers} from "
                f"checkpoint.json (pass --max-workers N to override)"
            )
    else:
        effective_max_workers = DEFAULT_MAX_WORKERS

    # Single source of truth for test-side eval config. Reused at every
    # eval_candidate / eval_run call site below (--eval-only and
    # --eval-test-set paths) so the test pipeline can't silently drift
    # from the training pipeline's eval_timeout or max_workers.
    test_eval_config = RoboPhDEvalConfig(
        eval_timeout=EVAL_TIMEOUT,
        max_workers=effective_max_workers,
    )

    train = load_paper_finder("validation")
    logger.info(f"Training pool (validation split): {len(train)} samples")

    # RoboPhD's ExternalEvaluatorDomain JSON-serializes each example to
    # compute a stable id (SHA256 of the dict). Inspect's Sample is a
    # pydantic model; flatten to plain dicts at the boundary; the
    # evaluator reconstructs Sample.
    train = [s.model_dump() for s in train]

    if args.eval_agent and not args.eval_only:
        raise SystemExit("--eval-agent requires --eval-only")

    # --eval-only: skip optimization, evaluate an agent from --resume on test.
    # By default uses the best-Elo agent (via eval_run); --eval-agent overrides
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
            eval_result = eval_candidate(
                evaluator=test_evaluator,
                dataset=test_data,
                candidate=candidate,
                config=test_eval_config,
            )
        else:
            eval_result = eval_run(
                evaluator=test_evaluator,
                dataset=test_data,
                experiment_dir=args.resume,
                config=test_eval_config,
            )

        logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} samples)")
        results_filename = (
            f"test_results_{args.eval_agent}.json" if args.eval_agent else "test_results.json"
        )
        summary_path, per_problem_path = _write_test_results(
            eval_result=eval_result,
            evaluator=test_evaluator,
            output_dir=Path(args.resume),
            agent_name=args.eval_agent or "best",
            summary_filename=results_filename,
        )
        logger.info(f"Test summary:    {summary_path}")
        logger.info(f"Test per-problem: {per_problem_path}")
        return

    # Build engine_overrides. Two principles:
    #
    # (1) User-explicit CLI values always propagate (this iteration of
    #     resume, or initial run — same behavior).
    # (2) On --resume with NO user value, don't pack anything: the
    #     original run's setting must survive (the resume path reapplies
    #     engine_overrides as a config delta, so a CLI default packed
    #     here would silently clobber the checkpoint's value).
    # (3) On INITIAL RUN with no user value, pack the task-specific
    #     default when it differs from RoboPhD's framework default.
    #
    # new_agent_test_rounds: PaperFinder wants 0 (Round 2 disabled —
    # 73% of samples invoke the GPT-4o judge, so per-agent training
    # evals are expensive; prefer more agents per budget). RoboPhD's
    # framework default is 1, so on initial runs we MUST pack the task
    # default; only on resume do we omit it. Override via
    # --engine-config '{"new_agent_test_rounds": 1}'.
    #
    # examples_per_iteration: ceil(train_pool / 5), i.e. 14 for the
    # 66-sample pool — below RoboPhD's framework default of 20. Each
    # iteration re-samples from only 66 queries, so at 20/iteration the
    # eval budget reuses every example fast and concentrates that reuse;
    # capping the per-iteration draw at a fifth of the pool slows
    # per-example exposure to limit overfitting to the training
    # queries. Derived from len(train) rather than hardcoded so a future
    # thermometer holdout (see README) shrinks it automatically.
    # Override via --engine-config '{"examples_per_iteration": N}'.
    parsed_engine_config = (
        json.loads(args.engine_config) if args.engine_config else {}
    )
    is_resume = args.resume is not None
    engine_overrides: dict = {}
    if not is_resume:
        engine_overrides["new_agent_test_rounds"] = 0
        engine_overrides["examples_per_iteration"] = -(-len(train) // 5)
    # Route max_workers through engine_overrides for the RoboPhD engine (not the
    # dedicated RoboPhDConfig field, which only applies on a fresh run — the
    # resume path re-applies engine_overrides as a config delta but never reads
    # cfg.max_workers). With no flag, pack the task default ONLY on a fresh run
    # so resume inherits the checkpoint value instead.
    if args.max_workers is None and not is_resume:
        engine_overrides["max_workers"] = DEFAULT_MAX_WORKERS
    engine_overrides.update(parsed_engine_config)
    # Re-assert an explicit --max-workers AFTER merging --engine-config so the
    # flag wins over an --engine-config max_workers (matches the other tasks);
    # done last so it also takes effect on a training --resume.
    if args.max_workers is not None:
        engine_overrides["max_workers"] = args.max_workers

    # GEPA and Autoresearch validate-then-select: the val set drives their
    # keep/discard decisions, so it must NOT be the held-out test set —
    # validating against test selects the agent directly on the data the
    # --eval-test-set score then reports, a leak. Instead carve validation
    # out of the 66-sample train pool: pass no val_dataset and set
    # val_size to half the pool, so build_val_split splits train into
    # equal train/val halves. RoboPhD is unaffected — it has no separate
    # val set and never touches test until --eval-test-set.
    engine_val_size = len(train) // 2
    if args.engine == "gepa":
        cfg = GEPAConfig(
            evaluation_budget=args.evaluation_budget,
            val_size=engine_val_size,
            max_workers=effective_max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
            eval_timeout=EVAL_TIMEOUT,
        )
        cfg = apply_engine_config(cfg, parsed_engine_config)
        dataset = train
    elif args.engine == "autoresearch":
        cfg = AutoresearchConfig(
            evaluation_budget=args.evaluation_budget,
            val_size=engine_val_size,
            max_workers=effective_max_workers,
            seed=args.random_seed or 0,
            parent_experiments_dir=args.runs_dir,
            eval_timeout=EVAL_TIMEOUT,
        )
        cfg = apply_engine_config(cfg, parsed_engine_config)
        dataset = train
    else:
        dataset = train
        cfg = RoboPhDConfig(
            num_iterations=args.num_iterations,
            evaluation_budget=args.evaluation_budget,
            parent_experiments_dir=args.runs_dir,
            random_seed=args.random_seed,
            meta_evolution_strategy=args.meta_evolution_strategy,
            engine_overrides=engine_overrides,
            eval_timeout=EVAL_TIMEOUT,
            # Persisted into checkpoint.json's task_config every iteration,
            # so the knobs survive any interruption that leaves a resumable
            # checkpoint. On resume this is the same values re-resolved
            # through _enforce_immutable_on_resume (idempotent merge), or
            # the one-time bootstrap values.
            task_config_extras={PFB_TASK_CONFIG_KEY: resolved_runtime},
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
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
    logger.info(f"Experiment dir: {result.experiment_dir}")

    if args.eval_test_set:
        if not result.completed_normally:
            logger.info("Skipping test-set evaluation -- run ended early due to failure")
        else:
            test_data = [s.model_dump() for s in load_paper_finder("test")]
            logger.info(f"Test evaluation: {len(test_data)} samples")
            eval_result = eval_candidate(
                evaluator=test_evaluator,
                dataset=test_data,
                candidate=result.best_candidate,
                config=test_eval_config,
            )
            logger.info(f"Test score: {eval_result.mean_score:.3f} ({eval_result.num_examples} samples)")
            summary_path, per_problem_path = _write_test_results(
                eval_result=eval_result,
                evaluator=test_evaluator,
                output_dir=result.experiment_dir,
                agent_name="best",
                summary_filename="test_results.json",
            )
            logger.info(f"Test summary:    {summary_path}")
            logger.info(f"Test per-problem: {per_problem_path}")


if __name__ == "__main__":
    from RoboPhD.eval_utils import force_exit_if_threads_leaked
    try:
        main()
    finally:
        force_exit_if_threads_leaked()
