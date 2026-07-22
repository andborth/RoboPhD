#!/usr/bin/env python3
"""
Evolve PaperFindingBench agents (AstaBench, Standard tools tier) using
RoboPhD's optimize_anything() API.

Targets the AstaBench Literature Understanding leaderboard's PaperFindingBench
subtask. Validation = 66 samples (training pool), test = 267 samples (held out).

Credentials required:
    HF_ACCESS_TOKEN — gated allenai/asta-bench dataset
    ASTA_TOOL_KEY   — Asta MCP corpus tools (the leaderboard's Standard
                      kit; 10 req/s per endpoint). Hard-required — the
                      MCP suite is the task's only retrieval surface.
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
import re
import sys
import tempfile
import uuid
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
# task-specific runtime values (cost_threshold / cost_per_error).
# Persisted by the framework every iteration via
# RoboPhDConfig's task_config_extras, so the values survive any mid-run
# interruption that leaves a resumable checkpoint.
PFB_TASK_CONFIG_KEY = "paper_finder_runtime"

# Both knobs resolve through resolve_run_immutable independently, so a
# fully-missing store needs both supplied on the same invocation.
_ALL_FLAGS_NOTE = (
    "NOTE: when no values are stored, --cost-threshold, --cost-per-error "
    "and --cap-judge-to-estimate/--no-cap-judge-to-estimate must all be "
    "supplied on the same resume invocation — each is checked independently. "
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
    knobs must be enforced on this run.

    These knobs are training-only — the test evaluator runs with
    apply_cost_penalty=False and uncapped judging, so --eval-only never reads
    them. Enforcing their stored-value immutability on an eval-only resume
    would be wrong,
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
    scoring_mode: dict,
):
    """Write a summary JSON plus a sibling .per_problem.json for a test eval.

    Costs are split agent-vs-judge: `agent_cost_usd` is the candidate's
    own registry-handle spend; `other_cost_usd` is the benchmark's GPT-4o
    relevance judge (semantic queries only), outside agent control.

    `scoring_mode` (from `_set_test_cache_env`) is merged into the summary
    so every recorded score names the judge model, cache mode, and cap
    setting that produced it — capped/shared numbers are a slightly
    different basis than pristine/uncapped (official) ones. Summaries
    written before these fields existed are implicitly pristine+uncapped.
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
            **scoring_mode,
        }, f, indent=2)

    per_problem_path = summary_path.with_suffix(".per_problem.json")
    with open(per_problem_path, "w") as f:
        json.dump(per_problem, f, indent=2)

    return summary_path, per_problem_path


# The two judges a run may use. Stock GPT-4o is the default, astabench's
# hardcoded official judge, and the only basis comparable to leaderboard
# scores. gpt-5.6-luna is the sole approved alternate: it passed the
# calibration gate on 2026-07-20 (kappa 0.755, matched Perfect rates —
# see README "Training judge"). A unit test pins element 0 to astabench's
# GRADER_MODEL_NAME and both elements to evaluator.JUDGE_MODEL_IDS.
TRAINING_JUDGE_CHOICES = [
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-5.6-luna",
]


def _judge_slug(judge: str) -> str:
    """Filesystem-safe judge-model slug used in judge-cache filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", judge)


def _test_results_filename(eval_agent: str | None, judge: str, stock: str) -> str:
    """Result-file basename for a test eval.

    Stock-judge evals keep the historical names (test_results.json /
    test_results_<agent>.json) so past runs and tooling are unaffected.
    A non-stock judge gets a `.judge_<model>` suffix so its scores — a
    different, official-incomparable basis — can never collide with (or
    be mistaken for) a stock GPT-4o evaluation of the same agent, and a
    stock re-eval can be added alongside later."""
    base = f"test_results_{eval_agent}" if eval_agent else "test_results"
    if judge != stock:
        base += f".judge_{_judge_slug(judge.split('/', 1)[-1])}"
    return base + ".json"


def _judge_cache_dir(runs_dir: str) -> Path:
    """Cross-run judge-cache directory, shared by training and test caches."""
    return Path(runs_dir) / ".judge_cache"


def _set_test_cache_env(
    label: str, *, runs_dir: str, shared: bool, cap: bool, judge: str, stock: str
) -> dict:
    """Configure the judge env for a test / formal eval and return the
    scoring-mode record for test_results.json.

    ``judge`` is the effective grader (``--training-judge`` or the stock
    GPT-4o id in ``stock``). Stock evals clear the override env and are
    the ONLY basis comparable to official astabench scores; a non-stock
    judge (opt-in, e.g. the calibrated gpt-5.6-luna) sets the override —
    the evaluator then also installs the lenient output normalizer — and
    is for A/B comparisons and lineage triage, recorded as such in the
    returned scoring-mode dict and in the suffixed result filename
    (_test_results_filename). The verdict cache is judge-scoped either
    way, so verdicts from different judges can never mix. Official
    submissions are untouched by all of this (stock astabench code).

    The other two knobs:

    - ``shared`` (default; ``--no-shared-judge-cache`` turns it off):
      persistent verdict cache at ``<runs_dir>/.judge_cache/
      shared_test_<judge-slug>.json``. Score-comparable to fresh judging
      because verdicts are keyed by (query, paper, evidence-hash) and the
      file is scoped by judge model, so a hit only ever replays the stock
      judge's verdict on identical inputs. The file is dedicated to test
      evals — never training's ``shared_<slug>.json`` — so its provenance
      stays pure and a future dataset renumbering of the (currently
      disjoint) split query-id namespaces cannot leak verdicts across
      splits. ``shared=False`` uses a fresh EMPTY per-invocation file
      instead: every verdict rendered anew on the submitting agent's own
      grounded evidence, matching a fresh official environment exactly.

    - ``cap`` (the run-wide ``--cap-judge-to-estimate``, default on):
      judge only the top-estimate submitted papers per semantic query.
      Much cheaper, but a slightly different metric than official
      uncapped scoring (the rank term sees fewer grades), which is why
      the mode is recorded alongside the scores.
    """
    from evaluator import CACHE_PATH_ENV, CAP_JUDGE_ENV, TRAINING_GRADER_ENV

    is_stock = judge == stock
    if shared:
        path = _judge_cache_dir(runs_dir) / f"shared_test_{_judge_slug(judge)}.json"
        scope = "shared"
    else:
        fd, tmp = tempfile.mkstemp(prefix=f"pf_pristine_{label}_", suffix=".json")
        os.close(fd)
        os.unlink(tmp)  # start truly empty: init_references treats missing as {}
        path = Path(tmp)
        scope = "pristine"
    os.environ[CACHE_PATH_ENV] = str(path)
    if is_stock:
        os.environ.pop(TRAINING_GRADER_ENV, None)
    else:
        os.environ[TRAINING_GRADER_ENV] = judge
    if cap:
        os.environ[CAP_JUDGE_ENV] = "1"
    else:
        os.environ.pop(CAP_JUDGE_ENV, None)
    logger.info(
        f"Judge cache ({scope} {label}): {path}; judge: {judge}"
        f"{'' if is_stock else ' (NON-STOCK — not official-comparable)'}; "
        f"judging cap: {'on' if cap else 'off'}"
    )
    record_extra = {} if is_stock else {
        "judge_note": (
            "non-stock judge: scores are NOT comparable to official "
            "astabench results (stock GPT-4o); for A/B comparison and "
            "lineage triage only"
        ),
    }
    return {
        **record_extra,
        "judge_model": judge,
        "judge_cache": scope,
        "judge_cache_path": str(path),
        "cap_judge_to_estimate": cap,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    # Pull the resolved defaults from the evaluator so --help never drifts from
    # the actual constants (MIN_COST_THRESHOLD / COST_PER_ERROR).
    from evaluator import COST_PER_ERROR, MIN_COST_THRESHOLD, _fmt_cost

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

    p.add_argument("--cost-threshold", type=float, default=None,
                   help="Mean agent cost across an iteration's batch below "
                        "this is in the free zone (no penalty). Judge cost "
                        f"never counts. Default {_fmt_cost(MIN_COST_THRESHOLD)}."
                        "%(default).0s")
    p.add_argument("--cost-per-error", type=float, default=None,
                   help="Dollars of mean batch spend (over --cost-threshold) "
                        "that equals one fully-wrong query of penalty. "
                        f"Default {_fmt_cost(COST_PER_ERROR)}. See README "
                        "'Cost-penalty math'."
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
    p.add_argument("--cap-judge-to-estimate", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Judge only the top-estimate (recall depth) submitted "
                        "papers instead of all of them — in training AND in "
                        "internal test evals (--eval-test-set / --eval-only). "
                        "On by default; cuts judge cost with no measured score "
                        "change (recall reads only the top-estimate; the rank "
                        "term is empirically unaffected, though capped test "
                        "scores are a slightly different basis than official "
                        "uncapped scoring — the mode is recorded in "
                        "test_results.json). For training it is run-immutable "
                        "like --cost-threshold: locked for the run's lifetime, "
                        "so resume keeps the original setting; on --eval-only "
                        "the flag applies at eval time instead. Official "
                        "astabench submissions are always uncapped. "
                        "--no-cap-judge-to-estimate judges the full submission, "
                        "as official does."
                        "%(default).0s")
    p.add_argument("--no-shared-judge-cache", action="store_true",
                   help="Isolate the judge verdict cache per run/invocation "
                        "instead of sharing it across runs. Sharing is sound "
                        "because verdicts are keyed by (query, paper, "
                        "evidence-hash) and scoped by judge model, so cross-run "
                        "reuse only recovers identical judge inputs. Applies to "
                        "training (shared_<judge>.json) and to internal test "
                        "evals (shared_test_<judge>.json — a separate file, so "
                        "test verdicts only ever come from test evals); for "
                        "test evals this flag forces a pristine per-invocation "
                        "cache, i.e. submission-exact fresh judging (e.g. a "
                        "clean cost measurement)."
                        "%(default).0s")
    # Judge-model option. History: agreement with GPT-4o is evidence-style-
    # dependent and must be re-measured per lineage with
    # _check_judge_calibration.py (n=150, untruncated evidence).
    #   gpt-5.4-mini (2026-07-17): FAIL — kappa 0.63 vs the 0.7 gate, +24%
    #     Perfect-rate inflation on mature snippet-rich evidence.
    #   gpt-5.4-nano (2026-07-20): FAIL — kappa ~0.52, severe deflation
    #     (credited 51% of GPT-4o's Perfects).
    #   gpt-5.6-luna (2026-07-20): PASS — kappa 0.755, Perfect rates 31.3%
    #     vs 32.7%, 2/300 format repairs. First approved alternate.
    # The strict-parser blocker is now shipped: the evaluator installs
    # _judge_normalize's lenient extractor whenever a non-stock judge is
    # active (stock paths stay strict for official parity). Choices are
    # pinned to evaluator.JUDGE_MODEL_IDS by a unit test.
    p.add_argument("--training-judge", type=str, default=None,
                   choices=TRAINING_JUDGE_CHOICES,
                   help="Relevance-judge model for training AND (if passed "
                        "with --eval-test-set/--eval-only) internal test "
                        "evals. Default: stock GPT-4o — the official judge "
                        "and the only basis comparable to leaderboard "
                        "scores. gpt-5.6-luna passed the calibration gate "
                        "(kappa 0.755, 2026-07-20; ~2x cheaper — see README "
                        "'Training judge'). Non-stock test results are "
                        "written to judge-suffixed files. Official "
                        "submissions always use stock GPT-4o regardless. "
                        "Each judge has its own verdict-cache namespace.")
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
        CACHE_PATH_ENV,
        CAP_JUDGE_ENV,
        COST_PER_ERROR,
        MIN_COST_THRESHOLD,
        PaperFinderEvaluator,
        TRAINING_GRADER_ENV,
        _fmt_cost,
        load_paper_finder,
    )
    # The stock grader in effect when no --training-judge override is set; used
    # to scope the shared judge cache by judge model.
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME as _STOCK_GRADER

    def _set_training_cache_env() -> str:
        """Point the judge cache at a training cache file, never astabench's
        package global.

        Verdicts are keyed by (query, paper, evidence-hash), which is a sound
        judge input, so reuse across runs is legitimate — a verdict is only
        reused when the identical grounded evidence recurs. So the default is a
        cache SHARED across runs (recovering cross-run reuse the old bare-pid
        key gave unsoundly), scoped by judge model: the model is in the filename
        so a gpt-4o run and a gpt-5.4-mini run can never read each other's
        verdicts (a verdict is a function of the judge, not just the input).

        --no-shared-judge-cache falls back to per-run isolation: derived from
        the resumed dir on resume (so a resumed run keeps its verdicts), else a
        fresh per-invocation file. Subprocess workers inherit the env var, so
        all workers of a run share one file within the run either way."""
        # Resolved run-immutable value (persisted in paper_finder_runtime),
        # NOT the raw CLI flag — a resume restores the run's original judge.
        judge = training_judge
        slug = _judge_slug(judge)
        cache_dir = _judge_cache_dir(args.runs_dir)
        if not args.no_shared_judge_cache:
            path = cache_dir / f"shared_{slug}.json"
            scope = "shared"
        elif args.resume:
            path = Path(args.resume) / f"judge_cache_{slug}.json"
            scope = "per-run/resume"
        else:
            path = cache_dir / f"run_{slug}_{uuid.uuid4().hex}.json"
            scope = "per-run"
        os.environ[CACHE_PATH_ENV] = str(path)
        logger.info(f"Judge cache ({scope}, judge={judge}): {path}")
        # Top-estimate judging cap (training-only cost saver). Uses the
        # run-immutable resolved value, so a resume applies the run's original
        # setting rather than this launch's flag.
        if cap_judge_to_estimate:
            os.environ[CAP_JUDGE_ENV] = "1"
            logger.info("Judging capped to top-estimate per semantic query")
        else:
            os.environ.pop(CAP_JUDGE_ENV, None)
        # Opt-in cheaper training judge. Compared against the stock id, not
        # flag truthiness: an EXPLICIT --training-judge gpt-4o must behave
        # exactly like the default — no override env, no lenient normalizer
        # (strict-parser parity with official scoring).
        if judge != _STOCK_GRADER:
            os.environ[TRAINING_GRADER_ENV] = judge
            logger.info(f"Training relevance judge: {judge}")
        else:
            os.environ.pop(TRAINING_GRADER_ENV, None)
        return str(path)

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
    # The top-estimate judging cap changes the training scoring basis (how many
    # papers the rank term sees), so like the cost knobs it must be immutable
    # across a run — a mid-run flip would make later iterations' Elo
    # incomparable. Resolved and persisted the same way; resume keeps the
    # original setting unless a disagreeing flag is passed (a hard error).
    cap_judge_to_estimate = _enforce_immutable_on_resume(
        cli_value=args.cap_judge_to_estimate,
        stored_value=checkpoint_pfb.get("cap_judge_to_estimate"),
        default_value=True,
        name="cap-judge-to-estimate",
        on_resume=on_resume,
        fmt=str,
    )
    # The training judge changes the training scoring basis (whose verdicts
    # Elo is computed on) AND the verdict-cache namespace, so it is
    # run-immutable like the knobs above — a flagless --resume silently
    # flipping a luna-trained campaign back to stock GPT-4o would
    # contaminate every later iteration's Elo. Stored as the resolved
    # model id (never None). Legacy checkpoints lack the key: resuming one
    # requires stating the judge explicitly once (a one-time bootstrap,
    # locked thereafter) — deliberately never a silent default, because a
    # pre-persistence run may have trained under either judge.
    training_judge = _enforce_immutable_on_resume(
        cli_value=args.training_judge,
        stored_value=checkpoint_pfb.get("training_judge"),
        default_value=_STOCK_GRADER,
        name="training-judge",
        on_resume=on_resume,
        fmt=str,
    )

    # ASTA_TOOL_KEY is validated by the evaluator's constructor preflight
    # (hard-required alongside the three provider keys); fail here first
    # with the same friendly message so a missing key never reaches
    # dataset loading.
    if not os.environ.get("ASTA_TOOL_KEY"):
        raise SystemExit(
            "ASTA_TOOL_KEY is not set. The Asta MCP corpus tools are the "
            "task's only retrieval surface; export ASTA_TOOL_KEY in the "
            "shell that launches the run (see README 'Credentials')."
        )

    resolved_runtime = {
        "cost_threshold": cost_threshold,
        "cost_per_error": cost_per_error,
        "cap_judge_to_estimate": cap_judge_to_estimate,
        "training_judge": training_judge,
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
        apply_cost_penalty=True,  # training: penalty fires
        min_cost_threshold=cost_threshold,
        cost_per_error=cost_per_error,
    )
    test_evaluator = evaluator.with_overrides(apply_cost_penalty=False)

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
        scoring_mode = _set_test_cache_env(
            "eval_only",
            runs_dir=args.runs_dir,
            shared=not args.no_shared_judge_cache,
            cap=cap_judge_to_estimate,
            judge=args.training_judge or _STOCK_GRADER,
            stock=_STOCK_GRADER,
        )
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
        results_filename = _test_results_filename(
            args.eval_agent, args.training_judge or _STOCK_GRADER, _STOCK_GRADER
        )
        summary_path, per_problem_path = _write_test_results(
            eval_result=eval_result,
            evaluator=test_evaluator,
            output_dir=Path(args.resume),
            agent_name=args.eval_agent or "best",
            summary_filename=results_filename,
            scoring_mode=scoring_mode,
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

    # Training judge cache: per-run file, never the package global.
    _set_training_cache_env()

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
            scoring_mode = _set_test_cache_env(
                "test_set",
                runs_dir=args.runs_dir,
                shared=not args.no_shared_judge_cache,
                cap=cap_judge_to_estimate,
                judge=args.training_judge or _STOCK_GRADER,
                stock=_STOCK_GRADER,
            )
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
                summary_filename=_test_results_filename(
                    None, args.training_judge or _STOCK_GRADER, _STOCK_GRADER
                ),
                scoring_mode=scoring_mode,
            )
            logger.info(f"Test summary:    {summary_path}")
            logger.info(f"Test per-problem: {per_problem_path}")


if __name__ == "__main__":
    from RoboPhD.eval_utils import force_exit_if_threads_leaked
    try:
        main()
    finally:
        force_exit_if_threads_leaked()
