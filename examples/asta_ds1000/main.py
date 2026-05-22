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
import ast
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable

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

# Default per-iteration sample size. Single source of truth: referenced
# from _build_dataset's return tuple AND from the argparse help text,
# so changing this constant updates both the run-time default and the
# user-facing default-listed-in-help. 20 (reverted from a 10 experiment):
# DS-1000 is binary-graded, and n=10 produced too many correctness ties
# at the top of the leaderboard for Elo selection to track real accuracy
# gaps — see Appendix B (Selection Noise Analysis) of the RoboPhD paper.
# Overfitting pressure from the higher per-problem hit count is mitigated
# by (a) the reworded objective that no longer rewards narrow
# specialization and (b) the --new-agent-test-rounds default of 0,
# which skips Deep Focus's round-2 refinement eval and keeps each
# agent's per-iteration training exposure comparable to the n=10 /
# 1-round regime. Pass --new-agent-test-rounds 1 to opt back into
# Round 2 (which also activates the Round-2-aware framing in
# objective.md).
DEFAULT_EXAMPLES_PER_ITERATION = 20

# Default iteration cap. Same single-source-of-truth pattern as
# DEFAULT_EXAMPLES_PER_ITERATION above — referenced from
# _build_dataset's return tuple and from the argparse help text via
# f-string interpolation, so changing this constant updates both.
DEFAULT_NUM_ITERATIONS = 20


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

    # Iteration-bounded: examples/iter=DEFAULT_EXAMPLES_PER_ITERATION,
    # num_iterations=DEFAULT_NUM_ITERATIONS → 150 evals at the default.
    # Set evaluation_budget high enough not to bind.
    return train, test, DEFAULT_EXAMPLES_PER_ITERATION, 999_999, DEFAULT_NUM_ITERATIONS


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
    # Head+tail truncation so the tail (where tracebacks carry the real
    # failure line) survives. Keeps the byte budget close to the prior
    # 500-char head-only cap.
    from evaluator import _head_tail_truncate

    def _trunc(s):
        return _head_tail_truncate(s, head=100, tail=400) if s else None

    per_problem = []
    total_agent_cost = 0.0
    n_fallback_used = 0
    diagnostics_list = eval_result.per_example_diagnostics or []
    scores_list = eval_result.per_example_scores or []
    for i, diag in enumerate(diagnostics_list):
        diag = diag or {}
        score = scores_list[i] if i < len(scores_list) else None
        agent_c = diag.get("agent_cost_usd") or 0.0
        total_agent_cost += agent_c
        err = diag.get("error")
        fallback_used = bool(diag.get("fallback_used"))
        if fallback_used:
            n_fallback_used += 1
        primary_error = diag.get("primary_error")
        per_problem.append({
            "sample_id": diag.get("sample_id"),
            "score": score,
            "raw_score": diag.get("raw_score"),
            "cost_penalty": diag.get("cost_penalty"),
            "agent_cost_usd": agent_c,
            "library": diag.get("library"),
            "error": _trunc(err),
            "fallback_used": fallback_used,
            "primary_error": _trunc(primary_error),
        })

    summary_path = output_dir / summary_filename
    n_problems = eval_result.num_examples
    mean_agent_cost = (total_agent_cost / n_problems) if n_problems else 0.0
    with open(summary_path, "w") as f:
        json.dump({
            "agent": agent_name,
            "phase": phase,
            "mean_test_score": eval_result.mean_score,
            "total_test_score": eval_result.total_score,
            "total_test_problems": n_problems,
            "test_eval_cost_usd": evaluator.total_eval_cost,
            "test_eval_agent_cost_usd": total_agent_cost,
            "mean_test_agent_cost_usd": mean_agent_cost,
            "n_fallback_used": n_fallback_used,
            # Empty for the default test path (apply_cost_penalty=False,
            # aggregator returns mean_raw with no annotation). Populated
            # if a future test mode opts into a non-default aggregator.
            "aggregate_explanation": getattr(eval_result, "aggregate_explanation", ""),
        }, f, indent=2)

    per_problem_path = summary_path.with_suffix(".per_problem.json")
    with open(per_problem_path, "w") as f:
        json.dump(per_problem, f, indent=2)

    n_total = len(per_problem)
    if n_total and n_fallback_used:
        # Only emit when fallback actually fired — a clean run shouldn't
        # spend a log line on "Fallback used: 0/N".
        rate = 100.0 * n_fallback_used / n_total
        logger.info(
            f"Fallback used: {n_fallback_used}/{n_total} ({rate:.1f}%)"
        )

    return summary_path, per_problem_path


def _resume_needs_stronger_flag(
    resume_dir: Path, gated_names: Iterable[str]
) -> bool:
    """Return True iff any agent.py in `<resume_dir>/agents/*/` imports
    a gated handle name from `model_registry`.

    Used at resume time to auto-set ASTA_DS1000_ALLOW_STRONGER_MODELS
    when --allow-stronger-models was omitted but the resumed run's
    agent pool needs the gated handles. Without this, eval workers
    crash on import in their per-sample subprocesses and the test eval
    yields a uniform 0.000 score with no surfaced exception.

    AST-based (ImportFrom name match) rather than substring scan, so
    occurrences in comments / docstrings don't trigger false positives.
    The cost asymmetry of false positives is mild (an unused env var
    on a non-stronger run does no harm), but AST is the cleaner
    contract. Files that fail to parse are skipped, not raised.
    """
    agents_dir = resume_dir / "agents"
    if not agents_dir.is_dir():
        return False
    gated_set = set(gated_names)
    for p in agents_dir.glob("*/agent.py"):
        try:
            tree = ast.parse(p.read_text())
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "model_registry"
            ):
                for alias in node.names:
                    if alias.name in gated_set:
                        return True
    return False


def _build_stronger_rows(allow_stronger_models: bool) -> str:
    """Return the three stronger-tier table rows for background.md,
    or empty string when --allow-stronger-models is off.

    Pure function — no I/O, no env-var reads, no global state. Tests
    import this directly and pin the output string, so the row shape
    is verified end-to-end without scanning main.py's source for
    string-literal patterns (which would couple to construction style).

    Row shape mirrors the cheap-tier 5-column shape:
        Handle | Input | Output | Default reasoning_effort | Available overrides

    The Default column uses a unified vocabulary across all 9 handles
    (cheap-tier in background.md, strong-tier here):
        "none"          — no reasoning by default (cheap-tier non-Google)
        `"low"`         — we pinned "low" (Gemini Flash family + Pro Preview)
        model-managed   — reasoning always on, level chosen by provider
                          (GPT-5.5, Opus 4.7)
    Pro Preview shares the Gemini Flash family's reasoning_effort shape
    (default `"low"`, opt-up to `"high"`), so its Available-overrides
    column matches theirs exactly. The prose under the table covers the
    "cannot disable below low" fact for all three Geminis at once.

    No trailing prose caveat — the dollar columns carry cost asymmetry
    and the Default column carries the always-on-reasoning fact. Keeping
    the agent-facing surface free of `temperature` references matches
    the broader direction of recommending only `reasoning_effort` and
    `max_tokens`.
    """
    if not allow_stronger_models:
        return ""
    return (
        '| `GPT_5_5` | 5.00 | 30.00 | model-managed | `"low"`, `"medium"`, `"high"` |\n'
        '| `CLAUDE_OPUS_4_7` | 5.00 | 25.00 | model-managed | `"low"`, `"medium"`, `"high"` |\n'
        '| `GEMINI_3_1_PRO_PREVIEW` | 2.00 | 12.00 | `"low"` | `"low"`, `"high"` |'
    )


def _test_rounds_framing(rounds: int) -> str:
    """Return objective.md paragraph 3's last-sentence framing for the
    given new_agent_test_rounds value.

    Two branches, both written from the evolutionary agent's POV:

    - rounds == 0: the agent only gets one pass per iteration, so we
      can't honestly promise an in-iteration revision opportunity. The
      framing falls back to the long-arc "future iterations" framing
      alone — true but a weaker anti-overfit signal.

    - rounds >= 1: RoboPhD's Deep Focus Round 2 re-evaluates each new
      agent on a fresh batch of training examples within the same
      iteration. The framing tells the agent it'll see results on a
      different batch and have a chance to refine before the agent is
      tested on entirely new batches in future iterations — explicit
      anti-overfit incentive that anchors the agent on generalization.

    The wording avoids RoboPhD-internal jargon ("Round 2", "session")
    and consistently uses "iteration", which is the vocabulary already
    in objective.md and the rest of the agent-facing prompt surface.
    """
    if rounds >= 1:
        return (
            "After you construct your agent, you will see how it "
            "performs on a different batch of examples within this "
            "iteration. You will then have the opportunity to refine "
            "your agent so that it can be tested on entirely new "
            "batches of examples in future iterations."
        )
    return (
        "After you construct your agent, it will be tested on entirely "
        "new batches of examples in future iterations."
    )


def parse_args():
    # ArgumentDefaultsHelpFormatter auto-appends "(default: <value>)" to
    # any flag whose help string does not already contain "%(default)".
    # For most flags this is what we want. To opt OUT for a specific flag
    # (e.g. when default=None is an implementation detail and the
    # behavioral default is described inline), append "%(default).0s" to
    # the help string — the .0s precision truncates the rendered value to
    # empty so nothing extra appears in --help. See --new-agent-test-rounds
    # below for the canonical example.
    p = argparse.ArgumentParser(
        description="Evolve DS-1000 agents on AstaBench (Standard tools, Docker sandbox)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=["experiment", "final"], default="experiment",
                   help="experiment: 90-sample held-out test (~10%% of ds1000_test). "
                        "final: all 900 ds1000_test samples (leaderboard metric).")

    p.add_argument("--engine", choices=["robophd", "gepa", "autoresearch"], default="robophd")
    p.add_argument("--num-iterations", type=int, default=None,
                   help=f"Override the default iteration cap ({DEFAULT_NUM_ITERATIONS})")
    p.add_argument("--examples-per-iteration", type=int, default=None,
                   help=f"Override the default per-iteration sample size ({DEFAULT_EXAMPLES_PER_ITERATION})")
    p.add_argument("--evaluation-budget", type=int, default=None,
                   help="Override the default evaluation budget (iter-bounded)")
    p.add_argument("--new-agent-test-rounds", type=int, default=None,
                   help="Number of Deep-Focus refinement rounds per new agent. "
                        "0 (default) skips Round 2 — each agent is evaluated once "
                        "per iteration. >=1 evaluates each new agent on a fresh "
                        "batch of training examples within the same iteration; "
                        "doubles per-agent training exposure and swaps "
                        "objective.md to the Round-2-aware framing that gives "
                        "the agent extra incentive to avoid overfitting to the "
                        "visible batch."
                        # The %(default).0s trick silences argparse's auto
                        # "(default: None)" append for this flag without
                        # affecting other flags' defaults display:
                        # ArgumentDefaultsHelpFormatter checks for the
                        # substring "%(default)" in the help to decide
                        # whether to auto-append; including it (with .0s
                        # truncating the rendered value to empty) opts out.
                        # User-facing default is 0, stated in-line above —
                        # the implementation-detail None default would just
                        # confuse readers if surfaced here.
                        "%(default).0s")

    p.add_argument("--no-allow-stronger-models", dest="allow_stronger_models",
                   action="store_false", default=True,
                   help="Restrict evolution to the cheap+standard tier. By "
                        "default the three stronger handles (GPT_5_5, "
                        "CLAUDE_OPUS_4_7, GEMINI_3_1_PRO_PREVIEW) are also "
                        "exposed — evolution can use them where they help, "
                        "and the cost-per-error penalty disciplines overuse."
                        # Suppress argparse's auto "(default: True)" suffix.
                        # The dest defaults to True (stronger models on), so
                        # the auto suffix is technically right but reads as
                        # "this flag is on by default" — confusing because
                        # passing the flag *disables* the tier. The behavioral
                        # default is already described in the help text above.
                        # See the --new-agent-test-rounds canonical example.
                        "%(default).0s")
    p.add_argument("--cost-threshold", type=float, default=None,
                   help="Mean cost across an iteration's batch below this "
                        "is in the free zone (no penalty). Default $0.04.")
    p.add_argument("--cost-per-error", type=float, default=None,
                   help="Dollars of mean batch spend (over --cost-threshold) "
                        "that equals one wrong answer of penalty. Default "
                        "$0.01 puts cost in active-pull territory. Set "
                        "large (e.g. $1, $10) for pure-tiebreaker semantics "
                        "where the penalty sorts within accuracy ties but "
                        "never overrides a 1-problem accuracy gap. Applied "
                        "to the iteration aggregate (not per example); "
                        "test-path scores are raw 0/1 fractions regardless.")

    p.add_argument("--max-workers", type=int, default=12,
                   help="Parallel eval workers. Each evaluation runs in its "
                        "own subprocess to bypass inspect.eval's process-global "
                        "singleton lock, so this is real parallelism. 12 fits "
                        "20 examples/iteration in ~2 waves on M-series Macs.")
    p.add_argument("--runs-dir", default="../robophd_runs",
                   help="Root directory for experiment output (default: %(default)s)")
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
                        "evaluate. Requires --eval-only. Defaults to the best-Elo agent. "
                        "Output file is suffixed with the agent name.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--extend", type=int, default=None)
    p.add_argument("--from-iteration", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    # Set the stronger-models env var BEFORE importing evaluator (which
    # transitively triggers model_registry's import-time check in each
    # per-sample subprocess). os.environ writes propagate to children
    # via inheritance — the asta_ds1000 evaluator's subprocess.Popen
    # calls don't pass env=, so the parent env is used.
    #
    # On --resume, also auto-detect: if any agent in the resumed run
    # imports a gated handle, the original training run had
    # --allow-stronger-models, and the resume must too — otherwise the
    # gated handles aren't created in the eval workers and every per-
    # problem subprocess crashes on import for a uniform 0.000 score.
    # Self-healing rather than checkpoint-persisted so historical runs
    # also benefit. The gated-handle list is owned by model_registry
    # (GATED_HANDLE_NAMES) so the registry stays the single source of
    # truth — no hardcoded list in main.py.
    if args.allow_stronger_models:
        os.environ["ASTA_DS1000_ALLOW_STRONGER_MODELS"] = "1"
    elif args.resume:
        from model_registry import GATED_HANDLE_NAMES
        if _resume_needs_stronger_flag(Path(args.resume), GATED_HANDLE_NAMES):
            os.environ["ASTA_DS1000_ALLOW_STRONGER_MODELS"] = "1"
            logger.info(
                "Auto-enabled --allow-stronger-models on resume: "
                "resumed run's agent pool imports a gated handle"
            )

    from evaluator import Ds1000Evaluator

    # Resolve cost-penalty knobs once: CLI override or evaluator default.
    # The same values feed both the evaluator and the markdown
    # interpolation below — keeping them in lockstep means the agent's
    # CLAUDE.md never disagrees with the actual scoring function.
    from evaluator import MIN_COST_THRESHOLD, COST_PER_ERROR
    cost_threshold = (
        args.cost_threshold if args.cost_threshold is not None
        else MIN_COST_THRESHOLD
    )
    cost_per_error = (
        args.cost_per_error if args.cost_per_error is not None
        else COST_PER_ERROR
    )

    def _fmt_cost(x: float) -> str:
        return f"${x:.2f}" if x == round(x, 2) else f"${x:.4f}"

    def _build_cost_penalty_table(threshold: float, cpe: float) -> str:
        """Generate the cost-penalty table substituted into background.md.

        Each row spans 1× cpe of mean cost; the breakeven correct-count
        grows by 1 per row. We stop at 3+ (the "decisive" tier) and
        emit a trailing pattern-continuation row so the agent doesn't
        have to extrapolate from numerics alone.
        """
        rows = [
            "| Mean cost | Effect on score |",
            "|---|---|",
            f"| ≤ {_fmt_cost(threshold)} | No effect on score — two free-zone "
            f"agents with the same raw accuracy score identically, "
            f"regardless of their actual spend |",
            f"| {_fmt_cost(threshold)}–{_fmt_cost(threshold + cpe)} | "
            f"Tiebreaker — lose tied accuracy to a cheaper agent; "
            f"**need 1+ more correct** to win |",
            f"| {_fmt_cost(threshold + cpe)}–{_fmt_cost(threshold + 2*cpe)} | "
            f"**Need 2+ more correct** than a free-zone agent to win |",
            f"| {_fmt_cost(threshold + 2*cpe)}–{_fmt_cost(threshold + 3*cpe)} | "
            f"**Need 3+ more correct** than a free-zone agent to win "
            f"(in practice, a decisive penalty) |",
            f"| … | Each additional {_fmt_cost(cpe)} of mean spend adds 1 "
            f"to the breakeven correct-count |",
        ]
        return "\n".join(rows)

    # Stronger-model rows, conditional on --allow-stronger-models.
    # When the flag is off, the placeholder collapses to empty string
    # and the background.md table stays at six rows; when on, three
    # rows are appended. Construction is delegated to a pure helper
    # so tests can pin the output string without scanning main.py's
    # source for string-literal patterns.
    stronger_rows = _build_stronger_rows(args.allow_stronger_models)

    # Resolve effective new_agent_test_rounds from CLI flag + the
    # --engine-config overlay. The flag is the primary surface; the
    # JSON overlay still wins as the escape hatch (matches the
    # precedence used for engine_overrides below). Resolving it here
    # means the framing in objective.md tracks the actual runtime
    # value — without this, --engine-config '{"new_agent_test_rounds":1}'
    # without the dedicated flag would run Round 2 but still ship the
    # 0-round wording, weakening the anti-overfit prompt.
    #
    # Coerce the JSON-typed value to int at the parse site. JSON has no
    # int type distinct from number, and a stringified value like
    # '{"new_agent_test_rounds":"1"}' would otherwise propagate as a
    # string — silently mis-routing both the framing branch (string vs
    # int comparison) and RoboPhDConfig validation downstream. int(...)
    # accepts numeric values and JSON strings of digits; non-numeric
    # input fails loud here rather than mid-render.
    parsed_engine_config = (
        json.loads(args.engine_config) if args.engine_config else {}
    )
    if "new_agent_test_rounds" in parsed_engine_config:
        parsed_engine_config["new_agent_test_rounds"] = int(
            parsed_engine_config["new_agent_test_rounds"]
        )
    # `args.new_agent_test_rounds` is None when the CLI flag was not
    # set — defaults to 0 here at the read site for the framing path.
    # The None-as-default pattern is intentional: it lets the
    # engine_overrides packing below distinguish "user set this" from
    # "framework default", so --resume doesn't silently clobber the
    # original run's value with the CLI default.
    effective_test_rounds = parsed_engine_config.get(
        "new_agent_test_rounds",
        args.new_agent_test_rounds if args.new_agent_test_rounds is not None else 0,
    )

    test_rounds_framing = _test_rounds_framing(effective_test_rounds)

    # ${STRONGER_MODELS_TABLE_ROWS} matches with its trailing newline so
    # that off-mode collapses the entire line (no stray blank line that
    # would terminate the markdown table early); on-mode puts the rows
    # plus a trailing newline back in.
    stronger_replacement = stronger_rows + "\n" if stronger_rows else ""

    cost_penalty_table = _build_cost_penalty_table(cost_threshold, cost_per_error)

    # Per-example timeout: must match the value passed to RoboPhDConfig
    # below. The evaluator derives a slightly-shorter subprocess_timeout
    # internally so subprocesses get killed BEFORE RoboPhD's reaper
    # would leak the thread. Defined here (above _interpolate) because
    # the ${EVAL_TIMEOUT_MIN} substitution below derives its value from
    # it and _interpolate is invoked a few lines down.
    #
    # 30-minute cap. Wall-clock time isn't a leaderboard criterion;
    # this is just a catch for runaway processes. The evaluator SIGKILLs
    # the per-problem subprocess at 1770s (= 29 min) — EVAL_TIMEOUT minus
    # the 30s reaper buffer (see evaluator.py:564, subprocess_timeout =
    # max(eval_timeout - 30, 60)) — so the agent's true budget is 29 min.
    # Raised from 20→30 min: a 3-model serial fan-out blew the old 1170s
    # cap on 8/20 problems, scored a misleading 60%, and the evolution
    # lineage misattributed the latency death as a reasoning regression.
    # 30 min keeps it generous enough to not be the binding constraint
    # while still catching genuine runaways.
    EVAL_TIMEOUT = 1800

    def _interpolate(text: str) -> str:
        return (
            text
            .replace("${COST_PENALTY_TABLE}", cost_penalty_table)
            .replace("${COST_THRESHOLD}", _fmt_cost(cost_threshold))
            .replace("${COST_PER_ERROR}", _fmt_cost(cost_per_error))
            .replace("${STRONGER_MODELS_TABLE_ROWS}\n", stronger_replacement)
            .replace("${TEST_ROUNDS_FRAMING}", test_rounds_framing)
            # True per-problem budget the agent experiences: EVAL_TIMEOUT
            # minus the 30s reaper buffer, floored to whole minutes
            # (1770 // 60 = 29). Floored & buffer-aware so the doc never
            # over-promises the wall-clock the agent actually gets.
            .replace("${EVAL_TIMEOUT_MIN}", str((EVAL_TIMEOUT - 30) // 60))
        )

    objective = _interpolate((HERE / "objective.md").read_text().strip())
    background = _interpolate((HERE / "background.md").read_text().strip())

    # Seed is loaded early so it can plumb into the test evaluator as
    # `fallback_candidate` below. Training keeps fallback_candidate=None
    # so error signals stay visible to the evolution loop; test/inference
    # opts in so a single primary failure doesn't tank that example's
    # score on the leaderboard or in --eval-test-set / --eval-only runs.
    seed = {"agent.py": (HERE / "seeds" / "baseline" / "agent.py").read_text()}

    # Two evaluator instances. Training applies the bounded cost penalty
    # (a tiebreaker between correctness-tied agents); test paths report
    # raw 0/1 so evolved agents land at their true point on the Pareto
    # cost-vs-score curve.
    evaluator = Ds1000Evaluator(
        eval_timeout=EVAL_TIMEOUT,
        apply_cost_penalty=True,  # training: penalty fires
        min_cost_threshold=cost_threshold,
        cost_per_error=cost_per_error,
    )
    test_evaluator = evaluator.with_overrides(
        apply_cost_penalty=False,
        fallback_candidate=seed,
    )

    # Single source of truth for test-side eval config. Reused at every
    # eval_candidate / eval_run call site below (--eval-only and
    # --eval-test-set paths) so the test pipeline can't silently drift
    # from the training pipeline's eval_timeout. Future test-side knobs
    # (max_workers, test_repeats, ...) belong on this object.
    test_eval_config = RoboPhDEvalConfig(eval_timeout=EVAL_TIMEOUT)

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
            eval_result = eval_candidate(
                evaluator=test_evaluator,
                dataset=test,
                candidate=candidate,
                config=test_eval_config,
            )
        else:
            eval_result = eval_run(
                evaluator=test_evaluator,
                dataset=test,
                experiment_dir=args.resume,
                config=test_eval_config,
            )

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

    num_iterations = args.num_iterations if args.num_iterations is not None else default_iterations
    evaluation_budget = args.evaluation_budget if args.evaluation_budget is not None else default_budget

    # Build engine_overrides. Two principles:
    #
    # (1) User-explicit CLI values always propagate (this iteration of
    #     resume, or initial run — same behavior).
    # (2) On --resume with NO user value, don't pack anything: the
    #     original run's setting must survive. api.py:327 reapplies
    #     engine_overrides as a delta on the resume iteration, so any
    #     CLI default packed here would silently clobber it.
    # (3) On INITIAL RUN with no user value, pack DS-1000's task-
    #     specific default if it differs from RoboPhD's framework
    #     default (config_manager.py:get_defaults). Otherwise the
    #     task's intent is silently lost to the framework default.
    #
    # examples_per_iteration: DS-1000 default (20) == RoboPhD default
    # (20), so omitting it is safe — the framework default lands at
    # the right value anyway. Only pack when the user explicitly set
    # something different.
    #
    # new_agent_test_rounds: DS-1000 wants 0 (Round 2 disabled by
    # default to keep training exposure comparable to the n=10 +
    # 1-round regime — see DEFAULT_EXAMPLES_PER_ITERATION comment).
    # RoboPhD's framework default is 1. So on initial runs we MUST
    # pack the task default; only on resume do we omit it.
    #
    # parsed_engine_config was already loaded above for
    # effective_test_rounds; reuse it to avoid parsing JSON twice.
    is_resume = args.resume is not None
    engine_overrides: dict = {}
    if args.examples_per_iteration is not None:
        engine_overrides["examples_per_iteration"] = args.examples_per_iteration
    if args.new_agent_test_rounds is not None:
        engine_overrides["new_agent_test_rounds"] = args.new_agent_test_rounds
    elif not is_resume:
        # Initial run with no user-set value: stamp DS-1000's task
        # default into iteration 1's config so it persists into the
        # checkpoint and is inherited by future iterations.
        engine_overrides["new_agent_test_rounds"] = 0
    engine_overrides.update(parsed_engine_config)

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
    logger.info(f"Best agent: Elo {result.best_score:.0f}")
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
                config=test_eval_config,
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
