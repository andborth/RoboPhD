"""
PaperFindingBench evaluator for AstaBench (Standard tools tier).

Each candidate is an Inspect `@solver` factory exported as `make_solver` in
agent.py. The evaluator runs the candidate against one PaperFindingBench
sample at a time via `inspect.eval()` with a 1-sample dataset, attaches the
Asta MCP corpus tools (ASTA_TOOL_KEY required — the Standard tier's only
sanctioned retrieval surface), and reports the scorer's adjusted-F1 score
along with cost/usage diagnostics.

Cost model: agents call LLMs only through model_registry handles; their
spend is billed to `agent_cost_usd` and (during training) penalized above
a mean-cost threshold via `aggregate()`. The benchmark's own GPT-4o
relevance judge — which the scorer invokes per predicted paper on
semantic queries — runs inside the same inspect.eval, so its usage
appears in the same log. It is split out by model ID into
`other_cost_usd`: reported for the audit trail, never penalized, because
the agent does not control the judge.

Bypassing inspect.eval() and constructing TaskState manually was considered
for lower per-call latency but rejected: AstaBench's submission tarball
machinery, cost accounting, and the scorer's get_llm_relevance for semantic
queries all assume an Inspect-driven runtime. Profile first; switch only if
the loop is the bottleneck.
"""

import importlib.util
import io
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import use_tools

from astabench.evals.paper_finder.paper_finder_utils import (
    detailed_reference_path,
    get_inserted_before_per_dataset_type,
)
from astabench.evals.paper_finder.relevance import (
    GRADER_MODEL_NAME as _ASTABENCH_GRADER_MODEL_NAME,
)
from astabench.evals.paper_finder.task import (
    normalize_corpus_id,
    pf_final_score_name_per_type,
    score_paper_finder_with_all_name,
)

logger = logging.getLogger(__name__)

# Tracks whether we've already warned about a model litellm couldn't price.
# Once-per-process to avoid log spam across hundreds of evals.
_unpriced_models_warned: set[str] = set()
_live_map_fallback_warned: set[str] = set()

_BUNDLED_PRICE_MAP: dict | None = None

# Model IDs whose usage belongs to the BENCHMARK, not the agent: the
# scorer's relevance judge for semantic queries. Split out of
# agent_cost_usd into other_cost_usd (reported, never penalized).
#
# The split is an exact string match on the inspect model_usage key, and
# it is sound only while (a) this set matches the judge astabench
# actually calls, and (b) no model_registry handle resolves to a judge
# ID. (a) is enforced by the module-load assert below — if astabench
# bumps its grader, this import fails loudly instead of silently
# billing judge spend to agents. (b) is enforced by a unit test
# (unit_tests/test_model_registry.py).
JUDGE_MODEL_IDS = frozenset({"openai/gpt-4o-2024-11-20"})
if _ASTABENCH_GRADER_MODEL_NAME not in JUDGE_MODEL_IDS:
    raise RuntimeError(
        f"astabench's paper_finder relevance judge is "
        f"{_ASTABENCH_GRADER_MODEL_NAME!r}, which is not in this "
        f"evaluator's JUDGE_MODEL_IDS {sorted(JUDGE_MODEL_IDS)}. The "
        f"agent-vs-judge cost split would misattribute judge spend to "
        f"agents. Update JUDGE_MODEL_IDS to include the new grader "
        f"(and re-check that no model_registry handle uses that ID)."
    )


def _bundled_price_map() -> dict:
    """litellm's BUNDLED price map — the leaderboard's cost basis.

    `astabench score` runs with LITELLM_LOCAL_MODEL_COST_MAP=True, which
    makes litellm price from the JSON snapshot shipped inside the
    installed package rather than the live remote map. We load that same
    snapshot directly (env-var-free: the flag only takes effect at
    litellm import time, which may already have happened) so internal
    costs match what the leaderboard will bill. Empty dict if litellm
    isn't installed or the snapshot can't be read — callers fall back
    to litellm.cost_per_token.
    """
    global _BUNDLED_PRICE_MAP
    if _BUNDLED_PRICE_MAP is None:
        try:
            import json as _json
            from pathlib import Path as _Path
            import litellm as _litellm
            snapshot = (
                _Path(_litellm.__file__).parent
                / "model_prices_and_context_window_backup.json"
            )
            _BUNDLED_PRICE_MAP = _json.loads(snapshot.read_text())
        except Exception as exc:
            _BUNDLED_PRICE_MAP = {}
            # Loud one-shot (cached {} skips this branch on later calls):
            # without the snapshot every model silently reprices on the
            # LIVE map — reintroducing the internal-vs-official basis
            # drift this loader exists to prevent (see _estimate_cost).
            logger.warning(
                "Could not load litellm's bundled price snapshot (%s). "
                "ALL internal costs will fall back to the live price map, "
                "which can diverge from the leaderboard's billing basis "
                "(e.g. gemini-3.1-flash-lite: live $0.25/$1.50 vs bundled "
                "$0.45/$2.70 per M). Fix the snapshot path/litellm install "
                "before trusting cost-penalized results.",
                exc,
            )
    return _BUNDLED_PRICE_MAP


def _head_tail_truncate(s: str, head: int = 200, tail: int = 1500) -> str:
    """Truncate `s` to `head` + marker + `tail` chars when too long.

    Inspect-AI wraps provider errors as f"\\nRequest:\\n{request}\\n\\n{error}"
    (_model.py:1007). The load-bearing upstream message is at the END;
    request JSON in the middle is typically large but uninformative.
    A flat `s[:N]` slice would land entirely inside the request body and
    hide every distinct failure mode behind an opaque tail-less string.
    Head + tail preserves both the "Request:" preamble (for context) and
    the actual provider response.

    The `head + tail + 50` short-string threshold: when `s` is only
    slightly longer than `head + tail`, truncating would replace those
    few extra chars with a marker that's actually longer than what we'd
    save. The marker `f"\\n... (N chars truncated) ...\\n"` is ~27 chars
    of fixed text plus 1-7 digits for N — call it ~28-35 chars total.
    Break-even sits around `len(s) = head + tail + 30`, where the
    truncation saves zero or one char. The `+ 50` cutoff gives a ~15-char
    safety margin past break-even so we only truncate when the savings
    is meaningful (~15+ chars). Pure: no I/O, no global state, no
    dependencies beyond `len()` and string slicing.
    """
    if len(s) <= head + tail + 50:
        return s
    truncated = len(s) - head - tail
    return s[:head] + f"\n... ({truncated} chars truncated) ...\n" + s[-tail:]


# Iteration-level score during training (apply_cost_penalty=True), via
# PaperFinderEvaluator.aggregate:
#   score = SCORE_SCALE * mean_raw - penalty_pts
#   errors_equivalent = max(0, mean_cost - MIN_COST_THRESHOLD) / COST_PER_ERROR
#   penalty_pts = errors_equivalent * (SCORE_SCALE / n)
# mean_raw is the mean of per-example F1 scores (continuous in [0, 1]:
# adjusted F1 on semantic queries, standard F1 on specific/metadata).
# n is the batch size. The penalty applies to the MEAN agent cost across
# the iteration's batch, not per example — matches the leaderboard's
# mean-cost Pareto framing and rewards selective routing (expensive
# models on hard queries, cheap on easy) as long as the batch average
# stays in the free zone. One error_equivalent of penalty equals one
# fully-wrong query of raw score (SCORE_SCALE/n points), so the penalty
# is expressible in the same units the agent already cares about. Only
# agent spend counts — the benchmark's relevance-judge cost
# (other_cost_usd) is excluded. Score is unbounded below. Test-path
# scores (apply_cost_penalty=False) skip SCORE_SCALE entirely and report
# mean_raw as a [0, 1] fraction — leaderboard parity.
SCORE_SCALE = 100.0
# Free-zone width $0.10: the Standard-tier leaderboard reference (generic
# ReAct + GPT-5 Mini) bills ~$0.06/query, and competitive paper-finding
# pipelines are evidence-heavy (query decomposition, snippet extraction,
# per-candidate filtering). A tighter zone would penalize the most
# promising evolution direction; $0.10 covers the whole mini/flash tier
# with margin while still decisively penalizing Opus-tier ReAct
# ($1.49-$3.38/query on the leaderboard).
MIN_COST_THRESHOLD = 0.10
# Dollars of mean batch spend (over threshold) that equals one
# fully-wrong query of penalty. PFB scores are continuous F1 (typical
# per-query 0.2-0.4), so one error-equivalent is a LARGER unit than in a
# binary benchmark — $0.02 (half DS-1000's slope) keeps cost an active
# pull without collapsing evolution into degenerate zero-LLM agents.
# At n=20 a $0.30-mean Sonnet-tier pipeline pays 50 pts — still
# prohibitive. Set this large (e.g. $1, $10) to recover pure-tiebreaker
# semantics. Score is unbounded below — catastrophically expensive
# agents land well negative, which is intentional.
COST_PER_ERROR = 0.02


def _fmt_cost(x: float) -> str:
    """Format a dollar amount with the fewest decimals that lose nothing.

    Two decimals when exact at two ($0.04, $0.10); otherwise out to four
    with trailing zeros trimmed ($0.044, $0.0425). Single source for both
    audiences the evolution AI must reconcile: the per-eval diagnostics
    here and the interpolated background.md/objective.md docs (main.py
    imports this), so the threshold reads identically everywhere.
    """
    return f"${x:.2f}" if x == round(x, 2) else f"${x:.4f}".rstrip("0")


def _elapsed_seconds(t0: float) -> float:
    """Monotonic wall-clock seconds since ``t0``, 3dp.

    Single source of truth for the ``eval_wall_clock_seconds`` rounding
    precision — used by both the subprocess and non-subprocess eval
    paths so a future precision tweak changes one place, not four.
    """
    return round(time.monotonic() - t0, 3)


def _judge_verdicts_markdown(
    submitted_ids: list[str], query_id: str, known_good: set[str]
) -> str | None:
    """Per-paper judge verdicts for a semantic query, in submitted order.

    Sourced from astabench's persistent judge cache
    (detailed_reference.json), which `update_references` appends to after
    every scoring pass — read the FILE directly, not
    get_normalizer_references(), whose module-global snapshot goes stale
    the moment this eval writes fresh verdicts. Zero extra LLM calls.

    Verdict per submitted paper:
      - in the query's known_to_be_good list → pre-seeded Perfect (the
        scorer never LLM-judges these, so they're absent from the cache)
      - in the cache → the judge's stored label, verbatim
      - neither → "(no verdict recorded)" (judge parse failure, or the
        scorer didn't run)

    Returns None when there's nothing to report (no submissions, or the
    cache is unreadable and nothing is known-good).
    """
    if not submitted_ids:
        return None
    try:
        with open(detailed_reference_path) as f:
            cache = json.load(f).get(query_id) or {}
    except (OSError, json.JSONDecodeError):
        cache = {}

    # The cache stores astabench's internal label strings
    # ("perfectly_relevant_papers", ...); translate to the human labels.
    pretty = {
        "perfectly_relevant_papers": "Perfectly Relevant",
        "highly_relevant_papers": "Highly Relevant",
        "somewhat_relevant_papers": "Somewhat Relevant",
        "not_relevant_papers": "Not Relevant",
    }
    perfect_raw = "perfectly_relevant_papers"

    lines = []
    n_perfect = n_lower = n_unknown = 0
    for i, pid in enumerate(submitted_ids[:250], 1):  # scorer reads first 250
        if pid in known_good:
            verdict = "Perfectly Relevant (known-good)"
            n_perfect += 1
        elif pid in cache:
            raw = str(cache[pid])
            verdict = pretty.get(raw, raw)
            if raw == perfect_raw:
                n_perfect += 1
            else:
                n_lower += 1
        else:
            verdict = "(no verdict recorded)"
            n_unknown += 1
        lines.append(f"{i}. {pid} — {verdict}")
    n_submitted = len(lines)
    lines.append(
        f"\n{n_perfect} Perfect / {n_lower} lower / {n_unknown} no verdict, "
        f"of {n_submitted} submitted"
    )
    return "\n".join(lines)


def load_paper_finder(split: str = "validation") -> list[Sample]:
    """Load the PaperFindingBench split as a list of Inspect Samples.

    Thin wrapper around astabench's loader. `split` is "validation" (66) or
    "test" (267). Litqa2 splits are also accepted by the loader but are out
    of scope for this example.
    """
    from astabench.evals.paper_finder.task import load_paper_finder_bench
    ds = load_paper_finder_bench(split)
    return list(ds)


def _build_tools(sample_id: str):
    """Return the Asta MCP corpus tools to attach for this sample.

    The factory applies the sample's date cutoff (no future-paper leaks)
    and wraps every tool in astabench's make_retry_wrapper (429/529/504 +
    BrokenResourceError, exponential backoff), so agents never see a
    transient rate limit.
    """
    inserted_before = get_inserted_before_per_dataset_type(sample_id)
    from astabench.tools import make_asta_mcp_tools
    return make_asta_mcp_tools(insertion_date=inserted_before)


def _import_candidate_solver(agent_code: str) -> Any:
    """Materialize candidate agent.py into a temp module and return its
    @solver factory function (must be exported as `make_solver`).

    Cleans up both the temp file and the sys.modules entry after the
    factory is captured. Function.__globals__ keeps a direct reference to
    the module's __dict__ so the factory remains executable after the
    sys.modules cleanup.
    """
    mod_name = f"_paper_finder_candidate_{uuid.uuid4().hex}"
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(agent_code)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
            if not hasattr(mod, "make_solver"):
                raise RuntimeError(
                    "candidate agent.py must define a function named "
                    "`make_solver` decorated with @solver "
                    "(see seeds/baseline/agent.py)"
                )
            return mod.make_solver
        finally:
            sys.modules.pop(mod_name, None)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class PaperFinderEvaluator:
    """RoboPhD evaluator for PaperFindingBench.

    Contract: __call__(candidate, example, *, problem_dir=None) -> (score, diagnostics).

    `candidate` is a {"agent.py": "<source>"} mapping (RoboPhD's standard
    file_mapping shape).
    `example` is an Inspect Sample or its dict form (Sample.model_dump()) —
    main.py converts before calling so RoboPhD's domain layer can SHA256
    stable IDs.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        subprocess_isolation: bool = True,
        eval_timeout: int = 600,
        apply_cost_penalty: bool = True,
        min_cost_threshold: float = MIN_COST_THRESHOLD,
        cost_per_error: float = COST_PER_ERROR,
    ):
        # Hard requirement: every provider key the registry references
        # must be set, even if the seed only uses one. Evolution can
        # produce an agent that calls Claude or Gemini at any iteration,
        # and the failure mode if the key is missing is a 401 mid-run
        # (worst time to discover the gap). Fail loudly at startup
        # instead. OPENAI_API_KEY is doubly required: the benchmark's
        # relevance judge (GPT-4o) runs inside every semantic-query
        # scoring pass regardless of which models the agent uses.
        # ANTHROPIC accepts either ANTHROPIC_API_KEY or
        # ANTHROPIC_API_KEY_FOR_ROBOPHD per RoboPhD convention; the
        # FOR_ROBOPHD variant lets the user's Claude Code CLI keep
        # using its own subscription credentials while RoboPhD uses a
        # separate API key (see model_registry.py).
        missing = []
        if not os.environ.get("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if not (os.environ.get("ANTHROPIC_API_KEY") or
                os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")):
            missing.append("ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_FOR_ROBOPHD)")
        if not os.environ.get("GOOGLE_API_KEY"):
            missing.append("GOOGLE_API_KEY")
        if not os.environ.get("ASTA_TOOL_KEY"):
            missing.append("ASTA_TOOL_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required env vars: {', '.join(missing)}. "
                f"PaperFindingBench evolution may pick any of nine solver "
                f"models across three providers (OpenAI, Anthropic, "
                f"Google); see model_registry.py for the full handle "
                f"list. OPENAI_API_KEY additionally powers the "
                f"benchmark's GPT-4o relevance judge, and ASTA_TOOL_KEY "
                f"is the Asta MCP corpus tools' API key — the Standard "
                f"tier's only retrieval surface. All must be set in the "
                f"shell that launches the run."
            )

        # Resolve registry handles at construction time (after the env-var
        # check so the friendly error fires first if keys are missing).
        # Storing self._default_model surfaces any registry-resolution
        # failure here rather than waiting for the first .generate()
        # inside inspect.eval. inspect.eval() requires a model argument
        # even though PFB agents call registry handles directly.
        from model_registry import GPT_5_4_MINI as _DEFAULT_MODEL
        self._default_model = _DEFAULT_MODEL

        self.subprocess_isolation = subprocess_isolation
        # True for training (RoboPhD's Elo competition): the iteration
        # aggregator (PaperFinderEvaluator.aggregate) returns SCORE_SCALE
        # * mean_raw_f1 - penalty(mean_agent_cost). False for test paths
        # (eval_candidate / eval_run / --eval-test-set / --eval-only):
        # aggregate returns mean_raw_f1 as a [0, 1] fraction with no
        # penalty so the agent lands at its true point on the Pareto
        # cost-vs-score curve. Per-example scores stay raw F1 in both
        # modes; agent_cost_usd and other_cost_usd are recorded in
        # diagnostics in both modes for the audit trail.
        self.apply_cost_penalty = apply_cost_penalty
        # Catch the misconfiguration class up-front: cost_per_error <= 0
        # would either divide by zero or sign-flip the penalty. Negative
        # threshold would mean every agent pays a penalty.
        if min_cost_threshold < 0:
            raise ValueError(
                f"min_cost_threshold must be >= 0; got {min_cost_threshold}"
            )
        if cost_per_error <= 0:
            raise ValueError(
                f"cost_per_error must be > 0; got {cost_per_error}"
            )
        self.min_cost_threshold = min_cost_threshold
        self.cost_per_error = cost_per_error

        # Subprocess kill-after timeout MUST be less than RoboPhD's
        # eval_timeout. RoboPhD's reaper writes "EVAL TIMEOUT" and scores
        # 0, but cannot interrupt Python threads — the thread keeps
        # blocking until our subprocess_timeout fires. If
        # subprocess_timeout > eval_timeout, the gap leaves MCP/judge-LLM
        # resources tied up after RoboPhD gave up. Buffer of 30s gives
        # Python time to SIGKILL the subprocess and the thread to return
        # cleanly.
        self.eval_timeout = eval_timeout
        self.subprocess_timeout = max(eval_timeout - 30, 60)

        # inspect.eval() insists on a log dir; use a per-evaluator temp dir
        # so multiple parallel evaluators don't fight over the same path.
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="paper_finder_eval_")
        self.total_eval_cost = 0.0     # agent spend only (penalized bucket)
        self.total_judge_cost = 0.0    # relevance-judge spend (audit only)
        self._cost_lock = threading.Lock()

    # -- Construction helpers -------------------------------------------------

    def with_overrides(self, **overrides: Any) -> "PaperFinderEvaluator":
        """Return a sibling evaluator inheriting this one's config.

        Used by main.py to derive the test evaluator from the training
        one with `apply_cost_penalty=False`. Any future constructor
        field added to PaperFinderEvaluator must also be added here.
        """
        base = {
            "eval_timeout": self.eval_timeout,
            "apply_cost_penalty": self.apply_cost_penalty,
            "min_cost_threshold": self.min_cost_threshold,
            "cost_per_error": self.cost_per_error,
            "subprocess_isolation": self.subprocess_isolation,
        }
        base.update(overrides)
        return PaperFinderEvaluator(**base)

    # -- Iteration-level aggregator -------------------------------------------

    def aggregate(self, per_example_results: list[dict]) -> tuple[float, str]:
        """Combine per-example scores into an iteration-level aggregate.

        Per-example `score` is raw F1 (continuous in [0, 1]). This method
        is called once per (agent, iteration) by RoboPhD after all
        per-example evals have completed (or been read from cache),
        and its scalar return becomes the agent's `average_score` for
        the iteration — what Elo compares.

        Two scale regimes:

        - **Training** (`apply_cost_penalty=True`): scale raw mean F1
          to a percentage (SCORE_SCALE × mean_raw), then subtract an
          error-equivalent cost penalty when the batch's *mean* agent
          cost exceeds `min_cost_threshold`. One "error" of penalty
          equals SCORE_SCALE/n score points (matching one fully-wrong
          query); the number of errors charged is `(mean_cost -
          threshold) / cost_per_error`. Judge cost (other_cost_usd)
          never enters. Score is unbounded below — a catastrophically
          expensive agent can land well negative, which is intentional.

        - **Test** (`apply_cost_penalty=False`): no scaling, no penalty.
          Returns mean_raw as a [0, 1] fraction — matches leaderboard
          format.

        The explanation string is non-empty in training mode (free-zone
        OR breach) so the report layer surfaces the scale and any
        penalty. In test mode the empty string keeps the default
        single-column report layout.
        """
        if not per_example_results:
            return 0.0, ""
        n = len(per_example_results)
        mean_raw = sum(r.get("score", 0.0) for r in per_example_results) / n

        # Test path: leaderboard-format fraction, no penalty, no
        # explanation (default-aggregator-style return).
        if not self.apply_cost_penalty:
            return mean_raw, ""

        # Training path: report on a percentage scale. Both branches
        # below (free-zone and breach) return on this scale and surface
        # an explanation so readers always know the scale they're
        # looking at — the empty-string fast path is reserved for test
        # mode.
        base = SCORE_SCALE * mean_raw

        # Cost can arrive under either key depending on whether the
        # caller is the training path (domain normalized → eval_cost)
        # or the test path (raw diagnostic → agent_cost_usd). The test
        # path exits above before this read, but the coalesce keeps
        # the aggregator invariant to the call site. Judge spend is
        # never under either key (it's other_cost_usd), so it cannot
        # leak into the penalty.
        def _cost_of(r: dict) -> float:
            return r.get("eval_cost") or r.get("agent_cost_usd") or 0.0
        mean_cost = sum(_cost_of(r) for r in per_example_results) / n

        if mean_cost <= self.min_cost_threshold:
            explanation = (
                f"Mean agent cost ${mean_cost:.4f} within free zone (threshold "
                f"{_fmt_cost(self.min_cost_threshold)}); no penalty applied. "
                f"Raw mean F1 {mean_raw:.4f} reported as percentage: {base:.3f}."
            )
            return base, explanation

        cost_excess = mean_cost - self.min_cost_threshold
        errors_equivalent = cost_excess / self.cost_per_error
        penalty = errors_equivalent * (SCORE_SCALE / n)
        final = base - penalty
        explanation = (
            f"Mean agent cost ${mean_cost:.4f} exceeded threshold {_fmt_cost(self.min_cost_threshold)} "
            f"by ${cost_excess:.4f} = {errors_equivalent:.2f} errors of penalty "
            f"(cost_per_error=${self.cost_per_error:.4f}); subtracted "
            f"{penalty:.3f} score pts from raw {base:.3f} → final {final:.3f} (percentage)."
        )
        return final, explanation

    # -- RoboPhD evaluator contract ------------------------------------------
    # RoboPhD invokes the evaluator object directly:
    #   evaluator(candidate, example, problem_dir=...) -> (score, diagnostics)

    def __call__(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        if self.subprocess_isolation:
            return self._evaluate_via_subprocess(candidate, example)
        return self.evaluate(candidate, example, problem_dir=problem_dir)

    def _evaluate_via_subprocess(self, candidate: dict, example) -> tuple[float, dict]:
        """Run one evaluation in a fresh Python subprocess.

        Inspect-AI's `inspect.eval()` raises if two calls are in flight
        in the same Python process. Subprocess isolation gives us real
        parallelism across RoboPhD's worker threads; each subprocess has
        its own process-global state.

        `subprocess_isolation=False` falls back to in-process execution.
        That path is correct for single-threaded callers (smoke tests,
        the subprocess worker calling itself) but will deadlock/error if
        multiple threads enter it concurrently.
        """
        if isinstance(example, dict):
            example_dict = example
        elif hasattr(example, "model_dump"):
            example_dict = example.model_dump()
        else:
            return 0.0, {
                "error.md": f"_evaluate_via_subprocess expects Sample or dict; got {type(example).__name__}"
            }

        worker_path = Path(__file__).resolve().parent / "_eval_worker.py"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as inf:
            # `default=str` is a defensive belt-and-braces: candidates SHOULD
            # be `{"agent.py": "<source>"}` and examples should round-trip
            # through Sample.model_dump() cleanly. If a future caller passes
            # objects that aren't JSON-native, default=str silently
            # stringifies them so the worker can still parse the JSON.
            # Known fidelity loss; the right fix is to enforce
            # JSON-serializable types at the boundary, not to remove default=str.
            json.dump({
                "candidate": candidate,
                "example": example_dict,
                "apply_cost_penalty": self.apply_cost_penalty,
                "min_cost_threshold": self.min_cost_threshold,
                "cost_per_error": self.cost_per_error,
            }, inf, default=str)
            inf_path = inf.name
        out_path = inf_path + ".out"

        # Note: `subprocess_stderr` in the diagnostic dict is truncated to the
        # last 2000 chars to avoid bloating context for downstream evolution.
        # For deep-traceback debugging, re-run with subprocess_isolation=False
        # to see the full traceback in-process.

        # Wall-clock for this problem's subprocess eval. Recorded into
        # diagnostics → result.json on EVERY return path (success,
        # timeout, non-zero exit, bad payload) so a latency problem is
        # a visible recorded number instead of an invisible cliff
        # (timeouts otherwise surface as score 0 / $0 / no stdout,
        # which the evolution loop has historically misattributed as a
        # reasoning regression rather than "ran out of time").
        _t0 = time.monotonic()

        try:
            proc = subprocess.Popen(
                [sys.executable, str(worker_path), inf_path, out_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # New session so we can killpg on timeout: SIGKILL goes
                # to the worker AND every grandchild it spawned. Plain
                # proc.kill() (what subprocess.run does) only hits the
                # immediate child, leaving orphan grandchildren that
                # can keep our stdout/stderr pipes open and wedge the
                # parent's post-kill drain for hours. POSIX-only;
                # silently ignored on Windows, which the rest of the
                # codebase already assumes we're not running on.
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.subprocess_timeout)
            except subprocess.TimeoutExpired as e:
                # Capture elapsed NOW, before killpg + the bounded 30s
                # post-kill drain. This is ≈ subprocess_timeout — i.e.
                # how long the agent's subprocess actually ran before we
                # killed it — NOT the parent-observed total (which would
                # additionally include up to ~30s of teardown drain and
                # mislead an analyst into thinking the agent ran longer).
                timed_out_after = _elapsed_seconds(_t0)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                # Bounded drain: pipes should EOF immediately once the
                # whole group is dead. If something is somehow still
                # holding the writer end, give up rather than hang.
                # On POSIX, communicate() preserves internal buffers
                # across calls — drain_exc.stdout/stderr is a superset
                # of the outer e.stdout/e.stderr (everything captured
                # up through the second timeout), so it's the right
                # source to prefer.
                try:
                    stdout, stderr = proc.communicate(timeout=30)
                except subprocess.TimeoutExpired as drain_exc:
                    proc.kill()
                    stdout = drain_exc.stdout or e.stdout or ""
                    stderr = drain_exc.stderr or e.stderr or ""
                    if not stderr:
                        stderr = "<post-kill drain timed out>"
                return 0.0, {
                    "error.md": f"subprocess timed out after {self.subprocess_timeout}s",
                    "subprocess_stderr": (stderr or "")[-2000:],
                    "eval_wall_clock_seconds": timed_out_after,
                }
            if proc.returncode != 0:
                return 0.0, {
                    "error.md": f"subprocess failed (exit {proc.returncode})",
                    "subprocess_stderr": (stderr or "")[-2000:],
                    "eval_wall_clock_seconds": _elapsed_seconds(_t0),
                }
            try:
                with open(out_path) as f:
                    payload = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                return 0.0, {
                    "error.md": f"subprocess produced no valid output: {type(e).__name__}: {e}",
                    "subprocess_stderr": (stderr or "")[-2000:],
                    "eval_wall_clock_seconds": _elapsed_seconds(_t0),
                }

            score = float(payload.get("score", 0.0))
            diagnostics = payload.get("diagnostics", {}) or {}
            diagnostics["eval_wall_clock_seconds"] = _elapsed_seconds(_t0)
            agent_only_cost = diagnostics.get("agent_cost_usd", 0.0) or 0.0
            judge_cost = diagnostics.get("other_cost_usd", 0.0) or 0.0
            with self._cost_lock:
                self.total_eval_cost += agent_only_cost
                self.total_judge_cost += judge_cost
            return score, diagnostics
        finally:
            for p in (inf_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def evaluate(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        _t0 = time.monotonic()
        agent_code = candidate.get("agent.py", "")
        if not agent_code:
            return 0.0, {"error.md": "candidate missing agent.py"}

        # RoboPhD's domain layer wants JSON-serializable examples (it
        # SHA256s them for stable IDs), so main.py converts Sample to
        # dict via .model_dump() before passing to optimize_anything().
        # Reconstruct here, and fail loud on anything else (a programmatic
        # caller bypassing main.py would otherwise hit a deep AttributeError
        # later instead of seeing the boundary mismatch).
        if isinstance(example, dict):
            example = Sample(**example)
        elif not isinstance(example, Sample):
            raise TypeError(
                f"evaluate() expects Sample or dict (from Sample.model_dump); "
                f"got {type(example).__name__}"
            )

        # Note: PaperFinder does NOT strip any fields from
        # example.metadata before passing to the solver. AstaBench's
        # json_to_sample for paper_finder puts {query, score_type,
        # raw_query} into metadata, and a leaderboard agent would receive
        # all three. score_type in particular is a legitimate input —
        # agents should branch on specific/metadata/semantic queries
        # because they call for different retrieval strategies (narrow
        # lookup vs author-filter vs broad retrieval + evidence).

        try:
            solver_factory = _import_candidate_solver(agent_code)
        except Exception as e:
            return 0.0, {
                "error.md": f"candidate import failed: {type(e).__name__}: {e}",
                "eval_wall_clock_seconds": _elapsed_seconds(_t0),
            }

        try:
            tools = _build_tools(example.id)
        except Exception as e:
            return 0.0, {
                "error.md": f"tool init failed: {type(e).__name__}: {e}",
                "eval_wall_clock_seconds": _elapsed_seconds(_t0),
            }

        # Build a 1-sample Task wrapping just this example.
        # The scorer factory wraps score_paper_finder via @scorer; calling
        # it with () instantiates the Scorer that Task expects.
        scorer = score_paper_finder_with_all_name(
            pf_final_score_name_per_type["paper_finder_bench"]
        )()
        task = Task(
            dataset=MemoryDataset([example]),
            solver=solver_factory(),
            setup=[use_tools(tools)],
            scorer=scorer,
        )

        # Capture the candidate's stdout (print() lines surface as diagnostics,
        # mirroring docfinqa/protein_go's agent_stdout convention).
        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                logs = inspect_eval(
                    task,
                    model=self._default_model,
                    display="none",
                    log_dir=self._log_dir,
                    log_format="json",
                    log_level="warning",
                )
        except Exception as e:
            return 0.0, {
                "error.md": f"inspect.eval crashed: {type(e).__name__}: {e}",
                "agent_stdout": captured.getvalue(),
                "eval_wall_clock_seconds": _elapsed_seconds(_t0),
            }

        log = logs[0]
        score, diagnostics = self._extract_score_and_diagnostics(
            log, example, captured.getvalue()
        )
        diagnostics["eval_wall_clock_seconds"] = _elapsed_seconds(_t0)
        return score, diagnostics

    # -- Result extraction ---------------------------------------------------

    def _extract_score_and_diagnostics(
        self, log, example: Sample, agent_stdout: str
    ) -> tuple[float, dict]:
        diagnostics: dict[str, Any] = {
            "score_type": example.metadata.get("score_type"),
            "agent_stdout": agent_stdout,
            "sample_id": str(example.id),
        }

        # Save the agent-visible query into the problem dir so a reader of
        # any per-problem dir can see what the agent was asked.
        raw_query = example.metadata.get("raw_query") or ""
        if raw_query:
            diagnostics["query.md"] = str(raw_query)

        # Expose the gold scorer criteria to evolution. The AGENT never
        # sees state.target at runtime; this only flows into post-hoc
        # diagnostics that evolution reads when deciding how to mutate.
        # For specific/metadata queries the target is the gold corpus_id
        # list; for semantic queries it's the relevance criteria dict.
        if example.target:
            diagnostics["gold_criteria.md"] = str(example.target)

        # samples is None if eval failed catastrophically
        if not getattr(log, "samples", None):
            diagnostics["error.md"] = "no samples in eval log"
            return 0.0, diagnostics

        sample_log = log.samples[0]

        # Sample-level errors (e.g. a tool call raised) end up here, not in
        # the outer try/except, because inspect.eval catches and records them.
        sample_err = getattr(sample_log, "error", None)
        if sample_err is not None:
            err_msg = getattr(sample_err, "message", None) or str(sample_err)
            diagnostics["error.md"] = _head_tail_truncate(err_msg)

        scores = getattr(sample_log, "scores", None) or {}
        # PaperFindingBench attaches a single scorer per sample; multiple
        # scorer entries would mean the upstream task definition changed
        # and we'd be silently picking whichever lands first.
        if len(scores) > 1:
            logger.warning(
                f"Sample {example.id} produced {len(scores)} scores; "
                f"expected 1. Using first floatable. Names: {list(scores)}"
            )
        score_value = 0.0
        for sc in scores.values():
            v = getattr(sc, "value", 0)
            try:
                score_value = float(v)
                break
            except (TypeError, ValueError):
                pass

        # Cost from inspect's model usage tracking, split agent-vs-judge.
        # The scorer's GPT-4o relevance judge runs inside this same
        # inspect.eval on semantic queries, so its usage shares the log.
        # Exact-match on JUDGE_MODEL_IDS routes judge spend to
        # other_cost_usd (reported, never penalized — the agent doesn't
        # control it); everything else is agent spend.
        agent_cost_usd = 0.0
        judge_cost_usd = 0.0
        usage_summary: dict[str, Any] = {}
        cost_by_model_usd: dict[str, float] = {}
        try:
            stats = getattr(log, "stats", None)
            model_usage = getattr(stats, "model_usage", None) if stats else None
            if model_usage:
                for model_name, u in model_usage.items():
                    counts = {
                        "input_tokens": getattr(u, "input_tokens", 0),
                        "output_tokens": getattr(u, "output_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                        # Reasoning ("thinking") tokens are reported
                        # SEPARATELY from output_tokens by Gemini models
                        # (total = input + output + reasoning) but billed
                        # at the output rate. Dropping them here would
                        # silently underprice Gemini agents by the full
                        # thinking spend. Persisted into usage_summary so
                        # recorded runs retain the full accounting.
                        "reasoning_tokens": getattr(u, "reasoning_tokens", None) or 0,
                    }
                    usage_summary[model_name] = counts
                    model_cost = self._estimate_cost(model_name, counts)
                    if model_name in JUDGE_MODEL_IDS:
                        judge_cost_usd += model_cost
                    else:
                        agent_cost_usd += model_cost
                        cost_by_model_usd[model_name] = model_cost
        except Exception:
            pass

        # Per-example score is raw F1. The cost penalty applies at the
        # iteration level via PaperFinderEvaluator.aggregate — computing
        # it per example would diverge from the leaderboard's mean-cost
        # framing and would punish selective-routing strategies whose
        # batch-mean cost lands in the free zone.
        with self._cost_lock:
            self.total_eval_cost += agent_cost_usd
            self.total_judge_cost += judge_cost_usd

        diagnostics["score"] = score_value           # raw F1 in [0, 1]
        diagnostics["raw_score"] = score_value       # alias kept for readability in result.json
        diagnostics["cost_usd"] = agent_cost_usd     # agent spend (→ eval_cost, penalized bucket)
        diagnostics["agent_cost_usd"] = agent_cost_usd
        diagnostics["other_cost_usd"] = judge_cost_usd  # relevance judge (reported, never penalized)
        diagnostics["usage"] = usage_summary
        diagnostics["cost_by_model_usd"] = cost_by_model_usd  # agent models only; plumbed to result.json
        diagnostics["agent_output"] = (
            getattr(sample_log.output, "completion", "")[:1000] if sample_log.output else ""
        )

        # Judge-verdict surfacing (semantic queries only): which of the
        # submitted papers the benchmark judge accepted, in submitted
        # order — lets evolution separate recall misses from judge
        # rejections and audit its ranking. Best-effort: parse failures
        # or an unreadable cache just skip the diagnostic.
        try:
            if str(example.metadata.get("score_type", "")).startswith("semantic"):
                completion = (
                    getattr(sample_log.output, "completion", "") if sample_log.output else ""
                )
                payload = json.loads(completion[completion.index("{"):])
                submitted = [
                    normalize_corpus_id(str(r.get("paper_id", "")))
                    for r in (payload.get("output") or {}).get("results") or []
                ]
                try:
                    known_good = {
                        normalize_corpus_id(str(x))
                        for x in json.loads(str(example.target)).get("known_to_be_good") or []
                    }
                except (json.JSONDecodeError, TypeError, AttributeError):
                    known_good = set()
                verdicts = _judge_verdicts_markdown(
                    [s for s in submitted if s], str(example.id), known_good
                )
                if verdicts:
                    diagnostics["judge_verdicts.md"] = verdicts
        except Exception:
            pass

        return score_value, diagnostics

    # -- Cost helpers ---------------------------------------------------------

    @staticmethod
    def _estimate_cost(model_name: str, counts: dict) -> float:
        """Estimate cost on the LEADERBOARD'S price basis.

        Prices with litellm's BUNDLED local price map — the same table
        `astabench score` uses (agenteval runs under
        LITELLM_LOCAL_MODEL_COST_MAP=True) — NOT the live remote map.
        The two can diverge (a bundled map can lag true provider
        prices); decision: track Ai2's numbers for comparability with
        other leaderboard systems, even where they lag true provider
        prices — revisit only when Ai2 moves. Falls back to
        litellm.cost_per_token (live map + registry overlay) for models
        absent from the bundled map.

        Returns 0.0 if litellm isn't installed, can't price the model,
        or raises. Logs a one-shot warning the first time a model with
        non-trivial usage prices to $0 — that pattern almost always
        means litellm's pricing table doesn't know about a recent
        model, in which case the cost penalty silently never fires.
        Loud failure beats silently-zero spend tracking.

        Provider-prefix translation: Inspect-AI requires `google/...`
        for routing Google models, but litellm prices them under
        `gemini/...`. We normalize at the cost-pricing boundary so
        the registry's `google/...` strings price correctly.
        """
        try:
            import litellm
        except ImportError:
            return 0.0
        try:
            # Registers pricing for registry models newer than the
            # installed litellm, which would otherwise price to $0 and
            # never trip the cost penalty. Rates are single-sourced from
            # RoboPhD/config.py's SUPPORTED_MODELS.
            from RoboPhD.runner_utils import register_supported_model_pricing
            register_supported_model_pricing()
        except ImportError:
            # Standalone evaluator use without RoboPhD on sys.path — the
            # one-shot $0 warning below still flags unpriced models.
            pass
        input_tokens = counts.get("input_tokens", 0)
        output_tokens = counts.get("output_tokens", 0)
        total_tokens = counts.get("total_tokens", 0)
        reasoning_tokens = counts.get("reasoning_tokens", 0) or 0
        # Gemini reports reasoning ("thinking") tokens separately and
        # EXCLUDES them from output_tokens (total = input + output +
        # reasoning), yet Google bills them at the output rate. Mirror
        # agenteval's rule (agenteval/log.py compute_model_cost): fold
        # reasoning into completion tokens only when the token arithmetic
        # proves output excludes it — OpenAI reports reasoning already
        # inside output_tokens (input == total - output), and Anthropic's
        # cache-token pattern must not be mistaken for reasoning, so the
        # strict equality guard is load-bearing.
        completion_tokens = output_tokens
        if (
            reasoning_tokens
            and input_tokens == total_tokens - output_tokens - reasoning_tokens
        ):
            completion_tokens = output_tokens + reasoning_tokens
        litellm_name = (
            "gemini/" + model_name[len("google/"):]
            if model_name.startswith("google/")
            else model_name
        )
        try:
            bundled = _bundled_price_map()
            entry = bundled.get(litellm_name) or bundled.get(
                litellm_name.split("/", 1)[-1]
            )
            if entry and entry.get("input_cost_per_token") is not None:
                # Leaderboard basis: flat bundled-map rates, reasoning
                # billed at the output rate (matches agenteval).
                cost = (
                    input_tokens * (entry.get("input_cost_per_token") or 0.0)
                    + completion_tokens * (entry.get("output_cost_per_token") or 0.0)
                )
            else:
                # Model not in the bundled map (e.g. newer than the
                # installed litellm): fall back to the live map plus the
                # registry overlay so the cost penalty still fires.
                # One-shot warning per model — this is expected for
                # brand-new models, but it means this model's cost is
                # NOT on the leaderboard's billing basis, so it must be
                # visible.
                if litellm_name not in _live_map_fallback_warned:
                    _live_map_fallback_warned.add(litellm_name)
                    logger.warning(
                        "%r is not in litellm's bundled price snapshot; "
                        "pricing it from the live map instead. Its cost is "
                        "NOT on the leaderboard's billing basis and may "
                        "diverge from an official astabench score.",
                        litellm_name,
                    )
                pin, pout = litellm.cost_per_token(
                    model=litellm_name,
                    prompt_tokens=input_tokens,
                    completion_tokens=completion_tokens,
                )
                cost = (pin or 0.0) + (pout or 0.0)
        except Exception:
            cost = 0.0

        if cost == 0.0 and (input_tokens + output_tokens) > 100 and model_name not in _unpriced_models_warned:
            _unpriced_models_warned.add(model_name)
            logger.warning(
                "litellm priced %r at $0 despite %d input + %d output tokens. "
                "Cost penalty will not fire for this model. Update litellm or "
                "patch cost_per_token mapping.",
                model_name, input_tokens, output_tokens,
            )
        return cost
