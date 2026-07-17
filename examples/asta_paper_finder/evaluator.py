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

import fcntl
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

import _grounding as grounding

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
#
# gpt-5.4-nano is included because training runs may override the grader to
# it for cost (see _apply_training_grader / $PF_TRAINING_GRADER_MODEL). It is
# never an agent-selectable model_registry handle, so listing it here only
# routes its spend to the judge bucket; it does not enable it as a solver.
JUDGE_MODEL_IDS = frozenset({
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-5.4-nano",
})
if _ASTABENCH_GRADER_MODEL_NAME not in JUDGE_MODEL_IDS:
    raise RuntimeError(
        f"astabench's paper_finder relevance judge is "
        f"{_ASTABENCH_GRADER_MODEL_NAME!r}, which is not in this "
        f"evaluator's JUDGE_MODEL_IDS {sorted(JUDGE_MODEL_IDS)}. The "
        f"agent-vs-judge cost split would misattribute judge spend to "
        f"agents. Update JUDGE_MODEL_IDS to include the new grader "
        f"(and re-check that no model_registry handle uses that ID)."
    )


def _safe_cache_rmw(path: str, query_id: str, judgements: dict[str, str]) -> None:
    """Synchronous locked read-merge-write of the judge cache.

    Module-level (not a closure) so the multi-process stress gate
    (_check_cache_stress.py) can exercise exactly this code, and so the
    async wrapper can push it off the event loop via asyncio.to_thread.
    """
    with open(path + ".lock", "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            try:
                text = Path(path).read_text()
            except OSError:
                text = ""
            cache: dict = {}
            if text:
                try:
                    cache = json.loads(text)
                except json.JSONDecodeError as e:
                    # Torn tail from a pre-fix writer: everything up
                    # to e.pos is typically a complete JSON object.
                    try:
                        cache = json.loads(text[:e.pos]) if e.pos else {}
                    except json.JSONDecodeError:
                        cache = {}
                    logger.warning(
                        "judge cache at %s was corrupt (%s); recovered "
                        "%d queries from the valid prefix", path, e, len(cache),
                    )
            if isinstance(cache.get(query_id), dict):
                cache[query_id].update(judgements)
            else:
                cache[query_id] = dict(judgements)
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(path) or ".", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as tmp_f:
                    json.dump(cache, tmp_f)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _install_safe_judge_cache() -> None:
    """Replace astabench's `update_references` with a multiprocess-safe
    version, in this process's namespaces only (no astabench file on
    disk is touched; official `astabench eval` submissions run stock
    code in a separate process and never see this).

    Why: upstream does a read-modify-write of the package-global
    detailed_reference.json guarded only by inspect's PER-PROCESS
    concurrency() lock, and writes non-atomically. Our evals run in
    parallel subprocesses, so concurrent writers tore the file, and
    astabench's init_references (which catches only FileNotFoundError)
    then raised JSONDecodeError through scorer init — zeroing every
    affected eval (run asta_paper_finder_20260713_125809, iterations
    11-14). Fix: cross-process flock over the whole read-merge-write,
    valid-prefix recovery on a corrupt read (upstream silently drops
    the entire cache), and atomic tempfile+rename writes so readers
    can never observe a torn file. All workers are subprocesses on one
    host, so flock suffices.

    Patches BOTH bindings: the origin in paper_finder_utils and the
    from-import binding in eval.py that get_llm_relevance actually
    calls. Idempotent via the marker attribute.
    """
    from astabench.evals.paper_finder import eval as _pf_eval
    from astabench.evals.paper_finder import paper_finder_utils as _pf_utils

    if getattr(_pf_utils.update_references, "_robophd_safe_cache", False):
        return

    async def _safe_update_references(query_id: str, judgements: dict[str, str]) -> None:
        import asyncio
        # Read the path at call time so tests can monkeypatch it. Push
        # the blocking flock + file I/O off the event loop so a writer
        # waiting on the lock doesn't stall in-flight work on this
        # worker's loop.
        await asyncio.to_thread(
            _safe_cache_rmw, _pf_utils.detailed_reference_path, query_id, judgements
        )

    _safe_update_references._robophd_safe_cache = True  # type: ignore[attr-defined]
    _pf_utils.update_references = _safe_update_references
    _pf_eval.update_references = _safe_update_references


_install_safe_judge_cache()
# Install AFTER the safe cache writer so the grounded judge's dynamic
# update_references lookup resolves to the multiprocess-safe version.
grounding.install_grounded_judge()


# Environment variable naming the judge-verdict cache file. When set, the
# cache is redirected away from astabench's package-global
# detailed_reference.json to this path. main.py points it at a per-run file
# for training (so verdicts never leak across runs) and at a fresh empty file
# for each test / formal eval (so every verdict is rendered on the submitting
# agent's own evidence — pristine, matching a fresh official environment).
CACHE_PATH_ENV = "PF_JUDGE_CACHE_PATH"

# Re-exported for main.py: env var that enables the training-only top-estimate
# judging cap (defined in _grounding, read by the grounded judge).
CAP_JUDGE_ENV = grounding.CAP_JUDGE_ENV


def _apply_cache_redirect() -> None:
    """Point astabench's judge cache at $PF_JUDGE_CACHE_PATH, if set.

    Runs at import in every process that imports this module — importantly,
    inside each evaluation subprocess, which inherits the variable from the
    parent. Both the writer (_safe_cache_rmw) and the reader (init_references)
    resolve the path through _pf_utils.detailed_reference_path at call time, so
    reassigning that one attribute redirects both. get_normalizer_references
    memoizes the loaded cache on first use; we clear the memo so a redirect set
    before scoring is honored even if something read the cache during import."""
    path = os.environ.get(CACHE_PATH_ENV)
    if not path:
        return
    from astabench.evals.paper_finder import eval as _pf_eval
    from astabench.evals.paper_finder import paper_finder_utils as _pf_utils
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except OSError:
        pass
    _pf_utils.detailed_reference_path = path
    _pf_eval._normalizer_reference = None
    _pf_eval._detailed_reference = None


_apply_cache_redirect()


# Environment variable naming the TRAINING relevance-judge model. When set, the
# scorer's grader is overridden to this model for cost control; the held-out
# test / formal path leaves it unset so scoring uses astabench's stock GPT-4o
# grader. main.py only sets this when the user opts in via --training-judge,
# and clears it on every test path. Enabling nano is gated on the calibration
# study (_check_judge_calibration.py) showing high GPT-4o agreement — training
# optimizes against this judge but test scores on GPT-4o.
TRAINING_GRADER_ENV = "PF_TRAINING_GRADER_MODEL"


def _apply_training_grader() -> None:
    """Override astabench's relevance grader with $PF_TRAINING_GRADER_MODEL, if
    set. relevance.GRADER_MODEL_NAME is read at judge time, and the grounded
    judge calls load_relevance_judgement dynamically, so reassigning the module
    attribute is enough. Warns if the model isn't priced in the bundled map —
    an unpriced grader would silently zero the reported judge cost."""
    model = os.environ.get(TRAINING_GRADER_ENV)
    if not model:
        return
    if model not in JUDGE_MODEL_IDS:
        raise RuntimeError(
            f"{TRAINING_GRADER_ENV}={model!r} is not in JUDGE_MODEL_IDS "
            f"{sorted(JUDGE_MODEL_IDS)}; its spend would be misbilled to the "
            f"agent. Add it to JUDGE_MODEL_IDS (and confirm it is not a "
            f"model_registry solver handle) before using it as the grader."
        )
    from astabench.evals.paper_finder import relevance as _pf_rel
    _pf_rel.GRADER_MODEL_NAME = model
    if _bundled_price_map().get(model.split("/", 1)[-1]) is None:
        logger.warning(
            "training grader %s is not in the bundled litellm price map; "
            "its judge cost may report as $0", model,
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


# Applied at import (after _bundled_price_map is defined, which its price check
# needs). No-op unless $PF_TRAINING_GRADER_MODEL is set.
_apply_training_grader()


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
# Free-zone width $0.06: set at the Standard-tier leaderboard reference
# (generic ReAct + GPT-5 Mini, ~$0.06/query) to target Pareto-dominating the
# two leaderboard points sitting at $0.06 and $0.063 — an agent that matches
# their score while staying in this zone wins on the cost axis — and to pull
# default spend down. Tighter than the earlier $0.10, so evidence-heavy
# pipelines (query decomposition, snippet extraction, per-candidate filtering)
# now feel the penalty sooner; the mini/flash tier still fits, and Opus-tier
# ReAct ($1.49-$3.38/query) remains decisively penalized. Immutable per run
# (task_config_extras), so in-flight runs keep their stored threshold on
# resume; only fresh runs pick up this default.
MIN_COST_THRESHOLD = 0.06
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


def _lookup_k_estimate(query_id: str) -> int | None:
    """The benchmark's per-query estimate of total relevant papers (the
    semantic recall denominator K), read from astabench's normalizer
    reference ONLY when it is already initialized — the scorer loads it
    during eval, so it is present whenever component metrics exist. Never
    calls get_normalizer_references(), which would trigger a fresh
    reference load just to render a diagnostic."""
    try:
        from astabench.evals.paper_finder import eval as pf_eval

        ref = pf_eval._normalizer_reference
        return ref.get(query_id) if ref else None
    except Exception:
        return None


def _score_calculation_markdown(
    score_type: str,
    score_meta: dict,
    submitted: list[str],
    gold_ids: list[str],
    k_estimate: int | None,
) -> str | None:
    """The worked calculation behind this query's F1: the scorer's own
    component metrics plugged into the formulas documented in
    background.md "Scoring (per query)".

    Every float comes verbatim from the scorer's Score.metadata (computed
    by astabench's calc_standard_f1 / calc_adjusted_f1) — nothing is
    re-derived, so the numbers cannot drift from the real score. Only the
    matched/missed id lists are derived locally, from the parsed
    submission and the gold ids, as illustration. Returns None when the
    expected component keys are absent (scorer didn't run its normal
    path).
    """
    if score_type.startswith("semantic"):
        rank = score_meta.get("rank")
        recall = score_meta.get("estimated_recall_at_estimate")
        f1 = score_meta.get("adjusted_f1")
        if rank is None or recall is None or f1 is None:
            return None
        hits_full = score_meta.get("relevant_predictions_at_full")
        if k_estimate is None:
            recall_full = score_meta.get("estimated_recall_at_full") or 0
            if hits_full and recall_full:
                k_estimate = round(hits_full / recall_full)
        n_top_k: int | None = None
        if k_estimate:
            n_top_k = round(recall * k_estimate)
            recall_note = (
                f"({n_top_k} of K={k_estimate} estimated relevant found in "
                f"your top K; only grade-3 papers count)"
            )
        else:
            recall_note = "(K unknown; only grade-3 papers count)"
        lines = [
            f"rank   = {rank:.4f}   (order quality: lower-bound-corrected "
            f"nDCG over the judged grades)",
            f"recall = {recall:.4f}   {recall_note}",
            f"score  = harmonic(rank, recall) = {f1:.4f}",
        ]
        if hits_full is not None and n_top_k is not None and hits_full > n_top_k:
            lines += [
                "",
                f"{int(hits_full) - n_top_k} more Perfect paper(s) ranked "
                f"below position K earned no recall credit — ordering cost "
                f"you recall.",
            ]
        lines += [
            "",
            '(per-paper grades: judge_verdicts.md · formulas: background.md '
            '"Scoring (per query)")',
        ]
        return "\n".join(lines)

    # specific_f1 / metadata_f1: exact-match standard F1.
    precision = score_meta.get("precision")
    recall = score_meta.get("known_recall_at_full")
    f1 = score_meta.get("standard_f1")
    if precision is None or recall is None or f1 is None:
        return None
    hits = score_meta.get("relevant_predictions_at_full")
    unique_submitted = list(dict.fromkeys(submitted))[:250]
    n_sub = len(unique_submitted)
    n_gold = len(gold_ids)
    gold_set = set(gold_ids)
    submitted_set = set(unique_submitted)
    matched = [pid for pid in unique_submitted if pid in gold_set]
    missed = [g for g in gold_ids if g not in submitted_set]

    def _frac(num, den, value: float) -> str:
        # Show the hits/N fraction only when it reproduces the scorer's
        # float — on any mismatch the scorer's own number wins, alone.
        if num is None or not den or abs(num / den - value) > 1e-6:
            return f"{value:.4f}"
        return f"{int(num)}/{den} = {value:.4f}"

    hits_shown = int(hits) if hits is not None else len(matched)
    lines = [
        f"submitted: {n_sub} unique paper id(s) (scorer reads first 250) "
        f"· gold: {n_gold}",
        f"hits (submitted ∩ gold): {hits_shown}"
        + (f" → {', '.join(matched)}" if matched else ""),
        f"missed gold ids: {', '.join(missed) if missed else '(none)'}",
        "",
        f"precision = hits / #submitted = {_frac(hits, n_sub, precision)}",
        f"recall    = hits / #gold      = {_frac(hits, n_gold, recall)}",
        f"score     = harmonic(precision, recall) = {f1:.4f}",
    ]
    return "\n".join(lines)


def _judge_verdicts_markdown(
    submitted_ids: list[str], query_id: str, known_good: set[str]
) -> str | None:
    """Per-paper judge verdicts for a semantic query, in submitted order.

    Sourced from the grounded judge's in-process record of THIS evaluation
    (grounding.last_judgements()), not the persistent cache file — so the
    verdicts reflect the judgement rendered on this agent's own (grounded)
    evidence, never a stale entry another agent wrote for the same paper.
    Zero extra LLM calls.

    Verdict per submitted paper:
      - in the query's known_to_be_good list → pre-seeded Perfect (the
        scorer never LLM-judges these)
      - judged this eval → the judge's label, verbatim
      - unjudged while the judge demonstrably ran → "(judge call failed —
        excluded from scoring)", plus a footer line saying it is judge-side
        and neutral, so evolution doesn't misread the gap as agent-caused
        or as a 0
      - unjudged with no judgements at all (scorer didn't run) →
        "(no verdict recorded)"

    Returns None when there's nothing to report (no submissions, or no
    grounded judgements were produced and nothing is known-good).
    """
    if not submitted_ids:
        return None
    raw_judgements = grounding.last_judgements()
    if not raw_judgements and not known_good:
        return None
    # Normalize judgement keys to match the normalized submitted_ids.
    judged = {normalize_corpus_id(str(k)): str(v) for k, v in raw_judgements.items()}

    # The judge stores astabench's internal label strings
    # ("perfectly_relevant_papers", ...); translate to the human labels.
    pretty = {
        "perfectly_relevant_papers": "Perfectly Relevant",
        "highly_relevant_papers": "Highly Relevant",
        "somewhat_relevant_papers": "Somewhat Relevant",
        "not_relevant_papers": "Not Relevant",
    }
    perfect_raw = "perfectly_relevant_papers"

    # Training caps judging at the top-`cap` submitted papers (recall depth);
    # deeper papers are intentionally not judged, so label them as such rather
    # than as a judge failure — otherwise evolution would chase "missing"
    # verdicts on papers that cannot affect the score.
    cap = grounding.last_cap()

    lines = []
    n_perfect = n_lower = n_unknown = n_beyond = 0
    for i, pid in enumerate(submitted_ids[:250], 1):  # scorer reads first 250
        if pid in known_good:
            verdict = "Perfectly Relevant (known-good)"
            n_perfect += 1
        elif pid in judged:
            raw = judged[pid]
            verdict = pretty.get(raw, raw)
            if raw == perfect_raw:
                n_perfect += 1
            else:
                n_lower += 1
        elif cap is not None and i > cap:
            verdict = "(beyond scored depth — not judged)"
            n_beyond += 1
        elif judged:
            # The judge ran (other papers have verdicts) but this one has
            # none: a judge-side call failure. The scorer excludes the paper
            # from both the rank sequence and recall — neither credited nor
            # penalized.
            verdict = "(judge call failed — excluded from scoring)"
            n_unknown += 1
        else:
            verdict = "(no verdict recorded)"
            n_unknown += 1
        lines.append(f"{i}. {pid} — {verdict}")
    n_submitted = len(lines)
    tail = f" / {n_beyond} beyond scored depth" if n_beyond else ""
    lines.append(
        f"\n{n_perfect} Perfect / {n_lower} lower / {n_unknown} no verdict"
        f"{tail}, of {n_submitted} submitted"
    )
    if judged and n_unknown:
        lines.append(
            "Judge-call failures are a judge-side error (rare, ~1%), unrelated "
            "to your submission or evidence: the paper is excluded from both "
            "the rank sequence and recall — neither credited nor penalized."
        )
    return "\n".join(lines)


def _evidence_grounding_markdown() -> str | None:
    """Feedback on evidence passages discarded this eval for failing the
    grounding check.

    A passage is discarded when it is not verbatim-derivable from text the agent
    retrieved for that paper (fabricated, paraphrased, or for a paper never
    retrieved). Discarding is per-passage: `full` = every passage dropped (the
    paper is judged Not Relevant with no judge call); `partial` = some passages
    dropped, the grounded ones still judged. Surfacing the offending passages
    lets the next iteration's evolution fix its evidence construction — quote
    retrieved text verbatim, joined by ` ... `. Returns None when nothing was
    discarded (the healthy case)."""
    blanked = grounding.last_blanked()
    if not blanked:
        return None
    n_full = sum(1 for b in blanked if b[3] == "full")
    n_partial = len(blanked) - n_full
    lines = [
        f"{len(blanked)} paper(s) had evidence discarded as not verbatim-"
        f"derivable from retrieved text: {n_full} fully (→ Not Relevant, no "
        f"judge call), {n_partial} partially (grounded passages kept and judged, "
        f"ungrounded ones dropped). Quote retrieved corpus text verbatim, joined "
        f"by ` ... ` (≤3 passages).",
        "",
    ]
    for pid, dropped, raw, kind in blanked[:50]:
        first = (dropped[0] if dropped else (raw or ""))[:160]
        lines.append(f"- {pid} [{kind}]: dropped passage → {first!r}")
    if len(blanked) > 50:
        lines.append(f"- … and {len(blanked) - 50} more")
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


def _wrap_tools_for_provenance(tools: list) -> list:
    """Wrap each corpus tool so its result payload is recorded for evidence
    grounding, then handed back to the agent unchanged.

    This is the only seam we own between the Asta MCP tools and the agent, so
    it is where we learn what text the agent actually retrieved for each paper.
    Wrapping via ToolDef round-trips the tool's name/description/parameters
    (the schema inspect needs), the same mechanism astabench uses to install
    its retry wrapper. Recording is best-effort and never alters the result.
    """
    from inspect_ai.tool import ToolDef

    wrapped = []
    for t in tools:
        td = ToolDef(t)
        orig = td.tool

        def _make(orig_call):
            async def _recording(*args, **kwargs):
                result = await orig_call(*args, **kwargs)
                grounding.record_tool_result(result)
                return result
            return _recording

        td.tool = _make(orig)
        wrapped.append(td.as_tool())
    return wrapped


def _build_tools(sample_id: str):
    """Return the Asta MCP corpus tools to attach for this sample.

    The factory applies the sample's date cutoff (no future-paper leaks)
    and wraps every tool in astabench's make_retry_wrapper (429/529/504 +
    BrokenResourceError, exponential backoff), so agents never see a
    transient rate limit. We add an outer wrapper that records each result
    payload for evidence grounding (see _grounding.py).
    """
    inserted_before = get_inserted_before_per_dataset_type(sample_id)
    from astabench.tools import make_asta_mcp_tools
    tools = make_asta_mcp_tools(insertion_date=inserted_before)
    return _wrap_tools_for_provenance(tools)


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

        # Clear grounding state before this sample retrieves anything. In the
        # default subprocess path this process is fresh, but the in-process
        # path (tests, smoke runs) reuses one process across samples.
        grounding.reset()

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
        score_obj = None  # kept for its .metadata / .explanation (score_calculation.md)
        for sc in scores.values():
            v = getattr(sc, "value", 0)
            try:
                score_value = float(v)
                score_obj = sc
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

        # Score-calculation surfacing (all query types): the scorer's own
        # component metrics (precision/recall/hits, or rank/recall/K)
        # rendered as the background.md formulas with this query's numbers
        # filled in — so evolution sees WHY the F1 is what it is, not just
        # the final float. Plus, semantic-only, the per-paper judge
        # verdicts, which separate recall misses from judge rejections and
        # let evolution audit its ranking. Best-effort: parse failures or
        # an unreadable cache just skip the affected diagnostic.
        try:
            score_type = str(example.metadata.get("score_type", ""))
            completion = (
                getattr(sample_log.output, "completion", "") if sample_log.output else ""
            )
            try:
                payload = json.loads(completion[completion.index("{"):])
                submitted = [
                    normalize_corpus_id(str(r.get("paper_id", "")))
                    for r in (payload.get("output") or {}).get("results") or []
                ]
                submitted = [s for s in submitted if s]
            except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
                submitted = []
            try:
                target = json.loads(str(example.target))
                gold_ids = [
                    normalize_corpus_id(str(x))
                    for x in target.get("corpus_ids") or []
                ]
                known_good = {
                    normalize_corpus_id(str(x))
                    for x in target.get("known_to_be_good") or []
                }
            except (json.JSONDecodeError, TypeError, AttributeError):
                gold_ids, known_good = [], set()

            score_meta = (
                dict(getattr(score_obj, "metadata", None) or {})
                if score_obj is not None else {}
            )
            if score_meta:
                calc = _score_calculation_markdown(
                    score_type, score_meta, submitted, gold_ids,
                    _lookup_k_estimate(str(example.id)),
                )
                if calc:
                    diagnostics["score_calculation.md"] = calc
            else:
                # No component metrics means the scorer bailed before its
                # normal path — e.g. "Agent output has an invalid format".
                # Its explanation is the only record of why the query
                # scored 0; surface it instead of a silent zero.
                explanation = (
                    getattr(score_obj, "explanation", None)
                    if score_obj is not None else None
                )
                if explanation:
                    diagnostics["score_calculation.md"] = (
                        f"score = {score_value:g} — the scorer produced no "
                        f"component metrics. Scorer explanation:\n\n"
                        f"{_head_tail_truncate(str(explanation))}"
                    )

            if score_type.startswith("semantic"):
                verdicts = _judge_verdicts_markdown(
                    submitted, str(example.id), known_good
                )
                if verdicts:
                    diagnostics["judge_verdicts.md"] = verdicts
                grounding_md = _evidence_grounding_markdown()
                if grounding_md:
                    diagnostics["evidence_grounding.md"] = grounding_md
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
