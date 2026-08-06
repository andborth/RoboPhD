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
# gpt-5.6-luna is included because training runs (and opt-in internal test
# evals) may override the grader to it for cost (see _apply_training_grader /
# $PF_TRAINING_GRADER_MODEL). It is the ONLY approved alternate judge: it
# passed the calibration gate on 2026-07-20 (PERFECT/not kappa 0.755 vs the
# 0.7 bar, Perfect rates 31.3% vs GPT-4o's 32.7%, 2/300 format repairs).
# gpt-5.4-nano was removed the same day: it FAILED calibration (kappa ~0.52,
# severe Perfect-deflation — credited only 51% of GPT-4o's Perfects) and was
# never used in any recorded run. Neither judge is an agent-selectable
# model_registry handle, so listing here only routes spend to the judge
# bucket; it does not enable them as solvers.
JUDGE_MODEL_IDS = frozenset({
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-5.6-luna",
})

# Judge-only price entries for models newer than the pinned litellm 1.88.1
# (which must not move — it is the leaderboard's billing basis for AGENT
# models). Judge spend is internal accounting (other_cost_usd, never
# penalized, never leaderboard-exposed), so a local table is sound here in a
# way it would not be for solver pricing. Rates from OpenAI's pricing page,
# 2026-07-31 (luna repriced 80% down that day: $1.00/$0.10/$6.00 ->
# $0.20/$0.02/$1.20; the earlier rates are what every campaign cost figure
# recorded before that date was computed on).
# cached_input applies to input_tokens_cache_read (measured
# essentially unused by the astabench judge prompt — 0.007% on the v0_0_7
# official run — but priced correctly if it ever engages). KNOWN GAP:
# the gpt-5.6 family also BILLS cache writes at 1.25x input (older
# models write free); we don't price writes because Inspect's pinned
# OpenAI provider predates the cache_write_tokens usage field (observed
# always 0 alongside nonzero reads). At the ~1% caching activity the
# judge prompt structure allows, the under-billing is <<1% of judge
# cost — revisit only if caching ever becomes material.
JUDGE_PRICE_OVERRIDES = {
    "gpt-5.6-luna": {
        "input_cost_per_token": 0.20e-6,
        "cached_input_cost_per_token": 0.02e-6,
        "output_cost_per_token": 1.20e-6,
    },
}
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


def _install_tool_transport_hardening() -> None:
    """Patch astabench's tool transport, in this process's namespaces only
    (no astabench file on disk is touched; official `astabench eval`
    submissions run stock code in a separate process and never see this).

    Two patches, motivated by run asta_paper_finder_20260724_193339
    (iterations 9-11: HTTP 500 burst windows from the shared Asta MCP
    endpoint — 100+ tool errors in one 14-query batch, exact-match queries
    zeroed when every retrieval probe died inside one window):

    1. `_is_retryable_error`: upstream retries only {429, 529, 504}. 500s
       arrive in contention windows and are transient, so 500/502/503 are
       added. Resolved as a module global inside the retry loop, so the
       patch reaches wrappers created before or after install.
    2. `make_retry_wrapper`: each underlying attempt first acquires a
       global per-endpoint launch slot (tool_pacer; ~8 launches/s shared
       across ALL eval subprocesses via PF_TOOL_PACER_PATH). Pacing sits
       INSIDE the retry ladder so retries are paced too — retries during
       a burst are otherwise the biggest burst amplifier. Per-agent
       self-throttles cannot do this: no agent can see the other workers.

    Idempotent via marker attributes; signature of make_retry_wrapper is
    preserved (the transport docs pin test inspects it).
    """
    import functools

    from astabench.tools import asta_tools as _at

    import tool_pacer as _tp

    if getattr(_at.make_retry_wrapper, "_robophd_paced", False):
        return

    _orig_retryable = _at._is_retryable_error
    _orig_make_retry_wrapper = _at.make_retry_wrapper
    _RETRYABLE_5XX = {500, 502, 503}

    def _retryable_with_5xx(error: Exception) -> bool:
        import httpx
        for leaf in _at._unravel_exception_group(error):
            if isinstance(leaf, httpx.HTTPStatusError) and hasattr(leaf, "response"):
                if leaf.response.status_code in _RETRYABLE_5XX:
                    return True
                break  # defer non-5xx status decisions to upstream
        return _orig_retryable(error)

    @functools.wraps(_orig_make_retry_wrapper)
    def _paced_make_retry_wrapper(td, *args, **kwargs):
        orig_call = td.tool
        name = td.name

        async def _paced(*a, **kw):
            await _tp.pace(name)
            return await orig_call(*a, **kw)

        td.tool = _paced
        return _orig_make_retry_wrapper(td, *args, **kwargs)

    _retryable_with_5xx._robophd_5xx = True  # type: ignore[attr-defined]
    _paced_make_retry_wrapper._robophd_paced = True  # type: ignore[attr-defined]
    _at._is_retryable_error = _retryable_with_5xx
    _at.make_retry_wrapper = _paced_make_retry_wrapper


_install_safe_judge_cache()
_install_tool_transport_hardening()
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
EVIDENCE_CAP_ENV = grounding.EVIDENCE_CAP_ENV


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


# Environment variable naming the relevance-judge model in force. When set,
# the scorer's grader is overridden to this model; when unset, scoring uses
# astabench's stock GPT-4o grader. Despite the PF_TRAINING_ name (kept for
# checkpoint/cache compatibility) it is set on BOTH phases: main.py writes
# --training-judge for training and --test-judge for held-out evals, each
# resolved independently. Since 2026-07-28 the training default is luna, so a
# flagless run sets this during training and clears it for the test eval —
# training optimizes against the cheap basis, the headline scores on GPT-4o.
# Any judge named here must have passed _check_judge_calibration.py; see
# JUDGE_MODEL_IDS for the approved set and the models rejected from it.
TRAINING_GRADER_ENV = "PF_TRAINING_GRADER_MODEL"
# Judge-prompt profile for the alternate judge: "no-prose" drops the
# snippet/summary prose the scorer never reads (validated for luna only —
# see _judge_normalize.NO_PROSE_JUDGE_TEMPLATE). Only honored alongside
# TRAINING_GRADER_ENV; a profile without a judge override is a hard error
# (the stock GPT-4o basis must stay byte-identical to official scoring,
# and 4o failed the no-prose calibration outright).
TRAINING_GRADER_PROMPT_ENV = "PF_TRAINING_GRADER_PROMPT"


def _apply_training_grader() -> None:
    """Override astabench's relevance grader with $PF_TRAINING_GRADER_MODEL, if
    set. relevance.GRADER_MODEL_NAME is read at judge time, and the grounded
    judge calls load_relevance_judgement dynamically, so reassigning the module
    attribute is enough.

    When the override is active, the lenient judge-output normalizer
    (_judge_normalize.install) is patched in as well: alternate judges emit
    rare near-JSON that astabench's strict parser would silently drop as
    Not Relevant (luna: 2/300 in calibration). Stock GPT-4o paths — the
    default test eval, official submissions, and any run that opts training
    back onto GPT-4o — are never patched, preserving strict-parser parity
    with astabench.

    Warns if the model isn't priced in the bundled map AND has no
    JUDGE_PRICE_OVERRIDES entry — an unpriced grader would silently zero
    the reported judge cost."""
    model = os.environ.get(TRAINING_GRADER_ENV)
    prompt_profile = os.environ.get(TRAINING_GRADER_PROMPT_ENV)
    if not model:
        if prompt_profile:
            raise RuntimeError(
                f"{TRAINING_GRADER_PROMPT_ENV}={prompt_profile!r} is set without "
                f"{TRAINING_GRADER_ENV}. A judge-prompt profile only applies to "
                f"the alternate training judge — the stock GPT-4o basis must "
                f"stay byte-identical to official scoring (and gpt-4o failed "
                f"the no-prose calibration: +18.5% Perfect inflation)."
            )
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
    import _judge_normalize
    _judge_normalize.install()
    if prompt_profile:
        if prompt_profile != "no-prose":
            raise RuntimeError(
                f"unknown {TRAINING_GRADER_PROMPT_ENV}={prompt_profile!r} "
                f"(supported: 'no-prose')"
            )
        _judge_normalize.install_no_prose_prompt()
    bare = model.split("/", 1)[-1]
    if _bundled_price_map().get(bare) is None and bare not in JUDGE_PRICE_OVERRIDES:
        logger.warning(
            "training grader %s is not in the bundled litellm price map and "
            "has no JUDGE_PRICE_OVERRIDES entry; its judge cost may report "
            "as $0", model,
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
#   errors_equivalent = max(0, mean_cost - min_cost_threshold) / cost_per_error
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
# Default cost_per_error, expressed as a FRACTION of the free zone
# rather than in dollars: the dollars that make a sensible penalty slope
# scale with the threshold they sit beside. The same $0.02 that is a mild
# tiebreaker against a $0.12 free zone is a wall against a $0.033 one, and
# campaigns now routinely move the threshold by 4x between runs, so a
# fixed-dollar default silently changes the penalty's character every time.
# 10% is the ratio the recent campaigns converged on by hand (v0_0_8 ran
# $0.033/$0.003 = 9.1%; ds1000's sharp-cap arm ran exactly 10%).
#
# It is a real sharpening of the old $0.02 default: at the $0.06 threshold
# the slope goes $0.02 -> $0.006, i.e. 3.3x steeper. ds1000's two-arm
# result for a 3.33x sharpening (0.001 -> 0.0003, same threshold) was
# ~-3.9pp accuracy for 51% cheaper inference — expect evolution to retreat
# further under the cap rather than buy through it.
#
# Sizing, in the units the penalty is denominated in: PFB scores are
# continuous F1 (typical per-query 0.2-0.4), so one error-equivalent is a
# LARGER unit than in a binary benchmark. At the default threshold a n=20
# $0.30-mean Sonnet-tier pipeline pays 200 pts — decisively prohibitive.
# Set cost_per_error large (e.g. $1, $10) to recover pure-tiebreaker
# semantics. Score is unbounded below — catastrophically expensive agents
# land well negative, which is intentional.
COST_PER_ERROR_FRACTION = 0.10


def default_cost_per_error(min_cost_threshold: float) -> float:
    """The default penalty slope for a given free-zone width.

    Single source for both the evaluator's own default and main.py's
    --cost-per-error default, so the two can never drift into an
    explicit-default-is-not-default trap. Rounded for the same reason as
    runner_utils.parse_dollars_or_percent: float noise in the product
    would compare unequal to a stored value on --resume.
    """
    return round(min_cost_threshold * COST_PER_ERROR_FRACTION, 12)


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


# Stands in for markdown_evidence in submission.json on results beyond the
# judging cap. Public: _check_judge_calibration.py imports it to skip these
# entries when re-judging (the text must never be scored as real evidence).
EVIDENCE_OMITTED_MARKER = "(evidence omitted — beyond scored depth, never judged)"


def _submission_json(
    completion: str,
    score_type: str,
    judge_cap: int | None,
    persist_full_evidence: bool = False,
) -> str | None:
    """The agent's full submitted payload, for the submission.json diagnostic.

    Replaces the old 1000-char agent_output preview, which forced evolution
    and _check_judge_calibration.py alike to reconstruct submissions from
    fragments (run 20260716_072622 flagged the truncation in 10 of 22
    iterations' reflections).

    Content-faithful, with size bounded by structure rather than a byte cap:
    - a parseable payload is stored pretty-printed, results in submitted
      order;
    - on semantic queries judged under a cap, results beyond the scored
      depth are never judged and affect neither rank nor recall, so their
      markdown_evidence (the bulk of the bytes) is replaced with
      EVIDENCE_OMITTED_MARKER — paper_id and position are kept, so
      "gold was buried at position N" stays diagnosable. This is the
      TRAINING behaviour and is deliberately frozen: it is the contract
      background.md states to evolution sessions, and evolution reads
      these files. persist_full_evidence=True suppresses the trimming and
      is set ONLY on the test evaluator (main.py's with_overrides), where
      nothing evolution sees is affected — it makes a test eval
      rejudgeable offline at any depth, including the uncapped basis
      official astabench scoring uses;
    - specific/metadata payloads are kept whole (the scorer never reads
      evidence there, and agents submit "" for it; the ids are the whole
      object of study);
    - a completion that json.loads rejects (the scorer's own primary parse)
      is stored raw — the malformation is the information.
    """
    if not completion:
        return None
    try:
        payload = json.loads(completion)
        results = (payload.get("output") or {}).get("results")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return completion
    if not isinstance(results, list):
        return completion
    if (
        not persist_full_evidence
        and judge_cap is not None
        and str(score_type).startswith("semantic")
    ):
        for r in results[judge_cap:]:
            if isinstance(r, dict) and r.get("markdown_evidence"):
                r["markdown_evidence"] = EVIDENCE_OMITTED_MARKER
    return json.dumps(payload, indent=2, ensure_ascii=False)


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


def _semantic_score_components(
    score_meta: dict, k_estimate: int | None
) -> dict | None:
    """Derived semantic score components shared by the prose and JSON
    score diagnostics — the single home of the K-fallback derivation
    (K from hits/recall when no estimate is provided, grade-3-in-top-K
    from recall·K), so the two renderings cannot drift. Returns None
    when the scorer didn't produce its normal component metrics."""
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
    n_top_k = round(float(recall) * k_estimate) if k_estimate else None
    return {
        "score": float(f1),
        "rank": float(rank),
        "recall": float(recall),
        "k_estimate": int(k_estimate) if k_estimate else None,
        "grade3_in_top_k": n_top_k,
        "grade3_at_full": int(hits_full) if hits_full is not None else None,
    }


def _exact_score_components(
    score_meta: dict, submitted: list[str], gold_ids: list[str]
) -> dict | None:
    """Derived exact-match score components shared by the prose and JSON
    score diagnostics — the single home of the dedup/cap and
    matched/missed derivations. Returns None when the scorer didn't
    produce its normal component metrics."""
    precision = score_meta.get("precision")
    recall = score_meta.get("known_recall_at_full")
    f1 = score_meta.get("standard_f1")
    if precision is None or recall is None or f1 is None:
        return None
    unique_submitted = list(dict.fromkeys(submitted))[:250]
    gold_set = set(gold_ids)
    submitted_set = set(unique_submitted)
    hits = score_meta.get("relevant_predictions_at_full")
    return {
        "score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "hits": int(hits) if hits is not None else None,
        "n_submitted": len(unique_submitted),
        "n_gold": len(gold_ids),
        "matched_gold_ids": [p for p in unique_submitted if p in gold_set],
        "missed_gold_ids": [g for g in gold_ids if g not in submitted_set],
    }


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
        c = _semantic_score_components(score_meta, k_estimate)
        if c is None:
            return None
        k = c["k_estimate"]
        n_top_k = c["grade3_in_top_k"]
        hits_full = c["grade3_at_full"]
        if k:
            recall_note = (
                f"({n_top_k} of K={k} estimated relevant found in "
                f"your top K; only grade-3 papers count)"
            )
        else:
            recall_note = "(K unknown; only grade-3 papers count)"
        lines = [
            f"rank   = {c['rank']:.4f}   (order quality: lower-bound-corrected "
            f"nDCG over the judged grades)",
            f"recall = {c['recall']:.4f}   {recall_note}",
            f"score  = harmonic(rank, recall) = {c['score']:.4f}",
        ]
        if hits_full is not None and n_top_k is not None and hits_full > n_top_k:
            lines += [
                "",
                f"{hits_full - n_top_k} more Perfect paper(s) ranked "
                f"below position K earned no recall credit — ordering cost "
                f"you recall.",
            ]
        lines += [
            "",
            '(per-paper grades: judge_verdicts.md · formulas: the task '
            'documentation\'s "Scoring (per query)" section)',
        ]
        return "\n".join(lines)

    # specific_f1 / metadata_f1: exact-match standard F1.
    c = _exact_score_components(score_meta, submitted, gold_ids)
    if c is None:
        return None
    matched = c["matched_gold_ids"]
    missed = c["missed_gold_ids"]

    def _frac(num, den, value: float) -> str:
        # Show the hits/N fraction only when it reproduces the scorer's
        # float — on any mismatch the scorer's own number wins, alone.
        if num is None or not den or abs(num / den - value) > 1e-6:
            return f"{value:.4f}"
        return f"{int(num)}/{den} = {value:.4f}"

    hits_shown = c["hits"] if c["hits"] is not None else len(matched)
    lines = [
        f"submitted: {c['n_submitted']} unique paper id(s) (scorer reads "
        f"first 250) · gold: {c['n_gold']}",
        f"hits (submitted ∩ gold): {hits_shown}"
        + (f" → {', '.join(matched)}" if matched else ""),
        f"missed gold ids: {', '.join(missed) if missed else '(none)'}",
        "",
        f"precision = hits / #submitted = "
        f"{_frac(c['hits'], c['n_submitted'], c['precision'])}",
        f"recall    = hits / #gold      = "
        f"{_frac(c['hits'], c['n_gold'], c['recall'])}",
        f"score     = harmonic(precision, recall) = {c['score']:.4f}",
    ]
    return "\n".join(lines)


def _score_meta_json(
    score_type: str,
    score_meta: dict,
    submitted: list[str],
    gold_ids: list[str],
    k_estimate: int | None,
) -> str | None:
    """Machine-readable sibling of score_calculation.md: the same scorer
    component metrics, serialized instead of rendered into prose, so
    analysis scripts read exact floats rather than regexing formula
    text. Same sources and same None conditions as the markdown; no
    extra LLM calls.
    """
    if score_type.startswith("semantic"):
        c = _semantic_score_components(score_meta, k_estimate)
    else:
        c = _exact_score_components(score_meta, submitted, gold_ids)
    if c is None:
        return None
    return json.dumps({"score_type": score_type, **c}, indent=2)


def _verdict_states(
    submitted_ids: list[str], known_good: set[str]
) -> tuple[int | None, list[tuple[int, str, str, str | None]]] | None:
    """Per-paper judge outcome states shared by the prose and JSON verdict
    diagnostics — the single home of the status-inference ordering
    (known-good → judged → beyond-cap → judge-failed → no-record), so the
    two renderings cannot drift.

    Sourced from the grounded judge's in-process record of THIS evaluation
    (grounding.last_judgements()), not the persistent cache file — so the
    verdicts reflect the judgement rendered on this agent's own (grounded)
    evidence, never a stale entry another agent wrote for the same paper.
    Zero extra LLM calls.

    Training caps judging at the top-`cap` submitted papers (recall depth);
    deeper papers are intentionally not judged and get their own state
    rather than a judge-failure — otherwise evolution would chase
    "missing" verdicts on papers that cannot affect the score.

    Returns (cap, states) with states as (position, paper_id, status,
    raw_label) in submitted order — status one of "judged", "known_good"
    (pre-seeded Perfect, never LLM-judged), "duplicate" (repeat of a
    verdict-holding paper: the scorer's recall window slices the raw
    submission filtered only by verdict membership, so the repeat consumes
    a top-K slot while adding nothing), "beyond_scored_depth",
    "judge_call_failed" (judge-side error: the paper is excluded from both
    the rank sequence and recall — neither credited nor penalized), or
    "no_verdict_recorded" — or None when there is nothing to report (no
    submissions, or no grounded judgements and nothing known-good)."""
    if not submitted_ids:
        return None
    raw_judgements = grounding.last_judgements()
    if not raw_judgements and not known_good:
        return None
    # Normalize judgement keys to match the normalized submitted_ids.
    judged = {
        normalize_corpus_id(str(k)): str(v) for k, v in raw_judgements.items()
    }
    cap = grounding.last_cap()
    states = []
    seen: set[str] = set()
    for i, pid in enumerate(submitted_ids[:250], 1):  # scorer reads first 250
        if pid in seen and (pid in judged or pid in known_good):
            # A repeat of a paper WITHOUT a verdict is position-noise the
            # scorer ignores entirely; those keep their positional status.
            status, label = "duplicate", None
        elif pid in known_good:
            status, label = "known_good", "perfectly_relevant_papers"
        elif pid in judged:
            status, label = "judged", judged[pid]
        elif cap is not None and i > cap:
            status, label = "beyond_scored_depth", None
        elif judged:
            status, label = "judge_call_failed", None
        else:
            status, label = "no_verdict_recorded", None
        seen.add(pid)
        states.append((i, pid, status, label))
    return cap, states


def _judge_verdicts_markdown(
    submitted_ids: list[str], query_id: str, known_good: set[str]
) -> str | None:
    """Prose rendering of the per-paper judge verdicts for a semantic
    query, in submitted order, with a summary-count footer. State
    semantics, sources, and None conditions live in _verdict_states; a
    judge-call failure additionally gets a footer line saying it is
    judge-side and neutral, so evolution doesn't misread the gap as
    agent-caused or as a 0."""
    res = _verdict_states(submitted_ids, known_good)
    if res is None:
        return None
    _cap, states = res

    # The judge stores astabench's internal label strings
    # ("perfectly_relevant_papers", ...); translate to the human labels.
    pretty = {
        "perfectly_relevant_papers": "Perfectly Relevant",
        "highly_relevant_papers": "Highly Relevant",
        "somewhat_relevant_papers": "Somewhat Relevant",
        "not_relevant_papers": "Not Relevant",
    }
    perfect_raw = "perfectly_relevant_papers"
    status_text = {
        "duplicate": "(duplicate submission — consumes a recall slot, adds nothing)",
        "beyond_scored_depth": "(beyond scored depth — not judged)",
        "judge_call_failed": "(judge call failed — excluded from scoring)",
        "no_verdict_recorded": "(no verdict recorded)",
    }

    lines = []
    n_perfect = n_lower = n_unknown = n_beyond = n_dup = 0
    for i, pid, status, label in states:
        if status == "known_good":
            verdict = "Perfectly Relevant (known-good)"
            n_perfect += 1
        elif status == "judged":
            verdict = pretty.get(label, label)
            if label == perfect_raw:
                n_perfect += 1
            else:
                n_lower += 1
        elif status == "duplicate":
            verdict = status_text[status]
            n_dup += 1
        elif status == "beyond_scored_depth":
            verdict = status_text[status]
            n_beyond += 1
        else:
            verdict = status_text[status]
            n_unknown += 1
        lines.append(f"{i}. {pid} — {verdict}")
    n_submitted = len(lines)
    tail = f" / {n_beyond} beyond scored depth" if n_beyond else ""
    dup_tail = f" / {n_dup} duplicate" if n_dup else ""
    lines.append(
        f"\n{n_perfect} Perfect / {n_lower} lower / {n_unknown} no verdict"
        f"{dup_tail}{tail}, of {n_submitted} submitted"
    )
    if n_dup:
        lines.append(
            "Duplicate submissions are pure waste: each repeat of a paper "
            "that already has a verdict consumes one top-K recall slot and "
            "adds nothing to rank or recall — submit each paper once."
        )
    if any(status == "judge_call_failed" for _, _, status, _ in states):
        lines.append(
            "Judge-call failures are a judge-side error (rare, ~1%), unrelated "
            "to your submission or evidence: the paper is excluded from both "
            "the rank sequence and recall — neither credited nor penalized."
        )
    return "\n".join(lines)


def _judge_verdicts_json(
    submitted_ids: list[str], known_good: set[str]
) -> str | None:
    """Machine-readable sibling of judge_verdicts.md: per-paper judge
    outcomes in submitted order, with the markdown's parenthetical
    per-paper states carried as an explicit `status` field and labels
    kept as the judge's raw strings ("perfectly_relevant_papers", ...).
    State semantics, sources, and None conditions live in
    _verdict_states."""
    res = _verdict_states(submitted_ids, known_good)
    if res is None:
        return None
    cap, states = res
    papers = [
        {"position": i, "paper_id": pid, "status": status, "label": label}
        for i, pid, status, label in states
    ]
    return json.dumps({"scored_depth_cap": cap, "papers": papers}, indent=2)


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
        f"by ` ... ` (≤8 passages).",
        "",
    ]
    for pid, dropped, raw, kind in blanked[:50]:
        first = (dropped[0] if dropped else (raw or ""))[:160]
        lines.append(f"- {pid} [{kind}]: dropped passage → {first!r}")
    if len(blanked) > 50:
        lines.append(f"- … and {len(blanked) - 50} more")
    return "\n".join(lines)


def _evidence_truncation_markdown() -> str | None:
    """Feedback on evidence clipped by the per-paper char cap
    (PF_EVIDENCE_CHAR_CAP, training-only).

    The cap is ENFORCED, not advisory: characters beyond it never reach
    the grounding check or the judge, so criteria support that lives past
    the cap earns nothing. Surfacing which papers were clipped (and by
    how much) is what lets evolution adapt — select and densify the
    strongest passages within the cap instead of concatenating
    everything. Returns None when nothing was clipped (cap off, or all
    evidence compliant — the target end-state)."""
    truncated = grounding.last_truncations()
    if not truncated:
        return None
    cap = truncated[0][2]
    total_cut = sum(orig - capped for _, orig, capped in truncated)
    lines = [
        f"{len(truncated)} paper(s) had evidence truncated at the enforced "
        f"{cap}-character cap ({total_cut:,} chars discarded unseen — the "
        f"judge and grounding check never read past the cap). Fit your "
        f"strongest per-criterion passages within the cap.",
        "",
    ]
    for pid, orig, capped in truncated[:50]:
        lines.append(f"- {pid}: {orig:,} → {capped:,} chars")
    if len(truncated) > 50:
        lines.append(f"- … and {len(truncated) - 50} more")
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


def _tool_failure_summary(exc: BaseException) -> str | None:
    """Root cause of a leaked tool-call failure, or None to re-raise as-is.

    Errors that escape astabench's retry wrapper reach the agent as anyio
    TaskGroup ExceptionGroups whose str() is "unhandled errors in a
    TaskGroup (1 sub-exception)" — the actual cause (rate limit with the
    retry budget exhausted, transport timeout, broken connection) is only
    in the leaves. Run asta_paper_finder_20260716_072622 showed evolution
    misreading exactly that string as generic tool flakiness for 20
    iterations; naming the cause is what lets it respond. Only exception
    groups are rewritten — a bare exception is already legible.
    """
    if not isinstance(exc, BaseExceptionGroup):
        return None
    import anyio
    import httpx

    leaves: list[BaseException] = []

    def _collect(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                _collect(sub)
        else:
            leaves.append(e)

    _collect(exc)

    def _label(leaf: BaseException) -> str:
        if isinstance(leaf, httpx.HTTPStatusError):
            code = leaf.response.status_code
            if code == 429:
                return "HTTP 429 rate-limited (retry budget exhausted)"
            if code in (500, 502, 503):
                return (
                    f"HTTP {code} server error (retried with backoff; "
                    f"retry budget exhausted)"
                )
            return f"HTTP {code}"
        if isinstance(leaf, httpx.TimeoutException):
            return f"transport timeout ({type(leaf).__name__})"
        if isinstance(leaf, anyio.BrokenResourceError):
            return "connection broken mid-call (BrokenResourceError)"
        msg = str(leaf).strip()
        return f"{type(leaf).__name__}: {msg[:200]}" if msg else type(leaf).__name__

    return "; ".join(dict.fromkeys(_label(leaf) for leaf in leaves)) or None


def _wrap_tools_for_provenance(tools: list) -> list:
    """Wrap each corpus tool so its result payload is recorded for evidence
    grounding, then handed back to the agent unchanged.

    This is the only seam we own between the Asta MCP tools and the agent, so
    it is where we learn what text the agent actually retrieved for each paper.
    Wrapping via ToolDef round-trips the tool's name/description/parameters
    (the schema inspect needs), the same mechanism astabench uses to install
    its retry wrapper. Recording is best-effort and never alters the result.

    Failures that leak through the retry wrapper are re-raised with the root
    cause named (see _tool_failure_summary). Only Exception is caught:
    a BaseExceptionGroup carrying cancellation must propagate untouched or
    asyncio's cancellation semantics break.
    """
    from inspect_ai.tool import ToolDef

    wrapped = []
    for t in tools:
        td = ToolDef(t)
        orig = td.tool

        def _make(orig_call):
            async def _recording(*args, **kwargs):
                try:
                    result = await orig_call(*args, **kwargs)
                except Exception as e:
                    summary = _tool_failure_summary(e)
                    if summary:
                        raise RuntimeError(f"tool call failed: {summary}") from e
                    raise
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
    TRANSIENT rate limit — sustained overrun still raises once the retry
    budget is spent. We add an outer wrapper that records each result
    payload for evidence grounding (see _grounding.py) and re-raises
    leaked failures with the root cause named (_tool_failure_summary).
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
        cost_per_error: float | None = None,
        persist_full_evidence: bool = False,
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
        # Defaulted here rather than in the signature because it is
        # relative to another argument (COST_PER_ERROR_FRACTION of the
        # free zone) — a signature default would freeze it at the
        # MIN_COST_THRESHOLD constant and silently mis-scale any caller
        # that moved the threshold.
        if cost_per_error is None:
            cost_per_error = default_cost_per_error(min_cost_threshold)
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
        # Test-only. Keeps evidence for results beyond the judging cap so a
        # stored test eval can be rejudged offline at any depth later.
        # Default False = the frozen TRAINING behaviour that background.md
        # describes to evolution sessions; main.py flips it on the test
        # evaluator alone, where evolution never reads the output.
        self.persist_full_evidence = persist_full_evidence

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
            "persist_full_evidence": self.persist_full_evidence,
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
        # path (tests, smoke runs) reuses one process across samples. Judge
        # format-repair counters reset alongside for the same reason.
        grounding.reset()
        import _judge_normalize
        _judge_normalize.reset()

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
                        # Prompt-cache splits (subset of input_tokens).
                        # Informational for most models; priced at the
                        # cached rate for JUDGE_PRICE_OVERRIDES entries.
                        # Measured near-zero for the astabench judge
                        # prompt (varying doc text precedes any stable
                        # prefix), but recorded so caching can never be
                        # silently overbilled.
                        "input_tokens_cache_read": getattr(u, "input_tokens_cache_read", None) or 0,
                        "input_tokens_cache_write": getattr(u, "input_tokens_cache_write", None) or 0,
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

            submission = _submission_json(
                completion, score_type,
                grounding.last_cap() if score_type.startswith("semantic") else None,
                persist_full_evidence=self.persist_full_evidence,
            )
            if submission:
                diagnostics["submission.json"] = submission

            score_meta = (
                dict(getattr(score_obj, "metadata", None) or {})
                if score_obj is not None else {}
            )
            if score_meta:
                k_estimate = _lookup_k_estimate(str(example.id))
                calc = _score_calculation_markdown(
                    score_type, score_meta, submitted, gold_ids, k_estimate,
                )
                if calc:
                    diagnostics["score_calculation.md"] = calc
                meta_json = _score_meta_json(
                    score_type, score_meta, submitted, gold_ids, k_estimate,
                )
                if meta_json:
                    diagnostics["score_meta.json"] = meta_json
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
                verdicts_json = _judge_verdicts_json(submitted, known_good)
                if verdicts_json:
                    diagnostics["judge_verdicts.json"] = verdicts_json
                grounding_md = _evidence_grounding_markdown()
                if grounding_md:
                    diagnostics["evidence_grounding.md"] = grounding_md
                truncation_md = _evidence_truncation_markdown()
                if truncation_md:
                    diagnostics["evidence_truncation.md"] = truncation_md
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
        # Judge-only override table (models newer than the pinned litellm;
        # see JUDGE_PRICE_OVERRIDES). Cache-aware: cache-read tokens bill
        # at the cached rate, the remainder at the full input rate. Checked
        # before the bundled map so the pinned-litellm gap can't zero a
        # judge's cost; agent models are never in this table, so the
        # leaderboard billing basis is untouched.
        override = JUDGE_PRICE_OVERRIDES.get(litellm_name.split("/", 1)[-1])
        if override:
            cache_read = counts.get("input_tokens_cache_read", 0) or 0
            cache_read = min(cache_read, input_tokens)
            return (
                (input_tokens - cache_read) * override["input_cost_per_token"]
                + cache_read * override.get(
                    "cached_input_cost_per_token", override["input_cost_per_token"]
                )
                + completion_tokens * override["output_cost_per_token"]
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
