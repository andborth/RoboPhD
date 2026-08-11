#!/usr/bin/env python3
"""Offline re-judge of a completed test-set eval with a different judge model.

A test eval persists everything the judge saw under test_problems/<sid>/
(ordered submission.json with the exact evidence, gold_criteria.md,
judge_verdicts.json with the scored-depth cap, score_meta.json with the
recall denominator K). This tool replays those stored submissions through a
new judge basis (model + prompt profile) and recomputes adjusted F1 —
without re-running the agent.

Scoring uses CANONICAL (submission-order) verdict ordering for every pass.
The live scoring path builds the nDCG grade list with cache-hit verdicts at
their submission position but freshly-judged ones appended after the loop,
so stored per-query rank values depend on judge-cache state at eval time.
A new judge cold-misses everything, so comparing it against the stored
aggregate would conflate the judge change with the ordering artifact.
Therefore the default A/B recomputes a stock GPT-4o baseline under the same
canonical ordering (cheap: the stock test cache already holds nearly all
verdicts) and reports the new judge against THAT.

Verdicts are read from and written to the same shared per-basis test caches
a real `--eval-only --training-judge` run uses (shared_test_<basis>.json),
so a later live eval with the new judge starts warm.

Usage:
    python rejudge_test.py <run_dir> --judge openai/gpt-5.6-luna
        [--from-eval-log PATH] [--k-from RUN_DIR] [--cap-to-k] [--no-baseline]
        [--limit N] [--uncapped] [--concurrency 4] [--retries 1]
        [--no-cache-write] [--dry-run] [--force]

--from-eval-log replays an OFFICIAL astabench .eval log instead of the run's
own test_problems/. That is the only stored source of an UNCAPPED,
leaderboard-basis submission set for runs completed before 2026-08-06, since
submission.json omitted beyond-cap evidence until then — and it is the more
useful basis regardless, being the submissions and depth that actually
reached the board. run_dir is still required: it supplies k_estimate per
query (a property of the query, not the agent) and hosts the diagnostics,
which land in a dedicated rejudge_officiallog/<sid>/ tree rather than in
test_problems/. The stock-4o A/B arm is skipped in this mode — it would be a
full cold pass (~$186 on a 250-deep agent) and the log already carries
official 4o scores. Measured on -012: 48,255 scoreable docs, ~$14 on luna.

--k-from borrows k_estimate from another completed run of the same split.
Required with --from-eval-log for any run that timed out on a semantic query:
the official log carries all 267 samples, but K is stored only in the run's
own test_problems/, which lacks the failed ones (-010 lacks semantic_242;
-011 lacks four), so a same-dir replay hard-errors on exactly those queries.
K is a per-query benchmark constant — verified identical for every shared sid
across -010/-011/-012/-013/-014 — so any 267-complete run serves.

By default the replay stops at the stored scored_depth_cap, matching how the
eval judged. --uncapped judges the whole submission instead — the basis
official astabench scoring uses. k_estimate still governs the recall window,
so the extra verdicts reach only the rank term. Outputs carry an .uncapped
tag; the verdict cache is shared with the capped pass, since a verdict is
keyed by (query, paper, evidence) and does not depend on judging depth.

--cap-to-k is the mirror image, for --from-eval-log only: official submissions
were judged uncapped, and this replays them at the INTERNAL depth instead
(cap = k_estimate, which is the internal rule exactly — verified equal on all
194 of -013's semantic queries). Run a log plain and then again with
--cap-to-k and the pair isolates the depth axis with the agent draw held
fixed, which no comparison between an internal eval and an official one can
do. The second pass is free: same cache, no new verdicts. Without it, depth is
confounded with agent draw in every internal-vs-official comparison.

--dry-run reports, per basis, how much of this run's judging the cache
already covers and what the remainder would cost. A cold target cache is
normal — cache keys include an evidence hash, so a new agent lineage shares
almost nothing with earlier work — and the cost is what decides whether to
proceed.

The judge-prompt profile is a property of the judge (main._prompt_for_judge:
gpt-4o -> stock, luna -> no-prose), matching what a live --training-judge /
--test-judge eval would run — a rejudge exists to predict those. Outputs
from before this derivation (stock-prompt luna) remain on disk under their
own basis-suffixed names and caches.

The two test_results.rejudge_* outputs are non-clobbering (--force to
overwrite). The per-problem judge_verdicts.rejudge_<basis>.json diagnostics
are deliberately NOT gated: an interrupted run must be able to resume, and a
same-basis rerun replays identical cached verdicts anyway.

Requires OPENAI_API_KEY (both supported judges are OpenAI models).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Stale judge-override env would make evaluator's import-time
# _apply_training_grader() install the lenient normalizer, which is not
# reversible — and the stock baseline pass must run with the strict parser.
for _k in ("PF_TRAINING_GRADER_MODEL", "PF_TRAINING_GRADER_PROMPT"):
    os.environ.pop(_k, None)

from evaluator import (  # noqa: E402
    EVIDENCE_OMITTED_MARKER,
    PaperFinderEvaluator,
    TRAINING_GRADER_ENV,
    TRAINING_GRADER_PROMPT_ENV,
    _apply_training_grader,
    _safe_cache_rmw,
)
from main import (  # noqa: E402
    JUDGE_CHOICES,
    _judge_basis_slug,
    _judge_cache_dir,
    _prompt_for_judge,
)
import _grounding  # noqa: E402
from _check_judge_calibration import _extract_results_lenient  # noqa: E402

from astabench.evals.paper_finder import relevance as rel  # noqa: E402
from astabench.evals.paper_finder.eval import (  # noqa: E402
    _calc_any_f,
    lower_bound_corrected_ndcg,
)
from astabench.evals.paper_finder.paper_finder_utils import (  # noqa: E402
    MAX_RESULTS_TO_CONSIDER,
)
from astabench.evals.paper_finder.relevance import (  # noqa: E402
    Relevance,
    RelevanceCriterion,
    bench_rj_2_pf_rj,
)

STOCK = JUDGE_CHOICES[0]


@dataclass
class Sample:
    sid: str
    score_type: str
    stored_score: float
    problem_dir: Path
    carry: bool = False
    carried_reason: str = ""
    results: list = field(default_factory=list)  # [(paper_id, evidence)] in submission order
    known_good: set = field(default_factory=set)
    criteria: list = field(default_factory=list)  # raw relevance_criteria dicts
    cap: int | None = None
    k_estimate: int | None = None


def load_run(run_dir: Path, limit: int | None = None) -> list[Sample]:
    """Load every test problem; exact-match and unreconstructable queries are
    marked carry (stored score reused verbatim). Hard-stops on missing data
    that would silently skew the aggregate."""
    tp = run_dir / "test_problems"
    if not tp.is_dir():
        raise SystemExit(f"{tp} not found — is this a completed test-eval run dir?")
    scrubbed = sorted(p.parent.name for p in tp.glob("*/evidence_grounding.md"))
    if scrubbed:
        raise SystemExit(
            f"{len(scrubbed)} problems had evidence scrubbed by the grounding "
            f"check at eval time (e.g. {scrubbed[0]}); submission.json holds "
            f"pre-scrub evidence there, so an offline rejudge would judge text "
            f"the original judge never saw. Refusing."
        )
    samples: list[Sample] = []
    for pdir in sorted(tp.iterdir()):
        if not pdir.is_dir():
            continue
        try:
            result = json.loads((pdir / "result.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"{pdir.name}: unreadable result.json ({e})")
        sid = result.get("sample_id") or pdir.name
        s = Sample(
            sid=sid,
            score_type=result.get("score_type") or "",
            stored_score=float(result.get("score") or 0.0),
            problem_dir=pdir,
        )
        if not sid.startswith("semantic"):
            # specific_*/metadata_* are exact-match scored — no LLM judge.
            s.carry, s.carried_reason = True, "exact_match"
            samples.append(s)
            continue
        try:
            text = (pdir / "submission.json").read_text()
        except OSError:
            text = ""
        try:
            results = (json.loads(text).get("output") or {}).get("results") or []
            results = [r for r in results if isinstance(r, dict)]
        except (json.JSONDecodeError, AttributeError):
            results = _extract_results_lenient(text)
        pairs = [
            (str(r.get("paper_id", "")).strip(), r.get("markdown_evidence") or "")
            for r in results
            if str(r.get("paper_id", "")).strip()
        ]
        if not pairs:
            s.carry, s.carried_reason = True, "no_submission"
            samples.append(s)
            continue
        try:
            gold = json.loads((pdir / "gold_criteria.md").read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"{sid}: unreadable gold_criteria.md ({e})")
        s.known_good = {str(x) for x in (gold.get("known_to_be_good") or [])}
        s.criteria = gold.get("relevance_criteria") or []
        try:
            s.cap = json.loads((pdir / "judge_verdicts.json").read_text()).get(
                "scored_depth_cap"
            )
        except (OSError, json.JSONDecodeError):
            s.cap = None
        try:
            s.k_estimate = json.loads((pdir / "score_meta.json").read_text()).get(
                "k_estimate"
            )
        except (OSError, json.JSONDecodeError):
            s.k_estimate = None
        if s.k_estimate is None:
            s.k_estimate = s.cap
        if s.k_estimate is None:
            raise SystemExit(
                f"{sid}: no k_estimate in score_meta.json and no scored_depth_cap "
                f"in judge_verdicts.json — cannot compute recall"
            )
        s.results = pairs[:MAX_RESULTS_TO_CONSIDER]
        samples.append(s)
    if limit is not None:
        semantic = [s for s in samples if not s.carry][:limit]
        samples = semantic  # smoke mode: aggregate over the selected subset only
    return samples


def load_eval_log(
    eval_path: Path,
    run_dir: Path,
    limit: int | None = None,
    k_from: Path | None = None,
):
    """Load samples from an OFFICIAL astabench .eval log instead of a RoboPhD
    run's test_problems/.

    Why this exists: official runs judge the FULL submission, and their logs
    keep the real evidence for every result (our own submission.json omitted
    beyond-cap evidence until 2026-08-06). So an official log is the only
    place a stored, uncapped, leaderboard-basis submission set can be found
    for runs already completed — which makes it the right place to measure a
    judge-basis change against the numbers that actually reached the board.

    The log lacks two things, sourced separately:

      * k_estimate per query, from <k_source>/test_problems/<sid>/
        score_meta.json, where <k_source> is ``--k-from`` when given and
        run_dir otherwise. K is a property of the QUERY (the benchmark's
        recall denominator, read from astabench's normalizer reference), not
        of the agent — verified across -010/-011/-012/-013/-014: every
        shared sid agrees, zero mismatches, and the three 267-complete runs
        each total 19,860. So any completed run of the same split serves,
        and ``--k-from`` exists because a run's OWN test_problems is missing
        exactly the samples that timed out (-010 lacks semantic_242; -011
        lacks four) while the official log has all 267 — so replaying a run
        against its own dir hard-errors on precisely the queries it failed.

      * a home for the per-problem verdict diagnostics: a dedicated
        <run_dir>/rejudge_officiallog/<sid>/ tree, NOT test_problems/.
        Deliberate — these verdicts grade the official submission, not this
        run's internal eval, and writing them into test_problems/ would have
        to mkdir the very sids the eval never produced, inflating the
        directory count that scripts read as "samples evaluated".

    Samples carry cap=None: official judging is uncapped by construction.
    """
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(eval_path))
    if not log.samples:
        raise SystemExit(f"{eval_path}: no samples in log")
    tp = (k_from or run_dir) / "test_problems"
    if not tp.is_dir():
        raise SystemExit(
            f"{tp} not found — --from-eval-log needs a completed run dir of "
            f"the same test split for k_estimate (this run's own dir, or "
            f"another one via --k-from)"
        )
    diag_root = run_dir / "rejudge_officiallog"
    samples: list[Sample] = []
    for x in log.samples:
        sid = str(x.id)
        score_type = (x.metadata or {}).get("score_type") or ""
        score_obj = next(iter((x.scores or {}).values()), None)
        stored = float(getattr(score_obj, "value", 0.0) or 0.0)
        s = Sample(
            sid=sid, score_type=score_type, stored_score=stored,
            problem_dir=diag_root / sid,
        )
        if not sid.startswith("semantic"):
            s.carry, s.carried_reason = True, "exact_match"
            samples.append(s)
            continue
        completion = (getattr(x.output, "completion", "") or "") if x.output else ""
        try:
            results = (json.loads(completion).get("output") or {}).get("results") or []
            results = [r for r in results if isinstance(r, dict)]
        except (json.JSONDecodeError, AttributeError):
            results = _extract_results_lenient(completion)
        pairs = [
            (str(r.get("paper_id", "")).strip(), r.get("markdown_evidence") or "")
            for r in results
            if str(r.get("paper_id", "")).strip()
        ]
        if not pairs:
            s.carry, s.carried_reason = True, "no_submission"
            samples.append(s)
            continue
        target = x.target if isinstance(x.target, str) else (x.target or [""])[0]
        try:
            gold = json.loads(target)
        except (json.JSONDecodeError, TypeError):
            raise SystemExit(f"{sid}: unparseable target in {eval_path.name}")
        s.known_good = {str(v) for v in (gold.get("known_to_be_good") or [])}
        s.criteria = gold.get("relevance_criteria") or []
        try:
            s.k_estimate = json.loads(
                (tp / sid / "score_meta.json").read_text()
            ).get("k_estimate")
        except (OSError, json.JSONDecodeError):
            s.k_estimate = None
        if s.k_estimate is None:
            raise SystemExit(
                f"{sid}: no k_estimate at {tp / sid / 'score_meta.json'} — "
                f"cannot compute recall. Pass --k-from <run_dir> pointing at a "
                f"267-complete run of the same test split (K is a per-query "
                f"benchmark constant, so any of them serves)."
            )
        s.cap = None  # official judging is uncapped
        s.results = pairs[:MAX_RESULTS_TO_CONSIDER]
        samples.append(s)
    if limit is not None:
        samples = [s for s in samples if not s.carry][:limit]
    return samples


def plan_sample(sample: Sample, cache_q: dict):
    """Mirror the live judging loop (sans grounding, a no-op for persisted
    evidence): per position decide preset verdict / judge call / exclusion.

    Returns (order, preset, to_judge, statuses):
      order    — first-occurrence submission order of scoreable paper_ids
      preset   — {pid: label} for known-good / cache-hit / empty-evidence
      to_judge — [(pid, evidence, cache_key)] needing a judge call
      statuses — per-position (pid, status) for the verdict diagnostic
    """
    order: list[str] = []
    preset: dict[str, str] = {}
    to_judge: list[tuple[str, str, str]] = []
    statuses: list[tuple[str, str]] = []
    seen: set[str] = set()
    for idx, (pid, ev) in enumerate(sample.results):
        if sample.cap is not None and idx >= sample.cap:
            statuses.append((pid, "beyond_scored_depth"))
            continue
        if pid in seen:
            # A later duplicate can neither add a verdict nor move the first
            # occurrence's position (dict semantics upstream).
            statuses.append((pid, "duplicate"))
            continue
        seen.add(pid)
        if pid in sample.known_good:
            order.append(pid)
            preset[pid] = Relevance.PERFECT.value
            statuses.append((pid, "known_good"))
            continue
        if ev == EVIDENCE_OMITTED_MARKER:
            # Beyond-cap remnant in submission.json; belt-and-braces with the
            # cap check above.
            statuses.append((pid, "beyond_scored_depth"))
            continue
        if not ev.strip():
            order.append(pid)
            preset[pid] = Relevance.NOT_RELEVANT.value
            statuses.append((pid, "empty_evidence"))
            continue
        order.append(pid)
        ckey = _grounding.cache_key(pid, ev)
        hit = cache_q.get(ckey)
        if hit is not None:
            preset[pid] = hit
            statuses.append((pid, "cached"))
        else:
            to_judge.append((pid, ev, ckey))
            statuses.append((pid, "pending"))
    return order, preset, to_judge, statuses


def score_sample(sample: Sample, judgements: dict[str, str]) -> dict:
    """Adjusted F1 from canonical-order judgements. Same math as astabench's
    calc_adjusted_f1, with K read from the stored k_estimate instead of the
    normalizer reference (no HF download, no astabench global state)."""
    grades = [bench_rj_2_pf_rj[v] for v in judgements.values()]
    rank = lower_bound_corrected_ndcg(grades)["rank"]
    k = sample.k_estimate or 0
    # Upstream parity (calc_recall_at_k): the K-window slices the RAW
    # submission, so a duplicated pid inside the window consumes two of the
    # K slots. Deduping here would diverge from official scoring.
    top_k = set(
        [pid for pid, _ in sample.results if pid in judgements][:k]
    )
    hits = sum(
        1
        for pid, label in judgements.items()
        if label == Relevance.PERFECT.value and pid in top_k
    )
    recall = hits / k if k else 0.0
    return {
        "score": _calc_any_f([rank, recall]),
        "rank": rank,
        "recall": recall,
        "k_estimate": k,
        "grade3_in_top_k": hits,
    }


class _UsageMeter:
    """Accumulate judge token usage by wrapping relevance.get_model — the
    only usage source in this direct-call path (there is no inspect eval
    log). load_relevance_judgement resolves get_model from its module
    namespace at call time, so rebinding the attribute captures every call."""

    def __init__(self):
        self.by_model: dict[str, dict] = {}
        self._orig = None

    def install(self):
        self._orig = rel.get_model
        meter = self

        def metered_get_model(name, *a, **kw):
            return _MeteredModel(meter._orig(name, *a, **kw), meter, str(name))

        rel.get_model = metered_get_model

    def uninstall(self):
        if self._orig is not None:
            rel.get_model = self._orig
            self._orig = None

    def add(self, model: str, usage) -> None:
        acc = self.by_model.setdefault(model, {})
        for f in (
            "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
            "input_tokens_cache_read", "input_tokens_cache_write",
        ):
            acc[f] = acc.get(f, 0) + (getattr(usage, f, None) or 0)

    def cost(self) -> float:
        return sum(
            PaperFinderEvaluator._estimate_cost(m, c)
            for m, c in self.by_model.items()
        )


class _MeteredModel:
    def __init__(self, real, meter, name):
        self._real, self._meter, self._name = real, meter, name

    async def generate(self, *a, **kw):
        out = await self._real.generate(*a, **kw)
        usage = getattr(out, "usage", None)
        if usage is not None:
            self._meter.add(str(getattr(self._real, "name", self._name)), usage)
        return out

    def __getattr__(self, item):
        return getattr(self._real, item)


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError:
        return {}
    except json.JSONDecodeError as e:
        print(f"WARNING: judge cache {path} unparseable ({e}); treating as empty")
        return {}


async def _judge_batch(criteria_raw: list, docs: list) -> dict:
    """One judge call over (pid, evidence) docs; {pid: label}. Resolved via
    the relevance module at call time so grader/normalizer patches apply."""
    criteria = [
        RelevanceCriterion(
            name=c["name"], description=c["description"], weight=c["weight"]
        )
        for c in criteria_raw
    ]
    entities = [rel.Document(corpus_id=pid, markdown=ev) for pid, ev in docs]
    return await rel.load_relevance_judgement(entities, criteria)


async def _process_sample(sample, cache, cache_path, sem, args, progress):
    cache_q = cache.get(sample.sid) or {}
    order, preset, to_judge, statuses = plan_sample(sample, cache_q)
    fresh: dict[str, str] = {}
    if to_judge:
        async with sem:
            pending = to_judge
            for _attempt in range(args.retries + 1):
                res = await _judge_batch(
                    sample.criteria, [(p, e) for p, e, _ in pending]
                )
                fresh.update(res)
                # The judge silently drops docs whose response fails to
                # parse or names the wrong criteria; re-ask just those.
                pending = [t for t in pending if t[0] not in fresh]
                if not pending:
                    break
        if fresh and not args.no_cache_write:
            writes = {ck: fresh[p] for p, _e, ck in to_judge if p in fresh}
            if writes:
                await asyncio.to_thread(
                    _safe_cache_rmw, str(cache_path), sample.sid, writes
                )
    # Canonical ordering: every verdict at its submission position.
    merged = {**preset, **fresh}
    judgements = {pid: merged[pid] for pid in order if pid in merged}
    scored = score_sample(sample, judgements)
    n_failed = sum(1 for p, _e, _c in to_judge if p not in fresh)

    def _final_status(pid: str, st: str) -> str:
        if st == "pending":
            return "judged" if pid in fresh else "judge_call_failed"
        return st

    # 1-based positions to match the sibling judge_verdicts.json the
    # evaluator writes — consumers diff the two files per position. Same
    # duplicate rule as the evaluator's _verdict_states (by rule, not just
    # by name): a repeat of a VERDICT-HOLDING paper is "duplicate" with
    # label None — including a repeat that fell beyond the cap, since the
    # recall window filters by membership, position-blind — while a repeat
    # of a no-verdict paper is scorer-invisible and mirrors its first
    # occurrence's outcome.
    papers = []
    first_status: dict[str, str] = {}
    for i, (pid, st) in enumerate(statuses, 1):
        if st == "duplicate":
            if pid in judgements:
                status, label = "duplicate", None
            else:
                status = first_status.get(pid, "judge_call_failed")
                label = None
        elif st == "beyond_scored_depth" and pid in judgements:
            # judgements holds in-cap pids only, so this pid was already
            # scored at an earlier position: a beyond-cap repeat.
            status, label = "duplicate", None
        else:
            status = _final_status(pid, st)
            label = judgements.get(pid)
            first_status.setdefault(pid, status)
        papers.append(
            {"position": i, "paper_id": pid, "status": status, "label": label}
        )
    # Overwritten (not --force-gated) by design: an interrupted run must be
    # able to resume, and a same-basis rerun replays identical cached
    # verdicts anyway; different bases write different filenames.
    _tag = ".uncapped" if getattr(args, "uncapped", False) else ""
    _tag += ".capk" if getattr(args, "cap_to_k", False) else ""
    _tag += ".officiallog" if getattr(args, "from_eval_log", None) else ""
    verdict_path = (
        sample.problem_dir / f"judge_verdicts.rejudge_{args.basis}{_tag}.json"
    )
    # test_problems/<sid>/ already exists on the run's own path; the
    # --from-eval-log path writes to a rejudge_officiallog/<sid>/ tree that
    # does not (see load_eval_log on why it is separate).
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(
        json.dumps({"scored_depth_cap": sample.cap, "papers": papers}, indent=2)
    )

    progress["done"] += 1
    print(
        f"  [{progress['done']}/{progress['total']}] {sample.sid}: "
        f"{scored['score']:.3f} (judged {len(fresh)}, cached "
        f"{sum(1 for _p, st in statuses if st == 'cached')}"
        f"{f', failed {n_failed}' if n_failed else ''})"
    )
    return {
        "sample_id": sample.sid,
        **scored,
        "n_fresh": len(fresh),
        "n_cached": sum(1 for _p, st in statuses if st == "cached"),
        "n_failed": n_failed,
    }


async def run_pass(samples, judge, judge_prompt, cache_path, args):
    """One full judging+scoring sweep for one judge basis. Returns
    {sid: row} for semantic samples plus pass-level usage/cost/cache stats."""
    if judge == STOCK and judge_prompt == "stock":
        # Stock basis must stay byte-identical to official scoring: env unset,
        # strict parser. Must run before any alternate-judge pass because the
        # lenient-normalizer install is not reversible — a plain raise, not an
        # assert, so python -O cannot strip the guard.
        if rel.GRADER_MODEL_NAME != STOCK:
            raise RuntimeError(
                "stock pass must run before any alternate-judge pass: grader "
                f"is already {rel.GRADER_MODEL_NAME!r} and the lenient "
                "normalizer may be installed"
            )
    else:
        os.environ[TRAINING_GRADER_ENV] = judge
        if judge_prompt != "stock":
            os.environ[TRAINING_GRADER_PROMPT_ENV] = judge_prompt
        _apply_training_grader()

    semantic = [s for s in samples if not s.carry]
    cache = _load_cache(cache_path)
    meter = _UsageMeter()
    meter.install()
    progress = {"done": 0, "total": len(semantic)}
    try:
        sem = asyncio.Semaphore(args.concurrency)
        rows = await asyncio.gather(
            *[
                _process_sample(s, cache, cache_path, sem, args, progress)
                for s in semantic
            ]
        )
    finally:
        meter.uninstall()
    return {
        "rows": {r["sample_id"]: r for r in rows},
        "judge_cost_usd": meter.cost(),
        "usage_by_model": meter.by_model,
    }


# Judge-call token model, for the --dry-run cost estimate only. Nothing here
# feeds a recorded number; the real cost comes from the eval's own usage.
#
# Calibrated against the one fully-measured official run, v0_0_7: the upstream
# report puts it at ~1,000 input / 165 output tokens per verdict on 976
# chars/paper of evidence, and the submit script records $192 of judge spend
# over 194 x 250 = 48,500 verdicts. A 700-token scaffold (instructions +
# criteria, everything that is not the evidence itself) reproduces that $192
# to within 1%; 750 overshoots by 4%.
_JUDGE_SCAFFOLD_TOKENS = 700
_CHARS_PER_TOKEN = 4
# Output is near-constant per verdict: 165 measured on the stock prose prompt,
# and no-prose cut output tokens 65% (2026-07-23 study).
_JUDGE_OUTPUT_TOKENS = {"stock": 165, "no-prose": round(165 * 0.35)}


def _estimate_judge_cost(to_judge_evidence, judge: str, judge_prompt: str) -> float:
    """Rough dollar cost of judging these documents fresh.

    Priced through PaperFinderEvaluator._estimate_cost so the estimate uses
    the same rate table (and the same JUDGE_PRICE_OVERRIDES entries) the real
    eval bills against — a second pricing implementation here could disagree
    with the run it is predicting.
    """
    n = len(to_judge_evidence)
    if not n:
        return 0.0
    evidence_tokens = sum(len(ev) for ev in to_judge_evidence) / _CHARS_PER_TOKEN
    tin = n * _JUDGE_SCAFFOLD_TOKENS + evidence_tokens
    tout = n * _JUDGE_OUTPUT_TOKENS.get(judge_prompt, 165)
    return PaperFinderEvaluator._estimate_cost(
        judge, {"input_tokens": tin, "output_tokens": tout,
                "total_tokens": tin + tout}
    )


def _fresh_evidence(samples, cache) -> list[str]:
    """Evidence text for every doc this basis would have to judge fresh."""
    out: list[str] = []
    for s in samples:
        if s.carry:
            continue
        _order, _preset, to_judge, _statuses = plan_sample(s, cache.get(s.sid) or {})
        out.extend(ev for _pid, ev, _ckey in to_judge)
    return out


def _coverage(samples, cache) -> tuple[int, int]:
    """(cache hits, total scoreable docs) for this run's submissions."""
    hits = total = 0
    for s in samples:
        if s.carry:
            continue
        cache_q = cache.get(s.sid) or {}
        _order, _preset, to_judge, statuses = plan_sample(s, cache_q)
        n_cached = sum(1 for _p, st in statuses if st == "cached")
        hits += n_cached
        total += n_cached + len(to_judge)
    return hits, total


def _aggregate(samples, rows: dict) -> float:
    """Plain unweighted mean over all loaded problems (matches the test
    aggregate: no cost penalty, no weighting)."""
    scores = [
        rows[s.sid]["score"] if s.sid in rows else s.stored_score for s in samples
    ]
    return sum(scores) / len(scores) if scores else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-judge a completed test eval's stored submissions with "
        "a different judge basis; no agent re-run."
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--judge", required=True, choices=JUDGE_CHOICES,
        help="judge model; its prompt profile follows (gpt-4o -> stock, "
             "luna -> no-prose), matching live --training-judge/--test-judge "
             "evals",
    )
    ap.add_argument(
        "--no-baseline", dest="baseline", action="store_false", default=True,
        help="skip the canonical stock-GPT-4o A/B pass",
    )
    ap.add_argument("--limit", type=int, help="first N semantic queries (smoke)")
    ap.add_argument(
        "--from-eval-log", type=Path, metavar="PATH",
        help="rejudge an OFFICIAL astabench .eval log instead of the run's "
             "own test_problems/. Official logs keep full evidence for every "
             "result and were judged UNCAPPED, so this measures a judge-basis "
             "change against the submissions and depth that actually reached "
             "the leaderboard. run_dir is still required: it supplies "
             "k_estimate per query (unless --k-from overrides that) and hosts "
             "the diagnostics under rejudge_officiallog/. The stock-4o "
             "A/B arm is skipped here (it would be a full cold pass, ~$190); "
             "the log's own official scores are the 4o reference.",
    )
    ap.add_argument(
        "--k-from", type=Path, metavar="RUN_DIR",
        help="source k_estimate from ANOTHER completed run's test_problems/ "
             "instead of run_dir's. Needed with --from-eval-log whenever the "
             "run being replayed timed out on a semantic query: the official "
             "log has all 267 samples but the run's own test_problems/ lacks "
             "the failed ones, and K is only stored there (-010 lacks "
             "semantic_242; -011 lacks four). Safe because K is a per-query "
             "benchmark constant, not an agent property — verified identical "
             "across every run that stores it. Point it at a 267-complete "
             "run. Diagnostics and outputs still go to run_dir.",
    )
    ap.add_argument(
        "--uncapped", action="store_true",
        help="judge EVERY submitted paper instead of stopping at the stored "
             "scored_depth_cap — the basis official astabench scoring uses. "
             "k_estimate still governs the recall window, so only the rank "
             "term sees the extra verdicts. Costs ~2.6x the capped pass "
             "(measured on -013: 18,574 in-cap vs 48,500 submitted). Outputs "
             "carry an .uncapped tag so they never clobber a capped pass; the "
             "verdict CACHE is shared, since a verdict is keyed by "
             "(query, paper, evidence) and does not depend on depth.",
    )
    ap.add_argument(
        "--cap-to-k", action="store_true",
        help="with --from-eval-log: judge the official submissions only down to "
             "k_estimate, reproducing the INTERNAL capped depth on the official "
             "submission set. Isolates the depth axis -- pair it with a plain "
             "(uncapped) pass over the same log and the only difference is how "
             "deep the judge went, with the agent draw held fixed. Costs nothing "
             "extra after that pass: a verdict is keyed by "
             "(query, paper, evidence) and does not depend on depth, so the "
             "capped arm replays entirely from the warm cache. k_estimate IS the "
             "internal cap -- verified equal on all 194 of -013's semantic "
             "queries. Outputs carry a .capk tag.",
    )
    ap.add_argument("--concurrency", type=int, default=4,
                    help="queries judged in flight (each judges all its docs at once)")
    ap.add_argument("--retries", type=int, default=1,
                    help="re-ask passes for docs the judge dropped")
    ap.add_argument("--no-cache-write", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="no LLM calls: doc counts + per-basis cache coverage")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    # No separate prompt knob: the profile is a property of the judge
    # (single-sourced from main.py), so a rejudge always measures a basis a
    # live eval can actually produce.
    args.judge_prompt = _prompt_for_judge(args.judge)
    run_dir = args.run_dir.resolve()
    args.basis = _judge_basis_slug(args.judge, args.judge_prompt)
    stock_basis = _judge_basis_slug(STOCK, "stock")
    is_stock_target = args.basis == stock_basis
    cache_dir = _judge_cache_dir(str(run_dir.parent.parent))
    if not cache_dir.is_dir() and not args.dry_run:
        print(f"NOTE: {cache_dir} does not exist yet (unusual for a completed "
              f"run — check run_dir depth); it will be created on first write")
    target_cache = cache_dir / f"shared_test_{args.basis}.json"
    stock_cache = cache_dir / f"shared_test_{stock_basis}.json"

    if args.cap_to_k and not args.from_eval_log:
        raise SystemExit(
            "--cap-to-k only applies to --from-eval-log: the run's own path "
            "already replays at the stored scored_depth_cap, which is what "
            "--cap-to-k reconstructs. Drop the flag."
        )
    if args.cap_to_k and args.uncapped:
        raise SystemExit(
            "--cap-to-k and --uncapped ask for opposite depths. An official log "
            "is uncapped by default, so run it plain for the uncapped arm and "
            "again with --cap-to-k for the capped one."
        )
    if args.k_from and not args.from_eval_log:
        raise SystemExit(
            "--k-from only applies to --from-eval-log: the run's own path reads "
            "K from the same test_problems/<sid>/ dir it reads the submission "
            "from, so borrowing K from elsewhere there would pair one run's "
            "recall denominator with another run's stored submission."
        )
    if args.from_eval_log:
        if not args.from_eval_log.is_file():
            raise SystemExit(f"{args.from_eval_log}: not a file")
        k_from = args.k_from.resolve() if args.k_from else None
        samples = load_eval_log(
            args.from_eval_log, run_dir, args.limit, k_from=k_from
        )
        if k_from:
            print(f"--k-from: k_estimate sourced from {k_from}")
        args.baseline = False  # see --from-eval-log help
        print(f"--from-eval-log: {args.from_eval_log.name} "
              f"(uncapped official submissions; stock-4o arm skipped, "
              f"comparison is against the log's official scores)")
    else:
        samples = load_run(run_dir, args.limit)
    if args.cap_to_k:
        # load_eval_log leaves cap=None (official judging is uncapped). Setting
        # it to k_estimate reconstructs the internal capped depth on the
        # official submission set -- cap == k_estimate is the internal rule,
        # not an approximation of it. plan_sample then marks everything past k
        # beyond_scored_depth, exactly as an internal eval did.
        for s in samples:
            if not s.carry:
                s.cap = s.k_estimate
        print(f"--cap-to-k: judging to k_estimate "
              f"({sum(s.cap or 0 for s in samples if not s.carry):,} of "
              f"{sum(len(s.results) for s in samples if not s.carry):,} "
              f"submitted papers)")
    if args.uncapped:
        # AFTER load_run: k_estimate falls back to cap when score_meta lacks
        # it (load_run:188), so clearing cap earlier would break the recall
        # window. Clearing it here leaves k_estimate resolved and only widens
        # judging depth — exactly the official split.
        # A rejudge cannot go deeper than the eval persisted. _submission_json
        # replaces markdown_evidence with EVIDENCE_OMITTED_MARKER on every
        # result beyond the judging cap, so the text those papers would be
        # judged on does not exist on disk. Clearing s.cap alone would sail
        # past the depth check at plan_sample and then be caught by the marker
        # check immediately below it — a silent no-op that still reported
        # "uncapped". Refuse instead.
        omitted = sum(
            1
            for s in samples
            if not s.carry
            for _pid, ev in s.results
            if ev == EVIDENCE_OMITTED_MARKER
        )
        if omitted:
            raise SystemExit(
                f"--uncapped: {omitted:,} submitted papers beyond the judging "
                f"cap have no persisted evidence (EVIDENCE_OMITTED_MARKER in "
                f"submission.json), so they cannot be judged offline.\n"
                f"  * Test evals run after 2026-08-11 keep that evidence "
                f"(PaperFinderEvaluator persist_full_evidence, which main.py "
                f"sets on the TEST evaluator only) and do support --uncapped. "
                f"NOT 2026-08-06, when the flag landed: it was dropped at the "
                f"subprocess boundary until 2026-08-11, so every test eval in "
                f"between trimmed regardless -- check for the marker rather "
                f"than trusting the run date.\n"
                f"  * For an older run, --from-eval-log <official .eval> "
                f"replays the official submissions, which were never trimmed "
                f"and were judged uncapped.\n"
                f"  * Or drop --uncapped for the capped basis."
            )
        for s in samples:
            s.cap = None
        print("--uncapped: full evidence present; judging whole submissions")
    semantic = [s for s in samples if not s.carry]
    carried = [s for s in samples if s.carry]
    print(f"Loaded {len(samples)} problems: {len(semantic)} semantic to rejudge, "
          f"{len(carried)} carried "
          f"({sum(1 for s in carried if s.carried_reason == 'exact_match')} "
          f"exact-match)")

    if args.dry_run:
        # With an empty cache every scoreable doc plans as a judge call.
        n_docs = sum(len(plan_sample(s, {})[2]) for s in semantic)
        print(f"Judge-scoreable docs (excl. known-good/empty): {n_docs}")
        for label, path, model, profile in (
            ("target", target_cache, args.judge, args.judge_prompt),
            ("stock", stock_cache, STOCK, "stock"),
        ):
            cache = _load_cache(path)
            hits, total = _coverage(samples, cache)
            pct = hits / total * 100 if total else 0.0
            fresh = _fresh_evidence(samples, cache)
            cost = _estimate_judge_cost(fresh, model, profile)
            print(f"  {label:6} cache {path.name}: {hits}/{total} ({pct:.1f}%)")
            print(f"         {len(fresh):,} fresh call(s) -> ~${cost:.2f}")
        if not args.baseline:
            print("  (--no-baseline: the stock arm will not run)")
        print("  Estimates, not quotes: token counts are modelled from evidence "
              "length (see _JUDGE_SCAFFOLD_TOKENS), priced on the same rate "
              "table the eval bills against.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (both judges are OpenAI)")

    suffix = f".limit{args.limit}" if args.limit is not None else ""
    # Output tag only. The verdict cache stays basis-keyed: a verdict is
    # keyed by (query, paper, evidence) and is identical whether or not the
    # paper fell beyond the cap, so an uncapped pass warms the same cache.
    suffix = ".uncapped" + suffix if args.uncapped else suffix
    suffix = ".capk" + suffix if args.cap_to_k else suffix
    suffix = ".officiallog" + suffix if args.from_eval_log else suffix
    out_summary = run_dir / f"test_results.rejudge_{args.basis}{suffix}.json"
    out_per_problem = (
        run_dir / f"test_results.rejudge_{args.basis}{suffix}.per_problem.json"
    )
    for p in (out_summary, out_per_problem):
        if p.exists() and not args.force:
            raise SystemExit(f"{p} exists; pass --force to overwrite")

    # Stock pass first (strict parser; the alternate judge's lenient
    # normalizer install is one-way).
    baseline_result = None
    if is_stock_target:
        target_result = asyncio.run(
            run_pass(samples, STOCK, "stock", stock_cache, args)
        )
    else:
        if args.baseline:
            print(f"\n=== Pass 1/2: canonical baseline ({STOCK}) ===")
            baseline_result = asyncio.run(
                run_pass(samples, STOCK, "stock", stock_cache, args)
            )
        print(f"\n=== Judging with {args.judge} ({args.judge_prompt}) ===")
        target_result = asyncio.run(
            run_pass(samples, args.judge, args.judge_prompt, target_cache, args)
        )

    stored_agg = sum(s.stored_score for s in samples) / len(samples)
    target_agg = _aggregate(samples, target_result["rows"])
    baseline_agg = (
        _aggregate(samples, baseline_result["rows"]) if baseline_result else None
    )

    per_problem = []
    for s in samples:
        row = target_result["rows"].get(s.sid)
        base_row = (baseline_result or {"rows": {}})["rows"].get(s.sid)
        base_score = (
            base_row["score"] if base_row
            else (s.stored_score if s.carry else None)
        )
        new_score = row["score"] if row else s.stored_score
        per_problem.append({
            "sample_id": s.sid,
            "score_type": s.score_type,
            "status": "carried" if s.carry else "rejudged",
            "stored_score": s.stored_score,
            "baseline_canonical_score": base_score,
            "new_score": new_score,
            "delta_vs_baseline": (
                new_score - base_score if base_score is not None else None
            ),
            "n_judge_failures": row["n_failed"] if row else 0,
        })
    out_per_problem.write_text(json.dumps(per_problem, indent=2))

    summary = {
        "source_run": str(run_dir),
        "source_eval_log": str(args.from_eval_log) if args.from_eval_log else None,
        "k_estimate_source": str(args.k_from.resolve()) if args.k_from else None,
        "judge": args.judge,
        "judge_prompt": args.judge_prompt,
        "scored_depth": (
            "uncapped (full submission)" if args.uncapped
            else "capped to k_estimate on an UNCAPPED official submission set "
                 "(--cap-to-k; depth-isolated arm)" if args.cap_to_k
            else "uncapped (full submission, as official judging does)"
                 if args.from_eval_log
            else "capped at stored scored_depth_cap"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "aggregate": target_agg,
        "n_problems": len(samples),
        "n_semantic_rejudged": len(semantic),
        "n_carried": len(carried),
        "judge_cost_usd": round(target_result["judge_cost_usd"], 4),
        "usage_by_model": target_result["usage_by_model"],
        "cache_path": str(target_cache),
        "cache_writes_enabled": not args.no_cache_write,
        "stored_aggregate": stored_agg,
        "stored_aggregate_note": (
            "as-scored at eval time; rank ordering there depends on judge-cache "
            "state, so compare against baseline_canonical, not this"
        ),
        "baseline_canonical": (
            {
                "judge": STOCK,
                "aggregate": baseline_agg,
                "judge_cost_usd": round(baseline_result["judge_cost_usd"], 4),
            }
            if baseline_result
            else None
        ),
        "limit": args.limit,
    }
    out_summary.write_text(json.dumps(summary, indent=2))

    def _fmt(v):
        return f"{v:.5f}" if v is not None else "    -    "

    print(f"\n{'':12} {'stored':>9} {'canonical-4o':>12} "
          f"{args.judge.split('/')[-1]:>16} {'delta':>8}")
    groups = sorted({s.score_type.split("_")[0] for s in samples})
    for grp in groups + ["ALL"]:
        sel = samples if grp == "ALL" else [
            s for s in samples if s.score_type.startswith(grp)
        ]
        st = sum(s.stored_score for s in sel) / len(sel)
        ba = (
            _aggregate(sel, baseline_result["rows"]) if baseline_result else None
        )
        nw = _aggregate(sel, target_result["rows"])
        delta = f"{nw - ba:+.5f}" if ba is not None else "   -"
        print(f"{grp:12} {st:9.5f} {_fmt(ba):>12} {nw:16.5f} {delta:>8}")
    print(f"\nJudge spend: ${target_result['judge_cost_usd']:.2f} ({args.judge})"
          + (f" + ${baseline_result['judge_cost_usd']:.2f} (baseline)"
             if baseline_result else ""))
    print(f"Wrote {out_summary.name}, {out_per_problem.name}, and per-problem "
          f"judge_verdicts.rejudge_{args.basis}.json files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
