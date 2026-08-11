"""Tests for the offline rejudge tool (rejudge_test.py).

Covers the canonical-ordering scorer (the whole point of the tool: scores
must not depend on judge-cache state), the judging plan (cap, known-good,
omitted/empty evidence, duplicates, cache-key parity with the live judge),
the judge-drop retry path, cache write-through, exact-match carry, and the
CLI guards (non-clobber; prompt profile derived from the judge, not a flag).
"""

import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _grounding  # noqa: E402
import rejudge_test as rt  # noqa: E402
from astabench.evals.paper_finder import relevance as rel  # noqa: E402

PERFECT = rel.Relevance.PERFECT.value
NOT_REL = rel.Relevance.NOT_RELEVANT.value
SOMEWHAT = rel.Relevance.SOMEWHAT.value


def _sample(tmp_path, results, known_good=(), cap=None, k=2, sid="semantic_1"):
    pdir = tmp_path / sid
    pdir.mkdir(parents=True, exist_ok=True)
    return rt.Sample(
        sid=sid, score_type="semantic_f1", stored_score=0.5, problem_dir=pdir,
        results=list(results), known_good=set(known_good),
        criteria=[{"name": "Relevance Criterion", "description": "d", "weight": 1.0}],
        cap=cap, k_estimate=k,
    )


def _args(**over):
    base = dict(retries=1, no_cache_write=False, basis="testbasis", concurrency=2)
    base.update(over)
    return SimpleNamespace(**base)


# --- canonical scorer -------------------------------------------------------

def _dcg(grades):
    return sum(g / math.log(i + 1) for i, g in enumerate(grades, 1))


def _rank(grades):
    mx, mn = _dcg(sorted(grades, reverse=True)), _dcg(sorted(grades))
    return (_dcg(grades) - mn) / (mx - mn) if mx != mn else 0.0


def test_score_sample_hand_computed():
    s = _sample(Path("/tmp"), [("A", "a"), ("B", "b"), ("C", "c")], k=2)
    judgements = {"A": PERFECT, "B": NOT_REL, "C": PERFECT}
    out = rt.score_sample(s, judgements)
    # rank over grades in dict (submission) order [3, 0, 3]
    expected_rank = _rank([3, 0, 3])
    assert out["rank"] == pytest.approx(expected_rank)
    # recall: top-2 judged papers are A, B; only A is PERFECT → 1/2
    assert out["recall"] == pytest.approx(0.5)
    assert out["grade3_in_top_k"] == 1
    f = [expected_rank, 0.5]
    assert out["score"] == pytest.approx(len(f) / sum(1 / x for x in f))


def test_score_sample_order_changes_rank():
    """Same verdict multiset, different order → different rank. This is the
    ordering sensitivity the canonical scorer exists to pin down."""
    s = _sample(Path("/tmp"), [("A", "a"), ("B", "b"), ("C", "c")], k=3)
    good_first = rt.score_sample(s, {"A": PERFECT, "B": SOMEWHAT, "C": NOT_REL})
    good_last = rt.score_sample(s, {"A": NOT_REL, "B": SOMEWHAT, "C": PERFECT})
    assert good_first["rank"] > good_last["rank"]


def test_score_sample_zero_when_no_perfect():
    s = _sample(Path("/tmp"), [("A", "a")], k=2)
    out = rt.score_sample(s, {"A": NOT_REL})
    assert out["recall"] == 0.0 and out["score"] == 0.0


def test_score_sample_k_truncation_uses_submission_order():
    """A PERFECT paper beyond the first K judged positions counts for rank
    but not recall."""
    s = _sample(Path("/tmp"), [("A", "a"), ("B", "b"), ("C", "c")], k=1)
    out = rt.score_sample(s, {"A": NOT_REL, "B": PERFECT, "C": PERFECT})
    assert out["grade3_in_top_k"] == 0  # top-1 judged paper is A
    assert out["recall"] == 0.0


def test_score_sample_duplicate_consumes_k_slot_upstream_parity():
    """Upstream's calc_recall_at_k slices the RAW submission, so a duplicated
    pid inside the K-window consumes two slots and pushes the next judged
    paper out. Deliberately preserved — official-scoring parity beats local
    tidiness (astabench eval.py:143-147)."""
    s = _sample(Path("/tmp"), [("A", "a"), ("A", "a2"), ("B", "b")], k=2)
    out = rt.score_sample(s, {"A": NOT_REL, "B": PERFECT})
    # window is [A, A]; B's PERFECT is outside it despite being 2nd judged
    assert out["grade3_in_top_k"] == 0
    assert out["recall"] == 0.0


# --- judging plan -----------------------------------------------------------

def test_plan_cap_breaks_and_marks_beyond():
    s = _sample(Path("/tmp"), [("A", "a"), ("B", "b"), ("C", "c")], cap=1)
    order, preset, to_judge, statuses = rt.plan_sample(s, {})
    assert order == ["A"]
    assert [p for p, _, _ in to_judge] == ["A"]
    assert [st for _, st in statuses] == [
        "pending", "beyond_scored_depth", "beyond_scored_depth"
    ]


def test_plan_known_good_empty_omitted_duplicate():
    s = _sample(
        Path("/tmp"),
        [("G", "junk"), ("E", "   "), ("O", rt.EVIDENCE_OMITTED_MARKER),
         ("A", "real"), ("A", "real again")],
        known_good={"G"},
    )
    order, preset, to_judge, statuses = rt.plan_sample(s, {})
    assert order == ["G", "E", "A"]  # omitted excluded, duplicate collapsed
    assert preset == {"G": PERFECT, "E": NOT_REL}
    assert [p for p, _, _ in to_judge] == ["A"]
    assert statuses == [
        ("G", "known_good"), ("E", "empty_evidence"),
        ("O", "beyond_scored_depth"), ("A", "pending"), ("A", "duplicate"),
    ]


def test_plan_cache_hit_uses_grounding_cache_key():
    """Cache-key parity with the live judge: a verdict stored under
    _grounding.cache_key must be found, and land at submission position."""
    ev = "some evidence text"
    cache_q = {_grounding.cache_key("B", ev): SOMEWHAT}
    s = _sample(Path("/tmp"), [("A", "a"), ("B", ev)])
    order, preset, to_judge, statuses = rt.plan_sample(s, cache_q)
    assert order == ["A", "B"]
    assert preset == {"B": SOMEWHAT}
    assert [p for p, _, _ in to_judge] == ["A"]
    # evidence variant misses: key includes the evidence hash
    assert rt.plan_sample(_sample(Path("/tmp"), [("B", ev + "!")]), cache_q)[2]


# --- judging + retry + verdict file ----------------------------------------

def _run_process(tmp_path, sample, judge_impl, **args_over):
    calls = []

    async def fake_judge(entities, criteria):
        calls.append([d.corpus_id for d in entities])
        return judge_impl(entities)

    orig = rel.load_relevance_judgement
    rel.load_relevance_judgement = fake_judge
    try:
        args = _args(**args_over)
        row = asyncio.run(
            rt._process_sample(
                sample, {}, tmp_path / "cache.json",
                asyncio.Semaphore(2), args, {"done": 0, "total": 1},
            )
        )
    finally:
        rel.load_relevance_judgement = orig
    return row, calls


def test_dropped_doc_retried_then_failed(tmp_path):
    """First call drops B (parse failure upstream); retry re-asks ONLY B.
    Still missing after retries → judge_call_failed, absent from scoring."""
    s = _sample(tmp_path, [("A", "a"), ("B", "b"), ("C", "c")], k=3)

    def impl(entities):
        return {d.corpus_id: PERFECT for d in entities if d.corpus_id != "B"}

    row, calls = _run_process(tmp_path, s, impl, no_cache_write=True)
    assert calls == [["A", "B", "C"], ["B"]]  # retry asks only the missing doc
    assert row["n_failed"] == 1
    verdicts = json.loads(
        (s.problem_dir / "judge_verdicts.rejudge_testbasis.json").read_text()
    )
    # 1-based positions, matching the evaluator's judge_verdicts.json sibling
    assert [p["position"] for p in verdicts["papers"]] == [1, 2, 3]
    by_pid = {p["paper_id"]: p for p in verdicts["papers"]}
    assert by_pid["B"]["status"] == "judge_call_failed"
    assert by_pid["B"]["label"] is None
    assert by_pid["A"]["status"] == "judged"
    # B excluded from both rank sequence and recall: grades are [3, 3]
    assert row["rank"] == pytest.approx(_rank([3, 3]))


def test_duplicate_statuses_match_evaluator_rule(tmp_path):
    """Verdict-file duplicate semantics follow the evaluator's
    _verdict_states rule, not just its status name: a repeat of a
    verdict-holding paper is 'duplicate' with label None — including a
    repeat beyond the cap (the recall window filters by membership,
    position-blind) — while a repeat of a judge-failed paper mirrors the
    first occurrence's outcome (scorer-invisible, no slot consumed)."""
    s = _sample(
        tmp_path, [("F", "f"), ("F", "f2"), ("A", "a"), ("A", "x")],
        k=3, cap=3,
    )

    def impl(entities):
        # F fails every ask; A judges fine.
        return {d.corpus_id: PERFECT for d in entities if d.corpus_id != "F"}

    _run_process(tmp_path, s, impl, no_cache_write=True)
    verdicts = json.loads(
        (s.problem_dir / "judge_verdicts.rejudge_testbasis.json").read_text()
    )
    rows = [(p["status"], p["label"]) for p in verdicts["papers"]]
    assert rows == [
        ("judge_call_failed", None),   # F, first occurrence
        ("judge_call_failed", None),   # F repeat: mirrors, NOT "duplicate"
        ("judged", PERFECT),           # A
        ("duplicate", None),           # A repeat beyond cap: still a dup
    ]


def test_canonical_order_despite_fresh_after_cached(tmp_path):
    """Fresh verdicts must land at submission position, not appended after
    cache hits — identical submissions score identically at any cache
    warmth."""
    ev_b = "cached evidence"
    s = _sample(tmp_path, [("A", "a"), ("B", ev_b), ("C", "c")], k=3)
    cache = {"semantic_1": {_grounding.cache_key("B", ev_b): NOT_REL}}

    async def fake_judge(entities, criteria):
        return {d.corpus_id: PERFECT for d in entities}

    orig = rel.load_relevance_judgement
    rel.load_relevance_judgement = fake_judge
    try:
        row = asyncio.run(
            rt._process_sample(
                s, cache, tmp_path / "cache.json",
                asyncio.Semaphore(2), _args(no_cache_write=True),
                {"done": 0, "total": 1},
            )
        )
    finally:
        rel.load_relevance_judgement = orig
    # grades in submission order [A=3, B=0, C=3]; appended-after ordering
    # would have produced [0, 3, 3] (rank 0 under lower-bound correction)
    assert row["rank"] == pytest.approx(_rank([3, 0, 3]))


def test_cache_write_through_and_opt_out(tmp_path):
    s = _sample(tmp_path, [("A", "evidence a")], k=1)
    row, _ = _run_process(
        tmp_path, s, lambda ents: {d.corpus_id: PERFECT for d in ents}
    )
    cache = json.loads((tmp_path / "cache.json").read_text())
    assert cache == {"semantic_1": {_grounding.cache_key("A", "evidence a"): PERFECT}}

    s2 = _sample(tmp_path, [("A", "evidence a")], k=1, sid="semantic_2")
    (tmp_path / "cache.json").unlink()
    _run_process(tmp_path, s2, lambda ents: {d.corpus_id: PERFECT for d in ents},
                 no_cache_write=True)
    assert not (tmp_path / "cache.json").exists()


# --- load_run + aggregate ---------------------------------------------------

def _mk_problem(tp, sid, results=None, score=0.5, cap=None, k=2,
                write_score_meta=True):
    d = tp / sid
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "sample_id": sid, "score": score,
        "score_type": sid.split("_")[0] + "_f1", "error": None,
    }))
    if sid.startswith("semantic"):
        (d / "submission.json").write_text(json.dumps({"output": {"results": [
            {"paper_id": p, "markdown_evidence": e} for p, e in (results or [])
        ]}}))
        (d / "gold_criteria.md").write_text(json.dumps({
            "known_to_be_good": [], "known_to_be_bad": [],
            "relevance_criteria": [
                {"name": "Relevance Criterion", "description": "d", "weight": 1.0}
            ],
        }))
        (d / "judge_verdicts.json").write_text(
            json.dumps({"scored_depth_cap": cap, "papers": []})
        )
        if write_score_meta:
            (d / "score_meta.json").write_text(json.dumps({"k_estimate": k}))
    return d


def test_load_run_carry_and_semantic(tmp_path):
    tp = tmp_path / "run" / "test_problems"
    _mk_problem(tp, "metadata_1", score=0.9)
    _mk_problem(tp, "semantic_1", [("A", "ev")], score=0.4)
    samples = rt.load_run(tmp_path / "run")
    by_sid = {s.sid: s for s in samples}
    assert by_sid["metadata_1"].carry and by_sid["metadata_1"].stored_score == 0.9
    assert not by_sid["semantic_1"].carry
    assert by_sid["semantic_1"].results == [("A", "ev")]


def test_load_run_k_fallback_to_cap_then_error(tmp_path):
    tp = tmp_path / "run" / "test_problems"
    _mk_problem(tp, "semantic_1", [("A", "ev")], cap=7, write_score_meta=False)
    samples = rt.load_run(tmp_path / "run")
    assert samples[0].k_estimate == 7
    tp2 = tmp_path / "run2" / "test_problems"
    _mk_problem(tp2, "semantic_1", [("A", "ev")], cap=None, write_score_meta=False)
    with pytest.raises(SystemExit, match="k_estimate"):
        rt.load_run(tmp_path / "run2")


def test_load_run_refuses_scrubbed_evidence(tmp_path):
    tp = tmp_path / "run" / "test_problems"
    d = _mk_problem(tp, "semantic_1", [("A", "ev")])
    (d / "evidence_grounding.md").write_text("scrubbed")
    with pytest.raises(SystemExit, match="scrub"):
        rt.load_run(tmp_path / "run")


def test_aggregate_mixes_rejudged_and_carried(tmp_path):
    tp = tmp_path / "run" / "test_problems"
    _mk_problem(tp, "metadata_1", score=1.0)
    _mk_problem(tp, "semantic_1", [("A", "ev")], score=0.4)
    samples = rt.load_run(tmp_path / "run")
    rows = {"semantic_1": {"score": 0.6}}
    assert rt._aggregate(samples, rows) == pytest.approx((1.0 + 0.6) / 2)


# --- CLI guards -------------------------------------------------------------

def test_judge_prompt_not_separately_settable(monkeypatch, capsys):
    """The prompt profile is a property of the judge (main._prompt_for_judge);
    a --judge-prompt flag must not exist, or a rejudge could measure a basis
    no live eval can produce."""
    monkeypatch.setattr(sys, "argv", [
        "rejudge_test.py", "/nonexistent", "--judge", rt.STOCK,
        "--judge-prompt", "no-prose",
    ])
    with pytest.raises(SystemExit):
        rt.main()
    assert "unrecognized arguments" in capsys.readouterr().err


def test_k_from_rejected_without_from_eval_log(tmp_path, monkeypatch, capsys):
    """On the run's own path, K and the submission come from the same
    test_problems/<sid>/ dir. Borrowing K there would pair one run's recall
    denominator with another run's stored submission."""
    run_dir = tmp_path / "runs" / "robophd" / "run1"
    _mk_problem(run_dir / "test_problems", "metadata_1", score=1.0)
    monkeypatch.setattr(sys, "argv", [
        "rejudge_test.py", str(run_dir), "--judge", "openai/gpt-5.6-luna",
        "--k-from", str(run_dir),
    ])
    with pytest.raises(SystemExit, match="only applies to --from-eval-log"):
        rt.main()


def test_non_clobber_without_force_and_derived_basis(tmp_path, monkeypatch):
    """Luna's basis derives to no-prose (matching live evals), so the
    non-clobber gate must guard the _noprose-suffixed filename."""
    tp = tmp_path / "runs" / "robophd" / "run1" / "test_problems"
    _mk_problem(tp, "metadata_1", score=1.0)
    run_dir = tmp_path / "runs" / "robophd" / "run1"
    from main import _prompt_for_judge
    basis = rt._judge_basis_slug(
        "openai/gpt-5.6-luna", _prompt_for_judge("openai/gpt-5.6-luna")
    )
    assert basis.endswith("_noprose")
    existing = run_dir / f"test_results.rejudge_{basis}.json"
    existing.write_text("{}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")
    monkeypatch.setattr(sys, "argv", [
        "rejudge_test.py", str(run_dir), "--judge", "openai/gpt-5.6-luna",
    ])
    with pytest.raises(SystemExit, match="--force"):
        rt.main()


def test_stock_pass_refuses_patched_grader():
    """The stock baseline must run before any alternate-judge pass: if the
    grader global has already been swapped, run_pass must refuse — with a
    real raise (an assert would vanish under python -O)."""
    orig = rel.GRADER_MODEL_NAME
    rel.GRADER_MODEL_NAME = "openai/gpt-5.6-luna"
    try:
        with pytest.raises(RuntimeError, match="stock pass"):
            asyncio.run(rt.run_pass([], rt.STOCK, "stock", Path("/tmp/x.json"),
                                    _args()))
    finally:
        rel.GRADER_MODEL_NAME = orig


def test_basis_slug_matches_main():
    from main import _judge_basis_slug
    assert _judge_basis_slug("openai/gpt-5.6-luna", "no-prose").endswith("_noprose")
    assert _judge_basis_slug(rt.STOCK, "stock") == "openai_gpt-4o-2024-11-20"


# --- --dry-run cost estimate --------------------------------------------------
#
# A 0.2% cache hit rate means ~18.5k fresh judge calls, which is the number a
# dry run exists to surface BEFORE you commit. Reporting only the hit rate
# left the reader to price it themselves.


def test_cost_scales_with_the_number_of_calls():
    a = rt._estimate_judge_cost(["x" * 1000], "openai/gpt-5.6-luna", "no-prose")
    b = rt._estimate_judge_cost(["x" * 1000] * 10, "openai/gpt-5.6-luna", "no-prose")
    assert b == pytest.approx(a * 10)


def test_cost_scales_with_evidence_length():
    short = rt._estimate_judge_cost(["x" * 500], "openai/gpt-5.6-luna", "no-prose")
    long = rt._estimate_judge_cost(["x" * 5000], "openai/gpt-5.6-luna", "no-prose")
    assert long > short, "longer evidence must cost more — it is input tokens"


def test_no_prose_is_cheaper_than_stock_on_identical_evidence():
    """The whole point of the profile: it drops output tokens the scorer
    never reads."""
    ev = ["x" * 2000] * 50
    assert (rt._estimate_judge_cost(ev, "openai/gpt-5.6-luna", "no-prose")
            < rt._estimate_judge_cost(ev, "openai/gpt-5.6-luna", "stock"))


def test_nothing_to_judge_costs_nothing():
    assert rt._estimate_judge_cost([], "openai/gpt-5.6-luna", "no-prose") == 0.0


def test_the_token_model_reproduces_the_one_measured_official_run():
    """Calibration guard. The scaffold constant was fitted to v0_0_7: $192 of
    judge spend over 194 x 250 verdicts at 976 chars/paper, per the submit
    script's cost table and the upstream report's token counts. If a later
    edit moves _JUDGE_SCAFFOLD_TOKENS or the chars-per-token ratio, this is
    what says the estimate no longer matches reality.
    """
    evidence = ["x" * 976] * (194 * 250)
    cost = rt._estimate_judge_cost(evidence, "openai/gpt-4o-2024-11-20", "stock")
    assert cost == pytest.approx(192.0, rel=0.05), (
        f"model predicts ${cost:.0f} against the measured $192 for v0_0_7; "
        f"the token model has drifted from the run it was calibrated on"
    )


def test_pricing_goes_through_the_evaluator_not_a_second_table():
    """A local rate table here could disagree with the eval it is predicting —
    and would have missed the 2026-07-31 luna reprice."""
    import ast

    src = (Path(__file__).resolve().parent.parent / "rejudge_test.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_estimate_judge_cost")
    body = ast.unparse(fn)
    assert "_estimate_cost" in body, "must reuse the evaluator's pricing path"
    assert "e-6" not in body and "0.20" not in body, (
        "per-token rates hardcoded here would drift from JUDGE_PRICE_OVERRIDES"
    )


# --- load_eval_log (official astabench .eval logs) ---------------------------
#
# This loader parses an EXTERNAL log format across several branches, so each
# branch is pinned: an upstream shape change must fail here rather than
# silently produce a short or mis-scored sample set.

def _log_sample(sid, *, results=None, score=0.5, target=None, completion=None):
    """One inspect_ai EvalSample, shaped as load_eval_log reads it."""
    if completion is None:
        completion = json.dumps({"output": {"query_id": sid, "results": [
            {"paper_id": p, "markdown_evidence": e} for p, e in (results or [])
        ]}})
    if target is None:
        target = json.dumps({
            "known_to_be_good": ["777"], "known_to_be_bad": [],
            "relevance_criteria": [
                {"name": "Relevance Criterion", "description": "d", "weight": 1.0}
            ],
        })
    return SimpleNamespace(
        id=sid,
        metadata={"score_type": sid.split("_")[0] + "_f1"},
        scores={"score_paper_finder": SimpleNamespace(value=score, metadata={})},
        output=SimpleNamespace(completion=completion),
        target=target,
    )


@pytest.fixture
def fake_eval_log(monkeypatch):
    """Patch inspect_ai.log.read_eval_log; load_eval_log imports it at call
    time, so the patch lands regardless of import order."""
    import inspect_ai.log as ial

    def _install(samples):
        monkeypatch.setattr(
            ial, "read_eval_log", lambda _p: SimpleNamespace(samples=samples)
        )
    return _install


def _eval_log_run(tmp_path, sids=("semantic_1",), k=2, cap=5):
    """Minimal run_dir scaffolding: load_eval_log needs only score_meta.json."""
    tp = tmp_path / "run" / "test_problems"
    for sid in sids:
        _mk_problem(tp, sid, [("X", "ev")] if sid.startswith("semantic") else None,
                    cap=cap, k=k)
    log = tmp_path / "official.eval"
    log.write_text("")
    return tmp_path / "run", log


def test_load_eval_log_semantic_and_carry(tmp_path, fake_eval_log):
    run_dir, log = _eval_log_run(tmp_path, ("semantic_1", "metadata_1"))
    fake_eval_log([
        _log_sample("semantic_1", results=[("A", "ev-a"), ("B", "ev-b")], score=0.4),
        _log_sample("metadata_1", score=0.9),
    ])
    by_sid = {s.sid: s for s in rt.load_eval_log(log, run_dir)}
    m = by_sid["metadata_1"]
    assert m.carry and m.carried_reason == "exact_match" and m.stored_score == 0.9
    s = by_sid["semantic_1"]
    assert not s.carry
    assert s.results == [("A", "ev-a"), ("B", "ev-b")]
    assert s.stored_score == 0.4                     # the OFFICIAL score
    assert s.known_good == {"777"}                   # from the log's target
    assert s.criteria[0]["name"] == "Relevance Criterion"
    assert s.k_estimate == 2                         # from run_dir score_meta
    # Diagnostics land in a dedicated tree, NOT test_problems/: these verdicts
    # grade the OFFICIAL submission, and a run that timed out has no
    # test_problems/<sid>/ for the failed queries — creating them would inflate
    # the directory count that scripts read as "samples evaluated".
    assert s.problem_dir == run_dir / "rejudge_officiallog" / "semantic_1"


def test_load_eval_log_is_uncapped_even_when_the_run_had_a_cap(tmp_path, fake_eval_log):
    """Load-bearing: official judging is uncapped, so plan_sample must not
    stop at the internal run's scored_depth_cap."""
    run_dir, log = _eval_log_run(tmp_path, ("semantic_1",), cap=1)
    fake_eval_log([_log_sample("semantic_1", results=[("A", "a"), ("B", "b")])])
    s = rt.load_eval_log(log, run_dir)[0]
    assert s.cap is None
    _order, _preset, to_judge, statuses = rt.plan_sample(s, {})
    assert len(to_judge) == 2
    assert not any(st == "beyond_scored_depth" for _p, st in statuses)


def test_load_eval_log_empty_log_and_missing_scaffolding(tmp_path, fake_eval_log):
    run_dir, log = _eval_log_run(tmp_path)
    fake_eval_log([])
    with pytest.raises(SystemExit, match="no samples"):
        rt.load_eval_log(log, run_dir)
    fake_eval_log([_log_sample("semantic_1", results=[("A", "a")])])
    with pytest.raises(SystemExit, match="test_problems"):
        rt.load_eval_log(log, tmp_path / "nonexistent_run")


def test_load_eval_log_missing_k_estimate_is_fatal(tmp_path, fake_eval_log):
    """K is the recall denominator; guessing it would silently mis-score."""
    tp = tmp_path / "run" / "test_problems"
    _mk_problem(tp, "semantic_1", [("X", "ev")], cap=None, write_score_meta=False)
    log = tmp_path / "official.eval"; log.write_text("")
    fake_eval_log([_log_sample("semantic_1", results=[("A", "a")])])
    with pytest.raises(SystemExit, match="--k-from"):
        rt.load_eval_log(log, tmp_path / "run")


# --- --k-from ----------------------------------------------------------------
#
# A run that timed out on a semantic query has no test_problems/<sid>/ for it,
# but the official log carries all 267 samples — and K is stored only in
# test_problems. So replaying such a run against its own dir dies on exactly
# the queries it failed (-010: semantic_242; -011: four). K is a per-query
# benchmark constant, so --k-from borrows it from a 267-complete run.


def test_k_from_supplies_a_sid_the_run_never_evaluated(tmp_path, fake_eval_log):
    run_dir, log = _eval_log_run(tmp_path, ("semantic_1",), k=11)  # no semantic_2
    complete = tmp_path / "complete"
    _mk_problem(complete / "test_problems", "semantic_1", [("X", "ev")], k=11)
    _mk_problem(complete / "test_problems", "semantic_2", [("X", "ev")], k=22)
    fake_eval_log([
        _log_sample("semantic_1", results=[("A", "a")]),
        _log_sample("semantic_2", results=[("B", "b")]),
    ])

    by_sid = {s.sid: s for s in rt.load_eval_log(log, run_dir, k_from=complete)}

    assert by_sid["semantic_1"].k_estimate == 11
    assert by_sid["semantic_2"].k_estimate == 22
    # Outputs still belong to the run being replayed, not to the K donor.
    assert by_sid["semantic_2"].problem_dir == (
        run_dir / "rejudge_officiallog" / "semantic_2"
    )
    assert not (run_dir / "test_problems" / "semantic_2").exists()
    assert not (complete / "rejudge_officiallog").exists()


def test_k_from_without_test_problems_is_fatal(tmp_path, fake_eval_log):
    run_dir, log = _eval_log_run(tmp_path)
    fake_eval_log([_log_sample("semantic_1", results=[("A", "a")])])
    with pytest.raises(SystemExit, match="test_problems"):
        rt.load_eval_log(log, run_dir, k_from=tmp_path / "not_a_run")


# --- --cap-to-k ---------------------------------------------------------------
#
# Official logs are judged uncapped; --cap-to-k replays them at the internal
# depth so the depth axis can be isolated with the agent draw held fixed.


def test_cap_to_k_sets_cap_from_k_estimate(tmp_path, fake_eval_log, monkeypatch):
    run_dir, log = _eval_log_run(tmp_path, ("semantic_1", "metadata_1"), k=2, cap=99)
    fake_eval_log([
        _log_sample("semantic_1", results=[("A", "a"), ("B", "b"), ("C", "c")]),
        _log_sample("metadata_1"),
    ])
    samples = rt.load_eval_log(log, run_dir)
    sem = next(s for s in samples if not s.carry)
    assert sem.cap is None, "official judging is uncapped before --cap-to-k"

    # What main() does under --cap-to-k. cap comes from k_estimate, NOT from
    # the run's stored scored_depth_cap (99 here) -- the official submission
    # set has no cap of its own to inherit.
    for s in samples:
        if not s.carry:
            s.cap = s.k_estimate
    assert sem.cap == 2
    _order, _preset, to_judge, statuses = rt.plan_sample(sem, {})
    assert len(to_judge) == 2, "third paper is beyond k"
    assert [st for _p, st in statuses][-1] == "beyond_scored_depth"


def test_cap_to_k_rejected_off_the_eval_log_path(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "robophd" / "run1"
    _mk_problem(run_dir / "test_problems", "metadata_1", score=1.0)
    monkeypatch.setattr(sys, "argv", [
        "rejudge_test.py", str(run_dir), "--judge", "openai/gpt-5.6-luna",
        "--cap-to-k",
    ])
    with pytest.raises(SystemExit, match="only applies to --from-eval-log"):
        rt.main()


def test_cap_to_k_and_uncapped_are_mutually_exclusive(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "robophd" / "run1"
    _mk_problem(run_dir / "test_problems", "metadata_1", score=1.0)
    log = tmp_path / "official.eval"
    log.write_text("")
    monkeypatch.setattr(sys, "argv", [
        "rejudge_test.py", str(run_dir), "--judge", "openai/gpt-5.6-luna",
        "--from-eval-log", str(log), "--cap-to-k", "--uncapped",
    ])
    with pytest.raises(SystemExit, match="opposite depths"):
        rt.main()


def test_capk_tag_keeps_arms_from_clobbering(tmp_path):
    """The capped arm must not overwrite the uncapped pass's verdict file."""
    s = _sample(tmp_path, [("A", "a"), ("B", "b"), ("C", "c")], k=3)
    names = set()
    for uncapped, cap_to_k in ((False, False), (False, True)):
        s.problem_dir = tmp_path / f"{uncapped}{cap_to_k}"
        _run_process(
            tmp_path, s, lambda ents: {d.corpus_id: PERFECT for d in ents},
            no_cache_write=True, uncapped=uncapped, cap_to_k=cap_to_k,
            from_eval_log=Path("official.eval"),
        )
        names |= {p.name for p in s.problem_dir.iterdir()}
    assert names == {
        "judge_verdicts.rejudge_testbasis.officiallog.json",
        "judge_verdicts.rejudge_testbasis.capk.officiallog.json",
    }


def test_verdict_diagnostic_creates_its_parent(tmp_path):
    """rejudge_officiallog/<sid>/ does not exist before the first write."""
    s = _sample(tmp_path, [("A", "a"), ("B", "b"), ("C", "c")], k=3)
    s.problem_dir = tmp_path / "rejudge_officiallog" / s.sid  # never created
    _row, _calls = _run_process(
        tmp_path, s, lambda ents: {d.corpus_id: PERFECT for d in ents},
        no_cache_write=True,
    )
    assert (s.problem_dir / "judge_verdicts.rejudge_testbasis.json").is_file()


def test_load_eval_log_unparseable_target_is_fatal(tmp_path, fake_eval_log):
    run_dir, log = _eval_log_run(tmp_path)
    fake_eval_log([_log_sample("semantic_1", results=[("A", "a")], target="not json")])
    with pytest.raises(SystemExit, match="target"):
        rt.load_eval_log(log, run_dir)


def test_load_eval_log_empty_submission_carries(tmp_path, fake_eval_log):
    run_dir, log = _eval_log_run(tmp_path)
    fake_eval_log([_log_sample("semantic_1", results=[])])
    s = rt.load_eval_log(log, run_dir)[0]
    assert s.carry and s.carried_reason == "no_submission"


def test_load_eval_log_falls_back_to_lenient_extraction(tmp_path, fake_eval_log,
                                                        monkeypatch):
    """A completion the strict parse rejects must reach the lenient
    extractor rather than silently becoming an empty submission."""
    run_dir, log = _eval_log_run(tmp_path)
    called = {}
    def _lenient(text):
        called["text"] = text
        return [{"paper_id": "Z", "markdown_evidence": "salvaged"}]
    monkeypatch.setattr(rt, "_extract_results_lenient", _lenient)
    fake_eval_log([_log_sample("semantic_1", completion="{truncated json...")])
    s = rt.load_eval_log(log, run_dir)[0]
    assert called["text"].startswith("{truncated")
    assert s.results == [("Z", "salvaged")]


def test_load_eval_log_limit_and_max_results(tmp_path, fake_eval_log):
    run_dir, log = _eval_log_run(tmp_path, ("semantic_1", "semantic_2", "metadata_1"))
    # carry FIRST, so a plain samples[:limit] would wrongly return it
    fake_eval_log([
        _log_sample("metadata_1"),
        _log_sample("semantic_1", results=[("A", "a")]),
        _log_sample("semantic_2", results=[("B", "b")]),
    ])
    limited = rt.load_eval_log(log, run_dir, limit=1)
    assert [s.sid for s in limited] == ["semantic_1"]   # semantic only, carries dropped
    over = rt.MAX_RESULTS_TO_CONSIDER + 5
    fake_eval_log([_log_sample(
        "semantic_1", results=[(f"p{i}", "e") for i in range(over)]
    )])
    s = rt.load_eval_log(log, run_dir)[0]
    assert len(s.results) == rt.MAX_RESULTS_TO_CONSIDER
