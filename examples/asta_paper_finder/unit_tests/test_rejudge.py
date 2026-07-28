"""Tests for the offline rejudge tool (rejudge_test.py).

Covers the canonical-ordering scorer (the whole point of the tool: scores
must not depend on judge-cache state), the judging plan (cap, known-good,
omitted/empty evidence, duplicates, cache-key parity with the live judge),
the judge-drop retry path, cache write-through, exact-match carry, and the
CLI guards (non-clobber, no-prose-with-stock rejection).
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

def test_no_prose_with_stock_judge_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "rejudge_test.py", "/nonexistent", "--judge", rt.STOCK,
        "--judge-prompt", "no-prose",
    ])
    with pytest.raises(SystemExit):
        rt.main()
    assert "no-prose" in capsys.readouterr().err


def test_non_clobber_without_force(tmp_path, monkeypatch):
    tp = tmp_path / "runs" / "robophd" / "run1" / "test_problems"
    _mk_problem(tp, "metadata_1", score=1.0)
    run_dir = tmp_path / "runs" / "robophd" / "run1"
    basis = rt._judge_basis_slug("openai/gpt-5.6-luna", "stock")
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
