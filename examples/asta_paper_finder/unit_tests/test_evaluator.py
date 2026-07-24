"""Unit tests for the asta_paper_finder evaluator.

Covers the pieces most likely to regress silently:

  - `aggregate()` math: free zone, breach arithmetic, unbounded-below,
    test-mode fraction, the eval_cost/agent_cost_usd coalesce.
  - Cost-knob validation at construction.
  - The agent-vs-judge cost split on a synthetic inspect log: judge
    usage (gpt-4o-2024-11-20) must land in other_cost_usd and NEVER in
    cost_usd / cost_by_model_usd.
  - `_head_tail_truncate` slice/marker behavior.
  - AST/source guards: `subprocess` is imported (the archived example
    shipped without it and crashed on the first eval), the killpg
    machinery is intact, and no diagnostics key uses bare "error"
    (the framework's failure detection reads "error.md").

Aggregate/split tests build instances via object.__new__ so they run
without provider keys or an astabench-importable environment beyond
what evaluator.py itself needs at module import.
"""
import ast
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PFB_DIR = REPO_ROOT / "examples" / "asta_paper_finder"
EVALUATOR_SRC = (PFB_DIR / "evaluator.py").read_text()


@pytest.fixture(scope="module")
def ev_mod():
    """Import evaluator.py once per module (heavy astabench import).

    PFB_DIR stays on sys.path for the module's lifetime — the
    constructor lazily does `from model_registry import ...`, which
    needs the example dir importable at construction time, not just at
    module-import time. (Run these tests as their own pytest invocation;
    the flat `evaluator` module name collides across examples.)
    """
    if str(PFB_DIR) not in sys.path:
        sys.path.insert(0, str(PFB_DIR))
    import evaluator  # noqa: E402
    return evaluator


def _bare_evaluator(ev_mod, *, apply_cost_penalty=True,
                    min_cost_threshold=0.10, cost_per_error=0.02):
    """Instance without running __init__ (no provider keys needed)."""
    ev = object.__new__(ev_mod.PaperFinderEvaluator)
    ev.apply_cost_penalty = apply_cost_penalty
    ev.min_cost_threshold = min_cost_threshold
    ev.cost_per_error = cost_per_error
    ev.total_eval_cost = 0.0
    ev.total_judge_cost = 0.0
    ev._cost_lock = threading.Lock()
    return ev


# --- aggregate() ------------------------------------------------------------


def test_aggregate_empty_batch(ev_mod):
    ev = _bare_evaluator(ev_mod)
    assert ev.aggregate([]) == (0.0, "")


def test_aggregate_free_zone_scales_to_percentage(ev_mod):
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.5, "eval_cost": 0.02}] * 20
    score, explanation = ev.aggregate(results)
    assert score == pytest.approx(50.0)
    assert "free zone" in explanation


def test_aggregate_breach_exact_arithmetic(ev_mod):
    """mean_cost 0.20 → excess 0.10 → 5 error-equivalents → 5 * (100/20)
    = 25 pts off a base of 100 * 0.4 = 40 → 15."""
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.4, "eval_cost": 0.20}] * 20
    score, explanation = ev.aggregate(results)
    assert score == pytest.approx(40.0 - 25.0)
    assert "exceeded threshold" in explanation


def test_aggregate_unbounded_below(ev_mod):
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.1, "eval_cost": 5.0}] * 10
    score, _ = ev.aggregate(results)
    assert score < -1000


def test_aggregate_test_mode_returns_fraction_no_explanation(ev_mod):
    ev = _bare_evaluator(ev_mod, apply_cost_penalty=False)
    results = [{"score": 0.5, "eval_cost": 99.0}] * 4
    assert ev.aggregate(results) == (0.5, "")


def test_aggregate_coalesces_agent_cost_usd_key(ev_mod):
    """Test-path diagnostics carry agent_cost_usd instead of eval_cost;
    the aggregator must read either."""
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.4, "agent_cost_usd": 0.20}] * 20
    score, _ = ev.aggregate(results)
    assert score == pytest.approx(15.0)


def test_aggregate_never_reads_other_cost_usd(ev_mod):
    """Judge spend must not leak into the penalty: a batch whose only
    cost is other_cost_usd stays in the free zone."""
    ev = _bare_evaluator(ev_mod)
    results = [{"score": 0.5, "other_cost_usd": 10.0}] * 20
    score, explanation = ev.aggregate(results)
    assert score == pytest.approx(50.0)
    assert "free zone" in explanation


# --- constructor knob validation ---------------------------------------------


@pytest.mark.parametrize("kwargs,match", [
    ({"min_cost_threshold": -0.01}, "min_cost_threshold"),
    ({"cost_per_error": 0.0}, "cost_per_error"),
    ({"cost_per_error": -1.0}, "cost_per_error"),
])
def test_constructor_rejects_bad_knobs(ev_mod, monkeypatch, kwargs, match):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("ASTA_TOOL_KEY", "test-key")
    with pytest.raises(ValueError, match=match):
        ev_mod.PaperFinderEvaluator(**kwargs)


def test_constructor_requires_provider_keys(ev_mod, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_FOR_ROBOPHD", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ev_mod.PaperFinderEvaluator()


def test_constructor_requires_asta_tool_key(ev_mod, monkeypatch):
    """ASTA_TOOL_KEY is unconditionally required: the MCP suite is the
    task's only retrieval surface (the public-S2 search fallback was
    removed — its sole real-world contribution was the
    asta_paper_finder_20260710_081139 429 budget-burn)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("ASTA_TOOL_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ASTA_TOOL_KEY"):
        ev_mod.PaperFinderEvaluator()


# --- judge-cost split ---------------------------------------------------------


def _fake_log(model_usage: dict, *, score=0.5, score_metadata=None,
              explanation=None):
    usage_objs = {
        name: SimpleNamespace(
            input_tokens=u[0], output_tokens=u[1],
            total_tokens=u[0] + u[1], reasoning_tokens=0,
        )
        for name, u in model_usage.items()
    }
    sample_log = SimpleNamespace(
        error=None,
        scores={"scorer": SimpleNamespace(
            value=score, metadata=score_metadata, explanation=explanation,
        )},
        output=SimpleNamespace(completion="{}"),
    )
    return SimpleNamespace(
        samples=[sample_log],
        stats=SimpleNamespace(model_usage=usage_objs),
    )


def _fake_sample(ev_mod):
    from inspect_ai.dataset import Sample
    return Sample(
        id="semantic_2",
        input="find papers",
        target="criteria",
        metadata={"score_type": "semantic", "raw_query": "find papers"},
    )


def test_judge_split_routes_costs(ev_mod, monkeypatch):
    """gpt-4o judge usage → other_cost_usd; agent model → cost_usd +
    cost_by_model_usd; judge never in the agent buckets."""
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 1.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log({
        "openai/gpt-4o-2024-11-20": (5000, 500),   # the judge
        "openai/gpt-5.4-mini": (1000, 100),         # the agent
    })
    score, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    assert score == 0.5
    assert diag["other_cost_usd"] == pytest.approx(1.0)
    assert diag["cost_usd"] == pytest.approx(1.0)
    assert diag["agent_cost_usd"] == pytest.approx(1.0)
    assert "openai/gpt-4o-2024-11-20" not in diag["cost_by_model_usd"]
    assert "openai/gpt-5.4-mini" in diag["cost_by_model_usd"]
    # Usage summary keeps BOTH models for the audit trail.
    assert set(diag["usage"]) == {
        "openai/gpt-4o-2024-11-20", "openai/gpt-5.4-mini",
    }
    # Running totals split the same way.
    assert ev.total_eval_cost == pytest.approx(1.0)
    assert ev.total_judge_cost == pytest.approx(1.0)


def test_judge_ids_match_astabench(ev_mod):
    """The module-load assert enforces this, but pin it as a test too so
    a refactor that drops the assert still gets caught."""
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME
    assert GRADER_MODEL_NAME in ev_mod.JUDGE_MODEL_IDS


# --- judge-verdict surfacing ---------------------------------------------------


@pytest.fixture()
def verdict_cache(ev_mod):
    """Seed the grounded judge's in-process verdict record for one test.

    _judge_verdicts_markdown now reads THIS eval's judgements
    (grounding.last_judgements()), not the persistent cache file, so tests
    populate that record. Yields a `seed(judgements, blanked=None)` callable
    and clears the record before and after."""
    g = ev_mod.grounding
    g.reset()

    def _seed(judgements, blanked=None, cap=None):
        g._LAST.clear()
        g._LAST.update(
            query_id="semantic_9",
            judgements=dict(judgements),
            blanked=list(blanked or []),
            cap=cap,
        )

    yield _seed
    g.reset()


def test_judge_verdicts_markdown_renders_in_submitted_order(ev_mod, verdict_cache):
    verdict_cache({
        "111": "perfectly_relevant_papers",
        "222": "not_relevant_papers",
    })
    md = ev_mod._judge_verdicts_markdown(
        ["222", "333", "111", "444"], "semantic_9", known_good={"333"}
    )
    lines = md.splitlines()
    assert lines[0] == "1. 222 — Not Relevant"
    assert lines[1] == "2. 333 — Perfectly Relevant (known-good)"
    assert lines[2] == "3. 111 — Perfectly Relevant"
    # The judge ran (other verdicts exist), so a missing verdict is labeled a
    # judge-side failure and the footer says it is neutral — evolution must
    # not read the gap as agent-caused or as a 0.
    assert lines[3] == "4. 444 — (judge call failed — excluded from scoring)"
    assert "2 Perfect / 1 lower / 1 no verdict, of 4 submitted" in md
    assert "neither credited nor penalized" in md


def test_judge_verdicts_markdown_labels_beyond_cap(ev_mod, verdict_cache):
    """Papers past the top-estimate judging cap are labeled as not-judged, not
    as a judge failure — so evolution doesn't chase 'missing' verdicts."""
    verdict_cache({"111": "perfectly_relevant_papers"}, cap=2)
    md = ev_mod._judge_verdicts_markdown(
        ["111", "222", "333", "444"], "semantic_9", known_good=set()
    )
    lines = md.splitlines()
    assert lines[0] == "1. 111 — Perfectly Relevant"
    assert lines[1] == "2. 222 — (judge call failed — excluded from scoring)"  # within cap, judge ran
    assert lines[2] == "3. 333 — (beyond scored depth — not judged)"
    assert lines[3] == "4. 444 — (beyond scored depth — not judged)"
    assert "2 beyond scored depth" in md


def test_judge_verdicts_markdown_handles_no_record(ev_mod, verdict_cache):
    """No in-process verdicts: known-good still reported; else None."""
    md = ev_mod._judge_verdicts_markdown(["111"], "semantic_9", known_good={"111"})
    assert "known-good" in md
    # No judgements and no known-good → nothing to report.
    assert ev_mod._judge_verdicts_markdown(["111"], "semantic_9", known_good=set()) is None


def test_judge_verdicts_markdown_empty_submission(ev_mod, verdict_cache):
    assert ev_mod._judge_verdicts_markdown([], "semantic_9", set()) is None


def _fake_log_with_submission(model_usage, results, **score_kwargs):
    import json as _json
    log = _fake_log(model_usage, **score_kwargs)
    log.samples[0].output = SimpleNamespace(completion=_json.dumps({
        "output": {"query_id": "q", "results": results}
    }))
    return log


def test_extract_emits_judge_verdicts_for_semantic(ev_mod, monkeypatch, verdict_cache):
    verdict_cache({"999": "highly_relevant_papers"})
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log_with_submission(
        {"openai/gpt-5.4-mini": (100, 10)},
        [{"paper_id": "CorpusId:999", "markdown_evidence": "x"}],
    )
    _, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    assert "judge_verdicts.md" in diag
    assert "1. 999 — Highly Relevant" in diag["judge_verdicts.md"]


def test_extract_no_verdicts_for_non_semantic(ev_mod, monkeypatch, verdict_cache):
    from inspect_ai.dataset import Sample
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log_with_submission(
        {"openai/gpt-5.4-mini": (100, 10)},
        [{"paper_id": "999", "markdown_evidence": "x"}],
    )
    sample = Sample(
        id="specific_7", input="q", target='{"corpus_ids": ["1"]}',
        metadata={"score_type": "specific_f1", "raw_query": "q"},
    )
    _, diag = ev._extract_score_and_diagnostics(log, sample, "")
    assert "judge_verdicts.md" not in diag


# --- score-calculation surfacing -------------------------------------------------


def test_score_calc_specific_renders_fractions(ev_mod):
    meta = {"standard_f1": 0.5, "precision": 1 / 3,
            "known_recall_at_full": 1.0, "relevant_predictions_at_full": 1}
    md = ev_mod._score_calculation_markdown(
        "specific_f1", meta, ["123456789", "111", "222"], ["123456789"], None)
    assert "submitted: 3 unique paper id(s)" in md
    assert "gold: 1" in md
    assert "hits (submitted ∩ gold): 1 → 123456789" in md
    assert "missed gold ids: (none)" in md
    assert "precision = hits / #submitted = 1/3 = 0.3333" in md
    assert "recall    = hits / #gold      = 1/1 = 1.0000" in md
    assert "score     = harmonic(precision, recall) = 0.5000" in md


def test_score_calc_specific_missed_gold_and_fraction_fallback(ev_mod):
    """Missed gold ids are listed; a fraction that does not reproduce the
    scorer's float is dropped in favor of the float alone."""
    meta = {"standard_f1": 0.0, "precision": 0.0,
            "known_recall_at_full": 0.25,  # inconsistent with 0/2 on purpose
            "relevant_predictions_at_full": 0}
    md = ev_mod._score_calculation_markdown(
        "metadata_f1", meta, ["999"], ["1", "2"], None)
    assert "missed gold ids: 1, 2" in md
    assert "precision = hits / #submitted = 0/1 = 0.0000" in md
    assert "recall    = hits / #gold      = 0.2500" in md  # no fraction shown


def test_score_calc_semantic_with_k(ev_mod):
    meta = {"adjusted_f1": 0.5306, "rank": 0.7312,
            "estimated_recall_at_estimate": 5 / 12,
            "estimated_recall_at_full": 7 / 12,
            "relevant_predictions_at_full": 7}
    md = ev_mod._score_calculation_markdown("semantic_f1", meta, [], [], 12)
    assert "rank   = 0.7312" in md
    assert "recall = 0.4167" in md
    assert "5 of K=12 estimated relevant" in md
    assert "score  = harmonic(rank, recall) = 0.5306" in md
    # 7 Perfect overall but only 5 within top K → the ordering-cost line.
    assert "2 more Perfect paper(s) ranked below position K" in md
    assert "judge_verdicts.md" in md  # pointer to the per-paper grades


def test_score_calc_semantic_k_derived_or_unknown(ev_mod):
    # No K passed in: derived from relevant_predictions / recall_at_full.
    meta = {"adjusted_f1": 0.4, "rank": 0.8,
            "estimated_recall_at_estimate": 0.5,
            "estimated_recall_at_full": 0.5,
            "relevant_predictions_at_full": 6}
    md = ev_mod._score_calculation_markdown("semantic_f1", meta, [], [], None)
    assert "K=12" in md
    assert "below position K" not in md  # all 6 Perfects are within top K

    # Zero hits: K underivable → labeled unknown, no count phrasing.
    meta = {"adjusted_f1": 0.0, "rank": 0.0,
            "estimated_recall_at_estimate": 0.0,
            "estimated_recall_at_full": 0.0,
            "relevant_predictions_at_full": 0}
    md = ev_mod._score_calculation_markdown("semantic_f1", meta, [], [], None)
    assert "K unknown" in md


def test_score_calc_missing_components_returns_none(ev_mod):
    assert ev_mod._score_calculation_markdown(
        "semantic_f1", {}, [], [], None) is None
    assert ev_mod._score_calculation_markdown(
        "specific_f1", {"precision": 1.0}, [], [], None) is None


def test_extract_emits_score_calculation_for_specific(ev_mod, monkeypatch):
    from inspect_ai.dataset import Sample
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log_with_submission(
        {"openai/gpt-5.4-mini": (100, 10)},
        [{"paper_id": "CorpusId:123", "markdown_evidence": ""},
         {"paper_id": "999", "markdown_evidence": ""}],
        score=2 / 3,
        score_metadata={"standard_f1": 2 / 3, "precision": 0.5,
                        "known_recall_at_full": 1.0,
                        "relevant_predictions_at_full": 1},
    )
    sample = Sample(
        id="specific_7", input="q", target='{"corpus_ids": ["123"]}',
        metadata={"score_type": "specific_f1", "raw_query": "q"},
    )
    _, diag = ev._extract_score_and_diagnostics(log, sample, "")
    md = diag["score_calculation.md"]
    assert "hits (submitted ∩ gold): 1 → 123" in md  # CorpusId: prefix normalized
    assert "precision = hits / #submitted = 1/2 = 0.5000" in md
    assert "recall    = hits / #gold      = 1/1 = 1.0000" in md
    assert "judge_verdicts.md" not in diag


def test_extract_emits_score_calculation_for_semantic(ev_mod, monkeypatch,
                                                      verdict_cache):
    verdict_cache({"999": "perfectly_relevant_papers"})
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log_with_submission(
        {"openai/gpt-5.4-mini": (100, 10)},
        [{"paper_id": "999", "markdown_evidence": "x"}],
        score=0.5,
        score_metadata={"adjusted_f1": 0.5, "rank": 1.0,
                        "estimated_recall_at_estimate": 1 / 3,
                        "estimated_recall_at_full": 1 / 3,
                        "relevant_predictions_at_full": 1},
    )
    _, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    assert "score_calculation.md" in diag
    assert "judge_verdicts.md" in diag  # both diagnostics coexist
    assert "K=3" in diag["score_calculation.md"]
    assert "harmonic(rank, recall) = 0.5000" in diag["score_calculation.md"]


def test_extract_surfaces_scorer_explanation_when_no_metadata(ev_mod, monkeypatch):
    """A scorer-side format rejection (value=0, no component metrics) must
    not be a silent zero — its explanation is the only record of why."""
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log(
        {"openai/gpt-5.4-mini": (100, 10)}, score=0.0,
        explanation="Agent output has an invalid format: boom",
    )
    score, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    assert score == 0.0
    assert "invalid format: boom" in diag["score_calculation.md"]


def test_extract_no_score_calculation_without_metadata_or_explanation(ev_mod, monkeypatch):
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log({"openai/gpt-5.4-mini": (100, 10)})
    _, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    assert "score_calculation.md" not in diag


def test_score_metadata_keys_match_astabench(ev_mod, monkeypatch):
    """Introspection pin: _score_calculation_markdown reads the scorer's
    component metrics by key name, and a missing key degrades the
    diagnostic to absent (best-effort by design). Every other test here
    fabricates that metadata, so an astabench key rename would keep the
    suite green while silently dropping the diagnostic — this test runs
    the REAL calc_standard_f1 / calc_adjusted_f1 and asserts the keys the
    renderer depends on. Same guard class as the module-load assert on
    GRADER_MODEL_NAME.

    get_normalizer_references is monkeypatched to a canned reference so
    the calc functions run pure-Python — no reference load, no network.
    """
    from astabench.evals.paper_finder import eval as pf_eval
    from astabench.evals.paper_finder.datamodel import (
        ExpectedAgentOutput, SingleResult,
    )
    from astabench.evals.paper_finder.relevance import Relevance

    monkeypatch.setattr(
        pf_eval, "get_normalizer_references",
        lambda: ({"semantic_1": 4, "specific_1": 1}, {}),
    )

    sem_output = ExpectedAgentOutput(query_id="semantic_1", results=[
        SingleResult(paper_id="11", markdown_evidence=""),
        SingleResult(paper_id="22", markdown_evidence=""),
    ])
    sem = pf_eval.calc_adjusted_f1(
        sem_output, "semantic_1",
        {"11": Relevance.PERFECT.value, "22": Relevance.NOT_RELEVANT.value},
        pf_eval.KTypes.ESTIMATED, pf_eval.KValues.AT_ESTIMATE,
    )
    assert {"adjusted_f1", "rank", "estimated_recall_at_estimate",
            "estimated_recall_at_full", "relevant_predictions_at_full"} <= set(sem)

    spec_output = ExpectedAgentOutput(query_id="specific_1", results=[
        SingleResult(paper_id="11", markdown_evidence=""),
    ])
    spec = pf_eval.calc_standard_f1(
        spec_output, "specific_1", {"11": Relevance.PERFECT.value},
        pf_eval.KTypes.KNOWN, pf_eval.KValues.AT_FULL,
    )
    assert {"standard_f1", "precision", "known_recall_at_full",
            "relevant_predictions_at_full"} <= set(spec)

    # And the renderer actually produces output from the real metadata.
    assert ev_mod._score_calculation_markdown(
        "semantic_f1", sem, [], [], None) is not None
    assert ev_mod._score_calculation_markdown(
        "specific_f1", spec, ["11"], ["11"], None) is not None


# --- submission.json ------------------------------------------------------------


def _submission_payload(n, evidence="some verbatim passage"):
    import json as _json
    return _json.dumps({"output": {"query_id": "q", "results": [
        {"paper_id": str(100 + i), "markdown_evidence": evidence}
        for i in range(n)
    ]}})


def test_submission_json_semantic_trims_beyond_cap(ev_mod):
    import json as _json
    out = ev_mod._submission_json(_submission_payload(4), "semantic_f1", 2)
    results = _json.loads(out)["output"]["results"]
    assert [r["paper_id"] for r in results] == ["100", "101", "102", "103"]
    assert results[0]["markdown_evidence"] == "some verbatim passage"
    assert results[1]["markdown_evidence"] == "some verbatim passage"
    assert results[2]["markdown_evidence"] == ev_mod.EVIDENCE_OMITTED_MARKER
    assert results[3]["markdown_evidence"] == ev_mod.EVIDENCE_OMITTED_MARKER
    assert out.count("\n") > 4  # pretty-printed, grep-able


def test_submission_json_uncapped_and_nonsemantic_kept_whole(ev_mod):
    import json as _json
    # No cap (uncapped judging): everything verbatim.
    out = ev_mod._submission_json(_submission_payload(3), "semantic_f1", None)
    results = _json.loads(out)["output"]["results"]
    assert all(r["markdown_evidence"] == "some verbatim passage" for r in results)
    # specific/metadata: never trimmed, even with a stale cap value.
    out = ev_mod._submission_json(_submission_payload(3), "specific_f1", 1)
    results = _json.loads(out)["output"]["results"]
    assert all(r["markdown_evidence"] == "some verbatim passage" for r in results)


def test_submission_json_raw_on_malformed_and_none_on_empty(ev_mod):
    # The scorer's own primary parse (json.loads) rejects this → stored raw.
    raw = 'Here are my results: {"output": {"results": []}}'
    assert ev_mod._submission_json(raw, "semantic_f1", 5) == raw
    assert ev_mod._submission_json("[1, 2]", "semantic_f1", 5) == "[1, 2]"
    assert ev_mod._submission_json("", "semantic_f1", 5) is None


def test_extract_emits_submission_json_and_no_agent_output(ev_mod, monkeypatch,
                                                           verdict_cache):
    verdict_cache({"999": "perfectly_relevant_papers"}, cap=1)
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log_with_submission(
        {"openai/gpt-5.4-mini": (100, 10)},
        [{"paper_id": "999", "markdown_evidence": "kept"},
         {"paper_id": "888", "markdown_evidence": "dropped"}],
    )
    import json as _json
    _, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    assert "agent_output" not in diag
    results = _json.loads(diag["submission.json"])["output"]["results"]
    assert results[0]["markdown_evidence"] == "kept"       # within cap=1
    assert results[1]["markdown_evidence"] == ev_mod.EVIDENCE_OMITTED_MARKER


def test_calibration_loader_reads_submission_json_and_skips_marker(ev_mod, tmp_path):
    """_load_docs must prefer submission.json over the legacy agent_output
    head, and must never treat the omission marker as real evidence to
    re-judge."""
    import importlib.util
    import json as _json

    spec = importlib.util.spec_from_file_location(
        "_check_judge_calibration", PFB_DIR / "_check_judge_calibration.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pdir = tmp_path / "iteration_001" / "agent_x" / "problems" / "semantic_9"
    pdir.mkdir(parents=True)
    (pdir / "gold_criteria.md").write_text(_json.dumps(
        {"relevance_criteria": [{"name": "n", "description": "d", "weight": 1.0}]}
    ))
    (pdir / "submission.json").write_text(_json.dumps({"output": {"results": [
        {"paper_id": "1", "markdown_evidence": "real evidence"},
        {"paper_id": "2", "markdown_evidence": ev_mod.EVIDENCE_OMITTED_MARKER},
        {"paper_id": "3", "markdown_evidence": ""},
    ]}}))
    # A stale legacy file must be ignored when submission.json exists.
    (pdir / "agent_output").write_text(
        '{"output": {"results": [{"paper_id": "9", "markdown_evidence": "stale"}]}}'
    )

    per_query = mod._load_docs(tmp_path)
    assert per_query["semantic_9"]["docs"] == [("1", "real evidence")]


# --- tool-failure legibility ----------------------------------------------------


def _leaf_429():
    import httpx
    req = httpx.Request("GET", "https://asta-tools.allen.ai/mcp/v1")
    resp = httpx.Response(429, request=req)
    return httpx.HTTPStatusError("429", request=req, response=resp)


def test_tool_failure_summary_names_rate_limit(ev_mod):
    """The exact leak evolution misread for 20 iterations in run
    asta_paper_finder_20260716_072622: a TaskGroup ExceptionGroup whose
    str() hides the 429 leaf."""
    eg = ExceptionGroup("unhandled errors in a TaskGroup", [_leaf_429()])
    assert ev_mod._tool_failure_summary(eg) == (
        "HTTP 429 rate-limited (retry budget exhausted)"
    )


def test_tool_failure_summary_nested_leaves_and_dedup(ev_mod):
    import anyio
    import httpx
    inner = ExceptionGroup("inner", [_leaf_429(), httpx.ReadTimeout("read")])
    eg = ExceptionGroup("outer", [inner, anyio.BrokenResourceError(), _leaf_429()])
    s = ev_mod._tool_failure_summary(eg)
    assert s.count("HTTP 429") == 1  # deduped across the nesting
    assert "transport timeout (ReadTimeout)" in s
    assert "connection broken mid-call (BrokenResourceError)" in s


def test_tool_failure_summary_generic_leaf_and_bare_exception(ev_mod):
    eg = ExceptionGroup("outer", [ValueError("boom")])
    assert ev_mod._tool_failure_summary(eg) == "ValueError: boom"
    # A bare exception is already legible — never rewritten.
    assert ev_mod._tool_failure_summary(RuntimeError("plain")) is None


async def _tool_that_raises(exc):
    async def search_papers(query: str) -> str:
        """Search for papers.

        Args:
            query: The search query.

        Returns:
            Search results.
        """
        raise exc
    return search_papers


def test_wrapped_tool_reraises_with_named_cause(ev_mod):
    import asyncio
    from inspect_ai.tool import ToolDef  # noqa: F401 - ensures importable

    eg = ExceptionGroup("unhandled errors in a TaskGroup", [_leaf_429()])
    tool = asyncio.run(_tool_that_raises(eg))
    [wrapped] = ev_mod._wrap_tools_for_provenance([tool])
    with pytest.raises(RuntimeError, match="HTTP 429 rate-limited"):
        asyncio.run(wrapped(query="x"))


def test_wrapped_tool_propagates_cancellation_group(ev_mod):
    """A BaseExceptionGroup carrying cancellation is not an Exception and
    must pass through untouched — rewriting it would break asyncio's
    cancellation semantics."""
    import asyncio

    beg = BaseExceptionGroup("cancelled", [asyncio.CancelledError()])
    tool = asyncio.run(_tool_that_raises(beg))
    [wrapped] = ev_mod._wrap_tools_for_provenance([tool])
    with pytest.raises(BaseExceptionGroup) as exc_info:
        asyncio.run(wrapped(query="x"))
    assert not isinstance(exc_info.value, RuntimeError)


def test_transport_docs_match_astabench_defaults(ev_mod):
    """background.md's 'Tool-call transport' numbers (connect 5 s, read
    300 s, 10 retry attempts, ~5 min worst-case backoff) document
    astabench's actual client/retry defaults — pin both ends so an
    upstream bump can't silently stale the doc."""
    import inspect as pyinspect
    from astabench.tools import asta_tools

    sig = pyinspect.signature(asta_tools.create_server_streamable_http)
    assert sig.parameters["timeout"].default == 5
    assert sig.parameters["sse_read_timeout"].default == 300

    sig = pyinspect.signature(asta_tools.make_retry_wrapper)
    assert sig.parameters["max_retries"].default == 10
    base = sig.parameters["base_delay"].default
    mult = sig.parameters["backoff_multiplier"].default
    cap = sig.parameters["max_delay"].default
    worst_case_sleep = sum(min(base * mult**i, cap) for i in range(10))
    assert 240 <= worst_case_sleep <= 360  # the doc's "~5 minutes"

    background = (PFB_DIR / "background.md").read_text()
    for needle in ("**Connect: 5 s**", "**Response read: 300 s**",
                   "up to 10 attempts", "10 requests/second"):
        assert needle in background, f"background.md lost: {needle}"


# --- safe judge-cache writer ----------------------------------------------------


def _pf_utils(ev_mod):
    from astabench.evals.paper_finder import paper_finder_utils
    return paper_finder_utils


def test_safe_cache_patch_installed_on_both_bindings(ev_mod):
    """The from-import binding in eval.py is the one get_llm_relevance
    actually calls; patching only the origin module would be a no-op."""
    from astabench.evals.paper_finder import eval as pf_eval
    from astabench.evals.paper_finder import paper_finder_utils
    assert getattr(paper_finder_utils.update_references, "_robophd_safe_cache", False)
    assert getattr(pf_eval.update_references, "_robophd_safe_cache", False)


def _run_update(ev_mod, qid, judgements):
    import asyncio
    from astabench.evals.paper_finder import paper_finder_utils
    asyncio.run(paper_finder_utils.update_references(qid, judgements))


@pytest.fixture()
def tmp_cache_path(ev_mod, monkeypatch, tmp_path):
    from astabench.evals.paper_finder import paper_finder_utils
    p = tmp_path / "detailed_reference.json"
    monkeypatch.setattr(paper_finder_utils, "detailed_reference_path", str(p))
    return p


def test_safe_cache_merges_and_stays_valid(ev_mod, tmp_cache_path):
    import json as _json
    _run_update(ev_mod, "semantic_1", {"a": "perfectly_relevant_papers"})
    _run_update(ev_mod, "semantic_1", {"b": "not_relevant_papers"})
    _run_update(ev_mod, "semantic_2", {"c": "highly_relevant_papers"})
    data = _json.loads(tmp_cache_path.read_text())
    assert data["semantic_1"] == {
        "a": "perfectly_relevant_papers", "b": "not_relevant_papers",
    }
    assert data["semantic_2"] == {"c": "highly_relevant_papers"}


def test_safe_cache_recovers_valid_prefix(ev_mod, tmp_cache_path):
    """A pre-fix torn write leaves valid JSON + trailing garbage; the
    safe writer must recover the prefix instead of dropping the cache
    (upstream's behavior)."""
    import json as _json
    good = _json.dumps({"semantic_9": {"x": "perfectly_relevant_papers"}})
    tmp_cache_path.write_text(good + '"stale-tail-garbage"}')
    _run_update(ev_mod, "semantic_10", {"y": "not_relevant_papers"})
    data = _json.loads(tmp_cache_path.read_text())
    assert data["semantic_9"] == {"x": "perfectly_relevant_papers"}  # recovered
    assert data["semantic_10"] == {"y": "not_relevant_papers"}       # merged


def test_safe_cache_concurrent_writers_never_corrupt(ev_mod, tmp_cache_path):
    """8 threads x 20 updates each, every call opening its own lock fd
    (flock serializes across fds, hence across threads AND processes).
    The file must parse after the storm and contain all 160 entries."""
    import json as _json
    errors = []

    def writer(tid):
        try:
            for i in range(20):
                _run_update(ev_mod, f"q_{tid}_{i}", {"p": "perfectly_relevant_papers"})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    data = _json.loads(tmp_cache_path.read_text())
    assert len(data) == 160


def test_apply_cache_redirect_env(ev_mod, monkeypatch, tmp_path):
    """$PF_JUDGE_CACHE_PATH redirects both the writer and the reader path and
    clears the memoized cache, so a per-run / pristine cache is honored."""
    from astabench.evals.paper_finder import paper_finder_utils, eval as pf_eval
    # Snapshot via monkeypatch so the global path is restored at teardown
    # (the redirect sets it directly, not through monkeypatch).
    monkeypatch.setattr(
        paper_finder_utils, "detailed_reference_path",
        paper_finder_utils.detailed_reference_path,
    )
    monkeypatch.setattr(pf_eval, "_detailed_reference", pf_eval._detailed_reference)
    target = tmp_path / "nested" / "run_cache.json"
    monkeypatch.setenv(ev_mod.CACHE_PATH_ENV, str(target))
    # seed a stale memo to prove the redirect clears it
    pf_eval._detailed_reference = {"stale": {}}
    ev_mod._apply_cache_redirect()
    assert paper_finder_utils.detailed_reference_path == str(target)
    assert pf_eval._detailed_reference is None
    assert target.parent.is_dir()  # created eagerly


def test_apply_cache_redirect_noop_without_env(ev_mod, monkeypatch):
    from astabench.evals.paper_finder import paper_finder_utils
    monkeypatch.delenv(ev_mod.CACHE_PATH_ENV, raising=False)
    before = paper_finder_utils.detailed_reference_path
    ev_mod._apply_cache_redirect()
    assert paper_finder_utils.detailed_reference_path == before


def test_apply_training_grader_override(ev_mod, monkeypatch):
    """$PF_TRAINING_GRADER_MODEL overrides the grader; the id must be in
    JUDGE_MODEL_IDS so its spend stays in the judge bucket. The lenient
    output normalizer must be installed alongside — alternate judges emit
    rare near-JSON that astabench's strict parser would otherwise drop as
    Not Relevant."""
    import _judge_normalize
    from astabench.evals.paper_finder import relevance
    monkeypatch.setattr(relevance, "GRADER_MODEL_NAME", relevance.GRADER_MODEL_NAME)
    monkeypatch.setattr(
        relevance, "extract_json_from_response", relevance.extract_json_from_response
    )
    monkeypatch.setenv(ev_mod.TRAINING_GRADER_ENV, "openai/gpt-5.6-luna")
    ev_mod._apply_training_grader()
    assert relevance.GRADER_MODEL_NAME == "openai/gpt-5.6-luna"
    assert relevance.extract_json_from_response is _judge_normalize._lenient_extract_json


def test_apply_training_grader_rejects_unbilled_model(ev_mod, monkeypatch):
    """A grader not in JUDGE_MODEL_IDS would misbill judge spend to the agent —
    fail loudly rather than silently. gpt-5.4-nano is deliberately in this
    class now: it FAILED the 2026-07-20 calibration (kappa ~0.52, severe
    Perfect-deflation) and was removed from JUDGE_MODEL_IDS."""
    for bad in ("openai/gpt-5.4-mini", "openai/gpt-5.4-nano"):
        monkeypatch.setenv(ev_mod.TRAINING_GRADER_ENV, bad)
        with pytest.raises(RuntimeError, match="JUDGE_MODEL_IDS"):
            ev_mod._apply_training_grader()


def test_apply_training_grader_noop_without_env(ev_mod, monkeypatch):
    """Stock path: no grader override AND no normalizer patch — official
    parity requires astabench's strict parser untouched."""
    from astabench.evals.paper_finder import relevance
    monkeypatch.delenv(ev_mod.TRAINING_GRADER_ENV, raising=False)
    before_grader = relevance.GRADER_MODEL_NAME
    before_extract = relevance.extract_json_from_response
    ev_mod._apply_training_grader()
    assert relevance.GRADER_MODEL_NAME == before_grader
    assert relevance.extract_json_from_response is before_extract


def test_extract_emits_judge_repairs_as_string_diagnostic(ev_mod, monkeypatch,
                                                          verdict_cache):
    """judge_format_repairs must be emitted as a STRING under a .md key:
    the domain layer persists string diagnostics as files but drops
    unknown dict keys from result.json's fixed schema — the original
    dict-shaped emission never reached disk (0 of 64 semantic problems
    in the first luna run). A revert to the dict shape would silently
    reproduce that vanishing."""
    import _judge_normalize
    verdict_cache({"999": "perfectly_relevant_papers"})
    monkeypatch.setattr(
        ev_mod.PaperFinderEvaluator, "_estimate_cost",
        staticmethod(lambda model_name, counts: 0.0),
    )
    ev = _bare_evaluator(ev_mod)
    log = _fake_log_with_submission(
        {"openai/gpt-5.4-mini": (100, 10)},
        [{"paper_id": "999", "markdown_evidence": "x"}],
    )
    # Simulate the lenient normalizer having repaired one judge response.
    _judge_normalize.reset()
    _judge_normalize._REPAIR["recovered"] = 1
    _judge_normalize._REPAIR["strict_ok"] = 4
    try:
        _, diag = ev._extract_score_and_diagnostics(log, _fake_sample(ev_mod), "")
    finally:
        _judge_normalize.reset()
    assert "judge_format_repairs" not in diag  # the dict shape must stay gone
    md = diag["judge_format_repairs.md"]
    assert isinstance(md, str)
    assert "1 recovered" in md and "5 judge responses" in md


def test_apply_training_grader_no_prose_profile(ev_mod, monkeypatch):
    """PF_TRAINING_GRADER_PROMPT=no-prose swaps the judge template — but
    only alongside the judge override; alone it must hard-error (the
    stock GPT-4o basis stays byte-identical to official scoring, and 4o
    failed the no-prose calibration)."""
    import _judge_normalize
    from astabench.evals.paper_finder import relevance
    monkeypatch.setattr(relevance, "GRADER_MODEL_NAME", relevance.GRADER_MODEL_NAME)
    monkeypatch.setattr(
        relevance, "extract_json_from_response", relevance.extract_json_from_response
    )
    monkeypatch.setattr(
        relevance,
        "relevance_criteria_judgement_prompt_with_relevant_snippets_after",
        relevance.relevance_criteria_judgement_prompt_with_relevant_snippets_after,
    )
    monkeypatch.setenv(ev_mod.TRAINING_GRADER_ENV, "openai/gpt-5.6-luna")
    monkeypatch.setenv(ev_mod.TRAINING_GRADER_PROMPT_ENV, "no-prose")
    ev_mod._apply_training_grader()
    assert (relevance.relevance_criteria_judgement_prompt_with_relevant_snippets_after
            is _judge_normalize.NO_PROSE_JUDGE_TEMPLATE)

    # Profile without judge override: refuse.
    monkeypatch.delenv(ev_mod.TRAINING_GRADER_ENV)
    with pytest.raises(RuntimeError, match="without"):
        ev_mod._apply_training_grader()

    # Unknown profile: refuse.
    monkeypatch.setenv(ev_mod.TRAINING_GRADER_ENV, "openai/gpt-5.6-luna")
    monkeypatch.setenv(ev_mod.TRAINING_GRADER_PROMPT_ENV, "verbose")
    with pytest.raises(RuntimeError, match="unknown"):
        ev_mod._apply_training_grader()


def test_evidence_truncation_markdown(ev_mod, verdict_cache):
    """evidence_truncation.md renders clipped papers + total, and is absent
    when nothing was clipped (cap off or all evidence compliant)."""
    g = ev_mod.grounding
    g._LAST.update(truncated=[("111", 3391, 2500), ("222", 2600, 2500)])
    md = ev_mod._evidence_truncation_markdown()
    assert "2 paper(s)" in md and "2500-character cap" in md
    assert "111: 3,391 → 2,500" in md and "222: 2,600 → 2,500" in md
    assert f"{(3391-2500)+(2600-2500):,} chars discarded" in md
    g.reset()
    assert ev_mod._evidence_truncation_markdown() is None


def test_judge_price_override_prices_luna(ev_mod):
    """litellm 1.88.1 (pinned — the leaderboard billing basis) predates
    gpt-5.6-luna; JUDGE_PRICE_OVERRIDES must price it so judge cost never
    silently reports $0. Cache-read tokens bill at the cached rate."""
    cost = ev_mod.PaperFinderEvaluator._estimate_cost(
        "openai/gpt-5.6-luna",
        {"input_tokens": 1_000_000, "output_tokens": 100_000, "total_tokens": 1_100_000},
    )
    assert cost == pytest.approx(1.00 + 0.60)  # $1/M in + $6/M out
    cost = ev_mod.PaperFinderEvaluator._estimate_cost(
        "openai/gpt-5.6-luna",
        {"input_tokens": 1_000_000, "output_tokens": 100_000,
         "total_tokens": 1_100_000, "input_tokens_cache_read": 500_000},
    )
    assert cost == pytest.approx(0.5 * 1.00 + 0.5 * 0.10 + 0.60)


# --- _head_tail_truncate ------------------------------------------------------


def test_truncate_short_string_passes_through(ev_mod):
    s = "x" * 100
    assert ev_mod._head_tail_truncate(s) == s


def test_truncate_long_string_keeps_head_and_tail(ev_mod):
    s = "H" * 200 + "M" * 5000 + "T" * 1500
    out = ev_mod._head_tail_truncate(s)
    assert out.startswith("H" * 200)
    assert out.endswith("T" * 1500)
    assert "chars truncated" in out
    assert len(out) < len(s)


def test_truncate_marker_reports_count(ev_mod):
    s = "a" * 200 + "b" * 1000 + "c" * 1500
    out = ev_mod._head_tail_truncate(s)
    assert "(1000 chars truncated)" in out


# --- source-level guards ------------------------------------------------------


def test_subprocess_is_imported():
    """Finding A regression guard: the archived example used
    subprocess.run without importing subprocess — first eval crashed
    with NameError."""
    tree = ast.parse(EVALUATOR_SRC)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" in imported
    assert "signal" in imported
    assert "time" in imported


def test_killpg_machinery_intact():
    """A refactor that drops start_new_session/killpg/bounded drain
    silently re-introduces the orphan-grandchild hang class."""
    assert "start_new_session=True" in EVALUATOR_SRC
    assert "os.killpg" in EVALUATOR_SRC
    assert "proc.communicate(timeout=30)" in EVALUATOR_SRC


def test_no_bare_error_diagnostics_key():
    """Every failure-path diagnostic must use "error.md" — the framework's
    failure detection (domains/external/domain.py) checks that exact key,
    and bare "error" records crashes as error:false in result.json."""
    for bad in ('"error":', "'error':", '["error"]', "diagnostics[\"error\"]"):
        assert bad not in EVALUATOR_SRC, (
            f"bare {bad} found in evaluator.py — use \"error.md\""
        )
    assert '"error.md"' in EVALUATOR_SRC


# --- _estimate_cost / bundled price map ---------------------------------------
#
# Ported from asta_ds1000's unit_tests/test_evaluator.py (ae1e410): the
# pricing machinery here is a byte-for-byte copy of ds1000's, and the
# repo's self-contained-examples convention means the ds1000 tests do
# NOT protect this copy — these duplicates do. If the two copies are
# ever deliberately diverged, these tests fail loudly on this side,
# which is the alarm we want (see README "Code work" for the
# extract-to-shared-module endpoint). Two invariants under pin:
#
# 1. Reasoning-token fold: Gemini reports reasoning separately and
#    excludes it from output_tokens; billed at the output rate. The
#    strict token-arithmetic guard must fire for the Gemini pattern
#    only — OpenAI (reasoning inside output) and Anthropic (cache
#    tokens inflating total) must NOT fold.
#
# 2. Bundled-map basis: internal costs are priced from litellm's BUNDLED
#    snapshot (what `astabench score` bills under
#    LITELLM_LOCAL_MODEL_COST_MAP=True), NOT the live map. Models absent
#    from the snapshot fall back to the live map WITH a one-shot warning;
#    a snapshot that fails to load warns loudly instead of silently
#    repricing everything on the live map.
#
# All tests inject a fixture price map (and stub litellm.cost_per_token
# for the fallback path) so they are hermetic against the installed
# litellm's actual map contents.

FIXTURE_PRICES = {
    # litellm-1.88.1 bundled rates for the v0_0_6 models (per token)
    "gemini/gemini-3.1-flash-lite": {
        "input_cost_per_token": 4.5e-07, "output_cost_per_token": 2.7e-06,
    },
    "gpt-5.4-mini": {
        "input_cost_per_token": 7.5e-07, "output_cost_per_token": 4.5e-06,
    },
    "claude-haiku-4-5-20251001": {
        "input_cost_per_token": 1e-06, "output_cost_per_token": 5e-06,
    },
    "gpt-5.4-2026-03-05": {
        "input_cost_per_token": 2.5e-06, "output_cost_per_token": 1.5e-05,
    },
}


@pytest.fixture()
def bundled_fixture_map(ev_mod, monkeypatch):
    """Point the bundled-map cache at FIXTURE_PRICES for one test."""
    monkeypatch.setattr(ev_mod, "_BUNDLED_PRICE_MAP", dict(FIXTURE_PRICES))
    return ev_mod


def _est(mod, model, **counts):
    return mod.PaperFinderEvaluator._estimate_cost(model, counts)


def test_gemini_reasoning_tokens_are_billed(bundled_fixture_map):
    """Gemini pattern (input == total - output - reasoning): fold fires."""
    mod = bundled_fixture_map
    cost = _est(
        mod, "google/gemini-3.1-flash-lite",
        input_tokens=1_000_000, output_tokens=100_000,
        total_tokens=1_300_000, reasoning_tokens=200_000,
    )
    expected = 1_000_000 * 4.5e-07 + (100_000 + 200_000) * 2.7e-06
    assert cost == pytest.approx(expected)


def test_openai_reasoning_not_double_billed(bundled_fixture_map):
    """OpenAI pattern (reasoning already inside output): fold must NOT fire."""
    mod = bundled_fixture_map
    with_r = _est(
        mod, "gpt-5.4-mini",
        input_tokens=1000, output_tokens=500, total_tokens=1500,
        reasoning_tokens=300,
    )
    without_r = _est(
        mod, "gpt-5.4-mini",
        input_tokens=1000, output_tokens=500, total_tokens=1500,
        reasoning_tokens=0,
    )
    assert with_r == pytest.approx(without_r)


def test_anthropic_cache_pattern_not_mistaken_for_reasoning(bundled_fixture_map):
    """Cache tokens inflate total (input excludes them); no reasoning fold."""
    mod = bundled_fixture_map
    cost = _est(
        mod, "claude-haiku-4-5-20251001",
        input_tokens=1000, output_tokens=500, total_tokens=2500,
        reasoning_tokens=0,
    )
    expected = 1000 * 1e-06 + 500 * 5e-06
    assert cost == pytest.approx(expected)


def test_bundled_map_wins_over_live_map(bundled_fixture_map, monkeypatch):
    """When the snapshot has the model, litellm.cost_per_token is never hit."""
    mod = bundled_fixture_map
    import litellm

    def _boom(**kwargs):
        raise AssertionError("live map must not be consulted for bundled models")

    monkeypatch.setattr(litellm, "cost_per_token", _boom)
    cost = _est(
        mod, "gpt-5.4-2026-03-05",
        input_tokens=1000, output_tokens=100, total_tokens=1100,
        reasoning_tokens=0,
    )
    assert cost == pytest.approx(1000 * 2.5e-06 + 100 * 1.5e-05)


def test_missing_model_falls_back_to_live_map_with_warning(
    bundled_fixture_map, monkeypatch, caplog
):
    mod = bundled_fixture_map
    import litellm

    monkeypatch.setattr(
        litellm, "cost_per_token", lambda **kw: (0.001, 0.002)
    )
    mod._live_map_fallback_warned.discard("brand-new-model")
    with caplog.at_level("WARNING"):
        cost = _est(
            mod, "brand-new-model",
            input_tokens=1000, output_tokens=100, total_tokens=1100,
            reasoning_tokens=0,
        )
    assert cost == pytest.approx(0.003)
    assert any("NOT on the leaderboard's billing basis" in r.message for r in caplog.records)
    # One-shot: second call must not warn again
    with caplog.at_level("WARNING"):
        caplog.clear()
        _est(
            mod, "brand-new-model",
            input_tokens=1000, output_tokens=100, total_tokens=1100,
            reasoning_tokens=0,
        )
    assert not any("billing basis" in r.message for r in caplog.records)


def test_snapshot_load_failure_warns_loudly(ev_mod, monkeypatch, caplog):
    """A missing/renamed snapshot must not silently reprice on the live map."""
    monkeypatch.setattr(ev_mod, "_BUNDLED_PRICE_MAP", None)

    import litellm
    real_file = litellm.__file__
    monkeypatch.setattr(litellm, "__file__", "/nonexistent/litellm/__init__.py")
    try:
        with caplog.at_level("WARNING"):
            result = ev_mod._bundled_price_map()
    finally:
        monkeypatch.setattr(litellm, "__file__", real_file)
    assert result == {}
    assert any(
        "bundled price snapshot" in r.message for r in caplog.records
    ), "snapshot load failure must warn, not silently drift to the live map"


def test_v0_0_6_official_cost_regression(bundled_fixture_map):
    """Golden regression: the recorded model_usage from asta_ds1000's
    officially-scored v0_0_6 run must reproduce the official astabench
    per-problem cost ($0.004280335) on the frozen fixture rates. The
    fixture is ds1000 provenance, but it pins THIS module's copy of the
    pricing function (reasoning fold + bundled basis, end to end) —
    which is byte-identical and must stay on the same billing basis."""
    mod = bundled_fixture_map
    usage = {
        "google/gemini-3.1-flash-lite": dict(
            input_tokens=1_465_500, output_tokens=163_410,
            total_tokens=1_862_265, reasoning_tokens=233_355,
        ),
        "gpt-5.4-mini": dict(
            input_tokens=1_171_114, output_tokens=79_480,
            total_tokens=1_250_594, reasoning_tokens=0,
        ),
        "claude-haiku-4-5-20251001": dict(
            input_tokens=254_413, output_tokens=42_332,
            total_tokens=296_745, reasoning_tokens=0,
        ),
        "gpt-5.4-2026-03-05": dict(
            input_tokens=141_163, output_tokens=4_439,
            total_tokens=145_602, reasoning_tokens=0,
        ),
    }
    total = sum(_est(mod, m, **c) for m, c in usage.items())
    assert total / 900 == pytest.approx(0.004280335, abs=1e-6)
