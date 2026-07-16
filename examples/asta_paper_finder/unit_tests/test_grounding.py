"""Tests for evidence grounding (_grounding.py).

Covers provenance extraction from the two payload shapes that matter
(nested paper lists; snippet_search where quotable text is a sibling of the
paper block), the honest-vs-fabricated matching boundary, evidence-hash cache
keying, and the end-to-end behavior of the patched get_llm_relevance:
ungrounded evidence is judged Not Relevant with NO judge call, grounded
evidence is judged and cached under a composite (paper, evidence-hash) key.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _grounding as g  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    g.reset()
    yield
    g.reset()


# --- provenance extraction -------------------------------------------------

def test_records_nested_paper_list():
    g.record_tool_result({"data": [
        {"corpusId": 249209614, "title": "CogVideo: Text-to-Video Generation",
         "abstract": "Autoregressive transformers revolutionized generation.",
         "tldr": {"text": "A 9B text-to-video model."}},
        {"corpusId": 111, "title": "Other", "abstract": "Unrelated protein work."},
    ]})
    assert g.check_evidence("249209614", "Autoregressive transformers revolutionized generation")[0]
    assert g.check_evidence("249209614", "A 9B text-to-video model")[0]
    # text from a DIFFERENT paper must not ground this one
    assert not g.check_evidence("249209614", "Unrelated protein work")[0]


def test_records_nested_tldr_despite_hash_paperid():
    """Regression: S2 payloads carry a hex `paperId` alongside the numeric
    `corpusId`. paperId must NOT be treated as a corpus id — otherwise a
    spurious digit-run from the hash registers a phantom second cid, the
    single-cid vacuum is skipped, and nested fields (tldr.text) are dropped."""
    g.record_tool_result({"data": [{
        "paperId": "077c713bccd9d2c7fde68d4cbde06ab0f07a6855",
        "corpusId": 235187266,
        "title": "ConSERT",
        "abstract": "learning high-quality sentence representations",
        "tldr": {"model": "tldr@v2.0.0",
                 "text": "ConSERT is presented, a contrastive framework"},
    }]})
    # both the direct abstract AND the nested tldr must ground
    assert g.check_evidence("235187266", "learning high-quality sentence representations")[0]
    assert g.check_evidence("235187266", "ConSERT is presented, a contrastive framework")[0]


def test_records_snippet_sibling_text():
    # snippet_search: the quotable text is a sibling of the paper block, not a child
    g.record_tool_result([
        {"paper": {"corpusId": 777, "title": "Snippet Paper"},
         "snippet": {"text": "Continual learning over dynamic corpora requires index rewriting."},
         "score": 0.9},
    ])
    assert g.check_evidence("777", "Continual learning over dynamic corpora requires index rewriting")[0]
    assert g.check_evidence("777", "Snippet Paper")[0]


def test_records_json_string_payload():
    import json
    g.record_tool_result(json.dumps({"data": [{"corpusId": 5, "title": "Hello World Paper"}]}))
    assert g.check_evidence("5", "Hello World Paper")[0]


def test_records_contenttext_list_payload():
    """The real Asta MCP tools return a list of inspect ContentText objects,
    each .text a JSON string for one paper. Regression guard: synthetic dict
    payloads alone would miss this and blank every honest agent."""
    import json
    from types import SimpleNamespace

    def content(obj):  # ContentText duck: has a .text str
        return SimpleNamespace(type="text", text=json.dumps(obj))

    g.record_tool_result([
        content({"corpusId": 278715187, "title": "Efficient Attention via Pre-Scoring",
                 "abstract": "Prioritizing informative keys in transformers."}),
        content({"corpusId": 111, "title": "Other Paper"}),
    ])
    assert g.check_evidence("278715187", "Prioritizing informative keys in transformers")[0]
    assert g.check_evidence("278715187", "Efficient Attention via Pre-Scoring")[0]


def test_opaque_string_payload_is_ignored():
    g.record_tool_result("some non-JSON tool message")
    assert not g.check_evidence("5", "anything")[0]


# --- matching boundary -----------------------------------------------------

def test_honest_title_dash_abstract_passes():
    g.record_tool_result({"data": [
        {"corpusId": 42, "title": "Deep Nets",
         "abstract": "We train very deep residual networks on ImageNet."},
    ]})
    ok, fail = g.check_evidence(
        "42", "Deep Nets — We train very deep residual networks on ImageNet"
    )
    assert ok and not fail


def test_fabricated_evidence_fails():
    g.record_tool_result({"data": [{"corpusId": 42, "title": "Deep Nets",
                                    "abstract": "We train residual networks."}]})
    ok, fail = g.check_evidence("42", "This paper proves P=NP via quantum language models.")
    assert not ok and fail


def test_evidence_for_unretrieved_paper_fails():
    ok, fail = g.check_evidence("999999", "Any claim at all.")
    assert not ok and fail


def test_empty_evidence_is_grounded():
    assert g.check_evidence("1", "")[0]
    assert g.check_evidence("1", "   ")[0]


def test_cross_boundary_match_is_deterministic():
    """A passage that straddles two spans must ground on a fixed (sorted) join
    order, not on nondeterministic set iteration. Spans 'aaa' and 'zzz': under
    sorted order the blob is 'aaa zzz', so 'aaa zzz' grounds and 'zzz aaa' does
    not — the same answer every run, regardless of insertion or hash order."""
    for _ in range(3):
        g.reset()
        # record in different insertion orders; outcome must not change
        g.record_tool_result({"data": [{"corpusId": 9, "title": "zzz", "abstract": "aaa"}]})
        assert g.check_evidence("9", "aaa zzz")[0] is True
        assert g.check_evidence("9", "zzz aaa")[0] is False


def test_unicode_and_whitespace_normalized():
    g.record_tool_result({"data": [{"corpusId": 7,
                                    "abstract": "café  résumé\nnaïve"}]})
    # different unicode presentation + reflowed whitespace still matches
    assert g.check_evidence("7", "café résumé naïve")[0]


def test_punctuation_style_normalized():
    """Smart quotes / dashes differing only in style must still ground."""
    # retrieved text uses ASCII apostrophe and hyphen
    g.record_tool_result({"data": [{"corpusId": 8,
                                    "abstract": "don't over-fit the model"}]})
    # agent quotes with a curly apostrophe and an en-dash
    assert g.check_evidence("8", "don’t over–fit the model")[0]
    # and the reverse direction (retrieved curly, agent ASCII)
    g.reset()
    g.record_tool_result({"data": [{"corpusId": 8,
                                    "abstract": "the “attention” mechanism"}]})
    assert g.check_evidence("8", 'the "attention" mechanism')[0]


# --- partial blanking ------------------------------------------------------

def test_scrub_keeps_grounded_drops_ungrounded():
    g.record_tool_result({"data": [{"corpusId": 12, "title": "Real Title",
                                    "abstract": "a real retrieved sentence"}]})
    # one grounded passage, one fabricated, joined by ' ... '
    scrubbed, dropped = g.scrub_evidence(
        "12", "a real retrieved sentence ... an invented claim about nothing")
    assert scrubbed == "a real retrieved sentence"      # kept the grounded one
    assert dropped == ["an invented claim about nothing"]


def test_scrub_fully_grounded_returns_evidence_unchanged():
    g.record_tool_result({"data": [{"corpusId": 12, "abstract": "alpha beta gamma"}]})
    scrubbed, dropped = g.scrub_evidence("12", "alpha beta")
    assert scrubbed == "alpha beta" and dropped == []


def test_scrub_nothing_grounded_returns_empty():
    g.record_tool_result({"data": [{"corpusId": 12, "abstract": "real text"}]})
    scrubbed, dropped = g.scrub_evidence("12", "invented one ... invented two")
    assert scrubbed == "" and len(dropped) == 2


# --- cache keying ----------------------------------------------------------

def test_cache_key_evidence_sensitive():
    assert g.cache_key("42", "abc def") == g.cache_key("42", "abc def")
    assert g.cache_key("42", "abc") != g.cache_key("42", "abc def")
    # blanked evidence collapses to one key per paper
    assert g.cache_key("42", "") == g.cache_key("42", "   ")
    assert g.paper_id_of(g.cache_key("249209614", "x")) == "249209614"


# --- end-to-end through the patched judge ----------------------------------

def _install_and_get_patched():
    g.install_grounded_judge()
    from astabench.evals.paper_finder import task as t
    return t.get_llm_relevance


def _output(results):
    return SimpleNamespace(
        query_id="semantic_9",
        results=[SimpleNamespace(paper_id=pid, markdown_evidence=ev) for pid, ev in results],
    )


def test_e2e_ungrounded_blanked_without_judge_call(monkeypatch):
    from astabench.evals.paper_finder import eval as pe
    from astabench.evals.paper_finder import relevance as rel

    judge_calls = {"n": 0}

    async def _fake_judge(docs, criteria):
        judge_calls["n"] += 1
        return {d.corpus_id: rel.Relevance.PERFECT.value for d in docs}

    monkeypatch.setattr(rel, "load_relevance_judgement", _fake_judge)
    monkeypatch.setattr(pe, "get_normalizer_references", lambda: ({}, {}))

    get_llm_relevance = _install_and_get_patched()
    metric = SimpleNamespace(known_to_be_good=set(), relevance_criteria=[])

    # Paper 42 retrieved; evidence is fabricated → must be blanked, no judge call.
    g.reset()
    g.record_tool_result({"data": [{"corpusId": 42, "abstract": "real retrieved text here"}]})
    out = _output([("42", "totally invented claim never retrieved")])
    judgements = asyncio.run(get_llm_relevance(out, metric))

    assert judgements["42"] == rel.Relevance.NOT_RELEVANT.value
    assert judge_calls["n"] == 0, "ungrounded evidence must not reach the judge"
    assert any(b[0] == "42" and b[3] == "full" for b in g.last_blanked())


def test_e2e_partial_blanking_judges_scrubbed_evidence(monkeypatch):
    from astabench.evals.paper_finder import eval as pe
    from astabench.evals.paper_finder import relevance as rel

    seen_markdown = {}

    async def _fake_judge(docs, criteria):
        for d in docs:
            seen_markdown[d.corpus_id] = d.markdown
        return {d.corpus_id: rel.Relevance.PERFECT.value for d in docs}

    monkeypatch.setattr(rel, "load_relevance_judgement", _fake_judge)
    monkeypatch.setattr(pe, "get_normalizer_references", lambda: ({}, {}))

    get_llm_relevance = _install_and_get_patched()
    metric = SimpleNamespace(known_to_be_good=set(), relevance_criteria=[])

    g.reset()
    g.record_tool_result({"data": [{"corpusId": 42, "abstract": "a grounded phrase"}]})
    # one grounded passage + one fabricated, joined by ' ... '
    out = _output([("42", "a grounded phrase ... a fabricated one")])
    judgements = asyncio.run(get_llm_relevance(out, metric))

    # paper is JUDGED (not zeroed), and the judge saw ONLY the grounded passage
    assert judgements["42"] == rel.Relevance.PERFECT.value
    assert seen_markdown["42"] == "a grounded phrase"
    # recorded as a partial drop, listing the fabricated passage
    partial = [b for b in g.last_blanked() if b[0] == "42"]
    assert partial and partial[0][3] == "partial"
    assert partial[0][1] == ["a fabricated one"]


def test_e2e_grounded_judged_and_cached(monkeypatch):
    from astabench.evals.paper_finder import eval as pe
    from astabench.evals.paper_finder import relevance as rel
    from astabench.evals.paper_finder import paper_finder_utils as pfu

    writes = {}

    async def _fake_judge(docs, criteria):
        return {d.corpus_id: rel.Relevance.PERFECT.value for d in docs}

    async def _fake_update(qid, judgements):
        writes.setdefault(qid, {}).update(judgements)

    cache = {}
    monkeypatch.setattr(rel, "load_relevance_judgement", _fake_judge)
    monkeypatch.setattr(pfu, "update_references", _fake_update)
    monkeypatch.setattr(pe, "get_normalizer_references", lambda: ({}, cache))

    get_llm_relevance = _install_and_get_patched()
    metric = SimpleNamespace(known_to_be_good=set(), relevance_criteria=[])

    g.reset()
    g.record_tool_result({"data": [{"corpusId": 42, "abstract": "real retrieved text here"}]})
    out = _output([("42", "real retrieved text here")])
    judgements = asyncio.run(get_llm_relevance(out, metric))

    assert judgements["42"] == rel.Relevance.PERFECT.value
    # cached under a composite (paper, evidence-hash) key, not the bare paper id
    written = writes["semantic_9"]
    assert list(written.keys()) == [g.cache_key("42", "real retrieved text here")]
    assert "42" not in written


def test_e2e_known_good_is_perfect_regardless_of_evidence(monkeypatch):
    from astabench.evals.paper_finder import eval as pe
    from astabench.evals.paper_finder import relevance as rel

    async def _fake_judge(docs, criteria):
        return {}

    monkeypatch.setattr(rel, "load_relevance_judgement", _fake_judge)
    monkeypatch.setattr(pe, "get_normalizer_references", lambda: ({}, {}))

    get_llm_relevance = _install_and_get_patched()
    metric = SimpleNamespace(known_to_be_good={"7"}, relevance_criteria=[])

    g.reset()  # nothing retrieved; gold paper with junk evidence still Perfect
    out = _output([("7", "junk evidence for a gold paper")])
    judgements = asyncio.run(get_llm_relevance(out, metric))
    assert judgements["7"] == rel.Relevance.PERFECT.value


# --- the real tool-wrapping seam -------------------------------------------

def test_wrap_tools_records_and_preserves_interface():
    """_wrap_tools_for_provenance must round-trip a real inspect tool (name +
    params intact), return the payload unchanged, and record it for grounding."""
    import evaluator
    from inspect_ai.tool import tool, ToolDef

    @tool
    def fake_search():
        async def execute(query: str):
            """Search papers.

            Args:
                query: the search string
            """
            return {"data": [{"corpusId": 314159, "title": "Prov Test",
                              "abstract": "grounded via the real wrapper"}]}
        return execute

    wrapped = evaluator._wrap_tools_for_provenance([fake_search()])
    td = ToolDef(wrapped[0])
    assert td.name == "fake_search"
    assert "query" in td.parameters.properties

    g.reset()
    res = asyncio.run(wrapped[0]("anything"))
    assert res["data"][0]["corpusId"] == 314159  # unchanged
    assert g.check_evidence("314159", "grounded via the real wrapper")[0]
