"""End-to-end smoke test of all three routes with fake tools + fake models."""
import asyncio
import json
import sys
import types

# ---- stubs (same as test_helpers)
mr = types.ModuleType("model_registry")


class _FakeModel:
    def __init__(self):
        self.replies = []  # queue; falls back to last

    async def generate(self, prompt, config=None):
        r = self.replies.pop(0) if len(self.replies) > 1 else (self.replies[0] if self.replies else "")
        return types.SimpleNamespace(completion=r if isinstance(r, str) else r(prompt))


mr.GPT_5_4 = _FakeModel()
mr.GPT_5_4_MINI = _FakeModel()
sys.modules["model_registry"] = mr

inspect_solver = types.ModuleType("inspect_ai.solver")


class TaskState:
    pass


inspect_solver.TaskState = TaskState
inspect_solver.Generate = object
inspect_solver.solver = lambda fn: fn
inspect_tool = types.ModuleType("inspect_ai.tool")


class ToolDef:
    def __init__(self, t):
        self.name = getattr(t, "tool_name", "")


inspect_tool.ToolDef = ToolDef
sys.modules["inspect_ai"] = types.ModuleType("inspect_ai")
sys.modules["inspect_ai.solver"] = inspect_solver
sys.modules["inspect_ai.tool"] = inspect_tool

import agent  # noqa: E402


class Item:
    def __init__(self, obj):
        self.text = json.dumps(obj)


class Tool:
    def __init__(self, name, fn):
        self.tool_name = name
        self.fn = fn

    def __call__(self, **kw):
        async def run():
            return self.fn(**kw)

        return run()


def paper(cid, title, year=2023, ncites=100, nauth=4, venue="NeurIPS"):
    return {
        "corpusId": cid,
        "paperId": f"hash{cid}",
        "title": title,
        "abstract": f"Abstract of {title}. This paper explicitly discusses clustering attention in transformers for efficiency.",
        "tldr": {"text": f"TLDR {cid}"},
        "year": year,
        "venue": venue,
        "journal": {"name": "Some Journal"},
        "authors": [{"name": f"A{j} Author{cid}"} for j in range(nauth)],
        "citationCount": ncites,
    }


PAPERS = [paper(i, f"Paper number {i} about clustering attention") for i in range(1, 60)]


def fake_rel(**kw):
    return [Item(p) for p in PAPERS[: min(int(kw.get("limit", 20)), len(PAPERS))]]


def fake_snip(**kw):
    ids = kw.get("paper_ids")
    if ids:
        cid = ids.split(":")[-1]
        return [Item({"data": [{"score": 1.0, "paper": {"corpusId": cid, "title": "t"},
                                "snippet": {"text": f"targeted body passage for {cid}"}}]})]
    return [Item({"data": [
        {"score": 1.0, "paper": {"corpusId": str(100 + i), "title": f"Snip paper {i}"},
         "snippet": {"text": f"snippet text {i}"}} for i in range(int(kw.get("limit", 10)))
    ]})]


def fake_batch(**kw):
    out = []
    for pid in kw.get("ids", []):
        cid = str(pid).split(":")[-1]
        d = {"corpusId": cid, "title": f"Fetched {cid}", "abstract": f"Fetched abstract {cid}",
             "year": 2022, "authors": [{"name": "X Y"}] * 4, "citationCount": 77,
             "venue": "NeurIPS", "journal": {"name": "J"}}
        if "references" in (kw.get("fields") or ""):
            # odd cids cite both the RoBERTa target and a paper by the cited author
            d["references"] = (
                [{"paperId": "hashTARGET", "corpusId": 9999}, {"paperId": "hash3001"}]
                if int(cid) % 2 == 1
                else [{"paperId": "other"}]
            )
        out.append(Item(d))
    return out


def fake_title_search(**kw):
    t = kw.get("title", "")
    if "RoBERTa" in t or "DistilBERT" in t:
        return [Item({"data": [{"corpusId": 9999, "paperId": "hashTARGET", "title": t, "matchScore": 100}]})]
    return [Item({"corpusId": 500, "paperId": "hash500", "title": t, "matchScore": 90,
                  "year": 2012, "authors": [{"name": "Alex K"}], "abstract": "the paper itself"})]


def fake_citations(**kw):
    return [Item({"citingPaper": paper(2000 + i, f"Citing paper {i}", year=2022 + i % 2, ncites=40, nauth=5)})
            for i in range(30)]


def fake_authors(**kw):
    return [Item({"authorId": "77", "name": kw.get("name", ""), "paperCount": 300})]


def fake_author_papers(**kw):
    return [Item(paper(3000 + i, f"Authored paper {i}", year=2021, ncites=15, venue="Sci J")) for i in range(20)]


def make_state(sid, stype, query):
    s = TaskState()
    s.sample_id = sid
    s.metadata = {"raw_query": query, "score_type": stype}
    s.tools = [
        Tool("search_papers_by_relevance", fake_rel),
        Tool("search_paper_by_title", fake_title_search),
        Tool("snippet_search", fake_snip),
        Tool("get_paper", lambda **kw: []),
        Tool("get_paper_batch", fake_batch),
        Tool("get_citations", fake_citations),
        Tool("search_authors_by_name", fake_authors),
        Tool("get_author_papers", fake_author_papers),
    ]
    s.output = types.SimpleNamespace(completion=None)
    return s


async def main():
    solve = agent.make_solver()

    # ---------- semantic ----------
    mr.GPT_5_4.replies = [
        json.dumps({
            "criteria": [
                {"name": "c1", "description": "must discuss clustering attention", "weight": 0.6},
                {"name": "c2", "description": "must show efficiency gains", "weight": 0.4},
            ],
            "keyword_queries": [f"kw{i}" for i in range(10)],
            "snippet_queries": ["Papers should show clustering attention.", "s2", "s3"],
            "year_min": None, "year_max": None,
        }),
        json.dumps({"keyword_queries": ["fresh1", "fresh2"]}),  # gap-fill
    ]
    # triage: everything gets mixed grades so gap-fill triggers, then enrichment
    mr.GPT_5_4_MINI.replies = ["\n".join(f"{i}: 3 1" for i in range(0, 500))]
    st = make_state("sem1", "semantic_f1", "clustering attention transformers?")
    st = await solve(st, None)
    out = json.loads(st.output.completion)
    rs = out["output"]["results"]
    assert out["output"]["query_id"] == "sem1"
    assert len(rs) > 50, len(rs)
    assert all(isinstance(r["paper_id"], str) and "markdown_evidence" in r for r in rs)
    assert any("targeted body passage" in r["markdown_evidence"] or "Abstract" in r["markdown_evidence"] for r in rs)
    print(f"SEMANTIC OK: {len(rs)} results\n")

    # ---------- specific ----------
    mr.GPT_5_4.replies = [
        json.dumps({"canonical_name": "MS2", "candidate_titles": ["MS2: Multi-Document Summarization of Medical Studies"],
                    "author_hints": ["DeYoung"], "year_hint": 2021, "confidence": 0.6}),
        json.dumps({"indices": [0, 1], "confidence": 0.5, "alternates": [2]}),
    ]
    st = make_state("spec1", "specific_f1", "the MS² DeYong2021 paper")
    st = await solve(st, None)
    out = json.loads(st.output.completion)
    rs = out["output"]["results"]
    assert 1 <= len(rs) <= 5, len(rs)
    assert all(r["markdown_evidence"] == "" for r in rs)
    print(f"SPECIFIC OK: {len(rs)} results {[r['paper_id'] for r in rs]}\n")

    # ---------- metadata: cites-paper with ref verification ----------
    mr.GPT_5_4.replies = [json.dumps({
        "authors": [], "venues": ["NeurIPS", "Neural Information Processing Systems"],
        "venue_constraint": None, "years_allowed": [2022, 2023], "year_min": None, "year_max": None,
        "cites_paper_title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "cites_author": None, "exclude_coauthor": None,
        "min_citations": 30, "min_authors": 4, "max_authors": None, "topic_keywords": None})]
    mr.GPT_5_4_MINI.replies = [lambda p: ",".join(str(i) for i in range(120))]  # venue LLM approves all
    st = make_state("meta1", "metadata_f1", 'NeurIPS papers 2022-2023 that cite the "RoBERTa" paper')
    st = await solve(st, None)
    out = json.loads(st.output.completion)
    rs = out["output"]["results"]
    assert len(rs) > 0, "metadata must never submit 0"
    print(f"METADATA cites-paper OK: {len(rs)} results\n")

    # ---------- metadata: author base + cites_author + exclude self-cites ----------
    mr.GPT_5_4.replies = [json.dumps({
        "authors": ["David Harel"], "venues": [], "venue_constraint": "journal articles (not conference proceedings)",
        "years_allowed": [], "year_min": None, "year_max": None,
        "cites_paper_title": None, "cites_author": "Gera Weiss", "exclude_coauthor": "Gera Weiss",
        "min_citations": 10, "min_authors": None, "max_authors": None,
        "topic_keywords": None})]
    mr.GPT_5_4_MINI.replies = [lambda p: ",".join(str(i) for i in range(120))]
    st = make_state("meta2", "metadata_f1", "Journal articles by David Harel citing papers by Gera Weiss")
    st = await solve(st, None)
    out = json.loads(st.output.completion)
    rs = out["output"]["results"]
    assert len(rs) > 0
    # ref-verification: fake batch gives odd cids refs containing hashTARGET
    assert all(int(r["paper_id"]) % 2 == 1 for r in rs), [r["paper_id"] for r in rs]
    print(f"METADATA cites-author OK: {len(rs)} results\n")

    # ---------- crash fallback ----------
    mr.GPT_5_4.replies = [""]
    st = make_state("sem2", "semantic_f1", "anything")
    st.metadata = {"raw_query": "anything", "score_type": "semantic_f1"}
    st.tools = [Tool("search_papers_by_relevance", fake_rel)]  # missing tools -> route raises
    st = await solve(st, None)
    out = json.loads(st.output.completion)
    assert len(out["output"]["results"]) > 0
    print("FALLBACK OK\n")

    print("E2E ALL PASSED")


asyncio.run(main())
