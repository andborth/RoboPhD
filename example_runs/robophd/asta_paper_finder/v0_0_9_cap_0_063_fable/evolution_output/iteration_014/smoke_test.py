"""Stub harness: fake state.tools + fake model_registry, asserts the schema
and PRINTS the built evidence so content bugs are visible, not just length."""
import asyncio, json, sys, types, random

fake_reg = types.ModuleType("model_registry")
class _Resp:
    def __init__(s, c): s.completion = c
class _M:
    def __init__(s, n): s.n = n
    async def generate(s, prompt, config=None):
        if "planning a literature search" in prompt:
            return _Resp(json.dumps({
                "keyword_queries": [f"kw {i}" for i in range(8)],
                "snippet_queries": ["sentence a", "sentence b", "sentence c"],
                "criteria": [{"name": "A", "description": "The paper must do A.", "weight": 0.4},
                             {"name": "B", "description": "The paper must do B.", "weight": 0.3},
                             {"name": "C", "description": "The paper must do C.", "weight": 0.3}],
                "exclusions": [], "oldest_first": False}))
        if "Numbered relevance criteria" in prompt:
            idxs = [int(l.split("]")[0][1:]) for l in prompt.splitlines() if l.startswith("[")]
            if "starved needle topic" in prompt:   # force the thin-pool gate open
                return _Resp("\n".join(f"{i}: 0,0,0" for i in idxs))
            return _Resp("\n".join(f"{i}: {random.choice([3,1,0])},{random.choice([3,1,0])},3" for i in idxs))
        if "found very few relevant papers" in prompt:
            return _Resp(json.dumps({
                "queries": ["alternate wording one", "alternate wording two"],
                "titles": ["Guessed Exact Paper Title One", "Guessed Exact Paper Title Two"]}))
        if "rate EACH criterion" in prompt:
            idxs = [int(l.split("]")[0][1:]) for l in prompt.splitlines() if l.startswith("[")]
            return _Resp("\n".join(f"{i}: 3,3,1" for i in idxs))
        if "Parse this literature-search request" in prompt and "RoBERTa" in prompt:
            return _Resp(json.dumps({
                "authors": [], "venues": ["NeurIPS"], "years": [2022, 2023],
                "year_min": None, "year_max": None, "min_citations": 30,
                "max_citations": None, "min_authors": 4, "max_authors": None,
                "cited_papers": ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"],
                "cites_venues": [], "topic": None, "keyword_query": "language model pretraining"}))
        if "looking for a specific known paper" in prompt:
            return _Resp(json.dumps({
                "name": "SPIKE", "title_guesses": ["SPIKE syntactic search"],
                "keyword_queries": ["SPIKE extractive search system"],
                "author_surnames": ["Shlain", "Schlain"], "year": 2020}))
        if "Parse this literature-search request" in prompt and "NAACL 2010" in prompt:
            return _Resp(json.dumps({
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                "authors_mode": "any", "venues": ["NAACL"], "years": [2010, 2012],
                "year_min": None, "year_max": None, "min_citations": None,
                "max_citations": None, "min_authors": 2, "max_authors": None,
                "cited_papers": [], "cites_venues": [], "topic": None,
                "keyword_query": "language model"}))
        if "Parse this literature-search request" in prompt and "SPLASH" in prompt:
            return _Resp(json.dumps({
                "authors": [], "venues": ["SPLASH"], "years": [],
                "year_min": 2019, "year_max": None, "min_citations": None,
                "max_citations": None, "cited_papers": [], "cites_venues": ["NeurIPS"],
                "topic": None, "keyword_query": "graph neural network"}))
        return _Resp("{}")
for h in ["GPT_5_4","GPT_5_4_MINI","GPT_5_5","CLAUDE_HAIKU_4_5","CLAUDE_SONNET_4_6",
          "CLAUDE_OPUS_4_8","GEMINI_3_1_FLASH_LITE","GEMINI_3_5_FLASH","GEMINI_3_1_PRO_PREVIEW"]:
    setattr(fake_reg, h, _M(h))
sys.modules["model_registry"] = fake_reg

import agent as AG

class CT:
    def __init__(s, t): s.text = t
ABS = ("We present a system that does A on long inputs. " * 14)
BODY = ("This section shows the method does A explicitly and reports B results. " * 8)
def mk(i):
    return {"paperId": f"h{i}", "corpusId": 1000 + i, "title": f"Paper {i} about A and B",
            "abstract": ABS if i % 4 else "", "tldr": {"text": f"Tldr for paper {i}."},
            "year": 2020 + i % 5, "venue": "NeurIPS"}
async def rel(keyword=None, fields=None, limit=20, venues=None):
    return [CT(json.dumps(mk(i))) for i in range(limit)]
async def snip(query=None, limit=10, paper_ids=None, venues=None):
    ids = [int(x.split(":")[1]) - 1000 for x in (paper_ids or "").split(",") if ":" in x] or list(range(limit))
    data = [{"score": 1.0 / (1 + j), "paper": {"corpusId": str(1000 + ids[j % len(ids)]), "title": "t"},
             "snippet": {"text": BODY, "section": "body"}} for j in range(limit)]
    return [CT(json.dumps({"data": data}))]
async def batch(ids=None, fields=None):
    out = []
    for x in (ids or []):
        i = int(str(x).split(":")[-1].lstrip("h")) - (1000 if "CorpusId" in str(x) else 0)
        r = mk(max(0, i)); r["abstract"] = ABS
        if "citations" in (fields or ""): r["citations"] = [{"paperId": f"h{k}"} for k in range(300, 340)]
        if "references" in (fields or ""): r["references"] = [{"paperId": f"h{k}"} for k in range(340, 380)]
        out.append(CT(json.dumps(r)))
    return out
_TITLE_N = [0]
async def by_title(title=None, fields=None, venues=None):
    _TITLE_N[0] += 1
    return [CT(json.dumps(mk(7000 + _TITLE_N[0])))]   # novel id per call
async def get_paper(paper_id=None, fields=None): return [CT(json.dumps(mk(2)))]
async def get_cit(paper_id=None, fields=None, limit=100): return [CT(json.dumps({"citingPaper": mk(i)})) for i in range(20)]
_AUTH_IDS = {}
async def auth(name=None, fields=None, limit=10):
    aid = _AUTH_IDS.setdefault(name, str(len(_AUTH_IDS) + 1))
    return [CT(json.dumps({"authorId": aid, "name": name, "paperCount": 99}))]
async def auth_p(author_id=None, paper_fields=None, limit=100):
    # disjoint per-author sets plus 4 shared papers: distinguishes union
    # ("any" mode) from intersection ("all" mode) in the author route
    base = int(author_id) * 40
    recs = [mk(base + i) for i in range(30)] + [mk(9000 + j) for j in range(4)]
    return [CT(json.dumps(r)) for r in recs]
NAMES = {"search_papers_by_relevance": rel, "snippet_search": snip, "get_paper_batch": batch,
         "search_paper_by_title": by_title, "get_paper": get_paper, "get_citations": get_cit,
         "search_authors_by_name": auth, "get_author_papers": auth_p}
class TD:
    def __init__(s, t): s.name = getattr(t, "_nm", "")
AG.ToolDef = TD
class ST:
    def __init__(s, md):
        s.tools = []
        for n, f in NAMES.items():
            f._nm = n; s.tools.append(f)
        s.metadata = md; s.sample_id = "t1"; s.input = md["raw_query"]
        s.output = types.SimpleNamespace(completion="")

async def main():
    for st_type, q in [("semantic_f1", "long video description research"),
                       ("semantic_f1", "starved needle topic research"),
                       ("specific_f1", "SPIKE syntactic search paper"),
                       ("metadata_f1", "NeurIPS papers 2022-2023 citing RoBERTa with 30+ citations"),
                       ("metadata_f1", "A SPLASH 2019 and beyond paper that cites any NeurIPS"),
                       ("metadata_f1", 'NAACL 2010 or 2012 papers co-authored by one of the authors of the "BERT" paper')]:
        state = ST({"score_type": st_type, "raw_query": q})
        out = await AG.make_solver()(state, None)
        d = json.loads(out.output.completion)
        rs = d["output"]["results"]
        assert set(d) == {"output"} and {"query_id", "results"} <= set(d["output"])
        assert all(isinstance(r["paper_id"], str) and isinstance(r["markdown_evidence"], str) for r in rs)
        assert all(len(r["markdown_evidence"]) <= 2500 for r in rs)
        L = sorted(len(r["markdown_evidence"]) for r in rs)
        print(f"OK {st_type}: n={len(rs)} evlen med={L[len(L)//2] if L else 0} max={L[-1] if L else 0} "
              f">=2200: {sum(1 for x in L if x >= 2200)}")
        if "BERT" in q:
            assert len(rs) > 20, f"any-mode union expected >20 papers, got {len(rs)} (intersection bug?)"
            print(f"   any-author union OK: {len(rs)} papers submitted")
        if st_type == "semantic_f1":
            print("--- first evidence, passage by passage ---")
            for pz in rs[0]["markdown_evidence"].split(" ... "):
                print(f"   [{len(pz):4d}] {pz[:110]!r}")
asyncio.run(main())
print("ALL OK")
