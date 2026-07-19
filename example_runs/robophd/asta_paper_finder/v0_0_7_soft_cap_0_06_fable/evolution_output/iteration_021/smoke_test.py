"""Offline smoke test for iter21_reverse_cite.

Stubs out inspect_ai / model_registry so agent.py imports without the
benchmark harness, asserts the semantic stack is byte-level unchanged from
iter20/iter18, then exercises the new machinery end-to-end:

  1. metadata_31-style: author base + cites_author, complete reverse-citer
     set -> membership gates hard, non-citers pruned exactly.
  2. metadata_42-style: single cite target with a capped (1000) citer list
     and a dead references API -> tier-2 fail-open keeps the filtered
     mention-channel set instead of discarding it.
  3. specific-path ambiguous alias: planner-title records enter the
     submission, alias hedge fills by citation count, cap 7.

Run: /opt/anaconda3/envs/robophd_demo/bin/python smoke_test.py
"""

import asyncio
import json
import sys
import types

# ---- stub the harness modules agent.py imports at module scope
inspect_ai = types.ModuleType("inspect_ai")
solver_mod = types.ModuleType("inspect_ai.solver")
tool_mod = types.ModuleType("inspect_ai.tool")
model_mod = types.ModuleType("inspect_ai.model")


class TaskState:
    pass


class ToolDef:
    def __init__(self, t):
        self.name = getattr(t, "__name__", "tool")


solver_mod.Generate = object
solver_mod.TaskState = TaskState
solver_mod.solver = lambda fn: fn
tool_mod.ToolDef = ToolDef
model_mod.GenerateConfig = object
sys.modules["inspect_ai"] = inspect_ai
sys.modules["inspect_ai.solver"] = solver_mod
sys.modules["inspect_ai.tool"] = tool_mod
sys.modules["inspect_ai.model"] = model_mod

reg = types.ModuleType("model_registry")
for h in ("GPT_5_4", "GPT_5_4_MINI", "CLAUDE_SONNET_4_6"):
    setattr(reg, h, object())
sys.modules["model_registry"] = reg

import agent as A  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAIL.append(name)


class Item:
    def __init__(self, obj):
        self.text = json.dumps(obj)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def paper(cid, title="p", venue="J", year=2020, ncit=20, nauth=1, abstract="a"):
    return {
        "corpusId": cid,
        "title": title,
        "venue": venue,
        "year": year,
        "citationCount": ncit,
        "authors": [{"name": f"A{i}"} for i in range(nauth)],
        "abstract": abstract,
    }


def make_state(sample_id, tools):
    st = TaskState()
    st.sample_id = sample_id
    st.metadata = {}
    st.output = types.SimpleNamespace(completion="")
    st.tools = tools
    return st


def submitted_ids(st):
    return [r["paper_id"] for r in json.loads(st.output.completion)["output"]["results"]]


def named(name):
    def deco(fn):
        fn.__name__ = name
        return fn
    return deco


# ---- canned LLM responses, keyed by prompt shape
GEN_RESPONSES = {}


async def fake_gen(model, prompt, retries=1, label="other"):
    for key, resp in GEN_RESPONSES.items():
        if key in prompt:
            return resp
    return "{}"


A._gen = fake_gen
A._venue_llm_filter_orig = A._venue_llm_filter


async def allow_all_venues(constraint, venue_strs):
    return set(venue_strs)


A._venue_llm_filter = allow_all_venues


print("semantic stack: iter20/iter18 values, untouched")
check("pool geometry", A.POOL_CAP == 320 and A.POOL_MERGE_HEAD == 240)
check("snippet limit", A.SNIP_INIT_LIMIT == 100)
check("triage depths", A.SIM_DEPTH == 55 and A.RESCUE_MAX == 22 and A.VERIFY_TOP == 26)
check("head geometry", A.HEAD == 100 and A.FULL_COVER_DEPTH == 36 and A.PER_CRIT_DEPTH == 70)
check("tail sweep kept", A.TAIL_SWEEP_END == 250 and A.SNIPPET_TIMEOUT == 90)


# ================================================================ test 1
print("\nmetadata_31-style: reverse-citer membership gates hard when complete")

GEN_RESPONSES.clear()
GEN_RESPONSES["Parse this scholarly paper search request"] = json.dumps({
    "authors": ["David Harel"], "venues": [],
    "venue_constraint": "journal articles (not conference proceedings)",
    "years_allowed": [], "year_min": None, "year_max": None,
    "cites_paper_titles": [], "cites_author": "Gera Weiss", "cites_venue": None,
    "exclude_coauthor": "Gera Weiss", "min_citations": 10,
    "min_authors": None, "max_authors": None, "topic_keywords": None,
})

harel_papers = [paper(str(100 + i), f"harel {i}", venue="J Comput", ncit=20) for i in range(10)]
weiss_cids = ["901", "902"]
# harel papers 100-103 cite Weiss (they appear in his papers' citer lists)
citers_by_target = {
    "901": [{"citingPaper": {"corpusId": c, "title": "x"}} for c in ("100", "101", "300")],
    "902": [{"citingPaper": {"corpusId": c, "title": "x"}} for c in ("102", "103")],
}


@named("search_authors_by_name")
async def t1_find_auth(name, fields="", limit=20):
    if "Harel" in name:
        return [Item({"authorId": "H1", "name": name, "paperCount": 300})]
    return [Item({"authorId": "W1", "name": name, "paperCount": 90})]


@named("get_author_papers")
async def t1_author_papers(author_id, paper_fields="", limit=100):
    if author_id == "H1":
        return [Item(p) for p in harel_papers]
    return [Item({"corpusId": c, "title": f"weiss {c}", "paperId": f"hash{c}"}) for c in weiss_cids]


@named("get_citations")
async def t1_citations(paper_id, fields="", limit=1000):
    cid = paper_id.split(":")[1]
    return [Item(x) for x in citers_by_target.get(cid, [])]


@named("get_paper_batch")
async def t1_batch(ids, fields=""):
    raise RuntimeError("ToolError: 'NoneType' object is not iterable")


@named("search_papers_by_relevance")
async def t1_rel(keyword, fields="", limit=20, venues=""):
    return []


@named("snippet_search")
async def t1_snip(query, limit=10, venues="", paper_ids=""):
    return [Item({"data": []})]


@named("search_paper_by_title")
async def t1_title(title, fields="", venues=""):
    return [Item({"data": []})]


st = make_state("t1", [t1_find_auth, t1_author_papers, t1_citations, t1_batch, t1_rel, t1_snip, t1_title])
st = run(A._solve_metadata(st, "Journal articles by David Harel with at least 10 citations, citing papers by Gera Weiss"))
got = submitted_ids(st)
check("exactly the 4 Weiss-citing Harel papers", sorted(got) == ["100", "101", "102", "103"], str(got))


# ================================================================ test 2
print("\nmetadata_42-style: cap-hit + dead refs -> tier-2 fail-open keeps the filtered set")

GEN_RESPONSES.clear()
GEN_RESPONSES["Parse this scholarly paper search request"] = json.dumps({
    "authors": [], "venues": ["NeurIPS"], "venue_constraint": "published at NeurIPS",
    "years_allowed": [2022, 2023], "year_min": None, "year_max": None,
    "cites_paper_titles": ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"],
    "cites_author": None, "cites_venue": None, "exclude_coauthor": None,
    "min_citations": 30, "min_authors": 4, "max_authors": None, "topic_keywords": None,
})

# 1000 recency-window citers (2024-25): cap hit, all fail the year filter
capped_citers = [
    {"citingPaper": paper(str(50000 + i), f"new citer {i}", venue="arXiv", year=2024, ncit=5, nauth=4)}
    for i in range(1000)
]
# 30 mention-channel candidates passing every observable filter; half never
# name the target in title/abstract (the common case: cited only in the body),
# so they can survive ONLY through the tier-2 fail-open
mention_docs = [
    paper(
        str(70000 + i),
        f"roberta downstream {i}" if i < 15 else f"downstream task study {i}",
        venue="NeurIPS", year=2022, ncit=60, nauth=5,
    )
    for i in range(30)
]


@named("search_paper_by_title")
async def t2_title(title, fields="", venues=""):
    return [Item({"corpusId": 555, "paperId": "hash555", "title": title})]


@named("get_citations")
async def t2_citations(paper_id, fields="", limit=1000):
    return [Item(x) for x in capped_citers]


@named("search_papers_by_relevance")
async def t2_rel(keyword, fields="", limit=100, venues=""):
    return [Item(d) for d in mention_docs]


@named("snippet_search")
async def t2_snip(query, limit=10, venues="", paper_ids=""):
    return [Item({"data": []})]


@named("get_paper_batch")
async def t2_batch(ids, fields=""):
    if "references" in (fields or ""):
        raise RuntimeError("ToolError: 'NoneType' object is not iterable")
    return [Item({"corpusId": i.split(":")[1], "title": "t"}) for i in ids]


@named("search_authors_by_name")
async def t2_find_auth(name, fields="", limit=20):
    return []


@named("get_author_papers")
async def t2_author_papers(author_id, paper_fields="", limit=100):
    return []


st = make_state("t2", [t2_title, t2_citations, t2_rel, t2_snip, t2_batch, t2_find_auth, t2_author_papers])
st = run(A._solve_metadata(st, 'NeurIPS papers 2022-2023 that cite the "RoBERTa" paper cited by at least 30 other papers written by more than 3 authors'))
got = submitted_ids(st)
check("all 30 filtered mention candidates kept", len(got) == 30 and set(got) == {str(70000 + i) for i in range(30)},
      f"n={len(got)}")


# ================================================================ test 3
print("\nspecific ambiguous: planner-title records + citation-ranked hedge, cap 7")

GEN_RESPONSES.clear()
GEN_RESPONSES["A user refers to one specific published paper"] = json.dumps({
    "canonical_name": "SPIKE",
    "candidate_titles": ["SPIKE: interpretation A", "SPIKE: interpretation B"],
    "author_hints": [], "year_hint": None, "confidence": 0.2,
})
GEN_RESPONSES["Which candidate IS that exact paper"] = json.dumps(
    {"indices": [0], "confidence": 0.4, "alternates": []}
)

title_hits = {
    "SPIKE: interpretation A": paper("11", "SPIKE: interpretation A", ncit=100),
    "SPIKE: interpretation B": paper("12", "SPIKE: interpretation B", ncit=90),
}
alias_docs = [
    paper("21", "SPIKE: obscure homonym", ncit=5),
    paper("22", "SPIKE: the classic", ncit=5000),
    paper("23", "SPIKE: mid homonym", ncit=300),
    paper("31", "unrelated spiking networks", ncit=9000),
]


@named("search_paper_by_title")
async def t3_title(title, fields="", venues=""):
    d = title_hits.get(title)
    return [Item(d)] if d else [Item({"data": []})]


@named("search_papers_by_relevance")
async def t3_rel(keyword, fields="", limit=20, venues=""):
    return [Item(d) for d in alias_docs]


@named("snippet_search")
async def t3_snip(query, limit=10, venues="", paper_ids=""):
    return [Item({"data": []})]


st = make_state("t3", [t3_title, t3_rel, t3_snip])
st = run(A._solve_specific(st, "the SPIKE paper"))
got = submitted_ids(st)
check("primary first", got[0] == "11", str(got))
check("planner-title record for interpretation B submitted", "12" in got, str(got))
check("citation-ranked hedge: classic (22) in, before mid (23)",
      "22" in got and (("23" not in got) or got.index("22") < got.index("23")), str(got))
check("non-alias-titled homonym (31) excluded", "31" not in got, str(got))
check("cap 7", len(got) <= 7, str(got))


print()
if FAIL:
    print(f"SMOKE FAILED: {FAIL}")
    sys.exit(1)
print("SMOKE OK")
