"""Offline smoke test for iter20_cite_proof.

Stubs out inspect_ai / model_registry so agent.py imports without the
benchmark harness, asserts the semantic stack is byte-level unchanged from
iter18 (constants + absent machinery), then exercises the three fixes:
_batch_bisect poison-id salvage, body-mention verification rescuing a
metadata_42-style candidate set, and conjunction augmentation adding
both-target-verified extras on a metadata_26-style cap-hit query.

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


class TaskState:  # noqa: D101
    pass


class ToolDef:  # noqa: D101
    def __init__(self, t):
        self.name = getattr(t, "__name__", "tool")


def solver(fn):  # noqa: D103
    return fn


solver_mod.Generate = object
solver_mod.TaskState = TaskState
solver_mod.solver = solver
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

SRC = open("agent.py").read()
import ast as _ast  # noqa: E402

_doc = _ast.get_docstring(_ast.parse(SRC)) or ""
CODE = "\n".join(SRC.splitlines()[len(_doc.splitlines()) + 2 :])
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


print("semantic stack: iter18 values, untouched")
check("pool geometry", A.POOL_CAP == 320 and A.POOL_MERGE_HEAD == 240)
check("snippet limit (iter18, not the iter19 revert)", A.SNIP_INIT_LIMIT == 100)
check("triage depths", A.SIM_DEPTH == 55 and A.RESCUE_MAX == 22 and A.VERIFY_TOP == 26)
check("head geometry", A.HEAD == 100 and A.FULL_COVER_DEPTH == 36 and A.PER_CRIT_DEPTH == 70)
check("tail sweep kept", A.TAIL_SWEEP_END == 250 and A.SNIPPET_TIMEOUT == 90)
check("no gap-fill machinery (iter18 removed it)", "t1gap" not in CODE and "GAP_MIN_PERFECT" not in CODE)
check("14-query planner kept", "up to 14" in CODE or "14 query" in CODE or CODE.count("keyword_queries") > 0)

print("\n_batch_bisect: poison-id salvage")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


calls = []


async def batch_poison(ids, fields):
    calls.append(list(ids))
    if "CorpusId:666" in ids:
        raise RuntimeError("Paper xyz is newer than the date cutoff")
    return [Item({"corpusId": i.split(":")[1], "title": f"t{i}"}) for i in ids]


batch_poison.__name__ = "get_paper_batch"
ids = [f"CorpusId:{n}" for n in [1, 2, 3, 666, 5, 6, 7, 8]]
got = run(A._batch_bisect(batch_poison, ids, "corpusId,title", "test", chunk=8, attempts=1))
got_cids = {d["corpusId"] for d in got}
check("salvages all non-poison ids", got_cids == {"1", "2", "3", "5", "6", "7", "8"}, str(got_cids))
check("poison id alone is lost", "666" not in got_cids)

print("\n_phrase_in / _conj_names")
check("word-bounded hit", A._phrase_in("t5", "We fine-tune T5-base on Spider."))
check("no substring false positive", not A._phrase_in("t5", "We use mt51 embeddings."))
check("multiword phrase", A._phrase_in("neural module networks", "using Neural Module Networks (NMN)"))
targets = [
    {"cid": "204838007", "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"},
    {"cid": "52815560", "title": "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task"},
]
names = A._conj_names("paper citing the T5 paper and the spider paper", targets)
check("informal names assigned (T5 by elimination, spider by title match)", names == ["T5", "spider"], str(names))

print("\n_snip_mention_verify")


async def snip_scoped(query=None, paper_ids=None, limit=None, venues=None):
    out = []
    for pid in (paper_ids or "").split(","):
        cid = pid.split(":")[1]
        if cid == "101":
            out.append({"score": 1, "paper": {"corpusId": cid}, "snippet": {"text": "We build on RoBERTa embeddings."}})
        elif cid == "102":
            out.append({"score": 1, "paper": {"corpusId": cid}, "snippet": {"text": "unrelated passage about optics"}})
    return [Item({"data": out})]


snip_scoped.__name__ = "snippet_search"
ok = run(A._snip_mention_verify(snip_scoped, ["101", "102", "103"], "RoBERTa"))
check("mentioning paper verified", ok == {"101"}, str(ok))

print("\n_solve_metadata scenario A: refs false-negatives, body mentions rescue (metadata_42-style)")

# tool stubs -----------------------------------------------------------
CITERS_RECENT = [{"citingPaper": {"corpusId": str(900000 + i), "title": f"recent {i}", "year": 2025,
                                  "venue": "arXiv", "authors": [{"name": "A"}] * 5, "citationCount": 2}}
                 for i in range(1000)]
MENTION_CANDS = [{"corpusId": str(100 + i), "title": f"NeurIPS pretraining paper {i}", "year": 2022,
                  "venue": "Neural Information Processing Systems",
                  "authors": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}],
                  "citationCount": 100, "abstract": "We study large language models."}
                 for i in range(30)]


def make_tools(conj=False):
    async def search_paper_by_title(title=None, fields=None, venues=None):
        if "Spider" in (title or ""):
            return [Item({"data": [{"paperId": "hashS", "corpusId": 52815560, "title": targets[1]["title"]}]})]
        if "Text-to-Text" in (title or ""):
            return [Item({"data": [{"paperId": "hashT", "corpusId": 204838007, "title": targets[0]["title"]}]})]
        return [Item({"paperId": "hashR", "corpusId": 198953378, "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach"})]

    async def get_citations(paper_id=None, fields=None, limit=None):
        if conj:
            # disjoint recency windows per target -> tiny intersection (5),
            # both lists cap at 1000 (the metadata_26 shape)
            base = 800000 if "204838007" in (paper_id or "") else 800995
            return [Item({"citingPaper": {"corpusId": str(base + i), "title": f"c{i}", "year": 2025,
                                          "venue": "arXiv", "authors": [{"name": "A"}],
                                          "citationCount": 1}}) for i in range(1000)]
        return [Item(c) for c in CITERS_RECENT]

    async def search_papers_by_relevance(keyword=None, fields=None, limit=None, venues=None):
        if conj:
            return [Item({"corpusId": str(500 + i), "title": f"text-to-SQL study {i}", "year": 2024,
                          "venue": "arXiv", "authors": [{"name": "A"}], "citationCount": 3,
                          "abstract": "sql generation"}) for i in range(10)]
        return [Item(d) for d in MENTION_CANDS]

    async def snippet_search(query=None, paper_ids=None, limit=None, venues=None):
        out = []
        if paper_ids:  # scoped verification call
            for pid in paper_ids.split(","):
                cid = pid.split(":")[1]
                n = int(cid)
                if conj:
                    # 502/503 mention BOTH T5 and spider; 505 mentions only T5
                    if n in (502, 503):
                        out.append({"score": 1, "paper": {"corpusId": cid},
                                    "snippet": {"text": "We prompt T5 on the Spider benchmark." if "spider" in (query or "").lower() or "t5" in (query or "").lower() else ""}})
                    elif n == 505 and "t5" in (query or "").lower():
                        out.append({"score": 1, "paper": {"corpusId": cid},
                                    "snippet": {"text": "T5 is our backbone."}})
                else:
                    # every even-numbered candidate mentions RoBERTa in body
                    if n % 2 == 0:
                        out.append({"score": 1, "paper": {"corpusId": cid},
                                    "snippet": {"text": "Following RoBERTa, we pretrain with dynamic masking."}})
        return [Item({"data": out})]

    async def get_paper_batch(ids=None, fields=None):
        if "references" in (fields or ""):
            # refs return truncated, id-less reference lists -> false negatives
            return [Item({"corpusId": i.split(":")[1], "references": [{"title": "some other work"}]}) for i in ids]
        return [Item({"corpusId": i.split(":")[1], "title": "t", "year": 2022, "venue": "x",
                      "authors": [{"name": "A"}] * 4, "citationCount": 50}) for i in ids]

    async def get_paper(paper_id=None, fields=None):
        return [Item({"corpusId": paper_id.split(":")[1]})]

    async def search_authors_by_name(name=None, fields=None, limit=None):
        return [Item({"authorId": "1", "name": name, "paperCount": 10})]

    async def get_author_papers(author_id=None, paper_fields=None, limit=None):
        return [Item({"data": []})]

    tools = [search_paper_by_title, get_citations, search_papers_by_relevance, snippet_search,
             get_paper_batch, get_paper, search_authors_by_name, get_author_papers]
    for t in tools:
        t.__name__ = t.__name__
    return tools


class Out:  # noqa: D101
    completion = ""


def make_state(query, conj=False):
    st = TaskState()
    st.sample_id = "test"
    st.metadata = {"raw_query": query, "score_type": "metadata_f1"}
    st.tools = make_tools(conj)
    st.output = Out()
    return st


PLAN_A = {
    "authors": [], "venues": ["NeurIPS", "Neural Information Processing Systems"],
    "venue_constraint": None, "years_allowed": [2022, 2023], "year_min": None, "year_max": None,
    "cites_paper_titles": ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"],
    "cites_author": None, "cites_venue": None, "exclude_coauthor": None,
    "min_citations": 30, "min_authors": 4, "max_authors": None, "topic_keywords": None,
}
PLAN_B = {
    "authors": [], "venues": [], "venue_constraint": None, "years_allowed": [],
    "year_min": None, "year_max": None,
    "cites_paper_titles": [targets[0]["title"], targets[1]["title"]],
    "cites_author": None, "cites_venue": None, "exclude_coauthor": None,
    "min_citations": None, "min_authors": None, "max_authors": None, "topic_keywords": None,
}
_plan = {"v": PLAN_A}


async def fake_gen(model, prompt, retries=1, label="other"):
    if "Parse this scholarly paper search request" in prompt:
        return json.dumps(_plan["v"])
    if "venue" in prompt.lower():
        # venue classifier: accept the NeurIPS venue string
        return json.dumps(["Neural Information Processing Systems"])
    return ""


A._gen = fake_gen

st = make_state('NeurIPS papers 2022-2023 that cite the "RoBERTa" paper')
run(A._solve_metadata(st, st.metadata["raw_query"]))
sub = json.loads(st.output.completion)["output"]["results"]
sub_ids = {r["paper_id"] for r in sub}
even_mentioners = {str(100 + i) for i in range(30) if (100 + i) % 2 == 0}
check("body-mention-verified candidates kept (were dropped by refs check)",
      even_mentioners <= sub_ids, f"kept {len(sub_ids)}")
check("refs-unverified non-mentioners dropped", "101" not in sub_ids, str(sorted(sub_ids))[:150])
check("submission non-empty and bounded", 0 < len(sub) <= 250)
check("exact-match evidence is empty string", all(r["markdown_evidence"] == "" for r in sub))

print("\n_solve_metadata scenario B: conjunction cap-hit augmentation (metadata_26-style)")
_plan["v"] = PLAN_B
st2 = make_state("paper citing the T5 paper and the spider paper", conj=True)
run(A._solve_metadata(st2, st2.metadata["raw_query"]))
sub2 = {r["paper_id"] for r in json.loads(st2.output.completion)["output"]["results"]}
check("both-target-verified extras admitted", {"502", "503"} <= sub2, str(sorted(sub2))[:200])
check("single-target mentioner excluded", "505" not in sub2)
check("non-mentioning conjunction candidates excluded", "500" not in sub2 and "501" not in sub2)

print("\nwiring greps")
check("ref-verify uses snippet channel", "body-mention-verified" in CODE and "_snip_mention_verify" in CODE)
check("conjunction augmentation gated on cap_hit", "cap_hit and len(inter) < 40" in CODE)
check("all three batch sites bisect", CODE.count("_batch_bisect(") >= 4)
check("no direct provider imports", all(s not in CODE for s in ("import openai", "import anthropic", "import litellm")))

print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
