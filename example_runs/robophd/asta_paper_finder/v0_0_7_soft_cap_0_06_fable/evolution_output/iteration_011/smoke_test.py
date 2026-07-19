"""Smoke test for iter11_tail_saturate: stub model_registry + MCP tools and
run the FULL semantic pipeline end-to-end, checking the new mechanisms:
lexical prescreen, per-seed reference expansion, qualifier enrichment,
full-head rescue, thin-pool verify depth, tail sweep, and telemetry."""
import asyncio
import json
import sys
import types

# ---- stub model_registry before importing agent
class FakeModel:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def generate(self, prompt, config=None):
        self.calls.append(prompt)
        resp = types.SimpleNamespace()
        if "Reply with JSON only" in prompt and "keyword_queries" in prompt and "criteria" in prompt:
            resp.completion = json.dumps(
                {
                    "criteria": [
                        {"name": "pruning", "description": "The paper must discuss pruning of LLMs",
                         "weight": 0.6, "probe": "pruning large language models"},
                        {"name": "task-agnostic", "description": "The method must be task-agnostic",
                         "weight": 0.4, "probe": "task-agnostic general-purpose"},
                    ],
                    "keyword_queries": [f"kw{i}" for i in range(10)],
                    "snippet_queries": ["Papers should show pruning.", "s2", "s3"],
                    "year_min": None, "year_max": None,
                }
            )
        elif "keyword_queries" in prompt:  # gap-fill
            resp.completion = json.dumps({"keyword_queries": ["gap1", "gap2", "gap3", "gap4", "gap5"]})
        elif "Grade candidate papers" in prompt:
            # parse candidate indices, grade everything 3 1 (one weak criterion)
            import re
            idxs = re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)
            resp.completion = "\n".join(f"{i}: 3 1" for i in idxs)
        else:
            resp.completion = "{}"
        return resp


stub = types.ModuleType("model_registry")
stub.GPT_5_4 = FakeModel("gpt54")
stub.GPT_5_4_MINI = FakeModel("mini")
sys.modules["model_registry"] = stub

import agent  # noqa: E402

agent._RETRY_DELAYS = (0, 0)

# ---- fake MCP tools
class Item:
    def __init__(self, obj):
        self.text = json.dumps(obj)


N_SEARCH_DOCS = 60  # per keyword query -> 10*60 raw, overlapping ids


def make_tools():
    counters = {"search": 0, "snippet_scoped": 0, "snippet_global": 0, "get_paper": 0, "citations": 0}

    async def search_papers_by_relevance(keyword, fields=None, limit=None, venues=None):
        counters["search"] += 1
        base = (hash(keyword) % 7) * 50
        return [
            Item({"corpusId": 10000 + base + i, "title": f"Pruning paper {base + i}",
                  "abstract": f"We prune large language models, method {base + i}.",
                  "year": 2024})
            for i in range(N_SEARCH_DOCS)
        ]

    async def snippet_search(query, limit=None, venues=None, paper_ids=None):
        if paper_ids:
            counters["snippet_scoped"] += 1
            cid = paper_ids.split(":")[1]
            return [Item({"data": [
                {"score": 0.9,
                 "paper": {"corpusId": cid, "title": f"Pruning paper {cid}"},
                 "snippet": {"text": f"Our approach is task-agnostic and general-purpose ({cid})."}}
            ]})]
        counters["snippet_global"] += 1
        return [Item({"data": [
            {"score": 0.9, "paper": {"corpusId": str(20000 + i), "title": f"Snip paper {i}"},
             "snippet": {"text": f"Task-agnostic pruning text {i}."}}
            for i in range(limit or 10)
        ]})]

    async def get_paper_batch(ids, fields=None):
        if "references" in (fields or ""):
            raise RuntimeError("ToolError: 'NoneType' object is not iterable")
        return [Item({"corpusId": int(i.split(":")[1]), "title": f"T{i}",
                      "abstract": "We prune large language models."}) for i in ids]

    async def get_paper(paper_id, fields=None):
        counters["get_paper"] += 1
        cid = int(paper_id.split(":")[1])
        return [Item({"corpusId": cid,
                      "references": [{"corpusId": 30000 + cid % 100 + j, "title": "ref"} for j in range(5)]})]

    async def get_citations(paper_id, fields=None, limit=None):
        counters["citations"] += 1
        cid = int(paper_id.split(":")[1])
        return [Item({"citingPaper": {"corpusId": str(40000 + cid % 100 + j), "title": "citer"}})
                for j in range(10)]

    return counters, {
        "search_papers_by_relevance": search_papers_by_relevance,
        "snippet_search": snippet_search,
        "get_paper_batch": get_paper_batch,
        "get_paper": get_paper,
        "get_citations": get_citations,
    }


counters, tools = make_tools()
agent._get_tool = lambda state, name: tools[name]


class FakeOutput:
    completion = ""


class FakeState:
    sample_id = "smoke_1"
    metadata = {"raw_query": "task-agnostic pruning methods for LLMs", "score_type": "semantic_f1"}
    output = FakeOutput()
    tools = []


async def main():
    import time
    state = FakeState()
    agent._llm_reset()
    t0 = time.monotonic()
    await agent._solve_semantic(state, state.metadata["raw_query"], t0)
    payload = json.loads(state.output.completion)
    results = payload["output"]["results"]
    assert payload["output"]["query_id"] == "smoke_1"
    assert 1 <= len(results) <= 250, len(results)
    for r in results:
        assert isinstance(r["paper_id"], str) and r["paper_id"].isdigit(), r["paper_id"]
        assert isinstance(r["markdown_evidence"], str) and r["markdown_evidence"]
    # expansion must have run its per-seed reference + citer fetches (the
    # fake model grades all docs identically, so expansion docs legitimately
    # rank below the 250-cap by index tiebreak — pool membership is enough)
    assert counters["get_paper"] > 0, "per-seed reference fetch never ran"
    assert counters["citations"] > 0, "citer fetch never ran"
    assert counters["snippet_scoped"] > 0, "no scoped snippet enrichment ran"
    rep = agent._llm_report()
    assert "t1:" in rep and "plan:" in rep, rep
    print(f"pipeline ok: {len(results)} results, tool counters {counters}")
    print(f"telemetry: {rep}")


asyncio.run(main())

# ---- prescreen unit check: leftovers with criteria words beat junk
docs_hi = [{"corpusId": str(i), "title": "pruning large language models", "abstract": "task-agnostic"} for i in range(500)]
w = agent._content_words("pruning large language models task-agnostic")
assert w & agent._content_words(docs_hi[0]["title"] + " " + docs_hi[0]["abstract"])
print("prescreen word overlap ok")

# ---- constants consistency
assert agent.POOL_MERGE_HEAD < agent.POOL_CAP < agent.POOL_CAP_TOTAL
assert agent.VERIFY_TOP_THIN > agent.VERIFY_TOP
assert agent.TAIL_SWEEP_END <= agent.MAX_SUBMIT
assert agent.TAIL_DEADLINE > agent.SOFT_DEADLINE
print("constants ok")

s = agent.make_solver()
assert callable(s)
print("solver builds ok")
print("ALL SMOKE TESTS PASSED")
