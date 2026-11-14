"""Smoke test for iter12_body_conjunction: stub model_registry + MCP tools and
run the FULL semantic pipeline end-to-end, checking the new mechanisms:
5 per-query snippet source lists, snippet-doc pool share, sim skip for
snippetless-weak head papers, reference field-variant probe (both the
dead-server case and the subfield-works case), tail sweep, telemetry."""
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
                    "snippet_queries": ["Papers should show pruning.", "s2", "s3", "s4", "s5"],
                    "year_min": None, "year_max": None,
                }
            )
        elif "keyword_queries" in prompt:  # gap-fill
            resp.completion = json.dumps({"keyword_queries": ["gap1", "gap2", "gap3", "gap4", "gap5"]})
        elif "Grade candidate papers" in prompt:
            # parse candidate indices; mixed grades so the sim-skip branch has
            # both weak (<= SIM_SKIP_W) and strong stage-1 papers
            import re
            idxs = re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)
            resp.completion = "\n".join(
                f"{i}: 3 1" if int(i) % 3 else f"{i}: 1 0" for i in idxs
            )
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


def make_tools(refs_mode="dead"):
    """refs_mode: 'dead' = every references variant fails;
    'subfield' = only the dotted-subfield variant works."""
    counters = {
        "search": 0, "snippet_scoped": 0, "snippet_global": 0,
        "get_paper_plain_refs": 0, "get_paper_subfield_refs": 0, "citations": 0,
        "batch_plain_refs": 0, "batch_subfield_refs": 0,
        "snippet_global_limits": [],
    }

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
        counters["snippet_global_limits"].append(limit)
        base = (hash(query) % 5) * 30
        return [Item({"data": [
            {"score": 0.9, "paper": {"corpusId": str(20000 + base + i), "title": f"Snip paper {base + i}"},
             "snippet": {"text": f"Task-agnostic pruning text {base + i}."}}
            for i in range(limit or 10)
        ]})]

    async def get_paper_batch(ids, fields=None):
        f = fields or ""
        if "references." in f:
            counters["batch_subfield_refs"] += 1
            if refs_mode != "subfield":
                raise RuntimeError("ToolError: 'NoneType' object is not iterable")
            return [Item({"corpusId": int(i.split(":")[1]),
                          "references": [{"corpusId": 30000 + j, "title": "ref"} for j in range(5)]})
                    for i in ids]
        if "references" in f:
            counters["batch_plain_refs"] += 1
            raise RuntimeError("ToolError: 'NoneType' object is not iterable")
        return [Item({"corpusId": int(i.split(":")[1]), "title": f"T{i}",
                      "abstract": "We prune large language models."}) for i in ids]

    async def get_paper(paper_id, fields=None):
        f = fields or ""
        cid = int(paper_id.split(":")[1])
        if "references." in f:
            counters["get_paper_subfield_refs"] += 1
            if refs_mode != "subfield":
                raise RuntimeError("ToolError: 'NoneType' object is not iterable")
            return [Item({"corpusId": cid,
                          "references": [{"corpusId": 30000 + cid % 100 + j, "title": "ref"} for j in range(5)]})]
        if "references" in f:
            counters["get_paper_plain_refs"] += 1
            raise RuntimeError("ToolError: 'NoneType' object is not iterable")
        return [Item({"corpusId": cid, "title": "T", "abstract": "A"})]

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


class FakeOutput:
    completion = ""


class FakeState:
    sample_id = "smoke_1"
    metadata = {"raw_query": "task-agnostic pruning methods for LLMs", "score_type": "semantic_f1"}
    output = FakeOutput()
    tools = []


async def run_semantic(refs_mode):
    import time
    counters, tools = make_tools(refs_mode)
    agent._get_tool = lambda state, name: tools[name]
    state = FakeState()
    state.output = FakeOutput()
    agent._llm_reset()
    await agent._solve_semantic(state, state.metadata["raw_query"], time.monotonic())
    payload = json.loads(state.output.completion)
    results = payload["output"]["results"]
    assert payload["output"]["query_id"] == "smoke_1"
    assert 1 <= len(results) <= 250, len(results)
    for r in results:
        assert isinstance(r["paper_id"], str) and r["paper_id"].isdigit(), r["paper_id"]
        assert isinstance(r["markdown_evidence"], str) and r["markdown_evidence"]
    return counters, results


async def main():
    # ---- refs dead (the observed live-server behavior)
    counters, results = await run_semantic("dead")
    # 5 global snippet calls at the new limit
    assert counters["snippet_global"] == 5, counters
    assert all(l == agent.SNIP_INIT_LIMIT for l in counters["snippet_global_limits"]), counters
    # probe stops after ONE plain + ONE subfield failure (not one per seed)
    assert counters["get_paper_plain_refs"] == 1, counters
    assert counters["get_paper_subfield_refs"] == 1, counters
    assert counters["citations"] > 0, "citer fetch never ran"
    assert counters["snippet_scoped"] > 0, "no scoped snippet enrichment ran"
    # snippet-sourced docs must reach the submission (they get pool share now)
    snip_sourced = [r for r in results if r["paper_id"].startswith("2000") or r["paper_id"].startswith("200")]
    assert any(int(r["paper_id"]) >= 20000 and int(r["paper_id"]) < 30000 for r in results), \
        "no snippet-sourced docs in submission"
    rep = agent._llm_report()
    assert "t1:" in rep and "plan:" in rep, rep
    print(f"dead-refs pipeline ok: {len(results)} results, counters {counters}")
    print(f"telemetry: {rep}")

    # ---- refs available under the subfield variant: probe commits, rest reuse it
    counters2, _ = await run_semantic("subfield")
    assert counters2["get_paper_plain_refs"] == 1, counters2
    assert counters2["get_paper_subfield_refs"] == agent.EXPAND_SEEDS, counters2
    print(f"subfield-refs pipeline ok: counters {counters2}")

    # ---- _fetch_references probe logic directly (metadata path)
    counters3, tools3 = make_tools("dead")
    refs = await agent._fetch_references(tools3["get_paper_batch"],
                                         [{"corpusId": str(100 + i)} for i in range(30)])
    assert refs == {}, refs
    assert counters3["batch_plain_refs"] == 1 and counters3["batch_subfield_refs"] == 1, counters3
    counters4, tools4 = make_tools("subfield")
    refs = await agent._fetch_references(tools4["get_paper_batch"],
                                         [{"corpusId": str(100 + i)} for i in range(30)])
    assert len(refs) == 30, len(refs)
    print("metadata _fetch_references probe ok (dead + subfield)")


asyncio.run(main())

# ---- constants consistency
assert agent.POOL_MERGE_HEAD < agent.POOL_CAP < agent.POOL_CAP_TOTAL
assert agent.VERIFY_TOP_THIN > agent.VERIFY_TOP
assert agent.TAIL_SWEEP_END <= agent.MAX_SUBMIT
assert agent.TAIL_DEADLINE > agent.SOFT_DEADLINE
assert agent.HEAD == 100 and agent.RESCUE_MAX == 24 and agent.EXPAND_CAP == 120
print("constants ok")

s = agent.make_solver()
assert callable(s)
print("solver builds ok")
print("ALL SMOKE TESTS PASSED")
