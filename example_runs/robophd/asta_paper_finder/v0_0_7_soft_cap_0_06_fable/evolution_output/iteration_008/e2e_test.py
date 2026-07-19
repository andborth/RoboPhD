"""Offline end-to-end control-flow test for iter8_resilient_lean.

Fakes the MCP tools and model handles to exercise:
  1. semantic path under a transient 502 outage (fan-out retry recovers)
  2. metadata path whose channels all return empty (terminal fallback ladder)
  3. specific path with an ambiguous alias (multi-referent hedge)
  4. solver-level fallback when the route raises
"""
import asyncio
import json
import re
import sys
import types

for name in ("inspect_ai", "inspect_ai.solver", "inspect_ai.tool", "model_registry"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["inspect_ai.solver"].Generate = object
sys.modules["inspect_ai.solver"].TaskState = object
sys.modules["inspect_ai.solver"].solver = lambda f: f
sys.modules["inspect_ai.tool"].ToolDef = lambda t: t  # tools carry .name directly
sys.modules["model_registry"].GPT_5_4 = object()
sys.modules["model_registry"].GPT_5_4_MINI = object()
sys.modules["model_registry"].GEMINI_3_1_FLASH_LITE = object()

import agent

agent._RETRY_DELAYS = (0.01, 0.02)

_real_sleep = asyncio.sleep
async def _fast_sleep(t):
    await _real_sleep(min(t, 0.01))
asyncio.sleep = _fast_sleep  # retry/backoff sleeps run instantly in tests


class Item:
    def __init__(self, obj):
        self.text = json.dumps(obj)


class FakeTool:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def __call__(self, **kwargs):
        return self._fn(**kwargs)


class FakeModel:
    """Answers planner/grader/verifier prompts by shape."""

    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, config=None):
        self.calls += 1
        if '"keyword_queries"' in prompt and '"criteria"' in prompt:
            out = {
                "criteria": [
                    {"name": "topic", "description": "The paper must address grasping datasets", "weight": 0.6, "probe": "task-oriented grasping dataset"},
                    {"name": "annotations", "description": "The paper must include grasp annotations", "weight": 0.4, "probe": "object task grasp annotations"},
                ],
                "keyword_queries": [f"grasp query {i}" for i in range(12)],
                "snippet_queries": ["Papers should present task-oriented grasping datasets."],
                "year_min": None, "year_max": None,
            }
            return types.SimpleNamespace(completion=json.dumps(out))
        if "Grade candidate papers" in prompt:
            idxs = re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)
            lines = "\n".join(f"{i}: 3 3" for i in idxs)
            return types.SimpleNamespace(completion=lines)
        if "canonical_name" in prompt:
            out = {
                "canonical_name": "SPIKE",
                "candidate_titles": [
                    "SPIKE: A Parallel Environment for Solving Banded Linear Systems",
                    "SPIKE: A GPU Optimized Spiking Neural Network Framework",
                ],
                "author_hints": [], "year_hint": None, "confidence": 0.4,
            }
            return types.SimpleNamespace(completion=json.dumps(out))
        if "potentially ambiguous" in prompt:
            return types.SimpleNamespace(completion=json.dumps({"indices": [0, 1, 2, 3], "confidence": 0.5}))
        if "Parse this scholarly paper search request" in prompt:
            out = {"authors": [], "venues": ["SPLASH"], "venue_constraint": "published at SPLASH",
                   "years_allowed": [], "year_min": 2019, "year_max": None,
                   "cites_paper_title": None, "cites_author": None, "exclude_coauthor": None,
                   "min_citations": None, "min_authors": None, "max_authors": None,
                   "topic_keywords": None}
            return types.SimpleNamespace(completion=json.dumps(out))
        if "NEW keyword queries" in prompt:
            return types.SimpleNamespace(completion=json.dumps({"keyword_queries": ["fresh q1", "fresh q2"]}))
        return types.SimpleNamespace(completion="")


class FakeState:
    def __init__(self, sample_id, score_type, query, tools):
        self.sample_id = sample_id
        self.metadata = {"score_type": score_type, "raw_query": query}
        self.tools = tools
        self.output = types.SimpleNamespace(completion="")


def paper(cid, title=None):
    return {
        "corpusId": cid,
        "paperId": f"hash{cid}",
        "title": title or f"Paper {cid} on task-oriented grasping",
        "abstract": f"Abstract of paper {cid} about task-oriented grasping datasets with annotations.",
        "tldr": {"text": f"TLDR {cid}."},
        "year": 2021,
        "venue": "CoRL",
        "authors": [{"name": "Alice Smith"}],
        "citationCount": 10,
    }


fake = FakeModel()
agent.GPT_5_4 = fake
agent.GPT_5_4_MINI = fake
agent.TRIAGE_MODEL = fake


# ---- 1. semantic under a transient outage ---------------------------------
outage = {"until": 14}  # first 14 search calls 502; then service recovers
counters = {"search": 0}

def make_semantic_tools():
    def search(**kw):
        async def run():
            counters["search"] += 1
            if counters["search"] <= outage["until"]:
                raise RuntimeError("tool call failed: HTTP 502")
            base = (counters["search"] * 7) % 50
            return [Item(paper(1000 + base + i)) for i in range(20)]
        return run()

    def snippet(**kw):
        async def run():
            if counters["search"] <= outage["until"]:
                raise RuntimeError("tool call failed: HTTP 502")
            data = [
                {"score": 0.9, "paper": {"corpusId": str(2000 + i), "title": f"Snip paper {i}"},
                 "snippet": {"text": f"task-oriented grasping dataset passage {i}", "section": "abstract"}}
                for i in range(5)
            ]
            return [Item({"data": data})]
        return run()

    def batch(**kw):
        async def run():
            ids = kw.get("ids") or []
            return [Item(paper(int(i.split(":")[1]))) for i in ids]
        return run()

    return [
        FakeTool("search_papers_by_relevance", search),
        FakeTool("snippet_search", snippet),
        FakeTool("get_paper_batch", batch),
        FakeTool("search_paper_by_title", lambda **kw: None),
        FakeTool("get_citations", lambda **kw: None),
        FakeTool("search_authors_by_name", lambda **kw: None),
        FakeTool("get_author_papers", lambda **kw: None),
        FakeTool("get_paper", lambda **kw: None),
    ]


solve = agent.make_solver()

state = FakeState("semantic_test", "semantic_f1", "What are some datasets for task-oriented grasping?", make_semantic_tools())
asyncio.run(solve(state, None))
sub = json.loads(state.output.completion)["output"]
assert sub["query_id"] == "semantic_test"
assert len(sub["results"]) >= agent.POOL_MIN_OK, f"outage recovery failed: {len(sub['results'])} results"
assert all(r["markdown_evidence"] for r in sub["results"][:5]), "head evidence missing"
assert all(isinstance(r["paper_id"], str) for r in sub["results"])
print(f"semantic outage-recovery OK ({len(sub['results'])} results after {counters['search']} search calls)")


# ---- 2. metadata with empty channels --------------------------------------
final_kw_calls = {"n": 0}

def make_empty_meta_tools():
    def rel(**kw):
        async def run():
            # everything empty until the terminal fallback query fires
            if kw.get("limit") == 30:
                final_kw_calls["n"] += 1
                return [Item(paper(3000 + i)) for i in range(15)]
            return []
        return run()

    def snippet(**kw):
        async def run():
            return [Item({"data": []})]
        return run()

    async def none_(**kw):
        return None

    return [
        FakeTool("search_papers_by_relevance", rel),
        FakeTool("snippet_search", snippet),
        FakeTool("get_paper_batch", lambda **kw: none_()),
        FakeTool("search_paper_by_title", lambda **kw: none_()),
        FakeTool("get_citations", lambda **kw: none_()),
        FakeTool("search_authors_by_name", lambda **kw: none_()),
        FakeTool("get_author_papers", lambda **kw: none_()),
        FakeTool("get_paper", lambda **kw: none_()),
    ]


state2 = FakeState("metadata_test", "metadata_f1", "A SPLASH 2019 and beyond paper that cites any NeurIPS", make_empty_meta_tools())
asyncio.run(solve(state2, None))
sub2 = json.loads(state2.output.completion)["output"]
assert 0 < len(sub2["results"]) <= 12, f"terminal fallback failed: {len(sub2['results'])}"
assert final_kw_calls["n"] >= 1
assert all(r["markdown_evidence"] == "" for r in sub2["results"])
print(f"metadata never-empty OK ({len(sub2['results'])} results)")


# ---- 3. specific ambiguous alias hedge ------------------------------------
spike_titles = [
    "SPIKE: A Parallel Environment for Solving Banded Linear Systems",
    "A parallel hybrid banded system solver: the SPIKE algorithm",
    "SPIKE: A GPU Optimized Spiking Neural Network Inference Framework",
    "SPIKE: Extractive Search over Scientific Text",
]

def make_specific_tools():
    def title_search(**kw):
        async def run():
            return [Item(paper(4000, spike_titles[0]))]
        return run()

    def rel(**kw):
        async def run():
            return [Item(paper(4000 + i, t)) for i, t in enumerate(spike_titles)]
        return run()

    def snippet(**kw):
        async def run():
            return [Item({"data": []})]
        return run()

    return [
        FakeTool("search_paper_by_title", title_search),
        FakeTool("search_papers_by_relevance", rel),
        FakeTool("snippet_search", snippet),
        FakeTool("get_paper_batch", lambda **kw: None),
        FakeTool("get_citations", lambda **kw: None),
        FakeTool("search_authors_by_name", lambda **kw: None),
        FakeTool("get_author_papers", lambda **kw: None),
        FakeTool("get_paper", lambda **kw: None),
    ]


state3 = FakeState("specific_test", "specific_f1", "the SPIKE paper", make_specific_tools())
asyncio.run(solve(state3, None))
sub3 = json.loads(state3.output.completion)["output"]
ids3 = [r["paper_id"] for r in sub3["results"]]
assert 3 <= len(ids3) <= 10, f"hedge should submit several records, got {ids3}"
assert "4001" in ids3, "hedge missed a verifier-listed non-head-title work"
print(f"specific ambiguous hedge OK ({len(ids3)} records: {ids3})")


# ---- 4. solver fallback when the route raises -----------------------------
def make_broken_then_ok_tools():
    calls = {"n": 0}

    def search(**kw):
        async def run():
            calls["n"] += 1
            if calls["n"] <= 4:
                raise RuntimeError("tool call failed: HTTP 502")
            return [Item(paper(5000 + i)) for i in range(10)]
        return run()

    async def broken(**kw):
        raise RuntimeError("tool call failed: HTTP 502")

    return [
        FakeTool("search_papers_by_relevance", search),
        FakeTool("snippet_search", lambda **kw: broken()),
        FakeTool("get_paper_batch", lambda **kw: broken()),
        FakeTool("search_paper_by_title", lambda **kw: broken()),
        FakeTool("get_citations", lambda **kw: broken()),
        FakeTool("search_authors_by_name", lambda **kw: broken()),
        FakeTool("get_author_papers", lambda **kw: broken()),
        FakeTool("get_paper", lambda **kw: broken()),
    ]


state4 = FakeState("specific_broken", "specific_f1", "the SPIKE paper", make_broken_then_ok_tools())
asyncio.run(solve(state4, None))
sub4 = json.loads(state4.output.completion)["output"]
assert len(sub4["results"]) > 0, "solver fallback should still submit"
print(f"solver fallback OK ({len(sub4['results'])} results)")

print("ALL E2E TESTS PASSED")
