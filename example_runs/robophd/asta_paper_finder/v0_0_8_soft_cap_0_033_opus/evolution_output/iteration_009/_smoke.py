"""Smoke test for iter9_rerank_rich_v1. Stubs inspect_ai + model_registry so the
agent imports without the real deps, then exercises the changed reranker and the
end-to-end semantic path."""
import sys, types, json, asyncio, importlib.util, os

# ---- stub inspect_ai ----
insp = types.ModuleType("inspect_ai")
solver_mod = types.ModuleType("inspect_ai.solver")
tool_mod = types.ModuleType("inspect_ai.tool")
model_mod = types.ModuleType("inspect_ai.model")

def solver(fn): return fn
class TaskState: pass
class Generate: pass
solver_mod.solver = solver
solver_mod.TaskState = TaskState
solver_mod.Generate = Generate

class ToolDef:
    def __init__(self, t): self._t = t
    @property
    def name(self): return getattr(self._t, "_name", "?")
tool_mod.ToolDef = ToolDef

class GenerateConfig:
    def __init__(self, **kw): pass
model_mod.GenerateConfig = GenerateConfig

sys.modules["inspect_ai"] = insp
sys.modules["inspect_ai.solver"] = solver_mod
sys.modules["inspect_ai.tool"] = tool_mod
sys.modules["inspect_ai.model"] = model_mod

# ---- stub model_registry ----
mr = types.ModuleType("model_registry")
class _Resp:
    def __init__(self, c): self.completion = c
class _Model:
    def __init__(self, name): self.name = name; self.handler = None
    async def generate(self, prompt, config=None):
        if _Model.handler: return _Resp(_Model.handler(prompt))
        return _Resp("{}")
_Model.handler = None
for h in ["GPT_5_4_MINI","GPT_5_4","GPT_5_5","CLAUDE_HAIKU_4_5","CLAUDE_SONNET_4_6",
          "CLAUDE_OPUS_4_8","GEMINI_3_1_FLASH_LITE","GEMINI_3_5_FLASH","GEMINI_3_1_PRO_PREVIEW"]:
    setattr(mr, h, _Model(h))
sys.modules["model_registry"] = mr

# ---- import agent ----
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("agent", os.path.join(here, "agent.py"))
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def run(coro): return asyncio.get_event_loop().run_until_complete(coro)


# ---- Test 1: reranker feeds abstract + reorders by 0-10 score ----
def test_rerank():
    cands = []
    for i in range(6):
        c = agent.Cand(str(1000 + i))
        c.title = f"Paper {i}"
        c.abstract = f"ABSTRACT_MARKER_{i} discussing topic and secondary aspect."
        c.score = 6 - i  # retrieval order 0..5
        cands.append(c)

    seen_prompt = {}
    def handler(prompt):
        seen_prompt["p"] = prompt
        # give paper 5 the top score, paper 0 the lowest -> reverse of retrieval order
        return json.dumps({str(i): i * 2 for i in range(6)})  # 0,2,4,6,8,10
    agent.GPT_5_4_MINI.handler = handler
    _Model.handler = handler

    out = run(agent._rerank("q", ["facet a", "facet b"], list(cands)))
    agent.GPT_5_4_MINI.handler = None; _Model.handler = None

    assert "ABSTRACT_MARKER_5" in seen_prompt["p"], "abstract not fed to reranker"
    assert "0-10" in seen_prompt["p"], "0-10 scale not in prompt"
    order = [c.cid for c in out]
    assert order[0] == "1005" and order[-1] == "1000", f"not reordered by score: {order}"
    print("  test_rerank OK ->", order)


# ---- Test 2: empty LLM output -> retrieval order preserved ----
def test_rerank_fallback():
    cands = []
    for i in range(6):
        c = agent.Cand(str(2000 + i)); c.title = f"T{i}"; c.abstract = "x"; c.score = 6 - i
        cands.append(c)
    def handler(prompt): return ""   # empty -> {}
    agent.GPT_5_4_MINI.handler = handler; _Model.handler = handler
    out = run(agent._rerank("q", ["f"], list(cands)))
    agent.GPT_5_4_MINI.handler = None; _Model.handler = None
    assert [c.cid for c in out] == [c.cid for c in cands], "fallback did not keep order"
    print("  test_rerank_fallback OK")


# ---- Test 3: partial output (< half) -> fallback ----
def test_rerank_partial():
    cands = []
    for i in range(10):
        c = agent.Cand(str(3000 + i)); c.title = f"T{i}"; c.abstract = "x"; c.score = 10 - i
        cands.append(c)
    def handler(prompt): return json.dumps({"0": 9, "1": 1})  # only 2/10
    agent.GPT_5_4_MINI.handler = handler; _Model.handler = handler
    out = run(agent._rerank("q", ["f"], list(cands)))
    agent.GPT_5_4_MINI.handler = None; _Model.handler = None
    assert [c.cid for c in out] == [c.cid for c in cands], "partial should fall back"
    print("  test_rerank_partial OK")


# ---- Test 4: end-to-end semantic path with fake tools ----
def test_semantic_e2e():
    class Tool:
        def __init__(self, name, fn): self._name = name; self._fn = fn
        async def __call__(self, **kw): return self._fn(**kw)

    class CT:
        def __init__(self, text): self.text = text

    def snippet_search(**kw):
        rows = [{"score": 1.0, "paper": {"corpusId": "9001", "title": "RAG survey"},
                 "snippet": {"text": "retrieval augmented generation architectures overview"}}]
        return [CT(json.dumps({"data": rows, "retrievalVersion": "1"}))]

    def search_papers_by_relevance(**kw):
        return [CT(json.dumps({"corpusId": 9001, "title": "RAG survey",
                               "abstract": "A survey of retrieval-augmented LM architectures.",
                               "tldr": {"text": "survey of RAG"}})),
                CT(json.dumps({"corpusId": 9002, "title": "Dense retriever",
                               "abstract": "Dense passage retrieval methods.", "tldr": None}))]

    def get_paper_batch(**kw):
        return [CT(json.dumps({"corpusId": 9002, "title": "Dense retriever",
                               "abstract": "Dense passage retrieval methods.", "tldr": None}))]

    tools = [
        Tool("snippet_search", snippet_search),
        Tool("search_papers_by_relevance", search_papers_by_relevance),
        Tool("get_paper_batch", get_paper_batch),
    ]
    st = agent.TaskState()
    st.tools = tools
    st.sample_id = "semantic_test"
    st.metadata = {"score_type": "semantic_f1", "raw_query": "RAG architectures"}

    def handler(prompt):
        if '"queries"' in prompt and '"facets"' in prompt:
            return json.dumps({"queries": ["RAG architectures"], "facets": ["retrieval", "architecture"]})
        if "0-10" in prompt:  # rerank
            return json.dumps({"0": 9, "1": 6})
        return "{}"
    agent.GPT_5_4_MINI.handler = handler; _Model.handler = handler
    res = run(agent._solve_semantic(st, "RAG architectures"))
    agent.GPT_5_4_MINI.handler = None; _Model.handler = None
    assert isinstance(res, list) and res, "no results"
    for r in res:
        assert "paper_id" in r and "markdown_evidence" in r
    print(f"  test_semantic_e2e OK -> {len(res)} results, first={res[0]['paper_id']}")


if __name__ == "__main__":
    test_rerank()
    test_rerank_fallback()
    test_rerank_partial()
    test_semantic_e2e()
    print("ALL SMOKE TESTS PASSED")
