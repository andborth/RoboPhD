"""Offline smoke test for iter19_breadth_restored.

Stubs out inspect_ai / model_registry so agent.py imports without the
benchmark harness, then exercises the mechanisms this iteration changed:
the planner's reverted 10-query parse, the restored gap-fill round (source
assertions), the compact triage format, evidence assembly with containment
dedup, the ambiguous-specific hedge, and the metadata conjunction parsing.

Run: /opt/anaconda3/envs/robophd_demo/bin/python smoke_test.py
"""

import asyncio
import json
import re
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

# CODE = the source with the module docstring stripped. The docstring is a
# changelog and legitimately names things this iteration REMOVED (e.g. the
# per-stage cost table mentions the deleted t1gap stage), so grep-the-source
# assertions about removed machinery must run against CODE, not SRC.
import ast as _ast

_doc = _ast.get_docstring(_ast.parse(SRC)) or ""
CODE = "\n".join(SRC.splitlines()[len(_doc.splitlines()) + 2 :])
FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAIL.append(name)


print("constants: breadth at full triage quality")
check("pool REVERTED to iter13 geometry", A.POOL_CAP == 320 and A.POOL_MERGE_HEAD == 240)
check("triage view NOT cheapened", A.T1_TITLE == 110 and A.T1_BODY == 170)
check("chunk NOT enlarged", A.GRADE_CHUNK == 32)
check("snippet limit reverted to iter13", A.SNIP_INIT_LIMIT == 50)
check("gap-fill constants restored", A.GAP_MIN_PERFECT == 20 and A.POOL_CAP_TOTAL == 380)
check("gap-fill block restored", 'label="t1gap"' in CODE and 'label="gap"' in CODE)
check("gap-fill caps at POOL_CAP_TOTAL", "POOL_CAP_TOTAL" in CODE)
check("gap-fill gated on predicted-perfect", "n_perfect < GAP_MIN_PERFECT" in CODE)
check("verify depth restored", A.VERIFY_TOP == 26 and A.VERIFY_TOP_THIN == 30)
check("sim/rescue depth restored", A.SIM_DEPTH == 55 and A.RESCUE_MAX == 22)
check("verify depth <= head", A.VERIFY_TOP_THIN <= A.HEAD)
check("pool exceeds head", A.POOL_CAP > A.HEAD)
check("tail sweep reaches submission end", A.TAIL_SWEEP_END == A.MAX_SUBMIT)
check("sim depth below head", A.SIM_DEPTH < A.HEAD)

print("\n_cut keeps passages verbatim")
long_abs = " ".join(f"word{i}" for i in range(600))
c = A._cut(long_abs, 2000)
check("cut is a verbatim substring", c in long_abs)
check("cut respects the cap", len(c) <= 2000)
check("short text untouched", A._cut("hello there", 500) == "hello there")

print("\n_redundant (containment dedup)")
check("exact repeat caught", A._redundant("Deep learning for NLP", ["Deep learning for NLP"]))
check("truncated form caught", A._redundant("Deep learning", ["Deep learning for NLP tasks"]))
check("case/punct tolerated", A._redundant("deep learning, for NLP!", ["Deep learning for NLP"]))
check("distinct passage kept", not A._redundant("Vanilla RNN gradients vanish", ["Deep learning for NLP"]))
check("empty is redundant", A._redundant("   ", ["anything"]))

print("\n_evidence")
doc = {
    "title": "LSTM versus RNN",
    "tldr": {"text": "LSTMs outperform RNNs on language modelling benchmarks."},
    "abstract": "We compare LSTMs against vanilla RNNs on language modelling. " * 4,
    "_snippets": [
        "We compare LSTMs against vanilla RNNs on language modelling.",  # inside abstract
        "Vanishing gradients explain the observed gap.",
        "Vanishing gradients explain the observed gap.",  # exact repeat
        "Perplexity drops from 120 to 94 on Penn Treebank.",
    ],
}
crit = [{"name": "gap", "description": "reasons for the difference", "weight": 1.0,
         "probe": "vanishing gradients", "probe2": "exploding gradients"}]
ev = A._evidence(doc, crit)
parts = [p.strip() for p in ev.split(" ... ")]
check("no duplicate passages", len(parts) == len(set(parts)), parts)
check("abstract-contained snippet dropped", not any(p == doc["_snippets"][0] for p in parts))
check("distinct snippets survive", any("Vanishing gradients" in p for p in parts))
check("at most 8 passages", len(parts) <= 8)
src_text = f"{doc['title']} {doc['tldr']['text']} {doc['abstract']} {' '.join(doc['_snippets'])}"
for p in parts:
    check(f"grounded: {p[:34]!r}", p in src_text)
# tldr slot economy: abstract + >=3 distinct snippets -> tldr dropped
check("tldr dropped when abstract+3snips", not any("outperform" in p for p in parts))
check("no title, no abstract", A._evidence({}, None) == "")
check("title only", A._evidence({"title": "T"}, None) == "T")

print("\n_grade_chunk parsing (compact format + local->global index mapping)")


async def fake_gen(model, prompt, retries=1, label="other"):
    return fake_gen.reply


real_gen = A._gen
A._gen = fake_gen
A._llm_reset()

crits = [{"name": "a", "description": "d1", "weight": 0.5}, {"name": "b", "description": "d2", "weight": 0.5}]
chunk = [(412, "paper one"), (413, "paper two"), (414, "paper three")]

fake_gen.reply = "1:31\n2:03\n3:11"
got = asyncio.run(A._grade_chunk(crits, chunk))
check("compact grades map to global indices", got == {412: [3, 1], 413: [0, 3], 414: [1, 1]}, got)

fake_gen.reply = "1: 3 1\n2: 0 3\n3: 1 1"
check("legacy spaced form still parses", asyncio.run(A._grade_chunk(crits, chunk)) == {412: [3, 1], 413: [0, 3], 414: [1, 1]})

fake_gen.reply = "1:31\n99:00\n3:11"
got = asyncio.run(A._grade_chunk(crits, chunk))
check("stray local index ignored", set(got) == {412, 414}, got)

fake_gen.reply = "1:322"
got = asyncio.run(A._grade_chunk(crits, chunk))
check("grade 2 folded to 1", got[412] == [3, 1], got)

print("\n_plan_semantic (reverted 10-query parse)")
fake_gen.reply = json.dumps({
    "criteria": [
        {"name": "topic", "description": "The paper must address X.", "weight": 0.6,
         "probe": "we address X", "probe2": "X is studied"},
        {"name": "conn", "description": "The paper must connect X to Y.", "weight": 0.6,
         "probe": "X applied to Y", "probe2": "Y via X"},
    ],
    "keyword_queries": [f"query {i}" for i in range(20)],
    "snippet_queries": ["Sentence one.", "Sentence two."],
    "year_min": None, "year_max": None,
})
plan = asyncio.run(A._plan_semantic("find X in Y", False))
check("keyword queries capped at 10", len(plan["keyword_queries"]) == 10, len(plan["keyword_queries"]))
check("criteria weights normalized", abs(sum(c["weight"] for c in plan["criteria"]) - 1.0) < 1e-6)
check("probe2 preserved", plan["criteria"][0]["probe2"] == "X is studied")
check("planner asks for 10 diverse queries", "keyword_queries: 10 DIVERSE" in SRC and "14 DIVERSE" not in SRC)
check("planner prompt is iter13 wording", "different phrasings, synonyms, method names, and sub-aspects" in SRC)

fake_gen.reply = "not json at all"
plan = asyncio.run(A._plan_semantic("find X in Y", False))
check("planner fallback criteria", plan["criteria"][0]["weight"] == 1.0)
check("planner fallback keywords nonempty", plan["keyword_queries"])
A._gen = real_gen

print("\nmetadata path: conjunction + venue filter wiring")
check("cites_paper_titles list prompt", '"cites_paper_titles"' in SRC)
check("semicolon split guard", re.search(r're\.split\(r"\\s\*;\\s\*"', SRC) is not None)
check("intersection fallback present", "_cites_count" in SRC)
check("chunked venue filter kept", "CV_LLM_CHUNK" in SRC and A.CV_LLM_MAX == 400)
check("no alphabetical [:120] venue truncation", "if v})[:120]" not in SRC and "if v})[:CV_LLM_MAX]" in SRC)
check("mention-channel fallback for capped citers", "citationCount" in SRC)

print("\nspecific path: widened ambiguous hedge")
check("alias search at tool max", "name_limit = 100 if ambiguous else 20" in SRC)
check("shortlist widened", "[:48]" in SRC)
check("ambiguous backstop re-tightened to 5", "len(results) < 5" in SRC and "len(results) < 12" not in SRC)
check("ambiguous submit cap re-tightened to 8", "results[: 8 if ambiguous else 5]" in SRC)
check("ambiguous alternates re-tightened", "n_extra = 6 if ambiguous" in SRC)
check("ambiguous punt fallback re-tightened", "scored[: 5 if ambiguous else 3]" in SRC)
check("_alias_titled prefix match", A._alias_titled("SPIKE", "SPIKE: A Simulator for Things"))
check("_alias_titled exact match", A._alias_titled("SPIKE", "spike"))
check("_alias_titled rejects others", not A._alias_titled("SPIKE", "SPIKED: unrelated"))

print("\nmisc helpers unchanged")
check("_cid strips prefix", A._cid({"corpusId": "CorpusId:123"}) == "123")
check("_cid int cast", A._cid({"corpusId": 456}) == "456")
check("_json_block on fenced json", A._json_block('```json\n{"a": 1}\n```') == {"a": 1})
check("_weighted caps at 1", A._weighted([{"weight": 1.0}], [3]) == 1.0)
check("make_solver returns callable", callable(A.make_solver()))

print()
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("all smoke tests passed")
