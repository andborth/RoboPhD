"""Offline smoke test for iter16_pool_breadth.

Stubs out inspect_ai / model_registry so agent.py imports without the
benchmark harness, then exercises the pure helpers and the two mechanisms
this iteration changed: evidence assembly (dedup + longer passages) and the
multi-target citation-intersection parsing.

Run: /opt/anaconda3/envs/robophd_demo/bin/python smoke_test.py
"""

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

FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAIL.append(name)


print("constants")
check("pool widened", A.POOL_CAP == 640 and A.POOL_MERGE_HEAD == 460)
check("pool ceiling above cap", A.POOL_CAP_TOTAL > A.POOL_CAP)
check("triage view trimmed", A.T1_TITLE == 85 and A.T1_BODY == 105)
check("chunk enlarged", A.GRADE_CHUNK == 48)
check("snippet limit at tool max", A.SNIP_INIT_LIMIT == 100)
check("sim depth capped below head", A.SIM_DEPTH < A.HEAD)
check("evidence caps exceed grader cut", A.EV_ABSTRACT_CUT > A.SIM_CUT and A.EV_SNIPPET_CUT > A.SIM_CUT)
check("tail sweep reaches submission end", A.TAIL_SWEEP_END == A.MAX_SUBMIT)
check("gap-fill constant gone", not hasattr(A, "GAP_MIN_PERFECT"))

print("\n_cut keeps passages verbatim")
long_abs = " ".join(f"word{i}" for i in range(600))
c = A._cut(long_abs, A.EV_ABSTRACT_CUT)
check("cut is a verbatim substring", c in long_abs)
check("cut respects the cap", len(c) <= A.EV_ABSTRACT_CUT)
check("short text untouched", A._cut("hello there", 500) == "hello there")

print("\n_redundant (global passage dedup)")
check("exact repeat caught", A._redundant("Deep learning for NLP", ["Deep learning for NLP"]))
check("truncated form caught", A._redundant("Deep learning", ["Deep learning for NLP tasks"]))
check("longer superset caught", A._redundant("Deep learning for NLP tasks", ["Deep learning"]))
check("case/punct tolerated", A._redundant("deep learning, for NLP!", ["Deep learning for NLP"]))
check("distinct passage kept", not A._redundant("Vanilla RNN gradients vanish", ["Deep learning for NLP"]))
check("empty is redundant", A._redundant("   ", ["anything"]))

print("\n_evidence")
doc = {
    "title": "LSTM versus RNN",
    "tldr": "LSTM versus RNN",  # duplicate of title -> must be dropped
    "abstract": "We compare LSTMs against vanilla RNNs on language modelling. " * 4,
    "_snippets": [
        "We compare LSTMs against vanilla RNNs on language modelling.",  # inside abstract
        "Vanishing gradients explain the observed gap.",
        "Vanishing gradients explain the observed gap.",  # exact repeat
        "Perplexity drops from 120 to 94 on Penn Treebank.",
    ],
}
ev = A._evidence(doc, [{"name": "gap", "description": "reasons for the difference", "weight": 1.0, "probe": "vanishing gradients"}])
parts = [p.strip() for p in ev.split(" ... ")]
check("no duplicate passages", len(parts) == len(set(parts)), parts)
check("title dedup dropped tldr", sum(1 for p in parts if p == "LSTM versus RNN") == 1)
check("abstract-contained snippet dropped", not any(p == doc["_snippets"][0] for p in parts))
check("distinct snippets survive", any("Vanishing gradients" in p for p in parts))
check("at most 8 passages", len(parts) <= 8)
for p in parts:
    src = f"{doc['title']} {doc['tldr']} {doc['abstract']} {' '.join(doc['_snippets'])}"
    check(f"grounded: {p[:34]!r}", p in src)

print("\nempty / degenerate docs")
check("no title, no abstract", A._evidence({}, None) == "")
check("title only", A._evidence({"title": "T"}, None) == "T")

print("\ncites_paper_titles parsing (metadata conjunction)")


def parse_titles(plan):
    """Mirror of the parsing block in _solve_metadata."""
    raw = plan.get("cites_paper_titles")
    if isinstance(raw, str):
        raw = [raw]
    if not raw:
        raw = [plan.get("cites_paper_title")] if plan.get("cites_paper_title") else []
    out = []
    for t in raw:
        if isinstance(t, str):
            for part in re.split(r"\s*;\s*", t):
                part = part.strip()
                if part and part not in out:
                    out.append(part)
    return out[:4]


check("list form", parse_titles({"cites_paper_titles": ["T5 paper", "Spider paper"]}) == ["T5 paper", "Spider paper"])
check("legacy scalar", parse_titles({"cites_paper_title": "BERT"}) == ["BERT"])
check("semicolon-joined split (the metadata_26 bug)",
      parse_titles({"cites_paper_titles": ["Exploring the Limits; Spider: Text-to-SQL"]})
      == ["Exploring the Limits", "Spider: Text-to-SQL"])
check("string instead of list", parse_titles({"cites_paper_titles": "BERT"}) == ["BERT"])
check("absent -> empty", parse_titles({}) == [])
check("dedupes repeats", parse_titles({"cites_paper_titles": ["A", "A", "B"]}) == ["A", "B"])
check("caps at 4", len(parse_titles({"cites_paper_titles": list("ABCDEFG")})) == 4)

print("\ncitation intersection semantics")


def intersect(citer_lists, n_targets):
    """Mirror of the intersection block."""
    sets = [{d["id"] for d in l} for l in citer_lists]
    all_docs = {}
    for l in citer_lists:
        for d in l:
            all_docs.setdefault(d["id"], dict(d))
    for cid, d in all_docs.items():
        d["_cites_count"] = sum(1 for s in sets if cid in s)
    need = n_targets
    inter = [d for d in all_docs.values() if d["_cites_count"] >= need]
    while not inter and need > 1:
        need -= 1
        inter = [d for d in all_docs.values() if d["_cites_count"] >= need]
    return sorted(inter, key=lambda d: -d["_cites_count"]), need


a = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
b = [{"id": "2"}, {"id": "3"}, {"id": "4"}]
got, need = intersect([a, b], 2)
check("intersection only", {d["id"] for d in got} == {"2", "3"} and need == 2)
got, need = intersect([[{"id": "1"}], [{"id": "9"}]], 2)
check("empty intersection falls back to union", len(got) == 2 and need == 1)
got, need = intersect([a], 1)
check("single target keeps all", len(got) == 3 and need == 1)

print("\nkeyword-query dedup (planner breadth)")


def dedup(kws):
    out, seen = [], set()
    for k in kws:
        key = frozenset(w for w in A._norm(k).split() if w not in A._STOP)
        if not key:
            key = frozenset({A._norm(k)})
        if key not in seen:
            seen.add(key)
            out.append(k)
    return out


check("stopword-only variants collapse", dedup(["neural machine translation", "the neural machine translation"]) == ["neural machine translation"])
check("genuinely different survive", len(dedup(["neural machine translation", "statistical machine translation"])) == 2)
check("order preserved", dedup(["zebra topic", "alpha topic"]) == ["zebra topic", "alpha topic"])
# short model/dataset names must not be swallowed by the length filter
check("short names stay distinctive", len(dedup(["T5 finetuning", "RL finetuning"])) == 2)
check("case-insensitive collapse", dedup(["BERT Pretraining", "bert pretraining"]) == ["BERT Pretraining"])

print("\n_grade_chunk parsing (compact format + local->global index mapping)")


async def fake_gen(model, prompt, retries=1, label="other"):
    return fake_gen.reply


A._gen = fake_gen
A._llm_reset()
import asyncio  # noqa: E402

crits = [{"name": "a", "description": "d1", "weight": 0.5}, {"name": "b", "description": "d2", "weight": 0.5}]
# chunk carries GLOBAL pool indices; the model is shown LOCAL numbers 1..N
chunk = [(412, "paper one"), (413, "paper two"), (414, "paper three")]

fake_gen.reply = "1:31\n2:03\n3:11"
got = asyncio.run(A._grade_chunk(crits, chunk))
check("compact grades map to global indices", got == {412: [3, 1], 413: [0, 3], 414: [1, 1]}, got)

fake_gen.reply = "1: 3 1\n2: 0 3\n3: 1 1"
check("legacy spaced form still parses", asyncio.run(A._grade_chunk(crits, chunk)) == {412: [3, 1], 413: [0, 3], 414: [1, 1]})

fake_gen.reply = "1:31\n99:00\n3:11"   # out-of-range local index must be ignored
got = asyncio.run(A._grade_chunk(crits, chunk))
check("stray index ignored, not misattributed", set(got) == {412, 414}, got)

fake_gen.reply = "2 is highly relevant overall"   # unparseable
got = asyncio.run(A._grade_chunk(crits, chunk))
check("unparseable chunk falls back to partial", got == {412: [1, 1], 413: [1, 1], 414: [1, 1]}, got)

fake_gen.reply = "1:322"   # grade 2 is not a judge output -> folded to 1
got = asyncio.run(A._grade_chunk(crits, chunk))
check("grade 2 folded to 1", got[412] == [3, 1], got)

print("\nmisc helpers unchanged")
check("_cid strips prefix", A._cid({"corpusId": "CorpusId:123"}) == "123")
check("_cid int cast", A._cid({"corpusId": 456}) == "456")
check("_json_block on fenced json", A._json_block('```json\n{"a": 1}\n```') == {"a": 1})
check("_weighted caps at 1", A._weighted([{"weight": 1.0}], [3]) == 1.0)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("all smoke tests passed")
