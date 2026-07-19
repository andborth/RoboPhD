"""Offline unit tests for agent.py helpers (model_registry / inspect_ai stubbed)."""
import asyncio
import sys
import types

# ---- stub model_registry and inspect_ai before importing agent
mr = types.ModuleType("model_registry")


class _FakeModel:
    def __init__(self, reply=""):
        self.reply = reply

    async def generate(self, prompt, config=None):
        return types.SimpleNamespace(completion=self.reply)


mr.GPT_5_4 = _FakeModel()
mr.GPT_5_4_MINI = _FakeModel()
sys.modules["model_registry"] = mr

inspect_solver = types.ModuleType("inspect_ai.solver")


class TaskState:  # minimal
    pass


def solver(fn):
    return fn


inspect_solver.TaskState = TaskState
inspect_solver.Generate = object
inspect_solver.solver = solver
inspect_tool = types.ModuleType("inspect_ai.tool")


class ToolDef:
    def __init__(self, t):
        self.name = getattr(t, "name", "")


inspect_tool.ToolDef = ToolDef
inspect_pkg = types.ModuleType("inspect_ai")
sys.modules["inspect_ai"] = inspect_pkg
sys.modules["inspect_ai.solver"] = inspect_solver
sys.modules["inspect_ai.tool"] = inspect_tool

import agent  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL: {name}"
    ok += 1
    print(f"  ok: {name}")


# _cid
check("cid int", agent._cid({"corpusId": 123}) == "123")
check("cid prefixed", agent._cid({"corpusId": "CorpusId:456"}) == "456")
check("cid missing", agent._cid({}) == "")
check("cid non-dict", agent._cid(None) == "")

# _ascii / superscripts
check("ascii sup", agent._ascii("MS² paper") == "MS2 paper")
check("ascii quotes", agent._ascii("“x” – ‘y’") == '"x" - \'y\'')

# _cut verbatim property
s = "word " * 100
c = agent._cut(s, 57)
check("cut substring", c in s and len(c) <= 57 and not c.endswith(" wor"))
check("cut short", agent._cut("abc", 10) == "abc")

# _auth_names dict/str mix
check(
    "auth mix",
    agent._auth_names({"authors": [{"name": "A B"}, "C D", 5, {"name": ""}]}) == ["A B", "C D"],
)

# _surname/_first_initial
check("surname", agent._surname("Gera Weiss") == "weiss")
check("first initial", agent._first_initial("Gera Weiss") == "g")
check("surname empty", agent._surname("") == "")

# _has_author
doc = {"authors": [{"name": "David Harel"}, {"name": "G. Weiss"}]}
check("has_author yes", agent._has_author(doc, "Gera Weiss"))
check("has_author initial mismatch", not agent._has_author(doc, "Tom Weiss"))
check("has_author no", not agent._has_author(doc, "Alice Smith"))

# _json_block
check("json fence", agent._json_block('```json\n{"a": 1}\n```') == {"a": 1})
check("json embedded", agent._json_block('text {"a": {"b": 2}} tail') == {"a": {"b": 2}})
check("json none", agent._json_block("no json here") is None)

# _parse_items with data wrapper


class Item:
    def __init__(self, text):
        self.text = text


items = [Item('{"data": [{"corpusId": 1}, {"corpusId": 2}]}'), Item('{"corpusId": 3}'), Item("not json")]
check("parse_items", [agent._cid(d) for d in agent._parse_items(items)] == ["1", "2", "3"])

# _refs_contain
refs = [{"paperId": "abc", "corpusId": 9}, {"paperId": None}]
check("refs hash", agent._refs_contain(refs, {"abc"}, set()))
check("refs cid", agent._refs_contain(refs, set(), {"9"}))
check("refs miss", not agent._refs_contain(refs, {"zzz"}, {"7"}))

# _short_name_of
check("short quoted", agent._short_name_of('papers citing the "RoBERTa" paper', "whatever") == "RoBERTa")
check(
    "short colon",
    agent._short_name_of("x", "DistilBERT, a distilled version of BERT: smaller") == "DistilBERT",
)
check(
    "short colon long",
    agent._short_name_of("x", "RoBERTa: A Robustly Optimized BERT Pretraining Approach") == "RoBERTa",
)

# _weighted / grade parsing
crit = [{"weight": 0.5, "description": "a"}, {"weight": 0.5, "description": "b"}]
check("weighted", abs(agent._weighted(crit, [3, 3]) - 1.0) < 1e-9)
check("weighted partial", abs(agent._weighted(crit, [3, 1]) - (0.5 + 0.5 / 3)) < 1e-9)


async def test_grade_chunk():
    agent.GPT_5_4_MINI.reply = "0: 3 3\n1: 1 0\n2: 3 1"
    out = await agent._grade_chunk(crit, [(0, "t"), (1, "t"), (2, "t")])
    check("grade parse", out == {0: [3, 3], 1: [1, 0], 2: [3, 1]})
    agent.GPT_5_4_MINI.reply = "garbage"
    out = await agent._grade_chunk(crit, [(5, "t")])
    check("grade fallback", out == {5: [1, 1]})


asyncio.run(test_grade_chunk())


# _evidence assembly + verbatim passages
doc = {
    "title": "T",
    "tldr": {"text": "TL"},
    "abstract": "A " * 800,
    "_snippets": ["s1", "s1", "s2"],
}
ev = agent._evidence(doc)
parts = ev.split(" ... ")
check("evidence parts", parts[0] == "T" and parts[1] == "TL" and len(parts) <= 8)
check("evidence dedup snippets", parts.count("s1") == 1 and "s2" in parts)

# _venue helpers
check("venue alias", agent._venue_ok_substring("Neural Information Processing Systems", ["NeurIPS"]))
check("venue miss", not agent._venue_ok_substring("ICML", ["Nature"]))
check("venue_str journal", agent._venue_str({"venue": "V", "journal": {"name": "J"}}) == "V | J")

# _author_ok
check("author_ok", agent._author_ok({"authors": [{"name": "David Harel"}]}, ["D. Harel"]))
check("author_ok fail", not agent._author_ok({"authors": [{"name": "Bob Jones"}]}, ["David Harel"]))

# _title_sim
check("title sim dup", agent._title_sim("ImageNet classification with deep convolutional neural networks",
                                        "ImageNet Classification with Deep Convolutional Neural Networks") > 0.99)


# _plan_semantic normalization
async def test_plan():
    agent.GPT_5_4.reply = (
        '{"criteria": [{"name": "n1", "description": "d1", "weight": 2},'
        '{"name": "n2", "description": "d2", "weight": 2}],'
        '"keyword_queries": ["q1", "q2"], "snippet_queries": ["s1"], "year_min": "2020"}'
    )
    plan = await agent._plan_semantic("query")
    check("plan weights normalized", abs(sum(c["weight"] for c in plan["criteria"]) - 1.0) < 1e-9)
    check("plan year", plan["year_min"] == 2020)
    agent.GPT_5_4.reply = "not json"
    plan = await agent._plan_semantic("query about MS² things")
    check("plan fallback criteria", plan["criteria"][0]["weight"] == 1.0)
    check("plan fallback kw", len(plan["keyword_queries"]) >= 1)


asyncio.run(test_plan())

print(f"\nALL {ok} CHECKS PASSED")
