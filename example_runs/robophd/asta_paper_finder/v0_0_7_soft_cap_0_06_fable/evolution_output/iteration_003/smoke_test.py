"""Smoke-test agent.py's pure-python helpers with model_registry/inspect_ai stubbed."""
import asyncio
import sys
import types


def _solver(f):
    return f


for name, attrs in [
    ("inspect_ai", {}),
    ("inspect_ai.solver", {"Generate": object, "TaskState": object, "solver": _solver}),
    ("inspect_ai.tool", {"ToolDef": object}),
    ("model_registry", {"GPT_5_4": object(), "GPT_5_4_MINI": object()}),
]:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m

import agent

# _json_block
assert agent._json_block('```json\n{"a": 1}\n```') == {"a": 1}
assert agent._json_block('x {"criteria": [{"description": "d", "weight": 0.4}]} y')["criteria"][0]["weight"] == 0.4
assert agent._json_block("no json here") is None

# _cid
assert agent._cid({"corpusId": 123}) == "123"
assert agent._cid({"corpusId": "CorpusId:456"}) == "456"
assert agent._cid({}) == ""

# _cut stays a substring
abs_text = "word " * 500
c = agent._cut(abs_text, 100)
assert c in abs_text and len(c) <= 100

# _evidence: title, tldr, abstract, deduped snippets, max 8 passages
doc = {
    "title": "T1",
    "tldr": {"text": "TL"},
    "abstract": "AB " * 10,
    "_snippets": ["s1 unique", "s1 unique", "s2 other", "s3", "s4", "s5", "s6", "s7"],
}
ev = agent._evidence(doc)
parts = ev.split(" ... ")
assert parts[0] == "T1" and parts[1] == "TL"
assert len(parts) <= 8
assert ev.count("s1 unique") == 1  # dedupe

# _grade_chunk parsing (stub _gen)
criteria = [
    {"name": "a", "description": "A", "weight": 0.4},
    {"name": "b", "description": "B", "weight": 0.4},
    {"name": "c", "description": "C", "weight": 0.2},
]


async def _fake_gen(model, prompt, retries=1):
    return "0: 3 3 3\n1: 3, 1, 0\n2: 3 2 3\ngarbage\n3: 1"


agent._gen = _fake_gen
out = asyncio.get_event_loop().run_until_complete(
    agent._grade_chunk(criteria, [(0, "x"), (1, "y"), (2, "z"), (3, "w")])
)
assert out[0] == [3, 3, 3]
assert out[1] == [3, 1, 0]
assert out[2] == [3, 1, 3]  # 2 maps to 1
assert 3 not in out  # wrong arity dropped

# _weighted
assert abs(agent._weighted(criteria, [3, 3, 3]) - 1.0) < 1e-9
assert abs(agent._weighted(criteria, [3, 3, 1]) - (0.4 + 0.4 + 0.2 / 3)) < 1e-9
assert agent._weighted(criteria, [0, 0, 0]) == 0.0

# unparseable chunk defaults to partial
async def _fake_gen_empty(model, prompt, retries=1):
    return ""


agent._gen = _fake_gen_empty
out = asyncio.get_event_loop().run_until_complete(agent._grade_chunk(criteria, [(0, "x")]))
assert out == {0: [1, 1, 1]}

# _venue helpers
assert agent._venue_ok_substring("Neural Information Processing Systems", ["NeurIPS"])
assert agent._venue_ok_substring("Nature", ["Nature"])
assert not agent._venue_ok_substring("ICML", ["NeurIPS"])
assert agent._venue_str({"venue": "V", "journal": {"name": "J"}}) == "V | J"
assert agent._venue_str({}) == ""

# _author_ok
assert agent._author_ok({"authors": [{"name": "D. Harel"}]}, ["David Harel"])
assert not agent._author_ok({"authors": [{"name": "Jane Smith"}]}, ["David Harel"])
assert agent._author_ok({"authors": []}, [])

# _title_sim
assert agent._title_sim("Attention Is All You Need", "attention is all you need") > 0.95

# _parse_items unwraps {"data": [...]} and skips junk
class _CT:
    def __init__(self, text):
        self.text = text


items = [_CT('{"data": [{"corpusId": 1}, {"corpusId": 2}]}'), _CT('{"corpusId": 3}'), _CT("not json")]
docs = agent._parse_items(items)
assert [agent._cid(d) for d in docs] == ["1", "2", "3"]

print("ALL SMOKE TESTS PASSED")
