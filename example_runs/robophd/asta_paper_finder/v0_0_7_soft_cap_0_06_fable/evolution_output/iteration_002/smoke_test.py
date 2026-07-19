"""Smoke-test agent.py's pure-python helpers with model_registry/inspect_ai stubbed."""
import sys, types, json

# stub the runtime-only imports
for name, attrs in [
    ("inspect_ai", {}),
    ("inspect_ai.solver", {"Generate": object, "TaskState": object, "solver": lambda f: f}),
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
assert agent._json_block('blah {"candidates": [{"title": "T", "confidence": 0.9}]} end')["candidates"][0]["title"] == "T"
assert agent._json_block("no json here") is None

# _cid
assert agent._cid({"corpusId": 123}) == "123"
assert agent._cid({"corpusId": "CorpusId:456"}) == "456"
assert agent._cid({}) == ""

# _cut stays a substring
abs_text = "word " * 500
c = agent._cut(abs_text, 100)
assert c in abs_text and len(c) <= 100

# _evidence passages
doc = {"title": "T1", "tldr": {"text": "TL"}, "abstract": "AB " * 10, "_snippets": ["s1", "s2", "s3"]}
ev = agent._evidence(doc)
assert ev.startswith("T1 ... TL ... AB") and "s1" in ev and "s3" not in ev

# _venue_ok with alias
assert agent._venue_ok({"venue": "Neural Information Processing Systems"}, ["NeurIPS"])
assert agent._venue_ok({"venue": "", "journal": {"name": "Nature"}}, ["Nature"])
assert not agent._venue_ok({"venue": "ICML"}, ["NeurIPS"])
assert agent._venue_ok({"venue": "anything"}, [])

# _author_ok
assert agent._author_ok({"authors": [{"name": "D. Harel"}]}, ["David Harel"])
assert not agent._author_ok({"authors": [{"name": "Jane Smith"}]}, ["David Harel"])
assert agent._author_ok({"authors": []}, [])

# _title_sim
assert agent._title_sim("Attention Is All You Need", "attention is all you need") > 0.95

# _parse_items with data wrapper and plain
class CT:
    def __init__(self, text): self.text = text
items = [CT(json.dumps({"data": [{"corpusId": 1}]})), CT(json.dumps({"corpusId": 2})), CT(json.dumps({"data": []}))]
docs = agent._parse_items(items)
assert [agent._cid(d) for d in docs] == ["1", "2"]

# grade-chunk regex path
import re
text = "0: 3\n1: 0\n 2 - 2\nnoise\n3. 1"
grades = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"^\s*(\d+)\s*[:.\-]\s*([0-3])\b", text, re.MULTILINE)}
assert grades == {0: 3, 1: 0, 2: 2, 3: 1}

print("SMOKE_OK")
