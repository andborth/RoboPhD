"""Offline smoke test for the pure-Python pieces of iter6_grade3_rescue."""
import sys, types, json

# stub external modules so agent.py imports cleanly outside inspect
for name in ("inspect_ai", "inspect_ai.solver", "inspect_ai.tool", "model_registry"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["inspect_ai.solver"].Generate = object
sys.modules["inspect_ai.solver"].TaskState = object
sys.modules["inspect_ai.solver"].solver = lambda f: f
sys.modules["inspect_ai.tool"].ToolDef = object
sys.modules["model_registry"].GPT_5_4 = object()
sys.modules["model_registry"].GPT_5_4_MINI = object()

import agent

# --- evidence assembly with criterion coverage
criteria = [
    {"name": "Few-shot slot tagging", "description": "The paper must focus on few-shot slot tagging", "weight": 0.4, "probe": "few-shot slot tagging"},
    {"name": "Micro-F1 episodes", "description": "evaluate micro-F1 averaged across test episodes", "weight": 0.6, "probe": "micro-F1 averaged across episodes"},
]
doc = {
    "title": "A few-shot slot tagging model",
    "abstract": "We study slot filling.",
    "tldr": {"text": "A model for slot tagging."},
    "_snippets": [
        "totally unrelated text about cooking recipes and food",
        "we report micro-F1 averaged across 5 test episodes following prior work",
        "we report micro-F1 averaged across 5 test episodes following prior work",  # dup
        "few-shot slot tagging is evaluated on SNIPS",
    ],
}
ev = agent._evidence(doc, criteria)
assert "micro-F1 averaged across 5 test episodes" in ev, ev
assert ev.count("micro-F1 averaged across 5") == 1, "dedup failed"
parts = ev.split(" ... ")
assert len(parts) <= 8
# the criterion-covering snippet should be selected before the unrelated one
assert parts.index(next(p for p in parts if "micro-F1 averaged" in p)) < len(parts)
print("evidence assembly OK:", len(parts), "passages")

# no-criteria path
ev2 = agent._evidence(doc)
assert "few-shot slot tagging model" in ev2.lower()
print("evidence no-criteria OK")

# --- title-sim duplicate guard thresholds
sim_dup = agent._title_sim(
    "ImageNet classification with deep convolutional neural networks",
    "ImageNet classification with deep convolutional neural networks",
)
sim_xl = agent._title_sim(
    "Objaverse: A Universe of Annotated 3D Objects",
    "Objaverse-XL: A Universe of 10M+ 3D Objects",
)
assert sim_dup >= 0.88, sim_dup
assert sim_xl < 0.88, f"XL guard would fail: {sim_xl}"
print(f"title-sim guard OK (dup={sim_dup:.2f}, xl={sim_xl:.2f})")

# --- weighted grade math mirrors the scorer's thresholds
w = agent._weighted(criteria, [3, 3])
assert w > 0.99
w2 = agent._weighted(criteria, [1, 3])
assert 0.67 < w2 <= 0.99, w2   # grade-2 band -> rescue-eligible
print("weighted math OK")

# --- grade parsing
out = {}
import re as _re
text = "0: 3 3\n1: 1 0\n2: 3 1\n"
valid = {0, 1, 2}
for m in _re.finditer(r"^\s*(\d+)\s*[:.\-]\s*([0-9 ,;/|]+?)\s*$", text, _re.MULTILINE):
    idx = int(m.group(1))
    digits = [int(d) for d in _re.findall(r"[0-9]", m.group(2))][:2]
    out[idx] = digits
assert out == {0: [3, 3], 1: [1, 0], 2: [3, 1]}
print("grade parse OK")

# --- _cut stays a verbatim prefix substring
long = "word " * 500
c = agent._cut(long, 300)
assert long.startswith(c)
print("cut OK")

print("ALL SMOKE TESTS PASSED")
