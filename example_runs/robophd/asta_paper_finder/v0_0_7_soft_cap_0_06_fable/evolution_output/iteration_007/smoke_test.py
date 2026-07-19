"""Offline smoke test for the pure-Python pieces of iter7_simview_breadth."""
import asyncio
import sys
import types

# stub external modules so agent.py imports cleanly outside inspect
for name in ("inspect_ai", "inspect_ai.solver", "inspect_ai.tool", "model_registry"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["inspect_ai.solver"].Generate = object
sys.modules["inspect_ai.solver"].TaskState = object
sys.modules["inspect_ai.solver"].solver = lambda f: f
sys.modules["inspect_ai.tool"].ToolDef = object
sys.modules["model_registry"].GPT_5_4 = object()
sys.modules["model_registry"].GPT_5_4_MINI = object()
sys.modules["model_registry"].GEMINI_3_1_FLASH_LITE = object()

import agent

criteria = [
    {"name": "Few-shot slot tagging", "description": "The paper must focus on few-shot slot tagging", "weight": 0.4, "probe": "few-shot slot tagging"},
    {"name": "Micro-F1 episodes", "description": "evaluate micro-F1 averaged across test episodes", "weight": 0.6, "probe": "micro-F1 averaged across episodes"},
]

# --- headline fix: the sim-view must ALWAYS include snippets, even when the
# abstract alone is longer than any prefix cut (iter6's failure mode)
doc = {
    "title": "A few-shot slot tagging model",
    "abstract": "We study slot filling in dialogue systems. " * 40,  # ~1700 chars
    "tldr": {"text": "A model for slot tagging."},
    "_snippets": [
        "totally unrelated text about cooking recipes and food",
        "we report micro-F1 averaged across 5 test episodes following prior work",
        "we report micro-F1 averaged across 5 test episodes following prior work",  # dup
        "few-shot slot tagging is evaluated on SNIPS",
    ],
}
sv = agent._sim_view(doc, criteria)
assert "micro-F1 averaged across 5 test episodes" in sv, "sim-view lost the criterion snippet"
assert sv.count("micro-F1 averaged across 5") == 1, "sim-view dedup failed"
assert "few-shot slot tagging is evaluated" in sv, "sim-view lost second criterion snippet"
assert len(sv) < 2200, f"sim-view too long: {len(sv)}"
# abstract must be trimmed to SV_ABS, not dominate
assert sv.index("micro-F1 averaged") < len(sv), sv
print(f"sim-view OK ({len(sv)} chars, snippets visible)")

sv_nocrit = agent._sim_view(doc)
assert "micro-F1 averaged" in sv_nocrit
print("sim-view no-criteria OK")

# --- submitted evidence still assembles with criterion coverage
ev = agent._evidence(doc, criteria)
parts = ev.split(" ... ")
assert len(parts) <= 8
assert any("micro-F1 averaged" in p for p in parts)
print("evidence assembly OK:", len(parts), "passages")

# --- duplicate-record similarity: containment-aware
lecun = "Gradient-based learning applied to document recognition"
lecun_junk = "PROC OF THE IEEE NOVEMBER Gradient Based Learning Applied to Document Recognition"
assert agent._dup_sim(lecun, lecun_junk) >= 0.96, agent._dup_sim(lecun, lecun_junk)
xl = agent._dup_sim(
    "Objaverse: A Universe of Annotated 3D Objects",
    "Objaverse-XL: A Universe of 10M+ 3D Objects",
)
assert xl < 0.88, f"XL guard would fail: {xl}"
alexnet = agent._dup_sim(
    "ImageNet classification with deep convolutional neural networks",
    "ImageNet Classification with Deep Convolutional Neural Networks",
)
assert alexnet >= 0.96, alexnet
short_sub = agent._dup_sim("Deep Learning", "Deep Learning for Computer Vision")
assert short_sub < 0.88, f"short-title containment leak: {short_sub}"
print(f"dup-sim OK (lecun_junk={agent._dup_sim(lecun, lecun_junk):.2f}, xl={xl:.2f}, short={short_sub:.2f})")

# --- weighted grade math mirrors the scorer's thresholds
assert abs(agent._weighted(criteria, [3, 3]) - 1.0) < 1e-9
assert agent._weighted(criteria, [3, 1]) < 0.99  # one weak criterion != perfect
print("weighted math OK")

# --- _grade_chunks: fallback on unparseable primary model
class FakeModel:
    def __init__(self, text):
        self.text = text
    async def generate(self, prompt):
        return types.SimpleNamespace(completion=self.text)

entries = [(0, "paper zero text"), (1, "paper one text"), (2, "paper two text")]

async def run_grades():
    bad = FakeModel("no grades here")
    good = FakeModel("0: 3 3\n1: 1 0\n2: 0 0")
    # primary fails -> fallback parses all
    out = await agent._grade_chunks(criteria, entries, 25, model=bad, fallback=good)
    assert out[0] == [3, 3] and out[1] == [1, 0] and out[2] == [0, 0], out
    # both fail -> default partial [1,1]
    out2 = await agent._grade_chunks(criteria, entries, 25, model=bad, fallback=bad)
    assert all(out2[i] == [1, 1] for i in range(3)), out2
    # primary parses -> no fallback needed
    out3 = await agent._grade_chunks(criteria, entries, 25, model=good, fallback=None)
    assert out3[0] == [3, 3], out3

asyncio.run(run_grades())
print("grade-chunks fallback OK")

# --- _cut keeps verbatim substrings
s = "alpha beta gamma delta epsilon"
c = agent._cut(s, 15)
assert s.startswith(c), (s, c)
print("cut OK")

print("ALL SMOKE TESTS PASSED")
