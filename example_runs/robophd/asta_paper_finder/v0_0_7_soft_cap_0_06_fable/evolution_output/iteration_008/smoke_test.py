"""Offline smoke test for iter8_resilient_lean (stubbed inspect_ai/model_registry)."""
import asyncio
import sys
import time
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

# speed up retry sleeps for the test
agent._RETRY_DELAYS = (0.01, 0.02)

criteria = [
    {"name": "Few-shot slot tagging", "description": "The paper must focus on few-shot slot tagging", "weight": 0.4, "probe": "few-shot slot tagging"},
    {"name": "Micro-F1 episodes", "description": "evaluate micro-F1 averaged across test episodes", "weight": 0.6, "probe": "micro-F1 averaged across episodes"},
]

# --- retrying _safe_tool: factory called fresh per attempt, recovers from 502s
calls = {"n": 0}

async def _flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise RuntimeError("tool call failed: HTTP 502")
    return "recovered"

async def run_safe_tool():
    out = await agent._safe_tool(lambda: _flaky(), "flaky", attempts=3)
    assert out == "recovered", out
    assert calls["n"] == 3, calls

    # permanent failure -> None, no raise
    async def _dead():
        raise RuntimeError("tool call failed: HTTP 502")
    out2 = await agent._safe_tool(lambda: _dead(), "dead", attempts=2)
    assert out2 is None

    # bare coroutine still works one-shot (legacy form)
    async def _ok():
        return 42
    out3 = await agent._safe_tool(_ok(), "bare")
    assert out3 == 42

    # timeout path with factory retry
    async def _slow_then_fast():
        calls["n"] += 1
        if calls["n"] < 5:
            await asyncio.sleep(1.0)
        return "fast"
    calls["n"] = 3
    out4 = await agent._safe_tool(lambda: _slow_then_fast(), "slow", timeout=0.05, attempts=3)
    assert out4 == "fast", out4

asyncio.run(run_safe_tool())
print("retrying _safe_tool OK")

# --- sim-view must ALWAYS include snippets (iter7's headline fix, retained)
doc = {
    "title": "A few-shot slot tagging model",
    "abstract": "We study slot filling in dialogue systems. " * 40,
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
assert len(sv) < 2000, f"sim-view too long: {len(sv)}"
print(f"sim-view OK ({len(sv)} chars, snippets visible)")

ev = agent._evidence(doc, criteria)
parts = ev.split(" ... ")
assert len(parts) <= 8 and any("micro-F1 averaged" in p for p in parts)
print("evidence assembly OK:", len(parts), "passages")

# --- duplicate-record similarity: containment-aware guard retained
lecun = "Gradient-based learning applied to document recognition"
lecun_junk = "PROC OF THE IEEE NOVEMBER Gradient Based Learning Applied to Document Recognition"
assert agent._dup_sim(lecun, lecun_junk) >= 0.96
xl = agent._dup_sim(
    "Objaverse: A Universe of Annotated 3D Objects",
    "Objaverse-XL: A Universe of 10M+ 3D Objects",
)
assert xl < 0.88, xl
assert agent._dup_sim("Deep Learning", "Deep Learning for Computer Vision") < 0.88
print("dup-sim OK")

# --- weighted grade math mirrors the scorer's thresholds
assert abs(agent._weighted(criteria, [3, 3]) - 1.0) < 1e-9
assert agent._weighted(criteria, [3, 1]) < 0.99
print("weighted math OK")

# --- _grade_chunks fallback behavior retained
class FakeModel:
    def __init__(self, text):
        self.text = text
    async def generate(self, prompt):
        return types.SimpleNamespace(completion=self.text)

entries = [(0, "paper zero text"), (1, "paper one text"), (2, "paper two text")]

async def run_grades():
    bad = FakeModel("no grades here")
    good = FakeModel("0: 3 3\n1: 1 0\n2: 0 0")
    out = await agent._grade_chunks(criteria, entries, 25, model=bad, fallback=good)
    assert out[0] == [3, 3] and out[1] == [1, 0] and out[2] == [0, 0], out
    out2 = await agent._grade_chunks(criteria, entries, 25, model=bad, fallback=bad)
    assert all(out2[i] == [1, 1] for i in range(3)), out2

asyncio.run(run_grades())
print("grade-chunks fallback OK")

# --- ambiguity gate logic (mirrors _solve_specific's inline expression)
def gate(name, author_hints, year_hint):
    return (
        not author_hints
        and year_hint is None
        and 0 < len(agent._norm(name).split()) <= 2
    )

assert gate("SPIKE", [], None) is True                     # "the SPIKE paper" -> hedge
assert gate("BART", ["Lewis"], None) is False              # author cue -> exact path
assert gate("MS2", ["DeYoung"], 2021) is False             # citation key -> exact path
assert gate("Deep Residual Learning Networks", [], None) is False  # long alias -> exact
print("ambiguity gate OK")

# --- _cut keeps verbatim substrings
s = "alpha beta gamma delta epsilon"
assert s.startswith(agent._cut(s, 15))
print("cut OK")

# --- constants sanity (cost trim actually applied)
assert agent.POOL_CAP == 440 and agent.HEAD == 110 and agent.RESCUE_MAX == 28
assert agent.T1_TITLE == 90 and agent.T1_BODY == 160 and agent.NARROW_TOP == 20
print("constants OK")

print("ALL SMOKE TESTS PASSED")
