"""Validate iter12 extraction: unescape + reindent + widened assign-wrap. Fix-or-no-op."""
import importlib.util, sys

spec = importlib.util.spec_from_file_location("agent", "agent.py")
# Stub the inspect/model imports so agent.py loads standalone.
import types
for name, attrs in {
    "inspect_ai.model": ["GenerateConfig"],
    "inspect_ai.solver": ["Generate", "TaskState", "solver"],
    "model_registry": ["GPT_5_4", "GPT_5_4_MINI"],
}.items():
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, (lambda f: f) if a == "solver" else object())
    sys.modules[name] = mod
sys.modules["inspect_ai"] = types.ModuleType("inspect_ai")

spec = importlib.util.spec_from_file_location("agent", "agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

E = agent._extract_code
H = agent._has_top_level_assign

# Prompts
TOP = "a = np.array([1,2,3])\nresult = ...  # put solution in this variable\nBEGIN SOLUTION\n<code>\n[insert]\n</code>"
FUNC = "def f(x):\n    ### BEGIN SOLUTION\n    [insert]\n    ### END SOLUTION\n    return result"
ARR_TARGET = "arr = ...  # solution\nBEGIN SOLUTION"

passed = failed = 0
def check(name, got, want):
    global passed, failed
    ok = got == want
    passed += ok; failed += not ok
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        print("   got : " + repr(got))
        print("   want: " + repr(want))

# --- _has_top_level_assign unit ---
assert H("x = 5") is True
assert H("np.array([1,2], dtype=int)") is False          # kwarg only
assert H("pd.Series(d, index=idx)") is False
assert H("a == b") is False
assert H("a <= b") is False
assert H("(x := 5)") is False                             # walrus in expr
assert H("d = {1: 2, 3: 4}") is True                      # top-level assign, dict RHS
assert H("f('a=b')") is False                             # = inside string
assert H("a[i] = 5") is True                              # subscript target assign
assert H("result") is False
print("has_top_level_assign unit: OK")

# 1. unescape: &lt; must become <
check("unescape", E("<code>\nresult = a[a &lt; 3]\n</code>", TOP),
      "<code>\nresult = a[a < 3]\n</code>")

# 2. function-body reindent: unindented return -> 4-space
check("func reindent", E("<code>\nreturn x + 1\n</code>", FUNC),
      "<code>\n    return x + 1\n</code>")

# 3. function body already indented -> untouched
check("func keep indent", E("<code>\n    return x + 1\n</code>", FUNC),
      "<code>\n    return x + 1\n</code>")

# 4. assign-wrap: bare expr, no `=` at all (451 class)
check("wrap bare 451", E("<code>\nnp.zeros((20,10,10,2))\n</code>", ARR_TARGET),
      "<code>\narr = np.zeros((20,10,10,2))\n</code>")

# 5. NEW: bare expr WITH kwarg -> now wraps (iter11 would have MISSED this)
check("wrap kwarg expr", E("<code>\nnp.array([1, 2], dtype=int)\n</code>", TOP),
      "<code>\nresult = np.array([1, 2], dtype=int)\n</code>")

# 6. NEW: bare expr with dict literal kwarg
check("wrap dict kwarg", E("<code>\npd.Series(data, index=idx)\n</code>", TOP),
      "<code>\nresult = pd.Series(data, index=idx)\n</code>")

# 7. already assigned -> untouched
check("no-op assigned", E("<code>\nresult = a[a != 0]\n</code>", TOP),
      "<code>\nresult = a[a != 0]\n</code>")

# 8. assignment with kwarg RHS -> untouched (has top-level =)
check("no-op assigned kwarg", E("<code>\nresult = np.array([1], dtype=int)\n</code>", TOP),
      "<code>\nresult = np.array([1], dtype=int)\n</code>")

# 9. multi-line answer -> not wrapped
check("no-op multiline", E("<code>\nx = a + 1\nresult = x * 2\n</code>", TOP),
      "<code>\nx = a + 1\nresult = x * 2\n</code>")

# 10. statement (for-loop) -> not wrapped
check("no-op statement", E("<code>\nfor i in range(3): pass\n</code>", TOP),
      "<code>\nfor i in range(3): pass\n</code>")

# 11. function-style problem never gets assign-wrap (no placeholder)
check("func no wrap", E("<code>\n    x + 1\n</code>", FUNC),
      "<code>\n    x + 1\n</code>")

# 12. markdown fence stripping
check("fence strip", E("```python\nresult = a.sum()\n```", TOP),
      "<code>\nresult = a.sum()\n</code>")

# 13. comparison expr as bare answer wraps correctly (== not treated as assign)
check("wrap comparison", E("<code>\n(a == b)\n</code>", TOP),
      "<code>\nresult = (a == b)\n</code>")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
