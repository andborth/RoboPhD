"""One-off correctness gate for Ds1000Evaluator.

Builds two synthetic candidates against the same DS-1000 sample:
  A. Emits the reference solution → must score 1.0
  B. Emits `result = None` → must score 0.0

If either fails, the evaluator is broken — debug before running any
optimization. Runs in-process (subprocess_isolation=False) for clean
tracebacks.
"""

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from evaluator import Ds1000Evaluator, load_ds1000


# Pick a deterministic sample. validation[0] is what loaded earlier.
samples = load_ds1000("validation")
sample = samples[0]
sample_dict = sample.model_dump()

# Reference solution lives in sample.target as a markdown-fenced block.
# inspect_evals.ds1000.record_to_sample wraps it as ```python\n{code}\n```.
target_str = sample.target
match = re.search(r"```python\s*\n(.*?)\n```", target_str, re.DOTALL)
if not match:
    raise SystemExit(f"unable to extract reference from target: {target_str!r}")
reference_code = match.group(1)

print(f"Sample id: {sample.id}")
print(f"Library:   {sample.metadata.get('library')}")
print(f"Reference: {reference_code!r}")
print()


def make_agent_source(emitted_code: str) -> str:
    """Build an agent.py that always emits a fixed `<code>...</code>` body."""
    # Use repr() to safely embed arbitrary code as a Python string literal.
    return (
        "from inspect_ai.solver import solver\n"
        "\n"
        "@solver\n"
        "def make_solver():\n"
        "    async def solve(state, generate):\n"
        f"        state.output.completion = {emitted_code!r}\n"
        "        return state\n"
        "    return solve\n"
    )


agent_a = make_agent_source(f"<code>\n{reference_code}\n</code>")
agent_b = make_agent_source("<code>\nresult = None\n</code>")

# In-process evaluator for clean tracebacks.
evaluator = Ds1000Evaluator(subprocess_isolation=False)

print("== Candidate A (reference solution) ==")
score_a, diag_a = evaluator.evaluate({"agent.py": agent_a}, sample_dict)
print(f"score = {score_a}")
print(f"diagnostics keys: {sorted(diag_a.keys())}")
for k in ("error", "agent_stdout"):
    if k in diag_a:
        v = diag_a[k]
        print(f"  {k}: {v[:600] if isinstance(v, str) else v!r}")
if "test_result.md" in diag_a:
    print("test_result.md (first 400 chars):")
    print(diag_a["test_result.md"][:400])
print()

print("== Candidate B (result = None) ==")
score_b, diag_b = evaluator.evaluate({"agent.py": agent_b}, sample_dict)
print(f"score = {score_b}")
if "test_result.md" in diag_b:
    print("test_result.md (first 400 chars):")
    print(diag_b["test_result.md"][:400])
print()

print("== Candidate C (metadata introspection) ==")
# Verify the scrubbing wrapper hides code_context and perturbation_type
# but leaves library visible. Candidate emits a dummy answer and
# stashes the visible metadata keys into agent_stdout via print().
agent_c = (
    "from inspect_ai.solver import solver\n"
    "\n"
    "@solver\n"
    "def make_solver():\n"
    "    async def solve(state, generate):\n"
    "        keys = sorted(state.metadata.keys())\n"
    "        print(f'AGENT_VISIBLE_METADATA_KEYS={keys}')\n"
    "        state.output.completion = '<code>\\nresult = None\\n</code>'\n"
    "        return state\n"
    "    return solve\n"
)
score_c, diag_c = evaluator.evaluate({"agent.py": agent_c}, sample_dict)
visible_keys_line = ""
for line in (diag_c.get("agent_stdout") or "").splitlines():
    if "AGENT_VISIBLE_METADATA_KEYS=" in line:
        visible_keys_line = line.split("AGENT_VISIBLE_METADATA_KEYS=", 1)[1]
        break
print(f"agent saw metadata keys: {visible_keys_line}")
print()

print("== Verdict ==")
ok_a = score_a == 1.0
ok_b = score_b == 0.0
ok_c = "library" in visible_keys_line and "code_context" not in visible_keys_line and "perturbation_type" not in visible_keys_line
print(f"A (reference scores 1.0):                 {'PASS' if ok_a else 'FAIL'}  got {score_a}")
print(f"B (result=None scores 0.0):               {'PASS' if ok_b else 'FAIL'}  got {score_b}")
print(f"C (agent sees library, not code_context): {'PASS' if ok_c else 'FAIL'}  saw {visible_keys_line!r}")
sys.exit(0 if (ok_a and ok_b and ok_c) else 1)
