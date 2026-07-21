"""DS-1000 solver — strong-prompt + execution-verified retry.

Strategy (see reasoning.md):
  1. Generate a solution with GPT_5_4_MINI using a DS-1000-aware guidance
     preamble (cheapest reasoning setting — stays deep in the free zone).
  2. When the visible setup code is reconstructable, run
     `setup + candidate` inside the free `python_session` sandbox to catch
     crashes (ImportErrors, NameErrors, wrong-API calls, …).
  3. Only on a detected crash, retry ONCE with reasoning + the traceback
     fed back. This concentrates the (small) extra spend on the problems
     that actually need it, keeping the batch mean well under $0.003.
"""

import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI


# --------------------------------------------------------------------------
# Guidance preamble — teaches the model the DS-1000 conventions that the
# hidden scorer (exact value + sometimes exact dtype/form) rewards.
# --------------------------------------------------------------------------
GUIDE = """\
You are an expert Python data scientist solving a DS-1000 problem. You are given a
problem with a code skeleton; write ONLY the Python code that goes in the solution
slot so that the requested target variable (e.g. `result`, `df`, `C`) holds the
correct value, or so the given function returns it.

Output format (STRICT):
- Emit exactly one `<code> ... </code>` block and NOTHING else — no prose, no
  markdown ``` fences, no `BEGIN/END SOLUTION` markers.
- Inside the tags: executable Python only. Do NOT repeat the setup code that is
  already given; write only the new code.
- Assign to the EXACT variable name shown in the skeleton. If the slot is inside a
  `def`, keep the indentation and `return` the value.

Correctness rules (the scorer compares exact values, and sometimes exact dtype/shape):
- Prefer the simplest, most idiomatic library call — DS-1000 reference answers are
  short and canonical, not clever. Match the shape, ordering, index, AND dtype the
  reference would naturally produce.
- Pandas: preserve the row index unless told otherwise; watch dtypes (a naive idiom
  may yield object/string or float columns — reproduce that). For "most frequent
  value per row" use `df.mode(axis=1)`.
- HONOR STYLE CONSTRAINTS. If the problem says "without a for loop", "not one by
  one", "efficiently", "vectorized", or mentions avoiding loops, you MUST NOT use
  `for` or `while` — use vectorized NumPy/Pandas (boolean masks, broadcasting,
  np.logical_and/np.where, groupby, etc.). If it asks to use a specific function,
  call that exact function by name.
- Read the request precisely. For matplotlib, set the EXACT property asked for:
  "hatch" means the `hatch=` kwarg (e.g. `plt.scatter(x, y, hatch='*')`), which is
  different from `marker=`. Do not call `plt.show()`.

Environment (modern package versions):
- scipy: `simps`/`cumtrapz`/`trapz` were removed — use `scipy.integrate.simpson`,
  `cumulative_trapezoid`, `trapezoid`.
- pandas: `DataFrame.append` was removed — use `pd.concat`.
- Use current, non-deprecated APIs throughout.
"""

RETRY_NOTE = """\

Your previous solution FAILED when executed — it raised this error:
<traceback>
{tb}
</traceback>
Diagnose and fix the cause (wrong/removed API, undefined name, wrong indentation,
etc.). Re-read the problem. Output only the corrected solution inside <code></code>.
"""

_SOLUTION_MARKERS = ("BEGIN SOLUTION", "SOLUTION START")


def _extract_code(text: str) -> str:
    """Pull clean Python out of the model's raw response."""
    s = (text or "").strip()
    if "<code>" in s:
        s = s.split("<code>", 1)[1]
        s = s.split("</code>", 1)[0]
    elif "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            block = parts[1]
            first, _, rest = block.partition("\n")
            if first.strip().lower() in ("python", "py", ""):
                block = rest
            s = block
    for m in ("### BEGIN SOLUTION", "### END SOLUTION", "BEGIN SOLUTION", "END SOLUTION"):
        s = s.replace(m, "")
    return s.strip("\n")


def _extract_setup(prompt: str):
    """Reconstruct the runnable setup shown in the prompt (before the solution slot).

    Returns the setup code string, or None when it can't be reconstructed safely
    (e.g. matplotlib prompts without a <code> block) — in which case we skip
    verification rather than risk a spurious error.
    """
    idx = -1
    for marker in _SOLUTION_MARKERS:
        i = prompt.find(marker)
        if i != -1:
            idx = i
            break
    pre = prompt if idx == -1 else prompt[:idx]
    lo = pre.rfind("<code>")
    if lo == -1:
        return None
    setup = pre[lo + len("<code>"):]
    c = setup.find("</code>")
    if c != -1:
        setup = setup[:c]
    return setup.rstrip()


async def _run_check(py, setup: str, solution: str):
    """Execute `setup + solution` in an isolated namespace inside the sandbox.

    Returns (ok: bool, traceback: str). ok=True means it ran without raising.
    """
    prog = setup + "\n" + solution
    # If the skeleton defines a function holding the solution, call it so the
    # body actually executes (surfacing errors inside it).
    m = re.search(r"^\s*def\s+(\w+)\s*\(", setup, re.M)
    if m and (m.group(1) + "(") not in solution:
        prog = prog + "\n" + m.group(1) + "()\n"

    driver = (
        "import traceback\n"
        f"_PROG = {prog!r}\n"
        "_ns = {}\n"
        "try:\n"
        "    exec(compile(_PROG, '<solution>', 'exec'), _ns)\n"
        "    print('__VERIFY_OK__')\n"
        "except Exception:\n"
        "    print('__VERIFY_ERR__')\n"
        "    print(traceback.format_exc())\n"
    )
    try:
        out = await py(code=driver)
    except Exception as e:  # sandbox hiccup — don't let it break the answer
        return True, f"(verifier unavailable: {e})"
    out = out if isinstance(out, str) else str(out)
    ok = ("__VERIFY_OK__" in out) and ("__VERIFY_ERR__" not in out)
    return ok, out


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={lib}")

        prompt = state.input
        base = GUIDE + "\n\n---\n\nProblem:\n\n" + prompt

        # ---- First pass: cheapest setting, strong guidance ----------------
        resp = await GPT_5_4_MINI.generate(base, config=GenerateConfig(max_tokens=2048))
        solution = _extract_code(resp.completion or "")

        # ---- Best-effort execution verification ---------------------------
        setup = _extract_setup(prompt)
        py = None
        if setup is not None:
            try:
                py = next(t for t in state.tools if ToolDef(t).name == "python_session")
            except (StopIteration, Exception):
                py = None

        if py is not None and setup is not None and solution.strip():
            ok, tb = await _run_check(py, setup, solution)
            print(f"  verify pass#1 ok={ok}")
            if not ok:
                # ---- Retry ONCE with reasoning + the traceback ------------
                retry_prompt = base + RETRY_NOTE.format(tb=tb[-1500:])
                try:
                    resp2 = await GPT_5_4_MINI.generate(
                        retry_prompt,
                        config=GenerateConfig(reasoning_effort="low", max_tokens=3000),
                    )
                    sol2 = _extract_code(resp2.completion or "")
                    if sol2.strip():
                        ok2, _ = await _run_check(py, setup, sol2)
                        print(f"  verify pass#2 ok={ok2}")
                        if ok2:
                            solution = sol2
                except Exception as e:
                    print(f"  retry failed: {e}")

        if not solution.strip():
            solution = (resp.completion or "").strip()

        state.output.completion = f"<code>\n{solution}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
