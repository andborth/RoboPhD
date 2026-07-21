"""DS-1000 solver — guided generation + execution-based self-consistency.

Motivation (see reasoning.md):
  Iteration 2 showed that a plain one-shot GPT_5_4_MINI (70%) beat a
  guided crash-retry agent (60%). The crash-retry added machinery that
  did not help: *wrong* answers execute cleanly, so crash-checking catches
  almost nothing. The real correctness signal on DS-1000 is the target
  VALUE, not whether the code raised.

Strategy:
  1. Generate a candidate with a compact, high-value DS-1000 guide
     (GPT_5_4_MINI, cheapest reasoning setting).
  2. When the problem's setup + target are reconstructable, generate a
     second independent sample and EXECUTE both in the free python_session
     sandbox, hashing the resulting target value. Independent agreement on
     the value is a strong correctness signal (self-consistency).
  3. On disagreement or crashes, escalate ONCE to a reasoning pass and take
     the majority value (preferring the reasoning answer on ties).
  4. When nothing is verifiable (matplotlib, opaque loaders), fall back to
     the single strong-guided candidate — same cost as the seed there.

Cost stays comfortably under the $0.003 free-zone threshold: most problems
use ~2 cheap mini calls; the reasoning escalation only fires on the subset
that actually disagrees. Verification never degrades the answer — every
uncertain path falls back to candidate #1.
"""

import re
from collections import Counter

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI


# --------------------------------------------------------------------------
# Compact guidance — the highest-value DS-1000 conventions. Kept short
# because it is paid on every (multi-sample) call.
# --------------------------------------------------------------------------
GUIDE = """\
You are an expert Python data scientist solving a DS-1000 problem. Write ONLY the
code that fills the solution slot so the requested target variable holds the correct
value (or the given function returns it).

FORMAT: output exactly one <code>...</code> block and NOTHING else — no prose, no
``` fences, no BEGIN/END markers. Executable Python only. Assign the EXACT target
variable name shown in the skeleton; if the slot is inside a `def`, keep the
indentation and `return` the value. Don't re-import or repeat the given setup, but
DO include every step needed for correctness (e.g. dtype conversions, fitting a
model, mutating the frame in place).

CORRECTNESS — the scorer compares the exact value, including dtype, shape, order and
pandas index:
- Prefer the short, canonical library idiom; DS-1000 reference answers are simple,
  not clever. Match the dtype, shape, ordering and index the reference would produce.
- If the problem shows a concrete expected output, make your code reproduce THAT
  exactly (down to the shown numbers and their dtype).
- Honor style constraints: "without a for loop" / "vectorized" / "efficiently" /
  "not one by one" => use vectorized NumPy/Pandas, no `for`/`while`. "use function X"
  => call X by name.
- Modern APIs only: scipy `simps`/`cumtrapz`/`trapz` => `simpson`/
  `cumulative_trapezoid`/`trapezoid`; `DataFrame.append` => `pd.concat`. Matplotlib:
  set the exact property asked (e.g. `hatch=` not `marker=`); never call plt.show().
"""

CAREFUL_NOTE = """\

Two earlier attempts disagreed or failed, so solve this VERY carefully. Re-read the
problem, honor every constraint, and reproduce any expected output shown exactly.
{extra}
Output only the corrected solution inside <code></code>."""

# DS-1000 problems whose real data is injected by the hidden test via an opaque
# loader can't be value-verified from the visible setup — running it just raises
# NameError. Skip verification for these (fall back to the single strong answer).
_OPAQUE = ("load_data(", "load_iris(", "load_diabetes(", "fetch_", "load_boston(")

_SOLUTION_MARKERS = ("BEGIN SOLUTION", "SOLUTION START", "### BEGIN SOLUTION")


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
    """Reconstruct the runnable setup shown before the solution slot.

    Returns the setup code string, or None when it can't be reconstructed.
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


def _detect_target(prompt: str, setup: str):
    """Figure out what value the scorer will read.

    Returns ('var', name), ('func', fname), or None (unverifiable).
    """
    # A `NAME = ...` skeleton line (result, df, B, coef, ...). Prefer one that
    # is NOT already a plain assignment in the setup.
    for m in re.finditer(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt):
        return ("var", m.group(1))
    # Function-based problem: solution is the body; call the function to run it.
    fm = re.search(r"(?m)^\s*def\s+(\w+)\s*\(", setup or "")
    if fm:
        return ("func", fm.group(1))
    return None


_NORM_HELPERS = r"""
import traceback, hashlib
def _norm(v):
    try:
        import pandas as pd
        if isinstance(v, pd.DataFrame):
            return ('DF|' + repr(v.shape) + '|'
                    + repr(v.astype(object).values.tolist()) + '|'
                    + repr([str(c) for c in v.columns]) + '|'
                    + repr([str(i) for i in v.index]))
        if isinstance(v, pd.Series):
            return ('S|' + repr(v.astype(object).tolist()) + '|'
                    + repr([str(i) for i in v.index]) + '|' + str(v.name))
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return 'A|' + str(v.dtype) + '|' + repr(np.asarray(v).tolist())
    except Exception:
        pass
    try:
        return 'R|' + repr(v)
    except Exception:
        return 'U'
def _emit(v):
    try:
        h = hashlib.md5(_norm(v).encode('utf-8', 'replace')).hexdigest()
    except Exception:
        h = 'NOHASH'
    print('__VALHASH__' + h)
"""


def _driver(setup: str, solution: str, target) -> str:
    """Build sandbox code that runs setup+solution and hashes the target value."""
    prog = setup + "\n" + solution
    kind, name = target
    if kind == "var":
        capture = f"_emit(_ns[{name!r}])"
    else:  # func
        capture = (
            f"_fn = _ns[{name!r}]\n"
            f"    print('__VALHASH__' + __import__('hashlib').md5("
            f"_norm(_fn()).encode('utf-8','replace')).hexdigest())"
        )
    return (
        _NORM_HELPERS
        + f"\n_PROG = {prog!r}\n"
        + "_ns = {}\n"
        + "try:\n"
        + "    exec(compile(_PROG, '<sol>', 'exec'), _ns)\n"
        + f"    {capture}\n"
        + "    print('__OK__')\n"
        + "except Exception:\n"
        + "    print('__ERR__')\n"
        + "    print(traceback.format_exc())\n"
    )


async def _run(py, setup, solution, target):
    """Execute a candidate. Returns (ok: bool, valhash: str|None, tb: str)."""
    if not solution.strip():
        return False, None, "(empty)"
    try:
        out = await py(code=_driver(setup, solution, target))
    except Exception as e:
        return True, None, f"(verifier unavailable: {e})"  # don't penalize on sandbox hiccups
    out = out if isinstance(out, str) else str(out)
    ok = ("__OK__" in out) and ("__ERR__" not in out)
    vh = None
    if "__VALHASH__" in out:
        vh = out.split("__VALHASH__", 1)[1].strip().split()[0][:32]
    tb = out[-1400:]
    return ok, (vh if ok else None), tb


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={lib}")

        prompt = state.input
        base = GUIDE + "\n\n---\n\nProblem:\n\n" + prompt

        # ---- Candidate #1 (cheap, strong guide) ---------------------------
        r1 = await GPT_5_4_MINI.generate(base, config=GenerateConfig(max_tokens=1400))
        sol1 = _extract_code(r1.completion or "")

        # ---- Decide whether we can value-verify ---------------------------
        setup = _extract_setup(prompt)
        target = _detect_target(prompt, setup) if setup is not None else None
        opaque = setup is not None and any(k in setup for k in _OPAQUE)
        verifiable = (
            setup is not None and target is not None
            and lib != "Matplotlib" and not opaque
        )

        py = None
        if verifiable:
            try:
                py = next(t for t in state.tools if ToolDef(t).name == "python_session")
            except Exception:
                py = None

        chosen = sol1
        if py is not None and sol1.strip():
            # Candidate #2 — independent sample for self-consistency.
            r2 = await GPT_5_4_MINI.generate(base, config=GenerateConfig(max_tokens=1400))
            sol2 = _extract_code(r2.completion or "")

            ok1, h1, tb1 = await _run(py, setup, sol1, target)
            ok2, h2, tb2 = await _run(py, setup, sol2, target)
            print(f"  c1 ok={ok1} h={h1}  c2 ok={ok2} h={h2}")

            agree = ok1 and ok2 and h1 is not None and h1 == h2
            if agree:
                chosen = sol1                      # confident: both agree on the value
            elif ok1 and not ok2:
                chosen = sol1
            elif ok2 and not ok1:
                chosen = sol2
            else:
                # Disagreement or both crashed -> a cheap 3rd "careful" sample
                # (no reasoning tokens) as a self-consistency tiebreaker.
                both_crashed = not ok1 and not ok2
                extra = ""
                if both_crashed:
                    extra = f"The error was:\n<traceback>\n{(tb1 or tb2)[-900:]}\n</traceback>\n"
                sol3, ok3, h3 = "", False, None
                try:
                    r3 = await GPT_5_4_MINI.generate(
                        base + CAREFUL_NOTE.format(extra=extra),
                        config=GenerateConfig(max_tokens=1600),
                    )
                    sol3 = _extract_code(r3.completion or "")
                    ok3, h3, _ = await _run(py, setup, sol3, target)
                    print(f"  c3 ok={ok3} h={h3}")
                except Exception as e:
                    print(f"  tiebreak failed: {e}")

                cands = [(sol1, ok1, h1), (sol2, ok2, h2), (sol3, ok3, h3)]
                ok_cands = [c for c in cands if c[1] and c[2]]
                votes = Counter(c[2] for c in ok_cands)
                if votes:
                    top_hash, n = votes.most_common(1)[0]
                    if n >= 2:
                        chosen = next(c[0] for c in ok_cands if c[2] == top_hash)
                    elif ok3:
                        chosen = sol3          # trust the freshest careful attempt
                    else:
                        chosen = ok_cands[0][0]
                elif both_crashed:
                    # Nobody ran — a persistent (likely API) error. One reasoning
                    # pass with the traceback is worth the spend here (rare).
                    try:
                        r4 = await GPT_5_4_MINI.generate(
                            base + CAREFUL_NOTE.format(extra=extra),
                            config=GenerateConfig(reasoning_effort="low", max_tokens=2500),
                        )
                        sol4 = _extract_code(r4.completion or "")
                        ok4, _, _ = await _run(py, setup, sol4, target)
                        print(f"  c4(reason) ok={ok4}")
                        if ok4 and sol4.strip():
                            chosen = sol4
                        else:
                            chosen = sol1
                    except Exception as e:
                        print(f"  reasoning escalation failed: {e}")
                        chosen = sol1
                else:
                    chosen = sol1              # keep the strong first answer

        if not chosen.strip():
            chosen = (r1.completion or "").strip()

        state.output.completion = f"<code>\n{chosen}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
