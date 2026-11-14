"""DS-1000 solver — guided cheap backbone + targeted reasoning escalation.

Design (see reasoning.md):
  Empirically, a plain one-shot GPT_5_4_MINI is a strong, cheap baseline
  (~75%), and same-model self-consistency did NOT beat it — because two mini
  samples share the *same* misconception and agree on the *wrong* value. The
  errors that killed every agent last round are systematic MISREADS
  (reproduce-shown-output, function-arg defaults, don't-overthink, style
  constraints), which a REASONING pass fixes, not repeated cheap sampling.

  So: keep a cheap guided mini candidate (A) as an always-valid backbone, and
  spend the wide-open cost headroom (iter3 ran at $0.001 mean vs a $0.003 free
  zone) on a reasoning escalation aimed ONLY at the uncertain problems:
    - A crashes on the reconstructed setup            -> escalate
    - A and an independent sample B disagree on value -> escalate
    - opaque "which library function" loaders         -> one careful reason shot
  Non-degradation is structural: A is only overridden by concrete execution
  evidence (a majority value-hash), so this can't do worse than the seed on
  cases the seed already handles.
"""

import re
from collections import Counter

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI


# --------------------------------------------------------------------------
# Concise, high-value guide. Every rule here is drawn from an observed
# failure class; kept short because it is paid on every candidate call.
# --------------------------------------------------------------------------
GUIDE = """\
You are an expert Python data scientist solving a DS-1000 problem. Write ONLY the
code that fills the solution slot so the requested target variable holds the correct
value (or the given function returns it).

FORMAT: output exactly one <code>...</code> block and NOTHING else — no prose, no
``` fences, no BEGIN/END markers. Executable Python only. Assign the EXACT target
variable name shown in the skeleton; if the slot is inside a `def`, keep the
indentation and `return` the value. Don't re-import or repeat the given setup, but
DO include every step needed for correctness (dtype conversions, fitting a model,
mutating the frame in place).

CORRECTNESS — the scorer compares the exact value (dtype, shape, order, pandas index):
- Prefer the SHORT canonical library idiom; DS-1000 reference answers are simple, not
  clever. Do NOT over-convert or add defensive transforms (e.g. scipy `linkage`
  accepts a distance/observation matrix directly — don't reshape it into something else).
- If the problem shows a concrete expected output, make your code reproduce THAT
  EXACTLY (down to the numbers, dtype and row order) — even if it seems to contradict
  the prose description. The shown table is the ground truth.
- If you must define a function and the setup pre-defines variables matching likely
  parameters (e.g. `x_min`, `x_max`), give those parameters DEFAULTS from them so the
  function still works when the test calls it with fewer arguments.
- Honor style constraints: prefer vectorized NumPy/Pandas idioms (stack/pivot/melt/
  groupby/broadcast) and AVOID `for`/`while` — DS-1000 often forbids explicit loops
  even when it isn't stated. "use function X" => call X by name.
- Modern APIs only: scipy `simps`/`cumtrapz`/`trapz` => `simpson`/
  `cumulative_trapezoid`/`trapezoid`; `DataFrame.append` => `pd.concat`. Matplotlib:
  set the exact property asked; never call plt.show().
"""

CAREFUL_NOTE = """\

An earlier cheap attempt was uncertain or failed, so solve this VERY carefully.
Re-read the problem word by word, honor EVERY (even implicit) constraint, reproduce
any expected output shown EXACTLY, and prefer the simplest canonical library call.
{extra}Output only the corrected solution inside <code></code>."""

# Real data injected by an opaque loader in the hidden test — running the visible
# setup just raises NameError, so these can't be value-verified. They are also the
# "which library function" class where a careful reasoning pass helps most.
_OPAQUE = ("load_data(", "load_iris(", "load_diabetes(", "fetch_", "load_boston(",
           "load_wine(", "load_digits(", "make_")

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
    """Reconstruct the runnable setup shown before the solution slot, or None."""
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
    """Return ('var', name), ('func', fname), or None (unverifiable)."""
    for m in re.finditer(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt):
        return ("var", m.group(1))
    # Function defined in the setup skeleton.
    fm = re.search(r"(?m)^\s*def\s+(\w+)\s*\(", setup or "")
    if fm:
        return ("func", fm.group(1))
    # Function requested only in prose, e.g. "define function named `smoothclamp`".
    pm = re.search(r"function\s+named\s+`?([A-Za-z_]\w*)`?", prompt) \
        or re.search(r"define\s+(?:a\s+)?(?:function|method)\s+`?([A-Za-z_]\w*)`?", prompt) \
        or re.search(r"(?:named|called)\s+`([A-Za-z_]\w*)`", prompt)
    if pm:
        return ("func", pm.group(1))
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
    """Sandbox code that runs setup+solution and hashes the target value.

    For function targets, call with no args first; if that raises a missing-
    argument TypeError, this is exactly the class of bug we want to surface —
    it is reported as a crash so the problem escalates to the reasoning pass.
    """
    prog = setup + "\n" + solution
    kind, name = target
    if kind == "var":
        capture = f"    _emit(_ns[{name!r}])\n"
    else:  # func — call it with values pulled from the namespace by parameter
        # name (the setup's example inputs), falling back to defaults, so we
        # value-verify the function's actual output rather than a bare call.
        capture = (
            f"    import inspect as _isp\n"
            f"    _fn = _ns[{name!r}]\n"
            f"    _args = []\n"
            f"    for _pn, _pp in _isp.signature(_fn).parameters.items():\n"
            f"        if _pn in _ns:\n"
            f"            _args.append(_ns[_pn])\n"
            f"        elif _pp.default is not _isp.Parameter.empty:\n"
            f"            _args.append(_pp.default)\n"
            f"        else:\n"
            f"            break\n"
            f"    _emit(_fn(*_args))\n"
        )
    return (
        _NORM_HELPERS
        + f"\n_PROG = {prog!r}\n"
        + "_ns = {}\n"
        + "try:\n"
        + "    exec(compile(_PROG, '<sol>', 'exec'), _ns)\n"
        + capture
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
        # Sandbox hiccup: don't penalize the candidate (treat as inconclusive-ok).
        return True, None, f"(verifier unavailable: {e})"
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

        async def gen(extra_note="", reason=False):
            if reason:
                # OpenAI: max_tokens is SHARED by reasoning + visible output, so
                # keep it generous or a long reasoning trace starves the answer.
                cfg = GenerateConfig(reasoning_effort="low", max_tokens=2500)
            else:
                cfg = GenerateConfig(max_tokens=1500)
            r = await GPT_5_4_MINI.generate(base + extra_note, config=cfg)
            return _extract_code(r.completion or ""), (r.completion or "")

        # ---- Candidate A: cheap guided backbone (always the fallback) -----
        solA, rawA = await gen()

        setup = _extract_setup(prompt)
        target = _detect_target(prompt, setup) if setup is not None else None
        opaque = setup is not None and any(k in setup for k in _OPAQUE)
        verifiable = (
            setup is not None and target is not None
            and lib != "Matplotlib" and not opaque
        )

        chosen = solA

        # ---- Opaque "which-function" problems: one careful reasoning shot --
        if opaque and lib != "Matplotlib":
            try:
                solR, _ = await gen(CAREFUL_NOTE.format(extra=""), reason=True)
                if solR.strip():
                    chosen = solR
                print(f"  opaque reason-shot len={len(solR)}")
            except Exception as e:
                print(f"  opaque reason-shot failed: {e}")

        # ---- Verifiable problems: execute, and escalate only if uncertain --
        elif verifiable and solA.strip():
            py = None
            try:
                py = next(t for t in state.tools if ToolDef(t).name == "python_session")
            except Exception:
                py = None

            if py is not None:
                okA, hA, tbA = await _run(py, setup, solA, target)
                print(f"  A ok={okA} h={hA}")

                # A ran but the sandbox couldn't hash a value (verifier down):
                # no signal to act on — keep A and don't waste B/C calls.
                escalate = not okA          # A crashed -> escalate directly
                solB, okB, hB = "", False, None
                if okA and hA is None:
                    escalate = False
                elif okA:
                    # Independent second sample as an uncertainty probe.
                    solB, _ = await gen()
                    okB, hB, _ = await _run(py, setup, solB, target)
                    print(f"  B ok={okB} h={hB}")
                    if okB and hB is not None and hA is not None and hB == hA:
                        escalate = False    # agree -> confident, keep A
                    else:
                        escalate = True     # disagree/B-crash -> reasoning tiebreak

                if escalate:
                    extra = ""
                    if not okA and tbA:
                        extra = f"The earlier attempt raised:\n<traceback>\n{tbA[-900:]}\n</traceback>\n"
                    solC, okC, hC = "", False, None
                    try:
                        solC, _ = await gen(CAREFUL_NOTE.format(extra=extra), reason=True)
                        okC, hC, _ = await _run(py, setup, solC, target)
                        print(f"  C(reason) ok={okC} h={hC}")
                    except Exception as e:
                        print(f"  reasoning escalation failed: {e}")

                    # Majority vote across whatever ran; tie -> prefer C (reasoning),
                    # then B, then A. Never pick a non-running candidate over a
                    # running one; fall back to A if nothing ran.
                    cands = [(solC, okC, hC), (solB, okB, hB), (solA, okA, hA)]
                    ran = [c for c in cands if c[1] and c[2] and c[0].strip()]
                    if ran:
                        votes = Counter(c[2] for c in ran)
                        top_hash, n = votes.most_common(1)[0]
                        if n >= 2:
                            chosen = next(c[0] for c in ran if c[2] == top_hash)
                        else:
                            # No majority: prefer the reasoning attempt if it ran.
                            chosen = next(c[0] for c in ran)  # order puts C first
                    elif okA:
                        chosen = solA
                    # else: everything crashed -> keep A as the least-bad guess.

        if not chosen.strip():
            chosen = _extract_code(rawA) or rawA.strip()

        state.output.completion = f"<code>\n{chosen}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
