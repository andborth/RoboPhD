"""DS-1000 solver: generate -> universal self-verify (free sandbox) -> retry -> escalate.

Strategy (see reasoning.md):
  * Strong instruction prompt nudging vectorized, single-library-call idioms,
    explicit imports, exact dtype matching, and literal attribute->API mapping
    (especially for matplotlib: hatch vs marker vs linestyle vs color).
  * Parse the problem to find the setup `<code>` block, the target variable, AND
    any function-body `def f(...):` the solution must complete.
  * Verify EVERY runnable problem in the free `python_session`:
      - target = ... family  -> run setup+candidate, check the var is assigned.
      - def f(...): family    -> run setup+candidate AND call f() (when all params
                                 have defaults), so runtime errors / missing imports
                                 actually fire (this is the iter2 blind spot).
      - matplotlib / other    -> run setup+candidate, check it executes cleanly.
    On failure, feed the traceback back and retry, escalating the model.
  * Spend quality budget (stronger model + reasoning) on families the sandbox can
    only confirm "runs" but not "is correct" (matplotlib), and on non-runnable
    setups (undefined helpers) where verification is impossible.

Only `*.generate()` calls are metered; `python_session` verification is free, so
we verify heavily. Mean cost stays far inside the $0.08 free zone.
"""

import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI, GPT_5_4, CLAUDE_SONNET_4_6


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

_RULES = """You are an expert Python data-science engineer solving a DS-1000 problem.
Write ONLY the new Python code to append after the setup already shown, so the
requested variable / plot ends up correct.

Rules:
- Output a single ```python ... ``` code block containing ONLY the new solution
  code. Do not repeat imports/data already defined in the setup unless needed.
- Assign EXACTLY the variable name the problem asks for (often `result`). Match the
  expected TYPE and DTYPE, not just the values — the grader compares results
  exactly (e.g. a column of integers and the same values stored as strings differ).
- IMPORTS: only the modules imported in the setup are available in the test
  program. If your solution uses anything else (scipy, sklearn, itertools, math,
  ...), import it yourself inside the solution.
- Use the {library} library idiomatically. Prefer concise, vectorized, single
  library-call solutions. Avoid explicit Python for/while loops and
  list-comprehensions when a library call exists — graders sometimes reject manual
  element-by-element code as non-idiomatic. If the problem names a specific
  function, use that exact function.
- Read the request LITERALLY and map each described attribute to the exact API
  argument. For matplotlib especially, distinguish marker vs hatch vs linestyle vs
  color vs fill/edgecolor — e.g. "star hatch" means hatch='*' (NOT marker='*'),
  "dashed" means linestyle='--', etc.
- Write code robust to the input's actual shape/dtype (choose functions that also
  accept 1-D arrays); do not assume a shape the problem didn't state.
- The code must run without error. No prose, no explanation.

PROBLEM:
"""


def _build_prompt(state: TaskState) -> str:
    library = state.metadata.get("library", "the appropriate")
    return _RULES.format(library=library) + state.input


def _retry_prompt(base: str, prev_code: str, error: str) -> str:
    return (
        base
        + "\n\nYour previous attempt was:\n```python\n"
        + prev_code
        + "\n```\nBut running it produced this error:\n"
        + error[-1500:]
        + "\n\nReturn a corrected solution (single ```python block, code only)."
    )


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _extract_code(text: str) -> str:
    """Pull executable code out of a model reply, preserving indentation."""
    s = (text or "").strip()
    m = re.search(r"```(?:python|py)?[^\n]*\n(.*?)```", s, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"<code>\s*\n?(.*?)</code>", s, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    s = re.sub(r"^\s*(BEGIN|END)\s+SOLUTION\s*$", "", s, flags=re.MULTILINE)
    return s.strip("\n")


def _all_params_have_defaults(params: str) -> bool:
    """True if every positional parameter has a default (so func() is callable).
    Conservative: any ambiguity -> False (we then skip the auto-call)."""
    params = params.strip()
    if not params:
        return True
    # Split on top-level commas only (avoid breaking inside (), [], {}).
    depth = 0
    parts, cur = [], ""
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    for raw in parts:
        p = raw.strip()
        if not p or p.startswith("*"):  # empty, *args, **kwargs, bare *
            continue
        if "=" not in p:
            return False
    return True


# Markers that delimit the end of the setup code / start of instructions. Some
# prompts (e.g. function-body ones) have no real closing </code> for the setup —
# their only </code> is inside the literal "`<code>...</code>`" instruction text —
# so we must also cut on these prose boundaries.
_END_MARKERS = (
    "</code>",
    "Write the remaining python code",
    "Put your answer inside",
    "\nBEGIN SOLUTION",
    "# SOLUTION START",
    "\nSOLUTION START",
)


def _extract_setup(prompt: str) -> str:
    """Pull the runnable setup code out of the prompt, robust to missing closing
    </code> tags and to bare (no-tag) matplotlib 'SOLUTION START' prompts."""
    # A *real* setup tag opens as `<code>\n`; the literal "`<code>...</code>`" in
    # the instruction prose is inline, so only match an opener followed by EOL.
    m = re.search(r"<code>[ \t]*\n", prompt)
    rest = prompt[m.end():] if m else prompt
    cuts = [rest.find(m) for m in _END_MARKERS]
    cuts = [c for c in cuts if c != -1]
    end = min(cuts) if cuts else len(rest)
    return rest[:end].strip("\n")


def _parse_problem(prompt: str):
    """Return (setup_code, target_var, func_name).

    target_var: name from a top-level `X = ...` placeholder, else None.
    func_name:  name of a `def NAME(...):` the solution completes whose params all
                have defaults (so it can be auto-called), else None.
    """
    setup = _extract_setup(prompt)

    m = re.search(r"^([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt, re.MULTILINE)
    target = m.group(1) if m else None

    func_name = None
    if target is None:
        fm = re.search(r"^def\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*:",
                       setup, re.MULTILINE | re.DOTALL)
        if fm and _all_params_have_defaults(fm.group(2)):
            func_name = fm.group(1)
    return setup, target, func_name


# --------------------------------------------------------------------------- #
# Sandbox verification (free)
# --------------------------------------------------------------------------- #

def _get_py(state: TaskState):
    try:
        return next(t for t in state.tools if ToolDef(t).name == "python_session")
    except Exception:
        return None


def _looks_like_error(out: str) -> bool:
    return ("Traceback (most recent call last)" in out) or bool(
        re.search(r"^\w*(Error|Exception):", out or "", re.MULTILINE)
    )


async def _setup_runnable(py, setup: str) -> bool:
    """Can the setup execute on its own? (False when it uses undefined helpers
    like load_data())."""
    if not setup.strip():
        return False
    try:
        out = await py(code=setup + "\nprint('__SETUP_OK__')")
    except Exception:
        return False
    return ("__SETUP_OK__" in out) and not _looks_like_error(out)


async def _verify(py, setup: str, candidate: str, target, func_name):
    """Run setup+candidate fresh and return (ok, output).

    target    -> also assert the variable got assigned.
    func_name -> also call the function so its body actually executes.
    otherwise -> just confirm the program runs without a traceback.
    """
    pre = ""
    if target:
        pre = f"globals().pop({target!r}, None)\n"
    code = pre + setup + "\n" + candidate + "\n"
    if func_name and not target:
        code += f"__vr_call__ = {func_name}()\n"
    code += "print('__VERIFY_OK__')\n"
    if target:
        code += f"print('__TGT__', {target!r} in dir())\n"
    try:
        out = await py(code=code)
    except Exception as e:  # pragma: no cover - defensive
        return False, f"harness error: {e}"
    if _looks_like_error(out) or "__VERIFY_OK__" not in out:
        return False, out
    if target and "__TGT__ True" not in out:
        return False, out + "\n(NameError: target variable was not assigned)"
    return True, out


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

async def _gen(model, prompt: str, reasoning: str | None, max_tokens: int) -> str:
    cfg_kwargs = {"max_tokens": max_tokens}
    if reasoning:
        cfg_kwargs["reasoning_effort"] = reasoning
    try:
        resp = await model.generate(prompt, config=GenerateConfig(**cfg_kwargs))
        return _extract_code(resp.completion or "")
    except Exception as e:  # pragma: no cover - defensive
        print(f"  generate error: {e}")
        return ""


async def _tiered(py, base, setup, target, func_name, attempts):
    """Generate -> verify -> retry/escalate. Return best candidate found."""
    final = ""
    last_err = ""
    prev_code = ""
    for i, (model, reasoning, mt) in enumerate(attempts):
        p = base if i == 0 else _retry_prompt(base, prev_code, last_err)
        cand = await _gen(model, p, reasoning, mt)
        if not cand:
            continue
        final = cand  # fallback = most recent (error-informed) attempt
        prev_code = cand
        ok, out = await _verify(py, setup, cand, target, func_name)
        if ok:
            print(f"  attempt {i}: verified OK")
            return cand
        last_err = out
        print(f"  attempt {i}: failed verification")
    return final


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input
        library = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={library}")

        base = _build_prompt(state)
        setup, target, func_name = _parse_problem(prompt)
        py = _get_py(state)

        runnable = False
        if py is not None:
            runnable = await _setup_runnable(py, setup)
        # Does verification gate CORRECTNESS (var assigned / function actually
        # executed), or only "the program runs"?
        gates_correctness = runnable and (target is not None or func_name is not None)
        print(f"  target={target} func={func_name} runnable={runnable} "
              f"gates_correctness={gates_correctness}")

        final = ""

        if gates_correctness:
            # Cheap workhorse with error-informed retry, then a stronger escalate.
            final = await _tiered(py, base, setup, target, func_name, attempts=[
                (GPT_5_4_MINI, "low", 8192),
                (GPT_5_4_MINI, "low", 8192),
                (GPT_5_4, "low", 8192),
            ])
        elif runnable:
            # Matplotlib / no target: sandbox can only confirm it runs, not that
            # it's correct, so buy quality up front with a stronger reasoning model
            # and still verify it executes (catches bad kwargs / API misuse).
            final = await _tiered(py, base, setup, target, func_name, attempts=[
                (GPT_5_4, "medium", 4096),
                (GPT_5_4_MINI, "low", 8192),
            ])
        elif target is not None or func_name is not None:
            # Setup not runnable (undefined helpers): can't verify; spend on a
            # strong model up front.
            final = await _gen(GPT_5_4, base, "low", 8192)
            if not final:
                final = await _gen(GPT_5_4_MINI, base, "low", 8192)
        else:
            # Non-runnable, no target/func: strong single shot.
            final = await _gen(GPT_5_4, base, "medium", 4096)
            if not final:
                final = await _gen(GPT_5_4_MINI, base, "low", 8192)

        if not final.strip():
            final = "result = None"

        state.output.completion = f"<code>\n{final}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
