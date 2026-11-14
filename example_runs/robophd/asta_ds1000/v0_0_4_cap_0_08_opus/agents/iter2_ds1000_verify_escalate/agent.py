"""DS-1000 solver: generate -> self-verify (free sandbox) -> retry -> escalate.

Strategy (see reasoning.md):
  * A strong instruction prompt that nudges toward vectorized, single library
    call idioms (no manual loops where avoidable) and shape-robust code.
  * Parse the problem to find the setup `<code>` block and the target variable.
  * For the common top-level-assignment family, RUN `setup + candidate` in the
    free `python_session`. If it raises, feed the traceback back and retry the
    same model, then escalate to a different strong model. Stop on first clean
    run.
  * Fall back gracefully for function-body / matplotlib / non-runnable-setup
    problems.

Only `*.generate()` calls are metered; `python_session` verification is free,
so we lean on it heavily. Costs stay far inside the $0.08 free zone.
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
Write ONLY the new Python code to append after the setup already shown, so that
the requested variable ends up holding the correct value.

Rules:
- Output a single ```python ... ``` code block containing ONLY the new solution
  code. Do not repeat imports/data already defined in the setup unless needed.
- Assign EXACTLY the variable name the problem asks for (e.g. result / df / X).
- Use the {library} library idiomatically. Prefer concise, vectorized, single
  library-call solutions. If the task can be done without explicit Python
  for/while loops or list-comprehensions, do that — graders sometimes reject
  manual element-by-element loops as non-idiomatic.
- Write code robust to the input's actual shape/dtype (e.g. choose functions
  that also accept 1-D arrays); do not assume a shape the problem didn't state.
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
        + "\n\nReturn corrected solution code only (single ```python block)."
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
    # Strip stray solution markers if present.
    s = re.sub(r"^\s*(BEGIN|END)\s+SOLUTION\s*$", "", s, flags=re.MULTILINE)
    return s.strip("\n")


def _parse_problem(prompt: str):
    """Return (setup_code, target_var). target_var is None when there is no
    top-level `X = ...` placeholder (function-body / matplotlib families)."""
    blocks = re.findall(r"<code>(.*?)</code>", prompt, re.DOTALL)
    setup = blocks[0].strip("\n") if blocks else ""
    m = re.search(r"^([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt, re.MULTILINE)
    target = m.group(1) if m else None
    return setup, target


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


async def _verify(py, setup: str, candidate: str, target):
    """Run setup+candidate fresh. Return (ok, output)."""
    pre = ""
    if target:
        pre = f"globals().pop({target!r}, None)\n"
    code = pre + setup + "\n" + candidate + "\nprint('__VERIFY_OK__')"
    if target:
        code += f"\nprint('__TGT__', {target!r} in dir())"
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


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input
        library = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={library}")

        base = _build_prompt(state)
        setup, target = _parse_problem(prompt)
        py = _get_py(state)

        verifiable = False
        if py is not None and target is not None:
            verifiable = await _setup_runnable(py, setup)
        print(f"  target={target} verifiable={verifiable}")

        final = ""

        if verifiable:
            # Tiered: mini -> mini(retry) -> Claude(escalate), verifying each.
            attempts = [
                (GPT_5_4_MINI, "low", 8192),
                (GPT_5_4_MINI, "low", 8192),
                (CLAUDE_SONNET_4_6, "low", 4096),
            ]
            last_err = ""
            prev_code = ""
            for i, (model, reasoning, mt) in enumerate(attempts):
                p = base if i == 0 else _retry_prompt(base, prev_code, last_err)
                cand = await _gen(model, p, reasoning, mt)
                if not cand:
                    continue
                final = cand  # fallback = most recent (error-informed) attempt
                prev_code = cand
                ok, out = await _verify(py, setup, cand, target)
                if ok:
                    print(f"  attempt {i}: verified OK")
                    final = cand
                    break
                last_err = out
                print(f"  attempt {i}: failed verification")
        elif target is not None:
            # Family 1 but setup not runnable (undefined helpers): can't verify,
            # so spend quality budget up front with a stronger model.
            final = await _gen(GPT_5_4, base, "low", 8192)
            if not final:
                final = await _gen(GPT_5_4_MINI, base, "low", 8192)
        else:
            # Function-body / matplotlib: no target var to verify against.
            final = await _gen(GPT_5_4_MINI, base, "low", 8192)
            if not final:
                final = await _gen(GPT_5_4_MINI, base, None, 4096)

        if not final.strip():
            final = "result = None"

        state.output.completion = f"<code>\n{final}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
