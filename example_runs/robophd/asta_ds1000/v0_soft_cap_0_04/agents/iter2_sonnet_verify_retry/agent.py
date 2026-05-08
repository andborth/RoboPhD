"""DS-1000 solver: Claude Sonnet 4.6 with python_session smoke verification.

Strategy:
1. Generate a candidate solution using Claude Sonnet 4.6 with a strong DS-1000
   system prompt covering output format, idiomatic library use, and variable
   naming conventions.
2. Smoke-test the candidate inside `python_session`: run the prompt's setup
   `<code>` block, append the candidate, and verify the target variable exists
   without raising.
3. If the smoke test fails, do one retry with the traceback as feedback.
4. Emit the final `<code>...</code>` block.
"""

from __future__ import annotations

import re
from typing import Optional

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6


SYSTEM_PROMPT = """You are an expert Python data-science programmer solving DS-1000 problems.

OUTPUT FORMAT (strict):
- Respond with EXACTLY one `<code>...</code>` block and NOTHING else.
- No prose, no markdown fences, no chain-of-thought, no `BEGIN SOLUTION` markers.
- The code inside the tags will be appended directly to the prompt's setup code, then run.

CODING RULES:
- Do NOT redeclare imports or variables that the prompt's setup `<code>` block already defines.
- Use the EXACT target variable name the prompt asks for (sometimes `result`, but often `weights`, `centered_scaled_data`, `transformed_df`, `slope`, etc.). Read the prompt carefully — the line `<name> = ... # put solution in this variable` tells you the name.
- Prefer idiomatic, vectorized library calls over manual loops/reimplementations. Many DS-1000 problems enforce this via hidden style tests, so a `for`-loop solution can fail even when output is correct.
  - NumPy: prefer broadcasting, fancy indexing, `np.where`, `np.unique`, `np.einsum`, etc.
  - Pandas: prefer `groupby`, `pivot`, `melt`, `apply`, vector ops.
  - Sklearn: prefer top-level functions like `preprocessing.scale`, `metrics.pairwise_distances` for 1D-friendly behavior; `StandardScaler` etc. require 2D inputs.
  - SciPy: use `scipy.stats`, `scipy.optimize`, `scipy.sparse` directly.
- Handle the obvious edge cases: 1D vs 2D arrays, NaNs, negative integers, empty groups, sorted vs unsorted.
- If the prompt shows an example with expected output, mentally trace your solution against it before answering.
- Keep solutions compact — typically 1–6 lines."""


_VAR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*\.\.\.", re.MULTILINE)
_SETUP_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$", re.MULTILINE)


def _detect_target_var(prompt: str) -> str:
    """Find the variable name the prompt asks the solver to populate.

    Looks for `<name> = ... # put solution in this variable`. Falls back to
    `result` if not found (the most common case).
    """
    m = _VAR_RE.search(prompt)
    if m:
        return m.group(1)
    return "result"


def _extract_setup_code(prompt: str) -> str:
    """Pull the first `<code>...</code>` block from the prompt.

    DS-1000 prompts have a setup block that defines `import` statements and
    input variables. We need this verbatim to run the smoke test.
    """
    matches = _SETUP_BLOCK_RE.findall(prompt)
    if not matches:
        return ""
    # First block is the setup. Later blocks (e.g. the empty one before
    # BEGIN SOLUTION) are not useful.
    return matches[0].strip()


def _extract_solution_code(text: str) -> str:
    """Strip wrappers and return just the executable Python.

    Handles: <code>...</code>, ```python ... ```, ```...```, raw code.
    """
    s = (text or "").strip()
    # Prefer the contents of <code> tags if present.
    m = re.search(r"<code>(.*?)</code>", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        # Strip markdown fences if present.
        s = _FENCE_RE.sub("", s).strip()
    return s.strip()


def _wrap(code: str) -> str:
    """Wrap a bare Python snippet in `<code>...</code>` for state.output."""
    return f"<code>\n{code}\n</code>"


async def _smoke_test(
    py_tool,
    setup: str,
    solution: str,
    target_var: str,
) -> Optional[str]:
    """Run setup + solution in the sandbox; return error string or None.

    Two-phase:
      Phase A: try to run setup alone. If it fails (e.g. uses `load_data()`
      from the test harness), we skip the smoke test entirely — we can't
      meaningfully verify the candidate without inputs.
      Phase B: run setup + solution together, check the target variable
      exists and the code didn't raise.

    Returns None on PASS or SKIP; returns a short error string on FAIL.
    """
    setup_program = (
        "import traceback as _tb\n"
        "try:\n"
        + _indent(setup or "pass") + "\n"
        "    print('SETUP_OK')\n"
        "except Exception:\n"
        "    print('SETUP_FAIL')\n"
        "    _tb.print_exc()\n"
    )
    try:
        setup_out = await py_tool(code=setup_program)
    except Exception:
        return None  # tool itself broke — skip
    if "SETUP_OK" not in str(setup_out):
        # Setup needs the test harness; we can't verify here. Skip.
        return None

    # Phase B — fresh sandbox cell that re-runs setup AND the candidate.
    program = (
        "import traceback as _tb\n"
        "try:\n"
        + _indent(setup or "pass") + "\n"
        + _indent(solution) + "\n"
        f"    _ok_target = {target_var!r} in dir() or {target_var!r} in globals()\n"
        "    if not _ok_target:\n"
        f"        print('SMOKE_FAIL: target variable {target_var} was not defined')\n"
        "    else:\n"
        f"        _val = eval({target_var!r})\n"
        "        print('SMOKE_OK:', type(_val).__name__)\n"
        "except Exception:\n"
        "    print('SMOKE_FAIL:')\n"
        "    _tb.print_exc()\n"
    )
    try:
        out = await py_tool(code=program)
    except Exception as e:
        return f"python_session call raised: {type(e).__name__}: {e}"
    out_str = str(out)
    if "SMOKE_OK:" in out_str:
        return None
    if "SMOKE_FAIL" in out_str:
        return out_str[-1500:]
    return None


def _indent(code: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in code.splitlines())


def _build_user_prompt(state_input: str) -> str:
    # Inline the system-style instructions at the top of the user prompt
    # for portability across model handles and Inspect AI versions.
    return SYSTEM_PROMPT + "\n\n---\n\n" + state_input


def _build_retry_prompt(state_input: str, prev_solution: str, error: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n\n---\n\n"
        + state_input
        + "\n\n---\n"
        + "Your previous attempt produced this code:\n"
        + "<code>\n"
        + prev_solution
        + "\n</code>\n\n"
        + "When this was appended to the setup code and run, it failed:\n"
        + "```\n"
        + error
        + "\n```\n\n"
        + "Fix the code. Output ONLY the corrected `<code>...</code>` block."
    )


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sample_id = state.sample_id
        library = state.metadata.get("library", "?")
        print(f"[{sample_id}] library={library}")

        target_var = _detect_target_var(state.input)
        setup = _extract_setup_code(state.input)
        print(f"[{sample_id}] target_var={target_var} setup_len={len(setup)}")

        # Locate the python_session tool (may not be registered in some envs).
        py_tool = None
        for t in state.tools:
            try:
                if ToolDef(t).name == "python_session":
                    py_tool = t
                    break
            except Exception:
                continue

        # Step 1: primary generation.
        config = GenerateConfig(temperature=0.0)
        try:
            resp = await CLAUDE_SONNET_4_6.generate(
                _build_user_prompt(state.input),
                config=config,
            )
            primary_text = resp.completion or ""
        except Exception as e:
            print(f"[{sample_id}] primary generate failed: {e}")
            primary_text = ""

        candidate = _extract_solution_code(primary_text)
        print(f"[{sample_id}] primary candidate len={len(candidate)}")

        # Step 2: smoke-test the candidate.
        if py_tool is not None and candidate:
            err = await _smoke_test(py_tool, setup, candidate, target_var)
            if err is not None:
                print(f"[{sample_id}] smoke-test failed; retrying with feedback")
                # Step 3: one retry with traceback feedback.
                try:
                    retry_resp = await CLAUDE_SONNET_4_6.generate(
                        _build_retry_prompt(state.input, candidate, err),
                        config=config,
                    )
                    retry_text = retry_resp.completion or ""
                    retry_candidate = _extract_solution_code(retry_text)
                    if retry_candidate:
                        # Verify retry; if it fails too, prefer the one that
                        # at least executed cleanly (otherwise keep retry —
                        # the model saw the error).
                        retry_err = await _smoke_test(
                            py_tool, setup, retry_candidate, target_var
                        )
                        if retry_err is None:
                            candidate = retry_candidate
                            print(f"[{sample_id}] retry passed smoke test")
                        elif err and not retry_err:
                            candidate = retry_candidate
                        else:
                            # Both failed — keep retry, since model saw error.
                            candidate = retry_candidate
                            print(f"[{sample_id}] retry still fails; keeping it anyway")
                except Exception as e:
                    print(f"[{sample_id}] retry failed: {e}")
            else:
                print(f"[{sample_id}] smoke test passed")
        else:
            print(f"[{sample_id}] no python_session or empty candidate; skipping smoke test")

        if not candidate:
            # Last-resort fallback so we always emit something.
            candidate = f"{target_var} = None"

        state.output.completion = _wrap(candidate)
        print(f"[{sample_id}] emitted {len(state.output.completion)} chars")
        return state

    return solve
