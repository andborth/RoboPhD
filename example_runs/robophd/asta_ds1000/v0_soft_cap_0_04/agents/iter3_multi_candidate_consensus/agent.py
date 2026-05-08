"""DS-1000 solver: multi-candidate generation with smoke-test consensus.

Strategy:
1. Generate 3 candidates in parallel from diverse model/temperature settings
   (Claude Sonnet 4.6 @ 0, Claude Sonnet 4.6 @ 0.6, GPT-5.4-mini @ 0).
2. Smoke-test each candidate inside `python_session`. The smoke program
   `exec()`s the candidate so SyntaxError/IndentationError are catchable;
   for function-body problems it re-indents the candidate, closes the open
   `def`, and (when args have defaults) calls it to capture the return value.
3. Three-tier smoke status: OK (parsed and ran cleanly, with REPR captured),
   PARSE_OK (parsed but call failed or args unknown), FAIL (parse error).
4. Among OK candidates, prefer a value-consensus pick (≥2 candidates'
   computed REPR matches). Otherwise fall back to preference order:
   Sonnet@0 → Sonnet@0.6 → Mini@0.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, Tuple

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6, GPT_5_4_MINI


SYSTEM_PROMPT = """You are an expert Python data-science programmer solving DS-1000 problems.

OUTPUT FORMAT (strict):
- Respond with EXACTLY one `<code>...</code>` block and NOTHING else.
- No prose, no markdown fences, no chain-of-thought, no `BEGIN SOLUTION` markers, no `### END SOLUTION`.
- The code inside the tags is appended directly to the prompt's setup code, then run.

TWO PROBLEM SHAPES — read the prompt carefully to tell which:

(A) Module-level completion (most common). The prompt ends with something like
    `result = ... # put solution in this variable` followed by `BEGIN SOLUTION`
    and an empty `<code>` block. You must write module-level statements that
    define the target variable. Example: `result = a[a != 0]`.

(B) Function-body completion. The prompt's last `<code>` block defines a
    function whose body ends with `### BEGIN SOLUTION` and is otherwise empty:
        def f(A=example_a, B=example_b):
            # return the solution in this function
            # result = f(A, B)
            ### BEGIN SOLUTION
    Your code goes INSIDE THE FUNCTION BODY. EVERY line must start with at
    least 4 spaces of indentation. Do NOT redeclare `def f(...)`. Do NOT
    write module-level statements after the function. End with `return ...`.
    Example for the f(A,B) case:
        <code>
            return tf.reduce_sum(tf.square(A - B), axis=1)
        </code>

CODING RULES:
- Use the EXACT target variable name the prompt asks for. The line
  `<name> = ... # put solution in this variable` tells you the name. It is
  often `result` but sometimes `weights`, `transformed_df`, `cluster_labels`,
  `b`, `c`, `slope`, `df_out`, etc.
- Do NOT redeclare imports or variables that the setup `<code>` block defines
  (e.g. don't `import numpy as np` if setup already did).
- DO NOT hard-code shapes/sizes/values from the example. Derive them from the
  inputs: `n = a.shape[0]`, `len(B)`, `df.columns.nlevels`, etc. The hidden
  test feeds DIFFERENT inputs of different sizes.
- Style constraints: when the prompt says "without a loop", "without a for",
  "vectorized", "the efficient way", "not one by one", or implies idiomatic
  library use, DO NOT use `for` or `while`. Use `np.where`, boolean masks,
  `np.searchsorted`, `np.einsum`, `groupby`, `apply`, etc. instead.
- Function definitions: when the problem says "define function named foo as
  solution", the test will call your function with the SAME signature shown
  in the prompt's preamble. If the preamble references `x_min`/`x_max` as
  globals, your function takes only the variable args; don't add the globals
  as parameters.
- Prefer idiomatic, vectorized library calls over manual reimplementations.
  Some problems enforce this via hidden style tests (a `for`-loop solution
  fails even when output matches).
- Match the requested return type / container exactly: list vs. tuple vs.
  ndarray vs. Series vs. DataFrame. Convert at the end if needed
  (`.tolist()`, `.to_numpy()`, `.reshape(-1)`).
- Sklearn: prefer top-level functions like `preprocessing.scale`,
  `metrics.pairwise_distances` for 1D-friendly behavior; `StandardScaler`
  requires 2D inputs.
- Pandas: prefer `groupby`, `pivot`, `melt`, `apply`, vector ops.
- SciPy: use `scipy.stats`, `scipy.optimize`, `scipy.sparse`,
  `scipy.cluster.hierarchy.linkage` (handles raw distance matrices) directly.
- TensorFlow / PyTorch: respect tensor dtype; convert inputs if scoring uses
  float comparisons.
- Keep solutions compact — typically 1–6 lines."""


# ---- prompt parsing ---------------------------------------------------------

_VAR_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*\.\.\.\s*(?:#\s*put solution.*)?$",
    re.MULTILINE,
)
_RETURN_VAR_RE = re.compile(
    r"#\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*[A-Za-z_][A-Za-z_0-9]*\s*\(",
)
_CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
# Robust: stop at the FIRST of `</code>`, `\nWrite the remaining` (the prompt's
# trailing instructions), `\nBEGIN SOLUTION` (a marker line outside `<code>`),
# or end-of-string. Function-body problems leave the setup `<code>` unclosed
# and the strict regex would otherwise match through to a `</code>` in the
# explanatory prose.
_CODE_BLOCK_LOOSE_RE = re.compile(
    r"<code>(.*?)(?=</code>|\nWrite the remaining|\nBEGIN SOLUTION|\Z)",
    re.DOTALL,
)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$", re.MULTILINE)
_DEF_LINE_RE = re.compile(
    r"^(\s*)def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\((.*?)\)\s*:",
    re.DOTALL | re.MULTILINE,
)
_BEGIN_SOL_RE = re.compile(r"###\s*BEGIN SOLUTION\s*$", re.MULTILINE)
# Match the entire BEGIN SOLUTION line including leading whitespace and the
# trailing newline, so substitution replaces the whole line cleanly.
_BEGIN_SOL_LINE_RE = re.compile(r"^[ \t]*###\s*BEGIN SOLUTION[ \t]*\n?", re.MULTILINE)


def _detect_target_var(prompt: str) -> str:
    """Find the variable name the prompt asks the solver to populate."""
    m = _VAR_RE.search(prompt)
    if m:
        return m.group(1)
    # Function-body case: look for `# result = f(...)`
    for code in _CODE_BLOCK_LOOSE_RE.findall(prompt):
        m2 = _RETURN_VAR_RE.search(code)
        if m2:
            return m2.group(1)
    # Sometimes the prompt says "put score in `b`, put prediction in `c`"
    if "put score in" in prompt or "put prediction in" in prompt:
        # Match the last variable mentioned that way; default to result.
        m3 = re.findall(r"put\s+\w+\s+in\s+`([A-Za-z_][A-Za-z_0-9]*)`", prompt)
        if m3:
            return m3[0]
    return "result"


def _extract_setup_code(prompt: str) -> str:
    """First `<code>` block from the prompt (setup imports/values).

    Uses the loose regex which stops at the first of `</code>`, the prompt's
    trailing instructions, or end-of-string — handling both well-formed
    blocks and unclosed function-body setups uniformly.
    """
    matches = _CODE_BLOCK_LOOSE_RE.findall(prompt)
    if matches:
        return matches[0].strip()
    return ""


def _detect_function_body(prompt: str) -> Optional[Tuple[str, str, str]]:
    """If the prompt's setup defines a function whose body is open at
    `### BEGIN SOLUTION`, return (func_name, signature, indent_str).

    Otherwise None.
    """
    blocks = _CODE_BLOCK_LOOSE_RE.findall(prompt)
    for block in blocks:
        # Look for `def NAME(...):` followed by `### BEGIN SOLUTION` somewhere
        # later, with the body before BEGIN SOLUTION being only comments/empty
        # lines (i.e. the function is unfilled).
        if not _BEGIN_SOL_RE.search(block):
            continue
        m = _DEF_LINE_RE.search(block)
        if not m:
            continue
        # Find indent of body — one level deeper than `def`
        def_indent = m.group(1) or ""
        body_indent = def_indent + "    "
        return (m.group(2), m.group(3), body_indent)
    return None


def _extract_solution_code(text: str) -> str:
    """Pull executable Python from the model's response."""
    s = (text or "").strip()
    m = re.search(r"<code>(.*?)</code>", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        s = _FENCE_RE.sub("", s).strip()
    # Strip stray BEGIN/END SOLUTION markers
    s = re.sub(r"^\s*###?\s*(BEGIN|END)\s*SOLUTION\s*$", "", s, flags=re.MULTILINE)
    return s.strip("\n")


def _wrap(code: str) -> str:
    return f"<code>\n{code}\n</code>"


def _indent(code: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line.strip() else line for line in code.splitlines())


def _reindent_to(code: str, target_indent: str) -> str:
    """Re-indent a code block to the target indentation level.

    If the model returned indented body code, normalize the leading common
    indent to zero, then add target_indent.
    """
    lines = code.splitlines()
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank:
        return code
    # Common leading whitespace
    common = min((len(ln) - len(ln.lstrip(" "))) for ln in nonblank)
    stripped = "\n".join(ln[common:] if ln.strip() else "" for ln in lines)
    return "\n".join(target_indent + ln if ln.strip() else "" for ln in stripped.splitlines())


# ---- prompt builders --------------------------------------------------------


def _build_user_prompt(state_input: str) -> str:
    return SYSTEM_PROMPT + "\n\n---\n\n" + state_input


# ---- smoke testing & value capture ------------------------------------------


def _build_smoke_program(
    setup: str,
    solution: str,
    target_var: str,
    fn_info: Optional[Tuple[str, str, str]],
) -> str:
    """Assemble a smoke-test program.

    Uses `exec()` on a code string so that SyntaxError / IndentationError
    inside the candidate are caught by the surrounding `try/except`. Stubs
    `load_data` so problems whose setup calls it can still parse.

    On success prints `SMOKE_OK::TYPE::...` and `SMOKE_OK::REPR::...`.
    On failure prints `SMOKE_FAIL` followed by the traceback.

    For function-body problems, the candidate is re-indented to body level,
    the `### BEGIN SOLUTION` line is replaced with the body, and the
    function is called with default args to verify signature.
    """
    if fn_info is None:
        # Module-level completion: assemble setup + solution.
        # Two-phase: define target_var via exec, then probe in same scope.
        full_code = (setup or "") + "\n" + (solution or "")
        program = (
            "import traceback as _tb\n"
            "_setup_code = " + repr(full_code) + "\n"
            "_ns = {}\n"
            "def load_data():\n"
            "    return None\n"
            "_ns['load_data'] = load_data\n"
            "try:\n"
            "    exec(_setup_code, _ns)\n"
            f"    if {target_var!r} not in _ns:\n"
            f"        print('SMOKE_FAIL: target {target_var} not defined')\n"
            "    else:\n"
            f"        _v = _ns[{target_var!r}]\n"
            "        try:\n"
            "            _r = repr(_v)\n"
            "        except Exception:\n"
            "            _r = '<repr failed>'\n"
            "        print('SMOKE_OK::TYPE::', type(_v).__name__)\n"
            "        print('SMOKE_OK::REPR::', _r[:1200])\n"
            "except Exception:\n"
            "    print('SMOKE_FAIL:')\n"
            "    _tb.print_exc()\n"
        )
        return program

    # Function-body: re-indent solution to body level and inject in place of
    # the `### BEGIN SOLUTION` marker. Then attempt to call the function.
    func_name, _signature, body_indent = fn_info
    body = _reindent_to(solution, body_indent)
    # Strip duplicate function def (some models redeclare it).
    body = re.sub(
        rf"^\s*def\s+{re.escape(func_name)}\s*\([^)]*\)\s*:.*?(?=\n\S|\Z)",
        "",
        body,
        flags=re.DOTALL | re.MULTILINE,
    ).strip("\n")
    if not body.strip():
        body = body_indent + "pass"
    if "BEGIN SOLUTION" in (setup or ""):
        setup_with_body = _BEGIN_SOL_LINE_RE.sub(body + "\n", setup or "", count=1)
    else:
        setup_with_body = (setup or "") + "\n" + body
    # Two-phase: parse-time first (catches IndentationError / SyntaxError);
    # then attempt to call the function with sensible defaults.
    program = (
        "import traceback as _tb\n"
        "_setup_code = " + repr(setup_with_body) + "\n"
        "_ns = {}\n"
        "def load_data():\n"
        "    return None\n"
        "_ns['load_data'] = load_data\n"
        "_parse_ok = False\n"
        "try:\n"
        "    exec(_setup_code, _ns)\n"
        "    _parse_ok = True\n"
        "except Exception:\n"
        "    print('SMOKE_FAIL: parse/exec error')\n"
        "    _tb.print_exc()\n"
        "if _parse_ok:\n"
        f"    if {func_name!r} not in _ns:\n"
        f"        print('SMOKE_FAIL: function {func_name} not defined')\n"
        "    else:\n"
        f"        _f = _ns[{func_name!r}]\n"
        "        import inspect as _ins\n"
        "        try:\n"
        "            _sig = _ins.signature(_f)\n"
        "            _can_call = all(\n"
        "                p.default is not _ins.Parameter.empty\n"
        "                for p in _sig.parameters.values()\n"
        "                if p.kind in (_ins.Parameter.POSITIONAL_OR_KEYWORD,\n"
        "                              _ins.Parameter.POSITIONAL_ONLY,\n"
        "                              _ins.Parameter.KEYWORD_ONLY)\n"
        "            )\n"
        "        except Exception:\n"
        "            _can_call = False\n"
        "        if _can_call:\n"
        "            try:\n"
        "                _val = _f()\n"
        "                try:\n"
        "                    _r = repr(_val)\n"
        "                except Exception:\n"
        "                    _r = '<repr failed>'\n"
        "                print('SMOKE_OK::TYPE::', type(_val).__name__)\n"
        "                print('SMOKE_OK::REPR::', _r[:1200])\n"
        "            except Exception:\n"
        "                print('SMOKE_PARSE_OK: function defined but call failed')\n"
        "                _tb.print_exc()\n"
        "        else:\n"
        "            # Function requires args we can't infer — accept the\n"
        "            # parse as success.\n"
        "            print('SMOKE_OK::TYPE::', 'function')\n"
        "            print('SMOKE_OK::REPR:: <function-defined>')\n"
    )
    return program


def _parse_smoke_output(out: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (status, payload).

    status:
      - 'OK': candidate parsed AND ran cleanly; payload is REPR string for
        consensus voting.
      - 'PARSE_OK': candidate parsed but couldn't be exercised (function
        needs args we can't infer, or runtime error in call); payload is
        the traceback excerpt.
      - 'FAIL': parse/exec error (e.g., IndentationError, SyntaxError);
        payload is the traceback excerpt.
      - None: smoke output unparseable.
    """
    s = str(out)
    if "SMOKE_OK::REPR::" in s:
        idx = s.rfind("SMOKE_OK::REPR::")
        repr_str = s[idx + len("SMOKE_OK::REPR::") :].strip()
        return ("OK", repr_str)
    if "SMOKE_OK::TYPE::" in s:
        return ("OK", "")
    if "SMOKE_PARSE_OK" in s:
        return ("PARSE_OK", s[-1500:])
    if "SMOKE_FAIL" in s:
        return ("FAIL", s[-1500:])
    return (None, s[-1500:])


# ---- main solver ------------------------------------------------------------


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sample_id = state.sample_id
        library = state.metadata.get("library", "?")
        print(f"[{sample_id}] library={library}")

        target_var = _detect_target_var(state.input)
        setup = _extract_setup_code(state.input)
        fn_info = _detect_function_body(state.input)
        print(
            f"[{sample_id}] target_var={target_var} "
            f"function_body={fn_info[0] if fn_info else 'no'} "
            f"setup_len={len(setup)}"
        )

        # Locate python_session
        py_tool = None
        for t in state.tools:
            try:
                if ToolDef(t).name == "python_session":
                    py_tool = t
                    break
            except Exception:
                continue

        prompt = _build_user_prompt(state.input)

        # ---- Generate 3 candidates in parallel ------------------------------
        async def _gen(model, temp):
            try:
                resp = await model.generate(
                    prompt, config=GenerateConfig(temperature=temp)
                )
                return _extract_solution_code(resp.completion or "")
            except Exception as e:
                print(f"[{sample_id}] generate failed (temp={temp}): {e}")
                return ""

        cand_specs = [
            ("sonnet@0", CLAUDE_SONNET_4_6, 0.0),
            ("sonnet@0.6", CLAUDE_SONNET_4_6, 0.6),
            ("mini@0", GPT_5_4_MINI, 0.0),
        ]
        candidates_raw = await asyncio.gather(
            *[_gen(model, temp) for _, model, temp in cand_specs]
        )
        candidates = [
            (name, code) for (name, _, _), code in zip(cand_specs, candidates_raw) if code
        ]
        for name, code in candidates:
            print(f"[{sample_id}] candidate {name} len={len(code)}")

        if not candidates:
            # Last-resort fallback so we always emit something valid.
            state.output.completion = _wrap(f"{target_var} = None")
            return state

        # ---- Smoke-test each candidate --------------------------------------
        smoke_results: list[Tuple[str, str, Optional[str], Optional[str]]] = []
        # (name, code, status, payload)

        if py_tool is not None:
            for name, code in candidates:
                program = _build_smoke_program(setup, code, target_var, fn_info)
                try:
                    out = await py_tool(code=program)
                    status, payload = _parse_smoke_output(str(out))
                except Exception as e:
                    status, payload = ("FAIL", f"tool error: {e}")
                smoke_results.append((name, code, status, payload))
                print(f"[{sample_id}] smoke {name}: {status}")
        else:
            print(f"[{sample_id}] no python_session; skipping smoke test")
            for name, code in candidates:
                smoke_results.append((name, code, None, None))

        # ---- Pick best candidate --------------------------------------------
        ok_passers = [(n, c, p) for (n, c, s, p) in smoke_results if s == "OK"]
        parse_ok = [(n, c, p) for (n, c, s, p) in smoke_results if s == "PARSE_OK"]
        passers = ok_passers or parse_ok

        chosen_name = ""
        chosen_code = ""

        if passers:
            # If we have multiple passers, look for value consensus on REPR.
            if len(passers) >= 2:
                from collections import Counter

                reprs = [p for (_, _, p) in passers if p]
                if reprs:
                    counts = Counter(reprs)
                    top_repr, top_count = counts.most_common(1)[0]
                    if top_count >= 2:
                        # Pick the first passer matching the consensus REPR,
                        # preferring sonnet@0 by ordering.
                        order = {n: i for i, (n, *_rest) in enumerate(cand_specs)}
                        consensus = sorted(
                            [(n, c) for (n, c, p) in passers if p == top_repr],
                            key=lambda nc: order.get(nc[0], 99),
                        )
                        chosen_name, chosen_code = consensus[0]
                        print(
                            f"[{sample_id}] consensus pick {chosen_name} "
                            f"({top_count}/{len(reprs)} agree)"
                        )

            if not chosen_code:
                # Fall back to the first passer in preference order.
                order = {n: i for i, (n, *_rest) in enumerate(cand_specs)}
                passers_sorted = sorted(passers, key=lambda nc: order.get(nc[0], 99))
                chosen_name, chosen_code, _ = passers_sorted[0]
                print(f"[{sample_id}] no consensus; picked first passer {chosen_name}")
        else:
            # No candidate passed. Smoke is uninformative (typically because
            # setup uses `load_data()` and our stub returns None, breaking
            # downstream unpacking). Fall back to preference-order pick: the
            # Sonnet@0 candidate is the safest single choice.
            order = {n: i for i, (n, *_rest) in enumerate(cand_specs)}
            ranked = sorted(smoke_results, key=lambda r: order.get(r[0], 99))
            primary = ranked[0]
            chosen_name, chosen_code = primary[0], primary[1]
            print(f"[{sample_id}] all smoke failed; preference pick {chosen_name}")

        if not chosen_code:
            chosen_code = f"{target_var} = None"
            chosen_name = "fallback"

        # ---- Emit ------------------------------------------------------------
        # For function-body problems, the candidate must be indented body code.
        # Most models produce indented bodies already; if a candidate is
        # un-indented (module-level), re-indent it before emitting.
        if fn_info is not None:
            _, _, body_indent = fn_info
            # Strip any duplicate `def` the model emitted.
            cleaned = re.sub(
                rf"^\s*def\s+{re.escape(fn_info[0])}\s*\([^)]*\)\s*:\s*\n",
                "",
                chosen_code,
                count=1,
                flags=re.MULTILINE,
            )
            # Detect: are all non-blank lines indented?
            nonblank = [ln for ln in cleaned.splitlines() if ln.strip()]
            already_indented = bool(nonblank) and all(
                ln.startswith(body_indent) or ln.startswith(" ") for ln in nonblank
            )
            if not already_indented:
                cleaned = _reindent_to(cleaned, body_indent)
            chosen_code = cleaned

        state.output.completion = _wrap(chosen_code)
        print(
            f"[{sample_id}] emitted {len(state.output.completion)} chars from {chosen_name}"
        )
        return state

    return solve
