"""DS-1000 solver: multi-candidate + smoke + judge + retry.

Strategy:
1. Generate 3 candidates in parallel from diverse model/temperature settings:
   Sonnet@0, GPT-5.4@0, Sonnet@0.6.
2. Smoke-test each candidate inside `python_session`. The smoke program
   `exec()`s the candidate so SyntaxError/IndentationError are catchable;
   for function-body problems it re-indents the candidate, closes the open
   `def`, and (when args have defaults) calls it to capture the return value.
3. Three-tier smoke status: OK (parsed and ran cleanly, with REPR captured),
   PARSE_OK (parsed but call failed or args unknown), FAIL (parse error).
4. Picking:
   - REPR consensus among OK passers (>=2 agree) -> consensus pick.
   - Else, OK passers with different REPRs -> ask Sonnet@0 to judge.
   - Else, all FAIL -> retry Sonnet@0 with traceback feedback.
   - Else, fall back to preference order Sonnet@0 -> GPT-5.4@0 -> Sonnet@0.6.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, Tuple

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6, GPT_5_4


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

CODING RULES:
- Use the EXACT target variable name the prompt asks for. The line
  `<name> = ... # put solution in this variable` tells you the name. It is
  often `result` but sometimes `weights`, `transformed_df`, `cluster_labels`,
  `b`, `c`, `slope`, `df_out`, etc.
- Do NOT redeclare imports or variables that the setup `<code>` block defines.
  In particular, if the setup contains `x, y = load_data()` or
  `gridsearch, testX, testY = load_data()`, USE those variables — do not
  recreate them with synthetic data.
- DO NOT hard-code shapes/sizes/values from the example. Derive them from the
  inputs: `n = a.shape[0]`, `len(B)`, `df.columns.nlevels`, `min(a.shape)`,
  etc. The hidden test feeds DIFFERENT inputs of different sizes.
- HONOR METHOD HINTS in the prompt. When the question explicitly mentions or
  suggests a specific method/function/library, use it.
    * "Perhaps using Simpson rule?" -> use `scipy.integrate.simpson`, not trapz.
    * "find frequent value in each row" / "mode" -> use `df.mode(axis=1)`.
    * "without a for loop" / "vectorized" / "the efficient way" /
      "not one by one" -> NO `for` or `while`; use vector ops, `np.where`,
      boolean masks, `np.searchsorted`, `np.einsum`, `groupby`, `apply`, etc.
    * "without using X" -> avoid X, even if it would be the obvious approach.
- PREFER NAMED LIBRARY METHODS over hand-rolled reimplementations. Some
  problems enforce this via hidden style tests (a `for`-loop or manual-vote
  solution can fail even when output matches). When pandas/numpy/scipy/sklearn
  has a one-liner for what you're doing, use it.
- Function signatures: when the problem says "define function named foo as
  solution", the test will call your function with the SAME arg list as in
  any preamble snippet. If the preamble references `x_min`/`x_max` (etc.) as
  module-level globals, your function takes only the variable args, OR uses
  defaults bound to those globals: `def foo(x, mi=x_min, mx=x_max): ...`.
  Do NOT add the globals as required positional parameters.
- Match the requested return type / container exactly: list vs. tuple vs.
  ndarray vs. Series vs. DataFrame. Convert at the end if needed
  (`.tolist()`, `.to_numpy()`, `.reshape(-1)`).
- DTYPE matters: tests use `np.testing.assert_array_equal` / `assert_frame_equal`
  which compare dtype too. If a reference uses `df.loc[i, col] = ...`, the
  resulting column is float64 even when values are integers. When a count
  column is added next to floats, prefer float over int.
- Sklearn: `preprocessing.scale`, `metrics.pairwise_distances` accept 1D;
  `StandardScaler` requires 2D. Use top-level functions when possible.
- Pandas: prefer `groupby`, `pivot`, `melt`, `apply`, `mode`, vector ops.
  `.str.join('|').str.get_dummies()` is the idiom for one-hot a list-of-strings
  column WITHOUT exploding rows.
- SciPy: use `scipy.stats`, `scipy.optimize`, `scipy.sparse`,
  `scipy.cluster.hierarchy.linkage`, `scipy.integrate.simpson` directly.
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
_BEGIN_SOL_LINE_RE = re.compile(r"^[ \t]*###\s*BEGIN SOLUTION[ \t]*\n?", re.MULTILINE)


def _detect_target_var(prompt: str) -> str:
    """Find the variable name the prompt asks the solver to populate."""
    m = _VAR_RE.search(prompt)
    if m:
        return m.group(1)
    for code in _CODE_BLOCK_LOOSE_RE.findall(prompt):
        m2 = _RETURN_VAR_RE.search(code)
        if m2:
            return m2.group(1)
    if "put score in" in prompt or "put prediction in" in prompt:
        m3 = re.findall(r"put\s+\w+\s+in\s+`([A-Za-z_][A-Za-z_0-9]*)`", prompt)
        if m3:
            return m3[0]
    return "result"


def _extract_setup_code(prompt: str) -> str:
    matches = _CODE_BLOCK_LOOSE_RE.findall(prompt)
    if matches:
        return matches[0].strip()
    return ""


def _detect_function_body(prompt: str) -> Optional[Tuple[str, str, str]]:
    """If the prompt's setup defines a function whose body is open at
    `### BEGIN SOLUTION`, return (func_name, signature, indent_str)."""
    blocks = _CODE_BLOCK_LOOSE_RE.findall(prompt)
    for block in blocks:
        if not _BEGIN_SOL_RE.search(block):
            continue
        m = _DEF_LINE_RE.search(block)
        if not m:
            continue
        def_indent = m.group(1) or ""
        body_indent = def_indent + "    "
        return (m.group(2), m.group(3), body_indent)
    return None


def _extract_solution_code(text: str) -> str:
    s = (text or "").strip()
    m = re.search(r"<code>(.*?)</code>", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        s = _FENCE_RE.sub("", s).strip()
    s = re.sub(r"^\s*###?\s*(BEGIN|END)\s*SOLUTION\s*$", "", s, flags=re.MULTILINE)
    return s.strip("\n")


def _wrap(code: str) -> str:
    return f"<code>\n{code}\n</code>"


def _reindent_to(code: str, target_indent: str) -> str:
    lines = code.splitlines()
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank:
        return code
    common = min((len(ln) - len(ln.lstrip(" "))) for ln in nonblank)
    stripped = "\n".join(ln[common:] if ln.strip() else "" for ln in lines)
    return "\n".join(target_indent + ln if ln.strip() else "" for ln in stripped.splitlines())


# ---- prompt builders --------------------------------------------------------


def _build_user_prompt(state_input: str) -> str:
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
        + "When this was appended to the setup code and run, it failed with:\n"
        + "```\n"
        + error[-1500:]
        + "\n```\n\n"
        + "Fix the code. Output ONLY the corrected `<code>...</code>` block."
    )


def _build_judge_prompt(state_input: str, candidates: list) -> str:
    """Ask Sonnet which candidate is most likely correct.

    `candidates` is a list of (label, code, status, smoke_repr_or_traceback).
    """
    parts = [
        SYSTEM_PROMPT,
        "\n\n---\n\nORIGINAL PROBLEM:\n",
        state_input,
        "\n\n---\n\nThree candidate solutions were generated. Each was smoke-tested in a sandbox; their computed values (or errors) are shown.\n\n",
    ]
    letters = ["A", "B", "C", "D", "E"]
    for i, (label, code, status, payload) in enumerate(candidates):
        letter = letters[i] if i < len(letters) else f"#{i}"
        parts.append(f"=== Candidate {letter} ({label}, smoke={status}) ===\n")
        parts.append("```python\n" + code + "\n```\n")
        if payload:
            parts.append(f"Smoke output:\n```\n{payload[:800]}\n```\n")
        parts.append("\n")
    parts.append(
        "Pick the candidate most likely to PASS the hidden test. Pay close attention to:\n"
        "- Whether the candidate honors any METHOD HINTS in the question (e.g. Simpson rule, mode, simpson, groupby).\n"
        "- Whether sizes are derived from inputs (not hardcoded from the example).\n"
        "- Whether function signatures match the test's expected call shape.\n"
        "- Whether dtype/return-type matches what the test will compare against.\n"
        "- Whether the candidate uses idiomatic library methods over hand-rolled reimplementations.\n\n"
        "Output ONLY the chosen candidate's full `<code>...</code>` block. No explanation."
    )
    return "".join(parts)


# ---- smoke testing & value capture ------------------------------------------


def _build_smoke_program(
    setup: str,
    solution: str,
    target_var: str,
    fn_info: Optional[Tuple[str, str, str]],
) -> str:
    if fn_info is None:
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

    func_name, _signature, body_indent = fn_info
    body = _reindent_to(solution, body_indent)
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
        "            print('SMOKE_OK::TYPE::', 'function')\n"
        "            print('SMOKE_OK::REPR:: <function-defined>')\n"
    )
    return program


def _parse_smoke_output(out: str) -> Tuple[Optional[str], Optional[str]]:
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
            ("gpt54@0", GPT_5_4, 0.0),
            ("sonnet@0.6", CLAUDE_SONNET_4_6, 0.6),
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
            state.output.completion = _wrap(f"{target_var} = None")
            return state

        # ---- Smoke-test each candidate --------------------------------------
        smoke_results: list = []  # (name, code, status, payload)

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

        order = {n: i for i, (n, *_rest) in enumerate(cand_specs)}

        # ---- Pick best candidate --------------------------------------------
        ok_passers = [(n, c, s, p) for (n, c, s, p) in smoke_results if s == "OK"]
        parse_ok = [(n, c, s, p) for (n, c, s, p) in smoke_results if s == "PARSE_OK"]
        fail = [(n, c, s, p) for (n, c, s, p) in smoke_results if s == "FAIL"]
        passers = ok_passers or parse_ok

        chosen_name = ""
        chosen_code = ""

        # Step 1: REPR consensus among OK passers.
        if len(ok_passers) >= 2:
            from collections import Counter

            reprs = [p for (_, _, _, p) in ok_passers if p]
            if reprs:
                counts = Counter(reprs)
                top_repr, top_count = counts.most_common(1)[0]
                if top_count >= 2:
                    consensus = sorted(
                        [(n, c) for (n, c, _, p) in ok_passers if p == top_repr],
                        key=lambda nc: order.get(nc[0], 99),
                    )
                    chosen_name, chosen_code = consensus[0]
                    print(
                        f"[{sample_id}] consensus pick {chosen_name} "
                        f"({top_count}/{len(reprs)} agree)"
                    )

        # Step 2: judge step when OK passers disagree.
        if not chosen_code and len(ok_passers) >= 2:
            judge_prompt = _build_judge_prompt(state.input, ok_passers)
            try:
                jresp = await CLAUDE_SONNET_4_6.generate(
                    judge_prompt, config=GenerateConfig(temperature=0.0)
                )
                jcode = _extract_solution_code(jresp.completion or "")
                if jcode:
                    chosen_name = "judge"
                    chosen_code = jcode
                    print(f"[{sample_id}] judge picked a candidate")
            except Exception as e:
                print(f"[{sample_id}] judge failed: {e}")

        # Step 3: fall back to first passer (OK or PARSE_OK).
        if not chosen_code and passers:
            passers_sorted = sorted(passers, key=lambda r: order.get(r[0], 99))
            chosen_name, chosen_code = passers_sorted[0][0], passers_sorted[0][1]
            print(f"[{sample_id}] no consensus/judge; first passer {chosen_name}")

        # Step 4: retry path — all candidates failed smoke (informative).
        if not chosen_code and fail and py_tool is not None:
            # Use the Sonnet@0 candidate's traceback for retry feedback.
            sonnet_fail = next(
                (r for r in fail if r[0] == "sonnet@0"), fail[0]
            )
            prev_code = sonnet_fail[1]
            tb = sonnet_fail[3] or ""
            try:
                rresp = await CLAUDE_SONNET_4_6.generate(
                    _build_retry_prompt(state.input, prev_code, tb),
                    config=GenerateConfig(temperature=0.0),
                )
                rcode = _extract_solution_code(rresp.completion or "")
                if rcode:
                    chosen_name = "retry"
                    chosen_code = rcode
                    print(f"[{sample_id}] retry produced new candidate")
            except Exception as e:
                print(f"[{sample_id}] retry failed: {e}")

        # Step 5: ultimate fallback — preference-order pick.
        if not chosen_code:
            ranked = sorted(smoke_results, key=lambda r: order.get(r[0], 99))
            primary = ranked[0]
            chosen_name, chosen_code = primary[0], primary[1]
            print(f"[{sample_id}] last-resort preference pick {chosen_name}")

        if not chosen_code:
            chosen_code = f"{target_var} = None"
            chosen_name = "fallback"

        # ---- Emit ------------------------------------------------------------
        if fn_info is not None:
            _, _, body_indent = fn_info
            cleaned = re.sub(
                rf"^\s*def\s+{re.escape(fn_info[0])}\s*\([^)]*\)\s*:\s*\n",
                "",
                chosen_code,
                count=1,
                flags=re.MULTILINE,
            )
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
