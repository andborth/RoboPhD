"""DS-1000 solver: exec-verify-ensemble.

Pipeline per problem:
  1. Guided prompt (format rules, target-variable rule, example-output rule,
     style-token warning, matplotlib cheat-sheet, function-body rule).
  2. Function-completion problems (### BEGIN SOLUTION inside an open def, with
     undefined load_data()) can't be executed standalone: single-model path with
     a local compile gate and one repair round.
  3. All other problems: two candidates from different model families
     (GPT_5_4_MINI + CLAUDE_HAIKU_4_5) generated concurrently, both executed in
     the free sandbox against the skeleton parsed from the prompt; crashing /
     target-missing candidates get one traceback-guided repair.
  4. Agreement check: deep value comparison of the target variable (or
     figure-state dump comparison for Matplotlib). Agreement -> submit A.
  5. Disagreement -> GPT_5_4 arbiter sees both candidates plus their *executed*
     values/errors and writes the final solution (also executed before adoption).
  6. Matplotlib: a reflection pass reviews the dumped figure properties.
  7. Final code containing for/while tokens gets a vectorized-rewrite attempt
     (adopted only if verified value-equal or an idiom hint is present).

All sandbox interaction degrades gracefully: on any harness failure the agent
falls back to submitting candidate A.
"""

import ast
import asyncio
import io
import re
import tokenize

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_HAIKU_4_5, GPT_5_4, GPT_5_4_MINI

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

GUIDANCE = """You are an expert Python data scientist solving a DS-1000 problem.
Output ONLY executable Python code inside <code></code> tags - no prose, no markdown fences, no BEGIN/END SOLUTION markers.

Rules:
1. Your code is appended after the given setup code. Do not repeat the setup lines.
2. Assign the final answer to the exact variable indicated by the "put solution in this variable" line (e.g. result, df, weights). The hidden grader reads that variable.
3. Study the example input/output shown in the problem and make sure your code reproduces the expected output EXACTLY (values, dtype, shape, ordering, index/columns). When the wording is ambiguous, trust the example output over your first reading.
4. Hidden tests rerun your code on different data of the same kind - generalize (do not hardcode example values), but only along dimensions the problem implies.
5. Prefer vectorized library idioms over Python for/while loops: some graders reject any solution whose source contains a for/while token, even inside comprehensions. If the problem says to use a specific function, actually call it; if it says "without X", do not use X.
"""

MPL_GUIDANCE = """6. Matplotlib: follow the comment instructions literally.
Marker codes: 'd' = thin diamond, 'D' = diamond, '*' = star, 's' = square, 'o' = circle, '+' = plus, 'x' = x, '|' = vline, '^'/'v' = triangles.
A "hatch" is a fill pattern, not a marker: "star hatch" means passing hatch='*' (scatter/bar accept hatch=...).
Distinguish marker/line/face/edge colors, markersize vs linewidth, and do exactly what is asked - nothing more.
"""

FUNC_GUIDANCE = """6. IMPORTANT: the setup code ends inside a function definition (### BEGIN SOLUTION appears inside a def). Output ONLY the remaining indented body of that function - indent every line with 4 spaces and end with the appropriate return statement. Do not repeat the def line and do not write any top-level (unindented) code.
"""

MARKER_CHEATSHEET = (
    "Marker code reference: 'd'=thin diamond, 'D'=diamond, '*'=star, 's'=square, "
    "'o'=circle, '+'=plus, 'x'=x, '|'=vline, '^'/'v'=triangles. "
    "A \"hatch\" is a fill pattern (hatch='*'), not a marker."
)

IDIOM_HINT_RE = re.compile(
    r"idiomatic|vectoriz|efficient|one[- ]?lin|without.{0,25}loop|no loop|"
    r"not one by one|elegant|pandas way|numpy way|clean(est)? way",
    re.I,
)

# --------------------------------------------------------------------------
# Sandbox harness (installed once per sample; python_session is stateful)
# --------------------------------------------------------------------------

_HARNESS = '''
import io as _io, contextlib as _ctx, traceback as _tb, sys as _sys
try:
    import matplotlib as _mpl
    _mpl.use("Agg")
except Exception:
    pass

_ns_store = {}

def _ds_run(_skeleton, _code):
    import numpy as _np, random as _rnd
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except Exception:
        pass
    _rnd.seed(42)
    _np.random.seed(42)
    _ns = {}
    _buf = _io.StringIO()
    _err = None
    try:
        with _ctx.redirect_stdout(_buf):
            exec(_skeleton + chr(10) + _code, _ns)
    except BaseException:
        _err = _tb.format_exc()
    return _ns, _err, _buf.getvalue()

def _ds_eq(a, b):
    import numpy as _np
    try:
        import pandas as _pd
    except Exception:
        _pd = None
    if a is b:
        return True
    if _pd is not None and isinstance(a, _pd.DataFrame) and isinstance(b, _pd.DataFrame):
        try:
            _pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-8)
            return True
        except Exception:
            return False
    if _pd is not None and isinstance(a, _pd.Series) and isinstance(b, _pd.Series):
        try:
            _pd.testing.assert_series_equal(a, b, check_dtype=False, atol=1e-8)
            return True
        except Exception:
            return False
    if "torch" in _sys.modules:
        import torch as _torch
        if isinstance(a, _torch.Tensor) and isinstance(b, _torch.Tensor):
            try:
                return a.shape == b.shape and bool(_torch.allclose(a.double(), b.double(), atol=1e-8))
            except Exception:
                return False
    if "scipy" in _sys.modules:
        try:
            from scipy import sparse as _sp
            if _sp.issparse(a) and _sp.issparse(b):
                return a.shape == b.shape and abs(a - b).max() <= 1e-8
        except Exception:
            pass
    if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
        try:
            _a, _b = _np.asarray(a), _np.asarray(b)
            if _a.shape != _b.shape:
                return False
            try:
                return bool(_np.allclose(_a, _b, atol=1e-8, equal_nan=True))
            except Exception:
                return bool(_np.array_equal(_a, _b))
        except Exception:
            return False
    if isinstance(a, float) and isinstance(b, float):
        return (a != a and b != b) or abs(a - b) <= 1e-8 * max(1.0, abs(a), abs(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_ds_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_ds_eq(a[k], b[k]) for k in a)
    try:
        r = (a == b)
        if isinstance(r, _np.ndarray):
            return bool(r.all())
        return bool(r)
    except Exception:
        return False

def _ds_summ(v, maxlen=700):
    try:
        extra = ""
        if hasattr(v, "shape"):
            extra = " shape=" + str(getattr(v, "shape", None)) + " dtype=" + str(getattr(v, "dtype", ""))
        r = repr(v)
        if len(r) > maxlen:
            r = r[:maxlen] + "...[truncated]"
        return "type=" + type(v).__name__ + extra + " value=" + r
    except Exception as e:
        return "<unreprable " + type(v).__name__ + ": " + str(e) + ">"

def _ds_fig_report():
    try:
        import matplotlib.pyplot as plt
        parts = []
        for fnum in plt.get_fignums():
            fig = plt.figure(fnum)
            for j, ax in enumerate(fig.axes):
                parts.append(
                    "fig%s.ax%s: title=%r xlabel=%r ylabel=%r xscale=%s yscale=%s"
                    % (fnum, j, ax.get_title(), ax.get_xlabel(), ax.get_ylabel(),
                       ax.get_xscale(), ax.get_yscale())
                )
                for k, ln in enumerate(ax.lines[:8]):
                    parts.append(
                        "  line%s: marker=%r ls=%r lw=%s color=%r n=%s label=%r"
                        % (k, ln.get_marker(), ln.get_linestyle(), ln.get_linewidth(),
                           ln.get_color(), len(ln.get_xdata()), ln.get_label())
                    )
                for k, c in enumerate(ax.collections[:8]):
                    try:
                        hatch = c.get_hatch()
                    except Exception:
                        hatch = "?"
                    try:
                        n = len(c.get_offsets())
                    except Exception:
                        n = "?"
                    parts.append("  collection%s: type=%s hatch=%r n=%s" % (k, type(c).__name__, hatch, n))
                if ax.patches:
                    parts.append("  patches: n=%s types=%s" % (len(ax.patches), sorted({type(p).__name__ for p in ax.patches})))
                leg = ax.get_legend()
                if leg is not None:
                    parts.append("  legend: %s" % [t.get_text() for t in leg.get_texts()])
                parts.append(
                    "  xticklabels=%s yticklabels=%s"
                    % ([t.get_text() for t in ax.get_xticklabels()][:12],
                       [t.get_text() for t in ax.get_yticklabels()][:12])
                )
        return chr(10).join(parts) if parts else "<no figures>"
    except Exception as e:
        return "<report failed: %s>" % e

def _ds_check(tag, skeleton, code, target, mpl=False):
    ns, err, out = _ds_run(skeleton, code)
    _ns_store[tag] = ns
    print("@@ERR@@", repr(None if err is None else err[-1200:]))
    print("@@HASV@@", repr(target in ns))
    print("@@VAL@@", repr(_ds_summ(ns[target]) if target in ns else "<missing>"))
    print("@@OUT@@", repr(out[-400:]))
    if mpl:
        print("@@FIG@@", repr(_ds_fig_report()))

def _ds_compare(tag1, tag2, target):
    a = _ns_store.get(tag1, {})
    b = _ns_store.get(tag2, {})
    ok = target in a and target in b and _ds_eq(a[target], b[target])
    print("@@EQ@@", repr(bool(ok)))

print("HARNESS_READY")
'''

# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def extract_code(text: str) -> str:
    """Pull the code out of a model response."""
    if not text:
        return ""
    blocks = re.findall(r"<code>(.*?)</code>", text, re.S)
    if blocks:
        code = blocks[-1]
    elif "<code>" in text:
        code = text.rsplit("<code>", 1)[1]
    else:
        fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        code = fences[-1] if fences else text
    code = re.sub(r"^\s*(BEGIN|END) SOLUTION.*$", "", code, flags=re.M)
    return code.strip("\n").rstrip()


def parse_problem(prompt: str):
    """Return (skeleton_code, target_variable, func_mode).

    func_mode marks the DS-1000 function-completion format: the last <code>
    block before '### BEGIN SOLUTION' is unclosed and ends inside a def whose
    body the answer must complete with indented code. Those skeletons usually
    call an undefined load_data(), so they can't be executed standalone.
    """
    func_mode = False
    skeleton = ""
    if "### BEGIN SOLUTION" in prompt:
        pre = prompt.split("### BEGIN SOLUTION")[0]
        closed = re.findall(r"<code>(.*?)</code>", pre, re.S)
        tail = pre.rsplit("<code>", 1)[1] if "<code>" in pre else ""
        if tail and "</code>" not in tail:
            func_mode = True
            skeleton = "\n".join([b.strip("\n") for b in closed] + [tail.strip("\n")])
        else:
            skeleton = "\n".join(b.strip("\n") for b in closed)
    elif "BEGIN SOLUTION" in prompt:
        pre = prompt.split("BEGIN SOLUTION")[0]
        blocks = re.findall(r"<code>(.*?)</code>", pre, re.S)
        skeleton = "\n".join(b.strip("\n") for b in blocks)
    elif "# SOLUTION START" in prompt:
        skeleton = prompt.split("# SOLUTION START")[0]
    else:
        blocks = re.findall(r"<code>(.*?)</code>", prompt, re.S)
        skeleton = blocks[0].strip("\n") if blocks else ""
    m = re.search(
        r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\..*put solution in this variable",
        prompt,
        re.M,
    )
    target = m.group(1) if m else "result"
    return skeleton, target, func_mode


def has_loop_tokens(code: str) -> bool:
    """Mirror the style grader: look for for/while NAME tokens (not in strings)."""
    try:
        toks = [
            t.string
            for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type == tokenize.NAME
        ]
        return "for" in toks or "while" in toks
    except Exception:
        return bool(re.search(r"\b(for|while)\b", code))


def compiles_ok(skeleton: str, code: str) -> bool:
    try:
        compile(skeleton + "\n" + code + "\n", "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def compile_err(skeleton: str, code: str) -> str:
    try:
        compile(skeleton + "\n" + code + "\n", "<candidate>", "exec")
        return ""
    except SyntaxError as e:
        return f"SyntaxError when appended to the setup code: {e}"


def parse_markers(output: str) -> dict:
    """Parse @@KEY@@ <repr> lines printed by the harness."""
    out = {}
    for line in str(output).splitlines():
        m = re.match(r"@@(\w+)@@ (.*)$", line)
        if m:
            try:
                out[m.group(1)] = ast.literal_eval(m.group(2))
            except Exception:
                out[m.group(1)] = m.group(2)
    return out


def norm_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())


def clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


# --------------------------------------------------------------------------
# LLM helpers
# --------------------------------------------------------------------------


async def gen(model, prompt: str, max_tokens: int = 1200) -> str:
    try:
        resp = await model.generate(prompt, config=GenerateConfig(max_tokens=max_tokens))
        return resp.completion or ""
    except Exception as e:
        print(f"  generate error: {e}")
        return ""


def repair_prompt(problem: str, code: str, err: str, note: str) -> str:
    return f"""Your Python solution to the problem below fails when executed.

Problem:
{clip(problem, 5000)}

Your code:
{clip(code, 2000)}

Execution error:
{clip(err, 1200)}
{note}
Fix it. Output ONLY the corrected Python code inside <code></code> tags."""


def arbiter_prompt(problem, a_code, a_desc, b_code, b_desc) -> str:
    return f"""Two independently written solutions to the same problem were executed and they disagree. Determine the correct solution.

Problem:
{clip(problem, 5000)}

--- Candidate A ---
{clip(a_code, 2000)}
Execution: {a_desc}

--- Candidate B ---
{clip(b_code, 2000)}
Execution: {b_desc}

Check each candidate against the problem's requirements and ESPECIALLY against any example output shown in the problem text. Reply with at most 3 short sentences of analysis, then the final correct solution (candidate A, candidate B, or your own corrected version) inside <code></code> tags."""


def rewrite_prompt(problem: str, code: str) -> str:
    return f"""Rewrite this solution so its source contains NO `for` or `while` tokens anywhere (no loops, no comprehensions, no generator expressions) - use vectorized library operations instead. Keep the behavior and the indentation level identical.

Problem:
{clip(problem, 4000)}

Current solution:
{clip(code, 2000)}

Output ONLY the rewritten Python code inside <code></code> tags."""


def reflect_prompt(problem: str, code: str, fig_report: str) -> str:
    return f"""A matplotlib task was solved with the code below; the resulting figure state was inspected programmatically.

Task:
{clip(problem, 4000)}

Code:
{clip(code, 1500)}

Actual figure state:
{clip(fig_report, 1500)}

{MARKER_CHEATSHEET}

Does the figure literally satisfy EVERY requirement in the task? If yes, reply with exactly OK. If not, reply with the full corrected code inside <code></code> tags."""


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        problem = state.input
        library = state.metadata.get("library", "")
        mpl = library == "Matplotlib"

        skeleton, target, func_mode = parse_problem(problem)
        print(
            f"[{state.sample_id}] library={library} mpl={mpl} func_mode={func_mode} "
            f"target={target} skeleton_lines={len(skeleton.splitlines())}"
        )

        guidance = GUIDANCE + (MPL_GUIDANCE if mpl else "") + (FUNC_GUIDANCE if func_mode else "")
        full_prompt = guidance + "\n" + problem

        def emit(code: str, tag) -> TaskState:
            if not code:
                code = "result = None"
            code = code.replace("<code>", "").replace("</code>", "").strip("\n")
            state.output.completion = f"<code>\n{code}\n</code>"
            print(f"  emitted {len(state.output.completion)} chars (tag={tag})")
            return state

        # ---- Function-completion format: no standalone execution possible. ----
        # The seed went 7/7 on these with a single cheap call, so keep this path
        # simple: one model, local compile gate, one repair, cross-family fallback.
        if func_mode:
            a_code = extract_code(await gen(GPT_5_4_MINI, full_prompt))
            if a_code and compiles_ok(skeleton, a_code):
                return emit(a_code, "A")
            err = compile_err(skeleton, a_code) if a_code else "empty completion"
            note = (
                "Remember: your code completes an open function body - every line "
                "must be indented with 4 spaces, ending with a return statement.\n"
            )
            fixed = extract_code(
                await gen(GPT_5_4_MINI, repair_prompt(problem, a_code, err, note))
            )
            if fixed and compiles_ok(skeleton, fixed):
                return emit(fixed, "A-repaired")
            b_code = extract_code(await gen(CLAUDE_HAIKU_4_5, full_prompt))
            if b_code and compiles_ok(skeleton, b_code):
                return emit(b_code, "B")
            return emit(a_code or fixed or b_code, "A-unrepaired")

        # ---- Standard format: dual generation + sandbox verification. ----
        try:
            py = next(t for t in state.tools if ToolDef(t).name == "python_session")
        except StopIteration:
            py = None

        async def install_harness():
            if py is None:
                return False
            try:
                out = await py(code=_HARNESS)
                return "HARNESS_READY" in str(out)
            except Exception as e:
                print(f"  harness install failed: {e}")
                return False

        resp_a, resp_b, harness_ok = await asyncio.gather(
            gen(GPT_5_4_MINI, full_prompt),
            gen(CLAUDE_HAIKU_4_5, full_prompt),
            install_harness(),
        )
        a_code = extract_code(resp_a)
        b_code = extract_code(resp_b)
        print(f"  candA={len(a_code)}ch candB={len(b_code)}ch harness={harness_ok}")

        async def check(tag: str, code: str) -> dict:
            """Run skeleton+code in a fresh namespace; return marker dict."""
            cell = f"_ds_check({tag!r}, {skeleton!r}, {code!r}, {target!r}, mpl={mpl!r})"
            try:
                return parse_markers(await py(code=cell))
            except Exception as e:
                return {"ERR": f"sandbox failure: {e}", "HASV": False, "VAL": "<none>", "FIG": ""}

        async def compare(tag1: str, tag2: str) -> bool:
            try:
                out = parse_markers(await py(code=f"_ds_compare({tag1!r}, {tag2!r}, {target!r})"))
                return bool(out.get("EQ"))
            except Exception:
                return False

        def is_clean(info: dict) -> bool:
            if info is None or info.get("ERR") is not None:
                return False
            return True if mpl else bool(info.get("HASV"))

        def describe(info: dict) -> str:
            if info is None:
                return "not executed"
            if info.get("ERR"):
                return "ERROR: " + str(info["ERR"])[:600]
            if mpl:
                return "figure state:\n" + str(info.get("FIG", ""))[:900]
            return f"`{target}` = " + str(info.get("VAL"))[:800]

        final_code, final_tag = a_code, None
        a_info = b_info = None

        verifiable = harness_ok and bool(skeleton.strip()) and bool(a_code or b_code)
        if verifiable:
            skel_info = await check("S", "")
            if skel_info.get("ERR") is not None:
                print(f"  skeleton itself fails: {str(skel_info['ERR'])[:120]}")
                verifiable = False

        if verifiable:
            same_code = bool(a_code) and norm_code(a_code) == norm_code(b_code)

            async def check_and_repair(tag, code, model):
                if not code:
                    return code, {"ERR": "empty completion", "HASV": False}
                info = await check(tag, code)
                if not is_clean(info):
                    err = info.get("ERR") or f"code ran but never assigned the variable `{target}`"
                    note = "" if mpl else f"Remember: assign the final answer to the variable `{target}`.\n"
                    fixed = extract_code(
                        await gen(model, repair_prompt(problem, code, str(err), note))
                    )
                    if fixed:
                        fixed_info = await check(tag, fixed)
                        if is_clean(fixed_info) or info.get("ERR"):
                            print(f"  repaired candidate {tag}")
                            return fixed, fixed_info
                return code, info

            if same_code:
                a_code, a_info = await check_and_repair("A", a_code, GPT_5_4_MINI)
                b_code, b_info = a_code, a_info
            else:
                (a_code, a_info), (b_code, b_info) = await asyncio.gather(
                    check_and_repair("A", a_code, GPT_5_4_MINI),
                    check_and_repair("B", b_code, CLAUDE_HAIKU_4_5),
                )

            a_ok, b_ok = is_clean(a_info), is_clean(b_info)
            agree = False
            if a_ok and b_ok:
                if same_code:
                    agree = True
                elif mpl:
                    agree = bool(a_info.get("FIG")) and a_info.get("FIG") == b_info.get("FIG")
                else:
                    agree = await compare("A", "B")
            print(f"  a_ok={a_ok} b_ok={b_ok} agree={agree}")

            if a_ok and (agree or not b_ok):
                final_code, final_tag = a_code, "A"
            elif b_ok and not a_ok:
                final_code, final_tag = b_code, "B"
            else:
                # Both clean but disagreeing, or both broken: arbitrate with evidence.
                arb = extract_code(
                    await gen(
                        GPT_5_4,
                        arbiter_prompt(problem, a_code, describe(a_info), b_code, describe(b_info)),
                        max_tokens=1000,
                    )
                )
                if arb:
                    arb_info = await check("ARB", arb)
                    if is_clean(arb_info):
                        final_code, final_tag = arb, "ARB"
                    elif a_ok:
                        final_code, final_tag = a_code, "A"
                    elif b_ok:
                        final_code, final_tag = b_code, "B"
                    else:
                        final_code, final_tag = arb, "ARB"
                else:
                    final_code, final_tag = (a_code, "A") if a_ok or not b_ok else (b_code, "B")
                print(f"  arbitrated -> {final_tag}")

            # Matplotlib reflection: verify the figure literally matches the request.
            if mpl and final_tag in ("A", "B", "ARB"):
                chosen_info = {"A": a_info, "B": b_info}.get(final_tag)
                if chosen_info is None:
                    chosen_info = await check(final_tag, final_code)
                fig = str(chosen_info.get("FIG", ""))
                if fig and not chosen_info.get("ERR"):
                    reflect = await gen(
                        GPT_5_4_MINI, reflect_prompt(problem, final_code, fig), max_tokens=800
                    )
                    if reflect and not (
                        reflect.strip().upper().startswith("OK") and "<code>" not in reflect
                    ):
                        fixed = extract_code(reflect)
                        if fixed and norm_code(fixed) != norm_code(final_code):
                            fixed_info = await check("MPLFIX", fixed)
                            if is_clean(fixed_info):
                                print("  adopted matplotlib reflection fix")
                                final_code, final_tag = fixed, "MPLFIX"
        else:
            # No sandbox verification available (e.g. skeleton needs the hidden
            # test env). Prefer A; consult the arbiter only on real disagreement.
            a_c = bool(a_code) and compiles_ok(skeleton, a_code)
            b_c = bool(b_code) and compiles_ok(skeleton, b_code)
            if not a_code and b_code:
                final_code, final_tag = b_code, "B"
            elif a_c and b_c and norm_code(a_code) != norm_code(b_code):
                arb = extract_code(
                    await gen(
                        GPT_5_4,
                        arbiter_prompt(problem, a_code, "not executed", b_code, "not executed"),
                        max_tokens=1000,
                    )
                )
                if arb and compiles_ok(skeleton, arb):
                    final_code, final_tag = arb, "ARB"
                print(f"  unverified arbitration -> {final_tag}")
            elif not a_c and b_c:
                final_code, final_tag = b_code, "B"

        # Style pass: try to eliminate for/while tokens from the final answer.
        if not mpl and final_code and has_loop_tokens(final_code):
            hint = bool(IDIOM_HINT_RE.search(problem))
            rewritten = extract_code(await gen(GPT_5_4_MINI, rewrite_prompt(problem, final_code)))
            if (
                rewritten
                and not has_loop_tokens(rewritten)
                and compiles_ok(skeleton, rewritten)
            ):
                adopt = False
                if verifiable:
                    rw_info = await check("RW", rewritten)
                    if is_clean(rw_info):
                        if final_tag in ("A", "B", "ARB"):
                            same_val = True if mpl else await compare(final_tag, "RW")
                            adopt = same_val or hint
                        else:
                            adopt = hint
                else:
                    adopt = hint
                if adopt:
                    print(f"  adopted loop-free rewrite (hint={hint})")
                    final_code = rewritten

        if not final_code:
            final_code = extract_code(resp_a) or extract_code(resp_b)

        return emit(final_code, final_tag)

    return solve
