"""DS-1000 solver: iter3-verified-mini.

Design (see reasoning.md):
  1. Guided prompt -> single GPT_5_4_MINI generation (cheap).
  2. Deterministic sanitizers: tag/marker stripping, func-mode body fixer
     (indent an unindented body, truncate trailing top-level junk), compile gate.
  3. Free sandbox execution of skeleton+candidate (error / target presence /
     value summary / matplotlib figure state). Skeletons that cannot run
     standalone (undefined load_data()) get a one-call synthesized setup so
     verification is still possible.
  4. Traceback-guided repair on crash/missing target (mini); if still broken,
     ONE GPT_5_4 escalation with the execution evidence.
  5. Example-anchored verify pass: mini sees problem + code + executed value
     (or figure state) and either replies OK or emits a correction, adopted
     only if it executes cleanly and changes the value.
  6. Loop-token style pass: vectorized rewrite adopted only if verified
     value-equal or an idiom hint is present.

Everything degrades gracefully: on any harness/LLM failure the agent falls
back to the best earlier candidate. Only model calls are metered; the
sandbox is free.
"""

import ast
import asyncio
import io
import re
import tokenize

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI

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
6. These problems come from StackOverflow and the grader follows the accepted answer. Prefer the most direct, standard call to the library named in the question, and do NOT add preprocessing/transformation steps the problem does not explicitly ask for (e.g. if asked to cluster a given matrix with scipy, feed the matrix to the clustering API as-is rather than converting it first).
"""

MPL_GUIDANCE = """7. Matplotlib: follow the comment instructions literally.
Marker codes: 'd' = thin diamond, 'D' = diamond, '*' = star, 's' = square, 'o' = circle, '+' = plus, 'x' = x, '|' = vline, '^'/'v' = triangles.
A "hatch" is a fill pattern, not a marker: "star hatch" means passing hatch='*' (scatter/bar accept hatch=...).
Distinguish marker/line/face/edge colors, markersize vs linewidth, and do exactly what is asked - nothing more.
"""

FUNC_GUIDANCE = """7. IMPORTANT: the setup code ends inside a function definition (### BEGIN SOLUTION appears inside a def). Output ONLY the remaining indented body of that function - indent every line with 4 spaces and end with the appropriate return statement. Do not repeat the def line and write NO top-level (unindented) code: no prints, no example calls, nothing after the function body.
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
# Parsing / sanitizing helpers
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
    code = re.sub(r"^\s*#*\s*(BEGIN|END) SOLUTION.*$", "", code, flags=re.M)
    return code.strip("\n").rstrip()


def parse_problem(prompt: str):
    """Return (skeleton_code, target_variable, func_mode).

    func_mode marks the DS-1000 function-completion format: the last <code>
    block before '### BEGIN SOLUTION' is unclosed and ends inside a def whose
    body the answer must complete with indented code.
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


def fix_func_body(code: str) -> str:
    """Sanitize a function-completion body.

    - If the whole body is unindented, indent it by 4 spaces.
    - Drop a repeated leading `def ...:` line (dedent would be handled by the
      compile gate + repair if the model indented relative to it).
    - Indent stray leading top-level lines (usually imports meant for the body).
    - Truncate at the first top-level statement AFTER indented body lines
      (trailing junk like `print(f())` that crashes the hidden test).
    """
    if not code:
        return code
    lines = code.splitlines()
    nonempty = [l for l in lines if l.strip()]
    if not nonempty:
        return code

    def indented(l):
        return l.startswith((" ", "\t"))

    if all(not indented(l) for l in nonempty):
        return "\n".join(("    " + l if l.strip() else l) for l in lines)

    out = []
    seen_indented = False
    for l in lines:
        if l.strip() and not indented(l):
            if seen_indented:
                break  # trailing top-level junk after the body
            if l.lstrip().startswith("def "):
                continue  # repeated def line
            out.append("    " + l)  # leading top-level line -> pull into body
        else:
            if l.strip():
                seen_indented = True
            out.append(l)
    return "\n".join(out).rstrip()


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


def synth_prompt(problem: str, skeleton: str, err: str) -> str:
    return f"""The setup code for a data-science problem cannot run here (for example it calls an undefined load_data()).

Problem:
{clip(problem, 4000)}

Setup code:
{clip(skeleton, 1500)}

Error:
{clip(err, 600)}

Write replacement setup code that defines the SAME variables with the SAME names, filling in concrete example data taken from the problem text (or small plausible data of the right type/shape if none is shown). Do NOT solve the problem itself. Output ONLY the setup code inside <code></code> tags."""


def escalate_prompt(problem: str, code: str, desc: str, target: str) -> str:
    return f"""A candidate solution to the problem below was executed after the problem's setup code and it is broken.

Problem:
{clip(problem, 5000)}

Candidate code:
{clip(code, 2000)}

Execution result:
{clip(desc, 1200)}

Write the correct solution. It is appended after the setup code (do not repeat the setup lines) and must assign the final answer to `{target}`. Check it against any example output shown in the problem. Reply with at most 2 short sentences of analysis, then the solution inside <code></code> tags."""


def verify_prompt(problem: str, code: str, evidence: str, target: str, mpl: bool) -> str:
    what = "Actual figure state produced:" if mpl else f"Actual value of `{target}` produced:"
    extra = MARKER_CHEATSHEET + "\n\n" if mpl else ""
    return f"""A candidate solution to a data-science problem is shown below, together with what it actually produced when executed after the problem's setup code (using the example data).

Problem:
{clip(problem, 5000)}

Candidate solution (appended after the setup code):
{clip(code, 2000)}

{what}
{clip(evidence, 1300)}

{extra}Check the candidate against EVERY requirement of the problem, and ESPECIALLY against any example/expected output shown in the problem text - the values, dtype, shape, ordering and index/columns must match the example EXACTLY. Hidden tests rerun the code on different data of the same kind, so it must generalize and must not hardcode example values. Also honor any usage constraint ("use function X", "without X", no explicit loops when a vectorized way is implied).

If the candidate is correct, reply with exactly OK.
If not, reply with the full corrected solution inside <code></code> tags (executable Python only, appended after the setup code)."""


def rewrite_prompt(problem: str, code: str) -> str:
    return f"""Rewrite this solution so its source contains NO `for` or `while` tokens anywhere (no loops, no comprehensions, no generator expressions) - use vectorized library operations instead. Keep the behavior and the indentation level identical.

Problem:
{clip(problem, 4000)}

Current solution:
{clip(code, 2000)}

Output ONLY the rewritten Python code inside <code></code> tags."""


def is_ok_reply(text: str) -> bool:
    return bool(text) and text.strip().upper().startswith("OK") and "<code>" not in text


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

        calls = 0  # LLM call budget guard (hard cap on the cost tail)
        MAX_CALLS = 7

        async def lgen(model, prompt, max_tokens=1200):
            nonlocal calls
            if calls >= MAX_CALLS:
                print("  call budget exhausted")
                return ""
            calls += 1
            return await gen(model, prompt, max_tokens)

        def emit(code: str, tag) -> TaskState:
            if not code:
                code = f"{target} = None"
            code = code.replace("<code>", "").replace("</code>", "").strip("\n")
            state.output.completion = f"<code>\n{code}\n</code>"
            print(f"  emitted {len(state.output.completion)} chars (tag={tag}, calls={calls})")
            return state

        # ---- Function-completion format: no standalone execution possible. ----
        if func_mode:
            code = fix_func_body(extract_code(await lgen(GPT_5_4_MINI, full_prompt)))
            if not code or not compiles_ok(skeleton, code):
                err = compile_err(skeleton, code) if code else "empty completion"
                note = (
                    "Remember: your code completes an open function body - every line "
                    "must be indented with 4 spaces, ending with a return statement. "
                    "Write NO top-level code (no prints, no example calls).\n"
                )
                fixed = fix_func_body(
                    extract_code(await lgen(GPT_5_4_MINI, repair_prompt(problem, code, err, note)))
                )
                if fixed and compiles_ok(skeleton, fixed):
                    code = fixed
                    print("  func repair adopted")
            # Style pass: compile-gate only (no execution), so require an idiom hint.
            if code and has_loop_tokens(code) and IDIOM_HINT_RE.search(problem):
                rw = fix_func_body(extract_code(await lgen(GPT_5_4_MINI, rewrite_prompt(problem, code))))
                if rw and not has_loop_tokens(rw) and compiles_ok(skeleton, rw):
                    code = rw
                    print("  func loop-free rewrite adopted")
            return emit(code, "func")

        # ---- Standard format: generate + sandbox-verify. ----
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

        resp_a, harness_ok = await asyncio.gather(
            lgen(GPT_5_4_MINI, full_prompt),
            install_harness(),
        )
        code = extract_code(resp_a)
        print(f"  cand={len(code)}ch harness={harness_ok}")

        eff_skeleton = {"v": skeleton}

        async def check(tag: str, c: str) -> dict:
            cell = f"_ds_check({tag!r}, {eff_skeleton['v']!r}, {c!r}, {target!r}, mpl={mpl!r})"
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

        def is_clean(info) -> bool:
            if info is None or info.get("ERR") is not None:
                return False
            return True if mpl else bool(info.get("HASV"))

        def describe(info) -> str:
            if info is None:
                return "could not be executed in this environment"
            if info.get("ERR"):
                return "ERROR: " + str(info["ERR"])[:700]
            if mpl:
                return str(info.get("FIG", ""))[:1100]
            return str(info.get("VAL"))[:900]

        # -- Establish an executable skeleton (synthesize setup data if needed) --
        verifiable = harness_ok and bool(skeleton.strip()) and bool(code)
        if verifiable:
            skel_info = await check("S", "")
            if skel_info.get("ERR") is not None:
                print(f"  skeleton fails: {str(skel_info['ERR'])[-160:]}")
                synth = extract_code(
                    await lgen(
                        GPT_5_4_MINI,
                        synth_prompt(problem, skeleton, str(skel_info["ERR"])),
                        max_tokens=900,
                    )
                )
                verifiable = False
                if synth:
                    eff_skeleton["v"] = synth
                    skel_info2 = await check("S", "")
                    if skel_info2.get("ERR") is None:
                        verifiable = True
                        print("  synthesized setup adopted")
                    else:
                        eff_skeleton["v"] = skeleton
                        print("  synthesized setup also fails; unverified mode")

        info = None
        tag = "A"
        if verifiable:
            info = await check(tag, code)

            # -- Repair on crash / missing target variable --
            if not is_clean(info):
                err = info.get("ERR") or f"the code ran but never assigned the variable `{target}`"
                note = "" if mpl else f"Remember: assign the final answer to the variable `{target}`.\n"
                fixed = extract_code(
                    await lgen(GPT_5_4_MINI, repair_prompt(problem, code, str(err), note))
                )
                if fixed:
                    finfo = await check("R", fixed)
                    if is_clean(finfo) or info.get("ERR"):
                        code, info, tag = fixed, finfo, "R"
                        print("  repair adopted")

            # -- Escalate once to GPT_5_4 if still broken --
            if not is_clean(info):
                esc = extract_code(
                    await lgen(
                        GPT_5_4,
                        escalate_prompt(problem, code, describe(info), target),
                        max_tokens=1000,
                    )
                )
                if esc:
                    einfo = await check("E", esc)
                    if is_clean(einfo) or (info.get("ERR") and not einfo.get("ERR")):
                        code, info, tag = esc, einfo, "E"
                        print("  escalation adopted")
        else:
            # Unverified: compile gate + one repair on syntax errors only.
            if code and not compiles_ok(skeleton, code):
                err = compile_err(skeleton, code)
                fixed = extract_code(
                    await lgen(GPT_5_4_MINI, repair_prompt(problem, code, err, ""))
                )
                if fixed and compiles_ok(skeleton, fixed):
                    code, tag = fixed, "R"
                    print("  syntax repair adopted")

        # -- Example-anchored verify pass --
        if code:
            vresp = await lgen(
                GPT_5_4_MINI,
                verify_prompt(problem, code, describe(info), target, mpl),
                max_tokens=1000,
            )
            if vresp and not is_ok_reply(vresp):
                vfix = extract_code(vresp)
                if vfix and norm_code(vfix) != norm_code(code):
                    if verifiable:
                        vinfo = await check("V", vfix)
                        if is_clean(vinfo):
                            same = False
                            if is_clean(info) and not mpl:
                                same = await compare(tag, "V")
                            if not same or not is_clean(info):
                                code, info, tag = vfix, vinfo, "V"
                                print("  verify fix adopted")
                            else:
                                print("  verify fix value-identical; kept original")
                    elif compiles_ok(skeleton, vfix):
                        code, tag = vfix, "V-unexec"
                        print("  verify fix adopted (unverified)")

        # -- Loop-token style pass --
        if not mpl and code and has_loop_tokens(code):
            hint = bool(IDIOM_HINT_RE.search(problem))
            rw = extract_code(await lgen(GPT_5_4_MINI, rewrite_prompt(problem, code)))
            if rw and not has_loop_tokens(rw) and compiles_ok(skeleton, rw):
                adopt = False
                if verifiable:
                    rw_info = await check("RW", rw)
                    if is_clean(rw_info):
                        same_val = await compare(tag, "RW") if is_clean(info) else False
                        adopt = same_val or hint
                else:
                    adopt = hint
                if adopt:
                    code = rw
                    print(f"  loop-free rewrite adopted (hint={hint})")

        if not code:
            code = extract_code(resp_a)

        return emit(code, tag)

    return solve
