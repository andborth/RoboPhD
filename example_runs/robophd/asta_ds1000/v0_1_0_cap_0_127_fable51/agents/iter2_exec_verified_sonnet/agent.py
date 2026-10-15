"""exec-verified-sonnet: DS-1000 solver.

Pipeline per sample:
  1. parse the prompt (skeleton code, target names, function-body context)
  2. generate a candidate with Claude Sonnet 4.6 + DS-1000-specific guidelines
  3. execute skeleton + candidate in the sandbox (subprocess, fresh namespace)
  4. repair on crash (<= 2 rounds)
  5. execution-informed review (KEEP or corrected block; crashing revisions discarded)
  6. loop guard: vectorised rewrite if `for`/`while` tokens remain
  7. emit a clean <code>...</code> block
"""

import io
import json
import re
import tokenize

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6

# ----------------------------------------------------------------------------
# cost bookkeeping (approximate; only for diagnostics / soft cap)
# ----------------------------------------------------------------------------
PRICE_IN = 3.00 / 1e6
PRICE_OUT = 15.00 / 1e6
SOFT_CAP = 0.11  # skip optional steps beyond this estimated spend per sample


class Budget:
    def __init__(self):
        self.spent = 0.0
        self.calls = 0

    def add(self, resp):
        self.calls += 1
        try:
            u = resp.usage
            i = getattr(u, "input_tokens", 0) or 0
            o = getattr(u, "output_tokens", 0) or 0
            r = getattr(u, "reasoning_tokens", 0) or 0
            self.spent += i * PRICE_IN + (o + r) * PRICE_OUT
        except Exception:
            self.spent += 0.03


# ----------------------------------------------------------------------------
# prompt parsing
# ----------------------------------------------------------------------------
MARKER_RE = re.compile(r"^[ \t]*#*[ \t]*(BEGIN SOLUTION|SOLUTION START).*$", re.M)
TAIL_RE = re.compile(r"\n*Write the remaining python code to append.*$", re.S)


def parse_prompt(text: str):
    """Return (skeleton_code, marker_line, targets, func_name, in_function)."""
    body = TAIL_RE.sub("", text)
    m = MARKER_RE.search(body)
    pre = body[: m.start()] if m else body
    marker_line = m.group(0) if m else ""

    if "<code>" in pre:
        blocks = []
        pos = 0
        while True:
            s = pre.find("<code>", pos)
            if s < 0:
                break
            s += len("<code>")
            e = pre.find("</code>", s)
            if e < 0:
                blocks.append(pre[s:])
                break
            blocks.append(pre[s:e])
            pos = e + len("</code>")
        skeleton = "\n".join(b.strip("\n") for b in blocks)
        outside = re.sub(r"<code>.*?(</code>|$)", "", pre, flags=re.S)
    else:
        # matplotlib style: everything after "A:" (or the whole thing) is code
        k = pre.find("\nA:")
        skeleton = pre[k + 3:] if k >= 0 else pre
        skeleton = skeleton.strip("\n")
        outside = ""

    targets = []
    for mm in re.finditer(r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=\s*\.\.\.", outside, re.M):
        targets += [t.strip() for t in mm.group(1).split(",")]
    for mm in re.finditer(r"put (?:the )?\w+ in [`'\"]?([A-Za-z_]\w*)[`'\"]?", outside):
        if mm.group(1) not in ("this", "these", "the", "a"):
            targets.append(mm.group(1))
    func_name = None
    fm = re.search(r"define (?:a )?function named [`'\"]?([A-Za-z_]\w*)", outside)
    if fm:
        func_name = fm.group(1)
    for mm in re.finditer(r"example variable [`'\"]?([A-Za-z_]\w*)", outside):
        targets.append(mm.group(1))
    # de-dup, keep order
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]

    in_function = bool(marker_line) and marker_line[:1] in (" ", "\t")
    if in_function:
        dm = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", skeleton, re.M)
        if dm:
            func_name = dm[-1]
    return skeleton, marker_line, targets, func_name, in_function


def first_assigned_names(skeleton: str):
    names = []
    for line in skeleton.splitlines():
        mm = re.match(r"^([A-Za-z_]\w*)\s*=[^=]", line)
        if mm:
            names.append(mm.group(1))
    return names


# ----------------------------------------------------------------------------
# code extraction / cleanup
# ----------------------------------------------------------------------------
def extract_code(text: str) -> str:
    s = text or ""
    m = re.search(r"<code>(.*?)</code>", s, re.S)
    if m:
        code = m.group(1)
    else:
        m = re.search(r"<code>(.*)$", s, re.S)
        if m:
            code = m.group(1)
        else:
            fm = re.search(r"```(?:python|py)?\n(.*?)```", s, re.S)
            code = fm.group(1) if fm else s
    code = re.sub(r"```(?:python|py)?", "", code)
    lines = []
    for ln in code.split("\n"):
        if re.match(r"^\s*#*\s*(BEGIN|END) SOLUTION\s*$", ln):
            continue
        if re.match(r"^\s*#\s*SOLUTION (START|END)\s*$", ln):
            continue
        lines.append(ln.rstrip())
    code = "\n".join(lines).strip("\n")
    return code


def fix_indent(code: str, in_function: bool) -> str:
    if not in_function or not code.strip():
        return code
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    if first[:1] not in (" ", "\t"):
        code = "\n".join(("    " + ln) if ln.strip() else ln for ln in code.splitlines())
    return code


def has_loop_tokens(code: str) -> bool:
    try:
        toks = [t.string for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type == tokenize.NAME]
    except Exception:
        return bool(re.search(r"\b(for|while)\b", code))
    return "for" in toks or "while" in toks


# ----------------------------------------------------------------------------
# sandbox execution
# ----------------------------------------------------------------------------
CHECK_SNIPPET = r'''
print("<<<CHECK>>>")
def _show(v):
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import pandas as _pd
    except Exception:
        _pd = None
    t = type(v).__module__ + "." + type(v).__name__
    try:
        if _pd is not None and isinstance(v, _pd.DataFrame):
            s = v.to_string(max_rows=30, max_cols=25)
            s += "\n[shape] %s\n[dtypes] %s\n[index] %s %s\n[columns] %s" % (
                v.shape, ", ".join("%s:%s" % (c, d) for c, d in v.dtypes.items()),
                type(v.index).__name__, list(v.index[:12]), list(v.columns[:25]))
        elif _pd is not None and isinstance(v, _pd.Series):
            s = v.to_string(max_rows=30)
            s += "\n[shape] %s [dtype] %s [name] %r\n[index] %s %s" % (
                v.shape, v.dtype, v.name, type(v.index).__name__, list(v.index[:12]))
        elif _np is not None and isinstance(v, _np.ndarray):
            s = _np.array2string(v, max_line_width=120, threshold=200)
            s += "\n[shape] %s [dtype] %s" % (v.shape, v.dtype)
        else:
            s = repr(v)
            if len(s) > 2500:
                s = s[:2500] + " ...[truncated]"
            if hasattr(v, "shape"):
                try:
                    s += "\n[shape] %s" % (tuple(v.shape),)
                except Exception:
                    pass
            if hasattr(v, "dtype"):
                try:
                    s += " [dtype] %s" % (v.dtype,)
                except Exception:
                    pass
    except Exception as _e:
        s = "<repr failed: %r>" % (_e,)
    return "type=%s\n%s" % (t, s)

_targets = __TARGETS__
_func = __FUNC__
_inputs = __INPUTS__
_INFUNC = __INFUNC__
_g = globals()
for _n in _targets:
    if _n in _g and _n != _func:
        print("--- %s:" % _n)
        print(_show(_g[_n]))
    elif _n != _func:
        print("--- %s: NOT DEFINED" % _n)
if _func:
    if _func in _g and callable(_g[_func]):
        import inspect as _ins
        try:
            _sig = _ins.signature(_g[_func])
            print("--- function %s%s defined" % (_func, _sig))
            _done = False
            if not _INFUNC and _inputs and _inputs[0] in _g:
                # DS-1000 convention: the hidden test calls X(<primary input>) with one argument
                try:
                    _r = _g[_func](_g[_inputs[0]])
                    print("--- call %s(%s) [as the hidden test does] ->" % (_func, _inputs[0]))
                    print(_show(_r))
                    _done = True
                except TypeError as _e:
                    print("--- WARNING: calling %s(%s) with ONE positional argument (as the hidden test does) failed: %r" % (_func, _inputs[0], _e))
            if not _done:
                _args = []
                _k = 0
                for _p in _sig.parameters.values():
                    if _p.default is not _ins.Parameter.empty:
                        break
                    if _p.name in _g:
                        _args.append(_g[_p.name])
                    elif _k < len(_inputs) and _inputs[_k] in _g:
                        _args.append(_g[_inputs[_k]])
                    _k += 1
                _r = _g[_func](*_args)
                print("--- call %s(%d positional arg(s)) ->" % (_func, len(_args)))
                print(_show(_r))
        except Exception as _e:
            print("--- calling %s failed: %r" % (_func, _e))
    else:
        print("--- function %s: NOT DEFINED" % _func)
if not _targets and not _func:
    try:
        import matplotlib.pyplot as _plt
        _figs = _plt.get_fignums()
        print("--- figures: %s" % _figs)
        for _fn in _figs:
            _f = _plt.figure(_fn)
            for _ax in _f.get_axes():
                _leg = _ax.get_legend()
                print("axes: title=%r xlabel=%r ylabel=%r xlim=%s ylim=%s lines=%d patches=%d collections=%d texts=%d legend=%s xticks=%s yticks=%s" % (
                    _ax.get_title(), _ax.get_xlabel(), _ax.get_ylabel(),
                    _ax.get_xlim(), _ax.get_ylim(), len(_ax.get_lines()),
                    len(_ax.patches), len(_ax.collections), len(_ax.texts),
                    [t.get_text() for t in _leg.get_texts()] if _leg else None,
                    list(_ax.get_xticks())[:8], list(_ax.get_yticks())[:8]))
    except Exception as _e:
        print("--- figure dump failed: %r" % (_e,))
print("<<<END>>>")
'''

RUNNER = r'''
import subprocess, sys, json, os, tempfile
_prog = json.loads(%s)
_dir = "/workspace"
try:
    os.makedirs(_dir, exist_ok=True)
    with open(os.path.join(_dir, "_cand.py"), "w") as _f:
        _f.write(_prog)
except Exception:
    _dir = tempfile.mkdtemp()
    with open(os.path.join(_dir, "_cand.py"), "w") as _f:
        _f.write(_prog)
try:
    _r = subprocess.run([sys.executable, os.path.join(_dir, "_cand.py")], capture_output=True, text=True,
                        timeout=150, cwd=_dir)
    _out, _err, _rc = _r.stdout, _r.stderr, _r.returncode
except subprocess.TimeoutExpired as _e:
    _out, _err, _rc = (_e.stdout or ""), "TIMEOUT: the program did not finish within 150s", -1
if isinstance(_out, bytes): _out = _out.decode("utf-8", "replace")
if isinstance(_err, bytes): _err = _err.decode("utf-8", "replace")
print("@@JSON@@" + json.dumps({"rc": _rc, "stdout": _out[-6000:], "stderr": _err[-4000:]}))
'''


def build_program(skeleton, marker_line, code, targets, func_name, inputs, in_function=False):
    check = (CHECK_SNIPPET
             .replace("__TARGETS__", repr(list(targets)))
             .replace("__FUNC__", repr(func_name))
             .replace("__INPUTS__", repr(list(inputs)))
             .replace("__INFUNC__", repr(bool(in_function))))
    header = ("import warnings\nwarnings.filterwarnings('ignore')\n"
              "try:\n    import matplotlib\n    matplotlib.use('Agg')\nexcept Exception:\n    pass\n")
    parts = [header, skeleton]
    if marker_line and marker_line.strip().startswith("#"):
        parts.append(marker_line)
    n_pre = len("\n".join(parts).splitlines())  # lines before the candidate starts
    parts.append(code)
    parts.append(check)
    return "\n".join(parts) + "\n", n_pre


def _strip_noise(err: str) -> str:
    keep = []
    for ln in err.splitlines():
        if "Warning" in ln and ("warnings.warn" in ln or ln.strip().startswith("/")):
            continue
        if ln.strip() == "warnings.warn(":
            continue
        keep.append(ln)
    return "\n".join(keep)


async def run_candidate(py, skeleton, marker_line, code, targets, func_name, inputs, in_function=False):
    """Return dict(ok: bool|None, report: str, out: str). ok=None → could not execute at all."""
    if py is None:
        return {"ok": None, "report": "(sandbox unavailable)", "out": ""}
    prog, n_pre = build_program(skeleton, marker_line, code, targets, func_name, inputs, in_function)
    runner = RUNNER % json.dumps(json.dumps(prog))
    try:
        raw = await py(code=runner)
    except Exception as e:
        return {"ok": None, "report": f"(sandbox error: {e!r})", "out": ""}
    raw = str(raw)
    k = raw.rfind("@@JSON@@")
    if k < 0:
        return {"ok": None, "report": "(no execution report)\n" + raw[-1500:], "out": ""}
    try:
        line = raw[k + 8:].strip().splitlines()[0]
        res = json.loads(line)
    except Exception:
        return {"ok": None, "report": "(unparseable execution report)\n" + raw[-1500:], "out": ""}
    stdout, stderr, rc = res["stdout"], _strip_noise(res["stderr"]), res["rc"]
    if "<<<CHECK>>>" in stdout:
        pre, post = stdout.split("<<<CHECK>>>", 1)
        post = post.replace("<<<END>>>", "").strip()
        rep = ""
        if pre.strip():
            rep += "[program stdout]\n" + pre.strip()[-1500:] + "\n"
        rep += "[values after running the solution]\n" + post[-5000:]
        if rc != 0:
            rep += "\n[stderr]\n" + stderr[-2500:]
        ok = rc == 0 and "NOT DEFINED" not in post
        return {"ok": ok, "report": rep, "out": post}
    rep = "[program crashed: exit code %s]\n" % rc
    if stdout.strip():
        rep += "[stdout]\n" + stdout.strip()[-1500:] + "\n"
    rep += "[stderr]\n" + stderr[-3000:]
    # crash raised from the skeleton itself (e.g. load_data undefined, missing module)?
    lines = [int(x) for x in re.findall(r'_cand\.py", line (\d+)', stderr)]
    if lines and lines[-1] <= n_pre:
        return {"ok": None, "report": "(the skeleton itself cannot run here)\n" + rep, "out": ""}
    return {"ok": False, "report": rep, "out": ""}


def skeleton_unrunnable(report: str) -> bool:
    """Crash caused by the skeleton itself (e.g. load_data undefined), not by the candidate."""
    return "load_data" in report and "NameError" in report


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Python data scientist solving DS-1000 benchmark problems.

You are given a problem: a question (often a StackOverflow-style post with an example and the desired output) and a code skeleton. You write ONLY the code that goes in the solution slot. Your code is appended verbatim to the skeleton and then executed by a hidden test harness on SEVERAL DIFFERENT inputs (other values, shapes, sizes, column names, lengths, etc.); the harness then compares the designated output variable(s) to a reference solution. Score is all-or-nothing.

OUTPUT FORMAT (strict):
- Respond with exactly one block: <code> ... </code> containing executable Python only. No prose, no markdown fences, no BEGIN/END SOLUTION markers.
- Do not repeat the skeleton code (imports/data setup already exist), except for extra imports you need.
- No print statements, no plt.show() unless already in the skeleton, no debugging output, no extra side effects.

RULES THAT DETERMINE PASS/FAIL (learned from the grader):
1. Assign exactly the variable(s) the skeleton asks for (e.g. `result = ...`, `df = ...`, `proba = ...`, "put score in `b`"). Keep the exact names. Store the raw/natural output of the operation asked for — do NOT post-process beyond the question (e.g. if asked for probabilities, assign the full predict_proba output, not one column). Do ONLY what the final instruction asks; no additional demonstrations, no extra mutations of the given objects.
2. Match the desired output shown in the question EXACTLY: type (DataFrame vs Series vs ndarray vs list vs tensor), shape/orientation, column names and order, index (kept vs reset), element order, dtype where visible. The sample output in the question is the spec — reproduce it even if it looks arbitrary (e.g. an unusual column ordering).
3. Be general: the hidden tests reuse your code on other inputs. Never hard-code example values, lengths, shapes, bit widths, or names that are not fixed by the problem; derive everything from the given variables. (E.g. do not use uint8/unpackbits for "m bits" when m may exceed 8; do not assume the example number of rows.)
4. Prefer idiomatic, vectorised library calls. Some hidden tests tokenise your code and FAIL it if it contains the tokens `for` or `while` — this includes list/dict comprehensions and generator expressions. Avoid loops and comprehensions entirely unless the problem genuinely cannot be solved without iteration. If the question names a specific function/method/approach to use, use it (some tests check that the function name appears in your code).
5. Function problems: if the skeleton shows a function header (e.g. `def f(a=example_a):` ending at BEGIN SOLUTION), write ONLY the indented function body (including the `return`). If the instructions say "define function named `X` as solution", the hidden test calls `X(<primary input>)` with ONE positional argument (the main input variable from the skeleton) and the other given variables (parameters/thresholds) are available as globals — so make the primary input the only required parameter and read the others from the enclosing scope (or give them keyword defaults equal to the given variables).
6. When the output variable has the same name as an input (e.g. `df = ...`, `a = ...`), assign the transformed object to that name. Prefer operations that return new objects; if you must mutate in place, make sure the named variable ends up holding the right object.
7. sklearn: never call load_data() again; use default hyper-parameters unless the question specifies them; honour requested random_state values; use the estimator/approach named in the question.
8. Use current APIs for the installed library versions (given below): no removed pandas methods (`DataFrame.append`, `Series.iteritems`, `.ix`), no deprecated numpy aliases (`np.float`, `np.int`), etc.
9. matplotlib: act on the current figure/axes via `plt` (or the given `ax`/`fig`) exactly as the comment asks (labels, titles, legend, ticks, colours, etc.); do not create new figures unless asked; do not call plt.close().
10. torch/tensorflow: return tensors (not numpy) unless asked; keep dtypes; use the API of the version stated in the question.

Think carefully about what the question is literally asking (read phrases like "before/after", "rows/columns", "ascending/descending", "inclusive/exclusive" precisely), then write the simplest correct, general solution."""


def user_prompt(problem: str, library: str, versions: str) -> str:
    return (f"Library: {library}\nInstalled versions: {versions}\n\n"
            f"{problem}\n\n"
            "Respond with exactly one <code>...</code> block containing only the code to append.")


def repair_prompt(problem, code, report):
    return (f"The following DS-1000 problem was attempted. The candidate solution CRASHED when appended to "
            f"the skeleton and executed on the example inputs.\n\n### Problem\n{problem}\n\n"
            f"### Candidate solution\n<code>\n{code}\n</code>\n\n### Execution report\n{report}\n\n"
            "Fix the solution. Keep all the rules (exact target variable names, general over other inputs, "
            "no loops/comprehensions, no extra side effects or prints). Respond with exactly one "
            "<code>...</code> block containing only the corrected code to append.")


def review_prompt(problem, code, report):
    return (f"Review a candidate solution to a DS-1000 problem. The candidate was appended to the skeleton "
            f"and executed on the example inputs; the resulting values are shown.\n\n### Problem\n{problem}\n\n"
            f"### Candidate solution\n<code>\n{code}\n</code>\n\n### Execution report\n{report}\n\n"
            "Check, in order:\n"
            "(a) Does the produced value match what the question asks for and any desired output shown "
            "in the question — type, shape/orientation, column names and order, index, element order, "
            "values, dtype (strings vs numbers)? Compare the shown sample output cell by cell if one exists.\n"
            "(b) Are the exact requested variable name(s) assigned, with the raw result (no extra "
            "post-processing such as selecting a column, rounding, converting to list)?\n"
            "(c) Does the code do ONLY what is asked — no additional mutations, demonstrations, prints?\n"
            "(d) Will it work unchanged on other hidden inputs (different sizes/values/names/number of "
            "rows, m > 8, empty groups, etc.) — nothing hard-coded from the example?\n"
            "(e) If a function was requested: is the signature what the hidden test will call "
            "(`f(<primary input>)` with other given names as globals, or exactly the given header)?\n"
            "(f) Does it avoid `for`/`while` (including comprehensions) unless iteration is unavoidable, "
            "and use any function the question explicitly names?\n\n"
            "If every check passes, respond with exactly: KEEP\n"
            "Otherwise respond with ONLY a corrected <code>...</code> block (full replacement code to "
            "append, no prose). Be conservative: change the code only if you identify a concrete defect; "
            "do not rewrite working code for style alone.")


def loop_prompt(problem, code):
    return (f"The following DS-1000 solution uses `for`/`while` (loops or comprehensions). The hidden "
            f"grader may reject any code containing those tokens. Rewrite it using vectorised library "
            f"operations (numpy/pandas/torch/tf functions, broadcasting, np.vectorize, apply/map, "
            f"itertools, etc.) with IDENTICAL behaviour and output. Keep the same variable names.\n\n"
            f"### Problem\n{problem}\n\n### Current solution\n<code>\n{code}\n</code>\n\n"
            "Respond with exactly one <code>...</code> block containing only the rewritten code to "
            "append, containing no `for` or `while` tokens anywhere.")


async def llm(model, messages, budget: Budget, effort=None, max_tokens=3000):
    cfg = GenerateConfig(max_tokens=max_tokens)
    if effort:
        cfg = GenerateConfig(max_tokens=max_tokens, reasoning_effort=effort)
    resp = await model.generate(messages, config=cfg)
    budget.add(resp)
    return resp.completion or ""


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sid = state.sample_id
        library = state.metadata.get("library", "?")
        problem = state.input if isinstance(state.input, str) else str(state.input)
        print(f"[{sid}] library={library}")
        budget = Budget()

        py = None
        try:
            py = next(t for t in state.tools if ToolDef(t).name == "python_session")
        except Exception:
            print("  python_session unavailable")

        skeleton, marker_line, targets, func_name, in_function = parse_prompt(problem)
        inputs = first_assigned_names(skeleton)
        print(f"  targets={targets} func={func_name} in_function={in_function} inputs={inputs[:4]}")

        # library versions from the sandbox (free)
        versions = "pandas 2.x, numpy 1.26+, scikit-learn 1.x, scipy 1.x, matplotlib 3.x"
        if py is not None:
            try:
                lib = library.lower()
                vcode = ("import importlib\n"
                         "mods=['numpy','pandas','scipy','sklearn','matplotlib']"
                         + ("+['torch']" if "torch" in lib else "")
                         + ("+['tensorflow']" if "tensorflow" in lib else "") + "\n"
                         "out=[]\n"
                         "for m in mods:\n"
                         "    try:\n"
                         "        out.append(m+' '+importlib.import_module(m).__version__)\n"
                         "    except Exception as e:\n"
                         "        out.append(m+' ?')\n"
                         "print('VERS:'+', '.join(out))")
                v = str(await py(code=vcode))
                k = v.find("VERS:")
                if k >= 0:
                    versions = v[k + 5:].strip().splitlines()[0]
            except Exception as e:
                print(f"  version probe failed: {e!r}")
        print(f"  versions: {versions}")

        async def execute(code):
            return await run_candidate(py, skeleton, marker_line, code, targets, func_name, inputs, in_function)

        # ---- 1. generate -------------------------------------------------
        msgs = [ChatMessageSystem(content=SYSTEM_PROMPT),
                ChatMessageUser(content=user_prompt(problem, library, versions))]
        code = ""
        try:
            code = extract_code(await llm(CLAUDE_SONNET_4_6, msgs, budget, effort="medium", max_tokens=4000))
        except Exception as e:
            print(f"  generation failed: {e!r}")
        if not code.strip():
            try:
                code = extract_code(await llm(CLAUDE_SONNET_4_6, msgs, budget, max_tokens=4000))
            except Exception as e:
                print(f"  generation retry failed: {e!r}")
        code = fix_indent(code, in_function)
        print(f"  candidate:\n{code}")

        # ---- 2. execute + repair ----------------------------------------
        run = await execute(code)
        print(f"  exec ok={run['ok']}\n{run['report'][:1500]}")
        can_execute = run["ok"] is not None and not skeleton_unrunnable(run["report"])
        attempts = 0
        while can_execute and run["ok"] is False and attempts < 2 and budget.spent < SOFT_CAP:
            attempts += 1
            try:
                fixed = fix_indent(extract_code(await llm(
                    CLAUDE_SONNET_4_6,
                    [ChatMessageSystem(content=SYSTEM_PROMPT),
                     ChatMessageUser(content=repair_prompt(problem, code, run["report"]))],
                    budget, effort="low", max_tokens=4000)), in_function)
            except Exception as e:
                print(f"  repair call failed: {e!r}")
                break
            if not fixed.strip():
                break
            frun = await execute(fixed)
            print(f"  repair {attempts} ok={frun['ok']}\n{frun['report'][:1200]}")
            code, run = fixed, frun
            if frun["ok"]:
                break

        # ---- 3. execution-informed review --------------------------------
        if budget.spent < SOFT_CAP and code.strip():
            report = run["report"] if can_execute else "(the skeleton needs hidden data, e.g. load_data(); the code could not be executed here — review statically)"
            try:
                rv = await llm(CLAUDE_SONNET_4_6,
                               [ChatMessageSystem(content=SYSTEM_PROMPT),
                                ChatMessageUser(content=review_prompt(problem, code, report))],
                               budget, effort="low", max_tokens=4000)
            except Exception as e:
                print(f"  review call failed: {e!r}")
                rv = "KEEP"
            if "<code>" in rv:
                revised = fix_indent(extract_code(rv), in_function)
                if revised.strip() and revised.strip() != code.strip():
                    rrun = await execute(revised) if can_execute else {"ok": None, "report": "", "out": ""}
                    print(f"  review proposed revision ok={rrun['ok']}\n{revised}\n{rrun['report'][:1200]}")
                    if rrun["ok"] or (rrun["ok"] is None) or (rrun["ok"] is False and run["ok"] is False):
                        code, run = revised, rrun
                    else:
                        print("  revision crashed; keeping original")
            else:
                print("  review: KEEP")

        # ---- 4. loop guard ----------------------------------------------
        if has_loop_tokens(code) and budget.spent < SOFT_CAP and not in_function:
            try:
                nl = fix_indent(extract_code(await llm(
                    CLAUDE_SONNET_4_6,
                    [ChatMessageSystem(content=SYSTEM_PROMPT),
                     ChatMessageUser(content=loop_prompt(problem, code))],
                    budget, effort="low", max_tokens=4000)), in_function)
            except Exception as e:
                print(f"  loop rewrite failed: {e!r}")
                nl = ""
            if nl.strip() and not has_loop_tokens(nl):
                if can_execute:
                    nrun = await execute(nl)
                    same = nrun["ok"] and nrun["out"].strip() == run["out"].strip()
                    print(f"  loop-free rewrite ok={nrun['ok']} same_output={same}")
                    if nrun["ok"] and (same or run["ok"] is False):
                        code, run = nl, nrun
                    elif nrun["ok"] and not same:
                        print("  loop-free rewrite differs in output; keeping original")
                else:
                    code = nl
            else:
                print("  loop-free rewrite still has loops or empty; keeping original")

        state.output.completion = f"<code>\n{code}\n</code>"
        print(f"  final:\n{code}")
        print(f"  llm_calls={budget.calls} est_cost=${budget.spent:.4f}")
        return state

    return solve
