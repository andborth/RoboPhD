"""ensemble-exec-vote: DS-1000 solver (iteration 3).

Pipeline per sample:
  1. parse the prompt (skeleton code, target names, function-body context)
  2. generate THREE candidates in parallel from diverse strong models
     (GPT-5.4 medium reasoning, Claude Opus 4.8, Gemini 3.1 Pro) with a
     DS-1000-specific system prompt
  3. execute every candidate in the sandbox (subprocess, fresh namespace) and
     dump the target variable(s) with type/shape/dtype/index
  4. crashing candidates get one repair round each (cheap, Sonnet)
  5. vote: candidates whose executed outputs agree form a majority -> pick the
     loop-free one from the highest-priority model; no majority -> an Opus 4.8
     judge sees problem + candidates + executed outputs and picks/corrects
  6. loop guard: if the pick contains `for`/`while` tokens, prefer an agreeing
     loop-free candidate, else request vectorised rewrites (2 tries, 2 models)
     and adopt only if the output is unchanged
  7. emit a clean <code>...</code> block
"""

import asyncio
import io
import json
import re
import tokenize

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import (
    CLAUDE_OPUS_4_8,
    CLAUDE_SONNET_4_6,
    GEMINI_3_1_PRO_PREVIEW,
    GPT_5_4,
)

# ----------------------------------------------------------------------------
# cost bookkeeping (approximate; only for diagnostics / soft cap)
# ----------------------------------------------------------------------------
PRICES = {  # $/token (input, output)
    "gpt54": (2.50 / 1e6, 15.00 / 1e6),
    "opus": (5.00 / 1e6, 25.00 / 1e6),
    "gemini": (2.00 / 1e6, 12.00 / 1e6),
    "sonnet": (3.00 / 1e6, 15.00 / 1e6),
}
SOFT_CAP = 0.10  # skip optional LLM steps beyond this estimated spend per sample


class Budget:
    def __init__(self):
        self.spent = 0.0
        self.calls = 0
        self.log = []

    def add(self, resp, key):
        self.calls += 1
        pin, pout = PRICES.get(key, (5.0 / 1e6, 30.0 / 1e6))
        try:
            u = resp.usage
            i = getattr(u, "input_tokens", 0) or 0
            o = getattr(u, "output_tokens", 0) or 0
            r = getattr(u, "reasoning_tokens", 0) or 0
            c = i * pin + (o + r) * pout
            self.log.append(f"{key}: in={i} out={o} reas={r} ${c:.4f}")
        except Exception:
            c = 0.03
            self.log.append(f"{key}: usage unavailable, assumed ${c:.4f}")
        self.spent += c


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

You are given a problem: a question (usually a StackOverflow post with an example and the desired output) and a code skeleton. You write ONLY the code that goes in the solution slot. Your code is appended verbatim to the skeleton and then executed by a hidden test harness on SEVERAL DIFFERENT inputs (other values, shapes, sizes, column names, number of levels, lengths, dates, etc.); the harness then compares the designated output variable(s) to a reference solution. The reference solution is essentially the accepted StackOverflow answer. Score is all-or-nothing.

OUTPUT FORMAT (strict):
- Respond with exactly one block: <code> ... </code> containing executable Python only. No prose, no markdown fences, no BEGIN/END SOLUTION markers.
- Do not repeat the skeleton code (imports/data setup already exist), except for extra imports you need.
- No print statements, no plt.show() unless already in the skeleton, no debugging output, no extra side effects.

RULES THAT DETERMINE PASS/FAIL (learned from the grader):
1. Assign exactly the variable(s) the skeleton asks for (e.g. `result = ...`, `df = ...`, `proba = ...`, "put score in `b`"). Keep the exact names. Store the raw/natural output of the operation asked for — do NOT post-process beyond the question (e.g. if asked for probabilities, assign the full predict_proba output, not one column). Do ONLY what the final instruction asks; no additional demonstrations, no extra mutations of the given objects.
2. Match the desired output shown in the question EXACTLY: type (DataFrame vs Series vs ndarray vs list vs tensor), shape/orientation, column names and order, index (kept vs reset), element order, dtype where visible. The sample output in the question is the spec — reproduce it even if it looks arbitrary (e.g. an unusual column ordering or row order).
3. Be general: the hidden tests reuse your code on other inputs. Never hard-code example values, lengths, shapes, number of columns/levels, bit widths, or names that are not fixed by the problem; derive everything from the given variables (e.g. use df.columns / df.columns.nlevels / len(...) rather than literal counts; do not use uint8/unpackbits for "m bits" when m may exceed 8).
4. Prefer the most conventional, commonly recommended approach for the library named in the question — the one an accepted StackOverflow answer would give — rather than a theoretically more rigorous or cleverer one. Do not add transformations the question did not ask for. When the question describes steps in order (e.g. "convert the date to this format, then sort"), implement them literally in that order with the obvious operations, even if another ordering seems smarter (e.g. if dates are reformatted to strings before sorting, the sort is on those strings — and that is what the expected output reflects).
5. Prefer idiomatic, vectorised library calls. Some hidden tests tokenise your code and FAIL it if it contains the tokens `for` or `while` — this includes list/dict/set comprehensions and generator expressions. Avoid loops and comprehensions entirely unless the problem genuinely cannot be solved without iteration (use stack/unstack/melt/pivot, add_prefix/add_suffix, str accessors, np.vectorize, apply/map, broadcasting, itertools, etc.). If the question names a specific function/method/approach to use, use it (some tests check that the function name appears in your code).
6. Function problems: if the skeleton shows a function header (e.g. `def f(a=example_a):` ending at BEGIN SOLUTION), write ONLY the indented function body (including the `return`). If the instructions say "define function named `X` as solution", the hidden test calls `X(<primary input>)` with ONE positional argument (the main input variable from the skeleton) and the other given variables (parameters/thresholds) are available as globals — so make the primary input the only required parameter and read the others from the enclosing scope (or give them keyword defaults equal to the given variables).
7. When the output variable has the same name as an input (e.g. `df = ...`, `a = ...`), assign the transformed object to that name. Prefer operations that return new objects; if you must mutate in place, make sure the named variable ends up holding the right object.
8. sklearn: never call load_data() again; use default hyper-parameters unless the question specifies them; honour requested random_state values; use the estimator/approach named in the question.
9. Use current APIs for the installed library versions (given below): no removed pandas methods (`DataFrame.append`, `Series.iteritems`, `.ix`), no deprecated numpy aliases (`np.float`, `np.int`), etc.
10. matplotlib: act on the current figure/axes via `plt` (or the given `ax`/`fig`) exactly as the comment asks (labels, titles, legend, ticks, colours, etc.); do not create new figures unless asked; do not call plt.close().
11. torch/tensorflow: return tensors (not numpy) unless asked; keep dtypes; use the API of the version stated in the question.
12. scipy/numpy: when the question wants labels, indices, or lists, use the canonical function for that (e.g. 0-based labels from scipy.cluster.hierarchy.cut_tree / sklearn, 1-based from fcluster) in the form the expected output shows; convert to a plain list only if the question asks for a list.

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


def judge_prompt(problem, cands, can_execute):
    parts = [
        "Several candidate solutions to one DS-1000 problem were produced by different models. ",
        "Each was appended to the skeleton and executed on the example inputs; the resulting values of the "
        "target variable(s) are shown." if can_execute else
        "The skeleton needs hidden data (e.g. load_data()), so the candidates could not be executed here; "
        "judge them statically.",
        f"\n\n### Problem\n{problem}\n",
    ]
    for c in cands:
        parts.append(f"\n### Candidate {c['tag']}\n<code>\n{c['code']}\n</code>\n")
        if can_execute:
            parts.append(f"Execution report for {c['tag']}:\n{c['run']['report'][:3500]}\n")
    parts.append(
        "\nThe hidden grader re-runs the chosen code on OTHER inputs (different values, sizes, column names, "
        "number of columns/levels, dates, etc.) and compares the target variable(s) to a reference solution "
        "that is essentially the accepted StackOverflow answer, using strict equality (type, shape, column "
        "names/order, index, element order, dtype). Some graders also reject code containing the tokens "
        "`for`/`while` (including comprehensions).\n\n"
        "Decide which candidate is most likely to pass. Check, in order: (a) output matches the desired "
        "output in the question exactly (compare cell by cell if one is shown); (b) exact requested variable "
        "names assigned with the raw result, no extra post-processing; (c) does ONLY what is asked — no extra "
        "mutations, prints, transformations the question did not request; (d) general over other hidden "
        "inputs — nothing hard-coded from the example (counts, level numbers, names, bit widths), and the "
        "steps follow the literal order the question describes (e.g. reformat then sort); (e) correct "
        "function signature if a function was requested; (f) no `for`/`while` tokens unless unavoidable; "
        "(g) the most conventional approach for the named library.\n\n"
        "Respond with EITHER exactly one line `CHOOSE: <tag>` naming the best candidate, OR — only if every "
        "candidate has a concrete defect you can name — a single corrected <code>...</code> block (full "
        "replacement code to append, no prose). Be conservative; do not rewrite working code for style alone.")
    return "".join(parts)


def loop_prompt(problem, code):
    return (f"The following DS-1000 solution contains the token `for` or `while` (a loop, a list/dict/set "
            f"comprehension, or a generator expression). The hidden grader tokenises the code and REJECTS it "
            f"if either token appears anywhere, even inside a comprehension or a string. Rewrite it using "
            f"vectorised library operations (stack/unstack/melt/pivot, add_prefix/add_suffix, str accessors, "
            f"numpy/pandas/torch/tf functions, broadcasting, np.vectorize, apply/map with a lambda or "
            f"operator functions, itertools, functools.reduce, etc.) with IDENTICAL behaviour and output "
            f"(same values, types, dtypes, column names/order, index). Keep the same variable names.\n\n"
            f"### Problem\n{problem}\n\n### Current solution\n<code>\n{code}\n</code>\n\n"
            "Respond with exactly one <code>...</code> block containing only the rewritten code to "
            "append. Before answering, scan your code: it must contain NO `for` and NO `while` token at all.")


async def llm(model, key, messages, budget: Budget, effort=None, max_tokens=3000):
    if effort:
        cfg = GenerateConfig(max_tokens=max_tokens, reasoning_effort=effort)
    else:
        cfg = GenerateConfig(max_tokens=max_tokens)
    resp = await model.generate(messages, config=cfg)
    budget.add(resp, key)
    return resp.completion or ""


# generators: (tag, model handle, price key, reasoning effort, max_tokens, priority)
GENERATORS = [
    ("A", CLAUDE_OPUS_4_8, "opus", None, 3500, 0),
    ("B", GPT_5_4, "gpt54", "medium", 6000, 1),
    ("C", GEMINI_3_1_PRO_PREVIEW, "gemini", None, 3500, 2),
]


def norm_out(s: str) -> str:
    """Normalise an execution dump for agreement comparison."""
    s = s or ""
    s = re.sub(r"(\d+\.\d{6})\d+", r"\1", s)          # trim long float tails
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n+", "\n", s)
    return s.strip()


def norm_code(c: str) -> str:
    lines = [ln.split("#", 1)[0].rstrip() for ln in (c or "").splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


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
        versions = "pandas 2.2, numpy 1.26, scikit-learn 1.4, scipy 1.15, matplotlib 3.10"
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

        # ---- 1. generate three candidates in parallel ---------------------
        msgs = [ChatMessageSystem(content=SYSTEM_PROMPT),
                ChatMessageUser(content=user_prompt(problem, library, versions))]

        async def gen(tag, model, key, effort, mt):
            try:
                txt = await llm(model, key, msgs, budget, effort=effort, max_tokens=mt)
                return fix_indent(extract_code(txt), in_function)
            except Exception as e:
                print(f"  generation {tag} failed: {e!r}")
                return ""

        raw = await asyncio.gather(*[gen(t, m, k, e, mt) for (t, m, k, e, mt, _p) in GENERATORS])
        cands = []
        for (tag, _m, key, _e, _mt, prio), code in zip(GENERATORS, raw):
            if code.strip():
                cands.append({"tag": tag, "key": key, "prio": prio, "code": code, "run": None})
                print(f"  candidate {tag} ({key}):\n{code}")
        if not cands:
            # last resort: one more try with Sonnet
            try:
                code = fix_indent(extract_code(await llm(CLAUDE_SONNET_4_6, "sonnet", msgs, budget,
                                                         effort="medium", max_tokens=3500)), in_function)
            except Exception as e:
                print(f"  fallback generation failed: {e!r}")
                code = ""
            cands.append({"tag": "S", "key": "sonnet", "prio": 3, "code": code, "run": None})

        # ---- 2. execute every candidate -----------------------------------
        for c in cands:
            c["run"] = await execute(c["code"])
            print(f"  exec {c['tag']} ok={c['run']['ok']}\n{c['run']['report'][:1200]}")
        can_execute = any(c["run"]["ok"] is not None for c in cands) and not any(
            skeleton_unrunnable(c["run"]["report"]) for c in cands)

        # ---- 3. repair crashed candidates (one cheap round each) ----------
        if can_execute:
            for c in cands:
                if c["run"]["ok"] is False and budget.spent < SOFT_CAP:
                    try:
                        fixed = fix_indent(extract_code(await llm(
                            CLAUDE_SONNET_4_6, "sonnet",
                            [ChatMessageSystem(content=SYSTEM_PROMPT),
                             ChatMessageUser(content=repair_prompt(problem, c["code"], c["run"]["report"]))],
                            budget, effort="low", max_tokens=3500)), in_function)
                    except Exception as e:
                        print(f"  repair {c['tag']} failed: {e!r}")
                        continue
                    if fixed.strip():
                        frun = await execute(fixed)
                        print(f"  repair {c['tag']} ok={frun['ok']}\n{frun['report'][:800]}")
                        if frun["ok"]:
                            c["code"], c["run"] = fixed, frun
                            c["repaired"] = True

        # ---- 4. vote / judge ----------------------------------------------
        def rank_key(c):
            return (has_loop_tokens(c["code"]), c["prio"])

        chosen = None
        if can_execute:
            good = [c for c in cands if c["run"]["ok"]]
            groups = {}
            for c in good:
                groups.setdefault(norm_out(c["run"]["out"]), []).append(c)
            sizes = sorted((len(v) for v in groups.values()), reverse=True)
            print(f"  runnable={len(good)}/{len(cands)} agreement groups={sizes}")
            if sizes and sizes[0] >= 2 and (len(sizes) == 1 or sizes[0] > sizes[1]):
                grp = max(groups.values(), key=len)
                chosen = sorted(grp, key=rank_key)[0]
                print(f"  majority of {len(grp)} agrees -> candidate {chosen['tag']}")
            pool = good if good else cands
        else:
            pool = cands
            # identical code (modulo comments) counts as agreement
            groups = {}
            for c in cands:
                groups.setdefault(norm_code(c["code"]), []).append(c)
            grp = max(groups.values(), key=len)
            if len(grp) >= 2:
                chosen = sorted(grp, key=rank_key)[0]
                print(f"  {len(grp)} identical candidates -> {chosen['tag']}")

        if chosen is None and len(pool) == 1 and can_execute and pool[0]["run"]["ok"]:
            chosen = pool[0]
            print(f"  only one runnable candidate -> {chosen['tag']}")

        if chosen is None and budget.spent < SOFT_CAP:
            try:
                jv = await llm(CLAUDE_OPUS_4_8, "opus",
                               [ChatMessageSystem(content=SYSTEM_PROMPT),
                                ChatMessageUser(content=judge_prompt(problem, pool, can_execute))],
                               budget, max_tokens=4000)
            except Exception as e:
                print(f"  judge failed: {e!r}")
                jv = ""
            m = re.search(r"CHOOSE:\s*(?:candidate\s*)?([A-Za-z])\b", jv, re.I)
            if "<code>" in jv:
                new = fix_indent(extract_code(jv), in_function)
                nrun = await execute(new) if can_execute else {"ok": None, "report": "", "out": ""}
                print(f"  judge wrote a new solution ok={nrun['ok']}\n{new}\n{nrun['report'][:800]}")
                if nrun["ok"] or nrun["ok"] is None:
                    chosen = {"tag": "J", "key": "opus", "prio": -1, "code": new, "run": nrun}
                else:
                    print("  judge's solution crashed; falling back to candidates")
            if chosen is None and m:
                tag = m.group(1).upper()
                for c in pool:
                    if c["tag"] == tag:
                        chosen = c
                        print(f"  judge chose candidate {tag}")
            if chosen is None:
                print(f"  judge output unusable: {jv[:200]!r}")

        if chosen is None:
            chosen = sorted(pool, key=rank_key)[0]
            print(f"  fallback pick -> {chosen['tag']}")

        code, run = chosen["code"], chosen["run"]

        # ---- 5. loop guard ------------------------------------------------
        if has_loop_tokens(code) and not in_function:
            # a loop-free candidate with identical executed output?
            alt = None
            if can_execute and run["ok"]:
                for c in cands:
                    if c is not chosen and c["run"]["ok"] and not has_loop_tokens(c["code"]) \
                            and norm_out(c["run"]["out"]) == norm_out(run["out"]):
                        alt = c
                        break
            if alt is not None:
                print(f"  loop guard: switching to loop-free agreeing candidate {alt['tag']}")
                code, run = alt["code"], alt["run"]
            else:
                for model, key, effort in ((CLAUDE_OPUS_4_8, "opus", None), (GPT_5_4, "gpt54", "medium")):
                    if budget.spent >= SOFT_CAP:
                        break
                    try:
                        nl = fix_indent(extract_code(await llm(
                            model, key,
                            [ChatMessageSystem(content=SYSTEM_PROMPT),
                             ChatMessageUser(content=loop_prompt(problem, code))],
                            budget, effort=effort, max_tokens=5000)), in_function)
                    except Exception as e:
                        print(f"  loop rewrite ({key}) failed: {e!r}")
                        continue
                    if not nl.strip() or has_loop_tokens(nl):
                        print(f"  loop rewrite ({key}) still has loops or empty")
                        continue
                    if can_execute:
                        nrun = await execute(nl)
                        same = bool(nrun["ok"]) and norm_out(nrun["out"]) == norm_out(run["out"])
                        print(f"  loop rewrite ({key}) ok={nrun['ok']} same_output={same}")
                        if nrun["ok"] and (same or not run["ok"]):
                            code, run = nl, nrun
                            break
                    else:
                        code = nl
                        break

        state.output.completion = f"<code>\n{code}\n</code>"
        print(f"  final:\n{code}")
        print(f"  llm_calls={budget.calls} est_cost=${budget.spent:.4f}")
        for ln in budget.log:
            print(f"    {ln}")
        return state

    return solve
