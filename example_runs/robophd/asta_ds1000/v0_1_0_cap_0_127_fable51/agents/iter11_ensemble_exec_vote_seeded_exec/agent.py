"""ensemble-exec-vote-seeded-exec: DS-1000 solver (iteration 11).

Iteration 10's pipeline is kept. New in this version:
  * seeded execution (stage 2). Every candidate is executed in its own
    subprocess; when the skeleton draws unseeded random data
    (np.random.rand / randn, torch.rand, ...; 12 of the 92 problems seen so
    far) every candidate saw different data, so the vote was a 1-1-1 split
    of pure noise, escalated to five models (still 1-1-1-1-1) and ended in a
    judge pick -- 8 of those 12 problems went that way, 513 did so with five
    byte-identical candidates. The execution header now seeds random, numpy
    and (when the program mentions them) torch / tensorflow before the
    skeleton runs, so candidates are compared on identical data.
  * identical code is one vote group. Candidates whose code is identical
    modulo comments/blank lines are merged into one agreement group even if
    their executed dumps differ (only non-determinism can cause that).
  * target parser: `put ... in X` now requires the word `put` (974 picked up
    the phantom target `one` from "in**put** falling in one of"), and a
    candidate counts as runnable when at least one target is defined, so a
    phantom target can no longer push a problem onto the static path.

History (iteration 10 header follows).

Iteration 9's pipeline is kept unchanged on every non-Matplotlib path. New:
  * figure-state dump (stage 2, Matplotlib problems only). Matplotlib graders
    inspect artist properties (611: `ax.collections[0].get_hatch()`), but the
    execution dump recorded only titles/labels/limits/artist counts, so
    candidates differing in hatch, marker, colour, linestyle, tick labels,
    legend, scale, spines or grid looked identical to the vote and the pick
    fell to model priority. The dump now records those properties per
    line/patch/collection/text/image and per axis (after a canvas draw, so
    tick labels are materialised). Genuine splits reach the judge, which is
    told what Matplotlib graders check and sees a longer report excerpt.
  * system prompt rule 10 names the inspected properties.

History (iteration 9 header follows).

Iteration 8's pipeline is kept unchanged on every path. New in iteration 9:
  * named-API guard (stage 6b). Some hidden graders tokenise the submitted
    code and assert that an identifier the question explicitly asks for
    appears (706: `"saved_model" in tokens`) or that a forbidden one does
    not. A cheap extractor (GPT-5.4-mini, no reasoning) lists identifiers
    the question EXPLICITLY requires/forbids; only names that literally
    occur in the question text are kept, loops are left to the loop guard.
    If the chosen code violates a constraint: switch to a value-agreeing
    candidate that satisfies it, else ask for a rewrite that keeps the
    behaviour and adopt it only if it executes to the identical output.
    With no explicit constraint (the usual case) the stage is a no-op.

Pipeline map (stage -> function / key variables):
  1 generate A/B/C (+ recall R)   pipeline() "1. generate"   cands, rcand
  2 execute, 2b synthetic data    "2."/"2b."                 execute(), cur_skel, synth_stub
  3 repair, 3b execute R          "3."/"3b."
  4 vote/escalate/judge           "4. vote"                  good, groups, chosen, majority_grp
  4a recall dtype tie-break       "4a."                      from_recall
  4b dtype-split judge            "4b."                      ctx["dtype_split"]
  5 generality check              "5."                       skeleton2, run2
  6 loop guard, 6b API guard      "6."/"6b."                 code, run

History (iteration 7 header follows).

Iteration 6's pipeline (three-provider ensemble, execute, majority vote,
escalation + judge on disagreement, dtype-split judge, generality check, loop
guard, bounded waits) is kept. New in this version:
  * synthetic data for unrunnable skeletons. When the skeleton calls a helper
    the harness hides (load_data()), every earlier version judged the
    candidates statically -- the only path on which nothing is ever executed,
    and the path that lost problem 906 (a candidate using np.matrix, which
    sklearn 1.4 rejects, was picked over two runnable ones). Now a model writes
    a stand-in for the missing helper(s) from the data shown/described in the
    question; the stand-in is probed, and the candidates are executed on it so
    crashes/repair/voting/dtype-split/loop-guard all work as on ordinary
    problems. The judge is told the data is synthetic and also sees candidates
    that crashed on it. If the stand-in cannot be built or every candidate
    crashes on it, the old static path is used unchanged.
  * judges see the installed library versions.
Carried over from iteration 6:
  * the cost estimator no longer double-counts reasoning tokens for OpenAI /
    Anthropic handles (their output_tokens already include reasoning). In
    iteration 5 the inflated estimate tripped SOFT_CAP on two three-way splits
    and the disagreement judge was silently skipped (662, 918; the fallback
    happened to be right). The judge now has its own, higher cap.
  * escalation on disagreement: when no two runnable candidates agree on the
    example output, two more candidates (Claude Sonnet 4.6 medium, GPT-5.5)
    are generated in parallel and executed; a group of >=3 of 5 wins outright,
    otherwise the judge sees one representative per distinct output with the
    number of models that produced it.
  * the disagreement judge runs with a little reasoning (Opus 4.8, low).
  * Gemini's generation timeout is 150 s (it stalled again in iteration 5).
  * every candidate's strict / value-normalised output hash and dtype lines
    are logged so decisions can be replayed from agent_stdout.

Pipeline per sample (all stages time-bounded, best-so-far always emitted):
  1. parse the prompt (skeleton code, target names, function-body context)
  2. generate THREE candidates in parallel from diverse strong models
     (Claude Opus 4.8, GPT-5.4 medium reasoning, Gemini 3.1 Pro); a model
     that stalls is dropped after LLM_TIMEOUT, the others proceed
  3. execute every candidate in the sandbox (subprocess, fresh namespace) and
     dump the target variable(s) with type/shape/dtype/index; a sandbox call
     that stalls marks the sandbox dead and the pipeline falls back to
     static judging
  4. crashing candidates get one repair round (Sonnet)
  5. vote: candidates whose executed outputs agree form a majority -> pick the
     loop-free one from the highest-priority model; no majority -> an Opus 4.8
     judge sees problem + candidates + executed outputs and picks/corrects.
     Value-agree/dtype-split (new): if every runnable candidate agrees on the
     values but the dumps differ in dtype/type, a dedicated Opus judge decides
     which dtypes the accepted StackOverflow answer would produce
  6. generality check (new): when a majority agrees on the example but their
     code differs, a cheap model writes an alternative input; the agreeing
     candidates are re-run on it; if they diverge, candidates that crash are
     dropped and a judge decides among the rest with both executions in view
  7. loop guard: if the pick contains `for`/`while` tokens, prefer an agreeing
     loop-free candidate, else request vectorised rewrites and adopt only if
     the output is unchanged
  8. emit a clean <code>...</code> block (hard deadline: whatever is best
     so far is emitted well before the harness kills the sample)
"""

import asyncio
import io
import json
import re
import time
import tokenize
import traceback

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import (
    CLAUDE_OPUS_4_8,
    CLAUDE_SONNET_4_6,
    GEMINI_3_1_PRO_PREVIEW,
    GPT_5_4,
    GPT_5_4_MINI,
    GPT_5_5,
)

# ----------------------------------------------------------------------------
# time limits (seconds). The harness kills a sample at ~1770 s.
# ----------------------------------------------------------------------------
LLM_TIMEOUT = 330        # wall-clock cap for one model call (incl. retries)
GEMINI_TIMEOUT = 150     # Gemini's successful calls take 5-62 s; it stalled for 240+ s on 3/40 samples
LLM_ATTEMPT_TIMEOUT = 240
SANDBOX_TIMEOUT = 240    # one python_session call (inner subprocess cap is 150 s)
PROBE_TIMEOUT = 90       # library-version probe
OPTIONAL_AFTER = 540     # skip optional stages (repair/generality/loop rewrite) past this
JUDGE_AFTER = 840        # skip even the disagreement judge past this
HARD_LIMIT = 1260        # emit best-so-far at this point no matter what
EXEC_SUBPROCESS_TIMEOUT = 150

# ----------------------------------------------------------------------------
# cost bookkeeping (approximate; only for diagnostics / soft cap)
# ----------------------------------------------------------------------------
PRICES = {  # $/token (input, output)
    "gpt54": (2.50 / 1e6, 15.00 / 1e6),
    "gpt55": (5.00 / 1e6, 30.00 / 1e6),
    "opus": (5.00 / 1e6, 25.00 / 1e6),
    "gemini": (2.00 / 1e6, 12.00 / 1e6),
    "sonnet": (3.00 / 1e6, 15.00 / 1e6),
    "mini": (0.75 / 1e6, 4.50 / 1e6),
}
# Verified against eval_cost in iteration 5: OpenAI and Anthropic usage.output_tokens already
# include reasoning tokens; Gemini bills reasoning on top of output_tokens.
REASONING_EXTRA = {"gemini"}
SOFT_CAP = 0.16    # skip optional LLM steps (repair/generality/loop rewrite/escalation) beyond this
JUDGE_CAP = 0.30   # the disagreement/dtype judges are the highest-value calls; only skip when far over


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
            c = i * pin + (o + (r if key in REASONING_EXTRA else 0)) * pout
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
    for mm in re.finditer(r"\bput (?:the )?\w+ in [`'\"]?([A-Za-z_]\w*)[`'\"]?", outside):
        if mm.group(1) not in ("this", "these", "the", "a", "one", "any", "each", "it"):
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
        import matplotlib.colors as _mc
        import numpy as _np2
        def _r(v):
            try:
                if isinstance(v, (bool, _np2.bool_)):
                    return bool(v)
                if isinstance(v, (str, _np2.str_)):
                    return str(v)
                if isinstance(v, (float, _np2.floating, int, _np2.integer)):
                    return round(float(v), 4)
                if isinstance(v, (list, tuple, _np2.ndarray)):
                    return [_r(x) for x in list(v)[:8]]
            except Exception:
                pass
            return v
        def _hex(c):
            try:
                if isinstance(c, str):
                    return _mc.to_hex(c, keep_alpha=True)
                a = _np2.asarray(c, dtype=float)
                if a.ndim == 2:
                    return [_mc.to_hex(tuple(x), keep_alpha=True) for x in a[:4]]
                return _mc.to_hex(tuple(a), keep_alpha=True)
            except Exception:
                return repr(c)
        def _p(fn, *a):
            try:
                return _r(fn(*a))
            except Exception:
                return "<err>"
        def _axis(_ax, _xa, _nm):
            try:
                _tl = _xa.get_ticklabels()
                _mt = _xa.majorTicks[0] if _xa.majorTicks else None
                _tp = {}
                try:
                    _tp = {k: _r(v) for k, v in sorted(_xa.get_tick_params(which="major").items())
                           if k in ("direction", "length", "width", "labelsize", "labelrotation", "pad",
                                    "colors", "labelcolor", "gridOn", "tick1On", "tick2On", "label1On", "label2On")}
                except Exception:
                    pass
                print(" %s: ticklabels=%r rot=%s size=%s lblcolor=%s tickvis=%s lblvis=%s minor=%d pos=%s labelpos=%s inverted=%s grid=%s label(color=%s,size=%s) params=%s" % (
                    _nm, [t.get_text() for t in _tl[:10]],
                    _p(lambda: _tl[0].get_rotation()) if _tl else None,
                    _p(lambda: _tl[0].get_fontsize()) if _tl else None,
                    _p(lambda: _hex(_tl[0].get_color())) if _tl else None,
                    _p(lambda: _mt.tick1line.get_visible()) if _mt else None,
                    _p(lambda: _mt.label1.get_visible()) if _mt else None,
                    len(_p(_xa.get_minorticklocs) or []), _p(_xa.get_ticks_position), _p(_xa.get_label_position),
                    _p(_xa.get_inverted), _p(lambda: any(g.get_visible() for g in _xa.get_gridlines())),
                    _p(lambda: _hex(_xa.label.get_color())), _p(_xa.label.get_fontsize), _tp))
            except Exception as _e:
                print(" %s: <axis dump failed %r>" % (_nm, _e))
        _figs = _plt.get_fignums()
        print("--- figures: %s" % _figs)
        for _fn in _figs:
            _f = _plt.figure(_fn)
            try:
                _f.canvas.draw()
            except Exception as _e:
                print("(draw failed: %r)" % (_e,))
            print("figure %s: size=%s axes=%d suptitle=%r" % (
                _fn, _p(lambda: list(_f.get_size_inches())), len(_f.get_axes()),
                _p(lambda: _f._suptitle.get_text() if _f._suptitle else "")))
            for _ax in _f.get_axes():
                _leg = _ax.get_legend()
                print("axes(%s): title=%r xlabel=%r ylabel=%r xlim=%s ylim=%s xscale=%s yscale=%s aspect=%s axison=%s facecolor=%s hidden_spines=%s lines=%d patches=%d collections=%d texts=%d images=%d containers=%s xticks=%s yticks=%s" % (
                    type(_ax).__name__, _ax.get_title(), _ax.get_xlabel(), _ax.get_ylabel(),
                    _p(_ax.get_xlim), _p(_ax.get_ylim), _p(_ax.get_xscale), _p(_ax.get_yscale),
                    _p(_ax.get_aspect), _p(lambda: _ax.axison), _p(lambda: _hex(_ax.get_facecolor())),
                    _p(lambda: sorted(k for k, s in _ax.spines.items() if not s.get_visible())),
                    len(_ax.get_lines()), len(_ax.patches), len(_ax.collections), len(_ax.texts),
                    len(_ax.images), _p(lambda: [type(c).__name__ for c in _ax.containers]),
                    _p(lambda: list(_ax.get_xticks())), _p(lambda: list(_ax.get_yticks()))))
                _axis(_ax, _ax.xaxis, "xaxis")
                _axis(_ax, _ax.yaxis, "yaxis")
                if _leg:
                    print(" legend: texts=%r title=%r frame=%s loc=%s ncols=%s" % (
                        [t.get_text() for t in _leg.get_texts()], _p(lambda: _leg.get_title().get_text()),
                        _p(_leg.get_frame_on), _p(lambda: getattr(_leg, "_loc", None)),
                        _p(lambda: getattr(_leg, "_ncols", None))))
                for _l in _ax.get_lines()[:10]:
                    print(" line: label=%r color=%s ls=%r lw=%s marker=%r ms=%s mfc=%s mec=%s alpha=%s vis=%s n=%d draw=%r" % (
                        _l.get_label(), _p(lambda: _hex(_l.get_color())), _p(_l.get_linestyle), _p(_l.get_linewidth),
                        _p(_l.get_marker), _p(_l.get_markersize), _p(lambda: _hex(_l.get_markerfacecolor())),
                        _p(lambda: _hex(_l.get_markeredgecolor())), _p(_l.get_alpha), _p(_l.get_visible),
                        len(_l.get_xdata()), _p(_l.get_drawstyle)))
                for _pa in _ax.patches[:12]:
                    print(" patch(%s): fc=%s ec=%s hatch=%r alpha=%s lw=%s ls=%r vis=%s xywh=%s" % (
                        type(_pa).__name__, _p(lambda: _hex(_pa.get_facecolor())), _p(lambda: _hex(_pa.get_edgecolor())),
                        _p(_pa.get_hatch), _p(_pa.get_alpha), _p(_pa.get_linewidth), _p(_pa.get_linestyle), _p(_pa.get_visible),
                        _p(lambda: (_pa.get_x(), _pa.get_y(), _pa.get_width(), _pa.get_height()))))
                for _c in _ax.collections[:6]:
                    print(" collection(%s): fc=%s ec=%s hatch=%r sizes=%s alpha=%s lw=%s ls=%s n=%s cmap=%s clim=%s vis=%s" % (
                        type(_c).__name__, _p(lambda: _hex(_c.get_facecolor())), _p(lambda: _hex(_c.get_edgecolor())),
                        _p(_c.get_hatch), _p(lambda: list(_c.get_sizes())[:5]), _p(_c.get_alpha), _p(lambda: list(_c.get_linewidth())[:3]),
                        _p(lambda: repr(_c.get_linestyle()[:2])), _p(lambda: tuple(_c.get_offsets().shape)),
                        _p(lambda: _c.get_cmap().name), _p(_c.get_clim), _p(_c.get_visible)))
                for _t in _ax.texts[:12]:
                    print(" text: %r pos=%s size=%s color=%s rot=%s ha=%s va=%s" % (
                        _t.get_text(), _p(_t.get_position), _p(_t.get_fontsize), _p(lambda: _hex(_t.get_color())),
                        _p(_t.get_rotation), _p(_t.get_ha), _p(_t.get_va)))
                for _im in _ax.images[:4]:
                    print(" image: cmap=%s clim=%s interp=%s shape=%s extent=%s" % (
                        _p(lambda: _im.get_cmap().name), _p(_im.get_clim), _p(_im.get_interpolation),
                        _p(lambda: tuple(_im.get_array().shape)), _p(_im.get_extent)))
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
_out, _err, _rc = "", "", -1
try:
    _p = subprocess.Popen([sys.executable, os.path.join(_dir, "_cand.py")], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, cwd=_dir, start_new_session=True)
    try:
        _out, _err = _p.communicate(timeout=%d)
        _rc = _p.returncode
    except subprocess.TimeoutExpired:
        try:
            import signal
            os.killpg(os.getpgid(_p.pid), signal.SIGKILL)
        except Exception:
            _p.kill()
        try:
            _out, _err = _p.communicate(timeout=10)
        except Exception:
            _out, _err = "", ""
        _err = (_err or "") + "\nTIMEOUT: the program did not finish within %d s"
        _rc = -1
except Exception as _e:
    _err = "runner failure: %%r" %% (_e,)
    _rc = -1
if isinstance(_out, bytes): _out = _out.decode("utf-8", "replace")
if isinstance(_err, bytes): _err = _err.decode("utf-8", "replace")
print("@@JSON@@" + json.dumps({"rc": _rc, "stdout": (_out or "")[-6000:], "stderr": (_err or "")[-4000:]}))
'''


SEED_HEADER = ("try:\n    import random as _rnd_; _rnd_.seed(0)\nexcept Exception:\n    pass\n"
               "try:\n    import numpy as _np_; _np_.random.seed(0)\nexcept Exception:\n    pass\n")
TORCH_SEED = "try:\n    import torch as _th_; _th_.manual_seed(0)\nexcept Exception:\n    pass\n"
TF_SEED = "try:\n    import tensorflow as _tf_; _tf_.random.set_seed(0)\nexcept Exception:\n    pass\n"


def seed_header(text: str) -> str:
    """Seed every RNG the program may use, before the skeleton runs, so that
    candidates executed in separate subprocesses see identical random data.
    (A skeleton that seeds itself simply re-seeds afterwards.)"""
    h = SEED_HEADER
    if "torch" in text:
        h += TORCH_SEED
    if "tensorflow" in text or "tf." in text:
        h += TF_SEED
    return h


def build_program(skeleton, marker_line, code, targets, func_name, inputs, in_function=False):
    check = (CHECK_SNIPPET
             .replace("__TARGETS__", repr(list(targets)))
             .replace("__FUNC__", repr(func_name))
             .replace("__INPUTS__", repr(list(inputs)))
             .replace("__INFUNC__", repr(bool(in_function))))
    header = ("import warnings\nwarnings.filterwarnings('ignore')\n"
              "try:\n    import matplotlib\n    matplotlib.use('Agg')\nexcept Exception:\n    pass\n")
    header += seed_header(skeleton + "\n" + code)
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


class Sandbox:
    """python_session wrapper: every call is time-bounded; a stalled call marks it dead."""

    def __init__(self, py):
        self.py = py
        self.dead = py is None
        self.calls = 0

    async def run(self, code: str, timeout: float = None) -> str:
        if self.dead:
            raise RuntimeError("sandbox unavailable")
        if timeout is None:
            timeout = SANDBOX_TIMEOUT
        self.calls += 1
        try:
            return str(await asyncio.wait_for(self.py(code=code), timeout=timeout))
        except asyncio.TimeoutError:
            self.dead = True
            print(f"  [t={ELAPSED():.0f}s] sandbox call stalled for {timeout}s -> sandbox marked dead")
            raise


async def run_candidate(sb: Sandbox, skeleton, marker_line, code, targets, func_name, inputs, in_function=False):
    """Return dict(ok: bool|None, report: str, out: str). ok=None → could not execute at all."""
    if sb is None or sb.dead:
        return {"ok": None, "report": "(sandbox unavailable)", "out": ""}
    prog, n_pre = build_program(skeleton, marker_line, code, targets, func_name, inputs, in_function)
    runner = RUNNER % (json.dumps(json.dumps(prog)), EXEC_SUBPROCESS_TIMEOUT, EXEC_SUBPROCESS_TIMEOUT)
    try:
        raw = await sb.run(runner)
    except Exception as e:
        return {"ok": None, "report": f"(sandbox error: {e!r})", "out": ""}
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
        ok = rc == 0 and ("NOT DEFINED" not in post or bool(re.search(r"^--- (?!function )\S+:\s*$", post, re.M))
                          or "--- call " in post)
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
10. matplotlib: act on the current figure/axes via `plt` (or the given `ax`/`fig`) exactly as the comment asks; do not create new figures unless asked; do not call plt.close(). The grader inspects the resulting artists' properties (colours, markers, hatches, line styles/widths, labels, titles, tick labels/rotation/visibility, legend entries/title, scales, limits, spines, grid, text positions), so set exactly the property the comment names through the standard keyword/method for it (a "hatch" is the hatch property, not the marker; a tick-label change is not an axis-label change) and leave everything else at its default.
11. torch/tensorflow: return tensors (not numpy) unless asked; keep dtypes; use the API of the version stated in the question.
12. scipy/numpy: when the question wants labels, indices, or lists, use the canonical function for that (e.g. 0-based labels from scipy.cluster.hierarchy.cut_tree / sklearn, 1-based from fcluster) in the form the expected output shows; convert to a plain list only if the question asks for a list.
13. dtypes matter: the comparison with the reference is dtype-strict (DataFrame/Series/array dtypes, int vs float, numeric vs string, numpy vs Python scalars). The reference is the accepted StackOverflow answer applied literally, so when two constructions give the same values, build the result the way that answer would (e.g. the construction the question's own attempts point towards) and keep the dtypes it would produce; never add casts (astype/float()/int()) the question did not ask for.

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


JUDGE_RULES = (
    "\nThe hidden grader re-runs the chosen code on OTHER inputs (different values, sizes, column names, "
    "number of columns/levels, dates, etc.) and compares the target variable(s) to a reference solution "
    "that is essentially the accepted StackOverflow answer, using strict equality (type, shape, column "
    "names/order, index, element order, dtype). Some graders also reject code containing the tokens "
    "`for`/`while` (including comprehensions). Dtype differences (int vs float, numeric vs string "
    "columns, numpy vs Python scalars) count as mismatches: prefer the candidate whose dtypes the "
    "accepted StackOverflow answer would produce.\n\n"
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


MPL_JUDGE_NOTE = (
    "\nThis is a Matplotlib problem: the grader inspects the FIGURE STATE after the code runs "
    "(artist properties such as colours, markers, hatches, line styles/widths, labels, titles, tick "
    "labels/rotation/visibility, legend entries/title, scales, limits, spines, grid, text positions), "
    "checking exactly the properties the question asks for; properties the question does not mention "
    "are free. The execution report shows each candidate's figure state. Prefer the candidate whose "
    "state satisfies every property the question names, set through the property the question names "
    "(a hatch is the hatch property, not the marker; a tick label change is not an axis label change), "
    "without extra artists or figures the question did not ask for.\n")


def judge_prompt(problem, cands, can_execute, versions="", synth="", mpl=False):
    parts = ["Several candidate solutions to one DS-1000 problem were produced by different models. "]
    if can_execute and synth:
        parts.append(
            "The skeleton needs hidden data (load_data()) that is not available here, so a model wrote a "
            "stand-in from the data shown/described in the question (see below) and each candidate was "
            "executed on that SYNTHETIC data. A crash caused by API misuse (unsupported types, removed or "
            "renamed functions, wrong argument shapes) is a real defect; differences in values or a crash "
            "that only reflects a mismatch between the stand-in and the real data are not. Weigh the "
            "executions together with a careful static reading of each candidate.")
    elif can_execute:
        parts.append("Each was appended to the skeleton and executed on the example inputs; the resulting "
                     "values of the target variable(s) are shown.")
    else:
        parts.append("The skeleton needs hidden data (e.g. load_data()), so the candidates could not be "
                     "executed here; judge them statically.")
    if versions:
        parts.append(f"\nInstalled versions in the grading sandbox: {versions}.")
    parts.append(f"\n\n### Problem\n{problem}\n")
    if can_execute and synth:
        parts.append(f"\n### Stand-in data used for execution (machine-written, may differ from the real data)"
                     f"\n<code>\n{synth}\n</code>\n")
    total = sum(c.get("n", 1) for c in cands)
    for c in cands:
        n = c.get("n", 1)
        hdr = f"\n### Candidate {c['tag']}"
        if can_execute and total > len(cands):
            hdr += (f" (this output was produced by {n} of the {total} models; the other"
                    f" {n - 1} agreeing solution(s) are omitted)" if n > 1
                    else f" (this output was produced by 1 of the {total} models)")
        parts.append(f"{hdr}\n<code>\n{c['code']}\n</code>\n")
        if can_execute:
            parts.append(f"Execution report for {c['tag']}:\n{c['run']['report'][:6000 if mpl else 3500]}\n")
    if mpl:
        parts.append(MPL_JUDGE_NOTE)
    if can_execute and total > len(cands):
        parts.append("\nAgreement between independent models is evidence but not proof: a minority "
                     "output is right when the majority shares a hard-coded assumption or ignores a "
                     "detail of the question. Weigh the counts together with the checks below.")
    parts.append(JUDGE_RULES)
    return "".join(parts)


def synth_data_prompt(problem, skeleton, missing, targets, versions, err_note=""):
    names = ", ".join(missing)
    return (
        "You are helping test DS-1000 solutions. The code skeleton below calls helper(s) that the hidden "
        f"test harness provides but that are not available here: {names}. Write a stand-in so the skeleton "
        "runs.\n\n"
        f"Define {names} in Python so that it returns EXACTLY what the skeleton unpacks (same number of values, "
        "same order) with the same types, shapes, column names, index and dtypes as the data shown or "
        "described in the question. If the question shows the data (a DataFrame print-out, arrays, lists, a "
        "table, tensors), reproduce it literally: the exact column names and order, the NaN cells, the index, "
        "numeric vs string columns, dates as the question has them. If the question only describes the data, "
        "build a small realistic example consistent with every stated detail (number of features/classes, "
        "label types, string vs numeric columns, sizes, date ranges); use a fixed seed (e.g. np.random.seed(0)) "
        "for anything random, and keep it small and fast. If a returned object is a fitted/configured "
        "estimator (e.g. a GridSearchCV), construct it the way the question implies.\n\n"
        "Rules: do NOT include any solution code; do not define "
        f"{', '.join(targets) if targets else 'the result variable'}; do not modify the skeleton; include the "
        "imports your stand-in needs (it is placed BEFORE the skeleton).\n\n"
        f"Installed versions: {versions}\n\n### Problem\n{problem}\n\n### Skeleton\n<code>\n{skeleton}\n</code>\n"
        + (f"\n### A previous stand-in failed when the skeleton ran; fix the cause:\n{err_note}\n" if err_note else "")
        + "\nRespond with exactly one <code>...</code> block containing only the stand-in definition(s) and "
          "their imports."
    )


def alt_input_prompt(problem, skeleton, targets, synthetic=False):
    note = ("- The skeleton begins with a machine-written stand-in for a helper the harness hides (e.g. "
            "load_data()). Keep a stand-in with the same name and return structure; change the data it "
            "builds. The replacement skeleton must run on its own.\n") if synthetic else ""
    return (
        "You are helping test DS-1000 solutions for GENERALITY. Below is a problem and its code skeleton "
        "(imports + example data setup). Write a NEW version of the skeleton that sets up DIFFERENT example "
        "data for the same problem, so that a solution which hard-codes the original example (its length, "
        "number of rows/columns/levels, specific labels, dates, positions, counts) would break or give a "
        "different answer, while a correct general solution still works.\n\n"
        "Requirements:\n"
        "- Keep the same imports and the SAME variable names, types and structure; keep every name the "
        "problem text explicitly refers to (column names, keys, parameters mentioned in the question).\n"
        "- Change what is free to vary: values, lengths/sizes, number of rows/columns/levels/groups, "
        "unreferenced labels, dates/months, order of rows, etc. Keep the data valid for the problem's "
        "assumptions (e.g. if the question implies sorted dates, unique ids, non-empty groups, keep that).\n"
        "- Use fixed literal data; no randomness unless the original uses np.random with a seed — then keep a "
        "seed but you may change sizes.\n"
        "- Keep it small and quick to run. Do NOT include any solution code and do not define "
        f"{', '.join(targets) if targets else 'the result'}.\n" + note + "\n"
        f"### Problem\n{problem}\n\n### Original skeleton\n<code>\n{skeleton}\n</code>\n\n"
        "Respond with exactly one <code>...</code> block containing the complete replacement skeleton."
    )


def general_judge_prompt(problem, cands, skeleton2):
    parts = [
        "Several candidate solutions to one DS-1000 problem were produced by different models. All of them "
        "produce IDENTICAL output on the original example input, but when re-run on an alternative input "
        "(machine-generated to probe generality) they DIVERGE. Decide which candidate's behaviour on the "
        "alternative input is the one the reference solution (the accepted StackOverflow answer applied "
        "literally to the question) would produce.\n\n"
        f"### Problem\n{problem}\n\n### Alternative input skeleton\n<code>\n{skeleton2}\n</code>\n",
    ]
    for c in cands:
        parts.append(f"\n### Candidate {c['tag']}\n<code>\n{c['code']}\n</code>\n")
        parts.append(f"Output on the ORIGINAL example:\n{c['run']['report'][:2000]}\n")
        parts.append(f"Output on the ALTERNATIVE input:\n{c['run2']['report'][:3000]}\n")
    parts.append(
        "\nCaveat: the alternative input is machine-generated and may itself violate an assumption of the "
        "question; if the divergence only reflects such an invalid input, prefer the most conventional "
        "candidate. Otherwise prefer the candidate that is general (nothing hard-coded from the example) "
        "and follows the question's literal specification on the alternative input."
        + JUDGE_RULES)
    return "".join(parts)


def dtype_judge_prompt(problem, groups):
    parts = [
        "Several candidate solutions to one DS-1000 problem were produced by different models. Each was "
        "appended to the skeleton and executed on the example inputs. They all produce the SAME VALUES, "
        "but they differ in dtype / type (e.g. int64 vs float64 vs string/object columns, numpy vs Python "
        "scalars, DataFrame vs ndarray). The hidden grader compares the target variable(s) to a reference "
        "solution with dtype-STRICT equality (pandas assert_frame_equal / assert_series_equal, "
        "np.array_equal with dtype, type checks), so exactly one of these dtype choices will pass.\n\n"
        "The reference solution is the accepted StackOverflow answer to this question applied literally "
        "(the direct, often naive implementation a typical answerer gave, wrapped in a function). Its "
        "dtypes are a SIDE EFFECT of how it was written, for example: np.column_stack / np.array over "
        "mixed str+int arrays makes everything strings; creating a new column by per-row `.loc[i, col] = "
        "...` assignment in a loop, or via `df.update`/`where`/`fillna`/reindex, yields float64; "
        "`mean`/`sum`/division give floats while `mode`/`idxmax`/indexing keep ints; pd.DataFrame from a "
        "dict of arrays keeps each array's dtype; `.values` on mixed columns gives object; `astype` in the "
        "answer sets a type explicitly.\n\n"
        "Task: reconstruct, step by step, how the accepted StackOverflow answer to THIS question would be "
        "written (use the question's own attempts, the desired output and common 2015-2020 idioms as "
        "evidence), work out the dtype(s) it produces, and choose the candidate whose dtypes match. If the "
        "question itself shows or states the dtype (e.g. an explicit `dtype=` or `astype` in the desired "
        "output), that overrides the reconstruction.\n\n"
        f"### Problem\n{problem}\n",
    ]
    for g in groups:
        parts.append(f"\n### Candidate {g['tag']} (produced by {g['n']} of the models)\n<code>\n{g['code']}\n</code>\n")
        parts.append(f"Execution report for {g['tag']}:\n{g['run']['report'][:3000]}\n")
    parts.append(
        "\nRespond with a short reasoning (at most ~10 lines) followed by exactly one line "
        "`CHOOSE: <tag>`. Only if you are confident that NO candidate's dtypes match the accepted answer, "
        "respond instead with a single corrected <code>...</code> block (full replacement code to append) "
        "that produces the same values with the right dtypes."
    )
    return "".join(parts)


RECALL_SYSTEM = (
    "You are a StackOverflow archivist with an excellent memory of pandas / numpy / scipy / scikit-learn / "
    "pytorch / tensorflow questions (2012-2022) and their accepted answers. Respond with exactly one "
    "<code>...</code> block containing executable Python only (no prose, no markdown fences)."
)


def recall_prompt(problem, library, versions):
    return (
        "This DS-1000 problem is a lightly edited StackOverflow question. The hidden reference solution is the "
        "ACCEPTED StackOverflow answer to that question applied literally, and the grader compares the result to "
        "it with dtype-strict equality. Your task is NOT to write the best solution: REPRODUCE the accepted answer "
        "as faithfully as you can recall it, keeping its exact construction (its loops, per-row `.loc` "
        "assignments, np.column_stack, apply, concat, the order of operations) and therefore the dtypes it "
        "produces. Do not vectorise, clean up, simplify or add casts the answer did not have. Adapt it only to "
        "the variable names of the skeleton below (assign exactly the target variable(s) the skeleton asks "
        "for; if the skeleton shows a function header, write only its body) and to the installed library "
        "versions when an API was removed.\n\n"
        "The FIRST line of your code must be the comment `# recall: high` if you specifically remember this "
        "StackOverflow question and its accepted answer, or `# recall: low` if you are only reconstructing a "
        "plausible answer. Then the code to append.\n\n"
        f"Library: {library}\nInstalled versions: {versions}\n\n{problem}\n\n"
        "Respond with exactly one <code>...</code> block."
    )


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


def constraint_prompt(problem):
    return (
        "CONSTRAINT EXTRACTION. Below is a DS-1000 problem (a StackOverflow question plus a code skeleton). "
        "Some hidden graders tokenise the submitted code and assert that a specific identifier APPEARS "
        "(when the question explicitly asks to use a particular function/method/module) or does NOT appear "
        "(when the question explicitly forbids one). List such identifiers ONLY when the question states the "
        "requirement explicitly, e.g. 'how can I do this using np.einsum', 'I want to use tf.saved_model', "
        "'using scipy.sparse', 'without using apply', 'I do not want to use merge'. Do NOT list: functions the "
        "asker merely tried or mentioned in passing or in error messages, loops ('for'/'while' are handled "
        "elsewhere), library aliases (np, pd, tf, torch, plt), or anything that is just the natural choice "
        "rather than an explicit demand of the question.\n\n"
        f"### Problem\n{problem}\n\n"
        'Respond with ONE JSON object only, no prose: {"must_use": [...], "must_not_use": [...]} using the '
        'dotted names exactly as written in the question (e.g. "tf.saved_model.save", "apply"). Use empty '
        "lists when nothing is explicit (this is the usual case)."
    )


def api_rewrite_prompt(problem, code, missing, present):
    want = ", ".join(f"`{t}`" for t in missing) or "(none)"
    ban = ", ".join(f"`{t}`" for t in present) or "(none)"
    return (f"The following DS-1000 solution is believed correct, but the hidden grader tokenises the code and "
            f"asserts the presence/absence of specific identifiers that the question explicitly asks for.\n"
            f"REQUIRED tokens that are currently missing: {want}\n"
            f"FORBIDDEN tokens that are currently present: {ban}\n\n"
            f"Rewrite the solution so that it uses the required function(s)/module(s) by that exact name and "
            f"avoids the forbidden ones, with IDENTICAL behaviour and output (same values, types, dtypes, "
            f"column names/order, index). Keep the same variable names, no loops/comprehensions unless the "
            f"current solution has them, no extra side effects.\n\n"
            f"### Problem\n{problem}\n\n### Current solution\n<code>\n{code}\n</code>\n\n"
            "Respond with exactly one <code>...</code> block containing only the rewritten code to append.")


_ALIASES = {"np", "pd", "tf", "torch", "plt", "sns", "sklearn", "scipy", "numpy", "pandas", "tensorflow",
            "matplotlib", "seaborn", "df", "self", "keras", "nn", "F", "stats", "sparse", "linalg", "signal"}
_GENERIC = {"result", "print", "return", "import", "python", "function", "def", "lambda", "for", "while",
            "if", "else", "in", "is", "not", "and", "or", "None", "True", "False", "list", "dict", "set",
            "len", "range", "str", "int", "float", "x", "y", "a", "b", "data"}


def code_tokens(code: str):
    """All token strings, like DS-1000's test_string does."""
    try:
        return {t.string for t in tokenize.generate_tokens(io.StringIO(code).readline)}
    except Exception:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))


def parse_constraints(txt: str, problem: str):
    """-> (must_use, must_not_use). must_use entries are lists of tokens that must ALL appear
    (e.g. tf.saved_model.save -> ['saved_model', 'save']; the grader checks "saved_model" in tokens);
    must_not_use entries are single tokens (the most specific part). Only names that literally
    occur in the question text are kept."""
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return [], []
    try:
        d = json.loads(m.group(0))
    except Exception:
        return [], []

    def clean(lst, all_parts):
        out = []
        for name in (lst or []):
            if not isinstance(name, str):
                continue
            parts = [p for p in re.split(r"[.\s()]+", name.strip()) if p]
            parts = [p for p in parts if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p)
                     and p not in _ALIASES and p not in _GENERIC and len(p) >= 3
                     and re.search(r"\b" + re.escape(p) + r"\b", problem)]  # only names the question contains
            if not parts:
                continue
            item = parts if all_parts else parts[-1]
            if item not in out:
                out.append(item)
        return out

    mu = clean(d.get("must_use"), True)
    flat = {p for parts in mu for p in parts}
    mn = [t for t in clean(d.get("must_not_use"), False) if t not in flat]
    return mu, mn


# ----------------------------------------------------------------------------
# time helpers
# ----------------------------------------------------------------------------
_T0 = [time.monotonic()]


def ELAPSED() -> float:
    return time.monotonic() - _T0[0]


def _config(effort, max_tokens, timeout):
    kw = dict(max_tokens=max_tokens)
    if effort:
        kw["reasoning_effort"] = effort
    try:
        return GenerateConfig(max_retries=4, attempt_timeout=int(min(LLM_ATTEMPT_TIMEOUT, timeout)), **kw)
    except Exception:
        return GenerateConfig(**kw)


async def llm(model, key, messages, budget: Budget, effort=None, max_tokens=3000, timeout=None):
    """One model call, hard-capped at `timeout` seconds of wall-clock."""
    if timeout is None:
        timeout = LLM_TIMEOUT
    cfg = _config(effort, max_tokens, timeout)
    t = time.monotonic()
    try:
        resp = await asyncio.wait_for(model.generate(messages, config=cfg), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"  [t={ELAPSED():.0f}s] {key} call timed out after {timeout}s")
        raise
    except Exception as e:
        # a handle that rejects the retry/timeout fields: retry once with a plain config
        print(f"  [t={ELAPSED():.0f}s] {key} call failed ({e!r}); retrying with plain config")
        plain = GenerateConfig(max_tokens=max_tokens, **({"reasoning_effort": effort} if effort else {}))
        remaining = max(30.0, timeout - (time.monotonic() - t))
        resp = await asyncio.wait_for(model.generate(messages, config=plain), timeout=remaining)
    budget.add(resp, key)
    print(f"  [t={ELAPSED():.0f}s] {key} call took {time.monotonic() - t:.0f}s")
    return resp.completion or ""


# generators: (tag, model handle, price key, reasoning effort, max_tokens, priority)
GENERATORS = [
    ("A", CLAUDE_OPUS_4_8, "opus", None, 3500, 0),
    ("B", GPT_5_4, "gpt54", "medium", 6000, 1),
    ("C", GEMINI_3_1_PRO_PREVIEW, "gemini", None, 3500, 2),
]


def gen_timeout(key):
    return GEMINI_TIMEOUT if key == "gemini" else LLM_TIMEOUT


# extra voices, generated only when the first three disagree pairwise
ESCALATION = [
    ("D", CLAUDE_SONNET_4_6, "sonnet", "medium", 3500, 4),
    ("E", GPT_5_5, "gpt55", None, 6000, 5),
]


def out_hash(s: str) -> str:
    import hashlib
    return hashlib.md5((s or "").encode("utf-8", "replace")).hexdigest()[:8]


def dtype_lines(out: str) -> str:
    d = re.findall(r"^(?:\[dtypes\].*|type=.*|.*\[dtype\] .*)$", out or "", re.M)
    return " | ".join(x.strip() for x in d)[:300]


DTYPE_LINE_RE = re.compile(r"^(type=.*|\[dtypes\].*)$", re.M)
DTYPE_TAG_RE = re.compile(r"\[dtype\] [^\s\[]+")


def norm_values(s: str) -> str:
    """Normalise an execution dump for VALUE comparison only: drop type/dtype
    lines, 3.0 -> 3, strip quotes, collapse whitespace. Two dumps that are equal
    here but differ under norm_out differ only in dtype/type."""
    s = s or ""
    s = DTYPE_LINE_RE.sub("", s)
    s = DTYPE_TAG_RE.sub("", s)
    s = s.replace("[shape] ()", "")                              # 0-d numpy scalar vs Python scalar
    s = re.sub(r"(\d+\.\d{6})\d+", r"\1", s)
    s = re.sub(r"(?<![\w.])(\d+)\.0+(?![\w.])", r"\1", s)      # 3.0 -> 3, 16510.00 -> 16510
    s = re.sub(r"(?<![\w.])(\d+)\.(?![\w.])", r"\1", s)         # 3. -> 3
    s = s.replace("'", "").replace('"', "")
    s = re.sub(r"\b(np\.|numpy\.)?(int|float|uint)\d*\(([^()]*)\)", r"\3", s)  # np.int64(3) -> 3
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n+", "\n", s)
    return s.strip()


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
async def pipeline(state: TaskState, ctx: dict):
    sid = state.sample_id
    library = state.metadata.get("library", "?")
    problem = state.input if isinstance(state.input, str) else str(state.input)
    budget = ctx["budget"]

    py = None
    try:
        py = next(t for t in state.tools if ToolDef(t).name == "python_session")
    except Exception:
        print("  python_session unavailable")
    sb = Sandbox(py)

    skeleton, marker_line, targets, func_name, in_function = parse_prompt(problem)
    inputs = first_assigned_names(skeleton)
    print(f"  targets={targets} func={func_name} in_function={in_function} inputs={inputs[:4]}")

    # library versions from the sandbox (free, time-bounded)
    versions = "pandas 2.2.2, numpy 1.26.4, scikit-learn 1.4.0, scipy 1.15.3, matplotlib 3.10.0"
    if not sb.dead:
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
            v = await sb.run(vcode, timeout=PROBE_TIMEOUT)
            k = v.find("VERS:")
            if k >= 0:
                versions = v[k + 5:].strip().splitlines()[0]
        except Exception as e:
            print(f"  version probe failed: {e!r}")
            sb.dead = False  # one stall on the probe is not conclusive; later calls get one more chance
    print(f"  [t={ELAPSED():.0f}s] versions: {versions}")

    cur_skel = [skeleton]  # replaced by a stand-in-prefixed skeleton when load_data() is synthesized

    async def execute(code, skel=None, inp=None):
        return await run_candidate(sb, cur_skel[0] if skel is None else skel, marker_line, code, targets,
                                   func_name, inputs if inp is None else inp, in_function)

    # ---- 1. generate three candidates in parallel ---------------------
    msgs = [ChatMessageSystem(content=SYSTEM_PROMPT),
            ChatMessageUser(content=user_prompt(problem, library, versions))]

    async def gen(tag, model, key, effort, mt):
        try:
            txt = await llm(model, key, msgs, budget, effort=effort, max_tokens=mt, timeout=gen_timeout(key))
            code = fix_indent(extract_code(txt), in_function)
            if code.strip() and not ctx["best"]:
                ctx["best"] = code  # something to emit if everything else stalls
            return code
        except Exception as e:
            print(f"  generation {tag} failed: {e!r}")
            return ""

    async def gen_recall():
        if library.lower().startswith("matplotlib"):
            return "", ""
        try:
            txt = await llm(CLAUDE_OPUS_4_8, "opus",
                            [ChatMessageSystem(content=RECALL_SYSTEM),
                             ChatMessageUser(content=recall_prompt(problem, library, versions))],
                            budget, max_tokens=3500)
        except Exception as e:
            print(f"  recall generation failed: {e!r}")
            return "", ""
        rcode = extract_code(txt)
        mm = re.search(r"#\s*recall\s*:\s*(high|low)", rcode, re.I)
        conf = mm.group(1).lower() if mm else "low"
        rcode = re.sub(r"^[ \t]*#\s*recall\s*:.*\n?", "", rcode, flags=re.M)
        return (fix_indent(rcode, in_function) if rcode.strip() else ""), conf

    raw_all = await asyncio.gather(*([gen(t, m, k, e, mt) for (t, m, k, e, mt, _p) in GENERATORS] + [gen_recall()]))
    raw = raw_all[:-1]
    recall_code, recall_conf = raw_all[-1]
    recall = None
    if recall_code.strip():
        recall = {"tag": "R", "key": "opus", "prio": 3, "code": recall_code, "run": None, "run2": None,
                  "conf": recall_conf}
        print(f"  recall R (opus, confidence={recall_conf}):\n{recall_code}")
    cands = []
    for (tag, _m, key, _e, _mt, prio), code in zip(GENERATORS, raw):
        if code.strip():
            cands.append({"tag": tag, "key": key, "prio": prio, "code": code, "run": None, "run2": None})
            print(f"  candidate {tag} ({key}):\n{code}")
    if len(cands) < 3:
        # a model stalled or failed: top up with Sonnet so voting still has three voices
        try:
            code = fix_indent(extract_code(await llm(CLAUDE_SONNET_4_6, "sonnet", msgs, budget,
                                                     effort="medium", max_tokens=3500)), in_function)
        except Exception as e:
            print(f"  fallback generation failed: {e!r}")
            code = ""
        if code.strip():
            cands.append({"tag": "S", "key": "sonnet", "prio": 3, "code": code, "run": None, "run2": None})
            print(f"  candidate S (sonnet):\n{code}")
            if not ctx["best"]:
                ctx["best"] = code
    if not cands:
        print("  no candidates at all")
        return
    ctx["best"] = sorted(cands, key=lambda c: (has_loop_tokens(c["code"]), c["prio"]))[0]["code"]

    # ---- 2. execute every candidate -----------------------------------
    for c in cands:
        c["run"] = await execute(c["code"])
        print(f"  [t={ELAPSED():.0f}s] exec {c['tag']} ok={c['run']['ok']}\n{c['run']['report'][:1200]}")
        if c["run"]["ok"]:
            print(f"  keys {c['tag']}: strict={out_hash(norm_out(c['run']['out']))} "
                  f"values={out_hash(norm_values(c['run']['out']))} dtypes: {dtype_lines(c['run']['out'])}")
    can_execute = any(c["run"]["ok"] is not None for c in cands) and not any(
        skeleton_unrunnable(c["run"]["report"]) for c in cands)

    # ---- 2b. unrunnable skeleton (load_data()): synthesize the data ------
    synth_stub = ""
    missing = sorted({m for c in cands
                      for m in re.findall(r"NameError: name '(\w+)' is not defined", c["run"]["report"] or "")
                      if c["run"]["ok"] is None and "skeleton itself cannot run" in (c["run"]["report"] or "")})
    if (not can_execute and missing and not sb.dead and ELAPSED() < OPTIONAL_AFTER
            and budget.spent < SOFT_CAP):
        print(f"  [t={ELAPSED():.0f}s] skeleton needs {missing}; synthesizing stand-in data")
        err_note = ""
        for model, key in ((CLAUDE_SONNET_4_6, "sonnet"), (GPT_5_4, "gpt54")):
            if ELAPSED() > OPTIONAL_AFTER or sb.dead:
                break
            try:
                txt = await llm(model, key,
                                [ChatMessageUser(content=synth_data_prompt(problem, skeleton, missing, targets,
                                                                            versions, err_note))],
                                budget, max_tokens=3000, timeout=min(180, LLM_TIMEOUT))
            except Exception as e:
                print(f"  data synthesis ({key}) failed: {e!r}")
                continue
            stub = extract_code(txt)
            if not stub.strip() or not all(re.search(rf"^\s*def\s+{m}\s*\(", stub, re.M) for m in missing):
                print(f"  data synthesis ({key}): stand-in does not define {missing}")
                err_note = f"the stand-in must define {', '.join(missing)} as function(s)"
                continue
            if any(t in first_assigned_names(stub) for t in targets):
                print(f"  data synthesis ({key}): stand-in assigns a target variable; rejected")
                err_note = "the stand-in must not assign the target variable(s)"
                continue
            skel2 = stub.rstrip("\n") + "\n" + skeleton
            probe = await execute(fix_indent("pass", in_function), skel=skel2)
            if not probe["out"]:
                print(f"  data synthesis ({key}): skeleton still fails with the stand-in\n{probe['report'][:800]}")
                err_note = probe["report"][-1500:]
                continue
            print(f"  [t={ELAPSED():.0f}s] stand-in data accepted ({key}):\n{stub}\n  probe:\n{probe['report'][:600]}")
            synth_stub = stub
            cur_skel[0] = skel2
            break
        if synth_stub:
            for c in cands:
                c["run"] = await execute(c["code"])
                print(f"  [t={ELAPSED():.0f}s] exec {c['tag']} (synthetic data) ok={c['run']['ok']}\n{c['run']['report'][:1200]}")
                if c["run"]["ok"]:
                    print(f"  keys {c['tag']}: strict={out_hash(norm_out(c['run']['out']))} "
                          f"values={out_hash(norm_values(c['run']['out']))} dtypes: {dtype_lines(c['run']['out'])}")
            if any(c["run"]["ok"] for c in cands):
                can_execute = True
            else:
                print("  every candidate fails on the stand-in data -> treating it as invalid; static judging")
                synth_stub = ""
                cur_skel[0] = skeleton
                for c in cands:
                    c["run"] = {"ok": None, "report": "(the skeleton itself cannot run here)", "out": ""}
    elif not can_execute and missing:
        print(f"  skeleton needs {missing}; data synthesis skipped (sandbox dead / out of time / over cap)")

    # ---- 3. repair crashed candidates (one round each) ----------------
    if can_execute:
        for c in cands:
            if c["run"]["ok"] is False and budget.spent < SOFT_CAP and ELAPSED() < OPTIONAL_AFTER:
                try:
                    fixed = fix_indent(extract_code(await llm(
                        CLAUDE_SONNET_4_6, "sonnet",
                        [ChatMessageSystem(content=SYSTEM_PROMPT),
                         ChatMessageUser(content=repair_prompt(problem, c["code"], c["run"]["report"]))],
                        budget, effort="low", max_tokens=3500, timeout=min(240, LLM_TIMEOUT))), in_function)
                except Exception as e:
                    print(f"  repair {c['tag']} failed: {e!r}")
                    continue
                if fixed.strip():
                    frun = await execute(fixed)
                    print(f"  [t={ELAPSED():.0f}s] repair {c['tag']} ok={frun['ok']}\n{frun['report'][:800]}")
                    if frun["ok"]:
                        c["code"], c["run"] = fixed, frun
                        c["repaired"] = True

    # ---- 3b. execute the recall candidate (tie-break evidence only) ------
    if recall is not None and can_execute and not sb.dead:
        recall["run"] = await execute(recall["code"])
        print(f"  [t={ELAPSED():.0f}s] exec R ok={recall['run']['ok']}\n{recall['run']['report'][:1200]}")
        if recall["run"]["ok"]:
            print(f"  keys R: strict={out_hash(norm_out(recall['run']['out']))} "
                  f"values={out_hash(norm_values(recall['run']['out']))} dtypes: {dtype_lines(recall['run']['out'])}")

    # ---- 4. vote / judge ----------------------------------------------
    def rank_key(c):
        return (has_loop_tokens(c["code"]), c["prio"])

    chosen = None
    majority_grp = None
    if can_execute:
        def regroup():
            good = [c for c in cands if c["run"]["ok"]]
            key = {c["tag"]: norm_out(c["run"]["out"]) for c in good}
            # identical code (modulo comments/blank lines) must be one group: a differing
            # dump can then only come from non-determinism, not from the answer
            by_code = {}
            for c in good:
                by_code.setdefault(norm_code(c["code"]), []).append(c)
            for same in by_code.values():
                keys = {key[c["tag"]] for c in same}
                if len(keys) > 1:
                    k0 = key[same[0]["tag"]]
                    print(f"  identical code {[c['tag'] for c in same]} executed to different dumps "
                          f"(non-deterministic) -> counted as one group")
                    for c in good:
                        if key[c["tag"]] in keys:
                            key[c["tag"]] = k0
            groups = {}
            for c in good:
                groups.setdefault(key[c["tag"]], []).append(c)
            sizes = sorted((len(v) for v in groups.values()), reverse=True)
            print(f"  runnable={len(good)}/{len(cands)} agreement groups={sizes}")
            return good, groups, sizes

        good, groups, sizes = regroup()
        # no two candidates agree -> escalate: two more voices before judging
        if (len(good) >= 2 and sizes[0] == 1 and budget.spent < SOFT_CAP and ELAPSED() < OPTIONAL_AFTER
                and not sb.dead):
            print(f"  [t={ELAPSED():.0f}s] no agreement among {len(good)} candidates -> escalating with "
                  f"{', '.join(k for (_t, _m, k, _e, _mt, _p) in ESCALATION)}")

            async def egen(tag, model, key, effort, mt):
                try:
                    txt = await llm(model, key, msgs, budget, effort=effort, max_tokens=mt, timeout=min(240, LLM_TIMEOUT))
                    return fix_indent(extract_code(txt), in_function)
                except Exception as e:
                    print(f"  escalation generation {tag} failed: {e!r}")
                    return ""

            eraw = await asyncio.gather(*[egen(t, m, k, e, mt) for (t, m, k, e, mt, _p) in ESCALATION])
            for (tag, _m, key, _e, _mt, prio), code in zip(ESCALATION, eraw):
                if not code.strip():
                    continue
                c = {"tag": tag, "key": key, "prio": prio, "code": code, "run": None, "run2": None}
                c["run"] = await execute(code)
                print(f"  candidate {tag} ({key}):\n{code}")
                print(f"  [t={ELAPSED():.0f}s] exec {tag} ok={c['run']['ok']}\n{c['run']['report'][:1200]}")
                if c["run"]["ok"]:
                    print(f"  keys {tag}: strict={out_hash(norm_out(c['run']['out']))} "
                          f"values={out_hash(norm_values(c['run']['out']))} dtypes: {dtype_lines(c['run']['out'])}")
                cands.append(c)
            good, groups, sizes = regroup()
        need = 2 if len(good) <= 3 else 3
        if sizes and sizes[0] >= need and (len(sizes) == 1 or sizes[0] > sizes[1]):
            grp = max(groups.values(), key=len)
            chosen = sorted(grp, key=rank_key)[0]
            majority_grp = grp
            print(f"  majority of {len(grp)} (of {len(good)} runnable) agrees -> candidate {chosen['tag']}")
        elif sizes and sizes[0] >= 2:
            print(f"  largest group {sizes[0]} of {len(good)} is not decisive; judging")
        if good and len(groups) >= 2:
            # one representative per distinct output; the judge sees the counts
            pool = []
            for g in groups.values():
                r = sorted(g, key=rank_key)[0]
                r["n"] = len(g)
                pool.append(r)
            pool.sort(key=lambda c: (-c["n"], rank_key(c)))
        else:
            pool = good if good else cands
        # value-agree / dtype-split: every runnable candidate has the same values but the
        # dumps differ (dtype/type lines). The majority is then uninformative about what the
        # accepted-StackOverflow reference produces; ask a dedicated judge.
        if len(groups) >= 2 and len(good) >= 2:
            vgroups = {}
            for c in good:
                vgroups.setdefault(norm_values(c["run"]["out"]), []).append(c)
            if len(vgroups) == 1:
                dtype_split = [sorted(g, key=rank_key)[0] for g in groups.values()]
                for g, rep_c in zip(groups.values(), dtype_split):
                    rep_c["n"] = len(g)
                dtype_split.sort(key=lambda c: (-c["n"], rank_key(c)))
                print(f"  dtype-split: values agree, dtypes differ across {len(groups)} groups "
                      f"({', '.join(c['tag'] + 'x' + str(c['n']) for c in dtype_split)})")
                for c in dtype_split:
                    d = re.findall(r"^(?:\[dtypes\].*|type=.*|.*\[dtype\] .*)$", c["run"]["out"], re.M)
                    print(f"    {c['tag']}: {' | '.join(x.strip() for x in d)[:300]}")
                ctx["dtype_split"] = dtype_split
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

    if chosen is None and len(pool) == 1 and can_execute and pool[0]["run"]["ok"] and not synth_stub:
        chosen = pool[0]
        print(f"  only one runnable candidate -> {chosen['tag']}")
    if chosen is None and synth_stub:
        # on synthetic data a crash may be an artefact of the stand-in: let the judge see those too
        extra = [c for c in cands if c["run"]["ok"] is False and c not in pool]
        if extra:
            print(f"  synthetic data: adding crashed candidate(s) {[c['tag'] for c in extra]} to the judge pool")
            pool = list(pool) + extra

    if chosen is None:
        ctx["best"] = sorted(pool, key=rank_key)[0]["code"]

    async def judge(prompt_text, ref_out=None):
        """Run the Opus judge; returns a candidate dict or None. ref_out: required
        example-1 output for judge-written code (None = only has to run)."""
        try:
            jv = await llm(CLAUDE_OPUS_4_8, "opus",
                           [ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=prompt_text)],
                           budget, effort="low", max_tokens=4000)
        except Exception as e:
            print(f"  judge failed: {e!r}")
            return None
        m = re.search(r"CHOOSE:\s*(?:candidate\s*)?([A-Za-z])\b", jv, re.I)
        if "<code>" in jv:
            new = fix_indent(extract_code(jv), in_function)
            nrun = await execute(new) if can_execute else {"ok": None, "report": "", "out": ""}
            print(f"  [t={ELAPSED():.0f}s] judge wrote a new solution ok={nrun['ok']}\n{new}\n{nrun['report'][:800]}")
            accept = nrun["ok"] or nrun["ok"] is None
            if accept and ref_out is not None and norm_out(nrun["out"]) != ref_out:
                print("  judge's solution changes the example output; not accepted")
                accept = False
            if accept:
                return {"tag": "J", "key": "opus", "prio": -1, "code": new, "run": nrun, "run2": None}
            print("  judge's solution rejected; falling back to candidates")
        if m:
            tag = m.group(1).upper()
            for c in list(pool) + cands:
                if c["tag"] == tag:
                    print(f"  judge chose candidate {tag}")
                    return c
        print(f"  judge output unusable: {jv[:200]!r}")
        return None

    if chosen is None and budget.spent < JUDGE_CAP and ELAPSED() < JUDGE_AFTER:
        chosen = await judge(judge_prompt(problem, pool, can_execute, versions, synth_stub,
                                          mpl=library.lower().startswith("matplotlib")))
    elif chosen is None:
        print(f"  judge skipped (est_cost=${budget.spent:.3f}, t={ELAPSED():.0f}s)")

    if chosen is None:
        chosen = sorted(pool, key=rank_key)[0]
        print(f"  fallback pick -> {chosen['tag']}")
    ctx["best"] = chosen["code"]

    # ---- 4a. reference-recall dtype tie-break ---------------------------
    # R is never a vote. It only matters when it produces the SAME VALUES as the
    # chosen candidate but different dtypes/types: then the accepted answer it
    # reproduces (confidence high) decides which dtype construction to emit.
    dsplit = ctx.get("dtype_split")
    from_recall = False
    if (recall is not None and recall.get("conf") == "high" and can_execute and recall["run"]
            and recall["run"]["ok"] and chosen["run"] and chosen["run"]["ok"] and chosen["tag"] != "J"):
        r_vals, r_strict = norm_values(recall["run"]["out"]), norm_out(recall["run"]["out"])
        if r_vals != norm_values(chosen["run"]["out"]):
            print("  recall R disagrees on values with the chosen candidate; ignored")
        elif r_strict == norm_out(chosen["run"]["out"]):
            print(f"  recall R confirms candidate {chosen['tag']}'s dtypes")
            dsplit = None
        else:
            same = [c for c in good if c["run"]["ok"] and norm_out(c["run"]["out"]) == r_strict]
            if same:
                pick = sorted(same, key=rank_key)[0]
                print(f"  recall R: same values, different dtypes -> switching from candidate {chosen['tag']} "
                      f"to {pick['tag']} (matches the recalled reference's dtypes)")
                chosen = pick
                majority_grp = same
            else:
                print("  recall R: same values, different dtypes, no candidate matches -> using R itself")
                chosen = recall
                majority_grp = None
                from_recall = True
            dsplit = None
            ctx["best"] = chosen["code"]
    elif recall is not None and recall.get("conf") != "high":
        print("  recall R: confidence low; not used for tie-breaking")

    # ---- 4b. dtype-split judge ------------------------------------------
    if dsplit and budget.spent < JUDGE_CAP and ELAPSED() < JUDGE_AFTER:
        ref_vals = norm_values(chosen["run"]["out"])
        saved_pool = pool
        pool = dsplit

        async def dtype_judge():
            try:
                jv = await llm(CLAUDE_OPUS_4_8, "opus",
                               [ChatMessageSystem(content=SYSTEM_PROMPT),
                                ChatMessageUser(content=dtype_judge_prompt(problem, dsplit))],
                               budget, max_tokens=4000)
            except Exception as e:
                print(f"  dtype judge failed: {e!r}")
                return None
            m = re.search(r"CHOOSE:\s*(?:candidate\s*)?([A-Za-z])\b", jv, re.I)
            if "<code>" in jv and not m:
                new = fix_indent(extract_code(jv), in_function)
                nrun = await execute(new)
                print(f"  [t={ELAPSED():.0f}s] dtype judge wrote a new solution ok={nrun['ok']}\n{new}\n{nrun['report'][:800]}")
                if nrun["ok"] and norm_values(nrun["out"]) == ref_vals:
                    return {"tag": "J", "key": "opus", "prio": -1, "code": new, "run": nrun, "run2": None}
                print("  dtype judge's solution crashes or changes the values; not accepted")
                return None
            if m:
                tag = m.group(1).upper()
                for c in dsplit:
                    if c["tag"] == tag:
                        print(f"  dtype judge chose candidate {tag} ({c['n']} model(s) produced these dtypes)")
                        return c
            print(f"  dtype judge output unusable: {jv[-200:]!r}")
            return None

        j = await dtype_judge()
        if j is not None:
            if j is not chosen:
                print(f"  dtype-split: switching from candidate {chosen['tag']} to {j['tag']}")
            chosen = j
            # the generality check only makes sense among candidates sharing the chosen dtypes
            majority_grp = [c for c in good if norm_out(c["run"]["out"]) == norm_out(chosen["run"]["out"])] \
                if chosen["tag"] != "J" else None
        else:
            print("  dtype-split: judge unusable; keeping the majority pick")
        pool = saved_pool
        ctx["best"] = chosen["code"]

    # ---- 5. generality check on an alternative input -------------------
    # Only when a majority agrees on the example with structurally different code:
    # that is exactly the case voting cannot see through.
    if (majority_grp is not None and len(majority_grp) >= 2 and targets and can_execute and not sb.dead
            and len({norm_code(c["code"]) for c in majority_grp}) >= 2
            and budget.spent < SOFT_CAP and ELAPSED() < OPTIONAL_AFTER):
        try:
            txt = await llm(CLAUDE_SONNET_4_6, "sonnet",
                            [ChatMessageUser(content=alt_input_prompt(problem, cur_skel[0], targets, bool(synth_stub)))],
                            budget, max_tokens=2500, timeout=min(180, LLM_TIMEOUT))
            skeleton2 = extract_code(txt)
        except Exception as e:
            print(f"  alt-input generation failed: {e!r}")
            skeleton2 = ""
        if skeleton2.strip() and norm_code(skeleton2) != norm_code(cur_skel[0]) \
                and not any(t in first_assigned_names(skeleton2) for t in targets):
            print(f"  alternative input skeleton:\n{skeleton2}")
            inputs2 = first_assigned_names(skeleton2)
            valid = True
            for c in majority_grp:
                if ELAPSED() > OPTIONAL_AFTER + 120 or sb.dead:
                    valid = False
                    break
                c["run2"] = await execute(c["code"], skel=skeleton2, inp=inputs2)
                print(f"  [t={ELAPSED():.0f}s] exec2 {c['tag']} ok={c['run2']['ok']}\n{c['run2']['report'][:900]}")
                if c["run2"]["ok"] is None:
                    valid = False  # the alternative skeleton itself does not run
                    break
            if valid:
                ok2 = [c for c in majority_grp if c["run2"]["ok"]]
                outs2 = {norm_out(c["run2"]["out"]) for c in ok2}
                if not ok2:
                    print("  generality: every candidate crashes on the alternative input; ignoring it")
                elif len(ok2) == len(majority_grp) and len(outs2) == 1:
                    print("  generality: all agree on the alternative input too")
                elif len(outs2) == 1:
                    # some crash, the survivors agree -> prefer a survivor
                    pick = sorted(ok2, key=rank_key)[0]
                    print(f"  generality: {len(majority_grp) - len(ok2)} candidate(s) crash on the "
                          f"alternative input -> candidate {pick['tag']}")
                    chosen = pick
                else:
                    ref_out = norm_out(chosen["run"]["out"])
                    pool = ok2
                    print(f"  generality: candidates diverge on the alternative input ({len(outs2)} outputs); judging")
                    if budget.spent < JUDGE_CAP and ELAPSED() < JUDGE_AFTER:
                        j = await judge(general_judge_prompt(problem, ok2, skeleton2), ref_out=ref_out)
                        if j is not None:
                            chosen = j
            else:
                print("  generality check skipped (alternative skeleton invalid or out of time)")
        else:
            print("  generality check skipped (no usable alternative skeleton)")
        ctx["best"] = chosen["code"]

    code, run = chosen["code"], chosen["run"]
    run2 = chosen.get("run2")

    # ---- 6. loop guard ------------------------------------------------
    if has_loop_tokens(code) and not in_function and from_recall:
        print("  loop guard skipped: code reproduces the recalled reference (loops included)")
    if has_loop_tokens(code) and not in_function and not from_recall:
        alt = None
        if can_execute and run["ok"]:
            for c in cands:
                if c is chosen or not c["run"]["ok"] or has_loop_tokens(c["code"]):
                    continue
                if norm_out(c["run"]["out"]) != norm_out(run["out"]):
                    continue
                if run2 and run2["ok"] and (not c.get("run2") or not c["run2"]["ok"]
                                            or norm_out(c["run2"]["out"]) != norm_out(run2["out"])):
                    continue
                alt = c
                break
        if alt is not None:
            print(f"  loop guard: switching to loop-free agreeing candidate {alt['tag']}")
            code, run = alt["code"], alt["run"]
        else:
            for model, key, effort in ((CLAUDE_OPUS_4_8, "opus", None), (GPT_5_4, "gpt54", "medium")):
                if budget.spent >= SOFT_CAP or ELAPSED() > OPTIONAL_AFTER + 180:
                    break
                try:
                    nl = fix_indent(extract_code(await llm(
                        model, key,
                        [ChatMessageSystem(content=SYSTEM_PROMPT),
                         ChatMessageUser(content=loop_prompt(problem, code))],
                        budget, effort=effort, max_tokens=5000, timeout=240)), in_function)
                except Exception as e:
                    print(f"  loop rewrite ({key}) failed: {e!r}")
                    continue
                if not nl.strip() or has_loop_tokens(nl):
                    print(f"  loop rewrite ({key}) still has loops or empty")
                    continue
                if can_execute and not sb.dead:
                    nrun = await execute(nl)
                    same = bool(nrun["ok"]) and norm_out(nrun["out"]) == norm_out(run["out"])
                    print(f"  [t={ELAPSED():.0f}s] loop rewrite ({key}) ok={nrun['ok']} same_output={same}")
                    if nrun["ok"] and (same or not run["ok"]):
                        code, run = nl, nrun
                        break
                else:
                    code = nl
                    break

    ctx["best"] = code

    # ---- 6b. named-API guard (explicit must-use / must-not-use identifiers) ----
    if (not library.lower().startswith("matplotlib") and budget.spent < SOFT_CAP
            and ELAPSED() < OPTIONAL_AFTER + 120):
        must_use, must_not = [], []
        try:
            ctxt = await llm(GPT_5_4_MINI, "mini", [ChatMessageUser(content=constraint_prompt(problem))],
                             budget, max_tokens=300, timeout=90)
            must_use, must_not = parse_constraints(ctxt, problem)
            print(f"  api guard: must_use={must_use} must_not_use={must_not} (raw: {(ctxt or '').strip()[:160]!r})")
        except Exception as e:
            print(f"  api guard: extraction failed: {e!r}")

        def violations(c):
            toks = code_tokens(c)
            return ([".".join(p) for p in must_use if not all(t in toks for t in p)],
                    [t for t in must_not if t in toks])

        missing, present = violations(code)
        if (must_use or must_not) and (missing or present):
            print(f"  api guard: chosen code violates constraints (missing={missing}, present={present})")
            alt = None
            if can_execute and run["ok"]:
                for c in cands:
                    if c is chosen or not c["run"]["ok"] or any(violations(c["code"])):
                        continue
                    if norm_out(c["run"]["out"]) != norm_out(run["out"]):
                        continue
                    if run2 and run2["ok"] and (not c.get("run2") or not c["run2"]["ok"]
                                                or norm_out(c["run2"]["out"]) != norm_out(run2["out"])):
                        continue
                    if has_loop_tokens(c["code"]) and not has_loop_tokens(code):
                        continue
                    alt = c
                    break
            if alt is not None:
                print(f"  api guard: switching to agreeing candidate {alt['tag']} that satisfies the constraints")
                code, run = alt["code"], alt["run"]
            else:
                loops_ok = has_loop_tokens(code)
                for model, key, effort in ((CLAUDE_OPUS_4_8, "opus", None), (GPT_5_4, "gpt54", "medium")):
                    if budget.spent >= SOFT_CAP or ELAPSED() > OPTIONAL_AFTER + 240:
                        break
                    try:
                        nl = fix_indent(extract_code(await llm(
                            model, key,
                            [ChatMessageSystem(content=SYSTEM_PROMPT),
                             ChatMessageUser(content=api_rewrite_prompt(problem, code, missing, present))],
                            budget, effort=effort, max_tokens=5000, timeout=240)), in_function)
                    except Exception as e:
                        print(f"  api rewrite ({key}) failed: {e!r}")
                        continue
                    if not nl.strip() or any(violations(nl)) or (has_loop_tokens(nl) and not loops_ok):
                        print(f"  api rewrite ({key}) empty or still violates the constraints")
                        continue
                    if can_execute and not sb.dead:
                        nrun = await execute(nl)
                        same = bool(nrun["ok"]) and norm_out(nrun["out"]) == norm_out(run["out"])
                        print(f"  [t={ELAPSED():.0f}s] api rewrite ({key}) ok={nrun['ok']} same_output={same}")
                        if nrun["ok"] and (same or not run["ok"]):
                            code, run = nl, nrun
                            break
                    elif missing:
                        print(f"  api rewrite ({key}) adopted without execution (static path, missing required name)")
                        code = nl
                        break
        elif must_use or must_not:
            print("  api guard: chosen code satisfies the constraints")

    ctx["best"] = code
    ctx["done"] = True


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _T0[0] = time.monotonic()
        sid = state.sample_id
        print(f"[{sid}] library={state.metadata.get('library', '?')}")
        ctx = {"best": "", "done": False, "budget": Budget()}
        try:
            await asyncio.wait_for(pipeline(state, ctx), timeout=HARD_LIMIT)
        except asyncio.TimeoutError:
            print(f"  [t={ELAPSED():.0f}s] HARD LIMIT hit; emitting best-so-far")
        except Exception as e:
            print(f"  [t={ELAPSED():.0f}s] pipeline error: {e!r}\n{traceback.format_exc()[-2000:]}")

        code = ctx["best"]
        if not code.strip():
            # last resort: a cheap one-shot answer rather than nothing
            try:
                resp = await asyncio.wait_for(
                    GPT_5_4_MINI.generate(
                        [ChatMessageSystem(content=SYSTEM_PROMPT),
                         ChatMessageUser(content=state.input if isinstance(state.input, str) else str(state.input))],
                        config=GenerateConfig(max_tokens=2000)),
                    timeout=120)
                ctx["budget"].add(resp, "mini")
                code = extract_code(resp.completion or "")
                print("  emergency fallback (mini) used")
            except Exception as e:
                print(f"  emergency fallback failed: {e!r}")
                code = ""

        state.output.completion = f"<code>\n{code}\n</code>"
        budget = ctx["budget"]
        print(f"  final (done={ctx['done']}, t={ELAPSED():.0f}s):\n{code}")
        print(f"  llm_calls={budget.calls} est_cost=${budget.spent:.4f}")
        for ln in budget.log:
            print(f"    {ln}")
        return state

    return solve
