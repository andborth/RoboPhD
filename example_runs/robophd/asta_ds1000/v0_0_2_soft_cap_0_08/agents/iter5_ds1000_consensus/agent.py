"""DS-1000 solver — iter5_ds1000_consensus.

Base = iter4_ds1000_idiom_probe (100% on the iter4 draw): strong reasoning
model + deterministic format/indent preservation + idiom/loop detection +
invent-signature function probe + traceback repair + Opus escalation.

iter4's structural blind spot — shared by every prior agent — is that its
sandbox verification is CRASH-DETECTION ONLY. The single failure that
separated the agents in iter4 (problem 427) was runnable-but-wrong output
(shape (m,) instead of (1, m)); a traceback-only check can never see it.

This agent keeps iter4's pipeline verbatim and adds the missing layer:
CROSS-MODEL EXECUTION CONSENSUS. For runnable, non-Matplotlib problems it
generates candidates from three independent model families, executes each in
the sandbox, computes a float-rounded shape/dtype-aware canonical signature
of the target variable(s), and votes. Independent architectures rarely share
the same subtle runnable-but-wrong bias, so value agreement is strong
evidence of correctness and disagreement flags exactly the traps that beat
iter2/iter3. On disagreement it falls back to iter4's repair loop + an Opus
tie-breaker that also acts as a 4th independent vote.

Every stage falls back gracefully so a wrapped <code> block is always
emitted; worst case it degrades to iter4's behavior, never below it.
"""

import ast
import re
import textwrap

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import (
    CLAUDE_SONNET_4_6,
    CLAUDE_OPUS_4_7,
    GPT_5_5,
    GEMINI_3_1_PRO_PREVIEW,
)

MODEL = CLAUDE_SONNET_4_6
GEN_CONFIG = GenerateConfig(reasoning_effort="high", max_tokens=4096)
STRONG_MODEL = CLAUDE_OPUS_4_7
STRONG_CONFIG = GenerateConfig(reasoning_effort="high", max_tokens=4096)

# Independent model families for cross-model consensus. Order = tie-break
# priority (Sonnet has the strongest proven DS-1000 track record).
PANEL = [
    ("sonnet", CLAUDE_SONNET_4_6, GenerateConfig(reasoning_effort="high",
                                                 max_tokens=4096)),
    ("gpt5", GPT_5_5, GenerateConfig(reasoning_effort="medium",
                                     max_tokens=8192)),
    ("gemini", GEMINI_3_1_PRO_PREVIEW, GenerateConfig(max_tokens=4096)),
]

MAX_REPAIRS = 3
MAX_IDIOM_REGEN = 2

SYSTEM_GUIDANCE = """\
You are an expert Python data-science engineer solving a DS-1000 problem.

You are given a problem with a code skeleton. The skeleton (imports + input
setup) is ALREADY written and runs BEFORE your code. Write ONLY the new code
that produces the requested answer. Do not repeat imports or data setup.

OUTPUT CONTRACT (strict):
- Output exactly one block: <code> ... </code>.
- Inside: executable Python only. No prose, no explanation, no markdown ```
  fences, no "BEGIN/END SOLUTION" markers.
- Write the solution as NORMAL TOP-LEVEL Python with NO extra leading
  indentation (start every line at column 0, using only the indentation your
  own loops/blocks need). The harness will place it at the correct insertion
  point automatically.
- {target_line}
- Do NOT add driver/test calls (e.g. `result = f(...)`, `print(...)`) unless
  the problem explicitly shows that the function must be called in the
  solution. For a function-body insertion, emit only the body (ending in
  `return ...` if appropriate) and nothing else.
- Nothing outside the <code>...</code> tags.

THINK CAREFULLY — DS-1000 is full of traps. Before answering, check:
- READ LITERALLY. Negations/inversions ("the values that are 0", "excluding",
  "not", "drop", "reverse", "all but", "opposite") usually mean invert a mask
  or reverse an order, NOT the obvious quantity. Use the EXACT thing named:
  if it says "hatch", use the `hatch=` argument, not `marker=`; if it names a
  specific argument or function, use that exact one.
- FOLLOW LITERAL SKELETON COMMENT HINTS. If a skeleton comment says e.g.
  `# Save the model in "export/1"` or names a path/value, use exactly that.
- MATPLOTLIB MARKER/STYLE CODES are exact: thin diamond = 'd', (fat) diamond
  = 'D', star = '*', point = '.'. "hatch" patterns ('*','/','x',...) go in
  `hatch=`. Do not add markersize/linewidth/color/labels that were not asked
  for; do not call plt.show().
- EXACT OUTPUT TYPE & SHAPE: DataFrame vs Series vs ndarray vs scalar vs
  list; preserve dtype, index, column names and row/column order exactly as
  any "Desired Output" shows. 2D matrix vs single column matters — if the
  text asks for a "(1, m) matrix" or "(n, 1)" the answer must be 2D, not a
  flat vector. When the natural/idiomatic construction (e.g.
  `np.column_stack` of mixed string+int columns) upcasts dtype, match that
  idiomatic construction — the expected output may be object/string, not the
  naive int.
- IDIOM CONSTRAINTS: "without a loop", "vectorized", "the efficient/clean
  way", "not one by one", "most idiomatic", or a named function => you MUST
  use the idiomatic library call; a manual loop/reimplementation is rejected
  even if numbers are right. The grader may tokenize your code and reject ANY
  `for` or `while` token — that includes list/dict/set comprehensions and
  generator expressions. Prefer vectorized ops (`.map` with a format string,
  broadcasting, `np.where`, `.agg`, `.stack`).
- USE CURRENT, NON-DEPRECATED APIs: e.g. `scipy.integrate.simpson` (not the
  removed `simps`), `numpy`/`pandas`/`sklearn` modern signatures.
- WORKED EXAMPLES: if the prompt shows example input and desired output,
  mentally execute your solution on it and confirm it reproduces the output
  exactly before finalizing.
- Pick the correct well-known library function (right SciPy interpolator,
  right sklearn helper) rather than an approximate substitute.
{func_line}
Respond with ONLY the <code>...</code> block.
"""

_FUNC_GUIDANCE = """\
- THIS PROBLEM ASKS YOU TO DEFINE A FUNCTION but the skeleton does NOT give
  the `def` line. The hidden test will CALL your function. It passes only the
  PRINCIPAL example input the problem discusses; other skeleton-defined values
  (bounds, min/max, n, k, thresholds, config) are MODULE GLOBALS your function
  should reference directly, NOT extra parameters. Keep the signature minimal
  — usually a single argument matching the example input. Mirror the simplest
  signature the example implies."""


def _all_code_blocks(text: str):
    return re.findall(r"<code>(.*?)</code>", text, re.DOTALL)


def _detect_targets(prompt: str) -> str:
    targets = _target_names(prompt)
    if not targets:
        return ("Assign the variable the problem asks for (commonly "
                "`result`). Use that exact name; do not also print it.")
    if len(targets) == 1:
        return (f"Assign the exact target variable `{targets[0]}` (use that "
                f"exact name; do not rename or print it).")
    return ("Assign each exact target variable: "
            + ", ".join(f"`{t}`" for t in targets) + ".")


def _target_names(prompt: str):
    targets = []
    for m in re.finditer(
        r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\..*?put .*?in this variable",
        prompt, re.MULTILINE,
    ):
        targets.append(m.group(1))
    for m in re.finditer(r"put [^,]*? in `([A-Za-z_]\w*)`", prompt):
        targets.append(m.group(1))
    for m in re.finditer(r"#\s*([A-Za-z_]\w*)\s*=\s*\w+\(", prompt):
        targets.append(m.group(1))
    if not targets:
        for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.",
                             prompt, re.MULTILINE):
            targets.append(m.group(1))
    seen, uniq = set(), []
    for t in targets:
        if t not in seen and t not in ("def", "return"):
            seen.add(t)
            uniq.append(t)
    return uniq


_SENTINEL_RE = re.compile(
    r"Write the remaining python code to append|Put your answer inside",
    re.IGNORECASE,
)
_CODE_START_RE = re.compile(
    r"^\s*(import |from |def |class |@|#|[A-Za-z_]\w*\s*[:=]|"
    r"plt\.|with |for |if |try:|return |print\()"
)


def _parse_skeleton(prompt: str):
    """Return (skeleton_code, is_completion)."""
    m = _SENTINEL_RE.search(prompt)
    head = prompt[: m.start()] if m else prompt

    closed = re.findall(r"<code>(.*?)</code>", head, re.DOTALL)
    if closed:
        body, is_completion, had_tag = closed[-1], True, True
    else:
        idx = head.rfind("<code>")
        had_tag = idx != -1
        body = head[idx + len("<code>"):] if had_tag else head
        is_completion = False

    lines = body.split("\n")
    while lines and (
        not lines[-1].strip()
        or lines[-1].strip() in ("</code>", "```", "```python", "```py")
    ):
        lines.pop()

    if not had_tag:
        start = 0
        for i, ln in enumerate(lines):
            if _CODE_START_RE.match(ln):
                start = i
                break
        lines = lines[start:]

    return "\n".join(lines), is_completion


def _skeleton_block(prompt: str) -> str:
    return _parse_skeleton(prompt)[0]


def _base_indent(prompt: str) -> int:
    skel, is_completion = _parse_skeleton(prompt)
    if is_completion:
        return 0
    nonempty = [ln for ln in skel.split("\n") if ln.strip()]
    if not nonempty:
        return 0
    last = nonempty[-1]
    indent = len(last) - len(last.lstrip())
    if last.rstrip().endswith(":"):
        indent += 4
    return indent if 0 <= indent <= 16 else 0


def _extract_code(text: str) -> str:
    if not text:
        return ""
    blocks = _all_code_blocks(text)
    if blocks:
        code = blocks[-1]
    else:
        s = text.strip()
        fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", s, re.DOTALL)
        code = fence.group(1) if fence else s
    return code.strip("\n")


def _clean(code: str) -> str:
    code = re.sub(r"^\s*#*\s*(BEGIN|END)\s+SOLUTION\s*$", "",
                  code, flags=re.MULTILINE)
    code = re.sub(r"^\s*```(?:python|py)?\s*$", "", code,
                  flags=re.MULTILINE)
    return code.strip("\n")


def _reindent(code: str, base: int) -> str:
    code = _clean(code)
    if not code.strip():
        return code
    code = textwrap.dedent(code)
    if base <= 0:
        return code.rstrip()
    pad = " " * base
    out = []
    for ln in code.split("\n"):
        out.append(pad + ln if ln.strip() else "")
    return "\n".join(out).rstrip()


_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|^\w*Error:|Exception:",
    re.MULTILINE,
)


def _has_error(out: str) -> bool:
    return bool(out) and bool(_TRACEBACK_RE.search(out))


def _runnable(skel: str) -> bool:
    if not skel.strip():
        return False
    if re.search(r"\b(load_data|load_iris|load_\w+|fetch_\w+|read_csv|"
                 r"read_excel|read_pickle|read_table)\s*\(", skel):
        return False
    return True


_ENV_NOISE_RE = re.compile(
    r"MessageFactory|GetPrototype|could not be resolved|"
    r"DLL load failed|cannot import name|No module named|"
    r"libcu|CUDA|protobuf",
    re.IGNORECASE,
)


def _env_noise(out: str) -> bool:
    return bool(out) and bool(_ENV_NOISE_RE.search(out))


def _actionable_error(out: str) -> bool:
    """A real traceback whose deepest frame is in the executed program."""
    if not _has_error(out):
        return False
    return ('File "<string>"' in out
            or re.search(r"/ds1000/test_\d+\.py", out) is not None)


def _missing_skeleton_name(out: str, skel: str) -> bool:
    m = re.search(r"name '(\w+)' is not defined", out)
    if not m:
        return False
    name = m.group(1)
    return ("load_data" in skel or "load_data" in out
            or re.search(rf"\b{name}\s*=", skel) is not None)


# ---- Idiom / style-constraint detection -----------------------------------

_IDIOM_RE = re.compile(
    r"\bidiomatic\b|most idiomatic|without (a |an |using )?(for |while )?loop"
    r"|without loops?|no loops?|not one by one|one[- ]?liner|in one line"
    r"|vectoriz|the (most )?(efficient|clean(est)?) way|element[- ]?wise"
    r"|do not use .{0,20}loop|don't use .{0,20}loop",
    re.IGNORECASE,
)


def _has_idiom_constraint(prompt: str) -> bool:
    return bool(_IDIOM_RE.search(prompt or ""))


def _has_loop(code: str) -> bool:
    """True if code contains a for/while statement OR any comprehension /
    generator expression."""
    if not code or not code.strip():
        return False
    try:
        tree = ast.parse(textwrap.dedent(_clean(code)))
    except SyntaxError:
        return bool(re.search(r"\bfor\b|\bwhile\b", code))
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While,
                             ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)):
            return True
    return False


# ---- Invent-signature function detection ----------------------------------

_DEF_FUNC_RE = re.compile(
    r"define (?:a |the )?function (?:named |called )?[`']?(\w+)[`']?"
    r"|named [`']?(\w+)[`']? as solution"
    r"|#\s*(?:return the solution in this function)",
    re.IGNORECASE,
)


def _invent_function(prompt: str, skel: str):
    """If the problem asks to DEFINE a function and the skeleton does NOT
    already supply the `def` line, return (func_name, [example_input_vars])."""
    if re.search(r"^\s*def\s+\w+\s*\(", skel, re.MULTILINE):
        return None, []
    m = _DEF_FUNC_RE.search(prompt)
    if not m:
        return None, []
    name = next((g for g in m.groups() if g), None)
    if not name:
        return None, []
    config = re.compile(
        r"^(.*?_)?(min|max|lower|upper|lb|ub|low|high|n|k|size|len|num|"
        r"threshold|eps|tol|seed|bins|deg|order|alpha|beta|gamma|lr)$",
        re.IGNORECASE,
    )
    vars_ = []
    for vm in re.finditer(
        r"^([A-Za-z_]\w*)\s*=\s*(?!\.\.\.)\S", skel, re.MULTILINE
    ):
        v = vm.group(1)
        if v in ("def", "return", "import", "from") or config.match(v):
            continue
        if v not in vars_:
            vars_.append(v)
    return name, vars_[:2]


# ---- Canonical signature for execution consensus --------------------------

_SIG_MARK = "___DS1000_SIG___"
_OK_MARK = "___DS1000_RAN_OK___"


def _sig_harness(targets) -> str:
    """Sandbox code: print a float-rounded, shape/dtype-aware canonical
    signature of each target variable so numerically-equal answers from
    different models group together while shape/dtype/order differences
    split apart. Robust to any object type; never raises out of the run."""
    names = targets or ["result"]
    names_lit = repr(list(names))
    return (
        "\ndef __ds1000_sig(x):\n"
        "    try:\n"
        "        import numpy as _np\n"
        "    except Exception:\n"
        "        _np = None\n"
        "    try:\n"
        "        import pandas as _pd\n"
        "    except Exception:\n"
        "        _pd = None\n"
        "    try:\n"
        "        if _pd is not None and isinstance(x, _pd.DataFrame):\n"
        "            y = x.copy()\n"
        "            try:\n"
        "                num = y.select_dtypes('number').columns\n"
        "                y[num] = y[num].round(6)\n"
        "            except Exception:\n"
        "                pass\n"
        "            return ('DF|' + repr(list(y.columns)) + '|'\n"
        "                    + repr(list(y.index)) + '|'\n"
        "                    + y.to_csv())\n"
        "        if _pd is not None and isinstance(x, _pd.Series):\n"
        "            try:\n"
        "                xv = x.round(6)\n"
        "            except Exception:\n"
        "                xv = x\n"
        "            return ('SR|' + str(x.dtype) + '|'\n"
        "                    + repr(list(x.index)) + '|' + xv.to_csv())\n"
        "        if _np is not None and isinstance(x, _np.ndarray):\n"
        "            return ('ND|' + str(x.dtype) + '|' + str(x.shape)\n"
        "                    + '|' + _np.array2string(x, precision=6,\n"
        "                      threshold=100000, max_line_width=100000,\n"
        "                      suppress_small=True))\n"
        "        if _np is not None and isinstance(x, _np.generic):\n"
        "            return 'NS|' + repr(round(float(x), 6))\n"
        "        if hasattr(x, 'detach') and hasattr(x, 'cpu'):\n"
        "            try:\n"
        "                import numpy as _n2\n"
        "                a = x.detach().cpu().numpy()\n"
        "                return ('TZ|' + str(a.dtype) + str(a.shape) + '|'\n"
        "                        + _n2.array2string(a, precision=6,\n"
        "                          threshold=100000, max_line_width=100000,\n"
        "                          suppress_small=True))\n"
        "            except Exception:\n"
        "                return 'TZx|' + repr(x)\n"
        "        if hasattr(x, 'numpy') and not isinstance(x, type):\n"
        "            try:\n"
        "                import numpy as _n3\n"
        "                a = x.numpy()\n"
        "                return ('TF|' + str(getattr(a, 'dtype', '')) + '|'\n"
        "                        + _n3.array2string(_n3.asarray(a),\n"
        "                          precision=6, threshold=100000,\n"
        "                          max_line_width=100000,\n"
        "                          suppress_small=True))\n"
        "            except Exception:\n"
        "                return 'TFx|' + repr(x)\n"
        "        if isinstance(x, float):\n"
        "            return 'FL|' + repr(round(x, 6))\n"
        "        return 'PY|' + repr(x)\n"
        "    except Exception as _e:\n"
        "        return 'SIGERR|' + repr(_e)\n"
        f"for __n in {names_lit}:\n"
        "    try:\n"
        f"        print('{_SIG_MARK}', __n, '=', "
        "__ds1000_sig(eval(__n)))\n"
        "    except Exception as __e:\n"
        f"        print('{_SIG_MARK}', __n, '= MISSING|' + repr(__e))\n"
        f"print('{_OK_MARK}')"
    )


def _parse_sig(out: str):
    """Extract the concatenated canonical signature from sandbox output.
    Returns None if the run errored before/at the signature stage."""
    if not out or _OK_MARK not in out:
        return None
    if "= MISSING|" in out:
        return None
    if _has_error(out):
        return None
    esc = re.escape(_SIG_MARK)
    # Capture each signature value across possible newlines (numpy wraps 2D
    # arrays one row per line) up to the next marker / OK sentinel.
    parts = re.findall(
        rf"{esc} (\w+) = (.*?)(?=\n{esc} |\n{re.escape(_OK_MARK)}|\Z)",
        out, re.DOTALL,
    )
    if not parts:
        return None
    return "##".join(f"{n}={v.strip()}" for n, v in parts)


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input or ""
        library = state.metadata.get("library", "?")
        base = _base_indent(prompt)
        skel = _skeleton_block(prompt)
        idiom = _has_idiom_constraint(prompt)
        fname, fvars = _invent_function(prompt, skel)
        targets = _target_names(prompt)
        print(f"[{state.sample_id}] library={library} base={base} "
              f"idiom={idiom} func={fname} targets={targets}")

        target_line = _detect_targets(prompt)
        func_line = _FUNC_GUIDANCE if fname else ""
        sys_msg = SYSTEM_GUIDANCE.format(target_line=target_line,
                                         func_line=func_line)
        gen_prompt = f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}"

        async def _gen(p, model=MODEL, cfg=GEN_CONFIG):
            try:
                r = await model.generate(p, config=cfg)
                return _extract_code(r.completion or "")
            except Exception as e:  # noqa: BLE001
                print(f"  generate failed: {e!r}")
                return ""

        py = None
        if base == 0 and _runnable(skel):
            try:
                py = next(
                    (t for t in state.tools
                     if ToolDef(t).name == "python_session"),
                    None,
                )
            except Exception:  # noqa: BLE001
                py = None

        use_consensus = (
            py is not None and library != "Matplotlib"
        )

        async def _run_sig(code):
            """Execute skeleton + code in the sandbox; return (sig, out).
            sig is None if it crashed / target missing."""
            sol = _reindent(code, base)
            parts = [skel, sol]
            if fname and fvars:
                arg = fvars[0]
                parts.append(
                    f"\ntry:\n    {fname}({arg})\n"
                    f"except TypeError:\n    raise\n"
                    f"except Exception:\n    pass"
                )
            parts.append(_sig_harness(targets))
            program = "\n".join(parts)
            try:
                out = await py(code=program)
                out = out if isinstance(out, str) else str(out)
            except Exception as e:  # noqa: BLE001
                print(f"  sandbox call error: {e!r}")
                return None, ""
            return _parse_sig(out), out

        async def _idiom_fix(code):
            """Loop-free regeneration if an idiom constraint is present."""
            if not (idiom and code):
                return code
            for i in range(MAX_IDIOM_REGEN):
                if not _has_loop(code):
                    break
                print(f"  idiom: loop detected, regen {i + 1}")
                regen = (
                    f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}\n\n"
                    f"Your previous answer:\n<code>\n{code}\n</code>\n\n"
                    "It contains a `for`/`while` (a loop OR a "
                    "comprehension/generator expression). This problem has an "
                    "idiom/style constraint: the grader TOKENIZES your code "
                    "and rejects ANY `for` or `while` token — comprehensions "
                    "and generator expressions count. Rewrite the SAME "
                    "correct result using pure vectorized library operations. "
                    "No `for`, no `while`, no comprehensions. Respond with "
                    "ONLY the corrected <code>...</code> block."
                )
                new = await _gen(regen)
                if new:
                    code = new
                else:
                    break
            return code

        best = ""

        # ================= Cross-model execution consensus =================
        if use_consensus:
            candidates = []  # list of (label, code, sig, out)
            for label, model, cfg in PANEL:
                code = await _gen(gen_prompt, model, cfg)
                if not code:
                    print(f"  panel[{label}] empty")
                    continue
                code = await _idiom_fix(code)
                sig, out = await _run_sig(code)
                ok = sig is not None
                if ok and idiom and _has_loop(code):
                    ok = False  # idiom-violating answers can't win the vote
                print(f"  panel[{label}] ran={'OK' if ok else 'BAD'}")
                candidates.append((label, code, sig if ok else None, out))

            clean = [c for c in candidates if c[2] is not None]

            if clean:
                # Vote: group clean candidates by canonical signature.
                groups = {}
                for label, code, sig, out in clean:
                    groups.setdefault(sig, []).append((label, code))
                prio = {lbl: i for i, (lbl, _, _) in enumerate(PANEL)}

                def _group_key(item):
                    sig, members = item
                    best_prio = min(prio.get(m[0], 99) for m in members)
                    return (-len(members), best_prio)

                top_sig, top_members = sorted(
                    groups.items(), key=_group_key
                )[0]
                top_members.sort(key=lambda m: prio.get(m[0], 99))
                best = _reindent(top_members[0][1], base)
                agree = len(top_members)
                print(f"  consensus: {agree}/{len(clean)} agree "
                      f"(picked {top_members[0][0]})")

                # Tie / no-majority -> Opus tie-breaker = 4th vote.
                if agree < 2 and len(groups) > 1:
                    print("  no majority; Opus tie-breaker")
                    opus_code = await _gen(gen_prompt, STRONG_MODEL,
                                           STRONG_CONFIG)
                    if opus_code:
                        opus_code = await _idiom_fix(opus_code)
                        osig, _ = await _run_sig(opus_code)
                        if osig is not None and not (
                            idiom and _has_loop(opus_code)
                        ):
                            if osig in groups:
                                # Opus confirms an existing candidate.
                                conf = sorted(
                                    groups[osig],
                                    key=lambda m: prio.get(m[0], 99),
                                )[0]
                                best = _reindent(conf[1], base)
                                print(f"  Opus confirms {conf[0]}")
                            else:
                                best = _reindent(opus_code, base)
                                print("  Opus distinct; trusting Opus")
            else:
                # All panel candidates crashed -> iter4 repair + escalate.
                print("  panel all crashed; iter4 repair fallback")
                seed = candidates[0][1] if candidates else \
                    await _gen(gen_prompt)
                best = await _repair_escalate(
                    seed, py, skel, base, prompt, sys_msg, fname, fvars,
                    targets, _gen, _idiom_fix
                )

        # ================= iter4 single-model path =========================
        # Matplotlib, non-runnable skeletons, or sandbox unavailable.
        if not best:
            candidate = await _gen(gen_prompt)
            if not candidate:
                try:
                    r = await MODEL.generate(gen_prompt)
                    candidate = _extract_code(r.completion or "")
                except Exception as e:  # noqa: BLE001
                    print(f"  fallback generate failed: {e!r}")
            candidate = await _idiom_fix(candidate)
            best = _reindent(candidate, base) if candidate else ""

            if candidate and py is not None:
                best = await _repair_escalate(
                    candidate, py, skel, base, prompt, sys_msg, fname,
                    fvars, targets, _gen, _idiom_fix
                ) or best

        # ---- Final idiom guard on whatever we settled on ----
        if idiom and best and _has_loop(best):
            print("  idiom: final guard regen")
            regen = (
                f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}\n\n"
                f"Final answer must contain NO `for`/`while` token (no loops, "
                f"no comprehensions, no generator expressions). Previous:\n"
                f"<code>\n{textwrap.dedent(_clean(best))}\n</code>\n\n"
                "Rewrite vectorized. Respond with ONLY the <code>...</code> "
                "block."
            )
            new = await _gen(regen)
            if new and not _has_loop(new):
                best = _reindent(new, base)

        final = best or ""
        if not final:
            try:
                r = await MODEL.generate(gen_prompt)
                final = _reindent(_extract_code(r.completion or ""), base)
            except Exception:  # noqa: BLE001
                final = ""
        state.output.completion = f"<code>\n{final}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve


# ---- iter4 traceback-repair + Opus escalation (verbatim behavior) ---------

async def _repair_escalate(candidate, py, skel, base, prompt, sys_msg,
                           fname, fvars, targets, _gen, _idiom_fix):
    """Run iter4's traceback-driven repair loop, then escalate to the strong
    model on hard runnable problems. Returns the best reindented solution."""
    cur = candidate
    best = _reindent(cur, base)
    actionable_left = False
    out = ""
    for attempt in range(MAX_REPAIRS + 1):
        solution = _reindent(cur, base)
        parts = [skel, solution]
        if fname and fvars:
            arg = fvars[0]
            parts.append(
                f"\ntry:\n    {fname}({arg})\n"
                f"except TypeError:\n    raise\n"
                f"except Exception:\n    pass"
            )
        parts.append("print('___DS1000_RAN_OK___')")
        program = "\n".join(parts)
        try:
            out = await py(code=program)
            out = out if isinstance(out, str) else str(out)
        except Exception as e:  # noqa: BLE001
            print(f"  sandbox call error: {e!r}")
            break

        if "___DS1000_RAN_OK___" in out and not _has_error(out):
            best = solution
            print(f"  sandbox OK on attempt {attempt}")
            actionable_left = False
            break
        if _missing_skeleton_name(out, skel):
            print("  needs hidden data; keeping candidate")
            break
        if not _actionable_error(out) and _env_noise(out):
            print("  pure env noise; keeping candidate")
            break

        actionable_left = _actionable_error(out)
        if attempt == MAX_REPAIRS:
            print("  repairs exhausted")
            break

        err = out[-1800:]
        repair_prompt = (
            f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}\n\n"
            f"Your previous solution (shown at column 0) was:\n"
            f"<code>\n{cur}\n</code>\n\n"
            f"Running skeleton + your code (and calling any function the "
            f"grader would) produced this error:\n```\n{err}\n```\n\n"
            "Fix the bug. If it is a missing-argument TypeError on a function "
            "you defined, the hidden test calls it with only the principal "
            "input — use the other skeleton variables as globals and shrink "
            "the signature. Respond with ONLY the corrected <code>...</code> "
            "block: just the new solution code at column 0, no skeleton, no "
            "driver calls."
        )
        fixed = await _gen(repair_prompt)
        if fixed:
            cur = await _idiom_fix(fixed)
            best = _reindent(cur, base)
            print(f"  repaired (attempt {attempt + 1})")
        else:
            break

    if actionable_left:
        print("  escalating to strong model")
        esc_prompt = (
            f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}\n\n"
            f"A capable model's best attempt still fails:\n"
            f"<code>\n{cur}\n</code>\n\n"
            f"Last error:\n```\n{out[-1800:]}\n```\n\n"
            "Produce a correct solution. Obey literal skeleton comment hints "
            "(paths, names) and avoid deprecated APIs. Respond with ONLY the "
            "<code>...</code> block."
        )
        esc = await _gen(esc_prompt, STRONG_MODEL, STRONG_CONFIG)
        if esc:
            esc = await _idiom_fix(esc)
            sol = _reindent(esc, base)
            try:
                chk = "\n".join(
                    [skel, sol, "print('___DS1000_RAN_OK___')"]
                )
                o2 = await py(code=chk)
                o2 = o2 if isinstance(o2, str) else str(o2)
                if "___DS1000_RAN_OK___" in o2 and not _has_error(o2):
                    best = sol
                    print("  strong model OK")
                elif not _actionable_error(o2):
                    best = sol
            except Exception:  # noqa: BLE001
                best = sol
    return best
