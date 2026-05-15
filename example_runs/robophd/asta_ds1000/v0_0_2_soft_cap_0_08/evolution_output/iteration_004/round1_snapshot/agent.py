"""DS-1000 solver — iter4_ds1000_idiom_probe.

Base = iter3_ds1000_format_aware (80%, the best prior agent): strong reasoning
model + deterministic format/indent preservation + free sandbox
verify-and-repair. iter3's only systematic blind spots were:

  * style/idiom tests (`test_string` rejects a `for`/`while` token) — the code
    runs cleanly so the traceback-only sandbox check never flags it;
  * "define a function named X" problems where the function is defined but
    never called in the solution, so an arity mismatch with the hidden test's
    call site is invisible to the sandbox.

This agent keeps iter3's pipeline verbatim and adds three guarded layers:

  1. Idiom-constraint detection -> stronger prompt + AST loop-scan +
     loop-free regeneration.
  2. Invent-signature function detection -> prompt guidance + an active
     sandbox call-probe that exercises the function like the grader will.
  3. Smarter env-noise vs actionable-error repair triage + a final
     CLAUDE_OPUS_4_7 attempt on hard runnable problems.

Every stage falls back gracefully so a wrapped <code> block is always emitted.
"""

import ast
import re
import textwrap

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6, CLAUDE_OPUS_4_7

MODEL = CLAUDE_SONNET_4_6
GEN_CONFIG = GenerateConfig(reasoning_effort="high", max_tokens=4096)
STRONG_MODEL = CLAUDE_OPUS_4_7
STRONG_CONFIG = GenerateConfig(reasoning_effort="high", max_tokens=4096)
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
  any "Desired Output" shows. 2D matrix vs single column matters. When the
  natural/idiomatic construction (e.g. `np.column_stack` of mixed string+int
  columns) upcasts dtype, match that idiomatic construction — the expected
  output may be object/string, not the naive int.
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
    if not uniq:
        return ("Assign the variable the problem asks for (commonly "
                "`result`). Use that exact name; do not also print it.")
    if len(uniq) == 1:
        return (f"Assign the exact target variable `{uniq[0]}` (use that "
                f"exact name; do not rename or print it).")
    return ("Assign each exact target variable: "
            + ", ".join(f"`{t}`" for t in uniq) + ".")


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
    """A real traceback whose deepest frame is in the executed program.

    Distinguishes a fixable bug in the user's code (frame in `<string>`,
    which is how the test execs the program) from pure environment noise
    (protobuf / missing-module spam printed during import). If both are
    present we still want to repair the actionable part.
    """
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
    generator expression — i.e. anything that yields a `for`/`while` token
    the DS-1000 string test would reject."""
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
    already supply the `def` line, return (func_name, [example_input_vars]).
    Otherwise return (None, [])."""
    if re.search(r"^\s*def\s+\w+\s*\(", skel, re.MULTILINE):
        return None, []  # skeleton gives the signature; safe case
    m = _DEF_FUNC_RE.search(prompt)
    if not m:
        return None, []
    name = next((g for g in m.groups() if g), None)
    if not name:
        return None, []
    # Principal example inputs = first skeleton scalar/array assignments whose
    # names are not obviously config (bounds / size / hyperparams).
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


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input or ""
        library = state.metadata.get("library", "?")
        base = _base_indent(prompt)
        skel = _skeleton_block(prompt)
        idiom = _has_idiom_constraint(prompt)
        fname, fvars = _invent_function(prompt, skel)
        print(f"[{state.sample_id}] library={library} base={base} "
              f"idiom={idiom} func={fname}")

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

        candidate = await _gen(gen_prompt)
        if not candidate:
            try:
                r = await MODEL.generate(gen_prompt)
                candidate = _extract_code(r.completion or "")
            except Exception as e:  # noqa: BLE001
                print(f"  fallback generate failed: {e!r}")

        # ---- Idiom-constraint enforcement (pre-sandbox) ----
        if idiom and candidate:
            for i in range(MAX_IDIOM_REGEN):
                if not _has_loop(candidate):
                    break
                print(f"  idiom: loop detected, regen {i + 1}")
                regen = (
                    f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}\n\n"
                    f"Your previous answer:\n<code>\n{candidate}\n</code>\n\n"
                    "It contains a `for`/`while` (a loop OR a "
                    "comprehension/generator expression). This problem has an "
                    "idiom/style constraint: the grader TOKENIZES your code "
                    "and rejects ANY `for` or `while` token — comprehensions "
                    "and generator expressions count. Rewrite the SAME "
                    "correct result using pure vectorized library operations "
                    "(e.g. pandas `.map`/`.apply` with a format string or "
                    "lambda, `.stack`, `.agg`, numpy broadcasting, `np.where`,"
                    " `.str` accessors). No `for`, no `while`, no "
                    "comprehensions. Respond with ONLY the corrected "
                    "<code>...</code> block."
                )
                new = await _gen(regen)
                if new:
                    candidate = new
                else:
                    break

        best = _reindent(candidate, base) if candidate else ""

        # ---- Sandbox verify-and-repair (column-0 runnable problems) ----
        if candidate and base == 0 and _runnable(skel):
            py = None
            try:
                py = next(
                    (t for t in state.tools
                     if ToolDef(t).name == "python_session"),
                    None,
                )
            except Exception:  # noqa: BLE001
                py = None

            if py is not None:
                cur = candidate
                actionable_left = False
                for attempt in range(MAX_REPAIRS + 1):
                    solution = _reindent(cur, base)
                    parts = [skel, solution]
                    # Probe: actively call an invent-signature function the
                    # way the hidden grader will, so an arity mismatch becomes
                    # a real traceback instead of staying invisible.
                    if fname and fvars:
                        arg = fvars[0]
                        parts.append(
                            f"\ntry:\n    {fname}({arg})\n"
                            f"except TypeError as _e:\n"
                            f"    raise\nexcept Exception:\n    pass"
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
                        f"Running skeleton + your code (and calling any "
                        f"function the grader would) produced this error:\n"
                        f"```\n{err}\n```\n\n"
                        "Fix the bug. If it is a missing-argument TypeError on "
                        "a function you defined, the hidden test calls it with "
                        "only the principal input — use the other skeleton "
                        "variables as globals and shrink the signature. "
                        "Respond with ONLY the corrected <code>...</code> "
                        "block: just the new solution code at column 0, no "
                        "skeleton, no driver calls."
                    )
                    fixed = await _gen(repair_prompt)
                    if fixed:
                        cur = fixed
                        best = _reindent(cur, base)
                        print(f"  repaired (attempt {attempt + 1})")
                    else:
                        break

                # ---- Strong-model escalation on hard runnable problems ----
                if actionable_left:
                    print("  escalating to strong model")
                    esc_prompt = (
                        f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}\n\n"
                        f"A capable model's best attempt still fails:\n"
                        f"<code>\n{cur}\n</code>\n\n"
                        f"Last error:\n```\n{out[-1800:]}\n```\n\n"
                        "Produce a correct solution. Obey literal skeleton "
                        "comment hints (paths, names) and avoid deprecated "
                        "APIs. Respond with ONLY the <code>...</code> block."
                    )
                    esc = await _gen(esc_prompt, STRONG_MODEL, STRONG_CONFIG)
                    if esc:
                        sol = _reindent(esc, base)
                        try:
                            chk = "\n".join(
                                [skel, sol,
                                 "print('___DS1000_RAN_OK___')"]
                            )
                            o2 = await py(code=chk)
                            o2 = o2 if isinstance(o2, str) else str(o2)
                            if ("___DS1000_RAN_OK___" in o2
                                    and not _has_error(o2)):
                                best = sol
                                print("  strong model OK")
                            elif not _actionable_error(o2):
                                best = sol
                        except Exception:  # noqa: BLE001
                            best = sol

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

        final = best or _reindent(candidate, base) or candidate or ""
        state.output.completion = f"<code>\n{final}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
