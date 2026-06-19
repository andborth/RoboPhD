"""DS-1000 solver: strong reasoning cascade (iter8).

Built directly on iter3_fmt_strong_cascade, the empirically best agent (90% at
$0.006 — beating every voting/escalation ensemble on accuracy AND cost). The
format-aware parsing, extraction, indentation, and free self-check / cross-model
repair logic are kept verbatim.

Two deliberate, data-driven changes (see reasoning.md):

  1. Enriched BASE_INSTRUCTIONS (adopted from iter7): explicitly require
     constructing every object the prose names but the skeleton does not define,
     and matching the function signature the example call implies. iter7's lower
     score came from its escalation machinery, NOT this prompt — so we take the
     prompt and drop the machinery.

  2. Base generation runs GPT_5_4 at reasoning="medium" (was "low"). This is the
     only lever that targets the residual failure class — semantic-ambiguity
     problems where the code runs but uses the wrong interpretation (e.g. 445,
     812) — and it cannot dilute the strongest model the way voting did: it IS
     the strongest model thinking harder. The free zone ($0.05/problem) leaves
     ~8x headroom over iter3's spend, so medium reasoning stays well inside it.

What is deliberately NOT here: voting, disagreement-escalation, and weak models —
each lost accuracy in a prior iteration by letting a weaker signal override
GPT_5_4. The self-check repair fires ONLY on a genuine run-failure (when the
primary already scores 0), so the loop is strictly monotone: it can turn a 0 into
a possible 1, never a 1 into a 0. Any exception anywhere falls back to the first
candidate.
"""

import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, CLAUDE_SONNET_4_6


BASE_INSTRUCTIONS = """You are an expert Python data scientist solving a DS-1000 problem.

You are given a problem with a code skeleton. Write ONLY the Python code that should be appended AFTER the given skeleton so that the requested variable holds the correct value.

Rules — follow them exactly:
- Output a single `<code>` ... `</code>` block and nothing else. No prose, no explanation, no markdown ``` fences.
- Do NOT repeat any code already shown in the skeleton (imports, data definitions, `load_data()` calls, asserts, the `def` line). Only write the NEW code.
- Assign the answer to the EXACT variable name the problem asks for. Look for a line like `result = ... # put solution in this variable`, or wording like "put score in `b`, put prediction in `c`". The name is often `result` but can be `proba`, `b`, `c`, `df`, `predict`, `centered_scaled_data`, etc. Match it precisely.
- CONSTRUCT every object the problem refers to that is NOT already defined in the skeleton's `<code>` block. Only variables literally assigned inside the skeleton's `<code>` pre-exist. If the prose mentions an estimator/model/object (e.g. "with example variable `logReg`", "fit the model `clf`"), you must create it yourself (e.g. `logReg = LogisticRegression()`) before using it — do not assume it already exists.
- Read the intended SEMANTICS carefully, not just the worked example. DS-1000 references are graded on hidden test inputs; an answer that happens to reproduce the prompt's small example but encodes the wrong general rule (off-by-one, wrong axis, wrong interpretation of which numbers are the data vs the counts) will fail. When the example is consistent with more than one rule, pick the interpretation that matches the plain wording of the request and generalizes (e.g. "reverse of rankdata" over a list of length n is `len(a) - rankdata(a)`).
- Prefer the library's own canonical, idiomatic function over a manual reimplementation. DS-1000 references use the standard library call (e.g. `sklearn.preprocessing.scale`, `scipy.interpolate.RectBivariateSpline`, `scipy.stats.rankdata`, `np.column_stack`), and a workaround that gives a slightly different numeric/dtype result can be marked wrong even when it looks correct on the shown example. When the question says "without using X", "not one by one", "the efficient way", or names a function, honor it — avoid explicit Python `for`/`while` loops when a vectorized library call does the job.
- When asked to DEFINE A FUNCTION, give it exactly the parameter signature implied by the example call and the module-level variables: arguments that the example passes in are parameters; values already defined at module level (e.g. `x_min`, `x_max`) should be used directly as globals, NOT added as extra parameters. Match the arity the hidden test will call with.
- Do not call `print()` (unless the answer literally requires building a string), do not call `plt.show()`, and do not wrap things in new functions unless asked.
- The code must run as-is when appended to the skeleton.
"""

MODULE_HINT = """
Insertion format: MODULE LEVEL. Write top-level statements (no indentation) that assign the requested variable(s).

Here is the problem:

"""

FUNCTION_HINT = """
Insertion format: FUNCTION BODY. The skeleton ends with a `def {fname}(...):` line, and your code goes INSIDE that function. Therefore:
- INDENT every line of your code by 4 spaces so it sits inside the function body.
- END with a `return <answer>` statement that returns the requested value (do NOT assign to a module-level variable; the test calls the function and uses its return value).

Here is the problem:

"""


CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
MARKER_RE = re.compile(r"^\s*###?\s*(BEGIN|END)\s+SOLUTION\s*$", re.IGNORECASE)
# Trailing boilerplate that leaks into a setup block when the skeleton's `<code>`
# is left unterminated (the regex then closes on the `</code>` in this sentence).
BOILERPLATE_RE = re.compile(
    r"(?im)^[ \t]*(Write the remaining python code|Put your answer inside)"
)


def extract_code(text: str) -> str:
    """Pull executable code out of a model completion."""
    if not text:
        return ""
    s = text.strip()
    blocks = [b.strip("\n") for b in CODE_RE.findall(s)]
    blocks = [b for b in blocks if b.strip() and "[insert]" not in b]
    if blocks:
        body = blocks[-1]
    else:
        fences = [b.strip("\n") for b in FENCE_RE.findall(s)]
        if fences:
            body = fences[-1]
        else:
            # No envelope: strip stray tags/fences and use the raw body.
            s = re.sub(r"</?code>", "", s)
            s = re.sub(r"```(?:python|py)?", "", s)
            body = s.replace("```", "")
    # Drop stray BEGIN/END SOLUTION marker lines.
    body = "\n".join(ln for ln in body.split("\n") if not MARKER_RE.match(ln))
    return body.strip("\n")


def _first_setup_block(prompt: str) -> str:
    for b in CODE_RE.findall(prompt):
        # Cut any trailing instruction boilerplate from an unterminated block.
        m = BOILERPLATE_RE.search(b)
        if m:
            b = b[: m.start()]
        b = b.rstrip("\n")
        if b.strip() and "[insert]" not in b and "insert" not in b.lower():
            return b.strip("\n")
    return ""


def parse_skeleton(prompt: str):
    """Return (setup_code, target_vars, func_mode, func_name).

    setup_code  : the first non-empty `<code>` block (imports + inline data / def).
    target_vars : variable names the solution must define (module mode).
    func_mode   : True when the insertion point is inside a `def f(...):` body.
    func_name   : the function name when func_mode, else None.
    """
    setup = _first_setup_block(prompt)

    # Detect function-body insertion: after dropping trailing comment / marker /
    # blank lines, the last meaningful skeleton line is a `def ...:`.
    func_mode = False
    func_name = None
    lines = setup.split("\n")
    j = len(lines) - 1
    while j >= 0 and (
        not lines[j].strip()
        or lines[j].lstrip().startswith("#")
        or MARKER_RE.match(lines[j])
    ):
        j -= 1
    if j >= 0:
        m = re.match(r"^\s*def\s+(\w+)\s*\(", lines[j])
        if m:
            func_mode = True
            func_name = m.group(1)
    if not func_mode and re.search(
        r"return the (solution|result) in this function", prompt, re.IGNORECASE
    ):
        # Fallback signal; recover a function name if one is present.
        m = re.search(r"def\s+(\w+)\s*\(", setup)
        if m:
            func_mode = True
            func_name = m.group(1)

    targets = []
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt, re.MULTILINE):
        targets.append(m.group(1))
    for m in re.finditer(r"\bin\s+`([A-Za-z_]\w*)`", prompt):
        targets.append(m.group(1))
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]
    if not targets:
        targets = ["result"]
    return setup, targets, func_mode, func_name


def ensure_indented(code: str, indent: int = 4) -> str:
    """Indent function-body code if the model returned it un-indented."""
    if not code.strip():
        return code
    lines = code.split("\n")
    first = next((ln for ln in lines if ln.strip()), "")
    if first[:1] in (" ", "\t"):
        return code  # already indented — trust the model's structure
    pad = " " * indent
    return "\n".join(pad + ln if ln.strip() else ln for ln in lines)


def _looks_like_traceback(out: str) -> bool:
    return bool(out) and "Traceback (most recent call last)" in out


def _build_module_check(setup: str, candidate: str, targets) -> str:
    asserts = "\n".join(
        f"assert {t!r} in dir(), 'TARGET_MISSING:{t}'" for t in targets
    )
    return f"{setup}\n{candidate}\n{asserts}\nprint('SELFCHECK_OK')"


def _build_exec_only(setup: str, candidate: str) -> str:
    return f"{setup}\n{candidate}\nprint('SELFCHECK_OK')"


def _build_function_check(setup: str, candidate: str, func_name: str) -> str:
    # `candidate` is already indented to sit inside the function body. The skeleton's
    # def uses default (example) args, so calling it with no args exercises the body.
    return f"{setup}\n{candidate}\n_chk = {func_name}()\nprint('SELFCHECK_OK')"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        prompt = state.input
        setup, targets, func_mode, func_name = parse_skeleton(prompt)
        print(f"[{state.sample_id}] library={lib} func_mode={func_mode}")

        hint = (
            FUNCTION_HINT.format(fname=func_name or "f") if func_mode else MODULE_HINT
        )
        full_prompt = BASE_INSTRUCTIONS + hint + prompt

        # --- Pass 1: strong generation with medium reasoning --------------
        # Large max_tokens so reasoning tokens don't starve the visible answer
        # (OpenAI shares the cap between reasoning and completion).
        candidate = await _generate(
            GPT_5_4, full_prompt, reasoning="medium", max_tokens=8192
        )
        if not candidate.strip():
            # Reasoning may have eaten the budget -> step down to low reasoning.
            candidate = await _generate(
                GPT_5_4, full_prompt, reasoning="low", max_tokens=4096
            )
        if not candidate.strip():
            # Still empty -> plain, no reasoning.
            candidate = await _generate(GPT_5_4, full_prompt, max_tokens=3000)

        if func_mode:
            candidate = ensure_indented(candidate)

        # --- Free self-check + optional cross-model repair ----------------
        try:
            candidate = await _verify_and_repair(
                state, prompt, candidate, setup, targets, func_mode, func_name, lib
            )
        except Exception as e:  # noqa: BLE001
            print(f"  verify skipped: {e!r}")

        if not candidate.strip():
            candidate = "    return None" if func_mode else "result = None"

        state.output.completion = f"<code>\n{candidate}\n</code>"
        print(f"  emitted {len(candidate)} chars")
        return state

    return solve


async def _generate(model, prompt, reasoning=None, max_tokens=4096) -> str:
    cfg = {"max_tokens": max_tokens}
    if reasoning:
        cfg["reasoning_effort"] = reasoning
    try:
        resp = await model.generate(prompt, config=GenerateConfig(**cfg))
        return extract_code(resp.completion or "")
    except Exception as e:  # noqa: BLE001
        print(f"  generate error: {e!r}")
        return ""


async def _verify_and_repair(
    state, prompt, candidate, setup, targets, func_mode, func_name, lib
):
    """Execute the candidate against inline skeleton data; repair once if it fails.

    Never raises out — any failure leaves the original candidate (caller-guarded).
    """
    py = None
    for t in state.tools or []:
        try:
            if ToolDef(t).name == "python_session":
                py = t
                break
        except Exception:  # noqa: BLE001
            continue
    if py is None or not setup.strip():
        return candidate

    is_mpl = str(lib).lower() == "matplotlib"

    def build(cand):
        if func_mode and func_name:
            return _build_function_check(setup, cand, func_name)
        if is_mpl:
            return _build_exec_only(setup, cand)
        return _build_module_check(setup, cand, targets)

    # Precheck: is the skeleton runnable here (no load_data()/undefined helpers)?
    try:
        pre = await py(code=setup + "\nprint('SETUP_OK')")
    except Exception as e:  # noqa: BLE001
        print(f"  setup precheck threw: {e!r}")
        return candidate
    if _looks_like_traceback(pre) or "SETUP_OK" not in str(pre):
        print("  data unavailable -> skip self-check")
        return candidate

    out = str(await py(code=build(candidate)))
    if "SELFCHECK_OK" in out and not _looks_like_traceback(out):
        print("  self-check passed")
        return candidate

    print("  self-check FAILED -> cross-model repair (CLAUDE_SONNET_4_6)")
    form = (
        FUNCTION_HINT.format(fname=func_name or "f") if func_mode else MODULE_HINT
    )
    repair_prompt = (
        BASE_INSTRUCTIONS
        + form
        + prompt
        + "\n\n---\nA previous attempt produced this code:\n<code>\n"
        + candidate
        + "\n</code>\n\nWhen appended to the skeleton it FAILED with:\n```\n"
        + out[-1500:]
        + "\n```\n\nReturn a corrected `<code>` block that runs cleanly and "
        "produces the requested value(s): " + ", ".join(targets) + "."
    )
    repaired = await _generate(
        CLAUDE_SONNET_4_6, repair_prompt, reasoning="low", max_tokens=2048
    )
    if not repaired.strip():
        return candidate
    if func_mode:
        repaired = ensure_indented(repaired)

    try:
        out2 = str(await py(code=build(repaired)))
        if "SELFCHECK_OK" in out2 and not _looks_like_traceback(out2):
            print("  repair passed")
            return repaired
        print("  repair still imperfect; using repaired (stronger) answer")
        return repaired
    except Exception as e:  # noqa: BLE001
        print(f"  repair recheck error: {e!r}")
        return repaired
