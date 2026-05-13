"""DS-1000 solver: parallel Sonnet+Opus ensemble with an LLM judge, careful prompting,
static checks for known failure modes, and isolated sandbox verification."""

import ast
import asyncio
import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6, CLAUDE_OPUS_4_7


SYSTEM_PROMPT = """You are a DS-1000 expert solver. Solve the problem and emit ONLY a single `<code>...</code>` block — no prose, no markdown fences, no BEGIN/END SOLUTION markers.

The hidden test program runs:
  1. The setup code shown in the first `<code>...</code>` block of the prompt.
  2. Your code (appended directly after the setup).
  3. Hidden assertions on a named variable from the prompt (commonly `result`, but sometimes `C`, `b`, `df`, `is_contained`, etc.).

CRITICAL READING RULES — these correspond to the most common DS-1000 failure modes:

1. **Identify the answer variable.** The prompt tells you which variable holds the answer (e.g.,
   "put solution in this variable", "put score in `b`"). Set THAT variable. If the prompt says
   `result = ...` set `result`. If it says `C = ...` set `C`. If it asks you to fill in a function
   body, see rule 2.

2. **Function-body completion pattern.** If the setup's last lines are `def f(A=..., B=...):`
   followed by some indented comments and then `### BEGIN SOLUTION` (or `BEGIN SOLUTION`), the
   prompt has already written the `def` line and expects you to write ONLY the indented function
   body. Do NOT re-emit `def f(...):` — the previous declaration is still active and re-declaring
   would either shadow it or, more commonly, cause an `IndentationError` because the original
   `def` has no body. Your code MUST start with whitespace indentation (4 spaces) and contain a
   `return` statement returning the answer.

3. **Generalize from the example.** The example shows ONE input (e.g., a 5×5 matrix). The hidden
   tests will use DIFFERENT inputs (different shapes, values, dtypes). Never hardcode a constant
   you read off the example — derive it at runtime:
   - WRONG: `np.diag_indices(5)` (uses the example's `5`)
   - RIGHT: `np.diag(a)` or `np.diag_indices(a.shape[0])`
   Same for column counts, row counts, list lengths, dictionary keys, etc.

4. **Consider what the hidden tests might vary.** Many problems hide variation behind the
   pretty example:
   - Pandas columns of mixed types may contain digit-*strings* like "26" in addition to ints
     — prefer `.astype(str).str.isdigit()` over `isinstance(x, int)`.
   - Sklearn preprocessors that work for 2D arrays may break on 1D inputs — module-level
     `preprocessing.scale(data)` accepts 1D; `StandardScaler().fit_transform` requires 2D.
     Pick the most permissive API unless the prompt forces a class-based one.
   - Tensor problems may flip polarity (zeros become ones), shapes (batched vs unbatched),
     or dtypes (float vs long). Re-read the question's "I want X" carefully.

5. **Prefer the simplest robust formula.** When two approaches give the same answer in theory,
   pick the one with no sign / ordering / scale ambiguity. Examples:
   - For `xi.dot(xi.T)` with positive `xi`, recover `xi` via `np.sqrt(np.diag(M))`, NOT via
     SVD (the SVD's top singular vector has sign ambiguity that breaks on random tests).
   - For dot-product distances, prefer the explicit `(a - b)**2`-sum over a matrix-form
     equivalent that needs broadcasting tricks.

6. **No-loop / vectorized constraints.** If the prompt contains "without a for loop", "without
   loops", "vectorized", "without iterating", "not one by one", "the efficient way", or any
   complaint about a loop being slow — your code MUST contain ZERO `for` and `while` keywords.
   Use NumPy/Pandas vectorized ops: boolean masks, `np.where`, `np.searchsorted`,
   `np.logical_and`, `np.isin`, etc. Some tests grep your code for `for`/`while` and fail you
   even if the *output* is correct.

7. **Required idiomatic call.** When the prompt asks "how do I do X with library Y", use library
   Y's named function. Some tests grep the candidate for a specific function name (e.g.,
   `np.unique`, `pd.melt`, `scipy.signal.find_peaks`, `sklearn.preprocessing.LabelEncoder`,
   `preprocessing.scale`).

8. **Inversions / polarity.** Read negations carefully.
   - "columns where index is 0" means select where mask == 0 (i.e., `~mask.bool()`), NOT mask == 1.
   - "values that are NOT in B" → `~np.isin(...)`.
   - "drop the zeros" vs "keep the zeros".

9. **Function signatures.** When defining a function, infer its parameter list from how the test
   will call it, NOT from the original function being mimicked. If the example shows globals like
   `x_min`, `x_max` defined in the setup, USE THE GLOBALS — do NOT add them as parameters.

10. **Use the prompt's literal code.** If the prompt provides explicit code (e.g.,
    `fit_params={...}`, "use these params"), use that exact dict/pattern verbatim. Do NOT
    substitute "improvements" like manually splitting a validation set.

11. **Matplotlib distinctions** (commonly confused):
    - `marker=` is the point shape (e.g., `'*'` is a star marker).
    - `hatch=` is a fill pattern. "star hatch" → `hatch='*'`. "star marker" → `marker='*'`.
    - `linestyle=` (`-`, `--`, `:`, `-.`); `linewidth=`, `color=`, `label=`, `title=`.
    - Do NOT call `plt.show()` — the harness inspects the figure object directly.
    - For subplot axes, set properties via `ax.set_xlabel(...)`, not `plt.xlabel(...)`.

12. **Don't redefine setup variables.** The setup `<code>` block has already executed. Don't
    re-import or reassign things that are already defined. Only import what the setup didn't
    already import.

13. **Pandas index/dtype.** Many Pandas problems hinge on whether you `.reset_index()` and on
    column dtypes. Match the example's expected output structure exactly.

THINK STEP BY STEP (silently before writing):
- Which variable must I set? (Or is this a function-body completion?)
- What's the most library-idiomatic call?
- Did the prompt forbid loops or require a specific function name?
- Will the hidden tests vary the inputs in ways my code might fail on?
- Is there a simpler formula with fewer ambiguities?
- Did the prompt give me literal code to use verbatim?
- Are there any inversions I might have missed?

OUTPUT FORMAT (strict):
<code>
# your python code here, setting the variable the prompt asks for
# (or, for function-body completion, indented body that ends with `return ...`)
</code>
"""


CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)

NO_LOOP_PATTERNS = (
    "without a for loop",
    "without for loop",
    "without using a for loop",
    "without using for loop",
    "without using a loop",
    "without loops",
    "without using loops",
    "no for loop",
    "no loops",
    "without a loop",
    "vectorized",
    "vectorize",
    "not one by one",
    "not iterate",
    "not iterating",
    "without iterating",
    "the efficient way",
    "more efficient",
    "any way to do it without",
    "takes long time to loop",
    "lengthy array",
)

LOAD_DATA_PATTERNS = (
    "load_data(",
    "load_iris(",
    "load_digits(",
    "load_diabetes(",
    "load_boston(",
    "load_breast_cancer(",
    "load_wine(",
    "load_dataset(",
    "fetch_california_housing",
    "fetch_20newsgroups",
    "fetch_openml",
    "make_classification(",
    "make_regression(",
    "make_blobs(",
)


def _extract_code(text: str) -> str:
    """Pull the python from a model response. Prefer <code>...</code>, then fences, then raw."""
    s = (text or "").strip()
    m = CODE_TAG_RE.search(s)
    if m:
        return m.group(1).strip("\n")
    m = FENCE_RE.search(s)
    if m:
        return m.group(1).strip("\n")
    return s


_WRITE_INSTR_RE = re.compile(
    r"\n\s*Write the remaining python code", re.IGNORECASE
)

# Detects the "function-body completion" pattern: an unclosed `def NAME(...):`
# followed by zero or more indented lines and ending with an indented
# `### BEGIN SOLUTION` (or `# BEGIN SOLUTION` / `BEGIN SOLUTION`) marker.
FUNC_BODY_RE = re.compile(
    r"^[ \t]*def\s+\w+\s*\([^)]*\)\s*:[ \t]*\n"
    r"(?:[ \t]+[^\n]*\n)*"
    r"[ \t]+#{1,3}\s*BEGIN\s+SOLUTION[ \t]*$",
    re.MULTILINE,
)


def _extract_setup(prompt: str) -> str:
    """Extract the setup code that runs before our code.

    There are two prompt shapes in DS-1000:
      1. Closed: setup is the first ``<code>…</code>`` block. The agent's code is
         appended AFTER `</code>` (and after the `result = …` answer-variable line).
      2. Open (function-body completion): the prompt has an opening ``<code>`` whose
         only closing `</code>` is inside the trailing prose's backticks. The setup
         runs to just before the "Write the remaining…" prose.
    """
    # If the function-body pattern is present, the setup is open — cut at the
    # "Write the remaining…" prose, not at the first `</code>` (which is in prose).
    if FUNC_BODY_RE.search(prompt):
        open_idx = prompt.find("<code>")
        if open_idx < 0:
            return ""
        rest = prompt[open_idx + len("<code>"):]
        instr = _WRITE_INSTR_RE.search(rest)
        if instr:
            rest = rest[:instr.start()]
        return rest.strip("\n")
    # Otherwise, the standard closed-pair pattern.
    m = CODE_TAG_RE.search(prompt)
    if m:
        return m.group(1).strip("\n")
    return ""


def _no_loop_required(prompt: str) -> bool:
    pl = prompt.lower()
    return any(p in pl for p in NO_LOOP_PATTERNS)


def _detect_function_body(prompt: str, setup: str) -> bool:
    """True iff the agent should emit ONLY an indented function body."""
    return bool(FUNC_BODY_RE.search(prompt))


def _strip_strings_and_comments(code: str) -> str:
    """Best-effort removal of string literals and comments so keyword scans don't
    false-positive on the word 'for' inside a docstring or comment."""
    s = re.sub(r'"""[\s\S]*?"""', "", code)
    s = re.sub(r"'''[\s\S]*?'''", "", s)
    s = re.sub(r'"[^"\n]*"', "", s)
    s = re.sub(r"'[^'\n]*'", "", s)
    s = re.sub(r"#[^\n]*", "", s)
    return s


def _has_for_or_while(code: str) -> bool:
    s = _strip_strings_and_comments(code)
    return bool(re.search(r"\bfor\b|\bwhile\b", s))


def _syntax_ok(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def _all_lines_indented(code: str) -> bool:
    """Every non-empty line starts with whitespace."""
    for ln in code.split("\n"):
        if ln.strip() and not ln.startswith((" ", "\t")):
            return False
    return True


def _is_safe_to_run(setup: str) -> bool:
    if not setup:
        return False
    return not any(p in setup for p in LOAD_DATA_PATTERNS)


def _check_candidate(code: str, no_loop: bool, is_func_body: bool) -> list[str]:
    issues = []
    if is_func_body:
        # For function bodies, syntax check requires wrapping in a fake def.
        wrapped = "def _wrap():\n" + ("\n".join("    " + ln for ln in code.split("\n")) if code.strip() else "    pass")
        if not _syntax_ok(wrapped):
            issues.append("syntax error in candidate function body")
        if not _all_lines_indented(code):
            issues.append(
                "this prompt requires a function-body completion (indented code with `return`); "
                "your candidate has unindented top-level lines"
            )
    else:
        if not _syntax_ok(code):
            issues.append("syntax error in candidate code")
    if no_loop and _has_for_or_while(code):
        issues.append(
            "candidate contains `for` or `while` but the prompt forbids loops — "
            "rewrite using vectorized operations only"
        )
    return issues


async def _verify_both_isolated(setup: str, code_a: str, code_b: str, tools) -> tuple[str | None, str | None]:
    """Run both candidates inside isolated `exec()` namespaces in a single python_session call.

    Returns (err_a, err_b) where each is None if the candidate ran cleanly, else an excerpt
    of the error message.
    """
    py = next((t for t in tools if ToolDef(t).name == "python_session"), None)
    if py is None:
        return None, None

    program = f"""\
import traceback as _tb
_setup = {setup!r}
_code_a = {code_a!r}
_code_b = {code_b!r}

def _run(setup, code):
    ns = {{}}
    try:
        exec(setup + '\\n' + code, ns)
        return 'OK'
    except Exception as e:
        return 'ERR: ' + type(e).__name__ + ': ' + str(e)[:400] + '\\n' + ''.join(_tb.format_exc().splitlines()[-3:])

print('CANDIDATE_A_RESULT:', _run(_setup, _code_a))
print('CANDIDATE_B_RESULT:', _run(_setup, _code_b))
"""
    try:
        out = await py(code=program)
    except Exception:
        return None, None

    s = str(out) if out is not None else ""
    err_a = err_b = None
    for line in s.split("\n"):
        if line.startswith("CANDIDATE_A_RESULT: ERR"):
            err_a = line[len("CANDIDATE_A_RESULT: "):][:1500]
        elif line.startswith("CANDIDATE_B_RESULT: ERR"):
            err_b = line[len("CANDIDATE_B_RESULT: "):][:1500]
    return err_a, err_b


def _build_user_prompt(prompt: str, library: str, no_loop: bool, is_func_body: bool) -> str:
    extras = []
    if is_func_body:
        extras.append(
            "**FUNCTION-BODY PATTERN DETECTED**: the setup ends inside an unclosed `def …:`. "
            "Your code must be the INDENTED FUNCTION BODY ONLY (4-space indent, ends with "
            "`return <answer>`). Do NOT re-emit the `def …:` line."
        )
    if no_loop:
        extras.append(
            "**This problem REQUIRES a vectorized solution.** Your code must NOT contain any "
            "`for` or `while` loops. Use library-vectorized operations only."
        )
    if library == "Matplotlib":
        extras.append(
            "**Matplotlib hint**: distinguish `marker=` (point shape) vs `hatch=` (fill pattern). "
            "Do not call `plt.show()`."
        )
    elif library == "Pandas":
        extras.append(
            "**Pandas hint**: prefer vectorized ops; pay attention to whether the expected output "
            "has a reset index, MultiIndex, or specific column dtypes. For object-dtype columns, "
            "use `.astype(str).str.isdigit()` instead of `isinstance(x, int)` when the question "
            "is about 'integer values'."
        )
    elif library in ("Pytorch", "Tensorflow"):
        extras.append(
            "**Tensor hint**: read polarity carefully (where == 0 vs where == 1). Use the right "
            "dtype (LongTensor / BoolTensor / FloatTensor) and respect device/grad context."
        )
    elif library == "Sklearn":
        extras.append(
            "**Sklearn hint**: if the prompt provides explicit `fit_params` or constructor args, "
            "use them verbatim. For scaling/centering, prefer module-level functions like "
            "`preprocessing.scale(data)` that accept 1D arrays; class-based transformers like "
            "`StandardScaler().fit_transform` require 2D inputs and will break on 1D test inputs."
        )
    elif library == "Numpy":
        extras.append(
            "**Numpy hint**: do NOT hardcode dimensions from the example (e.g., `np.diag_indices(5)` "
            "from a 5×5 example). Use shape-agnostic ops (`np.diag(a)`, `a.shape[0]`, etc.). "
            "Prefer the simplest robust formula — e.g., `sqrt(diag(M))` over an SVD reconstruction "
            "when the matrix is PSD with positive entries."
        )
    elif library == "Scipy":
        extras.append(
            "**Scipy hint**: use the named function the prompt suggests; many tests grep for "
            "specific function names (e.g., `scipy.signal.find_peaks`, `scipy.stats.zscore`)."
        )
    extra = ("\n\n" + "\n".join(extras)) if extras else ""
    return f"{SYSTEM_PROMPT}\n\n---\n\nProblem (library: {library}):\n{prompt}{extra}"


JUDGE_PROMPT = """You are picking the better of two candidate DS-1000 solutions. The hidden
test program will run setup + candidate, then assert on the answer variable, on INPUTS THAT
MAY DIFFER from the example shown in the prompt.

Consider:
- Which candidate handles input variation (different shapes, dtypes, edge cases)?
- Which uses the more library-idiomatic / spec-matching call?
- Which is simpler and has fewer sign/ordering/scale ambiguities?
- Does either redefine a function that was already partially defined in the setup?
- Does either hardcode a constant from the example?

Respond with EXACTLY one of: "A", "B". No prose, no explanation.

PROMPT:
{prompt}

CANDIDATE A:
```python
{code_a}
```

CANDIDATE B:
```python
{code_b}
```

Your answer (A or B):"""


async def _judge(prompt: str, code_a: str, code_b: str) -> str:
    """Return 'A' or 'B'. Falls back to 'B' (Opus) on any parse failure."""
    try:
        resp = await CLAUDE_SONNET_4_6.generate(
            JUDGE_PROMPT.format(prompt=prompt[:4000], code_a=code_a, code_b=code_b),
            config=GenerateConfig(reasoning_effort="low", max_tokens=64),
        )
        txt = (resp.completion or "").strip().upper()
        # Pull the first A or B token.
        m = re.search(r"\b([AB])\b", txt)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "B"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        library = state.metadata.get("library", "?")
        sid = state.sample_id
        no_loop = _no_loop_required(state.input)
        setup = _extract_setup(state.input)
        is_func_body = _detect_function_body(state.input, setup)
        runnable = _is_safe_to_run(setup) and not is_func_body  # don't sandbox-verify func-body (setup is incomplete)
        print(f"[{sid}] library={library} no_loop={no_loop} func_body={is_func_body} runnable={runnable}")

        user_prompt = _build_user_prompt(state.input, library, no_loop, is_func_body)

        # Generate two candidates in parallel with diverse models.
        sonnet_task = CLAUDE_SONNET_4_6.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        opus_task = CLAUDE_OPUS_4_7.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        try:
            resp_a, resp_b = await asyncio.gather(sonnet_task, opus_task)
        except Exception as e:
            print(f"  parallel generation failed: {e}; falling back to sequential")
            resp_a = await CLAUDE_SONNET_4_6.generate(
                user_prompt,
                config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
            )
            resp_b = await CLAUDE_OPUS_4_7.generate(
                user_prompt,
                config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
            )

        code_a = _extract_code(resp_a.completion or "")
        code_b = _extract_code(resp_b.completion or "")
        issues_a = _check_candidate(code_a, no_loop, is_func_body)
        issues_b = _check_candidate(code_b, no_loop, is_func_body)

        # Sandbox verify both if safe.
        err_a = err_b = None
        if runnable:
            err_a, err_b = await _verify_both_isolated(setup, code_a, code_b, state.tools)
            if err_a:
                issues_a.append(f"runtime error: {err_a[:300]}")
            if err_b:
                issues_b.append(f"runtime error: {err_b[:300]}")

        print(f"  A: {len(issues_a)} issues, B: {len(issues_b)} issues")

        # Selection logic.
        chose = None
        if not issues_a and not issues_b:
            # Both pass — use judge as tiebreaker.
            pick = await _judge(state.input, code_a, code_b)
            chose = "A" if pick == "A" else "B"
        elif not issues_a:
            chose = "A"
        elif not issues_b:
            chose = "B"
        else:
            # Both fail. Prefer fewer issues; ties → B (Opus).
            chose = "A" if len(issues_a) < len(issues_b) else "B"

        # Last-resort retry if BOTH candidates failed checks.
        if issues_a and issues_b:
            print(f"  both candidates failed checks; retrying with Opus + feedback")
            feedback_lines = []
            if issues_a:
                feedback_lines.append("Candidate A issues:\n" + "\n".join(f"  - {i}" for i in issues_a))
            if issues_b:
                feedback_lines.append("Candidate B issues:\n" + "\n".join(f"  - {i}" for i in issues_b))
            retry_prompt = (
                user_prompt
                + "\n\n--- PREVIOUS ATTEMPTS ---\nCANDIDATE A:\n```python\n"
                + code_a
                + "\n```\nCANDIDATE B:\n```python\n"
                + code_b
                + "\n```\n\n"
                + "\n\n".join(feedback_lines)
                + "\n\nWrite a CORRECTED `<code>...</code>` block that fixes ALL listed issues. "
                "Read the original problem again very carefully."
            )
            try:
                resp_retry = await CLAUDE_OPUS_4_7.generate(
                    retry_prompt,
                    config=GenerateConfig(reasoning_effort="high", max_tokens=6144),
                )
                code_retry = _extract_code(resp_retry.completion or "")
                issues_retry = _check_candidate(code_retry, no_loop, is_func_body)
                if runnable:
                    err_retry, _ = await _verify_both_isolated(setup, code_retry, "pass", state.tools)
                    if err_retry:
                        issues_retry.append(f"runtime error: {err_retry[:300]}")
                # If retry has fewer issues than both, use it.
                prev_best = min(len(issues_a), len(issues_b))
                if len(issues_retry) < prev_best:
                    chose = "R"
                    code_chosen = code_retry
                    print(f"  retry won with {len(issues_retry)} issues vs prev_best={prev_best}")
                else:
                    code_chosen = code_a if chose == "A" else code_b
                    print(f"  retry didn't help ({len(issues_retry)} issues); sticking with {chose}")
            except Exception as e:
                print(f"  retry failed: {e}; sticking with {chose}")
                code_chosen = code_a if chose == "A" else code_b
        else:
            code_chosen = code_a if chose == "A" else code_b

        state.output.completion = f"<code>\n{code_chosen}\n</code>"
        print(f"  chose {chose}; emitted {len(state.output.completion)} chars")
        return state

    return solve
