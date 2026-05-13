"""DS-1000 solver: triple-model parallel ensemble (Sonnet + Opus + GPT-5) with
sandbox-output consensus voting and an LLM judge as fallback.

Pipeline per problem:
  1. Detect prompt shape: function-body completion, named-function, no-loop, answer var.
  2. Generate THREE candidates in parallel (Sonnet 4.6 high, Opus 4.7 high, GPT_5_4 high).
  3. Static checks: syntax (function-body aware), no-loop constraint, indentation.
  4. Sandbox-execute all three in isolated namespaces with the prompt's setup.
  5. Consensus: if 2+ candidates produce identical sandbox-output reprs, the largest
     group wins (Opus > Sonnet > GPT-5 preference inside the group).
  6. Else: judge (Sonnet medium) sees all three candidates + their sandbox outputs
     and picks one, or writes a corrected version.
  7. If all three fail static checks: Opus reflection retry with full feedback.

Cost estimate: ~$0.06-$0.10/problem — well under the $0.16 free zone.
"""

import ast
import asyncio
import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6, CLAUDE_OPUS_4_7, GPT_5_4


SYSTEM_PROMPT = """You are a DS-1000 expert solver. Solve the problem and emit ONLY a single `<code>...</code>` block — no prose, no markdown fences, no BEGIN/END SOLUTION markers.

The hidden test program runs:
  1. The setup code shown in the first `<code>...</code>` block of the prompt.
  2. Your code (appended directly after the setup).
  3. Hidden assertions on a named variable from the prompt (commonly `result`, but sometimes `C`, `b`, `df`, `is_contained`, etc.).

CRITICAL READING RULES — these are the most common DS-1000 failure modes:

1. **Identify the answer variable.** The prompt tells you which variable holds the answer (e.g.,
   "put solution in this variable", "put score in `b`"). Set THAT variable. If the prompt says
   `result = ...` set `result`. If it says `C = ...` set `C`. If it asks you to fill in a function
   body, see rule 2.

2. **Function-body completion pattern.** If the setup's last lines are `def f(A=..., B=...):`
   followed by indented comments and `### BEGIN SOLUTION` (or `BEGIN SOLUTION`), the prompt has
   already written the `def` line and expects you to write ONLY the indented function body. Do
   NOT re-emit `def f(...):` — the previous declaration is still active. Your code MUST start
   with 4-space indentation and contain a `return` statement.

3. **Named-function definition.** If the prompt says "define function named `X` as solution",
   you must define a `def X(...)` at top level. **CRITICAL**: infer the parameter list from how
   the test will call it, NOT from textbook signatures. If the setup defines globals (e.g.,
   `x_min`, `x_max`), the test will likely call `X(<the one varying arg>)` and read globals from
   the enclosing scope. Do NOT add extra parameters for things that already exist as globals.

4. **Generalize from the example.** The example shows ONE input (e.g., a 5×5 matrix). The hidden
   tests will use DIFFERENT inputs (different shapes, values, dtypes). Never hardcode a constant
   you read off the example — derive it at runtime:
   - WRONG: `np.diag_indices(5)` (hardcodes the example's `5`)
   - RIGHT: `np.diag(a)` or `np.diag_indices(a.shape[0])`
   Same for column counts, row counts, list lengths, dictionary keys.

5. **Consider hidden-test variation.** Many problems hide variation behind the pretty example:
   - Pandas object columns may contain digit-*strings* like `"26"` alongside ints — prefer
     `.astype(str).str.isdigit()` over `isinstance(x, int)`.
   - `preprocessing.scale(data)` works on 1D; `StandardScaler().fit_transform` requires 2D.
     Pick the most permissive API unless the prompt forces a class-based one.
   - Tensor problems may flip polarity (zeros↔ones), shapes (batched↔unbatched), or dtypes.

6. **Prefer the simplest robust formula.** When two approaches agree in theory, pick the one
   with fewer sign/ordering/scale ambiguities:
   - To recover `xi` from `xi.dot(xi.T)` for positive `xi`: use `np.sqrt(np.diag(M))`, NOT SVD.
     The top singular vector has sign ambiguity that breaks random tests.
   - Use explicit `(a - b)**2`-sum over matrix-broadcasting tricks.

7. **No-loop / vectorized constraints.** If the prompt contains "without a for loop", "without
   loops", "vectorized", "not one by one", "the efficient way", or any complaint about loop
   slowness — your code MUST contain ZERO `for` and `while` keywords. Some tests grep the
   submission for these and fail you even if the *output* is correct.

8. **Required idiomatic call.** "How do I do X with library Y" means use Y's named function.
   Tests sometimes grep the candidate for specific function names: `np.unique`, `pd.melt`,
   `scipy.signal.find_peaks`, `sklearn.preprocessing.LabelEncoder`, `preprocessing.scale`.

9. **Inversions / polarity.** Read negations carefully:
   - "columns where index is 0" → mask == 0 (`~mask.bool()`), NOT mask == 1
   - "values that are NOT in B" → `~np.isin(...)`
   - "drop the zeros" vs "keep the zeros"

10. **Use the prompt's literal code.** If the prompt provides explicit code (e.g.,
    `fit_params={...}`), use that exact dict verbatim. Do NOT substitute manual workarounds.

11. **Library API currency.** The sandbox runs current versions. Avoid deprecated names:
    - `scipy.integrate.simps` → **`scipy.integrate.simpson`** (`simps` is removed)
    - `scipy.integrate.trapz` → **`scipy.integrate.trapezoid`**
    - `np.float`, `np.int`, `np.bool` → **`np.float64`, `np.int64`, `bool`**
    - `np.product` → **`np.prod`**
    - `sklearn.cross_validation` → **`sklearn.model_selection`**
    - `df.append(...)` → **`pd.concat([df, ...])`**
    - `df.ix[...]` → **`df.loc[...]`** / **`df.iloc[...]`**

12. **Matplotlib markers (commonly confused — read CAREFULLY):**
    - `marker=` is the point shape; `hatch=` is the fill pattern (different concept!).
    - "diamond" → `marker='D'`; **"thin diamond"** → `marker='d'` (lowercase).
    - "star marker" → `marker='*'`; "star hatch" → `hatch='*'` (fill pattern).
    - "plus marker" → `marker='+'`; "plus filled" → `marker='P'`. "plus hatch" → `hatch='+'`.
    - "x marker" → `marker='x'`; "x filled" → `marker='X'`. "x hatch" → `hatch='x'`.
    - "pentagon" → `marker='p'`. "hexagon" → `marker='h'` or `'H'`. "octagon" → `marker='8'`.
    - "circle" → `marker='o'`; "square" → `marker='s'`; "triangle up/down/left/right" → `'^'`/`'v'`/`'<'`/`'>'`.
    - Hatch patterns are strings like `'/'`, `'\\\\'`, `'|'`, `'-'`, `'+'`, `'x'`, `'o'`, `'O'`, `'.'`, `'*'`.
    - `linestyle=` (`'-'`, `'--'`, `':'`, `'-.'`); `linewidth=`/`lw=` for thickness on a line;
      `markersize=`/`ms=` for marker size; `markeredgewidth=`/`mew=` for marker stroke width.
    - "thickness of N" on a marker usually means `markersize=N` (point size, not stroke).
    - Do NOT call `plt.show()` — the harness inspects the figure object directly.
    - For subplot axes: `ax.set_xlabel(...)`, not `plt.xlabel(...)`.

13. **Don't redefine setup variables.** The setup `<code>` block has already executed. Don't
    re-import or reassign things it already defined. Add only the imports the setup didn't make.

14. **Pandas index/dtype/aggregation.** Many Pandas problems hinge on small details:
    - Watch `.reset_index()`, MultiIndex levels, column order, dtypes.
    - **Row-wise mode + count**: `df['frequent'] = df.mode(axis=1)[0]` adds a column to df. If you
      then count matches across ALL columns, the new `'frequent'` column matches itself and
      inflates the count by 1. Either drop it from the comparison
      (`df.drop(columns='frequent').eq(df['frequent'], axis=0).sum(axis=1)`) or subtract 1.
    - For object columns, use `.astype(str).str.isdigit()` instead of `isinstance(x, int)`.
    - Use `pd.concat([...])` instead of the deprecated `df.append`.

15. **Sklearn nuances.**
    - For scaling/centering, prefer `preprocessing.scale(data)` (works on 1D).
      `StandardScaler().fit_transform` requires 2D and will break on 1D test inputs.
    - For label encoding, instantiate first: `LabelEncoder().fit_transform(col)`, not the
      class method `LabelEncoder.fit_transform(col)`.

16. **Tensor library hints.**
    - PyTorch: read polarity (zeros vs ones), dtype (`LongTensor` for indices/labels,
      `BoolTensor` for masks, `FloatTensor` for numerics), device, and grad context.
    - TensorFlow: most modern problems use TF2 eager (no `tf.Session`).

THINK STEP BY STEP (silently before writing):
- Which variable must I set? (Or is this a function-body completion / named-function def?)
- What's the most library-idiomatic call? Is it deprecated?
- Did the prompt forbid loops or require a specific function name?
- Will hidden tests vary the inputs in ways my code might fail on?
- Is there a simpler formula with fewer ambiguities?
- Did the prompt give me literal code to use verbatim?
- Are there any inversions I might have missed?
- If matplotlib: marker vs hatch? Lowercase variant (`'d'` thin diamond) vs uppercase?

OUTPUT FORMAT (strict):
<code>
# your python code here, setting the variable the prompt asks for
# (or, for function-body completion, indented body ending with `return ...`)
</code>
"""


CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)

NO_LOOP_PATTERNS = (
    "without a for loop", "without for loop", "without using a for loop",
    "without using for loop", "without using a loop", "without loops",
    "without using loops", "no for loop", "no loops", "without a loop",
    "vectorized", "vectorize", "not one by one", "not iterate", "not iterating",
    "without iterating", "the efficient way", "more efficient",
    "any way to do it without", "takes long time to loop", "lengthy array",
)

LOAD_DATA_PATTERNS = (
    "load_data(", "load_iris(", "load_digits(", "load_diabetes(",
    "load_boston(", "load_breast_cancer(", "load_wine(", "load_dataset(",
    "fetch_california_housing", "fetch_20newsgroups", "fetch_openml",
    "make_classification(", "make_regression(", "make_blobs(",
)


_WRITE_INSTR_RE = re.compile(r"\n\s*Write the remaining python code", re.IGNORECASE)

FUNC_BODY_RE = re.compile(
    r"^[ \t]*def\s+\w+\s*\([^)]*\)\s*:[ \t]*\n"
    r"(?:[ \t]+[^\n]*\n)*"
    r"[ \t]+#{1,3}\s*BEGIN\s+SOLUTION[ \t]*$",
    re.MULTILINE,
)

NAMED_FUNC_RE = re.compile(
    r"define\s+(?:a\s+)?function\s+(?:named|called)?\s*`?(\w+)`?\s+as\s+solution",
    re.IGNORECASE,
)

ANSWER_VAR_RE = re.compile(
    r"^\s*(\w+)\s*=\s*\.\.\.\s*#\s*put\s+(?:solution|the\s+solution|answer)",
    re.IGNORECASE | re.MULTILINE,
)

PUT_IN_VAR_RE = re.compile(r"put\s+\w+\s+in\s+`(\w+)`", re.IGNORECASE)


def _extract_code(text: str) -> str:
    s = (text or "").strip()
    m = CODE_TAG_RE.search(s)
    if m:
        return m.group(1).strip("\n")
    m = FENCE_RE.search(s)
    if m:
        return m.group(1).strip("\n")
    return s


def _extract_setup(prompt: str) -> str:
    if FUNC_BODY_RE.search(prompt):
        open_idx = prompt.find("<code>")
        if open_idx < 0:
            return ""
        rest = prompt[open_idx + len("<code>"):]
        instr = _WRITE_INSTR_RE.search(rest)
        if instr:
            rest = rest[:instr.start()]
        return rest.strip("\n")
    m = CODE_TAG_RE.search(prompt)
    if m:
        return m.group(1).strip("\n")
    return ""


def _no_loop_required(prompt: str) -> bool:
    pl = prompt.lower()
    return any(p in pl for p in NO_LOOP_PATTERNS)


def _detect_function_body(prompt: str) -> bool:
    return bool(FUNC_BODY_RE.search(prompt))


def _detect_named_function(prompt: str) -> str | None:
    m = NAMED_FUNC_RE.search(prompt)
    return m.group(1) if m else None


def _detect_answer_var(prompt: str) -> str | None:
    m = ANSWER_VAR_RE.search(prompt)
    if m:
        return m.group(1)
    m = PUT_IN_VAR_RE.search(prompt)
    if m:
        return m.group(1)
    return None


def _strip_strings_and_comments(code: str) -> str:
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
    for ln in code.split("\n"):
        if ln.strip() and not ln.startswith((" ", "\t")):
            return False
    return True


def _is_safe_to_run(setup: str) -> bool:
    if not setup:
        return False
    return not any(p in setup for p in LOAD_DATA_PATTERNS)


def _check_candidate(code: str, no_loop: bool, is_func_body: bool) -> list[str]:
    issues: list[str] = []
    if not code or not code.strip():
        issues.append("empty candidate code")
        return issues
    if is_func_body:
        wrapped = "def _wrap():\n" + "\n".join("    " + ln for ln in code.split("\n"))
        if not _syntax_ok(wrapped):
            issues.append("syntax error in candidate function body")
        if not _all_lines_indented(code):
            issues.append(
                "this prompt requires a function-body completion (indented code "
                "with `return`); your candidate has unindented top-level lines"
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


# ---------- Sandbox execution helpers ----------


async def _run_three_candidates(
    setup: str,
    codes: tuple[str, str, str],
    answer_var: str | None,
    tools,
) -> list[tuple[str | None, str | None]]:
    """Run three candidates in isolated exec() namespaces in a single python_session call.
    Returns [(err, value_repr), ...] for each candidate (3 elements).
    err is None on success; value_repr is the repr of the answer variable.
    """
    py = next((t for t in tools if ToolDef(t).name == "python_session"), None)
    if py is None:
        return [(None, None), (None, None), (None, None)]

    var = answer_var or "result"

    program = f"""\
import traceback as _tb

_SETUP = {setup!r}
_CODES = [{codes[0]!r}, {codes[1]!r}, {codes[2]!r}]
_VAR = {var!r}

def _run(setup, code):
    ns = {{}}
    try:
        exec(setup + '\\n' + code, ns)
        v = ns.get(_VAR, None)
        # Try to find figure objects too for matplotlib problems.
        try:
            r = repr(v)
            if len(r) > 600:
                r = r[:600] + '...'
        except Exception:
            r = '<unrepr ' + type(v).__name__ + '>'
        return 'OK', r
    except Exception as e:
        tail = ''.join(_tb.format_exc().splitlines()[-3:])
        return 'ERR', type(e).__name__ + ': ' + str(e)[:240] + ' | ' + tail[:200]

for _i, _c in enumerate(_CODES):
    _s, _r = _run(_SETUP, _c)
    print('AAA_SLOT_' + str(_i) + '_STATUS_AAA', _s)
    print('AAA_SLOT_' + str(_i) + '_VALUE_AAA', repr(_r))
"""
    try:
        out = await py(code=program)
    except Exception:
        return [(None, None), (None, None), (None, None)]

    s = str(out) if out is not None else ""

    results: list[tuple[str | None, str | None]] = []
    for i in range(3):
        status_match = re.search(rf"AAA_SLOT_{i}_STATUS_AAA (.*)", s)
        value_match = re.search(rf"AAA_SLOT_{i}_VALUE_AAA (.*)", s)
        if not status_match:
            results.append((None, None))
            continue
        status = status_match.group(1).strip()
        value = value_match.group(1).strip() if value_match else None
        unwrapped = None
        if value:
            try:
                unwrapped = ast.literal_eval(value)
            except Exception:
                unwrapped = value
        if status == "OK":
            results.append((None, unwrapped))
        else:
            results.append((unwrapped or "unknown error", None))
    return results


# ---------- Prompt construction ----------


def _build_user_prompt(
    prompt: str,
    library: str,
    no_loop: bool,
    is_func_body: bool,
    named_func: str | None,
) -> str:
    extras = []
    if is_func_body:
        extras.append(
            "**FUNCTION-BODY PATTERN DETECTED**: the setup ends inside an unclosed `def …:`. "
            "Your code must be the INDENTED FUNCTION BODY ONLY (4-space indent, ends with "
            "`return <answer>`). Do NOT re-emit the `def …:` line."
        )
    if named_func and not is_func_body:
        extras.append(
            f"**NAMED-FUNCTION DETECTED**: you must define `def {named_func}(...)` at top level. "
            "Infer the parameter list from how the test will call it — if the setup defines "
            "globals matching natural parameter names, the test will pass JUST the varying "
            "argument and read globals from the enclosing scope."
        )
    if no_loop:
        extras.append(
            "**This problem REQUIRES a vectorized solution.** Your code must NOT contain any "
            "`for` or `while` loops. Use library-vectorized operations only."
        )
    if library == "Matplotlib":
        extras.append(
            "**Matplotlib hint**: read marker vs hatch carefully. 'thin diamond' is `marker='d'` "
            "(lowercase), 'diamond' is `marker='D'`. 'star marker' is `marker='*'`, 'star hatch' "
            "is `hatch='*'`. 'thickness of N' for a marker usually means `markersize=N`. "
            "Do not call `plt.show()`. For subplot axes, use `ax.set_xlabel(...)`."
        )
    elif library == "Pandas":
        extras.append(
            "**Pandas hint**: prefer vectorized ops; watch reset_index, MultiIndex, dtypes. "
            "For object columns, use `.astype(str).str.isdigit()` for 'integer values'. "
            "When counting matches of a row-wise mode, exclude the mode column itself or subtract 1. "
            "Use `pd.concat([...])` instead of the deprecated `df.append`."
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
            "`preprocessing.scale(data)` (accepts 1D); class-based transformers need 2D. "
            "For LabelEncoder: instantiate first (`LabelEncoder().fit_transform(col)`)."
        )
    elif library == "Numpy":
        extras.append(
            "**Numpy hint**: do NOT hardcode dimensions from the example (e.g., "
            "`np.diag_indices(5)` from a 5×5 example). Use shape-agnostic ops (`np.diag(a)`, "
            "`a.shape[0]`). Prefer the simplest robust formula — e.g., `sqrt(diag(M))` over "
            "an SVD reconstruction when the matrix is PSD with positive entries."
        )
    elif library == "Scipy":
        extras.append(
            "**Scipy hint**: use the named function the prompt suggests; many tests grep for "
            "specific names (`scipy.signal.find_peaks`, `scipy.stats.zscore`). Use "
            "`scipy.integrate.simpson` (NOT the removed `simps`) and "
            "`scipy.integrate.trapezoid` (NOT `trapz`)."
        )
    extra = ("\n\n" + "\n".join(extras)) if extras else ""
    return f"{SYSTEM_PROMPT}\n\n---\n\nProblem (library: {library}):\n{prompt}{extra}"


# ---------- Judge ----------


JUDGE_PROMPT = """You are picking the best of three candidate DS-1000 solutions, or
writing a corrected version. The hidden test runs setup + candidate, then asserts on
the answer variable using INPUTS THAT MAY DIFFER from the example shown in the prompt.

All three candidates were executed in a sandbox with the example setup. Their outcomes:

CANDIDATE A (Sonnet):
```python
{code_a}
```
Sandbox result A: {result_a}

CANDIDATE B (Opus):
```python
{code_b}
```
Sandbox result B: {result_b}

CANDIDATE C (GPT-5):
```python
{code_c}
```
Sandbox result C: {result_c}

Consider:
- Which candidate handles input variation (different shapes, dtypes, edge cases)?
- Which uses the more library-idiomatic / spec-matching call?
- Which is simpler, with fewer sign/ordering/scale ambiguities?
- Does any candidate hardcode a constant from the example (e.g., `np.diag_indices(5)`)?
- Does any candidate use a wrong function signature?
- Does any candidate use a deprecated API (`scipy.integrate.simps`, `np.float`, `df.append`)?
- For matplotlib: did the candidate pick the right marker character (e.g., `'d'` for "thin
  diamond" vs `'D'` for "diamond"), the right kwarg (`marker=` vs `hatch=`), the right size kwarg?
- For pandas row-wise mode + count: does the candidate avoid double-counting the mode column?

PROBLEM:
{prompt}

Respond with EXACTLY one of these tokens on the first line:
- `A` — candidate A is correct
- `B` — candidate B is correct
- `C` — candidate C is correct
- `R` — all three candidates are wrong; you will write a corrected version

If you respond `R`, write the corrected code in a single `<code>...</code>` block
below your one-letter answer. Otherwise emit nothing after the letter.

Your answer:"""


async def _judge(
    prompt: str,
    code_a: str, code_b: str, code_c: str,
    result_a: str, result_b: str, result_c: str,
) -> tuple[str, str | None]:
    try:
        resp = await CLAUDE_SONNET_4_6.generate(
            JUDGE_PROMPT.format(
                code_a=code_a, code_b=code_b, code_c=code_c,
                result_a=result_a[:600], result_b=result_b[:600], result_c=result_c[:600],
                prompt=prompt[:6000],
            ),
            config=GenerateConfig(reasoning_effort="medium", max_tokens=2048),
        )
        txt = (resp.completion or "").strip()
        m = re.search(r"\b([ABCR])\b", txt[:80].upper())
        if not m:
            return "B", None
        choice = m.group(1)
        if choice == "R":
            corrected = _extract_code(txt)
            if corrected and corrected != txt.strip():
                return "R", corrected
            return "B", None
        return choice, None
    except Exception:
        return "B", None


# ---------- Consensus voting ----------


def _consensus_pick(
    codes: list[str],
    issues_list: list[list[str]],
    sandbox_results: list[tuple[str | None, str | None]],
    matplotlib_lenient: bool,
) -> str | None:
    """Try to pick a candidate by output consensus.

    Returns the chosen label ('A'/'B'/'C') or None if no consensus.
    Preference inside a tie: B (Opus) > A (Sonnet) > C (GPT-5).
    """
    # Build clean candidates list. Static-check issues are disqualifying.
    # For sandbox: if matplotlib_lenient, runtime errors don't disqualify.
    eligible = []  # list of (label, output_repr_or_None)
    labels = ("A", "B", "C")
    for label, code, issues, (err, val) in zip(labels, codes, issues_list, sandbox_results):
        if issues:
            continue
        if err is not None and not matplotlib_lenient:
            continue
        # If matplotlib_lenient and error, val is None; treat as eligible-but-unknown.
        eligible.append((label, val))

    if not eligible:
        return None

    # Bucket by output repr (None means "no usable output").
    buckets: dict[str | None, list[str]] = {}
    for label, val in eligible:
        buckets.setdefault(val, []).append(label)

    # Find the largest bucket with a NON-None key (real consensus); ties resolved by
    # preferring buckets that include B, then A, then C.
    def _bucket_priority(item):
        key, labs = item
        non_none = key is not None
        size = len(labs)
        has_b = "B" in labs
        has_a = "A" in labs
        return (non_none, size, has_b, has_a)

    best_key, best_labels = max(buckets.items(), key=_bucket_priority)
    # Require at least 2 candidates to call it consensus.
    if len(best_labels) >= 2 and best_key is not None:
        for pref in ("B", "A", "C"):
            if pref in best_labels:
                return pref
    return None


# ---------- Solver ----------


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        library = state.metadata.get("library", "?")
        sid = state.sample_id
        no_loop = _no_loop_required(state.input)
        setup = _extract_setup(state.input)
        is_func_body = _detect_function_body(state.input)
        named_func = None if is_func_body else _detect_named_function(state.input)
        answer_var = _detect_answer_var(state.input)
        matplotlib_lenient = library == "Matplotlib"
        runnable = _is_safe_to_run(setup) and not is_func_body
        print(
            f"[{sid}] library={library} no_loop={no_loop} func_body={is_func_body} "
            f"named_func={named_func} answer_var={answer_var} runnable={runnable}"
        )

        user_prompt = _build_user_prompt(state.input, library, no_loop, is_func_body, named_func)

        # Stage 1: parallel generation across 3 models.
        sonnet_task = CLAUDE_SONNET_4_6.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        opus_task = CLAUDE_OPUS_4_7.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        # GPT_5_4 with reasoning: max_tokens is shared with reasoning on OpenAI, so set generously.
        gpt_task = GPT_5_4.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=8192),
        )
        try:
            resp_a, resp_b, resp_c = await asyncio.gather(
                sonnet_task, opus_task, gpt_task, return_exceptions=True
            )
        except Exception as e:
            print(f"  parallel gen failed: {e}")
            resp_a = resp_b = resp_c = None

        # Normalize results — any exception becomes empty completion.
        def _safe_completion(resp) -> str:
            if isinstance(resp, Exception) or resp is None:
                return ""
            return resp.completion or ""

        code_a = _extract_code(_safe_completion(resp_a))
        code_b = _extract_code(_safe_completion(resp_b))
        code_c = _extract_code(_safe_completion(resp_c))

        codes = [code_a, code_b, code_c]
        issues_list = [
            _check_candidate(code_a, no_loop, is_func_body),
            _check_candidate(code_b, no_loop, is_func_body),
            _check_candidate(code_c, no_loop, is_func_body),
        ]

        # Stage 2: sandbox-run all three.
        sandbox_results: list[tuple[str | None, str | None]] = [(None, None)] * 3
        if runnable:
            try:
                sandbox_results = await _run_three_candidates(
                    setup, (code_a, code_b, code_c), answer_var, state.tools,
                )
            except Exception as e:
                print(f"  sandbox exec exception: {e}")
            for i, (err, val) in enumerate(sandbox_results):
                if err and not matplotlib_lenient:
                    issues_list[i].append(f"runtime: {err[:240]}")

        labels = ("A", "B", "C")
        for label, code, issues, (err, val) in zip(labels, codes, issues_list, sandbox_results):
            print(f"  {label}: {len(issues)} issues, val={str(val)[:60]!r}")

        # Stage 3: decision.
        chose: str | None = None
        code_chosen: str | None = None

        # 3a. Consensus by output.
        consensus = _consensus_pick(codes, issues_list, sandbox_results, matplotlib_lenient)
        if consensus:
            chose = consensus
            code_chosen = {"A": code_a, "B": code_b, "C": code_c}[consensus]
            print(f"  consensus → {chose}")
        else:
            # 3b. Filter out candidates with hard (static-check) issues.
            eligible_labels = [lab for lab, issues in zip(labels, issues_list) if not issues]
            if len(eligible_labels) == 0:
                # 3c. All failed static checks → reflection retry.
                print(f"  all 3 candidates failed static checks; retrying with Opus")
                feedback_lines = []
                for label, code, issues in zip(labels, codes, issues_list):
                    feedback_lines.append(
                        f"Candidate {label}:\n```python\n{code}\n```\nIssues: " + "; ".join(issues)
                    )
                retry_prompt = (
                    user_prompt
                    + "\n\n--- PREVIOUS ATTEMPTS ---\n"
                    + "\n\n".join(feedback_lines)
                    + "\n\nAll three attempts failed static checks. Read the original problem "
                    "very carefully. Write a CORRECTED `<code>...</code>` block."
                )
                try:
                    resp_r = await CLAUDE_OPUS_4_7.generate(
                        retry_prompt,
                        config=GenerateConfig(reasoning_effort="high", max_tokens=6144),
                    )
                    code_r = _extract_code(resp_r.completion or "")
                    issues_r = _check_candidate(code_r, no_loop, is_func_body)
                    if runnable and not issues_r:
                        # Re-run just this one.
                        retry_results = await _run_three_candidates(
                            setup, (code_r, "pass", "pass"), answer_var, state.tools,
                        )
                        err_r, _val_r = retry_results[0]
                        if err_r and not matplotlib_lenient:
                            issues_r.append(f"runtime: {err_r[:200]}")
                    prev_best = min(len(iss) for iss in issues_list)
                    if len(issues_r) < prev_best:
                        chose = "R"
                        code_chosen = code_r
                        print(f"  retry won ({len(issues_r)} issues)")
                    else:
                        # Fall back to candidate with fewest issues; prefer B then A then C.
                        best_label = min(
                            labels,
                            key=lambda lab: (
                                len(issues_list[labels.index(lab)]),
                                {"B": 0, "A": 1, "C": 2}[lab],
                            ),
                        )
                        chose = best_label
                        code_chosen = {"A": code_a, "B": code_b, "C": code_c}[chose]
                        print(f"  retry didn't help; using {chose}")
                except Exception as e:
                    print(f"  retry exception: {e}")
                    best_label = min(
                        labels,
                        key=lambda lab: (
                            len(issues_list[labels.index(lab)]),
                            {"B": 0, "A": 1, "C": 2}[lab],
                        ),
                    )
                    chose = best_label
                    code_chosen = {"A": code_a, "B": code_b, "C": code_c}[chose]
            elif len(eligible_labels) == 1:
                # Only one passed; use it.
                chose = eligible_labels[0]
                code_chosen = {"A": code_a, "B": code_b, "C": code_c}[chose]
                print(f"  only {chose} passed static checks")
            else:
                # Multiple eligible but sandbox outputs differ (or sandbox unavailable).
                # Send to judge.
                def _summary(label, issues, err, val):
                    if issues:
                        return f"FAIL: {issues[0]}"
                    if err:
                        return f"RUNTIME: {err[:200]}"
                    if val is not None:
                        return f"OK; repr={val!r}"
                    return "not executed"

                result_a_s = _summary("A", issues_list[0], sandbox_results[0][0], sandbox_results[0][1])
                result_b_s = _summary("B", issues_list[1], sandbox_results[1][0], sandbox_results[1][1])
                result_c_s = _summary("C", issues_list[2], sandbox_results[2][0], sandbox_results[2][1])

                pick, corrected = await _judge(
                    state.input, code_a, code_b, code_c,
                    result_a_s, result_b_s, result_c_s,
                )
                if pick == "R" and corrected:
                    issues_r = _check_candidate(corrected, no_loop, is_func_body)
                    if not issues_r:
                        chose = "R"
                        code_chosen = corrected
                    else:
                        # Fall back to best eligible (prefer B).
                        for pref in ("B", "A", "C"):
                            if pref in eligible_labels:
                                chose = pref
                                code_chosen = {"A": code_a, "B": code_b, "C": code_c}[pref]
                                break
                elif pick in eligible_labels:
                    chose = pick
                    code_chosen = {"A": code_a, "B": code_b, "C": code_c}[pick]
                else:
                    # Judge picked an ineligible one; fall back to best eligible.
                    for pref in ("B", "A", "C"):
                        if pref in eligible_labels:
                            chose = pref
                            code_chosen = {"A": code_a, "B": code_b, "C": code_c}[pref]
                            break
                print(f"  judge picked {pick}; final {chose}")

        # Final safety net.
        if not code_chosen or not code_chosen.strip():
            code_chosen = code_b or code_a or code_c or "result = None"
            if not chose:
                chose = "B" if code_b else ("A" if code_a else "C")
            print(f"  empty code_chosen → falling back to {chose}")

        state.output.completion = f"<code>\n{code_chosen}\n</code>"
        print(f"  chose {chose}; emitted {len(state.output.completion)} chars")
        return state

    return solve
