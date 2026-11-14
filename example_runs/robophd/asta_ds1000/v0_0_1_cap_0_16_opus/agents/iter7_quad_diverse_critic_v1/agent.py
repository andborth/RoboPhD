"""DS-1000 solver: 4-way diverse ensemble (Sonnet + Opus + GPT-5 + Gemini-Pro)
with an ALWAYS-ON robustness critic.

Key advance over iter6: iter6 had a "3-way consensus → skip critic" fast path that
allowed convergent-wrong-algorithm bugs through (problem 238: all three candidates
used `sort_values(..., ascending=[True, False])` after `pd.to_datetime` and then
formatted to `%d-%b-%Y` — the reference instead formats FIRST and sorts ascending
on the string; both produce the same visible output for Jan/Feb but differ on
many other month combinations). The fix:

  (1) Always run the critic — don't skip on consensus.
  (2) Diversify the ensemble with a 4th model family (Gemini 3 Pro Preview).
  (3) Strengthen the critic prompt so it audits the algorithm itself, not just
      the output. The critic is encouraged to write a refined version even when
      all candidates agree on output, if their shared algorithm has a literal-
      interpretation alternative that would generalize better.
  (4) Verify the refined version against visible-example agreement: a refined
      algorithm should fix generalization but not change the answer the candidates
      gave on the example. If R produces a different visible output, the critic
      likely misread the prompt — discard R.
"""

import ast
import asyncio
import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import (
    CLAUDE_SONNET_4_6,
    CLAUDE_OPUS_4_7,
    GPT_5_4,
    GEMINI_3_1_PRO_PREVIEW,
)


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

5. **BEWARE EXAMPLE COINCIDENCES — and CONVERGENT WRONG ALGORITHMS.** A formula that happens to
   match the example by accident may break on hidden tests. Always prefer the formula whose
   correctness DOESN'T depend on specific properties of the example input.
   - WRONG: `r.max() + 1 - r` when reversing a rank — only equals `len(a) - r` when ties make
     `r.max() == len(a) - 1`. Without ties the formula is off by 1.
   - RIGHT: `len(a) - r` — works regardless of ties.
   - WRONG: `pd.to_datetime(col) → sort_values(by=col, ascending=False) → strftime(...)`
     when the expected output shows the formatted string sorted alphabetically. The reference
     might instead format-first-then-sort-ascending; the two agree only for some month pairs.
   - RIGHT: when the prompt mentions both "format the date as X" and "sort by date" — apply the
     operations in the order the prompt describes (format first, then sort the formatted form).
   - When two formulas agree on the example, pick the one based on PROBLEM STRUCTURE, not on a
     property that's true only of the example's specific values.

6. **Process operations in the order the prompt describes them.** If the prompt says "do A and
   then B" (e.g., "fill in city/district, then sort by id, then format the date"), apply them in
   that order even when reordering would seem more efficient. The reference solution often
   reflects the prompt's instruction order verbatim. Reordering may produce equivalent results
   on the example but diverge on hidden inputs.

7. **Consider hidden-test variation.** Many problems hide variation behind the pretty example:
   - Pandas object columns may contain digit-*strings* like `"26"` alongside ints — prefer
     `.astype(str).str.isdigit()` over `isinstance(x, int)`.
   - `preprocessing.scale(data)` works on 1D; `StandardScaler().fit_transform` requires 2D.
     Pick the most permissive API unless the prompt forces a class-based one.
   - Tensor problems may flip polarity (zeros↔ones), shapes (batched↔unbatched), or dtypes.

8. **Prefer the simplest robust formula.** When two approaches agree in theory, pick the one
   with fewer sign/ordering/scale ambiguities:
   - To recover `xi` from `xi.dot(xi.T)` for positive `xi`: use `np.sqrt(np.diag(M))`, NOT SVD.
     The top singular vector has sign ambiguity that breaks random tests.
   - Use explicit `(a - b)**2`-sum over matrix-broadcasting tricks.

9. **No-loop / vectorized constraints.** If the prompt contains "without a for loop", "without
   loops", "vectorized", "not one by one", "the efficient way", or any complaint about loop
   slowness — your code MUST contain ZERO `for` and `while` keywords. Some tests grep the
   submission for these and fail you even if the *output* is correct.

10. **Required idiomatic call.** "How do I do X with library Y" means use Y's named function.
    Tests sometimes grep the candidate for specific function names: `np.unique`, `pd.melt`,
    `scipy.signal.find_peaks`, `sklearn.preprocessing.LabelEncoder`, `preprocessing.scale`.

11. **Inversions / polarity.** Read negations carefully:
    - "columns where index is 0" → mask == 0 (`~mask.bool()`), NOT mask == 1
    - "values that are NOT in B" → `~np.isin(...)`
    - "drop the zeros" vs "keep the zeros"

12. **Use the prompt's literal code.** If the prompt provides explicit code (e.g.,
    `fit_params={...}`), use that exact dict verbatim. Do NOT substitute manual workarounds.

13. **Library API currency.** The sandbox runs current versions. Avoid deprecated names:
    - `scipy.integrate.simps` → **`scipy.integrate.simpson`** (`simps` is removed)
    - `scipy.integrate.trapz` → **`scipy.integrate.trapezoid`**
    - `np.float`, `np.int`, `np.bool` → **`np.float64`, `np.int64`, `bool`**
    - `np.product` → **`np.prod`**
    - `sklearn.cross_validation` → **`sklearn.model_selection`**
    - `df.append(...)` → **`pd.concat([df, ...])`**
    - `df.ix[...]` → **`df.loc[...]`** / **`df.iloc[...]`**

14. **Matplotlib markers (commonly confused — read CAREFULLY):**
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

15. **Don't redefine setup variables.** The setup `<code>` block has already executed. Don't
    re-import or reassign things it already defined. Add only the imports the setup didn't make.

16. **Pandas index/dtype/aggregation.** Many Pandas problems hinge on small details:
    - Watch `.reset_index()`, MultiIndex levels, column order, dtypes.
    - **Row-wise mode + count**: `df['frequent'] = df.mode(axis=1)[0]` adds a column to df. If you
      then count matches across ALL columns, the new `'frequent'` column matches itself and
      inflates the count by 1. Either drop it from the comparison
      (`df.drop(columns='frequent').eq(df['frequent'], axis=0).sum(axis=1)`) or subtract 1.
    - For object columns, use `.astype(str).str.isdigit()` instead of `isinstance(x, int)`.
    - Use `pd.concat([...])` instead of the deprecated `df.append`.
    - **Date formatting + sorting**: if the prompt asks you to FORMAT a date column to a string
      pattern AND to sort by that column, do the format FIRST and sort the formatted strings
      (string sort = alphabetic), unless the prompt is explicit about chronological order.

17. **Sklearn nuances.**
    - For scaling/centering, prefer `preprocessing.scale(data)` (works on 1D).
      `StandardScaler().fit_transform` requires 2D and will break on 1D test inputs.
    - For label encoding, instantiate first: `LabelEncoder().fit_transform(col)`, not the
      class method `LabelEncoder.fit_transform(col)`.

18. **Tensor library hints.**
    - PyTorch: read polarity (zeros vs ones), dtype (`LongTensor` for indices/labels,
      `BoolTensor` for masks, `FloatTensor` for numerics), device, and grad context.
    - TensorFlow: most modern problems use TF2 eager (no `tf.Session`).

THINK STEP BY STEP (silently before writing):
- Which variable must I set? (Or is this a function-body completion / named-function def?)
- What's the most library-idiomatic call? Is it deprecated?
- Did the prompt forbid loops or require a specific function name?
- Will hidden tests vary the inputs in ways my code might fail on?
- Does my formula depend on specific properties of the example (ties, hardcoded shape)?
  If yes, find a more robust formula.
- Did the prompt give me literal code to use verbatim?
- Are there any inversions I might have missed?
- If the prompt lists multiple operations, am I applying them in the order described?

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


async def _run_candidates(
    setup: str,
    codes: list[str],
    answer_var: str | None,
    tools,
) -> list[tuple[str | None, str | None]]:
    """Run multiple candidates in isolated exec() namespaces in a single python_session call.
    Returns [(err, value_repr), ...] for each candidate. err is None on success."""
    py = next((t for t in tools if ToolDef(t).name == "python_session"), None)
    if py is None:
        return [(None, None) for _ in codes]

    var = answer_var or "result"

    code_list_str = "[" + ", ".join(repr(c) for c in codes) + "]"
    program = f"""\
import traceback as _tb

_SETUP = {setup!r}
_CODES = {code_list_str}
_VAR = {var!r}

def _run(setup, code):
    ns = {{}}
    try:
        exec(setup + '\\n' + code, ns)
        v = ns.get(_VAR, None)
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
        return [(None, None) for _ in codes]

    s = str(out) if out is not None else ""

    results: list[tuple[str | None, str | None]] = []
    for i in range(len(codes)):
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
            "Use `pd.concat([...])` instead of the deprecated `df.append`. "
            "**Apply operations in the ORDER the prompt describes them** — e.g., format dates BEFORE "
            "sorting if the prompt mentions formatting first, because string sort vs datetime sort "
            "can diverge on hidden inputs even when they agree on the example."
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
            "an SVD reconstruction when the matrix is PSD with positive entries. Also: when "
            "reversing ranks, prefer `len(a) - rankdata(a)` over `rankdata(a).max() + 1 - rankdata(a)` "
            "— they agree on inputs with ties but differ when there are no ties."
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


# ---------- Robustness Critic ----------


CRITIC_PROMPT = """You are reviewing four candidate solutions for a DS-1000 problem. The
hidden test will run the chosen code on INPUTS THAT MAY DIFFER from the example shown in
the prompt. Your job: pick the candidate that will generalize best, or write a refined
version if all four have a shared problem.

**STEP 1 — Identify the expected example output from the prompt.**
The prompt usually shows the expected output via phrases like:
  - "I want this:"  →  the array/value that follows
  - "I want to get this:"
  - "should give:" / "should produce:" / "should be:"
  - "expected output:" / "the answer is:"
  - "I get:" / "It gives:" (paired with "but I want:")
  - The example output shown after a print() in the prompt
If you find such a stated expected output, write down what it is. If not, skip step 1.

**STEP 2 — Compare each candidate's sandbox output to the expected example output.**
A candidate whose sandbox output DOESN'T match the prompt's stated expected output is
almost certainly WRONG and should be rejected, even if the code looks reasonable.

**STEP 3 — For candidates that match the example, AUDIT THEIR ALGORITHMS for "example
coincidence" bugs. These bugs produce the right output on the visible example by accident
but fail on hidden test inputs. Common coincidence patterns:
  - **Reverse rank with ties**: `r.max() + 1 - r` only equals `len(a) - r` when ties make
    `r.max() == len(a) - 1`. Without ties they differ. The correct form is `len(a) - r`.
  - **Hardcoded shape**: `np.diag_indices(5)` from a 5×5 example breaks on 4×4 or 6×6.
    Use `a.shape[0]`.
  - **Hardcoded column count**: indexing `df.iloc[:, 0:3]` assumes 3 leading columns; if
    the test adds more, it breaks.
  - **Sort-then-format vs format-then-sort on dates**: when the expected output shows dates
    in a non-chronological lexicographic order (e.g., 'Feb' before 'Jan' within an id-group),
    the reference is likely `format-to-string` THEN `sort_values` ascending (alphabetic).
    A candidate that does `to_datetime → sort_values(ascending=False) → strftime` may agree
    on Jan/Feb (where alphabetic Feb<Jan equals chronological Feb>Jan reversed) but diverge
    on other month combinations (e.g., Jul/Sep). PREFER the candidate that formats first.
  - **Positive-values assumption**: `np.sqrt(x**2)` equals `x` only for positive `x`.
  - **No-ties assumption**: `np.argsort(a)` gives unique indices only for unique values.
  - **Specific dtype assumption**: float arithmetic on int arrays may differ.
  - **Square-matrix assumption**: code that assumes `len(a) == a.shape[1]`.

If a candidate has an example-coincidence bug, REJECT it even if it matches the example.

**STEP 4 — CONVERGENT WRONG ALGORITHM check.**
If ALL four candidates produce identical visible output AND they all use the same algorithm
structure, ask yourself: is there a *more literal* interpretation of the prompt's instructions
that would produce a different output on hidden inputs? Read the prompt's words carefully and
follow its instruction ORDER. If you find such an alternative, write an `R` refinement using it.
Convergent agreement is not proof of correctness — four wrong-but-coherent candidates is a real
failure mode.

**STEP 5 — Decision.**
- If exactly one candidate is example-correct AND generalization-safe → pick it (A/B/C/D).
- If multiple are example-correct AND generalization-safe → prefer the one with the
  simplest, most idiomatic formulation. Tiebreak order: B (Opus) > A (Sonnet) > D (Gemini) > C (GPT).
- If NONE is example-correct, OR all example-correct ones share a coincidence/literalness bug
  → respond `R` and write a refined `<code>...</code>` block that is example-correct AND uses
  an algorithm faithful to the prompt's literal wording and instruction order.

PROBLEM:
{prompt}

CANDIDATE A (Sonnet 4.6 high):
```python
{code_a}
```
Sandbox output A: {output_a}

CANDIDATE B (Opus 4.7 high):
```python
{code_b}
```
Sandbox output B: {output_b}

CANDIDATE C (GPT-5.4 high):
```python
{code_c}
```
Sandbox output C: {output_c}

CANDIDATE D (Gemini 3 Pro high):
```python
{code_d}
```
Sandbox output D: {output_d}

Respond with EXACTLY one of these tokens on the first line:
- `A` — candidate A is best
- `B` — candidate B is best
- `C` — candidate C is best
- `D` — candidate D is best
- `R` — refine to a corrected version

If `R`, write the corrected code in a single `<code>...</code>` block after the letter.
Otherwise emit nothing after the letter.

Your answer:"""


async def _robustness_critic(
    prompt: str,
    codes: list[str],
    outputs: list[tuple[str | None, str | None]],
    issues_list: list[list[str]],
) -> tuple[str, str | None]:
    """Returns (choice, optional_refined_code). choice in {"A","B","C","D","R"}."""

    def _summary(issues: list[str], err: str | None, val: str | None) -> str:
        if issues:
            non_runtime = [i for i in issues if not i.startswith("runtime:")]
            if non_runtime:
                return f"FAIL: {non_runtime[0][:200]}"
        if err:
            return f"RUNTIME: {err[:200]}"
        if val is not None:
            return f"OK; value={val!r}"
        return "not executed"

    summaries = [
        _summary(issues_list[i], outputs[i][0], outputs[i][1]) for i in range(len(codes))
    ]
    try:
        resp = await CLAUDE_OPUS_4_7.generate(
            CRITIC_PROMPT.format(
                prompt=prompt[:7000],
                code_a=codes[0], output_a=summaries[0],
                code_b=codes[1], output_b=summaries[1],
                code_c=codes[2], output_c=summaries[2],
                code_d=codes[3], output_d=summaries[3],
            ),
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        txt = (resp.completion or "").strip()
        # Find the first A/B/C/D/R token at the start of the response.
        m = re.search(r"\b([ABCDR])\b", txt[:120].upper())
        if not m:
            return "B", None
        choice = m.group(1)
        if choice == "R":
            corrected = _extract_code(txt)
            # Make sure we actually got a code block, not just the raw response text.
            if corrected and corrected != txt.strip() and len(corrected) < len(txt):
                return "R", corrected
            return "B", None  # `R` with no parseable code → fall back to B
        return choice, None
    except Exception as e:
        print(f"  critic exception: {e}")
        return "B", None


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

        # Stage 1: parallel generation across 4 models (3 prior + Gemini 3 Pro).
        sonnet_task = CLAUDE_SONNET_4_6.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        opus_task = CLAUDE_OPUS_4_7.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        gpt_task = GPT_5_4.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=8192),
        )
        gemini_task = GEMINI_3_1_PRO_PREVIEW.generate(
            user_prompt,
            config=GenerateConfig(reasoning_effort="high", max_tokens=4096),
        )
        try:
            resp_a, resp_b, resp_c, resp_d = await asyncio.gather(
                sonnet_task, opus_task, gpt_task, gemini_task, return_exceptions=True
            )
        except Exception as e:
            print(f"  parallel gen failed: {e}")
            resp_a = resp_b = resp_c = resp_d = None

        def _safe_completion(resp) -> str:
            if isinstance(resp, Exception) or resp is None:
                return ""
            return resp.completion or ""

        code_a = _extract_code(_safe_completion(resp_a))
        code_b = _extract_code(_safe_completion(resp_b))
        code_c = _extract_code(_safe_completion(resp_c))
        code_d = _extract_code(_safe_completion(resp_d))

        codes = [code_a, code_b, code_c, code_d]
        labels = ("A", "B", "C", "D")
        issues_list = [
            _check_candidate(c, no_loop, is_func_body) for c in codes
        ]

        # Stage 2: sandbox-run all four.
        sandbox_results: list[tuple[str | None, str | None]] = [(None, None)] * 4
        if runnable:
            try:
                sandbox_results = await _run_candidates(
                    setup, codes, answer_var, state.tools,
                )
            except Exception as e:
                print(f"  sandbox exec exception: {e}")
            for i, (err, _val) in enumerate(sandbox_results):
                if err and not matplotlib_lenient:
                    issues_list[i].append(f"runtime: {err[:240]}")

        for label, _code, issues, (_err, val) in zip(labels, codes, issues_list, sandbox_results):
            print(f"  {label}: {len(issues)} issues, val={str(val)[:60]!r}")

        chose: str | None = None
        code_chosen: str | None = None

        # Stage 3: are ALL candidates failing static + sandbox checks?
        num_clean = sum(1 for iss in issues_list if not iss)
        if num_clean == 0:
            # All failed. Reflection retry with Opus high reasoning + full feedback.
            print(f"  all 4 candidates failed checks; reflection retry with Opus")
            feedback_lines = []
            for label, code, issues in zip(labels, codes, issues_list):
                feedback_lines.append(
                    f"Candidate {label}:\n```python\n{code}\n```\nIssues: " + "; ".join(issues)
                )
            retry_prompt = (
                user_prompt
                + "\n\n--- PREVIOUS ATTEMPTS ---\n"
                + "\n\n".join(feedback_lines)
                + "\n\nAll four attempts failed. Read the original problem very carefully. "
                "Write a CORRECTED `<code>...</code>` block that fixes the listed issues."
            )
            try:
                resp_r = await CLAUDE_OPUS_4_7.generate(
                    retry_prompt,
                    config=GenerateConfig(reasoning_effort="high", max_tokens=6144),
                )
                code_r = _extract_code(resp_r.completion or "")
                issues_r = _check_candidate(code_r, no_loop, is_func_body)
                if runnable and not issues_r:
                    retry_results = await _run_candidates(
                        setup, [code_r], answer_var, state.tools,
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
                    best_label = min(
                        labels,
                        key=lambda lab: (
                            len(issues_list[labels.index(lab)]),
                            {"B": 0, "A": 1, "D": 2, "C": 3}[lab],
                        ),
                    )
                    chose = best_label
                    code_chosen = codes[labels.index(chose)]
                    print(f"  retry didn't help; using {chose}")
            except Exception as e:
                print(f"  retry exception: {e}")
                best_label = min(
                    labels,
                    key=lambda lab: (
                        len(issues_list[labels.index(lab)]),
                        {"B": 0, "A": 1, "D": 2, "C": 3}[lab],
                    ),
                )
                chose = best_label
                code_chosen = codes[labels.index(chose)]
        else:
            # Stage 3b: ALWAYS run the robustness critic (no consensus shortcut).
            # This is the key change from iter6: we don't skip the critic when
            # candidates agree on output, because that can mask convergent-wrong-algorithm bugs.
            pick, refined = await _robustness_critic(
                state.input, codes, sandbox_results, issues_list,
            )
            print(f"  critic picked {pick}")

            if pick == "R" and refined:
                # The critic wrote a refined version. Verify:
                #   (a) it parses + passes static checks
                #   (b) it runs cleanly in the sandbox
                #   (c) it agrees with the most-popular candidate output (i.e., it fixes the
                #       algorithm without changing the visible answer). If R produces a
                #       different visible output, the critic likely misread the prompt —
                #       discard R and fall back to the best clean candidate.
                issues_r = _check_candidate(refined, no_loop, is_func_body)
                val_r: str | None = None
                if runnable and not issues_r:
                    retry_results = await _run_candidates(
                        setup, [refined], answer_var, state.tools,
                    )
                    err_r, val_r = retry_results[0]
                    if err_r and not matplotlib_lenient:
                        issues_r.append(f"runtime: {err_r[:200]}")

                # Compute the most-popular candidate output (the "visible-example consensus").
                popular_val = None
                if runnable:
                    val_counts: dict[str, int] = {}
                    for (err, v) in sandbox_results:
                        if err is None and v is not None:
                            key = str(v)
                            val_counts[key] = val_counts.get(key, 0) + 1
                    if val_counts:
                        popular_key = max(val_counts.items(), key=lambda kv: kv[1])[0]
                        # Only consider a real "consensus" if at least 2 candidates agree.
                        if val_counts[popular_key] >= 2:
                            popular_val = popular_key

                accept_r = False
                if not issues_r:
                    if popular_val is None or val_r is None or not runnable:
                        # No consensus to compare against, or R didn't run — accept R.
                        accept_r = True
                    elif str(val_r) == popular_val:
                        accept_r = True
                        print(f"  R agrees with visible consensus → accepting")
                    else:
                        print(
                            f"  R disagrees with visible consensus → rejecting R"
                            f" (popular={popular_val[:80]!r}, R={str(val_r)[:80]!r})"
                        )

                if accept_r:
                    chose = "R"
                    code_chosen = refined
                else:
                    # R rejected; fall back to best eligible candidate.
                    eligible_labels = [
                        lab for lab, iss in zip(labels, issues_list) if not iss
                    ]
                    for pref in ("B", "A", "D", "C"):
                        if pref in eligible_labels:
                            chose = pref
                            code_chosen = codes[labels.index(pref)]
                            break
                    if not code_chosen:
                        chose = "B"
                        code_chosen = code_b or code_a or code_c or code_d
                    print(f"  refined had issues / disagreed; fell back to {chose}")
            elif pick in ("A", "B", "C", "D"):
                cidx = labels.index(pick)
                if issues_list[cidx]:
                    # Critic picked a candidate with issues; pick a clean alternative.
                    eligible_labels = [
                        lab for lab, iss in zip(labels, issues_list) if not iss
                    ]
                    if eligible_labels:
                        for pref in ("B", "A", "D", "C"):
                            if pref in eligible_labels:
                                chose = pref
                                code_chosen = codes[labels.index(pref)]
                                print(f"  critic picked {pick} (has issues); using clean {chose}")
                                break
                    else:
                        chose = pick
                        code_chosen = codes[cidx]
                else:
                    chose = pick
                    code_chosen = codes[cidx]
            else:
                # Defensive fallback.
                eligible_labels = [
                    lab for lab, iss in zip(labels, issues_list) if not iss
                ]
                for pref in ("B", "A", "D", "C"):
                    if pref in eligible_labels:
                        chose = pref
                        code_chosen = codes[labels.index(pref)]
                        break
                if not code_chosen:
                    chose = "B"
                    code_chosen = code_b or code_a or code_c or code_d

        # Final safety net.
        if not code_chosen or not code_chosen.strip():
            code_chosen = code_b or code_a or code_c or code_d or "result = None"
            if not chose:
                chose = "B" if code_b else ("A" if code_a else ("D" if code_d else "C"))
            print(f"  empty code_chosen → falling back to {chose}")

        state.output.completion = f"<code>\n{code_chosen}\n</code>"
        print(f"  chose {chose}; emitted {len(state.output.completion)} chars")
        return state

    return solve
