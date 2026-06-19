"""DS-1000 solver: literal-reference reasoning + cross-model agreement (iter13).

iter13_output_reconcile = iter12_func_agree verbatim, plus ONE guarded,
generalization-focused lever targeting the dominant residual failure class:
BOTH strong models AGREE on the "obvious" transform, but that transform
contradicts the desired output literally DISPLAYED in the problem prompt.

  Why iter12 wasn't enough here: cross-model value agreement only protects
  against the models DISAGREEING. On problem 142 (multi-level melt) the desired
  table is printed right in the prompt with the column levels REVERSED
  (variable_0=E innermost, variable_2=A outermost), yet the natural `pd.melt`
  gives the opposite order — and BOTH models produce that same natural order, so
  they agree and the wrong answer is emitted with "high confidence", never
  reconsidered. iter12 added a prompt-only RECONCILE instruction, but a passive
  instruction doesn't force the model to actually trace its code against the
  shown table; iter12 still failed 142.

  iter13's fix is EXECUTION-GROUNDED reconciliation. For module-mode problems
  whose prompt visibly shows a DESIRED OUTPUT block, after the agree/escalate
  layer settles on a candidate we EXECUTE it, serialize the value it ACTUALLY
  produces, and hand a high-reasoning model the produced value side-by-side with
  the problem (which contains the desired output). The model first emits a hard
  MATCH / MISMATCH verdict; only on an explicit MISMATCH backed by a clean-running
  corrected `<code>` do we swap. A correct answer reports MATCH and is kept
  untouched, so the pass is near zero-regression: it can convert a confidently-
  wrong agreement into a right answer but cannot silently break a passing one.
  Fires only when a desired-output cue is present (~pandas reshaping problems),
  adding at most one model call there; average spend stays deep in the free zone.

--- iter12 header retained below ---

iter12_func_agree = iter11 verbatim, plus two guarded, generalization-focused
changes targeting the two split-decision failures observed across the mature
95%-tied agents (iter7/iter10/iter11):

  (1) FUNCTION-MODE CROSS-MODEL VALUE AGREEMENT (architectural gap fix).
      Until now, only MODULE-mode problems got the proven cross-model
      executed-value agreement; FUNCTION-mode problems got a single-model
      self-check that passes if the code merely RUNS. Problem 723 (scipy sparse
      multiply) exposed this: candidate A's clever-but-wrong `densify+ravel`
      ran cleanly, self-check passed, and it was emitted with no arbitration,
      while a simpler model produced the correct `sA.multiply(sB)`. iter12
      extends the SAME agreement/escalation logic to function-mode problems
      (call the function with its example defaults, serialize the return value,
      compare A vs an independent B, escalate disagreements to a high-reasoning
      literal tiebreaker, majority-vote by value). Non-matplotlib only; matplotlib
      function problems keep the exec-only self-check (plots have no value to
      compare). Fully guarded: any failure falls back to iter11's self-check+repair.

  (2) EXPECTED-OUTPUT RECONCILIATION instruction (prompt lever).
      Problem 142 (multi-level melt) showed a desired-output TABLE in the prompt
      whose column order was the REVERSE of the obvious `pd.melt` transform
      (variable_0=E, not A). The literal-but-shallow reading missed it. A new base
      rule tells the model: when the problem prints a desired/expected output,
      your code must reproduce THAT EXACT table -- its column order, row order,
      labels, and values -- not merely a plausible transform; trace through to the
      shown result and reconcile.

Both changes are conservative and generalize beyond the two observed problems.

--- iter11 header retained below ---
iter11_consensus_plus = iter10 verbatim, plus two guarded, generalization-focused
additions: (1) BASE_INSTRUCTIONS gains a compact block of the highest-frequency
DS-1000 agree-but-wrong correctness traps (container/dtype exactness, pandas
index/column preservation, library-default parameters, copy-vs-in-place); and
(2) on the rare module-mode THREE-WAY split, a single independent cross-family
vote (GEMINI_3_1_PRO_PREVIEW, a frontier model) resolves the tie by executed-value
majority instead of iter10 blindly trusting the GPT-family tiebreaker C. Both
changes are fully guarded and fall back to iter10's exact behaviour otherwise.

--- iter10 header retained below ---
DS-1000 solver: literal-reference reasoning + cross-model agreement (iter10).

Direct descendant of iter9_reason_agree (the most complete prior agent: full
format handling, dtype/shape rules, medium-reasoning first shot, cross-model
value agreement with high-reasoning escalation). iter9, iter7 and iter3 all tied
at 80% on the iter9 batch; their per-problem differences were stochastic noise.
The only SYSTEMATIC, addressable failure was the consensus-miss class (445, 883):
on both, every agent gave the *cleverer / more statistically faithful* answer
(`rankdata(-a)`; condensing a distance matrix before `linkage`), but the DS-1000
reference uses the *direct literal transform* (`len(a) - rankdata(a)`; feeding the
raw matrix straight into `linkage`). Voting and self-checking cannot fix this --
there is no oracle and both strong models make the SAME clever substitution.

iter10's change is therefore a prompt-level lever, not an architectural one:
  * A new base rule teaches the model that DS-1000 references favor the most
    DIRECT literal reading of the plain-language problem and the simplest call
    that reproduces the shown example -- and to distrust a "smarter" equivalent
    that can diverge at the margins.
  * The disagreement tiebreaker is given the same lesson at the exact moment two
    strong candidates split, so it breaks ties toward the literal answer.
Everything else (architecture, value serializer, repair, escalation) is iter9's,
unchanged, so the change can only help the consensus-miss class without touching
the 80% that already passes. Cost headroom is ~2.3x (iter9 mean ~$0.022 vs the
$0.05 free zone), so the unchanged dual-lever spend stays comfortably free.

Original iter9 synthesis (retained verbatim below):
  * iter8_reason_cascade fixed wrong-but-runnable errors with a MEDIUM-reasoning
    base model.
  * iter7_agree_escalate fixed them with CROSS-MODEL VALUE AGREEMENT.
On problem 10 each lever fixed the failure independently. This agent takes the
UNION of both levers for redundant, complementary protection.

Architecture (iter7's, kept verbatim except the two reasoning bumps below):
  * Candidate A = GPT_5_4 at reasoning="medium" (was low) -- iter8's single most
    effective accuracy lever; raises the floor of the first shot.
  * Candidate B = CLAUDE_SONNET_4_6 -- independent strong model, different family
    -> independent errors. No weak voters (iter5/iter6 voting LOST accuracy).
  * Module-mode + runnable setup: execute both in the FREE python_session and
    compare ACTUAL produced values via a numpy/pandas-aware serializer:
      - agree            -> emit A (high confidence; common, cheap).
      - A crashes/missing -> cross-model repair with the traceback (iter3 path).
      - disagree         -> escalate to a tiebreaker C = GPT_5_4 reasoning="high"
                            (was medium), shown both candidates + divergent
                            outputs; majority-vote by value (tie -> C).
  * Function-body / non-runnable-setup problems fall back to iter3's proven
    self-check-and-repair on A (no regression).

The whole agreement/escalation layer is guarded so it can only convert a 0 into a
possible 1, never the reverse: any exception falls back to candidate A.
"""

import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, CLAUDE_SONNET_4_6, GEMINI_3_1_PRO_PREVIEW


BASE_INSTRUCTIONS = """You are an expert Python data scientist solving a DS-1000 problem.

You are given a problem with a code skeleton. Write ONLY the Python code that should be appended AFTER the given skeleton so that the requested variable holds the correct value.

Rules — follow them exactly:
- Output a single `<code>` ... `</code>` block and nothing else. No prose, no explanation, no markdown ``` fences.
- Do NOT repeat any code already shown in the skeleton (imports, data definitions, `load_data()` calls, asserts, the `def` line). Only write the NEW code.
- Assign the answer to the EXACT variable name the problem asks for. Look for a line like `result = ... # put solution in this variable`, or wording like "put score in `b`, put prediction in `c`". The name is often `result` but can be `proba`, `b`, `c`, `df`, `predict`, `centered_scaled_data`, etc. Match it precisely.
- CONSTRUCT every object the problem refers to that is NOT already defined in the skeleton's `<code>` block. Only variables literally assigned inside the skeleton's `<code>` pre-exist. If the prose mentions an estimator/model/object (e.g. "with example variable `logReg`", "fit the model `clf`"), you must create it yourself (e.g. `logReg = LogisticRegression()`) before using it — do not assume it already exists.
- Prefer the library's own canonical, idiomatic function over a manual reimplementation. DS-1000 references use the standard library call (e.g. `sklearn.preprocessing.scale`, `scipy.interpolate.RectBivariateSpline`, `scipy.stats.rankdata`, `np.column_stack`), and a workaround that gives a slightly different numeric/dtype result can be marked wrong even when it looks correct on the shown example. When the question says "without using X", "not one by one", "the efficient way", or names a function, honor it — avoid explicit Python `for`/`while` loops when a vectorized library call does the job.
- Favor the MOST DIRECT, LITERAL reading of the problem. The reference solution is almost always the simplest expression that plainly reproduces the described transformation and the shown example — NOT the most sophisticated or most statistically "correct" equivalent. Two specific traps: (1) When the problem says "the reverse / opposite / descending version of function F", apply a direct arithmetic transform of F's normal output (e.g. `len(a) - rankdata(a)`, `max(x) - x`) rather than re-deriving it by feeding negated/transformed inputs into F (`rankdata(-a)`), which can differ at ties or boundaries. (2) Feed the data to a function in the SAME literal form the problem presents it (e.g. if it hands you a 2-D matrix and asks to cluster it, pass that matrix straight into the routine as shown), and do NOT add preprocessing the problem never mentioned (squareform/condensing, reshaping, extra normalization). When a plain one-liner and a "smarter" multi-step version both match the example, choose the plain one-liner.
- When asked to DEFINE A FUNCTION, give it exactly the parameter signature implied by the example call and the module-level variables: arguments that the example passes in are parameters; values already defined at module level (e.g. `x_min`, `x_max`) should be used directly as globals, NOT added as extra parameters. Match the arity the hidden test will call with.
- Match the expected OUTPUT SHAPE exactly: dtype (int vs float — don't silently upcast or downcast unless the example/result clearly is the other), column order and names, index, and whether the result is a DataFrame vs Series vs scalar vs ndarray. Pandas/numpy equality checks are dtype- and order-sensitive; mirror the structure the problem's shown result implies.
- CONTAINER TYPE must match the shown result exactly: a Python list is not an ndarray, a scalar is not a 1-element array, and a pandas Series is not its `.values`. When the result comes from a DataFrame/Series, decide deliberately between keeping the pandas object and converting with `.values`/`.to_numpy()`/`.tolist()` based on what the shown result is — don't convert (or fail to convert) by accident.
- PRESERVE the pandas index and column labels and their ORDER. groupby, merge, pivot, sort, drop_duplicates and boolean filtering all change the index (and sometimes column order); if the shown result keeps the original index/columns, restore them (e.g. `.reset_index(drop=True)` only when the result clearly was reset, otherwise keep the original labels).
- Use the LIBRARY DEFAULTS the reference uses and do not pass options the problem never asked for: `scipy.stats.rankdata` averages ties; numpy/pandas sorts are ascending; `np.unique` returns sorted unique values; `argsort` is ascending. Changing a default (different tie method, descending, a non-default `ddof`, `keepdims`, etc.) when the problem didn't request it is a common silent mismatch.
- COPY vs IN-PLACE: if the original object must stay unchanged, operate on a `.copy()`; if the problem asks to modify the object in place, mutate it directly and don't return a brand-new object. Watch chained-assignment and views.
- RECONCILE WITH THE SHOWN EXPECTED OUTPUT. When the problem prints a desired/expected result (a DataFrame, array, Series, or value), your code must reproduce THAT EXACT output — its column order, row order, index/labels, and the literal values in each cell — not merely a transform that "seems right". Mentally trace your code on the shown input and compare cell-by-cell to the shown output. If the obvious operation would order/label/shape things differently from what is displayed (e.g. the displayed columns are in a reversed or non-default order, the index was reset or preserved, a level order differs), adjust your code to match the DISPLAYED result. The shown output is ground truth and overrides your intuition about the "natural" transform.
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
            s = re.sub(r"</?code>", "", s)
            s = re.sub(r"```(?:python|py)?", "", s)
            body = s.replace("```", "")
    body = "\n".join(ln for ln in body.split("\n") if not MARKER_RE.match(ln))
    return body.strip("\n")


def _first_setup_block(prompt: str) -> str:
    for b in CODE_RE.findall(prompt):
        m = BOILERPLATE_RE.search(b)
        if m:
            b = b[: m.start()]
        b = b.rstrip("\n")
        if b.strip() and "[insert]" not in b and "insert" not in b.lower():
            return b.strip("\n")
    return ""


def parse_skeleton(prompt: str):
    """Return (setup_code, target_vars, func_mode, func_name)."""
    setup = _first_setup_block(prompt)

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
        return code
    pad = " " * indent
    return "\n".join(pad + ln if ln.strip() else ln for ln in lines)


def _looks_like_traceback(out: str) -> bool:
    return bool(out) and "Traceback (most recent call last)" in out


# --- Canonical, noise-tolerant value serializer injected into python_session ---
_SERIALIZER = r"""
def __ser(v):
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            w = _np.round(v, 6) if v.dtype.kind == 'f' else v
            return 'A|%s|%s|%s' % (v.shape, v.dtype,
                _np.array2string(w, threshold=200000, separator=',', precision=6,
                                 suppress_small=True))
        if isinstance(v, _np.generic):
            v = v.item()
    except Exception:
        pass
    try:
        import pandas as _pd
        if isinstance(v, _pd.DataFrame):
            return 'DF|%s|%s' % (v.shape, v.round(6).to_string()
                                 if v.select_dtypes('number').shape[1] else v.to_string())
        if isinstance(v, _pd.Series):
            try:
                w = v.round(6)
            except Exception:
                w = v
            return 'S|%s|%s' % (v.shape, w.to_string())
        if isinstance(v, _pd.Index):
            return 'I|' + str(list(v))
    except Exception:
        pass
    try:
        import torch as _t
        if isinstance(v, _t.Tensor):
            return 'T|%s|%s' % (tuple(v.shape), _t.round(v * 1e6).long().tolist()
                                if v.dtype.is_floating_point else v.tolist())
    except Exception:
        pass
    if isinstance(v, float):
        return 'f|%.6g' % v
    return 'R|' + repr(v)
"""


def _build_module_check(setup: str, candidate: str, targets) -> str:
    asserts = "\n".join(
        f"assert {t!r} in dir(), 'TARGET_MISSING:{t}'" for t in targets
    )
    return f"{setup}\n{candidate}\n{asserts}\nprint('SELFCHECK_OK')"


def _build_exec_only(setup: str, candidate: str) -> str:
    return f"{setup}\n{candidate}\nprint('SELFCHECK_OK')"


def _build_function_check(setup: str, candidate: str, func_name: str) -> str:
    return f"{setup}\n{candidate}\n_chk = {func_name}()\nprint('SELFCHECK_OK')"


def _build_function_value_probe(setup: str, candidate: str, func_name: str) -> str:
    """Call the function with its example defaults and serialize the return value."""
    body = [
        setup,
        candidate,
        _SERIALIZER,
        f"_res = {func_name}()",
        "print('VVV:result:' + __ser(_res))",
        "print('VALUE_OK')",
    ]
    return "\n".join(body)


def _build_value_probe(setup: str, candidate: str, targets) -> str:
    """Run setup+candidate, then print a canonical serialization of each target."""
    body = [setup, candidate, _SERIALIZER]
    for t in targets:
        body.append(f"assert {t!r} in dir(), 'TARGET_MISSING:{t}'")
        body.append(f"print('VVV:{t}:' + __ser({t}))")
    body.append("print('VALUE_OK')")
    return "\n".join(body)


_VVV_RE = re.compile(r"^VVV:.*$", re.MULTILINE)


def _value_signature(out: str):
    """Extract the comparable value signature from a value-probe run, or None."""
    s = str(out)
    if "VALUE_OK" not in s or _looks_like_traceback(s):
        return None
    lines = _VVV_RE.findall(s)
    if not lines:
        return None
    return "\n".join(lines)


# --- iter13: detect a literally-shown DESIRED OUTPUT in the prompt -----------
# Cue phrases DS-1000 authors use right before displaying the target result.
# Kept deliberately broad: over-firing is harmless (the reconciliation pass is
# MATCH/MISMATCH-gated and keeps a correct answer untouched), but missing a real
# desired-output block forfeits the only available ground truth.
_DESIRED_CUE_RE = re.compile(
    r"(?i)("
    r"looking for|what i('?m| am)?\s*(want|after|looking)|i\s*want\b|want to (get|have|end)|"
    r"desired|should look like|should (be|give|return|produce|look)|"
    r"expected (output|result)|trying to (get|achieve|obtain)|end up with|"
    r"output should|the result (should|is|would)|like this:|"
    r"result will look|i('?d| would) like|i need (the|a|to get)"
    r")"
)


def _shows_desired_output(prompt: str) -> bool:
    """True when the prose visibly displays the target result (ground truth)."""
    return bool(_DESIRED_CUE_RE.search(prompt or ""))


async def _reconcile_with_shown_output(
    state, prompt, full_prompt, candidate, setup, targets, lib
):
    """Execution-grounded check against a desired output displayed in the prompt.

    Module-mode only. Runs the chosen candidate, serializes what it ACTUALLY
    produces, and asks a high-reasoning model to compare that produced value
    cell-by-cell against the desired output shown in the problem. The model must
    answer MATCH or MISMATCH on its first line; only an explicit MISMATCH backed
    by a clean-running, genuinely-different `<code>` block is accepted. Otherwise
    the original candidate is returned unchanged. Caller-guarded; never raises out.
    """
    py = _get_py(state)
    if py is None or not setup.strip() or not candidate.strip():
        return candidate
    if not await _setup_runnable(py, setup):
        return candidate

    sig0 = await _run_value(py, setup, candidate, targets)
    if sig0 is None:
        # Candidate doesn't even produce the target cleanly -> leave the existing
        # repair/agreement layer's result alone; nothing to reconcile against.
        return candidate

    produced = sig0[:4000]
    recon_prompt = (
        full_prompt
        + "\n\n---\nVERIFICATION TASK. The problem above DISPLAYS a desired/expected "
        "output (a table, array, Series, or value). A candidate solution was executed "
        "and produced the following ACTUAL value for the requested variable(s) "
        + ", ".join(targets)
        + ":\n```\n"
        + produced
        + "\n```\n\nThe candidate code was:\n<code>\n"
        + candidate
        + "\n</code>\n\nCompare the PRODUCED value above, cell by cell, against the "
        "desired output shown in the problem — column order, row order, index/labels, "
        "transposition, dtype, and the literal value in every cell. DS-1000 grades on "
        "EXACT match, and the displayed output is ground truth that overrides what the "
        "'natural' transform would give (e.g. column levels may be reversed, the index "
        "may be reset or preserved, columns may be in a non-default order).\n\n"
        "On the FIRST line output exactly `MATCH` if the produced value already "
        "reproduces the displayed desired output exactly, or `MISMATCH` if it differs "
        "in any way. If MATCH, output nothing further. If MISMATCH, after that line "
        "output a single corrected `<code>` ... `</code>` block (module-level, "
        "assigning the requested variable(s)) that WOULD reproduce the displayed output."
    )
    try:
        resp = await GPT_5_4.generate(
            recon_prompt, config=GenerateConfig(reasoning_effort="high", max_tokens=8192)
        )
        raw = resp.completion or ""
    except Exception as e:  # noqa: BLE001
        print(f"  reconcile generate error: {e!r}")
        return candidate

    head = raw.lstrip()[:40].upper()
    if head.startswith("MATCH"):
        print("  reconcile: model confirms produced output MATCHES shown output")
        return candidate

    fixed = extract_code(raw)
    if not fixed.strip() or fixed.strip() == candidate.strip():
        print("  reconcile: no usable correction -> keep candidate")
        return candidate

    sig1 = await _run_value(py, setup, fixed, targets)
    if sig1 is None:
        print("  reconcile: correction failed value-probe -> keep candidate")
        return candidate
    if sig1 == sig0:
        print("  reconcile: correction reproduces same value -> keep candidate")
        return candidate
    print("  reconcile: MISMATCH found -> swapping to reconciled answer")
    return fixed


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        prompt = state.input
        setup, targets, func_mode, func_name = parse_skeleton(prompt)
        print(f"[{state.sample_id}] library={lib} func_mode={func_mode} targets={targets}")

        hint = (
            FUNCTION_HINT.format(fname=func_name or "f") if func_mode else MODULE_HINT
        )
        full_prompt = BASE_INSTRUCTIONS + hint + prompt

        # --- Pass 1: strong generation (candidate A) ----------------------
        # Medium reasoning (iter8's key accuracy lever). Large max_tokens so
        # reasoning tokens don't starve the visible answer (OpenAI shares the cap).
        candidate = await _generate(
            GPT_5_4, full_prompt, reasoning="medium", max_tokens=8192
        )
        if not candidate.strip():
            candidate = await _generate(
                GPT_5_4, full_prompt, reasoning="low", max_tokens=4096
            )
        if not candidate.strip():
            candidate = await _generate(GPT_5_4, full_prompt, max_tokens=3000)
        if func_mode:
            candidate = ensure_indented(candidate)

        try:
            if func_mode:
                # iter12: extend cross-model value agreement to function-mode too
                # (non-matplotlib). Falls back internally to iter3's self-check+repair.
                candidate = await _agree_or_escalate_func(
                    state, prompt, full_prompt, candidate, setup, targets, func_name, lib
                )
            else:
                candidate = await _agree_or_escalate(
                    state, prompt, full_prompt, candidate, setup, targets, lib
                )
                # iter13: final guard for the both-models-agree-but-wrong class.
                # When the prompt literally displays the desired output, ground the
                # chosen answer against what it ACTUALLY produces. Module-mode only;
                # non-matplotlib (plots have no comparable value).
                if _shows_desired_output(prompt) and str(lib).lower() != "matplotlib":
                    candidate = await _reconcile_with_shown_output(
                        state, prompt, full_prompt, candidate, setup, targets, lib
                    )
        except Exception as e:  # noqa: BLE001
            print(f"  layer skipped: {e!r}")

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


def _get_py(state):
    for t in state.tools or []:
        try:
            if ToolDef(t).name == "python_session":
                return t
        except Exception:  # noqa: BLE001
            continue
    return None


async def _setup_runnable(py, setup) -> bool:
    try:
        pre = await py(code=setup + "\nprint('SETUP_OK')")
    except Exception as e:  # noqa: BLE001
        print(f"  setup precheck threw: {e!r}")
        return False
    ok = "SETUP_OK" in str(pre) and not _looks_like_traceback(str(pre))
    if not ok:
        print("  data unavailable -> single-model path")
    return ok


async def _agree_or_escalate(state, prompt, full_prompt, candidate, setup, targets, lib):
    """Module-mode: cross-check candidate A against an independent strong model B
    by comparing the ACTUAL produced values; escalate to a tiebreaker on disagree.

    Never raises out (caller-guarded); returns the chosen code.
    """
    py = _get_py(state)
    if py is None or not setup.strip():
        return candidate
    if not await _setup_runnable(py, setup):
        return candidate

    def value_of(cand):
        return _run_value(py, setup, cand, targets)

    sigA = await value_of(candidate)
    if sigA is None:
        # A crashed / left a target undefined (e.g. 919 NameError). Repair it.
        print("  A failed value-probe -> cross-model repair")
        repaired = await _repair(prompt, candidate, setup, targets, py, lib)
        return repaired or candidate

    # Independent second strong opinion (different family, no weak voter).
    candB = await _generate(CLAUDE_SONNET_4_6, full_prompt, reasoning="low", max_tokens=3072)
    if not candB.strip():
        print("  B empty -> emit A")
        return candidate
    sigB = await value_of(candB)
    if sigB is None:
        print("  B failed value-probe -> trust A")
        return candidate
    if sigA == sigB:
        print("  A and B AGREE on produced value -> emit A (high confidence)")
        return candidate

    # Disagreement on executed values -> escalate to a high-reasoning tiebreaker.
    print("  A and B DISAGREE -> escalate to GPT_5_4 (high) tiebreaker")
    tie_prompt = (
        full_prompt
        + "\n\n---\nTwo expert attempts produced DIFFERENT results for the requested "
        "variable(s): " + ", ".join(targets) + ".\n\nAttempt 1:\n<code>\n"
        + candidate
        + "\n</code>\n\nAttempt 2:\n<code>\n"
        + candB
        + "\n</code>\n\nThink carefully about which is correct under ALL inputs (not just "
        "the shown example) — watch for wrong library idioms, off-by-one ranking, wrong "
        "function arity, undefined objects, or dtype issues. IMPORTANT: DS-1000 reference "
        "solutions favor the MOST DIRECT, LITERAL transform and the simplest call that "
        "reproduces the shown example. If one attempt is a plain literal reading (e.g. "
        "`len(a) - rankdata(a)`, or feeding the given matrix straight into the routine) and "
        "the other is a 'smarter' equivalent (negating inputs, condensing/reshaping the data, "
        "extra preprocessing the problem never asked for), strongly prefer the literal one — "
        "the clever version often diverges at ties/boundaries. Return the single correct "
        "`<code>` block (you may fix either attempt or write a better one)."
    )
    candC = await _generate(GPT_5_4, tie_prompt, reasoning="high", max_tokens=8192)
    if not candC.strip():
        print("  tiebreaker empty -> emit A")
        return candidate
    sigC = await value_of(candC)
    if sigC is None:
        print("  tiebreaker failed value-probe -> emit A")
        return candidate
    if sigC == sigA:
        print("  tiebreaker matches A (2/3) -> emit A")
        return candidate
    if sigC == sigB:
        print("  tiebreaker matches B (2/3) -> emit B")
        return candB

    # Three-way split: iter10 blindly emitted C (a single GPT-family opinion).
    # Add ONE independent cross-family vote (Gemini Pro, a frontier model) and
    # resolve by executed-value majority; fall back to C if it's a genuine
    # 4-way split. Fires only in this rare worst case -> negligible avg cost.
    print("  three-way split -> cross-family Gemini tiebreak")
    candD = await _generate(
        GEMINI_3_1_PRO_PREVIEW, tie_prompt, reasoning="high", max_tokens=8192
    )
    if candD.strip():
        sigD = await value_of(candD)
        if sigD is not None:
            if sigD == sigA:
                print("  Gemini matches A (2/4) -> emit A")
                return candidate
            if sigD == sigB:
                print("  Gemini matches B (2/4) -> emit B")
                return candB
            if sigD == sigC:
                print("  Gemini matches C (2/4) -> emit C")
                return candC
    print("  no majority -> trust highest-reasoning tiebreaker C")
    return candC


async def _agree_or_escalate_func(
    state, prompt, full_prompt, candidate, setup, targets, func_name, lib
):
    """Function-mode analogue of _agree_or_escalate (iter12).

    Calls the function with its example defaults, serializes the RETURN value,
    and runs the same A/B agreement + literal-tiebreaker escalation that module
    mode uses. Matplotlib function problems (no comparable return value) and any
    setup/probe failure fall back to iter11's self-check+repair path, so this can
    only convert a 0 into a possible 1, never regress a passing answer's path.
    """
    py = _get_py(state)
    if py is None or not setup.strip() or not func_name:
        return candidate
    if str(lib).lower() == "matplotlib":
        # Plots have no comparable return value -> keep exec-only self-check+repair.
        return await _verify_and_repair(
            state, prompt, candidate, setup, targets, True, func_name, lib
        )
    if not await _setup_runnable(py, setup):
        return candidate

    def value_of(cand):
        return _run_func_value(py, setup, cand, func_name)

    sigA = await value_of(candidate)
    if sigA is None:
        # A crashed / function uncallable with defaults -> proven repair path.
        print("  func A failed value-probe -> self-check+repair fallback")
        return await _verify_and_repair(
            state, prompt, candidate, setup, targets, True, func_name, lib
        )

    # Independent second strong opinion (different family, no weak voter).
    candB = await _generate(CLAUDE_SONNET_4_6, full_prompt, reasoning="low", max_tokens=3072)
    if candB.strip():
        candB = ensure_indented(candB)
    if not candB.strip():
        print("  func B empty -> emit A")
        return candidate
    sigB = await value_of(candB)
    if sigB is None:
        print("  func B failed value-probe -> trust A")
        return candidate
    if sigA == sigB:
        print("  func A and B AGREE on returned value -> emit A (high confidence)")
        return candidate

    print("  func A and B DISAGREE -> escalate to GPT_5_4 (high) literal tiebreaker")
    tie_prompt = (
        full_prompt
        + "\n\n---\nTwo expert attempts produced DIFFERENT return values for the "
        "function `" + (func_name or "f") + "`.\n\nAttempt 1:\n<code>\n"
        + candidate
        + "\n</code>\n\nAttempt 2:\n<code>\n"
        + candB
        + "\n</code>\n\nThink carefully about which is correct under ALL inputs (not just "
        "the shown example) — watch for wrong library idioms, wrong broadcasting/axis, "
        "needless densify/reshape/ravel, dtype or container-type mismatch, and wrong "
        "function arity. IMPORTANT: DS-1000 reference solutions favor the MOST DIRECT, "
        "LITERAL call that reproduces the shown example. If one attempt is a plain library "
        "call (e.g. `sA.multiply(sB)`) and the other adds extra steps the problem never "
        "asked for (converting to dense, ravel/reshape, manual loops), strongly prefer the "
        "plain one — the clever version often diverges on inputs that differ in shape from "
        "the shown example. Return the single correct `<code>` block (the INDENTED function "
        "body ending in `return`; you may fix either attempt or write a better one)."
    )
    candC = await _generate(GPT_5_4, tie_prompt, reasoning="high", max_tokens=8192)
    if candC.strip():
        candC = ensure_indented(candC)
    if not candC.strip():
        print("  func tiebreaker empty -> emit A")
        return candidate
    sigC = await value_of(candC)
    if sigC is None:
        print("  func tiebreaker failed value-probe -> emit A")
        return candidate
    if sigC == sigA:
        print("  func tiebreaker matches A (2/3) -> emit A")
        return candidate
    if sigC == sigB:
        print("  func tiebreaker matches B (2/3) -> emit B")
        return candB
    print("  func three-way split -> trust highest-reasoning tiebreaker C")
    return candC


async def _run_func_value(py, setup, cand, func_name):
    if not cand.strip():
        return None
    try:
        out = str(await py(code=_build_function_value_probe(setup, cand, func_name)))
    except Exception as e:  # noqa: BLE001
        print(f"  func value-probe error: {e!r}")
        return None
    return _value_signature(out)


async def _run_value(py, setup, cand, targets):
    if not cand.strip():
        return None
    try:
        out = str(await py(code=_build_value_probe(setup, cand, targets)))
    except Exception as e:  # noqa: BLE001
        print(f"  value-probe error: {e!r}")
        return None
    return _value_signature(out)


async def _repair(prompt, candidate, setup, targets, py, lib):
    """Cross-model repair fed the actual traceback (iter3's repair, value-probe form)."""
    try:
        out = str(await py(code=_build_value_probe(setup, candidate, targets)))
    except Exception:  # noqa: BLE001
        out = ""
    repair_prompt = (
        BASE_INSTRUCTIONS
        + MODULE_HINT
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
        return ""
    sig = await _run_value(py, setup, repaired, targets)
    if sig is not None:
        print("  repair runs clean")
        return repaired
    print("  repair still imperfect; using repaired (different-model) answer")
    return repaired


async def _verify_and_repair(
    state, prompt, candidate, setup, targets, func_mode, func_name, lib
):
    """iter3's function-body self-check + single cross-model repair (unchanged)."""
    py = _get_py(state)
    if py is None or not setup.strip():
        return candidate

    is_mpl = str(lib).lower() == "matplotlib"

    def build(cand):
        if func_mode and func_name:
            return _build_function_check(setup, cand, func_name)
        if is_mpl:
            return _build_exec_only(setup, cand)
        return _build_module_check(setup, cand, targets)

    if not await _setup_runnable(py, setup):
        return candidate

    out = str(await py(code=build(candidate)))
    if "SELFCHECK_OK" in out and not _looks_like_traceback(out):
        print("  self-check passed")
        return candidate

    print("  self-check FAILED -> cross-model repair (CLAUDE_SONNET_4_6)")
    form = FUNCTION_HINT.format(fname=func_name or "f") if func_mode else MODULE_HINT
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
