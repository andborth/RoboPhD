"""DS-1000 solver: literal reading + structure rules + cross-model agreement (iter19).

iter19_literal_struct_consensus is a STRICT SUPERSET of the two strongest lineage
agents. It starts from iter18_reason_agree_struct's code (architecture + structure/
dtype rules + function-mode cross-model agreement) and grafts back in the two assets
that made iter10_literal_consensus the iter18 batch leader (95%):
  1. The EMPHATIC literal-reading rule with two concrete traps (reverse/opposite of
     F -> direct arithmetic transform, NOT negated inputs; feed data to a routine in
     the SAME form it is presented, no unrequested preprocessing). This is the rule
     that won problem 706 (literal `tf.saved_model.save`) where iter18's softer
     "read literally" line let it drift to the wrong idiom.
  2. The literal lesson injected into the tiebreaker prompt, so a split between two
     strong candidates is broken toward the plain literal transform.
These sit ALONGSIDE iter18's structure/dtype rules (the #1 DS-1000 failure class:
container/dtype/order mismatch) and function-mode value agreement, which are
orthogonal — they catch the failure value-agreement cannot (right concept, wrong
shape) and extend the oracle to `def` problems. All prior agents cost ~$0.012-0.014
mean vs the $0.05 free zone (~3.5x headroom); every addition here is pure prompt text
or a guarded layer that only fires on a minority of problems -> free accuracy plays.

--- iter18 additions (retained) ---
  1. Structure/dtype prompt rules stacked onto the base instructions.
  2. Cross-model value agreement EXTENDED to function-mode problems; function
     candidates keep iter3's self-check+repair, then ADDITIVELY get the B-vs-A value
     comparison + high-reasoning tiebreaker, fully guarded (upgrade-only).
Everything below is iter9's docstring.

--- iter9 lineage ---
Synthesis of the two best iter8 agents, each of which scored 95%:
  * iter8_reason_cascade fixed wrong-but-runnable errors with a MEDIUM-reasoning
    base model.
  * iter7_agree_escalate fixed them with CROSS-MODEL VALUE AGREEMENT.
On problem 10 each lever fixed the failure independently. Cost headroom is ~6x
(mean ~$0.008 vs the $0.05 free zone), so this agent takes the UNION of both
levers for redundant, complementary protection instead of choosing one.

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

from model_registry import GPT_5_4, CLAUDE_SONNET_4_6


BASE_INSTRUCTIONS = """You are an expert Python data scientist solving a DS-1000 problem.

You are given a problem with a code skeleton. Write ONLY the Python code that should be appended AFTER the given skeleton so that the requested variable holds the correct value.

Rules — follow them exactly:
- Output a single `<code>` ... `</code>` block and nothing else. No prose, no explanation, no markdown ``` fences.
- Do NOT repeat any code already shown in the skeleton (imports, data definitions, `load_data()` calls, asserts, the `def` line). Only write the NEW code.
- Assign the answer to the EXACT variable name the problem asks for. Look for a line like `result = ... # put solution in this variable`, or wording like "put score in `b`, put prediction in `c`". The name is often `result` but can be `proba`, `b`, `c`, `df`, `predict`, `centered_scaled_data`, etc. Match it precisely.
- CONSTRUCT every object the problem refers to that is NOT already defined in the skeleton's `<code>` block. Only variables literally assigned inside the skeleton's `<code>` pre-exist. If the prose mentions an estimator/model/object (e.g. "with example variable `logReg`", "fit the model `clf`"), you must create it yourself (e.g. `logReg = LogisticRegression()`) before using it — do not assume it already exists.
- Prefer the library's own canonical, idiomatic function over a manual reimplementation. DS-1000 references use the standard library call (e.g. `sklearn.preprocessing.scale`, `scipy.interpolate.RectBivariateSpline`, `scipy.stats.rankdata`, `np.column_stack`), and a workaround that gives a slightly different numeric/dtype result can be marked wrong even when it looks correct on the shown example. When the question says "without using X", "not one by one", "the efficient way", or names a function, honor it — avoid explicit Python `for`/`while` loops when a vectorized library call does the job.
- When asked to DEFINE A FUNCTION, give it exactly the parameter signature implied by the example call and the module-level variables: arguments that the example passes in are parameters; values already defined at module level (e.g. `x_min`, `x_max`) should be used directly as globals, NOT added as extra parameters. Match the arity the hidden test will call with.
- MIRROR THE EXPECTED OUTPUT STRUCTURE EXACTLY — this is the most common reason a correct-looking answer is marked wrong. Pandas/numpy equality is dtype-, order-, and type-sensitive. Match: the dtype (do NOT silently upcast/downcast — keep an int column int; and remember some reference idioms force a wider dtype, e.g. `np.column_stack` / `np.array` of mixed string+number arrays produces a single object/string `<U` dtype, so a column of numbers becomes strings); the container type (DataFrame vs Series vs scalar vs ndarray vs dict vs list); column names and order; row order and index. If you built the wrong type (ndarray where a DataFrame is implied, or vice versa), convert to the one the problem implies.
- If the problem DISPLAYS a desired output (a printed table/dict/array), treat its exact STRUCTURE — column/row order, nesting, dtype — as authoritative, overriding what a "natural" library call would default to. But NEVER hardcode the displayed VALUES: they come from the shown example only, while the hidden test runs your code on DIFFERENT inputs. Always COMPUTE the result generally from the input variables.
- Favor the MOST DIRECT, LITERAL reading of the problem. The reference solution is almost always the simplest expression that plainly reproduces the described transformation and the shown example — NOT the most sophisticated or most statistically "correct" equivalent, and NOT a clever multi-step reimplementation that risks a different type or numeric result. Two specific traps: (1) When the problem says "the reverse / opposite / descending version of function F", apply a direct arithmetic transform of F's normal output (e.g. `len(a) - rankdata(a)`, `max(x) - x`) rather than re-deriving it by feeding negated/transformed inputs into F (`rankdata(-a)`), which can differ at ties or boundaries. (2) Feed the data to a function in the SAME literal form the problem presents it (e.g. if it hands you a 2-D matrix and asks to cluster it, pass that matrix straight into the routine as shown; if it asks to save a model as a SavedModel folder, call the literal `tf.saved_model.save`), and do NOT add preprocessing the problem never mentioned (squareform/condensing, reshaping, extra normalization). When a plain one-liner and a "smarter" multi-step version both match the example, choose the plain one-liner.
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


def _build_value_probe(setup: str, candidate: str, targets) -> str:
    """Run setup+candidate, then print a canonical serialization of each target."""
    body = [setup, candidate, _SERIALIZER]
    for t in targets:
        body.append(f"assert {t!r} in dir(), 'TARGET_MISSING:{t}'")
        body.append(f"print('VVV:{t}:' + __ser({t}))")
    body.append("print('VALUE_OK')")
    return "\n".join(body)


def _build_function_value_probe(setup: str, candidate: str, func_name: str) -> str:
    """Define setup+function body, call the function with no args, serialize return.

    If the function requires arguments the call raises -> the probe traceback makes
    the signature None, so agreement degrades safely to trusting candidate A.
    """
    body = [
        setup,
        candidate,
        _SERIALIZER,
        f"__val = {func_name}()",
        "print('VVV:return:' + __ser(__val))",
        "print('VALUE_OK')",
    ]
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
                # Function-body / def problems: keep iter3's proven self-check +
                # repair path (no regression), THEN additively apply cross-model
                # value agreement on the (possibly repaired) candidate.
                candidate = await _verify_and_repair(
                    state, prompt, candidate, setup, targets, func_mode, func_name, lib
                )
                candidate = await _agree_or_escalate(
                    state, prompt, full_prompt, candidate, setup, targets, lib,
                    func_mode, func_name,
                )
            else:
                candidate = await _agree_or_escalate(
                    state, prompt, full_prompt, candidate, setup, targets, lib,
                    func_mode, func_name,
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


async def _agree_or_escalate(state, prompt, full_prompt, candidate, setup, targets, lib,
                             func_mode=False, func_name=None):
    """Cross-check candidate A against an independent strong model B by comparing the
    ACTUAL produced values; escalate to a tiebreaker on disagree. Works for both
    module-mode (compare target vars) and function-mode (compare the function return).

    Never raises out (caller-guarded); returns the chosen code.
    """
    py = _get_py(state)
    if py is None or not setup.strip():
        return candidate
    if func_mode and not func_name:
        return candidate
    if not await _setup_runnable(py, setup):
        return candidate

    def value_of(cand):
        return _run_value(py, setup, cand, targets, func_mode, func_name)

    sigA = await value_of(candidate)
    if sigA is None:
        if func_mode:
            # A already passed iter3's self-check+repair; a no-arg probe miss here
            # just means the fn needs args -> don't second-guess, trust A.
            return candidate
        # A crashed / left a target undefined (e.g. 919 NameError). Repair it.
        print("  A failed value-probe -> cross-model repair")
        repaired = await _repair(prompt, candidate, setup, targets, py, lib)
        return repaired or candidate

    # Independent second strong opinion (different family, no weak voter).
    candB = await _generate(CLAUDE_SONNET_4_6, full_prompt, reasoning="low", max_tokens=3072)
    if func_mode:
        candB = ensure_indented(candB)
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
        "`len(a) - rankdata(a)`, or feeding the given data straight into the routine in its "
        "presented form) and the other is a 'smarter' equivalent (negating inputs, "
        "condensing/reshaping the data, extra preprocessing the problem never asked for, or "
        "a non-default function variant), strongly prefer the literal one — the clever "
        "version often diverges at ties/boundaries or in column order/dtype. Return the "
        "single correct `<code>` block (you may fix either attempt or write a better one)."
    )
    candC = await _generate(GPT_5_4, tie_prompt, reasoning="high", max_tokens=8192)
    if func_mode:
        candC = ensure_indented(candC)
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
    print("  three-way split -> trust highest-reasoning tiebreaker C")
    return candC


async def _run_value(py, setup, cand, targets, func_mode=False, func_name=None):
    if not cand.strip():
        return None
    if func_mode and func_name:
        code = _build_function_value_probe(setup, cand, func_name)
    else:
        code = _build_value_probe(setup, cand, targets)
    try:
        out = str(await py(code=code))
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
