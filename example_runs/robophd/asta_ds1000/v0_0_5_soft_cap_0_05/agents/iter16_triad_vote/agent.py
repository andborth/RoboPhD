"""DS-1000 solver: 3-family executed-value majority vote (iter16).

iter16_triad_vote = iter15_agree_sharpen's architecture and prompt VERBATIM, with
ONE principled change: the module-mode 2-model agree/escalate step is replaced by a
true MAJORITY VOTE over three independent strong candidates from three different
model families, compared by their ACTUAL produced value.

Why diversity, not depth:
  * iter15's notes document that the depth levers regressed: bumping candidate A to
    MEDIUM reasoning (iter9) over-engineered idiom answers and LOST problems; an
    execution-RECONCILE pass toward displayed values (iter14) hardcoded a shown
    example. So candidate A stays at reasoning="low" and the prompt is unchanged.
  * The proven, non-fragile lever is the executed-value AGREEMENT mechanism. The
    canonical way to strengthen an ensemble without over-engineering is to add a
    decorrelated voter, turning agreement into a majority vote.

Module-mode flow (runnable-setup problems):
  * Three INDEPENDENT candidates -- A = GPT_5_4 (low), B = CLAUDE_SONNET_4_6 (low),
    C = GEMINI_3_1_PRO_PREVIEW (low). Different families -> decorrelated errors.
  * Execute each in the FREE python_session; reduce to a canonical numpy/pandas-aware
    value signature.
  * MAJORITY by produced value: >=2 candidates with the same value -> emit a
    representative (source-quality order A->B->C). This directly overrules a single
    wrong model when the other two agree, with no extra LLM call -- the main gain
    over 2-model agreement. Candidates that crash / leave the target undefined are
    excluded from the vote.
  * No two agree (or <2 ran clean) -> escalate to a GPT_5_4 (medium) tiebreaker shown
    the clean candidates + divergent outputs; if its value matches one of the three
    it forms a 2-vote majority, else trust the higher-reasoning tiebreaker. If every
    candidate crashed, cross-model repair on A.

Function-body / non-runnable-setup problems keep iter15's self-check-and-repair path.

The whole vote/escalation layer is exception-guarded: any failure falls back to
candidate A, so it can only convert a 0 into a possible 1, never the reverse.
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
- When asked to DEFINE A FUNCTION, give it exactly the parameter signature implied by the example call and the module-level variables: arguments that the example passes in are parameters; values already defined at module level (e.g. `x_min`, `x_max`) should be used directly as globals, NOT added as extra parameters. Match the arity the hidden test will call with.
- MIRROR THE EXPECTED OUTPUT STRUCTURE EXACTLY. Pandas/numpy equality checks are dtype-, order-, and type-sensitive. Match: the dtype (don't silently upcast/downcast — e.g. don't turn an int column into float, and remember some reference idioms like `np.column_stack` of mixed string+number arrays produce an object/string dtype); the container type (DataFrame vs Series vs scalar vs ndarray vs dict vs list); column names and column order; row order and index. When the result is a numpy array but you built a pandas object (or vice versa), convert to the type the problem implies.
- If the problem DISPLAYS a desired output (a printed table, dict, or array), treat its exact STRUCTURE as authoritative — its column order, row order, nesting, and dtype override whatever a "natural" library call would produce by default (e.g. if a melt/stack shows the levels in reversed order, reorder to match). But NEVER hardcode the displayed VALUES: those come from the example input only, and the hidden test runs your code on DIFFERENT inputs. Always COMPUTE the result generally from the input variables so it is correct for any input.
- Read the problem's framing LITERALLY and answer the exact question asked. If it says the given numbers ARE the frequencies/weights/probabilities, operate on them directly; do not substitute a more "standard" procedure (re-binning, re-estimating, re-deriving) that the problem did not ask for.
- Prefer the SIMPLEST idiomatic solution that satisfies the request. Do not over-engineer: when a single library call (e.g. `sA.multiply(sB)`, `df / df.sum()`) produces the asked-for value, use it rather than a clever multi-step reimplementation that risks a different type or numeric result.
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
        candidate = await _generate(
            GPT_5_4, full_prompt, reasoning="low", max_tokens=4096
        )
        if not candidate.strip():
            candidate = await _generate(GPT_5_4, full_prompt, max_tokens=3000)
        if func_mode:
            candidate = ensure_indented(candidate)

        try:
            if func_mode:
                # Function-body / def problems: keep iter3's proven path.
                candidate = await _verify_and_repair(
                    state, prompt, candidate, setup, targets, func_mode, func_name, lib
                )
            else:
                candidate = await _vote_or_escalate(
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


async def _vote_or_escalate(state, prompt, full_prompt, candidate, setup, targets, lib):
    """Module-mode: 3-family majority vote over the candidates' EXECUTED values.

    Three independent strong candidates (different model families) are run in the
    free python_session and reduced to canonical value signatures. If >=2 agree, the
    majority value wins (a wrong single model is directly overruled). Otherwise a
    medium-reasoning tiebreaker decides. Never raises out (caller-guarded).
    """
    py = _get_py(state)
    if py is None or not setup.strip():
        return candidate
    if not await _setup_runnable(py, setup):
        return candidate

    async def value_of(cand):
        return await _run_value(py, setup, cand, targets)

    # Candidate A (already generated, GPT_5_4 low).
    sigA = await value_of(candidate)

    # Two more INDEPENDENT strong opinions from different families.
    candB = await _generate(CLAUDE_SONNET_4_6, full_prompt, reasoning="low", max_tokens=3072)
    candC = await _generate(GEMINI_3_1_PRO_PREVIEW, full_prompt, reasoning="low", max_tokens=3072)
    sigB = await value_of(candB) if candB.strip() else None
    sigC = await value_of(candC) if candC.strip() else None

    # Collect clean (ran + target defined) candidates in source-quality order.
    clean = []  # list of (label, code, sig)
    if sigA is not None:
        clean.append(("A", candidate, sigA))
    if sigB is not None:
        clean.append(("B", candB, sigB))
    if sigC is not None:
        clean.append(("C", candC, sigC))

    if not clean:
        # Everybody crashed / left target undefined -> repair candidate A.
        print("  all candidates failed value-probe -> cross-model repair")
        repaired = await _repair(prompt, candidate, setup, targets, py, lib)
        return repaired or candidate

    # Majority by produced value. Prefer the cluster representative earliest in
    # source-quality order (A->B->C) so emitted CODE comes from the stronger model.
    counts = {}
    for label, code, sig in clean:
        counts.setdefault(sig, []).append((label, code))
    best_sig, best_members = max(counts.items(), key=lambda kv: len(kv[1]))
    if len(best_members) >= 2:
        rep_label, rep_code = best_members[0]
        labels = "".join(l for l, _ in best_members)
        print(f"  MAJORITY {len(best_members)}/{len(clean)} agree ({labels}) -> emit {rep_label}")
        return rep_code

    # No two clean candidates agree (or only one ran clean) ----------------
    if len(clean) == 1:
        only_label, only_code, _ = clean[0]
        print(f"  only {only_label} ran clean -> emit {only_label}")
        return only_code

    # All clean candidates disagree -> escalate to a medium-reasoning tiebreaker
    # shown the clean attempts and their divergent outputs.
    print("  clean candidates DISAGREE -> escalate to GPT_5_4 (medium) tiebreaker")
    attempts = "\n\n".join(
        f"Attempt {label}:\n<code>\n{code}\n</code>" for label, code, _ in clean
    )
    tie_prompt = (
        full_prompt
        + "\n\n---\nSeveral expert attempts produced DIFFERENT results for the requested "
        "variable(s): " + ", ".join(targets) + ".\n\n" + attempts
        + "\n\nThink carefully about which is correct under ALL inputs (not just the "
        "shown example) — watch for wrong library idioms, off-by-one ranking, wrong "
        "function arity, undefined objects, or dtype issues. Return the single correct "
        "`<code>` block (you may fix any attempt or write a better one)."
    )
    candD = await _generate(GPT_5_4, tie_prompt, reasoning="medium", max_tokens=4096)
    if not candD.strip():
        print("  tiebreaker empty -> emit A-cluster representative")
        return clean[0][1]
    sigD = await value_of(candD)
    if sigD is None:
        print("  tiebreaker failed value-probe -> emit clean representative")
        return clean[0][1]
    for label, code, sig in clean:
        if sig == sigD:
            print(f"  tiebreaker matches {label} (2 votes) -> emit {label}")
            return code
    print("  tiebreaker forms its own value -> trust highest-reasoning tiebreaker")
    return candD


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
