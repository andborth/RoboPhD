"""DS-1000 solver: iter3's proven safe-repair core + multi-perspective consensus.

Lineage (verified on prior iterations):
  * iter3_safe_repair (95%, $0.002) is the champion: simplicity-first generation,
    func-body indentation normalization, execute the candidate, and repair ONLY on a real
    traceback — a clean run is NEVER second-guessed. Every stage falls back to the best
    clean candidate.
  * iter6/iter7 added LLM reconciliation of clean runs against the prompt's expected
    output and scored LOWER (85/90) at 2-3x the cost: free arbitration rewrote correct
    literal answers into wrong ones (667: dropped `x.assign`; 723: over-engineered a sparse
    multiply). Lesson: do NOT let a model rewrite a clean answer.

This solver keeps iter3's scaffold byte-for-byte and replaces ONLY the single generation
with a small prompt-perspective ensemble decided by OBJECTIVE execution agreement (no model
arbitration):

  1. Generate 3 candidates (cheap MINI) under 3 framings: idiom, general-robust, and
     exact-output-form. Distinct prompts guarantee diversity even if temperature is ignored;
     the output-form variant targets clean-but-WRONG dtype/shape misses (e.g. 129).
  2. Execute all three; group by exact probe output. >=2 agree -> ship a candidate with that
     output, preferring the simplicity variants and shortest code (literal answers win ties).
  3. No majority -> escalate ONCE to the stronger GPT_5_4; ship only if it runs clean, else
     fall back to the first clean candidate.
  4. All three error -> iter3's error-only repair on the first candidate.

Every shipped answer is a clean run (or iter3's repaired/fallback answer); a clean consensus
is never rewritten by an LLM, so the floor equals iter3. Cost stays well inside the $0.05
free zone.
"""

import re
import textwrap

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI, GPT_5_4

GEN_MODEL = GPT_5_4_MINI
STRONG_MODEL = GPT_5_4
GEN_CFG = GenerateConfig(reasoning_effort="low", max_tokens=2200)
STRONG_CFG = GenerateConfig(reasoning_effort="low", max_tokens=2600)

GUIDE = """You are an expert Python data-science programmer solving a DS-1000 problem.

You are given a problem and a code skeleton. Write ONLY the additional Python code
that, appended to the skeleton, makes the target variable hold the correct value.

OUTPUT FORMAT — put your final code inside a single <code>...</code> block:
- Inside the tags: executable Python ONLY. No prose, no markdown fences, no
  "BEGIN SOLUTION"/"END SOLUTION" markers, and no print() statements.
- Do NOT repeat the skeleton code that is already given — only the new code.

HOW TO THINK ABOUT DS-1000 (most failures come from ignoring these):
1. The reference solution is almost always SHORT, LITERAL, and IDIOMATIC. Prefer the
   simplest direct library call. Do NOT over-engineer: avoid adding value_counts,
   reindexing, manual loops, or extra reshaping unless the problem clearly requires it.
   When in doubt, take the most literal reading of the question (e.g. "N numbers ->
   probabilities" usually means just normalize those N numbers).
2. Prefer module-level convenience functions over heavier class-based APIs when both
   exist and give the same result — e.g. `sklearn.preprocessing.scale(data)` over
   `StandardScaler().fit_transform(data)`, `np.sort` over building a sorter object.
   The convenience function is usually what the reference uses and is more robust to
   the hidden test's input shape.
3. Write code that works for GENERAL inputs of the stated type, not only the worked
   example. The hidden test feeds different values/shapes (e.g. a 1-D array where the
   example looked 2-D). Avoid assumptions that only hold for the sample shown.
4. The hidden test compares the WHOLE object exactly. Match: same column names AND
   column order, same row/index order, and the same dtypes. An int-vs-object dtype or a
   swapped column order counts as WRONG even if the printed values look right.
5. Honor implicit constraints: phrases like "the efficient way", "without a loop",
   "not one by one", or "use <function>" mean some problems forbid for/while loops or
   require a specific function name to literally appear in your code.
6. Reproduce reference quirks: a pair [lo, hi] usually means an INCLUSIVE RANGE
   (lo <= x <= hi); np.column_stack/np.array on MIXED types coerces to strings;
   mode/value_counts tie-breaking picks the smaller label; groupby/pivot impose a sort
   order on the result.
7. For "complete this function" skeletons (`def f(...):` with a `### BEGIN SOLUTION`
   marker): write ONLY the function body, indented, ending in `return ...`. Do NOT call
   the function, print, or add markers.
"""

# Per-perspective addenda. Two simplicity-leaning, one exact-output-form. Distinct framings
# guarantee candidate diversity (so consensus is meaningful even at temperature 0).
PERSPECTIVES = [
    # A — pure idiom / simplest one-liner
    "\nPERSPECTIVE: Write the single most idiomatic, shortest direct library call that "
    "answers the literal question. Resist any extra step that is not strictly required.",
    # B — general robustness over the worked example
    "\nPERSPECTIVE: Make the code correct for GENERAL inputs of the stated type, not just "
    "the worked example (different shapes/values/lengths). Still keep it short and literal.",
    # C — exact output form (dtype/shape/order/quirks)
    "\nPERSPECTIVE: Match the EXACT final object the problem expects — same dtype (watch "
    "int-vs-float promotion), same shape, same column names AND order, same row/index "
    "order, and any displayed quirks. If the prompt shows the expected output, reproduce it "
    "exactly. Keep the code as simple as possible while hitting that exact form.",
]

REPAIR = """The candidate code below was appended to the DS-1000 skeleton and RAISED AN ERROR
when executed. Fix it so it runs cleanly AND is correct. Keep the solution as short and
idiomatic as possible. Output ONLY the final code inside a single <code>...</code> block
(no prose, no print statements, no skeleton repetition).

==== PROBLEM ====
{problem}

==== CANDIDATE CODE (errored) ====
{code}

==== EXECUTION ERROR ====
{output}
"""


def _extract_setup(prompt: str) -> str:
    """First <code>...</code> block in the prompt = the runnable skeleton."""
    m = re.search(r"<code>\s*\n?(.*?)</code>", prompt, re.DOTALL)
    setup = m.group(1) if m else ""
    setup = re.split(r"\n\s*#*\s*#+\s*BEGIN SOLUTION", setup)[0]
    setup = re.split(r"\nBEGIN SOLUTION", setup)[0]
    setup = re.split(r"\nWrite the remaining python code", setup)[0]
    return setup.rstrip()


def _extract_solution(text: str) -> str:
    """Pull the code the model wrote, stripping envelopes."""
    s = (text or "").strip()
    m = re.search(r"<code>\s*\n?(.*?)</code>", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        fence = re.search(r"```(?:python)?\s*\n?(.*?)```", s, re.DOTALL)
        if fence:
            s = fence.group(1)
    s = re.sub(r"^\s*###?\s*(BEGIN|END)\s+SOLUTION\s*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*(BEGIN|END)\s+SOLUTION\s*$", "", s, flags=re.MULTILINE)
    return s.strip("\n")


def _detect_target(prompt: str, setup: str):
    """Return ('var', name) or ('func', name) describing what to probe."""
    m = re.search(r"^\s*(\w+)\s*=\s*\.\.\.\s*#\s*put solution", prompt, re.MULTILINE)
    if m:
        return ("var", m.group(1))
    if "return the solution in this function" in prompt or re.search(
        r"###\s*BEGIN SOLUTION", prompt
    ):
        fm = re.search(r"def\s+(\w+)\s*\(", setup)
        return ("func", fm.group(1) if fm else "f")
    return ("var", "result")


def _indent4(code: str) -> str:
    """Indent every non-empty line by 4 spaces."""
    return "\n".join(("    " + ln) if ln.strip() else ln for ln in code.split("\n"))


def _body_compiles(body: str) -> bool:
    """True if `body` is a syntactically valid function body under `def f():`."""
    try:
        compile("def _probe_fn():\n" + body + "\n", "<body>", "exec")
        return True
    except SyntaxError:
        return False


def _normalize_func_body(solution: str) -> str:
    """Guarantee a function-completion body is validly indented under `def f():`."""
    raw = solution.strip("\n")
    if not raw.strip():
        return solution

    ded = textwrap.dedent(raw)
    cand_struct = _indent4(ded)  # preserves relative structure (correct for nested code)
    if _body_compiles(cand_struct):
        return cand_struct

    flat = "\n".join(ln.strip() for ln in ded.split("\n"))
    cand_flat = _indent4(flat)
    if _body_compiles(cand_flat):
        return cand_flat

    return cand_struct


def _build_program(setup: str, solution: str, kind: str, name: str) -> str:
    if kind == "func":
        target_expr = f"{name}()"
        preamble = ""
    else:
        target_expr = name
        preamble = f"globals().pop({name!r}, None)\n"
    probe = (
        '\nprint("===DS1000_PROBE===")\n'
        "try:\n"
        f"    _tg = {target_expr}\n"
        '    print("TYPE:", type(_tg).__name__)\n'
        "    try:\n"
        "        import pandas as _pd\n"
        "        if isinstance(_tg, _pd.DataFrame):\n"
        '            print("COLUMNS:", list(_tg.columns))\n'
        '            print("DTYPES:", dict(_tg.dtypes.astype(str)))\n'
        "        elif isinstance(_tg, _pd.Series):\n"
        '            print("DTYPE:", str(_tg.dtype))\n'
        "    except Exception:\n"
        "        pass\n"
        "    _r = repr(_tg)\n"
        "    print(_r[:2500])\n"
        "except Exception:\n"
        "    import traceback as _tb\n"
        '    print("PROBE_ERROR:")\n'
        "    print(_tb.format_exc()[:1800])\n"
    )
    return preamble + setup + "\n" + solution + "\n" + probe


def _errored(output: str) -> bool:
    """True only when execution actually raised — NOT a heuristic 'maybe wrong'."""
    o = output or ""
    return (
        "Traceback (most recent call last)" in o
        or "PROBE_ERROR:" in o
        or "===DS1000_PROBE===" not in o
    )


def _probe_key(output: str) -> str:
    """Normalize a probe output into a comparison key (drop the marker line)."""
    o = output or ""
    idx = o.find("===DS1000_PROBE===")
    if idx >= 0:
        o = o[idx:]
    return o.strip()


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input
        library = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={library}")

        setup = _extract_setup(prompt)
        kind, name = _detect_target(prompt, setup)

        def finalize(sol: str) -> str:
            return _normalize_func_body(sol) if kind == "func" else sol

        async def gen(addendum: str, model=GEN_MODEL, cfg=GEN_CFG) -> str:
            try:
                resp = await model.generate(GUIDE + addendum + "\n\n" + prompt, config=cfg)
                return _extract_solution(resp.completion or "")
            except Exception as e:
                print(f"  generate failed: {e}")
                return ""

        # ---- Stage 1: 3-perspective generation ---------------------------------
        candidates = []
        for i, persp in enumerate(PERSPECTIVES):
            c = await gen(persp)
            if c.strip():
                candidates.append(c)
            print(f"  cand[{i}]: {len(c)} chars")

        first = candidates[0] if candidates else ""

        def ship(sol: str, note: str) -> TaskState:
            state.output.completion = f"<code>\n{finalize(sol)}\n</code>"
            print(f"  {note}; emitted {len(sol)} chars")
            return state

        # Matplotlib (and empty) -> no result/return to probe; ship first candidate.
        if library == "Matplotlib" or not first.strip():
            return ship(first, "no exec (matplotlib/empty)")

        try:
            py = next(t for t in state.tools if ToolDef(t).name == "python_session")
        except Exception:
            return ship(first, "no sandbox")

        async def run(sol: str) -> str:
            try:
                program = _build_program(setup, finalize(sol), kind, name)
                return await py(code=program)
            except Exception as e:  # sandbox hiccup
                return f"PROBE_ERROR:\n{e}"

        # ---- Stage 2: execute every candidate, group by objective output -------
        clean = []  # list of (idx, code, probe_key)
        for i, c in enumerate(candidates):
            out = await run(c)
            if not _errored(out):
                clean.append((i, c, _probe_key(out)))
        print(f"  clean candidates: {len(clean)}/{len(candidates)}")

        if clean:
            # Tally objective probe outputs across the clean candidates.
            tally = {}
            for idx, code, key in clean:
                tally.setdefault(key, []).append((idx, code))
            # Pick the output with the most votes; tie-break toward simplicity variants
            # (lower idx, i.e. idiom/general before output-form) and shorter code.
            best_key = max(
                tally,
                key=lambda k: (
                    len(tally[k]),               # most agreement first
                    -min(i for i, _ in tally[k]),  # prefer earlier (simplicity) perspectives
                ),
            )
            group = tally[best_key]
            votes = len(group)
            if votes >= 2:
                # Consensus across independent framings — ship the shortest in the group.
                idx, code = min(group, key=lambda ic: (len(ic[1]), ic[0]))
                return ship(code, f"consensus {votes}/{len(clean)} (cand {idx})")
            # No agreement among clean candidates -> uncertain problem.

        # ---- Stage 3a: no consensus -> escalate ONCE to the strong model -------
        if clean:  # >=1 clean but no >=2 agreement
            print("  no consensus -> strong-model escalation")
            strong = await gen(PERSPECTIVES[2], model=STRONG_MODEL, cfg=STRONG_CFG)
            if strong.strip() and strong.strip() not in {c.strip() for _, c, _ in clean}:
                out_s = await run(strong)
                if not _errored(out_s):
                    return ship(strong, "strong escalation accepted (clean)")
                print("  strong escalation errored; falling back")
            # Fall back to the first clean candidate (never ship an errored answer).
            return ship(clean[0][1], "fallback to first clean candidate")

        # ---- Stage 3b: all candidates errored -> iter3 error-only repair -------
        print("  all candidates errored -> one repair pass")
        err_out = await run(first)
        try:
            repair_prompt = REPAIR.format(
                problem=prompt, code=first, output=(err_out or "")[:2500]
            )
            r2 = await STRONG_MODEL.generate(
                GUIDE + "\n\n" + repair_prompt, config=STRONG_CFG
            )
            fixed = _extract_solution(r2.completion or "")
        except Exception as e:
            print(f"  repair failed: {e}")
            fixed = ""

        if fixed.strip() and fixed.strip() != first.strip():
            out2 = await run(fixed)
            if not _errored(out2):
                return ship(fixed, "repair accepted (now runs clean)")
            print("  repair still errors; keeping first candidate")

        return ship(first, "no clean answer; shipped first candidate")

    return solve
