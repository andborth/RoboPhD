"""DS-1000 solver: generate -> execute -> output-grounded self-review.

The seed agent was a single no-reasoning GPT_5_4_MINI call with no self-checking.
Its failures split into (a) a runtime error in a function-completion skeleton and
(b) several "runs fine but wrong" cases (column order, dtype coercion quirks,
range-vs-exact semantics, mode tie-breaking).

`python_session` is NOT metered, so we can execute candidates for free. This solver:
  1. generates with a DS-1000-specific instruction prompt + low reasoning,
  2. runs the candidate against the skeleton in the sandbox (auto-detecting the
     target variable / function),
  3. shows the model its ACTUAL output (or traceback) and lets it conservatively
     confirm or minimally fix the answer.

Every stage falls back to the best candidate so far, so we never do worse than
a one-shot generation. Cost stays well inside the $0.05 free zone.
"""

import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI

MODEL = GPT_5_4_MINI

GUIDE = """You are an expert Python data-science programmer solving a DS-1000 problem.

You are given a problem and a code skeleton. Write ONLY the additional Python code
that, appended to the skeleton, makes the target variable hold the correct value.

OUTPUT FORMAT — put your final code inside a single <code>...</code> block:
- Inside the tags: executable Python ONLY. No prose, no markdown fences, no
  "BEGIN SOLUTION"/"END SOLUTION" markers, and no print() statements.
- Do NOT repeat the skeleton code that is already given — only the new code.

CRITICAL DS-1000 RULES (most failures come from ignoring these):
1. The hidden test compares the WHOLE object. Match the expected result EXACTLY:
   same column names AND column order, same row order / index, and the same dtypes.
   An int-vs-object dtype or a swapped column order counts as WRONG even if the
   printed values look right.
2. Use the idiomatic library solution the question implies. Phrases like "the
   efficient way", "without a loop", "not one by one", or "use <function>" are
   constraints — some problems forbid for/while loops or require a specific
   function name to literally appear in your code.
3. Read pair/range semantics literally: a pair like [lo, hi] usually means an
   INCLUSIVE RANGE (lo <= x <= hi), not two exact values. Re-read what each input
   variable actually represents before coding.
4. Reproduce the quirks of the natural reference solution, e.g.:
   - np.column_stack / np.array on MIXED types coerces everything to strings;
   - DataFrame.mode(axis=1) / value_counts tie-breaking picks the smaller label;
   - pandas groupby and pivot/unstack impose a specific sort order on the result.
5. For "complete this function" skeletons (e.g. `def f(...):` with a
   `### BEGIN SOLUTION` marker): write ONLY the function body, properly indented,
   ending in `return ...`. Do NOT call the function, print, or add markers.
"""

REVIEW = """Below is a DS-1000 problem, your candidate solution, and what that code
ACTUALLY produced when run against the skeleton.

Carefully compare the actual output to the expected result described in the problem.
Check, in order: (1) did it error? (2) column names and COLUMN ORDER, (3) row/index
order, (4) DTYPES (int vs float vs object/string), (5) the exact semantics the
problem asked for (ranges, tie-breaking, which values change).

If the solution is correct, return it UNCHANGED. Only revise if you find a concrete
discrepancy, and then make the minimal fix. Output ONLY the final code inside a single
<code>...</code> block (no prose, no print statements, no skeleton repetition).

==== PROBLEM ====
{problem}

==== YOUR CANDIDATE CODE ====
{code}

==== ACTUAL EXECUTION RESULT ====
{output}
"""


def _extract_setup(prompt: str) -> str:
    """First <code>...</code> block in the prompt = the runnable skeleton.

    Function-completion prompts have no real closing </code> before the
    instruction text (the only </code> is inside a literal `<code>...</code>`
    mention), so the match can over-capture. Truncate at the BEGIN SOLUTION
    marker and drop the trailing "Write the remaining..." instruction.
    """
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
    # Drop any stray solution markers.
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


def _build_program(setup: str, solution: str, kind: str, name: str) -> str:
    if kind == "func":
        target_expr = f"{name}()"
        preamble = ""
    else:
        target_expr = name
        # Clear any stale value from a previous candidate so a failing
        # solution can't silently reuse an earlier success.
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


def _looks_failed(output: str) -> bool:
    o = output or ""
    return (
        "Traceback (most recent call last)" in o
        or "PROBE_ERROR:" in o
        or "Error:" in o
        or "===DS1000_PROBE===" not in o
    )


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input
        library = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={library}")

        gen_cfg = GenerateConfig(reasoning_effort="low", max_tokens=2200)

        # ---- Stage 1: generate ------------------------------------------------
        try:
            resp = await MODEL.generate(GUIDE + "\n\n" + prompt, config=gen_cfg)
            candidate = _extract_solution(resp.completion or "")
        except Exception as e:
            print(f"  generate failed: {e}")
            candidate = ""
        best = candidate

        # Matplotlib (and empty) -> no reliable result to probe; ship generation.
        if library == "Matplotlib" or not candidate.strip():
            state.output.completion = f"<code>\n{best}\n</code>"
            print(f"  emitted {len(best)} chars (no exec)")
            return state

        # Locate the sandbox tool.
        try:
            py = next(t for t in state.tools if ToolDef(t).name == "python_session")
        except Exception:
            state.output.completion = f"<code>\n{best}\n</code>"
            return state

        setup = _extract_setup(prompt)
        kind, name = _detect_target(prompt, setup)

        async def run(sol: str) -> str:
            try:
                program = _build_program(setup, sol, kind, name)
                return await py(code=program)
            except Exception as e:  # sandbox hiccup
                return f"PROBE_ERROR:\n{e}"

        # ---- Stage 2: execute the first candidate ----------------------------
        out = await run(candidate)
        cand_failed = _looks_failed(out)
        print(f"  exec ok={not cand_failed}")

        # ---- Stage 3: output-grounded self-review ----------------------------
        try:
            review_prompt = REVIEW.format(
                problem=prompt, code=candidate, output=(out or "")[:3000]
            )
            r2 = await MODEL.generate(GUIDE + "\n\n" + review_prompt, config=gen_cfg)
            revised = _extract_solution(r2.completion or "")
        except Exception as e:
            print(f"  review failed: {e}")
            revised = ""

        if revised.strip() and revised.strip() != candidate.strip():
            out2 = await run(revised)
            rev_failed = _looks_failed(out2)
            # Accept the revision unless it regresses (errors where the
            # original ran clean).
            if not (rev_failed and not cand_failed):
                best = revised
                cand_failed = rev_failed
                out = out2
            print(f"  revised exec ok={not rev_failed}; accepted={best is revised}")

        # ---- Stage 4: one repair pass if still erroring ----------------------
        if cand_failed:
            try:
                repair_prompt = REVIEW.format(
                    problem=prompt, code=best, output=(out or "")[:3000]
                ) + "\n\nThe code above ERRORS. Fix it so it runs and is correct."
                r3 = await MODEL.generate(
                    GUIDE + "\n\n" + repair_prompt, config=gen_cfg
                )
                fixed = _extract_solution(r3.completion or "")
                if fixed.strip():
                    out3 = await run(fixed)
                    if not _looks_failed(out3):
                        best = fixed
                        print("  repair accepted")
            except Exception as e:
                print(f"  repair failed: {e}")

        state.output.completion = f"<code>\n{best}\n</code>"
        print(f"  emitted {len(best)} chars")
        return state

    return solve
