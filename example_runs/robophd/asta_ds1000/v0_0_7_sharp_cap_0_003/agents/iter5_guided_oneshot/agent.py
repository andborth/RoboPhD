"""DS-1000 solver: guided one-shot.

Iteration-4 evidence was unambiguous: the plain one-shot `GPT_5_4_MINI`
seed (80%) beat both elaborate verify/self-consistency/reasoning agents
(70% each) *and* was the cheapest. The complex agents lost simple
problems by generating over-clever, broken answers (e.g. calling
`list.index((name, SVC()))` which raises ValueError, or hand-rolling a
loop where a named library function was required).

So this agent keeps the winning architecture — a single cheap mini call,
no reasoning, no verification loop — and adds only a static expert guide
that fixes the *careful-reading / simplicity* failure classes seen across
all agents:

  * prefer the shortest idiomatic library call (don't over-engineer)
  * write only the requested new code; don't re-run setup / retrain
  * reproduce any shown expected output exactly (off-by-one, dtype, order)
  * honor implicit style constraints (use the named function; no loops)

There is no dynamic decision path, so there is nothing that can pick a
worse candidate than the model's direct answer.
"""

from inspect_ai.solver import Generate, TaskState, solver

from model_registry import GPT_5_4_MINI


# Concise expert guide prepended to every prompt. Kept short (cheap input
# tokens) but covers the general DS-1000 pathologies observed in the data.
GUIDE = """You are an expert Python data-science engineer solving a DS-1000 problem.
You must write the Python code that goes where `[insert]` / `... # put solution` appears,
so that the variable the problem asks for (`result`, `df`, `proba`, a returned function
value, or a plot) ends up correct.

Follow these rules — they matter as much as correctness:

1. PREFER THE SHORTEST IDIOMATIC SOLUTION. The intended answer is almost always a single
   direct library call or a one-line expression. Do NOT over-engineer, do NOT add extra
   steps, and do NOT build clever constructions (e.g. avoid `list.index((name, Obj()))`
   which fails because a fresh object is not equal to the stored one — use the literal
   index/position the prompt implies).

2. WRITE ONLY THE NEW CODE REQUESTED. Do not repeat, redefine, re-import, reload data, or
   re-train anything that the given setup already established — reference those existing
   variables directly. If the setup ends with a comment telling you what to do (e.g.
   "# Save the model in export/1"), that comment IS the exact task; do just that and
   nothing more. Never reference variables that are not defined in the visible setup.

3. REPRODUCE ANY SHOWN EXPECTED OUTPUT EXACTLY. If the problem displays a desired result
   (an array, a table, a number), mentally trace the given example and make your code
   produce that displayed output precisely — matching sign, order, off-by-one, and dtype.
   Do not "improve" or shift it.

4. HONOR STYLE / IDIOM CONSTRAINTS. Phrasing like "how do I do X with <library>",
   "the efficient way", "vectorized", or "without a loop" means you must use the library's
   named built-in function; never hand-roll a Python for/while loop or reimplement the
   function manually. For function-style problems ("define a function named f(...)"), give
   parameters DEFAULT values that match the variables in the setup, since the hidden test
   may call it with fewer arguments.

5. SMALL FIXES: prefer current APIs — scipy `simps→simpson`, `cumtrapz→cumulative_trapezoid`,
   `trapz→trapezoid`; pandas `DataFrame.append→pd.concat`. Include necessary dtype
   conversions and `.fit()` calls; don't drop needed steps for the sake of terseness.

OUTPUT FORMAT: reply with ONLY a single `<code> ... </code>` block containing executable
Python — no prose, no explanations, no markdown fences, no BEGIN/END SOLUTION markers.

Here is the problem:

"""


def _extract_code(text: str) -> str:
    """Pull the solution code out of the model response and emit a clean
    single `<code>...</code>` block for the scorer."""
    s = (text or "").strip()

    # Prefer content inside <code>...</code> if the model used the tags.
    if "<code>" in s and "</code>" in s:
        inner = s.split("<code>", 1)[1].split("</code>", 1)[0].strip("\n")
        if inner.strip():
            return f"<code>\n{inner}\n</code>"

    # Otherwise strip a markdown fence if present.
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            block = parts[1]
            # Drop a leading language tag like "python\n".
            if "\n" in block:
                first, rest = block.split("\n", 1)
                if first.strip().isalpha():
                    block = rest
            block = block.strip("\n")
            if block.strip():
                return f"<code>\n{block}\n</code>"

    # Fall back to the raw response (guide told the model to emit only code).
    return f"<code>\n{s}\n</code>"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={lib}")

        prompt = GUIDE + state.input
        resp = await GPT_5_4_MINI.generate(prompt)
        state.output.completion = _extract_code(resp.completion or "")

        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
