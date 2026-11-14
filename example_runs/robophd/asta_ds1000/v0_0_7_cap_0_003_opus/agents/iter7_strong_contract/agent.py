"""DS-1000 solver: strong one-shot + harness-contract framing (iter7_strong_contract).

Settled architecture (7 iterations of evidence): a *plain one-shot* wins. Every added
mechanism — verification, self-consistency, reasoning escalation — is net-negative
(wrong DS-1000 answers run clean, so verification catches nothing and swaps
correct-simple for over-clever-broken). Heavy prescriptive guides on the *mini* model
also backfire (iter5 collapsed to 40%). This agent adds neither: the output is still
exactly one candidate — the model's direct answer.

The refinement over iter6_strong_oneshot targets its two — and only two — losses on the
iter6 batch, both pure harness-contract slips, NOT reasoning errors:

  * 451: emitted a bare expression `np.zeros((20,10,10,2))` instead of the required
    `arr = np.zeros(...)`  ->  NameError: arr not defined.
  * 365: wrote the correct function body but ALSO appended driver code
    `a = f(a=example_a, ...); print(a)`  ->  NameError: example_a not defined
    (the hidden test supplies its own setup; example-only vars don't exist there).

Both are fixed by a short, uniform, format-ONLY preamble (assign the target variable in
full; write only the solution slot; no drivers/prints). Unlike iter5's prescriptive
per-case *reasoning* rules aimed at a mini model, these are contract clarifications that
mirror what correct answers already do, applied to a strong model that isn't confused by
a few lines of instruction. They preserve the strong model's reasoning wins (iter6 fixed
seed's 18 range-semantics and 999 tensor-shape misses) while killing 365/451.

Cost: a single GPT_5_4 call measured $0.0016/problem in iter6 — inside the $0.003 free
zone, so cost never trades against accuracy here. Default reasoning ("none") keeps output
short and avoids the over-thinking that produces over-clever answers.
"""

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

from model_registry import GPT_5_4, GPT_5_4_MINI


# Short, contract-ONLY framing. Two jobs: (1) format the output for correct extraction;
# (2) state the harness contract so the strong model stops dropping the target assignment
# (451) and stops adding driver/print code that references example-only setup vars (365).
# No per-case reasoning rules (those are what backfired in iter5).
PREAMBLE = (
    "You are an expert Python data-scientist solving one DS-1000 problem. Your code is "
    "appended immediately after the shown setup and run by a hidden test harness that "
    "supplies its OWN copies of the setup variables (often with different values). Write "
    "ONLY the code that fills the solution slot:\n"
    "- Make the requested target variable correct. If the skeleton shows "
    "`name = ... # put solution in this variable`, write the FULL assignment "
    "`name = <expression>` -- never a bare expression, and never omit it. For a "
    "function-style skeleton (`def f(...):`), write only the function body.\n"
    "- Reference the existing setup variables directly, but do NOT re-import, redefine, "
    "reload or re-train them, do NOT call the function yourself, and do NOT print.\n"
    "- Prefer the shortest idiomatic library call; do not over-engineer or add steps.\n\n"
    "Reply with ONLY a single <code> ... </code> block of executable Python -- no prose, "
    "no markdown fences, no BEGIN/END markers.\n\n"
    "Problem:\n"
)

# Output budget cap (cost safety). DS-1000 solutions are short; this is generous.
MAX_TOKENS = 1024


def _extract_code(text: str) -> str:
    """Pull the solution out of the model response and emit a single clean
    `<code>...</code>` block for the scorer."""
    s = (text or "").strip()

    # Prefer content inside <code>...</code> if present.
    if "<code>" in s and "</code>" in s:
        inner = s.split("<code>", 1)[1].split("</code>", 1)[0].strip("\n")
        if inner.strip():
            return f"<code>\n{inner}\n</code>"

    # Otherwise strip a markdown fence if the model used one.
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if "\n" in block:
                first, rest = block.split("\n", 1)
                if first.strip().isalpha():  # drop a leading "python" tag
                    block = rest
            block = block.strip("\n")
            if block.strip():
                return f"<code>\n{block}\n</code>"

    # Fall back to the raw response (preamble told the model to emit only code).
    return f"<code>\n{s}\n</code>"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={lib}")

        prompt = PREAMBLE + state.input
        cfg = GenerateConfig(max_tokens=MAX_TOKENS)

        completion = ""
        try:
            resp = await GPT_5_4.generate(prompt, config=cfg)
            completion = resp.completion or ""
        except Exception as e:  # provider hiccup: never hard-0 the problem
            print(f"  GPT_5_4 error ({e!r}); falling back to mini")

        if not completion.strip():
            try:
                resp = await GPT_5_4_MINI.generate(prompt, config=cfg)
                completion = resp.completion or ""
            except Exception as e:
                print(f"  mini fallback error ({e!r})")

        state.output.completion = _extract_code(completion)
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
