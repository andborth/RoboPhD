"""DS-1000 solver: strong one-shot (iter6_strong_oneshot).

Three iterations of evidence on this benchmark converge on one conclusion:
the *plain one-shot* architecture wins, and every added mechanism has been
net-negative.

  * Iter-4: plain GPT_5_4_MINI seed = 80% (cheapest); verify / reasoning
    machinery = 70% each.
  * Iter-5: plain seed = 75%; the same cheap one-shot with a long 5-rule
    "expert guide" collapsed to 40% (the guide dropped 7 problems the raw
    seed got right).

Both falsified levers are avoided here:
  - No verification / self-consistency (wrong DS-1000 answers run clean, so
    it catches nothing and swaps correct-simple for over-clever-broken).
  - No heavy prescriptive guide (iter5 proved it backfires).

The one untested, on-thesis lever is model *quality*. Every prior agent used
GPT_5_4_MINI. This agent keeps the winning single-call architecture but gives
it a stronger brain (GPT_5_4). That is neither machinery nor a guide — there
is still exactly one candidate, the model's direct answer — so it cannot
regress via the machinery failure mode, while it directly targets the visible
failure classes a frontier model handles better than mini: current-API usage
(scipy simpson vs deprecated simps), vectorization / "no for loop" idioms, and
careful reading of the expected output.

Cost: a single GPT_5_4 call on DS-1000's short prompts lands well inside the
$0.003 free zone. No reasoning tokens (default "none") keeps cost down and
avoids the over-thinking that produces over-clever answers; a max_tokens cap
bounds the tail.
"""

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

from model_registry import GPT_5_4, GPT_5_4_MINI


# Deliberately tiny framing (contrast: iter5's 5-rule guide, which backfired).
# Two jobs only: (1) bound output to a clean code block => short/cheap output
# and correct extraction; (2) a gentle nudge toward the shortest idiomatic
# call, countering a strong model's tendency to over-engineer. No prescriptive
# per-case rules (those are what misfired in iter5).
PREAMBLE = (
    "You are an expert Python data-science engineer. Write the code that goes "
    "where the solution is requested so the asked-for variable ends up correct. "
    "Prefer the shortest idiomatic library call; do not over-engineer or add "
    "extra steps. Reply with ONLY a single <code> ... </code> block of "
    "executable Python — no prose, no markdown fences, no BEGIN/END markers.\n\n"
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
