"""DS-1000 solver: raw strong one-shot (iter8_raw_strong).

Settled architecture (8 iterations of evidence): a *plain raw one-shot* wins, and every
added mechanism is net-negative. Verification / self-consistency catches nothing (wrong
DS-1000 answers run clean) and swaps correct-simple for over-clever-broken. Heavy guides
backfire (iter5's 5-rule guide collapsed mini to 40%). So this agent adds neither: the
output is exactly one candidate, the model's direct answer, from the *verbatim* problem
prompt with NO preamble — identical framing to the winning seed.

The single change vs the seed is the brain: GPT_5_4 instead of GPT_5_4_MINI. Every prior
strong-model agent (iter6, iter7) was handicapped by a preamble, and tracing each of their
losses showed they were all preamble/encoding artifacts, not reasoning errors:

  * 451 — preamble reframing made the model emit a bare `np.zeros(...)` instead of the
    required `arr = np.zeros(...)`. A raw continuation of the `arr = ...` skeleton fixes it.
  * 397 / 910 — iter7's "write only the function body" dropped indentation -> IndentationError.
    A raw continuation of the indented `def f(...): ### BEGIN SOLUTION` skeleton keeps it.
  * 808 — the "prefer the shortest idiomatic call" push produced an over-condensed broken
    one-liner. Removing the push avoids it.
  * 962 — the model HTML-escaped `<` as `&lt;` inside the <code> tags -> SyntaxError (confirmed
    to reach the executor). `html.unescape` on the extracted code fixes this class.

Removing the preamble removes 451/397/808/910; unescape removes 962. What remains is the
strong model's genuine reasoning edge (803: correct `cdist(centroids,data).argmin(axis=1)`
where mini hallucinated a `labels` var; plus prior 18/999/372/444). On the iter7 batch this
counterfactually recovers 451, 962, 803 -> 18/20, beating the seed's 17/20.

Cost: one GPT_5_4 call, default reasoning ("none"), ~ $0.0016/problem -- inside the $0.003
free zone. Mini fallback on provider error/empty so a hiccup never hard-zeros a problem.
"""

import html

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

from model_registry import GPT_5_4, GPT_5_4_MINI


# Output budget cap (cost safety only). DS-1000 solutions are short; billed on tokens
# actually used, so this is generous — truncation would hard-fail, extra headroom is free.
MAX_TOKENS = 2048


def _extract_code(text: str) -> str:
    """Emit a single clean `<code>...</code>` block for the scorer.

    Preserves the model's indentation (function-style problems need the body indented)
    and `html.unescape`s the code so an entity-escaped `<`/`>`/`&` (which some models emit
    inside <code> tags) does not become a SyntaxError.
    """
    s = (text or "").strip()
    inner = None

    # Prefer content inside <code>...</code> if present (take the first block).
    if "<code>" in s and "</code>" in s:
        inner = s.split("<code>", 1)[1].split("</code>", 1)[0]

    # Otherwise strip a markdown fence if the model used one.
    elif "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if "\n" in block:
                first, rest = block.split("\n", 1)
                if first.strip().isalpha():  # drop a leading language tag (e.g. "python")
                    block = rest
            inner = block

    # Fall back to the raw response (the prompt asked for only a code block).
    if inner is None:
        inner = s

    # Strip only newlines (keep leading spaces = indentation), then unescape entities.
    inner = html.unescape(inner.strip("\n"))

    if not inner.strip():
        # Degenerate case: unescape the whole response as a last resort.
        return f"<code>\n{html.unescape(s)}\n</code>"

    return f"<code>\n{inner}\n</code>"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={lib}")

        # Verbatim problem prompt — NO preamble (identical framing to the winning seed).
        prompt = state.input
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
