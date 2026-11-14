"""DS-1000 solver: strong one-shot + hardened extraction (iter9_strong_oneshot_unescape).

Eight iterations of evidence converge on one architecture and one leader:

  * Machinery (verify / self-consistency / reasoning escalation) is net-negative:
    wrong DS-1000 answers execute cleanly (the scorer compares the exact target
    VALUE, incl. dtype/shape/order/index), so crash-verification catches nothing,
    and the marginal vote/verify loop swaps correct-simple for over-clever-broken.
  * Heavy / prescriptive preambles are net-negative (iter5's 5-rule guide dropped
    mini to 40%; iter7's contract preamble caused IndentationError on function
    problems and over-condensed one-liners).

The most consistent TOP performer across the two most recent batches is
`iter6_strong_oneshot` (80% then 75%): a single GPT_5_4 call, no reasoning, no
verification, and a genuinely TINY preamble. That tiny preamble is not the
falsified "heavy guide" lever — it carries zero per-case reasoning rules, only a
clean output contract plus a gentle anti-over-engineering nudge.

This agent = iter6's winning recipe VERBATIM (same model, same preamble, same cap)
+ the ONE strictly-additive fix it lacked: `html.unescape`-ing the extracted code
(a prior batch confirmed a strong model HTML-escaping `<` as `&lt;` inside the
`<code>` tags -> SyntaxError with otherwise-correct logic), with indentation
preserved for function bodies. The only delta vs the batch leader is a format-only
safety net, so it can only match-or-exceed iter6 while removing a real loss class.

Cost: one GPT_5_4 call on DS-1000's short prompts ~ $0.002/problem, comfortably
inside the $0.003 free zone. Default reasoning ("none") keeps cost down and avoids
the over-thinking that produces over-clever answers; max_tokens bounds the tail.
Mini fallback on provider error/empty so a hiccup never hard-zeros a problem.
"""

import html

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

from model_registry import GPT_5_4, GPT_5_4_MINI


# iter6's tiny, proven preamble — VERBATIM. Two jobs only: (1) bound the output to
# a single clean <code> block (short/cheap output + correct extraction); (2) a gentle
# nudge toward the shortest idiomatic call, countering a strong model's tendency to
# over-engineer. No prescriptive per-case rules (those are what misfired in iter5/iter7).
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
    """Emit a single clean `<code>...</code>` block for the scorer.

    Preserves the model's indentation (function-style problems need the body
    indented) and `html.unescape`s the code so an entity-escaped `<`/`>`/`&`
    (which some models emit inside <code> tags) does not become a SyntaxError.
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

    # Fall back to the raw response (the preamble asked for only a code block).
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
