"""DS-1000 solver: strong one-shot + hardened extraction (iter10_strong_reindent).

Nine iterations of evidence converge on ONE architecture and one leader, and this
agent keeps that architecture untouched while adding only deterministic, format-only
extraction hardening (the empirically "pure-upside" lever class).

Settled findings (do not re-litigate):
  * Machinery (verify / self-consistency / reasoning escalation) is net-negative:
    wrong DS-1000 answers execute cleanly (the scorer compares the exact target
    VALUE — dtype/shape/order/index), so crash-verification catches nothing, and the
    marginal vote/verify loop swaps correct-simple for over-clever-broken.
  * Heavy / prescriptive preambles are net-negative (iter5's 5-rule guide dropped
    mini to 40%; iter7's contract preamble caused IndentationError + over-condensed
    one-liners). Only iter6's genuinely TINY output-contract preamble survives.

The most CONSISTENT top performer across every recent batch is `iter6_strong_oneshot`
(80% / 75% / 90%): a single GPT_5_4 call, no reasoning, no verification, iter6's tiny
preamble. This agent = iter6's recipe VERBATIM (same model, same preamble, same cap,
same mini fallback) + two strictly-additive, deterministic extraction fixes that touch
only the *form* of the emitted code, never the model's chosen answer:

  1. html.unescape — a confirmed loss class (962): a strong model HTML-escapes `<` as
     `&lt;` inside the <code> tags -> SyntaxError with otherwise-correct logic.

  2. Function-body reindent — the single most common removable loss class. DS-1000
     function-style problems insert the answer INSIDE a `def f(...):` body (the prompt
     shows an INDENTED `### BEGIN SOLUTION` marker). GPT_5_4 stochastically emits the
     body with or without that leading indentation; when it drops it, the result is a
     bare top-level `return ...` -> IndentationError with correct logic (confirmed 397;
     same class as 372/910 across prior batches). We read the required indentation
     straight from the prompt's marker line and re-indent the emitted block to match.
     This can ONLY fix or no-op: on a top-level (indent-0) problem it never fires; on a
     function problem it only adds indentation the answer already required, so a
     correctly-indented answer is left untouched and an under-indented one is repaired.

Neither fix inspects or changes the model's reasoning or picks among candidates — there
is still exactly ONE candidate (the model's direct answer), so the falsified
machinery/guide failure modes cannot recur. Both are the same category as iter9's
unescape, which the record labels pure-upside.

Cost: one GPT_5_4 call on DS-1000's short prompts ~ $0.0015-0.002/problem, comfortably
inside the $0.003 free zone. Default reasoning ("none") keeps cost down and avoids the
over-thinking that produces over-clever answers; max_tokens bounds the tail. Mini
fallback on provider error/empty so a hiccup never hard-zeros a problem.
"""

import html
import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

from model_registry import GPT_5_4, GPT_5_4_MINI


# iter6's tiny, proven preamble — VERBATIM. Two jobs only: (1) bound the output to a
# single clean <code> block (short/cheap output + correct extraction); (2) a gentle
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

# Lines that mark the insertion point in the DS-1000 skeleton. The LAST occurrence's
# leading whitespace tells us the indentation the answer must have (function-style
# problems put this marker inside the def body, indented).
_MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)(###\s*)?(BEGIN SOLUTION|SOLUTION START)\s*$"
)


def _required_indent(problem_input: str) -> int:
    """Leading-space count of the insertion marker in the skeleton (0 => top level)."""
    indent = 0
    for line in (problem_input or "").splitlines():
        m = _MARKER_RE.match(line)
        if m:
            indent = len(m.group("indent").replace("\t", "    "))
    return indent


def _reindent(code: str, target: int) -> str:
    """Ensure the block's minimum indentation is at least `target` spaces.

    Only ever ADDS indentation (never removes), and only when the block is currently
    under-indented — so a correctly-indented function body is left untouched and a
    top-level answer (target 0) is never modified. Relative indentation is preserved.
    """
    lines = code.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return code
    # Don't touch a block that starts by (re)defining the wrapper itself.
    if re.match(r"\s*(def |class |import |from |@)", nonempty[0]):
        return code
    cur_min = min(len(ln) - len(ln.lstrip(" ")) for ln in nonempty)
    if cur_min >= target:
        return code
    pad = " " * (target - cur_min)
    return "\n".join(pad + ln if ln.strip() else ln for ln in lines)


def _extract_code(text: str, problem_input: str = "") -> str:
    """Emit a single clean `<code>...</code>` block for the scorer.

    Preserves the model's indentation, html.unescapes entity-escaped characters, and
    re-indents an under-indented function body to the skeleton's insertion indentation.
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

    # Determine the required insertion indentation from the prompt.
    target = _required_indent(problem_input)
    # Safety net: a bare top-level `return` is never valid Python, so if the answer
    # looks like an unindented function body even when the marker wasn't detected,
    # give it a standard 4-space body indent.
    if target == 0:
        first = next(ln for ln in inner.split("\n") if ln.strip())
        if not first[:1].isspace() and re.match(r"return(\s|$)", first):
            target = 4
    if target > 0:
        inner = _reindent(inner, target)

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

        state.output.completion = _extract_code(completion, state.input)
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
