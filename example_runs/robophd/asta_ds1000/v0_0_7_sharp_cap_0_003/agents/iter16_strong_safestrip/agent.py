"""DS-1000 solver: strong one-shot + hardened extraction (iter16_strong_safestrip).

Fifteen iterations of evidence converge on ONE architecture and one leader. This agent
keeps that architecture untouched and carries only deterministic, format-only extraction
hardening — the empirically "pure-upside" lever class (same category as html.unescape,
function-body reindent, target-variable assignment, and truncation safety).

Settled findings (do NOT re-litigate — each was falsified with data across 15 iters):
  * Machinery (verify / self-consistency / reasoning escalation) is net-negative:
    wrong DS-1000 answers execute cleanly (the scorer compares the exact target VALUE —
    dtype/shape/order/index), so crash-verification catches nothing, and the marginal
    vote/verify loop swaps correct-simple for over-clever-broken. Two-sample
    self-consistency does not help because DS-1000 errors are *systematic* misreads
    (both samples share the same misconception and agree on the WRONG value).
  * Heavy / prescriptive preambles are net-negative (iter5's 5-rule guide dropped a
    mini model to 40%; iter7's contract preamble caused IndentationError + over-condensed
    one-liners). Only iter6's genuinely TINY output-contract preamble survives.
  * Model quality: GPT_5_4 (this recipe) is the most consistent leader across recent
    batches. No reasoning_effort — its default "none" keeps cost down AND avoids the
    over-thinking that produces over-clever, wrong answers.

The single-GPT_5_4 + iter6-tiny-preamble recipe, plus a stack of strictly-additive
deterministic extraction fixes, has led every recent batch. There is exactly ONE
candidate (the model's direct answer), so the falsified machinery/guide failure modes
cannot recur.

WHAT THIS ITERATION CHANGES vs iter15. iter15 carried six format fixes; iter12 (which
carried the first four and NOT the driver-strip) has been the most consistent cross-batch
winner, and across the two most recent batches the agent carrying the driver-strip scored
LOWEST of the group. Auditing the six fixes shows five are *provably* fix-or-no-op (they
can only alter code that was already guaranteed to fail), but the sixth — the trailing
driver-strip — used a `body_base = indent-of-first-line` criterion that has a
*constructible* misfire: if the model emits a correct body whose FIRST line is more
indented than a later line (or whose first non-blank line is a comment at a deeper
indent), `body_base` is set too high and a valid trailing body line can be cut. That is
the one plausible mechanism by which "more fixes → lower score" could be real rather than
noise. So this iteration REWRITES the driver-strip to a criterion that is provably
fix-or-no-op, keeping the real 365/910 class fixed while removing the tail risk:

  A line inside a `def` body MUST be indented at least to the skeleton's insertion
  indent (`target_indent`, read from the `### BEGIN SOLUTION` marker). We therefore strip
  a trailing line ONLY when (a) at least one NON-COMMENT emitted line already reaches
  `target_indent` — proving the body is genuinely indented and this is not a wholly
  under-indented body that `reindent` should repair — AND (b) the trailing line sits
  BELOW `target_indent`, i.e. structurally outside the def. A correct function body never
  has a trailing line below `target_indent`, so this can only fix or no-op. (Comments are
  excluded when deciding whether the body reaches `target_indent`, so a deeper-indented
  leading comment above an under-indented body no longer defeats the guard.)

The format-only fixes (touch only the *form* of the emitted code, never its reasoning
and never a choice among candidates):

  1. html.unescape — a strong model HTML-escapes `<` as `&lt;` inside the <code> tags
     -> SyntaxError with otherwise-correct logic (confirmed 962).

  2. Function-body reindent — function-style problems insert the answer INSIDE a
     `def f(...):` body (the prompt shows an INDENTED `### BEGIN SOLUTION` marker).
     GPT_5_4 stochastically emits the body without that leading indent -> a bare
     top-level `return ...` -> IndentationError with correct logic. We read the required
     indent from the marker and pad. Fix-or-no-op (never fires on indent-0 problems).

  3. Target-variable assignment (single-line). Top-level problems show a placeholder
     `TARGET = ...  # put solution in this variable`; GPT_5_4 stochastically emits a bare
     EXPRESSION -> `NameError: TARGET` (confirmed 451). We wrap it, guarded by a
     depth-aware top-level-`=` scan (a `=` inside a kwarg/dict/subscript does not block).

  4. Target-variable assignment (multi-line). Same class over lines: the target is bound
     NOWHERE in the block and the final unindented line is a bare value-expression ->
     wrap it (guaranteed NameError as-is, so fix-or-no-op).

  5. Truncation / stray-tag safety. A long answer cut at max_tokens emits an opening
     `<code>` with no close; we take everything after it (and, symmetrically, before a
     lone closing tag) instead of re-wrapping the whole leaked string.

  6. Trailing-driver strip (HARDENED this iteration; see above). Function-style only:
     drop trailing lines that sit structurally below the def body — appended driver calls
     to the just-defined function, never part of a correct body (fixes 365/910).

Consensus / split failures on recent batches stay unaddressed because none is a safe
lever: 238 (multi-step merge + exact date order), 812/77 (groupby/value traps), 420
(model invents extra required function params where the reference closes over setup
globals — a careful-reading trap; a prescriptive preamble rule for it was the falsified
iter5 lever), 432/398 (numpy.ma / ewm-alpha API), 706/667 (TensorFlow protobuf ENV
crashes that raise before the answer is compared — unwinnable infra). Chasing any of
these needs machinery or a guide, both repeatedly falsified.

Cost: one GPT_5_4 call on DS-1000's short prompts ~ $0.0013-0.0018/problem (measured),
well inside the $0.003 free zone. max_tokens bounds the tail. Mini fallback on provider
error/empty so a hiccup never hard-zeros a problem.
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

# The DS-1000 target-variable placeholder, e.g. `result = ... # put solution ...` or
# `arr = ...`. The literal `= ...` (ellipsis) RHS is the distinctive placeholder marker,
# so this rarely collides with prose. The LAST such line names the answer variable.
_TARGET_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.")

# Statement keywords: if a single emitted line starts with one of these it is already a
# complete statement, not a bare value-expression, so we must NOT wrap it as an rvalue.
_STMT_KW = re.compile(
    r"(return|import|from|def|class|for|while|if|elif|else|with|try|except|"
    r"finally|raise|assert|print|del|global|nonlocal|yield|pass|break|continue|@)"
    r"(\s|\(|:|$)"
)


def _indent_of(line: str) -> int:
    """Leading-space count of a line (tabs expanded to 4)."""
    expanded = line.replace("\t", "    ")
    return len(expanded) - len(expanded.lstrip(" "))


def _required_indent(problem_input: str) -> int:
    """Leading-space count of the insertion marker in the skeleton (0 => top level)."""
    indent = 0
    for line in (problem_input or "").splitlines():
        m = _MARKER_RE.match(line)
        if m:
            indent = len(m.group("indent").replace("\t", "    "))
    return indent


def _target_var(problem_input: str) -> str | None:
    """Name of the answer variable from the skeleton's `NAME = ...` placeholder.

    Returns the LAST match (the real placeholder sits just before BEGIN SOLUTION),
    or None for function-style / plotting problems that have no such placeholder.
    """
    name = None
    for line in (problem_input or "").splitlines():
        m = _TARGET_RE.match(line)
        if m:
            name = m.group(1)
    return name


def _strip_trailing_drivers(code: str, target_indent: int) -> str:
    """Function-style only: drop trailing lines that sit structurally below the def body.

    In function-insertion problems (target_indent > 0) the answer belongs INSIDE a `def`,
    so every real body line is indented at least to `target_indent` (the skeleton's
    `### BEGIN SOLUTION` marker indent). GPT_5_4 sometimes appends a driver line that calls
    the just-defined function (e.g. `transformed_df = Transform(df)`, echoing the
    skeleton's commented hint), dedented to top level below the body -> the concatenated
    program hits an IndentationError or runs a spurious driver (NameError on hidden-only
    vars / infinite recursion).

    HARDENED (provably fix-or-no-op). We strip a trailing line ONLY when:
      (a) at least one NON-COMMENT emitted line already reaches `target_indent` — proving
          the body is genuinely indented, so this is not a wholly under-indented body that
          `_reindent` should repair; AND
      (b) that trailing line sits BELOW `target_indent` — structurally outside the def.
    A correct function body never has a trailing line below `target_indent`, so this can
    only turn a guaranteed-failing block into a valid body; it never touches a passing
    answer. (Using `target_indent` rather than the first line's indent removes the
    misfire where a correct body's first line is deeper than a later line, or where a
    deeper-indented leading comment sits above the body.)
    """
    if target_indent <= 0:
        return code
    lines = code.split("\n")
    content = [(i, _indent_of(ln)) for i, ln in enumerate(lines) if ln.strip()]
    if len(content) < 2:
        return code
    # Does any NON-COMMENT line reach the required body indent? If not, the whole body is
    # under-indented (a job for _reindent, not for stripping) — leave it alone.
    reaches = any(
        ind >= target_indent
        for i, ind in content
        if not lines[i].lstrip().startswith("#")
    )
    if not reaches:
        return code
    # Mark the contiguous run of trailing lines that sit below the body indent.
    cut = None
    for i, ind in reversed(content):
        if ind < target_indent:
            cut = i
        else:
            break
    if cut is None:
        return code
    kept = lines[:cut]
    while kept and not kept[-1].strip():  # drop now-trailing blank lines
        kept.pop()
    return "\n".join(kept) if kept else code


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


def _has_top_level_assign(line: str) -> bool:
    """True if `line` contains a `=` assignment at statement level (bracket depth 0).

    A `=` inside (), [], {} or a string literal is a keyword argument / dict entry /
    subscript, NOT a statement-level assignment, so it does not count. `==`, `!=`,
    `<=`, `>=`, and the walrus `:=` are comparisons/expressions, not assignments.
    """
    depth = 0
    in_str = None
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if in_str is not None:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in "\"'":
            in_str = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "=" and depth == 0:
            prev = line[i - 1] if i > 0 else ""
            nxt = line[i + 1] if i + 1 < n else ""
            # Skip ==, !=, <=, >=, := (comparison / walrus, not a plain assignment)
            # and the second half of any such operator.
            if prev not in "=!<>:" and nxt != "=":
                return True
        i += 1
    return False


def _wrappable_expr(line: str, target: str) -> bool:
    """True if `line` (stripped) is a bare value-expression we may prepend `target = `.

    It must not already start with `target`, not begin with a statement keyword, and
    contain no statement-level assignment of its own.
    """
    if _STMT_KW.match(line):
        return False
    if re.match(rf"{re.escape(target)}\b", line):
        return False
    if _has_top_level_assign(line):
        return False
    return True


def _maybe_wrap_assignment(code: str, target: str | None) -> str:
    """Prepend `target = ` when the answer never binds the target variable.

    Two guarded, fix-or-no-op cases:

    Single-line (iter12): exactly one non-blank/non-comment code line that is a bare
    value-expression with no statement-level assignment -> wrap it.

    Multi-line (iter13): the target variable is bound NOWHERE in the block (word-
    boundary search) and the final, unindented code line is a bare value-expression
    -> wrap that final line. Requiring the target to be entirely absent guarantees the
    block would NameError as-is (a passing answer must bind the target), so wrapping
    can only fix or no-op and never overwrites an existing assignment.

    Never fires on function-style problems (they have no `NAME = ...` placeholder, so
    `target` is None).
    """
    if not target:
        return code
    code_lines = [
        ln for ln in code.split("\n") if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not code_lines:
        return code

    # --- Single-line case (iter12 behavior, a strict subset of the guards below). ---
    if len(code_lines) == 1:
        line = code_lines[0].strip()
        if not _wrappable_expr(line, target):
            return code
        return f"{target} = {line}"

    # --- Multi-line case (iter13 widening). ---
    # Only when the target is bound nowhere in the block (guaranteed NameError as-is).
    if re.search(rf"\b{re.escape(target)}\b", code):
        return code
    last_line = code_lines[-1]
    # Must be a top-level statement, not inside a loop/if body.
    if last_line[:1].isspace():
        return code
    if not _wrappable_expr(last_line.strip(), target):
        return code
    # Wrap the final non-blank/non-comment line in place, preserving other lines.
    out = code.split("\n")
    for i in range(len(out) - 1, -1, -1):
        if out[i].strip() and not out[i].lstrip().startswith("#"):
            out[i] = f"{target} = {out[i].strip()}"
            break
    return "\n".join(out)


def _extract_code(text: str, problem_input: str = "") -> str:
    """Emit a single clean `<code>...</code>` block for the scorer.

    Preserves the model's indentation, html.unescapes entity-escaped characters,
    strips appended function-driver lines, re-indents an under-indented function body to
    the skeleton's insertion indentation, and wraps a bare value-expression in the
    skeleton's target-variable assignment.
    """
    s = (text or "").strip()
    inner = None

    # Prefer content inside <code>...</code> if present (take the first block).
    if "<code>" in s and "</code>" in s:
        inner = s.split("<code>", 1)[1].split("</code>", 1)[0]

    # Tolerate a truncated / stray tag. A long answer cut off at max_tokens emits an
    # opening `<code>` with no close; take everything after it. A lone closing tag ->
    # take everything before it. Fix-or-no-op (well-formed answers hit the branch above
    # first and are untouched).
    elif "<code>" in s:
        inner = s.split("<code>", 1)[1]
    elif "</code>" in s:
        inner = s.split("</code>", 1)[0]

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
    target_indent = _required_indent(problem_input)
    # Safety net: a bare top-level `return` is never valid Python, so if the answer
    # looks like an unindented function body even when the marker wasn't detected,
    # give it a standard 4-space body indent.
    if target_indent == 0:
        first = next(ln for ln in inner.split("\n") if ln.strip())
        if not first[:1].isspace() and re.match(r"return(\s|$)", first):
            target_indent = 4
    if target_indent > 0:
        # Drop appended driver lines BEFORE reindenting (they distort cur_min).
        inner = _strip_trailing_drivers(inner, target_indent)
        inner = _reindent(inner, target_indent)
    else:
        # Only top-level problems carry a `NAME = ...` placeholder; wrap a bare
        # expression into the required assignment (fix-or-no-op, guarded).
        inner = _maybe_wrap_assignment(inner, _target_var(problem_input))

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
