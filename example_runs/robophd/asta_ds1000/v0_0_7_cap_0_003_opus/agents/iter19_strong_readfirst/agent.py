"""DS-1000 solver: converged single-shot leader + ONE generic careful-reading nudge (iter19).

Delta vs iter18 (the current batch leader, 85%): the entire converged pipeline — model
(GPT_5_4), mini fallback, the full stack of provably-safe deterministic extraction fixes,
the syntax-validity-guarded retry, and the wide token budgets — is kept BYTE-IDENTICAL. The
only change is a single GENERIC sentence added to the tiny preamble instructing the model to
reproduce a SHOWN expected output exactly (column/row order, index, dtype, shape). This
targets the one fail class the deterministic format fixes cannot reach and that recurs across
batches: the model picks the right library operation but the wrong ordering/shape when the
prompt displays the answer (iter18 fails 142, 238, 812 all fit this; 238 recurs in 21 batches).
The sentence is inert on problems that show no expected output, so regression surface is small.

Original iter18 header follows:
DS-1000 solver: converged single-shot leader + syntax retry + truncation hardening (iter18).

Seventeen iterations of evidence converge on ONE architecture and one leader
(iter17_strong_validretry / iter16_strong_safestrip, which BOTH scored 100% on the iter17
batch). This agent keeps that architecture and its full stack of provably-safe,
deterministic, format-only extraction fixes UNTOUCHED, and adds exactly ONE new, on-thesis,
fix-or-no-op refinement — a wider output budget to eliminate silent truncation:

  *** Truncation hardening (the one delta vs iter17) ***
  iter17's syntax-validity-guarded retry only catches truncation that produces a *SyntaxError*.
  But a long answer cut at max_tokens often yields parseable-but-INCOMPLETE code — it passes
  `ast.parse`, so the retry never fires, and the incomplete answer is silently kept → a wrong
  value the scorer marks 0. The root cause is the output-budget cap, not the extraction. So:
    - Raise the base cap 1024 -> 1600. DS-1000 solutions are short, so on the ~99% of
      problems that finish well under the cap this is BYTE-IDENTICAL to iter17 (same sampled
      text, same cost). It only helps the rare legitimately-long answer that iter17 would
      have truncated — shrinking BOTH truncation subclasses (syntax-error AND silent-parseable).
    - On the guarded retry, raise the cap to 2560 so a non-parsing truncation gets a real
      second chance to finish rather than truncating again at the same length.
  This adds NO candidate, NO reasoning, NO value-vote — it only lets a guaranteed/likely-0
  truncated output complete. Cost impact is negligible: the cap bills only tokens actually
  emitted, and measured mean spend ($0.0014) sits deep in the $0.003 free zone; the retry
  itself fires only on the rare guaranteed-0 output.

Settled findings carried forward (do NOT re-litigate — each falsified with data):
  * Machinery (verify / self-consistency / reasoning escalation) is net-negative: wrong
    DS-1000 answers execute cleanly (scorer compares the exact target VALUE —
    dtype/shape/order/index), so crash-verification catches nothing, and same-model
    self-consistency agrees on the WRONG value (errors are systematic misreads).
  * Heavy / prescriptive preambles are net-negative; only iter6's TINY output-contract
    preamble survives.
  * Model: GPT_5_4 (default reasoning "none") is the consistent cross-batch leader and the
    cheapest clean path. Frontier handles (GPT_5_5, Opus 4.8) carry MANDATORY managed
    reasoning billed at the output rate ($25-30/M) -> a few hundred reasoning tokens blow the
    $0.003 free zone -> priced out. Sonnet 4.6 (same tier, default "none") is an unvalidated
    coin flip against the proven leader -> rejected.

The format-only fixes (touch only the FORM of the emitted code, never its reasoning and
never a choice among candidates — all provably fix-or-no-op):

  1. html.unescape — a strong model HTML-escapes `<` as `&lt;` inside <code> -> SyntaxError
     with otherwise-correct logic (confirmed 962).
  2. Function-body reindent — insert-inside-`def` problems show an INDENTED `### BEGIN
     SOLUTION` marker; GPT_5_4 stochastically drops the leading indent -> IndentationError.
     Read the marker indent and pad. Fix-or-no-op (never fires on indent-0 problems).
  3. Target-variable assignment (single-line). Skeleton shows `TARGET = ...`; a bare
     expression -> NameError (confirmed 451). Wrap it, guarded by a depth-aware top-level-`=`
     scan (a `=` inside a kwarg/dict/subscript does not block).
  4. Target-variable assignment (multi-line). Target bound NOWHERE + final unindented line
     is a bare value-expr -> wrap it (guaranteed NameError as-is -> fix-or-no-op).
  5. Truncation / stray-tag safety. A long answer cut at max_tokens emits a lone `<code>`;
     take everything after it (symmetrically, before a lone closing tag).
  6. Trailing-driver strip (function-style only): drop trailing lines structurally below the
     def body — appended driver calls, never part of a correct body (fixes 365/910). Strips
     only when a non-comment line already reaches `target_indent` AND the trailing line sits
     below it -> provably cannot cut a valid body line.

  7. Syntax-validity-guarded retry. After the full pipeline, if the emitted code is EMPTY or
     does NOT parse (function-style bodies parsed wrapped in a probe `def`), the answer is a
     GUARANTEED 0 — resample ONCE and keep whichever candidate parses. For any well-formed
     primary output the retry NEVER fires -> byte-identical to iter16 on ~95% of problems.
     Checks only PARSEABILITY (an orthogonal guaranteed-0 class), never executes/votes on
     VALUES.

Mini fallback on provider error/empty so a hiccup never hard-zeros a problem.
"""

import ast
import html
import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

from model_registry import GPT_5_4, GPT_5_4_MINI


# iter6's tiny, proven preamble + ONE generic careful-reading nudge (iter19). Jobs:
# (1) bound the output to a single clean <code> block; (2) a gentle nudge toward the
# shortest idiomatic call, countering a strong model's tendency to over-engineer;
# (3) NEW: when the prompt SHOWS a sample of the expected output, match it exactly.
#
# The (3) sentence targets the dominant CROSS-BATCH-RECURRING fail class that the
# format-fix well cannot touch: problems that display the expected table/array/values
# and the model produces the right operation but the wrong COLUMN order / ROW order /
# index / dtype / shape (iter18 fails 142 col-reversal, 238 row+date-order, 812 shape;
# 238 alone recurs in 21 batches and perennially fails). It is GENERIC guidance, not a
# per-case rule (heavy prescriptive preambles were falsified), and it is inert on the
# many problems that show no expected output — so its regression surface is limited to
# problems where "match the shown output exactly" is precisely the intent anyway.
PREAMBLE = (
    "You are an expert Python data-science engineer. Write the code that goes "
    "where the solution is requested so the asked-for variable ends up correct. "
    "Prefer the shortest idiomatic library call; do not over-engineer or add "
    "extra steps. If the problem SHOWS a sample of the expected output (a table, "
    "array, Series, or explicit values), make your result reproduce it exactly — "
    "same column order, row order, index, dtype and shape — even where that "
    "contradicts the prose or your first instinct. Reply with ONLY a single "
    "<code> ... </code> block of executable Python — no prose, no markdown fences, "
    "no BEGIN/END markers.\n\n"
    "Problem:\n"
)

# Output budget caps (cost safety). DS-1000 solutions are short, so on nearly every problem
# the model finishes well under MAX_TOKENS and this is identical to a smaller cap; the wider
# budget only matters for the rare long answer that would otherwise truncate to a silent 0.
# The guarded retry gets a still-wider budget so a truncation-caused non-parse can finish.
MAX_TOKENS = 1600
RETRY_MAX_TOKENS = 2560

_MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)(###\s*)?(BEGIN SOLUTION|SOLUTION START)\s*$"
)
_TARGET_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.")
_STMT_KW = re.compile(
    r"(return|import|from|def|class|for|while|if|elif|else|with|try|except|"
    r"finally|raise|assert|print|del|global|nonlocal|yield|pass|break|continue|@)"
    r"(\s|\(|:|$)"
)


def _indent_of(line: str) -> int:
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
    """Answer variable from the skeleton's `NAME = ...` placeholder (LAST match)."""
    name = None
    for line in (problem_input or "").splitlines():
        m = _TARGET_RE.match(line)
        if m:
            name = m.group(1)
    return name


def _strip_trailing_drivers(code: str, target_indent: int) -> str:
    """Function-style only: drop trailing lines that sit structurally below the def body.

    Strip a trailing line ONLY when (a) some NON-COMMENT emitted line already reaches
    `target_indent` (body genuinely indented — not a whole-body under-indent for _reindent)
    AND (b) that trailing line sits BELOW `target_indent` (structurally outside the def).
    A correct function body never has a trailing line below `target_indent`, so this can
    only fix or no-op.
    """
    if target_indent <= 0:
        return code
    lines = code.split("\n")
    content = [(i, _indent_of(ln)) for i, ln in enumerate(lines) if ln.strip()]
    if len(content) < 2:
        return code
    reaches = any(
        ind >= target_indent
        for i, ind in content
        if not lines[i].lstrip().startswith("#")
    )
    if not reaches:
        return code
    cut = None
    for i, ind in reversed(content):
        if ind < target_indent:
            cut = i
        else:
            break
    if cut is None:
        return code
    kept = lines[:cut]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) if kept else code


def _reindent(code: str, target: int) -> str:
    """Ensure the block's minimum indentation is at least `target` spaces (only ADDS)."""
    lines = code.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return code
    if re.match(r"\s*(def |class |import |from |@)", nonempty[0]):
        return code
    cur_min = min(len(ln) - len(ln.lstrip(" ")) for ln in nonempty)
    if cur_min >= target:
        return code
    pad = " " * (target - cur_min)
    return "\n".join(pad + ln if ln.strip() else ln for ln in lines)


def _has_top_level_assign(line: str) -> bool:
    """True if `line` has a `=` assignment at statement level (bracket/string depth 0)."""
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
            if prev not in "=!<>:" and nxt != "=":
                return True
        i += 1
    return False


def _wrappable_expr(line: str, target: str) -> bool:
    if _STMT_KW.match(line):
        return False
    if re.match(rf"{re.escape(target)}\b", line):
        return False
    if _has_top_level_assign(line):
        return False
    return True


def _maybe_wrap_assignment(code: str, target: str | None) -> str:
    """Prepend `target = ` when the answer never binds the target variable (fix-or-no-op)."""
    if not target:
        return code
    code_lines = [
        ln for ln in code.split("\n") if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not code_lines:
        return code

    if len(code_lines) == 1:
        line = code_lines[0].strip()
        if not _wrappable_expr(line, target):
            return code
        return f"{target} = {line}"

    if re.search(rf"\b{re.escape(target)}\b", code):
        return code
    last_line = code_lines[-1]
    if last_line[:1].isspace():
        return code
    if not _wrappable_expr(last_line.strip(), target):
        return code
    out = code.split("\n")
    for i in range(len(out) - 1, -1, -1):
        if out[i].strip() and not out[i].lstrip().startswith("#"):
            out[i] = f"{target} = {out[i].strip()}"
            break
    return "\n".join(out)


def _build_inner(text: str, problem_input: str = "") -> str:
    """Return the INNER code (no <code> tags) after the full deterministic fix pipeline."""
    s = (text or "").strip()
    inner = None

    if "<code>" in s and "</code>" in s:
        inner = s.split("<code>", 1)[1].split("</code>", 1)[0]
    elif "<code>" in s:
        inner = s.split("<code>", 1)[1]
    elif "</code>" in s:
        inner = s.split("</code>", 1)[0]
    elif "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if "\n" in block:
                first, rest = block.split("\n", 1)
                if first.strip().isalpha():
                    block = rest
            inner = block

    if inner is None:
        inner = s

    inner = html.unescape(inner.strip("\n"))

    if not inner.strip():
        # Degenerate case: unescape the whole response as a last resort.
        return html.unescape(s)

    target_indent = _required_indent(problem_input)
    if target_indent == 0:
        first = next(ln for ln in inner.split("\n") if ln.strip())
        if not first[:1].isspace() and re.match(r"return(\s|$)", first):
            target_indent = 4
    if target_indent > 0:
        inner = _strip_trailing_drivers(inner, target_indent)
        inner = _reindent(inner, target_indent)
    else:
        inner = _maybe_wrap_assignment(inner, _target_var(problem_input))

    return inner


def _wrap(inner: str) -> str:
    return f"<code>\n{inner}\n</code>"


def _extract_code(text: str, problem_input: str = "") -> str:
    """Emit a single clean `<code>...</code>` block for the scorer."""
    return _wrap(_build_inner(text, problem_input))


def _parses(inner: str, problem_input: str = "") -> bool:
    """True if `inner` is syntactically valid Python in the way it will be concatenated.

    Function-style bodies (an indented `return ...`) are parsed wrapped in a probe `def`
    so a CORRECT indented body is never falsely flagged. Top-level answers parse as-is
    (undefined setup names are not a syntax error). Returns False on empty/garbled/truncated
    output — the guaranteed-0 class the retry targets.
    """
    if not inner.strip():
        return False
    src = inner
    if _required_indent(problem_input) > 0 or inner[:1].isspace():
        # Body will be concatenated inside a def; mirror that so an indented body is valid.
        src = "def _probe():\n" + inner
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False
    except Exception:
        # Any non-syntax parser hiccup: don't punish a candidate we can't disprove.
        return True


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={lib}")

        prompt = PREAMBLE + state.input
        cfg = GenerateConfig(max_tokens=MAX_TOKENS)

        # --- Primary: single strong-model shot (the settled, cross-batch-winning recipe). ---
        completion = ""
        primary_errored = False
        try:
            resp = await GPT_5_4.generate(prompt, config=cfg)
            completion = resp.completion or ""
        except Exception as e:  # provider hiccup: never hard-0 the problem
            primary_errored = True
            print(f"  GPT_5_4 error ({e!r}); falling back to mini")

        # Mini fallback ONLY when the primary call errored or returned nothing (unchanged
        # from the settled recipe — keeps behavior identical on normal outputs).
        if primary_errored or not completion.strip():
            try:
                resp = await GPT_5_4_MINI.generate(prompt, config=cfg)
                completion = resp.completion or ""
            except Exception as e:
                print(f"  mini fallback error ({e!r})")

        inner = _build_inner(completion, state.input)

        # --- Syntax-validity-guarded retry (fires only on guaranteed-0 output). -----------
        # For any well-formed primary output this branch is skipped entirely → byte-identical
        # to the settled recipe. A second candidate can only match-or-improve a guaranteed-0.
        # The retry uses a WIDER token budget so a truncation-caused non-parse can finish.
        if not _parses(inner, state.input):
            print("  primary output does not parse; one guarded resample")
            retry_cfg = GenerateConfig(max_tokens=RETRY_MAX_TOKENS)
            # Prefer a fresh strong sample; use mini if the strong call already failed.
            retry_model = GPT_5_4_MINI if primary_errored else GPT_5_4
            try:
                resp = await retry_model.generate(prompt, config=retry_cfg)
                retry_completion = resp.completion or ""
            except Exception as e:
                retry_completion = ""
                print(f"  retry error ({e!r})")
            if retry_completion.strip():
                retry_inner = _build_inner(retry_completion, state.input)
                # Keep the retry only if IT parses (never trade a parsing answer for a
                # non-parsing one; if neither parses we keep the original untouched).
                if _parses(retry_inner, state.input):
                    inner = retry_inner

        state.output.completion = _wrap(inner)
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
