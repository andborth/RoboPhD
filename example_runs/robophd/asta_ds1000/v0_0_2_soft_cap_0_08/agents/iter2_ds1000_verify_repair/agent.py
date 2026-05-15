"""DS-1000 solver: specialized prompt + strong reasoning model + free
sandbox verify-and-repair loop.

Pipeline per sample:
  1. Build a DS-1000-specialized prompt (output contract, target-variable
     detection, gotcha checklist).
  2. Generate a candidate with CLAUDE_SONNET_4_6 (reasoning="medium").
  3. If the skeleton setup is locally runnable (no load_data / missing
     symbols), execute skeleton+candidate in the free `python_session`
     sandbox; on a traceback, feed it back for up to 2 repair rounds.
  4. Emit the final `<code>...</code>` block.

Every stage is defensive: any failure falls back to the best candidate so
the agent always emits a wrapped code block.
"""

import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_SONNET_4_6

MODEL = CLAUDE_SONNET_4_6
GEN_CONFIG = GenerateConfig(reasoning_effort="medium", max_tokens=4096)
MAX_REPAIRS = 2

SYSTEM_GUIDANCE = """\
You are an expert Python data-science engineer solving a DS-1000 problem.

You are given a problem with a code skeleton. The skeleton (imports + input
setup) is ALREADY written and will run before your code. You must write ONLY
the new code that produces the requested answer variable(s).

OUTPUT CONTRACT (strict):
- Output exactly one block: <code> ... </code>.
- Inside the tags: executable Python only. No prose, no explanation, no
  markdown ``` fences, no "BEGIN/END SOLUTION" markers.
- Do NOT repeat any code already shown in the skeleton (imports, data setup).
  Assume those names already exist.
- Nothing outside the <code>...</code> tags.

ASSIGN THE EXACT TARGET VARIABLE(S): {target_hint}
Use that exact name. Do not rename it, do not also print it.

THINK CAREFULLY — DS-1000 is full of traps. Before answering, check:
- NEGATIONS / INVERSIONS: read literally. "the values that are 0", "columns
  corresponding to a 0 value", "excluding", "not", "drop", "reverse",
  "opposite", "all but" — these often mean you must invert/complement a mask
  or reverse an order, NOT take the obvious quantity.
- EXACT OUTPUT TYPE & SHAPE: match what is asked precisely. e.g. full 2D
  probability matrix vs a single column; DataFrame vs Series vs ndarray;
  preserve dtype; preserve index, column names and row/column order exactly
  as shown in any "Desired Output" example.
- IDIOM CONSTRAINTS: if the problem says "without a loop", "vectorized",
  "the efficient/clean way", "not one by one", or names a specific library
  function, you MUST use the idiomatic library call — a manual loop /
  reimplementation will be rejected even if the numbers are right.
- WORKED EXAMPLES: if the prompt shows example inputs and a desired output,
  mentally run your solution on them and confirm it reproduces the output
  exactly before finalizing.
- Pick the correct, well-known library function for the task (e.g. the right
  SciPy interpolator, the right sklearn helper) rather than an approximate
  substitute.

Respond with ONLY the <code>...</code> block.
"""


def _all_code_blocks(text: str):
    return re.findall(r"<code>(.*?)</code>", text, re.DOTALL)


def _detect_targets(prompt: str) -> str:
    """Find the answer variable name(s) the skeleton expects."""
    targets = []
    # Pattern: `name = ... # put solution in this variable`
    for m in re.finditer(
        r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\..*?put .*?in this variable",
        prompt,
        re.MULTILINE,
    ):
        targets.append(m.group(1))
    # Pattern: "put score in `b`, put prediction in `c`"
    for m in re.finditer(r"put [^,]*? in `([A-Za-z_]\w*)`", prompt):
        targets.append(m.group(1))
    # Generic "name = ..." right before BEGIN SOLUTION
    if not targets:
        for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt, re.MULTILINE):
            targets.append(m.group(1))
    seen, uniq = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return "the variable requested in the problem (usually `result`)."
    if len(uniq) == 1:
        return f"`{uniq[0]}`."
    return ", ".join(f"`{t}`" for t in uniq) + " (assign each one)."


def _extract_code(text: str) -> str:
    """Pull the actual solution code out of a model response."""
    if not text:
        return ""
    blocks = _all_code_blocks(text)
    if blocks:
        return blocks[-1].strip()
    s = text.strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", s, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Strip stray solution markers.
    s = re.sub(r"^\s*(BEGIN|END)\s+SOLUTION\s*$", "", s, flags=re.MULTILINE)
    return s.strip()


def _setup_block(prompt: str) -> str:
    """First <code>...</code> in the prompt = skeleton imports + data setup."""
    blocks = _all_code_blocks(prompt)
    return blocks[0].strip() if blocks else ""


def _runnable(setup: str) -> bool:
    """Heuristic: can we execute setup+candidate locally as-is?"""
    if not setup:
        return False
    # These need hidden data the sandbox doesn't have.
    if re.search(r"\b(load_data|load_iris|fetch_|read_csv|read_)\w*\(", setup):
        return False
    return True


_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|^\w*Error:|Exception:",
                           re.MULTILINE)


def _has_error(out: str) -> bool:
    return bool(out) and bool(_TRACEBACK_RE.search(out))


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.input or ""
        library = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={library}")

        target_hint = _detect_targets(prompt)
        sys_msg = SYSTEM_GUIDANCE.format(target_hint=target_hint)
        gen_prompt = f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}"

        candidate = ""
        try:
            resp = await MODEL.generate(gen_prompt, config=GEN_CONFIG)
            candidate = _extract_code(resp.completion or "")
        except Exception as e:  # noqa: BLE001
            print(f"  generate failed: {e!r}")

        if not candidate:
            # Last-ditch cheap retry without config.
            try:
                resp = await MODEL.generate(gen_prompt)
                candidate = _extract_code(resp.completion or "")
            except Exception as e:  # noqa: BLE001
                print(f"  fallback generate failed: {e!r}")

        best = candidate

        # ---- Free sandbox verify-and-repair loop ----
        setup = _setup_block(prompt)
        if candidate and _runnable(setup):
            try:
                py = next(
                    (t for t in state.tools
                     if ToolDef(t).name == "python_session"),
                    None,
                )
            except Exception:  # noqa: BLE001
                py = None

            if py is not None:
                cur = candidate
                for attempt in range(MAX_REPAIRS + 1):
                    program = (
                        f"{setup}\n{cur}\n"
                        "print('___DS1000_RAN_OK___')"
                    )
                    try:
                        out = await py(code=program)
                        out = out if isinstance(out, str) else str(out)
                    except Exception as e:  # noqa: BLE001
                        print(f"  sandbox call error: {e!r}")
                        break

                    if "___DS1000_RAN_OK___" in out and not _has_error(out):
                        best = cur
                        print(f"  sandbox OK on attempt {attempt}")
                        break

                    # Don't waste repair rounds on missing-data env limits.
                    if re.search(r"name '(\w+)' is not defined", out) and \
                       "load_data" in out:
                        print("  needs hidden data; skipping repair")
                        break

                    if attempt == MAX_REPAIRS:
                        print("  repairs exhausted")
                        break

                    err = out[-1500:]
                    repair_prompt = (
                        f"{sys_msg}\n\n=== PROBLEM ===\n{prompt}\n\n"
                        f"Your previous solution was:\n<code>\n{cur}\n</code>\n\n"
                        f"Running skeleton + your code produced this error:\n"
                        f"```\n{err}\n```\n\n"
                        "Fix the bug. Respond with ONLY the corrected "
                        "<code>...</code> block (only the new code, not the "
                        "skeleton)."
                    )
                    try:
                        rresp = await MODEL.generate(
                            repair_prompt, config=GEN_CONFIG
                        )
                        fixed = _extract_code(rresp.completion or "")
                        if fixed:
                            cur = fixed
                            print(f"  repaired (attempt {attempt + 1})")
                        else:
                            break
                    except Exception as e:  # noqa: BLE001
                        print(f"  repair generate failed: {e!r}")
                        break

        final = best or candidate or ""
        state.output.completion = f"<code>\n{final}\n</code>"
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
