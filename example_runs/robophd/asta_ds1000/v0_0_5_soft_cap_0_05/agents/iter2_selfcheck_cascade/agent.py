"""DS-1000 solver: self-verifying cascade.

Strategy (see reasoning.md):
  1. Generate a candidate with a cheap model (GPT_5_4_MINI, low reasoning)
     using a DS-1000-tuned instruction preamble.
  2. FREE self-check: for problems whose skeleton carries inline example data
     (~80% of DS-1000), execute `setup + candidate` inside `python_session`
     (not metered) and assert the requested target variable(s) are defined and
     the program runs without raising.
  3. On a detected execution failure, do ONE repair pass with a stronger model
     (GPT_5_4, low reasoning), feeding back the traceback. Escalation is rare,
     so mean cost stays far inside the free zone.

Everything in the verify/repair path is wrapped so it can never break the main
path: if `python_session` is unavailable or anything throws, we fall back to the
first candidate. A crashing solution scores 0 anyway, so the loop is monotone:
it can only convert a 0 into a possible 1, never the reverse.
"""

import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI, GPT_5_4


INSTRUCTIONS = """You are an expert Python data scientist solving a DS-1000 problem.

You are given a problem with a code skeleton. Write ONLY the Python code that should be appended AFTER the given skeleton so that the requested variable holds the correct value.

Rules — follow them exactly:
- Output a single `<code>` ... `</code>` block and nothing else. No prose, no explanation, no markdown ```` ``` ```` fences.
- Do NOT repeat any code already shown in the skeleton (imports, data definitions, `load_data()` calls, asserts). Only write the NEW code.
- Assign the answer to the EXACT variable name the problem asks for. Look for a line like `result = ... # put solution in this variable`, or wording like "put score in `b`, put prediction in `c`". The name is often `result` but can be `proba`, `b`, `c`, `df`, etc. Match it precisely.
- Prefer concise, idiomatic, vectorized library calls. Avoid explicit Python `for`/`while` loops when a library function does the job. If the problem says "without using X", "not one by one", "the efficient way", or names a specific function, honor that — a manual reimplementation can be rejected even when the output is right.
- Do not call `print()` (unless the answer literally requires building a string), do not call `plt.show()`, and do not wrap things in functions unless asked.
- The code must run as-is when appended to the skeleton.

Here is the problem:

"""


CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull executable code out of a model completion."""
    if not text:
        return ""
    s = text.strip()
    blocks = [b.strip() for b in CODE_RE.findall(s)]
    blocks = [b for b in blocks if b and "[insert]" not in b]
    if blocks:
        return blocks[-1].strip()
    fences = [b.strip() for b in FENCE_RE.findall(s)]
    if fences:
        return fences[-1].strip()
    # No envelope: strip stray tags/fences and return the raw body.
    s = re.sub(r"</?code>", "", s)
    s = re.sub(r"```(?:python|py)?", "", s)
    s = s.replace("```", "")
    return s.strip()


def parse_skeleton(prompt: str):
    """Return (setup_code, target_vars) parsed from the problem prompt.

    setup_code  : the first non-empty `<code>` block (imports + inline data).
    target_vars : variable names the solution must define.
    """
    blocks = [b.strip() for b in CODE_RE.findall(prompt)]
    setup = ""
    for b in blocks:
        if b and "[insert]" not in b and "insert" not in b.lower():
            setup = b
            break

    targets = []
    # `result = ... # put solution in this variable`  (and variants)
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt, re.MULTILINE):
        targets.append(m.group(1))
    # "put score in `b`, put prediction in `c`"  /  "put ... in `var`"
    for m in re.finditer(r"\bin\s+`([A-Za-z_]\w*)`", prompt):
        targets.append(m.group(1))
    # de-dup, preserve order
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]
    if not targets:
        targets = ["result"]
    return setup, targets


def _looks_like_traceback(out: str) -> bool:
    if not out:
        return False
    return "Traceback (most recent call last)" in out


def _build_check_program(setup: str, candidate: str, targets) -> str:
    asserts = "\n".join(
        f"assert {t!r} in dir(), 'TARGET_MISSING:{t}'" for t in targets
    )
    return f"{setup}\n{candidate}\n{asserts}\nprint('SELFCHECK_OK')"


# Matplotlib answers are scored on the rendered plot and usually define no
# named result variable, so we only verify the plotting code runs (no asserts).
def _build_exec_only_program(setup: str, candidate: str) -> str:
    return f"{setup}\n{candidate}\nprint('SELFCHECK_OK')"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        print(f"[{state.sample_id}] library={lib}")

        prompt = state.input
        full_prompt = INSTRUCTIONS + prompt

        # --- Pass 1: cheap generation -------------------------------------
        candidate = ""
        try:
            resp = await GPT_5_4_MINI.generate(
                full_prompt,
                config=GenerateConfig(reasoning_effort="low", max_tokens=4096),
            )
            candidate = extract_code(resp.completion or "")
        except Exception as e:  # noqa: BLE001
            print(f"  pass1 error: {e!r}")

        # Empty completion (e.g. budget consumed by reasoning) -> retry plain.
        if not candidate.strip():
            try:
                resp = await GPT_5_4_MINI.generate(
                    full_prompt, config=GenerateConfig(max_tokens=2048)
                )
                candidate = extract_code(resp.completion or "")
            except Exception as e:  # noqa: BLE001
                print(f"  pass1 retry error: {e!r}")

        # --- Free self-check + optional repair ----------------------------
        try:
            candidate = await _verify_and_repair(state, prompt, candidate)
        except Exception as e:  # noqa: BLE001
            print(f"  verify skipped: {e!r}")

        if not candidate.strip():
            candidate = "result = None"

        state.output.completion = f"<code>\n{candidate}\n</code>"
        print(f"  emitted {len(candidate)} chars")
        return state

    return solve


async def _verify_and_repair(state: TaskState, prompt: str, candidate: str) -> str:
    """Execute the candidate against inline skeleton data; repair once if it fails.

    Returns the (possibly repaired) candidate. Never raises out — any failure
    inside here leaves the original candidate untouched (handled by caller).
    """
    # Locate the python_session tool.
    py = None
    for t in state.tools or []:
        try:
            if ToolDef(t).name == "python_session":
                py = t
                break
        except Exception:  # noqa: BLE001
            continue
    if py is None:
        return candidate

    setup, targets = parse_skeleton(prompt)
    if not setup.strip():
        return candidate
    is_mpl = str(state.metadata.get("library", "")).lower() == "matplotlib"

    def build(cand):
        return (
            _build_exec_only_program(setup, cand)
            if is_mpl
            else _build_check_program(setup, cand, targets)
        )

    # Precheck: can the skeleton's data even run here? If setup references
    # load_data() / undefined helpers it raises -> data unavailable -> skip.
    try:
        pre = await py(code=setup + "\nprint('SETUP_OK')")
    except Exception as e:  # noqa: BLE001
        print(f"  setup precheck threw: {e!r}")
        return candidate
    if _looks_like_traceback(pre) or "SETUP_OK" not in str(pre):
        print("  data unavailable (load_data/undefined) -> skip self-check")
        return candidate

    # Run setup + candidate (+ target asserts unless matplotlib).
    out = str(await py(code=build(candidate)))
    if "SELFCHECK_OK" in out and not _looks_like_traceback(out):
        print("  self-check passed")
        return candidate

    print("  self-check FAILED -> repair with GPT_5_4")
    traceback_tail = out[-1500:]
    repair_prompt = (
        INSTRUCTIONS
        + prompt
        + "\n\n---\nA previous attempt produced this code:\n<code>\n"
        + candidate
        + "\n</code>\n\nWhen appended to the skeleton it FAILED with:\n```\n"
        + traceback_tail
        + "\n```\n\nReturn a corrected `<code>` block that runs cleanly and "
        "defines the requested variable(s): " + ", ".join(targets) + "."
    )
    try:
        resp = await GPT_5_4.generate(
            repair_prompt,
            config=GenerateConfig(reasoning_effort="low", max_tokens=4096),
        )
        repaired = extract_code(resp.completion or "")
    except Exception as e:  # noqa: BLE001
        print(f"  repair error: {e!r}")
        return candidate

    if not repaired.strip():
        return candidate

    # Verify the repair; keep it if it now passes, else keep whichever ran.
    try:
        out2 = str(await py(code=build(repaired)))
        if "SELFCHECK_OK" in out2 and not _looks_like_traceback(out2):
            print("  repair passed")
            return repaired
        # Neither passed cleanly; prefer the repaired (stronger model) answer.
        print("  repair still imperfect; using repaired answer")
        return repaired
    except Exception as e:  # noqa: BLE001
        print(f"  repair recheck error: {e!r}")
        return repaired
