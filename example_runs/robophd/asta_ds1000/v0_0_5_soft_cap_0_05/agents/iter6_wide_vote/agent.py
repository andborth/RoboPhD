"""DS-1000 solver: diverse-lens value-vote ensemble with Matplotlib voting.

Strategy (see reasoning.md):
  1. Parse the skeleton (setup block, target variable(s), function-body vs
     module-level insertion) with the robust iter3/iter4 parser.
  2. Generate a DECORRELATED ensemble in parallel from five candidates across
     model families (GPT_5_4, CLAUDE_SONNET_4_6, GEMINI_3_5_FLASH, plus two cheap
     extra voters GPT_5_4_MINI and CLAUDE_HAIKU_4_5). Each gets a distinct framing
     LENS (canonical-library / literal-reading / plain) so their errors
     decorrelate. The two cheap voters are ranked LAST: they add verified-value
     voting power (turning 2-of-3 ties into decisive 3-of-5 majorities) but never
     degrade the non-votable fallback, which always uses the strongest candidate.
  3. FREE value-level self-consistency: when the skeleton has runnable inline
     data, execute each candidate in an ISOLATED namespace inside `python_session`
     (not metered) and compute a canonical, tolerance-rounded SHA-1 signature of
     the target value(s). Candidates whose computed value agrees form a vote
     group; emit a representative of the largest group.
  4. MATPLOTLIB (no votable value): run each candidate, then take a
     normalized-code majority vote among the runnable ones; fall back to the
     strongest runnable candidate. A literal keyword-mapping hint is added to the
     prompt for this library.
  5. Adaptive escalation: if no two candidates agree, add a strong arbiter
     (CLAUDE_OPUS_4_8). If every candidate errors, do ONE traceback-fed repair.

Every advanced step is guarded: any failure degrades to a strong single
candidate. A crashing/empty solution scores 0 anyway, so the loop is monotone —
it can convert a 0 into a possible 1, rarely the reverse.
"""

import asyncio
import base64
import re

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import (
    GPT_5_4,
    GPT_5_4_MINI,
    CLAUDE_SONNET_4_6,
    CLAUDE_HAIKU_4_5,
    CLAUDE_OPUS_4_8,
    GEMINI_3_5_FLASH,
)


BASE_INSTRUCTIONS = """You are an expert Python data scientist solving a DS-1000 problem.

You are given a problem with a code skeleton. Write ONLY the Python code that should be appended AFTER the given skeleton so that the requested variable holds the correct value.

Rules — follow them exactly:
- Output a single `<code>` ... `</code>` block and nothing else. No prose, no explanation, no markdown ``` fences.
- Do NOT repeat any code already shown in the skeleton (imports, data definitions, `load_data()` calls, asserts, the `def` line). Only write the NEW code.
- Assign the answer to the EXACT variable name the problem asks for. Look for a line like `result = ... # put solution in this variable`, or wording like "put score in `b`, put prediction in `c`". The name is often `result` but can be `proba`, `b`, `c`, `df`, `centered_scaled_data`, etc. Match it precisely.
- Prefer the library's own canonical, idiomatic function over a manual reimplementation. DS-1000 references use the standard library call (e.g. `sklearn.preprocessing.scale`, `scipy.interpolate.RectBivariateSpline`, `np.column_stack`), and a workaround that gives a slightly different numeric/dtype result can be marked wrong even when it looks correct. When the question says "without using X", "not one by one", "the efficient way", or names a function, honor it — avoid explicit Python `for`/`while` loops when a vectorized library call does the job.
- Read the problem literally: match its exact interpretation, expected shape/dtype, and any worked example values shown in the prompt. Do not over-engineer.
- Do not call `print()` (unless the answer literally requires building a string), do not call `plt.show()`, and do not wrap things in new functions unless asked.
- The code must run as-is when appended to the skeleton.
"""

# Per-family lens, appended to BASE_INSTRUCTIONS to decorrelate the ensemble's
# errors. Each pushes the model toward a different (all valid) reading priority.
LENS_CANONICAL = """
Emphasis for this attempt: reach for the single most CANONICAL library function that exactly implements what is asked. Recall the precise API name and its default behaviour rather than hand-rolling logic; if a dedicated function exists (in numpy / pandas / scipy / sklearn / tensorflow / torch), use it.
"""
LENS_LITERAL = """
Emphasis for this attempt: read the wording with extreme literalness. Map every descriptive phrase to the exact argument, keyword, shape, dtype, axis, and ordering it names — do not paraphrase a requested option into a similar-but-different one. Reproduce any concrete example values shown in the prompt.
"""
LENS_PLAIN = ""

MODULE_HINT = """
Insertion format: MODULE LEVEL. Write top-level statements (no indentation) that assign the requested variable(s).

Here is the problem:

"""

FUNCTION_HINT = """
Insertion format: FUNCTION BODY. The skeleton ends with a `def {fname}(...):` line, and your code goes INSIDE that function. Therefore:
- INDENT every line of your code by 4 spaces so it sits inside the function body.
- END with a `return <answer>` statement that returns the requested value (do NOT assign to a module-level variable; the test calls the function and uses its return value).

Here is the problem:

"""

# Added for Matplotlib problems (no votable `result`; literal keyword mapping is
# the dominant failure mode — "star hatch" must become hatch='*', not marker).
MPL_HINT = """
This is a MATPLOTLIB problem. Map each descriptive instruction to the EXACT matplotlib API call / keyword argument, reading the wording literally:
- "hatch" (e.g. star/dot/cross hatch) -> the `hatch=` keyword (e.g. hatch='*'); it is NOT the same as `marker=`.
- "marker" -> `marker=`; "dashed/dotted line" -> `linestyle='--'`/`':'`; "no line" -> `linestyle=''`; "color" -> `color=`.
- legend / label / title / ticks / limits / grid / scale wording -> the matching matplotlib function (plt.legend, set_xlabel, plt.title, set_xticks, set_xlim, plt.grid, set_yscale, ...).
- Do NOT substitute a similar-looking keyword for the one that is requested.
There is usually no `result` variable — just issue the plotting calls on the given data.
"""


CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
MARKER_RE = re.compile(r"^\s*###?\s*(BEGIN|END)\s+SOLUTION\s*$", re.IGNORECASE)
BOILERPLATE_RE = re.compile(
    r"(?im)^[ \t]*(Write the remaining python code|Put your answer inside)"
)


def extract_code(text: str) -> str:
    """Pull executable code out of a model completion."""
    if not text:
        return ""
    s = text.strip()
    blocks = [b.strip("\n") for b in CODE_RE.findall(s)]
    blocks = [b for b in blocks if b.strip() and "[insert]" not in b]
    if blocks:
        body = blocks[-1]
    else:
        fences = [b.strip("\n") for b in FENCE_RE.findall(s)]
        if fences:
            body = fences[-1]
        else:
            s = re.sub(r"</?code>", "", s)
            s = re.sub(r"```(?:python|py)?", "", s)
            body = s.replace("```", "")
    body = "\n".join(ln for ln in body.split("\n") if not MARKER_RE.match(ln))
    return body.strip("\n")


def _first_setup_block(prompt: str) -> str:
    for b in CODE_RE.findall(prompt):
        m = BOILERPLATE_RE.search(b)
        if m:
            b = b[: m.start()]
        b = b.rstrip("\n")
        if b.strip() and "[insert]" not in b and "insert" not in b.lower():
            return b.strip("\n")
    return ""


def parse_skeleton(prompt: str):
    """Return (setup_code, target_vars, func_mode, func_name)."""
    setup = _first_setup_block(prompt)

    func_mode = False
    func_name = None
    lines = setup.split("\n")
    j = len(lines) - 1
    while j >= 0 and (
        not lines[j].strip()
        or lines[j].lstrip().startswith("#")
        or MARKER_RE.match(lines[j])
    ):
        j -= 1
    if j >= 0:
        m = re.match(r"^\s*def\s+(\w+)\s*\(", lines[j])
        if m:
            func_mode = True
            func_name = m.group(1)
    if not func_mode and re.search(
        r"return the (solution|result) in this function", prompt, re.IGNORECASE
    ):
        m = re.search(r"def\s+(\w+)\s*\(", setup)
        if m:
            func_mode = True
            func_name = m.group(1)

    targets = []
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*\.\.\.", prompt, re.MULTILINE):
        targets.append(m.group(1))
    for m in re.finditer(r"\bin\s+`([A-Za-z_]\w*)`", prompt):
        targets.append(m.group(1))
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]
    if not targets:
        targets = ["result"]
    return setup, targets, func_mode, func_name


def ensure_indented(code: str, indent: int = 4) -> str:
    """Indent function-body code if the model returned it un-indented."""
    if not code.strip():
        return code
    lines = code.split("\n")
    first = next((ln for ln in lines if ln.strip()), "")
    if first[:1] in (" ", "\t"):
        return code
    pad = " " * indent
    return "\n".join(pad + ln if ln.strip() else ln for ln in lines)


def _normalize_code(code: str) -> str:
    """Whitespace/quote/comment-insensitive normal form for code-level voting."""
    out = []
    for ln in code.split("\n"):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r"\s+", "", s)   # collapse all whitespace
        s = s.replace('"', "'")      # normalize quote style
        out.append(s)
    return "\n".join(out)


def _looks_like_traceback(out: str) -> bool:
    return bool(out) and "Traceback (most recent call last)" in out


# ---------------------------------------------------------------------------
# Value-signature harness executed inside python_session (free / not metered).
# ---------------------------------------------------------------------------

_SIG_DEFS = r"""
import base64 as _b64, hashlib as _hl
import numpy as _np
import pandas as _pd

def __round(o):
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, (list, tuple)):
        return [__round(x) for x in o]
    if isinstance(o, dict):
        return {k: __round(v) for k, v in o.items()}
    return o

def __sig(v):
    try:
        if isinstance(v, _pd.DataFrame):
            cols = list(map(str, v.columns))
            idx = list(map(str, v.index))
            try:
                vals = _np.round(v.to_numpy(dtype=float), 6).tolist()
            except Exception:
                vals = v.astype(str).to_numpy().tolist()
            return "DF|" + repr(cols) + "|" + repr(idx) + "|" + repr(vals)
        if isinstance(v, _pd.Series):
            idx = list(map(str, v.index))
            try:
                vals = _np.round(v.to_numpy(dtype=float), 6).tolist()
            except Exception:
                vals = v.astype(str).to_numpy().tolist()
            return "S|" + repr(idx) + "|" + repr(vals)
        if isinstance(v, _np.ndarray):
            try:
                return "A|" + repr(v.shape) + "|" + repr(_np.round(v.astype(float), 6).tolist())
            except Exception:
                return "A|" + repr(v.shape) + "|" + repr(v.tolist())
        if isinstance(v, bool):
            return "B|" + repr(v)
        if isinstance(v, (int, _np.integer)):
            return "I|" + repr(int(v))
        if isinstance(v, (float, _np.floating)):
            return "F|" + repr(round(float(v), 6))
        if isinstance(v, (list, tuple)):
            return "L|" + repr(__round(list(v)))
        if isinstance(v, dict):
            return "D|" + repr({str(k): __sig(val) for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))})
        return "O|" + repr(v)
    except Exception:
        try:
            return "O|" + repr(v)
        except Exception:
            return "ERR"

def __emit(payload):
    print("SIG=" + _hl.sha1(payload.encode("utf-8", "replace")).hexdigest())
"""


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _build_sig_program(setup, candidate, targets, func_mode, func_name) -> str:
    head = (
        _SIG_DEFS
        + f'\n__SETUP = _b64.b64decode("{_b64(setup)}").decode("utf-8")\n'
        + f'__CAND = _b64.b64decode("{_b64(candidate)}").decode("utf-8")\n'
        + "_ns = {}\n"
        "try:\n"
        "    exec(compile(__SETUP + '\\n' + __CAND, '<sol>', 'exec'), _ns, _ns)\n"
    )
    if func_mode and func_name:
        body = (
            f"    _r = _ns[{func_name!r}]()\n"
            "    __emit('FUNC=' + __sig(_r))\n"
        )
    else:
        body = (
            f"    __targets = {list(targets)!r}\n"
            "    _sigs = [t + '=' + __sig(_ns.get(t, None)) for t in __targets]\n"
            "    __emit('||'.join(_sigs))\n"
        )
    tail = (
        "except Exception as _e:\n"
        "    import traceback as _tb\n"
        "    print('RUNERR=' + repr(_e))\n"
        "    _tb.print_exc()\n"
    )
    return head + body + tail


def _extract_sig(out: str):
    """Return the SHA-1 sig string if the program reported one, else None."""
    if not out:
        return None
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("SIG="):
            return ln[4:].strip()
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def _generate(model, prompt, reasoning=None, max_tokens=4096) -> str:
    cfg = {"max_tokens": max_tokens}
    if reasoning:
        cfg["reasoning_effort"] = reasoning
    try:
        resp = await model.generate(prompt, config=GenerateConfig(**cfg))
        return extract_code(resp.completion or "")
    except Exception as e:  # noqa: BLE001
        print(f"  generate error ({getattr(model, 'name', model)}): {e!r}")
        return ""


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lib = state.metadata.get("library", "?")
        prompt = state.input
        setup, targets, func_mode, func_name = parse_skeleton(prompt)
        is_mpl = str(lib).lower() == "matplotlib"
        print(f"[{state.sample_id}] library={lib} func_mode={func_mode} targets={targets}")

        hint = (
            FUNCTION_HINT.format(fname=func_name or "f") if func_mode else MODULE_HINT
        )
        mpl_block = MPL_HINT if is_mpl else ""

        def build_prompt(lens):
            # Order: base rules -> per-family lens -> library block -> format hint -> problem.
            return BASE_INSTRUCTIONS + lens + mpl_block + hint + prompt

        # Common preamble (no lens) — reused for the arbiter/retry paths.
        common = build_prompt("")

        # Diverse-lens ensemble: each family is steered toward a different (valid)
        # reading priority so their errors decorrelate. Ranked strongest-first for
        # tie-breaking.
        # Ranked strongest-first: rank drives all tie-breaks and the non-votable
        # fallback. The last two are cheap extra voters — they add verified-value
        # voting power (where the value harness can run) but never degrade the
        # fallback, which always uses the strongest-ranked candidate.
        specs = [
            ("GPT_5_4", GPT_5_4, "low", 6000, LENS_CANONICAL),
            ("SONNET", CLAUDE_SONNET_4_6, "low", 4096, LENS_LITERAL),
            ("GEMINI", GEMINI_3_5_FLASH, None, 4096, LENS_PLAIN),
            ("GPT_MINI", GPT_5_4_MINI, "low", 4096, LENS_LITERAL),
            ("HAIKU", CLAUDE_HAIKU_4_5, "low", 4096, LENS_CANONICAL),
        ]
        results = await asyncio.gather(
            *(
                _generate(m, build_prompt(lens), reasoning=r, max_tokens=mt)
                for _, m, r, mt, lens in specs
            ),
            return_exceptions=True,
        )
        candidates = []  # list of (rank, name, code)
        for rank, ((name, _m, _r, _mt, _l), res) in enumerate(zip(specs, results)):
            code = res if isinstance(res, str) else ""
            if func_mode and code.strip():
                code = ensure_indented(code)
            if code.strip():
                candidates.append((rank, name, code))

        # Empty everything -> last-ditch plain GPT_5_4 retry.
        if not candidates:
            code = await _generate(GPT_5_4, common, max_tokens=3000)
            if func_mode and code.strip():
                code = ensure_indented(code)
            if code.strip():
                candidates.append((0, "GPT_5_4_retry", code))

        chosen = candidates[0][2] if candidates else ""

        # --- Vote / verify / repair (fully guarded) -----------------------
        try:
            chosen = await _vote_and_repair(
                state, prompt, common, candidates, setup, targets,
                func_mode, func_name, is_mpl,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  vote skipped: {e!r}")

        if not chosen.strip():
            chosen = "    return None" if func_mode else "result = None"

        state.output.completion = f"<code>\n{chosen}\n</code>"
        print(f"  emitted {len(chosen)} chars")
        return state

    return solve


def _find_py(state):
    for t in state.tools or []:
        try:
            if ToolDef(t).name == "python_session":
                return t
        except Exception:  # noqa: BLE001
            continue
    return None


async def _vote_and_repair(
    state, prompt, full_prompt, candidates, setup, targets, func_mode, func_name, is_mpl
):
    """Execute candidates, vote on the computed value, escalate/repair as needed.

    Never raises out — any failure leaves the best single candidate (caller-guarded).
    """
    if not candidates:
        return ""

    best_single = candidates[0][2]  # strongest model's candidate
    py = _find_py(state)
    if py is None or not setup.strip():
        return best_single

    # Precheck: can the skeleton even run here (no load_data/undefined helpers)?
    try:
        pre = await py(code=setup + "\nprint('SETUP_OK')")
    except Exception as e:  # noqa: BLE001
        print(f"  setup precheck threw: {e!r}")
        return best_single
    if _looks_like_traceback(pre) or "SETUP_OK" not in str(pre):
        print("  data unavailable -> skip value-vote")
        return best_single

    async def run_sig(code):
        """Return (sig, raw_output). sig is None when the candidate errored."""
        try:
            out = str(await py(code=_build_sig_program(setup, code, targets, func_mode, func_name)))
        except Exception as e:  # noqa: BLE001
            return None, f"exec exception: {e!r}"
        return _extract_sig(out), out

    # Matplotlib answers are scored on the plot (no votable value). Run each
    # candidate, then take a normalized-code majority vote among the runnable
    # ones; fall back to the strongest runnable candidate.
    if is_mpl:
        runnable = []  # (rank, name, code)
        for rank, name, code in candidates:
            prog = setup + "\n" + code + "\nprint('EXEC_OK')"
            try:
                out = str(await py(code=prog))
                if "EXEC_OK" in out and not _looks_like_traceback(out):
                    runnable.append((rank, name, code))
            except Exception:  # noqa: BLE001
                continue
        if not runnable:
            print("  mpl: no candidate ran clean -> best single")
            return best_single
        # Group runnable candidates by normalized code; majority wins.
        norm_groups = {}
        for rank, name, code in runnable:
            norm_groups.setdefault(_normalize_code(code), []).append((rank, name, code))
        best = None
        for members in norm_groups.values():
            members.sort(key=lambda m: m[0])
            key = (len(members), -members[0][0])
            if best is None or key > best[0]:
                best = (key, members)
        members = best[1]
        if len(members) >= 2:
            print(f"  mpl consensus ({len(members)}): {','.join(m[1] for m in members)}")
        else:
            print(f"  mpl: no majority -> strongest runnable {members[0][1]}")
        return members[0][2]

    # --- Value vote across the ensemble -------------------------------
    sigs = await asyncio.gather(*(run_sig(c[2]) for c in candidates))
    last_err = ""
    groups = {}  # sig -> list of (rank, name, code)
    for (rank, name, code), (sig, out) in zip(candidates, sigs):
        if sig is None:
            last_err = out
            print(f"  {name}: RUNERR")
            continue
        groups.setdefault(sig, []).append((rank, name, code))
        print(f"  {name}: sig={sig[:12]}")

    pick = _pick_from_groups(groups)
    if pick is not None:
        sig, members = pick
        if len(members) >= 2:
            names = ",".join(m[1] for m in members)
            print(f"  consensus ({len(members)}): {names}")
            return members[0][2]
        # Only one candidate ran (others errored) -> use it.
        if len(groups) == 1:
            print(f"  single runnable candidate: {members[0][1]}")
            return members[0][2]

    # --- No agreement: escalate with a strong arbiter -----------------
    if groups:
        print("  no consensus -> arbiter (CLAUDE_OPUS_4_8)")
        arb = await _generate(CLAUDE_OPUS_4_8, full_prompt, max_tokens=4096)
        if func_mode and arb.strip():
            arb = ensure_indented(arb)
        if arb.strip():
            a_sig, a_out = await run_sig(arb)
            if a_sig is not None:
                groups.setdefault(a_sig, []).append((-1, "OPUS", arb))
                pick = _pick_from_groups(groups)
                if pick is not None:
                    sig, members = pick
                    if len(members) >= 2:
                        print(f"  arbiter consensus: {','.join(m[1] for m in members)}")
                        return members[0][2]
                # Still all distinct -> trust the arbiter (strongest).
                print("  still split -> arbiter answer")
                return arb
            else:
                print(f"  arbiter RUNERR: {a_out[:120]}")
        # Arbiter failed; pick best-ranked runnable group member.
        best_run = _best_runnable(groups)
        if best_run is not None:
            return best_run
        return best_single

    # --- Nothing ran: one traceback-fed repair ------------------------
    print("  all candidates errored -> repair (GPT_5_4)")
    repair_prompt = (
        full_prompt
        + "\n\n---\nA previous attempt produced this code:\n<code>\n"
        + best_single
        + "\n</code>\n\nWhen appended to the skeleton it FAILED with:\n```\n"
        + (last_err or "")[-1500:]
        + "\n```\n\nReturn a corrected `<code>` block that runs cleanly and "
        "produces the requested value(s): " + ", ".join(targets) + "."
    )
    repaired = await _generate(GPT_5_4, repair_prompt, reasoning="low", max_tokens=4096)
    if func_mode and repaired.strip():
        repaired = ensure_indented(repaired)
    if not repaired.strip():
        return best_single
    r_sig, _ = await run_sig(repaired)
    if r_sig is not None:
        print("  repair runs clean")
        return repaired
    print("  repair still imperfect; using repaired (stronger) answer")
    return repaired


def _pick_from_groups(groups):
    """Return (sig, members) for the largest group, tie-broken by best rank."""
    if not groups:
        return None
    best = None
    for sig, members in groups.items():
        members_sorted = sorted(members, key=lambda m: m[0])
        size = len(members)
        top_rank = members_sorted[0][0]
        key = (size, -top_rank)  # bigger group wins; then stronger model
        if best is None or key > best[0]:
            best = (key, sig, members_sorted)
    return (best[1], best[2])


def _best_runnable(groups):
    """Return the code of the best-ranked candidate among all runnable groups."""
    best = None
    for members in groups.values():
        for rank, name, code in members:
            if best is None or rank < best[0]:
                best = (rank, code)
    return best[1] if best else None
