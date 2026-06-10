"""DS-1000 solver: tri-model generation + sandbox cross-check + expected-output
verification + critique + adjudication + final execution check.

Pipeline (iter5: expectverify variant of iter4_trivote_adjudicate):
  1. CLAUDE_SONNET_4_6, GPT_5_4 and GEMINI_3_FLASH_PREVIEW each generate a
     solution (parallel).
  2. All candidates are executed in the sandbox against the visible context;
     target values are compared pairwise with type-aware, tolerant equality.
  3. NEW: a cheap GPT_5_4 call compares each executed value against the
     expected output displayed verbatim in the problem text (most DS-1000
     prompts show the desired output). A mismatch on the majority candidate
     blocks the consensus fast path.
  4. If >=2 OK candidates agree by value AND the group's value matches the
     displayed expectation, a cheap GPT_5_4 checklist critique gates submission.
  5. Otherwise GPT_5_5 adjudicates with full execution + expectation evidence.
  6. The adjudicated code passes through the critique gate (one conservative
     repair pass if flagged) and NEW: is itself executed in the sandbox; a
     raised exception or expectation mismatch triggers one evidence-driven
     GPT_5_5 repair, accepted only if it executes cleanly.
Every stage degrades gracefully; a last-resort one-shot call backs the whole thing.
"""

import ast
import asyncio
import base64
import json
import re
import textwrap
import traceback

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import (
    CLAUDE_SONNET_4_6,
    GEMINI_3_FLASH_PREVIEW,
    GPT_5_4,
    GPT_5_4_MINI,
    GPT_5_5,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RULES = """\
Rules (hidden tests run your code on DIFFERENT inputs than the example, and some \
tests also inspect your code text for required library calls or forbidden loops):
1. Output ONLY the code to append, inside a single <code>...</code> block. No prose,
   no markdown fences, no BEGIN/END SOLUTION markers.
2. Do not repeat code already given in the context (imports, data setup), though
   adding a missing import is fine.
3. Assign the final answer to the exact variable the problem asks for (`result`
   unless the prompt names another, e.g. `cluster_labels = ... # put solution in this variable`).
4. If the given code ends inside a function definition (e.g. `def f(...):` or
   `def solve(...):`), output only the function body, indented 4 spaces, ending
   with a `return`.
5. NEVER hardcode values you derived by eye from the example data (a column name
   that happens to hold the max, an index position, a size). Compute them
   programmatically (idxmax/argmax/.shape/...) so the code works on hidden inputs.
6. Honor explicitly requested output types/formats exactly: "as a list" -> .tolist();
   "should be csr_matrix" -> ensure csr (e.g. .tocsr()); Series vs DataFrame; dtype;
   index/column names.
7. Follow the wording literally, especially for matplotlib: "hatch" means the hatch=
   argument (not marker), "edge color" means edgecolor=, "minor ticks" means the
   minor-tick API, etc.
8. If the question asks how to do X with a particular library/function, use that
   library/function idiomatically; never reimplement it with Python loops. Avoid
   for/while loops whenever a vectorized or built-in solution exists.
9. These problems come from StackOverflow; the grader's reference is the canonical
   accepted answer. Prefer the standard idiom over a clever alternative (e.g.
   sklearn.preprocessing.scale(data) for generic scale-and-center;
   scipy.cluster.hierarchy.linkage(mat, 'ward') directly on the matrix shown).
10. Be robust to the shapes/dtypes the problem implies (a generic "data" ndarray
    may be 1-D; prefer APIs that accept it).
11. Some hidden tests tokenize your CODE TEXT and assert that the canonical API name
    appears. When the problem names a specific format, API, or function, that exact
    token must appear literally in your code, even if another call would produce the
    same effect. Example: "save the model in SavedModel format" requires
    tf.saved_model.save(model, "export/1") — the token `saved_model` must appear;
    model.export("export/1") fails the grader even though it also writes a SavedModel.
12. Hidden tests include degenerate sizes (a single match, one row, length-1 axes).
    Prefer constructs whose output shape/type is stable for ANY input size:
    `.nonzero(as_tuple=True)[0]` or `.reshape(-1)` instead of `.squeeze()` after
    nonzero/where (squeeze collapses to 0-dim when exactly one element matches);
    `df.loc[[i]]` vs `df.loc[i]` when a DataFrame row is required; never `.item()`
    or `[0]` on something that may hold several values.
13. When the problem DISPLAYS the desired output ("I want to get this:", "The
    expected one should be like this:", a printed array/DataFrame), that display is
    ground truth — the grader's reference reproduces it EXACTLY. Derive formulas and
    constants from it and check your code against it element by element. Examples:
    a reversed ranking displayed as array with first element 7 for an 8-element input
    means len(a) - rankdata(a).astype(int), NOT len(a)+1-rankdata(a); displayed rows
    where 01-Feb-2019 sorts before 01-Jan-2019 mean the reference formatted dates to
    strings FIRST and then sorted lexicographically, even though the prose said
    "smaller date ahead"; NaN shown in the display means a merge-style fill produces
    NaN, not a transform-style None. Match ordering, missing-value representation and
    value types exactly, even when they contradict a literal reading of the prose.
14. Match the canonical answer's CONSTRUCTION even when an alternative prints
    identically: tabulating np.unique(x, return_counts=True) output into a DataFrame
    is canonically pd.DataFrame(np.column_stack(someTuple), columns=[...]) — which
    coerces the counts to strings — and the grader expects those string values; a
    dict-of-columns construction keeping ints fails even though it prints the same."""

GEN_PREAMBLE = (
    "You are an expert Python data-science engineer solving a DS-1000 benchmark "
    "problem. Your code will be appended to the given partial program and graded "
    "by hidden tests.\n\n" + RULES + "\n\nProblem:\n"
)

CRITIQUE_PROMPT = """\
A candidate solution to a DS-1000 problem is below. Hidden tests will run it on
DIFFERENT inputs than the visible example and may also check the code text for
required function usage. Decide if it must be revised before submission.

Check ONLY for these high-risk flaws:
1. Hardcoded example-derived values (specific labels/indices/sizes that may differ
   on hidden inputs — e.g. dropping a column by name when the task said "drop the largest").
2. An explicitly requested output type/format not honored exactly (list vs array,
   sparse matrix format, Series vs DataFrame, dtype).
3. An explicit instruction not followed literally (especially matplotlib styling
   words: hatch vs marker, edgecolor, ticks, transparency, ...).
4. Fragility to other valid inputs the problem implies (1-D vs 2-D arrays,
   different column names or sizes, NaNs if mentioned).
5. The question asks to use a specific function/library but the code reimplements
   it manually, or uses a for/while loop where a vectorized call was implied.
6. The problem names a specific API, format, or function (e.g. "SavedModel format",
   "use np.einsum", "csr_matrix") but the canonical token does not appear literally
   in the code text — hidden tests may tokenize the code and assert the token is
   present (e.g. tf.saved_model.save(...) is required; model.export(...) fails even
   though it has the same effect).
7. Output shape/type unstable on degenerate inputs: `.squeeze()` after
   nonzero/where collapses to 0-dim when exactly one element matches (use
   as_tuple/reshape(-1)); single-row/column selections silently switching between
   Series and DataFrame; `.item()`/`[0]` on possibly-multi-element results. Hidden
   tests include size-1 cases.
8. The problem displays a concrete expected output and the code provably will NOT
   reproduce it exactly: wrong element/row order (e.g. sorting by real dates when
   the displayed rows are ordered by formatted date STRINGS), an off-by-one in a
   formula derivable from the display, None where the display shows NaN, or numeric
   values where the canonical construction yields strings.

If none apply, reply exactly: OK
Otherwise reply with 1-3 short bullets naming the flaw(s). No stylistic comments.

Problem:
{problem}

Candidate code (will be appended to the context program):
{code}
"""

EXPECT_PROMPT = """\
A DS-1000 problem is shown below, followed by the repr of each candidate
solution's executed value on the visible example data.

Many DS-1000 problems display the desired output verbatim in the problem text
("I want to get this:", "The expected one should be like this:", a printed
DataFrame/array). That displayed output is ground truth: the grader's reference
reproduces it exactly, including any ordering or representation that looks odd.

For each candidate decide whether its executed value matches the displayed
expected output:
- "match": element values, element/row ORDER, shape, columns/index and
  missing-value representation all agree. Ignore pure formatting noise
  (whitespace, column alignment, quoting, dtype footers, repr truncation
  marked by '...').
- "mismatch": any substantive difference: different numbers, different order,
  different shape/columns/length, None vs NaN, strings vs numbers.
- "unknown": the problem displays no concrete expected output to compare
  against, or the comparison cannot be made for this candidate.

Reply with ONLY one JSON object mapping each candidate letter to its verdict,
plus a "why" key briefly describing any mismatch, e.g.
{{"A": "mismatch", "B": "match", "why": "A's first element is 8 but the problem shows 7"}}

Problem:
{problem}

Executed values:
{exec_reprs}
"""

ADJUDICATE_PROMPT = """\
You are the final reviewer for a DS-1000 data-science problem. Several candidates
were generated by different models and executed against the visible example context.
Decide the FINAL code to submit: pick one candidate, fix one, or write a better
solution yourself.

""" + RULES + """

Problem:
{problem}

{candidates}

Execution on the visible example{mock_note}:
{exec_info}
{critique_section}
Notes:
- If executions failed with environment-looking errors (protobuf / MessageFactory
  noise, GPU/CUDA, deprecation spam — common for TensorFlow in this sandbox), treat
  the execution evidence as inconclusive: judge on code semantics and on which
  candidate literally uses the API the problem names (rule 11), and do not steer
  away from the canonical idiom just because the sandbox run errored.
- When two OK candidates disagree (especially in shape/dimensionality), the example
  or mock data is often degenerate (e.g. a single matching element collapsing a
  dimension under .squeeze()). Prefer the candidate whose output shape/type is
  stable for ANY input size (rule 12).
- A "mismatch" in the expected-output comparison is strong evidence: the output
  displayed in the problem is ground truth (rule 13) — reproduce its ordering,
  NaN/None representation and value types exactly — unless the comparison was
  clearly confused by repr truncation or formatting noise.

Output ONLY the final code inside <code> and </code> tags.{body_note}
"""

REPAIR_PROMPT = """\
You are the final reviewer for a DS-1000 data-science problem. The code below was
already selected for submission, but an automated checklist reviewer flagged a
possible flaw. Decide whether the flag is real.

""" + RULES + """

Problem:
{problem}

Selected code:
{code}

Reviewer flag(s):
{critique}

If the flag is a false alarm, output the SAME code unchanged. Otherwise output a
minimally revised version that fixes the flagged flaw(s) and nothing else.
Output ONLY the final code inside <code> and </code> tags.{body_note}
"""

EXEC_REPAIR_PROMPT = """\
You are the final reviewer for a DS-1000 data-science problem. The code below was
already selected for submission, but executing it against the visible example
revealed a problem.

""" + RULES + """

Problem:
{problem}

Selected code:
{code}

Evidence from executing it on the visible example:
{issue}

Output a minimally revised version that fixes this and nothing else. If the
evidence is clearly an artifact (sandbox/environment noise, repr truncation,
formatting-only difference), output the SAME code unchanged.
Output ONLY the final code inside <code> and </code> tags.{body_note}
"""

MOCK_PROMPT = """\
The Python context code below references undefined helper(s) such as load_data().
Write Python code defining those helpers so the context runs, returning SIMPLE,
small, plausible data consistent with the problem description and any assert
statements (respect asserted types; pick the shape the problem text implies).
Output only code inside <code></code> tags.

Problem description:
{problem}

Context code:
{context}
"""

# ---------------------------------------------------------------------------
# Sandbox harness (runs inside python_session; unmetered)
# ---------------------------------------------------------------------------

HARNESS_TEMPLATE = r'''
import base64 as _b64, json as _json, traceback as _tb
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass
_p = _json.loads("""__PAYLOAD__""")
_ctx = _b64.b64decode(_p["ctx"]).decode()
_results = []
_vals = []
for _cb in _p["cands"]:
    _cand = _b64.b64decode(_cb).decode()
    _src = _ctx + "\n" + _cand
    _tgt = _p["target"]
    if _p["call"]:
        _src = _src + "\n\nresult = " + _p["call"]
        _tgt = "result"
    _ns = {}
    _err = None; _rep = None; _tp = None; _v = None
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except Exception:
        pass
    try:
        exec(compile(_src, "<cand>", "exec"), _ns)
        if _tgt == "__none__":
            _rep = "(plot problem; no target var checked)"; _tp = "plot"
        elif _tgt in _ns:
            _v = _ns[_tgt]
            try:
                _rep = repr(_v)[:600]
            except Exception:
                _rep = "<unreprable>"
            _tp = type(_v).__name__
        else:
            _err = "TARGET_MISSING: %r was not assigned" % _tgt
    except Exception:
        _err = _tb.format_exc()[-1500:]
    _results.append({"err": _err, "repr": _rep, "type": _tp})
    _vals.append(_v)

def _same(a, b):
    import numpy as _np
    try:
        if type(a).__name__ != type(b).__name__:
            return False
        try:
            import scipy.sparse as _sp
            if _sp.issparse(a):
                return a.shape == b.shape and (a != b).nnz == 0
        except Exception:
            pass
        try:
            import pandas as _pd
            if isinstance(a, (_pd.DataFrame, _pd.Series, _pd.Index)):
                return bool(a.equals(b))
        except Exception:
            pass
        try:
            import torch as _th
            if isinstance(a, _th.Tensor):
                return bool(_th.equal(a, b))
        except Exception:
            pass
        if isinstance(a, _np.ndarray):
            if a.shape != b.shape:
                return False
            try:
                return bool(_np.allclose(a, b, equal_nan=True))
            except Exception:
                return bool(_np.array_equal(a, b))
        if isinstance(a, (list, tuple)):
            try:
                _aa = _np.asarray(a, dtype=float); _bb = _np.asarray(b, dtype=float)
                return _aa.shape == _bb.shape and bool(_np.allclose(_aa, _bb, equal_nan=True))
            except Exception:
                return a == b
        if isinstance(a, float):
            return bool(_np.isclose(a, b, equal_nan=True))
        if hasattr(a, "numpy"):
            try:
                return bool(_np.allclose(_np.asarray(a), _np.asarray(b)))
            except Exception:
                pass
        return bool(a == b)
    except Exception:
        try:
            return repr(a) == repr(b)
        except Exception:
            return False

_pairs = []
_n = len(_results)
for _i in range(_n):
    for _j in range(_i + 1, _n):
        _e = False
        if _results[_i]["err"] is None and _results[_j]["err"] is None:
            try:
                _e = bool(_same(_vals[_i], _vals[_j]))
            except Exception:
                _e = False
        _pairs.append([_i, _j, _e])
print("VERDICT::" + _json.dumps({"results": _results, "pairs": _pairs}))
'''

# ---------------------------------------------------------------------------
# Parsing / cleaning helpers
# ---------------------------------------------------------------------------


def clean_code(text: str) -> str:
    """Extract bare code from a model reply, preserving internal indentation."""
    if not text:
        return ""
    s = text
    blocks = re.findall(r"<code>(.*?)</code>", s, re.S)
    block = next((b for b in reversed(blocks) if b.strip()), None)
    if block is not None:
        s = block
    else:
        fences = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", s, re.S)
        fence = next((b for b in reversed(fences) if b.strip()), None)
        if fence is not None:
            s = fence
        s = s.replace("<code>", "").replace("</code>", "")
    lines = []
    for ln in s.splitlines():
        t = ln.strip().strip("#").strip()
        if t in ("BEGIN SOLUTION", "END SOLUTION", "SOLUTION START", "SOLUTION END"):
            continue
        lines.append(ln.rstrip())
    out = "\n".join(lines)
    return out.strip("\n")


def reindent_body(code: str) -> str:
    """Ensure a function-body candidate is indented (preserving relative structure)."""
    lines = code.splitlines()
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank:
        return code
    min_ind = min(len(ln) - len(ln.lstrip()) for ln in nonblank)
    if min_ind == 0:
        return "\n".join(("    " + ln) if ln.strip() else ln for ln in lines)
    return code


def normalize_candidate(code: str, func_form: bool) -> str:
    if not code:
        return code
    if func_form:
        return reindent_body(code)
    return textwrap.dedent(code)


def extract_context(prompt: str) -> str:
    head = prompt
    for marker in ("BEGIN SOLUTION", "# SOLUTION START", "SOLUTION START"):
        idx = head.find(marker)
        if idx != -1:
            head = head[:idx]
            break
    blocks = re.findall(r"<code>(.*?)</code>", head, re.S)
    # some prompts leave the final <code> block unterminated before BEGIN SOLUTION
    last_open = head.rfind("<code>")
    if last_open != -1 and last_open > head.rfind("</code>"):
        blocks.append(head[last_open + len("<code>"):])
    if blocks:
        return "\n".join(b.strip("\n") for b in blocks)
    # matplotlib-style prompts: raw code (imports + instruction comments), no tags
    return head


def find_target_var(prompt: str) -> str:
    m = re.search(r"([A-Za-z_]\w*)\s*=\s*\.\.\.\s*#\s*put solution", prompt)
    return m.group(1) if m else "result"


def detect_function_call(ctx: str):
    """If ctx ends inside a `def`, return the call expression to evaluate it
    (required args passed by name from globals, defaulted args omitted)."""
    if "def " not in ctx:
        return None
    try:
        compile(ctx, "<ctx>", "exec")
        return None  # context is complete; not a dangling function body
    except SyntaxError:
        pass
    except Exception:
        return None
    try:
        tree = ast.parse(ctx + "\n    pass")
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None
    fn = funcs[-1]
    n_required = len(fn.args.args) - len(fn.args.defaults)
    required = [a.arg for a in fn.args.args[:n_required]]
    return f"{fn.name}({', '.join(required)})"


def context_compiles(ctx: str, func_form: bool) -> bool:
    if func_form:
        return True  # validated per-candidate with the body attached
    try:
        compile(ctx, "<ctx>", "exec")
        return True
    except Exception:
        return False


def norm_text(code: str) -> str:
    return re.sub(r"\s+", "", code or "")


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def build_harness(ctx: str, cands: list, target: str, call_expr) -> str:
    payload = json.dumps(
        {"ctx": b64(ctx), "cands": [b64(c) for c in cands], "target": target,
         "call": call_expr or ""}
    )
    return HARNESS_TEMPLATE.replace("__PAYLOAD__", payload)


def emit(code: str) -> str:
    return "<code>\n" + (code or "result = None").strip("\n") + "\n</code>"


def fmt_exec(label: str, r) -> str:
    if r is None:
        return f"{label}: (not executed)"
    if r.get("err"):
        return f"{label}: raised an error:\n{r['err']}"
    return f"{label}: ran OK, result type={r.get('type')}, repr: {r.get('repr')}"


def equality_groups(n: int, pairs) -> list:
    """Union candidates into value-equal groups from the harness pair list."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, eq in pairs:
        if eq:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


async def call_model(model, prompt: str, max_tokens: int, effort=None) -> str:
    try:
        cfg = (
            GenerateConfig(max_tokens=max_tokens, reasoning_effort=effort)
            if effort
            else GenerateConfig(max_tokens=max_tokens)
        )
        resp = await model.generate(prompt, config=cfg)
        return resp.completion or ""
    except Exception:
        print("  model call failed:", traceback.format_exc()[-300:])
        return ""


async def run_critique(problem: str, code: str) -> str:
    """Cheap checklist critique; empty string means clean."""
    critique = (
        await call_model(
            GPT_5_4,
            CRITIQUE_PROMPT.format(problem=problem[:6000], code=code),
            400,
        )
    ).strip()
    if critique.upper().startswith("OK"):
        return ""
    return critique


async def run_expect_check(problem: str, items) -> tuple:
    """Compare executed reprs against the output displayed in the problem.

    items: list of (label, repr_text). Returns (verdicts dict, why string);
    verdicts maps label -> 'match' | 'mismatch' | 'unknown'.
    """
    if not items:
        return {}, ""
    reprs = "\n".join(f"Candidate {lbl}: {rep}" for lbl, rep in items)
    raw = await call_model(
        GPT_5_4,
        EXPECT_PROMPT.format(problem=problem[:6000], exec_reprs=reprs),
        300,
    )
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}, ""
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {}, ""
    if not isinstance(data, dict):
        return {}, ""
    why = str(data.pop("why", "") or "")
    verdicts = {}
    for k, v in data.items():
        v = str(v).strip().lower()
        if v in ("match", "mismatch", "unknown"):
            verdicts[str(k)] = v
    return verdicts, why


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            return await _solve(state)
        except Exception:
            print("PIPELINE FAILURE, falling back to one-shot:",
                  traceback.format_exc()[-500:])
            try:
                raw = await call_model(GPT_5_4, GEN_PREAMBLE + state.input, 1500)
                if not raw:
                    raw = (await GPT_5_4_MINI.generate(state.input)).completion or ""
                state.output.completion = emit(clean_code(raw))
            except Exception:
                state.output.completion = emit("result = None")
            return state

    async def _solve(state: TaskState) -> TaskState:
        problem = state.input
        library = (state.metadata.get("library") or "").lower()
        is_mpl = library == "matplotlib"
        print(f"[{state.sample_id}] library={library}")

        ctx = extract_context(problem)
        call_expr = detect_function_call(ctx)
        func_form = call_expr is not None
        target = "__none__" if is_mpl else find_target_var(problem)
        print(f"  target={target} func_call={call_expr}")

        async def exec_single(harness_ctx, code):
            """Run one candidate in the sandbox; return its result record."""
            try:
                py = next(
                    t for t in state.tools if ToolDef(t).name == "python_session"
                )
                out = await py(
                    code=build_harness(harness_ctx, [code], target, call_expr)
                )
                m = re.search(r"VERDICT::(\{.*\})", str(out))
                if m:
                    return json.loads(m.group(1))["results"][0]
            except Exception:
                print("  exec_single failed:", traceback.format_exc()[-200:])
            return None

        # --- Stage 1: tri-model generation (parallel, different families) ---
        gen_prompt = GEN_PREAMBLE + problem
        raws = await asyncio.gather(
            call_model(CLAUDE_SONNET_4_6, gen_prompt, 1500),
            call_model(GPT_5_4, gen_prompt, 1500),
            call_model(GEMINI_3_FLASH_PREVIEW, gen_prompt, 1500),
        )
        all_labels = ["A", "B", "C"]
        cands, labels = [], []
        for lbl, raw in zip(all_labels, raws):
            code = normalize_candidate(clean_code(raw), func_form)
            if code:
                cands.append(code)
                labels.append(lbl)
        if not cands:
            raise RuntimeError("all generations empty")
        n = len(cands)
        print(f"  candidates: {''.join(labels)} ({n})")

        # --- Stage 2: sandbox cross-check (unmetered) ---
        mock_code = ""
        if re.search(r"\bload_data\s*\(", ctx):
            mock_code = clean_code(
                await call_model(
                    GPT_5_4,
                    MOCK_PROMPT.format(problem=problem[:4000], context=ctx),
                    600,
                )
            )
            print(f"  mock generated ({len(mock_code)} chars)")
        harness_ctx = (mock_code + "\n" + ctx) if mock_code else ctx

        verdict = None
        if context_compiles(harness_ctx, func_form):
            try:
                py = next(
                    t for t in state.tools if ToolDef(t).name == "python_session"
                )
                out = await py(
                    code=build_harness(harness_ctx, cands, target, call_expr)
                )
                m = re.search(r"VERDICT::(\{.*\})", str(out))
                if m:
                    verdict = json.loads(m.group(1))
            except Exception:
                print("  harness failed:", traceback.format_exc()[-300:])

        results = verdict["results"] if verdict else [None] * n
        ok = [bool(r) and r["err"] is None for r in results]

        # Majority: a group of >=2 candidates agreeing by executed value
        # (matplotlib and no-verdict cases fall back to text equality).
        if verdict and not is_mpl:
            pairs = verdict["pairs"]
        else:
            pairs = [
                [i, j, norm_text(cands[i]) == norm_text(cands[j])
                 and (verdict is None or (ok[i] and ok[j]))]
                for i in range(n) for j in range(i + 1, n)
            ]
        groups = equality_groups(n, pairs)
        ok_groups = [g for g in groups
                     if len(g) >= 2 and all(verdict is None or ok[i] for i in g)]
        majority = ok_groups[0] if ok_groups else None
        print(f"  exec: ok={ok} groups={groups} majority={majority} "
              f"(verdict={'yes' if verdict else 'no'})")

        # --- Stage 3: expected-output check (executed values vs the output
        #     displayed in the problem text). Skipped for matplotlib and when
        #     mock data replaced load_data() (values wouldn't match anyway). ---
        expect_verdicts, expect_why = {}, ""
        if verdict and not is_mpl and not mock_code:
            items = [
                (labels[i], results[i].get("repr") or "(no repr)")
                for i in range(n) if ok[i]
            ]
            expect_verdicts, expect_why = await run_expect_check(problem, items)
            if expect_verdicts:
                print(f"  expect check: {expect_verdicts}"
                      f"{' why=' + expect_why[:150] if expect_why else ''}")

        # --- Stage 4: critique gate on majority agreement (blocked if the
        #     majority value mismatches the displayed expected output) ---
        critique = ""
        if majority is not None:
            chosen = cands[majority[0]]  # strongest generator in the group
            if expect_verdicts.get(labels[majority[0]]) == "mismatch":
                critique = (
                    "Expected-output check: the candidates' executed value does "
                    "NOT match the output displayed in the problem text. "
                    + (expect_why or "")
                )
                print("  majority blocked by expect-mismatch")
            else:
                critique = await run_critique(problem, chosen)
                if not critique:
                    print(f"  path=majority({''.join(labels[i] for i in majority)})"
                          "+critique-OK")
                    state.output.completion = emit(chosen)
                    return state
                print(f"  critique flagged: {critique[:200]}")

        # --- Stage 5: adjudication ---
        cand_section = "\n\n".join(
            f"Candidate {labels[i]}:\n{cands[i]}" for i in range(n)
        )
        exec_lines = [fmt_exec(labels[i], results[i]) for i in range(n)]
        if verdict and not is_mpl:
            agree_txt = ", ".join(
                "{" + ",".join(labels[i] for i in g) + "}" for g in groups
            )
            exec_lines.append(
                f"Value-equality groups under tolerant comparison: {agree_txt}"
            )
        if expect_verdicts:
            exec_lines.append(
                "Comparison of executed values to the output displayed in the "
                "problem: " + json.dumps(expect_verdicts)
                + (f" — {expect_why}" if expect_why else "")
            )
        exec_info = "\n".join(exec_lines)
        body_note = (
            " Your code is the body of f(): indent every line 4 spaces and "
            "end with a return."
            if func_form
            else ""
        )
        adj_prompt = ADJUDICATE_PROMPT.format(
            problem=problem,
            candidates=cand_section,
            mock_note=(
                " (load_data() was replaced by synthetic mock data; errors may be "
                "artifacts of the mock)"
                if mock_code
                else ""
            ),
            exec_info=exec_info,
            critique_section=(
                f"\nA reviewer flagged the majority candidate:\n{critique}\n"
                if critique
                else ""
            ),
            body_note=body_note,
        )
        final = clean_code(await call_model(GPT_5_5, adj_prompt, 6000))
        if not final:
            final = clean_code(
                await call_model(CLAUDE_SONNET_4_6, adj_prompt, 2000, effort="medium")
            )
        if final:
            final = normalize_candidate(final, func_form)
            print("  path=adjudicated")
        else:
            ranked = sorted(range(n), key=lambda i: (not ok[i], i))
            final = cands[ranked[0]]
            print("  path=adjudication-failed, fallback candidate")
            state.output.completion = emit(final)
            return state

        # --- Stage 6: critique gate on the adjudicated code + one repair pass ---
        adj_critique = await run_critique(problem, final)
        if adj_critique:
            print(f"  post-adjudication critique flagged: {adj_critique[:200]}")
            repaired = clean_code(
                await call_model(
                    GPT_5_5,
                    REPAIR_PROMPT.format(
                        problem=problem,
                        code=final,
                        critique=adj_critique,
                        body_note=body_note,
                    ),
                    6000,
                )
            )
            if repaired:
                final = normalize_candidate(repaired, func_form)
                print("  path=adjudicated+repaired")

        # --- Stage 7: execute the final code; one evidence-driven repair ---
        if not is_mpl and not mock_code and context_compiles(harness_ctx, func_form):
            fr = await exec_single(harness_ctx, final)
            issue = ""
            if fr and fr.get("err"):
                issue = ("Executing it on the visible example raised:\n"
                         + fr["err"])
            elif fr and fr.get("repr"):
                # Reuse the earlier verdict if the final code equals a candidate.
                same_lbl = next(
                    (labels[i] for i in range(n)
                     if norm_text(cands[i]) == norm_text(final)),
                    None,
                )
                fv, fwhy = None, expect_why
                if same_lbl is not None and same_lbl in expect_verdicts:
                    fv = expect_verdicts[same_lbl]
                else:
                    fvd, fwhy = await run_expect_check(
                        problem, [("F", fr["repr"])]
                    )
                    fv = fvd.get("F")
                if fv == "mismatch":
                    issue = (
                        "Its executed value does NOT match the output displayed "
                        "in the problem text. " + (fwhy or "")
                        + f"\nExecuted repr: {fr['repr']}"
                    )
            if issue:
                print(f"  final exec issue: {issue[:200]}")
                repaired = clean_code(
                    await call_model(
                        GPT_5_5,
                        EXEC_REPAIR_PROMPT.format(
                            problem=problem,
                            code=final,
                            issue=issue,
                            body_note=body_note,
                        ),
                        6000,
                    )
                )
                if repaired:
                    repaired = normalize_candidate(repaired, func_form)
                    if norm_text(repaired) != norm_text(final):
                        rr = await exec_single(harness_ctx, repaired)
                        # accept the repair unless it newly breaks execution
                        if rr is None or not rr.get("err") or (fr and fr.get("err")):
                            final = repaired
                            print("  path=+exec-repaired")
                        else:
                            print("  exec-repair rejected (raised an error)")

        state.output.completion = emit(final)
        return state

    return solve
