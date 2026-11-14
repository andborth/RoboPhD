"""DS-1000 solver: dual-model generation + sandbox cross-check + critique + adjudication.

Pipeline (iter3: token-guard variant of iter2_crosscheck_adjudicate):
  1. CLAUDE_SONNET_4_6 and GPT_5_4 each generate a solution (parallel).
  2. Both candidates are executed in the sandbox against the visible context;
     the target variable is compared with type-aware, tolerance-aware equality.
  3. If both run and agree, a cheap GPT_5_4 checklist critique gates submission.
  4. On disagreement / errors / critique flags, GPT_5_5 adjudicates with full
     execution evidence and writes the final code.
  5. NEW: the adjudicated code also passes through the critique gate; if flagged
     (e.g. a required library token is missing), one GPT_5_5 repair pass runs.
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

from model_registry import CLAUDE_SONNET_4_6, GPT_5_4, GPT_5_4_MINI, GPT_5_5

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
    model.export("export/1") fails the grader even though it also writes a SavedModel."""

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

If none apply, reply exactly: OK
Otherwise reply with 1-3 short bullets naming the flaw(s). No stylistic comments.

Problem:
{problem}

Candidate code (will be appended to the context program):
{code}
"""

ADJUDICATE_PROMPT = """\
You are the final reviewer for a DS-1000 data-science problem. Two candidates were
generated by different models and executed against the visible example context.
Decide the FINAL code to submit: pick one candidate, fix one, or write a better
solution yourself.

""" + RULES + """

Problem:
{problem}

Candidate A:
{cand_a}

Candidate B:
{cand_b}

Execution on the visible example{mock_note}:
{exec_info}
{critique_section}
Note: if both executions failed with environment-looking errors (protobuf /
MessageFactory noise, GPU/CUDA, deprecation spam — common for TensorFlow in this
sandbox), treat the execution evidence as inconclusive: judge on code semantics and
on which candidate literally uses the API the problem names (rule 11), and do not
steer away from the canonical idiom just because the sandbox run errored.

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

_eq = False
if len(_results) == 2 and _results[0]["err"] is None and _results[1]["err"] is None:
    try:
        _eq = bool(_same(_vals[0], _vals[1]))
    except Exception:
        _eq = False
print("VERDICT::" + _json.dumps({"results": _results, "equal": _eq}))
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
    """Cheap checklist critique; empty string or OK-prefixed means clean."""
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

        # --- Stage 1: dual generation (parallel, different families) ---
        gen_prompt = GEN_PREAMBLE + problem
        raw_a, raw_b = await asyncio.gather(
            call_model(CLAUDE_SONNET_4_6, gen_prompt, 1500),
            call_model(GPT_5_4, gen_prompt, 1500),
        )
        cand_a = normalize_candidate(clean_code(raw_a), func_form)
        cand_b = normalize_candidate(clean_code(raw_b), func_form)
        if not cand_a and not cand_b:
            raise RuntimeError("both generations empty")
        if not cand_a:
            cand_a = cand_b
        if not cand_b:
            cand_b = cand_a

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
                    code=build_harness(harness_ctx, [cand_a, cand_b], target, call_expr)
                )
                m = re.search(r"VERDICT::(\{.*\})", str(out))
                if m:
                    verdict = json.loads(m.group(1))
            except Exception:
                print("  harness failed:", traceback.format_exc()[-300:])
        res_a = verdict["results"][0] if verdict else None
        res_b = verdict["results"][1] if verdict else None
        ok_a = bool(res_a) and res_a["err"] is None
        ok_b = bool(res_b) and res_b["err"] is None

        if verdict is None:
            agree = norm_text(cand_a) == norm_text(cand_b)
        elif is_mpl:
            agree = ok_a and ok_b and norm_text(cand_a) == norm_text(cand_b)
        else:
            agree = ok_a and ok_b and verdict["equal"]
        print(f"  exec: ok_a={ok_a} ok_b={ok_b} agree={agree} "
              f"(verdict={'yes' if verdict else 'no'})")

        # --- Stage 3: critique gate on agreement ---
        critique = ""
        if agree:
            critique = await run_critique(problem, cand_a)
            if not critique:
                print("  path=agree+critique-OK")
                state.output.completion = emit(cand_a)
                return state
            print(f"  critique flagged: {critique[:200]}")

        # --- Stage 4: adjudication ---
        exec_info = (
            fmt_exec("A", res_a)
            + "\n"
            + fmt_exec("B", res_b)
            + (
                f"\nResults equal under tolerant comparison: {verdict['equal']}"
                if verdict and not is_mpl
                else ""
            )
        )
        body_note = (
            " Your code is the body of f(): indent every line 4 spaces and "
            "end with a return."
            if func_form
            else ""
        )
        adj_prompt = ADJUDICATE_PROMPT.format(
            problem=problem,
            cand_a=cand_a,
            cand_b=cand_b,
            mock_note=(
                " (load_data() was replaced by synthetic mock data; errors may be "
                "artifacts of the mock)"
                if mock_code
                else ""
            ),
            exec_info=exec_info,
            critique_section=(
                f"\nA reviewer flagged the agreed candidate:\n{critique}\n"
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
            final = cand_a if ok_a or not ok_b else cand_b
            print("  path=adjudication-failed, fallback candidate")
            state.output.completion = emit(final)
            return state

        # --- Stage 5: critique gate on the adjudicated code + one repair pass ---
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
        state.output.completion = emit(final)
        return state

    return solve
