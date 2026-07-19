"""Offline smoke test for iter14_crit_coverage.

Stubs `inspect_ai` and `model_registry`, execs agent.py, and unit-tests the
helpers this iteration changed. No network, no LLM calls.

Run: /opt/anaconda3/envs/robophd_demo/bin/python smoke_test.py
"""

import sys
import types

# ---- stub the imports agent.py needs -------------------------------------
for name in ("inspect_ai", "inspect_ai.solver", "inspect_ai.tool", "inspect_ai.model", "model_registry"):
    sys.modules.setdefault(name, types.ModuleType(name))

sys.modules["inspect_ai.solver"].solver = lambda f: f
sys.modules["inspect_ai.solver"].Generate = object
sys.modules["inspect_ai.solver"].TaskState = object
sys.modules["inspect_ai.tool"].ToolDef = object
sys.modules["inspect_ai.model"].GenerateConfig = object
for h in ("GPT_5_4", "GPT_5_4_MINI", "CLAUDE_SONNET_4_6"):
    setattr(sys.modules["model_registry"], h, object())

A = {}
exec(compile(open("agent.py").read(), "agent.py", "exec"), A)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


# ---- the real semantic_7 criteria (the query this iteration targets) -----
SEM7 = [
    {"name": "Large Language Models (LLMs)", "weight": 0.3,
     "description": "The paper must discuss or evaluate large language models (LLMs), specifically in the context of their performance in text summarization tasks.",
     "probe": "large language models summarization", "probe2": "LLM summarizer"},
    {"name": "Text Summarization", "weight": 0.3,
     "description": "The paper must focus on text summarization as the primary task being evaluated, with LLMs being applied to this task.",
     "probe": "text summarization task", "probe2": "abstractive summarization"},
    {"name": "Reference-Based Human Evaluation", "weight": 0.2,
     "description": "The paper must include an analysis or discussion of reference-based human evaluation methods for assessing the quality of text summarization outputs from LLMs.",
     "probe": "reference-based human evaluation", "probe2": "compared against reference summaries"},
    {"name": "Reference-Free Human Evaluation", "weight": 0.2,
     "description": "The paper must include an analysis or discussion of reference-free human evaluation methods for assessing the quality of text summarization outputs from LLMs.",
     "probe": "reference-free human evaluation", "probe2": "without reference summaries"},
]

print("=== 1. distinctiveness weighting separates the near-identical pair ===")
vocabs = A["_crit_vocab"](SEM7)
check("4 vocabs", len(vocabs) == 4)
check("'based' unique to ref-based", vocabs[2].get("based", 0) > vocabs[2].get("evaluation", 1))
check("'free' unique to ref-free", vocabs[3].get("free", 0) > vocabs[3].get("evaluation", 1))
check("'evaluation' shared between the pair", vocabs[2].get("evaluation", 1) <= 0.5)
check("'summarization' near-zero weight (in all 4)", max(v.get("summarization", 1) for v in vocabs) <= 0.25)

print("=== 2. the niche pair is DECIDABLE: right text wins its own criterion ===")
mt = A["_crit_match"]
generic = A["_content_words"]("We conducted a human evaluation of summary quality produced by large language models.")
based_txt = A["_content_words"]("We use reference-based protocols scoring against gold reference summaries.")
free_txt = A["_content_words"]("We adopt a reference-free evaluation requiring no gold summary.")
# the decisive property: each specific text beats the generic one on ITS criterion
check("ref-based text beats generic on ref-based", mt(based_txt, vocabs[2]) > mt(generic, vocabs[2]))
check("ref-free text beats generic on ref-free", mt(free_txt, vocabs[3]) > mt(generic, vocabs[3]))
# ...and beats the OTHER specific text on its own criterion (the old raw-overlap
# rule could not do this — the two criteria share every word but one)
check("ref-based text preferred for ref-based", mt(based_txt, vocabs[2]) > mt(free_txt, vocabs[2]))
check("ref-free text preferred for ref-free", mt(free_txt, vocabs[3]) > mt(based_txt, vocabs[3]))
check("match scores are normalised to [0,1]",
      all(0.0 <= mt(t, v) <= 1.0 for t in (generic, based_txt, free_txt) for v in vocabs))

print("=== 3. _cov_score is weighted and bounded (TELEMETRY ONLY — see calibrate.py) ===")
cs = A["_cov_score"]
check("empty text -> 0.0", cs("", SEM7) == 0.0)
check("no criteria -> 0.0", cs("anything", []) == 0.0)
check("scores stay in [0,1]", all(0.0 <= cs(t, SEM7) <= 1.0 for t in
      ("", "large language models", "reference-free reference-based human evaluation summarization")))
check("richer text scores >= poorer text",
      cs("large language models text summarization reference-based reference-free human evaluation", SEM7)
      >= cs("large language models", SEM7))
# guard the refutation: _cov_score must NOT be wired into the ordering key.
# Strip comments first — _key2 carries a long comment explaining WHY it isn't
# there, which a naive substring check would trip over.
_src = open("agent.py").read()
_k2 = _src[_src.index("    def _key2(i: int):"):]
_k2 = _k2[: _k2.index("head_ranked = sorted")]
_k2_code = "\n".join(ln for ln in _k2.splitlines() if not ln.strip().startswith("#"))
check("_cov_score is NOT in _key2 code (calibrate.py refuted it)",
      "_cov(" not in _k2_code and "_cov_score" not in _k2_code)

print("=== 4. global dedup: snippets vs already-emitted title/abstract ===")
TITLE = "Benchmarking Large Language Models for News Summarization"
ABS = "Large language models have shown promise for automatic summarization but the reasons behind their successes are poorly understood."
doc = {
    "corpusId": "1", "title": TITLE, "abstract": ABS,
    "_snippets": [
        ABS,                                   # exact duplicate of the abstract
        TITLE,                                 # exact duplicate of the title
        ABS[:60],                              # contained in the abstract
        "We adopt a reference-free evaluation requiring no gold summary.",
        "We use reference-based protocols scoring against gold reference summaries.",
    ],
}
ev = A["_evidence"](doc, SEM7)
ps = [p for p in ev.split(" ... ") if p.strip()]
check("<= 8 passages", len(ps) <= 8)
norm = A["_norm"]
keys = [norm(p) for p in ps]
dup = sum(1 for i, k in enumerate(keys) if any(k in o or o in k for o in keys[:i]))
check("no duplicate/contained passages survive", dup == 0)
check("the ref-free snippet made it in", any("reference-free" in p for p in ps))
check("the ref-based snippet made it in", any("reference-based" in p for p in ps))
check("dedup freed slots: >= 3 distinct body passages", len(ps) >= 3)

print("=== 5. verbatim grounding: every passage traces to retrieved text ===")
src = " ".join([TITLE, ABS] + doc["_snippets"])
for p in ps:
    check(f"passage verbatim-derivable: {p[:40]!r}", p.rstrip(".") in src or p in src)

print("=== 6. _grade_view also dedups and carries snippets ===")
gv = A["_grade_view"](doc, SEM7)
gvp = [p for p in gv.split(" ... ") if p.strip()]
gk = [norm(p) for p in gvp]
check("grade view has no dup passages", sum(1 for i, k in enumerate(gk) if any(k in o or o in k for o in gk[:i])) == 0)
check("grade view contains a body snippet", any("reference" in p for p in gvp))

print("=== 7. _cover_snippets honours pre-covered criteria ===")
snips = ["We use reference-based protocols scoring against gold reference summaries.",
         "We adopt a reference-free evaluation requiring no gold summary.",
         "Unrelated filler about parallel training throughput."]
cov = set()
out = A["_cover_snippets"](snips, SEM7, 3, covered=cov, vocabs=vocabs)
check("both niche criteria credited", 2 in cov and 3 in cov)
check("room respected", len(out) == 3)
pre = {2}
cov2 = set(pre)
A["_cover_snippets"](snips, SEM7, 1, covered=cov2, vocabs=vocabs)
check("pre-covered criterion not re-proved", 3 in cov2)

print("=== 8. degenerate inputs don't raise ===")
for d in ({}, {"title": ""}, {"title": "T", "_snippets": []}, {"title": "T", "abstract": "A", "_snippets": [""]}):
    try:
        A["_evidence"](d, SEM7)
        A["_grade_view"](d, SEM7)
        A["_evidence"](d, None)
        A["_grade_view"](d, None)
        check(f"degenerate doc ok: {d}", True)
    except Exception as e:  # noqa: BLE001
        check(f"degenerate doc ok: {d} ({e!r})", False)

print("=== 9. constants sane / cost trims applied ===")
check("T1_BODY trimmed to 150", A["T1_BODY"] == 150)
check("SIM_DEPTH trimmed to 48", A["SIM_DEPTH"] == 48)
check("SIM_DEPTH <= HEAD", A["SIM_DEPTH"] <= A["HEAD"])
check("CRIT_DISTINCT_MIN in (0,1]", 0 < A["CRIT_DISTINCT_MIN"] <= 1.0)
check("CRIT_COVER_MIN in (0,1)", 0 < A["CRIT_COVER_MIN"] < 1.0)
check("CONJ_QUERIES >= 1", A["CONJ_QUERIES"] >= 1)

print("=== 10. all-shared-vocab criteria stay satisfiable ===")
same = [{"name": "x", "description": "identical text here", "weight": 0.5, "probe": "", "probe2": ""},
        {"name": "x", "description": "identical text here", "weight": 0.5, "probe": "", "probe2": ""}]
sv = A["_crit_vocab"](same)
check("identical criteria still reachable", A["_covers"](A["_content_words"]("identical text here"), sv[0]))
check("unrelated text does not satisfy them", not A["_covers"](A["_content_words"]("quantum chromodynamics"), sv[0]))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
