"""Offline checks for the iteration-13 agent: no tools, no LLM calls.

Covers the pieces this iteration actually changed — the grade view (which
must now contain body snippets), evidence packing, the criterion-coverage
snippet ordering, the venue-filter chunking, and the ranking key.
"""

import ast
import sys
import types

SRC = "agent.py"

# Stub the two imports the harness provides so the module can be exec'd here.
inspect_solver = types.ModuleType("inspect_ai.solver")
inspect_solver.Generate = object
inspect_solver.TaskState = object
inspect_solver.solver = lambda f: f
inspect_tool = types.ModuleType("inspect_ai.tool")
inspect_tool.ToolDef = lambda t: t
inspect_model = types.ModuleType("inspect_ai.model")
inspect_model.GenerateConfig = object
inspect_pkg = types.ModuleType("inspect_ai")
registry = types.ModuleType("model_registry")
for h in ("GPT_5_4", "GPT_5_4_MINI", "CLAUDE_SONNET_4_6"):
    setattr(registry, h, object())
sys.modules.update(
    {
        "inspect_ai": inspect_pkg,
        "inspect_ai.solver": inspect_solver,
        "inspect_ai.tool": inspect_tool,
        "inspect_ai.model": inspect_model,
        "model_registry": registry,
    }
)

src = open(SRC).read()
ast.parse(src)
mod = types.ModuleType("agent")
mod.__file__ = SRC
exec(compile(src, SRC, "exec"), mod.__dict__)

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


CRIT = [
    {"name": "slot tagging", "description": "The paper must address few-shot slot tagging.",
     "weight": 0.5, "probe": "few-shot slot tagging", "probe2": "few shot slot filling"},
    {"name": "micro-F1", "description": "The paper must evaluate with micro-F1 scores.",
     "weight": 0.25, "probe": "micro-F1 score", "probe2": "micro averaged F1"},
    {"name": "episodes", "description": "Micro-F1 must be averaged across test episodes.",
     "weight": 0.25, "probe": "averaged across test episodes", "probe2": "mean over evaluation episodes"},
]

DOC = {
    "corpusId": 12345,
    "title": "Vector Projection Network for Few-shot Slot Tagging",
    "abstract": "Few-shot slot tagging becomes appealing for rapid domain transfer. " * 30,
    "tldr": {"text": "A projection network for few-shot slot tagging."},
    "_snippets": [
        "We report the micro-F1 score averaged across 100 test episodes for every target domain.",
        "The support set contains k examples per label in the K-shot setting.",
        "All models use the same CNN+CRF architecture over subword representations.",
        "Hyper-parameters were tuned by grid search on the development set.",
        "Results on SNIPS and NER show consistent gains over the strongest baseline.",
    ],
}

# --- 1. the headline fix: the grade view must contain body snippets ---
gv = mod._grade_view(DOC, CRIT)
check("grade view is non-empty", bool(gv))
check(
    "grade view contains the criterion-proving snippet",
    "averaged across 100 test episodes" in gv,
    f"(len={len(gv)})",
)
check("grade view contains the title", DOC["title"][:40] in gv)
n_snip_in_gv = sum(1 for s in DOC["_snippets"] if s[:40] in gv)
check("grade view carries multiple snippets", n_snip_in_gv >= 2, f"({n_snip_in_gv} of 5)")
check("grade view stays compact", len(gv) < 1400, f"(len={len(gv)})")

# The old behaviour, for contrast: the first 600 chars of the submitted
# evidence never reached a snippet. This is the bug being fixed.
ev = mod._evidence(DOC, CRIT)
check(
    "regression guard: old 600-char cut saw no snippet",
    "averaged across 100 test episodes" not in ev[:600],
)

# --- 2. evidence packing ---
check("evidence has at most 8 passages", ev.count(" ... ") + 1 <= 8,
      f"({ev.count(' ... ') + 1})")
check("evidence drops tldr when abstract + >=3 snippets exist",
      "A projection network for few-shot" not in ev)
lean = dict(DOC, _snippets=[], abstract="")
check("evidence keeps tldr when there is no abstract",
      "A projection network for few-shot" in mod._evidence(lean, CRIT))
check("evidence still contains the proving snippet somewhere",
      "averaged across 100 test episodes" in ev)

# every passage must be verbatim-derivable from retrieved text (grounding)
sources = " ".join(
    [DOC["title"], DOC["abstract"], DOC["tldr"]["text"], *DOC["_snippets"]]
)
ungrounded = [p for p in ev.split(" ... ") if p not in sources]
check("every evidence passage is verbatim from retrieved text",
      not ungrounded, f"({len(ungrounded)} ungrounded)")

# --- 3. criterion coverage ordering ---
cov = mod._cover_snippets(DOC["_snippets"], CRIT, 2)
# the 0.5-weight criterion picks first (it gets the K-shot snippet), then the
# micro-F1 criterion claims its proving passage — both land inside room=2
check("coverage serves the heaviest criterion first",
      "K-shot setting" in cov[0], f"({cov[0][:50]!r})")
check("coverage claims the proving snippet within room=2",
      any("micro-F1" in s for s in cov), f"({[s[:40] for s in cov]})")
check("a lone slot goes to the heaviest criterion",
      "K-shot setting" in mod._cover_snippets(DOC["_snippets"], CRIT, 1)[0])
check("coverage returns exactly `room` snippets", len(cov) == 2)
check("coverage never duplicates", len(set(cov)) == len(cov))
check("coverage handles room > available",
      len(mod._cover_snippets(DOC["_snippets"], CRIT, 99)) == 5)
check("coverage handles empty input", mod._cover_snippets([], CRIT, 3) == [])

# --- 4. probe fallbacks ---
check("alt probe uses probe2", mod._crit_query_alt(CRIT, 1) == "micro averaged F1")
noalt = [{"name": "x", "description": "desc here", "weight": 1.0, "probe": "p", "probe2": ""}]
check("alt probe falls back to description", mod._crit_query_alt(noalt, 0) == "desc here")
check("primary probe unchanged", mod._crit_query(CRIT, 0) == "few-shot slot tagging")

# --- 5. degenerate docs must not raise ---
for bad in ({}, {"title": None, "abstract": None}, {"_snippets": [None, ""]},
            {"title": "T", "tldr": "plain string tldr"}):
    try:
        mod._grade_view(bad, CRIT)
        mod._evidence(bad, CRIT)
        mod._evidence(bad, None)
    except Exception as e:
        check(f"degenerate doc {bad}", False, repr(e))
        break
else:
    check("degenerate docs never raise", True)

# --- 6. constants are internally consistent ---
check("FULL_COVER_DEPTH <= HEAD", mod.FULL_COVER_DEPTH <= mod.HEAD)
check("SIM_DEPTH <= HEAD", mod.SIM_DEPTH <= mod.HEAD)
check("verify depth <= SIM_DEPTH (verified papers were sim-ranked)",
      max(mod.VERIFY_TOP, mod.VERIFY_TOP_THIN) <= mod.SIM_DEPTH)
check("POOL_CAP <= POOL_CAP_TOTAL", mod.POOL_CAP <= mod.POOL_CAP_TOTAL)
check("SIM_CUT is gone (graders read the grade view now)",
      not hasattr(mod, "SIM_CUT"))
check("grade view budget is bounded",
      mod.GV_TITLE + mod.GV_ABSTRACT + mod.GV_SNIP * mod.GV_SNIP_MAX <= 1200)

# --- 7. venue chunking: nothing is truncated below the old 120 cut ---
venues = [f"Venue {i:03d}" for i in range(260)] + ["Nature Methods", "npj Digital Medicine"]
groups = sorted({v for v in venues if v})[: mod.CV_LLM_MAX]
check("venue classifier sees the N-initial venues that used to be cut",
      "Nature Methods" in groups and "npj Digital Medicine" in groups,
      f"({len(groups)} distinct)")
check("venue chunk size splits the work", mod.CV_LLM_CHUNK < mod.CV_LLM_MAX)

# --- 8. substring venue matching still works (the union arm) ---
check("Nature Methods matches a 'Nature' venue token",
      mod._venue_ok_substring("Nature Methods", ["Nature"]))
check("SPLASH umbrella still resolves OOPSLA",
      mod._venue_matches("OOPSLA", "SPLASH"))
check("unrelated venue rejected", not mod._venue_ok_substring("ICML", ["Nature"]))

# --- 9. scoring-model sanity: the metric the ranking optimises ---
import math


def dcg(seq):
    return sum(g / math.log(i + 1) for i, g in enumerate(seq, 1))


def rank_of(g):
    lo, hi = dcg(sorted(g)), dcg(sorted(g, reverse=True))
    return 0.0 if hi == lo else (dcg(g) - lo) / (hi - lo)


def score(g, K):
    r = rank_of(g)
    rec = sum(1 for x in g[:K] if x == 3) / K
    return 0.0 if r == 0 or rec == 0 else 2 * r * rec / (r + rec)


base = [2, 2, 3, 2, 1, 1, 2, 1] + [1] * 8          # semantic_43-shaped, K=16
promoted = [3 if x == 2 else x for x in base]       # what grade-2 -> 3 buys
check("grade-2 -> grade-3 conversion dominates the score",
      score(promoted, 16) > 3 * score(base, 16),
      f"({score(base, 16):.3f} -> {score(promoted, 16):.3f})")
perfect_order = sorted(base, reverse=True)
check("perfect ordering alone barely moves the score",
      score(perfect_order, 16) < 1.4 * score(base, 16),
      f"({score(base, 16):.3f} -> {score(perfect_order, 16):.3f})")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
