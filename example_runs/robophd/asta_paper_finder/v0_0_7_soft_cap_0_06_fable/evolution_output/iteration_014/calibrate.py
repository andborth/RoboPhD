"""Calibrate CRIT_COVER_MIN against real iteration-13 submissions + verdicts.

The coverage proxy is useful only if it correlates with the judge's actual
grade. This scores every judged paper's SUBMITTED evidence with _cov_score
and reports mean coverage per judge grade, sweeping the threshold.

Run: /opt/anaconda3/envs/robophd_demo/bin/python calibrate.py
"""

import glob
import json
import os
import re
import sys
import types

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

ROOT = "../../iteration_013/agent_iter13_grade_view/problems"
GRADE_ORDER = ["Not Relevant", "Somewhat Relevant", "Highly Relevant", "Perfectly Relevant"]


def load(pdir):
    gold = json.load(open(os.path.join(pdir, "gold_criteria.md")))
    crit = gold.get("relevance_criteria") or []
    sub = json.load(open(os.path.join(pdir, "submission.json")))
    ev = {r["paper_id"]: r["markdown_evidence"] for r in sub["output"]["results"]}
    grades = {}
    for line in open(os.path.join(pdir, "judge_verdicts.md")):
        m = re.match(r"\s*\d+\.\s+(\d+)\s+—\s+(.*)", line.strip())
        if not m:
            continue
        g = m.group(2).strip()
        if "not judged" in g:
            continue
        for name in GRADE_ORDER:
            if g.startswith(name):
                grades[m.group(1)] = name
                break
    return crit, ev, grades


rows = []
for pdir in sorted(glob.glob(os.path.join(ROOT, "semantic_*"))):
    if not os.path.exists(os.path.join(pdir, "judge_verdicts.md")):
        continue
    crit, ev, grades = load(pdir)
    if not crit:
        continue
    vocabs = A["_crit_vocab"](crit)
    for pid, g in grades.items():
        text = ev.get(pid, "")
        if not text:
            continue
        rows.append((g, A["_cov_score"](text, crit, vocabs)))

print(f"judged papers with evidence: {len(rows)}\n")

for thr in (0.20, 0.25, 0.30, 0.34, 0.40, 0.45, 0.50):
    A["CRIT_COVER_MIN"] = thr
    # recompute with this threshold
    vals = {g: [] for g in GRADE_ORDER}
    for pdir in sorted(glob.glob(os.path.join(ROOT, "semantic_*"))):
        if not os.path.exists(os.path.join(pdir, "judge_verdicts.md")):
            continue
        crit, ev, grades = load(pdir)
        if not crit:
            continue
        vocabs = A["_crit_vocab"](crit)
        for pid, g in grades.items():
            t = ev.get(pid, "")
            if t:
                vals[g].append(A["_cov_score"](t, crit, vocabs))
    means = {g: (sum(v) / len(v) if v else float("nan")) for g, v in vals.items()}
    fullfrac = {g: (sum(1 for x in v if x > 0.99) / len(v) if v else float("nan")) for g, v in vals.items()}
    sep = means["Perfectly Relevant"] - means["Somewhat Relevant"]
    print(f"thr={thr:.2f}  mean cov by grade: " +
          "  ".join(f"{g.split()[0][:4]}={means[g]:.3f}" for g in GRADE_ORDER) +
          f"   | frac full: " +
          "  ".join(f"{g.split()[0][:4]}={fullfrac[g]:.2f}" for g in GRADE_ORDER) +
          f"   | Perf-Some sep={sep:+.3f}")

print("\ncounts by grade:", {g: len(v) for g, v in vals.items()})
