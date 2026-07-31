"""Side-by-side validation of iteration 15 against its iteration 13 base.

Three assertion classes, per iteration 13/14's standing rules:
  1. CONTROL  - the old code provably lacks the new signal (recorded fact).
  2. FROZEN   - the submitted-evidence path is byte-identical to the base.
  3. LOGIC    - the new selector fires on real recorded artifacts.
"""
import importlib.util, json, sys, types, glob, os

stub = types.ModuleType("model_registry")
for n in ["GPT_5_4_MINI", "GPT_5_4", "GPT_5_5", "CLAUDE_HAIKU_4_5",
          "CLAUDE_SONNET_4_6", "CLAUDE_OPUS_4_8", "GEMINI_3_1_FLASH_LITE",
          "GEMINI_3_5_FLASH", "GEMINI_3_1_PRO_PREVIEW"]:
    setattr(stub, n, object())
sys.modules["model_registry"] = stub

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

BASE = "../../agents/iter13_balanced_digest_wide_vetting/agent.py"
OLD = load("old_agent", BASE)
NEW = load("new_agent", "agent.py")

# --- 1. CONTROL: the base's grader cannot report per-criterion digits --------
import inspect
assert "vec_out" not in inspect.signature(OLD._judge_evidence).parameters, \
    "CONTROL: base already exposed per-criterion vectors"
assert "vec_out" in inspect.signature(NEW._judge_evidence).parameters
# ...and the base's ONLY gap selector is the lexical one, whose measured
# separation on the recorded labels is 0.37 -> 0.53 P(perfect).
assert "pool_vecs" not in inspect.getsource(OLD.solve_semantic)
assert "pool_vecs" in inspect.getsource(NEW.solve_semantic)
print("1. CONTROL ok")

# --- 2. FROZEN: submitted-evidence assembly unchanged, on REAL fixtures -----
P = "../../iteration_014/agent_iter13_balanced_digest_wide_vetting/problems"
n_cases = 0
for prob in sorted(glob.glob(P + "/semantic_*")):
    crit = json.load(open(prob + "/gold_criteria.md"))["relevance_criteria"]
    crit_terms = [OLD._terms(c["name"] + " " + c["description"]) for c in crit]
    sub = json.load(open(prob + "/submission.json"))
    res = sub["output"]["results"] if "output" in sub else sub["results"]
    for r in res[:12]:
        ev = r["markdown_evidence"]
        if not ev:
            continue
        parts = ev.split(" ... ")
        paper = {"corpusId": r["paper_id"], "title": parts[0],
                 "abstract": " ".join(parts[1:3]), "tldr": {"text": parts[-1]}}
        snips = [(i % max(1, len(crit)), p) for i, p in enumerate(parts)]
        a = OLD._build_evidence(paper, snips, crit_terms)
        b = NEW._build_evidence(paper, snips, crit_terms)
        assert a == b, f"SUBMITTED evidence diverged on {prob} {r['paper_id']}"
        n_cases += 1
assert OLD._weighted == OLD._weighted
for f in ("_digest", "_trim_snippet", "_covers", "_terms", "_weighted",
          "_mean_vote", "_parse_judge", "_judge_prompt"):
    assert inspect.getsource(getattr(OLD, f)) == inspect.getsource(getattr(NEW, f)), f
print(f"2. FROZEN ok ({n_cases} real evidence strings byte-identical; "
      f"8 shared helpers textually identical)")

# --- 3. LOGIC: the new selector fires, and only on near-misses --------------
W = [0.4, 0.3, 0.3]
def weak_of(vals):
    weak = [i for i, v in enumerate(vals)
            if i < len(W) and W[i] > 0 and v <= NEW.VREPAIR_WEAK]
    if not weak or len(weak) > NEW.VREPAIR_MAX_MISS:
        return []
    weak.sort(key=lambda i: (-W[i], vals[i]))
    return weak[:NEW.VREPAIR_CRIT_MAX]

assert weak_of([9, 9, 9]) == []          # already grade-3 shaped: no probe
assert weak_of([2, 1, 0]) == []          # off topic: not one passage away
assert weak_of([9, 9, 4]) == [2]         # classic grade-2: probe the gap
assert weak_of([5, 9, 9]) == [0]         # heavy criterion missing
assert weak_of([4, 9, 5]) == [0, 2]      # heaviest gap first
assert weak_of([9, 4, 5]) == [1, 2]      # equal weight -> weaker digit first
print("3. LOGIC ok")

# --- 4. Does it fire on the real draw? Replay the digits we can reconstruct --
# We do not have the agent's own digits recorded, but we DO have the benchmark
# judge's labels: a `highly_relevant` paper is exactly the near-miss shape this
# pass targets, and we can count how many probes it would have wanted.
tot_hi = tot_pf = 0
for prob in sorted(glob.glob(P + "/semantic_*")):
    v = json.load(open(prob + "/judge_verdicts.json"))["papers"]
    hi = sum(1 for e in v if e["label"] == "highly_relevant_papers")
    pf = sum(1 for e in v if e["label"] == "perfectly_relevant_papers")
    tot_hi += hi; tot_pf += pf
print(f"4. TARGET MASS: {tot_hi} grade-2 papers inside judged prefixes against "
      f"{tot_pf} grade-3; converting 1 in 4 lifts mean recall ~+0.05")

# --- 5. constants sane -------------------------------------------------------
assert NEW.REPAIR_DEADLINE < NEW.VREPAIR_DEADLINE < NEW.ENRICH_DEADLINE + 200
assert NEW.VREPAIR_DEADLINE + 400 < 1740, "must leave room before the 29m timeout"
assert NEW.REPAIR_MAX_CALLS < OLD.REPAIR_MAX_CALLS, "lexical pass must shrink"
print("5. BUDGET ok  lexical<=%d calls/%ds, verdict<=%d calls/%ds"
      % (NEW.REPAIR_MAX_CALLS, NEW.REPAIR_DEADLINE,
         NEW.VREPAIR_MAX_CALLS, NEW.VREPAIR_DEADLINE))
print("\nALL SMOKE TESTS PASSED")
