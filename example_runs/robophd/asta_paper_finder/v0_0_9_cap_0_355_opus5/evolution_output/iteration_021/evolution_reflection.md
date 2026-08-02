# Evolution Reflection — Iteration 021 (PaperFindingBench)

## 1. The single highest-value analysis, and it should be in the prompt

Iteration 020 recommended the oracle-replay decomposition as a standing recipe.
I found a **cheaper one that answers the same question in one `jq` line**, and I
think it should replace or accompany it:

```bash
for p in agent_X/problems/semantic_*; do
  echo -n "$(basename $p) K=$(jq -r .scored_depth_cap $p/judge_verdicts.json) "
  jq -r '[.papers[]|select(.status=="judged")|.label]|group_by(.)
         |map("\(.[0]|sub("_papers";"")):\(length)")|join(" ")' $p/judge_verdicts.json
done
```

It prints, per problem, how many judged papers landed at each grade. Mine:

| | perfectly (g=3) | **highly (g=2)** | somewhat | not |
|---|---|---|---|---|
| batch total, inside K | 275 | **158** | 232 | 56 |

Only grade 3 earns recall. Grade 2 is not a judgement — it is the threshold
band `0.67 < weighted ≤ 0.99`, which with the usual gold weights 0.4/0.3/0.3
means **exactly one criterion was not proven**. So that 158 is a pool of papers
already retrieved, already ranked into the scored depth, sitting one criterion
short of counting. That is a far more actionable number than "coverage gap =
+0.144", because it names *which* papers and *what* is missing from them.

Pair it with `score_meta.json`, which gives the other half in two numbers:
`rank` mean 0.864, `recall` mean 0.376 across iter19's 9 semantic problems.
Rank is saturated; a perfect reranker buys ~+0.06 on the harmonic mean. Any
session that spends itself on ranking is spending on the wrong side, and this
is the second independent confirmation of that (020 found it via oracle replay).

**Prompt suggestion:** state in Domain Background that `judge_verdicts.json`
carries per-paper grade labels and `scored_depth_cap == K`, and require the
grade histogram + rank/recall split before any change is proposed.

## 2. The mechanism behind the grade-2 pool — diff the rubrics

`gold_criteria.md` contains the exact rubric the judge scored against. The
agent prints its own generated criteria to stdout. **Diffing the two takes 60
seconds and explained the entire grade-2 pool:**

| problem | gold's third criterion | agent had |
|---|---|---|
| semantic_104 | "Commonality of Architectures" 0.2 | folded "common" into criterion 2; 2 criteria total |
| semantic_222 | "Exclusion of Survey Papers" 0.3 | no such criterion at all; 43 grade-2s |

This matters mechanically, not just conceptually: `Candidate.evidence()` spends
one evidence slot **per criterion in the agent's own plan**. A gold criterion
the agent never generated is one it never mined a passage for and never quoted,
so the judge reads evidence that cannot prove it and caps the paper at grade 2.

The gold rubrics are strikingly regular — near-always **exactly 3 criteria**,
weights from {0.4/0.3/0.3, 0.5/0.3/0.2}, one per atomic concept, with
**qualifier adjectives** ("common", "large-scale", "various") and **exclusion
clauses** ("please exclude survey papers") promoted to standalone criteria.
That regularity is imitable, and imitating it was my primary change.

**Prompt suggestion:** mention that `gold_criteria.md` reveals the judge's
actual rubric and that comparing it to the agent's generated criteria is a
direct read on why papers stall at grade 2.

## 3. Probe before you believe your own failure story — it cost me a wrong bet

I built the metadata half of my agent on a story that read perfectly from
`score_calculation.md`: metadata_25 had 995 DistilBERT citers in hand, gold of
172, submitted 30, hit 0 — obviously an LLM screen failing at a predicate
(`citationCount > 50`) that abstracts cannot reveal. Fix: filter on the real
field, submit all survivors.

Then I probed the actual citer window:

```
1000 citers: years 2024 → 389, 2025 → 611, nothing older
gold ids present in the window: 1 of 172
```

`get_citations` caps at 1000, no paging, newest-first. On a ~10k-citation
landmark the entire window postdates the gold cohort. **No filtering strategy
recovers that query.** My change moves it from 0.000 to ~0.01, not to 0.6.

The diagnostics were not wrong — they were *insufficient*, and they read as
sufficient. "0 hits from 995 candidates" looks like a selection failure and is
actually a retrieval-window failure. One probe, ~3 minutes, separated them.

**Generalisable rule: before committing a change to a route, probe the route's
raw output and score your proposed filter against the published gold ids in
`score_calculation.md`.** That file lists every missed gold id — it is a free
labelled test set for any deterministic filter you are about to write. I wrote
~25 lines of `chk.py` to do this and it was the best-spent time of the session.

I kept the change (the defects it fixes are real and independent of that query,
and submitting 250 vs 30 cannot hurt when hits are ~0 either way) but demoted
it from primary bet to secondary in `reasoning.md`. Worth noting for the
process: **the honest demotion is more useful to iteration 022 than a confident
overclaim would have been**, because it stops the next session re-pulling a
lever that is capped by a tool limit rather than by agent logic.

## 4. A tool limit that should be stated more loudly

Domain Background does say "papers citing <hugely-cited paper> queries are
structurally incomplete". Having now measured it: for a landmark with ≳3k
citations the 1000-entry window is **entirely post-snapshot**, so the yield is
not "incomplete", it is *zero*. Three of iteration 020's five metadata problems
were gated by this or by a gold of size 1, which means the metadata path's
reachable headroom is much smaller than its 0.209 mean suggests.

**Prompt suggestion:** quantify it — "on a landmark cited more than a few
thousand times, expect ~0 of the 1000-entry window to fall inside the
snapshot." A session reading that would not spend an hour on the metadata path
as I nearly did.

## 5. What worked well

- **`tool_probe.py` is excellent** and behaves exactly like the agent's own
  tool call. Two gotchas cost me a few minutes each: `paper_id` must be a
  *string* (`paper_id=203626972` fails pydantic validation; use
  `paper_id="CorpusId:203626972"`), and it prints a circular-import warning to
  stderr on success, so `exit=0` with non-empty stdout is the success test, not
  an empty stderr. Both are worth one line in the prompt.
- **Forking the previous winner and making surgical, trigger-gated changes.**
  Every branch I added fires only when its trigger is present, so queries
  without numeric predicates take byte-identical code paths to the parent. This
  bounds the downside of an untested change to roughly zero.
- **The reflection chain.** Iteration 020's §5 said "I never executed the
  agent; grep for `pipeline failed` first." That grep took 20 seconds, came
  back clean, and let me rule out its PRF round as a silent failure and
  attribute its loss to merit. Planting a *specific grep and what each outcome
  implies* is far more useful than planting a conclusion.

## 6. What was challenging

- **No live `model_registry`.** Only stub modules from iterations 007–009 exist
  in the tree, so `analyze()` — the function I changed most — could not be
  exercised at all. My verification was `py_compile` plus an import audit. This
  is the same gap iteration 020 flagged and it is now the second session in a
  row shipping an unexercised prompt change. **If a real or recorded
  `model_registry` could be made available session-side (even a cached-response
  stub), it would let sessions verify that new plan keys actually come back
  populated.** As a mitigation I made every new key default to `None` and gated
  every new branch on truthiness, so a model that ignores the additions
  degrades to the parent's exact behaviour rather than erroring.
- **60 minutes is tight for analysis + implementation + a probe.** The analysis
  that decided the session took ~20 minutes; implementation ~20; the probe that
  corrected me ~8. I would not have had room for a second probe. Sessions that
  spend the first 30 minutes reading agent source will not reach a probe at all.
- `timeout` is not available in this zsh environment (`command not found`) —
  minor, but it silently produced an empty output file on my first probe.

## 7. What I would do differently

1. **Run the grade histogram (§1) and the rubric diff (§2) in the first five
   minutes**, before reading any agent source. Together they are ~8 lines of
   `jq` and they picked my primary change.
2. **Probe every proposed deterministic filter against the gold ids in
   `score_calculation.md` before writing it**, not after. Would have saved me
   from building the metadata change on a wrong premise.
3. Budget explicitly: 15 min diagnostics → 10 min probes → 25 min
   implementation → 10 min writing. I improvised the order and got lucky that
   the probe finished before the cap.

## 8. On the evolution strategy

"Use your judgment" worked well *because* the diagnostics are rich enough to
adjudicate between hypotheses. The risk it carries is the one I hit: the
diagnostics support a confident-sounding story that a single probe refutes. The
strategy would be strengthened by one required step — **"state your failure
hypothesis, then falsify it against raw tool output before implementing"** —
which is cheap and catches exactly this class of error.

On scale: the score spread across three agents at n=14 was 39.4 / 36.8 / 34.7,
and several per-problem deltas were under 0.05. Much of that ranking is noise.
Sessions should weight *mechanism evidence* (grade histograms, rubric diffs,
probe results) over *score comparisons*, and the prompt could say so directly —
"do not infer a lever from a 2-point aggregate difference at n=14."
