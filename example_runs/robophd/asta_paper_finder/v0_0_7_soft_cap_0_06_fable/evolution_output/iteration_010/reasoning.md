# Iteration 10 — Analysis and Plan

Name: iter10_cite_expand

## What the iteration-9 data shows

**Standings.** iter4_judge_sim_ranker 35.59 ($0.041), iter9_verified_head 34.63
(raw 34.80, $0.0605 → 0.17-pt cost penalty), iter6_grade3_rescue 33.06 ($0.054).
The three agents are within noise of each other on raw F1, but iter9 had the most
solo wins (6, incl. semantic_174 at 0.697 vs 0.434 — the head-verify graft doing
exactly what it was designed to do). iter9 is the base; its $0.0005 cost overage
must be fixed, and the batch's biggest losses are structural, not noise.

**Where the score is actually lost** (iter9 numbers, K = judge depth):

| problem | score | binding term | diagnosis |
| --- | --- | --- | --- |
| metadata_25 | 0.010 | recall 1/172 | "papers citing DistilBERT after 2022, >50 citations": gold has 172 ids; `get_citations` returns only 1000 of ~8k citers (skewed away from the recent/highly-cited ones), the mention channel added just 150 candidates from 3 searches, and a `get_paper_batch` failure (`'NoneType' object is not iterable`, both attempts) silently dropped 7 of the last 34 candidates. Submitted 27, hit 1. |
| semantic_145 | 0.049 | rank 0.03 | gold criteria include **"Earliest instance w=0.2"** — the planner dropped the earliest aspect from its reconstructed criteria, and the year-ascending tiebreak in `_key2` sits *before* the weighted score, so old-but-weak papers were ordered above the actual grade-3 (judge found it at position 6 of 6 — worst possible order). |
| semantic_203 | 0.146 | recall 2/24 | pool simply lacked grade-3s: 2 Perfect / 9 Highly / 10 Somewhat in the judged 24. |
| semantic_100/22/170/193 | 0.20–0.38 | recall 0.14–0.25 at K=118–204 | rank is decent (0.6–0.75); the judged prefix has far fewer grade-3s than K. semantic_170: judge graded **52 Perfect but also 99 Highly** — the corpus holds ~204 relevant papers and we surface a quarter of them. |

**The single dominant pattern:** on large-K semantic queries recall, not rank, is
the binding term everywhere. Every agent already submits 250; what's missing is
genuinely-relevant papers inside the judged prefix. The iteration-9 reflection
called this out explicitly and named the one untried, free lever: **citation-graph
expansion** — references and citers of confirmed-strong candidates are prime
grade-3 material that keyword search misses (different vocabulary, older papers,
follow-up work).

## Changes in iter10_cite_expand (base: iter9_verified_head)

1. **Citation-graph expansion (new recall stage, the headline change).** After
   stage-1 triage + gap-fill, take the 8 strongest candidates (predicted-perfect
   first, then by weighted score), fetch their reference lists and their citers
   (`get_citations`, limit 60 — all free tool calls), hydrate up to 140 new
   corpus-ids via `get_paper_batch`, stage-1-triage them, and merge into the pool
   before head selection. Directly attacks the recall ceiling on large-K queries
   (semantic_170/193/100/22-shaped, the majority of the batch) and the thin-pool
   case (semantic_203). Cost: ~6 extra mini triage chunks ≈ +$0.004/query.

2. **Qualifier criteria in the plan prompt (semantic_145 fix, part 1).** The
   real judge's criteria include explicit qualifier criteria like "Earliest
   instance w=0.2". The planner is now told: when the request carries a
   superlative/temporal qualifier (earliest, first, original, most recent),
   emit it as its OWN criterion with weight 0.15–0.25. The judge-sim ranking
   then optimizes for the same thing the judge grades.

3. **Ordering fix (semantic_145 fix, part 2).** In `_key2`, the year-ascending
   "earliest" tiebreak moves from *before* the weighted score to *after* it.
   Old-but-weak candidates can no longer outrank strong ones; year only breaks
   ties among equally-scored papers.

4. **Metadata cites-paper breadth (metadata_25 fix).** The mention channel grows
   from 3 keyword searches to 5 (short name, topic, combined, the cited title's
   distinctive words, "using <short name>") and the scoped snippet search limit
   rises 50→100. Reference verification becomes **fail-open for candidates that
   explicitly mention the cited work's short name in their title/abstract** — a
   paper whose abstract names DistilBERT all but certainly cites it, and S2
   reference lists are sometimes truncated or unfetchable; hard-dropping those
   candidates traded recall for nothing. (Gold=172 means this query shape is
   recall-dominated; precision loss from a few false mentions is second-order.)

5. **`get_paper_batch` bisect fallback (metadata_25 fix, part 3).**
   `_fetch_references` now recursively splits a failing batch in half (down to
   size 3) instead of losing the whole group to one poison id / transient
   `'NoneType'` server error.

6. **Cost trim to get back inside the free zone.** iter9 ran $0.0605. HEAD
   150→120 (fewer stage-2 sim targets + enrichment calls), VERIFY_TOP 24→16
   (the GPT_5_4 verify keeps its mechanism — it won semantic_174 — at ~2/3 the
   cost). Together ≈ −$0.009/query, offsetting the +$0.004 expansion cost.
   Projected mean ≈ $0.052–0.055, safely under $0.06.

7. **Soft deadline 1150→1300s.** Observed max wall clock was 1135s against a
   1740s budget; the expansion stage needs ~60s and was worth protecting from
   the deadline gate. All stages remain individually deadline-gated.

Everything else — transport retry wrapper, alias hedging on specific queries,
cites-venue verification, never-empty fallbacks, band ordering, grade-2 rescue,
evidence assembly — is inherited unchanged from iter9_verified_head.

## Why this should score higher

- The recall term is the binding constraint on ~8 of 11 semantic queries in the
  iter-9 batch, and citation-graph expansion is the one retrieval modality no
  prior agent has tried; it is free at tool level and cheap at triage level.
  Even a 20% relative recall gain on large-K queries is worth ~+2 raw points.
- Two of the three worst problems (metadata_25 at 0.010, semantic_145 at 0.049)
  have identified root causes with targeted fixes; both shapes ("papers citing
  <famous paper> + filters", "earliest paper that …") recur across batches.
- The cost fix converts iter9's standing 0.17-pt penalty into zero while keeping
  the verify mechanism that produced its largest solo win.
- All new stages are additive/fail-open and deadline-gated, so the downside on
  queries the base already handles well is bounded.
