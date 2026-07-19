# Iteration 13 — Analysis and Plan

Name: iter13_grade_view

Base: `iter12_body_conjunction` (iteration-12 winner, 45.83, mean cost $0.0471).

## 1. What the data says

### Standings and where the points are

| Agent | Score | Cost |
| --- | --- | --- |
| iter12_body_conjunction | 45.83 | $0.0471 |
| iter11_tail_saturate | 44.04 | $0.0526 |
| iter6_grade3_rescue | 43.28 | $0.0435 |

All four `specific_f1` queries scored 1.000 for all three agents; both
`metadata_f1` queries were weak (0.125 / 0.000). Everything that separates
agents lives in the nine semantic queries, which score 0.09–0.49. That
matches the test mix (73% semantic), so semantic is where to work.

### The scoring shape: recall is the only term that moves

`score = harmonic(rank, recall)`, where `recall = |grade-3 papers in the
first K| / K` and `rank` is a lower-bound-corrected nDCG over the judged
grades. Only grade 3 counts toward recall — grade 2 ("Highly Relevant")
earns exactly nothing — and grade 3 requires essentially *every* weighted
criterion judged Perfectly Relevant, since it needs `weighted > 0.99`.

Observed iteration-12 values: rank 0.54–0.78, recall 0.05–0.23. Because
the harmonic mean is dominated by its smaller term, this asymmetry is
severe. I checked it numerically on a semantic_43-shaped grade vector
(`smoke_test.py`, section 9):

- converting the grade-2s to grade-3: **0.116 → 0.472** (4.1x)
- perfecting the ordering, grades unchanged: **0.116 → 0.118** (+2%)

So ranking work is nearly exhausted, and every point left is in
**grade-2 → grade-3 conversion**. The observed grade mass is exactly
there: semantic_170 returned 131 Highly against 49 Perfect; semantic_43's
judged top-16 was 4 Highly, 1 Perfect, 8 Somewhat.

I also extracted K from all 46 observed semantic queries: median 52,
p25 20, p75 134, min 6. The judge reads only the first K submissions
(`cap_judge_to_estimate: True`), so positions 0–20 are judged on every
single query while position 100+ is judged on roughly a quarter of them.
Effort should be concentrated at the top far more than it was.

### The bug that explains the plateau

Submitted evidence runs ~4000 chars — title, tldr, abstract cut to 1300,
then five 600-char body snippets. Every internal grader (stage-2 judge
simulation, the grade-2 rescue round, the GPT_5_4 head verify) graded
`_cut(evidence, SIM_CUT=600)`. I confirmed against the actual iteration-12
submissions: at char 600 all four inspected papers were still mid-abstract.

**No fetched body snippet has ever reached an internal grader.** The whole
enrichment cascade — fetch criterion-probe passages to prove what the
abstract leaves unstated, then re-grade — has been an open loop: it fetches
the proof, then grades a text that excludes it. The telemetry says so
plainly and was misread as a retrieval limit rather than a plumbing fault:

```
semantic_43: stage1 predicted-perfect: 0
             rescue promoted 0 to predicted-perfect
             head verify: 0/24 confirmed perfect by GPT_5_4
```

Zero at every stage — while the real judge, reading the full evidence,
returned 1 Perfect and 4 Highly on those same papers. The graders were
grading abstracts and the judge was grading abstracts-plus-passages.

### The metadata zero

`metadata_4` ("a Nature portfolio papers by David Harel") scored 0.000 for
all three agents; iteration 2's much cruder agent scored 0.500 on it. The
plan resolves 452 author papers and `venue_constraint` only — `venues` is
empty, so `_venue_ok_substring` is vacuous and the LLM classifier is the
sole gate. That classifier ran on `sorted(distinct)[:120]`: an
**alphabetical** truncation. With 452 papers spanning well over 120
distinct venues, the N-initial Nature venues sit past the cut. 46 papers
submitted, 0 of 3 gold. (I confirmed gold id 14549281 is "A metric for
odorant comparison", *Nature Methods* — squarely inside the retrieved
author set, dropped by the filter.)

## 2. Changes

**1. The grade view (headline).** New `_grade_view()` builds what internal
graders read: title(110) + abstract(300) + up to 4 criterion-matched body
snippets @180 chars. Stage-2 sim, rescue and verify all switch to it. This
closes the loop: a fetched passage proving a weak criterion can now
actually promote a paper, and the ranker's estimates stop being
abstract-only. Everything downstream — rescue, verify, ordering — was
built on a signal that structurally could not see the evidence it was
built to evaluate.

**2. Full-criterion coverage in the always-judged region.** Positions 0–35
now get one probe-scoped snippet call per criterion — *all* criteria, not
just the stage-1-weak ones. A criterion the abstract happens to satisfy
still gets demoted when the submitted text doesn't state it, and grade 3
needs all of them. Positions 36–99 keep the weak-criteria policy. Snippet
calls are free; this costs only wall clock (~250 calls at concurrency 10,
well inside the 29-minute budget).

**3. Two probe phrasings per criterion.** The planner emits `probe` and
`probe2` with explicitly different vocabulary. Enrichment uses `probe`;
the rescue round retries with `probe2`. Re-issuing the phrasing that
already failed just retrieves the same miss.

**4. Verify orders instead of only boosting.** The GPT_5_4 pass covers the
top 26 (30 on thin pools, up from 16/24) and its per-criterion weight now
overrides the cheap graders' for every paper it read, rather than only
floating confirmed perfects. Banding is now: verified-perfect, then
cheap-perfect-only, then everything else — so GPT_5_4-confirmed coverage
outranks an unverified guess, but a paper verify *demoted* also can't
outrank one it never read. This targets the semantic_192 shape (top-20 was
13 Somewhat and 4 Perfect while 15 Perfect papers existed further down).

**5. Evidence packing.** Drop the tldr when an abstract and ≥3 snippets
exist — it paraphrases the abstract and costs one of only 8 passage slots;
abstract 1300→1150; snippet room 5→6.

**6. Venue filter chunked, not truncated.** Up to 400 distinct venues,
classified in chunks of 100.

**7. Cost offsets.** POOL_CAP 360→320, POOL_CAP_TOTAL 420→380, EXPAND_CAP
120→100, and stage-2 sim runs on the top 55 of the head instead of all 100
(nDCG discounts position 90 to a sixth of position 1; half of queries never
judge that deep). Net modelled change ≈ +$0.002/semantic query — the
deeper GPT_5_4 verify is paid for by the sim depth cut and smaller pools —
landing near $0.053 at a 73% semantic mix, inside the $0.06 free zone.

## 3. Why this should score higher

The argument is not that a new heuristic might help; it's that the
existing cascade has been running disconnected. Three separate LLM stages
(sim, rescue, verify — roughly 45% of per-query spend) have been paying
full price to grade a text that omits the evidence they exist to check.
Reconnecting them makes previously-inert machinery functional at no extra
cost, and it compounds with change 2, which supplies the passages that
loop now consumes.

The direction is also the one the arithmetic endorses. Both headline
changes push on grade-2 → grade-3 conversion in the top ~35 positions,
which the numeric check values at ~4x, rather than on ordering, which it
values at ~2%. The prior iterations moved ordering repeatedly (tail sweep,
band reordering, boost keys) for gains in the 1–2 point range; that is
what the harmonic mean predicts for ordering work, and it is why the
plateau at 44–46 has held.

Residual risks, in order of concern:

- **The judge may be stricter than GPT_5_4 on the same text.** Then
  conversions land as grade 2 anyway and the gain is smaller than modelled.
  The change is still not negative: the graders read strictly more relevant
  text than before, so the ranking signal improves regardless.
- **Longer evidence could dilute.** Mitigated by dropping the redundant
  tldr and by capping the grade view at ~1200 chars; the grounding check is
  per-passage and independent, so extra passages cannot invalidate others.
- **Wall clock** from ~250 enrichment calls. The existing SOFT_DEADLINE /
  TAIL_DEADLINE gates already degrade gracefully, and concurrency 10 sits
  exactly at the documented endpoint budget.

`smoke_test.py` covers the changed surface offline (grade view contains
snippets, evidence stays ≤8 verbatim-grounded passages, coverage ordering,
probe fallbacks, degenerate docs, constant consistency, venue chunking,
and the scoring arithmetic above) — 33 checks, all passing.
