Name: iter16_feedcast

# Iteration 016 — analysis and plan

## What the data shows

Iteration 15 (14 queries: 4 specific, 1 metadata, 9 semantic): bodycast 0.590 raw, crosscast
0.584, headcast 0.518. The batch was an easy draw (the same lineage scored 0.29–0.45 on
iterations 10–14), so the absolute level means little; the per-problem mechanisms do.

### 1. bodycast is on the wrong side of the cost cliff for the test mix

| | specific/metadata | semantic |
| --- | --- | --- |
| bodycast mean cost | $0.014 | **$0.0906** (0.064–0.106) |
| crosscast mean cost | $0.013 | $0.0753 |

The batch mean ($0.0631) only looked fine because 5 of 14 queries were cheap exact-match
queries. The held-out set is 73 % semantic, so bodycast's expected mean is
0.73·0.0906 + 0.27·0.014 ≈ **$0.070**, i.e. ~1.1 fully-wrong-query equivalents of penalty.
Crosscast lands at ≈ $0.059. The free-zone bound for the semantic mean is
(0.063 − 0.27·0.014)/0.73 ≈ $0.081; I target ≈ $0.075 for margin. Cost is 80 % GPT-5.4-mini
grading (input-dominated: ~900 chars per candidate, 520–900 candidates per query) and 20 %
the GPT-5.4 head rerank.

### 2. Recall, not rank, is the semantic bottleneck — and the pool is the limit on large K

Per-query components (bodycast): rank 0.63–0.96, recall 0.14–0.34 on 7 of 9 queries. Judge
labels inside the scored window:

| query | K | perfect | highly | somewhat | not | our grade dist (3/2/1) |
| --- | --- | --- | --- | --- | --- | --- |
| semantic_193 | 180 | 26 | 29 | 55 | 70 | 12 / 38 / 93 |
| semantic_226 | 138 | 22 | 21 | 93 | 2 | 6 / 47 / 232 |
| semantic_8 | 222 | 73 | 2 | 87 | 60 | 8 / 72 / 227 |
| semantic_229 | 62 | 21 | 2 | 39 | 0 | 4 / 20 / 86 |
| semantic_160 | 30 | 6 | 13 | 10 | 1 | 23 / 82 / 137 |

On the large-K queries the judge finds far more perfect papers than our grader does (193: 26 vs
12; 229: 21 vs 4), and the judged window runs deep into our grade-1 and even ungraded tiers
(193: positions 144–250 are ungraded fusion leftovers, 54 of 80 judged there are Not). Stage-2
citation expansion, the only pool-widening step after the initial searches, is erratic: 14 new
citers on 193, 46 on 160 (get_citations timeouts and sparse citation graphs), versus 300+ on
the queries that scored well (87, 112, 8). The pool after the planner's 14 keyword + 8 snippet
queries simply does not contain enough of the ~140–230 relevant papers.

### 3. Two smaller mechanisms

- **semantic_140**: known-good 277244603 is newer than the 2025-06-01 cutoff (tool error on
  `get_paper`) — unreachable, so recall on that query is capped at 11/12 by construction.
- **semantic_193** ("published on or after 2023"): the planner made "The paper was published on
  or after 2023" a grading criterion. The gold criteria never contain dates (checked on every
  training query with a date phrase), and the grader cannot verify a date from an abstract, so
  this criterion only demotes true positives to grade 1–2 in our own ordering.

## Base and changes

Base: **iter15_bodycast** (best raw mean; its body-passage grading is mechanism-backed and I
keep it). `agent.py` = bodycast + `patch_agent.py` (uniqueness-asserted string patches,
`ast.parse`-checked) + two small in-session patches noted at the end of that file.

1. **Stage-2 pseudo-relevance feedback.** The titles of the top 6 grade-3 seeds (grade-2 fill
   when fewer) are issued as `search_papers_by_relevance` keyword queries (limit 40, run
   concurrently with the existing `get_citations` calls; punctuation stripped). New papers join
   the expansion list with the same 0.6 weight as citers, ranked by how many seed lists hit
   them, and are graded within the stage-2 cap. A known-relevant title is the most precise
   topical query available and reaches papers whose wording the planner's generic queries
   missed; it is free, fast (~1–2 s), and independent of the citation graph that starved stage 2
   on 193/160. The stdout line `stage2 graded>=2 ... (neighbour-only: n)` records how many
   grade≥2 papers only this step found, so the next session can measure it.
2. **Cost trims to put the semantic mean inside the free zone.** `GRADE_MAX` 680→600,
   `GRADE_EXPANSION` 160→120, `HEAD_RERANK` 36→32, `GRADE_ABSTRACT_CHARS` 440→400. Estimated
   saving ≈ $0.014 per semantic query → semantic mean ≈ $0.077, test-mix mean ≈ $0.060. The
   yield-adaptive stage-1 loop is unchanged (the third round is now 80 instead of 160), the head
   rerank still covers the whole window of every K ≤ 32 query, and the expansion cap is
   above what stage 2 produced on 4 of 9 queries.
3. **Planner criteria are content-only.** The plan prompt now says date/recency/venue/author
   requirements must never become a criterion (they cannot be judged from the text). Date
   phrases appear in ~4 of 48 training semantic queries; the change is a strict no-op elsewhere.

Specific and metadata paths are untouched (specific_7/9/24 at 1.0, metadata_31 0.75 for both
lineage agents; specific_33's 0.4 vs 0.667 was picker noise on an ambiguous nickname).

## Why this should score higher

- Under the real 73 % semantic mix, bodycast pays ~1.1 query-equivalents of cost penalty;
  this agent pays none. That alone is worth ≈ 1/N·110 points for a batch of N queries.
- Large-K semantic queries are lost in the pool (section 2). Title-neighbour expansion adds a
  second, citation-independent source of candidates exactly where our grade-3 seeds are, at no
  LLM cost beyond the (reduced) stage-2 cap; grade≥2 papers it finds move ahead of the grade-1
  and ungraded tiers that currently fill positions 60–250.
- The small-K machinery (head rerank with body passages, criterion-targeted evidence) that won
  semantic_140/155/160 for bodycast is kept intact.

## Smoke tests (`smoke/`, harness from iteration 15; no judge available)

Results are appended below after the runs finish.

### Results

Run 1 — `smoke/log_semantic_193.txt` (patches 1–2 only, planner unchanged; K=180 query where
bodycast scored 0.242):

- stage 1: 403 graded (360 + 43 forced), yield stop after one round — same shape as bodycast.
- stage 2 (new): `6 title searches -> 63 new neighbours`; 10 seeds → 171 new citers+neighbours,
  120 graded; `graded>=2: 4/120 (neighbour-only: 2)`. bodycast's stage 2 on this query graded
  only 14 citers. Neighbour yield (2/63) matches citer yield (2/57); the step is cost-neutral
  where citers already fill the cap and adds ≤ 120 graded candidates where they do not.
- head rerank line: **every** head paper rated `[3, 1, 3]` — criterion 2 was the planner's
  "published on or after 2023", which no abstract can establish, so 0 of 32 head papers could be
  predicted grade 3 and our own ordering of this query was driven by a non-content criterion.
  This is the mechanism behind bodycast's 12 grade-3 / 38 grade-2 on a query where the judge
  found 26 perfect + 29 highly in the window.

Run 2 — `smoke/log_semantic_193_v2.txt` (final file: + planner patch, GRADE_ROUND1 320):

- planner criteria are now content-only: "about adaptive query expansion", "uses LLMs for
  query expansion or reformulation", "connects adaptive query expansion with retrieval",
  "proposes/studies/evaluates a method rather than mentioning it in passing".
- stage 1: 354 graded (320 + 34 forced), yield stop; grade distribution `{3: 13, 2: 27, 1: 98}`
  vs run 1's `{3: 10, 2: 22, 1: 89}` with the same pool size (1494).
- head rerank: ratings are now `[1, 3, 3, 3]` for most of the head (criterion 1 "about
  *adaptive* query expansion" is read literally by GPT-5.4; the judge, which found 26 perfect
  papers here, is more lenient). The ordering is no longer hostage to a date criterion.
- stage 2: the three title searches that reported in the log timed out at 60 s — the keyword
  backend was slow at that moment (a session-side probe of the same title also hung) — so run 2
  added 0 neighbours and the step fell back to citers alone, as designed. The timeout is raised
  to 90 s (it runs in parallel with the 90 s `get_citations` calls, so no wall-clock is added)
  and the line now prints per-seed hit counts and the seed titles for the next session.

The smoke harness has no judge, so neither run yields an F1; what the runs establish is that
the new stage-2 step and the planner change behave as intended end-to-end and that the per-query
grading volume dropped (403 → 354 in stage 1, 171 → ≤ 120 in stage 2).
