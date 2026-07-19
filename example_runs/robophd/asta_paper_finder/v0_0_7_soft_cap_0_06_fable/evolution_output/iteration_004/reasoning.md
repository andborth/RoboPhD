# Iteration 4 Analysis and Plan

Name: iter4-judge-sim-ranker

## What the iteration-3 data shows

**Scores**: iter2_broad_recall_router 35.2, iter3_criteria_evidence_max 32.0 (but iter3 won 6
of 11 semantic queries head-to-head; its aggregate lost on a single crash: specific_44 scored
0.095 because of an `AttributeError` in the specific path's fallback, where iter2 got 1.0).
Both agents are far inside the cost free zone ($0.016 / $0.023 vs the $0.06 threshold), so
there is real headroom to spend more per semantic query.

**The single most important discovery** (from `judge_verdicts.md`): *the judge scores exactly
the first K submitted papers* — every entry past K is marked "beyond scored depth — not
judged". K is the recall denominator (observed 12–222) and is unknown at query time. So the
whole semantic game is: maximize the number of Perfectly-Relevant (grade-3) papers in every
prefix of the list, and submit the full 250 (deep K queries need depth; shallow K queries
ignore the tail for free).

**The dominant semantic loss mode is grade-2 ("Highly Relevant")**. Grade distribution within
scored depth across iter3's semantic queries:

| query | K | perfect | highly | somewhat |
|---|---|---|---|---|
| semantic_101 | 94 | 11 | 48 | 34 |
| semantic_170 | 204 | 36 | 85 | 57 |
| semantic_123 | 26 | 3 | 14 | 6 |
| semantic_7 | 18 | 2 | 11 | 6 |
| semantic_100 | 162 | 26 | 34 | 47 |

Grade 2 = every criterion Perfect except one judged "Somewhat" — and it earns **zero recall
credit**. Converting even a third of these to grade 3 would roughly double several scores.
Because the judge sees *only* `markdown_evidence`, conversion is partly an evidence problem:
the submitted text must explicitly demonstrate *every* weighted criterion.

**Ranking is misaligned with the judge.** In semantic_7 a known-good gold paper was retrieved
but ranked **42nd** — beyond K=18 — earning nothing. iter3 ranked using a per-criterion grade
computed from a 280-character abstract prefix, then enriched evidence *after* ranking, so the
ranking never saw the evidence the judge would see.

**Specific-path failures are identification failures, not list-size failures**:
- specific_9 "the MS^2 DeYong2021 paper": both agents hallucinated titles from LLM memory
  ("peptide sequencing"). The query contains a citation key — author "DeYong(≈DeYoung)" +
  year 2021 — that neither agent used as a retrieval channel.
- specific_15 (iter 2) "the AlphaGeometry paper": the real title ("Solving olympiad geometry
  without human demonstrations") doesn't contain the alias; corpus-grounded verification
  (iter3's addition) fixes this class.
- specific_44: iter3's crash — `authors` entries from snippet-search papers are plain strings,
  and `a.get("name")` raised.

**Metadata** (none in this batch; iteration-2 data): metadata_4 ("Nature portfolio ... Harel")
missed 2/3 gold because substring venue matching kept only flagship "Nature"; metadata_15 got
6/6 gold but precision 0.6 from workshop/venue over-inclusion. iter3's LLM venue classifier
addresses both directions; keep it.

## The iteration-4 agent

`iter4-judge-sim-ranker` keeps iter3's architecture (criteria prediction, per-criterion
grading, targeted snippet enrichment, route-per-score_type) and makes these changes:

### Semantic: rank with a judge simulation over the evidence itself
1. **Broader retrieval**: 8 keyword variants (was 6) at limit 100 + 2 snippet queries
   (was 1) — deep-K queries like semantic_108 (66/222 found) are recall-starved at pool 260.
   Pool cap 320.
2. **Stage 1 (cheap triage)**: per-criterion 0/1/3 grading of the whole pool with
   GPT_5_4_MINI on title + ~320 chars, as before. This orders the tail and picks the head.
3. **Enrich BEFORE ranking the head**: for the top ~90, fetch criterion-targeted snippets
   (free tool calls) aimed at each paper's weakest predicted criteria, then **assemble the
   exact evidence string that will be submitted**.
4. **Stage 2 (judge simulation)**: re-grade the head per-criterion with GPT_5_4_MINI, but on
   the *assembled evidence* — the same text the real judge will see. Rank: predicted
   all-perfect first, then weighted grade. This directly fixes the known-good-at-rank-42
   failure and prioritizes papers whose *evidence* proves every criterion, which is what
   recall actually pays for.
5. Submit 250 with the assembled evidence (head) / title+tldr+abstract evidence (tail).

Cost: ~$0.05–0.06 per semantic query (plan on GPT_5_4, both grading stages on mini). At 73%
semantic mix that lands ≈ $0.045 batch average — inside the free zone with margin.

### Specific: use the query's own cues, and don't crash
- Identification prompt now flags citation-key patterns (`AuthorYYYY`, possibly misspelled)
  and asks for author-name variants + year hint alongside candidate titles.
- New **author retrieval channel**: search each author-name variant, pull top profiles'
  papers, filter to the year hint — this finds "MS^2 (DeYoung 2021)" deterministically even
  when LLM memory fails.
- Robust author-name extraction (dict *or* string entries) — fixes the specific_44 crash.
- Corpus-grounded verification (kept from iter3) over all channels, with cues in the prompt;
  submit 1 paper at confidence ≥0.65, 2 at ≥0.35, else 3 (F1 vs 1-paper gold: right-in-2
  still earns 0.667).

### Metadata: keep iter3's plan + LLM venue classification, add relaxation
If the hard filters empty the set, relax (drop topic filter, then venue filter) and submit a
small best-effort set instead of an empty list (empty guarantees 0).

### Robustness
The whole route is wrapped in the keyword-search fallback as before; enrichment and stage-2
are additionally wrapped so a mid-pipeline failure degrades to the stage-1 ranking instead of
the fallback. A soft wall-clock guard (~15 min) skips remaining enrichment if snippet latency
piles up.

## Why this should beat both parents
- It keeps everything that made iter3 win 6/11 semantic queries and removes its two losses
  (the crash; head ordering blind to evidence).
- The judge-simulation ranking optimizes the exact quantity the scorer pays for (grade-3s in
  the top-K prefix ordered descending), rather than a proxy computed from truncated abstracts.
- Wider retrieval attacks the deep-K recall ceiling (largest single component of the batch
  mean: the K≥160 queries).
- The specific path gains a deterministic retrieval channel for the observed failure class
  (citation-key queries) worth up to +1.0 F1 per such query, at ~$0.01.
