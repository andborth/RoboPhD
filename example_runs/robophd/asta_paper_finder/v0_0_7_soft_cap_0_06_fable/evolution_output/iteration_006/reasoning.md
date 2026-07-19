# Iteration 6 — Analysis and Plan

Name: iter6_grade3_rescue

## What the iteration-5 data says

Scores: iter4_judge_sim_ranker 46.59 ($0.040), iter5_cite_verify_deep_evidence
46.36 after a 0.88-pt cost penalty (raw 47.24, $0.0625), iter2 37.35. iter5's
raw F1 is the best but it burned its advantage on cost; iter4 and iter5 are
statistically tied and split solo wins 4/7. The base architecture (route on
score_type; semantic = predict criteria → broad retrieval → per-criterion
triage → scoped snippet enrichment → judge-simulation rerank) is validated —
the question is where the remaining ~53 points are.

### Confirmed mechanics (from diagnostics, new this iteration)

`judge_verdicts.md` explicitly marks entries "(beyond scored depth — not
judged)": **only the first ~K submitted papers are judged at all** (K =
the hidden per-query relevant-count estimate; observed 14–304). Both the
rank term and recall are computed inside that prefix. Everything below
position K is dead weight — neither helps nor hurts. So the whole game is:
*fill positions 1..K with papers whose evidence earns grade 3*, for an
unknown K between ~15 and ~300.

### Loss decomposition (iter5, 11 semantic problems)

rank is healthy everywhere (0.68–0.85); recall is the binding term on every
single problem (0.05–0.32). Within the judged prefix:

| problem | K | grade-3 | grade-2 ("Highly") | score |
|---|---|---|---|---|
| semantic_43 | 16 | 1 | 6 | 0.104 |
| semantic_214 | 20 | 1 | 6 | 0.094 |
| semantic_98 | 22 | 5 | 8 | 0.358 |
| semantic_222 | 134 | 33 | **80** | 0.363 |
| semantic_221 | 146 | 55 | 12 | 0.523 |
| semantic_196 | 304 | 98 | 46 | 0.447 |

Grade-2 papers earn **zero** recall — and they are the single largest bucket
in the top-K on most problems. A grade-2 paper is one whose evidence
demonstrated most-but-not-all weighted criteria ("weighted" lands in
(0.67, 0.99]); converting even half of semantic_222's 80 grade-2s to grade-3
would roughly double its recall (0.25 → 0.55+) and its score. This
conversion — *rescue the near-misses* — is the highest-leverage move
available, and it is mostly free: snippet_search calls cost nothing; only
the re-grading LLM pass costs money.

Secondary observations:

- **specific_11 (0.667, iter5's only specific loss)**: query "the paper about
  the Objaverse dataset"; the multi-record verifier listed Objaverse-XL as a
  "duplicate record" of Objaverse at confidence 0.99. XL is a *different
  paper* (follow-up, different title, different year). iter4's simpler path
  scored 1.0. The duplicate-record feature is right (specific_20's gold in
  iteration 4 contained two AlexNet records) but needs a mechanical guard:
  true duplicate records share a near-identical title (~1.0 similarity for
  AlexNet's two records) and first author; "X" vs "X-XL" (~0.8) must be
  rejected.
- **semantic_186 regression (iter5 0.302 vs iter4 0.451, same K=68)**: iter5's
  stage-2 sim over snippet-heavy evidence demoted papers the judge liked
  (13 vs 21 grade-3 in top-K). Stage-2 re-ranking can *hurt* when the
  assembled evidence misleads the simulator. Mitigation: a paper predicted
  all-perfect by stage-1 (on abstract text — which the judge also sees,
  since the abstract is in the evidence) keeps top-band rank even if stage-2
  disagrees; stage-2/rescue can promote but not demote out of the band.
- **Cost structure**: 86% of iter5's spend is GPT_5_4_MINI grading — stage-1
  triage of ~430 docs plus stage-2 sim of 150 docs at 900 chars each. The
  $0.0625 mean must come down ~20% to sit safely inside the free zone while
  funding the rescue pass.
- Metadata path went untested (0 metadata queries in the batch) but was
  built directly against iteration-4's metadata failures (reference
  verification, inclusive years, relax ladder). Keep it as-is.

## The new agent: iter6_grade3_rescue

Base: iter5's code, with these changes.

### Semantic path

1. **Criterion probes**: the planner emits, per criterion, a short
   declarative "probe" phrase (e.g. "micro-F1 averaged across test
   episodes") likely to appear near supporting text in a paper's body.
   Scoped enrichment snippet queries use probes instead of full 40-word
   criterion descriptions — sharper passage retrieval for exactly the text
   the judge needs to see.
2. **Criterion-coverage evidence assembly** (free, pure Python): evidence =
   title + tldr + abstract + snippets chosen *greedily by lexical overlap
   with each criterion* (weightiest-uncovered criterion first), rather than
   arrival order. Ensures each weighted criterion has a supporting passage
   in the ≤8 slots when one exists, and drops off-topic snippets that
   dilute the judge's read.
3. **Grade-2 rescue round** (the headline change): after stage-2 sim, take
   papers in the top 80 of the ranked head predicted grade-2 (weighted in
   (0.5, 0.99], ≤2 weak criteria) — up to 30 of them — fire one scoped
   probe snippet call per weak criterion, rebuild evidence, re-sim only
   those papers, and merge promotion-only. Directly converts near-misses
   into grade-3s at the top of the list, which is where small-K queries
   live or die.
4. **Anti-regression band rule**: rank band 0 = all-perfect in *either*
   stage-1 (abstract) or the latest evidence-sim; within band, sort by the
   max weighted score across stages, tiebreak by stage-1 order. Protects
   against semantic_186-style sim-induced demotions while keeping the sim's
   promotions.
5. **Cost cuts to pay for it**: stage-1 triage text 260→200 chars
   (title 140→110); stage-2 evidence cut 900→700 chars and skips papers
   already all-perfect at stage-1 (their band is protected anyway);
   gap-fill adds ≤60 docs (was 90). Estimated semantic-query cost ≈ $0.06,
   batch mean ≈ $0.050–0.055 (11/14 semantic mix) — inside the free zone
   with margin.

### Specific path

Keep iter5's multi-channel gathering + verifier, add the mechanical
duplicate guard: extra "records of the same work" are accepted only when
title similarity ≥ 0.88 *and* first-author surname matches (or is missing).
Verifier prompt now states explicitly that follow-ups/extensions
("X-XL", "X 2.0") are different papers. This restores specific_11 (→1.0)
without giving back specific_20-style duplicate-record wins.

### Metadata path

Unchanged from iter5 (reference verification, venue LLM+alias union filter,
inclusive year bounds, relax ladder). It encodes every iteration-4 metadata
failure and has no observed regressions.

## Why this should beat both parents

- iter5's raw F1 was already best; this keeps its retrieval breadth and adds
  the one mechanism the verdict data says is worth the most (grade-2→3
  conversion at the top of the list), while removing its cost penalty.
- The two known regressions vs iter4 (specific_11, semantic_186) each get a
  targeted, mechanical fix rather than a rollback, so iteration-4's wins are
  preserved.
- All changes are query-type-generic (no per-problem tuning), so they should
  transfer to the held-out mix, which matches the training mix.
