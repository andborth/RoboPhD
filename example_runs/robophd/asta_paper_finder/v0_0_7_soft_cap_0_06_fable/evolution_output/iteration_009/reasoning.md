# Iteration 9 — Analysis and Plan

Name: iter9_verified_head

## What the iteration-8 data shows

**Standings.** iter6_grade3_rescue is the incumbent best (0.480 in iter 6, 0.300 in
iter 7, 0.322 in iter 8 — top or effectively tied-top in all three), so it is the
base. iter5 and iter8 each beat it on specific query shapes, which tells me which
mechanisms to graft rather than to redesign.

**Scorer mechanics (re-confirmed on iteration-8 diagnostics).** The semantic judge
grades exactly the first K submitted entries (K observed 6–232, hidden at query
time); recall counts only grade-3 papers in that prefix, and grade 3 requires
*every* weighted criterion judged Perfectly. On small-K queries (6–16) the whole
score sits in the first ~15 positions. Also visible in the verdicts: papers listed
in the query's `known_to_be_good` auto-grade "Perfectly Relevant (known-good)" —
pure retrieval-breadth wins.

**Judge noise is real.** On semantic_57 the *same paper* (276408140) with
near-identical evidence graded Perfectly for iter5/iter8 but only Highly for iter6.
So micro-optimizing evidence further has diminishing returns; the reliable levers
are (a) getting genuinely-perfect papers into the pool and (b) ordering the head of
the list correctly.

**Recurring all-zero problems** (same three queries zeroed every agent in both
iterations 7 and 8 — persistent structural gaps, not noise):

1. `specific_39` "the SPIKE paper" — gold has **5** corpus_ids spanning distinct
   works named SPIKE. Every agent picked one referent (all wrong) and submitted ≤2
   ids. Ambiguous aliases need genuine multi-referent hedging *with retrieval that
   actually surfaces alias-titled papers* (iter8's hedge existed but its candidate
   pool never contained a gold SPIKE).
2. `metadata_33` "A SPLASH 2019 and beyond paper that cites any NeurIPS" — no agent
   models a *cites-venue* constraint. Gold is a single paper; the query is fully
   mechanical: SPLASH-family papers ≥2019 whose reference list contains a NeurIPS
   paper. Tool calls are free, so forward reference-venue verification is feasible.
3. `semantic_77` (4-bit columnar quantization, K=12) — rank 0.81 but recall 0:
   stage-1 predicted 31 "perfect" papers, the judge graded the top 12 and gave no
   grade 3. The cheap triage is over-lenient exactly when true perfects are scarce.
   Also 4/10 keyword searches died on HTTP 502 (which the transport layer does
   **not** auto-retry), shrinking the pool.

**Where the graftable wins were:**
- iter8's **narrow-query GPT_5_4 verify** (re-grade top ~20 when predicted-perfect
  is scarce, boost confirmed-perfect papers to the very top) is the visible cause
  of its wins on semantic_43 (Perfectly at ranks 2 and 5) and semantic_101 — both
  queries where iter6's head ordering left grade-3s buried.
- iter8's **transport retry wrapper** (fresh-coroutine factory, 3 attempts,
  12s/40s backoff) directly addresses the 502 outages seen on semantic_77 and the
  ConnectionRefused failures on specific_39.
- iter5/iter8 beat iter6 on semantic_145 ("**earliest** paper comparing two
  dialogue systems…", K=6) by having the known-good older paper in their pool and
  near the top. "Earliest/first" queries need an old-year preference in the head
  ordering; relevance search is recency-skewed.

## The new agent: iter9_verified_head

Base = iter6_grade3_rescue, unchanged in its core semantic pipeline (criteria
reconstruction with probes, 10-way keyword + 3-way snippet fan-out, cheap
per-criterion triage, gap-fill, criterion-targeted snippet enrichment,
criterion-coverage evidence assembly, judge simulation with anti-regression bands,
grade-2 rescue). Changes:

1. **Transport resilience (graft from iter8).** `_safe_tool` now takes a zero-arg
   coroutine factory and retries 3× with 12s/40s backoff. All call sites converted
   to lambdas. Fixes the 502/connection-refused class of losses; tool calls are
   free and the 29-min budget is slack.

2. **High-fidelity head verify (widened from iter8).** When stage-1
   predicted-perfect ≤ 32 (semantic_77 had 31 — iter8's gate of 8 missed it),
   GPT_5_4 re-grades the top 24 papers on their exact submitted evidence.
   Confirmed-perfect papers are boosted above everything, ordered by the
   verifier's weighted score. Boost-only: nothing is demoted, so well-ordered
   queries can't get much worse. This targets the dominant loss mode — grade-3s
   buried below grade-2s inside the judged prefix on small-K queries — at ~$0.015
   per triggered query.

3. **Ambiguous-alias hedging on specific queries (fixed retrieval + wider
   submit).** When the reference has no author/year cues and interpretation
   confidence < 0.65: raise the alias relevance search to limit 40, float
   alias-titled candidates (title begins with the alias) to the front of the
   verifier's shortlist so distinct SPIKE-like works are actually visible, ask the
   verifier to enumerate the best record of *every* distinct work plausibly known
   as "the <alias> paper", and submit up to 8. Cued or confident references keep
   iter6's tight path (it scored 1.0 on specific_11 where the hedging iter8 got
   0.4 — precision matters when gold is 1–2 records).

4. **Metadata cites-venue constraint.** The plan schema gains `cites_venue`. When
   set: build the candidate base from per-venue-name searches (with and without
   the server-side venue filter), apply the year/venue filters, then fetch each
   surviving candidate's references, batch-resolve the referenced papers' venues,
   and keep only candidates citing ≥1 paper at the target venue. Umbrella-venue
   expansion (SPLASH ⇒ OOPSLA/Onward!/DLS/GPCE/SLE/PACMPL; NeurIPS ⇒ NIPS/full
   name) applied on both sides. Terminal never-empty fallback (content-word
   keyword search, submit ≤10) so the metadata path can never submit 0 papers.

5. **"Earliest/first" temporal cue.** Detected from the raw query; the plan prompt
   is told to include era-appropriate/classic terminology in keyword variants, and
   the head ordering uses year-ascending as the tiebreak inside the top band so
   old seminal papers (the plausible golds and known-goods) sit at the top of the
   judged prefix.

6. **Sharper band ordering (from iter8).** Within the top band, sim-validated
   perfects outrank stage1-only perfects (`boost > band > sim-perfect > weighted
   score > stage-1 position`).

## Why this should score higher

- The three persistent all-zero queries get structural fixes (worth up to ~0.1
  mean on a sample containing their query shapes) rather than hoping the sample
  avoids them.
- The head-verify graft is the mechanism behind iter8's semantic wins, attached to
  iter6's otherwise-stronger pipeline, with the gate widened to cover the
  scarce-perfect case it demonstrably missed.
- Transport retries convert observed catastrophic (pool-starving) failures into
  ordinary latency.
- Everything else — the parts that made iter6 the best agent three iterations
  running — is preserved, and every addition is promotion/boost-only or gated
  behind conditions that were losses before, bounding downside risk.
- Projected cost: iter6 measured $0.0425 mean; the verify pass adds ~$0.006 mean
  (triggered on roughly half of semantic queries), keeping the batch mean ≈ $0.049,
  safely inside the $0.06 free zone.
