# Iteration 17 — Analysis and Plan

Name: iter17_recall_stack

## What the iteration-16 data says

**Scores.** iter13_grade_view 41.74, iter12_body_conjunction 41.60,
iter16_pool_breadth 37.04 (all raw F1 as %, all inside the cost free zone).
iter13/iter12 have been the top two for three consecutive iterations;
iter16's headline change — a 2× candidate pool — lost 4.7 points.

**Why iter16 lost (attrib16.py, this directory).** Its pool widening *did*
fire (640 vs 320 candidates on every semantic query), yet grade-3 papers
missing from its submissions **rose** from 33 (iter13) to 44. The wider
field flowed through a triage cheapened three ways at once (title 110→85
chars, body 170→105, chunk 32→48), and with 640 candidates competing for
250 submission slots the noisier grades decided who got cut. Its stage-2
sim also self-cancelled where cheap stage-1 over-predicted perfect
(semantic_224: "judge-sim graded 0/0, skipped 63 stage1-perfect" → 21
threes vs iter12's 33; two of its zeros — semantic_145/155 — were single
grade-3 papers stranded below tiny K). Conclusion: **breadth without triage
quality is net negative.** The separable, verified parts of iter16 are
still worth keeping (see below).

**Where the score actually lives (measured on the iteration-16 verdicts):**

- The judge grades only the first ~K submitted positions (K observed 6–114,
  hidden at query time), and **virtually every judged grade-3 already sits
  inside K** for iter12/13 — ordering is near-exhausted, confirming
  iteration 16's decile analysis.
- Recall = (# grade-3 papers) / K is the binding term. The judged prefix is
  dominated by grade-2 papers (semantic_104: 47 Highly vs 2 Perfect inside
  K=56; semantic_160: 23 vs 5), and the union of all three agents' grade-3s
  covers only ~25–50% of K on most queries. So the remaining headroom is
  (a) retrieving perfect papers no phrasing yet surfaced, and (b) putting
  the *right* retrieved papers into the judged prefix — both of which run
  through triage quality.
- Cross-agent divergence on identical code (semantic_104: iter12 got 8
  threes, iter13 got 2 — 3 of the missing ones were *in* iter13's
  submission below judged depth, 4 were never retrieved) shows planner
  phrasing diversity and head-ordering stability are worth more than any
  single constant.
- **Cost risk:** iter13's semantic queries cost ~$0.083–0.085. This batch
  was 64% semantic → mean $0.054, but the test mix is 73% (and an 86%
  batch was observed in iteration 14) → projected mean $0.062+, *outside*
  the free zone. The new agent must cut semantic cost while widening
  retrieval.
- Non-semantic: specific_39 ("the SPIKE paper") scored 0 for every agent —
  gold is ~5 unrelated works sharing the acronym, and the tight 5–8-paper
  hedge built from LLM-guessed titles missed all of them. metadata fixes
  shipped in iter16 (citation conjunction, cap-starved citing queries)
  never got exercised this batch but were root-caused from real failures
  in iteration 15.

## The plan: iter13 base + verified pieces of iter16 + two exact-match fixes

Base = iter13_grade_view (best raw F1 two rounds running). Changes:

1. **Retrieval diversity at full triage quality.** Planner emits 14
   keyword queries (was 10) with *demanded categories* — direct phrasing,
   other-community synonyms, named methods/systems/datasets, task/
   application, adjacent subfield, survey — because iter16's own analysis
   showed iter12's misses were retrieved by siblings with different
   *phrasings*, not deeper lists. POOL_CAP 320→400 (not 640),
   POOL_MERGE_HEAD 240→300, SNIP_INIT_LIMIT 50→100 (tool max, free).
   Triage constants stay at iter13 quality: T1_TITLE 110, T1_BODY 170,
   GRADE_CHUNK 32.
2. **Compact triage output format** (from iter16, parse-identical): local
   1..N indices + unspaced digit grades. Output bills at 6× input; ~40%
   fewer output tokens across t1/sim/rescue/verify.
3. **Gap-fill round removed** — five observed firings moved
   predicted-perfect by a net +2; its ~$0.005/query funds the wider pool.
4. **Depth trims where nDCG says depth doesn't pay:** SIM_DEPTH 55→48,
   RESCUE_MAX 22→16, VERIFY_TOP 26→20, VERIFY_TOP_THIN 30→26.
5. **Metadata path = iter16's** (both fixes root-caused): multi-target
   `cites_paper_titles` with citer-set intersection (metadata_26's
   conjunction was never expressed → 0), and citationCount-ranked mention
   channels + widened submission when the recency-ordered 1000-citer cap
   starves a large gold set (metadata_25: 0.0099). **Keeps iter13's
   chunked venue filter** — iter16 still carried the alphabetical [:120]
   truncation that zeroed metadata_4 for all agents in iteration 13.
6. **Ambiguous-specific hedge widened** (specific_39): when the ambiguity
   gate fires (no author/year cues, interpretation confidence < 0.65),
   alias relevance-search limit 40→100, shortlist 40→48, alternates 6→8,
   alias-titled backstop fills to 12, submission cap 8→14. With gold ≈ 5,
   4 hits in 12 submitted is F1 0.47 vs the observed 0; unambiguous
   references keep the tight 5-cap that has scored 1.000 repeatedly.
7. **Free evidence wins:** abstract passage 1150→2000 chars, body snippets
   600→900 (verbatim substrings, scorer has no length cap), containment
   dedup so a paper's 8 passage slots aren't spent re-showing the judge
   text it already read, tail sweep to position 250, SNIPPET_TIMEOUT 75→90
   (enrich timeouts were visibly eating head-paper snippets in iter16
   stdout).

**Cost projection** (iter13 telemetry × price table): semantic ≈ $0.076 vs
$0.083–0.085 (pool +$0.004, gap-fill −$0.0045, verify trim −$0.005,
compact output −$0.002, sim/rescue trims −$0.002, plan +$0.001). At the
73% semantic test mix → batch mean ≈ $0.058, inside the free zone; even an
86%-semantic batch stays ≈ $0.067 (≈⅓ error-equivalent penalty, tolerable
tail risk vs the recall upside).

## Why this should beat the incumbents

- Against iter13/iter12: same triage quality and head machinery, plus more
  diverse retrieval angles (the measured largest failure bucket), plus a
  cheaper cost profile that survives semantic-heavy batches, plus metadata
  and ambiguous-specific fixes each worth up to a full point when those
  query shapes appear.
- Against iter16: keeps its four separable, evidence-backed wins while
  reverting the confounded bundle (cheap triage + oversized pool + gutted
  sim) that its own failure attribution shows lost grade-3s at submission.

## Verification

`smoke_test.py` (56 checks, all passing) stubs the harness and exercises:
the 14-query planner parse and fallbacks, compact-format grade parsing with
local→global index mapping, evidence assembly (containment dedup, tldr slot
economy, verbatim grounding, ≤8 passages), the ported metadata conjunction
parsing/intersection wiring, the chunked venue filter, and the widened
ambiguous-specific constants. `attrib16.py` (left in this directory)
reproduces the failure-attribution table that drove the design.
