# Iteration 11 — Analysis and Plan

Name: iter11_tail_saturate

## What the iteration-10 data shows

**Standings.** iter10_cite_expand 47.93 ($0.0577), iter5_cite_verify_deep_evidence
46.46 ($0.0557), iter4_judge_sim_ranker 39.10 ($0.0371). iter10 (the newest
graft on the iter6→iter9 lineage) won, went 4/4 on specific queries, and is the
base for this round. All three were inside the cost free zone, but iter10's
$0.0577 mean leaves almost no margin — and the batch was only 71% semantic;
a 73%+ semantic draw would push the same agent to ~$0.060.

**Per-problem diagnosis (iter10, from `score_calculation.md` + `judge_verdicts.md`):**

| problem | K | rank | recall | judged grade histogram (P/H/S/N) |
| --- | --- | --- | --- | --- |
| semantic_110 | 228 | 0.70 | 0.29 | 67/126/26/8 |
| semantic_170 | 204 | 0.74 | 0.22 | 46/118/38/3 |
| semantic_226 | 138 | 0.73 | 0.24 | 33/27/74/4 |
| semantic_22  | 118 | 0.59 | 0.19 | 23/0/53/42 |
| semantic_91  | 100 | 0.39 | 0.22 | 22/73/5/0 |
| semantic_219 | 58  | 0.55 | 0.09 | 5/12/36/5 |
| semantic_189 | 40  | 0.51 | 0.13 | 5/13/19/3 |
| semantic_233 | 34  | 0.81 | 0.21 | 7/2/24/1 |
| semantic_70  | 34  | 0.61 | 0.12 | 4/13/13/4 |
| semantic_203 | 24  | 0.64 | 0.08 | 2/12/9/1 |

Recall is the binding term on every semantic query (rank 0.39–0.81 vs recall
0.08–0.29). But the histograms split that into two distinct sub-causes:

1. **Grade-2 mass ("Highly Relevant" earns zero recall).** semantic_110 has
   126 Highly vs 67 Perfectly in the judged prefix; semantic_170 has 118 vs 46;
   semantic_91 73 vs 22; semantic_203 12 vs 2. A paper is Perfect only when
   *every* weighted criterion is judged Perfectly-supported by the submitted
   `markdown_evidence`; one Somewhat criterion caps it at grade 2. Papers
   beyond HEAD=120 in the submission get abstract-only evidence — yet on big-K
   queries the judge reads to position ~K (up to 228). Qualifier criteria
   ("task-agnostic", "explicitly compares X and Y") rarely live in abstracts;
   they live in body text that `snippet_search` can quote **for free**.

2. **Pool coverage.** stage-1 triage predicted 25 perfect on semantic_110 where
   the judge found 67 in the submitted prefix alone — triage is conservative and
   the corpus is rich; more genuinely-relevant candidates exist than we retrieve
   and keep. Meanwhile the merge cap (POOL_CAP=340) blindly discards hundreds of
   unique candidates by round-robin order before triage ever sees them.

**Two mechanical faults found in stdout:**

- **Reference fetching in citation expansion is dead.** Every
  `get_paper_batch(fields="corpusId,references")` call failed with the
  server-side `'NoneType' object is not iterable` error (all attempts, all
  groups, on semantic_110 and others). iter10's headline citation-graph
  expansion has been running on **citers only** — the references half (prior
  work, older vocabulary — exactly what keyword search misses) never arrived.
- Expansion netted only +60 docs against a 140 cap for the same reason.

## Changes in iter11_tail_saturate (base: iter10_cite_expand)

The theme: **saturate evidence for every judged position at zero LLM cost, and
widen the funnel with free retrieval** — funded by mechanical trims so the mean
stays at or below iter10's spend.

1. **Tail evidence sweep (new, FREE — the headline change).** After head
   ordering is final, entries in submission positions HEAD..235 get one scoped
   `snippet_search` call targeting their weak criteria (from stage-1 verdicts),
   and their submit-time evidence is rebuilt to cover every criterion. On broad
   queries (many uniques retrieved) the whole judged tail is swept; on narrow
   ones only the first 40. This attacks the 126-Highly-at-zero-credit problem
   exactly where no LLM spend is needed — snippet calls are free, the evidence
   assembler is lexical, and the judge reads what we quote. Deadline-gated
   per call so the 29-minute budget is never at risk.

2. **Qualifier coverage for stage-1-perfect head papers (FREE).** Enrichment
   previously skipped papers triaged all-3 — precisely the papers the judge
   later demotes to Highly when the abstract doesn't state a qualifier
   criterion. Now any criterion whose probe/description words are absent from
   the title+abstract gets a targeted snippet fetch before evidence assembly.

3. **References restored in citation expansion.** Per-seed `get_paper(...,
   fields="corpusId,references")` (fail-open, attempts=1, semaphore 6) replaces
   the poisoned `get_paper_batch` group call. Seeds 8→10, citer limit 60→70,
   cap 140→150. Expansion should now actually deliver the prior-work half of
   the graph.

4. **Lexical prescreen instead of a blind pool cap.** All uniques from the 10
   keyword + 3 snippet searches are merged (no cap), the first 240 keep their
   source-rank order, and the remaining pool slots (to POOL_CAP=360) are filled
   by criteria-word overlap — a free filter that beats source-rank position
   241+ at picking triage-worthy candidates.

5. **Deeper rescue + thin-pool verify.** Grade-2 rescue depth 80→the whole
   head; RESCUE_MAX 30→32. When predicted-perfect ≤ 10 (thin pools like
   semantic_203, K=24, where the entire score sits in the top two dozen
   slots), the GPT_5_4 head verify extends 16→24 papers.

6. **Cost funding + telemetry.** T1 triage body 200→170 chars, triage chunk
   25→32 (fewer prompt-overhead repeats), HEAD 120→110, SIM_CUT 700→640.
   Net per-semantic-query cost is modeled ≈ iter10 or slightly below
   (~$0.078 vs $0.081 on the heaviest shape), keeping the batch mean safely
   under $0.06 even at a 73% semantic mix. Per-stage LLM call/char counters
   now print at the end of every query so the next iteration can trim
   surgically instead of estimating (a standing request of two reflections).

Everything else — planning prompt, transport retry wrapper, alias hedging on
specific queries, metadata path with fail-open reference verification, band
ordering, never-empty fallbacks — is inherited unchanged from iter10.

## Why this should score higher

- The grade-2 mass is the largest untapped pot in the data: on the four
  biggest-K queries there are 344 judged Highly papers earning zero recall.
  Even a 15% flip rate on those queries (evidence was the missing piece, the
  paper was genuinely qualified) is worth roughly +3–4 raw points on this
  batch shape; the mechanism costs nothing but wall-clock we have (observed
  max 1135s vs 1740 budget, and every call is deadline-gated).
- Restoring references fixes a silent regression in the incumbent's best
  feature; prior-work expansion is the retrieval modality most likely to catch
  older-vocabulary grade-3s keyword search misses.
- The prescreen widens the funnel at zero incremental triage cost.
- All changes are additive, gated, and fail-open; the specific/metadata paths
  (4/4 and structurally fixed in iter10) are untouched, so downside on the
  27% exact-match share is nil.
