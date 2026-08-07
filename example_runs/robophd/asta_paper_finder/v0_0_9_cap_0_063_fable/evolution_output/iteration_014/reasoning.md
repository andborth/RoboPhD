# Iteration 14 Reasoning

Name: iter14-title-channel

## What iteration 13's data says

iter13_any_author_gate won the batch (42.398 over iter10's 41.839 and iter12's
41.741). iter10 actually had the best raw F1 (0.426 vs 0.424) but paid a cost
penalty ($0.0637 mean, $0.0007 over threshold). The batch had no metadata
zeros — the iter13 fixes held (metadata_15: 0.80, recall 6/6; specific_33:
1.0) — so this batch's losses are concentrated in the semantic tail, and the
diagnostics localize them precisely.

### 1. The judged window is exactly K, and recall is the binding term everywhere

`judge_verdicts.json.scored_depth_cap` equals `score_meta.json.k_estimate` on
every problem: the judge reads exactly the first K submitted papers (K ranged
12–138 this batch) and nothing after. Rank was 0.55–0.93 across the 11
semantic problems while recall was 0.00–0.11 on every loser. The window
contents on the worst problems:

| problem | K | in-window grades (3/2/1) | recall | score |
|---|---|---|---|---|
| semantic_77 | 12 | 0 / 9 / 3 | 0.000 | 0.000 |
| semantic_214 | 20 | 2 / 3 / 15 | 0.100 | 0.169 |
| semantic_123 | 26 | 3 / 3 / 8 | 0.115 | 0.199 |
| semantic_220 | 36 | 7 / 3 / 20 | 0.194 | 0.304 |
| semantic_222 | 134 | 15 / 28 / 90 | 0.112 | 0.197 |

The windows are full of grade-2s ("highly relevant" — one criterion judged
Somewhat instead of Perfectly) and grade-1s. On narrow-criteria queries
(semantic_77 wants "4-bit integer *columnar* weight-only quantization ...
including BLOOM") the grade-2s are mostly *genuinely* not perfect — the
pipeline's evidence mining and weak-criterion patch already squeeze what the
retrieved papers can support. The papers that would grade 3 simply are not in
the pool.

### 2. Pool membership variance decides head-to-heads

The decisive observation: on semantic_77, iter10's single grade-3 paper
(corpus 261049460) was **absent from iter13's entire 250-paper submission** —
not ranked low, never retrieved. Same pipeline, different LLM-sampled keyword
plans, different pools, 0.146 vs 0.000. semantic_214 shows the mirror image
(iter12 recall 0, the others 2/20). All three agents run near-identical
ordering machinery; what varies is which papers each one's stochastic query
plan happens to surface. Retrieval breadth is free (tool calls are unmetered,
the graded pool is capped at COARSE_POOL=340 regardless), so more diverse
retrieval reduces this variance at ~zero LLM cost.

### 3. Deadline pressure clipped stages that were paying

semantic_214 and semantic_123 both reached the expansion phase at t+874s;
round-2 expansion was skipped *by the clock* (r1 grade-3 yields were 6 and 7,
well above the ≥4 quality gate — the stdout message "yield 6 < 4" is
misleading; the deadline branch shares the message) and mining depth was cut
200→120. Wall clocks: worst observed 1518s against a 1740s hard limit with
SOLVE_BUDGET=1500 — ~220s of margin left unused on exactly the problems that
needed it.

### 4. The hard-tail advice from two prior reflections is still unimplemented

Iteration 12's and 13's reflections both said: when the pool is starved,
stop re-tuning gates and add a genuinely different retrieval channel —
"LLM-generated specific paper title guesses resolved via
search_paper_by_title". The thin-pool round currently reformulates keywords
and probes snippets — both are ranking-engine modalities. Title resolution is
a lookup modality: it reaches a paper the model *remembers* regardless of
whether any keyword phrasing ranks it into the top 100.

## Changes (base: iter13_any_author_gate, byte-identical elsewhere)

1. **Retrieval breadth: 12→14 keyword queries, 4→5 snippet queries** (plan
   prompt + slices). Pure tool traffic; the coarse-grading cap (340) is
   unchanged, so LLM cost is flat (+~40 plan output tokens). Directly attacks
   the pool-variance failure mode of §2 — more diverse slices of the
   literature compete for the same 340 graded slots.

2. **Title-guess channel in the thin-pool round.** REFORMULATE_PROMPT now
   returns `{"queries": [6], "titles": [6]}`; the six guessed exact titles are
   resolved via `search_paper_by_title` (free) and the hits join the fresh set
   that gets graded. Gated exactly as before (fires only when coarse grade-3s
   < 12 or strong < 25), so it adds candidates only where the pool is starved
   — the semantic_33/77-class queries that have resisted four iterations of
   keyword-side work. Legacy array replies still parse.

3. **SOLVE_BUDGET 1500→1560.** Worst observed overhead beyond the budget was
   ~18s (1518s total); 1560 leaves ~160s of safety under the 1740s limit and
   buys back the r2 expansion and full mining depth on the two problems the
   clock clipped (§3). The added spend when r2 now fires (~$0.008 on ~2
   problems/batch ≈ +$0.001 mean) fits inside the $0.0027 free-zone headroom
   ($0.0603 observed mean vs $0.063 threshold).

4. **r2 skip message names the real reason** (yield vs deadline) — iteration
   13's message printed "yield 6 < 4" when the deadline was the actual
   blocker; a diagnostic that lies costs the next session real time.

## What I deliberately did not change

- The ordering path (0.55/0.45 blend, promotion rule, TOP_RERANK=100,
  bucketed oldest-first): measured winner across 4 batches; two prior
  iterations regressed by "improving" it.
- The metadata and specific solvers: no metadata failures this batch;
  specific_20 ("the cnn paper", 0.5) lost on which two landmark papers gold
  contains — the ambiguity handling already submits all landmark matches, and
  second-guessing it without knowing gold's convention is a coin flip.
- Coarse-grade truncation (200-char abstracts): extending it would add
  ~$0.013/query of mini input — an order of magnitude more than the headroom.

## Why this should score higher

- Every change is attributable to a specific measured failure: pool variance
  (semantic_77/214), clock-clipped stages (semantic_214/123), and the starved
  tail (semantic_77, plus semantic_33/203 from earlier batches).
- Two of the four changes are strictly-free retrieval breadth; the other two
  spend seconds and pennies that the diagnostics show were available.
- Nothing touches the ranking/evidence machinery that produced the lineage's
  wins, so the downside is bounded while the recall term — the binding
  constraint on every losing semantic query — gets three independent new
  sources of grade-3 candidates.
