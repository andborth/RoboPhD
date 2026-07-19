# Iteration 15 — Analysis and Plan

Name: iter15_cite_inverse

Base: `iter14_crit_coverage` (iteration-14 winner, 38.93, mean cost $0.0602).

## 1. What the iteration-14 data says

### The standings, and what actually decided them

| Agent | Raw mean | Final | Mean cost | Penalty |
| --- | --- | --- | --- | --- |
| iter14_crit_coverage | 0.390 | 38.93 | $0.0602 | −0.07 |
| iter13_grade_view | **0.394** | 38.12 | $0.0635 | −1.26 |
| iter10_cite_expand | 0.380 | 36.08 | $0.0654 | −1.92 |

iter13 had the *highest raw score* and lost on cost. iter14's semantic-side
changes (dedup, criterion vocabulary, conjunction retrieval) were roughly
score-neutral on this batch — its win came from being $0.003/query cheaper.
**All three agents paid a cost penalty.** The threshold is $0.06; the winner
cleared it by $0.0002 — a coin flip, exactly the situation iter14's own
reasoning warned about. This batch was 12/14 semantic (versus the nominal
73%), which pushes mean cost up, and future batches can look like this one.
Cost margin is therefore worth real points and must come from the semantic
path, not from hoping for a friendlier mix.

### Failure inventory (iter14, raw scores)

- `semantic_77` = **0.000**: recall 0/12. Judge graded only 12 papers — and
  that is the structural fact of the round, see below. 4 Highly, 8 Somewhat,
  0 Perfect. The criteria ("4-bit integer **columnar** weight-only
  quantization") are ultra-specific; no submitted evidence proved every
  criterion. Cost was also the batch max ($0.097).
- `metadata_31` = 0.043: "Journal articles by David Harel … citing papers by
  Gera Weiss". The agent found the right author base (452 candidates → 79
  after filters; gold is 16 of them), then tried to verify the citation
  constraint by fetching each candidate's `references` — which **fails
  server-side in every observed run** (`'NoneType' object is not iterable`,
  both field variants, batched and per-paper). The ref check zeroed out, the
  relax ladder submitted 30 unverified papers, 1/16 hit. The failure is
  mechanical, not conceptual.
- The remaining semantic queries repeat the known pattern: recall binds the
  harmonic mean (rank 0.30–0.90 vs recall 0.05–0.26 everywhere), and the
  Highly-not-Perfect mass is large (semantic_123: 13 Highly vs 2 Perfect;
  semantic_101: 33 vs 24; semantic_203: 7 vs 3).

### A structural fact previous iterations only half-used

Checking `judge_verdicts.md` against `score_calculation.md` across all 12
semantic problems: **the judge grades exactly the first K submitted papers
and nothing else** (K = the per-query relevant-count estimate; observed 12,
12, 22, 24, 26, 34, 68, 94, 198, 206, 228, 232 this batch). Both rank and
recall are computed inside that prefix. Consequences:

- Everything past position K is invisible; submitting the full 250 is free
  insurance for large-K queries and costs nothing on small-K ones. (Already
  done.)
- On the seven queries with K ≤ 94, the whole score lives in the first ~2
  dozen positions — head ordering is *recall*-critical, not just
  rank-critical, because a grade-3-capable paper at position K+1 earns
  nothing. The existing head-verify/band machinery is aimed correctly.
- iter13's reasoning had measured median K=52 over 46 queries; the design
  (FULL_COVER_DEPTH=36, verify top ~26) is consistent with this. I keep it.

### What I decided NOT to pursue, and why

- **Better Perfect-vs-Highly ordering signals.** iter14's `calibrate.py`
  refuted every cheap lexical proxy against 1208 ground-truth verdicts
  (coverage, passage count, length, position — none separate grade 2 from 3;
  coverage actively inverts). The internal LLM graders are also proven
  miscalibrated on that exact boundary (semantic_77: 31 predicted perfect,
  judge says 0; verify said 3, judge says 0). More effort here has negative
  expected value at current information.
- **specific_39-style name collisions** — left alone for the reasons iter14
  documented; the specific path scores 1.000 on well-posed queries.
- **A redesign.** Fourteen iterations of head-to-head data show batch noise
  (±5 pts for identical code) dwarfs any single change; only mechanism-backed
  fixes have survived. Both changes below are mechanism-backed.

## 2. Changes

### Change 1 (headline): inverse-citations verification in the metadata path

For "papers citing work by author X" constraints, the agent currently needs
candidates' reference lists — an API surface that is dead (server error on
every variant, every run, both call sites). But the *inverse* surface works
fine and the citation-expansion code already uses it: `get_citations` on the
**cited author's papers** returns every citing paper's corpusId.

New `_citing_set()`: for each of the cited author's papers (we already fetch
their corpus ids to build the reference-match target set), call
`get_citations(fields="corpusId", limit=1000)` and union the citing ids —
free tool calls, concurrency-capped. A candidate cites X iff its id is in
that union. The set is *complete* when every probed target returned under
the 1000 cap (true for typical cited-author profiles: Gera Weiss ≈ 99
papers, none Harel-scale); completeness is tracked, and when the set is
incomplete (a mega-cited target) or any call failed, the existing fail-open
behavior (title-mention heuristic, relax ladder) is preserved.

This runs alongside the existing reference check (which fails open if the
server ever heals) and unions with it. On metadata_31 it would have reduced
79 venue-filtered candidates to the ones actually citing Weiss — the gold is
16 of those 79 — instead of submitting 30 blind for 0.043. The
"cites-specific-paper" shape is already covered by channel A (get_citations
on the cited paper) and is unaffected.

### Change 2: cost margin, taken from measured non-binding stages

Target ≈ $0.052 batch mean even on a 86%-semantic batch (this one), i.e.
semantic-query mean down ~$0.008–0.010 from $0.065. Cuts, chosen where the
score argument is weakest per token:

| Knob | Old → New | Why this is the right margin |
| --- | --- | --- |
| verify chunk size | 6 → 8 | pure prompt-overhead: criteria block repeated per chunk on GPT_5_4 (~$0.004/query on verify-triggered queries) |
| VERIFY_TOP / THIN | 26/30 → 24/28 | nDCG discount at position 25 is ~30% of position 1; last two slots are the cheapest to lose |
| T1_BODY | 150 → 140 | triage reads the abstract head; 140 chars keeps the sentence that states the topic |
| SIM_DEPTH | 48 → 44 | half of observed K ≤ 52; sim positions 45–48 are read on a minority of queries |
| EXPAND_CAP | 100 → 80 | expansion triage cost is linear in docs; observed marginal predicted-perfect gain from the last 20 docs ≈ 0 (semantic_77: +100 docs → +0 perfect) |
| RESCUE_MAX | 22 → 18 | rescue promoted 0–1 papers/query in every observed run |

Nothing that feeds pool breadth (keyword/snippet/conjunction channels,
POOL_CAP, gap-fill) is touched — pool composition is the recall lever and
recall binds the score.

### Kept verbatim

Planner, criterion vocabulary/coverage machinery, global passage dedup,
grade view, full-coverage enrichment, citation expansion, tail sweep, band
ordering, specific path, metadata relax ladder, transport wrapper,
telemetry.

## 3. Why this should score higher

- **Cost:** on a batch shaped like iteration 14's, the trims are worth
  ~0.7–1.9 points relative to the three agents' actual penalties, and they
  remove the recurring coin-flip at the threshold. This is the most certain
  gain available and it compounds with everything else.
- **Metadata:** the citing-author fix converts a measured 0.043 into a
  plausibly high score whenever the shape appears (it has appeared in
  multiple training batches: metadata_31 here, metadata_25 earlier). On a
  14-query batch one such query is worth ~4–6 points of raw swing; on the
  held-out set the metadata share is ~19%, and citation-constrained queries
  are a recurring sub-shape.
- **Risk control:** both changes are mechanism-backed; neither touches the
  semantic recall machinery that produced the lineage's wins. The trims cut
  from stages with measured near-zero marginal product (rescue promotions,
  expansion tail, verify depth), so the plausible downside is small and the
  batch-noise floor dominates it.

Residual risks: (a) the verify-depth and sim-depth trims could cost an
occasional head slot on mid-K queries — bounded by the nDCG discount; (b) an
extremely cited target author would make the inverse set incomplete — handled
by the completeness flag falling back to today's behavior; (c) batch mix
could swing raw scores ±5 regardless — unavoidable, which is exactly why the
cost margin (the only deterministic term) is worth locking in.
