# Iteration 7 — Analysis and Plan

Name: iter7_simview_breadth

## What the iteration-6 data says

Scores: iter6_grade3_rescue 48.01 ($0.052), iter5_cite_verify_deep_evidence
44.90 (raw 45.36, cost-penalized at $0.0613), iter4_judge_sim_ranker 43.35
($0.041). iter6's win decomposes into specific_20 (+0.50), semantic_123
(+0.117), semantic_192 (+0.066) — while it *lost* ground to iter5 on the
evidence-heavy large-K queries: semantic_87 (0.754 vs 0.852), semantic_110
(0.389 vs 0.447), semantic_100 (0.200 vs 0.241). Recall is the binding term
on every semantic problem (rank 0.45–0.93, recall 0.00–0.68).

### Finding 1 (headline): the internal sim never saw any snippets

iter6 grades stage-2 / rescue candidates on `_cut(evidence, SIM_CUT=700)`.
But `_evidence()` puts title + tldr + abstract (≈1300-char cut) *first*, so a
700-char prefix almost never reaches the snippets — including the snippets the
rescue round just fetched to prove the weak criteria. Confirmed in stdout:
"rescue round: 30 near-miss papers … rescue promoted 1" on problem after
problem. The entire snippet-enrichment machinery improved only the
judge-visible text; internal ranking and rescue promotion decisions ran on a
truncated abstract. This is the single most mechanistic, highest-leverage fix
available: give the sim a *structured view* of the same evidence the judge
reads (title + tldr + trimmed abstract + the criterion-selected snippets),
instead of a blind prefix cut.

### Finding 2: pool variance dominates large-K recall

On semantic_87 (K=206), iter5 had **41 judge-graded-Perfect papers that were
completely absent from iter6's 250-entry submission** (not low-ranked —
absent). Same on semantic_110 (24 absent) and semantic_100 (14 of 16 absent).
Both agents ran the same architecture with 340-doc pools; the difference is
stochastic keyword-query phrasing and pool caps (iter5's gap-fill added 90 vs
iter6's 60). The union of the two agents' Perfect sets would have scored far
higher than either. Conclusion: on broad queries the corpus has many more
grade-3-capable papers than a 340-doc pool captures; retrieval breadth (more
diverse queries + larger pool) converts directly into recall.

### Finding 3: cost structure funds the fix

~84% of iter6's spend is GPT_5_4_MINI, and stage-1 triage (grading ~400 docs
at ~310 chars each) is the bulk of it — yet stage-1 is the *least*
precision-critical stage (it only decides head membership and tail order; the
head is re-graded anyway). GEMINI_3_1_FLASH_LITE is 40% cheaper on both input
and output. Moving stage-1 there (with a parse-failure fallback to mini)
pays for: a 460-doc pool (12 keyword queries), a no-skip mini sim over the
head on the richer sim-view, and a doubled rescue round.

### Finding 4: specific_20 — hedging works, the duplicate guard is brittle

Gold for "the cnn paper" was AlexNet's *two* corpus records; the verifier
picked LeCun'98 (conf 0.73), and only iter6's low-confidence hedge (adding
the alternate interpretation) rescued 0.5. Meanwhile the title-sim guard
rejected a *true* duplicate record of LeCun ("PROC OF THE IEEE NOVEMBER
Gradient Based Learning …", ratio 0.81 < 0.88) purely because of boilerplate
prefix junk. Fix: similarity = max(SequenceMatcher ratio, token-set
containment), which scores the boilerplate-prefixed duplicate 1.0 while
still rejecting Objaverse-XL (0.86 < 0.88). Also: when confidence < 0.5,
pull duplicate records for the hedged alternate too (gold multi-record sets
have now appeared twice in training).

### Finding 5: small-K queries live or die in the top 25

semantic_203 (K=24) scored 0.000: the sim predicted perfect papers that the
judge graded 2, while the only judge-perfect-capable papers sat at ranks
135–164. semantic_123 (K=26) peaked at 0.191. These queries put their entire
score in the first ~25 positions, and the cheap sim is miscalibrated there.
When stage-1 predicts few perfect papers (≤8 — a reliable narrowness signal:
203 had 6, 123 had 9, 100 had 0), spend one GPT_5_4 pass re-grading the top
24 on their full sim-view and float its all-perfect picks to the very top.

## The new agent: iter7_simview_breadth

Base: iter6's code (keeping all accumulated robustness: relax ladders,
unicode normalization, fallback route, metadata path untouched). Changes:

1. **Sim-view fix**: `_sim_view()` builds the grading text as title +
   tldr(≤220) + abstract(≤450) + up to 4 criterion-selected snippets(≤240
   each) — every passage type the judge sees, within ~1400 chars. Stage-2,
   rescue, and narrow-verify all grade on this. Rescue promotions become
   real instead of impossible.
2. **Breadth**: 12 keyword queries (was 10), POOL_CAP 340→460 (gap-fill cap
   400→520). Stage-1 triage moves to GEMINI_3_1_FLASH_LITE; chunks that fail
   to parse are retried once with GPT_5_4_MINI, then default to partial.
3. **No stage-1-perfect skip**: the whole head (130) is sim'd with mini on
   the sim-view. Band rule kept promotion-only (band 0 = sim-all-3 OR
   stage1-all-3, preserving the semantic_186 protection) but within band 0,
   sim-validated papers sort above stage1-only-perfect ones — cheap-model
   optimism can no longer pollute the very top.
4. **Rescue expansion**: eligible depth = the whole sim'd head (was 80),
   max 36 papers (was 30), re-simmed on the sim-view.
5. **Free tail enrichment**: final positions HEAD..250 with decent stage-1
   scores and no snippets get one combined-probe scoped snippet call
   (≤70 calls, deadline-guarded) — judge-visible evidence upgrade at zero
   LLM cost, targeting K>150 queries whose tail is judged.
6. **Narrow-query verify**: if stage-1 predicted-perfect ≤ 8, GPT_5_4
   re-grades the top 24 on the sim-view; its all-perfect papers form a band
   above everything (ordered by its weighted score). ~$0.02, fires on ~20%
   of semantic queries.
7. **Specific path**: containment-aware duplicate-record similarity;
   alternate-record hedging pulls the alternate's duplicates when conf<0.5.
8. **Evidence assembly**: abstract cut 1300→1400, snippet cut 600→650
   (judge-visible, free).

## Cost projection

Semantic ≈ $0.010 planner + $0.026 stage-1 (flash-lite, 460–520 docs) +
$0.024 stage-2 (mini, 130 docs × ~950-char sim-view) + $0.008 rescue +
$0.004 amortized narrow-verify ≈ **$0.067–0.072**; with the 73% semantic mix
→ batch mean ≈ **$0.053–0.056**, inside the free zone with margin. Tool
calls (snippet enrichment, tail enrichment, bigger pool searches) are free.

## Why this should beat iter6

- The sim-view fix repairs a confirmed mechanical fault at the heart of the
  three stages that decide ranking — every semantic query benefits.
- Pool breadth attacks the largest measured loss (absent Perfect papers on
  large-K queries, worth ~0.1 raw F1 per affected query), funded by a
  cheaper triage model rather than new spend.
- Small-K queries — the current 0.0–0.2 scorers — get a targeted
  high-fidelity re-rank exactly where their whole score is decided.
- The specific-path changes convert both observed specific-query failure
  modes (multi-record golds, boilerplate-prefixed duplicates) into wins
  without disturbing verified behavior.
- All changes are mechanism-backed and query-type-generic; nothing is tuned
  to a specific training problem.
