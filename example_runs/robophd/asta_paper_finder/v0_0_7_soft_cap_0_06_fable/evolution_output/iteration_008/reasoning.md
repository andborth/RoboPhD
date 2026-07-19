# Iteration 8 — Analysis and Plan

Name: iter8_resilient_lean

## What the iteration-7 data says

Scores: iter6_grade3_rescue 30.033 (raw 0.300, $0.052), iter7_simview_breadth
29.495 (raw 0.310, $0.0641 → 1.455-pt cost penalty), iter2 18.056. iter7 had
the best retrieval architecture but lost the iteration on cost. Its raw win
is bigger than it looks: excluding one query it lost to a transport outage
(below), iter7 beat iter6 on 6 of 9 shared semantic queries (+0.53 net raw
F1). The architecture is right; the packaging (cost, resilience) is what
failed. So iteration 8 keeps iter7's pipeline and fixes the failure shells
around it.

### Finding 1 (headline): a transport outage zeroed a query, and the agent let it

On semantic_192, every one of iter7's 15 initial tool calls failed with
`HTTP 502` — an error code *not* in the transport layer's auto-retry set
(429/529/504). The agent shrugged, built a 0-doc pool, and submitted 0 papers
— after **14.6 seconds** of a **29-minute** budget. iter6 ran the same query
minutes earlier at 0.360. Cost of this single failure: ~2.6 points of batch
score. The same pattern zeroed metadata_33 for every agent ("0 candidates →
submitted 0", 12s runs).

Fixes (all free — tool calls cost nothing):
- `_safe_tool` now takes a coroutine *factory* and retries every failed tool
  call up to 3 times with backoff (12s/40s) — covering 502/500/503 and
  broken connections that the transport layer won't retry.
- Phase-level recovery: if the semantic candidate pool comes back tiny
  (< 40 docs), the whole search fan-out re-runs after a 60–120s sleep, up to
  3 extra rounds, deadline-guarded. An outage now has to last ~8 minutes to
  hurt us.
- The solver-level fallback route also retries with sleeps instead of giving
  up in one shot.
- Metadata path gains a terminal never-empty ladder: content-word keyword
  search (the raw query is often sentence-shaped, which zeroes the keyword
  engine) plus an NL `snippet_search` channel, submitting a small (≤12)
  best-effort list. An empty submission is a guaranteed 0; a 12-entry
  guess at 1 hit is 0.15+.

### Finding 2: the judge grades exactly the first K positions

Cross-checking `judge_verdicts.md` against `score_calculation.md` on four
queries: judged-count == K every time (K=12, 24, 68, 228). Everything below
position K is invisible to the scorer. K is unknown at query time, so
submitting 250 stays correct (covers K=228), but this hardens the case for
iter7's top-focused machinery (narrow-query GPT_5_4 verify when stage-1
predicts ≤8 perfect papers — semantic_203's K was 24 and iter7 was the only
agent to score on it). All of that is kept.

### Finding 3: specific-query golds are sometimes *sets of distinct works*

specific_39 ("the SPIKE paper") has **5 gold corpus_ids spanning different
works** that share the name SPIKE. iter7's single-referent verifier picked
one wrong record at confidence 0.95, and its duplicate-record guard actively
*rejected* a gold paper ('A parallel hybrid banded system solver: the SPIKE
algorithm', dup_sim 0.44). Meanwhile specific_7 ("BART by Lewis et al.") and
specific_9 ("MS^2 DeYong2021") — both with author cues and single-referent
golds — scored 1.0 with confident single submissions.

Fix: an ambiguity-gated hedge. When the reference has **no author/year cues**
and the alias is short (≤2 tokens), the verifier is asked a different
question: list *every* candidate that IS a paper named/presenting `<alias>`
(across fields and interpretations), and we submit all of them plus their
duplicate records, capped at 10, with the strict dup-sim guard disabled in
this mode. F1 math: with a 5-id gold, 8 submitted / 4 hits = 0.62 vs today's
0.00; with a 1-id gold the worst case is 2/(10+1) ≈ 0.18 vs a coin-flip
single guess. The gate keeps specific_7/9-style cued queries on the exact
single-record path that already scores 1.0.

### Finding 4: where the $0.064 went, and what to cut

Per-semantic-query spend: flash-lite stage-1 ≈ $0.035 (520-doc pool),
mini stage-2+rescue ≈ $0.036 (130-doc head), GPT_5_4 planner/verify ≈
$0.015. The cuts below trade ~5% of grading volume for a ~30% cost drop,
without touching the mechanisms that produced iter7's wins (sim-view,
breadth-via-cheap-triage, rescue, narrow-verify, free enrichment):

| Knob | iter7 | iter8 | Effect |
|---|---|---|---|
| POOL_CAP / TOTAL | 460 / 520 | 440 / 500 | −4% stage-1 input |
| T1 title+body chars | 110+200 | 90+160 | −26% stage-1 per-doc input |
| HEAD (simmed depth) | 130 | 110 | −15% stage-2 input |
| SV abstract/snip/nsnip | 450/240/4 | 420/220/3 | ~−15% sim-view size |
| RESCUE_MAX | 36 | 28 | −22% rescue input |
| NARROW_TOP | 24 | 20 | −17% verify input |

Projected semantic cost ≈ $0.060–0.066; batch mean at the 73% semantic mix
≈ **$0.048–0.052**, and still ≤ $0.058 on an 86%-semantic batch. That is
iter6's cost envelope with iter7's architecture. Retries add zero LLM cost.
The planner stays on GPT_5_4: its criteria reconstruction steers triage,
sim, evidence, and rescue — the worst place to go cheap.

## What is deliberately unchanged

- The sim-view (judge-aligned grading text) — iter7's confirmed-fix core.
- Band/promotion-only ranking, rescue round, narrow-query verify, free
  head/tail snippet enrichment, gap-fill round.
- The metadata constraint plan → channels → post-filter → relax ladder.
- Per-problem deltas like semantic_155 (iter7 0.127 vs iter6 0.241) look
  like pool-composition noise (iter7's own reflection documents this);
  no per-problem tuning.

## Why iter8 should beat both parents

1. iter7's raw F1 already led; removing the cost penalty alone flips the
   iteration-7 ranking (~+1.5 pts).
2. Transport resilience converts observed hard zeros into normal scores
   (semantic_192 alone was worth ~2.6 batch pts this round; 502 bursts will
   recur on held-out runs).
3. Never-empty fallbacks put a floor under metadata/specific outage cases.
4. The ambiguous-alias hedge targets a now twice-observed gold pattern
   (multi-work and multi-record gold sets) with strictly-positive expected
   value under the F1 formula, gated so cued single-answer queries are
   untouched.
