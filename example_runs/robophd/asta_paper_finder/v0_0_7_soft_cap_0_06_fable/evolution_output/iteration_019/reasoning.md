# Iteration 19 — Analysis and Plan

Name: iter19_breadth_restored

Base: `iter18_triage_first`'s codebase with its retrieval stack reverted to
`iter13_grade_view`'s. Equivalently: iter13 — tied-best raw F1 over the last
two batches — plus only iter18's output-side and exact-match-path
improvements.

## 1. What the data says

### Standings (iteration 18, all 14 queries semantic)

| Agent | Score | Raw F1 | Mean cost |
| --- | --- | --- | --- |
| iter18_triage_first | 34.04 | 0.340 | $0.0484 (free zone) |
| iter12_body_conjunction | 33.41 | 0.386 | $0.0745 (−5.19 pts penalty) |
| iter13_grade_view | 32.26 | 0.370 | $0.0732 (−4.71 pts penalty) |

iter18 "won" the batch on the cost penalty alone while losing raw F1 by 4.6
points to iter12. Two things make that penalty misleading for the held-out
test: (a) this batch was 14/14 semantic, an ~1%-probability draw from a
73%-semantic pool — at the test mix, iter12/13's ~$0.074/semantic projects to
~$0.055 mean, inside the free zone; (b) raw F1 is what generalizes.

### The pre-registered diagnostic fired

Iteration 18's reasoning ended with: run the grade-3 attribution against
iteration_018 and compare `not_retrieved`; "if it rises above iter13's 28.3%,
[the gap-fill removal] is the first thing to undo." I ran it (`attrib18.py`,
this directory). For every paper ANY agent got graded 3, classify why each
agent did or didn't get credit:

| agent | got_it | not_retrieved | evidence_lost | stranded |
| --- | --- | --- | --- | --- |
| iter12_body_conjunction | 57.2% | **33.2%** | 5.7% | 3.9% |
| iter13_grade_view | 54.4% | 35.2% | 5.1% | 5.4% |
| iter18_triage_first | 53.2% | **37.3%** | **4.4%** | 5.1% |

iter18 has the *worst* retrieval-breadth bucket of the three, and it is the
second consecutive batch where the iter17/18 retrieval stack (14-query
always-on planner, no gap-fill, SNIP_INIT_LIMIT 100) delivered worse breadth
than iter13's (10-query planner + reactive gap-fill + SNIP_INIT 50):
iteration 17 measured 35.1% vs 28.3% for the same comparison. Meanwhile
iter18's *output-side* changes are confirmed good — its `evidence_lost` is
the best of the three (4.4%), and it is by far the cheapest.

The mechanism is the same one iter16/17 established for pool widening, one
level upstream: `POOL_CAP` is fixed at 320, so a bigger retrieval union does
not add pool slots — it adds contenders for fixed prescreen slots, and every
extra contender is a chance for the lexical prescreen to displace a paper the
triage would have ranked into the head. 14 planner queries and 100-passage
initial snippet calls both grow the union; neither grows the pool.

### Base selection: iter12 vs iter13

Head-to-head over the 28 problems both ran in iterations 17–18: mean 0.389 vs
0.394, wins 11–13. Statistically tied. I base on iter13's stack because its
grade-view fix is mechanistically sound (internal graders read the fetched
snippets; iter12's rescue/verify grade an abstract-only view), and because
the iter18 codebase — which already contains the metadata-path fixes and
evidence improvements I want to keep — is iter13-derived, making the revert
surgical: three localized changes rather than a large cross-port from iter12.

## 2. Changes vs iter18_triage_first (all reverts toward the measured winner)

1. **Planner reverted to iter13's 10-query prompt** (byte-identical,
   verified). The 14-query diverse-category planner was the iter17-line's
   last non-reverted retrieval change; breadth was measurably worse in both
   batches it ran.
2. **Gap-fill restored** (byte-identical to iter13, verified):
   `POOL_CAP_TOTAL 380`, `GAP_MIN_PERFECT 20`. When stage-1 triage finds <20
   predicted-perfect, one GPT_5_4 call proposes 5 fresh queries and the new
   docs are triaged in (`t1gap`). Reactive breadth (~$0.005 only when it
   fires) beats always-on breadth (14 queries diluting every query's union) —
   that is precisely what the two attribution tables say.
3. **SNIP_INIT_LIMIT 100 → 50** (iter13 value), same fixed-slots argument.

Everything else in iter18 is kept, because each piece is output-side or
exact-match-path and never touches the semantic pool: compact triage output
(local indices, unspaced digit grades — parse-identical, output bills at 6×
input), longer submitted evidence (abstract 2000 / snippets 900) with
containment dedup (`evidence_lost` 5.1% → 4.4%), `SNIPPET_TIMEOUT` 90 and
tail sweep to 250 (free tool calls), the metadata conjunction/intersection
fixes and chunked venue filter, and the ambiguous-specific
retrieve-wide-submit-tight geometry.

## 3. Why this should score higher

- **Against iter18**: its only measured regression vs iter13 is retrieval
  breadth, and all three reverts target exactly that bucket, which is also
  the largest loss bucket overall (33–37% of all missed grade-3 credit).
  Everything iter18 measurably improved (evidence, cost mechanics, metadata
  path) is retained.
- **Against iter12/iter13**: same (tied-best) semantic retrieval and triage,
  plus iter18's better evidence assembly, minus ~$0.003/query of triage
  output cost. At iter13's measured $0.0732/semantic minus compact-format
  savings, this lands near $0.070/semantic → ~$0.052 batch mean at the
  73%-semantic test mix, inside the free zone where iter12/13 paid 4.7–5.2
  points of penalty on this batch. An all-semantic batch would cost ~0.5
  error-equivalents of penalty; the expected-mix case is comfortably free.
- **Posture**: at n=14 with ±5-point batch noise, every change here is either
  a revert toward the two-batch measured winner or an already-measured free
  improvement. No new mechanisms.

## 4. Verification

- `attrib18.py` (this directory) — the attribution run backing the tables
  above; re-run against iteration_019 next round. The direct test of this
  iteration's claim: `not_retrieved` should drop back to ≤35% (iter13's
  level or better) while `evidence_lost` stays ≤5%.
- Structural verification: AST-level diff vs iter18 shows exactly the three
  intended reverts (constants, planner prompt, gap-fill block); the restored
  gap-fill block and planner prompt are byte-identical to iter13's; pyflakes
  clean.
- `smoke_test.py` (65 checks, all passing) — iter18's harness-stubbed suite
  adapted to this iteration: asserts the reverted constants
  (SNIP_INIT_LIMIT 50, POOL_CAP_TOTAL 380, GAP_MIN_PERFECT 20), the restored
  gap-fill wiring (t1gap label, predicted-perfect gate, POOL_CAP_TOTAL cap),
  the 10-query planner parse and fallbacks, plus the inherited surface:
  compact grade parsing with local→global mapping, evidence assembly
  (containment dedup, verbatim grounding, ≤8 passages), metadata conjunction
  wiring, chunked venue filter, and the ambiguous-specific caps.
