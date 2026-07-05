# Iteration 8 Reasoning

Name: iter8_expected_diff_cascade

## What iteration 7 showed

`iter7_lean_audited_cascade` is the champion: 16/20 raw (80%), mean cost $0.0023
(inside the $0.003 free zone), beating both older agents on raw accuracy AND on
cost. Its architecture (dual cheap generation → sandbox execution → majority
vote with Haiku escalation → slim GPT_5_4 arbiter → example-output audit →
style passes) is validated across three iterations. I keep it and fix the one
subsystem that demonstrably failed.

### Failure autopsy (all four iter7 misses)

| Problem | Root cause | Was it detectable from the problem text? |
|---|---|---|
| 238 | Majority (B==C) picked datetime-sort ordering; the problem **prints** the expected table, which shows string-date ordering (01-Feb before 01-Jan for id=3) | Yes — printed table contradicts executed value |
| 445 | Submitted `[8,6,3,1,...]`; the problem **prints** `array([7, 6, 3, 1, 3, 6, 3, 1])` | Yes — first element differs |
| 269 | Column names built in wrong order (`A_1,A_2,A_3,...`); the problem **prints** the desired header `A_1,B_1,C_1,...` | Yes — header differs |
| 165 | Value/rendering identical to printed output; reference builds via `np.column_stack` so the counts column is *string* dtype | No — dtype is invisible in the printout |

The stdout logs show the audit RAN on all four and flash-lite triage replied
"MATCH" every time. **Detection, not coverage, is the weak link**: a
$0.0002-class LLM cannot reliably diff a rendered table against a printed
expected block. Three of four failures are recoverable with a comparison that
actually works; 165 is not detectable from the visible text and I accept it.

## The change: mechanical expected-output diffing

Replace the flash-lite MATCH/MISMATCH judgment with a two-step scheme:

1. **Extract once** (flash-lite, verbatim-copy task — easy for a weak model):
   quote the desired-output block the asker printed, or NONE. The extraction is
   validated mechanically: ≥60% of its lines must literally appear in the
   problem text (kills hallucinated blocks), it must have 5–400 comparison
   tokens, and it is discarded if it is ≥85% token-contained in the skeleton
   source (which would mean it's the *input* data, not the desired output —
   guards against grabbing the wrong block, which would otherwise be the one
   way this feature could flip a correct answer to wrong).

2. **Compare in Python** (free, deterministic): tokenize both the expected
   block and the sandbox-rendered value into canonicalized number/word streams
   (float-tolerant, noise words like dtype names filtered, commas/# stripped),
   then score = order-aware token containment of the expected sequence in the
   rendered sequence (difflib matching blocks / len(expected)). Containment
   rather than symmetric ratio so that a rendered index column or dtype footer
   never penalizes a correct answer.

Verified on the actual iter7 failures (scratch_token_test.py):
wrong answers score 0.70 (269), 0.824 (238), 0.875 (445); correct answers score
0.933–1.0 (0.933 is 269, whose printed expected contains asker typos —
`B_2_`, `14,5` for `14,15`).

The score drives three decision points:

- **Selection**: when candidates disagree, score each clean candidate against
  the expected block *before* escalating; a candidate with containment ≥0.98
  and ≥0.08 margin over every rival wins outright — this both overrides
  wrong-majority coalitions (the 238 failure mode) and skips the Haiku/arbiter
  spend when the printed output already settles the question.
- **Audit detection**: final containment <0.97 ⇒ mismatch (445: 0.875 caught,
  269: 0.70 caught, 238: 0.824 caught, vs flash-lite's three false MATCHes).
- **Fix adoption**: an audit fix is adopted only if it runs clean AND improves
  containment by ≥0.08 AND doesn't hardcode (≥4 numeric literals from the
  expected block, |v|>2, ⇒ auto-reject). The delta gate is what makes false
  positives safe: on 269's typo-laden expected block a correct original scores
  0.933 and no fix can reach 1.013, so noise can never displace a correct
  answer — it only costs one wasted mini call.

When no expected block exists (extraction says NONE or fails validation), the
audit falls back to iter7's flash-lite triage + line-score-gated fix,
preserving champion behavior exactly on that slice.

Also: the fixer prompt now includes the expected block verbatim plus a
mechanically computed first-difference pointer ("expected 7 but got 8 at
position 1") — far stronger repair signal than flash-lite's 12-word reason.

## Everything kept from iter7

Guidance prompt (incl. anti-hardcode, vectorize, SO-literalism rules),
function-completion single-mini path (100% across all batches), anti-hardcode
AST strip, traceback repair, majority vote + slim WINNER arbiter, signature
guard, matplotlib reflection, loop-free rewrite pass, graceful sandbox
degradation. The mechanical full-line match gate stays as the free first check
(now comma/#-normalized, strictly more permissive).

## Cost budget

iter7: $0.0023 mean, $0.0007 headroom. Extraction is one flash-lite call
(~900 in / ~250 out ≈ $0.0006) on the ~65% of problems that reach the triage
branch, minus the retired triage calls (~$0.0002 each), minus skipped Haiku
escalations/arbiter calls when the expected block settles a disagreement.
Net estimate: +$0.0002–0.0004 ⇒ ~$0.0026 mean, still inside the free zone.

## Expected effect

- 238/445/269-class (printed-output contradiction): detection now mechanical
  and demonstrated on all three; recovery needs only that mini can write a
  matching fix given the exact target and diff — likely for 238/269, plausible
  for 445. These classes recur across batches (iter5/6 also lost 238-style
  problems), so this generalizes.
- 165-class (invisible dtype quirk): unaddressed; no reliable signal exists.
- Regression risk: bounded by the ≥0.98 selection threshold, the ≥0.08
  adoption delta, the hardcode guard, and the skeleton-containment extraction
  guard; NONE-path behavior is byte-identical to the champion.
