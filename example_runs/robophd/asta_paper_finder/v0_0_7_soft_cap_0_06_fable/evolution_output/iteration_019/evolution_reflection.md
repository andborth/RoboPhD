# Evolution Reflection — Iteration 19 (iter19_breadth_restored)

## What worked well

1. **Pre-registered diagnostics are the single most valuable inheritance.**
   Iteration 18 left behind `attrib17.py` and an explicit instruction: "run
   this against iteration_018 and compare `not_retrieved` against 28.3% /
   35.1%; that single number is the direct test of this iteration's central
   claim." Ten minutes into this session I had a decisive, quantitative
   answer (37.3% — the predicted failure mode had occurred), and the entire
   iteration's design fell out of it. Every iteration should end by leaving a
   falsifiable prediction plus the script that tests it.

2. **The grade-3 attribution framing** (got_it / not_retrieved /
   evidence_lost / stranded, computed over the union of papers ANY agent got
   graded 3) is far more actionable than per-query score deltas. It converts
   "agent A beat agent B by 0.05" into "agent B never retrieved 20 more of
   the known-good papers," which names the subsystem to fix. Reuse it; it's
   cheap (pure parsing of judge_verdicts.md + submission.json).

3. **Reverting to the measured winner instead of inventing mechanisms.** At
   n=14 with ±5-point batch noise, the only defensible changes are (a)
   reverts toward measured winners and (b) provably free improvements
   (output-format compaction, longer verbatim evidence, tool-call knobs —
   tool calls cost nothing). The iter16→17→18 arc lost points three times by
   bundling a speculative retrieval change with genuinely good output-side
   changes; the win this round came from unbundling them.

4. **Building by surgical revert rather than cross-port.** I constructed the
   new agent by copying the newest codebase (iter18, which had the metadata
   fixes and evidence improvements) and reverting three localized retrieval
   changes back to iter13 — then verified the restored blocks were
   byte-identical to iter13's via a script, and the full code diff vs iter18
   contained exactly the three intended hunks. This is much safer than
   hand-merging features from two 2,200-line files.

5. **Stdout telemetry with per-stage token counts** (`llm-usage: t1:n=12,
   in~25996t,out~959t; ...`) made cost attribution trivial: a 20-line script
   priced every stage per query and identified which model each stage uses.
   Keep this telemetry in every agent; it's what makes cost engineering
   possible.

6. **Adapting the inherited smoke test** (65 harness-stubbed checks) took
   minutes and caught nothing this time — which is itself evidence the edits
   were clean. The stub-the-harness pattern (fake inspect_ai + model_registry
   modules) is the right way to test these agents offline.

## What was challenging / time-consuming

1. **Confounded bundles.** iter17 changed seven things at once; iter18
   reverted some. Attributing iter18's 4.6-point raw loss among {14-query
   planner, removed gap-fill, SNIP_INIT 100} was impossible from the data —
   all three touch pool composition. I reverted all three as a bundle toward
   the winner. The lesson upstream: **change one retrieval-affecting thing
   per iteration.** Output-side changes (evidence text, prompt format) can be
   bundled freely because they're separable by construction; pool-affecting
   changes cannot.

2. **Batch composition noise vs cost penalty.** Iteration 18's batch was
   14/14 semantic (an ~1%-probability draw from a 73%-semantic pool), which
   made iter12/13 look cost-penalized when at the true test mix they project
   inside the free zone. The reported "Score" column can therefore invert the
   true ranking. Always recompute: (semantic-only cost) × (expected semantic
   share) before believing a penalty, and design to ~$0.07/semantic rather
   than to the batch mean.

3. **error_index.json structure** wasn't self-describing (by_agent held only
   summary stats, not per-problem scores); I fell back to globbing
   `problems/*/result.json`. Also note cached problems are symlinked across
   iterations — identical scores for the same problem in consecutive
   iterations are re-used results, not replications, so don't double-count
   them as independent evidence.

## Advice to future instances

1. **Read the previous iteration's reasoning.md FIRST** (in
   `../iteration_NNN/`, the prior working dir — not just the agent
   docstrings). It contains the predictions, the cost model, and the "first
   thing to undo if X" instructions. The agent docstrings duplicate much of
   it, but the working dir also has the analysis scripts.

2. **Trust accumulated cross-batch evidence over single-batch deltas.** By
   iteration 18, iter12 and iter13 had 28 shared problems: mean 0.389 vs
   0.394, wins 11–13 — statistically tied, and both clearly above the
   iter17/18 line. A single batch had shown iter13 "winning by 2.3 points";
   the next showed the reverse. Maintain the head-to-head table across all
   available iterations before picking a base.

3. **Know the score mechanics cold; they dictate priorities.** Recall counts
   only grade-3 papers in the first K positions; grade 3 requires EVERY
   weighted criterion judged Perfectly Relevant from your submitted evidence
   alone; rank is nearly saturated (0.36–0.51) while recall runs 0.00–0.23.
   So the leverage order is: (1) retrieve the grade-3 papers at all
   (not_retrieved is 33–37% of lost credit), (2) get them evidence that
   proves every criterion, (3) order them into the top ~K. Cost engineering
   matters only after those.

4. **The established negative results — do not relitigate without new
   evidence:** widening POOL_CAP (320→400→640 all lost), widening the
   retrieval union into a fixed prescreen (14-query planner, SNIP_INIT 100),
   and wide ambiguous-specific submission (unconverted slots are pure
   precision loss). The recurring failure mode of this whole experiment is
   "more candidates for fixed slots." Conversely: reactive breadth
   (gap-fill when predicted-perfect < 20) is measured-good.

5. **Free levers checklist** (verified free by the scoring rules): longer
   verbatim evidence up to 8 passages (no length cap; grounding is
   per-passage); tool calls in any quantity; compact LLM output formats
   (output bills ~6× input); `print()` telemetry. Spend these before spending
   dollars.

6. **Leave your successor what you inherited:** (a) a falsifiable prediction
   with a threshold ("not_retrieved should drop to ≤35%"), (b) the script
   that tests it (`attrib18.py` — repoint one string), (c) the adapted smoke
   test, (d) per-stage cost telemetry intact.

## Process/prompt suggestions

1. **Include batch composition in the results summary.** The headline table
   gave Score and cost penalty but not "14/14 queries were semantic_f1" —
   the single fact that explains why the penalty was atypical. One line
   ("query mix: 0 specific / 0 metadata / 14 semantic") would prevent
   misreading the ranking.

2. **Provide per-problem scores in error_index.json** keyed by agent →
   problem → score, and mark which results were cache-reused from a prior
   iteration.

3. **Consider evaluating 2 variants per iteration when the change is
   separable** (e.g., base+outputchanges vs base+outputchanges+poolchange).
   The three-agent comparisons across iterations were the only reason
   attribution was possible at all; a deliberate ablation pair would be even
   better than the accidental lineage comparisons.

4. **A standing warning in the prompt** worth adding: "Batch mean cost is
   computed on the drawn batch; batches vary in query-type mix. Project cost
   to the 73%-semantic test mix before trading quality for cost." I nearly
   over-weighted the cost penalty before checking the mix.
