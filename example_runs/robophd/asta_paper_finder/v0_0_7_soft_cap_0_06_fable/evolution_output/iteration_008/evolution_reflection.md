# Evolution Reflection — Iteration 8 (iter8_resilient_lean)

## What worked well

1. **Reading failure *shapes*, not just scores.** The decisive findings this
   round were all in the diagnostics, not the score table:
   - semantic_192's 0.000 looked like a retrieval failure until
     `agent_stdout` showed every tool call 502ing and the agent quitting
     after 14.6s of a 29-minute budget. `eval_wall_clock_seconds` in
     `result.json` (14.6s vs the usual 700-1000s) was the tell — an
     anomalously fast run on a scored-zero query almost always means an
     infrastructure failure, not a strategy failure.
   - specific_39's gold (`gold_criteria.md`) had FIVE corpus ids for "the
     SPIKE paper" — a fact that instantly reframes the specific-path design
     from "identify the one right paper" to "hedge across referents", and
     stdout showed the dup-guard *rejecting* one of the gold ids.
   Ten minutes of `jq` over `result.json` + reading stdout of every
   zero-scoring problem was the highest-yield analysis step.

2. **Verifying scorer mechanics empirically.** Counting judged entries in
   `judge_verdicts.md` against K from `score_calculation.md` proved the
   judge grades exactly the first K positions (K=12..228). That is a
   materially different objective than "250 entries, order-weighted" and
   sharpens where effort belongs (top-of-list on narrow queries). Worth 5
   minutes; changes design priorities permanently.

3. **Separating architecture quality from aggregate score.** iter7 *lost*
   iteration 7 on final score but was clearly the better retriever
   (+0.53 net raw F1 over iter6 on shared semantic queries once the outage
   query is excluded). The right move was keep-the-architecture,
   fix-the-packaging (cost + resilience) — not revert to the "winner".
   Score tables alone would have said "iter6 won, build on iter6".

4. **Copy the parent file + surgical edits + offline tests.** The inherited
   1600-line agent encodes seven iterations of fixes (unicode, relax
   ladders, split author profiles, sim-view). I copied it wholesale and made
   ~25 targeted edits. The factory-conversion of `_safe_tool` touched every
   call site — a grep for unconverted sites plus a stubbed end-to-end test
   (fake tools, fake models, simulated 502 outage) caught one missed site
   and validated all four new control-flow paths in seconds, with zero eval
   spend. Writing the e2e harness (~200 lines) took under half an hour and
   is the cheapest insurance available; previous iterations only smoke-
   tested pure functions and would not have caught a fan-out-loop bug.

5. **Costing from measured data, not char arithmetic.** `cost_report.md`'s
   per-model, per-problem numbers let me back out per-stage costs and size
   the trim (~30% off semantic queries) with confidence that it lands at
   ~$0.050 mean. The iteration-7 reflection's request for per-stage token
   logging is still the right ask; per-model per-problem was almost enough.

## What was challenging / time-consuming

- **Path confusion at the experiment root.** `evolution_output/iteration_NNN/`
  (agent workspaces) vs `<root>/iteration_NNN/` (evaluation results) again
  cost the first several minutes. The prompt's relative paths resolve from
  the working directory; an absolute-path map at the top of the prompt
  would remove this recurring tax (iter7's reflection hit the same thing).
- **Converting ~25 `_safe_tool` call sites from coroutines to factories.**
  Mechanical but error-prone (lambda late-binding in loops requires
  default-arg capture). If the base agent had used factories from the
  start, retries would have been a 10-line change. Advice: design tool
  wrappers as `(factory, label)` from iteration 1 — retryability is a
  property you will eventually need.
- **Cost/score tension is the real optimization surface.** The batch mean
  cost target moves with the semantic share of the sample (71-86%
  observed), so a fixed per-query budget is the wrong mental model. I
  budgeted semantic queries to ~$0.060-0.065 so even an all-semantic batch
  stays under $0.06 mean. Future instances: compute the worst-case mix,
  not the observed mix.

## Tool/process assessment

- Diagnostics are excellent and sufficient: stdout + result.json +
  judge_verdicts + score_calculation + gold_criteria separated
  "never retrieved" / "buried" / "graded 2" / "infrastructure death"
  cleanly. No strategy_tools/ existed this round; the two scripts iter7's
  reflection requested (per-problem K/rank/recall/cost table; cross-agent
  Perfect-paper join) still do not exist and I rebuilt fragments by hand —
  third iteration running that this has been re-derived.
- `jq` + shell over `result.json` files was fast and adequate.

## Advice to future instances

1. **Triage every zero first, and check wall-clock before theorizing.**
   Sort problems by score ascending, read stdout of each zero. A zero with
   a 10-30s wall-clock is an outage/bug; a zero with 700s is a strategy
   gap. Fix the first kind before designing for the second — one recovered
   outage (~0.36 F1) outweighs most architectural tweaks.
2. **Treat resilience as scored behavior.** Tool calls are free and the
   budget is 29 minutes; typical runs use 12-18. Any path that can submit
   0 papers is a latent -1 query. Retry bursts, re-run fan-outs after
   sleeps, and keep a terminal never-empty ladder on every route.
3. **Exploit gold-structure knowledge.** Specific golds can be multi-work
   sets (SPIKE: 5) and multi-record sets (AlexNet: 2). F1 math makes
   hedging strictly positive-EV once ambiguity signals appear (short
   alias, no author/year cues). Conversely, cued references should stay
   single-record — both patterns are now confirmed in training data.
4. **The judge scores only the first K entries.** On narrow queries
   (stage-1 predicted-perfect ≤ 8), everything rides on the top ~25 —
   spend high-fidelity model calls there and nowhere else.
5. **Don't chase per-problem deltas between sibling agents.** Pool
   composition is stochastic; ±0.1 swings on individual semantic problems
   are noise. Trust only deltas with a mechanistic explanation (this is
   now a three-iteration-old lesson that keeps being re-earned).
6. **Budget to ~$0.050 mean assuming a worst-case semantic-heavy batch.**
   The penalty is a cliff; iter5 died at $0.0613 and iter7 at $0.0641.
   Raw-F1 wins of +0.01 are erased by a $0.004 cost overshoot.

## Suggestions for the process/prompts

- Provide the two analysis scripts as `strategy_tools/` (per-problem
  K/rank/recall/grade-histogram/cost/wall-clock table across agents;
  cross-agent judge-Perfect join). Requested twice now; each instance
  spends ~an hour rebuilding them.
- Put an explicit path map in the prompt: "evaluation results:
  `<abs>/iteration_NNN/`; agent sources: `<abs>/agents/<name>/`; your
  outputs: `<abs>/evolution_output/iteration_NNN/`".
- Surface `eval_wall_clock_seconds` and K in `error_index.json` — both are
  primary triage signals currently buried in per-problem files.
- Document the transport retry set (429/529/504) prominently, with the
  note that 502/500/503 are NOT auto-retried. That single sentence would
  have prevented the semantic_192 loss an iteration earlier.
- Consider running each agent's zero-score problems through a "was this an
  outage?" classifier in the error analysis report (wall-clock < 60s +
  tool-failure lines in stdout) so infrastructure deaths are labeled as
  such instead of polluting the strategy signal.
