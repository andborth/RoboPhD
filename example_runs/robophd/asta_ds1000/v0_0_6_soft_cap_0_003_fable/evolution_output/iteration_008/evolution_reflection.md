# Evolution Reflection — Iteration 8 (DS-1000)

## What worked well

**Failure autopsy before design.** The single highest-value activity was reading
the per-problem diagnostics (`problem.md`, `extracted_code.md`, `test_result.md`,
`reference.md`, `agent_stdout`) for every failed problem side by side. In
iteration 7 this immediately revealed that 3 of 4 failures were the *same*
failure mode (the executed value contradicted the expected output printed in
the problem, and the flash-lite triage said "MATCH" anyway). Aggregate reports
tell you *which* problems failed; only the per-problem artifacts tell you *why*,
and the "why" is usually one shared subsystem, not four unrelated bugs.

**agent_stdout as a pipeline trace.** The champion agent printed a one-line log
at every decision point (candidate sizes, agreement, escalation, audit verdict,
final tag). This made the autopsy take minutes instead of hours — I could grep
`audit:` across all 20 problems and instantly see the audit fired 13 times and
never correctly flagged a mismatch. Future instances: keep and extend this
logging convention; it is the main debugging surface, since you cannot rerun
old batches.

**Verifying the fix idea on real data before writing the agent.** Before
touching agent.py, I prototyped the token-containment scorer in a scratch
script against the *actual* failed problems' expected/actual outputs. This
caught two real bugs (NaN crashing `canon_num`; symmetric ratio unfairly
penalizing rendered index columns → switched to asymmetric containment) and let
me pick thresholds (0.97 flag, 0.08 adoption delta) from measured scores
(wrong: 0.70–0.88, correct: 0.93–1.0) rather than guesses. Ten minutes of
prototyping bought calibrated confidence.

**Incremental evolution of a validated champion.** The winning lineage
(iter6 → iter7 → iter8) changed exactly one subsystem per iteration and kept
everything else byte-identical, including preserving the old behavior as the
fallback path when the new signal is unavailable. This bounds regression risk:
the new agent can only differ from the champion on the slice where the new
signal exists. Rewrites from scratch lose the accumulated fixes encoded in
prompts and guards (each guidance rule in iter7's prompt is a fossil of a past
failure).

**Mechanical gates over LLM judgment for verification.** A recurring lesson
across the lineage: cheap LLMs are decent *generators* and *copiers* but bad
*judges*. Flash-lite could not spot `[8,...]` vs `[7,...]`. The reliable
pattern is: use the LLM for the easy subtask (verbatim-copy the expected block
out of the problem), then do the comparison deterministically in Python
(tokenize + difflib). Similarly, gate every LLM-proposed "fix" mechanically
(runs clean, strictly improves a measurable score, doesn't hardcode literals)
— iter5 lost a problem by trusting an ungated skeptic fix.

## What was challenging / time-consuming

- **Path confusion**: `../../iteration_007/` resolves to the experiment root,
  not `evolution_output/iteration_007/`. My first two Reads failed. The prompt
  says "paths relative to evolution workspace" but an absolute path or a
  one-line `tree -L 1` of the experiment root in the prompt would save a probe.
- **Distinguishing detectable from undetectable failures.** Problem 165 (the
  reference builds the answer via `np.column_stack`, silently making a counts
  column *string*-typed; visually identical printout) is undetectable from the
  visible text. Recognizing quickly that one failure class is not worth
  chasing — and saying so explicitly in reasoning.md — avoided sinking time
  into overfitted single-problem hacks.
- **Reasoning about false-positive risk of new gates.** Most design time went
  not into making the new audit catch the 3 failures, but into proving it
  couldn't *flip a correct answer to wrong* (asker typos in expected blocks,
  extraction grabbing the input-demo block, hardcoding fixes that ace the
  example). Every new verification signal needs an adversarial pass: "how could
  this signal be wrong, and what happens then?"
- **Sandbox write scope**: `/tmp` is blocked; scratch scripts must go in the
  iteration dir. Documented, but easy to forget.

## Tooling notes

- The provided artifacts (error_analysis_report, per-problem dirs, cost_report)
  were sufficient and well-organized. `reference.md` (the hidden reference
  solution for *past* problems) is gold — it's the only way to learn grader
  quirks like the string-date sort in 238 or the column_stack dtype in 165.
- No `strategy_tools/` existed this round; not missed.
- `jq`/`tree` fine. Everything needed was doable with Read/Bash/grep.

## What I would do differently

- Start with a cross-iteration failure taxonomy (5 minutes of grepping older
  iterations' error reports) before reading any code, to see which classes
  *recur* vs were one-offs. Recurring classes justify architecture; one-offs
  justify at most a guidance line.
- Budget cost arithmetic earlier. The free zone is a batch *average*
  ($0.003/problem), and the champion sat at $0.0023, so I had a concrete
  $0.0007 envelope for new calls. Knowing that number up front shaped the
  design (one cheap extraction call, reuse cached, mechanical everything else).

## Advice for future instances

1. **Read the failed problems' full artifacts first.** The aggregate report is
   a table of contents, not the analysis.
2. **Find the shared subsystem behind the failures.** Fix systems, not
   problems. A per-problem patch overfits; the batch changes every iteration.
3. **Prototype any new scoring/detection logic against real past data in a
   scratch script** and set thresholds from measured separation.
4. **Never let an LLM be the final judge of correctness when a deterministic
   check is constructible.** Execution equality, token containment, compile
   gates, AST guards — all beat a cheap model's opinion.
5. **Gate every automated "improvement" so it can only replace an answer when a
   measurable signal strictly improves**, and adversarially ask what makes the
   signal lie (hardcoding is the classic way to ace the example and fail
   hidden tests).
6. **Respect the cost model precisely**: compute your headroom in dollars per
   problem and spend it where the marginal-accuracy story is concrete. Staying
   under $0.003 is worth ~1 correct answer per $0.001 saved beyond it.
7. **Keep the champion's stdout-logging discipline** — one line per decision
   point, greppable tags. Your successors will autopsy your agent the way you
   autopsied your predecessors'.

## Suggestions for the prompt/process

- Give the absolute path (or a small tree) of the experiment root; the
  `../../` convention cost a couple of failed reads.
- Consider carrying a short "failure taxonomy so far" document across
  iterations (class → example problems → status: fixed-by/open/unfixable).
  Each instance currently reconstructs this from scratch; it's the single most
  reusable piece of cross-iteration knowledge.
- The requirement that reasoning.md include expected effects per failure class
  is good practice even where not mandated — it forces falsifiable predictions
  the next iteration can check against fresh results.
