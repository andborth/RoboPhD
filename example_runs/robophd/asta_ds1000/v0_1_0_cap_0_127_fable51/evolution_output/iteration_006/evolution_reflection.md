# Evolution reflection — iteration 6 (ensemble-exec-vote-escalate)

## What worked well

- **Treating a 20/20 batch as a logging exercise, not a scoring exercise.** Three agents at 100% carry
  no ranking signal. One shell loop over `agent_stdout` (`agreement groups=` + `judge chose|fallback pick`)
  per problem produced the decision-rule distribution in a minute and exposed the only real defect of the
  session: two three-way splits resolved by `fallback pick` because the judge never ran.
- **Checking the agent's own cost estimate against `eval_cost` in `result.json`.** The estimator added
  reasoning tokens on top for every provider; billed cost shows only Gemini does that. Two samples were
  enough to pin the formula to four digits and to prove the soft cap had silently disabled the judge.
  This is the kind of bug no amount of prompt tuning would have found.
- **Patching the previous agent with asserted string replacements** (`patch_agent.py`, each `rep()`
  asserts the anchor occurs exactly once). A 1137-line agent gained ~90 lines with a reviewable diff
  and no accidental edits. Fourth session in a row this pattern has paid off.
- **Extending the dry-run harness with per-candidate code overrides** (`A=<code> ... E=<code>`).
  Real problem artifacts could not produce a pairwise split locally (the local pandas agrees where the
  sandbox's did not), so hand-written candidates (`df1.head(2)`, `df1.head(3)`, ...) exercised every
  new branch (`[2,1,1,1]`, `[2,2,1]`, five distinct, stall during escalation) in seconds.
- **Spending the unused free budget only on the disagreeing ~25% of problems.** The free zone is a
  batch average; with $0.048 mean against $0.127, escalation plus judge reasoning on the hard subset
  costs ~$0.02 in the mean and leaves unanimous problems untouched.

## What was challenging / time-consuming

- **Timeouts hard-coded inside the new code.** The escalation used `timeout=240` instead of the
  module constant, so the stall test waited the full 240 s before I noticed; the env-override harness
  only works when every timeout is read from a module-level name at call time (the memory note
  already said this and I still did it once).
- **Local dumps differ from sandbox dumps**, so "replay iteration N's split" is not possible from
  artifacts; synthetic candidates are the only reliable way to test decision rules.
- **Foreground command limit (120 s)** is shorter than one stalled-model scenario; run every
  dry-run batch in the background from the start.

## Tools

- No `strategy_tools/` again. The de-facto tools remain: `jq -c '[.score,.eval_cost,.eval_wall_clock_seconds]'`
  over `result.json`, a grep loop over `agent_stdout`, `test_dryrun.py`, and `patch_agent.py`. All four
  now live in `evolution_output/iteration_006/` and are reusable as-is (the harness now stubs
  `GPT_5_5` and accepts `A=..E=` overrides).
- The memory directory was writable this session (previous sessions reported denials); I recorded the
  billing formula there.

## What I would do differently

- Read the model-call cost lines in `agent_stdout` (`opus: in=.. out=.. reas=..`) against `eval_cost`
  on the very first problem inspected; it takes thirty seconds and would have surfaced the estimator bug
  in iteration 3 already.
- Build the synthetic-candidate scenarios before touching the agent, so each patch is validated as it
  lands instead of in one batch at the end.

## Insights on the evolution strategy

- **At this accuracy level the remaining risk is concentrated in one branch**: the three-way split
  (~25% of problems). Unanimous and 2-of-3 majorities have been right on every completed problem except
  the two reference-dtype quirks. Future work should measure that branch specifically: how often it
  fires, how the five voices split after escalation, and whether the judge's pick agrees with the
  largest group. The new `keys <tag>: strict=.. values=..` log lines make that countable offline.
- **Guard rails can be the bug.** Soft caps, timeouts and fallbacks are added for safety and are
  exactly the code paths never exercised by a passing batch. Every cap should log when it fires, and
  the estimator behind it should be checked against ground truth once.
- **Clone detection still penalises faithful carry-over.** Identical 20/20 scores on a tiny batch are
  not evidence of identical agents; iteration 5's agent was flagged a clone of iteration 2's despite
  different code and behaviour. Expect the same for this agent on an easy batch.
- The irreducible class (284-style: no candidate produces the reference's accidental dtype) is still
  ~1 in 50 problems; I did not spend time on it, per the previous reflection's advice.

## Suggestions for the prompt / process

- Report, per agent and problem, the **decision path** (`unanimous / majority / escalated / judge /
  dtype-split / fallback / static`) and `est_cost` vs `eval_cost` in the iteration summary. Both are
  one grep away but every session rebuilds them.
- State the **per-provider billing rule** in CLAUDE.md (reasoning tokens are included in
  `output_tokens` for OpenAI/Anthropic, billed on top for Gemini). Agents that budget themselves need it.
- Ship `test_dryrun.py` and `patch_agent.py` as `strategy_tools/`; the harness has been copied by hand
  for five sessions.
- Carry the verified-mechanism table forward (165 string counts, 284 float column, 238 string-date
  sort, 883 cut_tree labels, 269 comprehension tokens, 142 hard-coded levels, 808/822/906 harness kills,
  662/918 judge skipped by the estimator).
- Fix clone detection to compare extracted code, not scores.
