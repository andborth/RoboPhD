# Evolution Reflection — Iteration 15 (asta_paper_finder)

## What worked well

**Reading judge_verdicts.md against score_calculation.md before anything
else.** Ten minutes of jq/grep over the per-problem diagnostics produced the
round's two decisive facts: (1) the judge grades *exactly the first K
submitted papers* (verified across all 12 semantic problems: judged count ==
K every time, K ranging 12–232), and (2) metadata_31's failure was a dead API
surface, not a strategy gap. Neither fact was visible from the score table.
The diagnostic tree is the highest-information-density artifact in the whole
experiment; the score table is the lowest.

**Following the lineage's paper trail.** reasoning.md + evolution_reflection
from iterations 13–14 saved me from two traps I would otherwise have walked
into: re-deriving a lexical Perfect-vs-Highly ordering proxy (refuted by
calibrate.py against 1208 ground-truth verdicts) and over-crediting semantic
changes that batch noise can't distinguish. A well-maintained docstring
changelog in a 2500-line agent is what makes grep-first navigation viable —
keep writing it.

**Mechanism-backed changes only.** The two changes I shipped both have a
measured defect behind them: (a) `references` fails server-side in every
observed run while `get_citations` (the inverse surface) works — so verify
"cites author X" by unioning citers of X's papers; (b) all three agents paid
a cost penalty and the winner cleared the threshold by $0.0002 — so trim
stages with measured near-zero marginal product (rescue promoted 0–1
papers/query; +100 expansion docs → +0 predicted-perfect). I dropped every
idea I couldn't tie to a mechanism, because 14 iterations of data show ±5
points of batch noise on identical code.

**Reusing and extending the inherited smoke test.** Copying iter14's
smoke_test.py, updating the two stale constant assertions, and adding stubbed
tests for the new `_citing_set` helper took ~15 minutes and caught a real
scoping bug during development (inverse membership initially tested only
against the 300-capped forward-check slice, not all kept candidates).

## What was challenging / time-consuming

- **The 2500-line agent.** Whole-file reads are impossible; the workflow that
  works is: grep the function/constant outline first, read only the 200-line
  regions you intend to change, and trust the docstring changelog for the
  rest.
- **Attributing score differences.** iter14 "won" iteration 14, but its raw
  mean was *below* iter13's — the entire win was cost. Without decomposing
  final score into raw-vs-penalty per agent, I'd have credited the semantic
  changes. Always separate the two terms before deciding what to build on.
- **Shell quoting friction.** Heredocs and `===` echo separators kept
  tripping zsh; small annoyance, real minutes lost. Use `--- ` separators or
  python one-liners.

## Insights about the task itself (for future instances)

1. **Judged depth = K, exactly.** Rank and recall are both computed inside
   the first K submitted positions. K is unknown at query time (observed
   12–232, median ~52). Implications: submit the full 250 always (free
   insurance for large K); head ordering is *recall*-critical on small-K
   queries because a grade-3 paper at position K+1 earns zero; effort spent
   ordering positions beyond ~100 buys almost nothing.
2. **Recall binds everywhere.** rank 0.30–0.90 vs recall 0.05–0.26 on every
   observed semantic query. The Highly-not-Perfect mass (papers one
   criterion short of grade 3) is the standing frontier — and per
   calibrate.py, no cheap lexical proxy identifies which papers will convert.
   Pool composition and criterion-targeted evidence are the levers left.
3. **The cost threshold is a cliff you will hit on semantic-heavy batches.**
   The nominal mix is 73% semantic; iteration 14's batch was 86%, and that
   alone pushed all three agents over $0.06. Budget the *semantic* per-query
   mean to ~$0.058 so an unlucky mix still clears.
4. **Tool-surface failures are score bugs.** Two of the biggest single-query
   losses in the lineage (metadata_4 venue truncation, metadata_31 dead
   references) were plumbing, not intelligence. When a query type scores
   near zero, read agent_stdout for tool errors before theorizing.
5. **The `references` field is dead; `get_citations` is alive.** Any
   citation-direction constraint should be verified via citers of the
   *target*, not references of the *candidates*. Track completeness (the
   1000-cap) and fail open when incomplete.

## What I would do differently

- **Verify the eval ran my new code path.** I shipped the inverse-citations
  fix on the strength of a stubbed unit test. If a metadata-citing query
  appears in the next batch, the first thing to check is its agent_stdout
  for the `inverse-citations:` line — and future instances should generally
  budget a few minutes to trace one real transcript per new mechanism.
- **Quantify trims against the cost telemetry before choosing them.** I did
  this (per-stage token counts from `llm-usage` lines × price table) and it
  changed my picks — the verify chunk size mattered more than any depth
  constant. Doing it first, rather than after drafting the plan, would have
  been faster.

## Suggestions for the process / prompts

- **Put the raw-score vs cost-penalty decomposition in the score summary
  table**, not only in prose "aggregate notes". The single most misleading
  number in the report is the final score when penalties differ.
- **State the judged-depth-equals-K fact in CLAUDE.md.** It is derivable
  ("first K judged papers") but easy to miss; two lineage iterations
  half-used it. One explicit sentence — "papers past position K are never
  judged" — would save every future instance the re-derivation.
- **Keep `calibrate.py`-style ground-truth checks as a standing norm.** The
  prompt could say: before wiring any new ranking/grading signal, correlate
  it with real judge verdicts from prior iterations' diagnostics. This one
  habit killed the lineage's worst almost-shipped change.
- **Batch noise dwarfs most changes at n=14.** If the harness could report a
  per-query paired comparison vs the parent agent (same queries, same
  round), sign-tests would replace "it won this batch" as the evidence
  standard.

## Tools

Filesystem + grep + jq + python one-liners covered everything; no missing
tooling. The strategy_tools directory was absent this round — the manual
workflow above is a fine substitute, but a script that dumps (per query:
K, rank, recall, grade histogram, cost split by model/stage) as one table
would compress the first hour of every session into one command.
