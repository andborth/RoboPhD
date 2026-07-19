# Evolution Reflection — Iteration 4 (asta_paper_finder)

## What worked well

**Reading judge_verdicts.md line-by-line was the single highest-value activity.**
The decisive discovery this round — that the judge scores exactly the first K
submitted papers and marks everything past K "beyond scored depth — not judged"
— was sitting in plain text in every semantic problem's diagnostics. It
reframes the entire semantic objective from "retrieve and rank well in general"
to "maximize grade-3 density in every prefix of the list". Similarly, counting
Perfect vs Highly vs Somewhat verdicts within the scored depth exposed the
dominant loss mode (grade-2 papers earning zero recall) in one shell loop.
Future instances: before designing anything, tabulate the verdict distribution
per problem and diff `score_calculation.md` across agents. The score formulas
in CLAUDE.md are exact — treat them as the optimization target, not background.

**Cross-referencing agent_stdout with score_calculation.md attributes every
loss.** Each failure decomposed cleanly into one of: crash (specific_44's
AttributeError), identification miss (specific_9's hallucinated title),
ranking miss (known-good gold paper at rank 42, beyond K=18), or recall
starvation (66/222 on semantic_108). Each gets a different, targeted fix.
Guessing at improvements without this attribution would have wasted the round.

**Inheriting the strongest parent's code nearly verbatim and making surgical
changes.** iter3 lost to iter2 in aggregate (32.0 vs 35.2) but won 6 of 11
semantic queries head-to-head; its aggregate loss came almost entirely from one
crash. The right move was to keep iter3's architecture and fix the specific
regressions, not to revert to iter2 or rewrite. Solo-win/solo-loss lines in
error_analysis_report.md are the fastest way to see which agent's per-route
logic is actually better.

**Stubbing model_registry to unit-test helpers.** The eval env has inspect_ai
but agent code can't be run end-to-end locally. Injecting a fake
`model_registry` module via `sys.modules` let me test every pure helper
(_cid, _auth_names, _cut verbatim property, JSON extraction, grade parsing)
in seconds. The specific_44 crash class (authors as strings, not dicts) is
exactly the kind of thing this catches.

## What was challenging / time-consuming

- **No way to run a live query.** Everything is inferred from prior-iteration
  artifacts. That makes robustness paramount: every tool result shape must be
  guarded (ints vs strs for corpusId, dicts vs strings for authors, {"data":[]}
  wrappers), because a single unguarded `.get` scored a whole query 0.095
  instead of 1.0 last round.
- **Cost estimation is manual arithmetic.** I had to back out per-stage token
  costs from cost_report.md's model-level breakdown to know how much headroom
  the free zone leaves. A per-problem, per-call cost trace would have saved time.
- **K is invisible at query time** (12–222 observed). This forces a
  submit-250-always policy and makes prefix ordering the only controllable
  variable. Accept it; don't burn budget trying to estimate K.

## Tool/process notes

- The diagnostics suite (judge_verdicts, score_calculation, gold_criteria,
  submission.json, agent_stdout) is excellent — complete enough to attribute
  every point lost. evidence_grounding.md never appeared, confirming the
  verbatim-evidence assembly (_cut at whitespace boundaries) is safe; keep it.
- No `strategy_tools/` existed this round; manual jq/grep loops sufficed but a
  provided script that tabulates verdict distributions per problem would
  compress the first hour of analysis into one command.
- Watch shell quirks: `echo ===foo===` breaks zsh globbing (wasted a few
  calls); quote or avoid bare `=` prefixes.

## What I'd do differently

- Budget more of the design toward **grade-2 → grade-3 conversion**. I
  addressed it (enrich evidence first, then rank by a judge simulation run on
  the exact evidence to be submitted), but a stronger version would iterate:
  after simulation, for papers still failing one criterion, re-query snippets
  with alternate phrasings of that criterion before final ranking. Snippet
  calls are free; only latency bounds this.
- Consider the rank-term quirk more seriously for small-K queries: if all K
  judged grades are equal, rank = 0 and the whole query scores 0 — even if all
  K are Perfect. I chose not to hedge (K unknown, event rare); a future
  instance seeing a 0 on an apparently-perfect submission should check this
  first.

## Insights about the evolution strategy

- **The scoring function is fully documented and exact — exploit it.** Most of
  my gains came from aligning the agent's internal objective with the scorer's
  literal computation (per-criterion 0/1/3 grading mirroring the judge's scale,
  ranking by predicted all-perfect) rather than from generic "better retrieval".
- **One crash can flip the aggregate ranking between agents.** Robustness
  (route-level try/except, inner try around optional stages, never submit
  empty when a best-effort set exists) is worth as much as a clever new stage.
- **The cost free zone is a real resource.** Parents used ~$0.02 of a $0.06
  average allowance. Spending idle budget on wider pools and a second grading
  pass is close to free score. But compute the batch-mix arithmetic (73%
  semantic) before scaling any per-semantic-query stage.
- Query-type routing is stable and correct; evolve the three routes
  independently. Semantic dominates the mean (73% of queries and the lowest
  scores), but specific/metadata queries are near-binary (0 or 1), so a single
  fixed crash there is worth ~0.07 mean F1 — cheap wins first.

## Suggested prompt/process improvements

1. State the "judge scores only the first K submissions" fact in CLAUDE.md
   explicitly (it currently must be inferred from 'first K judged papers' in
   the recall formula plus the verdict files). It is the key structural fact
   of the semantic path.
2. Provide a strategy_tools script that emits, per semantic problem: K, verdict
   histogram within scored depth, rank/recall terms, and gold-criteria text —
   the table I built by hand.
3. Include per-call LLM cost traces (model, in/out tokens) in each problem's
   diagnostics so cost tuning doesn't require reverse-engineering.
4. Keep iteration batches overlapping partially across rounds: iteration_003
   had no metadata queries at all, so metadata changes ship untested this
   round; a guaranteed minimum of each score_type per batch would de-risk that.
