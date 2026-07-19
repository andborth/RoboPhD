# Evolution Reflection — Iteration 5 (asta_paper_finder)

## What worked well

**Attribute every point lost before designing anything.** The single most
valuable hour was spent cross-referencing four diagnostics per problem:
`score_calculation.md` (which term is binding), `judge_verdicts.md` (grade
distribution and judged depth), `agent_stdout` (what the agent actually did),
and `gold_criteria.md` (what the judge actually wanted). Every failure in
iteration 4 decomposed into exactly one of five mechanisms:

1. Grade-2 saturation (semantic): dozens of "Highly Relevant" verdicts worth
   zero recall — one qualifier criterion undemonstrated by the evidence.
2. Pool starvation (semantic, large K): not enough genuinely relevant papers
   retrieved to fill K judged slots.
3. Structural citation-cap misses (metadata): `get_citations` returns a
   recency-skewed 1000-subset of a 10k+-citation target; gold sets are the
   highly-cited citers that subset can't contain.
4. Unimplemented constraints (metadata): "citing papers by author X, not
   self-citations" needs reference-set intersection, which no agent had.
5. Identification brittleness (specific): unicode superscripts breaking
   keyword search; duplicate corpus records of one work in gold; a confident
   single submission when the alias was ambiguous.

Each mechanism suggests its own fix; none is discoverable from the aggregate
score table. The one-shell-loop tabulation of Perfect/Highly/Somewhat counts
per problem (from judge_verdicts.md) is the fastest way to see mechanism 1
and 2 — I recommend future instances run it first thing.

**Treat the scoring formulas as the target, not background.** Concrete
consequences read directly off the formulas + diagnostics: the judge stops at
K, so the tail past ~160 is free insurance and ordering of the head is half
the semantic score; grade 3 requires *every* weighted criterion Perfect, so
evidence must cover the weakest criterion, not summarize the paper; on
exact-match paths precision dilutes gently (16 gold / 211 submitted = 0.14)
but an empty submission is a guaranteed 0, so relax-ladders beat strictness.

**Inherit the winner verbatim; change mechanisms, not style.** iter4 had zero
crashes across 14 problems and zero solo losses. I copied it wholesale and
made targeted edits. Rewrites discard accumulated robustness (the corpusId
int/str casts, the authors-as-strings guard, the {"data":[]} unwrapping) that
was paid for with real zeros in earlier iterations.

**Stub-and-test before shipping.** The eval env can't be exercised locally,
but stubbing `model_registry` and `inspect_ai` via `sys.modules` let me run
41 helper checks plus a 5-scenario end-to-end smoke test of all three routes
(including the never-submit-empty fallback). The smoke test caught a real
hygiene bug (grade parser accepting hallucinated indices outside the chunk)
and — instructively — a case where my *test fixture* was wrong but the agent's
relax ladder behaved correctly. Cheap insurance; strongly recommended.

## What was challenging / time-consuming

- **No live tool access during evolution.** Everything about tool behavior
  (does `references` include corpusId? is get_citations recency-ordered?) is
  inferred from CLAUDE.md notes and prior stdout. I hedged in code (match
  references by paperId hash OR corpusId; relax if verification empties the
  set), but one live probe query would have removed real design uncertainty.
- **Cost arithmetic is manual.** Estimating whether deepening the judge-sim
  head from 80 to 150 keeps the batch mean under $0.06 required backing
  tokens out of cost_report.md's per-model totals. A per-stage cost trace in
  stdout (tokens in/out per LLM call) would make this one grep.
- **Small per-iteration samples (14 problems) make deltas noisy.** Two of the
  three specific queries and three of the four metadata queries in iteration
  4 were near-zero for everyone; one fixed mechanism can swing the mean by
  several points, but so can query-mix luck. I prioritized fixes by expected
  transfer (mechanisms keyed to query *structure*), not by observed deltas.

## Tool/process notes

- The diagnostics suite is excellent and complete; I never needed anything it
  didn't provide. `judge_verdicts.md`'s "beyond scored depth — not judged"
  markers are the ground truth for K — use them.
- No `strategy_tools/` existed this round. The single most useful script to
  provide would be: per semantic problem, print K, rank, recall, and the
  Perfect/Highly/Somewhat/Not counts within K, across all agents — that's
  ~80% of the analysis phase automated.
- Shell quirk: quote arguments starting with `=` in zsh; and `ls` the
  experiment root first — the "paths relative to evolution workspace" in the
  prompt resolve two levels up, which cost me one wrong Read at the start.

## What I'd do differently

- **Probe the judge's leniency boundary empirically.** My stage-2 "judge
  simulation" prompt asserts strictness rules I inferred from verdicts. With
  training diagnostics for several iterations available, one could calibrate:
  collect (evidence, actual judge verdict) pairs from past submissions and
  check the simulator's agreement rate offline. I didn't have time; a future
  instance could build this in an hour and tune the ranking threshold with
  data instead of prose.
- **Mine all past iterations, not just the last one.** I used iteration 3's
  data only opportunistically (semantic_77's all-Somewhat zero). The union of
  all past problems (~40 distinct) is a better regression suite than the
  latest 14; a quick script replaying "would my new ranking policy have
  reordered iteration-3's submitted lists better?" is feasible offline.

## Insights about the evolution strategy

- Route-by-score_type is the correct top-level decomposition; the three query
  types share almost nothing. Improvements are separable, which makes
  iteration safe: a metadata fix cannot regress semantic.
- The highest-leverage semantic insight across all iterations: **the judge
  reads only your evidence, and only all-criteria-Perfect earns recall.**
  Everything else (pool size, enrichment, reranking) serves that sentence.
- The benchmark rewards mechanisms over tuning. Every big score jump in this
  lineage came from a structural capability (criteria prediction, evidence
  enrichment, judge simulation, reference verification), never from prompt
  polish. Budget analysis time accordingly.

## Prompt/process suggestions

1. Provide the verdict-distribution tabulation script (see above) as a
   standard strategy tool; it compresses the first hour into one command.
2. Add per-LLM-call token/cost logging guidance to CLAUDE.md (agents should
   print it; evaluators already track it) so cost headroom is visible per
   stage without manual reconstruction.
3. State explicitly in the prompt that judged depth equals K and is visible
   in judge_verdicts.md — it's the single most decision-relevant fact and
   currently must be rediscovered from diagnostics each lineage.
4. Consider carrying a short "verified environment facts" file forward across
   iterations (tool quirks confirmed from stdout: citation-cap ordering,
   reference field contents, venue filter behavior). Each iteration currently
   re-infers these from scratch; a curated fact list would compound instead.
