# Evolution Reflection — Iteration 12 (PaperFindingBench)

## What worked well

1. **Read the previous iteration's reflection FIRST.** Twelve iterations in,
   the chain-of-reflections is the single highest-value artifact. Iteration
   11's reflection handed me the scorer mechanics, the "recall is binding"
   diagnosis, the grade-histogram technique, AND an explicit self-doubt
   ("my reference fix is smoke-tested but not live-validated — if the server
   also chokes on per-seed fetches, the fix is a no-op"). That flagged
   exactly where to look, and the doubt turned out to be correct. Ten
   minutes of reading saved an hour of re-derivation.

2. **Verify the incumbent's features actually ran before building on them.**
   One grep over stdout (`grep 'expand-ref' problems/*/agent_stdout`) showed
   every reference fetch failing identically — iter11's headline "references
   restored" fix never worked, and citation expansion has been citers-only
   for THREE iterations. This is now a confirmed recurring failure mode:
   fail-open design + score noise makes dead machinery invisible except in
   stdout. Make this grep a mandatory first step, not a discovery.

3. **Grade-histogram + criteria reading, per problem.** The verdict
   histograms (Perfect/Highly/Somewhat/Not counts) split "low recall" into
   three distinct mechanisms with different fixes:
   - Somewhat-mass (137/193/219/226) → the pool lacks conjunction-satisfying
     papers → retrieval-modality problem;
   - all-Highly (104/7/222) → qualifier criterion never Perfect → evidence
     problem, or a reality limit (on 104 the agent's own GPT_5_4 verify
     agreed with the judge: 4/16 — most papers genuinely can't satisfy
     "discusses COMMON architectures");
   - Not-mass (semantic_8: 95 Not judged) → pool pollution / ordering.
   Reading the actual gold criteria for the failures then revealed the
   shared shape: a low-weight "explicit connection" criterion whose evidence
   lives in body text. That single observation produced this round's
   headline change.

4. **Mechanism-level before/after comparison beats score comparison.**
   iter11 beat iter10 by only +1.9 (inside noise), but the grade histograms
   on the queries the tail sweep targeted moved in the predicted direction
   on all four (Perfect counts AND rank up). That is much stronger evidence
   the mechanism works than the aggregate score, and it justified keeping
   the sweep and extending the same philosophy (free evidence/retrieval).

5. **The smoke-test harness with stubbed tools AND stubbed models,
   exercising BOTH branches of new logic.** I tested the reference probe in
   both worlds (server-dead and subfield-works) with call-count assertions
   (probe must fire exactly once per variant, not once per seed). Reusing
   iter11's harness took ~10 minutes; writing assertions on tool-call
   counters caught nothing this time but is exactly how iter9/iter10-style
   silent no-ops get caught pre-submission.

6. **Cost engineering as explicit arithmetic.** The llm-usage telemetry
   iter11 added (per-stage call/token counters in stdout) made trimming
   surgical for the first time: I could see t1=27k, sim=21k, rescue=6.4k
   input tokens per heavy query and price each trim. Keep that telemetry
   line forever. Design target: ≤$0.055 mean at a 73% semantic mix, not
   ≤$0.0599 at the observed mix.

## What was challenging

1. **Separating "evidence problem" from "reality problem" on all-Highly
   queries.** When 47 papers sit at Highly, you cannot tell from diagnostics
   alone how many would flip to Perfect with better evidence. The best
   available proxy: compare the agent's own high-fidelity verify verdicts
   with the judge's — when both say "not perfect," it's reality, stop
   spending there. semantic_104-shaped queries may be near their ceiling
   (K=56 but perhaps only a handful of true grade-3s exist and K is an
   overestimate; nobody has scored >0.13 on it).

2. **The noise floor.** 14-query batches, same-lineage scores swinging
   0.21–0.48 across iterations. Only mechanism-level evidence justifies
   design decisions. Resist the urge to explain away a -0.09 delta on one
   query (iter11 vs iter10 on semantic_104 was ordering luck on ONE paper).

3. **The references API.** Three iterations have now tried to fetch
   reference lists (batch, per-paper, and now a subfield variant probe).
   If iteration 13's stdout shows the subfield variant ALSO fails, write it
   in the reflection in bold and stop trying — treat references as
   permanently unavailable and design around citers-only expansion.

## Tooling notes

- jq/grep over `judge_verdicts.md`, `score_calculation.md`, `gold_criteria.md`,
  `submission.json`, `agent_stdout` covered everything. No strategy_tools
  existed. The two most valuable one-liners:
  - grade histogram: `grep -oE '(Perfectly|Highly|Somewhat|Not) Relevant' judge_verdicts.md | sort | uniq -c`
  - dead machinery: `grep -h 'failed\|skipped\|unavailable' problems/*/agent_stdout | sort | uniq -c`
- zsh gotcha: bare `====` separators in echo get glob-interpreted and kill
  the command; quote them.
- Paths: experiment artifacts live at `../../iteration_NNN/` and
  `../../agents/`; the PREVIOUS evolution workspace (reasoning, reflection,
  smoke tests) lives at `../iteration_NNN/`. Both are worth reading; the
  first Read attempt at a guessed path usually fails — start with `ls`.

## What I would do differently

1. **Check realized flip-rates for the previous round's headline change
   quantitatively, not just directionally.** I confirmed the tail sweep
   moved histograms but never computed "how many of the 126 Highly on
   semantic_110 flipped" (it was 21, minus other churn). The realized rate
   calibrates how much more to invest in the same lever.

2. **Spend more time on ordering.** Rank is 0.47–0.86 and only the first
   min(K, n) submitted positions are judged at all. On semantic_8, 95 of
   222 judged slots held Not-relevant papers — each one a wasted judged
   position. Nobody has yet attacked triage precision/calibration directly
   (stage-1 predicted 8 perfect where the judge found 47+ Highly on 104,
   and 31 predicted vs 0 judged on an earlier query). A well-calibrated
   orderer is worth several recall points on big-K queries for zero
   retrieval work.

## Insights about the evolution strategy

- **The winning pattern is now 12-for-12: incumbent + one new mechanism +
  mechanical fixes, everything gated and fail-open.** Every rewrite lost;
  every graft held or gained. With 14-query batches this is structural, not
  cultural: bold bets are unattributable, so they cannot be kept even when
  they work.
- **The frontier has moved: retrieval breadth → citation expansion →
  evidence saturation → (this round) retrieval MODALITY.** Keyword search
  sees titles/abstracts; conjunction queries are decided in body text;
  snippet_search is the only free tool that reads body text. If the
  body-conjunction change works, the next frontier is likely triage
  calibration / ordering (see above), then per-query budget allocation
  (spend more on thin-K queries where each position is worth ~4% of a
  query, less on broad ones).
- **Free levers first, always.** Tool calls are unmetered; LLM tokens are
  the only cost. Snippet share, citers, evidence assembly, lexical
  prescreens — all free. The cost budget should be spent almost entirely
  on grading/ordering decisions, not on retrieval.

## Suggestions for the process/prompt

1. **Ship the diagnostics digest script.** Four consecutive reflections
   have asked. ~40 lines of Python emitting per-(agent, problem):
   score_type, K, rank, recall, n_submitted, grade histogram, cost,
   wall-clock, stdout-error count. It would halve the analysis phase and
   make cross-iteration comparison mechanical.
2. **Maintain a cumulative `facts.md` at the experiment root** for
   agent-discovered ground truth: references field is broken server-side
   (batch AND per-paper; subfield variant status pending iteration 12's
   run); judge depth = min(K, n_submitted); grade 2 earns zero recall;
   interrogative keyword queries return zero hits; samples run concurrently
   so module globals are unsafe; `get_paper_batch` group calls die on one
   poison id (bisect). Each instance currently re-verifies a subset.
3. **Mandate the "did the incumbent's features run?" grep** in the Round-1
   instructions, with the exact command. Two of the last three iterations
   shipped partially-dead headline features that stdout exposed in seconds.
4. **State the cost design target in the prompt** ("engineer to ≤$0.055
   mean at a 73% semantic mix") instead of only the penalty table; every
   iteration re-derives the same arithmetic.
5. **A fixed 6-query anchor set replayed every iteration** would separate
   "agent improved" from "batch got easier" — the noise floor is still the
   biggest obstacle to attribution, and with 3 agents/iteration Elo
   converges too slowly.
