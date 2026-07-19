# Evolution Reflection — Iteration 11 (PaperFindingBench)

## What worked well

1. **Reading the previous iteration's reflection FIRST, before any diagnostics.**
   The iteration-10 reflection was again the highest-value artifact: it handed me
   the confirmed scorer mechanics, the "recall is binding" diagnosis, the
   incumbent-plus-grafts strategy, and the warning about the noise floor. I spent
   my time on *new* evidence instead of re-deriving old conclusions. This
   chain-of-reflections is working; preserve it.

2. **Grade-histogram analysis of `judge_verdicts.md`, not just scores.** The
   single most valuable table this round was Perfect/Highly/Somewhat/Not counts
   per problem. It split "recall is low" into two different mechanisms:
   - papers **missing from the pool** (semantic_22: 42 Not-Relevant judged), vs.
   - papers **retrieved but stuck at grade 2** (semantic_110: 126 Highly vs 67
     Perfectly — grade 2 earns ZERO recall). 344 judged-Highly papers across the
     four biggest-K queries is an enormous pot that pure retrieval work cannot
     touch; only evidence quality can. One `grep -c` loop over judge_verdicts.md
     found it in two minutes.

3. **Reading agent stdout for silent mechanical failures.** stdout showed every
   `get_paper_batch(fields="references")` call failing server-side ('NoneType'
   not iterable) — meaning the incumbent's headline feature (citation expansion)
   had been running at half strength for a full iteration and nobody noticed,
   because the stage is fail-open and the score deltas are inside noise. Lesson:
   **fail-open stages need loud telemetry**, or their regressions are invisible.
   Grep stdout for `failed` / `skipped` on every incumbent before building on it.

4. **Hunting for free levers before paid ones.** Tool calls cost nothing; only
   LLM tokens are metered. This round's headline change (tail evidence sweep)
   costs zero LLM tokens — it upgrades evidence for judged-but-unenriched
   submission positions with snippet_search calls. When the cost budget is
   pinned at the free-zone edge (mean $0.0577 vs $0.06 threshold), free levers
   are the only ones with unbounded headroom. Ask "what does the judge see for
   position 150?" not just "what did we retrieve?".

5. **Cost modeling per stage before committing.** I arithmetic-checked every
   addition against per-model prices and funded them with explicit trims
   (T1 body chars, chunk size, HEAD, SIM_CUT). The batch's semantic share
   varies (71% this batch, 73% in the test set), so I modeled cost at the
   *worst* plausible mix, not the observed one. iter9 lost points to a $0.0005
   overage; design to ~$0.055, not $0.0599.

6. **End-to-end smoke test with stubbed models AND stubbed tools.** Unlike
   prior rounds' unit-level smoke tests, I ran the full `_solve_semantic`
   against fakes, twice (narrow pool and wide pool), which proved every stage
   fires: prescreen, gap-fill, per-seed reference expansion, thin-pool verify
   depth, tail sweep, telemetry. It caught nothing catastrophic this time, but
   the wide-pool variant confirmed a branch (lexical prescreen) my first run
   never entered — silent non-execution is exactly how features die (see #3).
   Cost: ~15 minutes. Worth it every round.

7. **A concurrency trap avoided: module globals in Inspect solvers.** My first
   telemetry implementation used a module-global dict reset per sample. Inspect
   runs samples concurrently in one event loop, so that would cross-contaminate.
   ContextVar fixed it. Future instances: any per-query state in agent.py must
   be task-local or threaded through arguments — never module-level.

## What was challenging

1. **Attributing Highly-vs-Perfectly to evidence vs reality.** A judged-Highly
   paper either genuinely lacks a criterion (SEAP is task-*specific*; no
   evidence fixes that) or satisfies it in body text we never quoted. The
   diagnostics cannot separate these; only the flip rate after an evidence
   improvement will. I estimated a 15% flip rate by reading a handful of
   verdict/evidence pairs by hand — tedious, and the next iteration should
   check the realized rate on the same query shapes.

2. **Cost decomposition still requires guesswork.** The cost report gives
   per-model totals; mapping them to pipeline stages needed token arithmetic
   with ~30% error bars. I added per-stage call/char counters to agent stdout
   (`llm-usage:` line) so iteration 12 can trim surgically. Keep this line in
   any derived agent; it costs nothing.

3. **The noise floor still dominates single-iteration comparisons.** Same
   lineage swung 0.21–0.48 across batches. Only mechanism-level evidence
   (stdout traces, verdict histograms, solo-win breakdowns) justifies design
   decisions; the aggregate score alone justifies almost nothing.

## Tooling notes

- jq + grep over `score_calculation.md`, `judge_verdicts.md`, `submission.json`,
  `agent_stdout` covered everything. The rank/recall/K extraction is one grep;
  the grade histogram is one grep -o | sort | uniq -c. No strategy_tools
  existed this round; the two greps above belong in one.
- Paths: artifacts live at the *experiment root* (`../../iteration_NNN/`,
  `../../agents/`), not under `evolution_output/`. The prompt's tree is
  correct; my first Read guessed wrong anyway. State the absolute root once.

## What I would do differently

1. **Live-probe the one uncertain tool pattern.** My reference fix (per-seed
   `get_paper(fields="corpusId,references")`) is smoke-tested for logic but not
   validated against the live server — the same gap iteration 10's reflection
   admitted for its expansion. If the server also chokes on per-seed reference
   fetches, the fix is a no-op (fail-open, so harmless, but worthless). Ten
   minutes of live probing would have converted "should work" to "works".

2. **Write down the expected-gain arithmetic per change in reasoning.md.** I
   did it informally (score formula is known; a 15% Highly→Perfect flip on
   semantic_110 ≈ +0.10 on that query). Forcing the arithmetic ranks the
   backlog better than intuition and gives iteration 12 falsifiable
   predictions to check against realized results.

## Insights about the evolution strategy

- **The winning pattern is unchanged through 11 iterations: incumbent + one new
  mechanism + mechanical bug fixes, everything gated/fail-open.** Every rewrite
  underperformed its parent; every graft held or gained. With 14-query batches,
  bold architecture bets are unattributable.
- **The frontier has moved twice: retrieval breadth → citation expansion →
  evidence saturation.** Rank is 0.6–0.8 nearly everywhere (mature); pool
  recall improves slowly and costs LLM triage; the grade-2 mass is the current
  cheapest target because it is free to attack. If the tail sweep works, the
  next frontier is likely (a) triage calibration — stage-1 predicted 25 perfect
  where the judge found 67, so ordering inside the pool is noisy-conservative;
  (b) probe quality — the per-criterion probe phrases drive both enrichment and
  sweep, and nobody has evaluated whether they retrieve the right passages.
- **Watch for silently-dead machinery.** Two consecutive iterations shipped a
  feature that partially didn't run (iter9's batch drop, iter10's references).
  The combination of fail-open design (correct for scoring) and score noise
  (unavoidable) means stdout is the ONLY place regressions are visible.

## Suggestions for the process/prompt

1. **Ship the diagnostics digest.** Three reflections in a row have asked for a
   per-(agent, problem) table: score_type, K, rank, recall, n_submitted, grade
   histogram, cost, wall-clock, stdout-error count. It is ~40 lines of Python
   over files that already exist and would halve the analysis phase.
2. **Maintain cumulative `facts.md` at the experiment root** for
   agent-discovered ground truth that CLAUDE.md doesn't capture: judge depth =
   min(K, n_submitted); known-good ids auto-grade Perfect; grade 2 earns zero
   recall; `get_paper_batch(fields="references")` reliably 500s while other
   field sets work; interrogative keyword queries return zero hits;
   samples run concurrently so agent module globals are unsafe. Each instance
   currently re-verifies or re-discovers a subset of these.
3. **Require a "verify the incumbent's features actually ran" step.** Before
   grafting onto a base, grep its stdout for failure lines on every named
   feature. Two iterations of half-dead citation expansion would have been one.
4. **State the cost design target in the prompt** ("engineer to ≤ $0.055 mean
   at a 73% semantic mix"), not just the penalty table. Every recent iteration
   has independently re-derived the same margin arithmetic.
5. **Keep the fixed-anchor-set idea alive.** A 6-query replay set rerun each
   iteration would separate "agent improved" from "batch got easier" — the
   0.21–0.48 lineage swings make Elo the only trustworthy cross-iteration
   signal right now, and it converges slowly with 3 agents/iteration.
