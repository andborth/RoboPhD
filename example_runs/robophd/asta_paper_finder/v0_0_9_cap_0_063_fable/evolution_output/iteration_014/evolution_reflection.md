# Evolution Reflection — Iteration 14 (iter14-title-channel)

## What worked well

1. **Reading the previous session's reflection first, and actually following
   its triage order.** Iteration 13's reflection prescribed: (1) cross-agent
   zeros, (2) solo losses with mechanism fingerprints, (3) the structural
   tail — only then read code. Following that order verbatim meant I never
   read the 2000-line agent end-to-end; I read ~600 lines of it, all
   implicated by a specific diagnostic. The whole analysis phase took ~20 of
   my 60 minutes and every change traces to a measured failure.

2. **Discovering the judged-window fact from the verdict files, not the
   docs.** Tabulating `judge_verdicts.json` label counts against
   `score_meta.json.k_estimate` showed `scored_depth_cap == k_estimate` on
   every problem: the judge reads exactly the first K papers and nothing
   after. This reframes the whole semantic game — ordering beyond position K
   is irrelevant, recall is decided entirely by grade-3 density inside a
   window whose size you don't know (12–138 observed). One 10-line Python
   loop over the batch established this more crisply than any amount of
   formula-reading. Verify structural facts empirically across the whole
   batch; single-problem reads mislead.

3. **Cross-agent paper-level diffs, not score-level diffs.** The single most
   decisive observation this session: iter10's only grade-3 on semantic_77
   was *absent from iter13's entire 250-paper submission*. Score tables say
   "iter10 was luckier"; the paper-level diff says "the pools differ because
   the stochastic keyword plan differs, so retrieval breadth is the lever."
   That converted a vague variance complaint into two concrete free changes
   (more keyword/snippet queries; a title-resolution channel).

4. **Auditing stdout for stage clipping.** Grepping `failed|skipped|deadline`
   across the losing problems' `agent_stdout` found r2 expansion skipped *by
   the clock* on the two problems where its quality gate had actually passed
   (yields 6 and 7 vs threshold 4) — and found the skip message lying about
   the reason. Wall-clock data (worst 1518s vs 1740s limit) justified a
   budget raise with numbers, not hope.

5. **The inherited smoke test, eleventh consecutive session.** Copying
   `smoke_test.py`, adding one stub for the new reformulate reply shape and
   one forced-thin-pool semantic case took ~10 minutes and verified the new
   title channel end-to-end (guesses resolved → fresh candidates graded)
   plus all six existing paths. The stub-matching trap the last session
   warned about (matching on strings that appear in prompt templates) was
   avoided by matching on a query-specific phrase ("starved needle topic").

## What was challenging

- **Distinguishing "genuinely not grade-3" from "evidence failed to show
  it".** The losing windows are full of grade-2 papers. On narrow queries
  (semantic_77's "4-bit integer *columnar* weight-only quantization ...
  BLOOM") inspection of the actual submitted evidence showed the papers are
  genuinely adjacent (vector quantization, block-wise FP4), not
  under-evidenced. The evidence machinery is near its ceiling; only pool
  recall can move these. Future sessions should not re-attack the evidence
  builder without first checking a few grade-2s' evidence by hand.
- **The public S2 API rate limit.** Both attempts to resolve specific_20's
  missed gold id got 429s. I dropped it — right call under a 60-minute
  budget, but it left the specific-path ambiguity question ("which landmark
  papers does gold contain for 'the cnn paper'?") unanswered for another
  session. If you need the public API, batch every id into ONE call and make
  it early in the session; retries later collide with other users.
- **Diff-auditing a copied 2000-line file.** `diff base new | grep` with a
  whitelist of expected tokens confirmed the four changes introduced nothing
  else. Cheap and worth doing every time you base on a prior agent.

## On the provided tools

- The diagnostics quartet (`score_calculation.md`, `judge_verdicts.json`,
  `score_meta.json`, `agent_stdout`) answered every "which stage failed"
  question. Note the verdict-file schema changed at some point: top level is
  `{"scored_depth_cap": K, "papers": [...]}` — a bare-list assumption
  crashes. jq + python one-liners covered everything; I never needed
  tool_probe.py this session (the failures were pool-variance and clock
  issues, not tool-semantics issues).
- The error analysis report's solo-wins/solo-losses lists were the fastest
  entry point — they name exactly which problems to open first.

## What I'd do differently

- I spent ~5 minutes on specific_20 and metadata_15 (0.5 and 0.8 scores)
  before concluding both were near their ceiling. The score table already
  showed all three agents identical on them — cross-agent-identical scores
  on exact-match paths usually mean "gold convention, not code" and can be
  deprioritized in seconds.
- I would have liked one more pass to quantify how often the thin-pool gate
  fires across recent batches (it fired 2/11 semantic problems here) to size
  the title-channel's expected impact; ran out of clock.

## Insights about the evolution strategy

- **The lineage's residual semantic losses are now a retrieval-variance
  problem, not a pipeline problem.** Rank is 0.7–0.9 nearly everywhere;
  ordering and evidence changes cannot add much. What separates agents on a
  given problem is which papers their stochastic query plans happened to
  pull. Free levers (more queries, more channels, citation graph, title
  lookups) should be exhausted before any metered lever is touched.
- **Know the cost anatomy before proposing anything.** 86% of spend is the
  mini coarse-grader; its input length (340 papers × 200-char abstracts) is
  the real budget dial. Extending truncation by 200 chars costs ~$0.013/query
  — 5× the entire free-zone headroom. Conversely anything that only adds
  tool calls is genuinely free. Check `cost_report.md`'s by-model breakdown
  first; it prevents whole classes of bad ideas.
- **Small, attributable diffs keep winning.** The last four winning agents
  were each ≤120-line diffs over the previous winner, every line tied to a
  problem id. The one session that rewrote the ordering path regressed. With
  a 60-minute budget, 3–4 verified changes is the right scope; a rewrite is
  never finishable or verifiable in time.

## Suggestions for the prompt / process

1. **Document the judged-window fact in CLAUDE.md.** "On semantic_f1 the
   judge reads exactly the first `k_estimate` papers (= `scored_depth_cap`
   in judge_verdicts.json); papers past K are never judged and cost
   nothing." Three sessions have now partially rediscovered this; it changes
   what ordering work is worth doing.
2. **Add the judge_verdicts.json schema to CLAUDE.md** (top-level dict with
   `scored_depth_cap` + `papers`; labels are `perfectly_relevant_papers`
   etc.). Requested by iteration 13's reflection too.
3. **In the error report, add a per-problem "pool diff" line**: for each
   semantic problem, which judge-verified grade-3 corpus_ids appear in some
   agent's submission but not others'. This session derived it by hand and
   it was the highest-value signal; automated, it would take one minute off
   every future session's critical path.
4. **Carry a standing "cost anatomy" line in the report**: % of spend by
   call site (plan / coarse grade / rerank), not just by model. The by-model
   table almost shows it (mini ≈ grader+rerank) but call-site attribution
   would make the budget dial explicit.
5. **The stage-timing table (parse `[t+NNNs]` stamps) remains unbuilt** —
   requested by four sessions now. This session again read timestamps by
   grep. Even a crude "last stamp before submit" column per problem would
   have surfaced the deadline-clipping pattern instantly.
