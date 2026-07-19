# Evolution Reflection — Iteration 10 (PaperFindingBench)

## What worked well

1. **Reading the previous iteration's reflection first.** The iteration-9
   `evolution_reflection.md` was the single highest-value artifact this round.
   It handed me the confirmed scorer mechanics (judge grades exactly the first
   K entries; only grade-3 earns recall; known-good ids auto-grade Perfect), a
   named untried lever ("citation-graph expansion — untried and free"), and the
   warning that sub-0.1 per-problem deltas are noise. I spent almost no time
   rediscovering ground truth and nearly all of it on diagnostics-to-mechanism
   attribution. Whatever the process can do to make each instance's reflection
   land in front of the next instance, keep doing it.

2. **Diagnosing rank-vs-recall per problem before proposing anything.** Pulling
   K, rank, recall, and submission size for every semantic problem into one
   table (a 5-minute jq/grep pass over `score_calculation.md`) immediately
   showed recall is the binding term on ~8 of 11 semantic queries (K=118–204,
   recall 0.14–0.25, rank already 0.6–0.9). That single table redirected the
   whole iteration from "tune the head ordering" (tempting, visible) to
   "widen the pool" (actually binding). Cross-checking with the judge-verdict
   grade distributions (99 Highly vs 52 Perfectly on semantic_170) confirmed
   both sub-causes: papers missing from the pool AND retrieved papers stuck at
   grade 2.

3. **Root-causing the catastrophic scores instead of averaging over them.**
   metadata_25 (0.010) and semantic_145 (0.049) each had a precise, fixable
   cause visible in stdout + gold diffs: a get_citations 1000-cap plus a
   silently-failing get_paper_batch dropping candidates; and a gold criterion
   ("Earliest instance w=0.2") the planner never reconstructed, compounded by a
   year-first tiebreak that inverted the head. Two near-zero problems fixed
   structurally are worth more than ten micro-tunings, and the fixes generalize
   because the query *shapes* recur across batches.

4. **Basing on the incumbent and making every addition gated or fail-open.**
   With 14-query batches and heavy judge noise, un-gated changes are
   unattributable. Every change this round is either additive (expansion stage,
   extra mention searches), fail-open (reference verification keeps candidates
   that explicitly name the cited work), deadline-gated, or a pure bug fix
   (batch bisect). The downside on already-working queries is bounded.

5. **Cheap artifact verification.** py_compile + a stubbed `model_registry`
   import + a small async smoke test of the new code paths (~3 minutes total)
   caught a real design flaw: my first bisect stopped at group size 3 and still
   lost a poison id's partners; the smoke test's "recovered 6/7" output made
   that visible and the fix trivial. Testing *behavior* of new helpers, not
   just syntax, is worth the few minutes every time.

6. **Reading only targeted slices of the 1700-line base agent.** grep for
   function definitions first, then read the 3–4 hunks the plan touches. I
   never read the specific-query path this round because no diagnostic implicated
   it. This kept context spend proportional to evidence.

## What was challenging

1. **Cost accounting is guesswork.** The cost report gives per-model totals per
   agent, but attributing spend to pipeline *stages* (stage-1 triage vs stage-2
   sim vs verify) required inference from token-size arithmetic. iter9 missed
   the free zone by $0.0005 — a rounding error — and paid a penalty. Agents
   should print per-stage LLM call counts and rough token totals to stdout so
   the next instance can trim surgically instead of estimating.

2. **The noise floor.** Mean F1 swung 0.21–0.48 across iterations for the same
   agent lineage purely from query sampling. The score table alone cannot rank
   agents; only mechanism-level evidence (stdout traces, verdict diffs) can.
   I'd have wasted the round chasing iter4's nominal "win" (0.356 vs 0.348)
   had the solo-win/solo-loss breakdown not shown iter9's mechanisms were
   stronger.

3. **Diagnostics are scattered.** Building the K/rank/recall table means
   grepping 14 × 3 problem directories. It's mechanical and error-prone in
   zsh (globbing ate my `====` separators once; quote them).

## Tooling notes

- jq + grep over `score_calculation.md`, `judge_verdicts.md`, `submission.json`
  and `agent_stdout` covered everything I needed. No strategy_tools existed
  this round; the manual pass was fine but is exactly what a helper script
  should emit (see below).
- The relative paths in the prompt ("../../agents/") resolve from the iteration
  working dir, but my first Read attempt used a wrong absolute join. Trivial,
  but stating the absolute experiment root once in the prompt would remove the
  stumble.

## What I would do differently

1. **Test the expansion stage against the live tools before committing.** My
   citation-graph expansion is smoke-tested for logic but not validated against
   the real MCP server (e.g., whether get_citations on 8 seeds concurrently
   trips the shared rate limit into slow backoff). A 10-minute live probe of
   the one new tool-usage pattern would have de-risked the headline change.
   I chose not to spend the time; a future instance with tool access should.

2. **Quantify the expected gain per change.** I estimated "+2 raw points if
   expansion lifts recall 20% on large-K queries" only informally. Writing the
   arithmetic down per change (score formula is known!) would force-rank the
   backlog better than intuition.

## Suggestions for the process/prompt

1. **Ship a diagnostics digest script (or table) per iteration.** One file per
   iteration with a row per (agent, problem): score_type, K, rank, recall,
   n_submitted, grade histogram (P/H/S/N counts), cost, wall-clock, and a
   one-line stdout tail. This is ~40 lines of Python, would cut the analysis
   phase by half, and would make rank-vs-recall diagnosis instant. The
   iteration-9 reflection asked for the same thing; it still doesn't exist —
   whoever maintains the harness should actually add it.

2. **Maintain a cumulative `facts.md` + `regressions.md` at the experiment
   root.** Confirmed mechanics (judge depth = K; known-good auto-Perfect;
   get_citations 1000-cap with no paging; get_paper_batch 'NoneType' failures;
   interrogative queries return zero hits) and a list of query shapes currently
   scoring ~0, each with root cause and which iteration attempted a fix. Each
   instance re-assembles this from diagnostics; CLAUDE.md captures the static
   parts but agent-discovered, evolving facts have no designated home.

3. **Ask for per-mechanism attribution and a parent diff, not prose.** The most
   useful part of any reasoning.md is "agent A beat B on problem P because
   mechanism M, see stdout line L". Requiring that format — plus keeping the
   agent docstring changelog convention and noting each agent's parent — makes
   the next instance's grafting decisions nearly mechanical. The lineage here
   (iter6 → iter9 → iter10) worked because docstrings were honest changelogs.

4. **Make the cost target explicit in the prompt.** "Design to a $0.050–0.055
   mean" is a better instruction than the current penalty table alone; iter9
   lost points to a $0.0005 overage that a stated engineering margin would have
   prevented. Conversely, agents running at $0.041 are leaving free-zone
   headroom unspent — both directions are waste.

5. **Consider evaluating candidate agents on a fixed replay batch.** The
   14-query resample between iterations makes cross-iteration comparison
   nearly meaningless (0.21–0.48 swings). Even a small fixed anchor set (say 6
   queries rerun every iteration alongside the fresh sample) would separate
   "the agent improved" from "the sample got easier" at modest cost.

## Insights about the evolution strategy itself

- The winning pattern across all ten iterations is **incumbent + grafted
  mechanisms + gated additions**, not rewrites. Every rewrite-flavored agent
  (seed, iter3, iter7) underperformed its parent; every graft-based agent
  (iter5, iter6, iter9) held or gained. The 14-sample noise makes bold
  architectural bets unattributable, so evolution should move in small,
  bounded-downside steps with one genuinely new retrieval modality per round.
- Retrieval breadth beats scoring finesse at the current frontier. The judge-
  simulation/evidence machinery is mature (rank 0.6–0.9 nearly everywhere);
  the pool is what's starved (recall 0.14–0.25). Future instances should keep
  attacking retrieval diversity — citation graphs (this round), author
  expansion from strong candidates, venue-scoped sweeps — before touching the
  ranking head again.
