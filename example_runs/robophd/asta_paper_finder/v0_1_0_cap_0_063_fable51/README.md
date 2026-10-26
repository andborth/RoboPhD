# v0_1_0_cap_0_063_fable51 — PaperFindingBench (AstaBench) submission snapshot

Curated snapshot of RoboPhD run `asta_paper_finder_20261019_184607`
(`robophd-asta_paper_finder-021`, 2026-10-19 → 2026-10-20). Claude Fable 5.1 evolved the
agent under a $0.063/query training cost cap. The submitted agent is `iter16_feedcast`,
the run's max-Elo winner (Elo 1590). This README covers the official result, the agent,
the evolution run and the snapshot's contents.

**It Pareto-dominates Asta Paper Finder, so RoboPhD now holds every point on the
PaperFindingBench Pareto frontier.**

## Official result (2026-10-23)

| | Score | Cost/query |
| --- | --- | --- |
| **v0_1_0_cap_0_063_fable51** | **0.39934** ± 0.01769 | **$0.058193** ± 0.001962 |
| Asta Paper Finder | 0.397 | $0.0626 |
| v0_0_9_cap_0_063_fable (ours) | 0.38954 | $0.058059 |

- **Against Asta Paper Finder:** higher on score and 7% cheaper. The score margin (+0.0023)
  is far inside the stderr, so this is dominance at the board's point estimates. The cost
  gap ($0.0044 against a $0.0020 stderr) is the firmer axis.
- **Against our v0_0_9_cap_0_063_fable:** it displaces that entry at displayed precision
  only. Both show $0.058; at full precision this one is $0.00013/query more expensive.
- **Frontier:** this was the last PaperFindingBench Pareto point not held by RoboPhD.
  Every point on the frontier is now a RoboPhD entry: 0.220 @ $0.006, 0.376 @ $0.052,
  **0.399 @ $0.058**, 0.432 @ $0.251, 0.440 @ $0.279.

| Per type | Official | Internal | Δ (paired) |
| --- | --- | --- | --- |
| semantic_f1 (194) | 0.3309 | 0.3355 | −0.0047 ± 0.0060 |
| specific_f1 (38) | 0.8089 | 0.7536 | +0.0553 ± 0.0376 |
| metadata_f1 (34) | 0.3439 | 0.3676 | −0.0237 ± 0.0156 |
| **Overall** | **0.39934** | **0.39787** | **+0.0015** |

- **Transfer:** the internal test is stock GPT-4o, capped. Internal→official across our
  seven PFB submissions: +0.0025 / −0.0550 / −0.0077 / +0.0096 / +0.0020 / +0.0022 /
  +0.0015.
- **Specific queries:** internally this agent trailed v0_0_9_cap_0_063_fable by −0.093 on
  specific, which looked like a weakness. Officially the gap is −0.020, about a third of
  the category's SD. Two policies do cost it on specific: padding single-answer queries,
  and picking the wrong record when Semantic Scholar has duplicates. The internal draw
  inflated that cost. See `specific_drop_analysis` in
  `../robophd_runs/results/asta_paper_finder.json` (-021).
- **Isolation wrapper:** this was the first official run under the per-sample isolation
  wrapper (cad0f713), and it held. Per-sample median 751 s, max 1,430 s, none over the
  agent's 1,560 s soft limit.
- **Spend:** $209.81 (judge $194.27, agent $15.54). Agent cost was 100.6% of internal.
- **Final log:** 267/267 samples, 0 errors, 0 wrapper fallbacks, revision `c3e7588a` with
  dirty False.

## Architecture of `iter16_feedcast`

The agent routes each query by score type.

- **Semantic (73% of test):**
  - **Retrieval:** a GPT-5.4-mini planner writes keyword queries and criterion-targeted
    snippet queries, one per required aspect; the results are fused.
  - **Grading:** GPT-5.4-mini grades 520 candidates, then 160 more at a time while the tail
    still yields, up to 600.
  - **Expansion:** the pool widens with the top papers' citers and with pseudo-relevance
    feedback (their titles re-run as keyword searches).
  - **Rerank:** GPT-5.4 re-judges the top 32 per criterion.
  - **Evidence:** the top 100 carry verbatim body passages chosen to cover what the abstract
    misses.
- **Specific:** GPT-5.4 and Sonnet 4.6 planners run in parallel. A GPT-5.4 picker's
  confidence sets how many papers to submit, from 1 to 6.
- **Metadata:** GPT-5.4 parses structured filters. Retrieval uses both citation directions,
  and "cites X and Y" queries intersect the two citer sets.

**Models (3 handles, 2 providers):** `openai/gpt-5.4-mini`,
`openai/gpt-5.4-2026-03-05`, and `anthropic/claude-sonnet-4-6`. Sonnet serves only as the
second specific-query planner.

## The evolution run

The run was a cold start from the standard `baseline` seed:
- **Budget:** 600 fresh evals, 14 examples per iteration; 21 iterations, 604 fresh evals used.
- **Judge and settings:** luna no-prose training judge, capped; 2,500-character evidence cap;
  Deep Focus off.
- **Objective:** the full objective, unlike the ablation arms -016/-017.

The winning lineage is surecast (6) → headcast (7) → steadycast (9) → clearcast (10) →
citecast (11) → provecast (12) → crosscast (13) → bodycast (15) → **feedcast (16)**.
- **Winners:** 10 distinct iteration winners.
- **Failures:** no exceptions, zero-accuracy cases or clone detections.
- **Reachability guard:** fired at iteration 21.
- **Selection:** feedcast's Elo of 1590 is one point ahead of iter18_oldcast (1589). It is
  thinly validated, but it is the run's result under the method's selection rule.

**Naming.** `v0_1_0` marks the submission infrastructure as settled, matching DS-1000's
v0_1_0_cap_0_127_fable51. `cap_0_063` is the training cost gate, and `fable51` is the
evolution model; Fable 5.1 is never a solver.

## Snapshot layout (~16 MB)

- `agents/` holds all 20 pool agents. The submitted agent is `agents/iter16_feedcast/agent.py`
  (2,088 lines). Its docstring still says "iteration 013 (crosscast)" because iterations 15
  and 16 were string patches; the code is iteration 16's.
- `iteration_015/`, `iteration_016/` hold the winner's breeding data and debut round. Only
  validation-split gold is included: those 25 problems share no IDs or query text with the
  267 test queries.
- `evolution_output/CLAUDE.md` and `evolution_output/iteration_016/` are the Claude Code
  session that produced the winner.
- `checkpoint.json` and `final_report.md` are the full run record.
- `test_results.json` (+ `.per_problem.json`) is the internal held-out eval: stock GPT-4o,
  capped, 267 queries.

## Submission form

| Field | Value |
| --- | --- |
| Agent name | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-2026-03-05, gpt-5.4-mini, claude-sonnet-4-6 |
| URL | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |
| Tarball | `v0_1_0_cap_0_063_fable51.tar.gz` (219.1 MB; the form shows 209.0 MB, in binary units), uploaded 2026-10-25 |

Agent description as submitted on the form (2026-10-25):

> Evolved by RoboPhD with Claude Fable 5.1 as the evolution (but not a solver)
> model, under a \$0.063/query training cost cap on solver spend. The cap was set
> at exactly the price of the strongest entry we had not yet Pareto dominated, so evolution
> had to win on score at a price already known to be competitive. It came in at
> \$0.058/query.
>
> The agent routes each query by type. For semantic queries, a GPT-5.4-mini
> planner turns the request into keyword searches plus one body-text snippet
> probe per aspect the request requires. The benchmark's grading criteria are
> hidden at query time, so the agent infers its own. GPT-5.4-mini grades the
> fused pool in batches and keeps grading while the tail still yields relevant
> papers. It then widens the pool from the best papers found so far: their
> citers, and their titles re-run as searches. GPT-5.4 re-judges the top 32
> papers criterion by criterion. Each submitted paper carries verbatim passages
> from its body, chosen to cover what its abstract leaves out.
>
> For named-paper queries, GPT-5.4 and Claude Sonnet 4.6 plan independently, and
> a GPT-5.4 picker reconciles their guesses. The picker's confidence sets how
> many papers to submit. For metadata queries, GPT-5.4 parses the request into
> structured filters. The agent answers them from both citation directions, and
> resolves "cites X and Y" requests by intersecting citer sets.

## Reproduce

```bash
pip install litellm==1.88.1     # submission-scoring price map
# The commit must be pushed first: astabench stamps eval_spec.revision
caffeinate -i python scripts/asta_paper_finder_submit.py --only v0_1_0_cap_0_063_fable51
```

Re-running the same command resumes an interrupted eval-set.

## Eval incidents: two outages

The eval ran as three attempts against one eval-set, and all three are in the
tarball's `submit_stdout.log`:
- **Attempt 1:** about 6 h, until the OpenAI outage below.
- **Attempt 2:** a resume, stopped after 10 minutes because the account still had no
  credit. No sample completed.
- **Attempt 3:** a 3 h resume that finished the remaining 119 samples.

1. **OpenAI quota exhaustion, 2026-10-22 from 10:10:52Z.** Every OpenAI call returned 429.
   - **9 silent zeros** (semantic_182, 187, 188, 190, 191, 194, 195, 197, 198): both
     wrapper tiers timed out, and the wrapper submitted an empty result that scored 0.0
     with no sample error. Eval-set retry would have reused them.
   - **Invalidated before the resume** with `inspect_ai.log.invalidate_samples`, which
     records the author and reason. The script is
     [`official_eval_invalidate_outage_samples.py`](official_eval_invalidate_outage_samples.py).
   - **Kept:** the 148 samples finished before the onset. They are unchanged in the final log.
2. **Local DNS outages, 2026-10-23, about 4 minutes each.** Asta-tools calls failed; no LLM
   calls did. 13 metadata samples overlapped, for about −0.0012 net, mostly metadata_40.
   They were not re-run: the outage could only lower the score, and re-drawing chosen
   samples after seeing outcomes would be a second draw.

The superseded logs are kept outside the tarball, in `submissions/.../logs/backup_pre_invalidate/`
and `logs/backup_invalidated/`.

**Lesson, not fixed in this artifact.** The wrapper's empty-submission tier turns a
provider outage into zeros that look like successes. Raising there instead would let
Inspect's retry recover them.
