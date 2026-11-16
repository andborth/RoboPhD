# v0_1_1_cap_0_279_fable52 — PaperFindingBench (AstaBench) submission snapshot

Curated snapshot of RoboPhD run `asta_paper_finder_20261111_142949`
(`robophd-asta_paper_finder-023`, 2026-11-11 → 2026-11-13). Claude Fable 5.2 evolved the
agent under a $0.279/query training cost cap, with the full solver menu. The submitted
agent is `iter20_year_steered_expansion`, the run's max-Elo winner (Elo 1567). This README
covers the official and internal results, the agent, the evolution run and the snapshot's
contents.

The cap was set at the price of the board's top entry, our `v0_0_9_cap_0_355_fable`
(0.440 @ $0.279), so evolution had to win on score at that price. It does, on both axes:

> **Official 0.501 @ $0.232**, against the incumbent's **0.440 @ $0.279**. It also
> dominates the frontier entry at **0.432 @ $0.251**.

## Official result (2026-11-15)

`astabench eval` at commit `553b0a11` (clean), stock GPT-4o judge, uncapped, 267 queries.

| | Score | Cost/query |
| --- | --- | --- |
| **v0_1_1_cap_0_279_fable52** | **0.50140 ± 0.01755** | **$0.23231 ± 0.00588** |
| v0_0_9_cap_0_355_fable | 0.44025 ± 0.01790 | $0.278579 ± 0.009930 |
| v0_0_9_cap_0_355_opus5 | 0.43177 ± 0.01797 | $0.250635 ± 0.0089 |

| Per type | This agent | v0_0_9_cap_0_355_fable |
| --- | --- | --- |
| semantic_f1 (194) | **0.4394** | 0.3749 |
| specific_f1 (38) | 0.8465 | **0.8588** |
| metadata_f1 (35) | **0.4706** | 0.3483 |
| **Overall** | **0.50140** | 0.44025 |

- **Dominance:** it Pareto-dominates two frontier entries, both ours:
  - `v0_0_9_cap_0_355_fable` (0.440 @ $0.279), the top of the board: +0.061 on score and
    $0.046/query (17%) cheaper.
  - `v0_0_9_cap_0_355_opus5` (0.432 @ $0.251): +0.070 on score and $0.018/query (7%)
    cheaper.

  Neither is Pareto-optimal any longer; this entry replaces both on the frontier.
- **Internal→official:** +0.014 (0.48769 → 0.50140), above the −0.055 to +0.010 range of
  our earlier PFB submissions.
- **Clean run:** all 267 queries scored; no tool-init failures and no tool calls that
  exhausted their retries (10 HTTP 429s, all retried by the harness).
- **Spend:** $206.78 judge + $62.03 agent = **$268.80**
- **Time:** 9h08m

## Internal result

Internal held-out eval: stock GPT-4o judge, capped, 267 queries.

| | Score | Cost/query |
| --- | --- | --- |
| **v0_1_1_cap_0_279_fable52** (internal) | **0.48769** | **$0.23186** |
| v0_0_9_cap_0_355_fable (internal) | 0.4383 | $0.2780 |
| v0_0_9_cap_0_355_fable (official) | 0.44025 | $0.278579 |

| Per type | This agent | v0_0_9_cap_0_355_fable |
| --- | --- | --- |
| semantic_f1 (194) | **0.4274** | 0.3715 |
| specific_f1 (38) | 0.8202 | **0.8675** |
| metadata_f1 (35) | **0.4609** | 0.3423 |
| **Overall** | **0.48769** | 0.4383 |

- **Margin:** +0.049 internal on score, and $0.046/query cheaper.
- **Zeros:** 23 queries score 0; 13 of them are queries the predecessor also scored 0 on.
- **Spend:** agent $61.91, judge $74.15 (capped, shared cache).
- **Training-judge basis:** rejudged with the luna no-prose training judge (capped,
  canonical ordering), the agent scores 0.51027 on 267 queries
  (`test_results.rejudge_openai_gpt-5.6-luna_noprose.merged.json`). That is the comparison
  basis for this run's model-menu ablations, not a leaderboard number.

## Architecture of `iter20_year_steered_expansion`

The agent (1,811 lines) routes each query by score type. For every type, an Opus 4.8
planner first turns the request into a plan (searches, candidate titles, criteria or
filters), with Sonnet 4.6 as the fallback if the plan fails to parse.

- **Semantic (73% of test):**
  - **Retrieval:** the plan's keyword and snippet searches run in parallel; the results
    are fused.
  - **Grading:** GPT-5.4-mini rates every candidate against each relevance criterion the
    plan inferred, using only the paper's text: 3 if the text clearly satisfies the
    criterion, 1 if only partially or by implication, 0 if not at all. The plan also
    weights the criteria, and a paper's **score** is the weighted sum of its ratings
    divided by 3, capped at 1. A score of 1 means every criterion was rated 3.
  - **Depth:** candidates are ordered by their **retrieval rank**, a fusion of their ranks
    across all the searches. The first tranche is the top 320. If at least 25 of them
    score above 0.67, a second tranche of up to 200 is graded.
  - **Citation snowball:** the seeds are up to 10 graded papers with a score of 1, taken in
    retrieval-rank order (topped up to 5 with papers above 0.67 if needed). Their references
    and citers are graded, those linked to several seeds first: 120 candidates, then 120
    more if at least 12 of the first batch score 1. Both steps start only early enough in
    the time budget.
  - **Re-grade:** GPT-5.4 re-judges the 80 highest-scoring candidates above 0.6 (at least
    30).
  - **Evidence:** each submitted paper carries verbatim title, abstract and best-matching
    body snippets. Up to 250 papers are submitted.
- **Specific:** the plan proposes real titles; GPT-5.5 and Gemini 3.1 Pro give independent
  readings of what the request refers to; title, keyword and snippet searches gather
  candidates, and Opus 4.8 picks the paper(s).
- **Metadata:** the plan is a structured spec; the author, citation and venue tools build
  candidates; deterministic filters use citation counts back-dated to the benchmark's
  cutoff, and Sonnet 4.6 matches venue names and checks any topic the request states. For
  year-bounded requests (≤4 years), each expansion query is also issued once per requested
  year with the year appended, which steers the year-less search tool toward those years.
- **Time budget:** no new tool calls after 21 minutes; the solver cancels itself at 25.5
  minutes, inside the 29-minute limit.

**Models (6 handles, 3 providers):** `openai/gpt-5.4-mini`, `openai/gpt-5.4-2026-03-05`,
`openai/gpt-5.5`, `anthropic/claude-opus-4-8`, `anthropic/claude-sonnet-4-6`,
`google/gemini-3.1-pro-preview`. Internal-test agent spend by model: GPT-5.4-mini 36%,
GPT-5.4 35%, Opus 4.8 23%, GPT-5.5 5%, Sonnet 4.6 and Gemini under 1% each.

## The evolution run

The run was a cold start from the standard `baseline` seed:
- **Budget:** 600 fresh evals, 14 examples per iteration; 22 iterations, 604 fresh evals used.
- **Judge and settings:** luna no-prose training judge, capped; 2,500-character evidence cap;
  Deep Focus off; full solver menu.
- **Penalty:** `cost_threshold=0.279`, `cost_per_error=0.0279` (10% of the threshold).

The winning lineage is wide_evidence (6) → meta_gold_window (8) → gold_style_criteria (9) →
wide_deep_recall (10) → qualifier_criteria (11) → citation_snowball (13) →
snowball_hydration (14) → alt_readings_head_check (16) → one_record_per_work (17) →
stage1_keyed_order (18) → backdated_citation_counts (19) → **year_steered_expansion (20)**.
- **Winners:** 12 distinct iteration winners.
- **Failures:** no exceptions, zero-accuracy cases or clone detections.
- **Reachability guard:** fired at iteration 22 (no new agent could pass the leader).
- **Selection:** iter20's Elo of 1567 is 7 points ahead of iter11_qualifier_criteria (1560),
  on 3 rounds. It is thinly validated, but it is the run's result under the method's
  selection rule.

## Snapshot layout (~13 MB)

- `agents/` holds all 21 pool agents. The submitted agent is
  `agents/iter20_year_steered_expansion/agent.py`.
- `iteration_019/`, `iteration_020/` hold the winner's breeding data and debut round. Only
  validation-split gold is included: those 24 problems share no IDs or query text with the
  test queries.
- `evolution_output/CLAUDE.md` and `evolution_output/iteration_020/` are the Claude Code
  session that produced the winner.
- `checkpoint.json` and `final_report.md` are the full run record.
- `test_results.json` (+ `.per_problem.json`) is the internal held-out eval: stock GPT-4o,
  capped, original pass (267 rows, one of which never ran). `test_results.merged.*` is the
  267-query result quoted above, with that row replaced by its re-run.
  `test_results.rejudge_openai_gpt-5.6-luna_noprose.*` is the same submissions rejudged on
  the training judge; its `.merged.*` files are the 267-query result.
- `test_results.tool_init_rerun.*` is the re-run of the remaining query, judged by luna and
  by GPT-4o (see the note at the end).

## Submission form metadata

| Field | Value |
| --- | --- |
| Agent name | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-mini, gpt-5.4-2026-03-05, gpt-5.5, claude-opus-4-8, claude-sonnet-4-6, gemini-3.1-pro-preview |
| URL | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |
| Tarball | `v0_1_1_cap_0_279_fable52.tar.gz` |

Agent description as submitted on the form (2026-11-15):

> Evolved by RoboPhD with Claude Fable 5.2 as the evolution (but not a solver)
> model, under a \$0.279/query training cost cap on solver spend. The cap was set
> at exactly the price of the board's top entry (a prior RoboPhD submission), so evolution had to win on score
> at a price already known to be competitive. It came in at \$0.232/query.
>
> The agent routes each query by type, and a Claude Opus 4.8 planner turns every
> request into a plan. For semantic queries, the plan's keyword and body-text
> snippet searches run in parallel and their results are fused. The benchmark's
> grading criteria are hidden at query time, so the planner infers and weights
> its own. GPT-5.4-mini rates every candidate against each criterion, and grades a
> second tranche when the first proves rich. The agent then follows citations in
> both directions from the papers that satisfy every criterion, grading first the
> papers linked to several of them. GPT-5.4 re-judges up to 80 of the strongest
> candidates. Each submitted paper carries verbatim passages from its body.
>
> For named-paper queries, the planner proposes real titles, GPT-5.5 and Gemini
> 3.1 Pro give independent readings of what the request refers to, and Opus 4.8
> picks the paper or papers from the pooled candidates. For metadata queries, the
> planner writes a structured spec that deterministic filters apply, using
> citation counts back-dated to the benchmark's cutoff. For year-bounded
> requests, each search is also issued once per requested year, which steers the
> year-less search tool toward those years.

## Reproduce

```bash
pip install litellm==1.88.1     # submission-scoring price map
# The commit must be pushed first: astabench stamps eval_spec.revision
python scripts/asta_paper_finder_submit.py --only v0_1_1_cap_0_279_fable52 --limit 3   # smoke (~$3)
caffeinate -i python scripts/asta_paper_finder_submit.py --only v0_1_1_cap_0_279_fable52   # full (measured: $268.80, 9h08m)
```

Re-running the same command resumes an interrupted eval-set. Measured spend: $206.78
judge (250 papers per semantic query, uncapped) + $62.03 agent.

## Note: re-run of `semantic_236`

`semantic_236` failed at tool initialization on the internal pass, so the agent never ran
on it. It was re-run once on 2026-11-15 with the same agent, menu and cap, and judged by
both judges: 0.6179 on stock GPT-4o and 0.6677 on luna, at $0.285
(`test_results.tool_init_rerun.*`). The internal results above include it
(`test_results.merged.json`). The original 266 rows alone, with this query counted as 0,
give 0.48537 @ $0.23080 on GPT-4o (`test_results.json`); on luna the 266 give 0.50968.

This affects the internal eval only. The official run had no such failure: all 267 queries
ran and were scored, so the official result needs no correction.
