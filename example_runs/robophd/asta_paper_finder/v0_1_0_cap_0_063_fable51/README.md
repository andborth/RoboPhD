# v0_1_0_cap_0_063_fable51 — PaperFindingBench (AstaBench) submission snapshot

Curated snapshot of RoboPhD run `asta_paper_finder_20261019_184607`
(`robophd-asta_paper_finder-021`, 2026-10-19 → 2026-10-20). It is the
**Claude Fable 5.1** arm at the $0.063 gate, run with the full objective. The winner is
`iter16_feedcast` (Elo 1590, 5 Elo rounds), selected by the method's own
rule (max Elo).

## Why it was submitted

The $0.063 gate is Asta Paper Finder's price (0.397 @ $0.0626), and that's
the one non-RoboPhD entry still on the PaperFindingBench frontier. Two
earlier attempts at this gate came in under it officially: v0_0_9_cap_0_063_opus5 at 0.376 and
v0_0_9_cap_0_063_fable at 0.390. This run is the first projected to clear it.

Internal held-out test: **0.39787 mean adjusted F1 @ $0.05783/query**
(stock GPT-4o judge, capped). That's the same basis as v0_0_9_cap_0_063_fable's internal 0.38738,
so this run is +0.0105 at equal cost.

- **Projected official: ~0.403.** Uncapped judging adds +0.003 to +0.006, measured on
  three fixed official submission sets. One sample in the internal test lost its tool
  startup and scored 0, which should recover by about +0.0015.
- **Risk:** the agent's re-run draw has an SD of about 0.0075.
- **What clearing the target needs:** at least 0.398 at three decimals. That's likely (~75%). Three of the
  four historical internal→official deltas on this basis clear it.

If it lands there, it Pareto-dominates Asta Paper Finder (0.397 @ $0.063)
and both of our earlier $0.063 entries (0.376 @ $0.052, 0.390 @ $0.058).

## Naming and conventions

`v0_1_0`, matching DS-1000's v0_1_0_cap_0_127_fable51: the submission
infrastructure is now considered settled. Since the last shipped v0_0_9 run,
the per-sample isolation wrapper (cad0f713) and the Fable 5.1 / Opus 5.5
engine registrations have landed. `cap_0_063` is the $0.063/query mean training cost gate, with
cost_per_error $0.0063. `fable51` means Claude Fable 5.1 was the evolution
model; it is never a solver.

## Snapshot layout

Same curated-run convention as the sibling snapshots (~16 MB):
- `agents/` holds all 20 pool agents (19 evolved plus the `baseline` seed).
  The submitted agent is `agents/iter16_feedcast/agent.py` (2,088 lines). Its
  docstring header still reads "iteration 013 (crosscast)". Iterations 15
  and 16 evolved by applying string patches, so the header was never rewritten. The code is
  iteration 16's.
- `iteration_015/`, `iteration_016/` hold the winner's breeding data and
  debut round: per-problem diagnostics including judge verdicts, gold
  criteria and submissions. **Only validation-split gold is included:** the 25 embedded
  problem ids share no ids and no query text with the 267 held-out test queries.
- `evolution_output/CLAUDE.md` + `evolution_output/iteration_016/` are the
  Claude Code session that produced the winner: reasoning, reflection,
  `patch_agent.py`, and a smoke harness.
- `checkpoint.json`, `final_report.md` are the full run record (21 iterations,
  604 fresh evals; the reachability guard fired at iteration 21).
- `test_results.json` (+ `.per_problem.json`) is the internal held-out
  eval: stock GPT-4o, capped, 267 queries, 1 tool-startup failure.

## Lineage

The run is a cold start from the standard `baseline` seed:

| Setting | Value |
| --- | --- |
| Budget | 600 fresh evals, 14 examples per iteration |
| Deep Focus | off |
| Training judge | luna no-prose, capped |
| Training evidence cap | 2,500 characters |
| Objective | full (generalization sentence present, unlike ablation arms -016/-017) |

The winning lineage is surecast (6) → headcast (7) → steadycast (9) → clearcast (10) →
citecast (11) → provecast (12) → crosscast (13) → bodycast (15) →
**feedcast (16)**. There were 10 distinct iteration winners, and no exceptions,
zero-accuracy cases or clone detections. The final Elo is 1590, against 1589 for
iter18_oldcast, so the winner is thinly validated. It is still the run's result under the
method's selection rule.

## Architecture of `iter16_feedcast`

The agent routes each query by score type.

- **Semantic (73% of test).**
  - **Retrieval:** a GPT-5.4-mini planner writes keyword queries, plus criterion-targeted snippet
    queries (one per required aspect). The results are fused.
  - **Grading:** yield-adaptive bulk grading on GPT-5.4-mini grades 520 candidates, then 160 more at
    a time while the tail still yields, up to 600.
  - **Expansion (stage 2):** adds the top graded papers' citers plus pseudo-relevance feedback,
    which re-searches the titles of the top graded papers as keyword queries.
  - **Rerank:** the top 32 are re-judged per criterion by GPT-5.4.
  - **Evidence:** the top 100 get verbatim body passages from scoped `snippet_search`, chosen to cover
    the criteria the abstract misses.
  - **Iteration 16's own changes:** the pseudo-relevance feedback step, cost trims to bring the semantic
    mean inside the free zone, and content-only planner criteria (no date or venue
    criteria).
- **Specific:** GPT-5.4 and Sonnet 4.6 planners run in parallel. A GPT-5.4
  picker returns a confidence, and a submit-count rule turns that confidence into
  1–6 papers.
- **Metadata:** GPT-5.4 parses structured filters. Author and venue retrieval
  draws on both citation directions, and a "cites X and Y" query is answered by intersecting the
  citer sets of X and Y.

### Models (3 handles, 2 providers)

`openai/gpt-5.4-mini`, `openai/gpt-5.4-2026-03-05`,
`anthropic/claude-sonnet-4-6` (second specific-query planner only). All three are
already in the submit script's `AGENT_MODELS` pricing preflight.

## Internal results

| Metric | Value |
| --- | --- |
| Mean adjusted F1 (stock 4o, capped) | **0.39787** (106.23 / 267) |
| semantic_f1 / specific_f1 / metadata_f1 | 0.3355 / 0.7536 / 0.3676 |
| vs v0_0_9_cap_0_063_fable (same basis) | +0.021 / −0.093 / +0.073 |
| Agent cost/query | $0.0578 (92% of the $0.063 zone); p50 $0.063, max $0.104 |
| Wall clock/query | median 703 s, max 1,373 s (1,800 s cap) |
| Projected official | ~0.403 @ ~$0.058 |

**Known weak spot: specific queries.** Specific scores 28.64/38 against
32.17 for v0_0_9_cap_0_063_fable, which is −0.013 overall. The losses come from two systematic policies, not noise, so they will
recur officially:
1. **Padding single-answer queries.** The submit-count rule submits 2–6 papers unless the picker is
   confident and the query is unambiguous. On a query with one gold paper, F1 = 2/(k+1). This hits "the glue
   paper", "the lstm paper", "the rnn paper" and "the gru paper". The
   rule was learned from training nicknames that had 2–5 gold papers.
2. **Duplicate Semantic Scholar records.** LEGOBench and MS MARCO each have two records. A confident
   single pick chose the non-gold record and scored 0.

Full breakdown: `specific_drop_analysis` in
`../robophd_runs/results/asta_paper_finder.json` (-021).

## Submission form metadata

| Field | Value |
| --- | --- |
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-2026-03-05, gpt-5.4-mini, claude-sonnet-4-6 |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |

Draft agent description: finalize at upload, after the official score.

## Official-result risks (assessed pre-submission)

1. **First official run under the per-sample isolation wrapper**
   (cad0f713). All prior PFB submissions used the older wrapper. Before
   any spend, two gates passed on 2026-10-21: `_check_isolation.py` and `test_submit.py` (19/19). The
   `--limit 3` smoke test is the last of them.
2. **Narrow margin.** At the central projection the margin over 0.397 is
   +0.006, against an SD of about 0.0075 for the agent's re-run draw. A draw like
   v0_0_9_cap_0_063_opus5's (−0.008) would land at about 0.395 and miss.
3. **Cost.** The margin to the $0.063 gate is $0.005/query (8%). Internal
   costs are already on the litellm 1.88.1 billing basis, and
   v0_0_9_cap_0_063_fable repriced by −0.4% officially.
4. **Wall-clock tail.** The internal max was 1,373 s, and the agent's own soft limit
   is 1,560 s. The wrapper's fallback chain (primary 3,000 s → seed 1,500 s →
   schema-valid empty) bounds the damage if the official environment runs slower.

## Reproduce

```bash
pip install litellm==1.88.1     # submission-scoring price map
# Push the commit BEFORE the full run — astabench stamps eval_spec.revision
python scripts/asta_paper_finder_submit.py --only v0_1_0_cap_0_063_fable51 --limit 3   # smoke (~$3)
caffeinate -i python scripts/asta_paper_finder_submit.py --only v0_1_0_cap_0_063_fable51  # full (~$215)
```

## Submission status

- [ ] Official eval run (date: score @ cost, spend, wall clock, N/267, errors)
- [ ] Tarball uploaded (date, v0_1_0_cap_0_063_fable51.tar.gz)
- [ ] Official score/cost recorded in ../robophd_runs/results/asta_paper_finder.json
