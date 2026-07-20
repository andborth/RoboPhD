# v0_0_7_soft_cap_0_06_fable — PaperFindingBench (AstaBench) submission snapshot

First RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier). The `0_0_7`
patch continues the cross-benchmark submission sequence after DS-1000's
`v0_0_6_soft_cap_0_003_fable`; `soft_cap_0_06` names the training-time
mean-spend free zone ($0.06/query); `_fable` = evolved by Claude Fable 5.

## Snapshot layout (ds1000 precedent: a curated run snapshot, ~15MB)

- `agents/` — all 21 iteration winners (`agents/iter12_body_conjunction/agent.py` is the submitted one)
- `iteration_011/`, `iteration_012/` — the winner's breeding data and debut (full per-problem diagnostics incl. `submission.json` / `score_calculation.md` / `judge_verdicts.md`)
- `evolution_output/` — `CLAUDE.md` + `iteration_012/` only (the session that produced the winner — ds1000 precedent)
- `checkpoint.json`, `final_report.md`, `test_results.json`, `test_results.per_problem.json`

**Redistribution note (resolved 2026-07-19):** the iteration problem
dirs embed VALIDATION-split gold (`gold_criteria.md`, and gold-derived
content in `score_calculation.md` / `judge_verdicts.md`) from the
HF-token-gated PaperFindingBench dataset. Decision (repo owner): this
modest exposure of training-split data is acceptable to publish — the
held-out test split's gold appears nowhere in the snapshot. Not a
blocker for pushing.

## Lineage

- Run: `robophd-asta_paper_finder-003` (`../robophd_runs/robophd/asta_paper_finder_20260717_170858`)
- Engine: RoboPhD Elo (Fable-5 evolution, Deep Focus OFF — `new_agent_test_rounds=0`),
  evaluation budget 600 (615 used), 21 iterations, 14 examples/iteration
- Winner: `agents/iter12_body_conjunction` — crowned at iteration 12, held
  through 21 (8 Elo test rounds, final Elo 1581, train mean 37.83)
- Meta-evolution configured (opus-4.8, first@4, cadence 3) but never fired

## Architecture (2,097 lines, single `agent.py`)

Body-conjunction retrieval agent (see the agent's own header docstring for
the iteration-12 delta): a GPT-5.4 planner emits 5 snippet-search queries
(≥2 phrased as method-section statements of the query's implied
conjunction) plus keyword/alias queries; each snippet query is its own
round-robin candidate source. Citers-only expansion (references are dead
server-side; probed then skipped), lexical prescreen, GPT-5.4-mini staged
relevance grading with a rescue/verify cascade, band ordering, tail sweep
to saturate big-K queries, alias hedging on `specific_f1`, metadata relax
ladder, and a defensive transport wrapper. Models: `GPT_5_4_MINI`
(`openai/gpt-5.4-mini`, ~73% of spend) + `GPT_5_4`
(`openai/gpt-5.4-2026-03-05`, planner/adjudication).

## Internal results (basis: capped judging + shared test judge cache)

- Test (267 queries, no cost penalty at test time): **mean F1 0.3724**
  (semantic-heavy blend; internal per-type training means: specific 0.72,
  semantic 0.32, metadata 0.28)
- Mean agent cost **$0.0556/query** (inside the $0.06 free zone, ~8% margin)
- Judge spend (informational, never penalized): $88.38 with
  `cap_judge_to_estimate=true` + `shared_test_openai_gpt-4o-2024-11-20.json`
- Would Pareto-dominate the leaderboard's ReAct GPT-5 Mini point
  (0.220 @ $0.060)

## Submission form metadata

| Field | Value |
| --- | --- |
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-mini (grading cascade), gpt-5.4-2026-03-05 (planner/verify) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |

Agent description as submitted on the form:

> Evolved by RoboPhD with Claude Fable 5 as the evolution (but not a
> solver) model, under a $0.06/query soft training cost cap on solver
> spend. The evolved agent is a body-conjunction retrieval pipeline: a
> GPT-5.4 planner rewrites each query into snippet-search probes phrased
> as method-section statements of the query's implied claim — reaching
> conjunctions that live in paper body text where keyword search cannot
> see them — alongside keyword and alias queries, each feeding a
> round-robin candidate pool. GPT-5.4-mini then grades candidates
> against the query's relevance criteria in a staged cascade with
> citation expansion, near-miss rescue, and a GPT-5.4 verification pass,
> and submits a best-first ranked list with evidence quoted verbatim
> from retrieved text. Evolution was presented with the same menu of
> nine LLMs from Anthropic, OpenAI, and Gemini that we used in our
> DS-1000 submissions; it settled on just two. Same RoboPhD code base
> as DS-1000; this is its first literature-search task.

## Official result (2026-07-20)

**adjusted_f1_micro_avg = 0.3749** (stderr 0.0183) @ **$0.0533/query**
(litellm 1.88.1 bundled pricing). Per-type: semantic 0.3227, specific
0.7308, metadata 0.2778 — every mean within noise of its internal
counterpart, so the capped-vs-uncapped judging-basis risk resolved as a
non-issue. Dominates ReAct GPT-5 Mini (0.220 @ $0.06); matches ReAct
Opus (0.374) at 1.6% of its cost ($3.38/query). Run took ~19h wall at
`--max-samples 6` after two false starts whose wrapper ceilings (1500s,
then 2100s) sat inside the agent's evolved pacing under official-harness
contention — the final ceiling is 3000s (see script history).

## Official-result risks (assessed pre-submission)

- **Judging basis**: official `astabench eval` judges fresh and uncapped
  (all ~250 submitted papers/semantic query, no top-K cap, no cache). The
  rank term is then computed over all 250 grades instead of the capped
  top-K; internal calibration of the cap showed no measured score change,
  but the direction of any residual shift is unknown. Projected official
  judge spend: 194 × 250 ≈ 48.5K verdicts at a measured ~$0.0042/paper ≈
  **$205–245**, billed to the submitter's OpenAI key during the eval.
- **Judge pin**: the relevance judge is hardcoded in astabench
  (`openai/gpt-4o-2024-11-20`) — same snapshot internal testing used.
- **Repricing**: `astabench score` uses litellm's bundled price map
  (litellm 1.88.1 verified to price both models at the registry's rates:
  $0.75/$4.50 and $2.50/$15.00 per M). Both models are OpenAI, so the
  DS-1000 v0_0_6 Gemini reasoning-token trap does not apply. Free-zone
  margin if repriced: ~$0.0044/query.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_7_soft_cap_0_06_fable --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_7_soft_cap_0_06_fable            # full (~$200-270, 12-18h)
```

Then upload `submissions/asta_paper_finder/v0_0_7_soft_cap_0_06_fable.tar.gz`
via the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [x] Official eval run (2026-07-20: 0.3749 @ $0.0533/query)
- [ ] Tarball uploaded
- [x] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`
