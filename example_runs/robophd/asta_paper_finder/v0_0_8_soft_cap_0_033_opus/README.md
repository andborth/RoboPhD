# v0_0_8_soft_cap_0_033_opus — PaperFindingBench (AstaBench) submission snapshot

Second RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier), and the cheap
counterpart to `v0_0_7_soft_cap_0_06_fable`. The `0_0_8` patch continues
the cross-benchmark submission sequence; `soft_cap_0_033` names the
training-time mean-spend free zone ($0.033/query); `_opus` = evolved by
Claude Opus 4.8.

The two submissions are distinct Pareto points from the same benchmark,
not competitors: v0_0_7 buys score (0.3749 @ $0.0533), this one buys
price (0.2754 @ $0.006).

## The cost gate was aimed at a named competitor

The $0.033 free-zone threshold was not a round number. Smolagents Coder
GPT-5 Mini sits on the board at **0.172 @ $0.03**, so a free zone just
above $0.03 forces evolution to land at or under that agent's price —
securing the *cost* half of a Pareto-dominance claim by construction and
leaving only the *score* half to win. The run cleared 0.172 comfortably
at 0.2754.

Evolution then overshot the target: rather than spending up to the
ceiling it came in at **$0.006/query**, 5× under the gate, which also
undercuts **Smolagents Coder Llama 4 Scout 17B (0.070 @ $0.013)**. The
entry therefore dominates *both* cheap-end points rather than the one it
was aimed at — strictly higher score at strictly lower cost than every
listed agent at ≤$0.03, with no competitive neighbor anywhere near
$0.006.

Recording this because the technique generalizes: a cost gate
reverse-engineered from a specific leaderboard entry is a repeatable way
to buy one axis of a dominance claim outright, and this run is evidence
it works.

## The submitted agent is the platform's pick, not ours

`iter9_rerank_rich_v1` won on Elo (1589.209). The runner-up,
`iter14_richev_grade3rerank_v1`, finished **1.07 Elo behind** (1588.136)
with a **26% higher mean training score** (30.54 vs 24.21) and won the
four most recent iterations — a human selecting by hand could reasonably
have picked it.

We submitted the Elo winner anyway. The purpose of the exercise is to
measure what RoboPhD's own selection mechanism yields end-to-end;
substituting human judgment at the last step would test us, not the
platform. Noted here so the near-tie is on the record rather than looking
like an oversight — and so that if iter9 underperforms officially, the
gap to iter14 is a known, pre-registered candidate explanation.

## Snapshot layout (ds1000 precedent: a curated run snapshot, ~10MB)

- `agents/` — all 22 iteration winners (`agents/iter9_rerank_rich_v1/agent.py` is the submitted one)
- `iteration_008/`, `iteration_009/` — the winner's breeding data and debut (full per-problem diagnostics incl. `submission.json` / `score_calculation.md` / `judge_verdicts.md`)
- `evolution_output/` — `CLAUDE.md` + `iteration_009/` only (the session that produced the winner — ds1000 precedent)
- `checkpoint.json`, `final_report.md`, `test_results.json`, `test_results.per_problem.json`
- `test_results.judge_gpt-5.6-luna-noprose.json` — the luna/no-prose arm of the first complete stock-vs-luna pairing on any run (see Judge basis below). Keeps its judge-suffixed filename so it can never be mistaken for the official-comparable result.

**Redistribution note:** as with v0_0_7, the iteration problem dirs embed
VALIDATION-split gold (`gold_criteria.md`, and gold-derived content in
`score_calculation.md` / `judge_verdicts.md`) from the HF-token-gated
PaperFindingBench dataset. Verified before commit: the 24 problem IDs
embedded here have **zero** overlap with the 267 held-out test IDs, so no
test gold is published. Same decision and same clean result as v0_0_7
(25 IDs, zero overlap).

## Lineage

- Run: `robophd-asta_paper_finder-006` (`../robophd_runs/robophd/asta_paper_finder_20260724_004103`)
- Engine: RoboPhD Elo (Opus 4.8 evolution, Deep Focus OFF — `new_agent_test_rounds=0`),
  evaluation budget 600 (602 used), 22 iterations, 14 examples/iteration
- Winner: `agents/iter9_rerank_rich_v1` — crowned at iteration 9, took the
  Elo lead at iteration 15 and held it through 22 (7 Elo test rounds,
  final Elo 1589, train mean 24.21)
- 10 distinct winners across 22 iterations; meta-evolution configured
  (opus-4.8, first@4, cadence 3) but never fired ($0.00)
- First full-stack cheap-judge campaign: luna no-prose training judge +
  enforced 2500-char evidence cap + the $0.033/$0.003 cost gate

## Architecture (1,006 lines, single `agent.py`)

Facet-diverse retrieval with a rich-evidence reranker, descended from
`iter6_metadata_targeted_v1` with the semantic reranker as the single
delta: it judges each candidate on its already-enriched abstract rather
than one snippet, on a finer 0–10 judge-aligned scale, so all-aspect
papers sort above topic-only ones. That targets the `rank` term (measured
0.34–0.55 on ~40% of semantic queries, where grade-3 papers were
retrieved but poorly ordered) — and since score = harmonic(rank, recall),
better ordering lifts both terms. Inherited from iter6: metadata
citation-relation handling (citing/cited-by anchor resolution +
intersection), robust author+venue+year selection with explicit year-SET
support, targeted per-paper facet evidence via scoped `snippet_search`
over the top ~15 recall-critical candidates, and `snippet_search` as a
third candidate source on `specific`. Every guard degrades to proven
prior behavior on failure.

**Every LLM call goes through one model** — `GPT_5_4_MINI`
(`openai/gpt-5.4-mini`), no reasoning tier. Tool calls are free and used
generously. This single-model shape is the direct answer to the tight
cost gate; the $0.036+ lineages all reach for a second, stronger model.

## Internal results (basis: full stock GPT-4o re-eval)

- Test (267 queries, no cost penalty at test time): **mean F1 0.2754**
  — semantic 0.2080, specific 0.7456, metadata 0.1384
- Mean agent cost **$0.006/query** — 5.5× under the $0.033 free zone, so
  no penalty applied anywhere in the run
- Judge spend (informational, never penalized): $66.99 stock
- Whole campaign cost $98.65 (evolution $64.89 + train judge $30.08 +
  eval $3.68) — cheapest to date by ~2×

**Where this is weak, stated plainly:** semantic at 0.2080 sits well
below the $0.036+ lineages' ~0.32–0.37. $0.006/query is below the knee of
the cost-quality curve for semantic retrieval. `specific` (0.7456) holds
up at any budget. This is an honest cheap-point submission; the claim is
Pareto-dominance at the cheap end, not frontier score.

## Judge basis

Training and agent selection ran on the **luna no-prose** judge
(`openai/gpt-5.6-luna`), the calibrated cheap judge. The headline above
is a **full stock GPT-4o re-eval** (2026-07-24) so it is directly
comparable to runs -002 through -004 and to official astabench numbers.

The luna arm is preserved in
`test_results.judge_gpt-5.6-luna-noprose.json` (0.2890). The paired
per-query diff across all 267 is **+0.0136 overall (se 0.0091)** and
**+0.0173 semantic-only (se 0.0077)** — the first complete stock-vs-luna
pairing on any run, and the validation at full scale of the no-prose
calibration. The Pareto claim rests on the stock number, not the luna
one.

## Submission form metadata

| Field | Value |
| --- | --- |
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-mini (every call) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |

Agent description as submitted on the form:

> Evolved by RoboPhD with Claude Opus 4.8 as the evolution (but not a
> solver) model, under a $0.033/query soft training cost cap on solver
> spend — a cap set just above the cheapest competitive entry on the
> board so that evolution had to win on score at a price already known to
> be competitive. It answered by building a pipeline that runs every LLM
> call on a single cheap model, GPT-5.4-mini, and came in at $0.006/query,
> five times under its own cap. The agent retrieves facet-diverse
> candidates from keyword and snippet search, enriches abstracts, then
> reranks on a fine-grained judge-aligned scale so papers matching every
> aspect of a query sort above merely on-topic ones — ordering matters
> because the benchmark's semantic score is a harmonic mean of rank and
> recall. It adds scoped snippet searches to cover facets a candidate's
> text misses, handles citation-relation and author/venue/year metadata
> queries explicitly, and submits a best-first ranked list with evidence
> quoted verbatim from retrieved text. Same RoboPhD code base as our
> DS-1000 submissions and as v0_0_7 on this task; this entry targets the
> cheap end of the cost-quality frontier rather than peak score.

## Official-result risks (assessed pre-submission)

- **Judging basis**: official `astabench eval` judges fresh and uncapped
  (every submitted paper, no top-K cap, no cache); internal used
  `cap_judge_to_estimate=true` and a shared cache. v0_0_7 shifted
  *favorably* across this same boundary (0.3724 → 0.3749, cost repriced
  cheaper), so the precedent is good — but that is one data point.
  Projected official judge spend: 194 semantic queries × 203.5 submitted
  papers ≈ 39.5K verdicts. At the ~$0.0042/paper measured on v0_0_7 that
  is ~$166, but this agent ships much shorter evidence (750 chars/paper
  vs 976), so expect **~$140–175** — below v0_0_7's actual $192. Note the
  submit script's built-in projection assumes a flat 250 papers/query and
  will print a higher number (~$219); it overestimates for this agent.
- **No self-pacing**: unlike v0_0_7's agent (SOFT_DEADLINE 1300 /
  TAIL_DEADLINE 1550), this one has **no deadline constants at all**, no
  per-call timeouts, and swallows tool exceptions. Its runtime comes from
  a fixed work plan, not a clock. Internal wall clock was median 121.5s /
  p95 323.9s / max 560.5s (including in-process judge), so the wrapper's
  3000s ceiling is a wide margin — but it can only ever fire on a genuine
  hang, and firing means a seed-tier fallback for that sample.
- **Judge pin**: the relevance judge is hardcoded in astabench
  (`openai/gpt-4o-2024-11-20`) — the same judge the headline number above
  was computed with.
- **Repricing**: `astabench score` uses litellm's bundled price map
  (1.88.1 prices `gpt-5.4-mini` at $0.75/$4.50 per M). Single OpenAI
  model, so the DS-1000 v0_0_6 Gemini reasoning-token trap does not
  apply. At $0.006/query the free-zone margin is enormous; repricing
  cannot plausibly threaten the cost claim.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_8_soft_cap_0_033_opus --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_8_soft_cap_0_033_opus            # full (~$140-175, 12-19h)
```

Push the commit **before** the full run — `astabench eval` stamps the
commit SHA into `eval_spec.revision`, and the form points reviewers at
the GitHub URL above; an unpushed SHA resolves to nothing.

Then upload `submissions/asta_paper_finder/v0_0_8_soft_cap_0_033_opus.tar.gz`
via the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [ ] Official eval run
- [ ] Tarball uploaded
- [ ] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`
