# v0_0_8_soft_cap_0_033_opus — PaperFindingBench (AstaBench) submission snapshot

Second RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier), and the cheap
counterpart to `v0_0_7_soft_cap_0_06_fable`. The `0_0_8` patch continues
the cross-benchmark submission sequence; `soft_cap_0_033` names the
training-time mean-spend free zone (\$0.033/query); `_opus` = evolved by
Claude Opus 4.8.

The two submissions are distinct Pareto points from the same benchmark,
not competitors: v0_0_7 buys score (0.3749 @ \$0.0533), this one buys
price (0.2754 @ \$0.006).

## The cost gate was aimed at a named competitor

The \$0.033 free-zone threshold was not a round number — it is, to the
cent, a competitor's price. The board's frontier has two points at the
cheap end: Smolagents Coder Llama 4 Scout 17B at **0.070 @ \$0.013**
(cheapest) and Smolagents Coder GPT-5 Mini at **0.172 @ \$0.033**
(second-cheapest). The board rounds both displays down — to \$0.01 and
\$0.03 — so the true figures are what the gate was built from.

The gate was aimed at the **second**-cheapest, because it is the harder
and more valuable target: clearing \$0.033 on cost is easier than clearing
\$0.013, while 0.172 is the higher score bar of the two.

Setting the free zone *at* \$0.033 — exactly that competitor's price,
not above it — means any agent evolution is willing to keep spends at
most what that competitor spends. That buys the *cost* half of a
Pareto-dominance claim by construction, leaving only the *score* half to
win. The run cleared 0.172 comfortably, at 0.2754 internally and 0.2205
officially.

Evolution then overshot the target: rather than spending up to the
ceiling it came in at **\$0.0059/query**, **5.5× under the gate**, which
also undercuts Llama 4 Scout's \$0.013. The entry therefore dominates
*both* cheap-end frontier points rather than the one it was aimed at —
strictly higher score at strictly lower cost than every listed agent at
≤\$0.033. Only the first domination was by design; the second is a
consequence of undershooting, and would not have followed from the gate
alone. Nothing on the board sits near \$0.0059.

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
  (opus-4.8, first@4, cadence 3) but never fired (\$0.00)
- First full-stack cheap-judge campaign: luna no-prose training judge +
  enforced 2500-char evidence cap + the \$0.033/\$0.003 cost gate

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
cost gate; the \$0.036+ lineages all reach for a second, stronger model.

## Internal results (basis: full stock GPT-4o re-eval)

- Test (267 queries, no cost penalty at test time): **mean F1 0.2754**
  — semantic 0.2080, specific 0.7456, metadata 0.1384
- Mean agent cost **\$0.006/query** — 5.5× under the \$0.033 free zone, so
  no penalty applied anywhere in the run
- Judge spend (informational, never penalized): \$66.99 stock
- Whole campaign cost \$98.65 (evolution \$64.89 + train judge \$30.08 +
  eval \$3.68) — cheapest to date by ~2×

**Where this is weak, stated plainly:** semantic at 0.2080 sits well
below the \$0.036+ lineages' ~0.32–0.37. \$0.006/query is below the knee of
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
> solver) model, under a \$0.033/query soft training cost cap on solver
> spend — a cap set at exactly the price of the second-cheapest point on
> the board's cost-quality frontier, so that evolution had to win on score
> at a price already known to be competitive. It answered by building a pipeline that runs every LLM
> call on a single cheap model, GPT-5.4-mini, and came in at \$0.006/query,
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

## Official result (2026-07-25)

**adjusted_f1_micro_avg = 0.2205** (stderr 0.0155) @ **\$0.005943/query**
(litellm 1.88.1 bundled pricing). Per-type: semantic 0.1809, specific
0.4956, metadata 0.1410. Run took 1h32m at `--max-samples 6` — no false
starts, zero sample errors, zero tool errors, zero retries.

**The Pareto claim holds.** Against the two cheap-end frontier points:
dominates Smolagents Coder GPT-5 Mini (0.172 @ \$0.033) by +0.048 score at
1/5 the cost, and Smolagents Coder Llama 4 Scout 17B (0.070 @ \$0.013)
outright. It also dominates ReAct GPT-5 Mini (0.220 @ \$0.060) — equal
score at 10× lower cost, which satisfies dominance (equal-or-better on
every axis, strictly better on one). The dominance there rests entirely
on the cost axis: our 0.2205 and their 0.220 are the same number at the
board's displayed precision, so this is not a claim to a *higher* score.

Cost came in at **\$118.68** total — \$117.09 judge (98.7%), \$1.59 agent —
against a pre-run projection of \$140–175. The agent-side cost matched
internal prediction to four decimals.

### Internal over-predicted; two-thirds of it is `specific`

Weighting the per-type deltas by query count decomposes the −0.055:
**`specific` contributes −0.036 (65%)**, `semantic` −0.020 (36%),
`metadata` ≈0. Per-type figures alone are misleading here — `specific`'s
−0.250 lands on only 14% of queries while `semantic`'s −0.027 lands on
73%, so the smaller-looking metric carries a real share of the total.

`specific` is nonetheless the part that needs explaining: it is the
larger contributor, and unlike `semantic` it has no judging-basis story
available (see below). `semantic`'s −0.027 is statistically real
(t = −3.37) but confounded with the capped+cached → fresh+uncapped
change, so it cannot be attributed to agent behaviour on this evidence.

| | internal (stock) | official | Δ |
| --- | --- | --- | --- |
| overall | 0.2754 | 0.2205 | −0.055 (~3.4σ) |
| semantic | 0.2080 | 0.1809 | −0.027 |
| **specific** | **0.7456** | **0.4956** | **−0.250** |
| metadata | 0.1384 | 0.1410 | +0.003 |

`specific_f1` is deterministic exact-match — **the judge is never
invoked** — so the judging-basis story that explained v0_0_7's clean
transfer cannot apply. Three runs of this same agent settle what did
happen (luna internal / stock internal / official are three independent
agent runs; luna-vs-stock is a *pure* re-run, since the judge does not
touch this metric):

```
specific   luna 0.7325 | stock 0.7456 | official 0.4956
  luna -> stock       Δ +0.013   identical 33/38   sign-test p=1.00   t=+0.28
  stock -> OFFICIAL   Δ -0.250   identical 21/38   sign-test p=0.049  t=-3.11
metadata (the control, also judge-free)
  stock -> OFFICIAL   Δ +0.003   identical 29/35   sign-test p=0.69   t=+0.10
```

Two internal runs agree to within noise; official differs by ~19× that
scale, one-directionally (13 worse, 4 better; 12 queries nonzero in
*both* internals and exactly 0.0 officially). Metadata makes the
identical internal→official transition with no shift at all, which rules
out a general miscalibration of our harness on deterministic metrics.

**But the failure mode is chronic, not new.** Every query that failed in
both internal runs failed the *same* way — a topically-related paper
instead of the canonical one ("the gru paper" → three different plausible
GRU papers, none Cho et al.; "the gpt-3 paper" → *a survey on GPT-3*;
"the snli paper" → *Unpacking the Resilience of SNLI Contradiction
Examples*). On `specific_32` and `specific_13` an internal run made
exactly the official error. So the agent has a standing weakness — it
does not reliably separate "the canonical X paper" from "a paper about
X" — and what changed officially is the *rate*: 6, 5, then 17 failures
out of 38. The weakness is the vulnerability; the environment difference
is only the trigger.

**Consequence for selection, not just reporting:** iter9 was ranked
during evolution on a specific signal inflated by ~0.25 on 14% of the
query mix. Elo may have crowned it partly on a measurement that does not
survive contact with the official harness. The fix is an evolution
target: "the X paper" queries need explicit canonical-vs-derivative
disambiguation (prefer the earliest / most-cited paper that *introduces*
X over one that surveys or extends it). iter9 has no such logic and takes
what search ranks first, which is recency-biased.

## Official-result risks (assessed pre-submission)

- **Judging basis**: official `astabench eval` judges fresh and uncapped
  (every submitted paper, no top-K cap, no cache); internal used
  `cap_judge_to_estimate=true` and a shared cache. v0_0_7 shifted
  *favorably* across this same boundary (0.3724 → 0.3749, cost repriced
  cheaper), so the precedent is good — but that is one data point.
  Projected official judge spend: 194 semantic queries × 203.5 submitted
  papers ≈ 39.5K verdicts. At the ~\$0.0042/paper measured on v0_0_7 that
  is ~\$166, but this agent ships much shorter evidence (750 chars/paper
  vs 976), so expect **~\$140–175** — below v0_0_7's actual \$192. Note the
  submit script's built-in projection assumes a flat 250 papers/query and
  will print a higher number (~\$219); it overestimates for this agent.
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
  (1.88.1 prices `gpt-5.4-mini` at \$0.75/\$4.50 per M). Single OpenAI
  model, so the DS-1000 v0_0_6 Gemini reasoning-token trap does not
  apply. At \$0.006/query the free-zone margin is enormous; repricing
  cannot plausibly threaten the cost claim.

### How they resolved

Three of the four were non-issues: no self-pacing never bit (zero
errors, zero retries, 1h32m total, the 3000s ceiling never fired), the
judge pin held, and repricing reproduced the agent cost to four decimals.
Cost came in **under** the projection (\$118.68 vs \$140–175) — the
per-verdict figure inherited from v0_0_7 was too high for this agent's
shorter evidence.

The judging-basis risk is the one worth re-reading, because **it was
mis-scoped**. It was written as though a judging-basis shift were the
main threat to transfer. It wasn't: semantic (the judged metric) moved
only −0.027, while the metric that actually broke — `specific`, −0.250 —
has no judge in it at all. v0_0_7's clean transfer made the
capped-vs-uncapped question look like the thing to watch, and that
inherited confidence pointed at the wrong risk entirely.

The real lesson for the next submission: internal evals predict the
*judged* metric well, and a deterministic metric can still diverge badly
if the agent's behaviour is environment-sensitive. Do not read "internal
matched official" on one submission as a general guarantee.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_8_soft_cap_0_033_opus --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_8_soft_cap_0_033_opus            # full (~$140-175, 12-19h)
```

Push the commit **before** the full run — `astabench eval` stamps the
commit SHA into `eval_spec.revision`, and the form points reviewers at
the GitHub URL above; an unpushed SHA resolves to nothing.

A fresh full run first moves astabench's accumulated stock judge cache
aside and judges cold (`backup_stock_judge_cache` in the submit script):
warm-cache scoring permutes the nDCG verdict ordering, so cold is the
only reproducible official basis (../robophd_runs/docs/astabench_judge_ordering_issue.md).
Budget the full judge spend accordingly. This run's official score
predates that guard and was scored against whatever the stock cache
held at the time (v0_0_7's verdicts included).

Then upload `submissions/asta_paper_finder/v0_0_8_soft_cap_0_033_opus.tar.gz`
via the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [x] Official eval run (2026-07-25: 0.2205 @ \$0.005943/query, \$118.68 spend, 1h32m)
- [x] Tarball uploaded — submitted to AstaBench 2026-07-27, under official review
- [x] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`

Both RoboPhD entries are under review concurrently: v0_0_7 (0.3749 @
\$0.0533) and this one (0.2205 @ \$0.0059).
