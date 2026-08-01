# v0_0_9_cap_0_063_opus5 — PaperFindingBench (AstaBench) submission snapshot

Third RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier). `0_0_9` continues the
cross-benchmark patch sequence; `cap_0_063` names the training-time mean-spend
free zone ($0.063/query); `_opus5` = evolved by Claude Opus 5.

The name drops the `soft_`/`sharp_` prefix carried by `v0_0_7` and `v0_0_8`.
That prefix encoded the cost-penalty slope, which now standardizes at 10% of
the threshold by default, so it no longer distinguishes anything.

## Price convention

All prices here are quoted to **three decimals**, the leaderboard's precision.
Two entries that round to the same figure are a **cost tie**, and a tie is
decided on score alone. This matters immediately: see below.

## What this displaces

Unlike `v0_0_8`, this is not a new Pareto point beside the others — it takes a
slot from our own earlier entry. Internal 0.3839 @ **$0.053/query**, a cost tie
with `v0_0_7` (0.3749 @ $0.053; the two agree to four decimals, $0.0533, but
the tie is already settled at three), for **+0.009 more score**. On a tie the
score decides, so if the official run transfers it strictly displaces
`v0_0_7`. It also dominates Ai2's Asta v0 (0.3757 @ $0.063) on both axes.

It does **not** reach Asta Paper Finder (0.3972 @ $0.063), which remains the
next bar above and is the target for a future run.

## The cost gate is a competitor's price, again

Like `v0_0_8`'s $0.033, the $0.063 free zone is not a round number — it is a
competitor's price. Asta Paper Finder's cheaper frontier point is **$0.063**
(0.0626 raw, and three decimals is where the board stops). The gate is that
figure exactly, not a margin above it.

Setting the free zone *at* the competitor's price means evolution is willing to
spend at most what that competitor spends, which buys the **cost** half of a
Pareto-dominance claim by construction and leaves only the **score** half to
win — and under the tie rule, landing *on* $0.063 would still have been enough,
since a cost tie is decided on score. Above it the score bar jumps from our own
0.3749 to Ai2's 0.3972, so there is no free-zone incentive to cross.

Where this differs from v0_0_8 is which competitor was named. That gate aimed
at the *second-cheapest* board entry and the run then undershot it 5.5×, so the
domination that followed was partly accidental. Here the target is the frontier
point directly above our own, and the lineage did not undershoot: the winning
agent's docstring projects
"~$0.071/semantic query, ~$0.053 batch mean against the $0.063 free-zone
threshold"; the observed training batch mean was $0.054 and the 267-query test
mean $0.0533 — **$0.053** on the board. It engineered to the threshold and
landed on it, which is what makes that price deliberate rather than a leftover.

## Opus 5 spends its allowance; Opus 4.8 did not

This is our first leaderboard submission of any kind whose evolution model is
Claude Opus 5, and the cost behavior is visibly different. Cap utilization
(test-set mean agent spend as a fraction of the training free zone) across the
paper_finder campaign:

| Evolution model | runs | cap utilization | mean |
| --- | --- | --- | --- |
| opus-4.8 | 5 | 18%, 30%, 65%, 67%, 80% | **52%** |
| **opus-5** | 3 | 59%, 81%, **85%** | **75%** |
| fable-5 | 2 | 79%, 93% | 86% |

Opus 4.8 routinely left half its allowance unspent and its spread is enormous
(18–80%); Opus 5 lands in a tighter, higher band (59–85%) much closer to
Fable's. Since the free zone carries no penalty below the threshold, unspent
allowance is forgone score — every dollar under the cap could have bought
retrieval depth or evidence enrichment for free.

This run is the clearest case: 85% of cap, our **highest `specific_f1` across
ten runs** (0.8114), and the frontier point. Read it as a consistent behavioral
difference worth planning caps around, not as a proven causal claim — the three
Opus 5 runs sit at three different thresholds, and no controlled A/B has been
run.

## Snapshot layout (ds1000 precedent: a curated run snapshot, ~15MB)

- `agents/` — all 20 iteration winners + seed (`agents/iter15_verdict_repair/agent.py` is the submitted one)
- `iteration_014/`, `iteration_015/` — the winner's breeding data and debut (full per-problem diagnostics incl. `submission.json` / `score_calculation.md` / `judge_verdicts.md`)
- `evolution_output/` — `CLAUDE.md` + `iteration_015/` only (the session that produced the winner — ds1000 precedent)
- `checkpoint.json`, `final_report.md`, `test_results.json`, `test_results.per_problem.json`

**Redistribution note:** as with `v0_0_7` and `v0_0_8`, the iteration problem
dirs embed VALIDATION-split gold (`gold_criteria.md`, and gold-derived content
in `score_calculation.md` / `judge_verdicts.md`) from the HF-token-gated
PaperFindingBench dataset. Verified before commit: the **25** problem IDs
embedded here have **zero** overlap with the held-out test IDs. 266 of the 267
test IDs were checked explicitly; the 267th lost its directory to a per-query
timeout, and is a test-split ID that cannot appear in a training iteration by
construction. Same clean result as v0_0_7 (25 IDs) and v0_0_8 (24 IDs).

## Lineage

- Run: `robophd-asta_paper_finder-010` (`../robophd_runs/robophd/asta_paper_finder_20260730_031141`)
- Engine: RoboPhD Elo (Opus 5 evolution, Deep Focus OFF — `new_agent_test_rounds=0`),
  evaluation budget 600 (605 used), 22 iterations, 14 examples/iteration
- Winner: `agents/iter15_verdict_repair` — final Elo 1571, 5 Elo test rounds,
  train mean 45.655 @ $0.054
- 12 distinct winners across 22 iterations
- **First run with the Elo-reachability guard enabled.** It fired at iteration
  22, switching that iteration to `greedy`: no evolution session, 14 fresh
  evals instead of the usual ~28. No suppression — the winner was created at
  iteration 15. Replayed against five prior paper_finder runs beforehand
  (`scripts/elo_reachability.py`); it would have fired on four of them and
  suppressed no winner in any.
- Second run on the post-`ad7faf24` harness, where the live scorer rebuilds the
  nDCG grade list in canonical submission order — the internal score no longer
  depends on judge-cache warmth.

## Architecture (2,349 lines, single `agent.py`)

The thesis is one piece of arithmetic. The benchmark's grade-3 gate is
`weighted = min(1, Σ w_c·r_c/3) > 0.99` — essentially every criterion Perfect —
and only grade-3 papers earn recall. The predecessor ranked with that same
*additive* form on the judge's 3-valued scale and broke twice: over three
criteria the scale has 27 outputs, so on one query it tied **119 papers at
exactly 1.0** (order falling back to retrieval order, rank 0.534) and on
another predicted 0 grade-3 where the judge found 26; and additivity let a
heavy criterion buy off a missing light one, `(3,3,0)`=0.70 outranking a
balanced `(3,1,1)`=0.60. The gate is conjunctive; the ranker was not.

So: **0–9 per criterion** (same one digit, same token cost, ~1000× resolution,
no saturation), aggregated by **weighted geometric mean** — the conjunctive
form of the gate.

**No gold criteria reach the agent.** The benchmark's `relevance_criteria` are
gold-side, surfaced only post-hoc in a *training* problem's `gold_criteria.md`;
at query time the solver has just `state.metadata["raw_query"]`. Every
"criterion" above is therefore the agent's own: one planner call infers 2–4
requirements from the request text alone (`agent.py:1677`), and the ranker, the
evidence enrichment, and the repair targeting all score against that
reconstruction, never against the rubric the benchmark judge uses. The
reconstruction imitates the gold's *shape* — same fields, weights summing to
1.0, and `_weights()` defaults to splits observed across training diagnostics —
which is training-set calibration, not test-set access. The practical
consequence is a real bound on the ranking signal: it is only as good as one
`gpt-5.4-mini` call's recovery of a hidden rubric from a one-line request.

Grading is a three-band ensemble across two providers, with per-grader votes
kept separate and combined as a reliability-weighted mean; `UNVETTED_SHRINK`
stops a single lenient pass from leapfrogging a paper two skeptical graders
confirmed, and confirmation count is a tiebreaker strictly *below* the mean so
being looked at more often can never promote a weak paper.

The `specific` path unions two cross-provider title guesses after a diagnosed
1.000-vs-0.000 swing on identical code traced to sampling variance in a single
parametric recall call, and anchors the shortlist on the citation key's author
surname and year — the only signal in the loop not produced by the models that
generated the titles.

Iteration 15's own change replaces a lexical stem-overlap test for picking
evidence-repair targets with the pool grader's per-criterion digits, after
measuring the lexical test as nearly uninformative (P(perfectly_relevant) =
0.37 / 0.48 / 0.53 across uncovered-criteria buckets, 76% of 1047 judged papers
in one bucket), then spends free `snippet_search` calls on near-miss papers
sitting one criterion short of the gate.

Deliberate freeze: `_build_evidence` is kept byte-for-byte from iteration 11,
with a smoke test asserting equality, after a predecessor improved lexical
criterion coverage 0.48→0.58 while grade-3 count on one query went 5→0. The
submitted string is not the place to experiment.

### Models (4 handles, 2 providers)

| Handle | Provider | Role |
| --- | --- | --- |
| `gpt-5.4-mini` | OpenAI | semantic plan, query expansion, metadata venues, wide pool grader |
| `gpt-5.4-2026-03-05` | OpenAI | head-band grader, `specific` verifier, metadata plan, title guess A |
| `claude-haiku-4-5-20251001` | Anthropic | independent deep-band grader |
| `claude-sonnet-4-6` | Anthropic | title guess B |

First RoboPhD submission on this task to call Anthropic models; all four are
priced in the litellm 1.88.1 bundled map (`AGENT_MODELS` preflight).

## Internal results (basis: stock GPT-4o judge, canonical ordering)

| Metric | Value |
| --- | --- |
| Mean F1 (267 queries) | **0.3839** |
| `semantic_f1` (194) | 0.3227 |
| `specific_f1` (38) | **0.8114** — our best on this metric across ten runs |
| `metadata_f1` (35) | 0.2687 |
| Agent cost | $0.0533/query — **$0.053** on the board ($14.23 total) |
| Judge cost (internal, capped + cached) | $72.85 |
| Timeouts | 1 query (scored 0; would read 0.3852 at the semantic mean) |

## Judge basis

Trained against `openai/gpt-5.6-luna` with the `no-prose` prompt profile
(validated at luna's own rerun-noise floor; ~5.7× cheaper per verdict than
stock GPT-4o). Test-scored on stock `openai/gpt-4o-2024-11-20` — the
leaderboard's basis. Training and test judges are deliberately different: a
held-out judge means any judge-specific fitting is a headwind on the reported
number, not a tailwind.

Internal judging is **capped** (`cap_judge_to_estimate: true`, top-K only) and
cached. Official judging is uncapped — see risks.

## Submission form metadata

| Field | Value |
| --- | --- |
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-mini, gpt-5.4-2026-03-05, claude-haiku-4-5-20251001, claude-sonnet-4-6 |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |

Agent description as submitted on the form:

> Evolved by RoboPhD under a $0.063/query training cost cap on solver spend
> — set at exactly the price of the frontier entry directly above ours
> ($0.063), so that evolution had to win on score at a price already known
> to be competitive. It came in at $0.053/query, and is the first agent we
> have submitted that was evolved by Claude Opus 5. The
> agent's organizing insight is that the benchmark awards recall only for
> papers judged Perfect on essentially every criterion. The benchmark's
> criteria are hidden at query time — the agent sees only the
> natural-language request — so it first infers 2–4 requirements from that
> request alone, then grades each candidate 0–9 against its own inferred
> requirements and combines them geometrically rather than additively: one
> weak criterion drags the product down the way the benchmark's own gate
> does, where an additive score lets a strong criterion mask a missing one.
> Ranking runs as a three-band ensemble over two model
> providers, with votes kept per grader so a single lenient pass cannot
> outrank a paper two skeptical graders confirmed. It then spends free corpus
> searches on the papers sitting one criterion short of the threshold,
> fetching passages targeted at exactly the criterion each is missing.

## Official-result risks (assessed pre-submission)

**1. `specific_f1` is the metric to watch, not the judged one.** This is the
lesson v0_0_8 paid for. Internal→official transfer on the two completed
submissions:

| | semantic | specific | metadata | overall |
| --- | --- | --- | --- | --- |
| v0_0_7 | 0.3194 → 0.3227 | 0.7137 → 0.7308 | 0.2957 → 0.2778 | 0.3724 → **0.3749** |
| v0_0_8 | 0.2080 → 0.1809 | 0.7456 → **0.4956** | 0.1384 → 0.1410 | 0.2754 → **0.2205** |

v0_0_8's judged metric moved only −0.027. What broke was `specific` — −0.250,
a deterministic exact-match metric with no judge in it at all — because the
agent's title-resolution behavior was environment-sensitive. This entry posts
our **highest ever** `specific_f1` (0.8114), so it has the most to lose there:
a −0.250 repeat costs 38 × 0.250 / 267 = **−0.036 overall**, landing ~0.348 and
forfeiting the frontier slot to our own v0_0_7.

Mitigating: the cross-provider title-guess union exists precisely to
de-correlate that failure, and it is the mechanism v0_0_8's winner lacked.
Unproven officially either way.

**2. Uncapped official judging.** Internally only the top-K estimate was
judged; officially all 250 submitted papers per query are. The papers beyond K
have never been graded, so the official `rank` term sees grades we have not
observed. This agent ships the full 250 on every semantic query.

**3. Internal is not banked.** Two calibration points, +0.0025 and −0.0549.
Treat 0.3839 as a central estimate with a wide interval, not a result.

**4. Wrapper ceiling.** `PRIMARY_TIMEOUT_S = 3000` (50 min). The run had one
internal timeout at 1770s under a tighter internal cap; the official ceiling
gives headroom and that query may complete.

**5. Tool rate limits.** `tool_pacer.py` is staged and applied via `pace_tools`
at ~8 launches/s per endpoint. Measured during training: `snippet_search` ran
at **100% pacer utilization** while all seven other endpoints sat idle, with
only 3–4 concurrent workers. This agent is snippet-heavy and the official run
uses `--max-samples 6`.

## Official result (2026-07-31)

| Metric | Internal | **Official** | Delta |
| --- | --- | --- | --- |
| Mean F1 | 0.3839 | **0.3762** ± 0.0186 | −0.0077 |
| `semantic_f1` (194) | 0.3227 | 0.3110 | −0.0117 |
| `specific_f1` (38) | 0.8114 | **0.8114** | **±0.0000** |
| `metadata_f1` (35) | 0.2687 | 0.2653 | −0.0034 |
| Agent cost | $0.0533 | **$0.0524** ± 0.0018 | −$0.0009 |

**$0.052 @ 0.376 on the board.** Judge $197.32 + agent $13.99 = $211.31, 7.3 h
at `--max-samples 6`. Per-paper judge rate $0.00407 — v0_0_7's $0.0040, not
v0_0_8's $0.0030, despite near-identical evidence length. Evidence length alone
does not predict the judge bill; the submit script's $0.0040 ceiling was the
better estimator.

The pre-submission risk list held up well. Risk 1 (`specific_f1` is the metric
to watch) was the right thing to name and it came through at **zero drift** —
the failure that cost v0_0_8 −0.250 did not recur. Risk 2 (uncapped official
judging) cost −0.0117 on semantic, real but small.

### It takes the frontier slot from v0_0_7

| | score | $/prob |
| --- | --- | --- |
| v0_0_7 (2026-07-20) | 0.375 | 0.053 |
| **v0_0_9** | **0.376** | **0.052** |

Higher *and* cheaper, so v0_0_7 leaves the curve. It also newly dominates Ai2's
Asta v0 (0.376 @ $0.063) — tied on score at board precision, 17% cheaper.
v0_0_7 could not make that claim: at 0.375 it sat *below* Asta v0 on score and
was merely cheaper, so the two were incomparable. The +0.0013 gain is small but
it is exactly the increment that converts "incomparable" into "dominates".

Frontier after this entry (You.com excluded — no cost data):

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | **RoboPhD (v0_0_9)** | **0.376** | **0.052** | Standard |
| 3 | Asta Paper Finder | 0.397 | 0.063 | Custom interface |
| 4 | Asta Paper Finder | 0.433 | 0.355 | Custom interface |

Two of four slots, both Standard tier. No Standard-tier agent on the frontier
is anyone else's.

### Twins on the outside, substantially different agents

v0_0_9 edges v0_0_7 on both axes — +0.0013 score, −$0.0009 cost — so at board
precision the two entries look like the same agent nudged. They are not. The
headline near-identity hides a composition that differs sharply, and the
margin comes from one category while two others move against it:

| type | n | v0_0_7 | v0_0_9 | delta | effect on overall |
| --- | --- | --- | --- | --- | --- |
| semantic | 194 | 0.3227 | 0.3110 | −0.0117 | **−0.0085** |
| specific | 38 | 0.7308 | **0.8114** | **+0.0806** | **+0.0115** |
| metadata | 35 | 0.2778 | 0.2653 | −0.0125 | −0.0016 |

v0_0_9 is **worse on semantic**, which is 73% of the test set, and worse on
metadata. It wins because one 38-query category moved +0.08 — enough to cover
both losses and leave +0.0013.

The agents are near-identical in size (2,097 vs 2,349 lines, one `agent.py`
each) and opposite in thesis:

- **v0_0_7 blames retrieval.** *"The joint phrase … lives in body text that
  keyword search never sees."* Headline change is body-conjunction retrieval —
  more planner snippet queries, phrased as method-section statements, each with
  its own round-robin source list.
- **v0_0_9 blames ranking and evidence.** *"The papers are already on topic;
  the evidence is what is thin."* It leaves retrieval alone and rebuilds the
  ranker (conjunctive geometric aggregation) and the evidence-repair targeting.

One says *we aren't finding the right papers*; the other says *we're finding
them and can't tell them apart*. Each is measurably right about a different
part of the benchmark, and the irony is that the agent theorizing about
semantic ranking **lost** on semantic and won on the exact-match category it
barely discusses.

The likely mechanism for that win is documented rather than accidental:
v0_0_9's `specific` path unions two **cross-provider** title guesses (`gpt-5.4`
+ `claude-sonnet-4-6`), built after diagnosing a 1.000-vs-0.000 swing between
agents running *identical* resolution code, traced to sampling variance in one
parametric recall call. v0_0_7 has only single-provider alias hedging — no
second distribution to union against. Sonnet fired lightly in the official run
(7,971 input tokens, `specific` queries only) for +0.0806 on that category.
Hold it as a strong hypothesis, not a settled result: it is one A/B with many
other differences.

**Implication for the next run:** our two best agents disagree about where the
difficulty lives and each is right about a different part. Neither has both.
v0_0_7's retrieval front-end under v0_0_9's `specific` path would score roughly
0.3227 + 0.8114 + 0.2778 weighted ≈ **0.385** — most of the way to Asta's 0.397
at our price, and without a new thesis.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_opus5 --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_opus5             # full (measured: $211, 7.3h)
```

Push the commit **before** the full run — `astabench eval` stamps the commit
SHA into `eval_spec.revision`, and the form points reviewers at the GitHub URL
above; an unpushed SHA resolves to nothing.

A fresh full run first moves astabench's accumulated stock judge cache aside
and judges cold (`backup_stock_judge_cache` in the submit script): warm-cache
scoring permutes the nDCG verdict ordering, so cold is the only reproducible
official basis (`../robophd_runs/docs/astabench_judge_ordering_issue.md`).
Budget the full judge spend accordingly.

Projected judge spend, measured from this run's own submissions: 250
papers/query × 193 semantic queries = 48,250 papers at 747 chars each. At
v0_0_8's measured $0.0030/paper (750 chars — near-identical evidence length),
**~$145**; the submit script's ceiling formula prints ~$193.

Then upload `submissions/asta_paper_finder/v0_0_9_cap_0_063_opus5.tar.gz` via
the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [x] Official eval run (2026-07-31: 0.3762 @ $0.052396/query, $211.31 spend, 7.3h)
- [x] Tarball uploaded — submitted to AstaBench 2026-07-31, under official review
- [x] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`

Three RoboPhD entries are on the board once this clears review: v0_0_7 (0.375 @
$0.053, displaced by this one), v0_0_8 (0.220 @ $0.006) and this (0.376 @
$0.052).
