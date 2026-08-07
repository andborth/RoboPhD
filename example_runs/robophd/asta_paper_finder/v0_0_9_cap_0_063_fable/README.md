# v0_0_9_cap_0_063_fable — PaperFindingBench (AstaBench) submission snapshot

Sixth RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier).

> **The submitted agent is [`agents/iter14_title_channel/agent.py`](agents/iter14_title_channel/agent.py)**
> — iteration-14 winner of run `robophd-asta_paper_finder-013`, 2,068 lines,
> two OpenAI model handles. Everything below describes that file.

## The gate is Asta's price; the variable is the evolution model

The training free zone is set at **$0.063/query** — exactly the price of **Asta
Paper Finder's 0.397 @ $0.063**, the one non-RoboPhD entry still holding a
frontier slot. Evolution had to win on score at a price already known to be
competitive.

This is the third use of the competitor-priced-gate technique on this task. It
is the second run aimed at *this* target — but not a repeat of it: the gate,
the guard, the judges, the budget and the harness are all held fixed, and
**the evolution model is the one deliberate change**.

| submission | gate | evolution model | aimed at |
| --- | --- | --- | --- |
| `v0_0_8` | $0.033 | Opus 4.8 | the second-cheapest point on the frontier |
| `v0_0_9_cap_0_063_opus5` | $0.063 | **Opus 5** | Asta Paper Finder, 0.397 @ $0.063 |
| `v0_0_9_cap_0_355_opus5` | $0.355 | **Opus 5** | the board leader |
| `v0_0_9_cap_0_355_fable` | $0.355 | **fable-5** | the board leader |
| **this entry** | **$0.063** | **fable-5** | **Asta Paper Finder, 0.397 @ $0.063** |

Read down the last three rows and the design is visible: the $0.355 gate has
already been run under both evolution models, and this entry supplies the
missing fable-5 arm at $0.063. That is the point of the run — the two attempts
at Asta's price are a controlled pair, not a retry.

The board as it stands, before this entry:

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | RoboPhD (v0_0_9 @ cap 0.063, opus5) | 0.376 | 0.052 | Standard |
| 3 | **Asta Paper Finder** | **0.397** | **0.063** | Custom interface |
| 4 | RoboPhD (v0_0_9 @ cap 0.355, opus5) | 0.432 | 0.251 | Standard |
| 5 | RoboPhD (v0_0_9 @ cap 0.355, fable) | 0.440 | 0.279 | Standard |

`v0_0_7` (0.375 @ $0.053) is listed but off the frontier, displaced by our own
`cap_0_063`.

**Where this entry lands is not claimed here.** Internally it scores 0.38738 @
$0.05828, but internal is not banked — see
[Official-result risks](#official-result-risks-assessed-pre-submission). The
frontier table with this entry inserted, and the outcome against the target,
are filled in under [Official result](#official-result) after the run.

## What this run is for: the 2×2 is now complete

`-013` is the missing **fable-5 × $0.063** cell. All four cells sit on one
frozen stack — zero commits to `examples/asta_paper_finder/` between `-012` and
this run:

| | **$0.063 gate** | **$0.355 gate** |
| --- | --- | --- |
| **opus-5** | `-010` 0.38394 | `-011` 0.43177 (official) |
| **fable-5** | **`-013` 0.38738** | `-012` 0.44025 (official) |
| | **+0.0012** to fable-5 | **+0.0085** to fable-5 |

Same direction at both gates, roughly a seventh the size at the cheap one.

**Read the $0.063 delta as +0.0012, not +0.0034.** The raw 267-vs-267 means
differ by +0.0034, but `-010` carries one row with `sample_id: "None"` — a query
lost to its 1770s subprocess timeout, scored 0.0, identity unrecoverable
(by elimination, `semantic_242`). On the **266 matched query IDs** the gap is
**+0.0012**. Two-thirds of the raw headline is that timeout, which `-011`'s
official run already established is local contention rather than an agent
property. `-013` had **zero** test-set timeouts.

Two further caveats, both load-bearing: **n = 1 per cell**, and the $0.063
comparison is internal-basis while the $0.355 one is official-basis, where
uncapped judging grades every submitted paper rather than the top-K slice.

## The category shape replicates at both gates

Against `-010` on the 266 matched IDs:

| category | n | opus-5 (`-010`) | **fable-5 (`-013`)** | delta | score points (n × delta) |
| --- | --- | --- | --- | --- | --- |
| semantic | 193 | **0.3227** | 0.3126 | −0.0101 | **−1.94** |
| specific | 38 | 0.8114 | **0.8465** | +0.0351 | **+1.33** |
| metadata | 35 | 0.2687 | **0.2950** | +0.0262 | **+0.92** |
| **all** | **266** | 0.38539 | **0.38655** | | **+0.31** |

+0.31 / 266 = **+0.0012**, the difference between the two.

The signs are the same three as the $0.355 cell (metadata **+0.1202**, specific
**+0.0430**, semantic **−0.0184**). **fable-5 trades semantic away for the other
two categories at both price points.** The headline is a trade, not a uniform
gain — which is the same conclusion the $0.355 A/B reached, now replicated at a
gate 5.6× cheaper.

Semantic is close to even query-by-query — 89 solo wins against 96 solo losses
across 193 queries — so the −0.0101 is a small net rather than a systematic
collapse. But semantic carries 73% of the queries, so a small per-query loss
there cancels most of what the two small categories win.

## Cap-hugging replicates too

| gate | opus-5 | fable-5 |
| --- | --- | --- |
| $0.063 | $0.0533 (85%) | **$0.0583 (93%)** |
| $0.355 | $0.2506 (71%) | $0.2786 (78%) |

Four observations, same direction: **fable-5 spends closer to the line at both
price points, and the gap widens as the cap tightens.** At $0.063 it leaves only
7% of the free zone unused.

## Naming and conventions

**The version is `v0_0_9` again.** The patch number tracks the RoboPhD code
base, not the run, and continues the cross-benchmark sequence shared with
DS-1000. This run executed 2026-08-04 on a stack byte-identical to `-012`'s.
Three commits landed on 2026-08-06 — `rejudge_test.py --uncapped` /
`--from-eval-log`, and `evaluator.PERSIST_FULL_EVIDENCE` — but all are
test-time or offline tooling that touch neither the solver nor training, so the
stack is unchanged.

`cap_0_063` names the training free zone ($0.063/query); `_fable` the evolution
model, distinguishing it from `v0_0_9_cap_0_063_opus5` at the same gate. Two
entries at one gate is intended: they are the cheap arm of a deliberate 2×2.

Prices are quoted to **three decimals**, the leaderboard's comparison basis;
the two-decimal figures the board displays are presentation only.

## Snapshot layout (ds1000 precedent: a curated run snapshot, ~17MB)

- `agents/` — all 19 iteration winners + seed (`agents/iter14_title_channel/agent.py` is the submitted one)
- `iteration_013/`, `iteration_014/` — the winner's breeding data and debut
- `evolution_output/` — `CLAUDE.md` + `iteration_014/` only
- `checkpoint.json`, `final_report.md`, `test_results.json`, `test_results.per_problem.json`

**Redistribution note:** as with the five prior snapshots, the iteration problem
dirs embed VALIDATION-split gold. Verified before commit: the **27** problem IDs
embedded here have **zero** overlap with all **267** held-out test IDs. (Prior
counts: 25 / 24 / 25 / 26.) The source run had zero test-set timeouts, so all
267 test dirs existed to check against — `cap_0_063_opus5` could only check 266.

## Lineage

- Run: `robophd-asta_paper_finder-013` (`../robophd_runs/robophd/asta_paper_finder_20260804_143045`)
- Engine: RoboPhD Elo (**fable-5** evolution, Deep Focus OFF), budget 600 (616
  used), 21 iterations, 14 examples/iteration
- Winner: `agents/iter14_title_channel` — Elo **1607**, train mean 43.877 @
  $0.058, **8** Elo test rounds
- 12 distinct winners across 21 iterations
- **Reachability guard fired at iteration 20**, displacing `use_your_judgment`,
  so the last two iterations ran greedy. That is the guard working as designed:
  once no newly evolved agent can still reach the top of the ladder within the
  remaining budget, continued exploration is spend without upside. Fourth live
  firing.
- Clean run: 0 exceptions, 0 zero-accuracy cases, 0 clone detections, and
  **zero test-set timeouts**
- Evolution cost: $27.30 eval + $138.18 evolution + $9.85 training judge =
  **$175.33**

**The winner is the best-measured of the campaign.** Elo 1607 over **8** test
rounds, against the 2–5 rounds typical of prior winners. It was champion from
iteration 14 through the end of the run.

## Architecture of `iter14_title_channel` (2,068 lines, single `agent.py`)

The organizing problem is **pool starvation**: on hard semantic queries the
candidate pool never contains the papers the judge would grade Perfect, and no
amount of reranking recovers what was never retrieved. Every mechanism below
targets that, and the distinctive one is a retrieval channel the lineage did not
previously have.

**The title-guess channel.** Alongside keyword and `snippet_search` probes, the
plan step emits up to **six guesses at the exact titles** of papers that would
satisfy the query, and each is resolved through a by-title lookup. It fires on
starved pools only. The rationale, from the agent's own comment:

> title-guess channel: on starved pools, recalled specific papers reach targets
> that no keyword phrasing surfaces

This reaches papers a model has memorised but that no phrasing of the query
retrieves — a different failure mode from "the search terms were wrong".

**Expansion rounds return partial results rather than fail.** An earlier
lineage member wrapped each expansion round in `asyncio.wait_for`; a timeout
*discarded the whole round* (one query spent 420s and kept nothing, scoring
0.272 against 0.659 with the round completed). Rounds are now bounded by an
elapsed-time cutoff checked before each chunk fetch — chunks past the cutoff are
dropped, chunks already fetched are **returned**. A slow backend yields a
thin-but-graded round, never a total loss. This matters for the official run;
see risk 5.

**The thin-pool gate fires on grade-3 count, not total strength.** Coarse
grade-3 count predicted score monotonically across a batch (0 grade-3s → 0.09–0.15;
40+ → 0.50–0.71), but a `strong < 25` gate missed the worst starvation: one
query had **zero** grade-3s and 103 grade-2s, so `strong = 103` and no
reformulation round fired. Grade-2s earn no recall. The round now also triggers
on grade-3 count < 12, and adds alternate-wording `snippet_search` probes,
because body-text passages reach needle techniques that never surface in an
abstract.

**Predicted-grade-3 promotion.** Recall — grade-3s inside the top K — is the
binding constraint on every contested query. Papers the judge-mimic rerank
scores > 0.99 (all weighted criteria Perfect, mirroring the scorer's own
grade-3 condition) sort ahead of the blend order, but **only when 1–15 such
papers exist**; a rerank flagging dozens is not trusted. A false promotion costs
a few rank positions; a true one buys 1/K of recall.

**Conditional round-2 expansion**, kept from the parent as its one verified win:
round 2 runs only when round 1 yielded ≥ 4 grade-3 candidates. On 7 of 9 prior
round-2 runs it graded ~100 papers for zero grade-3s.

A note for readers of the source: the module docstring still carries the
**parent's** header, so the file self-identifies as `iter13-any-author-gate`.
The lineage text below that header is accurate; only the title line is stale.

**No gold reaches the agent.** It reads exactly two fields from the sample —
`metadata.get("raw_query")` and `metadata.get("score_type")` — plus `sample_id`
for the required output field. An AST parse of executable positions finds **zero**
integer constants above 100000 and **zero** corpus-id-shaped string literals.
The `relevance_criteria` the judge scores against are never visible at runtime;
the agent infers its own.

### Models (2 handles, one provider)

| Handle | Provider | Role |
| --- | --- | --- |
| `gpt-5.4-2026-03-05` | OpenAI | query planning, metadata parse, specific-query parse |
| `gpt-5.4-mini` | OpenAI | candidate grading, distillation, reformulation, rerank, metadata filtering |

Both priced in the litellm 1.88.1 bundled map. The expensive handle is reserved
for the three parse/plan calls; everything that scales with pool size runs on
mini.

## Internal results (basis: stock GPT-4o judge, canonical ordering)

| Metric | Value |
| --- | --- |
| Mean F1 (267 queries) | **0.38738** |
| `semantic_f1` (194) | 0.3141 |
| `specific_f1` (38) | 0.8465 |
| `metadata_f1` (35) | 0.2950 |
| Agent cost | $0.0583/query ($15.56 total) |
| Judge cost (internal, capped + cached) | $73.23 |
| Timeouts | **0** |

Judge spend is 4.7× the agent's own bill — the usual shape on this task at a
cheap gate.

### Still weak: metadata

0.2950 with **12 of 35 queries at zero**. That is a real improvement on `-010`'s
18 zeros at the same gate, but well short of `-012`'s 5:

| entry | gate | `metadata_f1` | zeros |
| --- | --- | --- | --- |
| `-010` (opus-5) | $0.063 | 0.2687 | 18 / 35 |
| **`-013` (fable-5)** | **$0.063** | **0.2950** | **12 / 35** |
| `-012` (fable-5) | $0.355 | 0.3423 | 5 / 35 |

The mechanism that drove `-012`'s metadata result — padding the submission to
250 on broad exact-match queries, derived from the F1 denominator — is
affordable for a $0.355 agent. At $0.063 this run gets partway there by other
means. Metadata remains the category with the most headroom: the public best is
0.6416.

## Judge basis

Trained against `openai/gpt-5.6-luna` with the `no-prose` profile; test-scored
on stock `openai/gpt-4o-2024-11-20`, the leaderboard's basis. A held-out judge
means any judge-specific fitting is a headwind on the reported number, not a
tailwind. Internal judging is capped (`cap_judge_to_estimate: true`); official
judging is uncapped — see risks.

Measured for this run: luna/no-prose grades semantic **+0.041** above stock
GPT-4o on identical submissions (0.35536 vs 0.31447 on a full rejudge). The
training signal therefore ran systematically optimistic on the only judged
category relative to the basis reported here. Read that as a level shift rather
than a demonstrated ranking distortion — a uniformly more generous judge need
not mis-order agents, which is what evolution actually consumes.

## Submission form metadata

| Field | Value |
| --- | --- |
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-2026-03-05, gpt-5.4-mini |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |

Agent description as entered on the form:

> Evolved by RoboPhD with Claude Fable 5 as the evolution (but not a solver)
> model, under a $0.063/query training cost cap on solver spend — set at exactly
> the price of the strongest entry we have not yet passed, so that evolution had
> to win on score at a price already known to be competitive. It came in at
> $0.058/query.
>
> The organizing problem is pool starvation: on hard queries the candidate pool
> never contains the papers a grader would call perfectly relevant, and no
> amount of reranking recovers what was never retrieved. The benchmark's grading
> criteria are hidden at query time — the agent sees only the natural-language
> request — so it infers its own, then watches how many candidates clear the top
> grade and treats a thin count as a retrieval failure rather than a hard query.
>
> Its distinctive channel is title guessing: the planner writes out up to six
> guesses at the exact titles of papers that would answer the request, and looks
> each one up directly. That reaches specifically-recalled papers no phrasing of
> a keyword search surfaces. Retrieval rounds are time-boxed to return whatever
> arrived rather than discard a slow round entirely, so a slow backend costs
> depth instead of the whole round.
>
> It uses two OpenAI models and nothing else, with the more capable one reserved
> for three planning calls; everything that scales with the size of the
> candidate pool runs on the cheaper model.

## Official-result risks (assessed pre-submission)

**1. Internal is not banked.** Transfer has run +0.0025 / −0.0550 / −0.0077 /
+0.0096 / +0.0020 across the five submitted runs. Four of five sit within
±0.010; `v0_0_8`'s −0.0550 is the outlier, and it spans the `ad7faf24` scoring
fix — its internal number was computed pre-fix and its official post-fix, a
basis change this run does not have.

**2. Uncapped official judging.** Internally only the top-K estimate is judged;
officially all 250 submitted papers per query are. This has cut both ways —
it *helped* `cap_0_355_opus5` (+0.0146 semantic, t = 2.03) and `cap_0_355_fable`
(+0.0033, t = 0.51), and *hurt* `cap_0_063` (−0.0131, t = −2.01). This agent
fills all 250 slots on every one of its 194 semantic queries, so it is exposed
either way. Note the tempting-but-unsupported inference: submission depth does
**not** separate the runs that gained from the one that lost — `-010` also
filled all 250. Whatever distinguishes them is grading-pool depth, which has not
been measured here.

**3. `specific_f1` is resampling noise, not signal.** 0.8465 here. The category
carries ~0.058 SD over 38 all-or-nothing queries, so a ±0.05 swing between runs
needs no mechanism to explain it. Do not read a modest official drop as a defect.

**4. Selection confidence is unusually strong.** The winner was measured over
**8** Elo test rounds against the 2–5 typical of prior winners, and held the
champion slot from iteration 14 to the end. Where prior submissions had to argue
that the Elo ladder had picked correctly against a higher-train-mean runner-up,
this one does not: `iter14_title_channel` leads the pool by 74 Elo.

**5. Wall-clock headroom is thin, and the failure is silent.** Max sample 1590s
against the 1800s internal cap, 32 of 267 samples at ≥1500s, median 1318s — all
at `max_workers=8`. Because this lineage's expansion rounds are time-boxed to
return *partial* results rather than fail, wall-clock pressure costs score
quietly rather than producing errors, so "zero timeouts" is not the same as
"no time pressure". The official run uses `--max-samples 6`, less contention
than training saw. `-011` measured that exact transition: four internal
timeouts at 8-way became a 1042s max / 651s median officially. That makes the
internal score a conservative floor for this agent specifically — and the
official per-sample distribution, already recorded in the `.eval` log, is the
check.

## Official result

*Pending — filled in after the official run. Board position, the frontier table
with this entry inserted, the Internal/Official/Delta table, the sixth transfer
calibration point, the wall-clock comparison against risk 5, and the sixth
cost-calibration point all land here.*

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_fable --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_fable             # full
```

Push the commit **before** the full run — `astabench eval` hard-fails if the
commit is not on the remote, since it stamps the SHA into `eval_spec.revision`.
The script's own preflight refuses to run against a dirty working tree.

Projected spend **$213–226**: judge $0.0041–0.0043/paper × 48,500 papers (194
semantic queries × 250) plus ~$15.6 agent. Judge-dominated — the agent side is a
quarter of `cap_0_355_fable`'s because of the gate. The script's printed
estimate (~$209) uses a hardcoded $0.0040/paper and has under-predicted both
prior full-250 runs; budget from the measured band.

Then upload `submissions/asta_paper_finder/v0_0_9_cap_0_063_fable.tar.gz` via
the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [ ] Official eval run
- [ ] Tarball uploaded
- [ ] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`
