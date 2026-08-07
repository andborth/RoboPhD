# ARCHIVED — the 2026-08-06 official run, superseded by a re-run

**This is not the submission record.** It is the README as it stood after the
first official run of `v0_0_9_cap_0_063_fable`, kept because that run is
informative about how these evals behave and because the analysis below is
cited elsewhere.

**Why it was superseded.** The run scored 0.36382 @ \$0.048119, but its agent
cost came in **17.4% below** what training (\$0.058) and internal eval
(\$0.0583) had both measured — 4.3x the largest internal-to-official cost
deviation across the five prior submissions, which span −4.0% to +1.6%. Paired
per-query analysis (Spearman(Δcost, Δscore) = +0.183, n=194, p≈0.01) tied the
cost shortfall to the score shortfall, and the effect was **episodic**:

| hour of run | mean Δcost | mean Δscore | share in worst quartile |
| --- | --- | --- | --- |
| t+0–2h | −0.0003 | **+0.0075** | 6% |
| t+8–10h | −0.0238 | −0.0217 | 48% |
| t+12–14h | −0.0026 | −0.0156 | 0% |

The first two hours reproduced the internal measurement; the middle ~10 hours
did not; the last two recovered. Roughly 85% of the run executed degraded. A
concurrent DS-1000 run (`asta_ds1000_20260806_230212`) covered t+5.3h–t+12.6h
and showed 2.6x the spend reduction inside its window, but the degradation
began ~3h before it, so it is a partial explanation at best.

The frontier claim also depended on the anomaly: at \$0.048 the entry is not
dominated and holds a slot; at the twice-measured \$0.058 it is strictly worse
than our own `cap_0_063_opus5` (0.376 @ \$0.052) and earns nothing.

**Artifacts** (gitignored, under `submissions/asta_paper_finder/`):
`v0_0_9_cap_0_063_fable/logs/full_test.degraded_20260806/` and
`v0_0_9_cap_0_063_fable.degraded_20260806.tar.gz`.

---

# v0_0_9_cap_0_063_fable — PaperFindingBench (AstaBench) submission snapshot

Sixth RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier).

> **The submitted agent is [`agents/iter14_title_channel/agent.py`](agents/iter14_title_channel/agent.py)**
> — iteration-14 winner of run `robophd-asta_paper_finder-013`, 2,068 lines,
> two OpenAI model handles. Everything below describes that file.

## The gate is Asta's price; the variable is the evolution model

The training free zone is set at **\$0.063/query** — exactly the price of **Asta
Paper Finder's 0.397 @ \$0.063**, the one non-RoboPhD entry still holding a
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

Read down the last three rows and the design is visible: the \$0.355 gate has
already been run under both evolution models, and this entry supplies the
missing fable-5 arm at \$0.063. 

The board as it stands, before this entry:

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | RoboPhD (v0_0_9 @ cap 0.063, opus5) | 0.376 | 0.052 | Standard |
| 3 | **Asta Paper Finder** | **0.397** | **0.063** | Custom interface |
| 4 | RoboPhD (v0_0_9 @ cap 0.355, opus5) | 0.432 | 0.251 | Standard |
| 5 | RoboPhD (v0_0_9 @ cap 0.355, fable) | 0.440 | 0.279 | Standard |

`v0_0_7` (0.375 @ \$0.053) is listed but off the frontier, displaced by our own
`cap_0_063`.

**Outcome: 0.364 @ \$0.048 — a new frontier slot, but the target stands.**
Asta's 0.397 needed clearing and was not cleared; this entry slots in below our
own `cap_0_063` at a slightly lower price. Six frontier slots, five ours. Full
numbers under [Official result](#official-result-2026-08-07).

## What this run is for: completing the 2×2

`-013` is the missing **fable-5 × \$0.063** cell. All four runs sit on one frozen
stack — zero commits to `examples/asta_paper_finder/` between `-012` and this
run.

Scored on the leaderboard's own basis, so all four cells are comparable:

| official mean F1 | **$0.063 gate** | **$0.355 gate** |
| --- | --- | --- |
| **opus-5** | `-010` **0.37620** | `-011` 0.43177 |
| **fable-5** | `-013` 0.36382 | `-012` **0.44025** |
| **delta to fable-5** | **−0.0124** | **+0.0085** |

**The signs are opposite, and that is the run's main result.** On the \$0.355
cell alone, fable-5 looked like the better evolution model by +0.0085. Filling
the cheap corner reverses it: at \$0.063 opus-5 wins by 0.0124, half again the
margin fable-5 held at the expensive gate.

So the honest reading of the completed design is **not** "fable-5 builds better
agents". It is that the evolution model interacts with the cost gate, and a
single cell cannot tell you which way. Had we stopped at the \$0.355 A/B — which
was the cleanest controlled comparison in the campaign, and which we published
as such — we would have banked a conclusion this run overturns.

**n = 1 per cell** still, so the interaction itself is one observation, not an
established effect. What is established is that the +0.0085 was not
generalisable.

## Two of the three categories do replicate

Official against official at the \$0.063 gate:

| category | n | opus-5 (`-010`) | **fable-5 (`-013`)** | delta | score points (n × delta) |
| --- | --- | --- | --- | --- | --- |
| semantic | 194 | **0.3110** | 0.2870 | −0.0240 | **−4.66** |
| specific | 38 | **0.8114** | 0.7850 | −0.0264 | **−1.00** |
| metadata | 35 | 0.2653 | **0.3300** | +0.0647 | **+2.26** |
| **all** | **267** | **0.37620** | 0.36382 | −0.0124 | **−3.39** |

−3.39 / 267 = **−0.0127**, the difference between the two (−0.0124 on the
reported means; the gap is per-type rounding).

Set beside the \$0.355 cell, two of three category signs hold:

| category | $0.063 | $0.355 | replicates? |
| --- | --- | --- | --- |
| metadata | **+0.0647** | **+0.1202** | ✅ fable-5 up at both |
| semantic | **−0.0240** | **−0.0184** | ✅ fable-5 down at both |
| specific | −0.0264 | +0.0430 | ❌ flips |

**fable-5 trades semantic for metadata at both price points** — that much
survives the completed design. Specific flips sign, but that is the category
carrying ~0.058 SD over 38 all-or-nothing queries, so both figures are inside
half a standard deviation and neither is evidence of anything.

What changes with the gate is whether the trade pays. At \$0.355 the metadata
gain is large enough (+0.1202 over 35 queries) to outrun the semantic loss over
194. At \$0.063 it is not: metadata contributes +2.26 score points against
semantic's −4.66. **Semantic carries 73% of the queries, so it decides the
headline whenever the metadata gain is merely good rather than enormous.**

## Cap-hugging replicates too

Cap utilisation is a **training** behaviour — it describes what evolution
optimised toward — so it is measured on each winner's mean training spend
against its own free zone:

| gate | opus-5 | **fable-5** |
| --- | --- | --- |
| $0.063 | $0.054 (**86%**) | **$0.058 (92%)** |
| $0.355 | $0.270 (**76%**) | **$0.303 (85%)** |

Four observations, same direction: **fable-5 evolves agents that spend closer to
the line at both price points**, by 6–9 points of the free zone. This is the one
claim in this document that the completed 2×2 leaves fully intact.

A caution on which number to quote. This agent's cost differs across all three
bases — train \$0.058, internal test \$0.0583, **official \$0.0481 (76% of cap)** —
and only the first says anything about evolution's behaviour, because the free
zone is a *training* signal and the training mean is what evolution optimised
against. Reading cap utilisation off the official figure would invert the
finding at this gate, and the reason that figure came in low is itself
[not established](#two-things-this-run-does-not-explain).

## Naming and conventions

**The version is `v0_0_9` again.** The patch number tracks the RoboPhD code
base, not the run, and continues the cross-benchmark sequence shared with
DS-1000. This run executed 2026-08-04 on a stack byte-identical to `-012`'s.
Three commits landed on 2026-08-06 — `rejudge_test.py --uncapped` /
`--from-eval-log`, and `evaluator.PERSIST_FULL_EVIDENCE` — but all are
test-time or offline tooling that touch neither the solver nor training, so the
stack is unchanged.

`cap_0_063` names the training free zone (\$0.063/query); `_fable` the evolution
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
  \$0.058, **8** Elo test rounds
- 12 distinct winners across 21 iterations
- **Reachability guard fired at iteration 20**, displacing `use_your_judgment`,
  so the last two iterations ran greedy. That is the guard working as designed:
  once no newly evolved agent can still reach the top of the ladder within the
  remaining budget, continued exploration is spend without upside. Fourth live
  firing.
- Clean run: 0 exceptions, 0 zero-accuracy cases, 0 clone detections, and
  **zero test-set timeouts**
- Evolution cost: \$27.30 eval + \$138.18 evolution + \$9.85 training judge =
  **\$175.33**

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
affordable for a \$0.355 agent. At \$0.063 this run gets partway there by other
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
> model, under a \$0.063/query training cost cap on solver spend — set at exactly
> the price of the strongest entry we have not yet passed, so that evolution had
> to win on score at a price already known to be competitive. It came in at
> \$0.058/query.
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

## Official result (2026-08-07)

**As displayed on the board: 0.364 @ \$0.048.** 267/267 samples, **zero errors**,
**14h33m**. Judge \$187.38 + agent \$12.85 = **\$200.23**. Full precision below,
since the transfer is larger than the board's rounding step and the cost moved
in an unusual direction.

| Metric | Internal | **Official** | Delta |
| --- | --- | --- | --- |
| Mean F1 | 0.38738 | **0.36382** ± 0.01863 | **−0.0236** |
| `semantic_f1` (194) | 0.3141 | 0.2870 | −0.0271 |
| `specific_f1` (38) | 0.8465 | 0.7850 | −0.0615 |
| `metadata_f1` (35) | 0.2950 | **0.3300** | **+0.0350** |
| Agent cost | $0.0583 | **$0.048119** ± 0.00178 | **−$0.0102** |

Sixth calibration point, and the second-worst: **+0.0025 / −0.0550 / −0.0077 /
+0.0096 / +0.0020 / −0.0236**. Note what a transfer number actually spans — a
fresh agent execution *and* a judging-basis change. Those are separated
[below](#uncapped-judging-is-real-but-small--and-it-explains-only-half-the-campaigns-cases),
and for this run the agent term is the larger one. The four middling ones still sit within ±0.010;
this run and `v0_0_8` are the two that do not.

`specific_f1` fell 0.0615, which is about one standard deviation on a category
of 38 all-or-nothing queries — [risk 3](#official-result-risks-assessed-pre-submission)
called this in advance and it needs no mechanism. `metadata_f1` **rose** 0.0350,
its best official figure at this gate. The drop is semantic: −0.0271 over 194
queries is **−5.26 of the −6.37 total score points**, 83% of it.

### Where it lands: a new frontier slot, but not the target

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | **this entry** | **0.364** | **0.048** | **Standard** |
| 3 | RoboPhD (v0_0_9 @ cap 0.063, opus5) | 0.376 | 0.052 | Standard |
| 4 | Asta Paper Finder | 0.397 | 0.063 | Custom interface |
| 5 | RoboPhD (v0_0_9 @ cap 0.355, opus5) | 0.432 | 0.251 | Standard |
| 6 | RoboPhD (v0_0_9 @ cap 0.355, fable) | 0.440 | 0.279 | Standard |

Nothing on the board scores ≥ 0.364 at ≤ \$0.048, so this is **not dominated**
and takes a sixth slot — five of six now ours. It dominates nothing either: it
is strictly below our own `cap_0_063` on score while being \$0.004 cheaper.

**The target was missed.** Asta Paper Finder's 0.397 @ \$0.063 needed 0.397 from
this entry and it returned 0.364. The gate has now been aimed at that entry
twice and cleared neither time. Asta keeps its slot.

### The 14h33m is the agent, not the venue

The run's length invites a latency story. It does not survive measurement.

**This agent is the slowest in the campaign by construction.** Comparing
semantic samples only, on each run's own internal evaluation:

| run | internal median | official median | official ÷ internal |
| --- | --- | --- | --- |
| `-012` ($0.355, fable) | 814s | 709s | **0.87×** |
| **`-013` (this entry)** | **1378s** | **1529s** | **1.11×** |

`-013` is **1.69× slower than `-012` before either agent ever ran officially**.
The title-guess channel, the extra reformulation probes and the deeper expansion
all cost round-trips. 267 samples at ~1500s over 6 concurrent slots is 14 hours
whatever the backend does.

**The official-side slowdown is real but small: 11%.** `-012` ran 13% *faster*
officially, so `-013` did lose something at the venue that `-012` gained — but
an 11% shift, not a collapse.

Three checks constrain what it can be:

- **No drift across the run.** Median agent-phase span by start-time quartile:
  1238s / 1305s / 1291s / 1229s, Pearson *r* = **+0.063** against elapsed hours.
  A degrading backend would show a trend; this is flat from first sample to last.
- **The task is retrieval-bound, and always has been.** Merging every model
  call's interval per sample, the share of agent wall time *not* inside any LLM
  call is **98.2% / 95.0% / 93.9% / 97.6%** for `-010` / `-011` / `-012` /
  `-013`. Model latency is a rounding error in all four; agent speed is tool
  round-trips. So a slow agent here means *more* round-trips, which is a design
  property, not a venue property.
- **Model-call latency was normal** (median 2.3s against 1.2–1.7s elsewhere) —
  OpenAI was not the bottleneck either.

**A correction to an earlier draft of this section.** It reported the official
run as 16% slower with "94% of queries above 1500s against 17%". Both figures
were wrong: the internal median was taken over all 267 samples (including the
fast `specific`/`metadata` ones) and compared against an official median over
the 194 `semantic` ones. Like-for-like the shift is 11%, and the ≥1500s counts
(29 → 183) exaggerate it further because the median sits just below the
threshold internally and just above it officially — an 11% shift drags most of
the distribution across the line.

**So risk 5 is not the explanation for the score.** Wall-clock pressure moved
by 11%, in a run whose agent was always going to be slow. The transfer needs a
different cause, and [risk 2](#official-result-risks-assessed-pre-submission)
supplies it.

### Uncapped judging is real but small — and it explains only half the campaign's cases

Semantic is the only category a judging basis can touch, and it carries **83% of
the score points lost** here (−0.0271 over 194 queries). The campaign has
attributed movements like that to the capped→uncapped switch. **That attribution
can be tested for free, and it only partly holds.**

`adjusted_f1 = harmonic(rank, recall@estimate)` — verified on all 194 samples.
`recall@estimate` is *already* K-windowed, so judging depth cannot move it;
capping can only change the nDCG `rank` term. The official run judged all 250
papers, so its own verdicts are enough to recompute what the **capped** score
would have been: truncate the ordered grade list to K and re-score. No LLM cost.

Validated before use: parsing the official per-paper verdicts back out and
recomputing `rank` reproduces the official value **exactly** on 151 of 194
samples. The mismatches are precisely the samples with skipped documents, where
a missing verdict shifts the list. Only verified samples are used below.

| run | gate | cap effect (capped − uncapped) | published internal → official | cap explains |
| --- | --- | --- | --- | --- |
| `-010` | $0.063 | −0.0008 | **−0.0117** | ✗ wrong direction |
| **`-013`** | **$0.063** | **−0.0023** | **−0.0271** | ✗ wrong direction |
| `-011` | $0.355 | −0.0071 | **+0.0130** | ✓ ~55% |
| `-012` | $0.355 | −0.0054 | **+0.0033** | ✓ all of it |

Two things follow, and they pull in opposite directions.

**The mechanism is real and gate-dependent, as the campaign supposed.** Capping
always *lowers* the score, and it lowers it 3–4× more for the deep-grading
\$0.355 agents (0.005–0.007) than for the cheap ones (0.001–0.002). Agents that
grade deeper do gain more from having their tail judged.

**But it is far too small to carry the observed transfers, and at this gate it
has the wrong sign.** The cheap-gate runs moved −0.0117 and −0.0271 while the
cap pushes *positive*. Whatever drove those, it worked against the cap and was
an order of magnitude larger. The residual is the agent itself: an official run
is a fresh stochastic execution, and on this evidence **run-to-run variance
dominates the scoring basis** at the cheap gate.

So risk 2 named a real exposure and correctly predicted the sign at \$0.355 — but
it is not the explanation for this run's −0.0236. An earlier version of this
section claimed it was, on a "2-for-2 in each direction" pattern that conflated
the judging change with a fresh agent run.

### Two things this run does not explain

**Why agent spend fell 17%** (\$0.0583 → \$0.0481). Prior runs' internal-to-
official cost agreement was within ±\$0.005; this is double the widest previous
move. The internal evaluator records only dollars, not tokens, so there is no
way here to separate "did less work" from "billed on a different price map".
Persisting per-model token counts on internal test evals would close this, and
is cheap.

**Judge-side dropouts are ruled out** as a factor: 51 documents skipped for
missing judgements across ~48,500 (**0.11%**), against `-011`'s recorded 0.13%.
Normal rate, no effect on the comparison.

Judge-side dropouts were normal and are not a factor: 51 documents skipped for
missing judgements across ~48,500 (**0.11%**), against `-011`'s recorded 0.13%.

### Cost calibration (sixth measured point)

Judge came in at **\$0.00386/paper** — \$187.38 over 194 semantic queries × 250
papers — against \$0.0040 / \$0.0030 / \$0.00407 / \$0.00426 / \$0.00420 for the five
prior runs. **The lowest of any full-250 agent.** Judge input tokens were 42.3M here
against `-012`'s 45.5M for the same paper count, so this agent submits ~7% less
evidence text per paper — a property of what it writes, not of what the venue
charged.

Total \$200.23 against a **\$213–226** pre-run estimate — **the first projection
this campaign to miss high**. The method (measured \$/paper band × 48,500 plus
the agent's own internal cost) is sound in shape; both of its inputs simply came
in low, the judge rate because this agent writes shorter evidence and the agent
term for reasons not established (see above).

Recomputing prior judge bills from logged tokens at gpt-4o list rates reproduces
`$203.61` and `$197.32` exactly, so the \$187.38 is on the same basis as the
series.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_fable --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_fable             # full
```

Push the commit **before** the full run — `astabench eval` hard-fails if the
commit is not on the remote, since it stamps the SHA into `eval_spec.revision`.
The script's own preflight refuses to run against a dirty working tree.

Measured spend: **\$187.38 judge + \$12.85 agent = \$200.23**, at \$0.00386/paper
over 194 semantic queries × 250 papers, in **14h33m** at `--max-samples 6`. That
came in *under* the \$213–226 projection — the first time this campaign's
estimate has missed high; see
[Cost calibration](#cost-calibration-sixth-measured-point).

Then upload `submissions/asta_paper_finder/v0_0_9_cap_0_063_fable.tar.gz` via
the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [x] Official eval run (2026-08-07: 0.36382 @ \$0.048119/query, \$200.23 spend, 14h33m, 267/267 no errors)
- [ ] Tarball uploaded
- [ ] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`
