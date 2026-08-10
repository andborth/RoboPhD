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

The competitor-priced gate is the standing construction on this task — every
submission since `v0_0_7` has set its free zone at, or just under, the price of
a named entry on the board. This is the **second run to put the gate at Asta
Paper Finder's \$0.063 exactly** (`v0_0_7`'s \$0.06 aimed at the same region
without matching the price). It is not a repeat of the first: the gate, the
guard, the judges, the budget and the harness are all held fixed, and **the
evolution model is the one deliberate change**.

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

Going in, the frontier held five slots, four of them ours. The board with this
run inserted is under
[Board position](#board-position-a-new-frontier-slot-target-not-cleared).

**It came in at 0.390 @ \$0.058** — a new frontier slot, cheaper than Asta's
entry and scoring below it. **The target is not cleared:** 0.0075 short, which
is well inside this run's own 0.0178 stderr. Full accounting, at full
precision, under [Official result](#official-result).

## What this run is for: completing the 2×2

`-013` is the missing **fable-5 × \$0.063** cell. All four runs sit on one frozen
stack — zero commits to `examples/asta_paper_finder/` between `-012` and this
run.

Scored on the leaderboard's own basis, so all four cells are comparable:

| official mean F1 | **$0.063 gate** | **$0.355 gate** |
| --- | --- | --- |
| **opus-5** | `-010` 0.376 | `-011` 0.432 |
| **fable-5** | **`-013` 0.390** | `-012` 0.440 |
| **delta to fable-5** | **+0.0133** | **+0.0085** |

Cells are the board's three-decimal basis; the delta row is the full-precision
difference, so it will not subtract exactly from the rounded cells above
(0.38954 − 0.37620 = 0.01334). The deltas keep the extra digit because they are
compared against the runs' ~0.018 stderrs below.

**The 2×2 closes with a consistent sign: fable-5 leads at both gates**, and by
more at the cheap one (+0.0133 against +0.0085). **n = 1 per cell**, so that
consistency is the whole result — not an effect size, and not evidence the gap
widens as the gate tightens. Both deltas sit inside the runs' own ~0.018
stderrs; what the design buys is that two independent cells point the same way.

## The category shape does NOT replicate at the cheap gate

Official-against-official versus `-010`, on all **267** IDs (both runs have
every query; `-010`'s timed-out sample exists only in its *internal*
per-problem file, recorded there as `sample_id: "None"`):

| category | n | opus-5 (`-010`) | **fable-5 (`-013`)** | delta | score points (n × delta) |
| --- | --- | --- | --- | --- | --- |
| semantic | 194 | 0.3110 | **0.3184** | **+0.0074** | **+1.44** |
| specific | 38 | 0.8114 | **0.8289** | +0.0175 | **+0.67** |
| metadata | 35 | 0.2653 | **0.3069** | +0.0416 | **+1.46** |
| **all** | **267** | 0.37620 | **0.38954** | **+0.01334** | **+3.56** |

+3.56 / 267 = **+0.01334**. **All three categories move the same way**, semantic
included.

For contrast, the same comparison at \$0.355 (metadata **+0.1202**, specific
**+0.0430**, semantic **−0.0184**) *is* a three-way trade. So the honest
statement is: **fable-5 beats opus-5 at both gates, but it does not do it the
same way at each.** One shape, two gates, is not a pattern.

**The internal basis points the other way, and that is the interesting part.**
Semantic reads −0.0101 there against +0.0074 officially, and the overall delta
is an order of magnitude smaller (+0.0012 vs +0.01334). Not a scoring error:
the capped→uncapped judging transition moves the two agents in *opposite*
directions. `-010` **lost** 0.0131 of semantic going uncapped; `-013`
**gained** 0.0042. Nine-tenths of the official gap between these two agents is
that divergence — a difference the capped basis cannot see at all.

**The rule that follows:** comparing two agents on internal per-problem files
measures the judging transition as much as it measures the agents, whenever
their grading depth differs. The \$0.355 cell's trade is computed from official
figures on both arms and stands; a cheap-gate trade computed from internal
files does not.

> **Historic note.** Until the official numbers landed, this section claimed the
> opposite — that the three-way trade replicated at both gates. That reading was
> internal-basis only, and is exactly the artifact the rule above describes.

## Cap-hugging replicates too

| gate | opus-5 | fable-5 |
| --- | --- | --- |
| $0.063 | $0.0533 (85%) | **$0.0583 (93%)** |
| $0.355 | $0.2506 (71%) | $0.2786 (78%) |

Four observations, same direction: **fable-5 spends closer to the line at both
price points, and the gap widens as the cap tightens.** At \$0.063 it leaves only
7% of the free zone unused.

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

## How the agent budgets its 29 minutes

The siblings have nothing like this — `v0_0_8`'s README records that its agent
has "no deadline constants at all". This lineage evolved a **graded degradation
ladder**, and it was never handed a number in seconds to build it from:
`background.md` says only that a query taking "more than **29 minutes** of
wall-clock" scores 0, rendered by the harness from `(EVAL_TIMEOUT - 30) // 60`.

**A self-imposed budget under that ceiling.** `SOLVE_BUDGET = 1560.0`, reasoned
in the source from "<30 s pre-solve overhead, leaving >200 s of margin for the
final evidence build + write" and calibrated on "iter13's worst wall clock was
1518 s".

**Gates ordered by what each stage is worth.** As `_remaining()` falls the agent
sheds the tail and protects retrieval: weak-criterion patch skipped below 330s,
judge-mimic rerank needs 90s, individual tool and LLM calls short-circuit at 20s
and 15s. Upstream, the thin-pool round needs 420s and expansion must leave
`EXP_RESERVE = 720` — itself derived from measured downstream need, "full-depth
mining ~300 s + rerank ~30 s + weak-criterion patch ~320 s". Mining depth
degrades in steps rather than switching off:

```python
mine_depth = (MINE_DEPTH if rem > 700 else       # 200 papers
              120 if rem > 540 else
              72 if rem > 360 else 0)
```

**Three checkpoints, so there is always something to submit** — after coarse
grading, after expansion, after mining+rerank — with a minimal `_fallback()` if
even the first is missed. Expansion feeds this: a predecessor wrapped each round
in `asyncio.wait_for` and a timeout discarded the *whole* round (one query spent
420s and kept nothing, scoring 0.272 against 0.659 completed). Rounds are now
bounded by a cutoff checked before each chunk fetch, so a slow backend costs
depth, not the round.

**The ordering is measurably right.** With the pacing disabled by the official
harness, 81 samples ran to their ceiling and still scored **+0.0100 against
internal** — the axe falls after the last stage that changes *which* papers are
submitted or in *what* order. Missing the first checkpoint is the failure that
hurts: 22 samples on the degraded run dropped to `_fallback()` at −0.1302 each.
See [Wall clock](#wall-clock--risk-5-checked).

Source caveat: the docstring says "budget 1500 s of the ~1770 s kill line", the
constant is 1560, and a trailing comment names 1740 — none of them the real
1770. The self-description drifted from the constants, as with the stale
docstring header noted above.

## Still weak: metadata

**12 of 35 queries score zero.** A real improvement on `-010`'s 18 at the same
gate, but well short of `-012`'s 5:

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

Each was resolved by the run; the figures live once, in
[Official result](#official-result).

1. **Internal is not banked.** Transfer had run within ±0.010 on four of five
   prior submissions. → *sixth calibration point.*
2. **Uncapped official judging.** Internal judges only the top-K estimate,
   official all 250 — and this agent fills all 250 on every semantic query, so
   it was exposed either way. → *semantic internal→official.*
3. **`specific_f1` is resampling noise.** Its SD over 38 all-or-nothing queries
   swamps any plausible move, so a drop is not a defect. → *−0.0176, a third of
   that SD.*
4. **Selection confidence is unusually strong.** 8 Elo test rounds against the
   2–5 typical of prior winners, champion from iteration 14 to the end, ahead by
   **74 Elo** — so unlike prior submissions this one never had to argue the
   ladder picked correctly. *(The one risk the run cannot settle.)*
5. **Wall-clock headroom is thin, and the failure is silent.** The stages
   degrade rather than error (see [How the agent budgets its 29
   minutes](#how-the-agent-budgets-its-29-minutes)), so "zero timeouts" is not
   "no time pressure". → *checked; not the answer this risk expected.*

## Official result

A first official run on 2026-08-06 was discarded: its agent cost landed 17.4%
below what training and internal eval had both measured, episodically — the
opening two hours matched internal, the middle ~10 did not. It is preserved
with its analysis in
[`README.degraded_run_2026-08-06.md`](README.degraded_run_2026-08-06.md), and
its log and tarball carry a `.degraded_20260806` suffix under
`submissions/asta_paper_finder/`. Its figures appear below only in the
[comparison table](#against-the-degraded-first-attempt).

**The re-run (2026-08-07 17:03 → 08-08 05:43, 12h40m) scored 0.38954 @
\$0.058059**, stderr 0.01782, cost stderr 0.001976. 267/267 samples, zero
errors, zero limit hits, zero seed-tier fallbacks, zero empty submissions.

### Board position: a new frontier slot, target not cleared

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | RoboPhD (v0_0_9 @ cap 0.063, opus5) | 0.376 | 0.052 | Standard |
| 3 | **RoboPhD (this entry)** | **0.390** | **0.058** | **Standard** |
| 4 | **Asta Paper Finder** | **0.397** | **0.063** | Custom interface |
| 5 | RoboPhD (v0_0_9 @ cap 0.355, opus5) | 0.432 | 0.251 | Standard |
| 6 | RoboPhD (v0_0_9 @ cap 0.355, fable) | 0.440 | 0.279 | Standard |

`v0_0_7` (0.375 @ \$0.053) is on the board but off the frontier, displaced by
our own `cap_0_063`.

Six frontier slots, **five ours**. This entry neither dominates nor is
dominated: it undercuts Asta by \$0.005 and trails it by 0.0075 on score, so
both stay on the curve. It does not displace our own \$0.052 point either —
that one is cheaper.

**The target was Asta's 0.397 and we did not reach it.** 0.0075 short against a
0.0178 stderr, so the two are not separated by this measurement; but "not
statistically distinguishable" is not "cleared", and the frontier records the
point estimate. Third slot on the board is what this run bought.

### Internal → official

| Metric | Internal (capped) | Official (uncapped) | Delta |
| --- | --- | --- | --- |
| Mean F1 (267) | 0.38738 | **0.38954** | **+0.00216** |
| `semantic_f1` (194) | 0.31413 | 0.31838 | +0.00425 |
| `specific_f1` (38) | 0.8465 | 0.8289 | −0.0176 |
| `metadata_f1` (35) | 0.2950 | 0.3069 | +0.0119 |
| Agent \$/query | 0.05828 | 0.058059 | −0.4% |

Internal basis: stock GPT-4o judge, canonical ordering, capped to the top-K
estimate. It cost \$73.23 in judging (capped + cached, against the official
run's uncapped \$201.92) and recorded **zero timeouts** at `max_workers=8`.

**Sixth transfer calibration point: +0.0022.** The series is now +0.0025 /
−0.0550 / −0.0077 / +0.0096 / +0.0020 / **+0.0022** — five of six inside
±0.010, with `v0_0_8`'s −0.0550 still the lone outlier and still explained by
the `ad7faf24` basis change.

`specific_f1` moved −0.0176, about a third of the category's ~0.058 SD over 38
all-or-nothing queries. That is risk 3 behaving exactly as written; no
mechanism needed.

**Uncapped judging helped a cheap-gate agent for the first time.** Semantic
+0.0042 (t = 0.66, n = 194) here, against −0.0131 (t = −2.01) for `-010` at the
same gate. The prior reading — expensive gates gain, cheap gates lose — was
2-for-2 and is now 3-for-4. Both of this run's and `-012`'s effects are
indistinguishable from zero (t = 0.66, t = 0.51); only `-011` (+0.0146) and
`-010` (−0.0131) are individually meaningful, and they point opposite ways. The
gate-dependence story should be treated as unsupported rather than confirmed.

### Cost

| | value |
| --- | --- |
| Agent | \$15.50 (\$0.058059/query) |
| Judge | \$201.92 (**\$0.00416/paper**, 194 × 250) |
| **Total** | **\$217.42** |

**Sixth cost calibration point: \$0.00416/paper**, against \$0.0040 / \$0.0030 /
\$0.00407 / \$0.00426 / \$0.00420. Total landed inside the \$213–226 pre-run
band, the second projection running that did not miss low.

### Against the degraded first attempt

| | score | agent \$/query |
| --- | --- | --- |
| degraded (2026-08-06) | 0.36382 | 0.048119 |
| **this run** | **0.38954** | **0.058059** |
| delta | **+0.02572** | **+20.7%** |

Paired per-sample, 111 W / 83 L: the first attempt cost roughly **0.026 of
score**. Agent spend is the axis the two differ on most — **99.6% of internal
here against 82.6% there** — and it is what flagged the problem before any
score comparison was possible.

### Wall clock — risk 5, checked

| | max | median | ≥1500s |
| --- | --- | --- | --- |
| internal (`max_workers=8`) | 1590s | 1318s | 32 / 267 |
| **official (`--max-samples 6`)** | **1534s** | **1256s** | **81 / 267** |

Risk 5 predicted the 6-way official run would see *less* contention than 8-way
training, and the median agrees (1318 → 1256s). **But samples at the ceiling
went 32 → 81**, which the risk did not anticipate, and the cause is not
contention at all.

Investigating it during this run turned up a defect in the submission wrapper,
not the agent: `astabench eval` runs its samples as asyncio tasks in **one
process**, while RoboPhD training and internal test run **one sample per
subprocess**. This agent keeps its deadline clock at module scope
(`_START`/`_DEADLINE`, `agent_inner.py:178`), so under the official harness
every starting sample re-stamped the shared clock, `_remaining()` never counted
down, and every trim gate the agent evolved — mining depth, rerank skip, patch
skip — was dead. 81 samples ran to their own `wait_for` ceiling instead of
self-pacing inside it.

**It cost this run nothing measurable.** Paired against internal, the 81
guillotined samples scored **+0.0100**, because the cut lands after the last
stage that changes which papers are submitted or in what order. It was
expensive on the degraded run, where 22 samples never reached their first
checkpoint and fell to the agent's minimal fallback at −0.1302 each — over half
that run's damage. The wrapper now loads a fresh module per sample
(`WRAPPER_TEMPLATE`, both submit scripts), restoring the regime the agent was
evolved under. **That change is not in this submission's artifact** — this
entry is exactly the agent and wrapper that produced 0.38954.

So risk 5's conclusion held (internal was a conservative floor, +0.0022) but
its reasoning did not: the headroom came from the guillotine landing on
low-value work, not from reduced contention.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_fable --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_063_fable             # full
```

Push the commit **before** the full run — `astabench eval` hard-fails if the
commit is not on the remote, since it stamps the SHA into `eval_spec.revision`.
The script's own preflight refuses to run against a dirty working tree.

Projected spend **\$213–226**: judge \$0.0041–0.0043/paper × 48,500 papers (194
semantic queries × 250) plus ~\$15.6 agent. Judge-dominated — the agent side is a
quarter of `cap_0_355_fable`'s because of the gate. The script's printed
estimate (~\$209) uses a hardcoded \$0.0040/paper and has under-predicted both
prior full-250 runs; budget from the measured band.

Then upload `submissions/asta_paper_finder/v0_0_9_cap_0_063_fable.tar.gz` via
the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [x] Official eval run — 2026-08-08, 0.38954 @ \$0.058059, 267/267, zero errors
- [x] Tarball built — `submissions/asta_paper_finder/v0_0_9_cap_0_063_fable.tar.gz`
      (229 MB, 2026-08-08 05:44; the `.degraded_20260806` tarball is the
      superseded first attempt and must not be uploaded)
- [ ] Tarball uploaded to HF Spaces
- [x] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`
