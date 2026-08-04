# v0_0_9_cap_0_355_fable — PaperFindingBench (AstaBench) submission snapshot

Fifth RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier).

> **The submitted agent is [`agents/iter18_cocite_largegold_v1/agent.py`](agents/iter18_cocite_largegold_v1/agent.py)**
> — iteration-18 winner of run `robophd-asta_paper_finder-012`, 2,781 lines,
> two OpenAI model handles. Everything below describes that file.

## It takes the top of the board

The previous submission, `v0_0_9_cap_0_355_opus5`, came within 0.001 of Asta
Paper Finder's best entry and sat *beside* it on the curve. This one clears it.
On the board's three-decimal basis:

> **Official 0.440 @ $0.279**, against the former leader's **0.433 @ $0.355**.

Higher score **and** 21% cheaper — domination on both axes, not a tie, so Asta's
top entry leaves the curve entirely. At full precision, 0.44025 ± 0.01790 against
0.43270, and $0.278579 ± 0.009930 against $0.3548. Full breakdown in
[Official result](#official-result-2026-08-04) below.

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | RoboPhD (v0_0_9 @ cap 0.063) | 0.376 | 0.052 | Standard |
| 3 | Asta Paper Finder | 0.397 | 0.063 | Custom interface |
| 4 | RoboPhD (v0_0_9 @ cap 0.355, opus5) | 0.432 | 0.251 | Standard |
| 5 | **this entry** | **0.440** | **0.279** | **Standard** |

**Rank 1, and five slots with four ours** — all Standard tier, against Ai2's
Custom interface. They retain a single slot, at 0.397 @ $0.063.

By the leaderboard's rules this **dominates** the incumbent: strictly better on
score and strictly cheaper, so the incumbent is no longer Pareto-optimal.

## The cleanest evolution-model A/B in the campaign

This run is identical to `v0_0_9_cap_0_355_opus5` in every deliberate respect —
same $0.355 gate, same reachability guard, same luna/no-prose training judge,
same stock-4o test judge, same 600 budget, same 14 examples/iteration, same
harness. **Only the evolution model differs**: fable-5 here, opus-5 there.

The comparison uses **official** figures, which are the ones that carry — both
entries have them, and they judge every submitted paper rather than the top-K
slice internal scoring grades. On that basis fable-5 leads by **+0.0085**, and
the margin is a three-way trade rather than a single category:

| category | n | opus-5 | **fable-5** | delta | score points (n × delta) |
| --- | --- | --- | --- | --- | --- |
| metadata | 35 | 0.2281 | **0.3483** | +0.1202 | **+4.21** |
| semantic | 194 | **0.3933** | 0.3749 | −0.0184 | **−3.58** |
| specific | 38 | 0.8158 | **0.8588** | +0.0430 | +1.63 |
| **all** | **267** | 0.43177 | **0.44025** | | **+2.27** |

The benchmark score is a plain mean over all 267 queries, so a category's
influence on the headline is its per-query delta multiplied by how many queries
it has — the last column. Those sum to +2.27 score points, and +2.27 / 267 =
**+0.0085**, the difference between the two entries' totals.

That weighting is why the last column matters more than the delta beside it.
Metadata has by far the largest rate difference at +0.120, but it applies to only
35 queries; semantic's −0.018 looks negligible until multiplied by 194.

Metadata is where fable-5 wins, decisively: **+0.120**, with zeros falling from
16 of 35 to 5. It gets there while spending *less* on that category —
$0.0056/query against opus-5's $0.0136, a **59% reduction** — so the gain is
structural rather than bought, and it comes from the F1-denominator padding rule
described under Architecture below.

Roughly **85% of that is handed back on semantic**, where opus-5's deeper
Sonnet-graded band wins by 0.018 across 194 queries. Specific returns about a
third of it.

## Naming and conventions

**The version is `v0_0_9` again, deliberately.** The patch number tracks the
RoboPhD code base, not the run. One commit landed between this run and
`cap_0_355_opus5` — `41933a6e`, a luna *pricing-table* update affecting
training-judge cost accounting, not the solver — so the stack is unchanged.
`cap_0_355` names the training free zone ($0.355/query); `_fable` the evolution
model, distinguishing it from the opus-5 entry at the same gate. Two entries at
one gate is intended: they are a deliberate A/B and both hold frontier slots.

Prices are quoted to **three decimals**, the leaderboard's precision; two entries
that round to the same figure are a cost tie decided on score.

## Snapshot layout (ds1000 precedent: a curated run snapshot, ~13MB)

- `agents/` — all 19 iteration winners + seed (`agents/iter18_cocite_largegold_v1/agent.py` is the submitted one)
- `iteration_017/`, `iteration_018/` — the winner's breeding data and debut
- `evolution_output/` — `CLAUDE.md` + `iteration_018/` only
- `checkpoint.json`, `final_report.md`, `test_results.json`, `test_results.per_problem.json`

**Redistribution note:** as with the four prior snapshots, the iteration problem
dirs embed VALIDATION-split gold. Verified before commit: the **26** problem IDs
embedded here have **zero** overlap with all 267 held-out test IDs.

## Lineage

- Run: `robophd-asta_paper_finder-012` (`../robophd_runs/robophd/asta_paper_finder_20260802_103017`)
- Engine: RoboPhD Elo (**fable-5** evolution, Deep Focus OFF), budget 600 (613
  used), 22 iterations, 14 examples/iteration
- Winner: `agents/iter18_cocite_largegold_v1` — Elo 1594, train mean 48.147 @
  $0.303, 5 Elo test rounds
- 12 distinct winners across 22 iterations
- Third run with the Elo-reachability guard enabled; it fired at iteration 21
  without suppressing the winner (created at 18). Three live firings now, three
  clean outcomes.
- **Zero test-set timeouts**, against four for the opus-5 run at the same gate.

## Architecture of `iter18_cocite_largegold_v1` (2,781 lines, single `agent.py`)

The organizing discipline is a hard two-tier cost split, stated in the agent's
own docstring:

> `GPT_5_4` is 96% of the bill and the heaviest semantic queries already run
> $0.37-0.39, so every addition above rides `GPT_5_4_MINI` (4% of the bill); no
> new `GPT_5_4` calls anywhere.

Expensive calls are reserved for planning, ranking, the judge-mimic rating and
the specific/metadata plans. Every capacity increase is bought on the cheap
handle.

**Overflow mini-grading.** `SEMANTIC_CAND_CAP` was silently discarding 150–640
retrieved candidates per query *before any grading*, on queries whose K reached
180–222 and whose recall was starving. Candidates 400–800 by retrieval strength
are now graded on mini behind a validated 0.85 discount — retrieval work already
paid for becomes candidate mass at ~$0.03/query, while the discount stops
mini-scale drift from displacing `GPT_5_4`-validated papers. Second-wave cap
170→250 on the same logic, and cite-expand backfills grade-7 seeds when fewer
than four clear grade 8.

**Padding derived from the F1 denominator.** On the exact-match paths:

> With F1 = 2H/(N+G), any candidate whose hit probability exceeds ~F1/2 pays, so
> on broad citing-X queries (heavily-cited seed, no venue filter,
> expected_result_count=many) the submission is now padded to 250 with the
> strict-filter-passing unverified candidates, seed-mentioning ones first.
> Narrow queries (venue-bearing, "one"/"few", light seeds) are untouched —
> padding there would collapse precision.

A threshold rule with a case split, read off the metric definition rather than
tuned from a batch delta. It costs nothing to apply — padding is list
construction, not inference — and it is what drives this run's metadata result.

### Models (2 handles, one provider)

| Handle | Provider | Role |
| --- | --- | --- |
| `gpt-5.4-2026-03-05` | OpenAI | planning, ranking, judge-mimic rating, metadata/specific plans |
| `gpt-5.4-mini` | OpenAI | overflow grading, author disambiguation, cite-expand |

Both priced in the litellm 1.88.1 bundled map. Worth noting against the opus-5
entry's three handles across two providers: the higher score did **not** come
from more model diversity.

### Tool-contract workarounds

Two undocumented Asta MCP behaviours evolution discovered and this agent encodes. Recorded
because they are load-bearing and because their coverage is finite:

- **`get_citations` rejects `tldr`.** Evolution found the citation-expansion
  channel had "been dead in every agent since iteration 6", returning zero citers
  on 25 of 27 measured agent × query pairs — silently, because the per-call
  failure was caught and logged as an ordinary tool error. Fixed with a
  probe-verified narrower field list. Four independent lineages in this campaign
  rediscovered this same defect. **How it was found is worth as much as the fix
  — see [Diagnostics](#the-diagnostics-are-part-of-what-evolved) below.**
- **`venues=` normalises some acronyms but not all.** Worked around with a
  26-entry `_VENUE_ALIASES` table sending both the acronym and the expanded
  official name. **The table is finite and ML/NLP-weighted**, so the metadata fix
  is partial exactly in the venues it omits — `metadata_8` (PADL 2012) recovered
  to 1.000, but `metadata_18` (POPL) only to 0.154 and `metadata_38` (CHI 2001)
  to 0.069.

### The diagnostics are part of what evolved

Whatever a solver prints while answering a query is captured and handed back to
the evolution model before the next round. That makes logging a design decision,
and this lineage treated it as one: the seed agent carried **5 `print()` calls
across 121 lines**, the submitted agent **80 across 2,781**. What grew was a
funnel — a counter at each retrieval stage reporting how many candidates entered
and how many survived.

That instrumentation is what exposed the `get_citations` defect, and the route
matters. The broken channel did not look broken from any single query: it
reported zero new papers, which is an ordinary result when a search genuinely
finds nothing, and three consecutive evolution sessions read the logs that way
and moved on. The session that caught it compared the same counter across all
three competing agents and every query at once — zero in 25 of the 27 pairs
where the channel ran. **An outcome that never varies across agents or inputs is
a defect signature, not a measurement.** Only then did the failure log supply the
cause, because it records the arguments that produced the error alongside it.

The counters pay again as verification. Each round evaluates a small sample of
queries, so a newly added channel may go several rounds without meeting a query
it applies to, and scores cannot confirm the change did anything at all. A
distinctive log line can: two channels here were confirmed live two iterations
after they were written, at no evaluation cost.

The guard that made the failure silent — returning an empty result on any
exception — is also what recorded it. The channel was quiet in its behaviour and
loud in its logs for five iterations, and closing that gap took a change in how
the evidence was read rather than any new evidence.

The residual gap is that this remains a manual sweep performed by the evolution
agent, re-derived from scratch each round. Iteration 12 asked for it to be
automated — a "zero across all queries" audit in the harness's
`error_analysis_report.md`, flagging any log marker reading `-> 0` on ≥80% of
queries — noting it "would have caught the dead channel six iterations ago, and
the class recurs": a cap set below the number of results required struck twice,
a batch-level crash three times.

Requests of this kind are common in `evolution_reflection.md` across the
campaign. In this run alone, one report change was asked for in seven
consecutive sessions, each recomputing the same aggregate by hand first. They go
unanswered because `error_analysis_report.md` is a generic report spanning all
tasks, so a diagnostic specific to one lineage is not something an evolution
session can add. Building such tooling is a Meta-evolution function, and
Meta-evolution was not enabled for this experimental campaign.

The residual gap is that this remains a manual sweep. Iteration 12 asked for a
"zero across all queries" audit in `error_analysis_report.md` — flag any stdout
marker matching `-> 0` on ≥80% of queries — noting it "would have caught the dead
channel six iterations ago, and the class recurs" (caps-below-K twice, the
references batch crash three times).

## Internal results (basis: stock GPT-4o judge, canonical ordering)

| Metric | Value |
| --- | --- |
| Mean F1 (267 queries) | **0.4383** |
| `semantic_f1` (194) | 0.3715 |
| `specific_f1` (38) | 0.8675 |
| `metadata_f1` (35) | **0.3423** — best across twelve runs |
| Agent cost | $0.2780/query — **$0.278** on the board ($74.22 total) |
| Judge cost (internal, capped + cached) | $70.36 |
| Timeouts | **0** |

### Still weak: metadata

0.3423 is our best by a wide margin (previous high 0.2960) but remains far below
the public best of 0.6416, and **5 of 35 queries still score zero**. Those five
are `get_citations` 1000-cap cases on hugely-cited seeds — the citing-paper list
comes back newest-first and truncated, so the gold papers are unreachable. That
subset is structurally unfixable through this route.

## Judge basis

Trained against `openai/gpt-5.6-luna` with the `no-prose` profile; test-scored on
stock `openai/gpt-4o-2024-11-20`, the leaderboard's basis. A held-out judge means
any judge-specific fitting is a headwind on the reported number, not a tailwind.
Internal judging is capped (`cap_judge_to_estimate: true`); official judging is
uncapped — see risks.

## Submission form metadata

| Field | Value |
| --- | --- |
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (Asta MCP corpus tools provided by the task) |
| Models (solver, from logs) | gpt-5.4-2026-03-05, gpt-5.4-mini |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |

Agent description as submitted on the form:

> Evolved by RoboPhD with Claude Fable 5 as the evolution (but not a solver)
> model, under a $0.355/query training cost cap on solver spend — set at exactly
> the price of the most capable entry on the board, so that evolution had to win
> on score at a price already known to be competitive. It came in at
> $0.278/query. The agent runs on a hard two-tier budget: one capable model
> handles planning, ranking and the judge-mimic rating, and every increase in
> capacity after that is bought on a cheap model rather than by spending more on
> the expensive one. That let it grade candidates its predecessors were
> discarding unseen — several hundred retrieved papers per query were being
> dropped before any grading, on exactly the queries where recall was starving.
> On author, venue and year requests it derives how many results to submit from
> the scoring formula itself: because F1 divides by the gold-set size, a
> speculative candidate pays for itself once its chance of being correct exceeds
> half the current score, so broad requests against large answer sets are filled
> out while narrow ones are kept tight.

## Official-result risks (assessed pre-submission)

**1. Internal is not banked.** Transfer has run +0.0025 / −0.0550 / −0.0077 /
+0.0096. Rank 1 rests on ~0.006 of headroom over the leader, which is inside the
range those deltas span.

**2. Uncapped official judging.** Internally only the top-K estimate is judged;
officially all ~249 submitted papers per query are. This cut both ways on prior
runs — it *helped* `cap_0_355_opus5` (+0.013 semantic) and hurt `cap_0_063`
(−0.0117). This agent submits deep, so it is exposed either way.

**3. `specific_f1` is resampling noise, not signal.** 0.868 here. The category
carries ~0.058 SD over 38 all-or-nothing queries, so a ±0.05 swing between runs
needs no mechanism to explain it. Do not read a modest official drop as a defect.

**4. Selection confidence — stronger than the raw means suggest.** The runner-up
`iter17_ambigunion_landmark_v1` posts a higher train *mean* (52.149 vs 48.147),
which invites the reading that the wrong agent was submitted. The pairwise record
says otherwise:

| iteration | iter18 | iter17 | winner |
| --- | --- | --- | --- |
| 18 | 41.532 | 40.142 | iter18 |
| 21 | 60.350 | 63.057 | iter17 |
| 22 | 59.668 | 58.497 | iter18 |

**iter18 wins 2–1 head-to-head**, and across those three shared rounds the means
are 53.850 vs 53.899 — a gap of 0.049. The 4-point overall difference comes
entirely from rounds they did *not* share (iter17 also ran iteration 17; iter18
also ran 19 and 20), so those means are over different query samples and are not
comparable. Correcting for precisely that is what the Elo ladder is for, and it
ranked iter18 ahead, 1593.8 to 1566.7.

This is worth reading as a worked example rather than a caveat: at this noise
level, raw batch means across unmatched samples mislead. The same lesson appears
in the ±10-point batch band and in the `specific_f1` analysis above.

**5. Tool rate limits.** `tool_pacer.py` is staged and applied at ~8 launches/s
per endpoint. `snippet_search` was measured at 100% pacer utilization during
training with only 3–4 concurrent workers; the official run uses
`--max-samples 6`.

## Official result (2026-08-04)

**As displayed on the board: 0.440 @ $0.279.** 267/267 samples, **zero errors**,
7h14m. Judge $203.61 + agent $74.38 = **$277.99**. Full precision below, since
the transfer delta is smaller than the board's rounding step.

| Metric | Internal | **Official** | Delta |
| --- | --- | --- | --- |
| Mean F1 | 0.4383 | **0.44025** ± 0.01790 | **+0.0020** |
| `semantic_f1` (194) | 0.3715 | **0.3749** | +0.0033 |
| `specific_f1` (38) | 0.8675 | 0.8588 | −0.0088 |
| `metadata_f1` (35) | 0.3423 | **0.3483** | +0.0060 |
| Agent cost | $0.2780 | $0.278579 ± 0.0099 | +$0.0006 |

Fifth calibration point, and the third positive: **+0.0025 / −0.0550 / −0.0077 /
+0.0096 / +0.0020**. Four of the five now sit within ±0.010, leaving `v0_0_8`'s
−0.0550 as the lone outlier rather than the pattern.

`specific_f1` moved −0.0088. That category carries roughly 0.058 SD across 38
all-or-nothing queries, so this is about a fifth of a standard deviation — the
quietest that metric has been across the five submissions, and not a signal.

### How strong is the margin over the incumbent?

The board ranks on point estimates, and on those this dominates. On the
statistics it is a lead, not a separation:

| | score | stderr | $/prob |
| --- | --- | --- | --- |
| this entry | **0.44025** | 0.01790 | **0.27858** |
| former leader (Asta Paper Finder) | 0.43270 | 0.01650 | 0.35480 |

The **+0.0076** score gap is well inside either interval, so the two are not
statistically distinguishable on score from single measurements. What is not in
doubt is the cost axis — **21% cheaper**, with cost stderr of $0.0099 against a
$0.076 gap — so the domination claim rests on a firm cost advantage plus a score
lead that is real in the recorded numbers but modest against the noise.

**Two separate evaluations of this agent cleared 0.4327**, though — internal at
0.4383 and official at 0.44025, by +0.0056 and +0.0076. They share the test set
and the judge model, so they are not independent estimates, but they differ in
the agent's own sampling and in judging depth (internal capped to top-K, official
uncapped). Clearing the bar under both conditions is better evidence than one run
clearing it once, and it addresses the obvious worry that a single lucky draw
crossed the line.

### Uncapped official judging helped, as it did for the opus-5 entry

Internal judging is capped to the top-K estimate; official judging grades every
submitted paper. Matched on identical query IDs:

| run | n | internal | official | delta | t |
| --- | --- | --- | --- | --- | --- |
| `cap_0_063` (opus5) | 193 | 0.3227 | 0.3095 | **−0.0131** | −2.01 |
| `cap_0_355` (opus5) | 190 | 0.3800 | 0.3946 | **+0.0146** | +2.03 |
| **`cap_0_355` (fable)** | 194 | 0.3715 | 0.3749 | **+0.0033** | +0.51 |

Both $0.355-gate runs move up on semantic; the $0.063 run moves down. That fits
the strategy difference — the 0.355 agents grade far deeper (900 stage-1
candidates for opus-5, mini overflow grading out to 800 here), so papers never
judged internally get graded officially and mostly help, whereas a shallower
curated list gains only tail papers that dilute the rank term.

Read as suggestive rather than established: n=3, and these are three different
agents whose cost, strategy and evolution model all co-vary with the gate. This
entry's own +0.0033 is at t = 0.51, indistinguishable from zero on its own; the
opus-5 entry carries the weight of the observation. Isolating it would need one
stored submission re-scored uncapped internally — same agent, same submissions,
only the cap changing.

### Cost calibration (fifth measured point)

Judge came in at **$0.00420/paper** — $203.61 over 194 semantic queries × 250
papers — against $0.0040 / $0.0030 / $0.00407 / $0.00426 for the four prior runs.

Total $277.99 against a **$268–283** pre-run estimate: inside the range, and the
first projection this campaign that did not miss low. The script's own printed
figure remains unreliable for a full-250 agent — it computes `194 × 250 ×
$0.0040 = $194` judge plus a hardcoded ~$15 agent, and both halves were wrong
here. Budget from the measured $0.0041–0.0043/paper band and from the agent's own
internal cost.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_355_fable --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_355_fable             # full (measured: $277.99, 7h14m)
```

Push the commit **before** the full run — `astabench eval` hard-fails if the
commit is not on the remote, since it stamps the SHA into `eval_spec.revision`.

Spend, now measured rather than projected: **$203.61 judge + $74.38 agent =
$277.99**, at $0.00420/paper over 194 semantic queries × 250 papers, in 7h14m at
`--max-samples 6`. See [Cost calibration](#cost-calibration-fifth-measured-point).

Then upload `submissions/asta_paper_finder/v0_0_9_cap_0_355_fable.tar.gz` via
the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [x] Official eval run (2026-08-04: 0.44025 @ $0.278579/query, $277.99 spend, 7h14m, 267/267 no errors)
- [ ] Tarball uploaded
- [ ] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`
