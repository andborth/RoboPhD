# v0_0_9_cap_0_355_fable — PaperFindingBench (AstaBench) submission snapshot

Fifth RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier).

## It would take the top of the board

The previous submission, `v0_0_9_cap_0_355_opus5`, came within 0.001 of Asta
Paper Finder's best entry and sat *beside* it on the curve. This one clears it.
On the board's three-decimal basis:

> **Internal 0.438 @ $0.278**, against the leader's **0.433 @ $0.355**.

Higher score **and** 22% cheaper — domination on both axes, not a tie. At full
precision, 0.4383 against 0.4327 and $0.2780 against $0.3548.

Projected frontier if it transfers (You.com omitted — no cost shown on the
board):

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | RoboPhD (v0_0_9 @ cap 0.063) | 0.376 | 0.052 | Standard |
| 3 | Asta Paper Finder | 0.397 | 0.063 | Custom interface |
| 4 | RoboPhD (v0_0_9 @ cap 0.355, opus5) | 0.432 | 0.251 | Standard |
| 5 | **this entry** | **0.438** | **0.278** | **Standard** |

**Five slots, four ours**, all Standard tier, with Ai2's top entry displaced
entirely and only 0.397 @ $0.063 remaining to them.

**Projected, not banked.** These are internal numbers; internal→official has run
+0.0025 / −0.0550 / −0.0077 / +0.0096 across the four submitted runs. A −0.006
outcome puts this level with the leader rather than above it — the difference
between rank 1 and a shared frontier.

## The cleanest evolution-model A/B in the campaign

This run is identical to `v0_0_9_cap_0_355_opus5` in every deliberate respect —
same $0.355 gate, same reachability guard, same luna/no-prose training judge,
same stock-4o test judge, same 600 budget, same 14 examples/iteration, same
harness. **Only the evolution model differs**: fable-5 here, opus-5 there.

And the entire +0.016 comes from **one category**:

| category | opus-5 | **fable-5** | points to fable |
| --- | --- | --- | --- |
| metadata (n=35) | 0.215 | **0.342** | **+4.46** |
| semantic (n=190/194) | 0.380 | 0.372 | −0.12 |
| specific (n=38) | 0.868 | 0.868 | −0.03 |
| | | **total** | **+4.31** |

Semantic and specific are a **wash** — 0.15 points across 232 queries. The whole
gap is 35 metadata questions, where zeros fell from **16 to 5**.

It did that while spending *less* there: metadata cost $0.0056/query against
opus-5's $0.0136, a **59% reduction**. The gain is structural, not bought.

So the honest reading is not "fable-5 builds better agents." It is that fable-5
found the metadata failure and opus-5 did not; on the 87% of the benchmark that
is not metadata, the two runs are indistinguishable.

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

## Architecture (2,781 lines, single `agent.py`)

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

Two undocumented Asta MCP behaviours this agent discovered and encodes. Recorded
because they are load-bearing and because their coverage is finite:

- **`get_citations` rejects `tldr`.** The agent found its citation-expansion
  channel had "been dead in every agent since iteration 6", returning zero citers
  on 25 of 27 measured agent × query pairs — silently, because the per-call
  failure was caught and logged as an ordinary tool error. Fixed with a
  probe-verified narrower field list. Four independent lineages in this campaign
  rediscovered this same defect.
- **`venues=` normalises some acronyms but not all.** Worked around with a
  26-entry `_VENUE_ALIASES` table sending both the acronym and the expanded
  official name. **The table is finite and ML/NLP-weighted**, so the metadata fix
  is partial exactly in the venues it omits — `metadata_8` (PADL 2012) recovered
  to 1.000, but `metadata_18` (POPL) only to 0.154 and `metadata_38` (CHI 2001)
  to 0.069.

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

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_355_fable --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_355_fable             # full (~$268-283)
```

Push the commit **before** the full run — `astabench eval` hard-fails if the
commit is not on the remote, since it stamps the SHA into `eval_spec.revision`.

Projected spend, measured from this run's own submissions: 249.1 papers/query ×
194 semantic queries at 704 chars each. The script prints `194 × 250 × $0.0040 =
$194` judge, but **that figure is not a ceiling** — the last two full-250 runs
came in above it ($197.32 and $206.38). Budget from the measured
$0.0041–0.0043/paper band: **~$194–209 judge plus ~$74 agent, so ~$268–283**.
Expect ~7h at `--max-samples 6`.

Then upload `submissions/asta_paper_finder/v0_0_9_cap_0_355_fable.tar.gz` via
the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [ ] Official eval run
- [ ] Tarball uploaded
- [ ] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`
