# v0_0_9_cap_0_355_opus5 — PaperFindingBench (AstaBench) submission snapshot

Fourth RoboPhD submission to the AstaBench PaperFindingBench leaderboard
(Literature Understanding category, Standard tools tier).

## Aimed at the top of the board

Our three earlier gates were set at the price of an entry we could plausibly
beat: $0.033 at the second-cheapest agent, $0.063 at the frontier point directly
above ours. This is the first run pointed at the **best** agent on the board —
Asta Paper Finder at **0.433 @ $0.355** — by setting the free zone to its price
exactly.

The construction is the same each time: put the gate at a competitor's price,
and the cost half of a Pareto claim is bought by definition, leaving only the
score half to win. What differs here is the target's difficulty. 0.433 is the
highest score anyone has posted.

It came within one displayed thousandth. On the board's own three-decimal basis:

> **Official 0.432 @ $0.251**, against the target's **0.433 @ $0.355**.

At full precision that is 0.43177 vs 0.43270 — a gap of **0.00093** on a stderr
of **0.018**, so the two scores are statistically indistinguishable, and ours
costs **29% less**. 2026-08-02, 267/267 samples, zero errors; full breakdown in
[Official result](#official-result-2026-08-02) below.

Neither dominates, so both stay on the curve, and this opens a new region of it
between Ai2's two entries:

| | entry | score | $/prob | tier |
| --- | --- | --- | --- | --- |
| 1 | RoboPhD (v0_0_8) | 0.220 | 0.006 | Standard |
| 2 | RoboPhD (v0_0_9 @ cap 0.063) | 0.376 | 0.052 | Standard |
| 3 | Asta Paper Finder | 0.397 | 0.063 | Custom interface |
| 4 | **this entry** | **0.432** | **0.251** | **Standard** |
| 5 | Asta Paper Finder | 0.433 | 0.355 | Custom interface |

**Three of five frontier slots, all Standard tier.** (Four RoboPhD entries are
listed once both pending clear, but `v0_0_7` is displaced by our own
`cap_0_063`, which is higher *and* cheaper.)

Internally the run scored 0.4222 @ $0.246, so official came in **+0.0096** — the
best transfer of the four submitted runs and the first meaningfully positive
one.

## What the extra budget actually bought

The interesting question about a 5.6× larger free zone is what evolution chose
to do with it. Per-model spend on semantic queries, winner against winner:

| | `cap_0_063` winner | `cap_0_355` winner | |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | $0.0289 (40%) | **$0.1727 (49%)** | **6.0×** |
| deep-band grader | `claude-haiku-4-5` $0.0211 (29%) | **`claude-sonnet-4-6` $0.1667 (48%)** | **7.9×**, and a model swap |
| `gpt-5.4` | $0.0216 (30%) | **$0.0101 (3%)** | **halved** |
| total | $0.0717 | $0.3495 | 4.9× |

Three decisions, none of them "the same thing, more expensive":

1. **Grade far more candidates.** Stage-1 breadth went from 250 papers to
   **900** — that is where the 6× mini spend went.
2. **Upgrade the grader that matters.** The deep band stayed about the same size
   (130 → 120 papers) but moved from Haiku 4.5 to Sonnet 4.6, a 3×-per-token
   model. Better judgment on the band that decides the top-K, not more of it.
3. **De-fund a tier to pay for it.** `gpt-5.4` fell from 30% of spend to 3%,
   surviving only as the planner and the `specific` verifier. The 44-paper
   GPT-5.4 head-reranker of the cheap agent is gone.

That maps onto the measured outcome. Grading 900 candidates instead of 250 is a
**recall** purchase (0.2245 → 0.2764); dropping the head-reranker is a **rank**
cost (0.7723 → 0.7352). Since recall is worth ~9× rank at the margin, trading
the second for the first is the right side of the exchange — and evolution paid
for its recall partly by selling ordering precision, which is a sharper move
than simply spending more.

Notably it did **not** buy a more expensive planner, and it did not buy more
retrieval — tool calls are free, so retrieval breadth was never budget-limited
in the first place. The whole increment went into how many candidates get graded
and how well.

$0.355 is Asta Paper Finder's top entry to three decimals, the same
competitor's-price construction as the $0.033 and $0.063 gates. The agent landed
at **$0.246**, 69% of the gate — unlike the $0.063 run, which engineered to its
threshold and landed on it.

## Naming and conventions

**The version is `v0_0_9` again, deliberately.** The patch number tracks the
RoboPhD code base, not the run. Nothing meaningful changed between this run and
`v0_0_9_cap_0_063_opus5` — same solver stack, same harness — so this is the same
code base at a different cost cap. `cap_0_355` names the training free zone
($0.355/query); `_opus5` = evolved by Claude Opus 5.

Prices throughout are quoted to **three decimals**, the leaderboard's precision;
two entries that round to the same figure are a cost tie decided on score.

## Snapshot layout (ds1000 precedent: a curated run snapshot, ~14MB)

- `agents/` — all 20 iteration winners + seed (`agents/iter21_gold_rubric_and_hard_predicates/agent.py` is the submitted one)
- `iteration_020/`, `iteration_021/` — the winner's breeding data and debut
- `evolution_output/` — `CLAUDE.md` + `iteration_021/` only
- `checkpoint.json`, `final_report.md`, `test_results.json`, `test_results.per_problem.json`

**Redistribution note:** as with the three prior snapshots, the iteration
problem dirs embed VALIDATION-split gold. Verified before commit: the **25**
problem IDs embedded here have **zero** overlap with the held-out test IDs.

## Lineage

- Run: `robophd-asta_paper_finder-011` (`../robophd_runs/robophd/asta_paper_finder_20260731_023655`)
- Engine: RoboPhD Elo (Opus 5 evolution, Deep Focus OFF), budget 600 (609 used),
  23 iterations, 14 examples/iteration, restarted once mid-run
- Winner: `agents/iter21_gold_rubric_and_hard_predicates` — Elo 1602, train mean
  49.428 @ $0.270, **3 Elo test rounds**
- 10 distinct winners across 23 iterations
- **Second run with the Elo-reachability guard enabled** (after
  `v0_0_9_cap_0_063_opus5`). It fired at iteration 22 — the same iteration as
  last time — without suppressing the winner, which was created at 21. Two live
  firings, two clean outcomes, consistent with the 121-run archive replay.

## Architecture (1,884 lines, single `agent.py`)

Two halves, both keyed to the scoring formula.

**Rubric reconstruction.** Grade 3 needs `weighted > 0.99` — essentially every
criterion Perfect — and only grade-3 papers earn recall, so what the criteria
*are* determines the ceiling. The planner is given conventions induced from
training diagnostics: default to exactly 3 criteria, weights from 0.4/0.3/0.3 or
0.5/0.3/0.2, one criterion per concept literally named, and the two rules it
identifies as most often missed — qualifiers ("common", "large-scale", "recent")
and negative constraints ("exclude survey papers") each become their **own**
criterion rather than being folded into another.

**Hard predicates.** The same call emits structured filters for the exact-match
paths: `years`, `year_min/max`, `min_citations`, `min_authors/max_authors`,
`venues` plus lowercase `venue_aliases`, and `cites_reference`. Metadata queries
become deterministic filters instead of keyword gymnastics.

A note on the agent's name, which is the evolution model's own and kept
unedited: `gold_rubric` describes the *target* of the reconstruction, not its
input. No gold reaches the agent. The benchmark's `relevance_criteria` are
gold-side and the solver sees only `state.metadata["raw_query"]`; every
criterion downstream is inferred. Verified by AST parse of executable positions
— no corpus-id-shaped constants, no `sample_id` branching beyond the required
`query_id` output field. The planner prompt does carry three worked examples of
query → criteria+weights, drawn from training `gold_criteria.md` (which the task
docs expose post-hoc); their topics appear 11 / 9 / 11 times in training queries
and zero times in the held-out 267. No test artifacts were reachable during
evolution, including across the mid-run restart.

Grading is staged across two providers — `gpt-5.4-mini` over stage 1,
`claude-sonnet-4-6` over the top 120 in stage 2, mini again on the tail — and the
specific path dual-verifies with `gpt-5.4` and Sonnet. Inherited from the iter7
lineage: solo evidence mining, one scoped `snippet_search` per (paper,
criterion), after measuring that grouped mining starved 60% of the head because
`paper_ids` is a scope filter rather than a per-paper allocation.

### Models (3 handles, 2 providers)

| Handle | Provider | Role |
| --- | --- | --- |
| `gpt-5.4-2026-03-05` | OpenAI | rubric reconstruction + hard predicates, specific verify A |
| `gpt-5.4-mini` | OpenAI | stage-1 grading, stage-2 tail |
| `claude-sonnet-4-6` | Anthropic | stage-2 deep grading (top 120), specific verify B |

All three are priced in the litellm 1.88.1 bundled map (`AGENT_MODELS` preflight).

## Internal results (basis: stock GPT-4o judge, canonical ordering)

| Metric | Value |
| --- | --- |
| Mean F1 (267 queries) | **0.4222** |
| `semantic_f1` (190 scored) | 0.3800 |
| `specific_f1` (38) | **0.8684** — best across eleven runs |
| `metadata_f1` (35) | **0.2149** — see below |
| Agent cost | $0.2461/query — **$0.246** on the board ($65.70 total) |
| Judge cost (internal, capped + cached) | $77.85 |
| Timeouts | **4** queries, all semantic, at the 1770s internal cap |

The semantic gain is a deliberate trade, not a measurement shift. Against
`cap_0_063` on the same 189 problems: rank **0.7723 → 0.7352**, recall
**0.2245 → 0.2764**. Recall is worth ~9× rank at the margin, so trading ordering
for pool coverage nets positive through the harmonic — exactly the mechanism the
lineage argues for.

### Known weakness: `metadata_f1`, and why

16 of 35 metadata queries scored zero, and the cause is diagnosed rather than
mysterious. It is **not** the hard predicates over-filtering. On 7 of those the
log reads `metadata filter kept 0 of N` for N = 206–1018, and the pool being
filtered was already wrong: every keyword search had returned zero documents,
with no warning line — the calls succeeded and came back empty.

The native `venues=` filter takes the canonical abbreviation and silently
returns empty for plausible expansions. Probed live: `venues="POPL"` returns
18–20 real POPL papers (corpus venue string `"ACM-SIGACT Symposium on Principles
of Programming Languages"`), while `venues="Principles of Programming
Languages"` and `venues="ACM SIGPLAN Symposium on Principles of Programming
Languages"` both return **0**. The pool then fills only from venue-agnostic
`snippet_search`, the venue post-filter correctly drops all of it, and the agent
falls back to ~20 unfiltered papers.

The aliases that *would* have matched are computed and used only in
post-filtering, never in retrieval. Headroom if fixed: roughly +0.013 overall
for the 9 addressable queries, +0.024 if all 16 were recoverable — though
`metadata_23/24/28/44/45` are limited by the `get_citations` 1000-cap on
hugely-cited targets and are not fixable this way.

## Infrastructure verification

Checked before submitting, because the score is a large jump:

- **`k_estimate` identical to `cap_0_063` on all 189 comparable problems** (mean
  101.9 both). The recall denominator is the benchmark's, unchanged — this is
  the check that would catch a scoring artifact.
- Judge: all 190 semantic problems have verdicts — **190, not 194, because this
  section describes the INTERNAL run**, which lost four semantic queries to the
  1770s subprocess cap. **23 failed calls out of 18,187 judged (0.13%)**.
- **Zero** problems had evidence scrubbed by the grounding check.
- Judge spend $77.85, in line with the two prior runs ($73.90, $72.85).
- 0 exceptions, 0 zero-accuracy cases, 0 clone detections.
- The four timeouts recorded $0.00 cost — killed mid-flight at the internal
  1770s cap. They **cost** roughly 0.006. The official wrapper ceiling is 3000s,
  so they may complete officially.

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
| Models (solver, from logs) | gpt-5.4-2026-03-05, gpt-5.4-mini, claude-sonnet-4-6 |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench PaperFindingBench leaderboard](https://allenai-asta-bench-leaderboard.hf.space/literature-understanding#h-paperfindingbench-leaderboard) |

Agent description as submitted on the form:

> Evolved by RoboPhD with Claude Opus 5 as the evolution (but not a solver)
> model, under a $0.355/query training cost cap on solver spend — set at exactly
> the price of the most capable entry on the board, so that evolution had to win
> on score at a price already known to be competitive. It came in at
> $0.251/query. The benchmark awards recall only for papers judged
> Perfect on essentially every criterion, and the criteria themselves are hidden
> at query time — the agent sees only the natural-language request. So it first
> reconstructs the grading rubric from that request alone, splitting out the
> parts most often missed: a qualifier like "common" or "large-scale" becomes
> its own requirement, as does an exclusion like "no survey papers". The same
> step emits hard structured filters — years, citation counts, author counts,
> venues — that turn author/venue/year requests into deterministic selection
> rather than keyword guessing. Candidates are then graded against the
> reconstructed rubric by a staged ensemble spanning two model providers, and
> evidence is mined one paper at a time so that a passage proving the hardest
> requirement is retrieved for each candidate individually rather than being won
> by whichever paper in a batch matched best.

## Official-result risks (assessed pre-submission)

**1. Uncapped official judging.** Internally only the top-K estimate was judged;
officially all 250 submitted papers per query are. This agent ships the full 250
on every semantic query. On `v0_0_9_cap_0_063_opus5` this cost −0.0117 on
semantic; this run's recall-heavy strategy submits deeper into the tail, so the
exposure is plausibly larger.

**2. `specific_f1` is the metric with the most to lose.** 0.8684 is our best
ever. The same metric collapsed −0.250 on v0_0_8 while its judged metric barely
moved, because title resolution proved environment-sensitive. It transferred at
zero drift on `cap_0_063`, whose cross-provider title-guess union this agent
also has.

**3. Selection confidence is lower than usual.** The winner carries **3** Elo
test rounds against 11 for the runners-up. That affects which agent was picked,
not whether 0.4222 is real — the test score is measured on all 267 problems —
but a different pick might have scored comparably.

**4. Timeouts.** Four internally, up from one. A heavier agent (48.8h summed
test wall clock) against a fixed ceiling.

**5. Tool rate limits.** `tool_pacer.py` is staged and applied at ~8 launches/s
per endpoint; `snippet_search` was measured at 100% pacer utilization during
training with only 3–4 concurrent workers, and this agent mines solo — one call
per (paper, criterion).

## Official result (2026-08-02)

**As displayed on the board: 0.432 @ $0.251.** 267/267 samples, **zero errors**,
6h58m. Full precision below, since the deltas are smaller than the board's
rounding and the stderr is what makes the comparison to the leader meaningful.

| Metric | Internal | **Official** | Delta |
| --- | --- | --- | --- |
| Mean F1 | 0.4222 | **0.43177** ± 0.01797 | **+0.0096** |
| `semantic_f1` (194) | 0.3800 | **0.393** | **+0.013** |
| `specific_f1` (38) | 0.8684 | 0.816 | −0.052 |
| `metadata_f1` (35) | 0.2149 | 0.228 | +0.013 |
| Agent cost | $0.2461 | **$0.250635** ± 0.0089 | +$0.0045 |

Judge $206.38 + agent $66.75 = **$273.13**.

This is the **best internal→official transfer** of the four submitted runs, and
the first meaningfully positive one: +0.0025 / −0.0550 / −0.0077 / **+0.0096**.

Two pre-submission risks resolved the opposite way to expectation:

- **Uncapped official judging helped.** It cost `cap_0_063` −0.0117 on semantic;
  here semantic went **up** 0.013. That fits the strategy — an agent that grades
  900 candidates and submits deep gains when all 250 are judged rather than only
  the internal top-K.
- **The four internal timeouts were a local artifact, not an agent property.**
  Officially every sample completed: max 1042s against the 3000s wrapper ceiling,
  median 651s. They were the internal 1770s cap under 8-way local contention.

`specific_f1` dropped 0.052, and that is resampling rather than a finding.
**36 of the 38 queries scored identically; two flipped 1.000 → 0.000** —
`specific_26` ("the gru paper") and `specific_40` ("the NEMO^2 paper"), both
single-target nickname resolutions where gold is one paper and the agent
internally submitted that one paper correctly. 2/38 = 0.0526, the whole delta.

Exact-match scoring is deterministic, but the submission is not: resolution runs
LLM title guesses, so each query is closer to a Bernoulli trial with a high
success rate than to a fixed measurement. At p ≈ 0.85 the standard deviation of
the category mean over 38 queries is ≈ 0.058, so a 0.052 move is under one SD
and needs no mechanism to explain it.

Worth separating from v0_0_8's −0.250 on the same metric, which is ~9.5 queries
flipping — about 4.3 SD, and therefore systematic rather than noise. The two are
not the same phenomenon.

### It lands one thousandth below the board leader

The frontier table is in [Aimed at the top of the board](#aimed-at-the-top-of-the-board)
above. Against the leader: **0.43177 vs 0.43270** — a gap of **0.00093** against
a stderr of 0.018, so statistically indistinguishable — at **29% lower cost**.
Neither dominates, so both stay on the curve. It dominates 14 of the 28 costed
entries.

The gate aimed at 0.433 @ $0.355 and produced 0.432 @ $0.251. The cost half of
the Pareto claim was bought by construction, as intended; the score half missed
by less than one part in four hundred.

### Cost calibration (fourth measured point)

Judge came in at **$0.00426/paper** — $206.38 over the official run's **194**
semantic queries × 250 papers — against $0.0040 / $0.0030 / $0.00407 for the
three prior runs.

The submit script's own projection is `194 × 250 × $0.0040 = $194` judge, and
actual was $206.38: a **6.0% under-estimate**. So the printed figure is no
longer reliably a ceiling. It over-predicts only for an agent shipping short
lists (~$219 printed against $118.68 actual for v0_0_8 at 203.5 papers/query);
for a full-250 agent it now runs slightly under — $194 against $197.32 for
`cap_0_063`, and $194 against $206.38 here. Budget from the measured
$0.0041–0.0043 band rather than from the printed number.

The flat-rate-under-the-250-cap model continues to hold, and evidence length
continues not to predict it: 765 chars/paper here billed at essentially the same
rate as 747 chars for `cap_0_063` and 976 for v0_0_7.

## Reproduce

```bash
pip install litellm==1.88.1   # submission-scoring price map
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_355_opus5 --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_9_cap_0_355_opus5             # full (measured: $273, 6h58m)
```

Push the commit **before** the full run — `astabench eval` hard-fails if the
commit is not on the remote, since it stamps the SHA into `eval_spec.revision`.

Spend, now measured rather than projected: 250 papers/query × 194 semantic
queries = 48,500 papers at 765 chars each, billing **$206.38 judge** at
$0.00426/paper, plus **$66.75 agent** = **$273.13**. The script's own projection
prints $194 judge, which under-shot by 6.0% — see
[Cost calibration](#cost-calibration-fourth-measured-point).

Then upload `submissions/asta_paper_finder/v0_0_9_cap_0_355_opus5.tar.gz` via
the HF Spaces form (https://huggingface.co/spaces/allenai/asta-bench-leaderboard).
Form metadata: Openness "Open source, closed weights"; Tools tier "Standard".

## Submission status

- [x] Official eval run (2026-08-02: 0.43177 @ $0.250635/query, $273.13 spend, 6h58m, 267/267 no errors)
- [x] Tarball uploaded — submitted to AstaBench 2026-08-02, under official review
- [x] Official score/cost recorded in `../robophd_runs/results/asta_paper_finder.json`

Two RoboPhD submissions are under review concurrently: `v0_0_9_cap_0_063_opus5`
(0.376 @ $0.052) and this one (0.432 @ $0.251). Once both clear, four RoboPhD
entries are listed and three hold frontier slots — `v0_0_7` is displaced by
`cap_0_063`, which is higher *and* cheaper.
