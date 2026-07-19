# Iteration 18 — Analysis and Plan

Name: iter18_triage_first

Base: `iter17_recall_stack`, with its one losing change reverted and the
winner's (`iter13_grade_view`) triage geometry restored. Equivalently: iter13
plus every iter17 change that was separable from pool widening.

## 1. What the data says

### Standings

| Agent | Score | Raw F1 | Mean cost |
| --- | --- | --- | --- |
| iter13_grade_view | 41.43 | 0.419 | $0.0613 (penalized 0.47 pts) |
| iter12_body_conjunction | 39.13 | 0.391 | $0.0577 |
| iter17_recall_stack | 37.05 | 0.371 | $0.0418 |

iter17 was iter13 plus seven changes, and it lost 4.4 points. iter13 has led
raw F1 in three of the last four iterations. The three exact-match queries
were identical across all agents (two 1.000, one 0.000), so the entire spread
lives in the eleven semantic queries — matching the 73%-semantic test mix.

### The decisive measurement: pool widening is a triage tax, not a recall lever

I re-ran the grade-3 attribution over the iteration-17 verdicts
(`attrib17.py`, this directory). For every paper *any* agent got graded 3,
classify why this agent did or didn't get credit:

| agent | got_it | not_retrieved | evidence_lost | stranded |
| --- | --- | --- | --- | --- |
| iter12_body_conjunction | 55.2% | 27.4% | 11.7% | 5.7% |
| iter13_grade_view | 52.7% | 28.3% | 12.5% | 6.5% |
| iter17_recall_stack | 49.7% | **35.1%** | 9.2% | 6.0% |

iter17 raised `POOL_CAP` 320→400 specifically to attack the largest bucket,
and **that bucket got worse** (28.3% → 35.1%). This is the *second consecutive
iteration* where widening the pool increased the count of grade-3 papers
missing from the submission: iter16 did it at 640 and lost 4.7 points the same
way.

The mechanism is now well established and I think it settles the question.
`MAX_SUBMIT` is 250 and the judge reads only the first ~K positions. A wider
pool does not add slots — it adds *contenders for fixed slots*. Every extra
candidate is one more chance for a noisy stage-1 grade to displace a true
grade-3 at the cut. Pool size is not a recall lever on this benchmark. Worse,
iter17 paid that tax while simultaneously trimming the machinery that resolves
the cut (`SIM_DEPTH` 55→48, `RESCUE_MAX` 22→16, `VERIFY_TOP` 26→20) to fund
it, which is the worst possible pairing: more contenders judged by a cheaper
judge.

### What in iter17 was actually good

The evidence changes look genuinely positive and are entirely separable from
the pool. On the 268 papers that both iter13 and iter17 submitted *and* the
judge graded, iter17's longer evidence promoted 31 papers to grade 3 and
demoted 20 — net +11 — and its `evidence_lost` bucket is the best of the three
agents (9.2% vs iter13's 12.5%). Longer verbatim evidence is free: the scorer
has no length cap, grounding is checked per-passage independently, and the
internal graders read the separate compact `_grade_view`, not this text.

This is confounded with iter17's other changes, so I am not claiming a
precise effect size. But the direction is consistent across two independent
measurements (the flip table and the attribution bucket), the mechanism is
sound, and the change cannot cost anything — which is enough to keep it.

### Cost

Per-stage costs, measured from the iteration-17 stdout telemetry × the price
table (both agents, semantic queries only):

| stage | iter13 $ | iter17 $ |
| --- | --- | --- |
| t1 | 0.0214 | 0.0251 |
| verify | 0.0212 | 0.0104 |
| plan | 0.0108 | 0.0114 |
| sim | 0.0096 | 0.0044 |
| t1exp | 0.0062 | 0.0061 |
| gap + t1gap | 0.0052 | 0 |
| rescue | 0.0037 | 0.0022 |
| **total** | **0.0781** | **0.0596** |

iter13's real problem is that $0.0781/semantic projects to $0.062+ at the 73%
test mix — outside the free zone. It was measured at $0.0613 on this batch and
paid a penalty. Any agent built on iter13's quality must also get cheaper.

## 2. Changes (all reverts — no new mechanisms)

1. **Pool reverted to iter13 geometry.** `POOL_CAP` 400→320,
   `POOL_MERGE_HEAD` 300→240. This is the headline. The 14-query
   diverse-category planner is **kept** — more retrieval *angles* feeding a
   fixed-width prescreen is a different thing from more candidates *surviving*
   it, and the larger union it builds (1719 uniques vs iter13's 1102 on
   semantic_189) gives the lexical prescreen a better field to pick 320 from.
2. **Triage depth restored to the winner's values.** `SIM_DEPTH` 48→55,
   `RESCUE_MAX` 16→22, `VERIFY_TOP` 20→26, `VERIFY_TOP_THIN` 26→30. These were
   trimmed to fund the pool; with the pool reverted they are affordable again,
   and they are precisely the stages that decide the 250-slot cut and the
   order of the judged prefix.
3. **Ambiguous-specific submission re-tightened**: alias backstop fill 12→5,
   submit cap 14→8, alternates 8→6, punt fallback 8→5. iter17 widened these
   betting that a ~5-paper ambiguous gold set rewards hedging. On the one
   observed case (specific_39, "the SPIKE paper") the wide list scored 0.000
   exactly as the tight list did — the extra slots filled with alias homonyms
   from unrelated fields (SARS-CoV-2 spike protein, spike-timing plasticity)
   rather than gold. Unconverted slots are pure precision loss in exact-match
   F1. The *upstream* widening is kept: alias search at the tool max and the
   48-entry shortlist are free and give the verifier more to choose from.
   Retrieve wide, submit tight.

**Kept from iter17** (each separable from the pool change, each independently
evidenced): the 14-query diverse-category planner; the compact triage output
format (local 1..N indices, unspaced digit grades — output bills at 6× input
and the digits parse identically); the longer evidence cuts with containment
dedup; the removed gap-fill round (five observed firings moved
predicted-perfect by a net +2, and the 14-query planner now does that breadth
job proactively for every query rather than reactively for a few);
`SNIP_INIT_LIMIT` 100 and `SNIPPET_TIMEOUT` 90; tail sweep to 250 (I checked
`_sweep_one` — it is snippet-only, no LLM spend); and the metadata path fixes
on top of iter13's chunked venue filter.

**Cost projection**, modelled as iter13's input volumes (pool and depths
reverted) with iter17's measured output-per-input ratios (compact format),
minus gap-fill: **$0.0698/semantic query → $0.053 batch mean at the 73%
semantic test mix**, inside the free zone. Even an 86%-semantic batch lands at
$0.061, a negligible 0.05-error penalty. iter13's quality at iter13-minus
cost.

## 3. Why this should score higher

The claim is narrow and mechanical: iter17 is iter13 plus seven changes, six
of which are free-or-cheaper and at worst neutral, and one of which —
`POOL_CAP` 320→400 — is measurably responsible for the loss. Reverting that
one change (and the depth trims that existed only to fund it) recovers
iter13's leading raw F1, while the six retained changes keep iter17's better
evidence quality and its much lower cost.

Against iter13 specifically: same pool, same triage depths, same verify, same
venue filter — plus longer evidence (its worst bucket was `evidence_lost` at
12.5%, and the evidence changes measurably improved exactly that), plus a cost
profile that clears the free zone where iter13 paid a penalty.

Against iter17: the two-iteration-consistent pool finding is reverted.

Residual risks, in order of concern:

- **Removing gap-fill could cost retrieval breadth**, which is the largest
  bucket. This is the one change I kept that isn't a pure revert, and it's the
  genuine risk. The mitigation is that the 14-query planner explicitly demands
  the same categories gap-fill was generating (synonyms, named methods,
  adjacent subfields), proactively for every query rather than reactively for
  the few below the trigger — the breadth moves upstream. If iteration 19 sees
  `not_retrieved` rise above iter13's 28.3%, this is the first thing to undo.
- **The evidence-length benefit is confounded** in iter17's bundle. If it is
  illusory, the change is still free by scoring rules, so the downside is zero
  rather than negative.
- **Batch noise is ±5 points on identical code** (iter12 scored 45.8 and 40.0
  on consecutive batches). Every change here is either a revert toward the
  measured winner or free, which is the only defensible posture at n=14.

## 4. Verification

`smoke_test.py` (62 checks, all passing) stubs the harness and exercises the
changed surface: the reverted constants, the compact-format grade parse with
local→global index mapping, evidence assembly (containment dedup, verbatim
grounding, ≤8 passages), the 14-query planner parse and fallbacks, the
metadata conjunction/intersection wiring, the chunked venue filter, and the
re-tightened ambiguous-specific caps. `pyflakes` is clean.

One test note worth recording: the inherited "gap-fill block gone" check
grepped the whole file, and my docstring's per-stage cost table legitimately
names the removed `t1gap` stage — so it failed on prose, not code. I confirmed
via AST that no gap-fill code path survives, then rescoped the check to run
against the source with the module docstring stripped. Grep-the-source
assertions about *removed* machinery have to exclude the changelog that
documents the removal.

`attrib17.py` (this directory) is the attribution script, re-pointed at
iteration_017; run it against iteration_018 next round and compare the
`not_retrieved` column against the 28.3% / 35.1% baselines above — that single
number is the direct test of this iteration's central claim.
