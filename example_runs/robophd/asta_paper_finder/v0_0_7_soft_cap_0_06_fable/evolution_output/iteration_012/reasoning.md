# Iteration 12 — Analysis and Plan

Name: iter12_body_conjunction

## What the iteration-11 data shows

**Standings.** iter11_tail_saturate 44.20 ($0.0585), iter10_cite_expand 42.34
($0.0577), iter4_judge_sim_ranker 34.91 ($0.0374). The iter11 grafts held:
the tail evidence sweep measurably moved the grade distributions it targeted
(semantic_110 Perfect 67→72 and rank 0.70→0.74; semantic_112 Perfect 161→172
and rank 0.57→0.76; semantic_8 Perfect 39→47 and rank 0.46→0.70; semantic_219
Perfect 5→7). iter11 is the base for this round. Cost margin is thin:
$0.0585 on a 71% semantic batch ⇒ ~$0.060 at the test set's 73% mix. This
round must trim, not just hold.

**Where the remaining points are.** Per-problem grade histograms
(judge_verdicts.md) split the low scorers into two failure shapes:

1. **All-Highly queries** — the qualifier criterion is never judged Perfect,
   and grade 2 earns zero recall: semantic_104 (47 Highly / 1 Perfect, K=56,
   score 0.035), semantic_7 (15 H / 1 P, K=18, 0.099), semantic_222
   (64 H / 33 P). On 104 even the agent's own GPT_5_4 head-verify confirmed
   only 4/16 — for single-architecture papers the "common/widely-used
   architectures" criterion is a *reality* limit, not just an evidence limit
   (the one judged Perfect was a survey whose abstract literally states a
   taxonomy). Partially attackable, not fully.

2. **Somewhat-mass conjunction queries** — the pool itself lacks papers
   satisfying the conjunction: semantic_137 ("generation vs discrimination",
   62 Somewhat judged, 6 Perfect, K=98), semantic_219 ("rejection sampling
   finetuning", 38 Somewhat, 7 Perfect, K=58), semantic_226 (66 Somewhat),
   semantic_193 (90 Somewhat). Gap-fill fired with decent vocabulary
   ("RFT best of N", "process reward model reranking") yet predicted-perfect
   never moved (9→9, 14→14). Diagnosis: the conjunction ("rejection sampling
   *used in* finetuning") typically lives in a paper's **method/analysis body
   text**, not its title/abstract — `search_papers_by_relevance` can't see it.
   `snippet_search` ranks body passages across the whole corpus and is free,
   but iter11 gives it only 3 queries × limit 35, merged as ONE round-robin
   source among 11 — body-matched candidates get ~9% of the pool while the
   10 keyword rephrasings (which all hit the same title/abstract surface)
   get 91%.

**Two mechanical faults found in stdout:**

- **References are dead server-side under every access path.** iter11's
  per-seed `get_paper(fields="corpusId,references")` "fix" fails identically
  to the batched call it replaced: `'NoneType' object is not iterable` on
  every call, every problem. Citation expansion has been **citers-only for
  three iterations**. The metadata cites-venue check on metadata_33 resolved
  0/0 referenced venues, relaxed, and submitted 13 papers against a 1-paper
  gold (F1 0.143 instead of a possible 1.0).
- One sim chunk of 8 came back unparsed on semantic_219 (defaulted to
  partial) — rare, tolerated.

## Changes in iter12_body_conjunction (base: iter11_tail_saturate)

Theme: **give body-text retrieval a real share of the funnel** (the frontier
failure is conjunction recall, and passage search is the only free tool that
sees where conjunctions are stated), plus reference-plumbing repair and cost
trims that restore margin.

1. **Body-conjunction retrieval (headline, FREE).**
   - Planner now emits **5 snippet queries** (was 3), of which ≥2 must state
     the implied connection/qualifier phrased the way a paper's method or
     analysis section would state it.
   - Snippet limit 35→50 per query.
   - Each snippet query becomes its **own source list** in the round-robin
     merge (5 of 15 lists ≈ a third of pool slots available to body-matched
     candidates, up from 1 of 11). Papers arriving this way carry their
     matched body passage, which flows straight into `markdown_evidence` for
     the connection criterion.
   - Per-doc snippet retention in the initial pass 2→3 passages.
   - Planner keyword instruction: 2 of the 10 queries must name specific
     well-known methods/systems/model families that instantiate the request
     (e.g. "ReST reinforced self-training", "reward ranked fine-tuning
     RAFT") — parametric-knowledge aliases that keyword rephrasing misses.

2. **Reference plumbing: probe field variants once, then commit (FREE).**
   Both `get_paper` and `get_paper_batch` reject `fields="corpusId,
   references"`. Both ref-fetch sites now probe two field variants on one
   paper first (`corpusId,references`, then the S2-subfield form
   `corpusId,references.corpusId,references.title`); whichever returns data
   is used for the rest; if both fail the stage skips quietly (citers-only
   expansion, relax ladder in metadata) instead of burning 10-26 doomed
   calls. Loud telemetry either way — if the subfield form works, the
   metadata cites-venue/cites-title verifications and the prior-work half of
   citation expansion come back from the dead; if not, we stop pretending.
   Citer limit 70→90 so the citers half can actually fill the expansion cap.

3. **Cost trims (fund #1, restore margin).** HEAD 110→100; RESCUE_MAX 32→24;
   EXPAND_CAP 150→120; SIM_CUT 640→600; stage-2 sim now **skips head papers
   with no snippets and stage-1 weighted ≤ 0.45** — their sim input is just
   title+tldr+abstract, i.e. nearly the text stage 1 already graded, so the
   verdict is reused instead of re-bought (~15-20% of sim volume). Modeled
   per-heavy-semantic ≈ $0.089, typical semantic ≈ $0.072 (was $0.080-0.100);
   batch mean at a 73% semantic mix ≈ $0.053-0.055. Verify depths untouched —
   thin pools are where whole scores live.

4. **Inherited unchanged:** planning prompt structure, triage/sim/rescue/
   verify cascade, banding order, tail sweep (it worked), alias hedging on
   specific queries, metadata relax ladder, transport wrapper, telemetry.

## Why this should score higher

- The Somewhat-mass conjunction queries (137, 193, 219, 226) average ~0.20
  and are ~30% of the semantic share; their binding term is recall of
  conjunction-satisfying papers, and body-passage retrieval is the only
  untried free modality that targets exactly where conjunctions are written.
  Even +4-6 true grade-3s on each of these query shapes is worth ~+2-3
  points on a 14-query batch.
- The reference repair either revives two scoring features for free (best
  case: metadata cites-checks regain precision — metadata_33-shaped queries
  go 0.14→1.0 when verification narrows 13→1-3; expansion regains prior-work
  recall) or eliminates wasted calls and log noise (worst case, no score
  change). Strictly non-negative.
- The trims keep the identical machinery on 95% of decisions while moving
  the modeled mean from the free-zone edge ($0.060 at test mix) to ~$0.054,
  removing the risk of a tiebreaker penalty that would cost a full
  query-equivalent.
- Specific/metadata exact-match paths (3/3 specifics for both leaders this
  round) are untouched except the strictly-fail-open reference repair.
