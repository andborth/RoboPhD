# Evolution Reflection — Iteration 6 (iter6_grade3_rescue)

## What worked well

1. **Reading the scorer's diagnostics before touching code.** The single most
   valuable 20 minutes was grepping `judge_verdicts.md` + `score_calculation.md`
   across all problems and both parent agents. It surfaced a decisive,
   non-obvious mechanic that CLAUDE.md only hints at: **only the first ~K
   submitted papers are judged** (entries below carry "(beyond scored depth —
   not judged)"), so the whole semantic game is "fill positions 1..K with
   grade-3 papers" for an unknown K of 14–304. Every design decision followed
   from that.

2. **Quantitative loss decomposition over vibes.** Tabulating grade counts
   (Perfect / Highly / Somewhat / Not) per problem per agent made the dominant
   loss bucket unambiguous: grade-2 "Highly Relevant" papers (zero recall) at
   the top of the list — e.g. 80 grade-2s vs 33 grade-3s in semantic_222's
   top-134. That pointed at a *conversion* mechanism (rescue near-misses with
   targeted snippet evidence) rather than more retrieval breadth, which would
   have been the naive move.

3. **Inheriting the best agent's code wholesale and making few, mechanism-backed
   edits.** iter4 vs iter5 differed by ~0.6 raw points on 14 queries — deep
   inside noise. Rewriting from scratch would discard several iterations of
   accumulated robustness (relax ladders, unicode normalization, empty-pool
   fallbacks, tool-error tolerance). Every edit I made maps to a specific
   observed failure with a named problem id.

4. **Fixing regressions mechanically, not by rollback.** specific_11
   (Objaverse-XL wrongly merged as a "duplicate record") got a Python guard
   (title_sim >= 0.88 + first-author match) that I *verified offline* keeps the
   AlexNet duplicate-record win (sim 1.00) and rejects XL (0.82). semantic_186
   (stage-2 sim demoting judge-liked papers) got a band rule: stage-1-perfect
   papers can't be demoted by the sim. Rollbacks would have traded one batch's
   wins for another's.

5. **Offline smoke tests with stubbed imports.** Stubbing `inspect_ai` and
   `model_registry` lets you unit-test evidence assembly, threshold math, and
   parsing in seconds without an eval run. The `_weighted` thresholds
   (0.67/0.99 bands) and the title-sim guard were both validated this way;
   both had realistic failure modes a live run would only reveal statistically.

## What was challenging / time-consuming

- **Cost accounting by hand.** The penalty structure makes cost a hard design
  constraint (iter5 lost 0.88 pts at $0.0625), but estimating a design's cost
  means token-arithmetic over pool sizes × chars-per-doc × price tables. My
  estimates matched observed costs only after calibrating against iter5's
  actuals. A strategy tool that, given (pool size, chars/doc, chunk overhead,
  model), prints projected per-query cost — or better, a per-call cost log in
  `agent_stdout` — would remove an error-prone step.
- **Attribution of small deltas.** With 14 queries/iteration and large per-query
  variance, iter4-vs-iter5 differences on single problems (0.30 vs 0.45) can be
  architecture or noise. I treated only *mechanistically explicable* deltas as
  signal (e.g. verdict-count shifts with same K). Resist tuning on ±0.1
  single-problem swings.
- **The batch composition shifts per iteration.** Iteration 5 had zero metadata
  queries, so iter5's whole metadata rewrite went unvalidated. I kept it on the
  strength of its iteration-4 failure analysis — but that's faith, not
  evidence. Future instances: check which routes actually ran before crediting
  or blaming them.

## Tools & data: assessment

- The per-problem diagnostics (`judge_verdicts.md`, `score_calculation.md`,
  `gold_criteria.md`, `agent_stdout`) are excellent — genuinely sufficient to
  separate "never retrieved" / "retrieved, ranked low" / "judged grade-2".
  `agent_stdout` printing pool sizes, predicted-perfect counts, and per-stage
  decisions (a habit of the parent agents) is what makes post-hoc analysis
  possible. Keep printing aggressively.
- No `strategy_tools/` existed this round; jq + shell over the problems tree
  was adequate but repetitive. The single most useful script to provide:
  per-problem table of (K, judged grade histogram, rank, recall, cost) across
  all agents.
- `evolution_reflection.md` files from prior iterations existed but I found the
  `reasoning.md` files more useful — they record *why* mechanisms exist, which
  prevents accidental removal of failure-driven code.

## What I'd do differently / advice to future instances

1. **Start with the verdict/K analysis, not the agent code.** Read
   `score_calculation.md` for every problem first; the code only matters once
   you know which term (rank vs recall vs grade conversion) is binding.
2. **Attack recall via grade conversion before pool growth.** Pools of 340
   already contain the gold on most queries; the papers are there, the evidence
   isn't. Scoped `snippet_search` calls are free — the only cost is re-grading.
3. **Mind the grade-3 cliff.** A paper earns recall only if *every* weighted
   criterion is judged Perfectly Relevant from your evidence text. Qualifier
   criteria ("explicitly connects X and Y", "not a survey", "micro-F1 averaged
   across episodes") are the usual blockers; hunt body passages that state them
   verbatim.
4. **Budget ≈ $0.05 mean, not $0.06.** The free-zone edge is a cliff and the
   batch mix varies; leave margin. 85%+ of spend is the cheap-model grading
   passes, so cost control = controlling (docs graded × chars/doc), not model
   choice at the planner.
5. **Never let a filter chain end empty.** Guaranteed-nonempty relax ladders
   and never-crash fallbacks are worth several points per iteration on their
   own; preserve them when refactoring.
6. **Test the specific-path duplicate logic offline** against both known cases
   (AlexNet two-records = gold contains both; Objaverse-XL = different paper).
   These two examples fully constrain the guard.

## Suggestions for the process/prompts

- **Document the scored-depth mechanic in CLAUDE.md.** "The judge judges
  approximately the first K submitted results (K = the hidden recall
  denominator); deeper entries are never judged" — I had to discover this from
  diagnostics; it reframes the entire semantic strategy and should be stated
  up front.
- **Carry a cumulative "validated mechanisms + known regressions" ledger**
  across iterations (one file, appended each round: mechanism → problem ids it
  fixed → problem ids it hurt). Each instance currently reconstructs this from
  scattered reasoning.md files and diffs.
- **Report per-route means** (specific/metadata/semantic) in the error analysis
  header. The batch mix shifts, so overall means mislead; route-level deltas
  are the comparable unit.
- **Include per-query agent cost in `error_index.json`** so cost-vs-quality
  trades can be analyzed per problem without cross-referencing cost_report.md.
