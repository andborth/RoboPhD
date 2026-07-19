# Evolution Reflection — Iteration 14 (asta_paper_finder)

## The single most valuable thing I did: calibrate before shipping

My planned headline change was to order results by lexical criterion coverage
of the submitted evidence. The story was airtight on paper: it mirrors the
judge's documented grade-3 rule, computed on the exact string the judge reads,
and the internal LLM graders were demonstrably miscalibrated on that call. Any
prior iteration would have shipped it.

Instead I wrote `calibrate.py` — 40 lines that score every one of the 1208
judged papers' *actual submitted evidence* against its query's *actual
criteria* and group by the judge's *actual grade*. Result: Highly-Relevant
papers score ABOVE Perfectly-Relevant at every threshold tried. The change
would have promoted grade-2 papers over the grade-3s that alone earn recall —
actively negative, wrapped in a convincing story.

**The lesson generalizes: every ranking/grading heuristic has a free ground-
truth test available.** Prior iterations' diagnostics contain submitted
evidence + criteria + per-paper judge verdicts. Before wiring any new signal
into ordering or grading, correlate it with real verdicts. Five minutes of
scripting either validates the change or kills it before the expensive eval
round. Iteration 13 did this for score *arithmetic* (which lever is worth
what); this round extended it to *signals*. Both should be standard practice.

## What else worked

- **Measuring defects instead of inferring them.** The dedup bug was claimed
  as "536/6308 passages redundant, 33.9% of papers waste a slot" — one script
  over real submissions. A measured defect can't be argued away, and fixing it
  is safe regardless of how much it gains.
- **Reading judge_verdicts.md histograms first.** "15 Highly, 3 Perfect, pool
  otherwise clean" (semantic_7) instantly localizes the failure to evidence
  rather than retrieval. The grade histogram per query is the fastest triage
  signal in the whole diagnostic tree.
- **Smoke test with stubbed imports.** 41 checks; caught two real bugs this
  round (a threshold semantics error and a coverage floor set above what a
  one-sentence passage can score). The failing smoke test is also what forced
  me to debug match-score distributions numerically instead of trusting my
  mental model.
- **Knowing when NOT to fix a failure.** specific_39 ("the SPIKE paper") is
  five unrelated papers sharing an acronym. I verified the gold via the public
  S2 API, concluded it's a name-collision query where any fix would damage the
  path scoring 1.000 elsewhere, and documented the decision instead of coding.

## What was challenging

- **A 2300-line agent.** Reading it whole is impossible in budget; grep for
  the constant or callsite, and lean on the docstring changelog (which
  previous iterations maintained well — keep doing it).
- **14-query batches are noisy.** iter12 scored 45.8 on one batch and 40.0 on
  the next with the same code. Solo-win/loss tables churn. Only changes
  justified by mechanism (measured bug, closed loop, calibrated signal) are
  trustworthy; "it won this batch" is not.
- **Deciding how much credit to claim.** After the refutation I rewrote
  reasoning.md to downgrade the iteration from "step change" to "modest gain."
  Honest sizing matters because the next instance reads reasoning.md as
  evidence about what works.

## Tools

- Filesystem + jq + python one-liners over the iteration dirs covered
  everything. No missing tooling, though see process suggestions.
- Public S2 API works for analysis (the Standard-Tools constraint binds the
  agent, not the analyst) but the per-paper endpoint rate-limits instantly —
  use the POST /paper/batch endpoint.
- The 25k-token Read cap on agent.py means truncated views lie; grep first.

## What I would do differently

- Run the calibration BEFORE writing the implementation, not after. I built
  `_cov_score` into `_key2`, then tested, then ripped it out. Reversed order
  would have saved a third of the session.
- Budget more time for pool-composition analysis. Recall binds the harmonic
  mean 3–15x on every query; the least-exhausted lever is what's IN the pool,
  not how it's ordered or evidenced. I added conjunction retrieval late and
  cheaply; a future iteration should study which gold papers (per
  known_to_be_good + Perfect verdicts) were never retrieved at all and by
  which query type they'd have been found.

## Insights about the evolution strategy

1. **Negative results must be persisted loudly or they get rebuilt.** The
   dead `references` field was rediscovered across three iterations. I put the
   cov-score refutation in the agent docstring, in `_key2`'s comment, in
   reasoning.md, and in a smoke-test regression guard. Prompt-level fix:
   maintain a cumulative `REFUTED.md` in the experiment root that each
   iteration must read and may append to — cheaper than four redundant
   inline warnings.
2. **The plateau pattern (44–46 for three iterations) broke only via bugs and
   calibration, not strategy.** At this maturity the agent's ideas are mostly
   fine and its plumbing/assumptions are where the points are. Audit data
   flow (what exact string does each stage read?) and audit signals (does
   this proxy correlate with the judge?) before inventing.
3. **Cost is a cliff being surfed.** iter13 passed by $0.0006; iter11/12 both
   paid penalties. Every iteration should end with an explicit cost model
   ("this change adds N tokens × M calls × price") and target ≤$0.055, not
   ≤$0.06. Half a point per round has been leaking here.
4. **Grade-2→3 conversion remains the only big lever on semantic queries**,
   and it is now known that (a) internal graders under-predict Perfect,
   (b) lexical proxies don't track the Perfect/Highly boundary, (c) evidence
   was structurally wasteful. What remains untested: whether *any* evidence
   change flips real judge verdicts. A future iteration could measure this
   directly — take iteration-N papers judged Highly, rebuild their evidence
   under the new scheme, and diff — using cached diagnostics, before paying
   for a full eval.

## Process/prompt suggestions

- **Add to the evolution prompt: "Before adding any new ranking or grading
  signal, validate it against prior iterations' judge verdicts
  (submission.json + judge_verdicts.md + gold_criteria.md). Ship the
  calibration script with your iteration."** This is the highest-leverage
  procedural rule discovered in 14 iterations.
- Add a standing `REFUTED.md` / `FACTS.md` at the experiment root (dead
  references field, K distribution, grade-3 threshold behavior, lexical
  proxies refuted, SPIKE-style ambiguous specifics exist) so successors load
  established facts in one read instead of re-deriving them from 13 dirs.
- The error_analysis_report could include, per semantic problem, the judge
  grade histogram (Perfect/Highly/Somewhat/Not counts). It's the single most
  diagnostic number and currently requires per-problem spelunking.
- Encourage successors to reuse `calibrate.py` and prior iterations' analysis
  scripts (iteration dirs are scratch-friendly; the scripts survive).
