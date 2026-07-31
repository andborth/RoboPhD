# Evolution Reflection — Iteration 015 (PaperFindingBench)

Session: 14 prior iterations, 3 agents (49.9 / 48.7 / 48.2), 14 problems
(9 semantic / 2 metadata / 3 specific), 60-minute cap, Round 1 only. Produced
`iter15_verdict_repair`. Split: ~28 min analysis, ~14 min patching, ~6 min
validation, ~12 min write-up. Diff against iteration 13: ~140 lines, one new
pipeline step, zero new LLM calls.

---

## 1. The finding I'd most want the next session to inherit

**Join `score_meta.json` to the judge-label histogram and look at the grade-2
column. It is the biggest addressable bucket in the dataset and no session had
looked at it.**

```
query        K   g3   g2   g1   g0   rank  recall  score
semantic_91 100   31  [64]   5    0  0.644  0.310  0.418   <- 0 not-relevant!
semantic_110 228 112  [57]  15   44  0.946  0.491  0.646
semantic_101  94   23  [30]  34    7  0.721  0.244  0.365
semantic_57   12    3   [7]   2    0  0.938  0.250  0.394
                        ^^^ 229 across the batch, against 419 grade-3
```

`recall` counts grade 3 only, and grade 3 needs `weighted > 0.99` — essentially
every weighted criterion Perfectly Relevant. So **a grade-2 paper is one the
judge found on-topic but could not see ONE criterion demonstrated in the evidence
text**, and it pays exactly zero. semantic_91 had *zero* not-relevant papers in
100 judged: retrieval was already right, and 100% of the missing 0.69 recall was
evidence that failed to show a criterion.

The one-line recipe:

```bash
for p in problems/semantic_*; do
  jq -r '"K=\(.k_estimate) g3=\(.grade3_in_top_k) rec=\(.recall)"' $p/score_meta.json
  jq -r '[.papers[].label] | group_by(.) | map("\(.[0])=\(length)") | join(" ")' \
     $p/judge_verdicts.json
done
```

Iterations 12–14 all named the K-sorted score list as their highest-value cut.
It is better with the label histogram beside it: K tells you *where* the loss is,
the histogram tells you *what kind* it is. grade-0-heavy (semantic_193: 81 of
180) is a retrieval failure; grade-2-heavy (semantic_91) is an evidence failure;
they want opposite fixes and the K column alone cannot tell them apart.

## 2. Validate the proxy before you optimise it — it takes five minutes and it
##    killed five iterations of work

Iteration 13's reflection said "an agent-computed proxy that has never been
validated against the recorded judge labels is a hypothesis, not a metric" and
asked someone to correlate it. Nobody had. I did, over 1047 recorded verdicts,
and the agent's own printed headline metric — lexical criterion coverage,
described in iteration 11's source as *"the number to move next iteration: it
upper-bounds grade-3 count, which is the whole of recall"* — is nearly flat:

```
uncovered criteria = 3  ->  P(perfectly_relevant) = 0.37   (n=794)
uncovered criteria = 2  ->  P(perfectly_relevant) = 0.48   (n=203)
uncovered criteria = 1  ->  P(perfectly_relevant) = 0.53   (n= 45)
```

0.16 of spread, 76% of papers in one bucket. Iterations 9–14 built a
criterion-harvest pass, a single-paper repair pass, and three digest budgets on
top of it; on semantic_91 the repair pass alone burned **355 s of wall clock**
(32 of 163 probes timing out at 100 s) selecting targets with this signal.

The mechanism is obvious once stated: `_covers` is stem overlap, and *"long-range
dependencies between words"* shares no stem with *"non-local syntactic
relations"*. It misses real gaps and invents fake ones.

**Concrete cost of the code being twenty lines away from its own validation set:**
five iterations. The validation is ~15 lines of Python joining `submission.json`
to `judge_verdicts.json` to `gold_criteria.md`, importable straight from the
agent module with a 6-line `model_registry` stub. Do it for every proxy the agent
prints before you move it.

## 3. Look for a signal the agent already computes and discards

Iteration 14's rule was "find where the agent estimates the wrong quantity, not
where it estimates the right quantity badly." The sibling rule that paid here:
**grep the pipeline for information that is computed, collapsed to a scalar, and
thrown away.**

`_judge_evidence` prompts for one digit per criterion, `_parse_judge` returns the
vector, `_weighted` collapses it, and the components are dropped on the floor.
That vector is a model's read of *the same text the benchmark judge reads*, on
*the axis that decides grade 3* — a far better answer to "which criterion is this
paper short on" than stem overlap, and it costs nothing because the grade pass
already ran. Acting on it is `snippet_search` calls, which are free.

Adding it was an optional `vec_out: dict | None = None` out-param: return type
and all four call sites unchanged. **When you need a second output from a
function with several call sites, a caller-owned dict beats changing the return
arity** — iteration 13 had to fix three call sites for exactly that.

Generalised: the highest-leverage changes in a mature pipeline are not new
components, they are **re-plumbing existing ones**. Free, cheap to validate,
small blast radius.

## 4. Cost is binding, and the free-lever framing is now the whole game

Iteration 14 said this; it held completely. iter13 sits at $0.0471 against a
$0.063 threshold, and its semantic queries alone run $0.073–0.079. My entire
change is **$0.00 of new LLM spend** — reused digits plus free tool calls, funded
by cutting wall clock from a pass I had measured to be near-useless.

The practical rule: **before designing, list which of your candidate changes cost
LLM tokens and which cost only wall clock.** Wall clock has ~600 s of real slack
(745–1158 s observed against 1740 s); the LLM budget has ~$0.005. Tool calls,
prompt wording, ordering arithmetic, estimator choice, and *reallocating an
existing pass's targets* are all free. Adding a grader is not.

Corollary that made my change fit at all: **fund a new pass by shrinking a
measured-useless one, not by adding on top.** That keeps wall clock flat and
makes the trade auditable in `reasoning.md`.

## 5. The validation pattern, now four-for-four

Iterations 12–14 each converged on this; I ran it and it caught nothing, which is
the point — it is cheap insurance on a 2100-line base you are patching blind.
`smoke_test.py` is left in the iteration dir. Four assertion classes:

1. **CONTROL** — `inspect.signature`/`getsource` proving the *base* lacks the new
   signal. Turns "I think the old code was missing this" into a recorded fact.
2. **FROZEN** — `_build_evidence` byte-identical across old and new on **108 real
   submitted evidence strings** reconstructed from `submission.json`, plus
   `getsource` equality on 8 shared helpers. Iteration 12 paid 3.6 points for
   skipping this. Real artifacts, not synthetic fixtures — iteration 13's rule.
3. **LOGIC** — a decision table for the new selector instantiated with the
   constants the code actually uses (`[9,9,9]`→no probe, `[2,1,0]`→no probe,
   `[9,9,4]`→probe c3, `[4,9,5]`→heaviest gap first). Iteration 14's warning
   about illustrative examples using the wrong constants applies here.
4. **BUDGET** — assert the deadline ordering leaves ≥500 s before the 29-minute
   timeout, and that the pass you claimed to shrink actually shrank
   (`NEW.REPAIR_MAX_CALLS < OLD.REPAIR_MAX_CALLS`). Ten seconds, catches the
   copy-paste error where you add a pass and forget to fund it.

Total: ~90 lines, ~6 minutes including debugging. Every session should ship it.

## 6. `tool_probe.py`

Used once, deliberately, and it paid — iteration 14's rule ("probe whenever you
are about to attribute a zero to a corpus property") generalises to **probe
whenever your change depends on the corpus returning something useful you have
not seen it return.** My whole change assumes a scoped single-paper
`snippet_search` on a criterion probe returns text that demonstrates the
criterion. One probe on a real grade-2 paper confirmed it returns four verbatim
body passages the title+abstract evidence never had. ~90 seconds.

Usage notes (iteration 14 reported the first two; confirming, plus one more):
- ~8 lines of `DEPRECATED`/circular-import noise on stderr before any JSON →
  `2>/dev/null`.
- Output is **concatenated top-level JSON objects, not an array**; `jq` chokes.
  A `json.JSONDecoder().raw_decode` loop is the reliable parse.
- `timeout` is not on PATH on this macOS box.

## 7. What I'd do differently / left on the table

- **I skipped the metadata path almost entirely** and it is ~14% of the draw with
  both queries at 0.000. I added a cheap non-empty guard (`metadata_33` had
  `pool=0` and *all three* fallbacks re-slice an empty pool — the last resort has
  to be a fresh retrieval, which none of them was). But `metadata_26`-class
  "cites A and B" queries are still structurally unreachable, and iteration 14
  already handed the next session the fix: intersect from the *less*-cited
  paper's citer list and check each candidate's `references`. That is a
  well-specified ~40-line change sitting unclaimed for two iterations now.
  **If you are the next session and semantic is saturating, take it.**
- **Ranking by "score mass movable per minute" worked** (iteration 14's §9
  advice). The grade-2 column was on screen at minute 12 and I went straight at
  it. I did *not* re-run the K-sorted table from scratch or sweep cross-iteration
  scores — both correctly skipped, on three sessions' advice.
- **`error_analysis_report.md` was again the least useful artifact.** Fifth
  consecutive session to say so. `score_meta.json` + `judge_verdicts.json` +
  `submission.json` + `gold_criteria.md` answered everything.

## 8. Requests for the harness (with running counts)

1. **`error_analysis_report.md` columns**: per-query `K` / `rank` / `recall`
   (4th request — highest-value cut for iterations 12, 13, 14 *and* 15
   independently) **and the judge-label histogram beside it** (8th request; §1
   explains why the pair is worth more than either alone). Both are `jq`
   one-liners over files that already exist. Four sessions have now hand-rolled
   this in their first ten minutes.
2. **A proxy-validation helper** (new, and I think now the highest-value item):
   `correlate(agent, proxy_fn)` joining `submission.json` → `judge_verdicts.json`
   → `gold_criteria.md` and printing P(label | proxy bucket). §2 shows five
   iterations of work built on an unvalidated proxy; ~15 lines would have caught
   it at iteration 10.
3. **A replay helper** — `load_submission(agent, problem)`, the `model_registry`
   stub, and `load_agent_module(name)` for the side-by-side import (4th request).
   Every session hand-rolls it; mine is in `smoke_test.py` if you want to lift it.
4. **A same-position label-diff report** (3rd request, from iteration 13).
5. **Record in CLAUDE.md**: the grade-2 bucket and its interpretation (§1); that
   grade-0-heavy vs grade-2-heavy prefixes want opposite fixes; the probe's
   stderr noise and concatenated-JSON format (2nd request); that
   `grade3_in_top_k == grade3_at_full` is tautological (6th).
6. **Raise queries per iteration, or report a bootstrap CI** (11th request). This
   draw: 14 problems, 3 saturated at 1.000 for every agent, 2 at 0.000 for every
   agent — **effective n = 9** — and the top-two gap was 1.2 points. Sessions are
   being selected on differences the harness cannot measure. This is the single
   biggest threat to the loop and it has been raised in every reflection since
   iteration 5.

## 9. One structural observation about the evolution strategy

The reflection chain is doing more work than the artifacts. Reading reflections
13 and 14 first (~7 minutes) handed me the derivative ratio, the frozen-evidence
rule, the proxy-trap warning, the probe rule, the cost-is-binding finding, and
two disproved dead ends I would otherwise have re-explored. That is the highest
return per minute available in this setup, by a wide margin.

But the chain also **propagates unexamined claims**. The lexical-coverage proxy
survived five iterations partly *because* each reflection passed it forward as
"the number to move." Iteration 13 flagged it as unvalidated; iterations 13 and
14 both then declined to validate it and optimised elsewhere. **Read the
reflections for leads, but treat any inherited number as a hypothesis until you
have joined it to `judge_verdicts.json` yourself.** The one claim I checked was
the one that had been load-bearing for five iterations, and it was wrong.
