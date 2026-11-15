# Evolution reflection — iteration 20 (agent: iter20_year_steered_expansion)

## Status to keep in mind when reading this

The agent is `iter19_backdated_citation_counts` with one metadata-path change. The semantic and
specific solvers are unchanged since iter18.

**Change:** for requests bounded to an explicit year list or a range of at most four years, the
metadata expansion issues each topic query a second time with a year appended (`"<topic> 2022"`).
All plain queries are kept and launched first; up to 100 year-suffixed variants follow with a
45 s timeout. The keyword-only metadata path gets up to 40 such variants.

| Check | Result |
| --- | --- |
| pyflakes, `ast.parse` | clean |
| `year_variants` on a toy list | output as intended |
| Year steering, probe 1 (`"neural networks 2023"`, NeurIPS filter) | 82 of 100 hits from 2023, 4 s |
| Year steering, probe 2 (`"masked language model representation learning 2022"`) | 34 of 100 from 2022, 15 from 2023; over 2 minutes |
| Year steering, probe 3 (`"graph neural networks 2022"`) | 125 s, no hits (error or empty; stderr was discarded) |
| The same keywords without a year (baseline) | **not run** |
| Any agent run with the new code | **none** |

The evidence for the change is weak: one clean probe, one weak and slow, one failed. Treat the
change as a hypothesis the next iteration's metadata results will test, not as an established gain.

## What worked well

- **A per-problem table from `result.json` + `score_meta.json` first** (score, cost, wall-clock,
  rank, recall, K, grade counts). One shell loop; it showed in a minute that semantic rank is
  already 0.86–0.99 and recall is the loss.
- **Comparing iter18 and iter19 on semantic queries as a noise measurement.** They share the
  semantic code, yet per-query scores differ by up to ±0.17. That stopped me from tuning a
  semantic prompt against a 10-query batch.
- **Probing the missed gold ids of `metadata_42` with `get_paper_batch`.** All 20 probed passed
  every filter (NeurIPS, 2022/2023, 46–6,397 citations, 4–22 authors). Together with the log line
  `expansion: 3633 found, 417 pass numeric filters` this points to a candidate shortage — though
  the log records counts, not ids, so "never candidates" is an inference.
- **Keeping plain queries first after the second probe came back slow**, so the worst case is
  iter19's candidate set plus a delay.

## What was challenging or time-consuming

- **The session ran right up to the 60-minute cap.** About 25 minutes went on semantic analysis
  that produced no change (criteria comparison, `semantic_145`, large-K queries).
- **The search backend was slow during the session.** Two of three probes took over two minutes.
  I launched them late, they went to the background, and the last result arrived after I had
  already written "ROUND 1 COMPLETE".
- **I discarded stderr on the probes** (`2>/dev/null`), so the failed third probe is unexplained.
- **No in-session agent run**, as in every earlier session.
- **The agent file is ~1,790 lines.** I read the planner prompt, the grader prompts and
  `solve_metadata` lines 1,400–1,700 only.

## What I got wrong or left open

- **No baseline for the steering probes.** I never ran the same keyword without the year, so I
  cannot say how much the year token improves the in-year share, nor whether it causes the
  slowness. The "11 % pass" figure I compared against includes citation and author filters and is
  not a like-for-like baseline.
- **The first probe used a two-word keyword; the agent's topic queries are 2–6 words.** The
  longer probe steered much less (34 %). The first probe is the less representative one.
- **Verification time is the likely binding constraint and I did not address it.** Reference-list
  verification ran at under one list per second on `metadata_42` (417 lists in 460 s, 154
  `get_paper_batch` failures, mostly `'NoneType' object is not iterable` forcing bisection). More
  candidates only help if they get verified.
- **Worst-case added delay is about 190 s** (100 variants, 24 in flight, 45 s timeout). On
  `metadata_42` the run ended at 990 s against a 1,260 s soft budget, so most of the slack could
  be consumed.
- **`semantic_145` (score 0 for all agents) is unsolved.** K = 6, all six judged papers "Highly";
  gold has an "Earliest instance" criterion (weight 0.2) and the two known-good papers were at
  positions 59 and 63.

## Observations worth carrying forward

- **Only the first K submitted papers are judged** (`scored_depth_cap` in `judge_verdicts.json`
  equals `k_estimate`). On small-K queries (6, 12, 14, 24 in this batch) the first handful of
  positions is the whole score, and if all K grades are equal the rank term is 0.
- **Known-good papers are graded Perfect without a judge call** (`status: known_good`), so they
  score even when a criterion cannot be shown from text.
- **For large-K queries, re-ranking deeper cannot raise recall much**: with 250 submitted and
  K = 222, nearly every candidate is already inside the scored depth. Recall there depends on
  candidates and evidence, not order.
- **The agent's criteria still diverge from gold in small ways** on most queries I read
  (`semantic_140`: an extra "evaluation of generated ideas" criterion; `semantic_203`: a strict
  "video domain" criterion where gold folds video into the task). I could not tie these to score.
- **Semantic cost is $0.27–0.35 per query**; the batch mean was $0.242 with 10 of 14 semantic.
  There is roughly $0.03–0.04 of mean headroom, not enough for another model stage.

## What I would do differently

- Run the steering probes in the first ten minutes, in pairs (with and without the year, same
  keyword, same venue), with stderr kept and each probe timed. Three pairs would have settled it.
- Decide on the path to work on within 15 minutes. The noise comparison was enough to rule out
  semantic prompt changes; I kept reading semantic logs after that.
- Log candidate ids at each metadata stage (the iter19 reflection asked for this too; I did not
  add it). Without it the next session again cannot join missed gold against candidates.
- Look at `batch_get` failures before adding candidates: fixing the `'NoneType'` error path may
  be worth more than any expansion change.

## Open questions for the next session

- Read the new log line `expansion queries: N -> M (years [...])` and the following
  `expansion: X found, Y pass numeric filters`. Did Y rise relative to 417 on a comparable query?
- How many `search_papers_by_relevance` failures does the metadata path log now, compared with
  25 before?
- Did verification still finish (`reference lists: a/b returned ... left=`), or did the added
  search time push it into the soft budget?
- Is the year-suffixed search slower than the plain one in general, or was the backend slow on
  the day?

## Suggestions for the prompt / process

- **State in the background doc that only the first K results are judged on `semantic_f1`** and
  that known-good papers are auto-Perfect. Both are visible only in `judge_verdicts.json`.
- **Report run-to-run variance.** A second run of the unchanged best agent each iteration (or a
  note of which results are cached) would tell sessions what size of difference is signal.
- **Allow one in-session smoke run on a single query.** This is the fifth reflection to ask.
- **Give `tool_probe.py` a clean-JSON mode and a printed elapsed time**, with errors on stdout.
- **Report per-type means and costs in `error_analysis_report.md`.**
- **Add per-criterion judge verdicts to `judge_verdicts.json`**; the "Highly, not Perfectly" gap
  cannot be diagnosed without them.
- **Document whether a year token in `keyword` is an intended ranking signal** if the tool owners
  know; it would replace three inconclusive probes.
