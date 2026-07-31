# PaperFindingBench relevance-judge: cost & robustness findings for the astabench team

Measurements from running PaperFindingBench evaluations at scale (several
full training campaigns + official test evals, astabench 0.5.4, July 2026).
Four findings, each independently actionable. Contact: RoboPhD project
(https://github.com/andborth/RoboPhD; submissions "RoboPhD" on the
leaderboard).

## 1. Judge cost is the dominant barrier to submitting, and it scales with evidence style

The scorer's GPT-4o judge bills the submitter during `astabench eval`. Two
of our submissions/candidates, identical protocol, measured:

| Agent lineage | evidence style | judge tokens/verdict | official judging cost (267 queries) |
| --- | --- | --- | --- |
| v0_0_7 (trained, $0.06 cost cap) | ~1,000 in / 165 out | $0.0040/verdict | **$192** (measured) |
| -005 candidate (trained, $0.12 cap) | ~2,100+ in / 350 out | $0.0124/verdict | **~$600** (projected from n=25 sample) |

Nothing in the task pushes back on evidence length (`markdown_evidence` is
unbounded and judging cost is invisible to the leaderboard's cost axis), so
richer agents make official evaluation itself several times more expensive
— for the submitter. At ~$600/submission, the Standard tier prices out the
independent researchers it seems designed for.

**Suggestions**, in ascending effort:
- Bless a calibrated cheaper judge (we have a full calibration dossier for
  `gpt-5.6-luna` vs your `gpt-4o-2024-11-20`: PERFECT/not kappa 0.755 on
  n=150 paired verdicts over real submission evidence, Perfect rates 31.3%
  vs 32.7%, and a 25-query full-pipeline A/B with score diff −0.027 ± 0.035;
  already shared with your team separately).

  **This got substantially stronger on 2026-07-31**, when luna repriced 80%
  down to $0.20/$1.20 per M against gpt-4o's $2.50/$10.00. The dossier's
  "~1/3 the judging cost" figure was computed at luna's old $1.00/$6.00; at
  the new rates the same measurements put it more than an order of magnitude
  below the stock judge. The $192 v0_0_7 submission above would judge for
  roughly $15, and the ~$600 evidence-rich case for roughly $50 — which
  changes the character of the barrier in §1 rather than just its size.
- Offer an AI2-side judging tier for submissions (predictions-only upload).
  The current process is already trust-based — submitters run the judge on
  their own keys and upload the resulting logs — so this centralizes cost
  without changing the trust model.

## 2. The judge prompt structurally defeats OpenAI prompt caching

`relevance.py` builds each judge call as
`prompt_template.format(criteria=doc_text)` where `doc_text` is the
`RelevanceJudgementInput` TypedDict — i.e. the prompt embeds the **Python
dict repr**, constructed `document` first. Consequences:

- The per-paper document text begins diverging immediately after the ~450
  shared instruction tokens — under OpenAI's 1024-token cache-eligibility
  prefix. Measured on our v0_0_7 official run: **3,328 cache-read tokens
  out of 46.2M judge input tokens (0.007%)**.
- Reordering the dict criteria-first is NOT sufficient: all of a query's
  judge calls are fired concurrently (`asyncio.gather`), so no request
  completes in time to write the prefix cache for its siblings. We measured
  the reorder alone: 1.0% → 1.8% cache reads.
- Harvesting the cache would need BOTH the reorder and a serialized
  prefix-warming first call per query (or staggered batches). Upside at
  GPT-4o's $1.25/M cached-input rate: roughly 30–40% off every submitter's
  judging bill, more for evidence-rich agents.

## 3. Most of the judge bill is prose the metric never reads

Per-verdict token profile (measured, gpt-5.6-luna on rich submission
evidence; GPT-4o proportions similar): ~1,400 input tokens vs ~535 output
tokens. At output:input price ratios of 4-6x, the judge's OUTPUT is
40-69% of total judging cost — and it consists almost entirely of the
mandated per-criterion `relevant_snippet` and the `relevance_summary`,
neither of which the scorer reads (only the relevance labels enter the
metric). We measured whether the prose is load-bearing, per judge model
(3-arm design: stock prompt twice for a same-prompt noise floor, plus a
labels-only prompt, 148 identical docs each):

- **gpt-4o-2024-11-20: the prose IS load-bearing.** Labels-only judging
  inflated Perfectly-Relevant verdicts +18.5% (54 -> 64), far beyond the
  rerun noise floor (54 -> 54). Do not drop the fields for the current
  judge without recalibrating the metric.
- **gpt-5.6-luna: it is not.** Labels-only stayed at the rerun noise
  floor on every measure (agreement 0.811 vs floor 0.838; Perfect drift
  +9% vs the floor's own +6%), while cutting output tokens 65% —
  $0.0022/verdict all-in, ~5.7x cheaper than the current stock judge on
  evidence-rich submissions. (Measured at luna's pre-2026-07-31 rates of
  $1.00/$6.00 per M; at the current $0.20/$1.20 the same measurement is
  ~$0.00044/verdict, ~28x. The token reduction and the agreement findings
  are rate-independent and unchanged.)
- Incidentally measured: gpt-4o is the noisier judge — on identical
  inputs it churns 22% of its own Perfect verdicts between reruns (luna:
  9%), and luna matches gpt-4o's Perfect set (0.82) better than gpt-4o
  reruns match themselves (0.778). Aggregate scores are stabler than
  per-verdict agreement suggests, but per-paper verdicts carry
  substantial judge noise.

Net recommendation: the affordable path is not trimming the current
judge's output (metric-shifting) but a calibrated modern judge with a
labels-only prompt — with any such change version-bumped.

(For completeness: we also measured the submitter-side alternative of
truncating evidence text — it trades badly. Halving evidence chars cut
input only 29% (prompt overhead is fixed) while destroying 21% of
Perfectly-Relevant verdicts, the only recall-earning grade.)

## 4. Judge verdicts are sensitive to prompt payload order (robustness finding)

While testing the reorder we found it is **not metric-neutral**: judging
the same 113 (criteria, evidence) docs with the same model under the two
dict orders gave 76.1% exact label agreement and shifted the
Perfectly-Relevant count from 35 to 42 (+20% — and Perfect is the only
recall-earning grade). Part of that spread is sampling temperature, but a
shift this size means the current prompt's dict-repr construction is a
load-bearing, presumably unintentional, detail of the metric.

**Suggestions**: pin the payload serialization explicitly (stable field
order, ideally clean JSON rather than dict repr), consider JSON mode for
the response (your strict parser currently drops format-deviant verdicts
as Not Relevant — we measure ~1% silent drops for non-GPT-4o judges, near
zero for GPT-4o), and treat any future prompt change as a scoring-version
bump.

## Appendix: verdict-cache writer under multi-process evaluation

`update_references` (detailed_reference.json writer) is not
multiprocess-safe (per-process lock, non-atomic write). Standard
single-process `astabench eval` is unaffected; frameworks that evaluate
samples in parallel worker processes (as ours does) corrupt the JSON —
the scorer then raises through init and zeroes subsequent evals. Our
mitigation (flock + valid-prefix recovery + atomic rename), stress-tested
at 400/400 concurrent updates where stock lost 387/400, is in
`examples/asta_paper_finder/evaluator.py` (`_safe_cache_rmw`) in the repo
above — a fix along those lines would harden the default for anyone
running parallel evals.
