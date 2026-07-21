# asta_ds1000 / v0_0_7_sharp_cap_0_003

RoboPhD evolution run on AstaBench's DS-1000 task. Headline submitted agent: **`iter12_strong_toplevel`**. Run id: `robophd-asta_ds1000-044`. Evolved by **Claude Opus 4.8** (the evolution model; not a solver).

**This is the sharp-penalty experiment.** Same run configuration family as v0_0_6 (budget 750, examples-per-iteration 20, Deep Focus off, same standard seed) with two deliberate changes: the evolution model reverts to the opus-4.8 default, and the cost-breach penalty is **3.33× sharper** — `cost_threshold=0.003` is unchanged, but `cost_per_error` drops 0.001 → **0.0003**. The motivation comes directly from the v0_0_6/043 postmortems: the penalty is `(mean-cost overage / cost_per_error)` error-equivalents (`examples/asta_ds1000/evaluator.py:721`), and at `cost_per_error=0.001` a rational optimizer *buys through* the cap — the interim run `robophd-asta_ds1000-043` (first honest-accounting $0.003 run, not submitted) landed at $0.00342 (+14% over cap) @ 0.7722, its $0.00042 overage costing only ~2.1 pts per 20-problem batch. At 0.0003 the same overage costs ~7 pts, more than the accuracy edge it buys.

Outcome: **the target was hit** — internal 900-test **0.7333 (660/900) @ $0.001677/problem**, which strictly Pareto-dominates the board's ReAct/GPT-5-Mini frontier point (0.710 @ $0.003 true cost): +2.3pp accuracy at 44% lower cost. That is the point v0_0_6 targeted and missed.

What's distinctive about this submission:

- **The cap stopped binding.** The prediction was cap-hugging at ~$0.0028–0.0029; instead evolution retreated to 56% of the cap. The sharp penalty didn't squeeze the previous architecture under the boundary — it changed the equilibrium. The lineage measured the spend-more alternatives as net-negative: verify/self-consistency/reasoning-escalation catch nothing because wrong DS-1000 answers execute cleanly (the scorer compares the exact target *value* — dtype/shape/order/index — so crash-verification is blind), and heavy prescriptive preambles measurably hurt (a 5-rule guide dropped mini to 40%; a contract preamble caused IndentationErrors). With marginal spend between $0.0017 and $0.003 buying nothing, the optimum is one strong call plus free deterministic repair.
- **Single strong one-shot + deterministic extraction hardening.** One `gpt-5.4-2026-03-05` generation under a genuinely tiny output-contract preamble (`gpt-5.4-mini` as on-error fallback), then three strictly-additive, format-only fixes that can only fix or no-op, never alter the model's chosen answer: (1) `html.unescape` — strong models HTML-escape `<` as `&lt;` inside `<code>` tags → SyntaxError with correct logic; (2) function-body reindent — the required indent is read from the prompt's indented `### BEGIN SOLUTION` marker and padded, fixing stochastic bare top-level `return` IndentationErrors; (3) target-variable assignment — when the skeleton shows a `NAME = ...` placeholder and the model emits a bare value-expression with no top-level assignment, prepend `TARGET = ` (this iteration's refinement: kwargs/dict-literals no longer suppress the fix). 302 lines total.
- **A clean two-arm incentive-design result.** Same seed (byte-identical standard), same config, n=1 per arm: `cost_per_error=0.001` → $0.00342 @ 0.772 (run 043, bought through the cap); `cost_per_error=0.0003` → $0.00168 @ 0.733 (this run, retreated far under). The sharper constraint cost ~3.9pp of accuracy for 51% cheaper inference. Design lesson: on batch-noisy costs, sharp soft-cap penalties behave like walls with a wide safety margin, not tight targets.

The `sharp_cap_0_003` tail names the training free-zone and penalty variant: `cost_threshold=0.003` with the sharpened `cost_per_error=0.0003` (v0_0_6's `soft_cap_0_003` = same threshold at the default-scaled `cost_per_error=0.001`).

## First post-fix parity check

This is the first submission whose internal costs were measured **entirely on the leaderboard's billing basis** — Gemini reasoning-token folding (`dde6356`) and litellm 1.88.1 bundled-map pricing (`b74f8e7`) were both live for the whole run. The official cost should therefore replicate the internal **$0.001677** up to Ai2-side litellm version drift, unlike v0_0_6 (internal $0.00294 → official $0.00428, +46%). Both solver models are OpenAI (`gpt-5.4-2026-03-05`, `gpt-5.4-mini`) and cleanly priced in the bundled map; OpenAI reasoning tokens are already inside `output_tokens`, so no folding subtlety applies. Whatever the official number is, the internal-vs-official delta is the end-to-end verdict on the accounting-parity fixes.

## Leaderboard score

**To be submitted as `RoboPhD`.** Official `astabench score` run: **pending**.

| | Official (`astabench score`) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | pending | 0.7333 (660/900) |
| Per-problem cost | pending | $0.001677 (post-fix leaderboard billing basis) |
| Wrapper fallbacks | pending | 0 / 900 |
| Submission name | `v0_0_7_sharp_cap_0_003` / form: `RoboPhD` | — |
| Pareto target | ReAct/GPT-5-Mini (0.710 @ $0.003 true cost): internally **strictly dominated** (+2.3pp at 44% lower cost). Also edges v0_0_6 (0.7322 @ $0.00428 official) on both axes. Our unsubmitted 043 point (0.772 @ $0.00342) is not dominated — the pair are adjacent frontier slots. | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (solver) | gpt-5.4-2026-03-05 (primary generator), gpt-5.4-mini (on-error fallback) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

Key stats (not required by the AstaBench form; recorded for our own tracking):

| Stat | Value |
|---|---|
| Evolved agent name | `iter12_strong_toplevel` |
| Lines of code | 302 (`agent.py`, single file) |
| Evolution model | claude-opus-4-8 |
| Cost target (`cost_threshold`) | $0.003/problem (training free zone; batch mean) |
| Cost-breach penalty (`cost_per_error`) | $0.0003 of mean overage per error-equivalent (3.33× sharper than v0_0_6's 0.001) |
