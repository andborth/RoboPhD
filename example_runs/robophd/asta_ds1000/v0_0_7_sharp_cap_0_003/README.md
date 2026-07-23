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

**To be submitted as `RoboPhD`.** Official `astabench score` run completed 2026-07-23 (third attempt; see "Official-run incidents" below):

| | Official (`astabench score`) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | **0.7367** (663/900, stderr 0.0147) | 0.7333 (660/900) |
| Per-problem cost | **$0.001671** (± 0.000026) | $0.001677 |
| Wrapper fallbacks | 0 / 900 (6 samples used the agent's internal gpt-5.4-mini on-error fallback; 4 scored correct) | 0 / 900 |
| Submission name | `v0_0_7_sharp_cap_0_003` / form: `RoboPhD` | — |
| Pareto outcome | ReAct/GPT-5-Mini (0.710 @ $0.003 true cost) is **strictly dominated officially**: +2.7pp accuracy at 44% lower cost. Also beats v0_0_6 (0.7322 @ $0.00428 official) on both axes. Our unsubmitted 043 point (0.772 @ $0.00342 internal) is not dominated — adjacent frontier slots. | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

**Parity verdict — the accounting fixes hold end-to-end.** This was the first submission whose internal costs were measured entirely on the leaderboard billing basis (post `dde6356`/`b74f8e7`), and the official numbers replicated both axes: accuracy +3 problems (well inside stderr), cost within 0.4% ($0.001671 vs $0.001677). Contrast v0_0_6: internal $0.00294 → official $0.00428 (+46%).

## Official-run incidents (attempts 1–2, archived in the working dir)

- **Attempt 1** (2026-07-21): OrbStack VM (16 GB cap) OOM-crashed under the 6-sandbox load ~1 h in; 547 samples preserved in `logs/archive_attempt1_oom_20260721/`.
- **Attempt 2** (2026-07-22): completed 900/900 but scored 0.607 — the sandbox images had been built in the two minutes before attempt 1's crash, and the VM died before page cache flushed: matplotlib 3.10.0's `__init__.py`/`pyplot.py` were committed as zero-byte files, so the scorer's own `generate_test_case` crashed on 117 matplotlib problems. Diagnosed via per-sample join against the internal eval (cost parity exact; non-matplotlib flips balanced 25 vs 28). Archived in `logs/archive_attempt2_corrupt_image_20260722/`.
- **Attempt 3** (2026-07-22→23, this result): fresh images on a 32 GB VM; clean 900/900.

**Run-over-run variance (four evals of the same agent, per-sample join on the common non-matplotlib subset, n=757):** aggregate accuracy is tightly stable — internal 0.7173, attempt 2 0.7213, attempt 3 0.7186 (0.4pp spread) — while per-problem outcomes churn: ~7–8% pairwise disagreement, 84/757 problems (11%) flip in at least one pair, canceling almost perfectly in aggregate. The reported stderr (±1.5pp) is dominated by problem sampling, not run stochasticity; the +2.7pp margin over ReAct/GPT-5-Mini is ~5× the observed run-over-run spread.

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (solver) | gpt-5.4-2026-03-05 (primary generator), gpt-5.4-mini (on-error fallback) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

**Agent description (as submitted on the form):**

> 302-line agent evolved by RoboPhD from the 46-line baseline seed, with Claude Opus 4.8 as the evolution (but not solver) model, trained under an ultra-low cost budget: a $0.003/problem soft cap with a sharp overage penalty. Under that constraint, evolution converged on radical simplicity — it measured verification, self-consistency, and ensemble machinery as net-negative on DS-1000 (wrong answers execute cleanly, so crash-checking catches nothing) and discarded them all. The evolved agent makes a single gpt-5.4 call under a compact ~480-token instruction prompt (gpt-5.4-mini as on-error fallback; fired on 6/900 samples), then applies three deterministic, format-only repair passes — HTML-entity unescaping, function-body re-indentation read from the prompt's marker, and target-variable assignment for bare expressions — each of which can fix or no-op but never alter the model's chosen answer. Mean cost: ~$0.0017/problem.

Key stats (not required by the AstaBench form; recorded for our own tracking):

| Stat | Value |
|---|---|
| Evolved agent name | `iter12_strong_toplevel` |
| Lines of code | 302 (`agent.py`, single file) |
| Evolution model | claude-opus-4-8 |
| Cost target (`cost_threshold`) | $0.003/problem (training free zone; batch mean) |
| Cost-breach penalty (`cost_per_error`) | $0.0003 of mean overage per error-equivalent (3.33× sharper than v0_0_6's 0.001) |
