# PaperFindingBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's PaperFindingBench task (Literature Understanding category, Standard tools tier). Validation = 66 samples (training pool), test = 267 samples (held out).

Reference points (Standard-tools tier, the one we compete on): generic ReAct spans 0.220 @ $0.06/query (GPT-5 Mini) to 0.374 @ $3.38/query (Opus). Ai2's custom-interface Asta Paper Finder sits outside the tier at 0.397 @ $0.06 — the gap between generic ReAct and that system is the "PaperFinder-specific structure" evolution is trying to discover.

## Leaderboard submissions

Snapshots live in `example_runs/robophd/asta_paper_finder/<name>/`; the pipeline is `scripts/asta_paper_finder_submit.py` (official `astabench eval` + `score` + tarball for manual HF-form upload — see the script docstring for cost/prereqs). Patch numbers continue the cross-benchmark sequence shared with DS-1000.

| Submission | Agent (run) | Gate | Internal test | Official |
| --- | --- | --- | --- | --- |
| `v0_0_7_soft_cap_0_06_fable` | `iter12_body_conjunction` (`-003`, fable-5) | $0.06 | 0.3724 @ $0.0556 | **0.3749 @ $0.0533** |
| `v0_0_8_soft_cap_0_033_opus` | `iter9_rerank_rich_v1` (`-006`, opus-4.8) | $0.033 | 0.2754 @ $0.0060 | **0.2205 @ $0.0059** |
| `v0_0_9_cap_0_063_opus5` | `iter15_verdict_repair` (`-010`, opus-5) | $0.063 | 0.3839 @ $0.0533 | **0.3762 @ $0.0524** |
| `v0_0_9_cap_0_355_opus5` | `iter21_gold_rubric_and_hard_predicates` (`-011`, opus-5) | $0.355 | 0.4222 @ $0.2461 | **0.4318 @ $0.2506** |
| `v0_0_9_cap_0_355_fable` | `iter18_cocite_largegold_v1` (`-012`, fable-5) | $0.355 | 0.4383 @ $0.2780 | **0.4403 @ $0.2786** |
| `v0_0_9_cap_0_063_fable` | `iter14_title_channel` (`-013`, fable-5) | $0.063 | 0.3874 @ $0.0583 | **0.3895 @ $0.0581** |

Five of the six hold frontier slots; `v0_0_7` is displaced by `cap_0_063_opus5`,
which is both higher-scoring and cheaper. Asta Paper Finder's 0.397 @ $0.063 is
the one non-RoboPhD entry still on the curve, and `cap_0_063_fable` now sits
just under it at 0.390 @ $0.058 — cheaper, lower-scoring, so both stay.

**The four `v0_0_9` entries form a completed 2×2** of {opus-5, fable-5} ×
{$0.063, $0.355} on one frozen stack. The evolution-model delta to fable-5 is
**+0.0085** at $0.355 and **+0.0133** at $0.063 — same sign in both cells, so
the "fable-5 builds better agents" reading that the $0.355 A/B supported does
survive the cheap cell. n = 1 per cell and both deltas sit inside their runs'
~0.018 stderrs, so consistency of sign is the claim, not an effect size. The
category signs do **not** replicate: the $0.355 cell is a three-way trade
(semantic −0.0184), while $0.063 is positive in all three. See
`example_runs/robophd/asta_paper_finder/v0_0_9_cap_0_063_fable/README.md`.

> **Correction (2026-08-08).** Until the re-run this table carried
> `cap_0_063_fable` at 0.3638 @ $0.0481, and this paragraph reported the
> cheap-gate delta as **−0.0124** — opposite signs, concluding the $0.355 A/B
> "does not survive the cheap cell". That first official run was degraded —
> agent spend at 82.6% of internal, episodic, **cause never established** — and
> has been superseded. The reversal was an artifact of it and is withdrawn.

Of the first two entries, v0_0_8's internal-to-official transfer was **not** clean, unlike v0_0_7's: −0.055 overall, of which `specific_f1` contributes −0.036 (65%) and `semantic_f1` −0.020 (36%). `specific` is the part needing explanation — it's the larger share, and it's a metric the judge never touches, so no judging-basis story is available; three runs of the same agent show it's systematic rather than variance (snapshot README has the analysis). `semantic`'s share is real but confounded with the capped→uncapped judging change. Don't read v0_0_7's clean transfer as a general guarantee that internal predicts official.

Those first two are distinct Pareto points, not successive attempts: v0_0_7 buys score, v0_0_8 buys price. v0_0_8's $0.033 training cost gate is, to the cent, the price of the **second**-cheapest point on the board's frontier (Smolagents Coder GPT-5 Mini, 0.172 @ $0.033 — the board rounds the display to $0.03). That was the harder target of the two cheap-end points, since the cheapest (Llama 4 Scout 17B, 0.070 @ $0.013) sets a lower score bar but a tighter cost one. Pinning the free zone at a competitor's exact price buys the cost half of a dominance claim by construction; evolution then came in 5.5× under the gate at $0.0059, which cleared $0.013 as well, so it dominates both.

```bash
pip install litellm==1.88.1
# Push the commit before the full run — astabench stamps the SHA into eval_spec.revision.
python scripts/asta_paper_finder_submit.py --only v0_0_8_soft_cap_0_033_opus --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_8_soft_cap_0_033_opus            # full (measured: $118.68, 1h32m)
```

## Setup

```bash
# 1. Install RoboPhD core dependencies (from repo root)
pip install -r requirements.txt

# 2. Install task-specific dependencies (astabench bundles inspect_ai + Asta MCP factories)
pip install -r examples/asta_paper_finder/requirements.txt
```

## Credentials

### Solver + judge LLM keys (all three required)

Evolution may pick any of nine solver models across three providers (see
`model_registry.py`), so the evaluator hard-requires all three keys at
startup. `OPENAI_API_KEY` is doubly required: the benchmark's GPT-4o
relevance judge scores every `semantic_f1` query.

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."   # or ANTHROPIC_API_KEY_FOR_ROBOPHD (preferred:
                                 # keeps Claude Code CLI on its own credentials)
export GOOGLE_API_KEY="..."
```

### `HF_ACCESS_TOKEN` — gated dataset

The validation/test splits live in HuggingFace dataset `allenai/asta-bench`, which is gated (approval is automatic once you accept the terms).

1. Have a HuggingFace account at https://huggingface.co/.
2. Visit https://huggingface.co/datasets/allenai/asta-bench and accept the conditions.
3. Generate a Read-type token at https://huggingface.co/settings/tokens.
4. Export it (and also as `HF_TOKEN`, which other parts of astabench expect):

```bash
export HF_ACCESS_TOKEN="hf_..."
export HF_TOKEN="hf_..."
```

Verify:
```bash
python -c "from astabench.evals.paper_finder.task import load_paper_finder_bench; \
           ds = load_paper_finder_bench('validation'); print(f'{len(ds)} samples')"
# expect: 66 samples
```

### `ASTA_TOOL_KEY` — Asta MCP corpus tools

The leaderboard's Standard tools tier requires the Asta MCP corpus tools, served by an AI2 endpoint (`https://asta-tools.allen.ai/mcp/v1`). Request a key via the form at https://share.hsforms.com/1L4hUh20oT3mu8iXJQMV77w3ioxm (linked from https://allenai.org/asta/resources/mcp). Rate limit: **10 requests/second per endpoint**; excess requests get HTTP 429.

```bash
export ASTA_TOOL_KEY="..."
```

`ASTA_TOOL_KEY` is **hard-required** — the MCP suite is the task's only retrieval surface, and both `main.py` and the evaluator's constructor error at startup without it. The MCP factory wraps every tool in astabench's `make_retry_wrapper`.

**That ladder is narrower than it looks, and agents DO see transient server errors.** `_is_retryable_error` (`astabench/tools/asta_tools.py:482`) retries only `{429, 529, 504}` plus `anyio.BrokenResourceError` — **500/502/503 are not in the retryable set**. And `make_retry_wrapper` short-circuits on `ToolError` *before* consulting retryability at all (`:601`, commented "ToolErrors are not retryable and usually mean the LLM did a bad request"), so a server-side failure arriving as a `ToolError` whose message reads `Internal Server Error` is re-raised on the first attempt and the ten-attempt ladder never engages.

Not hypothetical. It is the documented aggregate-contention signature — `tool_pacer.py`'s docstring records "windows of HTTP 500s in which every call to the endpoint fails" in run `asta_paper_finder_20260724_193339` iters 9-11, 100+ tool errors in one batch, exact-match queries zeroed — and it recurred in `asta_paper_finder_20260809_222409` iteration 5 at **162 / 75 / 139** errors across a single 14-problem batch, zeroing two exact-match queries that other agents scored 1.000 on. `tool_pacer` reduces how often the endpoint is pushed into that state; nothing in the stack retries this class once it happens.

## Running

```bash
# Quick smoke test (small budget)
python examples/asta_paper_finder/main.py --num-iterations 2 --evaluation-budget 20

# Full run with default config
python examples/asta_paper_finder/main.py

# With held-out test-set evaluation after optimization
python examples/asta_paper_finder/main.py --eval-test-set

# Other engines
python examples/asta_paper_finder/main.py --engine gepa
python examples/asta_paper_finder/main.py --engine autoresearch
```

### Multi-seed runs (`--seed-runs`)

Start from the winners of prior runs instead of `seeds/baseline`. Each
`LABEL=RUN_DIR` pair contributes that run's best-Elo agent, resolved from its
`checkpoint.json`, so the seeded agent cannot drift from what the run actually
produced. `LABEL` is a provenance tag you choose; the pool agent is named
`seed_<LABEL>`. Fresh runs only — seeds are fixed when a run starts and recovered
from the checkpoint thereafter, so `--resume` rejects the flag. Without it the run
seeds from `seeds/baseline` as the agent `baseline`, which is also the name to
pass to `--eval-agent` to baseline the seed.

```bash
# Seed from all four v0_0_9 winners (the {opus-5, fable-5} x {$0.063, $0.355} 2x2)
python examples/asta_paper_finder/main.py \
  --cost-threshold 0.063 \
  --seed-runs \
    063_opus5=example_runs/robophd/asta_paper_finder/v0_0_9_cap_0_063_opus5 \
    355_opus5=example_runs/robophd/asta_paper_finder/v0_0_9_cap_0_355_opus5 \
    355_fable=example_runs/robophd/asta_paper_finder/v0_0_9_cap_0_355_fable \
    063_fable=example_runs/robophd/asta_paper_finder/v0_0_9_cap_0_063_fable
```

Seeding more agents than `agents_per_iteration` (default 3) loses none of them:
untested agents have selection priority, so the 4th seed enters at iteration 2.
Don't raise `agents_per_iteration` to compensate — that widens every round-robin
for the whole run to buy what the selection ladder already does once.

Seeds far above the run's cost gate start pinned to the floor and stay there.
At `--cost-threshold 0.063` with the default slope, a $0.355-cell agent
(~$0.25/query) takes roughly 208 penalty points against a 100-point maximum
(see [Cost-penalty math](#cost-penalty-math)), so it loses every head-to-head
and stops being selected after its first appearance — it persists as
cross-pollination material for evolution sessions rather than as a competitor.
That is a reason to include such a seed, not a bug; `--cost-per-error` flattens
the slope if you want it competing.

## Model registry

Nine pre-resolved Inspect-AI Model handles live in `model_registry.py` (outside the candidate's `file_mapping`, which only contains `agent.py`), grouped by family into three tiers:

- OpenAI: `GPT_5_4_MINI`, `GPT_5_4`, `GPT_5_5`
- Anthropic: `CLAUDE_HAIKU_4_5`, `CLAUDE_SONNET_4_6`, `CLAUDE_OPUS_4_8`
- Google: `GEMINI_3_1_FLASH_LITE`, `GEMINI_3_5_FLASH`, `GEMINI_3_1_PRO_PREVIEW`

Evolved agents import handles and call `.generate(...)`; the provider/model strings stay outside the evolvable artifact, so evolution can't substitute an arbitrary model. The training cost penalty disciplines strong-tier overuse. The benchmark's relevance judge (`openai/gpt-4o-2024-11-20`, hardcoded in astabench) is deliberately NOT a handle — see "Agent-vs-judge cost split" below.

## Cost-penalty math

The iteration-aggregate score during training is:

```
errors_equivalent = max(0, mean_cost − $0.06) / $0.006
score = 100 · mean_F1 − errors_equivalent · (100 / n)
```

where `mean_cost` is the batch's mean **agent** spend (judge cost excluded) and `n` is the iteration batch size. One error-equivalent of penalty costs exactly one fully-wrong query of raw score, so the penalty lives in the agent's own currency (queries), not dollars. The free-zone width ($0.06) sits at the Standard-tier leaderboard reference (ReAct + GPT-5 Mini, ~$0.06/query), so an agent that matches those points' score while staying inside the zone wins on the cost axis. Above the threshold, the penalty is **unbounded** — a catastrophically expensive agent can score well negative, which is intentional.

### Writing `--cost-per-error` as a percentage

`--cost-per-error` accepts either dollars (`0.006`) or a **percentage of `--cost-threshold`** (`10%`), and its default is `10%` — not a fixed dollar figure. The reason is that the dollars which make a sensible penalty slope scale with the free zone they sit beside: the $0.02 that was a mild tiebreaker against a $0.12 threshold is a wall against a $0.033 one, and campaigns now move the threshold by 4× between runs. Recent runs had been re-deriving the ratio by hand anyway (v0_0_8 ran `$0.033`/`$0.003` = 9.1%; ds1000's sharp-cap arm ran exactly 10%), so 10% is the observed practice made into the default.

Conversion happens once, at the CLI boundary, against the *resolved* threshold — so on `--resume` a percentage is measured against the run's stored threshold, not this invocation's default. Everything downstream (the scoring function, `paper_finder_runtime` in `checkpoint.json`, the `${COST_PER_ERROR}` figure interpolated into the agent-facing docs) sees dollars only; the percentage form never reaches an agent or a stored record.

One input the relative default cannot serve: `--cost-threshold 0` (a legal threshold meaning "no free zone, penalize from the first cent") has no width to take a percentage of, so it must state its slope in dollars — `--cost-threshold 0 --cost-per-error 0.006`. The run stops with a message naming the threshold rather than the flag you did not pass.

Note this **sharpened the default** relative to the old flat `$0.02`: at the default threshold the slope is now $0.006, 3.3× steeper. ds1000's two-arm experiment on a 3.33× sharpening at a fixed threshold (`cost_per_error` 0.001 → 0.0003) cost ~3.9pp of accuracy for 51% cheaper inference, with evolution retreating well under the cap rather than buying through it — expect the same direction here.

The three knobs (`--cost-threshold`, `--cost-per-error`, `--cap-judge-to-estimate`) are independently tunable, resolved at run start, persisted into `checkpoint.json`'s `task_config.paper_finder_runtime`, and **immutable on `--resume`** (a disagreeing flag is a hard error). Exception: on `--eval-only`, `--cap-judge-to-estimate` is taken from the CLI (default on) rather than the stored value — test scoring is the eval-time caller's choice, and the mode used is recorded in `test_results.json`. The per-iteration `aggregate_explanation` (in `evaluation.json`) carries the resolved excess and error-count so failure analysis can read "good but expensive" off the page.

### Cost-penalty asymmetry

The training evaluator applies the formula above. The test evaluator (derived via `with_overrides(apply_cost_penalty=False)` in `main.py`) does not — test scores are raw mean F1 in `[0, 1]` for leaderboard parity.

### Agent-vs-judge cost split

On `semantic_f1` queries (73% of validation) the scorer runs a relevance judge over every returned paper, inside the same `inspect.eval` — so its token usage lands in the same log as the agent's. The evaluator splits by exact model ID against `JUDGE_MODEL_IDS` (both approved judges, so the split holds whichever is in force): judge usage → `other_cost_usd` (reported, never penalized — the agent doesn't control it); everything else → `agent_cost_usd` (the penalized bucket). The split is sound because the registry's model IDs never include a judge's, and agents can't mint arbitrary model IDs (the registry is the only sanctioned LLM surface). A module-load assert cross-checks astabench's `GRADER_MODEL_NAME` so an upstream judge bump fails loudly instead of silently misattributing cost.

### Relevance judges (`--training-judge`, `--test-judge`)

**Training and test judging are separate choices with separate defaults**, because they answer different questions: training wants the cheapest basis that ranks agents faithfully, a test eval wants the basis the leaderboard uses.

| Flag | Default | Scope |
| --- | --- | --- |
| `--training-judge` | `openai/gpt-5.6-luna` — the calibrated cheap basis | Training only. Run-immutable (persisted; a resume restores it), because a mid-campaign switch would contaminate Elo. |
| `--test-judge` | `openai/gpt-4o-2024-11-20` — astabench's official judge, **the only basis comparable to leaderboard results** | Held-out evals only (`--eval-test-set` / `--eval-only`). An eval-time choice, not run-immutable — a completed run can legitimately be re-scored on either basis, and the one used is recorded in `test_results.json`. |

So the default run gives you cheap training *and* an official-comparable headline with no flags at all. Choosing a non-stock `--test-judge` writes judge-suffixed result files carrying a `judge_note`, and every test eval logs its basis affirmatively — `(STOCK — official-comparable)` or `(NON-STOCK — not official-comparable)`.

These were one flag until 2026-07-28, which created an explicit-default-is-not-default trap: passing `--training-judge openai/gpt-5.6-luna` moved the *test* eval to luna, while relying on that same value as the default did not — identical training configurations producing different test bases depending on whether the default was spelled out. Two flags remove the trap by construction.

**The judge-prompt profile is derived, not selected.** `no-prose` is validated for luna and rejected for gpt-4o, so each judge has exactly one correct profile — `gpt-4o → stock`, `luna → no-prose` — and there is no `--judge-prompt` flag to mispair. The profile is still persisted in `paper_finder_runtime`, because it scopes the verdict cache and the test-result filename, and a resume restores the stored value rather than re-deriving it (so the one prose-luna campaign, and any checkpoint predating the knob, keep their original basis).

Calibration record:

| Candidate | Calibration vs GPT-4o (n=150, untruncated v0_0_7-lineage evidence) | Verdict |
| --- | --- | --- |
| gpt-5.4-mini (2026-07-17) | κ 0.63 (< 0.7 gate), **+24% Perfect-rate inflation** | FAIL |
| gpt-5.4-nano (2026-07-20) | κ ~0.52, severe deflation (credited 51% of GPT-4o's Perfects) | FAIL |
| **gpt-5.6-luna** (2026-07-20) | **κ 0.755**, Perfect rates 31.3% vs 32.7% (matched), 2/300 format repairs | **PASS** |

`openai/gpt-5.6-luna` is therefore the only non-stock judge either flag accepts — and, since the flip below, the training default. Why it exists: judging is the campaign's biggest cost line (the v0_0_7 campaign's training API spend was ~$164 — solver models $26.55 + judge $137.68, i.e. judging was ~84% of it — and the internal test eval added another $88.38 of judging; evolution-model usage is excluded here since it bills against a Claude Max plan, not API dollars), and luna prices at **$0.20/$1.20 per M** vs GPT-4o's $2.50/$10.00 — an order of magnitude cheaper at matched verdict fidelity (luna repriced 80% down on 2026-07-31, from $1.00/$6.00; campaign cost figures recorded before that date are on the old rates and are not restated). Agreement is evidence-style-dependent, so re-run `_check_judge_calibration.py` before trusting the gate on a materially different lineage.

Mechanics and invariants:

- Each flag governs exactly one phase — `--training-judge` never reaches a test eval, `--test-judge` never reaches training. Non-stock test results are written to judge-suffixed files (`test_results.judge_gpt-5.6-luna-noprose.json`) with a `judge_note` field, so they can never collide with, or be mistaken for, a stock GPT-4o evaluation of the same agent. **Official submissions always use stock GPT-4o** (stock astabench code in the submit pipeline), regardless of either flag.
- With a non-stock judge active, the evaluator installs a lenient output normalizer (`_judge_normalize.py`, shared with the calibration script): astabench's strict parser drops format-deviant verdicts as Not Relevant, and alternate judges deviate rarely but nonzero. Stock paths keep the strict parser untouched. Repair counts surface per-problem as `judge_format_repairs`.
- Verdict caches are judge-scoped (`shared_<judge>.json` / `shared_test_<judge>.json`), so verdicts from different judges never mix; the first luna campaign pays cold-cache judging.
- Judge pricing for models newer than the pinned litellm 1.88.1 comes from `evaluator.JUDGE_PRICE_OVERRIDES` (internal accounting only — agent-model pricing stays on the leaderboard's bundled-map basis, untouched).
- **Why luna's profile is `no-prose`.** It drops the judge's mandated `relevant_snippet`/`relevance_summary` output — prose the scorer never reads. Validated 2026-07-23 with a 3-arm study (stock×2 for a same-prompt rerun-noise floor + labels-only, 148 stored docs): luna's labels-only verdicts sat AT its noise floor (agreement 0.811 vs floor 0.838; Perfect drift +9% vs the floor's own +6%) while output tokens fell 65% → **$0.0022/verdict, ~5.7× cheaper than the stock GPT-4o basis** (measured at the then-current $1.00/$6.00 luna rates; after the 2026-07-31 repricing the same measurement is ~$0.00044/verdict, ~28× cheaper — the token reduction and agreement findings are unaffected). The same change is REJECTED for gpt-4o (+18.5% Perfect inflation vs a stable floor — snippet-writing is chain-of-thought for the older model), which is why the profile is pinned to the judge rather than chosen. A prompt variant is a distinct verdict basis, so it gets its own verdict-cache namespace (`…_noprose.json`) and test-result suffix and never shares files with prose-luna. Meta-finding from the same study: gpt-4o churns 22% of its own Perfect verdicts between reruns on identical inputs (luna: 9%), and luna matches gpt-4o's Perfect set (0.82) better than gpt-4o reruns match themselves (0.778).
- **Default-flip status: done (2026-07-28).** The blocking question was training *dynamics* under a cheaper judge, which the calibration gate — verdict agreement on fixed inputs — cannot answer. Run -006 answered it: a full 267-query stock-vs-luna pairing of the same agent came in at **+0.0136 overall (se 0.0091)**, and the cheap basis roughly quartered the training judge bill (**$30.08 vs -005's $128.32**). Fresh runs therefore train on luna/no-prose. For a few hours this flip did drag the *test* eval onto luna as well (one flag governed both phases), so a default `--eval-test-set` stopped being official-comparable; the `--test-judge` split later the same day removed that side effect, which is why the table above shows the two phases defaulting differently.

### Evidence char cap (`--evidence-char-cap`, default 2500)

Per-paper `markdown_evidence` length budget, **enforced during training only** (truncated before grounding/judging; test evals and official runs are uncapped — the evolved short-evidence behavior is what transfers). Why it exists: agents have no cost incentive toward brevity (judge spend is invisible to their reward), and the -005 lineage's uniformly bloated evidence (mean 3,391 chars/paper; p50 3,421 / p90 5,556) tripled official judging bills ($192 → ~$600). The 2,500 cap clips 66% of that lineage's papers for a −36.3% arbitrary char cut; evolved agents selecting to fit are expected to land ~−50%.

Gate experiment (2026-07-24, 148 stored iter13 docs, luna+no-prose basis, clip-at-cap vs full): **agreement 0.858 — above the 0.838 rerun-noise floor — with Perfect-rate drift −9%, inside the ±6–9% rerun band**; i.e. even *arbitrary* imposition of the 2,500 cap is statistically indistinguishable from judge noise (the cap shape trims only the longest evidence — much gentler than uniform truncation, which measurably destroyed Perfects at similar reduction). Clip-at-1850 perturbs beyond the floor (agreement 0.764) — don't tighten without re-gating. Clipping surfaces per-problem as `evidence_truncation.md`; the enforced contract line reaches agents via `${EVIDENCE_CAP_NOTE}` in background.md (present only when the cap is on). Run-immutable; `0` disables.

## Cost notes

- Asta MCP rate limit is 10 req/s per endpoint, shared across all workers. The seed's few sequential calls per query sit well under it at `--max-workers 8`, but evolved agents routinely fan out (run `20260716_072622`'s winner: up to 16 concurrent snippet calls per query; run `20260724_193339`'s champions launch 250-340 calls/query at concurrency 20), and sustained collective overrun surfaces as retry-backoff latency, leaked tool errors, HTTP 500 contention windows, and agent-side timeouts rather than clean 429s. Per-agent self-throttles cannot bound the aggregate — no agent can see the other workers — so the harness enforces a **global launch pacer** (`--tool-launch-rate`, default 8 launches/s per endpoint, cross-process slot state under `<runs_dir>/.tool_pacer/`; see `tool_pacer.py` and the gate `_check_pacer_stress.py`). The training transport also retries 500/502/503 (evaluator monkeypatch; stock astabench retries only 429/529/504). Both behaviors are documented to agents in background.md ("Tool-call transport"); leaked errors name their root cause (`evaluator._tool_failure_summary`) to make the signal legible. The pacer is also staged into leaderboard-submission wrappers (in-process mode) so official `astabench eval` runs — several samples concurrently in one process — respect the same budget. NOTE for run comparability: the pacer + 5xx retry are a harness change (2026-07-27); record `tool_launch_rate` when logging results, and treat pre-pacer runs' burst-related failures as a known environment difference.
- 73% of validation samples are `semantic_f1` queries that invoke the scorer's GPT-4o judge per predicted paper. **Judge cost typically dominates agent cost** for short pipelines. It's excluded from the penalty but still real dollars — budget accordingly.
- **The judge caches verdicts persistently**: astabench appends every judgement to a `detailed_reference.json` (by default *inside the installed package*) and replays it on later evals. `main.py` redirects that path via `PF_JUDGE_CACHE_PATH` to model-scoped files under `<runs_dir>/.judge_cache/`: training runs share `shared_<judge-slug>.json` across runs, and internal test evals (`--eval-test-set` / `--eval-only`) share a **dedicated** `shared_test_<judge-slug>.json` — separate from training's so every verdict in it came from a test eval, and safe because verdicts are keyed by (query, paper, evidence-hash) and scoped by judge model, so a hit only replays the stock judge's verdict on identical inputs. `--no-shared-judge-cache` opts out: training falls back to a per-run file; test evals fall back to a pristine empty per-invocation file, i.e. submission-exact fresh judging (the pre-cache behavior). The judging cap (`--cap-judge-to-estimate`, default on) likewise applies to both training and internal test evals; every `test_results.json` records `judge_model` / `judge_cache` / `cap_judge_to_estimate` so scores are never ambiguous about the basis that produced them. Upstream's cache writer is not multiprocess-safe (per-process lock, non-atomic write) and our parallel workers corrupted it — a torn file zeroes every eval, since astabench's reader raises through scorer init. The evaluator therefore monkeypatches `update_references` at import with a flock-serialized, atomic-rename writer (our processes only; official `astabench eval` submissions run stock code, single-process, uncapped, and fresh-judged — none of this machinery applies to a real submission). Consequences: judge cost per eval shrinks as coverage grows (a fully-cached sample bills $0 with a nonzero score), and repeated evals of the same submission are more deterministic than a fresh environment would be.

## Files

- `main.py` — `optimize_anything()` entry point
- `evaluator.py` — `PaperFinderEvaluator`; runs `inspect.eval()` on a 1-sample dataset per evaluation; cost split + iteration aggregator
- `_eval_worker.py` — subprocess worker (one per evaluation, for inspect.eval parallelism)
- `model_registry.py` — the nine solver-model handles (not in `file_mapping`)
- `seeds/baseline/agent.py` — minimal `@solver` factory exported as `make_solver`. Demonstrates a tool call, a registry-handle LLM call with an empty-completion guard, and the JSON output schema.
- `_check_evaluator.py` / `_check_seed.py` — credentialed sanity gates: judge-cache integrity, gold-ID leak scan over the evolution-facing artifacts, synthetic-candidate scorer checks, and a 3-sample live seed run with judge-split assertions.
- `_check_cache_stress.py` — key-less multi-process stress gate for the safe judge-cache writer (8 subprocesses × 50 updates; the process-level evidence the thread-based unit test can't provide).
- `objective.md` — what evolution should optimize (cost placeholders interpolated by main.py)
- `background.md` — task spec, score types, output schema, tools, model menu, cost-penalty table
- `unit_tests/` — aggregate math, judge-split soundness, registry invariants, doc-interpolation coverage
- `requirements.txt` — astabench (which bundles inspect_ai)

## Status

### Done

- [x] Scaffold matches existing examples (main.py, evaluator.py, seed, objective.md, background.md, requirements.txt)
- [x] Dataset loads (66 validation, 267 test) via astabench's loader
- [x] `inspect.eval()` runs end-to-end on a 1-sample dataset; sample-level errors surface cleanly into diagnostics
- [x] `ASTA_TOOL_KEY` acquired (2026-07-09); MCP tools verified live
- [x] ds1000-parity modernization: model registry, mean-cost penalty aggregator with train/test asymmetry, agent-vs-judge cost split, task_config_extras persistence, subprocess killpg machinery
- [x] Rate-limit hardening (2026-07-10): auto tool_source hard-errors without ASTA_TOOL_KEY (no silent search fallback); search kit wrapped in astabench's `make_retry_wrapper` for parity with the MCP kit
- [x] First full evolution run completed (2026-07-11, 16 iterations); recorded as non-comparable (task under active development) in `../robophd_runs/results/asta_paper_finder.non_comparable.json`
- [x] Scoring description corrected against the scorer source and pinned: plain per-query mean (not macro-by-group), semantic ordering/NDCG semantics incl. the uniform-grade zero, grade-3-only recall
- [x] Verified tool documentation in `background.md` from live-server probes: search matching semantics, return shapes, requestable fields, limits, snippet retrieval model
- [x] Seed reworked to the family criteria (short, MCP-only, empty-completion guards, semantic-list floor; 121 lines); sanity gates in `_check_evaluator.py` / `_check_seed.py`
- [x] EVAL_TIMEOUT 600s → 1800s (ds1000 parity — wall clock is a runaway backstop, not a criterion); default budget 600 (~20 iterations at observed cache-adjusted burn)
- [x] Judge-verdict surfacing: `judge_verdicts.md` per semantic problem lists the judge's verdict on every submitted paper in submitted order, sourced from astabench's persistent judgement cache (zero extra LLM calls). The most-requested diagnostic across the first run's reflections — separates recall misses from judge rejections.

### Code work (deferred)

- [ ] **Extract the pricing machinery to a shared module.** `_estimate_cost` / `_bundled_price_map` here are byte-for-byte copies of `asta_ds1000/evaluator.py`'s (the leaderboard billing basis is a cross-task concept, not task-specific). Both copies carry duplicated test suites (`test_evaluator.py`'s `_estimate_cost` section, ported from ds1000's `ae1e410`) so accidental drift fails loudly, but the right endpoint is one shared implementation — e.g. under `RoboPhD/`, with an import-fallback story for standalone evaluator use. Deferred until the ds1000 campaign is at a quiet point (extraction touches its evaluator).
- [ ] **Standard-Tools allowlist (AST scan).** The evaluator should reject candidates that import outside `{json, re, asyncio, dataclasses, ..., inspect_ai.*, model_registry}`. Without this, evolution could in principle introduce `import openai` and silently lose cost-accounting fidelity / the Standard Tools badge. ~30 lines of `ast.parse` walking.
- [x] **Submission pipeline: `scripts/asta_paper_finder_submit.py`** (ported from `asta_ds1000_submit.py`, 2026-07-19). Stages `agent.py` (resilience wrapper) + `agent_inner.py` + `seed_agent.py` + `model_registry.py`, re-runs official `astabench eval paper_finder_test` + `astabench score`, and tarballs *that* run's logs. Deltas from ds1000: no Docker, `ASTA_TOOL_KEY`/`HF_ACCESS_TOKEN` preflights, a litellm bundled-map pricing preflight over `AGENT_MODELS`, an uncapped-judging cost projection, `--limit N` smoke runs (log-isolated, never tarred), and a schema-valid empty submission as the wrapper's last resort (an empty string would route through the scorer's LLM re-parse). No seed-baseline entry: ASTA didn't post ds1000's seed on the leaderboard. Tests: `unit_tests/test_submit.py`. See "Leaderboard submissions" above.

### Design questions to revisit

- [ ] **Whether to add `paper_finder` (high-level MCP tool) to the allowlist.** Using it would make our agent an `ai2i_paper_finder` lookalike — defeats the purpose. Lean: exclude.

(Resolved: score-type stratification in Elo — the leaderboard headline turned out to be the plain per-query mean, `adjusted_f1_micro_avg`, so the Elo signal already matches it exactly; no stratification tension exists.)

(Resolved: held-out thermometer split — rejected as against the framework philosophy. Validation strategy belongs to the engines, not the task: GEPA/Autoresearch carve their own val split from the pool, and the RoboPhD engine's protections are per-iteration re-sampling, Elo, and the held-out test set. A task-side holdout would second-guess all three. The first run's tight train/test agreement supports leaving the full 66-sample pool with evolution.)
