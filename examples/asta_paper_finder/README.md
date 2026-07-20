# PaperFindingBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's PaperFindingBench task (Literature Understanding category, Standard tools tier). Validation = 66 samples (training pool), test = 267 samples (held out).

Reference points (Standard-tools tier, the one we compete on): generic ReAct spans 0.220 @ $0.06/query (GPT-5 Mini) to 0.374 @ $3.38/query (Opus). Ai2's custom-interface Asta Paper Finder sits outside the tier at 0.397 @ $0.06 — the gap between generic ReAct and that system is the "PaperFinder-specific structure" evolution is trying to discover.

## Leaderboard submissions

Snapshots live in `example_runs/robophd/asta_paper_finder/<name>/`; the pipeline is `scripts/asta_paper_finder_submit.py` (official `astabench eval` + `score` + tarball for manual HF-form upload — see the script docstring for cost/prereqs). Patch numbers continue the cross-benchmark sequence shared with DS-1000.

| Submission | Agent (run) | Internal test | Official |
| --- | --- | --- | --- |
| `v0_0_7_soft_cap_0_06_fable` | `iter12_body_conjunction` (`robophd-asta_paper_finder-003`, fable-5-evolved) | 0.3724 @ $0.0556/query | **0.3749 @ $0.0533/query** |

```bash
pip install litellm==1.88.1
python scripts/asta_paper_finder_submit.py --only v0_0_7_soft_cap_0_06_fable --limit 3   # smoke (~$3)
python scripts/asta_paper_finder_submit.py --only v0_0_7_soft_cap_0_06_fable            # full (~$200-270, 12-18h)
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

`ASTA_TOOL_KEY` is **hard-required** — the MCP suite is the task's only retrieval surface, and both `main.py` and the evaluator's constructor error at startup without it. The MCP factory wraps every tool in astabench's `make_retry_wrapper`, so agents never see transient 429/5xx.

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

## Model registry

Nine pre-resolved Inspect-AI Model handles live in `model_registry.py` (outside the candidate's `file_mapping`, which only contains `agent.py`), grouped by family into three tiers:

- OpenAI: `GPT_5_4_MINI`, `GPT_5_4`, `GPT_5_5`
- Anthropic: `CLAUDE_HAIKU_4_5`, `CLAUDE_SONNET_4_6`, `CLAUDE_OPUS_4_8`
- Google: `GEMINI_3_1_FLASH_LITE`, `GEMINI_3_5_FLASH`, `GEMINI_3_1_PRO_PREVIEW`

Evolved agents import handles and call `.generate(...)`; the provider/model strings stay outside the evolvable artifact, so evolution can't substitute an arbitrary model. The training cost penalty disciplines strong-tier overuse. The benchmark's relevance judge (`openai/gpt-4o-2024-11-20`, hardcoded in astabench) is deliberately NOT a handle — see "Agent-vs-judge cost split" below.

## Cost-penalty math

The iteration-aggregate score during training is:

```
errors_equivalent = max(0, mean_cost − $0.10) / $0.02
score = 100 · mean_F1 − errors_equivalent · (100 / n)
```

where `mean_cost` is the batch's mean **agent** spend (judge cost excluded) and `n` is the iteration batch size. One error-equivalent of penalty costs exactly one fully-wrong query of raw score, so the penalty lives in the agent's own currency (queries), not dollars. The free-zone width ($0.10) covers the Standard-tier leaderboard reference (ReAct + GPT-5 Mini, ~$0.06/query) with margin — a tighter zone would penalize the most promising evolution direction. Above the threshold, the penalty is **unbounded** — a catastrophically expensive agent can score well negative, which is intentional.

The three knobs (`--cost-threshold`, `--cost-per-error`, `--cap-judge-to-estimate`) are independently tunable, resolved at run start, persisted into `checkpoint.json`'s `task_config.paper_finder_runtime`, and **immutable on `--resume`** (a disagreeing flag is a hard error). Exception: on `--eval-only`, `--cap-judge-to-estimate` is taken from the CLI (default on) rather than the stored value — test scoring is the eval-time caller's choice, and the mode used is recorded in `test_results.json`. The per-iteration `aggregate_explanation` (in `evaluation.json`) carries the resolved excess and error-count so failure analysis can read "good but expensive" off the page.

### Cost-penalty asymmetry

The training evaluator applies the formula above. The test evaluator (derived via `with_overrides(apply_cost_penalty=False)` in `main.py`) does not — test scores are raw mean F1 in `[0, 1]` for leaderboard parity.

### Agent-vs-judge cost split

On `semantic_f1` queries (73% of validation) the scorer runs a GPT-4o relevance judge over every returned paper, inside the same `inspect.eval` — so its token usage lands in the same log as the agent's. The evaluator splits by exact model ID: judge usage → `other_cost_usd` (reported, never penalized — the agent doesn't control it); everything else → `agent_cost_usd` (the penalized bucket). The split is sound because the registry's model IDs never include the judge's, and agents can't mint arbitrary model IDs (the registry is the only sanctioned LLM surface). A module-load assert cross-checks astabench's `GRADER_MODEL_NAME` so an upstream judge bump fails loudly instead of silently misattributing cost.

## Cost notes

- Asta MCP rate limit is 10 req/s per endpoint, shared across all workers. The seed's few sequential calls per query sit well under it at `--max-workers 8`, but evolved agents routinely fan out (run `20260716_072622`'s winner: up to 16 concurrent snippet calls per query), and sustained collective overrun surfaces as retry-backoff latency, leaked tool errors, and agent-side timeouts rather than 429s. The limit and the transport timeouts are documented to agents in background.md ("Tool-call transport"), so evolving a self-throttle is the intended fix; leaked errors name their root cause (`evaluator._tool_failure_summary`) to make that signal legible.
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
