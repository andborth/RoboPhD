# PaperFindingBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's PaperFindingBench task (Literature Understanding category, Standard tools tier). Validation = 66 samples (training pool), test = 267 samples (held out).

Reference points (Standard-tools tier, the one we compete on): generic ReAct spans 0.220 @ $0.06/query (GPT-5 Mini) to 0.374 @ $3.38/query (Opus). Ai2's custom-interface Asta Paper Finder sits outside the tier at 0.397 @ $0.06 — the gap between generic ReAct and that system is the "PaperFinder-specific structure" evolution is trying to discover.

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

`ASTA_TOOL_KEY` is **required** unless you pass `--tool-source search` explicitly: with no key and no flag, the run hard-errors at startup. (It used to warn and silently fall back to the public-S2 tier — run `asta_paper_finder_20260710_081139` burned its budget on unauthenticated 429s that way.)

The explicit `search` mode uses the Python-importable `paper_search`/`snippet_search` factories, which send `ASTA_TOOL_KEY` as the Semantic Scholar `x-api-key` header (`astabench/tools/search.py`). For dev without the AI2 key, request a free personal Semantic Scholar API key at https://www.semanticscholar.org/product/api and set it as `ASTA_TOOL_KEY` — with no key at all you're on S2's unauthenticated shared pool and will throttle under any parallelism. Both tool tiers absorb transient 429/5xx identically: the MCP factory has astabench's `make_retry_wrapper` built in, and `_build_tools` applies the same wrapper to the search kit, so agents never see (and never need to evolve handling for) transient rate limits. Scores from the `search` fallback **do not match the leaderboard exactly** (the MCP tool set has different filtering and date semantics) but the structure is the same. The evaluator records `tool_source` ("mcp" or "search") in every diagnostics dict, and the resolved value is locked for the lifetime of a run (immutable on `--resume`).

Troubleshooting: 429 storms against `api.semanticscholar.org/graph/v1/...` in `error.md` files mean the run is on the `search` tier (check `checkpoint.json` → `task_config.paper_finder_runtime.tool_source`), not that the Asta MCP limit was hit.

## Running

```bash
# Quick smoke test (small budget)
python examples/asta_paper_finder/main.py --num-iterations 2 --evaluation-budget 20

# Full run with default config
python examples/asta_paper_finder/main.py

# With held-out test-set evaluation after optimization
python examples/asta_paper_finder/main.py --eval-test-set

# Force the public-S2 fallback even when ASTA_TOOL_KEY is set
python examples/asta_paper_finder/main.py --tool-source search

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

The two knobs (`--cost-threshold`, `--cost-per-error`) are independently tunable, resolved at run start, persisted into `checkpoint.json`'s `task_config.paper_finder_runtime`, and **immutable on `--resume`** (a disagreeing flag is a hard error). The per-iteration `aggregate_explanation` (in `evaluation.json`) carries the resolved excess and error-count so failure analysis can read "good but expensive" off the page.

### Cost-penalty asymmetry

The training evaluator applies the formula above. The test evaluator (derived via `with_overrides(apply_cost_penalty=False)` in `main.py`) does not — test scores are raw mean F1 in `[0, 1]` for leaderboard parity.

### Agent-vs-judge cost split

On `semantic_f1` queries (73% of validation) the scorer runs a GPT-4o relevance judge over every returned paper, inside the same `inspect.eval` — so its token usage lands in the same log as the agent's. The evaluator splits by exact model ID: judge usage → `other_cost_usd` (reported, never penalized — the agent doesn't control it); everything else → `agent_cost_usd` (the penalized bucket). The split is sound because the registry's model IDs never include the judge's, and agents can't mint arbitrary model IDs (the registry is the only sanctioned LLM surface). A module-load assert cross-checks astabench's `GRADER_MODEL_NAME` so an upstream judge bump fails loudly instead of silently misattributing cost.

## Cost notes

- Asta MCP rate limit is 10 req/s per endpoint, which comfortably supports the default `--max-workers 8`.
- 73% of validation samples are `semantic_f1` queries that invoke the scorer's GPT-4o judge per predicted paper. **Judge cost typically dominates agent cost** for short pipelines. It's excluded from the penalty but still real dollars — budget accordingly.
- **The judge caches verdicts persistently**: astabench appends every (query, paper) judgement to `detailed_reference.json` *inside the installed package* (`astabench/evals/paper_finder/`) and replays it on later evals. Consequences: judge cost per eval shrinks as coverage grows (a fully-cached sample bills $0 with a nonzero score); repeated evals of the same submission are more deterministic than a fresh environment would be; and a pip reinstall/upgrade of astabench silently wipes the cache, restoring full judge cost and re-rolling any verdicts that would land differently. The leaderboard's official rerun judges fresh.

## Files

- `main.py` — `optimize_anything()` entry point
- `evaluator.py` — `PaperFinderEvaluator`; runs `inspect.eval()` on a 1-sample dataset per evaluation; cost split + iteration aggregator
- `_eval_worker.py` — subprocess worker (one per evaluation, for inspect.eval parallelism)
- `model_registry.py` — the nine solver-model handles (not in `file_mapping`)
- `seeds/baseline/agent.py` — minimal `@solver` factory exported as `make_solver`. Demonstrates a tool call (MCP kit only), a registry-handle LLM call with an empty-completion guard, and the JSON output schema.
- `_check_evaluator.py` / `_check_seed.py` — credentialed sanity gates: gold-ID leak scan over the evolution-facing artifacts, synthetic-candidate scorer checks, and a 3-sample live seed run with judge-split assertions.
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

### Code work (deferred)

- [ ] **Extract the pricing machinery to a shared module.** `_estimate_cost` / `_bundled_price_map` here are byte-for-byte copies of `asta_ds1000/evaluator.py`'s (the leaderboard billing basis is a cross-task concept, not task-specific). Both copies carry duplicated test suites (`test_evaluator.py`'s `_estimate_cost` section, ported from ds1000's `ae1e410`) so accidental drift fails loudly, but the right endpoint is one shared implementation — e.g. under `RoboPhD/`, with an import-fallback story for standalone evaluator use. Deferred until the ds1000 campaign is at a quiet point (extraction touches its evaluator).
- [ ] **Standard-Tools allowlist (AST scan).** The evaluator should reject candidates that import outside `{json, re, asyncio, dataclasses, ..., inspect_ai.*, model_registry}`. Without this, evolution could in principle introduce `import openai` and silently lose cost-accounting fidelity / the Standard Tools badge. ~30 lines of `ast.parse` walking.
- [ ] **Submission pipeline: port `scripts/asta_ds1000_submit.py`.** Submissions re-run the official `astabench eval` + `astabench score` against staged `agent.py` + `model_registry.py` and tarball *that* run's logs (they do not collect our internal eval logs). Mostly a name swap, no Docker; remember a seed-baseline entry alongside the best agent.
- [ ] **Judge-explanation surfacing.** Only `gold_criteria.md` is exposed to evolution; the LLM judge's per-paper relevance verdicts (which paper was kept/dropped and why on `semantic_f1` queries) aren't surfaced yet. The most-requested diagnostic across the first run's iteration reflections — sessions could not distinguish recall misses from judge rejections.

### Design questions to revisit

- [ ] **Held-out thermometer split.** Currently all 66 validation samples feed evolution — at 14 examples/iteration over a ~20-iteration run, each example is drawn ~4× (each draw evaluated by up to 3 agents). Decide whether to hold out ~16 as a thermometer the evolution AI can't see, or accept the reuse. Evidence so far (tight train/test agreement in the first run, and the modest draw count) leans accept.
- [ ] **Whether to add `paper_finder` (high-level MCP tool) to the allowlist.** Using it would make our agent an `ai2i_paper_finder` lookalike — defeats the purpose. Lean: exclude.

(Resolved: score-type stratification in Elo — the leaderboard headline turned out to be the plain per-query mean, `adjusted_f1_micro_avg`, so the Elo signal already matches it exactly; no stratification tension exists.)
