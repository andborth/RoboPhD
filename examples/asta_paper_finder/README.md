# PaperFindingBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's PaperFindingBench task (Literature Understanding category, Standard tools tier). Validation = 66 samples (training pool), test = 267 samples (held out).

Reference target: best Standard-tools entry on the leaderboard is generic ReAct at adjusted-F1 0.374, $3.38/eval. The Standard frontier is loose because no published agent on it has PaperFinder-specific structure — that's the headroom we're after.

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

If `ASTA_TOOL_KEY` is unset, the evaluator falls back to the Python-importable `paper_search` and `snippet_search` factories. Note: those factories also use `ASTA_TOOL_KEY` as the Semantic Scholar `x-api-key` header (`astabench/tools/search.py`), so without *any* key you'll hit S2's unauthenticated rate limits (HTTP 429) within a handful of calls. For dev without the AI2 ASTA_TOOL_KEY, request a free personal Semantic Scholar API key at https://www.semanticscholar.org/product/api and set it as `ASTA_TOOL_KEY`. Scores from the `search` fallback **do not match the leaderboard exactly** (the MCP tool set has different filtering and date semantics) but the structure is the same. The evaluator records `tool_source` ("mcp" or "search") in every diagnostics dict, and the resolved value is locked for the lifetime of a run (immutable on `--resume`).

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

## Files

- `main.py` — `optimize_anything()` entry point
- `evaluator.py` — `PaperFinderEvaluator`; runs `inspect.eval()` on a 1-sample dataset per evaluation; cost split + iteration aggregator
- `_eval_worker.py` — subprocess worker (one per evaluation, for inspect.eval parallelism)
- `model_registry.py` — the nine solver-model handles (not in `file_mapping`)
- `seeds/baseline/agent.py` — minimal `@solver` factory exported as `make_solver`. Demonstrates a tool call, a registry-handle LLM call, and the JSON output schema.
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

### Verification (in progress)

- [ ] Confirm seed produces nonzero score on at least one sample of each `score_type` (specific, metadata, semantic) via `tool_source=mcp`.
- [ ] Measure realistic per-eval latency at concurrency (`--max-workers 8` against ~20 samples).
- [ ] Enumerate the live MCP tool surface (`for t in make_asta_mcp_tools(...): print(ToolDef(t).name, ToolDef(t).parameters)`) and backfill `background.md` with signatures and return shapes for each — same treatment `protein_go/background.md` gives its tool kit.

### Code work (deferred)

- [ ] **Extract the pricing machinery to a shared module.** `_estimate_cost` / `_bundled_price_map` here are byte-for-byte copies of `asta_ds1000/evaluator.py`'s (the leaderboard billing basis is a cross-task concept, not task-specific). Both copies carry duplicated test suites (`test_evaluator.py`'s `_estimate_cost` section, ported from ds1000's `ae1e410`) so accidental drift fails loudly, but the right endpoint is one shared implementation — e.g. under `RoboPhD/`, with an import-fallback story for standalone evaluator use. Deferred until the ds1000 campaign is at a quiet point (extraction touches its evaluator).
- [ ] **Standard-Tools allowlist (AST scan).** The evaluator should reject candidates that import outside `{json, re, asyncio, dataclasses, ..., inspect_ai.*, model_registry}`. Without this, evolution could in principle introduce `import openai` and silently lose cost-accounting fidelity / the Standard Tools badge. ~30 lines of `ast.parse` walking.
- [ ] **Submission tarball pipeline.** The leaderboard accepts tarballs of `.eval` log files. Each `inspect.eval()` call already writes one to the evaluator's `_log_dir`. A separate "package for submission" path could collect logs from a full test-set run.
- [ ] **Judge-explanation surfacing.** Only `gold_criteria.md` is exposed to evolution; the LLM judge's per-paper relevance verdicts (which paper was kept/dropped and why on `semantic_f1` queries) aren't surfaced yet.

### Design questions to revisit

- [ ] **Held-out thermometer split.** Currently all 66 validation samples feed evolution — 1500 budget / 66 ≈ 22.7× per-example reuse. Decide whether to hold out ~16 as a thermometer the evolution AI can't see, or accept the reuse.
- [ ] **Score-type stratification in Elo.** The scorer reports means *grouped by* `score_type`; RoboPhD's Elo sees a single float. Options: (a) stratify Elo match sampling by score_type, (b) trust the marginal mean. (b) is the current default; revisit if evolution overfits to semantic queries (the largest group).
- [ ] **Whether to add `paper_finder` (high-level MCP tool) to the allowlist.** Using it would make our agent an `ai2i_paper_finder` lookalike — defeats the purpose. Lean: exclude.
