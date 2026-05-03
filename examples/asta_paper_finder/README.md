# PaperFindingBench (AstaBench)

Evolves Inspect-AI `@solver` agents on AstaBench's PaperFindingBench task (Literature Understanding category, Standard tools tier). Validation = 66 samples, test = 267 samples.

Reference target: best Standard-tools entry on the leaderboard is generic ReAct at adjusted-F1 0.374, $3.38/eval. The Standard frontier is loose because no published agent on it has PaperFinder-specific structure — that's the headroom we're after.

## Setup

```bash
# 1. Install RoboPhD core dependencies (from repo root)
pip install -r requirements.txt

# 2. Install task-specific dependencies (astabench bundles inspect_ai + Asta MCP factories)
pip install -r examples/asta_paper_finder/requirements.txt
```

## Credentials

Two credentials gate this task. Both are environment variables.

### `HF_ACCESS_TOKEN` — gated dataset

The validation/test splits live in HuggingFace dataset `allenai/asta-bench`, which is gated.

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

The leaderboard's Standard tools tier requires the Asta MCP corpus tools, served by an AI2 endpoint. Request a key via the form linked from https://github.com/allenai/asta-bench (under `ASTA_TOOL_KEY`).

```bash
export ASTA_TOOL_KEY="..."
```

If `ASTA_TOOL_KEY` is unset, the evaluator falls back to the Python-importable `paper_search` and `snippet_search` factories. Note: those factories also use `ASTA_TOOL_KEY` as the Semantic Scholar `x-api-key` header (`astabench/tools/search.py`), so without *any* key you'll hit S2's unauthenticated rate limits (HTTP 429) within a handful of calls. For dev without the AI2 ASTA_TOOL_KEY, request a free personal Semantic Scholar API key at https://www.semanticscholar.org/product/api and set it as `ASTA_TOOL_KEY`. Scores from the `search` fallback **do not match the leaderboard exactly** (the MCP tool set has different filtering and date semantics) but the structure is the same. The evaluator records `tool_source` ("mcp" or "search") in every diagnostics dict so the mode is auditable.

### Solver LLM key

The candidate solver calls an LLM via `get_model()`. Set whichever provider key matches `--model`:
```bash
export OPENAI_API_KEY="..."     # for openai/gpt-4o-mini, the default
# or ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```

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

## Cost notes

- Asta MCP rate limit is ~4 req/sec on the default `ASTA_TOOL_KEY`, which caps useful `--max-workers` around 4–8. Default is 4.
- 73% of validation samples are `semantic_f1` queries that invoke the scorer's LLM judge per predicted paper. **Judge cost typically dominates agent cost** for short pipelines. Budget accordingly.
- At GPT-5 Mini-class rates ($0.06/eval observed on the leaderboard), a 1500-budget run is ~$90 in solver tokens plus judge cost.

## Files

- `main.py` — `optimize_anything()` entry point
- `evaluator.py` — `PaperFinderEvaluator`; runs `inspect.eval()` on a 1-sample dataset per evaluation
- `seeds/baseline/agent.py` — minimal `@solver` factory exported as `make_solver`. Demonstrates a tool call, an Inspect-tracked LLM call, and the JSON output schema.
- `objective.md` — what evolution should optimize
- `background.md` — task spec, score types, output schema, available tools, calling conventions, Standard Tools constraint
- `requirements.txt` — astabench (which bundles inspect_ai)

## Status

### Done

- [x] Scaffold matches existing examples (main.py, evaluator.py, seed, objective.md, background.md, requirements.txt)
- [x] Dataset loads (66 validation, 267 test) via astabench's loader
- [x] `inspect.eval()` runs end-to-end on a 1-sample dataset; sample-level errors surface cleanly into diagnostics (`error`, `agent_stdout`, `agent_output`, `cost_usd`, `usage`, `score_type`, `tool_source`)
- [x] One full eval cycle measured at ~8s wall-clock (most of which is Inspect's per-task overhead)
- [x] Seed demonstrates the three calling conventions evolution will mutate: tool-by-name lookup from `state.tools`, Inspect-tracked `get_model().generate()`, JSON output to `state.output.completion`

### Blocked on credentials

- [ ] **`ASTA_TOOL_KEY`** — request form is pending with AI2 (https://allenai-web-dev.allen.ai/asta/resources/mcp). Unblocks `tool_source=mcp` (the leaderboard's Standard tier). Until it arrives, smoke-testing with `tool_source=search` requires a personal Semantic Scholar key (free, ~5 min at https://www.semanticscholar.org/product/api) set into the same `ASTA_TOOL_KEY` env var; that path produces structurally-correct scores but uses S2's public surface, not the MCP corpus.

### Verification (do once a key is in hand)

- [ ] Confirm seed produces nonzero score on at least one sample of each `score_type` (specific, metadata, semantic). Expected to be low — the seed is a one-search, one-rerank baseline. Rough lower bound: > 0.0 on metadata/specific (paper_search alone often hits the gold ID for "the BART paper"–style queries).
- [ ] Measure realistic per-eval latency at concurrency. The 8s observed in dev was a single sample with a 429 short-circuit; real runs with the LLM judge on `semantic_f1` queries (48/66 of validation) will be slower. Probe with `--max-workers 4` against ~20 samples and report wall-clock.
- [ ] Confirm `usage` and `cost_usd` populate correctly when LLM calls actually run. The current pipeline reads `log.stats.model_usage` and prices via `litellm.cost_per_token`; both paths are cold until a real solver call lands.
- [ ] Enumerate the live MCP tool surface (`for t in make_asta_mcp_tools(...): print(ToolDef(t).name, ToolDef(t).parameters)`) and backfill `background.md` with signatures, return shapes, and short usage notes for each — same treatment `protein_go/background.md` gives its tool kit.

### Code work

- [ ] **Standard-Tools allowlist (AST scan).** Currently absent. The evaluator should reject candidates that import outside `{json, re, asyncio, dataclasses, ..., inspect_ai.*, astabench.tools.*}`. Without this, evolution could in principle introduce `import openai` or a custom search backend and silently lose cost-accounting fidelity / the Standard Tools badge. ~30 lines of `ast.parse` walking.
- [ ] **Per-eval cost cap.** Other examples (DocFinQA, ARC-AGI) cap per-problem solver cost at ~$0.10 with a soft penalty for overruns. PaperFinder's per-eval cost has a hard floor (the judge LLM on semantic queries is outside our control) and a soft ceiling (the agent's own LLM calls). Decide whether to add a budget similar to docfinqa's, and where the ceiling lives.
- [ ] **Decide on TaskState bypass.** If 8s/eval × 1500 budget is acceptable, leave `inspect.eval()` per call as-is. If we want to push throughput, switch to constructing TaskState manually and calling the scorer in-process. Profile first.
- [ ] **Submission tarball pipeline.** Out of scope for v1 but: the leaderboard accepts tarballs of `.eval` log files. Each `inspect.eval()` call already writes one to the evaluator's `_log_dir`. A separate "package for submission" path could collect logs from a full test-set run.
- [ ] **Judge-explanation surfacing** (parallel to `asta_discoverybench`'s `judge_explanation.md`). Currently only `gold_criteria.md` is exposed to evolution; the LLM judge's per-paper relevance verdicts (which paper was kept/dropped and why on `semantic_f1` queries) aren't surfaced. Defer until MCP is unblocked and we can test against the real scorer surface — but if PaperFinder ever runs at scale via the `search` fallback first, evolution will be working with weaker per-evaluation signal than DiscoveryBench's.
- [ ] **Cost penalty** (parallel to `asta_discoverybench`'s `apply_cost_penalty` / `cost_breached` / `cost_penalty_applied`). PaperFinder doesn't currently apply a per-example cost penalty during training. If we want the same training-vs-test-asymmetric soft penalty (multiply score by 0.9 on breach during ELO; raw HMS at test) we'd need: a `cost_budget` parameter, a way to disentangle agent spend from the LLM judge's per-paper fan-out on semantic queries, and the `apply_cost_penalty=True/False` toggle pattern from `asta_discoverybench/evaluator.py`. Defer until the design questions are resolved (and ideally until PaperFinder has been smoke-tested end-to-end with a real `ASTA_TOOL_KEY`).

### Design questions to revisit

- [ ] **Held-out thermometer split.** PLAN.md flagged this. Currently we're feeding all 66 validation samples to evolution, which is a lot of reuse (1500 budget / 66 examples ≈ 22.7× per-example reuse). Decide whether to hold out ~16 as a thermometer the evolution AI can't see, or accept the reuse.
- [ ] **Score-type stratification in ELO.** RoboPhD's ELO sees a single float per match. The PaperFindingBench scorer reports means *grouped by* `score_type`. Two options: (a) sample ELO matches stratified by score_type so each group gets balanced exposure, (b) trust that optimizing the marginal mean tracks the headline. (b) is the simpler default; revisit if evolution overfits to semantic queries (the largest group).
- [ ] **Whether to add `paper_finder` (high-level MCP tool) to the allowlist.** It exists; using it would make our agent essentially an `ai2i_paper_finder` lookalike. Defeats the purpose for evolution but might be defensible as a baseline benchmark. Lean: exclude.
