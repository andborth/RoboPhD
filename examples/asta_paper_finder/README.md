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

This example is a recent addition. What's verified:

- [x] Scaffold matches existing examples (main.py, evaluator.py, seed, objective.md, background.md, requirements.txt)
- [x] Dataset loads (66 validation, 267 test) via astabench's loader
- [x] `inspect.eval()` runs end-to-end on a 1-sample dataset; sample-level errors surface in diagnostics
- [x] One full eval cycle measured at ~8s wall-clock (most of which is Inspect's per-task overhead)

Open before treating scores as authoritative:

- [ ] End-to-end run with a working `ASTA_TOOL_KEY` (any S2 API key unblocks `tool_source=search`; AI2's MCP key unblocks `tool_source=mcp`)
- [ ] Standard-Tools allowlist enforcement (AST scan in evaluator) — currently absent; evolution could in principle import outside the allowed set
- [ ] Backfill `background.md` with the live MCP tool list (signatures, return shapes) once the MCP key is in hand
