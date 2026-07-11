# PaperFindingBench (AstaBench)

Each example is a literature-search query: a natural-language description of papers the user wants ("the BART paper", "papers by David Harel in Nature", "clustering-based attention in Transformers"), and a hidden gold relevance judgment. The agent returns a ranked list of Semantic Scholar `corpus_id`s; each query gets an F1 score in [0, 1] (standard F1 by exact match for specific/metadata queries, LLM-judged adjusted F1 for semantic — see the table below), and the overall score is the plain mean over queries.

## Three query types (`state.metadata["score_type"]`)

| score_type | meaning | gold form | scoring path | train / held-out test count |
| --- | --- | --- | --- | --- |
| `specific_f1` | "the X paper" — known target | `{"corpus_ids":[...]}` | exact-match against `corpus_ids` | 10 / ~40 |
| `metadata_f1` | author/year/venue filters | `{"corpus_ids":[...]}` | exact-match against `corpus_ids` | 8 / ~30 |
| `semantic_f1` | broad topical query | `{"known_to_be_good":[...], "known_to_be_bad":[...], "relevance_criteria":[{name, description, weight}, ...]}` | LLM judge over each predicted paper, weighted by `relevance_criteria` | 48 / ~200 |

All three paths produce a real-valued score in [0, 1]; differences in difficulty are large.

The query text is in `state.metadata["raw_query"]`. The full `state.input` wraps it in a longer instruction template; either is fair game.

## Output schema

The scorer reads `state.output.completion` as a JSON string with this shape:

```json
{
  "output": {
    "query_id": "<state.sample_id>",
    "results": [
      {
        "paper_id": "123456789",
        "markdown_evidence": "Title — first ~400 chars of abstract or key passage."
      },
      ...
    ]
  }
}
```

Notes on the schema:
- `paper_id` is a Semantic Scholar corpus_id as a **string** (not int). Sources sometimes return ints — cast.
- The scorer also accepts `"CorpusId:123456789"` and lowercases/strips it.
- `markdown_evidence` is shown to the LLM judge for `semantic_f1` queries; quality of evidence text directly affects the judge's verdict. Include the title and a relevant passage.
- List order: the scorer reads only your first 250 entries; beyond that cap, order semantics differ by query type. On `specific_f1`/`metadata_f1`, **order does not matter** — precision and recall are computed over the whole (capped) list. On `semantic_f1`, **order is half your score** — see "Scoring (per query)".

## Available tools (`state.tools`)

The PaperFindingBench task attaches the **Asta MCP corpus tools** — eight of them. A date-cutoff filter is applied task-side so results don't leak papers published after the benchmark snapshot.

To find a tool by name in `state.tools`:

```python
from inspect_ai.tool import ToolDef

def get_tool(state, name):
    return next(t for t in state.tools if ToolDef(t).name == name)
```

**Return shape (all MCP tools):** a `list` of ContentText objects whose `.text` attribute is a JSON string. Parse each item:

```python
docs = [json.loads(item.text) for item in raw_result]
```

### The eight tools and their parameters

| Tool | Parameters | Returns (per item) |
| --- | --- | --- |
| `search_papers_by_relevance` | `keyword` (str), `fields` (str, comma-sep), `limit` (int), `venues` (str) | one paper JSON per item: `paperId`, `corpusId` (int), `title`, plus whatever `fields` requests (`abstract`, `authors`, `year`, `venue`, `citationCount`, ...) |
| `search_paper_by_title` | `title` (str), `fields` (str), `venues` (str) | best title match: `paperId`, `title`, `matchScore` |
| `snippet_search` | `query` (str), `limit` (int), `venues` (str), `paper_ids` (str, comma-sep `CorpusId:...`) | ONE item whose JSON is `{"data": [{score, paper: {corpusId (str), title, authors, ...}, snippet...}], "retrievalVersion"}` |
| `get_paper` | `paper_id` (str), `fields` (str) | full metadata for one paper |
| `get_paper_batch` | `ids` (array), `fields` (str) | metadata for many papers at once |
| `get_citations` | `paper_id` (str), `fields` (str), `limit` (int) | citing papers |
| `search_authors_by_name` | `name` (str), `fields` (str), `limit` (int) | author records with `authorId` |
| `get_author_papers` | `author_id` (str), `paper_fields` (str), `limit` (int) | an author's papers |

Sharp edges, verified against the live server:
- `search_papers_by_relevance` is a **literal keyword search**: a full conversational query ("Could you suggest research that investigates ...?") returns ZERO hits. Distill queries to keyword phrases before searching (`snippet_search` is more tolerant of natural language).
- `search_papers_by_relevance` takes `keyword=`, `snippet_search` takes `query=` — read before calling.
- **Request `fields="title,abstract,corpusId"` explicitly** on the search tools; the default omits abstracts, which you need for `markdown_evidence`.
- `corpusId` is an **int** in `search_papers_by_relevance` results but a **str** inside `snippet_search`'s `paper` objects — cast to str for output.
- `snippet_search` is much slower than paper search (can take minutes on cold queries). Budget for it or scope it with `paper_ids` to a shortlist you already retrieved.
- The author/citation tools make `metadata_f1` queries (author/venue/year filters) tractable without keyword-search gymnastics.

### `tool_source=search` fallback (dev only)
When the evaluator runs with `tool_source=search` (no `ASTA_TOOL_KEY`), the kit is different: `paper_search(kquery=..., limit=...)` and `snippet_search(query=...)`, both returning plain `list[dict]` with `corpus_id`/`corpusId`, `title`, `abstract`. Leaderboard-comparable runs always use the MCP kit above; a robust agent can look for both tool names.

## LLM calls

The following model handles are available, imported from `model_registry`. Prices are the rates this benchmark's scoring bills:

| Handle | Input ($/M tok) | Output ($/M tok) | Default `reasoning_effort` | Available overrides |
| --- | --- | --- | --- | --- |
| `GPT_5_4_MINI` | 0.75 | 4.50 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `GPT_5_4` | 2.50 | 15.00 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `GPT_5_5` | 5.00 | 30.00 | model-managed | `"low"`, `"medium"`, `"high"` |
| `CLAUDE_HAIKU_4_5` | 1.00 | 5.00 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `CLAUDE_SONNET_4_6` | 3.00 | 15.00 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `CLAUDE_OPUS_4_8` | 5.00 | 25.00 | model-managed | `"low"`, `"medium"`, `"high"` |
| `GEMINI_3_1_FLASH_LITE` | 0.45 | 2.70 | `"low"` | `"low"`, `"high"` |
| `GEMINI_3_5_FLASH` | 1.50 | 9.00 | `"low"` | `"low"`, `"high"` |
| `GEMINI_3_1_PRO_PREVIEW` | 2.00 | 12.00 | `"low"` | `"low"`, `"high"` |

Setting `reasoning_effort` to any value in the "available overrides" column adds reasoning tokens above what the default already costs. For handles whose default is `"none"`, picking `"low"` is the cheapest opt-in step but it's still strictly more expensive than omitting `reasoning_effort` entirely. For the Gemini handles whose default is already `"low"`, the only opt-up is `"high"`. To stay at the cheapest path on any handle, omit the `reasoning_effort` field from `GenerateConfig`.

`max_tokens` is a universal output-budget cap accepted on every handle (an integer; no provider rejects or strips it). Pass it via `GenerateConfig(max_tokens=N)`. On Anthropic and Gemini handles, the cap applies to the visible completion only — reasoning tokens (when `reasoning_effort` is set) come on top of it. On OpenAI handles, the cap is shared between reasoning and visible tokens, so set it generously when combined with `reasoning_effort` or you may get an empty completion.

```python
from inspect_ai.model import GenerateConfig
from model_registry import GPT_5_4_MINI, CLAUDE_SONNET_4_6

# Default call (cheapest, no extra reasoning):
resp = await GPT_5_4_MINI.generate("Your prompt here")

# Opt into reasoning for a hard query and cap the output:
resp = await CLAUDE_SONNET_4_6.generate(
    "Your prompt here",
    config=GenerateConfig(reasoning_effort="low", max_tokens=2048),
)
text = resp.completion
```

`config` is optional. The two knobs to use are `reasoning_effort` (trades cost for quality on hard queries; see the per-handle table above for default and available values) and `max_tokens` (caps the output budget). All LLM calls must go through one of the handles above — never `get_model()` with a hardcoded string, and **never** a direct `openai` / `anthropic` / `litellm` client import, or you silently underreport cost and risk losing the Standard Tools badge.

## Standard Tools constraint

This benchmark targets the **Standard Tools** leaderboard tier. The agent may use only:
- Tools attached to `state.tools` (the Asta MCP corpus suite)
- LLM calls through `model_registry` handles (Inspect-tracked)
- Standard Python (json, re, asyncio, dataclasses, ...)

It must **not** import third-party search backends (Elasticsearch, Pinecone, custom indices), nor the AI2-internal Mabool client (`paper_finder_ai2i`), nor maintain its own paper cache. The evaluator may reject candidates that import outside an allowlist.

## Per-query cost

Your cost is the LLM calls your agent makes through `model_registry` handles (`agent_cost_usd`). Tool calls (`paper_search`, `snippet_search`, the rest of the MCP suite) are free, in unlimited quantity.

## The relevance judge (semantic queries)

On `semantic_f1` queries (73% of the training queries, and the same share of the held-out test set), the benchmark's scorer runs a GPT-4o relevance judge over every paper you return, grading each against the query's weighted `relevance_criteria`. Understanding the judge is a score lever: it reads each result's `markdown_evidence` when deciding relevance, so evidence quality (title + a passage that speaks to the criteria) directly moves your F1. For each training query, the exact criteria the judge scored against are visible post-hoc in that problem's `gold_criteria.md` diagnostic — read them when diagnosing why a semantic query scored low.

The judge's own LLM spend appears in each problem's `result.json` as `other_cost`. That field is informational only — it is never penalized, never counts toward your batch mean, and is outside your control. Do not optimize for it.

`specific_f1` and `metadata_f1` queries score by exact-match against `corpus_ids` — no judge at all.

## Iteration-aggregate score

Per-example scoring is continuous F1 in [0, 1]. At the end of each iteration, your batch is combined into a single score: your mean F1 (on a 0–100 scale) minus a cost penalty when your mean batch agent-spend exceeds the threshold. The penalty is expressed in fully-wrong-query units — each ${COST_PER_ERROR} of mean spend over ${COST_THRESHOLD} subtracts one error-equivalent (one query's worth of F1) from your score. Only `model_registry` handle calls are metered — tool calls are free.

${COST_PENALTY_TABLE}

## Time budget

Your agent times out and the query scores 0 if a single query takes more than **${EVAL_TIMEOUT_MIN} minutes** of wall-clock. This is a generous budget and is unlikely to be the binding constraint. Per-query wall-clock is recorded as `eval_wall_clock_seconds` in each problem's `result.json`.

## Scoring (per query)

The scorer (`astabench.evals.paper_finder.task.score_paper_finder`) returns a single float in [0, 1] per sample. The overall score is the plain mean of those per-query floats.

**`specific_f1` / `metadata_f1`** — standard F1, order-blind: precision = relevant fraction of what you returned, recall = fraction of the gold set you found, computed over your whole list (first 250 entries). Worked example:
- Predicted `corpus_ids` = `["123456789", "987654321"]`, gold = `["123456789"]`
- TP=1, FP=1, FN=0; precision=0.5, recall=1.0, F1=0.667

**`semantic_f1`** — the LLM judge grades each returned paper 0–3 against the query's weighted `relevance_criteria`, and the score is the harmonic mean of TWO terms, so a zero in either zeroes the query:

1. **Rank**: an NDCG over the grade sequence *in your submitted order*, normalized between the worst and best possible orderings of the papers you returned. Best-first ordering → 1.0; relevant papers buried behind irrelevant ones → toward 0.0. Ordering is not a tiebreaker here; a badly ordered list can zero the whole query even when the papers are good.
2. **Recall at estimated K**: only your first K papers count toward recall, where K is a per-query constant the benchmark ships (its estimate of the total relevant papers in the corpus; also the recall denominator). K is hidden from the agent and varies widely — across training queries the median is ≈30, ranging from 1 to ~200 — so you cannot tune to it; lead with your best and return a generously long list.

Two practical consequences of the rank normalization:
- Never return a **single paper** or a list the judge will grade **uniformly** (e.g. only your most confident hits, all judged perfect): identical grades make worst ordering equal best ordering, the rank term degenerates to 0, and the query scores 0 regardless of recall. Include your confident papers first, then plausible ones — grade variety in the tail is harmless (extra low-graded papers past K don't reduce recall) and protects the rank term.
- Order strictly best-first; it is half the score.

## Diagnostics

Any `print()` output from the agent is captured and included in evaluation diagnostics as `agent_stdout`. Use `print()` to log anything you think would be helpful for you to see when improving the agent in later rounds.
