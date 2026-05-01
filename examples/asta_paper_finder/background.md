# PaperFindingBench (AstaBench)

Each example is a literature-search query: a natural-language description of papers the user wants ("the BART paper", "papers by David Harel in Nature", "clustering-based attention in Transformers"), and a hidden gold relevance judgment. The agent returns a ranked list of Semantic Scholar `corpus_id`s; the scorer computes adjusted micro-F1 against the gold set, separately per query type, and a macro mean.

## Three query types (`state.metadata["score_type"]`)

| score_type | meaning | gold form | scoring path | val/test count |
| --- | --- | --- | --- | --- |
| `specific_f1` | "the X paper" — known target | `{"corpus_ids":[...]}` | exact-match against `corpus_ids` | 10 / ~40 |
| `metadata_f1` | author/year/venue filters | `{"corpus_ids":[...]}` | exact-match against `corpus_ids` | 8 / ~30 |
| `semantic_f1` | broad topical query | `{"known_to_be_good":[...], "known_to_be_bad":[...], "relevance_criteria":[{name, description, weight}, ...]}` | LLM judge over each predicted paper, weighted by `relevance_criteria` | 48 / ~200 |

All three paths produce a real-valued score in [0, 1]; differences in difficulty and per-eval cost are large.

The query text is in `state.metadata["raw_query"]`. The full `state.input` wraps it in a longer instruction template; either is fair game.

## Output schema

The scorer reads `state.output.completion` as a JSON string with this shape:

```json
{
  "output": {
    "query_id": "<state.sample_id>",
    "results": [
      {
        "paper_id": "212718077",
        "markdown_evidence": "Title — first ~400 chars of abstract or key passage."
      },
      ...
    ]
  }
}
```

Notes on the schema:
- `paper_id` is a Semantic Scholar corpus_id as a **string** (not int). Sources sometimes return ints — cast.
- The scorer also accepts `"CorpusId:212718077"` and lowercases/strips it.
- `markdown_evidence` is shown to the LLM judge for `semantic_f1` queries; quality of evidence text directly affects the judge's verdict. Include the title and a relevant passage.
- `results` should be ordered most-relevant-first and may contain up to 250 entries; the scorer auto-truncates per `score_type` (e.g. semantic uses "estimated K", specific uses K=full, litqa2 uses recall@30).

## Available tools (`state.tools`)

The PaperFindingBench task attaches the **Asta MCP corpus tools**. `paper_search` and `snippet_search` are the workhorses. The wider Asta tool surface is exposed via the same MCP connection. **Argument names are not always intuitive — `paper_search` uses `kquery`, `snippet_search` uses `query`. Read the description before calling.**

To find a tool by name in `state.tools`:

```python
from inspect_ai.tool import ToolDef

def get_tool(state, name):
    return next(t for t in state.tools if ToolDef(t).name == name)
```

### `paper_search`
**Signature:** `paper_search(kquery: str, limit: int = 50) -> list[dict]`
Search the Semantic Scholar paper index by free-text query. The task applies a date-cutoff filter (`inserted_before=2025-06-01`) so results don't leak future papers.

Each hit is a dict with at minimum:
- `corpus_id` (str), `corpusId` (int) — same value, both keys present
- `title` (str)
- `abstract` (str), `text` (str) — same
- `section_title` ("abstract")
- plus other Semantic Scholar metadata fields

### `snippet_search`
**Signature:** `snippet_search(query: str, limit: int = 50, corpus_ids: list[str] | None = None) -> list[dict]`
Snippet-level retrieval — returns short passages from papers along with their containing paper. `corpus_ids` (optional) restricts to a known shortlist, useful for evidence-extraction over candidates already retrieved by `paper_search`.

Note the parameter name is `query`, not `kquery`.

### Other Asta MCP tools (advertised by the MCP server when `ASTA_TOOL_KEY` is set)
The MCP server may also expose: `similar_papers`, `get_papers_metadata`, `paper_qa`, `corpus_qa`, `author_search_by_name`, `paper_citations`, `paper_references`, plus a high-level `paper_finder`. Discover their exact signatures at runtime via `ToolDef(tool).parameters`. Using `paper_finder` defeats the purpose of building an evolved agent — it's a sub-agent that does what we're trying to evolve.

## Calling the LLM

Use Inspect's tracked model API so token usage flows into the `.eval` log (which is what the leaderboard reads for cost):

```python
from inspect_ai.model import get_model

response = await get_model().generate(prompt)
text = response.completion
```

**Do not** import `openai` / `anthropic` / `litellm` directly. If you absolutely must (e.g., to use a model not in Inspect's registry), wrap the call with `record_model_usage_with_inspect(model_name, ModelUsage(...))` afterward, or you silently underreport cost and risk losing the Standard Tools badge.

## Standard Tools constraint

This benchmark targets the **Standard Tools** leaderboard tier. The agent may use only:
- Tools attached to `state.tools` (the Asta MCP corpus suite)
- Inspect-tracked LLM calls via `get_model()`
- Standard Python (json, re, asyncio, dataclasses, ...)

It must **not** import third-party search backends (Elasticsearch, Pinecone, custom indices), nor the AI2-internal Mabool client (`paper_finder_ai2i`), nor maintain its own paper cache. The evaluator may reject candidates that import outside an allowlist.

## Per-query cost

- `specific_f1` and `metadata_f1`: scoring is exact-match against `corpus_ids`, **no judge LLM**. Cost is whatever the agent itself spends on tool calls + LLM reasoning.
- `semantic_f1` (73% of the validation split): scoring invokes the LLM judge `get_llm_relevance` on every predicted paper, against each `relevance_criteria` entry. **Per-eval judge cost is non-trivial** — typically dominates the agent's own cost for short pipelines.

This means evolution sees a meaningfully different cost profile per query type, even when the *agent's* code is identical. Score-type-aware behavior (smaller k for semantic to limit judge cost; aggressive k for specific where judge isn't running) is a productive surface to mutate.

## Scoring (per query)

The scorer (`astabench.evals.paper_finder.task.score_paper_finder`) returns a single float in [0, 1] per sample. The headline benchmark score is the macro mean grouped by `score_type`, weighted equally across the three groups.

Worked example for `specific_f1`:
- Predicted `corpus_ids` = `["204960716", "13745324"]`, gold = `["204960716"]`
- TP=1, FP=1, FN=0; precision=0.5, recall=1.0, F1=0.667 (after AstaBench's adjustment for K-truncation)

For `semantic_f1`, each predicted paper is judged against each `relevance_criteria` by an LLM, producing a continuous relevance score; the final F1 is computed against the *estimated* total relevant set ("KTypes.ESTIMATED").

## Diagnostics

Any `print()` output from the solver is captured into `agent_stdout` in the evaluator's diagnostics dict — useful for logging which tool calls were made, what each returned, etc. Prefer concise structured prints (`f"[{state.sample_id}] step=rerank kept={n}"`) over dumping full tool responses.

## Headline benchmark target

| Tier | Reference agent | Score | Cost |
| --- | --- | --- | --- |
| Standard | ReAct + claude-opus-4-7 | **0.374** | $3.38 |
| Standard | ReAct + claude-opus-4-6 | 0.372 | $1.49 |
| Standard | ReAct + GPT-5 Mini | 0.220 | $0.06 |
| Custom interface | Asta Paper Finder + gpt-5-mini | 0.433 | $0.35 |

The Standard tier is what we compete on. The reference ReAct agents on this tier have *no PaperFindingBench-specific logic* — they're generic loops with the MCP tools attached. This is the headroom evolution is built to claim.
