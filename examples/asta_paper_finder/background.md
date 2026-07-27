# PaperFindingBench (AstaBench)

Each example is a literature-search query: a natural-language description of papers the user wants ("the BART paper", "papers by David Harel in Nature", "clustering-based attention in Transformers"), and a hidden gold relevance judgment. The agent returns a ranked list of Semantic Scholar `corpus_id`s; each query gets an F1 score in [0, 1] (standard F1 by exact match for specific/metadata queries, LLM-judged adjusted F1 for semantic — see the table below), and the overall score is the plain mean over queries.

## Three query types (`state.metadata["score_type"]`)

| score_type | meaning | gold form | scoring path | train count |
| --- | --- | --- | --- | --- |
| `specific_f1` | "the X paper" — known target | `{"corpus_ids":[...]}` | exact-match against `corpus_ids` | 10 |
| `metadata_f1` | author/year/venue filters | `{"corpus_ids":[...]}` | exact-match against `corpus_ids` | 8 |
| `semantic_f1` | broad topical query | `{"known_to_be_good":[...], "known_to_be_bad":[...], "relevance_criteria":[{name, description, weight}, ...]}` | LLM judge over each predicted paper, weighted by `relevance_criteria` | 48 |

The held-out test set has a similar query-type mix, so improvements weighted by these proportions generalize.

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
        "markdown_evidence": "A verbatim passage from retrieved text ... a second verbatim passage."
      },
      ...
    ]
  }
}
```

Notes on the schema:
- `paper_id` is a Semantic Scholar corpus_id as a **string** (not int). Sources sometimes return ints — cast.
- The scorer also accepts `"CorpusId:123456789"` and lowercases/strips it.
- `markdown_evidence` is the *only* text the judge sees for a paper on `semantic_f1` queries: it is handed your evidence plus the criteria and nothing else — not the title, abstract, or corpus record unless your evidence quotes them. It then judges, per criterion, how well that text supports the criterion (see "Scoring (per query)" for how the per-criterion judgments combine into the paper's 0–3 grade). On `specific_f1`/`metadata_f1` the content is never read (scoring is exact-match on `paper_id`).
- The field is **required on every result, all query types** (`SingleResult.markdown_evidence` is a non-optional string). Omitting the key fails output parsing and scores the whole query 0 — so on the exact-match paths, where the content is ignored, an empty string is valid, e.g. `{"paper_id": "123456789", "markdown_evidence": ""}`, but the key must be present.
- **Grounding requirement**: `markdown_evidence` must be up to 8 passages quoted **verbatim** from text you retrieved for that same paper (title/abstract/tldr/snippet returned by the tools), joined by ` ... `. Each passage is checked independently: any passage not verbatim-derivable from retrieved corpus text is discarded before the judge sees it, and the judge scores the paper on whatever grounded passages remain. If *every* passage is discarded, the paper is scored Not Relevant with no judge call. Punctuation/case/whitespace differences are tolerated; paraphrased or invented passages are not, and earn nothing.${EVIDENCE_CAP_NOTE}
- List order: the scorer reads only your first 250 entries; beyond that cap, order semantics differ by query type. On `specific_f1`/`metadata_f1`, **order does not matter** — precision and recall are computed over the whole (capped) list. On `semantic_f1`, **order is half your score** — see "Scoring (per query)".

## Available tools (`state.tools`)

The PaperFindingBench task attaches the **Asta MCP corpus tools** — eight of them. A date-cutoff filter is applied task-side so results don't leak papers published after the benchmark snapshot — with one gap: the citing-paper lists returned by `get_citations` are not filtered (see the search-semantics notes below).

To find a tool by name in `state.tools`:

```python
from inspect_ai.tool import ToolDef

def get_tool(state, name):
    return next(t for t in state.tools if ToolDef(t).name == name)
```

**Return shape (all MCP tools):** a `list` of ContentText objects whose `.text` attribute is a JSON string. Some tools wrap their payload in a `{"data": [...]}` object (see the table), so parse with a flattener:

```python
def parse_items(raw):
    docs = []
    for item in raw or []:
        doc = json.loads(item.text)
        docs.extend(doc["data"]) if "data" in doc else docs.append(doc)
    return docs
```

### The eight tools and their parameters

| Tool | Parameters | Returns |
| --- | --- | --- |
| `search_papers_by_relevance` | `keyword` (str), `fields` (str, comma-sep), `limit` (int, **1–100**), `venues` (str) | one paper JSON per item: `paperId`, `corpusId` (**int**), `title`, plus whatever `fields` requests (`abstract`, `authors`, `year`, `venue`, `citationCount`, ...) |
| `search_paper_by_title` | `title` (str), `fields` (str), `venues` (str) | ONE item: the single best match (`paperId`, `corpusId`, `title`, `matchScore`, + `fields`). **No match ⇒ the item is `{"data": []}`** — check for `paperId` before use |
| `snippet_search` | `query` (str), `limit` (int), `venues` (str), `paper_ids` (str, comma-sep, up to 100, `CorpusId:<id>` / `DOI:<doi>` / arXiv etc.) | ALWAYS one wrapper item: `{"data": [...], "retrievalVersion"}`; `limit` sets the length of `data`. Each `data` entry: `{score, paper: {corpusId (**str**), title, authors, openAccessInfo}, snippet: {text, section, snippetKind, snippetOffset, annotations}}` |
| `get_paper` | `paper_id` (str), `fields` (str) | full metadata for one paper |
| `get_paper_batch` | `ids` (array), `fields` (str) | metadata for many papers at once |
| `get_citations` | `paper_id` (str), `fields` (str), `limit` (int, max 1000, no offset/paging) | one item per citing paper, **wrapped**: `{"citingPaper": {paperId, corpusId (**str**), title, ...}}` — unwrap before reading |
| `search_authors_by_name` | `name` (str), `fields` (str), `limit` (int) | author records with `authorId`, `name`, `paperCount` |
| `get_author_papers` | `author_id` (str), `paper_fields` (str), `limit` (int) | an author's papers |

**Requestable `fields`** (verified; applies to the paper search/get tools): `abstract`, `authors`, `citationCount`, `citations`, `corpusId`, `externalIds`, `fieldsOfStudy`, `influentialCitationCount`, `isOpenAccess`, `journal`, `publicationDate`, `referenceCount`, `references`, `tldr`, `url`, `venue`, `year`. `paperId` and `title` are always returned. `tldr` is an auto-generated one-sentence summary of the paper — useful raw material for `markdown_evidence`. `citations`/`references` return nested arrays and can be heavy. An invalid field name raises a tool error (with an unhelpful message), so stick to this list.

### Search semantics (verified against the live server)

- **Term matching is lenient, with no query operators.** Extra or missing terms don't zero a result set (adding a gibberish term leaves the top hits unchanged); quoting a phrase does NOT enforce its presence (a quoted nonexistent phrase still returns full results); `-term` does not exclude; `OR` is treated as an ordinary token. Query text steers *ranking* only.
- **Interrogative/imperative framing returns ZERO hits.** "Could you suggest research that investigates X?" → 0 results, with or without punctuation; the bare noun phrase "X" → full results, even with articles and prepositions intact. Strip the question/request preamble; keyword or noun-phrase queries only. (`snippet_search` is tolerant of full natural-language queries — it's the right tool for sentence-shaped input.)
- **`limit` must be 1–100 on the search tools** — values outside that range are a loud tool error, not a clamp.
- `venues=` genuinely filters (comma-separated exact venue names, e.g. `"Nature,NeurIPS"`).
- **There is no year/date filter parameter.** For year-constrained queries (common in `metadata_f1`), request the `year` field and post-filter yourself.
- **Same person, multiple author IDs**: `search_authors_by_name("David Harel")` returns several fragmentary identities (one with 399 papers, others with 5–6). Disambiguate by `paperCount` or by checking the papers themselves before trusting `get_author_papers`.
- The docstring prose on some tools states stale defaults ("fields default is title", "limit default is 50") inherited from the upstream server — **trust the parameter defaults instead** (a rich field set including `corpusId`, limit 20). Also, the docstrings' "available fields" lists omit `corpusId` even though it's valid and essential — when trimming `fields`, always keep `title,abstract,corpusId`.
- `corpusId` is an **int** in `search_papers_by_relevance` results but a **str** in `snippet_search` and `get_citations` results — cast to str for output.
- `snippet_search` is a **passage-retrieval engine**, not a paper search: it ranks ~500-word chunks (title/abstract/body; `snippet.section` says which) across the whole corpus and returns the top `limit` passages, score-descending. `paper_ids` is a scope filter, not a per-paper allocation — multiple passages routinely come from the same paper (verified: 8-of-10 from one paper in a two-paper scope), so for evidence on *each* of a shortlist's candidates, call per-paper or raise `limit`; a single scoped call starves the weaker matches.
- `snippet_search` latency is variable (seconds to minutes on cold queries). Budget for it or scope it with `paper_ids` to a shortlist you already retrieved.
- `search_papers_by_relevance` takes `keyword=`, `snippet_search` takes `query=` — read before calling.
- The author/citation tools make `metadata_f1` queries (author/venue/year filters) tractable without keyword-search gymnastics.
- **`get_citations` is the one tool the snapshot date-cutoff does not cover.** Its citing-paper list comes back unfiltered, so it can include papers published after the benchmark snapshot; on the exact-match paths such papers are never in gold, so submitting them is a pure precision loss — post-filter by `year`/`publicationDate` yourself. (The nested `citations` *field* on the metadata tools **is** filtered — the gap is only this tool's own output.) Results are observed (not guaranteed) to arrive newest-first, so on a heavily-cited target the ≤1000-entry window can be dominated by post-snapshot citers while older citers stay unreachable — this ordering plus the 1000 cap makes "papers citing <hugely-cited paper>" queries structurally incomplete.

### Tool-call transport: timeouts, retries, and the rate limit

For information — the layers a corpus tool call passes through, bottom-up. In many cases none of these turn out to be binding constraints; they are documented so slow or failed calls are attributable rather than mysterious.

1. **Connect: 5 s** to establish the HTTP connection to the MCP server.
2. **Response read: 300 s** — the per-call ceiling. A call whose response takes longer (cold `snippet_search` is the usual case) raises, without retry.
3. **Automatic retries on transient errors**: HTTP 429/529/504 and server errors (500/502/503) and broken connections are retried with exponential backoff (up to 10 attempts, ~5 minutes of accumulated waiting worst-case; no overall per-call deadline). The backend enforces a rate limit of **10 requests/second per endpoint**, shared by everything using the key at once — including other concurrently running evaluations — so sustained bursts convert into backoff latency here rather than errors.${TOOL_LAUNCH_NOTE}
4. **The per-query wall-clock budget** ("Time budget" below) is the only other deadline anywhere in the stack.

A call that still fails raises into your code with the root cause named in the exception message (e.g. `HTTP 429 rate-limited (retry budget exhausted)`, a transport timeout, or a broken connection).

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

`max_tokens` is an output-budget cap accepted on every handle, passed via `GenerateConfig(max_tokens=N)`. On Anthropic and Gemini handles, the cap applies to the visible completion only — reasoning tokens (when `reasoning_effort` is set) come on top of it. **On OpenAI handles, the cap is shared between reasoning and visible tokens, so `max_tokens` is not recommended on OpenAI handles** — shape output length in the prompt instead.

```python
from inspect_ai.model import GenerateConfig
from model_registry import GPT_5_4, GPT_5_4_MINI, CLAUDE_SONNET_4_6

# Default call (cheapest, no extra reasoning):
resp = await GPT_5_4_MINI.generate("Your prompt here")

# Reasoning on an Anthropic/Gemini handle: max_tokens caps the VISIBLE
# completion only, so a modest cap is safe alongside reasoning.
resp = await CLAUDE_SONNET_4_6.generate(
    "Your prompt here",
    config=GenerateConfig(reasoning_effort="low", max_tokens=2048),
)

# Reasoning on an OpenAI handle: no max_tokens (not recommended on
# OpenAI handles — see the warning above).
resp = await GPT_5_4.generate(
    "Your prompt here",
    config=GenerateConfig(reasoning_effort="low"),
)

text = (resp.completion or "").strip()
print(f"completion len={len(text)}")  # empty ⇒ investigate, don't silently fall back
```

`config` is optional. The two knobs to use are `reasoning_effort` (trades cost for quality on hard queries; see the per-handle table above for default and available values) and `max_tokens` (caps the output budget). All LLM calls must go through one of the handles above — never `get_model()` with a hardcoded string, and **never** a direct `openai` / `anthropic` / `litellm` client import, or you silently underreport cost and risk losing the Standard Tools badge.

## Standard Tools constraint

This benchmark targets the **Standard Tools** leaderboard tier. The agent may use only:
- Tools attached to `state.tools` (the Asta MCP corpus suite)
- LLM calls through `model_registry` handles (Inspect-tracked)
- Standard Python (json, re, asyncio, dataclasses, ...)

It must **not** import third-party search backends (Elasticsearch, Pinecone, custom indices), nor the AI2-internal Mabool client (`paper_finder_ai2i`), nor call web APIs directly — including the public Semantic Scholar API (`api.semanticscholar.org`): the `state.tools` suite is the agent's only corpus access. Those tools enforce the benchmark's snapshot date-cutoff; the live public API does not, so calling it both breaks the Standard Tools tier and leaks post-snapshot papers the scorer treats as wrong. It must also not persist retrieval results across queries — every evaluation's papers must come through the tools (within-query in-memory bookkeeping over tool results is normal and fine). The evaluator may reject candidates that import outside an allowlist.

## Per-query cost

Your cost is the LLM calls your agent makes through `model_registry` handles, recorded per problem as `eval_cost` in `result.json`. Tool calls (`paper_search`, `snippet_search`, the rest of the MCP suite) are free, in unlimited quantity.

## The relevance judge (semantic queries)

On `semantic_f1` queries (73% of the training queries, and the same share of the held-out test set), the benchmark's scorer runs an LLM relevance judge over every paper you return, grading each against the query's weighted `relevance_criteria`. Understanding the judge is a score lever: it decides relevance from each result's `markdown_evidence` alone, so whether that text demonstrates each criterion directly moves your F1. Evidence is grounding-checked first (see the Output schema's grounding requirement): only evidence quoted verbatim from text you retrieved for that paper reaches the judge; anything else is discarded and the paper scored Not Relevant. For each training query, the exact criteria the judge scored against are visible post-hoc in that problem's `gold_criteria.md` diagnostic, `judge_verdicts.md` lists the judge's verdict on every paper you submitted in your submitted order, `score_calculation.md` shows the score formula with that query's actual component numbers filled in (see "Scoring (per query)"), `submission.json` preserves the full submitted payload (results beyond the scored depth keep their id and position, but their evidence text is replaced with an omission marker — it was never judged), and `evidence_grounding.md` (present only when something was discarded) names the papers whose evidence failed the grounding check and the offending passage — together they separate "the right papers were never retrieved" from "retrieved but rejected" from "evidence discarded as ungrounded" and let you audit your ranking.

The judge's own LLM spend appears in each problem's `result.json` as `other_cost`. That field is informational only — it is never penalized, never counts toward your batch mean, and is outside your control. Do not optimize for it.

`specific_f1` and `metadata_f1` queries score by exact-match against `corpus_ids` — no judge at all.

## Iteration-aggregate score

Per-example scoring is continuous F1 in [0, 1]. At the end of each iteration, your batch is combined into a single score: your mean F1 (on a 0–100 scale) minus a cost penalty when your mean batch agent-spend exceeds the threshold. The penalty is expressed in fully-wrong-query units — each ${COST_PER_ERROR} of mean spend over ${COST_THRESHOLD} subtracts one error-equivalent (one query's worth of F1) from your score. Only `model_registry` handle calls are metered — tool calls are free.

${COST_PENALTY_TABLE}

## Time budget

Your agent times out and the query scores 0 if a single query takes more than **${EVAL_TIMEOUT_MIN} minutes** of wall-clock. Per-query wall-clock is recorded as `eval_wall_clock_seconds` in each problem's `result.json`.

## Scoring (per query)

The scorer returns a single float in [0, 1] per sample; the overall score is the plain mean of those per-query floats. The formulas below are its exact computation. Each training problem's diagnostics include a `score_calculation.md` showing these formulas with that query's actual numbers filled in (precision/recall/hits on the exact-match paths; rank/recall/K on `semantic_f1`).

**`specific_f1` / `metadata_f1`** — every submitted id is graded Perfect iff it is in the gold `corpus_ids` set; order never enters. With `hits` = |submitted ∩ gold|, over your whole list (first 250 entries):

```
precision = hits / #submitted        recall = hits / #gold
score     = harmonic(precision, recall)          (0 if either is 0)
```

Worked example: predicted `corpus_ids` = `["123456789", "987654321"]`, gold = `["123456789"]` → precision 0.5, recall 1.0, F1 0.667.

**`semantic_f1`** — the LLM judge grades each judged paper 0–3 (0 = Not, 1 = Somewhat, 2 = Highly, 3 = Perfectly Relevant) against the query's weighted `relevance_criteria`. Let `g₁..gₙ` be those grades *in your submitted order*, and K the benchmark's per-query estimate of the total relevant papers in the corpus (K is the recall denominator; it varies widely per query and you do not have it at query time):

```
DCG(seq) = Σᵢ seq[i] / ln(i + 1)                 (i = 1-based position)
rank     = (DCG(g) − DCG(g sorted ascending))
           / (DCG(g sorted descending) − DCG(g sorted ascending))
           — defined as 0 when the denominator is 0 (all grades equal)
recall   = |{i ≤ K : gᵢ = 3}| / K                (first K judged papers, submitted order)
score    = harmonic(rank, recall)                (0 if either term is 0)
```

Each paper's grade `gᵢ` is itself derived from the judge's per-criterion verdicts. The judge does not read a paper holistically — it rates every criterion `c` (gold weights `w_c` sum to 1) as Not / Somewhat / Perfectly Relevant, i.e. `r_c ∈ {0, 1, 3}` (there is no per-criterion "Highly"), then:

```
weighted = min(1, Σ_c  w_c · r_c / 3)
gᵢ       = 0 if weighted ≤ 0.25;  1 if ≤ 0.67;  2 if ≤ 0.99;  else 3
```

Grade 2 ("Highly Relevant") exists only as this threshold band, never as a judge output. Because grade 3 needs `weighted > 0.99` and each `r_c/3 ≤ 1`, a paper reaches 3 — the only grade that earns recall — essentially only when every weighted criterion is judged Perfectly Relevant; a single unsupported criterion caps it at 2 and earns zero recall. The judge's `relevant_snippet` output does not enter the score.

## Diagnostics

Any `print()` output from the agent is captured and included in evaluation diagnostics as `agent_stdout`. Use `print()` to log anything you think would be helpful for you to see when improving the agent in later rounds.

${SESSION_ACCESS_NOTE}
