# Third-party notices — PaperFindingBench (AstaBench) example

Covers what `examples/asta_paper_finder/requirements.txt` brings in. Core framework
dependencies are in the repository's [`NOTICE.md`](../../NOTICE.md).

## Python packages

| Package | License | Project |
| --- | --- | --- |
| `astabench` | Apache-2.0 | https://github.com/allenai/asta-bench |

`astabench` bundles `inspect_ai` (MIT), the Asta MCP tool factories, the dataset loaders and
the scorer. Leaderboard submissions additionally require `litellm==1.88.1` (MIT), installed
separately — see the README's submission section.

## Services

- **Asta MCP corpus endpoint** (`ASTA_TOOL_KEY`) — AI2-operated; the task's only retrieval
  surface. Request a key via https://allenai.org/asta/resources/mcp. Rate limited to 10
  requests/second per endpoint. Paper metadata and abstracts returned by these tools come
  from Semantic Scholar.
- **OpenAI API** — required twice over: as a solver-model provider, and for the benchmark's
  GPT-4o relevance judge, which scores every `semantic_f1` query.
- **Anthropic and Google APIs** — additional solver models in `model_registry.py`. The
  evaluator hard-requires all three provider keys at startup because evolution may select
  any of them.

All governed by each provider's own terms.

## Datasets

**PaperFindingBench**, from `allenai/asta-bench`. **Not redistributed here.**

| | |
| --- | --- |
| Source | https://huggingface.co/datasets/allenai/asta-bench |
| License | Apache-2.0 (the AstaBench project) |
| Access | **Gated**: accept the terms on the dataset page (approval is automatic), then supply `HF_ACCESS_TOKEN` / `HF_TOKEN` |

Because the dataset is gated, its terms are something you accept directly from AI2. This
repository never redistributes the queries or gold labels: archived runs under
`example_runs/` and `submissions/` record `query_id`s, scores and cost metrics only.

Agent outputs archived in those runs do quote short passages from paper abstracts as
evidence. Those are retrieved live from the Asta MCP corpus at evaluation time and originate
with the papers' publishers — they are not benchmark data.
