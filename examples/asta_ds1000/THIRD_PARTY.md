# Third-party notices — DS-1000 (AstaBench) example

Covers what `examples/asta_ds1000/requirements.txt` brings in. Core framework dependencies
are in the repository's [`NOTICE.md`](../../NOTICE.md).

## Python packages

| Package | License | Project |
| --- | --- | --- |
| `astabench` | Apache-2.0 | https://github.com/allenai/asta-bench |
| `inspect_evals` | MIT | https://github.com/UKGovernmentBEIS/inspect_evals |

`astabench` bundles `inspect_ai` (MIT) and its Docker `python_session` sandbox. Leaderboard
submissions additionally require `litellm==1.88.1` (MIT), installed separately — see the
requirements file for why it cannot be a line in it.

## Services

- **Asta MCP corpus endpoint** (`ASTA_TOOL_KEY`) — AI2-operated; request a key via
  https://allenai.org/asta/resources/mcp. Rate limited to 10 requests/second per endpoint.
- **Solver models** — evolution picks from a multi-provider `model_registry`: OpenAI,
  Anthropic, and Google models. Your own keys for each; governed by each provider's terms.
- **Docker** — required for the `python_session` sandbox the agent executes code in.

## Datasets

**DS-1000**, via AstaBench's wrapper (which applies the canonical 100/900 val/test split).
**Not redistributed here.**

| | |
| --- | --- |
| Source | [`xlang-ai/DS-1000`](https://github.com/xlang-ai/DS-1000), distributed through [`allenai/asta-bench`](https://huggingface.co/datasets/allenai/asta-bench) |
| DS-1000 license | **CC BY-SA 4.0** |
| AstaBench license | Apache-2.0 |
| Access | The `allenai/asta-bench` dataset is **gated**: accept the terms on the dataset page (approval is automatic), then supply `HF_ACCESS_TOKEN` / `HF_TOKEN` |

**CC BY-SA 4.0 is share-alike.** Distributing an adapted version of DS-1000's problems
requires licensing that adaptation under the same terms. It does not reach RoboPhD's code,
which is not an adaptation of the data.

DS-1000's problems derive from StackOverflow questions, which carry their own CC BY-SA
lineage — part of why the dataset is share-alike.
