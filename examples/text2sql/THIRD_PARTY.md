# Third-party notices — Text2SQL example

Core framework dependencies are in the repository's [`NOTICE.md`](../../NOTICE.md).

## Python packages

**None beyond core.** This example has no `requirements.txt`: its only third-party import is
`litellm`, which is a core dependency (`RoboPhD/config.py`, `runner_utils.py`,
`llm_providers.py`) and comes from the root `requirements.txt`.

## Services

- **OpenAI API** — the solver model behind the agent's `llm()` callable. Your own API key;
  governed by [OpenAI's terms](https://openai.com/policies/terms-of-use).

## Datasets

**BIRD** (BIg Bench for LArge-scale Database Grounded Text-to-SQL Evaluation), fetched by
`benchmark_resources/download_bird.sh`. **Not redistributed here.**

| | |
| --- | --- |
| Source | https://bird-bench.github.io — reference implementation at [`AlibabaResearch/DAMO-ConvAI`](https://github.com/AlibabaResearch/DAMO-ConvAI) (MIT) |
| Data license | **CC BY-SA 4.0** |

**CC BY-SA 4.0 is share-alike.** Distributing an adapted version of BIRD's data requires
licensing that adaptation under the same terms. This does not reach RoboPhD's code, which is
not an adaptation of the data, but it does reach anything you build that embeds or derives
from the dataset itself. The SQLite databases the agent queries are BIRD's data.
