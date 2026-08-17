# Third-party notices — DocFinQA example

Covers what `examples/docfinqa/requirements.txt` brings in. Core framework dependencies are
in the repository's [`NOTICE.md`](../../NOTICE.md).

## Python packages

| Package | License | Project |
| --- | --- | --- |
| `huggingface_hub` | Apache-2.0 | https://github.com/huggingface/huggingface_hub |

`litellm` is not listed: it is a core dependency and comes from the root `requirements.txt`.

## Services

- **OpenAI API** — the solver model (`gpt-4.1-mini`) and embeddings
  (`text-embedding-3-small`). Your own API key; governed by
  [OpenAI's terms](https://openai.com/policies/terms-of-use).

## Datasets

**`kensho/DocFinQA`**, fetched from HuggingFace at run time. **Not redistributed here.**

| | |
| --- | --- |
| Source | https://huggingface.co/datasets/kensho/DocFinQA |
| License | MIT (per the dataset card's license tag) |
| Paper | DocFinQA: A Long-Context Financial Reasoning Dataset (ACL 2024) |

The underlying documents are SEC 10-K filings — US government-published corporate disclosures,
which are public records.
