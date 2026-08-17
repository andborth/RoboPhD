# Third-party notices — RoboPhD core

RoboPhD is licensed under the MIT License (see [`LICENSE`](LICENSE)). This file covers
third-party material that ships with, or is required by, **the core framework** — what you
get from `pip install -r requirements.txt` and use through `optimize_anything()`.

**Each example declares its own dependencies, datasets and services separately**, in
`examples/<domain>/THIRD_PARTY.md`. Installing an example's `requirements.txt` pulls in
licenses this file does not cover — including, for `protein_go`, GPL-3.0 components. Read
that example's file before redistributing anything built on it.

## Vendored code

One subtree in this repository is not ours and is not MIT:

| Path | Upstream | License |
| --- | --- | --- |
| `examples/cant_be_late/utils/` | [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa), carrying the SkyPilot spot simulator from [`UCB-ADRS/ADRS`](https://github.com/UCB-ADRS/ADRS) | MIT (gepa) over Apache-2.0 (ADRS) |

Full license texts and the provenance chain are in
[`examples/cant_be_late/utils/README.md`](examples/cant_be_late/utils/README.md) and
`examples/cant_be_late/utils/LICENSES/`. Nothing else in the repository vendors third-party
source.

## Core dependencies

Declared in [`requirements.txt`](requirements.txt). Licenses as published by each project;
this is a courtesy inventory, not a substitute for each package's own license, and it covers
direct dependencies only, not their transitive closure.

| Package | License | Project |
| --- | --- | --- |
| `anthropic` | MIT | https://github.com/anthropics/anthropic-sdk-python |
| `tqdm` | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| `func-timeout` | LGPL-3.0 (see note) | https://github.com/kata198/func_timeout |
| `psutil` | BSD-3-Clause | https://github.com/giampaolo/psutil |
| `mcp` | MIT | https://github.com/modelcontextprotocol/python-sdk |
| `litellm` | MIT | https://github.com/BerriAI/litellm |
| `tree-sitter` | MIT | https://github.com/tree-sitter/py-tree-sitter |
| `tree-sitter-bash` | MIT | https://github.com/tree-sitter/tree-sitter-bash |

With `--engine gepa`, [`requirements-gepa.txt`](requirements-gepa.txt) adds:

| Package | License | Project |
| --- | --- | --- |
| `gepa` | MIT | https://github.com/gepa-ai/gepa |
| `cloudpickle` | BSD-3-Clause | https://github.com/cloudpipe/cloudpickle |

### Copyleft in core

Two entries are not fully permissive. Both are weak, file- or library-level copyleft, and
RoboPhD neither modifies nor redistributes either package — `pip` installs them from PyPI
under their own terms, and the user remains free to inspect or replace them. The practical
obligation is disclosure, which is what this section is.

- **`func-timeout` — LGPL-3.0.** Imported as a library for per-evaluation timeouts. Note
  that its package metadata declares `LGPLv2`, but the LICENSE file it ships is *GNU Lesser
  General Public License, Version 3*, and its repository is labelled LGPL-3.0. The metadata
  is stale; automated license scanners will report the wrong version for this package.
- **`tqdm` — MPL-2.0 AND MIT.** The `AND` is conjunctive, not a choice: tqdm's own `LICENCE`
  places `files: *` under MPL-2.0 and specific historical contributions to `tqdm/_tqdm.py`
  under MIT, so the package contains both and both travel with it. MPL-2.0's reciprocity
  attaches to modified MPL-covered files that you distribute; we distribute none.

## Services and tools

These are prerequisites, not licensed components — the same category as requiring `git`.
Nothing here is bundled or redistributed, and each is governed by its provider's terms
rather than by an open-source license.

- **Claude Code CLI** — the evolution engine runs it as a subprocess for every domain, using
  your own Claude Max credentials. Governed by
  [Anthropic's terms](https://www.anthropic.com/legal/commercial-terms). Install per
  [the Claude Code docs](https://docs.anthropic.com/en/docs/claude-code).
- **Anthropic API** (`ANTHROPIC_API_KEY_FOR_ROBOPHD`) — the GEPA engine's reflection model.
  Same terms; required only with `--engine gepa`.

Solver-model providers (OpenAI, Google, OpenRouter) are per-example and are documented in
each example's `THIRD_PARTY.md`, not here. `cant_be_late` and `sudoku` make no LLM calls at
all.

## Datasets

The core framework ships and downloads **no datasets**. Every benchmark is fetched by the
user from its own publisher, under that publisher's terms, by the example that needs it —
see each example's `THIRD_PARTY.md`. Several carry share-alike terms (BIRD and DS-1000 are
CC BY-SA 4.0) and one is access-gated (`allenai/asta-bench`).

Run artifacts committed under `example_runs/` and `submissions/` record sample IDs, scores
and cost metrics. They contain no benchmark query text and no gold labels.
