# Example runs

Public-facing snapshots of RoboPhD evolution runs that produced AstaBench leaderboard submissions. The structure is:

```
example_runs/<engine>/<task>/<engine_version>_<descriptor>/
```

- **engine**: `robophd`, `gepa`, or `autoresearch` (which `optimize_anything()` engine drove the run; see [the project README](../README.md) for the engines).
- **task**: the AstaBench task name, e.g. `asta_ds1000`.
- **engine_version**: `v0` for the first generation of submissions; future structural changes get `v1`, `v2`, …
- **descriptor**:
  - For evolved runs: `soft_cap_<X>` where `<X>` is the cost-cap config given to evolution (the lever, not the result). `.` is encoded as `_` for filesystem safety, so `$0.04` becomes `soft_cap_0_04`.
  - For seed baselines: `seed_<model_handle>`, e.g. `seed_gpt54_mini`, `seed_sonnet_4_6`. The original run-internal seed hash is documented in the dir's README.

Layout differs by type. Evolved-run dirs have an `agents/` subdir holding the full lineage. Seed dirs are flat — one `agent.py` at the root + a README.

## What's in this repo vs HuggingFace

These snapshots are **lightweight** (~3MB per evolved run): the agent.py files, top-level run reports, `test_results_final.{json,per_problem.json}`, and one representative `iteration_NNN/` dir with its corresponding `evolution_output/iteration_NNN/` Claude Code session log. That's enough to verify the submitted score against the agent code and get a feel for what an iteration looks like.

The **bulk of each run** — all 15 `iteration_*/` dirs (~30MB), the full `evolution_output/` (~2MB of Claude Code transcripts), `meta_evolution_output/`, and the sandbox-denials log — lives at:

> `huggingface.co/datasets/<TBD>` *(coming soon — link will be added once the dataset repo is set up)*

Following the same `<engine>/<task>/<engine_version>_<descriptor>/` directory convention.

## Caveat on recorded scores

The `test_results_final.json` files in these snapshots were produced by **RoboPhD's internal scoring tooling**, which uses the same `inspect_evals.ds1000.ds1000_scorer` as the official AstaBench leaderboard but runs each sample in a subprocess-isolated `inspect.eval()` call (vs the leaderboard's single batched call across all 900 samples). We track the official scoring as closely as possible, but small variation is possible (sample ordering, concurrency, sandbox state). The leaderboard's verified score after `astabench eval` re-run is the canonical number — see the per-submission `README.md` for the URL once each entry lands.
