# Example runs

Public-facing snapshots of RoboPhD evolution runs that produced AstaBench leaderboard submissions. The structure is:

```
example_runs/<engine>/<task>/<version>_cap_<threshold>_<model>[_<variant>]/
```

- **engine**: `robophd`, `gepa`, or `autoresearch` (which `optimize_anything()` engine drove the run; see [the project README](../README.md) for the engines).
- **task**: the AstaBench task, e.g. `asta_ds1000`, `asta_paper_finder`.
- **version**: a patch-level sequence (`v0_0_1`, `v0_0_2`, …, `v0_1_0`) shared across both benchmarks. A bump marks a change in code state since the previous submission, so runs submitted from the same code state share a version (four PaperFindingBench snapshots are `v0_0_9`).
- **cap_\<threshold\>**: the per-query cost threshold (`cost_threshold`, the free zone) given to evolution: the lever, not the result. `.` is encoded as `_` for filesystem safety, so `$0.063` becomes `cap_0_063`. The descriptor names the threshold only. The penalty slope above it (`cost_per_error`) differed across the early runs, so two snapshots with similar thresholds were not necessarily trained under the same incentive. Each snapshot's README records its `cost_per_error`, and the paper tabulates the settings for every run: [Competing at Every Price Point with Agentic Evolution over a Menu of LLMs](https://arxiv.org/abs/2608.16207).
- **model**: the evolution model (never a solver model):
  - `opus`: Claude Opus 4.7 or 4.8
  - `fable`: Claude Fable 5
  - later models carry their version: `opus5`, `fable51`
- **variant** (optional): a configuration difference worth naming, e.g. `_deep_focus`.

Snapshots renamed into this convention say so on their README's first line ("Originally submitted as …"), since the leaderboard submission and its tarball used the original name.

## What's in a snapshot

- `agents/`: every iteration winner plus the seed, i.e. the full lineage. The submitted agent is named at the top of the snapshot's README.
- `checkpoint.json`, `final_report.md`: the run's configuration, Elo history and summary.
- Two representative `iteration_NNN/` dirs, each with its matching `evolution_output/iteration_NNN/` (the Claude Code session that produced that iteration's agent).
- Internal test results: `test_results_final.{json,per_problem.json}` for DS-1000, `test_results.{json,per_problem.json}` for PaperFindingBench (scored by a non-stock judge, a file carries a judge suffix, e.g. `test_results.judge_gpt-5.6-luna-noprose.json`).

That is enough to verify the internal score against the agent code and to see what an iteration looks like.

## Caveat on recorded scores

The test results in these snapshots come from **RoboPhD's internal scoring tooling**, not the leaderboard. The leaderboard's own `astabench eval` re-run is the canonical number; each snapshot's README records its official result.

- **DS-1000**: internal scoring uses the same `inspect_evals.ds1000.ds1000_scorer` as the leaderboard, but runs each sample in a subprocess-isolated `inspect.eval()` call rather than the leaderboard's single batched call across all 900 samples. Small differences are possible (sample ordering, concurrency, sandbox state).
- **PaperFindingBench**: internal judging is **capped**: the GPT-4o judge grades only the agent's estimated top-K papers per semantic query. Official judging is **uncapped** and grades every submitted paper, which moves the ranking term and can move the score in either direction.
