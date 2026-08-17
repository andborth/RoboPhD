# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

- **Recommended CLI Tools**: `jq` and `tree` (install via your package manager if not present)

## Project Overview

RoboPhD evolves AI agents to improve task performance without human intervention. The primary interface is the `optimize_anything()` API, with self-contained examples for each domain. Three optimization engines are available: RoboPhD (Elo-based evolution), GEPA (Pareto reflective), and Autoresearch (greedy hill-climbing).

**Active domains:**
- **ARC-AGI-1**: Abstract reasoning agents (Gemini via OpenRouter)
- **Can't Be Late**: Cloud scheduling strategies, no LLM calls (NSDI'24)
- **DocFinQA**: Retrieval + QA for long financial documents (GPT-4.1-mini)
- **Text2SQL**: SQL generation with `llm()` + `test_sql()` callables (BIRD benchmark)
- **Sudoku**: Pure-Python solver optimization, no LLM calls
- **DS-1000 (AstaBench)**: Inspect-AI `@solver` code agents, Docker `python_session`; agent picks from a multi-family `model_registry` (OpenAI GPT-5.4/5.5, Claude 4.x, Gemini 3.x)
- **Protein GO**: GO-MF term prediction from sequences via BLAST/ESM/`llm()` callables (Gemini 3.1 Flash Lite via OpenRouter)

**Paper**: [RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets](https://arxiv.org/abs/2604.04347)

## Domains

| Domain | Benchmark | Agent Files | Example |
|--------|-----------|-------------|---------|
| ARC-AGI-1 | ARC-AGI (HuggingFace) | `agent.py` | `examples/arc_agi_1/` |
| Can't Be Late | AWS spot traces (NSDI'24) | `agent.py` | `examples/cant_be_late/` |
| DocFinQA | DocFinQA (ACL 2024) | `agent.py` | `examples/docfinqa/` |
| Text2SQL | BIRD | `agent.py` + `analyze_db.py` | `examples/text2sql/` |
| Sudoku | sapientinc/sudoku-extreme | `agent.py` | `examples/sudoku/` |
| DS-1000 (AstaBench) | DS-1000 (AstaBench) | `agent.py` | `examples/asta_ds1000/` |
| Protein GO | CAFA Fmax (ProteInfer split + Price-149) | `agent.py` | `examples/protein_go/` |

Each example is self-contained: `main.py` (entry point), `evaluator.py` (scoring), `objective.md` + `background.md` (evolution context), and `seeds/baseline/` (seed agent).

## Key Commands

### Environment Setup
```bash
pip install -r requirements.txt

# Install Claude Code CLI (required for evolution — uses Claude Max auth, not an API key)
# See: https://docs.anthropic.com/en/docs/claude-code

# For the GEPA engine (adds gepa, dspy, datasets, cloudpickle)
pip install -r requirements-gepa.txt

# For the GEPA engine: Anthropic API key for the reflection model
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."
```

Per-example setup (solver API keys, dataset downloads, extra pip installs) lives in each `examples/<domain>/README.md`.

### Running Evolution

Each example is a standalone script calling `optimize_anything()`. Runs stop early when the `evaluation_budget` is exhausted (default 1500 evaluations).

```bash
# ARC-AGI evolution
python examples/arc_agi_1/main.py

# Can't Be Late evolution (download traces first)
bash examples/cant_be_late/download_traces.sh
python examples/cant_be_late/main.py

# DocFinQA evolution
python examples/docfinqa/main.py

# Text2SQL evolution (download BIRD dataset first)
bash benchmark_resources/download_bird.sh
python examples/text2sql/main.py

# Sudoku evolution
python examples/sudoku/main.py

# Quick smoke test (any domain)
python examples/arc_agi_1/main.py --num-iterations 2 --evaluation-budget 60
```

### Engine Selection

All examples support three optimization engines via `--engine`:

```bash
# RoboPhD Elo competition (default)
python examples/docfinqa/main.py

# GEPA Pareto-based reflective evolution
python examples/docfinqa/main.py --engine gepa

# Autoresearch single-session greedy hill-climbing
python examples/docfinqa/main.py --engine autoresearch
```

### Resume and Extend
```bash
# Resume from checkpoint
python examples/arc_agi_1/main.py \
  --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016

# Extend completed run with additional iterations
python examples/arc_agi_1/main.py \
  --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016 \
  --extend 5
```

### Test-Set Evaluation
```bash
# Evaluate best agent from a run on the held-out test set
python examples/arc_agi_1/main.py --eval-test-set  # after optimization

# Evaluate a prior run without re-optimizing
python examples/arc_agi_1/main.py \
  --eval-only --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016

# Evaluate a specific named agent from the pool (defaults to the best-Elo agent)
python examples/arc_agi_1/main.py \
  --eval-only --resume ../robophd_runs/robophd/arc_agi_1_20260322_183016 \
  --eval-agent iter12_some_agent
```

## System Architecture

The `optimize_anything()` API supports three engines, selected by config type:

- **`RoboPhDConfig`** (default): Multi-agent Elo competition with Deep Focus refinement
- **`GEPAConfig`**: Pareto-based reflective text evolution
- **`AutoresearchConfig`**: Single Claude Code session with greedy experimentation

### Seeding

`optimize_anything()` takes one seed parameter, `seed_agents`, holding the agents the run
starts from as `{agent name: source}`:

```python
seed_agents={"baseline": HERE / "seeds" / "baseline"}      # every example
seed_agents={"063_opus5": winner_dir, "355_fable": ...}    # several prior winners
seed_agents={"baseline": {"agent.py": src}}                # artifacts in memory
```

An agent's identity in this architecture **is its directory**, so the keys are agent names —
what shows up in Elo tables, reports, and `--eval-agent`. Values are either a `Path` to an
agent directory (walked by `candidate_utils.read_agent_dir`, skipping dot-prefixed entries
and `__pycache__`) or the artifacts themselves as `{relative path: contents}`. A bare `str`
is rejected: it would be ambiguous between the two, and the likely mistake is passing one
agent's artifacts where a pool of agents belongs. The number of artifacts per agent is free
(text2sql seeds two files); every agent in a pool must have the same set, since they define
one `file_mapping`.

Every seed enters at Elo 1500 and competes from iteration 1. GEPA and Autoresearch evolve a
single agent and require a one-entry pool rather than silently dropping the rest.

Seed names must be safe single path components and must **not** start with `iter<N>` —
`--from-iteration` archival parses that prefix as an iteration number of the *current* run
and would move the seed out of the pool. Prefix seeds instead (`seed_<label>`).

Seeding more agents than `agents_per_iteration` loses none of them: untested agents have
selection priority (`researcher.select_agents_for_iteration` Priority 3), so a 4th seed in a
3-slot run enters at iteration 2 rather than needing a wider round-robin.
`examples/asta_paper_finder/main.py --seed-runs LABEL=RUN_DIR ...` is the worked example —
it resolves each run's best-Elo agent via `runner_utils.find_best_agent`.

### How Evolution Works (RoboPhD Engine)

```
    ┌─────────────────────────────────────────────────────────────┐
    │                      ITERATION CYCLE                        │
    │                                                             │
    │  ┌──────────────────┐         ┌────────────────────┐        │
    │  │  EVOLUTION AI    │ Creates │  AGENT ARTIFACTS    │        │
    │  │  (Claude Code    │────────▶│  (per file_mapping) │        │
    │  │   CLI session)   │         └────────┬───────────┘        │
    │  └──────────────────┘                  │                    │
    │           ▲                             ▼                    │
    │           │                    ┌────────────────────┐        │
    │   Performance                  │  EVALUATOR FN      │        │
    │   data from                    │  Black-box scoring │        │
    │   prior iterations             │  (candidate,example)│       │
    │           │                    │   → (score, diag)  │        │
    │           │                    └────────┬───────────┘        │
    │           │                             │                    │
    │           │                             ▼                    │
    │  ┌────────┴─────────┐         ┌────────────────────┐        │
    │  │  AGENT RANKINGS  │◀────────│  Elo COMPETITION   │        │
    │  │  Top agents      │         │  Head-to-head on   │        │
    │  │  inform next     │         │  sampled problems   │        │
    │  │  evolution round │         └────────────────────┘        │
    │  └──────────────────┘                                       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

## Key Classes and Files

### API
- **`RoboPhD/api.py`**: `optimize_anything()`, `eval_candidate()`, `eval_run()`, config dataclasses
- **`RoboPhD/engines/`**: GEPA and Autoresearch engine wrappers

### Core
- **`RoboPhD/researcher.py`**: ParallelAgentResearcher — Elo evolution loop
- **`RoboPhD/elo_reachability.py`**: Elo update formula (single source) + the reachability search behind the Elo-reachability guard; no framework imports, so it is unit-testable standalone
- **`RoboPhD/evolution.py`**: Evolution strategy selector
- **`RoboPhD/deep_focus_evolution_manager.py`**: Multi-round evolution with testing
- **`RoboPhD/meta_evolution_manager.py`**: Meta-evolution for strategy improvement
- **`RoboPhD/domains/external/domain.py`**: Bridges evaluator functions into the evolution loop

### Utilities
- **`RoboPhD/eval_utils.py`**: `exec_with_stdout_capture`, `run_parallel_eval`, `retry_on_rate_limit`
- **`RoboPhD/candidate_utils.py`**: `extract_candidate` / `materialize_candidate`
- **`RoboPhD/runner_utils.py`**: `find_best_agent`, `to_litellm_model`, `CostTrackingLM`
- **`RoboPhD/config_manager.py`**: Delta-based configuration management
- **`RoboPhD/config.py`**: Model mappings, API keys

## Evolution System

### Evolution Strategies
- `RoboPhD/evolution_strategies/` — `cross_pollination`, `data_focus`, `refinement`, `use_your_judgment` (default)

### Advanced Configuration (via `--engine-config`)

With `--engine gepa` or `--engine autoresearch`, `--engine-config` keys map to `GEPAConfig` / `AutoresearchConfig` fields (unknown keys fail loudly):

```bash
# GEPA reflection model
python examples/docfinqa/main.py --engine gepa --engine-config '{"reflection_model": "fable-5"}'
```

For the default RoboPhD engine:

```bash
# Evolution schedule
python examples/arc_agi_1/main.py --engine-config '{
  "evolution_strategy": "data_focus",
  "config_schedule": {"3": {"evolution_strategy": "none"}}
}'

# Deep Focus tuning
python examples/text2sql/main.py --engine-config '{
  "new_agent_test_rounds": 2,
  "evolution_model": "opus-5"
}'

# Weighted random strategies
python examples/cant_be_late/main.py --engine-config '{
  "use_weighted_random": true,
  "weighted_random_configs": [
    [{"evolution_strategy": "data_focus"}, 50],
    [{"evolution_strategy": "refinement"}, 30],
    [{"evolution_strategy": "none"}, 20]
  ]
}'
```

### Elo System
- **K-factor**: 32, **Initial Elo**: 1500
- **Tie Handling**: 0.5 points each, random winner selection
- **Clone detection**: -200 Elo penalty for identical predictions
- **Elo update**: `RoboPhD/elo_reachability.calculate_elo_updates` — `researcher.py` delegates to it, so the ladder and the reachability projection cannot use different formulas

### Elo-Reachability Guard (on by default)

Late in a run a newly evolved agent can be arithmetically incapable of climbing from 1500 to the incumbent's rating before the budget runs out. `find_best_agent` selects by max Elo, so such an agent cannot be the run's output: it burns an evolution session *and* takes a round-robin slot that would otherwise compare the agents still in contention. When that is provable, the guard switches the iteration to `greedy` (no evolution, deterministic top-k by Elo).

On by default since 2026-08-05, on the strength of replaying it over the 121 eligible archived runs: it fires in 66% of them, and in no case would it have suppressed the run's own winning agent. To turn it off for a run:

```bash
python examples/asta_paper_finder/main.py \
  --engine-config '{"elo_reachability_guard": false}'
```

| Key | Default | Meaning |
|---|---|---|
| `elo_reachability_guard` | `true` | the switch |
| `elo_reachability_min_history` | `5` (= `TRAILING_WINDOW`) | completed iterations required before it can fire |

`min_history` does two jobs with one number, which is why it defaults to the horizon's averaging window rather than its own literal: iteration 1 runs no evolution and costs ~half a steady-state iteration, so a short trailing mean overstates the horizon ~2×; and it keeps the guard off smoke-test runs (`--num-iterations 3` or `5`, extended later), which would otherwise lose their *final* evolution round. It covers caps ≤5 — raise it if you routinely smoke-test longer.

**Mechanics** (`RoboPhD/elo_reachability.py`): reachability is existential, so the search returns on the first line of play where the new agent leads. `reachable=False` therefore means "no line exists", not "a heuristic didn't find one"; if the node budget runs out first the verdict fails safe (declines to fire) rather than firing unproven. The horizon takes both terminators — `evaluation_budget` and `num_iterations` — whichever binds first. Once fired it stays greedy, since the verdict only deteriorates as budget drains, but it is re-checked every iteration so `--extend` restores the displaced strategy. Decisions go through `ConfigManager.apply_delta` with source `elo_reachability`, landing in `config_change_history` and the checkpoint.

**King-of-the-Hill runs are exempt** (`agents_per_iteration: 2` with `oldest_agent_wins_ties: true`). A KotH run's result is whichever agent won the last round, not the Elo leader (`runner_utils.find_last_winner`), so a new agent becomes the output by winning one round whatever its rating and nothing is ever dead weight. ~20% of archived runs are this shape, so the exemption is load-bearing rather than an edge case.

**Resume respects the stored value, on or off.** A default flip changes new runs only: a checkpoint stores both the defaults snapshot in force when it was written and its resolved iteration-1 config, and either alone is enough to carry the old value forward — so an in-flight campaign resumed after a flip keeps whatever it was running under, and a run that explicitly disabled the guard stays disabled. No example packs this key into `engine_overrides` (which *is* re-applied as a delta on resume), so a flagless resume adds nothing.

The two halves of that differ deliberately. An **explicit** `elo_reachability_guard: true` on a KotH config raises at startup — you asked for something incoherent and should hear about it. The **default** being on merely leaves the guard inert there, at both the config layer and in `researcher._apply_reachability_guard`; failing a run that never mentioned the guard would quote a flag its operator never passed.

Replay it against a finished run before enabling it on a real one — free, and it refuses KotH runs the same way:

```bash
python scripts/elo_reachability.py ../robophd_runs/robophd/<run>
python scripts/elo_reachability.py <run> --min-history 0   # show what the floor suppresses
```

Measured over 121 eligible archived runs: fires in 66%, with no run's winning agent suppressed. Almost always on the final iteration (78 of 80 firings); reach scales with Elo spread, and the archive's widest leaders (~1650) buy only one extra iteration. The saving is one evolution session plus the round-robin slots it would have occupied — which matter most at the end, where they are what separates the remaining contenders.

## Development Tips

- **Unit Tests**: `pytest` covers RoboPhD core only. Run `bash scripts/run_tests.sh` for the full sweep — the example suites need one process each (see `pytest.ini` for why), so a bare `pytest` will not catch a regression in `examples/`.
- **Quick Test**: `python examples/cant_be_late/main.py --num-iterations 2 --evaluation-budget 60`
- **Check Progress**: Review `checkpoint.json` and `final_report.md` in the experiment dir
- **Debug Evaluation**: Check `iteration_XXX/agent_YYY/problems/` in the experiment dir
- **Evolution Output**: Check `evolution_output/iteration_XXX/` for Claude's reasoning
- **Run Outputs**: All runs land in `../robophd_runs/` (`robophd/` for Elo, `gepa/` for GEPA, `autoresearch/` for Autoresearch)
- **Cleanup**: `python scripts/cleanup_runs.py` to find and remove short/experimental runs
- **Reachability replay**: `python scripts/elo_reachability.py <run_dir>` replays the Elo-reachability guard over a finished run's `test_history` — what it would have done, and whether that would have suppressed the run's best agent

## Troubleshooting

### Memory (OOM) Errors
- **Symptom**: Process killed with "zsh: killed"
- **Solution**: Use `--max-workers 4` or reduce to 2

### Evolution Failures
- **Claude CLI not found**: Ensure Claude Code CLI is installed
- **Context too long**: Use `--engine-config '{"examples_per_iteration": 3}'`
- **Session errors**: Check Claude CLI authentication with `claude --version`

### Domain-Specific Issues
Domain-specific setup, datasets, and troubleshooting live in each example's README (`examples/<domain>/README.md`).

## License

MIT License - see LICENSE file for details.

Third-party attribution is split to match the dependency structure, and the split is
load-bearing rather than cosmetic — `protein_go` pulls in GPL-3.0 components that core does
not, and several datasets are share-alike or access-gated:

- `NOTICE.md` — core dependencies, the vendored subtree under
  `examples/cant_be_late/utils/` (the only third-party source in the repo), and the services
  the evolution loop needs.
- `examples/<domain>/THIRD_PARTY.md` — that example's packages, datasets, external binaries
  and model providers. Every example has one, including those needing nothing beyond core.

`RoboPhD/unit_tests/test_attribution_coverage.py` asserts every declared package appears in
the file covering it. It cannot check that a recorded license is *correct* — confirm those by
hand against `importlib.metadata` and the package's own LICENSE file, which do not always
agree (`func-timeout` declares LGPLv2 in metadata but ships LGPL-3.0).
