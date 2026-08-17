# Can't Be Late Example

Evolve cloud scheduling strategies that minimize cost while meeting deadlines, using spot vs on-demand instance selection.

Pure algorithmic optimization — no LLM calls needed for the solver. Only Claude Code is needed for evolution.

## Setup

```bash
# Install core dependencies (from repo root)
pip install -r requirements.txt

# Install this example's dependencies (the vendored simulator's — see
# utils/README.md for that subtree's provenance and license)
pip install -r examples/cant_be_late/requirements.txt

# Only if using --engine gepa (reflection model); the default Elo engine uses
# the Claude Code CLI's Claude Max auth and needs no API key.
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."

# Download trace data (~151MB)
bash examples/cant_be_late/download_traces.sh
```

## Quick Start

Run from the RoboPhD repo root:

```bash
# Smoke test (2 iterations, ~60 evaluations)
python examples/cant_be_late/main.py --num-iterations 2

# Full run (budget-limited, typically ~21 iterations)
python examples/cant_be_late/main.py

# With test-set evaluation after optimization
python examples/cant_be_late/main.py --eval-test-set
```

## Resume / Extend

```bash
# Resume from checkpoint
python examples/cant_be_late/main.py --resume ../robophd_runs/robophd/cant_be_late_20260401_120000

# Add 5 more iterations
python examples/cant_be_late/main.py --resume <dir> --extend 5
```

## Configuration

```bash
# Custom engine config
python examples/cant_be_late/main.py --engine-config '{"include_evolution_rankings": false}'

# Adjust concurrency
python examples/cant_be_late/main.py --max-workers 4

# Longer simulation timeout
python examples/cant_be_late/main.py --simulation-timeout 600
```

## About the Benchmark

From [Can't Be Late: Optimizing Spot Instance Savings under Deadlines](https://github.com/UCB-ADRS/ADRS) (Wu et al., NSDI 2024). The agent decides at each timestep whether to use SPOT (cheap, unreliable), ON_DEMAND (expensive, reliable), or NONE (wait). Scores are negative costs — higher (less negative) is better.
