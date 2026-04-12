# Sudoku Example

Evolve pure-Python Sudoku solvers that maximize speed while maintaining correctness.

Pure algorithmic optimization — no LLM calls needed for the solver. Only Claude Code is needed for evolution.

## Setup

```bash
# Install core dependencies (from repo root)
pip install -r requirements.txt

# Install datasets for HuggingFace puzzle loading
pip install datasets

# Set API key for evolution
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"
```

## Quick Start

Run from the RoboPhD repo root:

```bash
# Smoke test (2 iterations, ~60 evaluations)
python examples/sudoku/main.py --num-iterations 2

# Full run (budget-limited, typically ~21 iterations)
python examples/sudoku/main.py

# With test-set evaluation after optimization
python examples/sudoku/main.py --eval-test-set
```

## Resume / Extend

```bash
# Resume from checkpoint
python examples/sudoku/main.py --resume ../robophd_runs/robophd/sudoku_20260401_120000

# Add 5 more iterations
python examples/sudoku/main.py --resume <dir> --extend 5
```

## Configuration

```bash
# Custom engine config
python examples/sudoku/main.py --engine-config '{"include_evolution_rankings": false}'
```

## About the Benchmark

Puzzles from the [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) dataset on HuggingFace. Difficulty ratings range from 0 (easy) to 465 (extreme).

Scoring: `max(0.0, 1.0 - elapsed_seconds * 100)` for correct solutions, 0.0 for incorrect. Pure Python only — no ctypes, subprocess, or compiled extensions.
