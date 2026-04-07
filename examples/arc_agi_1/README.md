# ARC-AGI-1 Example

Evolve agents that solve ARC-AGI abstract reasoning puzzles using LLM calls.

## Setup

```bash
# Install core dependencies
pip install -r ../../requirements.txt
pip install -r ../../requirements-gepa.txt  # adds dspy, datasets

# Set API keys
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"   # for evolution (Claude Code)
export OPENROUTER_API_KEY="sk-or-..."              # for the solver (Gemini)
# Recommended: link your Google API key at https://openrouter.ai/settings/integrations
# to get your own Gemini rate limits (otherwise you share limits with all OpenRouter users)
```

## Quick Start

Run from the RoboPhD repo root:

```bash
# Smoke test (2 iterations, ~60 evaluations)
python examples/arc_agi_1/main.py --evaluation-budget 60 --num-iterations 2

# Full run (budget-limited, typically ~21 iterations)
python examples/arc_agi_1/main.py

# With test-set evaluation after optimization
python examples/arc_agi_1/main.py --eval-test-set
```

## Cost Tiers

| Config | Solver Model | Cost Budget | Flag |
|--------|-------------|-------------|------|
| Default | Gemini 2.5 Flash Lite | $0.10/problem | (none) |
| Paper | Gemini 3.1 Flash Lite | $0.25/problem | `--paper-config` |

The default uses a cheaper model and tighter budget, suitable for experimentation. To reproduce paper results:

```bash
python examples/arc_agi_1/main.py --paper-config
```

## Resume / Extend

```bash
# Resume from checkpoint
python examples/arc_agi_1/main.py --resume ../robophd_runs/robophd/optimize_anything_20260401_120000

# Add 5 more iterations
python examples/arc_agi_1/main.py --resume <dir> --extend 5

# Restart from iteration 10
python examples/arc_agi_1/main.py --resume <dir> --from-iteration 10
```

## Configuration

```bash
# Custom engine config
python examples/arc_agi_1/main.py --engine-config '{"include_evolution_rankings": false}'

# Adjust concurrency
python examples/arc_agi_1/main.py --max-workers 4
```
