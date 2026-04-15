# ARC-AGI-1 Example

Evolve agents that solve ARC-AGI abstract reasoning puzzles using LLM calls.

## Setup

```bash
# Install core dependencies
pip install -r ../../requirements.txt
pip install -r requirements.txt  # adds dspy, datasets (from examples/arc_agi_1/)

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
python examples/arc_agi_1/main.py --num-iterations 2

# Full run (budget-limited, typically ~21 iterations)
python examples/arc_agi_1/main.py

# With test-set evaluation after optimization
python examples/arc_agi_1/main.py --eval-test-set
```

## Configuration Tiers

Both tiers use **Gemini 3.1 Flash Lite** via OpenRouter but trade off accuracy against cost and speed:

| | Default | `--paper-config` |
|---|---|---|
| Reasoning effort | medium | high |
| Cost budget | $0.10/problem | $0.25/problem |
| Max LLM calls | 20 | 10 |

**Why two tiers?** On the seed agent, high reasoning scores 3.3x better (29% vs 8.7%) but costs 6.8x more ($0.027 vs $0.004/problem) and runs 7.5x slower. The default tier trades seed accuracy for cheaper evaluations, allowing more evolutionary cycles within the same dollar budget — and gives evolution 20 LLM calls to build multi-step strategies.

```bash
# Default: medium reasoning, 20 calls, $0.10/problem
python examples/arc_agi_1/main.py

# Paper: high reasoning, 10 calls, $0.25/problem (reproduces paper results)
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
