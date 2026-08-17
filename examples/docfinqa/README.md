# DocFinQA Example

Evolve agents that answer numerical questions over long financial documents (SEC 10-K filings).

## Setup

```bash
# Install core dependencies (from repo root)
pip install -r requirements.txt

# Install this example's dependencies
pip install -r examples/docfinqa/requirements.txt

# Set API keys
export OPENAI_API_KEY="sk-..."                     # for the solver (gpt-4.1-mini + text-embedding-3-small)
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."  # only if using --engine gepa (reflection model)
```

No data download needed — dataset loads automatically from HuggingFace.

## Quick Start

Run from the RoboPhD repo root:

```bash
# Smoke test (2 iterations)
python examples/docfinqa/main.py --num-iterations 2

# Full run (budget-limited, typically ~17 iterations)
python examples/docfinqa/main.py

# With test-set evaluation after optimization
python examples/docfinqa/main.py --eval-test-set
```

## Resume / Extend

```bash
# Resume from checkpoint
python examples/docfinqa/main.py --resume ../robophd_runs/robophd/docfinqa_20260401_120000

# Add 5 more iterations
python examples/docfinqa/main.py --resume <dir> --extend 5
```

## Configuration

```bash
# Custom engine config
python examples/docfinqa/main.py --engine-config '{"include_evolution_rankings": false}'

# Adjust concurrency (default 8 to avoid fd exhaustion)
python examples/docfinqa/main.py --max-workers 4

# Different models
python examples/docfinqa/main.py --model gpt-4.1 --embed-model text-embedding-3-large
```

## About the Benchmark

From [DocFinQA: A Long-Context Financial Reasoning Dataset](https://huggingface.co/datasets/kensho/DocFinQA) (Reddy et al., ACL 2024). Each problem provides a full SEC 10-K filing (~123K words) and a numerical question requiring arithmetic reasoning over tables and text.
