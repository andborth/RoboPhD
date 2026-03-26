# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- ~50GB disk space for BIRD dataset (Text2SQL domain only)
- Anthropic API key

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/RoboPhD.git
cd RoboPhD
```

## Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Configure API Keys

RoboPhD requires an Anthropic API key for task generation and evaluation.

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_anthropic_api_key_here"

# Reload your shell
source ~/.zshrc  # or ~/.bashrc
```

## Step 4: Install Claude Code CLI (Required for Evolution)

Evolution requires Claude Code CLI. Install it following the official documentation:
https://docs.anthropic.com/en/docs/claude-code

Verify installation:
```bash
claude --version
```

## Step 5: Download BIRD Dataset (Text2SQL Only)

Run the download script:

```bash
./benchmark_resources/download_bird.sh
```

This will download and extract:
- Training set (~40GB)
- Development set (~2GB)
- Test set metadata

**Manual download**: If the script fails, download from [BIRD Benchmark](https://bird-bench.github.io/) and extract to `benchmark_resources/datasets/`.

## Step 6: Verify Installation

Run a quick test:

```bash
# Can't Be Late (no API key needed — pure simulation)
bash scripts/download_cant_be_late_traces.sh
python scripts/run_robophd.py --task cant_be_late --num-iterations 2 \
  --engine-config '{"examples_per_iteration": 3}'

# Text2SQL (requires steps 5-6 + API key)
python scripts/run_robophd.py --task text2sql --num-iterations 2 \
  --engine-config '{"examples_per_iteration": 3}'

# List all valid parameters for a task
python scripts/run_robophd.py --task cant_be_late --list-params
```

If successful, you'll see iteration progress and a final report.

## Directory Structure After Installation

```
RoboPhD/
├── RoboPhD/                    # Core code
│   ├── text2sql_agents/        # Text2SQL seed agents
│   ├── arcagi1_agents/         # ARC-AGI-1 seed agents
│   ├── cant_be_late_agents/    # Can't Be Late seed agents
│   ├── docfinqa_agents/        # DocFinQA seed agents
│   ├── evolution_strategies/   # Evolution strategies (all domains)
│   ├── adapters/               # Task evaluators and adapters
│   ├── tasks/                  # Task registry
│   └── ...
├── scripts/                    # Entry points (run_robophd.py, eval_test_set.py, etc.)
├── benchmark_resources/
│   └── datasets/               # BIRD dataset (~40GB, Text2SQL only)
├── configs/                    # Configuration files
└── ../robophd_runs/            # Created during runs (outside repo)
    ├── robophd/                # RoboPhD ELO evolution runs
    ├── gepa/                   # GEPA optimization runs
    ├── agent_tests/            # Standalone agent evaluations
    └── results/                # Results JSON files and run symlinks
```

## Troubleshooting

### "Database is locked" errors
Run ground truth pre-computation:
```bash
python RoboPhD/tools/precompute_ground_truth.py
```

### Out of memory errors
Reduce concurrency:
```bash
python scripts/run_robophd.py --task cant_be_late --num-iterations 5 \
  --engine-config '{"max_concurrent": 2}'
```

### Claude CLI not found
Ensure Claude Code CLI is installed and in your PATH:
```bash
which claude
claude --version
```

### API rate limits
The system handles rate limits automatically. For high-throughput runs, consider using a paid API tier.

## Next Steps

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation, key commands, and configuration options.
