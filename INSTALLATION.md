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

> **Skip this step for CodeGen domain.** No dataset download is needed for CodeGen.

Run the download script:

```bash
./benchmark_resources/download_bird.sh
```

This will download and extract:
- Training set (~40GB)
- Development set (~2GB)
- Test set metadata

**Manual download**: If the script fails, download from [BIRD Benchmark](https://bird-bench.github.io/) and extract to `benchmark_resources/datasets/`.

## Step 6: Pre-compute Ground Truth (Text2SQL Only)

Pre-computing ground truth prevents "database is locked" errors during research runs:

```bash
# For train-filtered dataset (default)
python RoboPhD/tools/precompute_ground_truth.py

# For dev dataset
python RoboPhD/tools/precompute_ground_truth.py --dataset dev
```

## Step 7: Verify Installation

Run a quick test:

```bash
# Text2SQL (requires steps 5-6)
python RoboPhD/researcher.py \
  --num-iterations 1 \
  --config '{"examples_per_iteration": 1, "problems_per_context": 5}'

# Or verify with CodeGen domain (no dataset needed)
python RoboPhD/researcher.py \
  --num-iterations 1 \
  --config configs/codegen_small_test.json
```

If successful, you'll see iteration progress and a final report.

## Directory Structure After Installation

```
RoboPhD/
├── RoboPhD/                    # Core code
│   ├── agents/                 # Text2SQL agents
│   ├── codegen_agents/         # CodeGen agents
│   ├── evolution_strategies/          # Evolution strategies (all domains)
│   ├── evolution_strategies_text2sql/ # Text2SQL evolution strategies (legacy)
│   └── ...
├── benchmark_resources/
│   └── datasets/
│       ├── train/              # Training data (~40GB, Text2SQL)
│       ├── dev/                # Development data (~2GB, Text2SQL)
│       └── ...
├── configs/                    # Configuration files
└── evolution/                  # Created during runs
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
python RoboPhD/researcher.py --num-iterations 5 \
  --config '{"max_concurrent": 2}'
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

See [QUICKSTART.md](QUICKSTART.md) for a 5-minute tutorial on running your first evolution experiment.
