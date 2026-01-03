# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- ~50GB disk space for BIRD dataset
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

RoboPhD requires an Anthropic API key for SQL generation and evaluation.

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

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

## Step 5: Download BIRD Dataset

Run the download script:

```bash
./benchmark_resources/download_bird.sh
```

This will download and extract:
- Training set (~40GB)
- Development set (~2GB)
- Test set metadata

**Manual download**: If the script fails, download from [BIRD Benchmark](https://bird-bench.github.io/) and extract to `benchmark_resources/datasets/`.

## Step 6: Pre-compute Ground Truth (Recommended)

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
python RoboPhD/researcher.py \
  --num-iterations 1 \
  --config '{"databases_per_iteration": 1, "questions_per_database": 5}'
```

If successful, you'll see iteration progress and a final report.

## Directory Structure After Installation

```
RoboPhD/
├── RoboPhD/                    # Core code
│   ├── agents/                 # Pre-trained agents
│   ├── evolution_strategies/   # Evolution strategies
│   └── ...
├── benchmark_resources/
│   └── datasets/
│       ├── train/              # Training data (~40GB)
│       ├── dev/                # Development data (~2GB)
│       └── ...
├── configs/                    # Configuration files
└── output/                     # Created during runs
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
  --config '{"max_concurrent_dbs": 2}'
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
