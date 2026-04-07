# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- ~50GB disk space for BIRD dataset (Text2SQL domain only)
- Anthropic API key

## Step 1: Clone the Repository

```bash
git clone https://github.com/andborth/RoboPhD.git
cd RoboPhD
```

## Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-gepa.txt  # adds dspy, datasets (needed for ARC-AGI)
```

## Step 3: Configure API Keys

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_anthropic_api_key_here"

# For ARC-AGI (Gemini solver via OpenRouter)
export OPENROUTER_API_KEY="sk-or-..."

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

## Step 5: Download Domain-Specific Data

### ARC-AGI
No download needed — dataset loads automatically from HuggingFace.

### Can't Be Late
```bash
bash scripts/download_cant_be_late_traces.sh
```

### Text2SQL (BIRD dataset, ~50GB)
```bash
./benchmark_resources/download_bird.sh
```

### DocFinQA
No download needed — dataset loads automatically from HuggingFace.

## Step 6: Verify Installation

Run a quick test with ARC-AGI-1:

```bash
python examples/arc_agi_1/main.py --evaluation-budget 60 --num-iterations 2
```

If successful, you'll see iteration progress and a final report.

## Directory Structure After Installation

```
RoboPhD/
├── RoboPhD/                    # Core framework
│   ├── api.py                  # optimize_anything(), eval_candidate()
│   ├── researcher.py           # ELO evolution engine
│   ├── evolution_strategies/   # Evolution strategies (all domains)
│   ├── adapters/               # Shared utilities (candidate_utils, etc.)
│   └── ...
├── examples/                   # Self-contained benchmark examples
│   └── arc_agi_1/              # ARC-AGI-1 (more coming soon)
│       ├── main.py             # Entry point
│       ├── evaluator.py        # Domain evaluator
│       ├── background.md       # Domain description for evolution AI
│       └── seeds/              # Seed agents
├── scripts/                    # Utility scripts
├── benchmark_resources/
│   └── datasets/               # BIRD dataset (~50GB, Text2SQL only)
├── configs/                    # Configuration files
└── ../robophd_runs/            # Created during runs (outside repo)
    └── robophd/                # Experiment output directories
```

## Troubleshooting

### Out of memory errors
Reduce concurrency:
```bash
python examples/arc_agi_1/main.py --max-workers 4
```

### Claude CLI not found
Ensure Claude Code CLI is installed and in your PATH:
```bash
which claude
claude --version
```

### API rate limits
The system retries transient rate limits automatically. For sustained rate limiting, reduce `--max-workers`.

## Next Steps

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation, key commands, and configuration options.
