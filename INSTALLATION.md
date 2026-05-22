# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- ~50GB disk space for BIRD dataset (Text2SQL domain only)
- Anthropic API key

### Recommended CLI Tools

`jq` and `tree` are recommended but not essential. The evolution AI uses them to explore experiment data more effectively. Omitting them may impact results slightly.

```bash
# macOS
brew install jq tree

# Ubuntu/Debian/WSL
apt install jq tree
```

## Step 1: Clone the Repository

```bash
git clone https://github.com/andborth/RoboPhD.git
cd RoboPhD
```

## Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt

# For GEPA engine support
pip install -r requirements-gepa.txt

# For ARC-AGI (additional deps: dspy, datasets)
pip install -r examples/arc_agi_1/requirements.txt
```

## Step 3: Configure API Keys

Evolution itself uses the Claude Code CLI (Claude Max auth); see Step 4. The keys below are only for
the GEPA reflection model and for per-example solvers.

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)

# For the GEPA engine only (reflection model).
# Not needed for the default RoboPhD Elo engine or autoresearch.
# Also needed for the Text2SQL example (default eval model is haiku-4.5).
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."

# For ARC-AGI (Gemini solver via OpenRouter)
export OPENROUTER_API_KEY="sk-or-..."
# Recommended: link your Google API key at https://openrouter.ai/settings/integrations
# to get your own Gemini rate limits (otherwise you share limits with all OpenRouter users)

# For DocFinQA (gpt-4.1-mini + text-embedding-3-small)
export OPENAI_API_KEY="sk-..."

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
bash examples/cant_be_late/download_traces.sh
```

### Text2SQL (BIRD dataset, ~50GB)
```bash
bash benchmark_resources/download_bird.sh
```

### DocFinQA
No download needed — dataset loads automatically from HuggingFace.

### Sudoku
No download needed — dataset loads automatically from HuggingFace.

## Step 6: Verify Installation

Run a quick test:

```bash
python examples/cant_be_late/main.py --evaluation-budget 60 --num-iterations 2
```

If successful, you'll see iteration progress and a final report.

## Directory Structure After Installation

```
RoboPhD/
├── RoboPhD/                    # Core framework
│   ├── api.py                  # optimize_anything(), eval_candidate(), eval_run()
│   ├── researcher.py           # Elo evolution engine
│   ├── engines/                # GEPA + Autoresearch engine wrappers
│   ├── evolution_strategies/   # Evolution strategy prompts
│   └── ...
├── examples/                   # Self-contained benchmark examples
│   ├── arc_agi_1/              # ARC-AGI abstract reasoning
│   ├── cant_be_late/           # Cloud scheduling optimization
│   ├── docfinqa/               # Financial document QA
│   ├── sudoku/                 # Sudoku solver optimization
│   └── text2sql/               # SQL generation (BIRD benchmark)
│       ├── main.py             # Entry point (same structure in each)
│       ├── evaluator.py        # Domain evaluator
│       ├── background.md       # Domain description for evolution AI
│       └── seeds/              # Seed agents
├── scripts/                    # Utility scripts
├── benchmark_resources/
│   └── datasets/               # BIRD dataset (~50GB, Text2SQL only)
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
