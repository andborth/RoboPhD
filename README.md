# RoboPhD: Evolving AI Agents Without Human Domain Knowledge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01126-b31b1b.svg)](https://arxiv.org/abs/2601.01126)

RoboPhD evolves AI agents to improve task performance without human intervention or author-supplied domain knowledge. It implements a closed-loop evolution cycle where an Evolution agent designs new versions of task agents based on performance feedback. The system uses ELO-based evolutionary selection to continuously improve agents across iterations.

Originally developed for Text2SQL ([paper](https://arxiv.org/abs/2601.01126)), the framework has since been extended to code generation, demonstrating that the evolutionary approach generalizes across domains.

## Key Results: Skip-a-Tier Deployment

Evolution produces the largest gains on cheaper models, enabling an **evolved cheaper model to exceed a naive expensive model** at lower cost.

**Text2SQL (BIRD Benchmark)**

| Eval Model | Naive | Evolved | Delta | Naive Cost | Evolved Cost |
|---|---|---|---|---|---|
| Opus-4.5 | 69.0% | 71.3% | +2.3 | 1.61¢ | 3.13¢ |
| Sonnet-4.5 | 65.7% | 69.2% | +3.5 | 0.56¢ | 0.87¢ |
| Haiku-4.5 | 57.2% | 66.1% | +8.9 | 0.34¢ | 0.51¢ |

Evolved Haiku (66.1%, 0.51¢/query) exceeds naive Sonnet (65.7%, 0.56¢/query), and evolved Sonnet (69.2%, 0.87¢/query) exceeds naive Opus (69.0%, 1.61¢/query) — better accuracy at lower cost in both cases.

**CodeGen (LiveCodeBench v6)**

| Configuration | Before Critic | After Critic | Delta | Cost/Problem |
|---|---|---|---|---|
| Haiku coder + Naive haiku critic | 53.8% | 53.8% | +0.0% | 6.5¢ |
| Haiku coder + Naive sonnet critic | 53.8% | 56.9% | +3.1% | 16.4¢ |
| **Haiku coder + Evolved haiku critic** | **53.8%** | **58.0%** | **+4.2%** | **10.5¢** |

The evolved haiku critic exceeds naive sonnet on both accuracy (+4.2% vs +3.1%) and cost (10.5¢ vs 16.4¢), replicating the skip-a-tier pattern in a second domain.

## How It Works

RoboPhD uses AI throughout:

1. **Task Execution**: Claude agents execute domain tasks (SQL generation, code review)
2. **Evolution**: Claude Code agents evolve increasingly better task agents
3. **Infrastructure**: The authors used Claude Code to build the RoboPhD system

The system uses ELO-based evolutionary selection to continuously improve agents across iterations.

### Supported Domains

| Domain | Benchmark | What Evolves | Status |
|---|---|---|---|
| Text2SQL | BIRD | Database analysis scripts + SQL generation instructions | Published ([paper](https://arxiv.org/abs/2601.01126)) |
| CodeGen | LiveCodeBench v6 | Code review critic agents | Experimental |

## Quick Start

### Text2SQL Domain

```bash
# 1. Clone and install (requires conda)
git clone https://github.com/andborth/RoboPhD.git
cd RoboPhD
./setup.sh

# 2. Activate environment
conda activate robophd

# 3. Set your API key (Text2SQL only)
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"

# 4. Download BIRD dataset
./benchmark_resources/download_bird.sh

# 5. Pre-compute ground truth (prevents timeout warnings)
python RoboPhD/tools/precompute_ground_truth.py

# 6. Run a quick test
python RoboPhD/researcher.py \
  --num-iterations 2 \
  --config '{"examples_per_iteration": 2, "problems_per_context": 10}'
```

### CodeGen Domain

```bash
# Steps 1-2 same as above (clone, install)
# No API key or dataset download needed — uses Claude Code CLI

python RoboPhD/researcher.py \
  --num-iterations 2 \
  --config configs/codegen_small_test.json
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions and [QUICKSTART.md](QUICKSTART.md) for a 5-minute tutorial.

## Requirements

- Python 3.10+
- Anthropic API key (Text2SQL only)
- Claude Code CLI (required for evolution and for CodeGen inference) 
- ~50GB disk space for BIRD dataset (Text2SQL only)

## Included Agents

**Text2SQL** (`RoboPhD/agents/`)

| Agent | Description | Dev Accuracy |
|---|---|---|
| `naive` | Baseline agent | 57-69% |
| `opus_best` | Best Opus-4.5 evolved agent | 71.3% |
| `sonnet_best` | Best Sonnet-4.5 evolved agent | 69.2% |
| `haiku_best` | Best Haiku-4.5 evolved agent | 66.1% |

**CodeGen** (`RoboPhD/codegen_agents/`)

| Agent | Description |
|---|---|
| `naive_critic` | Baseline critic agent |
| `codegen_haiku_best` | Best evolved Haiku-4.5 critic (+4.2% on LiveCodeBench) |

## Evolution Strategies

Built-in strategies:
- `cross_pollination` - Combines patterns from multiple successful agents
- `refinement` - Iteratively improves a single agent

Strategies are loaded from `RoboPhD/evolution_strategies/` (all domains) or `RoboPhD/evolution_strategies_text2sql/` (Text2SQL legacy).

## Configuration

Use production configs for best results:

```bash
# Text2SQL: Primary production config (Opus evolution, Haiku eval)
python RoboPhD/researcher.py --num-iterations 20 \
  --config configs/primary_production.json

# Text2SQL: Experimental config with research-driven evolution and meta-evolution
python RoboPhD/researcher.py --num-iterations 30 \
  --config configs/experimental_using_research_driven_and_meta_evolution.json

# CodeGen: Evolution with meta-evolution
python RoboPhD/researcher.py --num-iterations 15 \
  --config configs/codegen_with_simple_meta_evolution.json
```

## Documentation

- [Paper](https://arxiv.org/abs/2601.01126) - RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution
- [CLAUDE.md](CLAUDE.md) - Comprehensive system documentation
- [INSTALLATION.md](INSTALLATION.md) - Detailed installation guide
- [QUICKSTART.md](QUICKSTART.md) - 5-minute getting started guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [Text2SQL Domain Guide](docs/claude/text2sql.md)
- [CodeGen Domain Guide](docs/claude/codegen.md)

## Citation

If you use RoboPhD in your research, please cite:

```bibtex
@article{borthwick2026robophd,
  title={RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution},
  author={Borthwick, Andrew and Ash, Steve},
  journal={arXiv preprint arXiv:2601.01126},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BIRD Benchmark](https://bird-bench.github.io/) for the Text-to-SQL dataset
- [LiveCodeBench](https://livecodebench.github.io/) for the code generation benchmark
- [Anthropic](https://www.anthropic.com/) for Claude API and Claude Code
