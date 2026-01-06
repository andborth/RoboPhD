# RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01126-b31b1b.svg)](https://arxiv.org/abs/2601.01126)

RoboPhD evolves AI agents to improve Text-to-SQL performance without human intervention or author-supplied domain knowledge. It implements a closed-loop evolution cycle with two coordinated components: a SQL Generation agent composed of a database analysis script and SQL generation instructions, and an Evolution agent that designs new versions of the SQL agent based on performance feedback. RoboPhD achieves **73.67% accuracy** on the BIRD benchmark test set.

## Key Results

| Model | Dev Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Opus-4.5 | 71.3% | 73.67% |
| Sonnet-4.5 | 69.2% | - |
| Haiku-4.5 | 66.1% | - |

## How It Works

RoboPhD uses AI throughout:

1. **SQL Generation**: Claude agents generate SQL from natural language questions
2. **Evolution**: Claude Code agents evolve increasingly better database analysis scripts and SQL generation instructions
3. **Infrastructure**: The authors used Claude Code to build the RoboPhD system

The system uses ELO-based evolutionary selection to continuously improve agents across iterations.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/RoboPhD.git
cd RoboPhD

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"

# 4. Download BIRD dataset
./benchmark_resources/download_bird.sh

# 5. Pre-compute ground truth (prevents timeout warnings)
python RoboPhD/tools/precompute_ground_truth.py

# 6. Run a quick test
python RoboPhD/researcher.py \
  --num-iterations 2 \
  --config '{"databases_per_iteration": 2, "questions_per_database": 10}'
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions and [QUICKSTART.md](QUICKSTART.md) for a 5-minute tutorial.

## Requirements

- Python 3.10+
- Anthropic API key
- Claude Code CLI (required for evolution)
- ~50GB disk space for BIRD dataset

## Included Agents

| Agent | Description | Dev Accuracy |
|-------|-------------|--------------|
| `naive` | Baseline agent | 57-69% |
| `opus_best` | Best Opus-4.5 evolved agent | 71.3% |
| `sonnet_best` | Best Sonnet-4.5 evolved agent | 69.2% |
| `haiku_best` | Best Haiku-4.5 evolved agent | 66.1% |

## Evolution Strategies

**Tool-only variants** (deterministic, recommended):
- `cross_pollination_tool_only` - Combines patterns from multiple successful agents
- `refinement_tool_only` - Iteratively improves a single agent
- `research_driven_tool_only` - Incorporates insights from academic papers

**Neutral variants** (allow LLM in database analysis):
- `cross_pollination_neutral`
- `refinement_neutral`
- `research_driven_neutral`

## Configuration

Use production configs for best results:

```bash
# Primary production config (Opus evolution, Haiku eval)
python RoboPhD/researcher.py --num-iterations 20 \
  --config configs/primary_production.json

# Experimental config with research-driven evolution and meta-evolution
python RoboPhD/researcher.py --num-iterations 30 \
  --config configs/experimental_using_research_driven_and_meta_evolution.json
```

## Documentation

- [Paper](https://arxiv.org/abs/2601.01126) - RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution
- [CLAUDE.md](CLAUDE.md) - Comprehensive system documentation
- [INSTALLATION.md](INSTALLATION.md) - Detailed installation guide
- [QUICKSTART.md](QUICKSTART.md) - 5-minute getting started guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

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
- [Anthropic](https://www.anthropic.com/) for Claude API and Claude Code
