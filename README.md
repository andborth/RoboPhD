# RoboPhD: Evolving AI Agents Without Human Domain Knowledge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01126-b31b1b.svg)](https://arxiv.org/abs/2601.01126)

RoboPhD evolves AI agents to improve task performance without human intervention or author-supplied domain knowledge. It implements a closed-loop evolution cycle where an Evolution agent designs new versions of task agents based on performance feedback, using ELO-based competition to select the best agents across iterations.

The system supports multiple optimization engines:
- **RoboPhD**: Multi-agent ELO competition with evolution strategies and deep focus testing
- **[GEPA](https://github.com/gepa-ai/gepa)**: Reflective text evolution with Pareto selection (Agrawal et al.; [paper](https://arxiv.org/abs/2507.19457))
- **[Autoresearch](https://github.com/karpathy/autoresearch)**: Single Claude Code session with greedy experimentation (Karpathy, 2025)

## Key Results

Tested across four benchmarks with diverse task types — abstract reasoning, cloud scheduling, SQL generation, and financial document QA:

| Benchmark | Train | Val | Test | Seed Baseline | RoboPhD Best | GEPA Best | Autoresearch Best | Published |
|-----------|-------|-----|------|--------------|-------------|-----------|-------------------|-----------|
| [ARC-AGI](https://arcprize.org/) | 200 | 200 | 400 | 26.5% | **65.1%** | 58.5% | 60.25% | — |
| [Can't Be Late](https://github.com/UCB-ADRS/ADRS/tree/main/openevolve/examples/ADRS/cant-be-late) | 2000 | — | 1080 | -96.48 | **-87.85** | -89.13 | -90.48 | — |
| [Text2SQL (BIRD)](https://bird-bench.github.io/) | 6601 | — | 1534 | 59.19% | **67.14%** | 61.15% | 65.38% | 66.1% |
| [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) | 5735 | 780 | 922 | 0.22% | **51.63%** | 37.85% | 45.88% | 42.6% |

*ARC-AGI and DocFinQA have designated val splits. For Can't Be Late and Text2SQL, GEPA and Autoresearch carve validation from the training set (typically 200 examples). RoboPhD combines all training data into a single pool and samples from it each iteration.*

RoboPhD holds #1 across all four benchmarks. It exceeds published results on Text2SQL (+1.0pp over prior best evolved Haiku) and DocFinQA (+3.3pp over GPT-3.5 + finetuned ColBERT).

## How It Works

RoboPhD uses AI throughout:

1. **Task Execution**: Solver agents execute domain tasks (SQL generation, puzzle solving, scheduling)
2. **Evolution**: Claude Code agents evolve increasingly better task agents
3. **Infrastructure**: The authors used Claude Code to build the RoboPhD system

```
    ┌─────────────────────────────────────────────────────────────┐
    │                      ITERATION CYCLE                        │
    │                                                             │
    │  ┌──────────────────┐         ┌────────────────────┐        │
    │  │  EVOLUTION AI    │ Creates │  AGENT ARTIFACTS    │        │
    │  │  (Claude Code    │────────▶│  (per file_mapping) │        │
    │  │   CLI session)   │         └────────┬───────────┘        │
    │  └──────────────────┘                  │                    │
    │           ▲                             ▼                    │
    │           │                    ┌────────────────────┐        │
    │   Performance                  │  EVALUATOR FN      │        │
    │   data from                    │  Black-box scoring │        │
    │   prior iterations             │  (candidate,example)│       │
    │           │                    │   → (score, diag)  │        │
    │           │                    └────────┬───────────┘        │
    │           │                             │                    │
    │           │                             ▼                    │
    │  ┌────────┴─────────┐         ┌────────────────────┐        │
    │  │  AGENT RANKINGS  │◀────────│  ELO COMPETITION   │        │
    │  │  Top agents      │         │  Head-to-head on   │        │
    │  │  inform next     │         │  sampled problems   │        │
    │  │  evolution round │         └────────────────────┘        │
    │  └──────────────────┘                                       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### Supported Domains

| Domain | Benchmark | What Evolves | Solver Model |
|---|---|---|---|
| ARC-AGI | ARC-AGI (HuggingFace) | `agent.py` — Python solver with `solve()` | Gemini Flash Lite (via OpenRouter) |
| Can't Be Late | AWS spot traces (NSDI'24) | `agent.py` — scheduling strategy class | Pure algorithmic (no LLM) |
| Text2SQL | BIRD | `eval_instructions.md` + `tools/analyze_db.py` + `verify_prompt.md` | Claude Haiku 4.5 |
| DocFinQA | DocFinQA (ACL 2024) | `agent.py` — retrieval + QA pipeline | GPT-4.1-mini + text-embedding-3-small |

Additional domains (CodeGen, AIME, CodeCritic) are available in the task registry but not actively maintained.

New domains are added via the task registry (`RoboPhD/tasks/`) — implement a `TaskDefinition` with an evaluator function, dataset builder, and file mapping.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/andborth/RoboPhD.git
cd RoboPhD
pip install -r requirements.txt

# 2. Install Claude Code CLI (required for evolution)
# See: https://docs.anthropic.com/en/docs/claude-code

# 3. Run a quick test (Can't Be Late — no API key needed)
bash scripts/download_cant_be_late_traces.sh
python scripts/run_robophd.py --task cant_be_late --num-iterations 3 \
  --engine-config '{"examples_per_iteration": 5}'

# 4. Text2SQL (requires Anthropic API key)
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"
python scripts/run_robophd.py --task text2sql --num-iterations 3 \
  --engine-config '{"examples_per_iteration": 5}'

# 5. List all valid parameters for a task
python scripts/run_robophd.py --task cant_be_late --list-params
```

### Running with GEPA

```bash
pip install -r requirements-gepa.txt

python scripts/run_gepa.py --task cant_be_late \
  --engine-config '{"evaluation_budget": 1500, "val_size": 200}' \
  --eval-test-set
```

### Test-Set Evaluation

```bash
# Auto-select best agent by ELO from a run
python scripts/eval_test_set.py --task cant_be_late \
  --run-dir ../robophd_runs/robophd/cant_be_late_20260313_230325

# Specify agent directly
python scripts/eval_test_set.py --task text2sql \
  --agent-dir RoboPhD/text2sql_agents/naive
```

## Evolution Strategies

Built-in strategies in `RoboPhD/evolution_strategies/`:
- `use_your_judgment` — Open-ended: study agents, data, and failure patterns (default)
- `data_focus` — Data-first: explore problem-level outputs before studying agent code
- `refinement` — Iteratively improve a single base agent
- `cross_pollination` — Combine patterns from multiple successful agents

Meta-evolution (`train_a_winner`) can generate additional strategies beyond these built-in options.

## Configuration

```bash
# Full run with deep focus testing and test-set evaluation
python scripts/run_robophd.py --task text2sql --num-iterations 20 --eval-test-set

# With meta-evolution
python scripts/run_robophd.py --task cant_be_late --num-iterations 22 --eval-test-set \
  --engine-config configs/robophd_engine/meta_evolution_starts_at_5.json

# Resume a run
python scripts/run_robophd.py --task cant_be_late \
  --resume ../robophd_runs/robophd/cant_be_late_20260313_230325 --extend 10
```

See [CLAUDE.md](CLAUDE.md) for comprehensive system documentation including all configuration parameters, deep focus evolution, evolution schedule control, and troubleshooting.

## Requirements

- Python 3.10+
- Claude Code CLI (required for evolution)
- For GEPA: `pip install -r requirements-gepa.txt`
- For ARC-AGI: `OPENROUTER_API_KEY` environment variable
- For Text2SQL: `ANTHROPIC_API_KEY_FOR_ROBOPHD` + ~50GB for BIRD dataset
- For DocFinQA: OpenAI API key (for gpt-4.1-mini and embeddings)
- For Can't Be Late: trace data via `bash scripts/download_cant_be_late_traces.sh`

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

- [ARC Prize](https://arcprize.org/) for the ARC-AGI benchmark
- [BIRD Benchmark](https://bird-bench.github.io/) for the Text2SQL dataset
- [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) (Reddy et al., ACL 2024)
- [Can't Be Late](https://github.com/UCB-ADRS/ADRS) (NSDI'24 AWS spot traces)
- [GEPA](https://github.com/gepa-ai/gepa) (Agrawal et al.)
- [Autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2025)
- [Anthropic](https://www.anthropic.com/) for Claude API and Claude Code
