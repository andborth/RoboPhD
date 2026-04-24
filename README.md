# RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2604.04347-b31b1b.svg)](https://arxiv.org/abs/2604.04347)

RoboPhD evolves AI agents to improve task performance without human intervention or author-supplied domain knowledge. It implements a closed-loop evolution cycle where an Evolution agent designs new versions of task agents based on performance feedback, using ELO-based competition to select the best agents across iterations.

## Key Results

Tested across four benchmarks with diverse task types — abstract reasoning, cloud scheduling, SQL generation, and financial document QA. All runs use a fixed budget of 1,500 evaluations. Scores show test set performance; numbers in parentheses are agent lines of code.

| Benchmark | Seed | RoboPhD | GEPA | Autoresearch |
|-----------|------|---------|------|--------------|
| [ARC-AGI](https://arcprize.org/) (%) | 27.8 (22) | **65.8** (1,013) | 58.5 (366) | 54.2 (304) |
| [Can't Be Late](https://github.com/UCB-ADRS/ADRS/tree/main/openevolve/examples/ADRS/cant-be-late) | -96.5 (31) | -90.7 (148) | -89.3 (142) | **-87.6** (87) |
| [Text2SQL (BIRD)](https://bird-bench.github.io/) (%) | 52.2 (96) | **64.5** (602) | 60.4 (498) | 60.7 (265) |
| [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) (%) | 17.7 (29) | **50.4** (825) | 40.0 (207) | 48.2 (198) |

*Can't Be Late scores are negative costs (higher = better).*

Using a single default configuration, RoboPhD outperforms both GEPA and Autoresearch on three of four benchmarks, losing only on Can't Be Late — the simplest task, where the winning solution required just 87 lines of code. On the three complex benchmarks, RoboPhD's multi-iteration Elo competition produces substantially larger agents (602–1,013 lines) that combine strategies discovered across many evolutionary cycles.

## How It Works

RoboPhD uses AI throughout:

1. **Task Execution**: Solver agents execute domain tasks (SQL generation, puzzle solving, scheduling)
2. **Evolution**: Claude Code agents evolve increasingly better task agents
3. **Infrastructure**: The authors used Claude Code to build the RoboPhD system

```
    ┌─────────────────────────────────────────────────────────────┐
    │                      ITERATION CYCLE                        │
    │                                                             │
    │  ┌──────────────────┐         ┌───────────────────-─┐       │
    │  │  EVOLUTION AI    │ Creates │  AGENT ARTIFACTS    │       │
    │  │  (Claude Code    │────────▶│  (per file_mapping) │       │
    │  │   CLI session)   │         └────────┬──────────-─┘       │
    │  └──────────────────┘                  │                    │
    │           ▲                            ▼                    │
    │           │                    ┌───────────────────-─┐      │
    │   Performance                  │  EVALUATOR FN       │      │
    │   data from                    │  Black-box scoring  │      │
    │   prior iterations             │  (candidate,example)│      │
    │           │                    │   → (score, diag)   │      │
    │           │                    └────────┬───────────-┘      │
    │           │                             │                   │
    │           │                             ▼                   │
    │  ┌────────┴─────────┐         ┌────────────────────┐        │
    │  │  AGENT RANKINGS  │◀────────│  ELO COMPETITION   │        │
    │  │  Top agents      │         │  Head-to-head on   │        │
    │  │  inform next     │         │  sampled problems  │        │
    │  │  evolution round │         └────────────────────┘        │
    │  └──────────────────┘                                       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### Supported Domains

| Domain | Benchmark | What Evolves | Solver Model |
|---|---|---|---|
| ARC-AGI | ARC-AGI (HuggingFace) | `agent.py` — Python solver with `solve()` | Gemini 3.1 Flash Lite (via OpenRouter) |
| Can't Be Late | AWS spot traces (NSDI'24) | `agent.py` — scheduling strategy class | Pure algorithmic (no LLM) |
| Text2SQL | BIRD | `agent.py` + `analyze_db.py` — SQL generation with `llm()` + `test_sql()` | Claude Haiku 4.5 |
| DocFinQA | DocFinQA (ACL 2024) | `agent.py` — retrieval + QA pipeline | GPT-4.1-mini + text-embedding-3-small |

Each domain has a self-contained example under [`examples/`](examples/) with evaluator, seed agent, and documentation. More examples coming soon.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/andborth/RoboPhD.git
cd RoboPhD
pip install -r requirements.txt
pip install -r requirements-gepa.txt  # for GEPA engine support
pip install -r examples/arc_agi_1/requirements.txt  # for ARC-AGI (dspy, datasets)

# 2. Install Claude Code CLI (required for evolution)
# See: https://docs.anthropic.com/en/docs/claude-code

# 3. Set API keys
# Evolution itself uses Claude Code CLI (Claude Max auth) — no API key needed.
export ANTHROPIC_API_KEY_FOR_ROBOPHD="sk-ant-..."  # GEPA engine + Text2SQL example
export OPENAI_API_KEY="sk-..."                     # DocFinQA (gpt-4.1-mini + embeddings)
export OPENROUTER_API_KEY="sk-or-..."              # ARC-AGI (Gemini via OpenRouter)
# The Anthropic key is not needed for the default RoboPhD ELO engine or autoresearch.
# Recommended: link your Google API key at https://openrouter.ai/settings/integrations
# to get your own Gemini rate limits (otherwise you share limits with all OpenRouter users).

# 4. DocFinQA — financial document QA (easiest to start with)
python examples/docfinqa/main.py --num-iterations 2

# 5. ARC-AGI-1 — abstract reasoning (requires OpenRouter key above)
python examples/arc_agi_1/main.py --num-iterations 2

# 6. Can't Be Late — cloud scheduling (no solver API key needed)
bash examples/cant_be_late/download_traces.sh
python examples/cant_be_late/main.py --num-iterations 2

# 7. Text2SQL — SQL generation from natural language (BIRD benchmark)
bash benchmark_resources/download_bird.sh
python examples/text2sql/main.py --num-iterations 2
```

## Optimize Anything API

Use `optimize_anything()` to evolve any text artifact with your own evaluator:

```python
from RoboPhD import optimize_anything, RoboPhDConfig

def evaluator(candidate, example, *, problem_dir=None):
    prompt = candidate["system_prompt"]
    # Call your LLM, run your code, score the result...
    score = 1.0 if correct else 0.0
    return score, {
        "score": score,
        "predicted_answer": predicted,
        # String values are written as files for the evolution AI to read
        "question.md": example["question"],
        "response.md": response_text,
    }

result = optimize_anything(
    evaluator=evaluator,
    dataset=[{"id": "1", "question": "...", "answer": "..."}],
    seed_candidate={"system_prompt": "Your initial prompt here"},
    objective="Maximize accuracy on my task",
    config=RoboPhDConfig(num_iterations=5, evaluation_budget=200),
)
print(result.best_candidate["system_prompt"])
print(f"Best ELO: {result.best_score}")
```

**Resume & extend** — `result.experiment_dir` points to the checkpoint directory, so you can always resume:

```python
# Resume from where it left off
result = optimize_anything(
    evaluator=evaluator, dataset=my_dataset, objective="Maximize accuracy",
    config=RoboPhDConfig(experiment_dir=result.experiment_dir),
)

# Extend by 5 more iterations
result = optimize_anything(
    evaluator=evaluator, dataset=my_dataset, objective="Maximize accuracy",
    config=RoboPhDConfig(experiment_dir=result.experiment_dir, extend_iterations=5),
)
```

Note: `seed_candidate` is only needed for the initial run — on resume, the file mapping is recovered from the checkpoint. `evaluator` and `dataset` are always required (they can't be serialized).

**Evaluating candidates** — use `eval_candidate()` to evaluate any candidate on a dataset:

```python
from RoboPhD import eval_candidate, RoboPhDEvalConfig

eval_result = eval_candidate(
    evaluator=evaluator,
    dataset=test_dataset,
    candidate=result.best_candidate,
    config=RoboPhDEvalConfig(test_repeats=3, max_workers=8),
)
print(f"Accuracy: {eval_result.mean_score:.1%} ({eval_result.num_examples} examples)")
```

See [`RoboPhD/api.py`](RoboPhD/api.py) for the full API reference.

## Evolution Strategies

Built-in strategies in `RoboPhD/evolution_strategies/`:
- `use_your_judgment` — Open-ended: study agents, data, and failure patterns (default)
- `data_focus` — Data-first: explore problem-level outputs before studying agent code
- `refinement` — Iteratively improve a single base agent
- `cross_pollination` — Combine patterns from multiple successful agents

## Configuration

```bash
# Full run with test-set evaluation
python examples/arc_agi_1/main.py --eval-test-set

# Use paper configuration (stronger model, higher cost budget)
python examples/arc_agi_1/main.py --paper-config

# Custom engine config
python examples/arc_agi_1/main.py --engine-config '{"include_evolution_rankings": false}'

# Resume a run
python examples/arc_agi_1/main.py --resume ../robophd_runs/robophd/optimize_anything_20260401_120000

# Extend by 5 more iterations
python examples/arc_agi_1/main.py --resume <dir> --extend 5
```

**Multi-engine support**: All examples support `--engine {robophd,gepa,autoresearch}` to select the optimization engine.

## Requirements

- Python 3.10+
- Claude Code CLI (required for evolution — uses Claude Max auth)
- `pip install -r requirements-gepa.txt` (only if using `--engine gepa`)

Per-example requirements (solver API keys, dataset downloads, extra pip installs) are documented in each `examples/<domain>/README.md`.

## Acknowledgments

RoboPhD builds on several excellent open-source projects and benchmarks:
- [GEPA](https://github.com/gepa-ai/gepa) (Agrawal et al., 2025) — reflective text evolution with Pareto selection
- [Autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026) — single-session greedy experimentation
- [ARC Prize](https://arcprize.org/) / [ARC-AGI](https://arxiv.org/abs/1911.01547) (Chollet, 2019) — abstract reasoning benchmark
- [BIRD](https://bird-bench.github.io/) (Li et al., 2024) — Text-to-SQL benchmark
- [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) (Reddy et al., 2024) — long-context financial QA benchmark
- [Can't Be Late](https://github.com/UCB-ADRS/ADRS) (Wu et al., 2024) — cloud spot instance scheduling

## Citation

If you use RoboPhD in your research, please cite:

```bibtex
@article{borthwick2026robophd,
  title={RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets},
  author={Borthwick, Andrew and Ash, Stephen and Galczak, Anthony},
  journal={arXiv preprint arXiv:2604.04347},
  year={2026}
}
```
