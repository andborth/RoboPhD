# RoboPhD: Evolving AI Agents Without Human Domain Knowledge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01126-b31b1b.svg)](https://arxiv.org/abs/2601.01126)

RoboPhD evolves AI agents to improve task performance without human intervention or author-supplied domain knowledge. It implements a closed-loop evolution cycle where an Evolution agent designs new versions of task agents based on performance feedback, using ELO-based competition to select the best agents across iterations.

The system supports multiple optimization engines:
- **RoboPhD**: Multi-agent ELO competition with evolution strategies and deep focus testing
- **[GEPA](https://github.com/gepa-ai/gepa)**: Reflective text evolution with Pareto selection (Agrawal et al.; [paper](https://arxiv.org/abs/2507.19457))
- **[Autoresearch](https://github.com/karpathy/autoresearch)**: Single Claude Code session with greedy experimentation (Karpathy, 2026)

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
| ARC-AGI | ARC-AGI (HuggingFace) | `agent.py` — Python solver with `solve()` | Gemini Flash Lite (via OpenRouter) |
| Can't Be Late | AWS spot traces (NSDI'24) | `agent.py` — scheduling strategy class | Pure algorithmic (no LLM) |
| Text2SQL | BIRD | `agent.py` + `analyze_db.py` — SQL generation with `llm()` + `test_sql()` | Claude Haiku 4.5 |
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

### Optimize Anything Programmatic API

One simple way to use RoboPhD is to use it's `optimize_anything()` API to evolve any text artifact with your own evaluator. This is inspired by GEPA's optimize_anything api. Here is the sketch of how to use it:

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

**Resume & extend** — The result's `completed_normally` attribute tells you whether the run finished as expected or ended early due to a failure. Either way, `result.experiment_dir` points to the checkpoint directory, so you can always resume from where it left off:

```python
result = optimize_anything(
    evaluator=evaluator, dataset=my_dataset, objective="Maximize accuracy",
    seed_candidate=seed, config=RoboPhDConfig(num_iterations=10),
)

if not result.completed_normally:
    # Resume from where the failed run left off
    result = optimize_anything(
        evaluator=evaluator, dataset=my_dataset, objective="Maximize accuracy",
        config=RoboPhDConfig(experiment_dir=result.experiment_dir),
    )

# Extend by 5 more iterations
result = optimize_anything(
    evaluator=evaluator, dataset=my_dataset, objective="Maximize accuracy",
    config=RoboPhDConfig(experiment_dir=result.experiment_dir, extend_iterations=5),
)

# Restart from iteration 3 (discards iterations 3+ and re-runs)
result = optimize_anything(
    evaluator=evaluator, dataset=my_dataset, objective="Maximize accuracy",
    config=RoboPhDConfig(experiment_dir=result.experiment_dir, from_iteration=3),
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

Try the included demo, which evolves a math-problem-solving prompt from a naive seed ("Solve the following math problem") into an optimized prompt with chain-of-thought reasoning:

```bash
# 1. Set your API key (used for both evolution and the Haiku solver)
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"

# 2. Run the demo (3 iterations, ~$1-2 in API costs)
python scripts/run_optimize_anything.py --num-iterations 3

# 3. Quick test with minimal budget
python scripts/run_optimize_anything.py --num-iterations 2 \
    --evaluation-budget 50 --examples-per-iteration 5

# 4. Demo with resume + extend (runs 2 iterations, extends by 1, then restarts from iteration 2)
python scripts/run_optimize_anything.py --demo-resume
```

See [`RoboPhD/api.py`](RoboPhD/api.py) for the full API reference, including `optimize_task()` for running registered benchmarks programmatically.

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
  title={RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets},
  author={Borthwick, Andrew and Ash, Stephen and Galczak, Anthony},
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
- [Autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026)
- [Anthropic](https://www.anthropic.com/) for Claude API and Claude Code
