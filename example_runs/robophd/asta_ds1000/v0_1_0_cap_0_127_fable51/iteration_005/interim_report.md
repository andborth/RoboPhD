# Parallel Agent Research - Interim Report (Iteration 5)

## Experiment Configuration

**Run ID**: asta_ds1000_20261014_081142
**Date**: 2026-10-14 10:31:40
**Iterations**: 5/999

### System Defaults (Iteration 0)

Default values when no configuration is specified:

- agents_directory: None
- agents_per_iteration: 3
- dataset: train-filtered
- debug_log_probability: 0.02
- dollar_budget: None
- domain: external
- elo_reachability_guard: True
- elo_reachability_min_history: 5
- eval_result_cache: True
- eval_timeout: 300
- evaluation_budget: None
- evolution_model: opus-5.5 (claude-opus-5-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- include_evolution_rankings: False
- include_prior_evolution: True
- initial_agents: ['naive']
- llm_call_timeout: 120
- max_workers: None
- meta_evolution_cadence: 3
- meta_evolution_domain: None
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5.5 (claude-opus-5-5)
- new_agent_test_round_offset: -2
- new_agent_test_rounds: 1
- oldest_agent_wins_ties: False
- random_agent_wins_ties: True
- runs_directory: None
- strategies_directory: None
- use_weighted_random: False

### Task Configuration

Task-specific parameters (from task defaults + --task-config overrides):

- ds1000_runtime: {'cost_threshold': 0.127, 'cost_per_error': 0.0127}
- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds/seeds_0hf_cch3
- domain: external
- eval_timeout: 1800
- evaluation_budget: 750
- evolution_model: fable-5.1 (claude-fable-5-1)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['baseline']
- max_workers: 10
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5.5 (claude-opus-5-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (6 wins - 85.7% of all wins)
- **iter2_exec_verified_sonnet** (4 wins, created iteration 2)
- **iter3_ensemble_exec_vote** (2 wins, created iteration 3)

#### **Initial** (1 wins - 14.3% of all wins)
- **baseline** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 5
- **Total wins counted**: 7 (includes ties)
- **Unique winning agents**: 3
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|-----------|----------|
| iter2_exec_verified_sonnet | - | **#1** 75.000 | **#1** 90.000 | **#1** 90.000 | **#1** 100.000 | **1537** | **1.00** |
| iter3_ensemble_exec_vote | - | - | #2 85.000 | **#1** 90.000 | **#1** 100.000 | 1504 | **1.33** |
| baseline | **#1** 60.000 | #2 65.000 | #3 80.000 | - | - | 1455 | 2.50 |
| iter4_ensemble_exec_vote_hardened | - | - | - | **#1** 90.000 | - | 1302* | **1.00** |
| iter5_ensemble_exec_vote_dtype_j... | - | - | - | - | **#1** 100.000 | 1302* | **1.00** |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration
- \* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200.*



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter2_exec_verified_sonnet | 1537 | 88.750 | $0.014 | 4 |
| iter3_ensemble_exec_vote | 1504 | 91.667 | $0.043 | 3 |
| baseline | 1455 | 68.333 | $0.001 | 3 |
| iter4_ensemble_exec_vote_hardened | 1302 | 90.000 | $0.042 | 1 |
| iter5_ensemble_exec_vote_dtype_judge | 1302 | 100.000 | $0.048 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | baseline | 60.000 | - | 1.5m | 1.5m |
| 2 | iter2_exec_verified_sonnet | 75.000 | 10.1m | 3.9m | 13.9m |
| 3 | iter2_exec_verified_sonnet | 90.000 | 9.3m | 48.0m | 57.3m |
| 4 | iter2_exec_verified_sonnet, iter3_ensemble_exec_vote (iter4_ensemble_exec_vote_hardened* excluded) | 90.000 | 16.8m | 22.2m | 39.0m |
| 5 | iter2_exec_verified_sonnet, iter3_ensemble_exec_vote (iter5_ensemble_exec_vote_dtype_judge* excluded) | 100.000 | 9.8m | 18.4m | 28.2m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter2_exec_verified_sonnet
- Elo Score: 1537
- Mean Score: 88.750


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | baseline | 1500 | Initial |
| 2 | iter2_exec_verified_sonnet | 1516 | use_your_judgment |
| 3 | iter2_exec_verified_sonnet | 1545 | use_your_judgment |
| 4 | iter2_exec_verified_sonnet | 1541 | use_your_judgment |
| 5 | iter2_exec_verified_sonnet | 1537 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 4/5
- **Total evolution time**: 46.0 minutes
- **Average evolution time**: 11.5 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 10.8m | 93.6% |
| Reflection | 0.7m | 6.4% |
| **Total** | **11.5m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 46.0m | 32.9% | 9.2m | 11.5m |
| Total Test Time | 93.9m | 67.1% | 18.8m | 18.8m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **140.0m** | **100.0%** | **28.0m** | **-** |

- **Evolution/Test ratio**: 0.49x
  - For every minute of testing, 0.49 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $42.26
  - **Evaluation Cost**: $4.73
  - **Evolution CLI Cost**: $37.54
- **Total CLI Calls**: 8
- **Total CLI Input Tokens**: 2,898
- **Total CLI Output Tokens**: 178,504
- **Total Cache Created**: 422,254
- **Total Cache Read**: 7,581,294

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $4.73 | 11.2% | - | - | - |
| Evolution (CLI) | $37.54 | 88.8% | 8 | 2,898 | 178,504 |
| **Total** | **$42.26** | **100%** | **8** | **2,898** | **178,504** |

**Answer: Evolution drives 88.8% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $9.05 | **$0.27** | **$8.77** | $0.27 | $4.27 | $4.50 | - | 36 | use_your_judgment | - |
| **3** | $9.35 | **$1.07** | **$8.28** | $1.07 | $4.01 | $4.27 | - | 49 | use_your_judgment | - |
| **4** | $12.60 | **$1.48** | **$11.11** | $1.48 | $5.41 | $5.70 | - | 43 | use_your_judgment | - |
| **5** | $11.26 | **$1.88** | **$9.38** | $1.88 | $4.55 | $4.83 | - | 50 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$42.26** | **$4.73** | **$37.54** | $4.73 | $18.24 | $19.29 | $0.00 | **198** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $4.56 | 48.6% | 690 | 42024 |
| Reflection | $4.82 | 51.4% | 34 | 2602 |
| **Total** | **$9.38** | **100%** | **724** | **44626** |

### Cache Performance

- **Cache Creation**: 422,254 tokens
- **Cache Hits**: 7,581,294 tokens (94.7% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅