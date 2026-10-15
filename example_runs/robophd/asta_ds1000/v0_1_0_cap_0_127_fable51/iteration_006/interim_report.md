# Parallel Agent Research - Interim Report (Iteration 6)

## Experiment Configuration

**Run ID**: asta_ds1000_20261014_081142
**Date**: 2026-10-14 11:03:42
**Iterations**: 6/999

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

#### **Use Your Judgment** (7 wins - 87.5% of all wins)
- **iter2_exec_verified_sonnet** (4 wins, created iteration 2)
- **iter3_ensemble_exec_vote** (2 wins, created iteration 3)
- **iter6_ensemble_exec_vote_escalate** (1 win, created iteration 6)

#### **Initial** (1 wins - 12.5% of all wins)
- **baseline** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 6
- **Total wins counted**: 8 (includes ties)
- **Unique winning agents**: 4
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter2_exec_verified_sonnet | - | **#1** 75.000 | **#1** 90.000 | **#1** 90.000 | **#1** 100.000 | #2 85.000 | **1534** | **1.20** |
| iter6_ensemble_exec_vote_escalate | - | - | - | - | - | **#1** 90.000 | 1533 | **1.00** |
| iter3_ensemble_exec_vote | - | - | #2 85.000 | **#1** 90.000 | **#1** 100.000 | #3 80.000 | 1474 | **1.75** |
| baseline | **#1** 60.000 | #2 65.000 | #3 80.000 | - | - | - | 1455 | 2.50 |
| iter4_ensemble_exec_vote_hardened | - | - | - | **#1** 90.000 | - | - | 1302* | **1.00** |
| iter5_ensemble_exec_vote_dtype_j... | - | - | - | - | **#1** 100.000 | - | 1302* | **1.00** |

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
| iter2_exec_verified_sonnet | 1534 | 88.000 | $0.013 | 5 |
| iter6_ensemble_exec_vote_escalate | 1533 | 90.000 | $0.065 | 1 |
| iter3_ensemble_exec_vote | 1474 | 88.750 | $0.042 | 4 |
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
| 6 | iter6_ensemble_exec_vote_escalate | 90.000 | 13.0m | 19.1m | 32.0m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter2_exec_verified_sonnet
- Elo Score: 1534
- Mean Score: 88.000


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | baseline | 1500 | Initial |
| 2 | iter2_exec_verified_sonnet | 1516 | use_your_judgment |
| 3 | iter2_exec_verified_sonnet | 1545 | use_your_judgment |
| 4 | iter2_exec_verified_sonnet | 1541 | use_your_judgment |
| 5 | iter2_exec_verified_sonnet | 1537 | use_your_judgment |
| 6 | iter2_exec_verified_sonnet | 1534 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 5/6
- **Total evolution time**: 59.0 minutes
- **Average evolution time**: 11.8 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 11.0m | 93.5% |
| Reflection | 0.8m | 6.5% |
| **Total** | **11.8m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 59.0m | 34.3% | 9.8m | 11.8m |
| Total Test Time | 113.0m | 65.7% | 18.8m | 18.8m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **172.0m** | **100.0%** | **28.7m** | **-** |

- **Evolution/Test ratio**: 0.52x
  - For every minute of testing, 0.52 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $52.83
  - **Evaluation Cost**: $6.72
  - **Evolution CLI Cost**: $46.11
- **Total CLI Calls**: 10
- **Total CLI Input Tokens**: 3,706
- **Total CLI Output Tokens**: 212,262
- **Total Cache Created**: 528,554
- **Total Cache Read**: 9,934,315

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $6.72 | 12.7% | - | - | - |
| Evolution (CLI) | $46.11 | 87.3% | 10 | 3,706 | 212,262 |
| **Total** | **$52.83** | **100%** | **10** | **3,706** | **212,262** |

**Answer: Evolution drives 87.3% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $9.05 | **$0.27** | **$8.77** | $0.27 | $4.27 | $4.50 | - | 36 | use_your_judgment | - |
| **3** | $9.35 | **$1.07** | **$8.28** | $1.07 | $4.01 | $4.27 | - | 49 | use_your_judgment | - |
| **4** | $12.60 | **$1.48** | **$11.11** | $1.48 | $5.41 | $5.70 | - | 43 | use_your_judgment | - |
| **5** | $11.26 | **$1.88** | **$9.38** | $1.88 | $4.55 | $4.83 | - | 50 | use_your_judgment | - |
| **6** | $10.56 | **$1.99** | **$8.57** | $1.99 | $4.16 | $4.41 | - | 42 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$52.83** | **$6.72** | **$46.11** | $6.72 | $22.40 | $23.70 | $0.00 | **240** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $4.48 | 48.6% | 707 | 39890 |
| Reflection | $4.74 | 51.4% | 34 | 2563 |
| **Total** | **$9.22** | **100%** | **741** | **42452** |

### Cache Performance

- **Cache Creation**: 528,554 tokens
- **Cache Hits**: 9,934,315 tokens (94.9% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅