# Parallel Agent Research - Interim Report (Iteration 10)

## Experiment Configuration

**Run ID**: asta_ds1000_20260618_155040
**Date**: 2026-06-18 19:21:51
**Iterations**: 10/999

### System Defaults (Iteration 0)

Default values when no configuration is specified:

- agents_directory: None
- agents_per_iteration: 3
- dataset: train-filtered
- debug_log_probability: 0.02
- dollar_budget: None
- domain: external
- eval_result_cache: True
- eval_timeout: 300
- evaluation_budget: None
- evolution_model: opus-4.8 (claude-opus-4-8)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- include_evolution_rankings: False
- initial_agents: ['naive']
- llm_call_timeout: 120
- max_workers: None
- meta_evolution_cadence: 3
- meta_evolution_domain: None
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_round_offset: -2
- new_agent_test_rounds: 1
- oldest_agent_wins_ties: False
- random_agent_wins_ties: True
- runs_directory: None
- strategies_directory: None
- use_weighted_random: False

### Task Configuration

Task-specific parameters (from task defaults + --task-config overrides):

- ds1000_runtime: {'cost_threshold': 0.05, 'cost_per_error': 0.01}
- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1800
- evaluation_budget: 750
- evolution_model: opus-4.8 (claude-opus-4-8)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_pl47_s10']
- max_workers: 10
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (13 wins - 92.9% of all wins)
- **iter3_fmt_strong_cascade** (5 wins, created iteration 3)
- **iter2_selfcheck_cascade** (3 wins, created iteration 2)
- **iter7_agree_escalate** (2 wins, created iteration 7)
- **iter5_diverse_vote_mpl** (1 win, created iteration 5)
- **iter9_reason_agree** (1 win, created iteration 9)
- **iter10_literal_consensus** (1 win, created iteration 10)

#### **Initial** (1 wins - 7.1% of all wins)
- **seed_pl47_s10** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 10
- **Total wins counted**: 14 (includes ties)
- **Unique winning agents**: 7
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter10_literal_consensus | - | - | - | - | - | - | - | - | - | **#1** 100.000 | **1534** | **1.00** |
| iter7_agree_escalate | - | - | - | - | - | - | #2 85.000 | **#1** 95.000 | **#1** 80.000 | #2 95.000 | 1520 | **1.50** |
| iter9_reason_agree | - | - | - | - | - | - | - | - | **#1** 80.000 | - | 1503 | **1.00** |
| iter3_fmt_strong_cascade | - | - | **#1** 75.000 | **#1** 90.000 | #2 85.000 | **#1** 80.000 | **#1** 90.000 | #3 90.000 | **#1** 80.000 | #3 90.000 | 1500 | **1.62** |
| iter5_diverse_vote_mpl | - | - | - | - | **#1** 95.000 | #3 70.000 | - | - | - | - | 1500 | 2.00 |
| iter2_selfcheck_cascade | - | **#1** 75.000 | **#1** 75.000 | **#1** 90.000 | #3 60.000 | - | - | - | - | - | 1495 | **1.50** |
| iter6_wide_vote | - | - | - | - | - | #2 75.000 | #3 80.000 | - | - | - | 1473 | 2.50 |
| seed_pl47_s10 | **#1** 40.000 | #2 60.000 | #3 70.000 | - | - | - | - | - | - | - | 1455 | 2.50 |
| iter8_reason_cascade | - | - | - | - | - | - | - | **#1** 95.000 | - | - | 1319* | **1.00** |
| iter4_exec_vote_ensemble | - | - | - | **#1** 90.000 | - | - | - | - | - | - | 1302* | **1.00** |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration
- \* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200.*



### Quick Summary

| Agent | Elo | Mean Score | Tests |
|-------|-----|------------|-------|
| iter10_literal_consensus | 1534 | 100.000 | 1 |
| iter7_agree_escalate | 1520 | 88.750 | 4 |
| iter9_reason_agree | 1503 | 80.000 | 1 |
| iter3_fmt_strong_cascade | 1500 | 85.000 | 8 |
| iter5_diverse_vote_mpl | 1500 | 82.500 | 2 |
| iter2_selfcheck_cascade | 1495 | 75.000 | 4 |
| iter6_wide_vote | 1473 | 77.500 | 2 |
| seed_pl47_s10 | 1455 | 56.667 | 3 |
| iter8_reason_cascade | 1319 | 95.000 | 1 |
| iter4_exec_vote_ensemble | 1302 | 90.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_pl47_s10 | 40.000 | - | 24.6m | 24.6m |
| 2 | iter2_selfcheck_cascade | 75.000 | 6.1m | 7.1m | 13.3m |
| 3 | iter2_selfcheck_cascade, iter3_fmt_strong_cascade | 75.000 | 6.3m | 11.5m | 17.9m |
| 4 | iter2_selfcheck_cascade, iter3_fmt_strong_cascade (iter4_exec_vote_ensemble* excluded) | 90.000 | 6.9m | 14.7m | 21.6m |
| 5 | iter5_diverse_vote_mpl | 95.000 | 5.9m | 12.2m | 18.1m |
| 6 | iter3_fmt_strong_cascade | 80.000 | 3.4m | 15.9m | 19.3m |
| 7 | iter3_fmt_strong_cascade | 90.000 | 7.2m | 34.0m | 41.2m |
| 8 | iter7_agree_escalate (iter8_reason_cascade* excluded) | 95.000 | 5.2m | 18.4m | 23.5m |
| 9 | iter7_agree_escalate +2 | 80.000 | 3.6m | 12.8m | 16.4m |
| 10 | iter10_literal_consensus | 100.000 | 4.8m | 10.4m | 15.1m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter10_literal_consensus
- Elo Score: 1534
- Mean Score: 100.000


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_pl47_s10 | 1500 | Initial |
| 2 | iter2_selfcheck_cascade | 1516 | use_your_judgment |
| 3 | iter2_selfcheck_cascade | 1530 | use_your_judgment |
| 4 | iter2_selfcheck_cascade | 1528 | use_your_judgment |
| 5 | iter5_diverse_vote_mpl | 1533 | use_your_judgment |
| 6 | iter3_fmt_strong_cascade | 1547 | use_your_judgment |
| 7 | iter3_fmt_strong_cascade | 1574 | use_your_judgment |
| 8 | iter3_fmt_strong_cascade | 1536 | use_your_judgment |
| 9 | iter3_fmt_strong_cascade | 1534 | use_your_judgment |
| 10 | iter10_literal_consensus | 1534 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 9/10
- **Total evolution time**: 49.5 minutes
- **Average evolution time**: 5.5 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 4.9m | 89.6% |
| Reflection | 0.6m | 10.4% |
| **Total** | **5.5m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 49.5m | 23.4% | 4.9m | 5.5m |
| Total Test Time | 161.7m | 76.6% | 16.2m | 16.2m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **211.2m** | **100.0%** | **21.1m** | **-** |

- **Evolution/Test ratio**: 0.31x
  - For every minute of testing, 0.31 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $18.15
  - **Evaluation Cost**: $3.01
  - **Evolution CLI Cost**: $15.14
- **Total CLI Calls**: 18
- **Total CLI Input Tokens**: 44,416
- **Total CLI Output Tokens**: 219,737
- **Total Cache Created**: 550,024
- **Total Cache Read**: 7,807,584

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $3.01 | 16.6% | - | - | - |
| Evolution (CLI) | $15.14 | 83.4% | 18 | 44,416 | 219,737 |
| **Total** | **$18.15** | **100%** | **18** | **44,416** | **219,737** |

**Answer: Evolution drives 83.4% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.72 | **$0.05** | **$1.67** | $0.05 | $1.52 | $0.15 | - | 37 | use_your_judgment | - |
| **3** | $2.42 | **$0.14** | **$2.28** | $0.14 | $1.62 | $0.66 | - | 54 | use_your_judgment | - |
| **4** | $2.06 | **$0.27** | **$1.79** | $0.27 | $1.61 | $0.18 | - | 55 | use_your_judgment | - |
| **5** | $2.12 | **$0.26** | **$1.86** | $0.26 | $1.69 | $0.17 | - | 44 | use_your_judgment | - |
| **6** | $1.83 | **$0.55** | **$1.28** | $0.55 | $1.14 | $0.14 | - | 49 | use_your_judgment | - |
| **7** | $2.33 | **$0.47** | **$1.85** | $0.47 | $1.68 | $0.18 | - | 41 | use_your_judgment | - |
| **8** | $1.66 | **$0.30** | **$1.35** | $0.30 | $1.20 | $0.16 | - | 41 | use_your_judgment | - |
| **9** | $1.91 | **$0.59** | **$1.33** | $0.59 | $1.18 | $0.14 | - | 37 | use_your_judgment | - |
| **10** | $2.10 | **$0.38** | **$1.72** | $0.38 | $1.56 | $0.16 | - | 37 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$18.15** | **$3.01** | **$15.14** | $3.01 | $13.21 | $1.93 | $0.00 | **415** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.47 | 87.3% | 4503 | 22425 |
| Reflection | $0.21 | 12.7% | 433 | 1990 |
| **Total** | **$1.68** | **100%** | **4935** | **24415** |

### Cache Performance

- **Cache Creation**: 550,024 tokens
- **Cache Hits**: 7,807,584 tokens (93.4% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅