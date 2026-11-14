# Parallel Agent Research - Interim Report (Iteration 10)

## Experiment Configuration

**Run ID**: asta_ds1000_20260614_194218
**Date**: 2026-06-14 23:47:06
**Iterations**: 10/20

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

- ds1000_runtime: {'cost_threshold': 0.08, 'cost_per_error': 0.01}
- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1800
- evaluation_budget: 620
- evolution_model: opus-4.8 (claude-opus-4-8)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_dmmf_ba5']
- max_workers: 6
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: fable-5 (claude-fable-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (11 wins - 84.6% of all wins)
- **iter3_ds1000_ensemble_judge** (5 wins, created iteration 3)
- **iter8_ds1000_strongjudge** (3 wins, created iteration 8)
- **iter2_ds1000_verify_escalate** (2 wins, created iteration 2)
- **iter6_ds1000_verified_judge** (1 win, created iteration 6)

#### **Initial** (2 wins - 15.4% of all wins)
- **seed_dmmf_ba5** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 10
- **Total wins counted**: 13 (includes ties)
- **Unique winning agents**: 5
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter8_ds1000_strongjudge | - | - | - | - | - | - | - | **#1** 100.000 | **#1** 95.000 | **#1** 100.000 | **1545** | **1.00** |
| iter3_ds1000_ensemble_judge | - | - | **#1** 100.000 | **#1** 85.000 | **#1** 100.000 | #2 90.000 | **#1** 90.000 | #2 85.000 | #3 80.000 | **#1** 100.000 | 1510 | **1.50** |
| iter6_ds1000_verified_judge | - | - | - | - | - | **#1** 95.000 | #3 85.000 | #2 85.000 | - | - | 1489 | 2.00 |
| iter5_ds1000_universal_verify | - | - | - | - | #2 95.000 | #3 80.000 | - | - | - | - | 1473 | 2.50 |
| iter2_ds1000_verify_escalate | - | **#1** 75.000 | #2 80.000 | **#1** 85.000 | #3 90.000 | - | - | - | - | - | 1471 | **1.75** |
| seed_dmmf_ba5 | **#1** 85.000 | **#1** 75.000 | #3 70.000 | - | - | - | - | - | - | - | 1469 | 2.00 |
| iter7_ds1000_triverified_judge | - | - | - | - | - | - | **#1** 90.000 | - | - | - | 1320* | **1.00** |
| iter9_ds1000_execverify_judge | - | - | - | - | - | - | - | - | **#1** 95.000 | - | 1319* | **1.00** |
| iter10_ds1000_dtypeverify | - | - | - | - | - | - | - | - | - | **#1** 100.000 | 1303* | **1.00** |
| iter4_ds1000_triensemble_vote | - | - | - | **#1** 85.000 | - | - | - | - | - | - | 1301* | **1.00** |

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
| iter8_ds1000_strongjudge | 1545 | 98.333 | 3 |
| iter3_ds1000_ensemble_judge | 1510 | 91.250 | 8 |
| iter6_ds1000_verified_judge | 1489 | 88.333 | 3 |
| iter5_ds1000_universal_verify | 1473 | 87.500 | 2 |
| iter2_ds1000_verify_escalate | 1471 | 82.500 | 4 |
| seed_dmmf_ba5 | 1469 | 76.667 | 3 |
| iter7_ds1000_triverified_judge | 1320 | 90.000 | 1 |
| iter9_ds1000_execverify_judge | 1319 | 95.000 | 1 |
| iter10_ds1000_dtypeverify | 1303 | 100.000 | 1 |
| iter4_ds1000_triensemble_vote | 1301 | 85.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_dmmf_ba5 | 85.000 | - | 8.0m | 8.0m |
| 2 | seed_dmmf_ba5, iter2_ds1000_verify_escalate | 75.000 | 6.8m | 10.7m | 17.5m |
| 3 | iter3_ds1000_ensemble_judge | 100.000 | 4.8m | 19.8m | 24.6m |
| 4 | iter3_ds1000_ensemble_judge, iter2_ds1000_verify_escalate (iter4_ds1000_triensemble_vote* excluded) | 85.000 | 3.5m | 20.7m | 24.2m |
| 5 | iter3_ds1000_ensemble_judge | 100.000 | 7.1m | 18.5m | 25.7m |
| 6 | iter6_ds1000_verified_judge | 95.000 | 4.5m | 23.7m | 28.2m |
| 7 | iter3_ds1000_ensemble_judge (iter7_ds1000_triverified_judge* excluded) | 90.000 | 4.5m | 19.2m | 23.7m |
| 8 | iter8_ds1000_strongjudge | 100.000 | 5.2m | 24.9m | 30.2m |
| 9 | iter8_ds1000_strongjudge (iter9_ds1000_execverify_judge* excluded) | 95.000 | 5.4m | 25.5m | 30.9m |
| 10 | iter8_ds1000_strongjudge, iter3_ds1000_ensemble_judge (iter10_ds1000_dtypeverify* excluded) | 100.000 | 4.7m | 27.1m | 31.8m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter8_ds1000_strongjudge
- Elo Score: 1545
- Mean Score: 98.333


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_dmmf_ba5 | 1500 | Initial |
| 2 | seed_dmmf_ba5 | 1500 | Initial |
| 3 | iter3_ds1000_ensemble_judge | 1531 | use_your_judgment |
| 4 | iter3_ds1000_ensemble_judge | 1528 | use_your_judgment |
| 5 | iter3_ds1000_ensemble_judge | 1557 | use_your_judgment |
| 6 | iter3_ds1000_ensemble_judge | 1552 | use_your_judgment |
| 7 | iter3_ds1000_ensemble_judge | 1564 | use_your_judgment |
| 8 | iter3_ds1000_ensemble_judge | 1543 | use_your_judgment |
| 9 | iter8_ds1000_strongjudge | 1549 | use_your_judgment |
| 10 | iter8_ds1000_strongjudge | 1545 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 9/10
- **Total evolution time**: 46.6 minutes
- **Average evolution time**: 5.2 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 4.6m | 89.2% |
| Reflection | 0.6m | 10.8% |
| **Total** | **5.2m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 46.6m | 19.0% | 4.7m | 5.2m |
| Total Test Time | 198.2m | 81.0% | 19.8m | 19.8m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **244.8m** | **100.0%** | **24.5m** | **-** |

- **Evolution/Test ratio**: 0.24x
  - For every minute of testing, 0.24 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $18.34
  - **Evaluation Cost**: $5.07
  - **Evolution CLI Cost**: $13.27
- **Total CLI Calls**: 18
- **Total CLI Input Tokens**: 35,577
- **Total CLI Output Tokens**: 207,415
- **Total Cache Created**: 490,999
- **Total Cache Read**: 5,962,790

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $5.07 | 27.6% | - | - | - |
| Evolution (CLI) | $13.27 | 72.4% | 18 | 35,577 | 207,415 |
| **Total** | **$18.34** | **100%** | **18** | **35,577** | **207,415** |

**Answer: Evolution drives 72.4% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $2.05 | **$0.03** | **$2.02** | $0.03 | $1.42 | $0.60 | - | 35 | use_your_judgment | - |
| **3** | $1.56 | **$0.26** | **$1.29** | $0.26 | $1.15 | $0.14 | - | 49 | use_your_judgment | - |
| **4** | $1.58 | **$0.58** | **$1.00** | $0.58 | $0.87 | $0.13 | - | 46 | use_your_judgment | - |
| **5** | $2.02 | **$0.21** | **$1.81** | $0.21 | $1.66 | $0.15 | - | 45 | use_your_judgment | - |
| **6** | $1.87 | **$0.55** | **$1.32** | $0.55 | $1.18 | $0.14 | - | 46 | use_your_judgment | - |
| **7** | $2.11 | **$0.80** | **$1.30** | $0.80 | $1.17 | $0.14 | - | 46 | use_your_judgment | - |
| **8** | $2.52 | **$0.88** | **$1.64** | $0.88 | $1.47 | $0.17 | - | 42 | use_your_judgment | - |
| **9** | $2.16 | **$0.68** | **$1.48** | $0.68 | $1.33 | $0.15 | - | 39 | use_your_judgment | - |
| **10** | $2.48 | **$1.07** | **$1.41** | $1.07 | $1.26 | $0.16 | - | 44 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$18.34** | **$5.07** | **$13.27** | $5.07 | $11.50 | $1.77 | $0.00 | **412** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.28 | 86.6% | 3546 | 21092 |
| Reflection | $0.20 | 13.4% | 407 | 1954 |
| **Total** | **$1.47** | **100%** | **3953** | **23046** |

### Cache Performance

- **Cache Creation**: 490,999 tokens
- **Cache Hits**: 5,962,790 tokens (92.4% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅