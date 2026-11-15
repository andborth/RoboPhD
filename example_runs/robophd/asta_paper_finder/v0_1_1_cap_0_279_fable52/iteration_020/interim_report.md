# Parallel Agent Research - Interim Report (Iteration 20)

## Experiment Configuration

**Run ID**: asta_paper_finder_20261111_142949
**Date**: 2026-11-12 18:22:24
**Iterations**: 20/999

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

- file_mapping: {'agent.py': 'agent.py'}
- paper_finder_runtime: {'cost_threshold': 0.279, 'cost_per_error': 0.0279, 'cap_judge_to_estimate': True, 'training_judge': 'openai/gpt-5.6-luna', 'judge_prompt': 'no-prose', 'evidence_char_cap': 2500, 'model_menu': ['GPT_5_4_MINI', 'GPT_5_4', 'GPT_5_5', 'CLAUDE_HAIKU_4_5', 'CLAUDE_SONNET_4_6', 'CLAUDE_OPUS_4_8', 'GEMINI_3_1_FLASH_LITE', 'GEMINI_3_5_FLASH', 'GEMINI_3_1_PRO_PREVIEW']}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds/seeds_88t55w0h
- domain: external
- eval_timeout: 1800
- evaluation_budget: 600
- evolution_model: fable-5.2 (claude-fable-5-2)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 14
- initial_agents: ['baseline']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5.5 (claude-opus-5-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (19 wins - 95.0% of all wins)
- **iter6_wide_evidence** (5 wins, created iteration 6)
- **iter14_snowball_hydration** (3 wins, created iteration 14)
- **iter11_qualifier_criteria** (2 wins, created iteration 11)
- **iter18_stage1_keyed_order** (2 wins, created iteration 18)
- **iter2_plan_grade_evidence** (1 win, created iteration 2)
- **iter3_rejudge_topk** (1 win, created iteration 3)
- **iter4_round2_repair** (1 win, created iteration 4)
- **iter5_focused_evidence** (1 win, created iteration 5)
- **iter8_meta_gold_window** (1 win, created iteration 8)
- **iter13_citation_snowball** (1 win, created iteration 13)
- **iter20_year_steered_expansion** (1 win, created iteration 20)

#### **Initial** (1 wins - 5.0% of all wins)
- **baseline** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 20
- **Total wins counted**: 20 (includes ties)
- **Unique winning agents**: 12
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter11_qualifier_criteria | - | - | - | - | - | - | - | - | - | - | **#1** 62.121 | #2 53.309 | - | - | - | **#1** 52.551 | #2 38.803 | #2 55.165 | - | - | **1561** | **1.60** |
| iter20_year_steered_expansion | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 56.701 | 1536 | **1.00** |
| iter18_stage1_keyed_order | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 58.081 | **#1** 58.948 | #3 54.992 | 1530 | **1.67** |
| iter14_snowball_hydration | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 64.074 | **#1** 63.663 | #3 49.521 | **#1** 41.957 | #3 50.618 | - | #2 56.025 | 1528 | **1.83** |
| iter6_wide_evidence | - | - | - | - | - | **#1** 51.708 | **#1** 41.252 | **#1** 57.394 | **#1** 51.684 | #2 58.825 | #3 54.406 | **#1** 60.611 | #3 50.850 | - | #3 52.339 | - | - | - | - | - | 1520 | **1.78** |
| iter8_meta_gold_window | - | - | - | - | - | - | - | #2 49.921 | #2 47.743 | **#1** 60.227 | #2 55.460 | - | #2 51.633 | #3 51.923 | - | - | - | - | - | - | 1511 | 2.00 |
| iter16_alt_readings_head_check | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 50.361 | - | - | - | - | 1505 | 2.00 |
| iter15_adaptive_depth | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 54.153 | - | - | - | - | - | 1504 | 2.00 |
| iter19_backdated_citation_counts | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 58.591 | - | 1503 | 2.00 |
| iter13_citation_snowball | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 52.747 | #2 52.851 | - | - | - | - | #3 48.265 | - | 1503 | 2.00 |
| iter4_round2_repair | - | - | - | **#1** 44.890 | #3 35.753 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1500 | 2.00 |
| iter5_focused_evidence | - | - | - | - | **#1** 44.156 | #2 47.030 | #2 39.971 | #3 49.064 | - | - | - | - | - | - | - | - | - | - | - | - | 1499 | 2.00 |
| iter3_rejudge_topk | - | - | **#1** 42.791 | #3 41.295 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1498 | 2.00 |
| iter2_plan_grade_evidence | - | **#1** 39.463 | #2 41.415 | #2 42.346 | #2 36.503 | #3 42.264 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter10_wide_deep_recall | - | - | - | - | - | - | - | - | - | #3 44.275 | - | - | - | - | - | - | - | - | - | - | 1474 | 3.00 |
| iter12_retry_recovery | - | - | - | - | - | - | - | - | - | - | - | #3 46.204 | - | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter17_one_record_per_work | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 37.658 | - | - | - | 1473 | 3.00 |
| iter9_gold_style_criteria | - | - | - | - | - | - | - | - | #3 45.078 | - | - | - | - | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter7_aligned_rank | - | - | - | - | - | - | #3 37.691 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1472 | 3.00 |
| baseline | **#1** 4.914 | #2 4.603 | #3 3.924 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter11_qualifier_criteria | 1561 | 52.390 | $0.247 | 5 |
| iter20_year_steered_expansion | 1536 | 56.701 | $0.202 | 1 |
| iter18_stage1_keyed_order | 1530 | 57.340 | $0.227 | 3 |
| iter14_snowball_hydration | 1528 | 54.310 | $0.244 | 6 |
| iter6_wide_evidence | 1520 | 53.230 | $0.180 | 9 |
| iter8_meta_gold_window | 1511 | 52.818 | $0.175 | 6 |
| iter16_alt_readings_head_check | 1505 | 50.361 | $0.278 | 1 |
| iter15_adaptive_depth | 1504 | 54.153 | $0.274 | 1 |
| iter19_backdated_citation_counts | 1503 | 58.591 | $0.242 | 1 |
| iter13_citation_snowball | 1503 | 51.288 | $0.250 | 3 |
| iter4_round2_repair | 1500 | 40.322 | $0.184 | 2 |
| iter5_focused_evidence | 1499 | 45.055 | $0.187 | 4 |
| iter3_rejudge_topk | 1498 | 42.043 | $0.158 | 2 |
| iter2_plan_grade_evidence | 1483 | 40.398 | $0.091 | 5 |
| iter10_wide_deep_recall | 1474 | 44.275 | $0.228 | 1 |
| iter12_retry_recovery | 1473 | 46.204 | $0.244 | 1 |
| iter17_one_record_per_work | 1473 | 37.658 | $0.311 | 1 |
| iter9_gold_style_criteria | 1473 | 45.078 | $0.141 | 1 |
| iter7_aligned_rank | 1472 | 37.691 | $0.166 | 1 |
| baseline | 1455 | 4.480 | $0.001 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | baseline | 4.914 | - | 1.9m | 1.9m |
| 2 | iter2_plan_grade_evidence | 39.463 | 18.3m | 30.3m | 48.6m |
| 3 | iter3_rejudge_topk | 42.791 | 17.6m | 44.9m | 62.6m |
| 4 | iter4_round2_repair | 44.890 | 13.1m | 59.2m | 72.4m |
| 5 | iter5_focused_evidence | 44.156 | 44.5m | 64.8m | 109.3m |
| 6 | iter6_wide_evidence | 51.708 | 17.4m | 59.0m | 76.4m |
| 7 | iter6_wide_evidence | 41.252 | 15.9m | 68.5m | 84.4m |
| 8 | iter6_wide_evidence | 57.394 | 28.8m | 52.6m | 81.3m |
| 9 | iter6_wide_evidence | 51.684 | 6.5m | 55.5m | 62.0m |
| 10 | iter8_meta_gold_window | 60.227 | 26.6m | 55.4m | 82.0m |
| 11 | iter11_qualifier_criteria | 62.121 | 25.9m | 54.8m | 80.7m |
| 12 | iter6_wide_evidence | 60.611 | 17.1m | 76.0m | 93.0m |
| 13 | iter13_citation_snowball | 52.747 | 14.8m | 75.4m | 90.1m |
| 14 | iter14_snowball_hydration | 64.074 | 6.6m | 71.0m | 77.6m |
| 15 | iter14_snowball_hydration | 63.663 | 4.2m | 74.0m | 78.3m |
| 16 | iter11_qualifier_criteria | 52.551 | 6.7m | 63.6m | 70.3m |
| 17 | iter14_snowball_hydration | 41.957 | 3.2m | 66.3m | 69.5m |
| 18 | iter18_stage1_keyed_order | 58.081 | 3.5m | 75.4m | 78.9m |
| 19 | iter18_stage1_keyed_order | 58.948 | 5.8m | 81.2m | 87.0m |
| 20 | iter20_year_steered_expansion | 56.701 | 10.0m | 77.0m | 86.9m |

## Best Agent: iter11_qualifier_criteria
- Elo Score: 1561
- Mean Score: 52.390


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | baseline | 1500 | Initial |
| 2 | iter2_plan_grade_evidence | 1516 | use_your_judgment |
| 3 | iter3_rejudge_topk | 1531 | use_your_judgment |
| 4 | iter4_round2_repair | 1533 | use_your_judgment |
| 5 | iter5_focused_evidence | 1533 | use_your_judgment |
| 6 | iter6_wide_evidence | 1533 | use_your_judgment |
| 7 | iter6_wide_evidence | 1563 | use_your_judgment |
| 8 | iter6_wide_evidence | 1590 | use_your_judgment |
| 9 | iter6_wide_evidence | 1614 | use_your_judgment |
| 10 | iter6_wide_evidence | 1604 | use_your_judgment |
| 11 | iter6_wide_evidence | 1565 | use_your_judgment |
| 12 | iter6_wide_evidence | 1593 | use_your_judgment |
| 13 | iter6_wide_evidence | 1555 | use_your_judgment |
| 14 | iter6_wide_evidence | 1555 | use_your_judgment |
| 15 | iter14_snowball_hydration | 1566 | use_your_judgment |
| 16 | iter11_qualifier_criteria | 1568 | use_your_judgment |
| 17 | iter11_qualifier_criteria | 1564 | use_your_judgment |
| 18 | iter11_qualifier_criteria | 1561 | use_your_judgment |
| 19 | iter18_stage1_keyed_order | 1566 | use_your_judgment |
| 20 | iter11_qualifier_criteria | 1561 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 19/20
- **Total evolution time**: 286.4 minutes
- **Average evolution time**: 15.1 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 14.5m | 96.3% |
| Reflection | 0.6m | 3.7% |
| **Total** | **15.1m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 286.4m | 19.2% | 14.3m | 15.1m |
| Total Test Time | 1206.7m | 80.8% | 60.3m | 60.3m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **1493.0m** | **100.0%** | **74.7m** | **-** |

- **Evolution/Test ratio**: 0.24x
  - For every minute of testing, 0.24 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $230.46
  - **Evaluation Cost**: $105.51
  - **Evolution CLI Cost**: $116.55
  - **Other (evaluator overhead) Cost**: $8.39
- **Total CLI Calls**: 38
- **Total CLI Input Tokens**: 726
- **Total CLI Output Tokens**: 374,708
- **Total Cache Created**: 1,514,226
- **Total Cache Read**: 29,506,169

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $105.51 | 45.8% | - | - | - |
| Evolution (CLI) | $116.55 | 50.6% | 38 | 726 | 374,708 |
| Other (evaluator overhead) | $8.39 | 3.6% | - | - | - |
| **Total** | **$230.46** | **100%** | **38** | **726** | **374,708** |

**Answer: Evolution drives 50.6% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | $0.00 | 14 | - | - |
| **2** | $8.41 | **$1.28** | **$6.91** | $1.28 | $3.34 | $3.57 | - | $0.22 | 24 | use_your_judgment | - |
| **3** | $9.87 | **$3.30** | **$6.19** | $3.30 | $2.97 | $3.21 | - | $0.38 | 34 | use_your_judgment | - |
| **4** | $12.31 | **$4.86** | **$6.92** | $4.86 | $3.32 | $3.60 | - | $0.53 | 31 | use_your_judgment | - |
| **5** | $16.19 | **$5.34** | **$10.46** | $5.34 | $5.10 | $5.36 | - | $0.39 | 32 | use_your_judgment | - |
| **6** | $11.42 | **$5.62** | **$5.27** | $5.62 | $2.52 | $2.75 | - | $0.53 | 32 | use_your_judgment | - |
| **7** | $12.38 | **$5.41** | **$6.51** | $5.41 | $3.12 | $3.39 | - | $0.46 | 34 | use_your_judgment | - |
| **8** | $12.12 | **$4.83** | **$6.86** | $4.83 | $3.29 | $3.58 | - | $0.42 | 29 | use_your_judgment | - |
| **9** | $9.87 | **$4.13** | **$5.38** | $4.13 | $2.55 | $2.83 | - | $0.36 | 27 | use_your_judgment | - |
| **10** | $13.92 | **$6.21** | **$7.07** | $6.21 | $3.39 | $3.68 | - | $0.63 | 28 | use_your_judgment | - |
| **11** | $13.45 | **$5.90** | **$6.93** | $5.90 | $3.31 | $3.62 | - | $0.62 | 28 | use_your_judgment | - |
| **12** | $12.16 | **$5.98** | **$5.76** | $5.98 | $2.70 | $3.06 | - | $0.42 | 27 | use_your_judgment | - |
| **13** | $11.06 | **$4.24** | **$6.45** | $4.24 | $3.10 | $3.35 | - | $0.37 | 21 | use_your_judgment | - |
| **14** | $12.16 | **$7.01** | **$4.69** | $7.01 | $2.21 | $2.48 | - | $0.46 | 27 | use_your_judgment | - |
| **15** | $12.89 | **$7.09** | **$5.33** | $7.09 | $2.52 | $2.81 | - | $0.47 | 28 | use_your_judgment | - |
| **16** | $13.10 | **$7.64** | **$5.16** | $7.64 | $2.45 | $2.71 | - | $0.30 | 29 | use_your_judgment | - |
| **17** | $13.72 | **$8.54** | **$4.52** | $8.54 | $2.14 | $2.38 | - | $0.66 | 28 | use_your_judgment | - |
| **18** | $11.00 | **$5.72** | **$4.80** | $5.72 | $2.28 | $2.52 | - | $0.48 | 27 | use_your_judgment | - |
| **19** | $13.38 | **$7.56** | **$5.44** | $7.56 | $2.59 | $2.85 | - | $0.39 | 32 | use_your_judgment | - |
| **20** | $11.04 | **$4.83** | **$5.90** | $4.83 | $2.81 | $3.10 | - | $0.31 | 27 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$230.46** | **$105.51** | **$116.55** | $105.51 | $55.71 | $60.84 | $0.00 | **$8.39** | **559** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $2.93 | 47.8% | 34 | 16722 |
| Reflection | $3.20 | 52.2% | 4 | 2999 |
| **Total** | **$6.13** | **100%** | **38** | **19721** |

### Cache Performance

- **Cache Creation**: 1,514,226 tokens
- **Cache Hits**: 29,506,169 tokens (95.1% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅