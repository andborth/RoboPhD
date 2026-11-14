# Cost Analysis - Iteration 9

**Total Evaluation Cost: $3.42**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total |
|-----|---------|------|-----|
| iter3_ensemble_judge_v1 | $0.12 | 16/20 | **$0.12** |
| iter7_quad_diverse_critic_v1 | $1.18 | 7/20 | **$1.18** |
| iter9_iter7_polish_v1 | $2.12 | - | **$2.12** |
| **Total** | **$3.42** | **23/40** | **$3.42** |

---

## Cost by Model

**iter3_ensemble_judge_v1** ($0.125 total)
- anthropic/claude-opus-4-7: $0.067 (53%)
- anthropic/claude-sonnet-4-6: $0.058 (47%)

**iter7_quad_diverse_critic_v1** ($1.176 total)
- anthropic/claude-opus-4-7: $0.593 (50%)
- anthropic/claude-sonnet-4-6: $0.258 (22%)
- openai/gpt-5.4-2026-03-05: $0.231 (20%)
- google/gemini-3.1-pro-preview: $0.095 (8%)

**iter9_iter7_polish_v1** ($2.121 total)
- anthropic/claude-opus-4-7: $1.036 (49%)
- openai/gpt-5.4-2026-03-05: $0.500 (24%)
- anthropic/claude-sonnet-4-6: $0.416 (20%)
- google/gemini-3.1-pro-preview: $0.169 (8%)


---

## Cost Insights

### Most Expensive Agents
1. iter9_iter7_polish_v1: $2.12 (avg $0.106/problem)
2. iter7_quad_diverse_critic_v1: $1.18 (avg $0.090/problem)
3. iter3_ensemble_judge_v1: $0.12 (avg $0.031/problem)

### Top 3 Most Expensive Tasks per Agent

**iter3_ensemble_judge_v1**
1. 419: $0.037 (claude-sonnet-4-6 $0.020, claude-opus-4-7 $0.017)
2. 883: $0.037 (claude-sonnet-4-6 $0.020, claude-opus-4-7 $0.017)
3. 262: $0.028 (claude-opus-4-7 $0.017, claude-sonnet-4-6 $0.011)

**iter7_quad_diverse_critic_v1**
1. 883: $0.214 (claude-opus-4-7 $0.115, gpt-5.4-2026-03-05 $0.072, claude-sonnet-4-6 $0.020, +1 more)
2. 446: $0.139 (claude-sonnet-4-6 $0.067, claude-opus-4-7 $0.051, gpt-5.4-2026-03-05 $0.014, +1 more)
3. 372: $0.132 (gpt-5.4-2026-03-05 $0.060, claude-opus-4-7 $0.040, claude-sonnet-4-6 $0.024, +1 more)

**iter9_iter7_polish_v1**
1. 883: $0.248 (claude-opus-4-7 $0.114, gpt-5.4-2026-03-05 $0.105, claude-sonnet-4-6 $0.021, +1 more)
2. 919: $0.197 (claude-opus-4-7 $0.097, gpt-5.4-2026-03-05 $0.072, claude-sonnet-4-6 $0.019, +1 more)
3. 446: $0.170 (claude-sonnet-4-6 $0.074, claude-opus-4-7 $0.072, gpt-5.4-2026-03-05 $0.015, +1 more)
