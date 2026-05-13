# Cost Analysis - Iteration 10

**Total Evaluation Cost: $3.76**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total |
|-----|---------|------|-----|
| iter10_idiomatic_loop_guard_v1 | $2.07 | - | **$2.07** |
| iter4_robust_consensus_v1 | $0.23 | 13/20 | **$0.23** |
| iter9_iter7_polish_v1 | $1.46 | 4/20 | **$1.46** |
| **Total** | **$3.76** | **17/40** | **$3.76** |

---

## Cost by Model

**iter10_idiomatic_loop_guard_v1** ($2.073 total)
- anthropic/claude-opus-4-7: $1.016 (49%)
- anthropic/claude-sonnet-4-6: $0.454 (22%)
- openai/gpt-5.4-2026-03-05: $0.420 (20%)
- google/gemini-3.1-pro-preview: $0.183 (9%)

**iter4_robust_consensus_v1** ($0.233 total)
- anthropic/claude-opus-4-7: $0.135 (58%)
- anthropic/claude-sonnet-4-6: $0.098 (42%)

**iter9_iter7_polish_v1** ($1.457 total)
- anthropic/claude-opus-4-7: $0.721 (49%)
- openai/gpt-5.4-2026-03-05: $0.325 (22%)
- anthropic/claude-sonnet-4-6: $0.277 (19%)
- google/gemini-3.1-pro-preview: $0.134 (9%)


---

## Cost Insights

### Most Expensive Agents
1. iter10_idiomatic_loop_guard_v1: $2.07 (avg $0.104/problem)
2. iter9_iter7_polish_v1: $1.46 (avg $0.091/problem)
3. iter4_robust_consensus_v1: $0.23 (avg $0.033/problem)

### Top 3 Most Expensive Tasks per Agent

**iter10_idiomatic_loop_guard_v1**
1. 919: $0.194 (claude-opus-4-7 $0.102, gpt-5.4-2026-03-05 $0.058, claude-sonnet-4-6 $0.023, +1 more)
2. 808: $0.186 (gpt-5.4-2026-03-05 $0.105, claude-opus-4-7 $0.052, claude-sonnet-4-6 $0.020, +1 more)
3. 10: $0.173 (claude-sonnet-4-6 $0.075, claude-opus-4-7 $0.059, gpt-5.4-2026-03-05 $0.029, +1 more)

**iter4_robust_consensus_v1**
1. 386: $0.048 (claude-opus-4-7 $0.025, claude-sonnet-4-6 $0.023)
2. 372: $0.038 (claude-sonnet-4-6 $0.019, claude-opus-4-7 $0.019)
3. 906: $0.036 (claude-opus-4-7 $0.021, claude-sonnet-4-6 $0.016)

**iter9_iter7_polish_v1**
1. 808: $0.193 (gpt-5.4-2026-03-05 $0.115, claude-opus-4-7 $0.050, claude-sonnet-4-6 $0.020, +1 more)
2. 129: $0.146 (claude-opus-4-7 $0.057, gpt-5.4-2026-03-05 $0.046, claude-sonnet-4-6 $0.034, +1 more)
3. 10: $0.110 (claude-opus-4-7 $0.048, gpt-5.4-2026-03-05 $0.028, claude-sonnet-4-6 $0.023, +1 more)
