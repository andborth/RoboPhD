# Cost Analysis - Iteration 14

**Total Evaluation Cost: $3.15**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter14_filemock_adjudicate | $1.07 | - | **$1.07** | $0.054 |
| iter6_inplace_expect_adjudicate | $1.04 | 16/20 | **$1.04** | $0.052 |
| iter8_refquirk_adjudicate | $1.03 | 11/20 | **$1.03** | $0.052 |
| **Total** | **$3.15** | **27/40** | **$3.15** | **$0.052** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter14_filemock_adjudicate** ($1.073 total)
- openai/gpt-5.5: $0.576 (54%)
- anthropic/claude-sonnet-4-6: $0.245 (23%)
- openai/gpt-5.4-2026-03-05: $0.184 (17%)
- google/gemini-3.5-flash: $0.069 (6%)

**iter6_inplace_expect_adjudicate** ($1.045 total)
- openai/gpt-5.5: $0.775 (74%)
- openai/gpt-5.4-2026-03-05: $0.143 (14%)
- anthropic/claude-sonnet-4-6: $0.127 (12%)

**iter8_refquirk_adjudicate** ($1.032 total)
- openai/gpt-5.5: $0.692 (67%)
- anthropic/claude-sonnet-4-6: $0.172 (17%)
- openai/gpt-5.4-2026-03-05: $0.167 (16%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter14_filemock_adjudicate**
1. 142: $0.120 (gpt-5.5 $0.084, claude-sonnet-4-6 $0.020, gpt-5.4-2026-03-05 $0.010, gemini-3.5-flash $0.006)
2. 838: $0.119 (gpt-5.5 $0.089, claude-sonnet-4-6 $0.013, gpt-5.4-2026-03-05 $0.010, gemini-3.5-flash $0.006)
3. 129: $0.119 (gpt-5.5 $0.093, claude-sonnet-4-6 $0.014, gemini-3.5-flash $0.007, gpt-5.4-2026-03-05 $0.005)
4. 444: $0.108 (gpt-5.5 $0.083, claude-sonnet-4-6 $0.013, gpt-5.4-2026-03-05 $0.007, gemini-3.5-flash $0.006)
5. 284: $0.084 (gpt-5.5 $0.058, claude-sonnet-4-6 $0.012, gpt-5.4-2026-03-05 $0.008, gemini-3.5-flash $0.006)

**iter6_inplace_expect_adjudicate**
1. 269: $0.259 (gpt-5.5 $0.246, gpt-5.4-2026-03-05 $0.007, claude-sonnet-4-6 $0.006)
2. 129: $0.160 (gpt-5.5 $0.140, gpt-5.4-2026-03-05 $0.012, claude-sonnet-4-6 $0.008)
3. 142: $0.103 (gpt-5.5 $0.078, gpt-5.4-2026-03-05 $0.015, claude-sonnet-4-6 $0.011)
4. 444: $0.069 (gpt-5.5 $0.057, gpt-5.4-2026-03-05 $0.006, claude-sonnet-4-6 $0.006)
5. 944: $0.068 (gpt-5.5 $0.058, claude-sonnet-4-6 $0.006, gpt-5.4-2026-03-05 $0.004)

**iter8_refquirk_adjudicate**
1. 883: $0.185 (gpt-5.5 $0.171, claude-sonnet-4-6 $0.008, gpt-5.4-2026-03-05 $0.005)
2. 944: $0.119 (gpt-5.5 $0.107, claude-sonnet-4-6 $0.007, gpt-5.4-2026-03-05 $0.005)
3. 284: $0.104 (gpt-5.5 $0.079, claude-sonnet-4-6 $0.013, gpt-5.4-2026-03-05 $0.013)
4. 445: $0.080 (gpt-5.5 $0.061, gpt-5.4-2026-03-05 $0.011, claude-sonnet-4-6 $0.007)
5. 919: $0.075 (gpt-5.5 $0.061, claude-sonnet-4-6 $0.010, gpt-5.4-2026-03-05 $0.004)
