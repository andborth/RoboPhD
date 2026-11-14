# Cost Analysis - Iteration 9

**Total Evaluation Cost: $0.73**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter3_fmt_strong_cascade | $0.08 | 15/20 | **$0.08** | $0.004 |
| iter7_agree_escalate | $0.21 | 8/20 | **$0.21** | $0.010 |
| iter9_reason_agree | $0.44 | - | **$0.44** | $0.022 |
| **Total** | **$0.73** | **23/40** | **$0.73** | **$0.012** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter3_fmt_strong_cascade** ($0.084 total)
- openai/gpt-5.4-2026-03-05: $0.076 (90%)
- anthropic/claude-sonnet-4-6: $0.008 (10%)

**iter7_agree_escalate** ($0.208 total)
- openai/gpt-5.4-2026-03-05: $0.144 (69%)
- anthropic/claude-sonnet-4-6: $0.064 (31%)

**iter9_reason_agree** ($0.442 total)
- openai/gpt-5.4-2026-03-05: $0.379 (86%)
- anthropic/claude-sonnet-4-6: $0.063 (14%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter3_fmt_strong_cascade**
1. 445: $0.011
2. 961: $0.008 (claude-sonnet-4-6 $0.005, gpt-5.4-2026-03-05 $0.003)
3. 269: $0.007
4. 576: $0.006 (claude-sonnet-4-6 $0.004, gpt-5.4-2026-03-05 $0.002)
5. 883: $0.005

**iter7_agree_escalate**
1. 804: $0.044 (gpt-5.4-2026-03-05 $0.040, claude-sonnet-4-6 $0.004)
2. 999: $0.017 (gpt-5.4-2026-03-05 $0.013, claude-sonnet-4-6 $0.003)
3. 440: $0.016 (gpt-5.4-2026-03-05 $0.012, claude-sonnet-4-6 $0.004)
4. 338: $0.015 (gpt-5.4-2026-03-05 $0.011, claude-sonnet-4-6 $0.004)
5. 398: $0.014 (gpt-5.4-2026-03-05 $0.011, claude-sonnet-4-6 $0.004)

**iter9_reason_agree**
1. 445: $0.147 (gpt-5.4-2026-03-05 $0.144, claude-sonnet-4-6 $0.003)
2. 883: $0.045
3. 999: $0.039 (gpt-5.4-2026-03-05 $0.036, claude-sonnet-4-6 $0.004)
4. 812: $0.031 (gpt-5.4-2026-03-05 $0.027, claude-sonnet-4-6 $0.004)
5. 804: $0.029 (gpt-5.4-2026-03-05 $0.024, claude-sonnet-4-6 $0.005)
