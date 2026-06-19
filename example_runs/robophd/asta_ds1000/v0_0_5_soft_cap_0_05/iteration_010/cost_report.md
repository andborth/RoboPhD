# Cost Analysis - Iteration 10

**Total Evaluation Cost: $0.50**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter10_literal_consensus | $0.25 | - | **$0.25** | $0.013 |
| iter3_fmt_strong_cascade | $0.10 | 15/20 | **$0.10** | $0.005 |
| iter7_agree_escalate | $0.15 | 8/20 | **$0.15** | $0.007 |
| **Total** | **$0.50** | **23/40** | **$0.50** | **$0.008** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter10_literal_consensus** ($0.255 total)
- openai/gpt-5.4-2026-03-05: $0.182 (72%)
- anthropic/claude-sonnet-4-6: $0.072 (28%)

**iter3_fmt_strong_cascade** ($0.099 total)
- openai/gpt-5.4-2026-03-05: $0.078 (79%)
- anthropic/claude-sonnet-4-6: $0.021 (21%)

**iter7_agree_escalate** ($0.145 total)
- openai/gpt-5.4-2026-03-05: $0.087 (60%)
- anthropic/claude-sonnet-4-6: $0.059 (40%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter10_literal_consensus**
1. 445: $0.053 (gpt-5.4-2026-03-05 $0.049, claude-sonnet-4-6 $0.004)
2. 372: $0.026
3. 918: $0.017 (gpt-5.4-2026-03-05 $0.011, claude-sonnet-4-6 $0.006)
4. 185: $0.015 (claude-sonnet-4-6 $0.008, gpt-5.4-2026-03-05 $0.006)
5. 446: $0.013 (gpt-5.4-2026-03-05 $0.009, claude-sonnet-4-6 $0.004)

**iter3_fmt_strong_cascade**
1. 446: $0.012
2. 445: $0.011
3. 910: $0.008 (claude-sonnet-4-6 $0.005, gpt-5.4-2026-03-05 $0.003)
4. 690: $0.007 (claude-sonnet-4-6 $0.005, gpt-5.4-2026-03-05 $0.003)
5. 918: $0.007 (claude-sonnet-4-6 $0.004, gpt-5.4-2026-03-05 $0.003)

**iter7_agree_escalate**
1. 446: $0.014 (gpt-5.4-2026-03-05 $0.010, claude-sonnet-4-6 $0.003)
2. 185: $0.012 (claude-sonnet-4-6 $0.007, gpt-5.4-2026-03-05 $0.005)
3. 383: $0.012 (gpt-5.4-2026-03-05 $0.008, claude-sonnet-4-6 $0.003)
4. 918: $0.010 (gpt-5.4-2026-03-05 $0.005, claude-sonnet-4-6 $0.005)
5. 432: $0.009 (gpt-5.4-2026-03-05 $0.006, claude-sonnet-4-6 $0.004)
