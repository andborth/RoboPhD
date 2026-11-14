# Cost Analysis - Iteration 12

**Total Evaluation Cost: $0.09**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter10_strong_reindent | $0.03 | 7/20 | **$0.03** | $0.002 |
| iter12_strong_toplevel | $0.03 | - | **$0.03** | $0.002 |
| iter6_strong_oneshot | $0.03 | 13/20 | **$0.03** | $0.002 |
| **Total** | **$0.09** | **20/40** | **$0.09** | **$0.002** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses. Cache does not affect this calculation.*

---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter10_strong_reindent**
1. 808: $0.003
2. 662: $0.002
3. 34: $0.002
4. 10: $0.002
5. 190: $0.002

**iter12_strong_toplevel**
1. 808: $0.004
2. 662: $0.002
3. 34: $0.002
4. 10: $0.002
5. 190: $0.002

**iter6_strong_oneshot**
1. 808: $0.004
2. 10: $0.003
3. 662: $0.002
4. 34: $0.002
5. 190: $0.002
