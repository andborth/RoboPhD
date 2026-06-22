# Cost Analysis - Iteration 8

**Total Evaluation Cost: $0.28**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter3_safe_repair_ds1000 | $0.04 | 9/20 | **$0.04** | $0.002 |
| iter7_grounded_reconcile_ds1000 | $0.11 | 3/20 | **$0.11** | $0.005 |
| iter8_perspective_consensus_ds1000 | $0.13 | - | **$0.13** | $0.006 |
| **Total** | **$0.28** | **12/40** | **$0.28** | **$0.005** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter3_safe_repair_ds1000** ($0.043 total)
- openai/gpt-5.4-mini: $0.043 (100%)

**iter7_grounded_reconcile_ds1000** ($0.109 total)
- openai/gpt-5.4-2026-03-05: $0.072 (67%)
- openai/gpt-5.4-mini: $0.036 (33%)

**iter8_perspective_consensus_ds1000** ($0.130 total)
- openai/gpt-5.4-mini: $0.081 (62%)
- openai/gpt-5.4-2026-03-05: $0.049 (38%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter3_safe_repair_ds1000**
1. 906: $0.006
2. 838: $0.006
3. 113: $0.005
4. 883: $0.005
5. 667: $0.004

**iter7_grounded_reconcile_ds1000**
1. 883: $0.022 (gpt-5.4-2026-03-05 $0.018, gpt-5.4-mini $0.004)
2. 838: $0.022 (gpt-5.4-2026-03-05 $0.019, gpt-5.4-mini $0.002)
3. 906: $0.016 (gpt-5.4-2026-03-05 $0.014, gpt-5.4-mini $0.002)
4. 667: $0.009 (gpt-5.4-2026-03-05 $0.007, gpt-5.4-mini $0.002)
5. 763: $0.007 (gpt-5.4-2026-03-05 $0.004, gpt-5.4-mini $0.003)

**iter8_perspective_consensus_ds1000**
1. 883: $0.027 (gpt-5.4-2026-03-05 $0.021, gpt-5.4-mini $0.006)
2. 838: $0.014 (gpt-5.4-mini $0.007, gpt-5.4-2026-03-05 $0.007)
3. 906: $0.012 (gpt-5.4-2026-03-05 $0.007, gpt-5.4-mini $0.006)
4. 804: $0.009 (gpt-5.4-2026-03-05 $0.005, gpt-5.4-mini $0.004)
5. 667: $0.007 (gpt-5.4-mini $0.004, gpt-5.4-2026-03-05 $0.003)
