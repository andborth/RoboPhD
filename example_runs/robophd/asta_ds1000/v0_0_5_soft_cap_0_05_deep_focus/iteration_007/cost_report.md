# Cost Analysis - Iteration 7

**Total Evaluation Cost: $0.27**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter3_safe_repair_ds1000 | $0.05 | 10/20 | **$0.05** | $0.002 |
| iter6_grounded_repair_ds1000 | $0.10 | 3/20 | **$0.10** | $0.005 |
| iter7_grounded_reconcile_ds1000 | $0.13 | - | **$0.13** | $0.006 |
| **Total** | **$0.27** | **13/40** | **$0.27** | **$0.005** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter3_safe_repair_ds1000** ($0.047 total)
- openai/gpt-5.4-mini: $0.047 (100%)

**iter6_grounded_repair_ds1000** ($0.097 total)
- openai/gpt-5.4-2026-03-05: $0.063 (66%)
- openai/gpt-5.4-mini: $0.033 (34%)

**iter7_grounded_reconcile_ds1000** ($0.129 total)
- openai/gpt-5.4-2026-03-05: $0.093 (72%)
- openai/gpt-5.4-mini: $0.036 (28%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter3_safe_repair_ds1000**
1. 940: $0.006
2. 822: $0.005
3. 841: $0.005
4. 826: $0.005
5. 667: $0.004

**iter6_grounded_repair_ds1000**
1. 822: $0.014 (gpt-5.4-2026-03-05 $0.012, gpt-5.4-mini $0.002)
2. 826: $0.013 (gpt-5.4-2026-03-05 $0.012, gpt-5.4-mini $0.002)
3. 977: $0.012 (gpt-5.4-2026-03-05 $0.011, gpt-5.4-mini $0.001)
4. 841: $0.012 (gpt-5.4-2026-03-05 $0.011, gpt-5.4-mini $0.001)
5. 940: $0.011 (gpt-5.4-2026-03-05 $0.010, gpt-5.4-mini $0.001)

**iter7_grounded_reconcile_ds1000**
1. 822: $0.020 (gpt-5.4-2026-03-05 $0.018, gpt-5.4-mini $0.002)
2. 826: $0.015 (gpt-5.4-2026-03-05 $0.013, gpt-5.4-mini $0.002)
3. 977: $0.013 (gpt-5.4-2026-03-05 $0.012, gpt-5.4-mini $0.001)
4. 940: $0.013 (gpt-5.4-2026-03-05 $0.011, gpt-5.4-mini $0.001)
5. 841: $0.012 (gpt-5.4-2026-03-05 $0.011, gpt-5.4-mini $0.001)
