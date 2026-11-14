# Cost Analysis - Iteration 3

**Total Evaluation Cost: $0.30**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter2_ds1000_verify_repair | $0.10 | 5/20 | **$0.10** | $0.005 |
| iter3_ds1000_format_aware | $0.19 | - | **$0.19** | $0.010 |
| seed_yyg6m9ud | $0.01 | 10/20 | **$0.01** | $0.001 |
| **Total** | **$0.30** | **15/40** | **$0.30** | **$0.005** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter2_ds1000_verify_repair**
1. 706: $0.011
2. 723: $0.011
3. 961: $0.010
4. 614: $0.009
5. 688: $0.008

**iter3_ds1000_format_aware**
1. 706: $0.047
2. 269: $0.023
3. 426: $0.021
4. 961: $0.012
5. 763: $0.012

**seed_yyg6m9ud**
1. 706: $0.001
2. 426: $0.001
3. 420: $0.001
4. 803: $0.001
5. 34: $0.001
