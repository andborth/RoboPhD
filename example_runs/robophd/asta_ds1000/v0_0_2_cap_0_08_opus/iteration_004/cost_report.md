# Cost Analysis - Iteration 4

**Total Evaluation Cost: $0.58**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter2_ds1000_verify_repair | $0.13 | 5/20 | **$0.13** | $0.007 |
| iter3_ds1000_format_aware | $0.24 | 3/20 | **$0.24** | $0.012 |
| iter4_ds1000_idiom_probe | $0.21 | - | **$0.21** | $0.010 |
| **Total** | **$0.58** | **8/40** | **$0.58** | **$0.010** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter2_ds1000_verify_repair**
1. 444: $0.039
2. 662: $0.018
3. 688: $0.008
4. 601: $0.007
5. 667: $0.006

**iter3_ds1000_format_aware**
1. 662: $0.069
2. 444: $0.058
3. 427: $0.024
4. 962: $0.017
5. 919: $0.009

**iter4_ds1000_idiom_probe**
1. 662: $0.073
2. 427: $0.018
3. 444: $0.016
4. 822: $0.014
5. 962: $0.010
