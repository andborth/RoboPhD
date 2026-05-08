# Cost Analysis - Iteration 13

**Total Evaluation Cost: $0.82**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total |
|-----|---------|------|-----|
| iter13_style_aware_lean | $0.45 | - | **$0.45** |
| iter8_robust_judge | $0.19 | 8/20 | **$0.19** |
| iter9_mpl_aware_judge | $0.18 | 13/20 | **$0.18** |
| **Total** | **$0.82** | **21/40** | **$0.82** |

---

## Cost Insights

### Most Expensive Agents
1. iter9_mpl_aware_judge: $0.18 (avg $0.026/problem)
2. iter13_style_aware_lean: $0.45 (avg $0.023/problem)
3. iter8_robust_judge: $0.19 (avg $0.015/problem)

### Top 3 Most Expensive Tasks per Agent

**iter13_style_aware_lean**
1. 238: $0.051
2. 269: $0.039
3. 961: $0.038

**iter8_robust_judge**
1. 961: $0.025
2. 446: $0.021
3. 861: $0.021

**iter9_mpl_aware_judge**
1. 238: $0.041
2. 961: $0.037
3. 446: $0.029
