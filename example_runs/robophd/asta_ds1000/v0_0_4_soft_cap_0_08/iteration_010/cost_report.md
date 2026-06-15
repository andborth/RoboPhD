# Cost Analysis - Iteration 10

**Total Evaluation Cost: $1.27**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter10_ds1000_dtypeverify | $0.57 | - | **$0.57** | $0.028 |
| iter3_ds1000_ensemble_judge | $0.26 | 14/20 | **$0.26** | $0.013 |
| iter8_ds1000_strongjudge | $0.45 | 2/20 | **$0.45** | $0.022 |
| **Total** | **$1.27** | **16/40** | **$1.27** | **$0.021** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter10_ds1000_dtypeverify** ($0.566 total)
- openai/gpt-5.4-2026-03-05: $0.466 (82%)
- anthropic/claude-sonnet-4-6: $0.100 (18%)

**iter3_ds1000_ensemble_judge** ($0.260 total)
- openai/gpt-5.4-2026-03-05: $0.195 (75%)
- anthropic/claude-sonnet-4-6: $0.065 (25%)

**iter8_ds1000_strongjudge** ($0.446 total)
- openai/gpt-5.4-2026-03-05: $0.358 (80%)
- anthropic/claude-sonnet-4-6: $0.088 (20%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter10_ds1000_dtypeverify**
1. 426: $0.131 (gpt-5.4-2026-03-05 $0.127, claude-sonnet-4-6 $0.005)
2. 812: $0.065 (gpt-5.4-2026-03-05 $0.060, claude-sonnet-4-6 $0.004)
3. 919: $0.064 (gpt-5.4-2026-03-05 $0.057, claude-sonnet-4-6 $0.007)
4. 836: $0.055 (gpt-5.4-2026-03-05 $0.050, claude-sonnet-4-6 $0.005)
5. 918: $0.047 (gpt-5.4-2026-03-05 $0.042, claude-sonnet-4-6 $0.004)

**iter3_ds1000_ensemble_judge**
1. 129: $0.063 (gpt-5.4-2026-03-05 $0.058, claude-sonnet-4-6 $0.005)
2. 919: $0.021 (gpt-5.4-2026-03-05 $0.015, claude-sonnet-4-6 $0.005)
3. 836: $0.019 (gpt-5.4-2026-03-05 $0.016, claude-sonnet-4-6 $0.003)
4. 446: $0.019 (gpt-5.4-2026-03-05 $0.016, claude-sonnet-4-6 $0.003)
5. 918: $0.016 (gpt-5.4-2026-03-05 $0.013, claude-sonnet-4-6 $0.003)

**iter8_ds1000_strongjudge**
1. 836: $0.078 (gpt-5.4-2026-03-05 $0.073, claude-sonnet-4-6 $0.004)
2. 129: $0.064 (gpt-5.4-2026-03-05 $0.058, claude-sonnet-4-6 $0.006)
3. 919: $0.055 (gpt-5.4-2026-03-05 $0.048, claude-sonnet-4-6 $0.007)
4. 918: $0.047 (gpt-5.4-2026-03-05 $0.043, claude-sonnet-4-6 $0.004)
5. 812: $0.035 (gpt-5.4-2026-03-05 $0.031, claude-sonnet-4-6 $0.004)
