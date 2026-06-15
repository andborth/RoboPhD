# Cost Analysis - Iteration 11

**Total Evaluation Cost: $1.55**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter11_ds1000_tridtype_judge | $0.90 | - | **$0.90** | $0.045 |
| iter3_ds1000_ensemble_judge | $0.23 | 16/20 | **$0.23** | $0.011 |
| iter8_ds1000_strongjudge | $0.42 | 9/20 | **$0.42** | $0.021 |
| **Total** | **$1.55** | **25/40** | **$1.55** | **$0.026** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter11_ds1000_tridtype_judge** ($0.900 total)
- openai/gpt-5.4-2026-03-05: $0.742 (82%)
- anthropic/claude-sonnet-4-6: $0.098 (11%)
- google/gemini-3.1-pro-preview: $0.060 (7%)

**iter3_ds1000_ensemble_judge** ($0.229 total)
- openai/gpt-5.4-2026-03-05: $0.164 (71%)
- anthropic/claude-sonnet-4-6: $0.065 (29%)

**iter8_ds1000_strongjudge** ($0.417 total)
- openai/gpt-5.4-2026-03-05: $0.331 (79%)
- anthropic/claude-sonnet-4-6: $0.086 (21%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter11_ds1000_tridtype_judge**
1. 165: $0.200 (gpt-5.4-2026-03-05 $0.192, claude-sonnet-4-6 $0.005, gemini-3.1-pro-preview $0.003)
2. 426: $0.161 (gpt-5.4-2026-03-05 $0.153, claude-sonnet-4-6 $0.005, gemini-3.1-pro-preview $0.003)
3. 38: $0.064 (gpt-5.4-2026-03-05 $0.056, claude-sonnet-4-6 $0.005, gemini-3.1-pro-preview $0.003)
4. 564: $0.060 (gpt-5.4-2026-03-05 $0.053, claude-sonnet-4-6 $0.004, gemini-3.1-pro-preview $0.003)
5. 420: $0.059 (gpt-5.4-2026-03-05 $0.052, claude-sonnet-4-6 $0.005, gemini-3.1-pro-preview $0.003)

**iter3_ds1000_ensemble_judge**
1. 420: $0.023 (gpt-5.4-2026-03-05 $0.020, claude-sonnet-4-6 $0.003)
2. 836: $0.019 (gpt-5.4-2026-03-05 $0.016, claude-sonnet-4-6 $0.003)
3. 822: $0.017 (gpt-5.4-2026-03-05 $0.013, claude-sonnet-4-6 $0.003)
4. 142: $0.017 (gpt-5.4-2026-03-05 $0.010, claude-sonnet-4-6 $0.007)
5. 124: $0.017 (gpt-5.4-2026-03-05 $0.013, claude-sonnet-4-6 $0.003)

**iter8_ds1000_strongjudge**
1. 836: $0.078 (gpt-5.4-2026-03-05 $0.073, claude-sonnet-4-6 $0.004)
2. 420: $0.046 (gpt-5.4-2026-03-05 $0.042, claude-sonnet-4-6 $0.004)
3. 142: $0.045 (gpt-5.4-2026-03-05 $0.036, claude-sonnet-4-6 $0.009)
4. 822: $0.044 (gpt-5.4-2026-03-05 $0.039, claude-sonnet-4-6 $0.005)
5. 812: $0.035 (gpt-5.4-2026-03-05 $0.031, claude-sonnet-4-6 $0.004)
