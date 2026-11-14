# Cost Analysis - Iteration 13

**Total Evaluation Cost: $2.76**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter12_thirdvote_adjudicate | $1.05 | 4/20 | **$1.05** | $0.052 |
| iter13_fnsig_adjudicate | $0.96 | - | **$0.96** | $0.048 |
| iter8_refquirk_adjudicate | $0.76 | 11/20 | **$0.76** | $0.038 |
| **Total** | **$2.76** | **15/40** | **$2.76** | **$0.046** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter12_thirdvote_adjudicate** ($1.046 total)
- openai/gpt-5.5: $0.643 (61%)
- anthropic/claude-sonnet-4-6: $0.210 (20%)
- openai/gpt-5.4-2026-03-05: $0.146 (14%)
- google/gemini-3.5-flash: $0.048 (5%)

**iter13_fnsig_adjudicate** ($0.960 total)
- openai/gpt-5.5: $0.519 (54%)
- anthropic/claude-sonnet-4-6: $0.227 (24%)
- openai/gpt-5.4-2026-03-05: $0.153 (16%)
- google/gemini-3.5-flash: $0.062 (6%)

**iter8_refquirk_adjudicate** ($0.756 total)
- openai/gpt-5.5: $0.439 (58%)
- openai/gpt-5.4-2026-03-05: $0.159 (21%)
- anthropic/claude-sonnet-4-6: $0.158 (21%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter12_thirdvote_adjudicate**
1. 129: $0.202 (gpt-5.5 $0.171, claude-sonnet-4-6 $0.013, gpt-5.4-2026-03-05 $0.012, gemini-3.5-flash $0.006)
2. 919: $0.124 (gpt-5.5 $0.101, claude-sonnet-4-6 $0.013, gemini-3.5-flash $0.006, gpt-5.4-2026-03-05 $0.003)
3. 284: $0.117 (gpt-5.5 $0.093, claude-sonnet-4-6 $0.011, gpt-5.4-2026-03-05 $0.008, gemini-3.5-flash $0.005)
4. 763: $0.115 (gpt-5.5 $0.096, claude-sonnet-4-6 $0.011, gemini-3.5-flash $0.005, gpt-5.4-2026-03-05 $0.003)
5. 812: $0.101 (gpt-5.5 $0.076, claude-sonnet-4-6 $0.010, gpt-5.4-2026-03-05 $0.010, gemini-3.5-flash $0.005)

**iter13_fnsig_adjudicate**
1. 919: $0.133 (gpt-5.5 $0.109, claude-sonnet-4-6 $0.014, gemini-3.5-flash $0.006, gpt-5.4-2026-03-05 $0.004)
2. 129: $0.131 (gpt-5.5 $0.106, claude-sonnet-4-6 $0.013, gpt-5.4-2026-03-05 $0.006, gemini-3.5-flash $0.006)
3. 812: $0.086 (gpt-5.5 $0.061, claude-sonnet-4-6 $0.011, gpt-5.4-2026-03-05 $0.008, gemini-3.5-flash $0.006)
4. 284: $0.083 (gpt-5.5 $0.054, claude-sonnet-4-6 $0.015, gpt-5.4-2026-03-05 $0.008, gemini-3.5-flash $0.006)
5. 763: $0.073 (gpt-5.5 $0.050, claude-sonnet-4-6 $0.012, gpt-5.4-2026-03-05 $0.006, gemini-3.5-flash $0.006)

**iter8_refquirk_adjudicate**
1. 812: $0.109 (gpt-5.5 $0.093, gpt-5.4-2026-03-05 $0.009, claude-sonnet-4-6 $0.007)
2. 284: $0.104 (gpt-5.5 $0.079, claude-sonnet-4-6 $0.013, gpt-5.4-2026-03-05 $0.013)
3. 763: $0.094 (gpt-5.5 $0.076, gpt-5.4-2026-03-05 $0.010, claude-sonnet-4-6 $0.008)
4. 919: $0.075 (gpt-5.5 $0.061, claude-sonnet-4-6 $0.010, gpt-5.4-2026-03-05 $0.004)
5. 492: $0.057 (gpt-5.5 $0.041, gpt-5.4-2026-03-05 $0.008, claude-sonnet-4-6 $0.008)
