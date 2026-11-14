# Cost Analysis - Iteration 12

**Total Evaluation Cost: $2.00** (+ Other $6.70 = $8.70 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter11_tail_saturate | $0.74 | $2.22 | 3/14 | **$2.96** | $0.053 |
| iter12_body_conjunction | $0.66 | $2.19 | - | **$2.85** | $0.047 |
| iter6_grade3_rescue | $0.61 | $2.28 | 8/14 | **$2.89** | $0.043 |
| **Total** | **$2.00** | **$6.70** | **11/28** | **$8.70** | **$0.048** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter11_tail_saturate** ($0.736 total)
- openai/gpt-5.4-mini: $0.523 (71%)
- openai/gpt-5.4-2026-03-05: $0.213 (29%)

**iter12_body_conjunction** ($0.659 total)
- openai/gpt-5.4-mini: $0.461 (70%)
- openai/gpt-5.4-2026-03-05: $0.198 (30%)

**iter6_grade3_rescue** ($0.609 total)
- openai/gpt-5.4-mini: $0.500 (82%)
- openai/gpt-5.4-2026-03-05: $0.109 (18%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter11_tail_saturate**
1. semantic_170: $0.099 (gpt-5.4-mini $0.076, gpt-5.4-2026-03-05 $0.023)
2. semantic_100: $0.098 (gpt-5.4-mini $0.073, gpt-5.4-2026-03-05 $0.025)
3. semantic_192: $0.092 (gpt-5.4-mini $0.070, gpt-5.4-2026-03-05 $0.022)
4. semantic_214: $0.090 (gpt-5.4-mini $0.068, gpt-5.4-2026-03-05 $0.022)
5. semantic_43: $0.086 (gpt-5.4-mini $0.065, gpt-5.4-2026-03-05 $0.021)

**iter12_body_conjunction**
1. semantic_100: $0.092 (gpt-5.4-mini $0.066, gpt-5.4-2026-03-05 $0.026)
2. semantic_192: $0.086 (gpt-5.4-mini $0.063, gpt-5.4-2026-03-05 $0.023)
3. semantic_43: $0.084 (gpt-5.4-mini $0.062, gpt-5.4-2026-03-05 $0.022)
4. semantic_214: $0.080 (gpt-5.4-mini $0.057, gpt-5.4-2026-03-05 $0.023)
5. semantic_233: $0.078 (gpt-5.4-mini $0.057, gpt-5.4-2026-03-05 $0.021)

**iter6_grade3_rescue**
1. semantic_100: $0.080 (gpt-5.4-mini $0.069, gpt-5.4-2026-03-05 $0.010)
2. semantic_214: $0.075 (gpt-5.4-mini $0.066, gpt-5.4-2026-03-05 $0.009)
3. semantic_192: $0.073 (gpt-5.4-mini $0.065, gpt-5.4-2026-03-05 $0.008)
4. semantic_98: $0.071 (gpt-5.4-mini $0.063, gpt-5.4-2026-03-05 $0.008)
5. semantic_138: $0.071 (gpt-5.4-mini $0.062, gpt-5.4-2026-03-05 $0.009)
