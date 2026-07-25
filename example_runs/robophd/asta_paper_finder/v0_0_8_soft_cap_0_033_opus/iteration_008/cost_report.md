# Cost Analysis - Iteration 8

**Total Evaluation Cost: $0.36** (+ Other $1.89 = $2.26 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter5_grade3_rerank_v1 | $0.22 | $0.96 | 3/14 | **$1.18** | $0.015 |
| iter6_metadata_targeted_v1 | $0.07 | $0.47 | 7/14 | **$0.54** | $0.005 |
| iter8_grade3_probe_v1 | $0.07 | $0.46 | - | **$0.54** | $0.005 |
| **Total** | **$0.36** | **$1.89** | **10/28** | **$2.26** | **$0.009** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter5_grade3_rerank_v1** ($0.216 total)
- openai/gpt-5.4-2026-03-05: $0.212 (98%)
- openai/gpt-5.4-mini: $0.004 (2%)

**iter6_metadata_targeted_v1** ($0.073 total)
- openai/gpt-5.4-mini: $0.073 (100%)

**iter8_grade3_probe_v1** ($0.075 total)
- openai/gpt-5.4-mini: $0.075 (100%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter5_grade3_rerank_v1**
1. semantic_222: $0.020 (gpt-5.4-2026-03-05 $0.020, gpt-5.4-mini $0.000)
2. semantic_203: $0.020 (gpt-5.4-2026-03-05 $0.020, gpt-5.4-mini $0.000)
3. semantic_125: $0.020 (gpt-5.4-2026-03-05 $0.020, gpt-5.4-mini $0.000)
4. semantic_152: $0.020 (gpt-5.4-2026-03-05 $0.019, gpt-5.4-mini $0.000)
5. semantic_108: $0.020 (gpt-5.4-2026-03-05 $0.019, gpt-5.4-mini $0.000)

**iter6_metadata_targeted_v1**
1. metadata_14: $0.006
2. semantic_222: $0.006
3. semantic_152: $0.006
4. semantic_189: $0.006
5. semantic_203: $0.006

**iter8_grade3_probe_v1**
1. metadata_14: $0.007
2. semantic_222: $0.006
3. semantic_203: $0.006
4. semantic_125: $0.006
5. semantic_152: $0.006
