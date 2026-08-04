# Cost Analysis - Iteration 18

**Total Evaluation Cost: $13.28** (+ Other $0.78 = $14.06 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter17_ambigunion_landmark_v1 | $4.99 | $0.24 | 2/14 | **$5.23** | $0.356 |
| iter18_cocite_largegold_v1 | $4.87 | $0.24 | - | **$5.11** | $0.348 |
| iter9_metafix_poolboost_v1 | $3.42 | $0.30 | 14/14 | **$3.72** | $0.244 |
| **Total** | **$13.28** | **$0.78** | **16/28** | **$14.06** | **$0.316** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter17_ambigunion_landmark_v1** ($4.987 total)
- openai/gpt-5.4-2026-03-05: $4.314 (87%)
- openai/gpt-5.4-mini: $0.673 (13%)

**iter18_cocite_largegold_v1** ($4.869 total)
- openai/gpt-5.4-2026-03-05: $4.258 (87%)
- openai/gpt-5.4-mini: $0.611 (13%)

**iter9_metafix_poolboost_v1** ($3.420 total)
- openai/gpt-5.4-2026-03-05: $3.420 (100%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter17_ambigunion_landmark_v1**
1. semantic_220: $0.448 (gpt-5.4-2026-03-05 $0.390, gpt-5.4-mini $0.058)
2. semantic_222: $0.442 (gpt-5.4-2026-03-05 $0.375, gpt-5.4-mini $0.067)
3. semantic_152: $0.436 (gpt-5.4-2026-03-05 $0.370, gpt-5.4-mini $0.066)
4. semantic_110: $0.434 (gpt-5.4-2026-03-05 $0.365, gpt-5.4-mini $0.069)
5. semantic_43: $0.412 (gpt-5.4-2026-03-05 $0.372, gpt-5.4-mini $0.040)

**iter18_cocite_largegold_v1**
1. semantic_110: $0.438 (gpt-5.4-2026-03-05 $0.369, gpt-5.4-mini $0.070)
2. semantic_222: $0.434 (gpt-5.4-2026-03-05 $0.365, gpt-5.4-mini $0.069)
3. semantic_43: $0.431 (gpt-5.4-2026-03-05 $0.387, gpt-5.4-mini $0.043)
4. semantic_220: $0.416 (gpt-5.4-2026-03-05 $0.367, gpt-5.4-mini $0.049)
5. semantic_174: $0.397 (gpt-5.4-2026-03-05 $0.350, gpt-5.4-mini $0.047)

**iter9_metafix_poolboost_v1**
1. semantic_222: $0.298
2. semantic_152: $0.286
3. semantic_43: $0.286
4. semantic_110: $0.280
5. semantic_220: $0.272
