# Cost Analysis - Iteration 17

**Total Evaluation Cost: $8.29** (+ Other $0.65 = $8.93 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter12_citerevive_multiref_v1 | $2.75 | $0.21 | 10/14 | **$2.97** | $0.197 |
| iter17_ambigunion_landmark_v1 | $3.25 | $0.20 | - | **$3.45** | $0.232 |
| iter9_metafix_poolboost_v1 | $2.28 | $0.23 | 11/14 | **$2.52** | $0.163 |
| **Total** | **$8.29** | **$0.65** | **21/28** | **$8.93** | **$0.197** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter12_citerevive_multiref_v1** ($2.754 total)
- openai/gpt-5.4-2026-03-05: $2.664 (97%)
- openai/gpt-5.4-mini: $0.090 (3%)

**iter17_ambigunion_landmark_v1** ($3.248 total)
- openai/gpt-5.4-2026-03-05: $2.885 (89%)
- openai/gpt-5.4-mini: $0.362 (11%)

**iter9_metafix_poolboost_v1** ($2.285 total)
- openai/gpt-5.4-2026-03-05: $2.284 (100%)
- openai/gpt-5.4-mini: $0.001 (0%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter12_citerevive_multiref_v1**
1. semantic_189: $0.363 (gpt-5.4-2026-03-05 $0.343, gpt-5.4-mini $0.020)
2. semantic_77: $0.357 (gpt-5.4-2026-03-05 $0.345, gpt-5.4-mini $0.012)
3. semantic_100: $0.356 (gpt-5.4-2026-03-05 $0.347, gpt-5.4-mini $0.009)
4. semantic_155: $0.338 (gpt-5.4-2026-03-05 $0.327, gpt-5.4-mini $0.011)
5. semantic_186: $0.335 (gpt-5.4-2026-03-05 $0.320, gpt-5.4-mini $0.015)

**iter17_ambigunion_landmark_v1**
1. semantic_189: $0.439 (gpt-5.4-2026-03-05 $0.370, gpt-5.4-mini $0.069)
2. semantic_77: $0.432 (gpt-5.4-2026-03-05 $0.389, gpt-5.4-mini $0.043)
3. semantic_155: $0.411 (gpt-5.4-2026-03-05 $0.344, gpt-5.4-mini $0.066)
4. semantic_100: $0.394 (gpt-5.4-2026-03-05 $0.349, gpt-5.4-mini $0.045)
5. semantic_193: $0.367 (gpt-5.4-2026-03-05 $0.341, gpt-5.4-mini $0.027)

**iter9_metafix_poolboost_v1**
1. semantic_100: $0.302
2. semantic_77: $0.296
3. semantic_189: $0.281
4. semantic_155: $0.276
5. semantic_193: $0.267
