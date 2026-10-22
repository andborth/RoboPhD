# Cost Analysis - Iteration 15

**Total Evaluation Cost: $2.30** (+ Other $0.62 = $2.93 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter13_crosscast | $0.74 | $0.25 | 5/14 | **$0.99** | $0.053 |
| iter15_bodycast | $0.88 | $0.24 | - | **$1.13** | $0.063 |
| iter7_headcast | $0.68 | $0.13 | 11/14 | **$0.81** | $0.049 |
| **Total** | **$2.30** | **$0.62** | **16/28** | **$2.93** | **$0.055** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter13_crosscast** ($0.740 total)
- openai/gpt-5.4-mini: $0.513 (69%)
- openai/gpt-5.4-2026-03-05: $0.215 (29%)
- anthropic/claude-sonnet-4-6: $0.012 (2%)

**iter15_bodycast** ($0.884 total)
- openai/gpt-5.4-mini: $0.628 (71%)
- openai/gpt-5.4-2026-03-05: $0.242 (27%)
- anthropic/claude-sonnet-4-6: $0.014 (2%)

**iter7_headcast** ($0.680 total)
- openai/gpt-5.4-mini: $0.473 (70%)
- openai/gpt-5.4-2026-03-05: $0.195 (29%)
- anthropic/claude-sonnet-4-6: $0.012 (2%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter13_crosscast**
1. semantic_87: $0.091 (gpt-5.4-mini $0.073, gpt-5.4-2026-03-05 $0.017)
2. semantic_112: $0.088 (gpt-5.4-mini $0.071, gpt-5.4-2026-03-05 $0.017)
3. semantic_226: $0.082 (gpt-5.4-mini $0.064, gpt-5.4-2026-03-05 $0.018)
4. semantic_160: $0.082 (gpt-5.4-mini $0.064, gpt-5.4-2026-03-05 $0.017)
5. semantic_229: $0.081 (gpt-5.4-mini $0.061, gpt-5.4-2026-03-05 $0.020)

**iter15_bodycast**
1. semantic_87: $0.106 (gpt-5.4-mini $0.083, gpt-5.4-2026-03-05 $0.023)
2. semantic_112: $0.104 (gpt-5.4-mini $0.083, gpt-5.4-2026-03-05 $0.021)
3. semantic_8: $0.101 (gpt-5.4-mini $0.081, gpt-5.4-2026-03-05 $0.020)
4. semantic_155: $0.096 (gpt-5.4-mini $0.072, gpt-5.4-2026-03-05 $0.024)
5. semantic_160: $0.090 (gpt-5.4-mini $0.070, gpt-5.4-2026-03-05 $0.020)

**iter7_headcast**
1. semantic_229: $0.071 (gpt-5.4-mini $0.055, gpt-5.4-2026-03-05 $0.016)
2. semantic_87: $0.071 (gpt-5.4-mini $0.055, gpt-5.4-2026-03-05 $0.017)
3. semantic_8: $0.071 (gpt-5.4-mini $0.055, gpt-5.4-2026-03-05 $0.016)
4. semantic_112: $0.070 (gpt-5.4-mini $0.054, gpt-5.4-2026-03-05 $0.016)
5. semantic_193: $0.069 (gpt-5.4-mini $0.051, gpt-5.4-2026-03-05 $0.018)
