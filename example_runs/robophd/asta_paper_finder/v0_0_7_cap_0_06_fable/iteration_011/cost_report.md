# Cost Analysis - Iteration 11

**Total Evaluation Cost: $2.13** (+ Other $13.46 = $15.59 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter10_cite_expand | $0.79 | $4.37 | 5/14 | **$5.16** | $0.056 |
| iter11_tail_saturate | $0.82 | $4.86 | - | **$5.68** | $0.058 |
| iter4_judge_sim_ranker | $0.52 | $4.23 | 9/14 | **$4.75** | $0.037 |
| **Total** | **$2.13** | **$13.46** | **14/28** | **$15.59** | **$0.051** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter10_cite_expand** ($0.789 total)
- openai/gpt-5.4-mini: $0.616 (78%)
- openai/gpt-5.4-2026-03-05: $0.173 (22%)

**iter11_tail_saturate** ($0.819 total)
- openai/gpt-5.4-mini: $0.624 (76%)
- openai/gpt-5.4-2026-03-05: $0.195 (24%)

**iter4_judge_sim_ranker** ($0.524 total)
- openai/gpt-5.4-mini: $0.446 (85%)
- openai/gpt-5.4-2026-03-05: $0.077 (15%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter10_cite_expand**
1. semantic_7: $0.091 (gpt-5.4-mini $0.072, gpt-5.4-2026-03-05 $0.019)
2. semantic_222: $0.090 (gpt-5.4-mini $0.072, gpt-5.4-2026-03-05 $0.019)
3. semantic_219: $0.089 (gpt-5.4-mini $0.070, gpt-5.4-2026-03-05 $0.019)
4. semantic_110: $0.088 (gpt-5.4-mini $0.067, gpt-5.4-2026-03-05 $0.021)
5. semantic_104: $0.084 (gpt-5.4-mini $0.066, gpt-5.4-2026-03-05 $0.018)

**iter11_tail_saturate**
1. semantic_7: $0.100 (gpt-5.4-mini $0.076, gpt-5.4-2026-03-05 $0.024)
2. semantic_137: $0.094 (gpt-5.4-mini $0.074, gpt-5.4-2026-03-05 $0.020)
3. semantic_222: $0.093 (gpt-5.4-mini $0.070, gpt-5.4-2026-03-05 $0.023)
4. semantic_219: $0.087 (gpt-5.4-mini $0.065, gpt-5.4-2026-03-05 $0.022)
5. semantic_104: $0.082 (gpt-5.4-mini $0.065, gpt-5.4-2026-03-05 $0.018)

**iter4_judge_sim_ranker**
1. semantic_219: $0.052 (gpt-5.4-mini $0.046, gpt-5.4-2026-03-05 $0.006)
2. semantic_222: $0.052 (gpt-5.4-mini $0.046, gpt-5.4-2026-03-05 $0.006)
3. semantic_110: $0.052 (gpt-5.4-mini $0.045, gpt-5.4-2026-03-05 $0.006)
4. semantic_137: $0.051 (gpt-5.4-mini $0.045, gpt-5.4-2026-03-05 $0.007)
5. semantic_193: $0.051 (gpt-5.4-mini $0.045, gpt-5.4-2026-03-05 $0.006)
