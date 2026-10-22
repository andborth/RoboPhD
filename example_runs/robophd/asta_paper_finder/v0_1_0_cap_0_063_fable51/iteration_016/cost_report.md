# Cost Analysis - Iteration 16

**Total Evaluation Cost: $2.44** (+ Other $0.69 = $3.13 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter11_citecast | $0.77 | $0.13 | 5/14 | **$0.90** | $0.055 |
| iter15_bodycast | $0.84 | $0.30 | 3/14 | **$1.14** | $0.060 |
| iter16_feedcast | $0.83 | $0.27 | - | **$1.10** | $0.059 |
| **Total** | **$2.44** | **$0.69** | **8/28** | **$3.13** | **$0.058** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter11_citecast** ($0.770 total)
- openai/gpt-5.4-mini: $0.519 (67%)
- openai/gpt-5.4-2026-03-05: $0.246 (32%)
- anthropic/claude-sonnet-4-6: $0.006 (1%)

**iter15_bodycast** ($0.838 total)
- openai/gpt-5.4-mini: $0.587 (70%)
- openai/gpt-5.4-2026-03-05: $0.244 (29%)
- anthropic/claude-sonnet-4-6: $0.007 (1%)

**iter16_feedcast** ($0.827 total)
- openai/gpt-5.4-mini: $0.602 (73%)
- openai/gpt-5.4-2026-03-05: $0.219 (26%)
- anthropic/claude-sonnet-4-6: $0.007 (1%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter11_citecast**
1. semantic_189: $0.080 (gpt-5.4-mini $0.058, gpt-5.4-2026-03-05 $0.022)
2. semantic_108: $0.080 (gpt-5.4-mini $0.058, gpt-5.4-2026-03-05 $0.022)
3. semantic_140: $0.080 (gpt-5.4-mini $0.056, gpt-5.4-2026-03-05 $0.023)
4. semantic_224: $0.078 (gpt-5.4-mini $0.058, gpt-5.4-2026-03-05 $0.020)
5. semantic_145: $0.071 (gpt-5.4-mini $0.047, gpt-5.4-2026-03-05 $0.024)

**iter15_bodycast**
1. semantic_108: $0.103 (gpt-5.4-mini $0.080, gpt-5.4-2026-03-05 $0.023)
2. semantic_172: $0.100 (gpt-5.4-mini $0.078, gpt-5.4-2026-03-05 $0.022)
3. semantic_189: $0.092 (gpt-5.4-mini $0.071, gpt-5.4-2026-03-05 $0.021)
4. semantic_193: $0.088 (gpt-5.4-mini $0.068, gpt-5.4-2026-03-05 $0.020)
5. semantic_145: $0.081 (gpt-5.4-mini $0.060, gpt-5.4-2026-03-05 $0.021)

**iter16_feedcast**
1. semantic_145: $0.095 (gpt-5.4-mini $0.076, gpt-5.4-2026-03-05 $0.019)
2. semantic_108: $0.094 (gpt-5.4-mini $0.076, gpt-5.4-2026-03-05 $0.018)
3. semantic_189: $0.092 (gpt-5.4-mini $0.074, gpt-5.4-2026-03-05 $0.019)
4. semantic_193: $0.087 (gpt-5.4-mini $0.069, gpt-5.4-2026-03-05 $0.018)
5. semantic_172: $0.085 (gpt-5.4-mini $0.068, gpt-5.4-2026-03-05 $0.017)
