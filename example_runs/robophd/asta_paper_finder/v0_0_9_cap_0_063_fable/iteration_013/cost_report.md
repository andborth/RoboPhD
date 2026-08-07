# Cost Analysis - Iteration 13

**Total Evaluation Cost: $2.55** (+ Other $0.77 = $3.31 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter10_deadline_guard | $0.89 | $0.24 | 4/14 | **$1.13** | $0.064 |
| iter12_salvage_rank | $0.81 | $0.27 | - | **$1.08** | $0.058 |
| iter13_any_author_gate | $0.84 | $0.26 | - | **$1.10** | $0.060 |
| **Total** | **$2.55** | **$0.77** | **4/14** | **$3.31** | **$0.061** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter10_deadline_guard** ($0.891 total)
- openai/gpt-5.4-mini: $0.776 (87%)
- openai/gpt-5.4-2026-03-05: $0.115 (13%)

**iter12_salvage_rank** ($0.809 total)
- openai/gpt-5.4-mini: $0.691 (85%)
- openai/gpt-5.4-2026-03-05: $0.118 (15%)

**iter13_any_author_gate** ($0.845 total)
- openai/gpt-5.4-mini: $0.723 (86%)
- openai/gpt-5.4-2026-03-05: $0.122 (14%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter10_deadline_guard**
1. semantic_222: $0.099 (gpt-5.4-mini $0.091, gpt-5.4-2026-03-05 $0.008)
2. semantic_174: $0.096 (gpt-5.4-mini $0.088, gpt-5.4-2026-03-05 $0.008)
3. semantic_57: $0.096 (gpt-5.4-mini $0.089, gpt-5.4-2026-03-05 $0.007)
4. semantic_214: $0.075 (gpt-5.4-mini $0.067, gpt-5.4-2026-03-05 $0.008)
5. semantic_123: $0.075 (gpt-5.4-mini $0.066, gpt-5.4-2026-03-05 $0.009)

**iter12_salvage_rank**
1. semantic_57: $0.092 (gpt-5.4-mini $0.084, gpt-5.4-2026-03-05 $0.008)
2. semantic_174: $0.086 (gpt-5.4-mini $0.078, gpt-5.4-2026-03-05 $0.008)
3. semantic_77: $0.079 (gpt-5.4-mini $0.070, gpt-5.4-2026-03-05 $0.009)
4. semantic_226: $0.075 (gpt-5.4-mini $0.068, gpt-5.4-2026-03-05 $0.007)
5. semantic_222: $0.069 (gpt-5.4-mini $0.060, gpt-5.4-2026-03-05 $0.009)

**iter13_any_author_gate**
1. semantic_226: $0.089 (gpt-5.4-mini $0.082, gpt-5.4-2026-03-05 $0.008)
2. semantic_222: $0.087 (gpt-5.4-mini $0.079, gpt-5.4-2026-03-05 $0.008)
3. semantic_57: $0.087 (gpt-5.4-mini $0.078, gpt-5.4-2026-03-05 $0.008)
4. semantic_174: $0.084 (gpt-5.4-mini $0.076, gpt-5.4-2026-03-05 $0.008)
5. semantic_137: $0.078 (gpt-5.4-mini $0.069, gpt-5.4-2026-03-05 $0.008)
