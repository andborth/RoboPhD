# Cost Analysis - Iteration 20

**Total Evaluation Cost: $8.33** (+ Other $0.47 = $8.80 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter14_snowball_hydration | $2.59 | $0.15 | 9/14 | **$2.75** | $0.185 |
| iter18_stage1_keyed_order | $2.90 | $0.16 | 6/14 | **$3.06** | $0.207 |
| iter20_year_steered_expansion | $2.83 | $0.15 | - | **$2.99** | $0.202 |
| **Total** | **$8.33** | **$0.47** | **15/28** | **$8.80** | **$0.198** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter14_snowball_hydration** ($2.594 total)
- openai/gpt-5.4-mini: $0.950 (37%)
- openai/gpt-5.4-2026-03-05: $0.892 (34%)
- anthropic/claude-opus-4-8: $0.744 (29%)
- anthropic/claude-sonnet-4-6: $0.008 (0%)

**iter18_stage1_keyed_order** ($2.902 total)
- openai/gpt-5.4-mini: $1.062 (37%)
- openai/gpt-5.4-2026-03-05: $0.920 (32%)
- anthropic/claude-opus-4-8: $0.744 (26%)
- openai/gpt-5.5: $0.134 (5%)
- google/gemini-3.1-pro-preview: $0.034 (1%)
- anthropic/claude-sonnet-4-6: $0.009 (0%)

**iter20_year_steered_expansion** ($2.834 total)
- openai/gpt-5.4-mini: $0.948 (33%)
- openai/gpt-5.4-2026-03-05: $0.893 (32%)
- anthropic/claude-opus-4-8: $0.765 (27%)
- openai/gpt-5.5: $0.182 (6%)
- google/gemini-3.1-pro-preview: $0.037 (1%)
- anthropic/claude-sonnet-4-6: $0.009 (0%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter14_snowball_hydration**
1. semantic_193: $0.332 (gpt-5.4-2026-03-05 $0.181, gpt-5.4-mini $0.103, claude-opus-4-8 $0.048)
2. semantic_112: $0.323 (gpt-5.4-2026-03-05 $0.145, gpt-5.4-mini $0.132, claude-opus-4-8 $0.046)
3. semantic_77: $0.317 (gpt-5.4-2026-03-05 $0.144, gpt-5.4-mini $0.122, claude-opus-4-8 $0.051)
4. semantic_214: $0.241 (gpt-5.4-mini $0.119, gpt-5.4-2026-03-05 $0.076, claude-opus-4-8 $0.046)
5. semantic_219: $0.241 (gpt-5.4-mini $0.116, gpt-5.4-2026-03-05 $0.079, claude-opus-4-8 $0.046)

**iter18_stage1_keyed_order**
1. semantic_120: $0.332 (gpt-5.4-2026-03-05 $0.148, gpt-5.4-mini $0.133, claude-opus-4-8 $0.051)
2. semantic_214: $0.311 (gpt-5.4-2026-03-05 $0.143, gpt-5.4-mini $0.119, claude-opus-4-8 $0.049)
3. semantic_193: $0.297 (gpt-5.4-2026-03-05 $0.134, gpt-5.4-mini $0.122, claude-opus-4-8 $0.041)
4. semantic_57: $0.290 (gpt-5.4-2026-03-05 $0.120, gpt-5.4-mini $0.115, claude-opus-4-8 $0.055)
5. semantic_112: $0.281 (gpt-5.4-mini $0.131, gpt-5.4-2026-03-05 $0.103, claude-opus-4-8 $0.047)

**iter20_year_steered_expansion**
1. semantic_77: $0.340 (gpt-5.4-2026-03-05 $0.145, gpt-5.4-mini $0.144, claude-opus-4-8 $0.051)
2. semantic_7: $0.319 (gpt-5.4-2026-03-05 $0.149, gpt-5.4-mini $0.120, claude-opus-4-8 $0.050)
3. semantic_112: $0.299 (gpt-5.4-mini $0.133, gpt-5.4-2026-03-05 $0.117, claude-opus-4-8 $0.050)
4. semantic_219: $0.275 (gpt-5.4-mini $0.116, gpt-5.4-2026-03-05 $0.108, claude-opus-4-8 $0.051)
5. semantic_193: $0.256 (gpt-5.4-2026-03-05 $0.133, gpt-5.4-mini $0.071, claude-opus-4-8 $0.052)
