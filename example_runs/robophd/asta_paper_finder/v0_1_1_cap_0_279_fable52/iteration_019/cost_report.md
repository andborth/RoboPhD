# Cost Analysis - Iteration 19

**Total Evaluation Cost: $10.03** (+ Other $0.52 = $10.55 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter13_citation_snowball | $3.27 | $0.17 | 6/14 | **$3.45** | $0.234 |
| iter18_stage1_keyed_order | $3.37 | $0.17 | 4/14 | **$3.55** | $0.241 |
| iter19_backdated_citation_counts | $3.39 | $0.18 | - | **$3.56** | $0.242 |
| **Total** | **$10.03** | **$0.52** | **10/28** | **$10.55** | **$0.239** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter13_citation_snowball** ($3.274 total)
- openai/gpt-5.4-2026-03-05: $1.434 (44%)
- openai/gpt-5.4-mini: $1.039 (32%)
- anthropic/claude-opus-4-8: $0.799 (24%)
- anthropic/claude-sonnet-4-6: $0.001 (0%)

**iter18_stage1_keyed_order** ($3.374 total)
- openai/gpt-5.4-mini: $1.209 (36%)
- openai/gpt-5.4-2026-03-05: $1.131 (34%)
- anthropic/claude-opus-4-8: $0.776 (23%)
- openai/gpt-5.5: $0.212 (6%)
- google/gemini-3.1-pro-preview: $0.044 (1%)
- anthropic/claude-sonnet-4-6: $0.001 (0%)

**iter19_backdated_citation_counts** ($3.387 total)
- openai/gpt-5.4-2026-03-05: $1.232 (36%)
- openai/gpt-5.4-mini: $1.187 (35%)
- anthropic/claude-opus-4-8: $0.776 (23%)
- openai/gpt-5.5: $0.166 (5%)
- google/gemini-3.1-pro-preview: $0.024 (1%)
- anthropic/claude-sonnet-4-6: $0.001 (0%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter13_citation_snowball**
1. semantic_152: $0.403 (gpt-5.4-2026-03-05 $0.200, gpt-5.4-mini $0.148, claude-opus-4-8 $0.055)
2. semantic_8: $0.322 (gpt-5.4-2026-03-05 $0.159, gpt-5.4-mini $0.110, claude-opus-4-8 $0.053)
3. semantic_226: $0.304 (gpt-5.4-2026-03-05 $0.156, gpt-5.4-mini $0.105, claude-opus-4-8 $0.042)
4. semantic_203: $0.297 (gpt-5.4-2026-03-05 $0.159, gpt-5.4-mini $0.091, claude-opus-4-8 $0.047)
5. semantic_145: $0.296 (gpt-5.4-2026-03-05 $0.148, gpt-5.4-mini $0.111, claude-opus-4-8 $0.037)

**iter18_stage1_keyed_order**
1. semantic_152: $0.352 (gpt-5.4-2026-03-05 $0.152, gpt-5.4-mini $0.145, claude-opus-4-8 $0.054)
2. semantic_120: $0.332 (gpt-5.4-2026-03-05 $0.148, gpt-5.4-mini $0.133, claude-opus-4-8 $0.051)
3. semantic_220: $0.326 (gpt-5.4-2026-03-05 $0.142, gpt-5.4-mini $0.138, claude-opus-4-8 $0.045)
4. semantic_226: $0.313 (gpt-5.4-2026-03-05 $0.142, gpt-5.4-mini $0.132, claude-opus-4-8 $0.039)
5. semantic_8: $0.300 (gpt-5.4-mini $0.129, gpt-5.4-2026-03-05 $0.120, claude-opus-4-8 $0.051)

**iter19_backdated_citation_counts**
1. semantic_152: $0.334 (gpt-5.4-2026-03-05 $0.145, gpt-5.4-mini $0.139, claude-opus-4-8 $0.050)
2. semantic_220: $0.318 (gpt-5.4-2026-03-05 $0.139, gpt-5.4-mini $0.137, claude-opus-4-8 $0.043)
3. semantic_120: $0.306 (gpt-5.4-2026-03-05 $0.138, gpt-5.4-mini $0.113, claude-opus-4-8 $0.054)
4. semantic_145: $0.300 (gpt-5.4-2026-03-05 $0.149, gpt-5.4-mini $0.114, claude-opus-4-8 $0.037)
5. semantic_226: $0.295 (gpt-5.4-2026-03-05 $0.144, gpt-5.4-mini $0.112, claude-opus-4-8 $0.040)
