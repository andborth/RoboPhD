# Cost Analysis - Iteration 14

**Total Evaluation Cost: $2.23** (+ Other $0.93 = $3.16 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter13_any_author_gate | $0.80 | $0.32 | 1/14 | **$1.12** | $0.057 |
| iter14_title_channel | $0.84 | $0.31 | - | **$1.15** | $0.060 |
| iter6_graph_recall | $0.59 | $0.30 | 10/14 | **$0.89** | $0.042 |
| **Total** | **$2.23** | **$0.93** | **11/28** | **$3.16** | **$0.053** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter13_any_author_gate** ($0.800 total)
- openai/gpt-5.4-mini: $0.692 (87%)
- openai/gpt-5.4-2026-03-05: $0.108 (13%)

**iter14_title_channel** ($0.838 total)
- openai/gpt-5.4-mini: $0.722 (86%)
- openai/gpt-5.4-2026-03-05: $0.116 (14%)

**iter6_graph_recall** ($0.588 total)
- openai/gpt-5.4-mini: $0.489 (83%)
- openai/gpt-5.4-2026-03-05: $0.099 (17%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter13_any_author_gate**
1. semantic_7: $0.094 (gpt-5.4-mini $0.086, gpt-5.4-2026-03-05 $0.008)
2. semantic_148: $0.092 (gpt-5.4-mini $0.084, gpt-5.4-2026-03-05 $0.009)
3. semantic_33: $0.083 (gpt-5.4-mini $0.076, gpt-5.4-2026-03-05 $0.007)
4. semantic_110: $0.076 (gpt-5.4-mini $0.068, gpt-5.4-2026-03-05 $0.009)
5. semantic_224: $0.072 (gpt-5.4-mini $0.064, gpt-5.4-2026-03-05 $0.008)

**iter14_title_channel**
1. semantic_33: $0.093 (gpt-5.4-mini $0.084, gpt-5.4-2026-03-05 $0.009)
2. semantic_148: $0.092 (gpt-5.4-mini $0.082, gpt-5.4-2026-03-05 $0.010)
3. semantic_7: $0.086 (gpt-5.4-mini $0.078, gpt-5.4-2026-03-05 $0.008)
4. semantic_224: $0.072 (gpt-5.4-mini $0.064, gpt-5.4-2026-03-05 $0.008)
5. semantic_170: $0.072 (gpt-5.4-mini $0.065, gpt-5.4-2026-03-05 $0.007)

**iter6_graph_recall**
1. semantic_170: $0.060 (gpt-5.4-mini $0.054, gpt-5.4-2026-03-05 $0.007)
2. semantic_110: $0.053 (gpt-5.4-mini $0.045, gpt-5.4-2026-03-05 $0.008)
3. semantic_148: $0.052 (gpt-5.4-mini $0.044, gpt-5.4-2026-03-05 $0.008)
4. semantic_104: $0.052 (gpt-5.4-mini $0.044, gpt-5.4-2026-03-05 $0.008)
5. semantic_8: $0.052 (gpt-5.4-mini $0.044, gpt-5.4-2026-03-05 $0.008)
