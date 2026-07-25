# Cost Analysis - Iteration 9

**Total Evaluation Cost: $0.21** (+ Other $1.72 = $1.93 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter3_evidence_rerank_v1 | $0.04 | $0.82 | 8/14 | **$0.86** | $0.003 |
| iter6_metadata_targeted_v1 | $0.08 | $0.59 | 6/14 | **$0.67** | $0.006 |
| iter9_rerank_rich_v1 | $0.09 | $0.31 | - | **$0.40** | $0.006 |
| **Total** | **$0.21** | **$1.72** | **14/28** | **$1.93** | **$0.005** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter3_evidence_rerank_v1**
1. semantic_77: $0.004
2. semantic_43: $0.004
3. semantic_152: $0.004
4. semantic_229: $0.004
5. semantic_123: $0.004

**iter6_metadata_targeted_v1**
1. semantic_77: $0.006
2. metadata_14: $0.006
3. semantic_43: $0.006
4. semantic_152: $0.006
5. semantic_123: $0.006

**iter9_rerank_rich_v1**
1. semantic_203: $0.008
2. semantic_98: $0.008
3. semantic_77: $0.008
4. semantic_33: $0.008
5. semantic_229: $0.007
