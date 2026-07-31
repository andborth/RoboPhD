# Cost Analysis - Iteration 14

**Total Evaluation Cost: $1.71** (+ Other $3.52 = $5.23 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter13_balanced_digest_wide_vetting | $0.66 | $1.16 | 3/14 | **$1.82** | $0.047 |
| iter14_conjunctive_vote | $0.66 | $1.05 | - | **$1.71** | $0.047 |
| iter7_criterion_repair | $0.39 | $1.32 | 13/14 | **$1.70** | $0.028 |
| **Total** | **$1.71** | **$3.52** | **16/28** | **$5.23** | **$0.041** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter13_balanced_digest_wide_vetting** ($0.659 total)
- openai/gpt-5.4-mini: $0.254 (39%)
- openai/gpt-5.4-2026-03-05: $0.215 (33%)
- anthropic/claude-haiku-4-5-20251001: $0.184 (28%)
- anthropic/claude-sonnet-4-6: $0.006 (1%)

**iter14_conjunctive_vote** ($0.663 total)
- openai/gpt-5.4-mini: $0.255 (39%)
- openai/gpt-5.4-2026-03-05: $0.215 (32%)
- anthropic/claude-haiku-4-5-20251001: $0.185 (28%)
- anthropic/claude-sonnet-4-6: $0.007 (1%)

**iter7_criterion_repair** ($0.389 total)
- openai/gpt-5.4-2026-03-05: $0.217 (56%)
- openai/gpt-5.4-mini: $0.172 (44%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter13_balanced_digest_wide_vetting**
1. semantic_110: $0.075 (gpt-5.4-mini $0.030, gpt-5.4-2026-03-05 $0.023, claude-haiku-4-5-20251001 $0.022)
2. semantic_7: $0.072 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.021, claude-haiku-4-5-20251001 $0.021)
3. semantic_57: $0.072 (gpt-5.4-mini $0.029, claude-haiku-4-5-20251001 $0.021, gpt-5.4-2026-03-05 $0.021)
4. semantic_172: $0.071 (gpt-5.4-mini $0.028, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.021)
5. semantic_193: $0.070 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.021, claude-haiku-4-5-20251001 $0.021)

**iter14_conjunctive_vote**
1. semantic_110: $0.073 (gpt-5.4-mini $0.030, claude-haiku-4-5-20251001 $0.022, gpt-5.4-2026-03-05 $0.022)
2. semantic_57: $0.073 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.022)
3. semantic_7: $0.072 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.021)
4. semantic_148: $0.071 (gpt-5.4-mini $0.028, gpt-5.4-2026-03-05 $0.021, claude-haiku-4-5-20251001 $0.021)
5. semantic_193: $0.070 (gpt-5.4-mini $0.029, claude-haiku-4-5-20251001 $0.021, gpt-5.4-2026-03-05 $0.021)

**iter7_criterion_repair**
1. semantic_110: $0.046 (gpt-5.4-2026-03-05 $0.025, gpt-5.4-mini $0.021)
2. semantic_7: $0.045 (gpt-5.4-2026-03-05 $0.023, gpt-5.4-mini $0.021)
3. semantic_170: $0.044 (gpt-5.4-2026-03-05 $0.024, gpt-5.4-mini $0.020)
4. semantic_148: $0.044 (gpt-5.4-2026-03-05 $0.023, gpt-5.4-mini $0.020)
5. semantic_172: $0.044 (gpt-5.4-2026-03-05 $0.023, gpt-5.4-mini $0.020)
