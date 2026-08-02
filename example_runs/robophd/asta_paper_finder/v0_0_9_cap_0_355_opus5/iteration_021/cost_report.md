# Cost Analysis - Iteration 21

**Total Evaluation Cost: $9.68** (+ Other $0.81 = $10.48 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter19_deep_screen_bulk_evidence | $3.48 | $0.27 | 5/14 | **$3.75** | $0.248 |
| iter21_gold_rubric_and_hard_predicates | $3.65 | $0.27 | - | **$3.92** | $0.260 |
| iter8_criterion_window_evidence | $2.55 | $0.26 | 12/14 | **$2.82** | $0.182 |
| **Total** | **$9.68** | **$0.81** | **17/28** | **$10.48** | **$0.230** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter19_deep_screen_bulk_evidence** ($3.477 total)
- openai/gpt-5.4-mini: $1.710 (49%)
- anthropic/claude-sonnet-4-6: $1.659 (48%)
- openai/gpt-5.4-2026-03-05: $0.107 (3%)

**iter21_gold_rubric_and_hard_predicates** ($3.646 total)
- openai/gpt-5.4-mini: $1.783 (49%)
- anthropic/claude-sonnet-4-6: $1.713 (47%)
- openai/gpt-5.4-2026-03-05: $0.150 (4%)

**iter8_criterion_window_evidence** ($2.552 total)
- anthropic/claude-sonnet-4-6: $1.611 (63%)
- openai/gpt-5.4-mini: $0.833 (33%)
- openai/gpt-5.4-2026-03-05: $0.108 (4%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter19_deep_screen_bulk_evidence**
1. semantic_222: $0.370 (gpt-5.4-mini $0.182, claude-sonnet-4-6 $0.180, gpt-5.4-2026-03-05 $0.008)
2. semantic_189: $0.368 (claude-sonnet-4-6 $0.182, gpt-5.4-mini $0.178, gpt-5.4-2026-03-05 $0.008)
3. semantic_123: $0.358 (gpt-5.4-mini $0.180, claude-sonnet-4-6 $0.171, gpt-5.4-2026-03-05 $0.008)
4. semantic_170: $0.348 (gpt-5.4-mini $0.175, claude-sonnet-4-6 $0.166, gpt-5.4-2026-03-05 $0.007)
5. semantic_57: $0.348 (gpt-5.4-mini $0.175, claude-sonnet-4-6 $0.166, gpt-5.4-2026-03-05 $0.007)

**iter21_gold_rubric_and_hard_predicates**
1. semantic_155: $0.379 (claude-sonnet-4-6 $0.185, gpt-5.4-mini $0.183, gpt-5.4-2026-03-05 $0.010)
2. semantic_189: $0.371 (gpt-5.4-mini $0.181, claude-sonnet-4-6 $0.179, gpt-5.4-2026-03-05 $0.010)
3. semantic_57: $0.367 (gpt-5.4-mini $0.178, claude-sonnet-4-6 $0.178, gpt-5.4-2026-03-05 $0.010)
4. semantic_170: $0.366 (gpt-5.4-mini $0.180, claude-sonnet-4-6 $0.176, gpt-5.4-2026-03-05 $0.010)
5. semantic_220: $0.365 (gpt-5.4-mini $0.184, claude-sonnet-4-6 $0.171, gpt-5.4-2026-03-05 $0.011)

**iter8_criterion_window_evidence**
1. semantic_222: $0.279 (claude-sonnet-4-6 $0.182, gpt-5.4-mini $0.088, gpt-5.4-2026-03-05 $0.008)
2. semantic_123: $0.266 (claude-sonnet-4-6 $0.170, gpt-5.4-mini $0.089, gpt-5.4-2026-03-05 $0.008)
3. semantic_140: $0.255 (claude-sonnet-4-6 $0.162, gpt-5.4-mini $0.086, gpt-5.4-2026-03-05 $0.007)
4. semantic_155: $0.254 (claude-sonnet-4-6 $0.164, gpt-5.4-mini $0.084, gpt-5.4-2026-03-05 $0.007)
5. semantic_57: $0.253 (claude-sonnet-4-6 $0.165, gpt-5.4-mini $0.081, gpt-5.4-2026-03-05 $0.006)
