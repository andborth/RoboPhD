# Cost Analysis - Iteration 20

**Total Evaluation Cost: $10.35** (+ Other $0.70 = $11.04 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter19_deep_screen_bulk_evidence | $3.33 | $0.23 | 1/14 | **$3.55** | $0.238 |
| iter20_prf_coverage_expansion | $4.09 | $0.23 | - | **$4.32** | $0.292 |
| iter9_bulk_passage_harvest | $2.94 | $0.23 | 12/14 | **$3.17** | $0.210 |
| **Total** | **$10.35** | **$0.70** | **13/28** | **$11.04** | **$0.246** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter19_deep_screen_bulk_evidence** ($3.325 total)
- openai/gpt-5.4-mini: $1.646 (50%)
- anthropic/claude-sonnet-4-6: $1.570 (47%)
- openai/gpt-5.4-2026-03-05: $0.110 (3%)

**iter20_prf_coverage_expansion** ($4.088 total)
- openai/gpt-5.4-mini: $2.416 (59%)
- anthropic/claude-sonnet-4-6: $1.556 (38%)
- openai/gpt-5.4-2026-03-05: $0.116 (3%)

**iter9_bulk_passage_harvest** ($2.935 total)
- anthropic/claude-sonnet-4-6: $1.558 (53%)
- openai/gpt-5.4-mini: $1.272 (43%)
- openai/gpt-5.4-2026-03-05: $0.106 (4%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter19_deep_screen_bulk_evidence**
1. semantic_77: $0.386 (claude-sonnet-4-6 $0.193, gpt-5.4-mini $0.185, gpt-5.4-2026-03-05 $0.008)
2. semantic_110: $0.370 (gpt-5.4-mini $0.183, claude-sonnet-4-6 $0.179, gpt-5.4-2026-03-05 $0.007)
3. semantic_222: $0.370 (gpt-5.4-mini $0.182, claude-sonnet-4-6 $0.180, gpt-5.4-2026-03-05 $0.008)
4. semantic_189: $0.368 (claude-sonnet-4-6 $0.182, gpt-5.4-mini $0.178, gpt-5.4-2026-03-05 $0.008)
5. semantic_22: $0.362 (gpt-5.4-mini $0.184, claude-sonnet-4-6 $0.170, gpt-5.4-2026-03-05 $0.008)

**iter20_prf_coverage_expansion**
1. semantic_110: $0.486 (gpt-5.4-mini $0.293, claude-sonnet-4-6 $0.184, gpt-5.4-2026-03-05 $0.009)
2. semantic_222: $0.478 (gpt-5.4-mini $0.293, claude-sonnet-4-6 $0.177, gpt-5.4-2026-03-05 $0.009)
3. semantic_189: $0.453 (gpt-5.4-mini $0.262, claude-sonnet-4-6 $0.183, gpt-5.4-2026-03-05 $0.008)
4. semantic_22: $0.443 (gpt-5.4-mini $0.269, claude-sonnet-4-6 $0.166, gpt-5.4-2026-03-05 $0.008)
5. semantic_77: $0.438 (gpt-5.4-mini $0.241, claude-sonnet-4-6 $0.188, gpt-5.4-2026-03-05 $0.008)

**iter9_bulk_passage_harvest**
1. semantic_77: $0.355 (claude-sonnet-4-6 $0.196, gpt-5.4-mini $0.150, gpt-5.4-2026-03-05 $0.008)
2. semantic_222: $0.333 (claude-sonnet-4-6 $0.180, gpt-5.4-mini $0.144, gpt-5.4-2026-03-05 $0.008)
3. semantic_110: $0.327 (claude-sonnet-4-6 $0.179, gpt-5.4-mini $0.141, gpt-5.4-2026-03-05 $0.006)
4. semantic_205: $0.323 (claude-sonnet-4-6 $0.172, gpt-5.4-mini $0.143, gpt-5.4-2026-03-05 $0.008)
5. semantic_189: $0.317 (claude-sonnet-4-6 $0.172, gpt-5.4-mini $0.138, gpt-5.4-2026-03-05 $0.008)
