# Cost Analysis - Iteration 15

**Total Evaluation Cost: $1.88** (+ Other $4.03 = $5.90 grand total)

**Agents Tested**: 3 agents
**Problems Tested**: 14 problems
**Total Tests**: 42 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Other | Cached | Total | Avg/Problem |
|-----|---------|-----|------|-----|-----------|
| iter11_ensemble_conjunctive_rank | $0.54 | $1.39 | 7/14 | **$1.93** | $0.039 |
| iter13_balanced_digest_wide_vetting | $0.67 | $1.43 | 5/14 | **$2.10** | $0.048 |
| iter15_verdict_repair | $0.66 | $1.21 | - | **$1.87** | $0.047 |
| **Total** | **$1.88** | **$4.03** | **12/28** | **$5.90** | **$0.045** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses (Other is excluded: it is outside the agent's control and never penalized). Cache does not affect this calculation.*

---

## Cost by Model

**iter11_ensemble_conjunctive_rank** ($0.543 total)
- openai/gpt-5.4-mini: $0.271 (50%)
- openai/gpt-5.4-2026-03-05: $0.156 (29%)
- anthropic/claude-haiku-4-5-20251001: $0.109 (20%)
- anthropic/claude-sonnet-4-6: $0.007 (1%)

**iter13_balanced_digest_wide_vetting** ($0.670 total)
- openai/gpt-5.4-mini: $0.257 (38%)
- openai/gpt-5.4-2026-03-05: $0.221 (33%)
- anthropic/claude-haiku-4-5-20251001: $0.187 (28%)
- anthropic/claude-sonnet-4-6: $0.006 (1%)

**iter15_verdict_repair** ($0.663 total)
- openai/gpt-5.4-mini: $0.255 (39%)
- openai/gpt-5.4-2026-03-05: $0.216 (33%)
- anthropic/claude-haiku-4-5-20251001: $0.185 (28%)
- anthropic/claude-sonnet-4-6: $0.006 (1%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter11_ensemble_conjunctive_rank**
1. semantic_214: $0.060 (gpt-5.4-mini $0.032, gpt-5.4-2026-03-05 $0.015, claude-haiku-4-5-20251001 $0.013)
2. semantic_110: $0.060 (gpt-5.4-mini $0.031, gpt-5.4-2026-03-05 $0.016, claude-haiku-4-5-20251001 $0.013)
3. semantic_205: $0.058 (gpt-5.4-mini $0.030, gpt-5.4-2026-03-05 $0.016, claude-haiku-4-5-20251001 $0.013)
4. semantic_189: $0.058 (gpt-5.4-mini $0.031, gpt-5.4-2026-03-05 $0.015, claude-haiku-4-5-20251001 $0.012)
5. semantic_2: $0.057 (gpt-5.4-mini $0.030, gpt-5.4-2026-03-05 $0.014, claude-haiku-4-5-20251001 $0.012)

**iter13_balanced_digest_wide_vetting**
1. semantic_214: $0.075 (gpt-5.4-mini $0.030, claude-haiku-4-5-20251001 $0.022, gpt-5.4-2026-03-05 $0.022)
2. semantic_110: $0.075 (gpt-5.4-mini $0.030, gpt-5.4-2026-03-05 $0.023, claude-haiku-4-5-20251001 $0.022)
3. semantic_189: $0.072 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.021)
4. semantic_205: $0.072 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.021)
5. semantic_172: $0.071 (gpt-5.4-mini $0.028, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.021)

**iter15_verdict_repair**
1. semantic_214: $0.075 (gpt-5.4-mini $0.030, gpt-5.4-2026-03-05 $0.023, claude-haiku-4-5-20251001 $0.022)
2. semantic_110: $0.074 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.022)
3. semantic_189: $0.072 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.021)
4. semantic_205: $0.072 (gpt-5.4-mini $0.029, gpt-5.4-2026-03-05 $0.022, claude-haiku-4-5-20251001 $0.021)
5. semantic_2: $0.071 (gpt-5.4-mini $0.028, claude-haiku-4-5-20251001 $0.021, gpt-5.4-2026-03-05 $0.021)
