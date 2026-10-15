# Cost Analysis - Iteration 6

**Total Evaluation Cost: $2.34**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter2_exec_verified_sonnet | $0.25 | 11/20 | **$0.25** | $0.012 |
| iter3_ensemble_exec_vote | $0.79 | 7/20 | **$0.79** | $0.040 |
| iter6_ensemble_exec_vote_escalate | $1.30 | - | **$1.30** | $0.065 |
| **Total** | **$2.34** | **18/40** | **$2.34** | **$0.039** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses. Cache does not affect this calculation.*

---

## Cost by Model

**iter2_exec_verified_sonnet** ($0.247 total)
- anthropic/claude-sonnet-4-6: $0.247 (100%)

**iter3_ensemble_exec_vote** ($0.793 total)
- anthropic/claude-opus-4-8: $0.404 (51%)
- openai/gpt-5.4-2026-03-05: $0.195 (25%)
- google/gemini-3.1-pro-preview: $0.195 (25%)

**iter6_ensemble_exec_vote_escalate** ($1.304 total)
- anthropic/claude-opus-4-8: $0.575 (44%)
- openai/gpt-5.4-2026-03-05: $0.330 (25%)
- google/gemini-3.1-pro-preview: $0.206 (16%)
- openai/gpt-5.5: $0.106 (8%)
- anthropic/claude-sonnet-4-6: $0.086 (7%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter2_exec_verified_sonnet**
1. 238: $0.028
2. 906: $0.021
3. 763: $0.013
4. 124: $0.012
5. 999: $0.012

**iter3_ensemble_exec_vote**
1. 238: $0.102 (claude-opus-4-8 $0.053, gpt-5.4-2026-03-05 $0.025, gemini-3.1-pro-preview $0.024)
2. 999: $0.084 (claude-opus-4-8 $0.049, gpt-5.4-2026-03-05 $0.018, gemini-3.1-pro-preview $0.017)
3. 420: $0.072 (claude-opus-4-8 $0.031, gpt-5.4-2026-03-05 $0.026, gemini-3.1-pro-preview $0.015)
4. 883: $0.061 (claude-opus-4-8 $0.031, gpt-5.4-2026-03-05 $0.017, gemini-3.1-pro-preview $0.013)
5. 426: $0.051 (gpt-5.4-2026-03-05 $0.020, gemini-3.1-pro-preview $0.017, claude-opus-4-8 $0.014)

**iter6_ensemble_exec_vote_escalate**
1. 906: $0.153 (claude-opus-4-8 $0.082, gpt-5.4-2026-03-05 $0.057, gemini-3.1-pro-preview $0.015)
2. 420: $0.144 (gpt-5.5 $0.064, claude-opus-4-8 $0.034, gpt-5.4-2026-03-05 $0.021, gemini-3.1-pro-preview $0.019, claude-sonnet-4-6 $0.006)
3. 238: $0.129 (claude-opus-4-8 $0.063, gemini-3.1-pro-preview $0.023, claude-sonnet-4-6 $0.022, gpt-5.4-2026-03-05 $0.022)
4. 883: $0.097 (gpt-5.4-2026-03-05 $0.051, claude-opus-4-8 $0.034, gemini-3.1-pro-preview $0.013)
5. 804: $0.087 (claude-opus-4-8 $0.044, gpt-5.5 $0.019, gemini-3.1-pro-preview $0.009, gpt-5.4-2026-03-05 $0.008, claude-sonnet-4-6 $0.006)
