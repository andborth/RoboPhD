# Cost Analysis - Iteration 5

**Total Evaluation Cost: $2.13**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter2_exec_verified_sonnet | $0.27 | 5/20 | **$0.27** | $0.013 |
| iter3_ensemble_exec_vote | $0.90 | 5/20 | **$0.90** | $0.045 |
| iter5_ensemble_exec_vote_dtype_judge | $0.96 | - | **$0.96** | $0.048 |
| **Total** | **$2.13** | **10/40** | **$2.13** | **$0.035** |

*Avg/Problem is Eval Cost divided by problems tested — the same agent-only basis the cost penalty uses. Cache does not affect this calculation.*

---

## Cost by Model

**iter2_exec_verified_sonnet** ($0.269 total)
- anthropic/claude-sonnet-4-6: $0.269 (100%)

**iter3_ensemble_exec_vote** ($0.900 total)
- anthropic/claude-opus-4-8: $0.433 (48%)
- openai/gpt-5.4-2026-03-05: $0.247 (27%)
- google/gemini-3.1-pro-preview: $0.220 (24%)

**iter5_ensemble_exec_vote_dtype_judge** ($0.960 total)
- anthropic/claude-opus-4-8: $0.387 (40%)
- openai/gpt-5.4-2026-03-05: $0.300 (31%)
- google/gemini-3.1-pro-preview: $0.221 (23%)
- anthropic/claude-sonnet-4-6: $0.051 (5%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter2_exec_verified_sonnet**
1. 10: $0.029
2. 662: $0.028
3. 836: $0.017
4. 919: $0.015
5. 962: $0.015

**iter3_ensemble_exec_vote**
1. 662: $0.094 (claude-opus-4-8 $0.055, gpt-5.4-2026-03-05 $0.023, gemini-3.1-pro-preview $0.017)
2. 918: $0.087 (gpt-5.4-2026-03-05 $0.040, claude-opus-4-8 $0.037, gemini-3.1-pro-preview $0.010)
3. 962: $0.078 (claude-opus-4-8 $0.054, gpt-5.4-2026-03-05 $0.012, gemini-3.1-pro-preview $0.012)
4. 961: $0.077 (claude-opus-4-8 $0.054, gemini-3.1-pro-preview $0.014, gpt-5.4-2026-03-05 $0.008)
5. 812: $0.058 (gpt-5.4-2026-03-05 $0.032, claude-opus-4-8 $0.013, gemini-3.1-pro-preview $0.013)

**iter5_ensemble_exec_vote_dtype_judge**
1. 961: $0.094 (claude-opus-4-8 $0.056, gemini-3.1-pro-preview $0.029, gpt-5.4-2026-03-05 $0.009)
2. 962: $0.083 (claude-opus-4-8 $0.055, gemini-3.1-pro-preview $0.017, gpt-5.4-2026-03-05 $0.012)
3. 662: $0.077 (gpt-5.4-2026-03-05 $0.054, claude-opus-4-8 $0.015, claude-sonnet-4-6 $0.008)
4. 10: $0.069 (gpt-5.4-2026-03-05 $0.025, gemini-3.1-pro-preview $0.022, claude-opus-4-8 $0.017, claude-sonnet-4-6 $0.005)
5. 918: $0.067 (gpt-5.4-2026-03-05 $0.042, claude-opus-4-8 $0.015, gemini-3.1-pro-preview $0.010)
