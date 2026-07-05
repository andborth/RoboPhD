# Cost Analysis - Iteration 8

**Total Evaluation Cost: $0.21**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter2_exec_verify_ensemble | $0.09 | 10/20 | **$0.09** | $0.005 |
| iter7_lean_audited_cascade | $0.06 | 1/20 | **$0.06** | $0.003 |
| iter8_expected_diff_cascade | $0.06 | - | **$0.06** | $0.003 |
| **Total** | **$0.21** | **11/40** | **$0.21** | **$0.004** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter2_exec_verify_ensemble** ($0.092 total)
- openai/gpt-5.4-2026-03-05: $0.047 (51%)
- anthropic/claude-haiku-4-5-20251001: $0.027 (30%)
- openai/gpt-5.4-mini: $0.018 (19%)

**iter7_lean_audited_cascade** ($0.055 total)
- openai/gpt-5.4-mini: $0.027 (49%)
- openai/gpt-5.4-2026-03-05: $0.012 (22%)
- google/gemini-3.1-flash-lite: $0.008 (15%)
- anthropic/claude-haiku-4-5-20251001: $0.008 (14%)

**iter8_expected_diff_cascade** ($0.064 total)
- openai/gpt-5.4-mini: $0.030 (47%)
- openai/gpt-5.4-2026-03-05: $0.013 (20%)
- google/gemini-3.1-flash-lite: $0.011 (17%)
- anthropic/claude-haiku-4-5-20251001: $0.010 (16%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter2_exec_verify_ensemble**
1. 10: $0.014 (gpt-5.4-2026-03-05 $0.007, claude-haiku-4-5-20251001 $0.004, gpt-5.4-mini $0.003)
2. 662: $0.011 (gpt-5.4-2026-03-05 $0.006, claude-haiku-4-5-20251001 $0.002, gpt-5.4-mini $0.002)
3. 961: $0.008 (gpt-5.4-2026-03-05 $0.005, claude-haiku-4-5-20251001 $0.002, gpt-5.4-mini $0.001)
4. 919: $0.008 (gpt-5.4-2026-03-05 $0.005, claude-haiku-4-5-20251001 $0.002, gpt-5.4-mini $0.001)
5. 999: $0.008 (gpt-5.4-2026-03-05 $0.005, claude-haiku-4-5-20251001 $0.003, gpt-5.4-mini $0.001)

**iter7_lean_audited_cascade**
1. 662: $0.009 (gpt-5.4-2026-03-05 $0.005, gpt-5.4-mini $0.002, claude-haiku-4-5-20251001 $0.002, gemini-3.1-flash-lite $0.000)
2. 961: $0.008 (gpt-5.4-2026-03-05 $0.003, claude-haiku-4-5-20251001 $0.003, gpt-5.4-mini $0.001, gemini-3.1-flash-lite $0.001)
3. 919: $0.007 (gpt-5.4-mini $0.004, gpt-5.4-2026-03-05 $0.002, gemini-3.1-flash-lite $0.000)
4. 940: $0.003 (gpt-5.4-2026-03-05 $0.002, gpt-5.4-mini $0.002, gemini-3.1-flash-lite $0.000)
5. 10: $0.003 (gpt-5.4-mini $0.002, gemini-3.1-flash-lite $0.001)

**iter8_expected_diff_cascade**
1. 961: $0.007 (gpt-5.4-2026-03-05 $0.004, claude-haiku-4-5-20251001 $0.001, gpt-5.4-mini $0.001, gemini-3.1-flash-lite $0.001)
2. 999: $0.006 (gpt-5.4-2026-03-05 $0.003, claude-haiku-4-5-20251001 $0.001, gemini-3.1-flash-lite $0.001, gpt-5.4-mini $0.001)
3. 919: $0.006 (gpt-5.4-mini $0.003, gpt-5.4-2026-03-05 $0.003, gemini-3.1-flash-lite $0.001)
4. 812: $0.006 (gpt-5.4-2026-03-05 $0.002, claude-haiku-4-5-20251001 $0.002, gpt-5.4-mini $0.001, gemini-3.1-flash-lite $0.001)
5. 662: $0.005 (gpt-5.4-mini $0.003, claude-haiku-4-5-20251001 $0.002, gemini-3.1-flash-lite $0.001)
