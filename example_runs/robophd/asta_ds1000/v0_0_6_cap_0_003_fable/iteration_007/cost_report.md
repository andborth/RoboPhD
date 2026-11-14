# Cost Analysis - Iteration 7

**Total Evaluation Cost: $0.19**

**Agents Tested**: 3 agents
**Problems Tested**: 20 problems
**Total Tests**: 60 (agent x problem pairs)

## Agent Cost Summary

| Agent | Eval Cost | Cached | Total | Avg/Problem |
|-----|---------|------|-----|-----------|
| iter2_exec_verify_ensemble | $0.07 | 16/20 | **$0.07** | $0.004 |
| iter6_audited_cascade | $0.07 | 5/20 | **$0.07** | $0.004 |
| iter7_lean_audited_cascade | $0.05 | - | **$0.05** | $0.002 |
| **Total** | **$0.19** | **21/40** | **$0.19** | **$0.003** |

*Avg/Problem is total cost divided by problems tested. Cache does not affect this calculation.*

---

## Cost by Model

**iter2_exec_verify_ensemble** ($0.071 total)
- openai/gpt-5.4-2026-03-05: $0.028 (40%)
- anthropic/claude-haiku-4-5-20251001: $0.027 (38%)
- openai/gpt-5.4-mini: $0.016 (22%)

**iter6_audited_cascade** ($0.073 total)
- openai/gpt-5.4-mini: $0.035 (48%)
- openai/gpt-5.4-2026-03-05: $0.017 (24%)
- anthropic/claude-haiku-4-5-20251001: $0.011 (15%)
- google/gemini-3.1-flash-lite: $0.010 (13%)

**iter7_lean_audited_cascade** ($0.046 total)
- openai/gpt-5.4-mini: $0.025 (54%)
- google/gemini-3.1-flash-lite: $0.010 (21%)
- anthropic/claude-haiku-4-5-20251001: $0.007 (16%)
- openai/gpt-5.4-2026-03-05: $0.004 (9%)


---

## Cost Insights

### Top 5 Most Expensive Tasks per Agent

**iter2_exec_verify_ensemble**
1. 238: $0.013 (gpt-5.4-2026-03-05 $0.008, claude-haiku-4-5-20251001 $0.004, gpt-5.4-mini $0.001)
2. 185: $0.009 (gpt-5.4-2026-03-05 $0.006, claude-haiku-4-5-20251001 $0.002, gpt-5.4-mini $0.001)
3. 906: $0.009 (gpt-5.4-2026-03-05 $0.005, gpt-5.4-mini $0.002, claude-haiku-4-5-20251001 $0.002)
4. 838: $0.008 (gpt-5.4-2026-03-05 $0.004, claude-haiku-4-5-20251001 $0.003, gpt-5.4-mini $0.001)
5. 761: $0.004 (gpt-5.4-2026-03-05 $0.003, claude-haiku-4-5-20251001 $0.001, gpt-5.4-mini $0.001)

**iter6_audited_cascade**
1. 446: $0.011 (gpt-5.4-2026-03-05 $0.006, gpt-5.4-mini $0.002, claude-haiku-4-5-20251001 $0.002, gemini-3.1-flash-lite $0.001)
2. 838: $0.008 (gpt-5.4-2026-03-05 $0.004, gpt-5.4-mini $0.003, gemini-3.1-flash-lite $0.001)
3. 238: $0.008 (gpt-5.4-mini $0.004, claude-haiku-4-5-20251001 $0.003, gemini-3.1-flash-lite $0.001)
4. 761: $0.006 (gpt-5.4-2026-03-05 $0.003, gpt-5.4-mini $0.002, claude-haiku-4-5-20251001 $0.001, gemini-3.1-flash-lite $0.000)
5. 445: $0.006 (gpt-5.4-2026-03-05 $0.002, gpt-5.4-mini $0.002, claude-haiku-4-5-20251001 $0.001, gemini-3.1-flash-lite $0.000)

**iter7_lean_audited_cascade**
1. 838: $0.006 (gpt-5.4-mini $0.003, gpt-5.4-2026-03-05 $0.003, gemini-3.1-flash-lite $0.000)
2. 238: $0.005 (claude-haiku-4-5-20251001 $0.003, gpt-5.4-mini $0.002, gemini-3.1-flash-lite $0.001)
3. 446: $0.004 (gemini-3.1-flash-lite $0.002, claude-haiku-4-5-20251001 $0.001, gpt-5.4-mini $0.001)
4. 906: $0.004 (gpt-5.4-mini $0.003, gemini-3.1-flash-lite $0.001)
5. 803: $0.004 (gpt-5.4-mini $0.002, claude-haiku-4-5-20251001 $0.001, gemini-3.1-flash-lite $0.000)
