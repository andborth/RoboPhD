---
name: iter5_reflection_refined_critic
description: Reflection-driven refinement combining iter2's proven analysis with calibrated eval instructions addressing TLE rationalization and logic bug gaps
execution_mode: tool_only
tool_command: python tools/problem_analyzer.py
tool_output_file: tool_output/critic_feedback.txt
---

# Reflection-Refined Problem Analyzer (Iteration 5)

This critic agent uses deterministic tool-only execution to provide comprehensive code analysis for competitive programming solutions.

## Evolution History

This agent was created through reflection-driven refinement, systematically incorporating learnings from iterations 2-4:

### Base: iter2_deep_analysis_critic (65% accuracy, ELO 1556)
- Proven AST-based complexity analysis
- Reliable constraint extraction
- Conservative sample test parsing
- Comprehensive pattern detection

### Key Improvements from Reflection Analysis

**From iter2 reflection**: Added stronger logic bug detection guidance in eval_instructions
**From iter3 reflection**: Preserved conservative "Could not parse" behavior for uncertain samples
**From iter4 reflection**: Added explicit anti-rationalization rules to prevent TLE false positives

## Analysis Components

The tool provides structured analysis in six sections:

### 1. COMPLEXITY ANALYSIS
- Maximum loop nesting depth
- Loop structure with line numbers and bounds
- Potential performance issues (expensive ops in loops)

### 2. CONSTRAINT & TLE RISK ASSESSMENT
- Extracted constraints from problem/reflection
- Operation count estimation
- Risk level: CRITICAL / HIGH / MODERATE / LOW
- Warning when constraints cannot be extracted but nested loops detected

### 3. CODE PATTERN ANALYSIS
- All detected patterns (no filtering)
- Recursion without memoization
- Missing recursion limit
- Expensive operations in loops
- Collection modification during iteration

### 4. REFLECTION VALIDATION
- Complexity claim mismatches
- Overconfident phrases detected
- Efficiency claims vs actual loop depth

### 5. SAMPLE TEST EXECUTION
- Conservative parsing (fails safe on uncertain formats)
- Whitespace-tolerant comparison
- PASS / WRONG_ANSWER / TIMEOUT / ERROR status

### 6. CRITICAL ISSUES SUMMARY
- Synthesized list of blocking issues
- Guidance for the critic model

## Execution Details

**Tool command**: `python tools/problem_analyzer.py`
**Output file**: `tool_output/critic_feedback.txt`
**Timeout**: 600 seconds

The tool reads:
- `solution.py` - The code to analyze
- `reflection.md` - The coder's self-assessment
- `problem.md` - The problem statement with constraints

## Error Recovery

If the tool fails (non-zero exit, missing output, output < 200 bytes):
1. The system clears `tool_output/` directory
2. Falls back to calling this agent with LLM
3. Agent should examine available files and provide analysis

For manual fallback analysis:
1. Read solution.py and identify loop structure
2. Read problem.md for constraints
3. Estimate complexity and TLE risk
4. Check reflection.md for overconfident claims
5. Write analysis to `./output/agent_output.txt`
