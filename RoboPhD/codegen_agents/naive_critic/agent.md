---
name: naive_critic
description: Baseline critic that provides generic code review
execution_mode: tool_only
tool_command: python tools/analyze_problem.py
tool_output_file: tool_output/analysis.txt
---

# Naive Code Critic (Tool-Only)

This agent uses tool-only execution to analyze the problem context.

## Process

1. **Run Analysis Tool**
   ```bash
   python tools/analyze_problem.py
   ```

2. **Read and Output Results**
   - Read the generated analysis from `tool_output/analysis.txt`
   - Write the complete output to `./output/agent_output.txt`

## Error Recovery

If the tool fails:

1. Check `reflection.md` exists
2. Verify Python environment
3. Examine any error messages in `tool_output/`
4. Fall back to reading reflection.md directly and outputting the categories section
