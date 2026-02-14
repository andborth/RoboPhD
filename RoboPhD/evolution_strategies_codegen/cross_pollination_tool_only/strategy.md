# Cross-Pollination Strategy for Code Generation Critics (Tool-Only Focus)

Having reviewed the files provided to you above, you are creating a new Claude Code critic agent primarily by combining successful elements from multiple existing critics using the three-artifact architecture.

**IMPORTANT: This strategy REQUIRES combining tool-only patterns** from multiple high-performing critics. Identify and merge the best tool-based techniques into a comprehensive, deterministic analysis system.

## Context
You're evolving a code generation critic system that consists of three distinct artifacts:
1. **eval_instructions.md** - Static code generation instructions passed directly to the coder model
2. **agent.md** - Problem analysis agent that examines coding problems and provides critic feedback
3. **tools/** - Python/shell scripts for problem analysis

## Domain: Code Generation with Coder/Critic Architecture

This is a code generation domain where:
- **Phase 1 Input**: Problem context (question, code_v1, approach_description)
- **Phase 1 Output**: Critic feedback to help improve the code
- **Phase 2**: Coder revises code based on critic feedback
- **Evaluation**: Binary pass/fail on hidden test cases

## Your Task
Create a new critic package by cross-pollinating successful **tool-only patterns** from the top-performing critics to achieve higher accuracy on problems you haven't seen yet.

**Note:** Although you are primarily using a cross-pollinating approach, you can use a new idea of your own if you think you see an opportunity.

## Tool-Only Cross-Pollination Strategy

When examining top-performing critics, look for:

### Identifying Techniques to Combine:
- Which code analysis techniques do different critics use effectively?
- Which bug-detection approaches consistently achieve the best results?
- How do different critics' tools complement each other?
- What patterns of effectiveness emerge across multiple critics?
- Which combinations of techniques might create synergies?

### Cross-Pollination Approach:
1. Identify Critic A's most effective bug-detection techniques
2. Identify Critic B's complementary edge-case analysis
3. Identify Critic C's unique code pattern recognition
4. **Your tool**: Combine these complementary techniques into one comprehensive analyzer

### Benefits of Tool-Only Cross-Pollination:
- Combine best deterministic algorithms from multiple sources
- Create "super-tool" that incorporates proven techniques
- Maintain speed ($0.00) and consistency of tool-only execution
- Easy to verify which techniques came from which critics

## Tool-Only Execution Mode

The system supports a **tool-only execution mode** where your Python/shell tool generates complete critic feedback that is directly used, bypassing the AI agent entirely. This is the REQUIRED approach for this strategy.

### How Tool-Only Works

**YAML Frontmatter** in agent.md:
```yaml
---
name: your-critic-name
description: Cross-pollinated tool combining best patterns from multiple critics
execution_mode: tool_only
tool_command: python tools/cross_pollinated_analyzer.py
tool_output_file: tool_output/critic_feedback.txt
---
```

**Execution Flow**:
1. System runs `tool_command` with 600-second timeout
2. System checks output file exists and >= 200 bytes
3. If successful: copies file to `output/agent_output.txt` (agent never called)
4. If failed: clears `tool_output/` and calls agent normally for error recovery

## Required Output Structure

You must create the following files:

### 1. reasoning.md
You are trying to achieve higher accuracy than any of the critics you are examining. Please give your analysis of what to combine and why, based on:
- Identification of complementary strengths across different critics
- Analysis of which critic excels at which types of problems
- **Which tool-based techniques from each critic are worth combining**
- Specific combinations that could address observed weaknesses
- Hypotheses about synergistic combinations

The following should also be included in your reasoning:

#### Performance Analysis
- Review of critic feedback from best performers
- Analysis of critic performance across problem types
- Identification of strengths and weaknesses

#### Error Analysis
{if error_analyzer}- Summary of key findings from error_analysis_report.md
- How analyzer findings influenced your design decisions
{else}- Analysis of error patterns from available analysis artifacts
- How your analysis of error patterns influenced your design decisions
{endif}

#### Tool-Only Cross-Pollination Analysis
- Which critics have the best tools and what makes them effective
- Which specific tool functions/algorithms to adopt from each critic
- How these tools complement each other
- Expected synergies from combining these approaches

### 2. eval_instructions.md
Complete code generation instructions that combine the best patterns from multiple critics. For example:
- Merge effective coding patterns from different sources
- Combine complementary bug-prevention rules
- Unify successful edge-case handling approaches
- Integrate proven testing strategies

### 3. tools/
Analysis tools as Python (.py) or shell (.sh) scripts:
- **PRIMARY GOAL**: Create a tool that combines the best analysis techniques from multiple critics
- Incorporate proven techniques from top-performing critics' tools
- Output to `tool_output/critic_feedback.txt`
- Add error handling with meaningful exit codes

Example cross-pollinated tool:
```python
#!/usr/bin/env python3
"""Cross-pollinated problem analyzer."""

import json
import sys

# Technique from Critic A: Bug pattern detection
def detect_common_bugs(code):
    bugs = []
    # ... detection logic ...
    return bugs

# Technique from Critic B: Edge case analysis
def analyze_edge_cases(question, code):
    edge_cases = []
    # ... analysis logic ...
    return edge_cases

# Technique from Critic C: Complexity analysis
def analyze_complexity(code):
    issues = []
    # ... complexity checks ...
    return issues

def analyze_problem(context_path):
    with open(context_path, 'r') as f:
        ctx = json.load(f)

    output_lines = []
    output_lines.append("# CRITIC FEEDBACK\n")

    # Apply all cross-pollinated techniques
    bugs = detect_common_bugs(ctx['code_v1'])
    edge_cases = analyze_edge_cases(ctx['question'], ctx['code_v1'])
    complexity_issues = analyze_complexity(ctx['code_v1'])

    # Combine findings into comprehensive feedback
    # ...

    with open('tool_output/critic_feedback.txt', 'w') as f:
        f.write('\n'.join(output_lines))

    return 0

if __name__ == "__main__":
    exit(analyze_problem("problem_context.json"))
```

### 4. agent.md
Problem analysis agent with YAML frontmatter:

```yaml
---
name: your-cross-pollinated-critic-name
description: Combined best techniques from multiple high-performing critics
execution_mode: tool_only
tool_command: python tools/cross_pollinated_analyzer.py
tool_output_file: tool_output/critic_feedback.txt
---
```

## Success Metrics

Your evolved package should:
- Successfully combine complementary strengths from multiple critics
- Use tool-only execution mode for deterministic, fast Phase 1
- Generate comprehensive critic feedback that helps coders fix bugs
- Address weaknesses observed in individual critics
- Be maintainable and allow tracing techniques back to their sources

## Your overall goal: Push critic accuracy higher through cross-pollination

Combine the best techniques from multiple critics to create a "super-critic" that outperforms any individual critic.

**Focus on identifying and combining proven techniques** into one comprehensive tool-only solution.

## Important Notes

- The final critic prompt will be: [tool output] + [eval_instructions]
- Tool has access to: problem_context.json
- Tool should write to: tool_output/critic_feedback.txt
- If tool succeeds: output used directly (fast, $0.00 cost)
- If tool fails: agent called for error recovery

Remember: The power of cross-pollination comes from thoughtful combination of complementary strengths, not just merging everything together.
