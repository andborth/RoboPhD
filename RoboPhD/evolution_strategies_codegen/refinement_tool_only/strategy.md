# Refinement Strategy for Code Generation Critics (Tool-Only Focus)

Having reviewed the files provided to you above, you are building a Claude Code agent that uses a three-artifact architecture with the goal of achieving higher critic accuracy than any previous agent. The strategy you will be using (in service of your overall goal of achieving higher accuracy on unseen coding problems) is to adopt one agent as a starting point and to make targeted changes on top of that agent.

**IMPORTANT: This strategy REQUIRES creating a tool-only agent** where deterministic Python/shell scripts generate complete problem analysis that bypasses the AI agent entirely. This approach offers:
- **Consistency**: Same problem always produces same analysis
- **Speed**: No LLM invocation costs or latency for Phase 1
- **Debuggability**: Clear, inspectable code instead of opaque AI reasoning
- **Reliability**: Deterministic behavior without prompt sensitivity

So in this strategy you will:

1. **Pick an agent to be your starting point**. This is your call. Maybe this is a current top-performer or maybe is an agent where you see an opportunity to fix it so that it would be the top performer.
2. **Copy the agent as a starting point for your new agent**. Copy over the whole three-artifact structure
3. **Refine as a tool-only agent**. Transform the agent to use the tool-only execution mode where a Python script generates complete analysis output that bypasses the AI agent.

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

The critic's job is to analyze the initial code (code_v1) and provide feedback that helps the coder produce a correct revision (code_v2).

## Tool-Only Execution Mode

The system supports a **tool-only execution mode** where your Python/shell tool generates complete critic feedback that is directly used, bypassing the AI agent entirely. This is the REQUIRED approach for this strategy.

### How Tool-Only Works

**YAML Frontmatter** in agent.md:
```yaml
---
name: your-agent-name
description: Brief description
execution_mode: tool_only
tool_command: python tools/problem_analyzer.py
tool_output_file: tool_output/critic_feedback.txt
---
```

**Execution Flow**:
1. System runs `tool_command` with 600-second timeout
2. System checks output file exists and >= 200 bytes
3. If successful: copies file to `output/agent_output.txt` (agent never called)
4. If failed: clears `tool_output/` and calls agent normally for error recovery

### Example Tool-Only Agent

**agent.md** (tool-only template):

Create an agent.md file with this structure:

```
---
name: your-critic-name
description: Brief description of your tool-only critic approach
execution_mode: tool_only
tool_command: python tools/problem_analyzer.py
tool_output_file: tool_output/critic_feedback.txt
---

# Your Critic Name (Tool-Only)

This critic uses deterministic tool-only execution for consistent, fast problem analysis.

## Process

1. **Run Analysis Tool**
   - Execute: python tools/problem_analyzer.py

2. **Read and Output Results**
   - Read the generated feedback from tool_output/critic_feedback.txt
   - Write the complete output to ./output/agent_output.txt

## Error Recovery

If the tool fails:

1. Check problem_context.json exists
2. Verify Python environment has required libraries
3. Examine any error messages in tool_output/
4. Fall back to manual analysis if needed
```

**tools/problem_analyzer.py** (template):
```python
#!/usr/bin/env python3
"""Problem analyzer for code generation critic."""

import json
import sys

def analyze_problem(context_path: str, output_file: str):
    """Generate critic feedback for a coding problem."""

    try:
        with open(context_path, 'r') as f:
            context = json.load(f)

        question = context.get('question', '')
        code_v1 = context.get('code_v1', '')
        approach = context.get('approach', '')

        output = []
        output.append("# CRITIC FEEDBACK\n")

        # Your analysis logic here
        # - Identify potential bugs in code_v1
        # - Check if approach matches problem requirements
        # - Suggest specific improvements

        # Write output
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))

        print(f"Analysis complete - wrote to {output_file}")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    exit(analyze_problem("problem_context.json", "tool_output/critic_feedback.txt"))
```

## Your Task
Create an evolved critic package using the three-artifact structure with tool-only execution to achieve higher accuracy on coding problems you haven't seen yet.

## Required Output Structure

You must create the following files:

### 1. reasoning.md
Your analysis of why your new critic is going to outperform previous critics. In your discussion, please note which agent you used as a starting point and describe how you plan to improve it using tool-only execution.

The following should also be included in your reasoning:

#### Performance Analysis
- Review of critic feedback from best performers
- Analysis of critic performance across problem types
- Identification of strengths and weaknesses

#### Error Analysis
{if error_analyzer}- Summary of key findings from error_analysis_report.md (generated by {error_analyzer_agent})
- How analyzer findings influenced your design decisions
{else}- Analysis of error patterns from available analysis artifacts
- How your analysis of error patterns influenced your design decisions
{endif}

### 2. eval_instructions.md
Complete code generation instructions for the coder model. For example, these instructions might include:
- Code writing principles and patterns
- Edge case handling rules
- Common bug patterns to avoid
- Output format requirements (clean code, proper formatting)
- Testing considerations

### 3. tools/
Analysis tools as Python (.py) or shell (.sh) scripts:
- **PRIMARY GOAL**: Create a tool that generates COMPLETE, SELF-CONTAINED critic feedback
- The tool should analyze the problem, code_v1, and approach
- Output to `tool_output/critic_feedback.txt`
- Add error handling with meaningful exit codes
- You should analyze the problem_context.json file

Example tool structure (tool-only):
```python
#!/usr/bin/env python3
import json
import sys

def analyze_problem(context_path):
    # Load problem context
    with open(context_path, 'r') as f:
        ctx = json.load(f)

    # Generate COMPLETE critic feedback
    output_lines = []

    # ... comprehensive analysis logic ...

    # Write to tool_output/
    with open('tool_output/critic_feedback.txt', 'w') as f:
        f.write('\n'.join(output_lines))

    print("Analysis complete")
    return 0  # Success

if __name__ == "__main__":
    exit_code = analyze_problem("problem_context.json")
    sys.exit(exit_code)
```

### 4. agent.md
Problem analysis agent with YAML frontmatter that MUST include these fields:

```yaml
---
name: your-unique-critic-name-here
description: Brief description of the approach you are taking
execution_mode: tool_only
tool_command: python tools/problem_analyzer.py
tool_output_file: tool_output/critic_feedback.txt
---
```

**IMPORTANT**: Include the `execution_mode: tool_only` configuration along with `tool_command` and `tool_output_file` to enable direct tool execution.

## Success Metrics

Your evolved package should:
- Use tool-only execution mode for deterministic, fast Phase 1
- Generate comprehensive critic feedback via Python tool
- Identify real bugs and issues in code_v1
- Provide actionable suggestions for the coder
- Consider edge cases and test scenarios
- Be maintainable and debuggable

## Your overall goal: Push critic accuracy higher with tool-only execution

You are an expert in code review and software engineering. Use your knowledge to build a critic package that will help coders produce more correct code than previous critics achieved.

**Focus on creating a robust, comprehensive Python tool that generates complete critic feedback** for direct use by the coder model.

## Important Notes

- The final critic prompt will be: [tool output] + [eval_instructions]
- Tool has access to: problem_context.json
- Tool should write to: tool_output/critic_feedback.txt
- If tool succeeds: output used directly (fast, $0.00 cost)
- If tool fails: agent called for error recovery

Remember: **Think harder** than you normally would about this. Gather the information you need and use your knowledge and experience to improve accuracy.
