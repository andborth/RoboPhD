# Refinement Strategy (Tool-Only Focus)

Having reviewed the files provided to you above, you are building a Claude Code agent that uses a three-artifact architecture with the goal of achieving higher accuracy than any previous agent. The strategy you will be using (in service of your overall goal of achieving higher accuracy on unseen databases) is to adopt one agent as a starting point and to make targeted changes on top of that agent.

**IMPORTANT: This strategy REQUIRES creating a tool-only agent** where deterministic Python/shell scripts generate complete database analysis files that bypass the AI agent entirely. This approach offers:
- **Consistency**: Same database always produces same analysis
- **Speed**: No LLM invocation costs or latency for Phase 1
- **Debuggability**: Clear, inspectable code instead of opaque AI reasoning
- **Reliability**: Deterministic behavior without prompt sensitivity

So in this strategy you will:

1. **Pick an agent to be your starting point**. This is your call. Maybe this is a current top-performer or maybe is an agent where you see an opportunity to fix it so that it would be the top performer.
2. **Copy the agent as a starting point for your new agent**. Copy over the whole three-artifact structure
3. **Refine as a tool-only agent**. Transform the agent to use the tool-only execution mode where a Python script generates complete analysis output that bypasses the AI agent.

## Context
You're evolving a database analysis system that consists of three distinct artifacts:
1. **eval_instructions.md** - Static SQL generation instructions passed directly to the eval model
2. **agent.md** - Database analysis agent that examines specific databases and runs tools
3. **tools/** - Python/shell scripts for database analysis

## Tool-Only Execution Mode

The system supports a **tool-only execution mode** where your Python/shell tool generates a complete analysis file that is directly copied to the agent output, bypassing the AI agent entirely. This is the REQUIRED approach for this strategy.

### How Tool-Only Works

**YAML Frontmatter** in agent.md:
```yaml
---
name: your-agent-name
description: Brief description
execution_mode: tool_only
tool_command: python tools/comprehensive_analyzer.py
tool_output_file: tool_output/schema_analysis.txt
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
name: your-agent-name
description: Brief description of your tool-only approach
execution_mode: tool_only
tool_command: python tools/your_analyzer.py
tool_output_file: tool_output/analysis.txt
---

# Your Agent Name (Tool-Only)

This agent uses deterministic tool-only execution for consistent, fast database analysis.

## Process

1. **Run Analysis Tool**
   - Execute: python tools/your_analyzer.py

2. **Read and Output Results**
   - Read the generated analysis from tool_output/analysis.txt
   - Write the complete output to ./output/agent_output.txt

## Error Recovery

If the tool fails:

1. Check database.sqlite exists
2. Verify Python environment has required libraries
3. Examine any error messages in tool_output/
4. Attempt to run the tool manually to see errors
5. Fall back to manual analysis if needed
```

**Notes for evolution AI:**
- Design the tool to generate comprehensive database analysis
- Tool output should contain everything the eval model needs for SQL generation
- Let your analysis guide what information to include

**tools/your_analyzer.py** (template):
```python
#!/usr/bin/env python3
"""Database analyzer for tool-only execution."""

import sqlite3
import sys

def analyze_database(db_path: str, output_file: str):
    """Generate comprehensive database analysis."""

    try:
        conn = sqlite3.connect(db_path)
        output = []

        # Your analysis logic here
        # Extract schemas, relationships, patterns, etc.
        # Based on your error analysis and accuracy goals

        output.append("# DATABASE ANALYSIS\n")
        # ... add your analysis sections ...

        # Write output
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))

        print(f"Analysis complete - wrote to {output_file}")
        conn.close()
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    exit(analyze_database("database.sqlite", "tool_output/analysis.txt"))
```

## Your Task
Create an evolved agent package using the three-artifact structure with tool-only execution to achieve higher accuracy on databases you haven't seen yet. In developing this agent, you can use any techniques you have previously learned, or you can come up with your own ideas for achieving your goal.

Details about your task are below:

## Required Output Structure

You must create the following files:

### 1. reasoning.md
Your analysis of why your new agent is going to outperform previous agents. In your discussion, please note which agent you used as a starting point and describe how you plan to improve it using tool-only execution.

The following should also be included in your reasoning:

#### Performance Analysis
- Review of system prompts from best performers
- Analysis of agent performance across databases
- Identification of strengths and weaknesses

#### Error Analysis
{if error_analyzer}- Summary of key findings from error_analysis_report.md (generated by {error_analyzer_agent})
- How analyzer findings influenced your design decisions
{else}- Analysis of error patterns from available error analysis artifacts (evaluation.json, error_analysis_report.md, etc.)
- How your analysis of error patterns influenced your design decisions
{endif}

### 2. eval_instructions.md
Complete SQL generation instructions for the eval model (which will be generating SQL in response to user questions and optional "evidence"). For example, these instructions might include:
- SQL writing principles and patterns
- Column selection rules (e.g., return ONLY requested columns)
- Error patterns to avoid (based on your error analysis)
- Output format requirements (clean SQL, no markdown)
- Common SQL patterns and examples
- As noted above, you will be starting from the eval_instructions.md of the starting-point model and then modifying them according to your strategy
- These instructions go DIRECTLY to the eval model

### 3. tools/
Analysis tools as Python (.py) or shell (.sh) scripts:
- **PRIMARY GOAL**: Create a tool that generates a COMPLETE, SELF-CONTAINED analysis file
- The tool should produce output that is ready to be directly used by the eval model
- Output to `tool_output/schema_analysis.txt` (or similar)
- When generating these tools, think about what information the eval model will need about this specific database
- Note that this database-specific information is distinct from the general instructions provided in eval_instructions.md
- Remember that the eval model will not have any information about the database beyond the information that your tool provides
- Add error handling with meaningful exit codes
- You can use standard Python libraries + sqlite3
- You can also use any other Python library or any command-line tool if you see that it is currently installed
- You should analyze the specific database found at ./database.sqlite
- The tools should incorporate the ideas you described in reasoning.md

Example tool structure (tool-only):
```python
#!/usr/bin/env python3
import sqlite3
import sys

def analyze_database(db_path):
    # Generate COMPLETE analysis
    output_lines = []

    # ... comprehensive analysis logic ...

    # Write to tool_output/
    with open('tool_output/schema_analysis.txt', 'w') as f:
        f.write('\n'.join(output_lines))

    print("Analysis complete")
    return 0  # Success

if __name__ == "__main__":
    exit_code = analyze_database("database.sqlite")
    sys.exit(exit_code)
```

### 4. agent.md
Database analysis agent with YAML frontmatter that MUST include these fields:

```yaml
---
name: your-unique-agent-name-here
description: Brief description of the approach you are taking with the overall three-artifact package
execution_mode: tool_only
tool_command: python tools/your_analyzer.py
tool_output_file: tool_output/schema_analysis.txt
---
```

**IMPORTANT**: Include the `execution_mode: tool_only` configuration along with `tool_command` and `tool_output_file` to enable direct tool execution.

Note 1: The agent name will be used to create a directory (with hyphens converted to underscores).
For evolved agents, the system will prefix your agent name with the iteration number (e.g., iter2_your_agent_name).

Note 2: As noted above, you will be evolving the agent.md of the starting-point model

Note 3: Please be sure that the name and description of your agent are different than those of the starting-point model

The agent instructions should be minimal since the tool does all the work:
- Brief explanation of tool-only execution
- Error recovery instructions (if tool fails, agent is called)
- Reference to the tool output format

## Success Metrics

Your evolved package should:
- Use tool-only execution mode for deterministic, fast Phase 1
- Generate comprehensive database analysis via Python tool
- Preserve good general-purpose SQL generation instructions (in eval_instructions.md)
- Provide rich database-specific context (in the tool output)
- Separate concerns cleanly between the above two elements
- Address specific failure patterns from your analysis
- Be maintainable and debuggable

## Your overall goal: Push accuracy higher with tool-only execution

You are an expert in the field of Text2SQL. Use your knowledge of the field, your analysis of what is bringing accuracy down with current agents, and your analysis of the system prompts that prior systems are generating to build an agent package that will achieve higher accuracy than previous agents on a set of databases that you haven't seen before.

**Focus on creating a robust, comprehensive Python tool that generates complete database analysis files** for direct use by the eval model.

## Important Notes

- The final system prompt will be: [tool output] + [eval_instructions]
- Tool has access to: database.sqlite
- Tool should write to: tool_output/schema_analysis.txt (or similar)
- If tool succeeds: output used directly (fast, $0.00 cost)
- If tool fails: agent called for error recovery
- Agent has access to: database.sqlite, tools/, tool_output/

Remember: **Think harder** than you normally would about this. Gather the information you need and use your knowledge and experience to improve accuracy. Use your judgment to pick an agent as a starting point that you think makes sense, then build a comprehensive tool-only solution that will achieve your goal of pushing accuracy higher.
