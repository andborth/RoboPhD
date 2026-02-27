# Refinement Strategy (Approach-Neutral)

Having reviewed the files provided to you above, you are building a Claude Code agent that uses a three-artifact architecture with the goal of achieving higher accuracy than any previous agent. The strategy you will be using (in service of your overall goal of achieving higher accuracy on unseen databases) is to adopt one agent as a starting point and to make targeted changes on top of that agent.

**You have full flexibility in choosing the best approach** for database analysis. Different approaches have different trade-offs, and you should select based on what will achieve the highest accuracy.

So in this strategy you will:

1. **Pick an agent to be your starting point**. This is your call. Maybe this is a current top-performer or maybe is an agent where you see an opportunity to fix it so that it would be the top performer.
2. **Copy the agent as a starting point for your new agent**. Copy over the whole three-artifact structure
3. **Choose your approach and refine**. Select the approach that best suits your accuracy goals and refine the agent accordingly.

## Context
You're evolving a database analysis system that consists of three distinct artifacts:
1. **eval_instructions.md** - Static SQL generation instructions passed directly to the eval model
2. **agent.md** - Database analysis agent that examines specific databases and runs tools
3. **tools/** - Python/shell scripts for database analysis

## Three Approaches to Database Analysis

You can choose any of these approaches or combine elements from multiple approaches:

### Approach 1: Tool-Only Execution

**Philosophy**: Deterministic Python/shell scripts generate complete analysis files that bypass the AI agent entirely.

**Pros**:
- Consistency: Same database always produces same analysis
- Speed: No LLM invocation for Phase 1 (1-5s vs 30-60s)
- Cost: $0.00 for Phase 1 analysis
- Debuggability: Clear, inspectable code

**Cons**:
- Less flexible for edge cases
- Requires upfront engineering of all analysis logic
- May miss nuanced patterns that an LLM would catch

**YAML Configuration**:
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
1. System runs tool_command with 600s timeout
2. If successful (exit 0, file >= 200 bytes): uses output directly
3. If failed: clears tool_output/ and calls agent for recovery

**Example Tool**:
```python
#!/usr/bin/env python3
import sqlite3

def analyze_database():
    # Generate COMPLETE analysis
    conn = sqlite3.connect("database.sqlite")
    # ... comprehensive analysis ...
    output = generate_full_analysis(conn)

    with open('tool_output/schema_analysis.txt', 'w') as f:
        f.write(output)
    return 0  # Success

if __name__ == "__main__":
    exit(analyze_database())
```

### Approach 2: Hybrid (Tools + Agent Reasoning)

**Philosophy**: Tools extract structured data; AI agent provides high-level analysis, pattern recognition, and natural language insights.

**Pros**:
- Combines deterministic analysis with flexible reasoning
- Agent can adapt to unusual schemas or patterns
- Tools handle tedious data extraction
- Agent provides strategic guidance

**Cons**:
- Requires LLM invocation ($0.10-2.00 per database)
- Slower than tool-only (30-60s typical)
- Prompt sensitivity can affect consistency

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Brief description
---
```
(No `execution_mode` field - uses normal agent execution)

**Example Agent Instructions**:

Create an agent.md file with this structure:

```
# Hybrid Schema Analyzer

## Process

1. Run Analysis Tools
   - python tools/schema_extractor.py    # Extract schemas
   - python tools/fk_detector.py          # Detect relationships
   - python tools/value_sampler.py        # Sample data

2. Review Tool Outputs
   - Read all files in tool_output/
   - Identify patterns and relationships
   - Note any anomalies or special cases

3. Generate High-Level Analysis
   - Database domain and purpose
   - Key relationships and patterns
   - Query strategy guidance
   - Special handling notes

4. Write Output
   Write comprehensive analysis to output/agent_output.txt
```

### Approach 3: Agent-Centric (Agent Orchestrates with Light Tools)

**Philosophy**: AI agent does most of the analysis, using simple tools for specific tasks (e.g., checking table existence, getting row counts).

**Pros**:
- Maximum flexibility and adaptability
- Can handle complex, unusual schemas
- Natural language reasoning about database
- Can make judgment calls on ambiguous cases

**Cons**:
- Most expensive ($0.50-2.00 per database)
- Prompt sensitive - results may vary
- Slowest approach (30-90s typical)
- Harder to debug than deterministic approaches

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Brief description
---
```

**Example Agent Instructions**:
```markdown
# Agent-Centric Schema Analyzer

You are responsible for comprehensive database analysis.

## Process

1. **Explore Database Structure**
   - Use SQL queries to understand schema
   - Identify tables, columns, types
   - Detect primary and foreign keys

2. **Optional: Use Helper Tools**
   - `python tools/quick_stats.py` for table statistics
   - `python tools/sample_data.py` for data samples

3. **Analyze and Document**
   - Identify database domain and purpose
   - Document all tables and relationships
   - Provide sample values and patterns
   - Give strategic SQL guidance

4. **Write Output**
   Write complete analysis to `output/agent_output.txt`
```

## Your Task
Create an evolved agent package using the three-artifact structure to achieve higher accuracy on databases you haven't seen yet. **Choose the approach that you believe will achieve the highest accuracy** based on your analysis of errors and performance data.

Details about your task are below:

## Required Output Structure

You must create the following files:

### 1. reasoning.md
Your analysis of why your new agent is going to outperform previous agents. In your discussion:
- Note which agent you used as a starting point
- Explain which approach you chose (tool-only, hybrid, agent-centric) and why
- Describe how you plan to improve the starting agent

The following should also be included in your reasoning:

#### Performance Analysis
- Review of system prompts from best performers
- Analysis of agent performance across databases
- Identification of strengths and weaknesses

#### Error Analysis
- Analysis of error patterns from available error analysis artifacts (evaluation.json, error_analysis_report.md, etc.)
- How your analysis of error patterns influenced your design decisions

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
- You can use standard Python libraries + sqlite3
- You can also use any other Python library or any command-line tool if you see that it is currently installed
- You should analyze the specific database found at ./database.sqlite
- Generate database-specific guidance
- Output to stdout or files in ./tool_output/
- Include error handling
- The tools should incorporate the ideas you described in reasoning.md

**Tool design depends on your chosen approach**:
- **Tool-only**: Create comprehensive tool that generates complete output file
- **Hybrid**: Create focused tools that extract specific data for agent review
- **Agent-centric**: Create simple helper tools for specific tasks

Example tool structure (tool-only):
```python
#!/usr/bin/env python3
import sqlite3
import sys

def analyze_database():
    # Generate COMPLETE analysis
    output = []
    # ... comprehensive logic ...
    with open('tool_output/schema_analysis.txt', 'w') as f:
        f.write('\n'.join(output))
    return 0

if __name__ == "__main__":
    sys.exit(analyze_database())
```

Example tool structure (hybrid):
```python
#!/usr/bin/env python3
import sqlite3

def extract_foreign_keys():
    # Extract structured FK data
    conn = sqlite3.connect("database.sqlite")
    # ... extraction logic ...
    with open('tool_output/foreign_keys.json', 'w') as f:
        json.dump(fk_data, f, indent=2)

if __name__ == "__main__":
    extract_foreign_keys()
```

### 4. agent.md
Database analysis agent with YAML frontmatter that MUST include these EXACT two fields:

**For tool-only approach**:
```yaml
---
name: your-unique-agent-name-here
description: Brief description of the approach you are taking with the overall three-artifact package
execution_mode: tool_only
tool_command: python tools/your_analyzer.py
tool_output_file: tool_output/schema_analysis.txt
---
```

**For hybrid or agent-centric**:
```yaml
---
name: your-unique-agent-name-here
description: Brief description of the approach you are taking with the overall three-artifact package
---
```

Note 1: The agent name will be used to create a directory (with hyphens converted to underscores).
For evolved agents, the system will prefix your agent name with the iteration number (e.g., iter2_your_agent_name).

Note 2: As noted above, you will be evolving the agent.md of the starting-point model

Note 3: Please be sure that the name and description of your agent are different than those of the starting-point model

**Agent instructions should match your chosen approach**:

**Hybrid approach template** (tools extract data, agent analyzes and synthesizes):
- Run analysis tools to extract structured data
- Review all tool outputs for completeness
- Analyze and synthesize: identify domain, patterns, relationships, edge cases
- Develop strategic guidance for SQL generation
- Generate comprehensive analysis combining tool outputs with insights
- Error handling: check tool_output/ for errors, fall back to manual analysis

Example agent.md process section:
1. Run tools: `python tools/schema_extractor.py`, `python tools/relationship_detector.py`, `python tools/value_sampler.py`
2. Review tool outputs in `tool_output/`
3. Analyze patterns and synthesize insights
4. Write analysis to `./output/agent_output.txt`
5. Error handling if tools fail

**Tool-only approach**: Minimal instructions (error recovery only)
**Agent-centric approach**: Comprehensive instructions for database analysis

## Success Metrics

Your evolved package should:
- Use the approach best suited for achieving highest accuracy
- Preserve good general-purpose SQL generation instructions (in eval_instructions.md)
- Provide rich database-specific context (via tool output and/or agent analysis)
- Separate concerns cleanly between the above two elements
- Address specific failure patterns from your analysis
- Be maintainable and debuggable

## Your overall goal: Push accuracy higher

You are an expert in the field of Text2SQL. Use your knowledge of the field, your analysis of what is bringing accuracy down with current agents, and your analysis of the system prompts that prior systems are generating to build an agent package that will achieve higher accuracy than previous agents on a set of databases that you haven't seen before.

Choose the approach that you believe will maximize accuracy based on your analysis. Don't feel constrained to any particular approach - use your judgment.

## Important Notes

- The final system prompt will be: [agent output or tool output] + [eval_instructions]
- Tool-only: tool output used directly (if successful)
- Hybrid/Agent-centric: agent generates output using tools and reasoning
- Agent has access to: database.sqlite, tools/, tool_output/

Remember: **Think harder** than you normally would about this. Gather the information you need and use your knowledge and experience to improve accuracy. Use your judgment to pick an agent as a starting point that you think makes sense, and then build a comprehensive solution that will maximize accuracy.
