# Cross-Pollination Strategy (Approach-Neutral)

Having reviewed the files provided to you above, you are creating a new Claude Code agent primarily by combining successful elements from multiple existing agents using the three-artifact architecture.

**You have full flexibility in choosing the best approach** for combining successful patterns. Different approaches have different trade-offs, and you should select based on what will achieve the highest accuracy.

## Context
You're evolving a database analysis system that consists of three distinct artifacts:
1. **eval_instructions.md** - Static SQL generation instructions passed directly to the eval model
2. **agent.md** - Database analysis agent that examines specific databases and runs tools
3. **tools/** - Python/shell scripts for database analysis

## Your Task
Create a new agent package by cross-pollinating successful patterns from the top-performing agents to achieve higher accuracy on databases you haven't seen yet.

**Note:** Although you are primarily using a cross-pollinating approach, you can use a new idea of your own if you think you see an opportunity.

## Three Approaches to Cross-Pollination

You can combine successful patterns using any of these approaches:

### Approach 1: Tool-Only Cross-Pollination

**Philosophy**: Identify and merge the best deterministic tools from multiple agents into one comprehensive analyzer.

**Pros**:
- Combines proven algorithms from multiple sources
- Maintains speed ($0.00) and consistency
- Easy to trace which technique came from which agent
- Can verify correctness of individual components

**Cons**:
- Limited to techniques that can be encoded as algorithms
- Less flexible for nuanced combinations

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Tool-only cross-pollination of [Agent A + Agent B + Agent C]
execution_mode: tool_only
tool_command: python tools/combined_analyzer.py
tool_output_file: tool_output/schema_analysis.txt
---
```

**Example**: Combine Agent A's FK detection + Agent B's value sampling + Agent C's partition detection

### Approach 2: Hybrid Cross-Pollination

**Philosophy**: Merge successful tool patterns from multiple agents, with an AI agent to orchestrate and synthesize.

**Pros**:
- Combines best deterministic analysis with flexible reasoning
- Agent can adapt combined techniques to specific databases
- Can handle cases where techniques conflict or need judgment
- Strategic synthesis of multiple tool outputs

**Cons**:
- Requires LLM invocation ($0.10-2.00 per database)
- Need careful prompts to effectively combine multiple approaches

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Hybrid cross-pollination combining tools and reasoning
---
```

**Example**: Tools from Agents A, B, C extract different aspects; AI agent synthesizes insights

### Approach 3: Agent-Centric Cross-Pollination

**Philosophy**: Combine successful reasoning strategies and high-level approaches from multiple agents.

**Pros**:
- Can merge nuanced reasoning patterns
- Flexible adaptation of multiple strategies
- Can apply different techniques based on database characteristics

**Cons**:
- Most expensive approach
- Harder to verify which techniques are being applied
- Prompt complexity increases with more merged patterns

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Agent-centric strategy merging multiple reasoning approaches
---
```

**Example**: Agent A's schema analysis strategy + Agent B's relationship reasoning + Agent C's query guidance

## Required Output Structure

You must create the following files:

### 1. reasoning.md
You are trying to achieve higher accuracy than any of the agents you are examining. Please give your analysis of what to combine and why, based on, for instance:
- Identification of complementary strengths across different agents
- Analysis of which agent excels at which types of problems
- Specific combinations that could address observed weaknesses
- Hypotheses about synergistic combinations
- Which cross-pollination approach you chose and why

The following should also be included in your reasoning:

#### Performance Analysis
- Review of system prompts from best performers
- Analysis of agent performance across databases
- Identification of strengths and weaknesses

#### Error Analysis
- Analysis of error patterns from available error analysis artifacts (evaluation.json, error_analysis_report.md, etc.)
- How your analysis of error patterns influenced your design decisions

#### Cross-Pollination Strategy
- Which agents you selected and what makes each one valuable
- What specific patterns/techniques to adopt from each
- How these patterns complement each other
- Why your chosen approach best combines these patterns

### 2. eval_instructions.md
Complete SQL generation instructions that combine the best patterns from multiple agents. For example, you might want to:
- Merge effective column selection rules from different sources
- Combine complementary SQL pattern recognitions
- Integrate different error-avoidance strategies
- Synthesize output format requirements
- Unify successful approaches to evidence handling

Notes:
- Write your own instructions, as appropriate, to achieve the goals laid out in reasoning.md
- Feel free to copy (or copy and then modify) instructions from other agents if you see something that you like
- These instructions go DIRECTLY to the eval model

### 3. tools/
Analysis tools combining successful patterns (design depends on your approach):

**Tool-only approach**: Create comprehensive tool merging best algorithms
```python
#!/usr/bin/env python3
"""Combined analyzer with techniques from Agents A, B, C."""
import sqlite3, sys

def technique_from_agent_a():
    # Best FK detection
    pass

def technique_from_agent_b():
    # Best value sampling
    pass

def analyze_database():
    output = apply_all_techniques()
    with open('tool_output/analysis.txt', 'w') as f:
        f.write(output)
    return 0

if __name__ == "__main__":
    sys.exit(analyze_database())
```

**Hybrid approach**: Focused tools that extract complementary data
```python
#!/usr/bin/env python3
"""Tool A's schema extraction."""
# Extract schemas Agent A's way
with open('tool_output/schemas_a.json', 'w') as f:
    json.dump(data, f)

"""Tool B's relationship detection."""
# Detect relationships Agent B's way
with open('tool_output/relationships_b.json', 'w') as f:
    json.dump(data, f)
```

**Notes**:
- Write your own scripts, as appropriate, to achieve the goals laid out in reasoning.md
- Feel free to copy (or copy and then modify) scripts from other agents if you see something that you like
- The tools should incorporate the cross-pollinated ideas you described in reasoning.md

## Tool Design Principles

Create **multiple focused tools** rather than one comprehensive tool:
- Each tool should have a single, clear purpose
- Tools should output to `tool_output/TOOL_NAME.json`
- Use an orchestrator script (`analyze_all.py`) to run all tools in sequence
- This enables easier debugging, evolution, and cross-pollination

**Benefits**:
- Future evolution can easily combine tools from multiple agents
- Failed tools don't cascade (schema analysis works even if value sampling fails)
- Incremental refinement: improve one tool without touching others

### 4. agent.md
Database analysis agent with appropriate YAML configuration:

**For tool-only**:
```yaml
---
name: your-unique-agent-name-here
description: Cross-pollinated tool combining patterns from multiple agents
execution_mode: tool_only
tool_command: python tools/combined_analyzer.py
tool_output_file: tool_output/analysis.txt
---
```

**For hybrid/agent-centric**:
```yaml
---
name: your-unique-agent-name-here
description: Cross-pollination of successful patterns from multiple agents
---
```

Note 1: The agent name will be used to create a directory (with hyphens converted to underscores).
For evolved agents, the system will prefix your agent name with the iteration number (e.g., iter2_your_agent_name).

Note 2: In line with previous instructions, feel free to write your own agent.md or to copy elements from the agent.md files of other agents as makes sense to achieve your goals

**Agent instructions should match your chosen approach**:

**Hybrid approach template** (tools from multiple agents, LLM synthesizes):
- Run combined analysis tools from multiple agents
- Review all tool outputs
- Synthesize cross-pollinated insights: identify complementary patterns, resolve conflicts
- Develop comprehensive understanding from multiple perspectives
- Create unified strategic guidance
- Generate analysis combining all tool outputs with synthesized insights
- Error handling: check tool_output/ for errors, fall back to manual analysis

Example agent.md process section:
1. Run tools: `python tools/agent_a_schema_tool.py`, `python tools/agent_b_relationship_tool.py`, `python tools/agent_c_value_tool.py`
2. Review outputs in `tool_output/`
3. Synthesize insights from multiple tools
4. Write analysis to `./output/agent_output.txt`
5. Error handling if tools fail

**Tool-only approach**: Minimal instructions (error recovery, technique summary)
**Agent-centric approach**: Instructions for applying multiple reasoning strategies

## Success Metrics

Your evolved package should:
- Effectively combine complementary strengths from multiple agents
- Use the approach best suited for cross-pollinating the selected patterns
- Preserve good general-purpose SQL generation instructions (in eval_instructions.md)
- Provide rich database-specific context from combined techniques
- Separate concerns cleanly between the elements
- Address specific failure patterns from your analysis
- Be maintainable and debuggable

## Your overall goal: Push accuracy higher through cross-pollination

You are an expert in the field of Text2SQL. Use your knowledge of the field, your analysis of what is bringing accuracy down with current agents, and **your analysis of which patterns from each agent work best** to build an agent package that will achieve higher accuracy than previous agents on a set of databases that you haven't seen before.

**Choose the cross-pollination approach that you believe will maximize accuracy** based on the specific patterns you're combining.

## Important Notes

- The final system prompt will be: [agent output or tool output] + [eval_instructions]
- Tool-only: tool output used directly (if successful)
- Hybrid/Agent-centric: agent generates output using tools and reasoning
- Agent has access to: database.sqlite, tools/, tool_output/

Remember: **Think harder** than you normally would about this. Review multiple top-performing agents, identify what makes each one effective, and choose the approach that best combines these strengths to achieve your goal of pushing accuracy higher.
