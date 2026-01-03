# Research-Driven Strategy (Approach-Neutral)

Having reviewed the files provided to you above, you are refining a Claude Code agent that uses a three-artifact architecture with the goal of achieving higher accuracy than any previous agent.

**You have full flexibility in choosing the best approach** for implementing research insights. Different approaches have different trade-offs, and you should select based on what will achieve the highest accuracy.

## Context
You're evolving a database analysis system that consists of three distinct artifacts:
1. **eval_instructions.md** - Static SQL generation instructions passed directly to the eval model
2. **agent.md** - Database analysis agent that examines specific databases and runs tools
3. **tools/** - Python/shell scripts for database analysis

## Required Research Reading

You MUST read exactly ONE paper from the available research in papers/:

### Paper Selection Strategy:
1. First, run the paper selection tool to choose your paper:
   ```bash
   python strategy_tools/select_research_paper.py
   ```
   This tool will:
   - Select the next unused paper from the strategy's pool
   - Track used papers in strategy_tools/papers_pool.json
   - Write the selected paper path to `selected_paper.txt` in the current directory

2. Read the selected paper from the path provided by the tool

3. The tool tracks which papers have been used (when --agents-directory is specified) to ensure comprehensive exploration of all available methods

The papers are from top BIRD methods achieving 71-77% accuracy.

## Your Mission

Create an evolved three-artifact agent package that improves SQL generation accuracy by incorporating proven techniques from academic research. **Choose the implementation approach that best suits the research insights and your accuracy goals.**

### Three Approaches for Implementing Research:

#### Approach 1: Tool-Only (Research Algorithms as Code)

**Best for**: Research techniques that can be encoded as deterministic algorithms

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Implements [Paper] insights via deterministic analysis
execution_mode: tool_only
tool_command: python tools/research_analyzer.py
tool_output_file: tool_output/analysis.txt
---
```

**Example**: If paper describes a schema linking algorithm with clear steps, implement as Python functions

**Pros**: Fast ($0.00), consistent, debuggable, verifiable correctness
**Cons**: Less flexible for nuanced research insights requiring judgment

#### Approach 2: Hybrid (Tools Extract, Agent Reasons)

**Best for**: Research combining structured analysis with high-level reasoning

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Hybrid implementation of [Paper] techniques
---
```

**Example**: Tools extract schema patterns (deterministic), agent applies paper's reasoning strategies (flexible)

**Pros**: Combines algorithmic rigor with adaptive reasoning
**Cons**: Moderate cost, requires careful prompt engineering

#### Approach 3: Agent-Centric (Agent Applies Research Principles)

**Best for**: Research insights about reasoning strategies, heuristics, or complex decision-making

**YAML Configuration**:
```yaml
---
name: your-agent-name
description: Agent applies [Paper] reasoning strategies
---
```

**Example**: Paper describes multi-step reasoning process that benefits from LLM flexibility

**Pros**: Maximum flexibility to apply nuanced research insights
**Cons**: Higher cost, prompt sensitive, harder to verify

### Creative Adaptation:
You're not limited to direct implementation. Consider:
- Combining techniques from multiple sections of the paper
- Simplifying complex research methods for practical use
- Creating hybrid approaches that blend paper techniques with existing success patterns
- Choosing tool-only for deterministic parts, agent reasoning for judgment calls

### How to think about the research:
- The key focus is on novel techniques that achieve high BIRD scores
- Consider which aspects are algorithmic vs require judgment
- Some standard techniques in papers might be overlooked basics worth adding
- Not everything is practical (e.g., fine-tuning LLMs) - use your judgment
- Think about the best implementation approach for each research insight

## Required Output Structure

You must create the following files:

### 1. reasoning.md
Your analysis of what to improve and why, based on:

#### Performance Analysis
- Review of system prompts from best performers
- Analysis of agent performance across databases
- Identification of strengths and weaknesses

#### Error Analysis
- Analysis of error patterns from available error analysis artifacts (evaluation.json, error_analysis_report.md, etc.)
- How your analysis of error patterns influenced your design decisions

#### Academic Paper Analysis
- Which paper you selected and its key contributions
- What you learned from the paper
- Ideas most applicable to the three-artifact architecture
- Which implementation approach you chose and why
- How this approach best captures the research insights

### 2. eval_instructions.md
Complete SQL generation instructions incorporating research insights:
- SQL writing principles and patterns from the paper
- Column selection rules informed by research
- Error patterns to avoid (based on paper and your error analysis)
- Output format requirements (clean SQL, no markdown)
- Common SQL patterns discovered in research
- These instructions go DIRECTLY to the eval model

### 3. tools/
Analysis tools implementing research techniques (design depends on your approach):

**Tool-only approach**:
```python
#!/usr/bin/env python3
"""Comprehensive analyzer implementing [Paper] algorithms."""
import sqlite3, sys

def research_algorithm_1():
    # Deterministic implementation
    pass

def analyze_database():
    output = generate_complete_analysis()
    with open('tool_output/analysis.txt', 'w') as f:
        f.write(output)
    return 0

if __name__ == "__main__":
    sys.exit(analyze_database())
```

**Hybrid approach**:
```python
#!/usr/bin/env python3
"""Focused tools for agent to review."""
def extract_schema_patterns():
    # Extract structured data
    with open('tool_output/patterns.json', 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    extract_schema_patterns()
```

## Tool Design Principles

Create **multiple focused tools** rather than one comprehensive tool:
- Each tool should have a single, clear purpose (e.g., one tool per research technique)
- Tools should output to `tool_output/TOOL_NAME.json`
- Use an orchestrator script (`analyze_all.py`) to run all tools in sequence
- This enables easier debugging, evolution, and combining techniques from different papers

**Benefits for research implementation**:
- Future evolution can combine this paper's technique A with another paper's technique B
- Failed tools don't cascade (schema linking works even if value normalization fails)
- Incremental development: implement and test one research technique at a time
- Baseline comparison: disable specific techniques to measure individual impact

### 4. agent.md
Database analysis agent with appropriate YAML configuration:

**For tool-only**:
```yaml
---
name: your-agent-name
description: [Paper]-informed tool-only analyzer
execution_mode: tool_only
tool_command: python tools/analyzer.py
tool_output_file: tool_output/analysis.txt
---
```

**For hybrid/agent-centric**:
```yaml
---
name: your-agent-name
description: [Paper]-informed hybrid analyzer
---
```

Agent instructions should match your chosen approach and explain how research insights are applied.

**Hybrid approach template** (tools extract data, agent applies research insights):
- Run research-informed analysis tools
- Review tool outputs for research technique results
- Apply research insights: analyze using principles from the paper
- Identify patterns using techniques from research
- Develop query guidance based on research findings
- Generate analysis combining tool outputs with research-informed insights
- Error handling: fall back to manual analysis applying research principles

Example agent.md process section:
1. Run tools: `python tools/research_informed_extractor.py`
2. Review tool outputs in `tool_output/`
3. Apply research insights from [Paper Name]
4. Write analysis to `./output/agent_output.txt`
5. Error handling if tools fail

## Success Metrics

Your evolved package should:
- Effectively implement research insights using the most appropriate approach
- Preserve good general-purpose SQL generation instructions (in eval_instructions.md)
- Provide rich database-specific context informed by research
- Address specific failure patterns from your analysis
- Achieve higher accuracy through research-informed design

## Your overall goal: Push accuracy higher with research insights

You are an expert in the field of Text2SQL. Read the research paper carefully, understand the key insights, and **implement them using the approach that will maximize accuracy**. Don't feel constrained to any particular implementation style - choose what works best for the specific research techniques.

## Important Notes

- The final system prompt will be: [agent output or tool output] + [eval_instructions]
- Tool-only: tool output used directly (if successful)
- Hybrid/Agent-centric: agent generates output using tools and reasoning
- Agent has access to: database.sqlite, tools/, tool_output/
- You can judiciously combine research with prior techniques.  For instance, you can use a strong performing agent as a baseline and make targeted improvements

Remember: **Think harder** than you normally would about this. Read the research paper carefully, understand the relevant insights, and build a system that best achieves your goal of increasing accuracy.  You are using your understanding of existing errors, your understanding of how baseline systems are working, and your reading of the latest academic research to help you achieve that goal.
