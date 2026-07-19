


## Previous Iteration Results (Iteration 14)

### Agent Scores

| Agent | Score | Raw / Total |
|-------|-------|-------------|
| iter10_cite_expand | 36.084 | 5.3/14 |
| iter13_grade_view | 38.120 | 5.5/14 |
| iter14_crit_coverage | 38.931 | 5.5/14 |

**Aggregate notes** (how each Score was derived from Raw / Total):

- **iter10_cite_expand**: Mean agent cost $0.0654 exceeded threshold $0.06 by $0.0054 = 0.27 errors of penalty (cost_per_error=$0.0200); subtracted 1.917 score pts from raw 38.002 → final 36.084 (percentage).
- **iter13_grade_view**: Mean agent cost $0.0635 exceeded threshold $0.06 by $0.0035 = 0.18 errors of penalty (cost_per_error=$0.0200); subtracted 1.260 score pts from raw 39.380 → final 38.120 (percentage).
- **iter14_crit_coverage**: Mean agent cost $0.0602 exceeded threshold $0.06 by $0.0002 = 0.01 errors of penalty (cost_per_error=$0.0200); subtracted 0.072 score pts from raw 39.003 → final 38.931 (percentage).



## Experiment Directory Structure

Experiment directory structure (paths relative to evolution workspace):

```
../../iteration_014/
  agent_<AGENT_NAME>/
    evaluation.json                ← Summary metrics for all examples
    problems/
      <problem_id>/               ← Per-problem directory (symlink if cached)
        result.json               ← Score and metadata for caching
        {key}.md                 ← Diagnostics from evaluator

Agent source code:
  ../../agents/
    <agent_name>/
      agent.py                       ← agent.py
```

**Agents tested (3):**
- agent_iter10_cite_expand (source: ../../agents/iter10_cite_expand/)
- agent_iter13_grade_view (source: ../../agents/iter13_grade_view/)
- agent_iter14_crit_coverage (source: ../../agents/iter14_crit_coverage/)

## Evolution Strategy: Use Your Judgment

# Use Your Judgment

Having reviewed the files provided to you above, your goal is to create a new agent that achieves the highest possible accuracy on problems you haven't seen yet.

**How you do this is entirely up to you.** You may refine an existing agent, combine ideas from multiple agents, or try something completely new. Choose whatever approach you believe will produce the best results based on what you see in the data.

## Your Task

Study the available agents, their performance data, and their failure patterns. Then create a new agent that outperforms all existing agents.

## Required Output Structure

You must create the following files:

### 1. reasoning.md
Your analysis and plan. Explain:
- What you observed in the performance data and agent artifacts
- What approach you chose and why
- Why you believe your agent will achieve higher accuracy

### 2. The artifacts listed in OUTPUT REQUIREMENTS
Create the artifacts specified in the evolution prompt above.

## Your overall goal: Maximize accuracy

Think hard about this. Study the data carefully, understand what works and what doesn't, and build the best agent you can.


## OUTPUT REQUIREMENTS

Create the following files in evolution_output/iteration_015/:

1. **reasoning.md** - Your analysis and improvement strategy
   Must include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

2. **agent.py** - agent.py

**Task objective**: Evolve a PaperFindingBench agent that, given a natural-language literature-search query, returns a list of Semantic Scholar corpus_ids maximizing adjusted micro-F1 against the query's hidden gold, using only the Standard tools (Asta MCP corpus + model_registry LLM handles).

Your primary goal is simple: maximize the score on held-out queries. The scoring function (described in Domain Background in CLAUDE.md) encodes this directly — retrieval quality is the dominant signal, and cost acts as a tiebreaker close to threshold but starts to actively trade off against score farther out. Cost means your LLM spend through `model_registry` handles (tool calls are free), and the free zone is the batch *average*, not per-query: you can spend more on hard queries and less on easy ones. Above $0.06 per query on average, every $0.02 of extra spend costs you one fully-wrong-query-equivalent of score; see the cost-penalty table in Domain Background for the breakeven math.

Each iteration draws a different sample of queries, and the final agent is evaluated on a held-out test set it has never seen. So, to rephrase your goal, your objective is to build an agent that generalizes to unseen queries — the visible batch is a training signal, not the target.

## Round 1: Analysis, Planning, and Implementation

Based on the evolution strategy guidance above, please complete both steps below.

### Step 1: Analysis and Planning

1. Analyze the provided data (agent performance, errors, patterns)
2. Develop a strategic improvement plan
3. Document your reasoning and planned changes

**Error Analysis Available:**
- Previous iteration error analysis: `../../iteration_014/error_analysis_report.md`

Create a file called `reasoning.md` with your analysis and plan.
Include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

1. `agent.py` - agent.py

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
