


## Previous Iteration Results (Iteration 5)

### Agent Scores

| Agent | Score | Raw / Total |
|-------|-------|-------------|
| iter2_broad_recall_router | 37.350 | 5.2/14 |
| iter4_judge_sim_ranker | 46.586 | 6.5/14 |
| iter5_cite_verify_deep_evidenc... | 46.363 | 6.6/14 |

**Aggregate notes** (how each Score was derived from Raw / Total):

- **iter2_broad_recall_router**: Mean agent cost $0.0175 within free zone (threshold $0.06); no penalty applied. Raw mean F1 0.3735 reported as percentage: 37.350.
- **iter4_judge_sim_ranker**: Mean agent cost $0.0403 within free zone (threshold $0.06); no penalty applied. Raw mean F1 0.4659 reported as percentage: 46.586.
- **iter5_cite_verify_deep_evidence**: Mean agent cost $0.0625 exceeded threshold $0.06 by $0.0025 = 0.12 errors of penalty (cost_per_error=$0.0200); subtracted 0.880 score pts from raw 47.243 → final 46.363 (percentage).



## Experiment Directory Structure

Experiment directory structure (paths relative to evolution workspace):

```
../../iteration_005/
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
- agent_iter2_broad_recall_router (source: ../../agents/iter2_broad_recall_router/)
- agent_iter4_judge_sim_ranker (source: ../../agents/iter4_judge_sim_ranker/)
- agent_iter5_cite_verify_deep_evidence (source: ../../agents/iter5_cite_verify_deep_evidence/)

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

Create the following files in evolution_output/iteration_006/:

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
- Previous iteration error analysis: `../../iteration_005/error_analysis_report.md`

Create a file called `reasoning.md` with your analysis and plan.
Include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

1. `agent.py` - agent.py

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
