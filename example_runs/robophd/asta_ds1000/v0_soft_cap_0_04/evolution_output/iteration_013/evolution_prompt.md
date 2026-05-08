


## Previous Iteration Results (Iteration 12)

### Agent Scores

| Agent | Score | Score Sum / Total |
|-------|-------|-------------------|
| iter12_version_aware_judge | 89.990 | 1799.8/20 |
| iter8_robust_judge | 95.000 | 1900.0/20 |
| iter9_mpl_aware_judge | 95.000 | 1900.0/20 |



## Experiment Directory Structure

Experiment directory structure (paths relative to evolution workspace):

```
../../iteration_012/
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
- agent_iter12_version_aware_judge (source: ../../agents/iter12_version_aware_judge/)
- agent_iter8_robust_judge (source: ../../agents/iter8_robust_judge/)
- agent_iter9_mpl_aware_judge (source: ../../agents/iter9_mpl_aware_judge/)

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

Create the following files in evolution_output/iteration_013/:

1. **reasoning.md** - Your analysis and improvement strategy
   Must include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

2. **agent.py** - agent.py

**Task objective**: Evolve a DS-1000 agent that, given a Python data-science problem prompt, produces a `<code>...</code>` block whose contents make the hidden test program's `result` variable match the reference under all hidden test inputs.

Your primary goal is simple: maximize the per-example score on held-out problems. The scoring function (described in Domain Background in CLAUDE.md) is designed to encode this directly — correctness is the dominant signal and cost acts only as a tiebreaker between agents tied on correctness. Don't worry at all about per-example costs below $0.04 — those sit in the free zone with no penalty. Above $0.04, feel free to go as high as you want if you see an opportunity to solve more problems. As an example, a $0.10/problem agent will always beat a $0.03/problem agent if it converts even one answer from wrong to right.

Each iteration draws a different sample of problems, and the final agent is evaluated on a held-out test set it has never seen. After you construct your agent, it will be tested on entirely new batches of examples in future iterations. So, to rephrase your goal, your objective is to build an agent which gets the highest possible score on problems you haven't seen before.

## Round 1: Analysis, Planning, and Implementation

Based on the evolution strategy guidance above, please complete both steps below.

### Step 1: Analysis and Planning

1. Analyze the provided data (agent performance, errors, patterns)
2. Develop a strategic improvement plan
3. Document your reasoning and planned changes

**Error Analysis Available:**
- Previous iteration error analysis: `../../iteration_012/error_analysis_report.md`

Create a file called `reasoning.md` with your analysis and plan.
Include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

1. `agent.py` - agent.py

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
