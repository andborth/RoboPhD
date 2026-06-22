


## Previous Iteration Results (Iteration 7)

### Agent Scores

| Agent | Score | Raw / Total |
|-------|-------|-------------|
| iter3_safe_repair_ds1000 | 95.000 | 19.0/20 |
| iter6_grounded_repair_ds1000 | 85.000 | 17.0/20 |
| iter7_grounded_reconcile_ds100... | 90.000 | 18.0/20 |

**Aggregate notes** (how each Score was derived from Raw / Total):

- **iter3_safe_repair_ds1000**: Mean cost $0.0024 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9500 reported as percentage: 95.000.
- **iter6_grounded_repair_ds1000**: Mean cost $0.0048 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.8500 reported as percentage: 85.000.
- **iter7_grounded_reconcile_ds1000**: Mean cost $0.0064 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.

**iter3_safe_repair_ds1000** failed problems (1): 129
**iter6_grounded_repair_ds1000** failed problems (3): 129, 667, 723
**iter7_grounded_reconcile_ds1000** failed problems (2): 723, 667


## Experiment Directory Structure

Experiment directory structure (paths relative to evolution workspace):

```
../../iteration_007/
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
- agent_iter3_safe_repair_ds1000 (source: ../../agents/iter3_safe_repair_ds1000/)
- agent_iter6_grounded_repair_ds1000 (source: ../../agents/iter6_grounded_repair_ds1000/)
- agent_iter7_grounded_reconcile_ds1000 (source: ../../agents/iter7_grounded_reconcile_ds1000/)

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

Create the following files in evolution_output/iteration_008/:

1. **reasoning.md** - Your analysis and improvement strategy
   Must include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

2. **agent.py** - agent.py

**Task objective**: Evolve a DS-1000 agent that, given a Python data-science problem prompt, produces a `<code>...</code>` block whose contents make the hidden test program's `result` variable match the reference under all hidden test inputs.

Your primary goal is simple: maximize the score on held-out problems. The scoring function (described in Domain Background in CLAUDE.md) encodes this directly — correctness is the dominant signal, and cost acts as a tiebreaker close to threshold but starts to actively trade off against correctness farther out. The free zone is the batch *average*, not per-problem: you can spend more on some problems and less on others. Above $0.05 per problem on average, every $0.01 of extra spend costs you one error-equivalent of score; see the cost-penalty table in Domain Background for the breakeven math.

Each iteration draws a different sample of problems, and the final agent is evaluated on a held-out test set it has never seen. After you construct your agent, you will see how it performs on a different batch of examples within this iteration. You will then have the opportunity to refine your agent so that it can be tested on entirely new batches of examples in future iterations. So, to rephrase your goal, your objective is to build an agent that generalizes to unseen problems — the visible batch is a training signal, not the target.

## Round 1: Analysis, Planning, and Implementation

Based on the evolution strategy guidance above, please complete both steps below.

### Step 1: Analysis and Planning

1. Analyze the provided data (agent performance, errors, patterns)
2. Develop a strategic improvement plan
3. Document your reasoning and planned changes

**Error Analysis Available:**
- Previous iteration error analysis: `../../iteration_007/error_analysis_report.md`

Create a file called `reasoning.md` with your analysis and plan.
Include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

1. `agent.py` - agent.py

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
