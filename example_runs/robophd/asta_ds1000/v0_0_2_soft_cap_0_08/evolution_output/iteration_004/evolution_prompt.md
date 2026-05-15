


## Previous Iteration Results (Iteration 3)

### Agent Scores

| Agent | Score | Raw / Total |
|-------|-------|-------------|
| iter2_ds1000_verify_repair | 75.000 | 15.0/20 |
| iter3_ds1000_format_aware | 80.000 | 16.0/20 |
| seed_yyg6m9ud | 65.000 | 13.0/20 |

**Aggregate notes** (how each Score was derived from Raw / Total):

- **iter2_ds1000_verify_repair**: Mean cost $0.0051 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.7500 reported as percentage: 75.000.
- **iter3_ds1000_format_aware**: Mean cost $0.0096 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.
- **seed_yyg6m9ud**: Mean cost $0.0005 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.6500 reported as percentage: 65.000.

**iter2_ds1000_verify_repair** failed problems (5): 706, 165, 269, 420, 723
**iter3_ds1000_format_aware** failed problems (4): 165, 269, 420, 706
**seed_yyg6m9ud** failed problems (7): 706, 763, 887, 269, 165, 420, 426


## Experiment Directory Structure

Experiment directory structure (paths relative to evolution workspace):

```
../../iteration_003/
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
- agent_iter2_ds1000_verify_repair (source: ../../agents/iter2_ds1000_verify_repair/)
- agent_iter3_ds1000_format_aware (source: ../../agents/iter3_ds1000_format_aware/)
- agent_seed_yyg6m9ud (source: ../../agents/seed_yyg6m9ud/)

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

Create the following files in evolution_output/iteration_004/:

1. **reasoning.md** - Your analysis and improvement strategy
   Must include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

2. **agent.py** - agent.py

**Task objective**: Evolve a DS-1000 agent that, given a Python data-science problem prompt, produces a `<code>...</code>` block whose contents make the hidden test program's `result` variable match the reference under all hidden test inputs.

Your primary goal is simple: maximize the score on held-out problems. The scoring function (described in Domain Background in CLAUDE.md) is designed to encode this directly — correctness is the dominant signal and cost acts only as a tiebreaker between agents tied on correctness. Don't worry about costs as long as your **mean** cost across the iteration's batch stays below $0.08 — the free zone applies to the batch average, not to individual problems. You can spend more on some problems and less on others; only the average matters. Above the threshold, feel free to spend as much as you want if you see an opportunity to solve more problems. As an example, an agent averaging $0.16/problem across the batch will always beat one averaging $0.07/problem if it converts even one answer from wrong to right.

Each iteration draws a different sample of problems, and the final agent is evaluated on a held-out test set it has never seen. After you construct your agent, it will be tested on entirely new batches of examples in future iterations. So, to rephrase your goal, your objective is to build an agent which gets the highest possible score on problems you haven't seen before.

## Round 1: Analysis, Planning, and Implementation

Based on the evolution strategy guidance above, please complete both steps below.

### Step 1: Analysis and Planning

1. Analyze the provided data (agent performance, errors, patterns)
2. Develop a strategic improvement plan
3. Document your reasoning and planned changes

**Error Analysis Available:**
- Previous iteration error analysis: `../../iteration_003/error_analysis_report.md`

Create a file called `reasoning.md` with your analysis and plan.
Include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

1. `agent.py` - agent.py

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
