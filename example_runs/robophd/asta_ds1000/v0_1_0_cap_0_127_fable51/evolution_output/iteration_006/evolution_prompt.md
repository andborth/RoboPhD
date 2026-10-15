


## Previous Iteration Results (Iteration 5)

### Agent Scores

| Agent | Score | Raw / Total |
|-------|-------|-------------|
| iter2_exec_verified_sonnet | 100.000 | 20.0/20 |
| iter3_ensemble_exec_vote | 100.000 | 20.0/20 |
| iter5_ensemble_exec_vote_dtype... | 100.000 | 20.0/20 |

**Aggregate notes** (how each Score was derived from Raw / Total):

- **iter2_exec_verified_sonnet**: Mean cost $0.0134 within free zone (threshold $0.127); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.
- **iter3_ensemble_exec_vote**: Mean cost $0.0450 within free zone (threshold $0.127); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.
- **iter5_ensemble_exec_vote_dtype_judge**: Mean cost $0.0480 within free zone (threshold $0.127); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.



## Experiment Directory Structure

Experiment directory structure (paths are relative to your shell's working directory — your session's workspace directory under `evolution_output/`, two levels below the experiment root):

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

Two sibling trees share iteration numbering — don't conflate them:
- `../../iteration_005/` — evaluation results (the tree above)
- `../../evolution_output/iteration_005/` — that iteration's evolution-session workspace (`reasoning.md`, `evolution_reflection.md`); your own cwd is the current iteration's directory in this tree

**Agents tested (3):**
- agent_iter2_exec_verified_sonnet (source: ../../agents/iter2_exec_verified_sonnet/)
- agent_iter3_ensemble_exec_vote (source: ../../agents/iter3_ensemble_exec_vote/)
- agent_iter5_ensemble_exec_vote_dtype_judge (source: ../../agents/iter5_ensemble_exec_vote_dtype_judge/)

## Prior Evolution Sessions

The evolution session that created iteration 5's agent recorded its thinking:

- `evolution_output/iteration_005/reasoning.md` — that session's analysis and plan
- `evolution_output/iteration_005/evolution_reflection.md` — its advice for future sessions

Earlier sessions (iterations 2–4) left the same artifacts under their own `evolution_output/iteration_NNN/` directories.

These prior session analyses are available to you. Feel free to form your own opinions.

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

**Task objective**: Evolve a DS-1000 agent that, given a Python data-science problem prompt, produces a `<code>...</code>` block whose contents make the hidden test program's `result` variable match the reference under all hidden test inputs.

Your primary goal is simple: maximize the score on held-out problems. The scoring function (described in Domain Background in CLAUDE.md) encodes this directly — correctness is the dominant signal, and cost acts as a tiebreaker close to threshold but starts to actively trade off against correctness farther out. The free zone is the batch *average*, not per-problem: you can spend more on some problems and less on others. Above $0.127 per problem on average, every $0.0127 of extra spend costs you one error-equivalent of score; see the cost-penalty table in Domain Background for the breakeven math.

Each iteration draws a different sample of problems, and the final agent is evaluated on a held-out test set it has never seen. After you construct your agent, it will be tested on entirely new batches of examples in future iterations. So, to rephrase your goal, your objective is to build an agent that generalizes to unseen problems — the visible batch is a training signal, not the target.

## Round 1: Analysis, Planning, and Implementation

**Time budget:** this session is capped at 60 minutes of wall clock.

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
