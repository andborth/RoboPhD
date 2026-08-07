


## Previous Iteration Results (Iteration 13)

### Agent Scores

| Agent | Score | Raw / Total |
|-------|-------|-------------|
| iter10_deadline_guard | 41.839 | 6.0/14 |
| iter12_salvage_rank | 41.741 | 5.8/14 |
| iter13_any_author_gate | 42.398 | 5.9/14 |

**Aggregate notes** (how each Score was derived from Raw / Total):

- **iter10_deadline_guard**: Mean agent cost $0.0637 exceeded threshold $0.063 by $0.0007 = 0.11 errors of penalty (cost_per_error=$0.0063); subtracted 0.768 score pts from raw 42.607 → final 41.839 (percentage).
- **iter12_salvage_rank**: Mean agent cost $0.0578 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.4174 reported as percentage: 41.741.
- **iter13_any_author_gate**: Mean agent cost $0.0603 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.4240 reported as percentage: 42.398.



## Experiment Directory Structure

Experiment directory structure (paths are relative to your shell's working directory — your session's workspace directory under `evolution_output/`, two levels below the experiment root):

```
../../iteration_013/
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
- `../../iteration_013/` — evaluation results (the tree above)
- `../../evolution_output/iteration_013/` — that iteration's evolution-session workspace (`reasoning.md`, `evolution_reflection.md`); your own cwd is the current iteration's directory in this tree

**Agents tested (3):**
- agent_iter10_deadline_guard (source: ../../agents/iter10_deadline_guard/)
- agent_iter12_salvage_rank (source: ../../agents/iter12_salvage_rank/)
- agent_iter13_any_author_gate (source: ../../agents/iter13_any_author_gate/)

Session helper scripts (read-only; run with e.g. `python ../../session_tools/<script>`): tool_probe.py

## Prior Evolution Sessions

The evolution session that created iteration 13's agent recorded its thinking:

- `evolution_output/iteration_013/reasoning.md` — that session's analysis and plan
- `evolution_output/iteration_013/evolution_reflection.md` — its advice for future sessions

Earlier sessions (iterations 2–12) left the same artifacts under their own `evolution_output/iteration_NNN/` directories.

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

Create the following files in evolution_output/iteration_014/:

1. **reasoning.md** - Your analysis and improvement strategy
   Must include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

2. **agent.py** - agent.py

**Task objective**: Evolve a PaperFindingBench agent that, given a natural-language literature-search query, returns a list of Semantic Scholar corpus_ids maximizing adjusted micro-F1 against the query's hidden gold, using only the Standard tools (Asta MCP corpus + model_registry LLM handles).

Your primary goal is simple: maximize the score on held-out queries. The scoring function (described in Domain Background in CLAUDE.md) encodes this directly — retrieval quality is the dominant signal, and cost acts as a tiebreaker close to threshold but starts to actively trade off against score farther out. Cost means your LLM spend through `model_registry` handles (tool calls are free), and the free zone is the batch *average*, not per-query: you can spend more on hard queries and less on easy ones. Above $0.063 per query on average, every $0.0063 of extra spend costs you one fully-wrong-query-equivalent of score; see the cost-penalty table in Domain Background for the breakeven math.

Each iteration draws a different sample of queries, and the final agent is evaluated on a held-out test set it has never seen. So, to rephrase your goal, your objective is to build an agent that generalizes to unseen queries — the visible batch is a training signal, not the target.

## Round 1: Analysis, Planning, and Implementation

**Time budget:** this session is capped at 60 minutes of wall clock.

Based on the evolution strategy guidance above, please complete both steps below.

### Step 1: Analysis and Planning

1. Analyze the provided data (agent performance, errors, patterns)
2. Develop a strategic improvement plan
3. Document your reasoning and planned changes

**Error Analysis Available:**
- Previous iteration error analysis: `../../iteration_013/error_analysis_report.md`

Create a file called `reasoning.md` with your analysis and plan.
Include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

1. `agent.py` - agent.py

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
