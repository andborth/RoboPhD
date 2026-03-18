# Data-Driven Discovery

Your goal is to create a new agent that achieves the highest possible accuracy on problems you haven't seen yet.

**Start from the data, not from the agents.** Before reading any agent code, explore the actual problem data to understand what drives performance.

## Step 1: Explore the Problem Data

Read problem-level outputs from recent iterations. Look at:
- Per-problem diagnostics and summaries in `../../iteration_NNN/agent_<name>/problems/<id>/`
- The error analysis report at `../../iteration_NNN/error_analysis_report.md`
- Score comparisons across agents on the same problems

Focus on understanding:
- **What makes a problem easy vs hard?** Look for patterns in problem characteristics (encoded in problem IDs, input parameters, or diagnostic files) that predict cost/score.
- **Where do agents diverge?** Find the specific problems where agents produce different scores. These are the problems where your improvements will matter.
- **What categorical structure exists?** Are there distinct problem regimes that might benefit from different strategies?

Read at least 5-10 problem-level outputs across different problem types before proceeding.

## Step 2: Form Hypotheses

Based on your data exploration, write down:
- What patterns distinguish high-scoring from low-scoring problems
- Whether different problem categories might need different approaches
- What specific failure modes you observed in the data

## Step 3: Study Existing Agents (Briefly)

Now read the top 2-3 agents' source code. But read them through the lens of your data hypotheses:
- Does the agent's logic address the patterns you found?
- What opportunities does it miss?
- Which problem categories does it handle well vs poorly?

## Step 4: Build Your Agent

Create a new agent informed by your data insights. Your agent should:
- Address the categorical patterns you discovered
- Handle different problem regimes appropriately
- Be grounded in specific observations, not generic improvements

## Required Output Structure

### 1. reasoning.md
Your analysis and plan. Structure it as:
- **Data observations**: What you found in the problem-level data (with specific examples)
- **Hypotheses**: What patterns you identified and why they matter
- **Agent design**: How your agent addresses these patterns
- Why you believe your agent will achieve higher accuracy

### 2. The artifacts listed in OUTPUT REQUIREMENTS
Create the artifacts specified in the evolution prompt above.

## Your overall goal: Discover structure in the data that existing agents miss

The biggest improvements come from understanding the problem domain better, not from tweaking existing code. Let the data guide you.
