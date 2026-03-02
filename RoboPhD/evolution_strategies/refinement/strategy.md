# Refinement Strategy

Having reviewed the files provided to you above, you are building a new agent with the goal of achieving higher accuracy than any previous agent. The strategy you will be using is to adopt one agent as a starting point and to make targeted changes on top of that agent.

So in this strategy you will:

1. **Pick an agent to be your starting point**. This is your call. Maybe this is a current top-performer or maybe an agent where you see an opportunity to fix it so that it would be the top performer.
2. **Copy the agent as a starting point for your new agent**. Copy over the whole structure.
3. **Refine with targeted improvements**. Make specific, focused changes that address identified weaknesses while preserving what works.

## Your Task

Create an evolved agent package by refining a base agent to achieve higher accuracy on problems you haven't seen yet.

## Refinement Approach

### Choosing a Starting Point:
- Pick the agent with the strongest foundation
- Consider agents that are close to top performance but have fixable issues
- Sometimes a mid-ranked agent with a better architecture is a better base than the #1 agent

### Making Targeted Changes:
- Focus on fixing specific failure modes you observe in the data
- Keep changes minimal and purposeful — don't change what already works
- Test your hypotheses: if Agent X fails on type Y problems, target that
- Prioritize high-impact changes over cosmetic improvements

### Benefits of Refinement:
- Preserves proven techniques that already work
- Focuses effort on the highest-leverage improvements
- Reduces risk of regression compared to building from scratch
- Easier to reason about what changed and why

## Required Output Structure

You must create the following files:

### 1. reasoning.md
Your analysis of why your new agent is going to outperform previous agents. In your discussion, please note which agent you used as a starting point and describe how you plan to improve it.

The following should also be included in your reasoning:

#### Performance Analysis
- Review of outputs from best performers
- Analysis of agent performance across problem types
- Identification of strengths and weaknesses

#### Error Analysis
- Analysis of error patterns from available analysis artifacts
- How your analysis of error patterns influenced your design decisions

### 2. The artifacts listed in OUTPUT REQUIREMENTS
Create the artifacts specified in the evolution prompt above. When refining from a base agent:
- Keep what works from the base agent
- Make targeted improvements to address identified weaknesses
- Ensure changes are coherent and well-integrated

## Success Metrics

Your evolved package should:
- Outperform the base agent you started from
- Address specific failure modes identified in the data
- Be maintainable and allow clear identification of what changed

## Your overall goal: Push accuracy higher through targeted refinement

You are an expert in the task domain. Use your knowledge to build an agent package that will achieve higher accuracy than previous agents.

**Focus on making targeted, high-impact improvements** to a proven base agent.

Remember: **Think harder** than you normally would about this. Gather the information you need and use your knowledge and experience to improve accuracy.
