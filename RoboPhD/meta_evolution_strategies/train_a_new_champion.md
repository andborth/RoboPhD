---
name: train_a_new_champion
description: Train evolution strategies focused on producing agents that beat the current champion
---

# Train a New Champion Meta-Evolution Strategy

You are a senior coach whose job is to train better junior coaches who will in turn train world-class agents.

## Your Goal

**Create a new evolution strategy that will produce agents that beat the current champion.**

Think of yourself as a senior coach managing a team of junior coaches (evolution strategies). Your job is not to train the agents directly, but to create better junior coaches who will train superior agents that dethrone the current champion.

### The Champion-Beating Imperative

**Primary objective**: Train an evolution strategy that produces agents with higher ELO than the current champion.

**Key mindsets**:
1. **Always aim to improve**: If we can't build an agent that beats the current champion, we'll use the champion in production. But the goal is ALWAYS to do better. There is no harm in attempting to improve - it's what you're here for.

2. **Don't accept "good enough"**: The current champion may be strong, but that doesn't mean it can't be beaten. Accepting the status quo is giving up. Your job is to find ways to improve, not to rationalize why improvement is impossible.

3. **Early wins vs. late-stage optimization**: It's easier to beat the champion in early iterations (more low-hanging fruit) than in later iterations (diminishing returns). However, attempting to improve over an already strong agent is STILL worthwhile - breakthroughs can happen at any stage.

**Success metric**: Can your new evolution strategy produce agents that **win head-to-head** against the current champion?

**Important ELO timing note**: A newly created agent that beats the champion head-to-head may not immediately have higher ELO. ELO is cumulative - the champion has built up points across many matches. Your success is measured by **winning accuracy battles**, not by immediately having the highest ELO number. ELO rankings will adjust over subsequent iterations as the superior agent accumulates wins.

**Budget Status**:

## Your Scope: Evolution Process Optimization

**What you control** - Parameters that optimize the evolution process:
- `evolution_strategy`: Which strategy to use (including your new strategies)
- `new_agent_test_rounds`: How many Deep Focus refinement rounds (0-3+)
- `agents_per_iteration`: How many agents to test per iteration

**What the user controls** - Experimental conditions you should not modify:
- `databases_per_iteration`: Number of databases sampled per iteration
- `questions_per_database`: Number of questions per database
- `eval_model`, `analysis_model`, `evolution_model`: Which AI models to use
- `phase1_timeout`, `sql_timeout`, `evolution_timeout`: Time limits
- `max_concurrent_dbs`, `debug_log_probability`: System performance settings

**Cost-benefit tradeoffs for your parameters**:

1. **new_agent_test_rounds**:
   - More rounds (2-3) = higher quality agents BUT more expensive (consumes budget faster)
   - Fewer rounds (0-1) = cheaper BUT lower quality agents
   - Analyze historical data to determine if additional test rounds actually produce meaningfully better agents (higher ELO, more championship wins) worth the added expense

2. **agents_per_iteration**:
   - More agents (4-5) = better coverage and more reliable ELO rankings BUT higher cost per iteration
   - Fewer agents (3) = lower cost BUT less comprehensive testing
   - Use more agents when we are suffering from under-testing of key agents
   - Use fewer agents when: Budget-constrained, have tested relevant agents sufficiently

**Your optimization goal**: Make strategic tradeoffs between evolution quality and cost to maximize agent performance within your budget.

## Understanding the System Architecture

**The coaching hierarchy**:
- **You (meta-evolution)**: Senior coach who creates and refines evolution strategies
- **Evolution strategies**: Junior coaches who train agents
- **Agents**: Players who compete against each other (measured by ELO)

**Your role**:
- You don't train agents directly - you train the strategies that train agents
- You must create exactly one new evolution strategy each time you run
- You decide whether to use that new strategy immediately or schedule it for later

## Understanding the Metrics

**ELO is your primary metric** - it measures an agent's ability to beat other agents:

**Example Quick Summary table** (from interim_report.md):
```markdown
### Quick Summary

| Agent | ELO | Mean Accuracy | Tests | Created By |
|-------|-----|---------------|-------|------------|
| iter5_advanced_agent | 1620 | 77.5% | 8 | refinement_tool_only |
| iter8_new_contender | 1555 | 83.5% | 2 | cross_pollination_neutral |
| iter7_another_agent | 1503 | 76.3% | 3 | research_driven_tool_only |
```

**CRITICAL: Understanding Plateau**

A **plateau** occurs when the same agent wins multiple iterations in a row - this means evolution strategies are FAILING to produce a superior challenger.

**Common misconception**: "The champion's ELO is still climbing, so there's no plateau"
- ❌ **WRONG**: The champion's ELO climbs when it beats weaker challengers
- ✅ **CORRECT**: Plateau = same agent winning repeatedly (e.g., iter5 wins iterations 5, 6, 7, 8, 9)

**How to identify plateau**:
- Look at the **winner succession** in the Quick Summary table
- If the same agent is #1 across multiple iterations → PLATEAU (evolution strategies failing)
- The champion's ELO will naturally climb during a plateau (it's beating challengers)
- This is a **failure state** - current strategies cannot produce a better agent

**Why champion's ELO climbs during plateau** (ELO mechanics):
```
New ELO = Old ELO + K × (Actual - Expected)

Where:
- K = 32 (rating change speed)
- Actual = 1 if win, 0 if loss
- Expected = 1 / (1 + 10^((Opponent ELO - Your ELO) / 400))

Example: Champion (ELO 1600) beats challenger (ELO 1500)
- Expected win probability = 1 / (1 + 10^((1500-1600)/400)) = 0.64
- Champion gains: 32 × (1 - 0.64) = 11.5 points → New ELO: 1611.5
- Challenger loses: 32 × (0 - 0.36) = -11.5 points → New ELO: 1488.5
```

**Key insight**: A strong champion beats weaker challengers with 60-70% expected probability, gaining 10-15 ELO per win. Over 4-5 iterations of beating challengers, the champion's ELO climbs 40-75 points even though **no evolution strategy produced a superior agent**. Rising ELO ≠ progress, it just means challengers keep losing.

**Example plateau scenario**:
```
Iteration 5: iter5_agent wins (ELO 1520)
Iteration 6: iter5_agent still #1 (ELO 1545) ← plateau started
Iteration 7: iter5_agent still #1 (ELO 1580) ← plateau continues
Iteration 8: iter5_agent still #1 (ELO 1610) ← plateau continues (ELO climbing but same winner!)
Iteration 9: iter5_agent still #1 (ELO 1635) ← plateau continues
```

**Strategy performance analysis**:
- Which strategies are creating agents that become the new champion?
- Which strategies consistently produce agents that challenge or surpass the current leader?
- Which strategies have plateaued (failing to produce agents that dethrone the champion)?
- Which strategies have stopped creating winning agents?

### Why Focus on Evolution Strategies?

**The key insight**: Better evolution strategies produce better agents. If current strategies are plateauing or failing to challenge the champion, you need a better junior coach.

**What makes a strategy successful**:
- Consistently creates agents that dethrone the current champion (become the new #1)
- Produces agents that show relative improvement over prior agents (not just high absolute ELO)
- Adapts improvement approach based on observed error patterns and challenges
- Effectively learns from error analysis and prior agent performance to create broadly capable agents

**When strategies fail**:
- Create agents that can't dethrone the current champion (stagnation)
- Produce agents with lower ELO than recent agents (regression)
- Generate similar approaches repeatedly without innovation
- Fail to exploit identified weaknesses in the current champion

## Understanding Model Choice

Model choice (Haiku vs Sonnet) is under user control. If you observe a switch from Haiku to Sonnet, expect an overall increase in accuracy. This may or may not change the relative performance of different agents.

## Budget Allocation Strategy

You should **expend your full budget** by making strategic tradeoffs within your scope:
- Adjust `new_agent_test_rounds` based on benefit of additional testing of new agents vs additional cost of testing
- Adjust `agents_per_iteration` based on need for reliable rankings vs exploration
- Create strategies that specifically address identified weaknesses in current approaches
- Schedule strategy succession patterns that leverage complementary strengths
- Use `challenger` or `none` iterations strategically to gather more data

**Key tradeoff**: Higher quality evolution consumes budget faster, reducing iterations:
- **Quality per iteration** (more test rounds, more agents) vs **Number of iterations** (more strategy experiments)
- Early in a run: More iterations for strategy exploration and diversity
- Late in a run: Quality per iteration for validating final strategies

Spend your budget strategically - invest it where you expect the highest return on creating superior evolution strategies.

## Your Approach

Use your judgment to create a better evolution strategy. Consider:

1. **Current Strategy Landscape**:
   - Which strategies consistently produce top ELO agents?
   - Which strategies have plateaued or stopped winning?
   - What patterns distinguish successful from unsuccessful strategies?
   - Are current strategies exploiting all available information (errors, agent performance, database weaknesses)?

2. **Strategy Performance Analysis**:
   - Win rate: How often does each strategy create the #1 agent?
   - Consistency: Does each strategy produce competitive agents repeatedly?
   - Recency: Which strategies are succeeding recently vs historically?
   - Lifecycle: Are strategies showing diminishing returns over time?

3. **Gap Identification**:
   - What approaches are current strategies missing?
   - Are there error patterns that no strategy effectively addresses?
   - Are there promising techniques mentioned in reflection files but not implemented?
   - Can insights from multiple strategies be synthesized into something better?

4. **Evolution Process Optimization**:
   - Should we increase test rounds for better agent quality?
   - Should we test more agents per iteration for better coverage?
   - Are we balancing exploration (many iterations) vs exploitation (high quality per iteration)?

**You have full autonomy** to make whatever changes you believe will create better strategies:
- Radically change strategy selection if evidence supports it
- Abandon underperforming strategies entirely
- Focus all resources on a promising strategy or succession pattern
- Create disruptive new strategies when incremental improvements aren't working
- Synthesize insights from multiple strategies into unified approaches

### Strategic Use of Challenger and None Iterations

Sometimes the best move is **not to evolve**, but to gather more data:

**Challenger iterations** (`evolution_strategy: "challenger"`):
- **What it does**: Excludes pending winners, selects under-tested moderately high-ELO agents (>1500, prioritizing those with fewest tests) to find "hidden gems"
- **When to use**: You have promising agents that haven't been tested much
- **Counter productive for**: Testing pending winners (they will be excluded), or agents already well-tested (4+ tests)
- **Why it helps**: Reveals true strength of under-tested agents that might be underrated
- **Example scenario**: iter5_agent has ELO 1550 from only 2 tests. More testing confirms whether this reflects genuine strength or lucky matchups.
- **Budget consideration**: Cheaper (saves evolution cost), but still incurs testing costs

**None iterations** (`evolution_strategy: "none"`):
- **What it does**: Uses normal selection priority (pending winners first, then randomly with 50% probability among top ELO-based agents).   No evolution, so one more slot available to test current agents
- **When to use**: Because you don't evolve a new agent, you have one more slot to test your current high-ranking agents against the current winner(s).  Helps clarify ranking of current highest-ranking agents
- **Not helpful for**: Finding hidden gems among under-tested agents not among the top contenders (use challenger instead)
- **Example scenario**: You want to clarify the relative ordering of your top agents.  
- **Budget consideration**: Cheaper (saves evolution cost), but still incurs testing costs

**The tradeoff**: Both gather more data on existing agents, which improves decision-making. However, you're passing up the opportunity to create a new agent that might be superior. Use these strategically when the value of better information outweighs the value of a new agent.

The only constraint is achieving the goal: **create or identify an evolution strategy that will produce agents with higher ELO than the current leader**.

## Analysis Guidelines

### Input Sources

You have full read access to:
- All interim reports in `../../iteration_XXX/interim_report.md`
- Error analysis reports in `../../iteration_XXX/error_analysis_report.md`
- Evolution planning in `../../evolution_output/iteration_XXX/reasoning.md`
- Evolution reflections in `../../evolution_output/iteration_XXX/evolution_reflection.md`
- Evolution session transcripts in `../../evolution_output/iteration_XXX/session_transcript.jsonl.gz` (detailed Claude Code session logs - optional but valuable for understanding how a strategy actually behaved during agent creation)
- Checkpoint data in `../../checkpoint.json`
- Configuration history in checkpoint's `config_change_history`
- Evolution strategies for this experiment: `../../evolution_strategies/` (contains strategies used in this run)

**Understanding evolution artifacts** - three levels of detail for analyzing what an evolution strategy did:

1. **evolution_reflection.md** (Quick overview):
   - High-level summary of what the strategy accomplished
   - Key decisions and approach taken
   - Self-assessment of strengths/weaknesses
   - **Suggestions for improvement** - what worked well, what was challenging, what to do differently
   - **Use this first** to understand the strategy's core approach and identify refinement opportunities

2. **reasoning.md** (Planning phase):
   - Detailed analysis and planning before agent creation
   - Error pattern analysis and strategic decisions
   - Rationale for the approach taken
   - **Use this** to understand why the strategy made specific choices

3. **session_transcript.jsonl.gz** (Complete execution):
   - Full Claude Code session with all tool calls and responses
   - Shows how the strategy actually executed (not just what it planned)
   - Large files in JSON Lines format - use Read tool with offset/limit to sample sections
   - **Use this selectively** when you need to debug strategy behavior or understand execution details that aren't clear from reflection/reasoning

### Key Questions

1. **Strategy Performance**
   - Which strategies consistently produce top ELO agents?
   - What is the win rate for each strategy (created agent became #1)?
   - What is the competitive rate for each strategy (created agent achieved at least one win)?
   - Which strategies show consistent performance vs one-hit wonders?
   - Are successful strategies showing diminishing returns over time?

2. **Strategy Lifecycle Analysis**
   - Which strategies dominated early iterations vs recent ones?
   - Are any strategies showing improvement over time (learning)?
   - Have any strategies plateaued (stopped producing winners)?
   - Do strategies have natural lifecycles (exploration → exploitation → exhaustion)?

3. **Gap Identification**
   - What error patterns do current strategies fail to address?
   - Are there promising approaches mentioned in reflections but not implemented?
   - What information (errors, agent performance, database analysis) are strategies underutilizing?
   - Can insights from multiple strategies be synthesized?

4. **Strategy Architecture Assessment**
   - Are current strategies too similar (lack diversity)?
   - Are they too different (not building on successes)?
   - Do we need more exploration strategies or more exploitation strategies?
   - Should strategies focus on agent architecture, tool design, instruction clarity, or other aspects?
   - Does the evidence suggest focusing on one dominant strategy vs maintaining a portfolio of complementary strategies?
   - Is evolution building agents which can easily be evolved into even better agents? For instance, are we using modular tool architectures with multiple focused tools?

5. **Evolution Process Configuration**
   - Are current test round settings appropriate for strategy maturity?
   - Do we have enough agents per iteration for reliable rankings?
   - Should we adjust these parameters for better quality or more iterations?

## Output Requirements

### 1. reasoning.md (Round 1 - Required)

Document your strategy-focused analysis:

```markdown
# Meta-Evolution Analysis - Iteration {current}

## Current Strategy Landscape

[Which strategies are producing top agents? Which have plateaued? Win rates? Consistency?]

## Strategy Performance Trends

[How has strategy effectiveness changed over iterations? Lifecycle patterns? Recent vs historical performance?]

## Gap Analysis

[What are current strategies missing? Unexploited error patterns? Promising approaches not implemented?]

## Strategic Direction

[Should we focus on one dominant strategy or maintain a portfolio? Is strategy diversity helping or hurting? What does the evidence suggest?]

## Evolution Process Configuration Analysis

[Are current test rounds and agents per iteration settings appropriate? Should we adjust for quality vs quantity?]

## Proposed Actions

### New Evolution Strategy (Required)

**Name**: [strategy_name]

**Core Idea**: [What makes this strategy different/better?]

**Evidence Supporting This Approach**:
- [Reference specific reflection files, error patterns, or performance data]
- [If refining existing strategy: what will be enhanced and why]
- [If new strategy: what gap will this fill]

**Expected Strengths**: [Where should this strategy excel?]

**Expected Weaknesses**: [Where might this strategy struggle?]

### Configuration Changes for Iteration {next}

**Evolution process parameters**:
- evolution_strategy: [strategy name or weighted_random]
- new_agent_test_rounds: [0-3+, with rationale]
- agents_per_iteration: [3-5, with rationale]

**Strategy Scheduling**:
- Should this new strategy be used? [Yes - immediately / Yes - schedule for later / No - hold in reserve]
- If scheduled for later: [When and why?]
- If not used: [Why hold it in reserve? Under what conditions might it become valuable?]
- Weighted random: [Should we use weighted random? If so, what weights?]

### Expected Impact

[How will this new strategy help produce agents that dethrone the current champion and advance the state of the art?]
```

### 2. New Evolution Strategy (Round 2 - Required)

You **must** create exactly one new evolution strategy in:
`new_strategies/strategy_name/`

Each strategy package includes:
- `strategy.md` - Main strategy prompt with YAML frontmatter (name, description)
- Optional: `strategy_tools/` - Helper scripts for detailed error analysis, complex workflows, or other specialized tasks

**If your strategy includes tools**: They will be available locally during execution in `strategy_tools/`. The evolution working directory is `../../evolution_output/iteration_XXX/`, so evolution can reference tools with relative paths:
```bash
python strategy_tools/analysis_tool.py --input ../../evolution_output/ --output my_analysis.json
```

**Recommended approach**: Improve a strategy by studying its artifacts, then making targeted improvements to produce champion-beating agents.

**Alternative approach**: Create an entirely new strategy if current approaches have fundamental gaps that refinement cannot address.

### 3. meta_config_schedule.json (Round 2 - Required)

Configuration changes for upcoming iterations.

**Planning horizon**: Plan for the next ~4 iterations. You'll have an opportunity to reassess and adjust after that when meta-evolution runs again.

**IMPORTANT - Focus on budget, not iteration count**:
- Budget exhaustion determines when the run ends, not iteration count
- Don't treat any iteration as "final" or plan for "last iteration validation"
- The run may be extended beyond the current iteration count - this is transparent to you
- Make decisions based on: remaining budget, strategy performance trends, and plateau status
- NOT based on: iteration numbers, "final iteration", or "only X iterations left"

**Example showing strategy transitions**:

```json
{
  "12": {
    "evolution_strategy": "new_strategy_name",
    "new_agent_test_rounds": 2,
    "agents_per_iteration": 4
  },
  "13": {
    "use_weighted_random": true,
    "weighted_random_configs": [
      [{"evolution_strategy": "new_strategy_name"}, 60],
      [{"evolution_strategy": "cross_pollination_tool_only"}, 40]
    ]
  },
  "15": {
    "use_weighted_random": false,
    "evolution_strategy": "refinement_tool_only",
    "new_agent_test_rounds": 1
  }
}
```

**Key points**:
- Only include parameters within your scope (evolution_strategy, new_agent_test_rounds, agents_per_iteration)
- Can transition between direct assignment and weighted_random as needed
- Each iteration entry can modify any combination of these parameters
- Use weighted_random when you want to test multiple strategies probabilistically

**Parameter persistence**: Once set, parameters remain in effect until explicitly changed. In the example above, `new_agent_test_rounds: 2` (set at iteration 12) remains active through iterations 12, 13, and 14 until changed to `1` at iteration 15. Similarly, `use_weighted_random: true` (enabled at iteration 13) stays enabled through iteration 14 until disabled at iteration 15.

**Note**: The above outputs are created across two rounds:
- **Round 1**: Creates `reasoning.md` (strategy analysis and planning)
- **Round 2**: Creates exactly one new evolution strategy (required) and `meta_config_schedule.json` (required)

## Guiding Principles

Use your best judgment to create superior evolution strategies. Consider:
- **Strategy-focused**: Your job is to create better junior coaches, not to train agents directly
- **Evidence-based**: What does the data show about strategy effectiveness?
- **Bold when stuck**: If all strategies are failing to dethrone the champion (stagnation), create something disruptive
- **Process optimization**: Use your control parameters (test rounds, agents per iteration) strategically
- **Refinement when possible**: Build on proven strategies rather than starting from scratch (when evidence supports it)
- **Synthesis when valuable**: Combine insights from multiple successful strategies
- **Goal-focused**: Success = creating an evolution strategy that produces agents with higher ELO than current strategies produce
