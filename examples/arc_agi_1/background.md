You are optimizing an ARC-AGI solving agent.

ARC-AGI task format:
- Each task has training examples (input/output pairs) and test inputs
- The (multi) agent(s) must infer the transformation pattern from training examples
- Competition allows maximum of 2 parallel output attempts per test input (pass if either matches)
- You can also use up to {max_llm_calls} LLM calls to solve the problem.
- Freely explore diverse strategies like multi agent systems, ensembles, voting, etc.

LLM cost:
- You are allowed to build an agent system with up to {max_llm_calls} LLM calls.

A per-problem cost budget of ${cost_budget} is enforced. Correct answers within budget score 1.0. Correct answers that exceed the budget are penalized to 0.9 (a 10% reduction). Incorrect answers score 0.0 regardless of cost.

The agent receives:
- train_in, train_out: Training examples (list of 2D grids)
- test_in: Test inputs (no ground truth given to agent)
- llm: Callable for LLM queries with token/call tracking. Signature: llm(prompt, temperature=1.0). temperature controls randomness (0.0 = deterministic, 1.0 = creative).

The agent must return:
{
    "train": [grid, ...],           # 1 prediction per train example
    "test": [[grid, grid], ...],    # up to 2 attempts per test example
}

We evaluate on both training (training_score) and test (test_score with 2 attempts).

Diagnostics: Any print() output from the agent is captured and included in evaluation diagnostics as agent_stdout. Use print() to log any information you think would be helpful for you to see in improving the agent in later rounds of testing and refinement.

## Scratch space

Your working directory is your iteration's evolution dir. If you need to drop a small test harness or scratch script while debugging, write it there directly — `/tmp` is outside the write scope, and there's no need to clean up after yourself afterwards. Leftover debugging artifacts in the iteration dir are useful for retrospective analysis.
