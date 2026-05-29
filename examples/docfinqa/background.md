The document is a complete SEC filing in clean markdown with tables preserved. Documents average 123K words (~250 pages); the relevant information is typically in a single section or table. Questions require numerical reasoning: ratios, differences, percentages, averages, and multi-step arithmetic.

Available tools:
  llm(prompt, temperature=0.0) -> str : Call a language model. temperature controls randomness (0.0 = deterministic, 1.0 = creative). Expensive (~$0.003-0.01 per call).
  embed(text) -> list[float] : Embed text for similarity search. Cheap (~$0.0001 per call).

Scoring: The program's `answer` variable is compared to the expected answer numerically with 1% relative tolerance. Unit labels (%, $, commas) are stripped before comparison, so the program should assign a raw number to `answer` (e.g., `answer = 36.5`, not `answer = '36.5%'`).

A per-problem cost budget of $0.10 is enforced. Correct answers within budget score 1.0. Correct answers that exceed the budget are penalized to 0.9 (a 10% reduction). Incorrect answers score 0.0 regardless of cost. The program output is executed via exec(); if it raises an exception the answer is counted as incorrect.

Diagnostics: Any print() output from the agent is captured and included in evaluation diagnostics as agent_stdout. Use print() to log any information you think would be helpful for you to see in improving the agent in later rounds of testing and refinement.
