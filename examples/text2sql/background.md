## BIRD Benchmark Text2SQL (Integrated Agent)

The Text2SQL domain generates SQL queries from natural language questions
against the BIRD benchmark.

### Architecture: analyze_db.py + agent.py

```
Phase 1 (Tool): analyze_db.py examines database.sqlite
  -> Produces schema analysis text (cached per code+database)

Phase 2 (Agent): agent.py receives analysis + question + callables
  -> Uses llm() and test_sql() to generate and refine SQL
  -> Returns final SQL string for scoring

Scoring: set(predicted_results) == set(ground_truth_results)
```

### What Evolution Controls

The evolved agent consists of two files:

1. **`analyze_db.py`** — Database analysis script.
   Reads `database.sqlite` from its working directory, performs schema
   analysis, and writes findings to `analysis.txt`.
   Runs as a subprocess. Common techniques: DDL extraction, sample data,
   foreign key mapping, column statistics.

2. **`agent.py`** — SQL generation agent with a `solve()` function.
   Receives the analysis output, the question, and two callables:
   - `llm(prompt)` — call the eval LLM (haiku-4.5), returns response text
   - `test_sql(sql)` — execute SQL against the database, returns formatted
     results string or error message. Limited to 5 calls per question.
   The agent returns the final SQL string to submit for scoring.

### Cost Budget

Per-question cost budget of $0.10 is enforced. Correct answers within budget
score 1.0. Correct answers that exceed the budget are penalized to 0.9
(a 10% reduction). Incorrect answers score 0.0 regardless of cost.

### Scoring

- **Correct**: `set(predicted_results) == set(ground_truth_results)`
- Row order is ignored, duplicates are removed (BIRD methodology)
- Score: 1.0 if match, 0.0 otherwise (before cost penalty)

### Dataset: BIRD Benchmark

- **train-filtered** (default): 6,601 questions across 69 databases
- **dev**: 1,534 questions across 11 databases
- All databases are SQLite

Diagnostics: Any print() output from agent.py is captured and included in evaluation diagnostics as agent_stdout. Any print() output from analyze_db.py is captured as tool_stdout. Use print() to log any information you think would be helpful for you to see in improving the agent in later rounds of testing and refinement.
