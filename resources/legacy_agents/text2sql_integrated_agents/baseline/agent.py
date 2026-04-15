import re


def solve(question, evidence, analysis, llm, test_sql):
    """Generate SQL for a BIRD benchmark question.

    Args:
        question: str — the natural language question
        evidence: str — contextual evidence/hints (may be empty)
        analysis: str — output from analyze_db.py for this database
        llm: callable(prompt) -> str — LLM call with cost tracking
        test_sql: callable(sql) -> str — execute SQL, returns results or error (max 5 calls)

    Returns:
        str — final SQL query to submit for scoring
    """
    # --- Demonstration: print() output is captured as agent_stdout ---
    print(f"Question: {question[:100]}")

    prompt = f"""Given this database schema analysis:
{analysis}

Question: {question}
{"Evidence: " + evidence if evidence else ""}

Generate a SQLite query that answers the question.
- Use SQLite syntax (LIMIT not TOP, || for concatenation)
- Follow the evidence literally when provided
- Return ONLY the SQL query, no explanation."""

    response = llm(prompt)

    # Extract SQL from response
    sql = response.strip()
    match = re.search(r"```(?:sql)?\n?(.*?)```", sql, re.DOTALL)
    if match:
        sql = match.group(1).strip()
    sql = sql.rstrip(";").strip()

    # Test the SQL
    result = test_sql(sql)
    print(f"SQL: {sql}")
    print(f"Result preview: {result[:200]}")

    return sql
