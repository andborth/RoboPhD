"""
Integrated Text2SQL adapter: agent.py controls SQL generation with llm() and test_sql().

Unlike text2sql (rigid pipeline: analyze_db → eval_instructions → verify_prompt),
this gives the agent full control. The agent receives:
- analysis: pre-computed output from analyze_db.py (subprocess, cached)
- llm(prompt) -> str: LLM call with cost tracking
- test_sql(sql) -> str: execute SQL and return formatted results

The agent returns the final SQL string for scoring.

Also captures stdout from both agent.py and analyze_db.py.
"""

import builtins
import hashlib
import io
import json
import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

from RoboPhD.adapters.gepa_docfinqa import _retry_on_rate_limit

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOOL_TIMEOUT = 300  # seconds for analyze_db.py subprocess
_SQL_TIMEOUT = 30  # seconds for test_sql() execution
_MAX_TEST_SQL_CALLS = 5
_COST_BUDGET = 0.10  # dollars per question
_OVER_BUDGET_PENALTY = 0.9

# ---------------------------------------------------------------------------
# File mapping
# ---------------------------------------------------------------------------

TEXT2SQL_INTEGRATED_FILE_MAPPING = {
    "agent_code": "agent.py",
    "database_analysis_code": "analyze_db.py",
}

# ---------------------------------------------------------------------------
# BACKGROUND and OBJECTIVE
# ---------------------------------------------------------------------------

OBJECTIVE = (
    "Optimize the database analysis tool and agent code to maximize "
    "execution accuracy on BIRD benchmark questions."
)

BACKGROUND = """\
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
   analysis, and writes findings to `tool_output/analysis.txt`.
   Runs as a subprocess (cached per code+database). Common techniques:
   DDL extraction, sample data, foreign key mapping, column statistics.

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

Diagnostics: Any print() output from agent.py is captured and included \
in evaluation diagnostics as agent_stdout. Any print() output from \
analyze_db.py is captured as tool_stdout. Use print() to log any \
information you think would be helpful for you to see in improving \
the agent in later rounds of testing and refinement."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_content(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate text with a note."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"


def _format_sql_results(results: List[Tuple], max_rows: int = 50) -> str:
    """Format SQL results as a readable string."""
    if not results:
        return "0 rows returned"
    total = len(results)
    shown = results[:max_rows]
    lines = [f"{total} rows returned:"]
    for row in shown:
        lines.append(str(row))
    if total > max_rows:
        lines.append(f"... ({total - max_rows} more rows)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cost tracker (simplified from DocFinQA)
# ---------------------------------------------------------------------------

class CostTracker:
    """Per-evaluation cost and call tracker with ordered trace."""

    def __init__(self, budget: float = _COST_BUDGET):
        self.budget = budget
        self.llm_cost = 0.0
        self.llm_calls = 0
        self.sql_calls = 0
        # Ordered trace: list of ("llm", {prompt, response, cost}) or ("sql", {sql, result})
        self.trace: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def total(self) -> float:
        return self.llm_cost


def make_tracked_llm(model: str, tracker: CostTracker):
    """Return an llm(prompt) -> str callable with cost tracking."""

    def llm(prompt: str) -> str:
        resp = _retry_on_rate_limit(
            lambda: litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        response_text = resp.choices[0].message.content or ""
        tracker.llm_cost += cost
        tracker.llm_calls += 1
        tracker.trace.append(("llm", {"prompt": prompt, "response": response_text, "cost": cost}))
        return response_text

    return llm


def make_tracked_test_sql(db_path: str, tracker: CostTracker, max_calls: int = _MAX_TEST_SQL_CALLS):
    """Return a test_sql(sql) -> str callable with call counting."""
    from RoboPhD.utilities.cached_sql_executor import CachedSQLExecutor, SQLTimeoutError

    executor = CachedSQLExecutor()

    def test_sql(sql: str) -> str:
        if tracker.sql_calls >= max_calls:
            result_str = f"ERROR: test_sql call limit reached ({max_calls} calls)"
            tracker.sql_queries.append(sql)
            tracker.sql_results.append(result_str)
            return result_str

        tracker.sql_calls += 1

        try:
            results = executor.execute_sql(
                sql, db_path, is_ground_truth=False, timeout_seconds=_SQL_TIMEOUT,
            )
            result_str = _format_sql_results(results)
        except SQLTimeoutError:
            result_str = f"ERROR: SQL timed out after {_SQL_TIMEOUT}s"
        except Exception as e:
            result_str = f"ERROR: {e}"

        tracker.trace.append(("sql", {"sql": sql, "result": result_str}))
        return result_str

    return test_sql


# ---------------------------------------------------------------------------
# Analysis tool (subprocess, cached)
# ---------------------------------------------------------------------------

class AnalysisToolRunner:
    """Runs analyze_db.py as a subprocess with caching.

    Adapted from Text2SQLEvaluator's _run_analysis_tool / _get_analysis.
    Caches (analysis_text, tool_stdout) as JSON.
    """

    def __init__(self, db_root: Path, cache_dir: Optional[Path] = None):
        self.db_root = db_root
        self._cache: Dict[str, str] = {}  # memory cache
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, code_hash: str, db_id: str) -> str:
        return f"integrated_{code_hash}_{db_id}"

    def get_analysis(self, code: str, db_id: str) -> Tuple[str, str]:
        """Get (analysis_text, tool_stdout) with caching."""
        code_hash = _hash_content(code)
        key = self._cache_key(code_hash, db_id)

        # Memory cache
        if key in self._cache:
            return self._parse_cached(self._cache[key])

        # Disk cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.txt"
            if cache_file.exists():
                cached = cache_file.read_text()
                self._cache[key] = cached
                return self._parse_cached(cached)

        # Run tool
        analysis, stdout = self._run_tool(code, db_id)
        cache_value = json.dumps({"analysis": analysis, "stdout": stdout})
        self._cache[key] = cache_value
        if self._cache_dir:
            (self._cache_dir / f"{key}.txt").write_text(cache_value)

        return analysis, stdout

    def _parse_cached(self, cached: str) -> Tuple[str, str]:
        try:
            obj = json.loads(cached)
            if isinstance(obj, dict) and "analysis" in obj:
                return obj["analysis"], obj.get("stdout", "")
        except (json.JSONDecodeError, TypeError):
            pass
        return cached, ""

    def _run_tool(self, code: str, db_id: str) -> Tuple[str, str]:
        """Run analyze_db.py subprocess, return (analysis, stdout)."""
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return f"ERROR: Database not found: {db_path}", ""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "analyze_db.py").write_text(code)
            (tmp / "database.sqlite").symlink_to(db_path.resolve())
            (tmp / "tool_output").mkdir()

            try:
                result = subprocess.run(
                    ["python", "analyze_db.py"],
                    cwd=str(tmp),
                    capture_output=True, text=True,
                    timeout=_TOOL_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                return f"ERROR: Tool timed out after {_TOOL_TIMEOUT}s for {db_id}", ""

            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else "(no stderr)"
                return f"ERROR: Tool exited with code {result.returncode} for {db_id}\n{stderr}", result.stdout or ""

            stdout = result.stdout or ""

            for output_path in [
                tmp / "tool_output" / "analysis.txt",
                tmp / "tool_output" / "schema.txt",
            ]:
                if output_path.exists() and output_path.stat().st_size >= 10:
                    return output_path.read_text(), stdout

            if stdout and len(stdout) >= 10:
                return stdout, stdout

            return f"ERROR: Tool produced no output for {db_id}", stdout


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

def run_agent(agent_code, question, evidence, analysis, llm, test_sql):
    """Execute agent code with stdout capture.

    Returns (sql_string, agent_stdout).
    """
    buf = io.StringIO()

    def _captured_print(*args, **kwargs):
        kwargs.setdefault("file", buf)
        print(*args, **kwargs)

    try:
        patched_builtins = dict(vars(builtins))
        patched_builtins["print"] = _captured_print
        namespace = {"print": _captured_print, "__builtins__": patched_builtins}
        exec(agent_code, namespace)

        solve_fn = namespace.get("solve")
        if not solve_fn:
            return "", buf.getvalue(), "No solve() function found in agent code"

        result = solve_fn(question, evidence, analysis, llm, test_sql)
        if not isinstance(result, str):
            result = str(result) if result is not None else ""

        return result, buf.getvalue(), None
    except Exception as e:
        return "", buf.getvalue(), str(e)


# ---------------------------------------------------------------------------
# Diagnostic formatting
# ---------------------------------------------------------------------------

def _format_trace(tracker: CostTracker) -> str:
    """Format LLM and test_sql calls in execution order."""
    lines = [
        f"# Agent Trace",
        f"",
        f"LLM calls: {tracker.llm_calls}, Cost: ${tracker.llm_cost:.4f} (budget: ${tracker.budget:.2f})",
        f"test_sql calls: {tracker.sql_calls}/{_MAX_TEST_SQL_CALLS}",
        "",
    ]

    for i, (call_type, data) in enumerate(tracker.trace, 1):
        if call_type == "llm":
            prompt = data["prompt"]
            response = data["response"]
            cost = data.get("cost", 0.0)
            lines.append(f"## Step {i}: LLM Call (${cost:.4f})")
            lines.append(f"### Prompt ({len(prompt)} chars)")
            lines.append(_truncate(prompt))
            lines.append(f"### Response ({len(response)} chars)")
            lines.append(_truncate(response))
        elif call_type == "sql":
            lines.append(f"## Step {i}: test_sql")
            lines.append(f"### SQL")
            lines.append(f"```sql\n{data['sql']}\n```")
            lines.append(f"### Result")
            lines.append(_truncate(data["result"]))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Text2SQLIntegratedEvaluator:
    """
    Integrated Text2SQL evaluator: agent.py with llm() and test_sql().

    Phase 1: analyze_db.py subprocess (cached)
    Phase 2: agent.py solve() with llm/test_sql callables
    Scoring: set-based result comparison (BIRD methodology)
    """

    def __init__(
        self,
        eval_model: str = "haiku-4.5",
        dataset: str = "train-filtered",
        work_dir: Optional[Path] = None,
        cost_budget: float = _COST_BUDGET,
        over_budget_penalty: float = _OVER_BUDGET_PENALTY,
        max_test_sql_calls: int = _MAX_TEST_SQL_CALLS,
    ):
        from RoboPhD.config import DATASET_PATHS

        self.eval_model = eval_model
        self.cost_budget = cost_budget
        self.over_budget_penalty = over_budget_penalty
        self.max_test_sql_calls = max_test_sql_calls

        paths = DATASET_PATHS[dataset]
        self.db_root = Path(paths["db_root"])

        self.work_dir = Path(work_dir) if work_dir else Path("text2sql_integrated_work")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._tool_runner = AnalysisToolRunner(
            db_root=self.db_root,
            cache_dir=self.work_dir / "phase1_cache",
        )

        self._eval_count = 0
        self._total_eval_cost = 0.0
        self._last_logged_count = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        agent_code = candidate.get("agent_code", "")
        analysis_code = candidate.get("database_analysis_code", "")

        db_id = example["db_id"]
        question = example["question"]
        evidence = example.get("evidence", "")
        ground_truth_sql = example["SQL"]

        with self._lock:
            self._eval_count += 1

        # Phase 1: Analysis tool (cached)
        analysis, tool_stdout = self._tool_runner.get_analysis(analysis_code, db_id)

        # Create callables
        tracker = CostTracker(budget=self.cost_budget)
        llm = make_tracked_llm(self.eval_model, tracker)
        db_path = str(self.db_root / db_id / f"{db_id}.sqlite")
        test_sql = make_tracked_test_sql(db_path, tracker, max_calls=self.max_test_sql_calls)

        # Phase 2: Agent execution
        predicted_sql, agent_stdout, agent_error = run_agent(
            agent_code, question, evidence, analysis, llm, test_sql,
        )

        # Track cost
        cost = tracker.total
        with self._lock:
            self._total_eval_cost += cost
            count = self._eval_count
            total_cost = self._total_eval_cost
            milestone = count // 50 * 50
            should_log = milestone > 0 and milestone > self._last_logged_count
            if should_log:
                self._last_logged_count = milestone
        if should_log:
            logger.info(
                f"Text2SQL-integrated evaluator: {milestone} evaluations "
                f"completed (${total_cost:.2f} spent)"
            )

        # Scoring
        if agent_error:
            score = 0.0
            comparison = {
                "matches": False, "status": "agent_error",
                "predicted_results": None, "ground_truth_results": None,
                "predicted_error": agent_error, "ground_truth_error": None,
            }
        else:
            # Clean SQL
            predicted_sql = (predicted_sql or "").strip()
            if predicted_sql.startswith("```"):
                import re
                match = re.search(r"```(?:sql)?\n?(.*?)```", predicted_sql, re.DOTALL)
                if match:
                    predicted_sql = match.group(1).strip()
            predicted_sql = predicted_sql.rstrip(";").strip()

            from RoboPhD.utilities.cached_sql_executor import compare_execution_results, CachedSQLExecutor
            executor = CachedSQLExecutor()
            comparison = compare_execution_results(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_path=db_path,
                executor=executor,
            )

            over_budget = cost > self.cost_budget
            if comparison["matches"]:
                score = 1.0 * (self.over_budget_penalty if over_budget else 1.0)
            else:
                score = 0.0

        # Build diagnostics — strings only
        diagnostics: Dict[str, Any] = {
            "question_id": example.get("question_id", ""),
            "db_id": db_id,
            "score": score,
            "cost_usd": cost,
            "question.md": f"# Question\n\n{question}\n\n## Evidence\n\n{evidence or '(none)'}",
            "analysis.md": f"# Database Analysis ({db_id})\n\n{_truncate(analysis, 5000)}",
            "predicted_sql.md": f"```sql\n{predicted_sql}\n```" if predicted_sql else "(no SQL returned)",
            "expected_sql.md": f"```sql\n{ground_truth_sql}\n```",
            "predicted_results.md": str(comparison.get("predicted_results", "(error)"))[:3000],
            "ground_truth_results.md": str(comparison.get("ground_truth_results", "(error)"))[:3000],
            "result_comparison.md": (
                f"Match: {comparison.get('matches', False)}\n"
                f"Status: {comparison.get('status', 'unknown')}\n"
                f"Cost: ${cost:.4f} (budget: ${self.cost_budget:.2f})"
            ),
            "agent_trace.md": _format_trace(tracker),
        }

        if agent_stdout.strip():
            diagnostics["agent_stdout"] = agent_stdout
        if tool_stdout.strip():
            diagnostics["tool_stdout"] = tool_stdout
        if agent_error:
            diagnostics["error.md"] = f"# Agent Error\n\n```\n{agent_error}\n```"
        if comparison.get("predicted_error"):
            diagnostics["predicted_error.md"] = comparison["predicted_error"]
        if comparison.get("ground_truth_error"):
            diagnostics["ground_truth_error.md"] = comparison["ground_truth_error"]

        # Write result.json
        if problem_dir is not None:
            problem_dir = Path(problem_dir)
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "result.json", "w") as f:
                json.dump({
                    "question_id": example.get("question_id", ""),
                    "db_id": db_id,
                    "score": score,
                    "status": comparison.get("status", "unknown"),
                    "cost": cost,
                    "llm_calls": tracker.llm_calls,
                    "sql_calls": tracker.sql_calls,
                    "error": agent_error,
                }, f, indent=2)

        return score, diagnostics

    @property
    def total_evaluations(self) -> int:
        return self._eval_count

    @property
    def total_eval_cost(self) -> float:
        return self._total_eval_cost
