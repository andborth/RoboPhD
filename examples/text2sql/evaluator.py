"""
Text2SQL Integrated evaluator — self-contained.

Two-phase evaluation:
  Phase 1: analyze_db.py subprocess (cached per code+database)
  Phase 2: agent.py solve() with llm() and test_sql() callables
Scoring: set-based SQL result comparison (BIRD methodology)
"""

import json
import logging
import os
import queue
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

from RoboPhD.eval_utils import retry_on_rate_limit, exec_with_stdout_capture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOOL_TIMEOUT = 300       # analyze_db.py subprocess timeout (seconds)
_SQL_TIMEOUT = 30         # test_sql() per-query timeout (seconds)
_MAX_TEST_SQL_CALLS = 5   # max test_sql() calls per question
_COST_BUDGET = 0.10       # per-question cost budget (USD)
_OVER_BUDGET_PENALTY = 0.9  # multiplier for correct-but-over-budget answers

# Dataset root. Defaults to the repo's benchmark_resources/datasets (relative to
# repo root), but can be relocated — e.g. to an external drive when the ~130GB
# BIRD dataset won't fit on the internal disk — via the BIRD_DATA_DIR env var.
# Keep this in sync with benchmark_resources/download_bird.sh, which honors the
# same variable, and with main.py's sandbox read carve-out.
_DATA_ROOT = os.environ.get("BIRD_DATA_DIR", "benchmark_resources/datasets")
_DATASET_PATHS = {
    "train": {
        "questions": f"{_DATA_ROOT}/train/train/train.json",
        "db_root": f"{_DATA_ROOT}/train/train/train_databases",
    },
    "train-filtered": {
        "questions": f"{_DATA_ROOT}/train-filtered/train_filtered.json",
        "db_root": f"{_DATA_ROOT}/train/train/train_databases",
    },
    "dev": {
        "questions": f"{_DATA_ROOT}/dev/dev_20240627/dev.json",
        "db_root": f"{_DATA_ROOT}/dev/dev_20240627/dev_databases",
    },
}


# ---------------------------------------------------------------------------
# SQL execution (inlined from cached_sql_executor)
# ---------------------------------------------------------------------------

class SQLTimeoutError(Exception):
    """Raised when a SQL query exceeds its timeout."""


def execute_sql_with_timeout(db_path: str, sql: str, timeout_seconds: int) -> List[Tuple]:
    """Execute SQL against a SQLite database with a thread-based timeout."""
    result_queue: queue.Queue = queue.Queue()
    exception_queue: queue.Queue = queue.Queue()

    def target():
        try:
            conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
            conn.execute("PRAGMA query_only = 1")
            conn.execute("PRAGMA read_uncommitted = 1")
            try:
                cursor = conn.cursor()
                cursor.execute(sql)
                result_queue.put(cursor.fetchall())
            finally:
                conn.close()
        except Exception as e:
            exception_queue.put(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise SQLTimeoutError(f"SQL query timed out after {timeout_seconds} seconds")
    if not exception_queue.empty():
        raise exception_queue.get()
    if not result_queue.empty():
        return result_queue.get()
    raise RuntimeError("SQL execution completed but no result returned")


def compare_sql_results(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: str,
    pred_timeout: int = 30,
    gt_timeout: int = 300,
) -> Dict[str, Any]:
    """Execute both SQLs and compare results using set comparison (BIRD methodology)."""

    # Execute predicted SQL
    pred_results = None
    pred_error = None
    try:
        pred_results = execute_sql_with_timeout(db_path, predicted_sql, pred_timeout)
    except SQLTimeoutError as e:
        pred_error = str(e)
    except Exception as e:
        pred_error = str(e)

    # Execute ground truth SQL
    gt_results = None
    gt_error = None
    try:
        gt_results = execute_sql_with_timeout(db_path, ground_truth_sql, gt_timeout)
    except SQLTimeoutError as e:
        gt_error = str(e)
    except Exception as e:
        gt_error = str(e)

    # Determine status
    if gt_error:
        status = "gt_error"
    elif pred_error:
        status = "pred_error"
    elif set(pred_results) == set(gt_results):
        status = "match"
    else:
        status = "mismatch"

    return {
        "matches": status == "match",
        "status": status,
        "predicted_results": _format_sql_results(pred_results),
        "ground_truth_results": _format_sql_results(gt_results),
        "predicted_error": pred_error,
        "ground_truth_error": gt_error,
    }


def _format_sql_results(results: Optional[List[Tuple]], max_rows: int = 50) -> Optional[str]:
    """Format SQL results as a readable string, with truncation."""
    if results is None:
        return None
    if not results:
        return "0 rows returned"
    total = len(results)
    lines = [f"{total} rows returned:"]
    for row in results[:max_rows]:
        lines.append(str(row))
    if total > max_rows:
        lines.append(f"... ({total - max_rows} more rows)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate text with a note."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"


# ---------------------------------------------------------------------------
# Cost tracker
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


# ---------------------------------------------------------------------------
# Tracked callables
# ---------------------------------------------------------------------------

def make_tracked_llm(model: str, tracker: CostTracker):
    """Return an llm(prompt) -> str callable with cost tracking.

    ``model`` must be a full litellm-compatible name (e.g.
    "claude-haiku-4-5-20251001"), not a short alias.
    """
    import os

    # Resolve an explicit api_key for Anthropic models and pass it to litellm,
    # rather than mutating os.environ["ANTHROPIC_API_KEY"]. The env mutation
    # would leak into Claude Code CLI subprocesses spawned later in the run
    # and could divert them onto API billing.
    api_key: Optional[str] = None
    if "claude" in model.lower() or model.startswith("anthropic/"):
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")
        )

    def llm(prompt: str) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": 300,
            "num_retries": 0,
        }
        if api_key:
            kwargs["api_key"] = api_key
        resp = retry_on_rate_limit(lambda: litellm.completion(**kwargs))
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

    def test_sql(sql: str) -> str:
        if tracker.sql_calls >= max_calls:
            result_str = f"ERROR: test_sql call limit reached ({max_calls} calls)"
            tracker.trace.append(("sql", {"sql": sql, "result": result_str}))
            return result_str

        tracker.sql_calls += 1

        try:
            results = execute_sql_with_timeout(db_path, sql, _SQL_TIMEOUT)
            result_str = _format_sql_results(results)
        except SQLTimeoutError:
            result_str = f"ERROR: SQL timed out after {_SQL_TIMEOUT}s"
        except Exception as e:
            result_str = f"ERROR: {e}"

        tracker.trace.append(("sql", {"sql": sql, "result": result_str}))
        return result_str

    return test_sql


# ---------------------------------------------------------------------------
# Analysis tool (subprocess in problem_dir)
# ---------------------------------------------------------------------------

def _run_analysis_tool(code: str, db_id: str, db_root: Path, problem_dir: Path) -> Tuple[str, str]:
    """Run analyze_db.py as a subprocess in the problem directory.

    Writes analyze_db.py and symlinks database.sqlite into problem_dir,
    then runs the script. The output file (analysis.txt) and all artifacts
    persist in problem_dir for reproducibility.

    Returns (analysis_text, stdout).
    """
    db_path = db_root / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return f"ERROR: Database not found: {db_path}", ""

    (problem_dir / "analyze_db.py").write_text(code)
    link = problem_dir / "database.sqlite"
    if not link.exists():
        link.symlink_to(db_path.resolve())

    try:
        result = subprocess.run(
            ["python", "analyze_db.py"],
            cwd=str(problem_dir),
            capture_output=True, text=True,
            timeout=_TOOL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: Tool timed out after {_TOOL_TIMEOUT}s for {db_id}", ""

    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else "(no stderr)"
        return f"ERROR: Tool exited with code {result.returncode} for {db_id}\n{stderr}", result.stdout or ""

    stdout = result.stdout or ""

    # Read analysis output — check common filenames
    for name in ["analysis.txt", "schema.txt"]:
        output_path = problem_dir / name
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

    Returns (sql_string, agent_stdout, error_or_None).
    """

    def _call_solve(ns):
        solve_fn = ns.get("solve")
        if not solve_fn:
            raise RuntimeError("No solve() function found in agent code")
        result = solve_fn(question, evidence, analysis, llm, test_sql)
        if not isinstance(result, str):
            result = str(result) if result is not None else ""
        return result

    try:
        namespace, stdout = exec_with_stdout_capture(agent_code, then=_call_solve)
        return namespace.get("_result", ""), stdout, None
    except Exception as e:
        return "", getattr(e, "stdout", ""), str(e)


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
        cost_budget: float = _COST_BUDGET,
        over_budget_penalty: float = _OVER_BUDGET_PENALTY,
        max_test_sql_calls: int = _MAX_TEST_SQL_CALLS,
    ):
        self.eval_model = eval_model
        self.cost_budget = cost_budget
        self.over_budget_penalty = over_budget_penalty
        self.max_test_sql_calls = max_test_sql_calls

        paths = _DATASET_PATHS[dataset]
        self.db_root = Path(paths["db_root"])

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
        agent_code = candidate.get("agent.py", "")
        analysis_code = candidate.get("analyze_db.py", "")

        db_id = example["db_id"]
        question = example["question"]
        evidence = example.get("evidence", "")
        ground_truth_sql = example["SQL"]

        with self._lock:
            self._eval_count += 1

        # Phase 1: Analysis tool (runs in problem_dir, or tempdir if standalone)
        if problem_dir is not None:
            analysis, tool_stdout = _run_analysis_tool(
                analysis_code, db_id, self.db_root, problem_dir,
            )
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                analysis, tool_stdout = _run_analysis_tool(
                    analysis_code, db_id, self.db_root, Path(td),
                )

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
                f"Text2SQL evaluator: {milestone} evaluations "
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
            match = re.search(r"```(?:sql)?\n?(.*?)```", predicted_sql, re.DOTALL)
            if match:
                predicted_sql = match.group(1).strip()
            predicted_sql = predicted_sql.rstrip(";").strip()

            comparison = compare_sql_results(
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql,
                db_path=db_path,
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_text2sql_dataset(dataset: str = "train-filtered") -> List[Dict[str, Any]]:
    """Load BIRD benchmark questions from local files.

    Args:
        dataset: "train-filtered" (6601 questions, 69 dbs),
                 "train" (full train set), or "dev" (1534 questions, 11 dbs).

    Requires: benchmark_resources/download_bird.sh
    """
    if dataset not in _DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            f"Available: {', '.join(sorted(_DATASET_PATHS))}"
        )

    paths = _DATASET_PATHS[dataset]
    questions_file = Path(paths["questions"])
    db_root = Path(paths["db_root"])

    if not questions_file.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_file}\n"
            f"Run benchmark_resources/download_bird.sh to download the BIRD dataset."
        )

    with open(questions_file) as f:
        questions = json.load(f)

    examples = []
    for q in questions:
        examples.append({
            "question_id": q.get("question_id", len(examples)),
            "db_id": q["db_id"],
            "question": q["question"],
            "SQL": q["SQL"],
            "evidence": q.get("evidence", ""),
        })

    examples.sort(key=lambda e: (e["db_id"], e["question_id"]))
    logger.info(
        f"Text2SQL dataset '{dataset}': {len(examples)} questions, "
        f"{len(set(e['db_id'] for e in examples))} databases"
    )
    return examples
