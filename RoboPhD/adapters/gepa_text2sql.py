"""
GEPA adapter for RoboPhD's Text2SQL domain (BIRD benchmark).

Tool-only Phase 1 (database analysis via Python script), LLM Phase 2
(SQL generation), and evolvable verification loop.

Candidate representation:
    {
        "eval_instructions": "...",       # System prompt for SQL generation
        "database_analysis_code": "...",  # Python script that analyzes database.sqlite
        "verify_prompt": "...",           # Evolvable verification prompt
    }

Usage:
    from RoboPhD.adapters.gepa_text2sql import (
        Text2SQLEvaluator,
        build_text2sql_dataset,
        TEXT2SQL_FILE_MAPPING,
    )
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from RoboPhD.adapters.debug_logging import maybe_debug_log

from RoboPhD.adapters.candidate_utils import extract_candidate, materialize_candidate  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File mappings: candidate dict keys -> agent directory paths
# ---------------------------------------------------------------------------

TEXT2SQL_FILE_MAPPING = {
    "eval_instructions": "eval_instructions.md",
    "database_analysis_code": "tools/analyze_db.py",
    "verify_prompt": "verify_prompt.md",
}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_text2sql_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a flat list of example dicts from BIRD benchmark questions.

    Each example contains the question text and metadata needed for evaluation.
    Ground truth SQL is included for scoring but NOT exposed to the evolution AI.

    Args:
        config: Merged config dict. Relevant keys:
            - dataset: "train-filtered" (default) or "dev"

    Returns:
        List of example dicts sorted by (db_id, question_id).
    """
    from RoboPhD.config import DATASET_PATHS

    dataset = config.get("dataset", "train-filtered")
    if dataset not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            f"Available: {', '.join(sorted(DATASET_PATHS))}"
        )

    paths = DATASET_PATHS[dataset]
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
        f"{len(set(e['db_id'] for e in examples))} databases, "
        f"db_root={db_root}"
    )
    return examples


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

# Temperature schedule for progressive verification
_PROGRESSIVE_TEMPS = [0.0, 0.2, 0.3]

# Tool execution timeout (seconds)
_TOOL_TIMEOUT = 300


def _hash_content(content: str) -> str:
    """Short hash of text content for cache keys."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _clean_sql(text: str) -> str:
    """Strip markdown fences and whitespace from LLM SQL output."""
    text = text.strip()
    # Remove ```sql ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first and last fence lines
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    # Remove trailing semicolons (BIRD evaluation doesn't need them)
    text = text.rstrip(";").strip()
    return text


class Text2SQLEvaluator:
    """
    Evaluator wrapping Text2SQL's tool-based analysis + LLM SQL generation.

    Implements: evaluator(candidate, example, *, problem_dir=None) -> (score, diagnostics)

    Phase 1 (tool): Run analyze_db.py against database.sqlite, cache per (code_hash, db_id).
    Phase 2 (LLM): Generate SQL using eval_instructions + analysis + question.
    Verification: Execute SQL, verify with evolvable prompt, retry if not CORRECT.
    Scoring: Compare predicted vs ground truth result sets (order-insensitive).
    """

    def __init__(
        self,
        eval_model: str = "haiku-4.5",
        dataset: str = "train-filtered",
        use_evidence: bool = True,
        max_verification_retries: int = 2,
        temperature_strategy: str = "fixed",
        output_dir: Optional[str] = None,
        work_dir: Optional[Path] = None,
        debug_log_probability: float = 0.0,
        debug_log_dir: Optional[Path] = None,
    ):
        from RoboPhD.config import DATASET_PATHS

        self.eval_model = eval_model
        self.use_evidence = use_evidence
        self.max_verification_retries = max_verification_retries
        self.temperature_strategy = temperature_strategy

        paths = DATASET_PATHS[dataset]
        self.db_root = Path(paths["db_root"])

        self.debug_log_probability = debug_log_probability
        self.debug_log_dir = Path(debug_log_dir) if debug_log_dir else None

        self.work_dir = Path(work_dir) if work_dir else Path("gepa_text2sql_work")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Disk-backed Phase 1 cache directory
        if output_dir:
            self._cache_dir = Path(output_dir) / "cache" / "phase1_tool_analysis"
        else:
            self._cache_dir = None  # fallback to in-memory
        self._memory_cache: Dict[str, str] = {}

        self._eval_count = 0
        self._total_eval_cost = 0.0
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()

        # Lazy-init provider
        self._provider = None
        self._provider_lock = threading.Lock()

        # Lazy-init SQL executor
        self._sql_executor = None
        self._sql_executor_lock = threading.Lock()

    def _get_provider(self):
        """Lazy-init LLM provider."""
        if self._provider is not None:
            return self._provider
        with self._provider_lock:
            if self._provider is None:
                from RoboPhD.llm_providers import get_provider
                self._provider = get_provider(self.eval_model)
            return self._provider

    def _get_sql_executor(self):
        """Lazy-init CachedSQLExecutor."""
        if self._sql_executor is not None:
            return self._sql_executor
        with self._sql_executor_lock:
            if self._sql_executor is None:
                from RoboPhD.utilities.cached_sql_executor import CachedSQLExecutor
                cache_dir = str(self.work_dir / "sql_cache")
                self._sql_executor = CachedSQLExecutor(cache_dir=cache_dir)
            return self._sql_executor

    # -- Phase 1: Tool-based database analysis --

    def _phase1_cache_key(self, code_hash: str, db_id: str) -> str:
        return f"{code_hash}_{db_id}"

    def _phase1_cache_get(self, key: str) -> Optional[str]:
        """Check disk cache, then memory cache."""
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.txt"
            if cache_file.exists():
                return cache_file.read_text()
        return self._memory_cache.get(key)

    def _phase1_cache_put(self, key: str, value: str) -> None:
        """Write to disk cache (if available) and memory cache."""
        with self._cache_lock:
            self._memory_cache[key] = value
            if self._cache_dir:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self._cache_dir / f"{key}.txt"
                cache_file.write_text(value)

    def _run_analysis_tool(self, code: str, db_id: str) -> str:
        """
        Run analyze_db.py against the database, return analysis text.

        Creates a temp directory with:
          - tools/analyze_db.py (the candidate code)
          - database.sqlite (symlink to actual db)
          - tool_output/ (for script output)
        """
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return f"ERROR: Database not found: {db_path}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write tool code
            tools_dir = tmp / "tools"
            tools_dir.mkdir()
            (tools_dir / "analyze_db.py").write_text(code)

            # Symlink database
            (tmp / "database.sqlite").symlink_to(db_path.resolve())

            # Create output directory
            (tmp / "tool_output").mkdir()

            try:
                result = subprocess.run(
                    ["python", "tools/analyze_db.py"],
                    cwd=str(tmp),
                    capture_output=True,
                    text=True,
                    timeout=_TOOL_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                return f"ERROR: Tool timed out after {_TOOL_TIMEOUT}s for {db_id}"

            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else "(no stderr)"
                return f"ERROR: Tool exited with code {result.returncode} for {db_id}\n{stderr}"

            # Read output — check common output locations
            for output_path in [
                tmp / "tool_output" / "analysis.txt",
                tmp / "tool_output" / "schema.txt",
            ]:
                if output_path.exists() and output_path.stat().st_size >= 10:
                    return output_path.read_text()

            # Fall back to stdout
            if result.stdout and len(result.stdout) >= 10:
                return result.stdout

            return f"ERROR: Tool produced no output for {db_id}"

    def _get_analysis(self, code: str, db_id: str) -> str:
        """Get analysis with caching."""
        code_hash = _hash_content(code)
        key = self._phase1_cache_key(code_hash, db_id)

        cached = self._phase1_cache_get(key)
        if cached is not None:
            return cached

        analysis = self._run_analysis_tool(code, db_id)
        self._phase1_cache_put(key, analysis)
        return analysis

    # -- Phase 2: LLM SQL generation --

    def _generate_sql(self, eval_instructions: str, analysis: str, question: str, evidence: str) -> Tuple[str, float]:
        """Generate initial SQL via LLM. Returns (sql, cost).

        Uses multi-breakpoint caching:
          Block 0: eval_instructions (fixed per agent)
          Block 1: analysis (fixed per agent+db, potentially large DDL)
          User prompt: question + evidence (varies per example)
        """
        provider = self._get_provider()

        user_parts = [f"## Question\n\n{question}"]
        if self.use_evidence and evidence:
            user_parts.append(f"\n## Evidence\n\n{evidence}")
        user_prompt = "\n".join(user_parts)

        response = provider.generate(
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1000,
            cached_blocks=[eval_instructions, f"## Database Analysis\n\n{analysis}"],
        )

        cost = self._compute_cost(response)

        maybe_debug_log(
            debug_log_probability=self.debug_log_probability,
            debug_log_dir=self.debug_log_dir,
            call_type="generation",
            model=self.eval_model,
            messages=[
                {"role": "system", "content": eval_instructions},
                {"role": "system", "content": f"## Database Analysis\n\n{analysis}"},
                {"role": "user", "content": user_prompt},
            ],
            response_text=response.text,
            metadata={"cost_usd": cost, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )

        return _clean_sql(response.text), cost

    # -- Verification loop --

    def _get_verification_temperature(self, attempt_num: int) -> float:
        if self.temperature_strategy == "progressive":
            return _PROGRESSIVE_TEMPS[min(attempt_num, len(_PROGRESSIVE_TEMPS) - 1)]
        elif self.temperature_strategy == "adaptive":
            return 0.0 if attempt_num == 0 else 0.2
        return 0.0  # "fixed" or default

    def _summarize_results(self, results, error_msg: Optional[str] = None) -> str:
        if error_msg:
            return f"SQL Error: {error_msg}"
        if results is None or len(results) == 0:
            return "Empty result set (0 rows)"
        if len(results) <= 10:
            return f"Complete results ({len(results)} rows):\n{results}"
        sample = results[:10]
        return f"Result summary: {len(results)} total rows\nFirst 10 rows:\n{sample}\n[...{len(results)-10} more rows]"

    def _verify_and_improve(
        self,
        verify_prompt: str,
        question: str,
        evidence: str,
        sql: str,
        summary: str,
        attempts: List[Dict],
        eval_instructions: str,
        analysis: str,
    ) -> Tuple[bool, str, float]:
        """Single verification call. Returns (is_correct, new_sql, cost).

        Uses multi-breakpoint caching to reuse the eval_instructions + analysis
        prefix from the generation call (guaranteed cache hit):
          Block 0: eval_instructions (fixed per agent)
          Block 1: analysis (fixed per agent+db — prefix hit from generation)
          Block 2: verify_prompt + format instructions (fixed per agent)
          User prompt: history + question + SQL + results (varies per call)
        """
        provider = self._get_provider()
        temperature = self._get_verification_temperature(len(attempts))

        # Build history
        history = ""
        if attempts:
            history = "Previous attempts:\n\n"
            for i, att in enumerate(attempts, 1):
                history += f"Attempt {i}:\nSQL: {att['sql']}\nResults: {att['summary']}\n"
                if att.get("feedback"):
                    history += f"Issues identified: {att['feedback']}\n"
                history += "\n"

        maybe_evidence = ""
        if self.use_evidence and evidence:
            maybe_evidence = f"\nEvidence: {evidence}"

        user_prompt = f"""{history}Question: {question}{maybe_evidence}

Current SQL: {sql}
Results: {summary}

Your response:"""

        # Verification block: verify_prompt + response format instructions.
        # Cached as block 2 after eval_instructions and analysis.
        verify_block = f"""{verify_prompt}

Respond with EXACTLY one of the following:
1. The single word: CORRECT
2. A new SQL query with no additional text

Do not include explanations, prefixes, or combine both responses."""

        response = provider.generate(
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=1000,
            cached_blocks=[eval_instructions, f"## Database Analysis\n\n{analysis}", verify_block],
        )
        cost = self._compute_cost(response)

        maybe_debug_log(
            debug_log_probability=self.debug_log_probability,
            debug_log_dir=self.debug_log_dir,
            call_type="verification",
            model=self.eval_model,
            messages=[
                {"role": "system", "content": eval_instructions},
                {"role": "system", "content": f"## Database Analysis\n\n{analysis}"},
                {"role": "system", "content": verify_block},
                {"role": "user", "content": user_prompt},
            ],
            response_text=response.text,
            metadata={"cost_usd": cost, "attempt": len(attempts), "temperature": temperature},
        )

        text = response.text.strip()

        if text.upper() == "CORRECT" or text.upper().startswith("CORRECT"):
            return True, sql, cost

        new_sql = _clean_sql(text)
        # Handle edge case: "CORRECT: SELECT ..."
        if new_sql.upper().startswith("CORRECT"):
            new_sql = new_sql[7:].strip().lstrip(":").strip()
            new_sql = _clean_sql(new_sql)
        # Handle edge case: "SELECT ...\nCORRECT" (CORRECT on its own line)
        elif new_sql.strip().split("\n")[-1].strip().upper() == "CORRECT":
            stripped = "\n".join(new_sql.strip().split("\n")[:-1]).strip()
            return True, stripped if stripped else sql, cost

        return False, new_sql, cost

    def _generate_with_verification(
        self,
        eval_instructions: str,
        analysis: str,
        question: str,
        evidence: str,
        verify_prompt: str,
        db_id: str,
    ) -> Tuple[str, float, List[Dict]]:
        """Generate SQL with verification loop. Returns (final_sql, total_cost, attempts)."""
        sql, total_cost = self._generate_sql(eval_instructions, analysis, question, evidence)
        attempts: List[Dict] = []

        executor = self._get_sql_executor()
        db_path = str(self.db_root / db_id / f"{db_id}.sqlite")

        for attempt_num in range(self.max_verification_retries):
            # Execute current SQL
            try:
                results = executor.execute_sql(sql, db_path, is_ground_truth=False, timeout_seconds=30)
                summary = self._summarize_results(results)
            except Exception as e:
                summary = self._summarize_results(None, str(e))

            is_correct, new_sql, cost = self._verify_and_improve(
                verify_prompt, question, evidence, sql, summary, attempts,
                eval_instructions=eval_instructions, analysis=analysis,
            )
            total_cost += cost

            attempts.append({
                "sql": sql,
                "summary": summary,
                "is_correct": is_correct,
                "feedback": "" if is_correct else "Model provided improved SQL",
            })

            if is_correct:
                return sql, total_cost, attempts

            sql = new_sql

        return sql, total_cost, attempts

    # -- Scoring --

    def _compute_cost(self, response) -> float:
        """Estimate cost from LLMResponse token counts."""
        from RoboPhD.llm_providers import get_model_pricing
        pricing = get_model_pricing(self.eval_model)
        input_cost = (response.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (response.output_tokens / 1_000_000) * pricing["output"]
        cache_write = (response.cache_creation_tokens / 1_000_000) * pricing.get("cache_write", 0)
        cache_read = (response.cache_read_tokens / 1_000_000) * pricing.get("cache_read", 0)
        return input_cost + output_cost + cache_write + cache_read

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a candidate on a single BIRD question.

        Args:
            candidate: {"eval_instructions": "...", "database_analysis_code": "...", "verify_prompt": "..."}.
            example: {"question_id": ..., "db_id": "...", "question": "...", "SQL": "...", "evidence": "..."}.
            problem_dir: Optional path for writing per-problem diagnostic files.

        Returns:
            (score, diagnostics) where score is 1.0 if result sets match, 0.0 otherwise.
        """
        eval_instructions = candidate.get("eval_instructions", "")
        analysis_code = candidate.get("database_analysis_code", "")
        verify_prompt = candidate.get("verify_prompt", "Review the SQL and results. Does this correctly answer the question?")

        db_id = example["db_id"]
        question = example["question"]
        evidence = example.get("evidence", "")
        ground_truth_sql = example["SQL"]

        with self._lock:
            self._eval_count += 1
            count = self._eval_count
        if count % 50 == 0:
            logger.info(f"Text2SQL evaluator: {count} evaluations completed (${self._total_eval_cost:.2f} spent)")

        # Phase 1: Tool-based analysis (cached)
        analysis = self._get_analysis(analysis_code, db_id)

        # Phase 2 + Verification
        try:
            if self.max_verification_retries > 0:
                predicted_sql, cost, attempts = self._generate_with_verification(
                    eval_instructions, analysis, question, evidence, verify_prompt, db_id,
                )
            else:
                predicted_sql, cost = self._generate_sql(eval_instructions, analysis, question, evidence)
                attempts = []
        except Exception as e:
            logger.error(f"SQL generation failed for {db_id}/{example['question_id']}: {e}")
            return 0.0, {"error": str(e), "question_id": example["question_id"], "db_id": db_id}

        with self._lock:
            self._total_eval_cost += cost

        # Scoring: compare result sets
        from RoboPhD.utilities.cached_sql_executor import compare_execution_results
        executor = self._get_sql_executor()
        db_path = str(self.db_root / db_id / f"{db_id}.sqlite")

        comparison = compare_execution_results(
            predicted_sql=predicted_sql,
            ground_truth_sql=ground_truth_sql,
            db_path=db_path,
            executor=executor,
            pred_timeout_seconds=30,
            gt_timeout_seconds=300,
        )

        score = 1.0 if comparison["matches"] else 0.0

        # Build diagnostics
        diagnostics = {
            "question_id": example["question_id"],
            "db_id": db_id,
            "score": score,
            "correct": score >= 0.5,
            "status": comparison["status"],
            "verification_retries": len([a for a in attempts if not a["is_correct"]]),
            "cost_usd": cost,
            # String diagnostics — keys are filenames written by ExternalEvaluatorDomain
            "question.md": f"# Question\n\n{question}\n\n## Evidence\n\n{evidence or '(none)'}",
            "analysis.md": f"# Database Analysis ({db_id})\n\n{analysis}",
            "predicted_sql.md": f"```sql\n{predicted_sql}\n```",
            "expected_sql.md": f"```sql\n{ground_truth_sql}\n```",
            "predicted_results.md": str(comparison.get("predicted_results", "(error)")),
            "ground_truth_results.md": str(comparison.get("ground_truth_results", "(error)")),
            "result_comparison.md": (
                f"Match: {comparison['matches']}\n"
                f"Status: {comparison['status']}\n"
                f"Predicted truncated: {comparison.get('predicted_results_truncated', False)}\n"
                f"GT truncated: {comparison.get('ground_truth_results_truncated', False)}\n"
                f"Predicted deduplicated: {comparison.get('predicted_results_deduplicated', False)}\n"
                f"GT deduplicated: {comparison.get('ground_truth_results_deduplicated', False)}"
            ),
        }

        rejected_attempts = [att for att in attempts if not att["is_correct"]]
        if rejected_attempts:
            parts = []
            for i, att in enumerate(rejected_attempts, 1):
                parts.append(
                    f"## Rejected Attempt {i}\n\n"
                    f"### SQL\n```sql\n{att['sql']}\n```\n\n"
                    f"### Results\n{att['summary']}"
                )
            diagnostics["verification_attempts.md"] = "\n\n".join(parts)

        if comparison.get("predicted_error"):
            diagnostics["predicted_error.md"] = comparison["predicted_error"]
        if comparison.get("ground_truth_error"):
            diagnostics["ground_truth_error.md"] = comparison["ground_truth_error"]

        # Write result.json if problem_dir provided (RoboPhD path)
        if problem_dir is not None:
            problem_dir = Path(problem_dir)
            problem_dir.mkdir(parents=True, exist_ok=True)
            result_entry = {
                "question_id": example["question_id"],
                "db_id": db_id,
                "correct": score >= 0.5,
                "score": score,
                "status": comparison["status"],
                "verification_retries": len([a for a in attempts if not a["is_correct"]]),
            }
            with open(problem_dir / "result.json", "w") as f:
                json.dump(result_entry, f, indent=2)

        return score, diagnostics

    @property
    def total_evaluations(self) -> int:
        return self._eval_count

    @property
    def total_eval_cost(self) -> float:
        return self._total_eval_cost
