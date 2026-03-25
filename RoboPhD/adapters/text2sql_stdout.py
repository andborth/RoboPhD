"""
Text2SQL adapter with stdout capture from analyze_db.py.

Subclasses Text2SQLEvaluator to capture subprocess stdout from the
database analysis tool and surface it as a diagnostic. This lets the
evolution AI design its own diagnostics by adding print() statements
to analyze_db.py.

Results are non-comparable with the base text2sql task — use the
text2sql_stdout task name for tracking.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from RoboPhD.adapters.gepa_text2sql import (
    Text2SQLEvaluator,
    TEXT2SQL_FILE_MAPPING,
    build_text2sql_dataset,
    _hash_content,
    _TOOL_TIMEOUT,
)

logger = logging.getLogger(__name__)


class Text2SQLStdoutEvaluator(Text2SQLEvaluator):
    """Text2SQLEvaluator with stdout capture from analyze_db.py.

    Overrides _run_analysis_tool to return (analysis, tool_stdout),
    _get_analysis to cache both, and __call__ to add tool_stdout to diagnostics.
    """

    def _run_analysis_tool(self, code: str, db_id: str) -> Tuple[str, str]:
        """Run analyze_db.py, return (analysis_text, tool_stdout)."""
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return f"ERROR: Database not found: {db_path}", ""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            tools_dir = tmp / "tools"
            tools_dir.mkdir()
            (tools_dir / "analyze_db.py").write_text(code)
            (tmp / "database.sqlite").symlink_to(db_path.resolve())
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
                return f"ERROR: Tool timed out after {_TOOL_TIMEOUT}s for {db_id}", ""

            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else "(no stderr)"
                return f"ERROR: Tool exited with code {result.returncode} for {db_id}\n{stderr}", result.stdout or ""

            stdout = result.stdout or ""

            # Read file output
            for output_path in [
                tmp / "tool_output" / "analysis.txt",
                tmp / "tool_output" / "schema.txt",
            ]:
                if output_path.exists() and output_path.stat().st_size >= 10:
                    return output_path.read_text(), stdout

            # Fall back to stdout as analysis (stdout is then both analysis and tool_stdout)
            if stdout and len(stdout) >= 10:
                return stdout, stdout

            return f"ERROR: Tool produced no output for {db_id}", stdout

    def _get_analysis(self, code: str, db_id: str) -> str:
        """Get analysis with caching. Also caches tool_stdout (retrievable via _get_tool_stdout)."""
        code_hash = _hash_content(code)
        key = self._phase1_cache_key(code_hash, db_id)

        cached = self._phase1_cache_get(key)
        if cached is not None:
            # Cache stores JSON {"analysis": ..., "stdout": ...} or plain text (legacy)
            try:
                obj = json.loads(cached)
                if isinstance(obj, dict) and "analysis" in obj:
                    return obj["analysis"]
            except (json.JSONDecodeError, TypeError):
                pass
            return cached

        analysis, tool_stdout = self._run_analysis_tool(code, db_id)

        cache_value = json.dumps({"analysis": analysis, "stdout": tool_stdout})
        self._phase1_cache_put(key, cache_value)

        return analysis

    def _get_tool_stdout(self, code: str, db_id: str) -> str:
        """Retrieve cached tool_stdout for a (code, db_id) pair."""
        code_hash = _hash_content(code)
        key = self._phase1_cache_key(code_hash, db_id)
        cached = self._phase1_cache_get(key)
        if cached:
            try:
                obj = json.loads(cached)
                if isinstance(obj, dict):
                    return obj.get("stdout", "")
            except (json.JSONDecodeError, TypeError):
                pass
        return ""

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, Any],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate with tool_stdout capture.

        Parent __call__ invokes our _get_analysis override which caches
        both analysis and stdout as JSON. We then retrieve stdout from
        the cache via _get_tool_stdout (cheap dict lookup, no re-execution).
        """
        score, diagnostics = super().__call__(candidate, example, problem_dir=problem_dir)

        analysis_code = candidate.get("database_analysis_code", "")
        tool_stdout = self._get_tool_stdout(analysis_code, example["db_id"])
        if tool_stdout.strip():
            diagnostics["tool_stdout"] = tool_stdout

        return score, diagnostics
