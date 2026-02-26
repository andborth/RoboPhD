"""
GEPA adapter for RoboPhD's codegen-critic domain.

Bridges between GEPA's optimize_anything() interface and RoboPhD's
agent-based critic evaluation pipeline.

Candidate representation:
    GEPA sees a dict[str, str] of named text components.
    RoboPhD uses agent directories (agent.md + eval_instructions.md + tools/).
    This adapter translates between the two.

Usage:
    from RoboPhD.adapters.gepa_codegen import (
        RoboPhDCodeGenEvaluator,
        build_codegen_dataset,
        extract_candidate,
        materialize_candidate,
        CODEGEN_FILE_MAPPING,
    )
"""

import hashlib
import json
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Re-export generic candidate utils for backwards compatibility
from RoboPhD.adapters.candidate_utils import extract_candidate, materialize_candidate  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File mappings: candidate dict keys -> agent directory paths
# ---------------------------------------------------------------------------

CODEGEN_FILE_MAPPING = {
    "eval_instructions": "eval_instructions.md",
    "tool_code": "tools/problem_analyzer.py",
}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

# Date cutoff matching CodeGenDomain.TEST_SET_CUTOFF
TEST_SET_CUTOFF = "2024-11-01"


def _bootstrap_cache(cache_dir: Path) -> None:
    """
    Bootstrap cache from HuggingFace dataset (problem.md + meta.json only).

    Called lazily when build_codegen_dataset() finds an empty cache.
    Solution.py is NOT created here — it's generated on demand by
    regenerate_coder() when the problem is first evaluated.
    """
    tools_dir = str(Path(__file__).parent.parent / "tools")
    project_root = str(Path(__file__).parent.parent.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from evaluate_livecodebench import load_dataset
    from run_critic_evaluation import ensure_cache_entry

    dataset = load_dataset()
    logger.info(f"Bootstrapping cache with {len(dataset)} problems from HuggingFace")

    cache_dir.mkdir(parents=True, exist_ok=True)
    for question_id, problem_data in sorted(dataset.items()):
        ensure_cache_entry(cache_dir, question_id, problem_data)

    logger.info(f"Cache bootstrapped: {len(dataset)} entries in {cache_dir}")


def _enumerate_cache(cache_dir: Path, split: str) -> List[Dict[str, str]]:
    """Enumerate cache entries matching the requested split."""
    examples = []
    for problem_dir in sorted(cache_dir.iterdir()):
        if not problem_dir.is_dir():
            continue

        meta_path = problem_dir / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        contest_date = meta.get("contest_date", "")

        # Apply date filtering
        if split == "test" and contest_date < TEST_SET_CUTOFF:
            continue
        if split == "evolution" and contest_date >= TEST_SET_CUTOFF:
            continue

        examples.append({"question_id": meta.get("question_id", problem_dir.name)})

    return examples


def build_codegen_dataset(
    cache_dir: Path,
    split: str = "evolution",
) -> List[Dict[str, str]]:
    """
    Build a flat list of example dicts from the codegen cache.

    Each example is lightweight: {"question_id": "abc314_c"}.
    Hidden test data is NOT included — the evaluator resolves it internally
    by question_id, preventing leakage to GEPA's reflection LM.

    Bootstraps the cache from HuggingFace if no entries match the requested
    split (creates problem.md + meta.json only; solution.py is generated on
    demand during evaluation).

    Args:
        cache_dir: Path to codegen_cache/<model>_v6/ directory.
        split: "evolution" (before cutoff) or "test" (on/after cutoff).

    Returns:
        List of example dicts sorted by question_id.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    examples = _enumerate_cache(cache_dir, split)
    if not examples:
        _bootstrap_cache(cache_dir)
        examples = _enumerate_cache(cache_dir, split)
    return examples


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RoboPhDCodeGenEvaluator:
    """
    GEPA-compatible evaluator wrapping RoboPhD's codegen-critic pipeline.

    Implements the signature: evaluator(candidate, example) -> (score, diagnostics)

    The evaluator holds heavy state (HuggingFace dataset, cache dir) and
    materializes the candidate to disk only when it changes between calls.
    """

    def __init__(
        self,
        coder_model: str = "haiku-4.5",
        critic_model: str = "haiku-4.5",
        cache_dir: Optional[Path] = None,
        work_dir: Optional[Path] = None,
        codegen_timeout: int = 1200,
        critic_timeout: int = 600,
        file_mapping: Optional[Dict[str, str]] = None,
        lmstudio_base_url: str = "http://localhost:1234",
    ):
        """
        Args:
            coder_model: Model for code generation (determines cache path if cache_dir not set).
            critic_model: Model for critic evaluation.
            cache_dir: Path to codegen_cache directory. Auto-derived from coder_model if None.
            work_dir: Working directory for materialized agents and evaluation output.
                     Defaults to ./gepa_codegen_work/.
            codegen_timeout: Timeout for code generation calls.
            critic_timeout: Timeout for critic calls.
            file_mapping: Candidate key -> file path mapping. Defaults to CODEGEN_FILE_MAPPING.
            lmstudio_base_url: Base URL for LM Studio models.
        """
        self.coder_model = coder_model
        self.critic_model = critic_model
        self.codegen_timeout = codegen_timeout
        self.critic_timeout = critic_timeout
        self.file_mapping = file_mapping or CODEGEN_FILE_MAPPING
        self.lmstudio_base_url = lmstudio_base_url

        # Resolve cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            cache_model_name = coder_model.replace("/", "--")
            runs_dir = Path("../robophd_runs")
            self.cache_dir = runs_dir / "codegen_cache" / f"{cache_model_name}_v6"

        # Working directory for materialized agents
        self.work_dir = Path(work_dir) if work_dir else Path("gepa_codegen_work")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded heavy state
        self._hf_dataset: Optional[Dict] = None
        self._candidate_hash: Optional[str] = None
        self._agent_dir: Optional[Path] = None
        self._evaluator = None  # CriticEvaluator instance
        self._eval_count = 0
        self._total_eval_cost = 0.0

        # Thread safety for concurrent __call__ from GEPA's ThreadPoolExecutor
        self._lock = threading.Lock()
        self._hf_lock = threading.Lock()

        # One-time sys.path setup for tools imports
        self._paths_configured = False

    def _ensure_paths(self) -> None:
        """Add tools directories to sys.path once."""
        if self._paths_configured:
            return
        tools_dir = str(Path(__file__).parent.parent / "tools")
        project_root = str(Path(__file__).parent.parent.parent)
        sys.path.insert(0, tools_dir)
        sys.path.insert(0, project_root)
        self._paths_configured = True

    def _load_hf_dataset(self) -> Dict:
        """Lazy-load HuggingFace dataset (for test evaluation)."""
        if self._hf_dataset is not None:
            return self._hf_dataset
        with self._hf_lock:
            if self._hf_dataset is None:
                self._ensure_paths()
                from evaluate_livecodebench import load_dataset
                self._hf_dataset = load_dataset()
        return self._hf_dataset

    def _candidate_fingerprint(self, candidate: Dict[str, str]) -> str:
        """Hash candidate contents to detect changes."""
        content = json.dumps(candidate, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _ensure_agent_materialized(self, candidate: Dict[str, str]) -> Path:
        """
        Materialize candidate to disk if it changed since last call.

        Returns path to the agent directory.
        """
        fp = self._candidate_fingerprint(candidate)
        if fp == self._candidate_hash and self._agent_dir is not None:
            return self._agent_dir

        # New candidate — materialize and reset evaluator
        agent_dir = self.work_dir / "current_agent"
        if agent_dir.exists():
            shutil.rmtree(agent_dir)

        materialize_candidate(candidate, agent_dir, self.file_mapping)
        self._candidate_hash = fp
        self._agent_dir = agent_dir
        self._evaluator = None  # Force re-creation with new agent
        return agent_dir

    def _get_evaluator(self, agent_dir: Path):
        """Get or create CriticEvaluator instance."""
        if self._evaluator is not None:
            return self._evaluator

        self._ensure_paths()
        from run_critic_evaluation import CriticEvaluator

        output_dir = self.work_dir / "eval_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        self._evaluator = CriticEvaluator(
            cache_dir=self.cache_dir,
            critic_agent_dir=agent_dir,
            output_dir=output_dir,
            coder_model=self.coder_model,
            critic_model=self.critic_model,
            codegen_timeout=self.codegen_timeout,
            critic_timeout=self.critic_timeout,
            lmstudio_base_url=self.lmstudio_base_url,
        )
        return self._evaluator

    def _collect_diagnostics(self, problem_dir: Path) -> Dict[str, str]:
        """
        Collect Actionable Side Information (ASI) from the problem directory.

        Returns the per-problem execution trace: the actual file contents
        providing the full input -> analysis -> decision -> outcome chain.
        """
        diagnostics = {}
        files_to_collect = [
            ("problem_statement", "problem.md"),
            ("solution_v1", "solution.py"),
            ("tool_analysis", "tool_output/analysis.txt"),
            ("critic_feedback", "feedback.md"),
            ("solution_v2", "solution_v2.py"),
            ("acceptance", "acceptance.md"),
            ("result", "result.json"),
        ]

        for key, filename in files_to_collect:
            filepath = problem_dir / filename
            if filepath.exists() and not filepath.is_symlink():
                try:
                    diagnostics[key] = filepath.read_text()
                except Exception:
                    pass
            elif filepath.is_symlink():
                # Resolve symlinks and read the actual content so GEPA's
                # reflection model sees the real data (not a placeholder).
                try:
                    diagnostics[key] = filepath.resolve().read_text()
                except Exception:
                    pass

        return diagnostics

    def __call__(
        self,
        candidate: Dict[str, str],
        example: Dict[str, str],
        *,
        problem_dir: Optional[Path] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a candidate on a single example.

        GEPA calls this as: evaluator(candidate, example) -> (score, diagnostics)
        RoboPhD calls with: evaluator(candidate, example, problem_dir=path)

        Args:
            candidate: Dict with "eval_instructions" and optionally "tool_code".
            example: Dict with "question_id".
            problem_dir: When provided (RoboPhD path), write per-problem artifacts
                and result.json here. When None (GEPA path), use internal work_dir.

        Returns:
            (score, diagnostics) where score is 1.0 if v2 passed, 0.0 otherwise.
        """
        question_id = example["question_id"]

        # Lock guards shared mutable state so concurrent threads don't race on
        # agent materialization. evaluate_problem() runs outside the lock since
        # it uses per-problem output dirs. NOTE: this assumes all concurrent
        # callers pass the same candidate (GEPA val sweeps). If callers ever
        # pass different candidates concurrently, the shared agent dir would be
        # overwritten mid-evaluation — that would need per-candidate dirs.
        with self._lock:
            self._eval_count += 1
            agent_dir = self._ensure_agent_materialized(candidate)
            evaluator = self._get_evaluator(agent_dir)

        # Resolve problem data from cache
        cache_problem_dir = self.cache_dir / question_id
        if not cache_problem_dir.exists():
            logger.warning(f"Cache dir not found for {question_id}")
            return 0.0, {"error": f"cache_dir_missing: {question_id}"}

        # Load problem data from HF dataset for test evaluation
        hf_dataset = self._load_hf_dataset()
        problem_data = hf_dataset.get(question_id, {})
        if not problem_data:
            logger.warning(f"Problem {question_id} not found in HF dataset")
            return 0.0, {"error": f"hf_dataset_missing: {question_id}"}

        # Determine output directory
        if problem_dir is not None:
            output_problem_dir = Path(problem_dir)
            output_problem_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_problem_dir = self.work_dir / "eval_output" / "problems" / question_id
            if output_problem_dir.exists():
                shutil.rmtree(output_problem_dir)
            output_problem_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        try:
            result = evaluator.evaluate_problem(
                cache_problem_dir=cache_problem_dir,
                output_problem_dir=output_problem_dir,
                problem_data=problem_data,
                question_id=question_id,
            )
        except Exception as e:
            logger.error(f"evaluate_problem failed for {question_id}: {e}")
            return 0.0, {"error": str(e), "question_id": question_id}

        # Accumulate cost
        eval_cost = result.get("timing", {}).get("total_cost_usd", 0.0)
        with self._lock:
            self._total_eval_cost += eval_cost

        # Score: 1.0 if v2 passed, 0.0 otherwise
        score = 1.0 if result.get("v2_passed", False) else 0.0

        # Write result.json for RoboPhD eval cache
        if problem_dir is not None:
            result_entry = {
                "question_id": question_id,
                "correct": score >= 0.5,
                "score": score,
                "v1_passed": result.get("v1_passed", False),
                "v2_passed": result.get("v2_passed", False),
                "improved": result.get("improved", False),
                "regressed": result.get("regressed", False),
            }
            with open(output_problem_dir / "result.json", "w") as f:
                json.dump(result_entry, f, indent=2)

        # Collect diagnostics (ASI)
        diagnostics = self._collect_diagnostics(output_problem_dir)
        diagnostics["result_json"] = json.dumps(result, indent=2)
        diagnostics["question_id"] = question_id
        diagnostics["score"] = score
        diagnostics["v1_passed"] = result.get("v1_passed", False)
        diagnostics["v2_passed"] = result.get("v2_passed", False)
        diagnostics["improved"] = result.get("improved", False)
        diagnostics["regressed"] = result.get("regressed", False)

        return score, diagnostics

    @property
    def total_evaluations(self) -> int:
        """Total number of evaluations performed."""
        return self._eval_count

    @property
    def total_eval_cost(self) -> float:
        """Total evaluation cost in USD."""
        return self._total_eval_cost
