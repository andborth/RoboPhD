"""
CodeCritic task definition: codegen variant with fresh-session revision.

Uses a fixed, pre-generated cache (no code generation) and replaces
session-forking revision with a fresh session that reads problem.md,
solution.py, and reflection.md before receiving critic feedback.
"""

from pathlib import Path
from typing import Any, Dict, List

from .base import TaskDefinition


def _resolve_cache_dir(config: Dict[str, Any]) -> Path:
    """Resolve codegen cache directory from config."""
    cache_dir = config.get("cache_dir")
    if cache_dir is not None:
        return Path(cache_dir)
    # Default: in-repo cache (self-contained, rebuild with scripts/rebuild_code_critic_cache.py)
    return Path(__file__).parent.parent / "data" / "code_critic" / "cache"


def _evaluator_factory(config: Dict[str, Any]):
    """Build a RoboPhDCodeGenEvaluator with fresh revision mode."""
    from RoboPhD.adapters.gepa_codegen import RoboPhDCodeGenEvaluator, CODEGEN_FILE_MAPPING

    cache_dir = _resolve_cache_dir(config)

    work_dir = config.get("work_dir")
    if work_dir is None:
        work_dir = Path(config.get("output_dir", "gepa_runs/work")) / "work"

    return RoboPhDCodeGenEvaluator(
        coder_model=config.get("coder_model", "haiku-4.5"),
        critic_model=config.get("critic_model") or config.get("coder_model", "haiku-4.5"),
        cache_dir=cache_dir,
        work_dir=work_dir,
        codegen_timeout=config.get("codegen_timeout", 1200),
        critic_timeout=config.get("critic_timeout", 600),
        file_mapping=CODEGEN_FILE_MAPPING,
        revision_mode="fresh",
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build codegen dataset from cache, filtering to entries with solution.py."""
    from RoboPhD.adapters.gepa_codegen import build_codegen_dataset

    cache_dir = _resolve_cache_dir(config)
    split = config.get("codegen_split", "evolution")
    examples = build_codegen_dataset(cache_dir, split=split)

    # Filter to entries that have both solution.py and reflection.md
    # (cache is read-only in fresh mode; reflection.md is needed for call 1)
    filtered = []
    for ex in examples:
        entry_dir = cache_dir / ex["question_id"]
        if (entry_dir / "solution.py").exists() and (entry_dir / "reflection.md").exists():
            filtered.append(ex)

    if len(filtered) < len(examples):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"code_critic: filtered {len(examples)} -> {len(filtered)} "
            f"(excluding entries without solution.py or reflection.md)"
        )

    return filtered


def make_code_critic_task() -> TaskDefinition:
    """Create the CodeCritic task definition."""
    from RoboPhD.adapters.gepa_codegen import CODEGEN_FILE_MAPPING
    from .codegen import _CODEGEN_BACKGROUND, _CODEGEN_OBJECTIVE

    return TaskDefinition(
        name="code_critic",
        description="Evolve critic agents with fresh-session revision on LiveCodeBench",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=CODEGEN_FILE_MAPPING,
        default_seed_agent="RoboPhD/codegen_agents/naive_critic",
        objective=_CODEGEN_OBJECTIVE,
        background=_CODEGEN_BACKGROUND,
        diagnostic_files={
            "problem.md": "Problem statement with examples",
            "solution.py": "Initial code solution (v1)",
            "reflection.md": "Coder's original reflection on the solution",
            "feedback.md": "Critic's verdict (CORRECT/INCORRECT) and feedback",
            "solution_v2.py": "Revised code after critic feedback (or symlink to v1 if CORRECT)",
            "acceptance.md": "Coder's explanation of changes",
            "tool_output/analysis.txt": "Tool-generated static analysis",
        },
        config_defaults={
            "coder_model": "haiku-4.5",
            "coder_model_tag": "",
            "critic_model": "haiku-4.5",
            "codegen_split": "evolution",
            "codegen_timeout": 1200,
            "codegen_call_timeout": 1200,
            "critic_timeout": 600,
            "evaluation_budget": 1500,
            "max_workers": 12,
        },
        test_overrides={"codegen_split": "test"},
    )
