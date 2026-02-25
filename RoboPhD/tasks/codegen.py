"""
CodeGen task definition: evolve critic agents for code review on LiveCodeBench.

Wires existing evaluator + dataset from the GEPA adapter.
"""

from pathlib import Path
from typing import Any, Dict, List

from .base import TaskDefinition


def _resolve_cache_dir(config: Dict[str, Any]) -> Path:
    """Resolve codegen cache directory from config."""
    cache_dir = config.get("cache_dir")
    if cache_dir is not None:
        return Path(cache_dir)
    cache_model_name = config.get("coder_model", "haiku-4.5").replace("/", "--")
    tag = config.get("coder_model_tag", "")
    if tag:
        cache_model_name = f"{cache_model_name}_{tag}"
    runs_dir = Path(config.get("runs_dir", "../robophd_runs"))
    return runs_dir / "codegen_cache" / f"{cache_model_name}_v6"


def _evaluator_factory(config: Dict[str, Any]):
    """Build a RoboPhDCodeGenEvaluator from merged config."""
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
    )


def _dataset_builder(config: Dict[str, Any]) -> List[Dict]:
    """Build codegen dataset from cache directory."""
    from RoboPhD.adapters.gepa_codegen import build_codegen_dataset

    cache_dir = _resolve_cache_dir(config)
    split = config.get("codegen_split", "evolution")
    return build_codegen_dataset(cache_dir, split=split)


def make_codegen_task() -> TaskDefinition:
    """Create the CodeGen task definition."""
    from RoboPhD.adapters.gepa_codegen import CODEGEN_FILE_MAPPING

    return TaskDefinition(
        name="codegen",
        description="Evolve critic agents for code review on LiveCodeBench",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=CODEGEN_FILE_MAPPING,
        default_seed_agent="RoboPhD/codegen_agents/naive_critic",
        objective=(
            "Optimize the critic agent to accurately identify incorrect code solutions "
            "and provide actionable feedback that helps the coder fix bugs. "
            "The eval_instructions guide the critic's verdict (CORRECT/INCORRECT) and feedback. "
            "The tool_code performs static analysis before the critic reviews the code."
        ),
        config_defaults={
            "coder_model": "haiku-4.5",
            "critic_model": "haiku-4.5",
            "codegen_timeout": 1200,
            "critic_timeout": 600,
            "runs_dir": "../robophd_runs",
        },
    )
