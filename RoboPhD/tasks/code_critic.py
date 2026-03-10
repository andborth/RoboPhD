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
    project_root = Path(__file__).parent.parent.parent
    return project_root / "RoboPhD" / "data" / "code_critic" / "cache"


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


_CODE_CRITIC_BACKGROUND = """\
## Coder/Critic Architecture (Fresh-Session Variant)

The CodeCritic domain is a variant of CodeGen that uses a **fixed, pre-generated
code cache** and a **fresh-session revision** instead of session forking.

### Key Differences from CodeGen

1. **No code generation**: Solutions are pre-cached. The evaluator never calls
   `regenerate_coder()` or `_bootstrap_cache()`.
2. **Fresh-session revision**: Instead of forking the coder's original session,
   a fresh Claude session reads problem.md, solution.py, and reflection.md to
   understand the solution, then receives critic feedback in a follow-up message.
3. **Persuasiveness test**: The coder forms an independent understanding of the
   solution before seeing feedback. This tests whether the critic can persuade
   a coder that has already internalized the solution approach.

### Workflow

```
Phase 1: (skipped — solution.py from cache)

Phase 2: Critic Review (tool-only + eval LLM)
  Tool analyzes code -> eval LLM produces verdict + feedback.
  Verdict: CORRECT or INCORRECT.

  If CORRECT -> v2 = symlink to v1, skip to evaluation.
  If INCORRECT -> proceed to Phase 3.

Phase 3: Fresh-Session Revision (Coder)
  Call 1: Fresh session reads problem.md, solution.py, reflection.md.
          Summarizes understanding of problem and solution approach.
  Call 2: Same session receives critic feedback. Optionally revises
          to solution_v2.py.

Phase 3.5: Acceptance Query
  Post-hoc query: ACCEPTED_ALL / ACCEPTED_SOME / REJECTED_ALL.

Phase 4: Evaluation (Ground Truth)
  Tests both v1 and v2 against hidden test suite.
  6-second timeout per test, one retry on timeout.
  Binary outcome: pass (all tests) / fail (any test fails).
```

### What The Critic Controls

The evolved critic consists of two files:

1. **`eval_instructions.md`** — Decision framework for the eval LLM.
   Guides the critic's verdict (CORRECT/INCORRECT) and feedback content.

2. **`tools/problem_analyzer.py`** — Static analysis script.
   Reads `solution.py` and `problem.md` from its working directory,
   performs analysis, and writes findings to `tool_output/analysis.txt`.

### Scoring

- **Improved**: v1 fail -> v2 pass (critic helped fix a bug)
- **Regressed**: v1 pass -> v2 fail (critic broke working code)
- The binary score for each problem: 1.0 if v2 passes, 0.0 if v2 fails

### Dataset: LiveCodeBench v6

Total problems: 1055 (May 2023 - April 2025).
Evolution split: 767 problems (May 2023 - Oct 2024).
Test split: 288 problems (Nov 2024 - Apr 2025), never seen during evolution.
"""


def make_code_critic_task() -> TaskDefinition:
    """Create the CodeCritic task definition."""
    from RoboPhD.adapters.gepa_codegen import CODEGEN_FILE_MAPPING

    return TaskDefinition(
        name="code_critic",
        description="Evolve critic agents with fresh-session revision on LiveCodeBench",
        evaluator_factory=_evaluator_factory,
        dataset_builder=_dataset_builder,
        file_mapping=CODEGEN_FILE_MAPPING,
        default_seed_agent="RoboPhD/codegen_agents/naive_critic",
        objective=(
            "Optimize the critic agent to accurately identify incorrect code solutions "
            "and provide actionable, persuasive feedback. The revision uses a fresh session "
            "where the coder independently reads the problem and solution before seeing "
            "feedback, testing whether the critic's suggestions are compelling enough to "
            "change the coder's mind."
        ),
        background=_CODE_CRITIC_BACKGROUND,
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
            "evaluation_budget": 1600,
            "max_workers": 12,
        },
        test_overrides={"codegen_split": "test"},
    )
