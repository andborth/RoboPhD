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


_CODEGEN_BACKGROUND = """\
## Coder/Critic Architecture

The CodeGen domain evolves a **critic agent** that reviews code and provides
feedback to the coder.

### Workflow

Phase 1: Initial Generation (Coder)
  Receives problem, generates initial solution (Code v1).
  Can execute code on visible examples only.

Phase 2: Critic Review
  Critic reviews code, produces verdict + feedback.
  Verdict: CORRECT or INCORRECT.
  If CORRECT → no revision, v2 = v1, skip to evaluation.
  If INCORRECT → proceed to Phase 3.

Phase 3: Revision (Coder)
  Receives critic feedback; has discretion to accept all, some, or none.
  Produces Code v2.

Phase 3.5: Acceptance Query
  Post-hoc query on the revision session. Coder explains its reasoning
  and categorizes: ACCEPTED_ALL / ACCEPTED_SOME / REJECTED_ALL.

Phase 4: Evaluation (Ground Truth)
  Tests both v1 and v2 against hidden test suite.
  6-second timeout per test.
  Binary outcome: pass (all tests) / fail (any test fails).

### What The Critic Controls

1. **`eval_instructions.md`** — Decision framework for the eval LLM.
   Guides the critic's verdict (CORRECT/INCORRECT) and feedback content.

2. **`tools/problem_analyzer.py`** — Static analysis script.
   Reads solution.py and problem.md, performs analysis, writes findings
   to tool_output/analysis.txt. The eval LLM uses this analysis
   alongside eval_instructions to render a verdict.

### What The Critic Does NOT Control

- The coder's initial solution (Phase 1)
- The coder's revision behavior (Phase 3) — the critic's feedback must be persuasive enough to guide the coder toward the right fix
- Test execution infrastructure (Phase 4)
- Problem selection or difficulty

### Visibility

Coder and critic can see: problem statement with examples, code execution
results on visible examples.

They cannot see: hidden test cases, edge cases, performance limits, or
whether the solution is actually correct beyond visible examples.

### Scoring

- Binary score: 1.0 if v2 passes all hidden tests, 0.0 otherwise.
  This is the score you are seeking to maximize.
- Improved: v1 fail → v2 pass (critic helped)
- Regressed: v1 pass → v2 fail (critic hurt)
- Verdict classification: TP (INCORRECT + v1 wrong), FP (INCORRECT + v1 right),
  TN (CORRECT + v1 right), FN (CORRECT + v1 wrong)
"""

_CODEGEN_OBJECTIVE = (
    "Optimize the critic agent to accurately identify incorrect code solutions "
    "and provide actionable feedback that helps the coder fix bugs. "
    "Your overall objective is to maximize the score on problems you have never seen before."
)


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
        objective=_CODEGEN_OBJECTIVE,
        background=_CODEGEN_BACKGROUND,
        diagnostic_files={
            "problem.md": "Problem statement with examples",
            "solution.py": "Initial code solution (v1)",
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
            "runs_dir": "../robophd_runs",
            "evaluation_budget": 1600,
            "max_workers": 12,
        },
        test_overrides={"codegen_split": "test"},
    )
