"""
Task definition: the common shape every task must provide.

A task is a benchmark + evaluator that both GEPA and RoboPhD can optimize.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class TaskDefinition:
    """
    Everything an engine needs to run optimization on a task.

    Attributes:
        name: Short identifier (e.g. "codegen").
        description: Human-readable description.
        evaluator_factory: (config) -> evaluator_fn.
            The returned evaluator has signature:
            evaluator(candidate: dict, example: dict, *, problem_dir: Path = None) -> (float, dict)
            Must be thread-safe: RoboPhD calls it concurrently from a ThreadPoolExecutor.
        dataset_builder: (config) -> [examples].
            Returns a flat list of example dicts (e.g. [{"question_id": "abc314_c"}, ...]).
        file_mapping: Candidate key -> agent file path.
            Maps flat candidate dict keys to relative paths inside agent directories.
        default_seed_agent: Path from project root to the default seed agent directory.
        objective: Concise optimization goal (1-2 sentences).
        background: Rich domain documentation for evolution context (multi-paragraph).
            Written as CLAUDE.md into evolution working directories for Claude Code.
            Passed as `background` param to GEPA. Empty string means no background.
        diagnostic_files: Per-problem diagnostic files written by the evaluator.
            Maps filename -> description (e.g. {"problem.md": "Problem statement"}).
            Used in evolution prompts to tell the AI what files are available.
        config_defaults: Merged into config for both engines (lowest priority).
    """

    name: str
    description: str
    evaluator_factory: Callable[..., Callable]
    dataset_builder: Callable[..., List[Dict]]
    file_mapping: Dict[str, str]
    default_seed_agent: str
    objective: str = ""
    background: str = ""
    diagnostic_files: Dict[str, str] = field(default_factory=dict)
    config_defaults: Dict[str, Any] = field(default_factory=dict)
    test_overrides: Dict[str, Any] = field(default_factory=dict)
    """Config overrides applied when building the test dataset (e.g. switching split)."""

    gepa_datasets_builder: Optional[Callable[..., Tuple[List[Dict], List[Dict]]]] = None
    """Optional pre-split dataset builder for GEPA: (config) -> (trainset, valset).
    When set, run_gepa.py uses these splits directly instead of shuffle + val_ratio."""
