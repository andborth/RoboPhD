"""
Task definition: the common shape every task must provide.

A task is a benchmark + evaluator that both GEPA and RoboPhD can optimize.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


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
        dataset_builder: (config) -> [examples].
            Returns a flat list of example dicts (e.g. [{"question_id": "abc314_c"}, ...]).
        file_mapping: Candidate key -> agent file path.
            Maps flat candidate dict keys to relative paths inside agent directories.
        default_seed_agent: Path from project root to the default seed agent directory.
        objective: Text describing what GEPA should optimize for.
        config_defaults: Merged into config for both engines (lowest priority).
    """

    name: str
    description: str
    evaluator_factory: Callable[..., Callable]
    dataset_builder: Callable[..., List[Dict]]
    file_mapping: Dict[str, str]
    default_seed_agent: str
    objective: str
    config_defaults: Dict[str, Any] = field(default_factory=dict)
