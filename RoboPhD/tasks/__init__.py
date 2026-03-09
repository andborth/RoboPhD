"""
Task registry: maps task names to TaskDefinition instances.

Usage:
    from RoboPhD.tasks import get_task, list_tasks

    task = get_task("codegen")
    print(task.name, task.file_mapping)
    print(list_tasks())
"""

from typing import Dict, List

from .base import TaskDefinition

_REGISTRY: Dict[str, TaskDefinition] = {}
_BUILTINS_REGISTERED = False


def register_task(task: TaskDefinition) -> None:
    """Register a task definition."""
    _REGISTRY[task.name] = task


def _ensure_builtins():
    """Lazily register built-in tasks on first access."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    from .codegen import make_codegen_task
    register_task(make_codegen_task())

    from .code_critic import make_code_critic_task
    register_task(make_code_critic_task())

    from .aime import make_aime_task
    register_task(make_aime_task())

    from .text2sql import make_text2sql_task
    register_task(make_text2sql_task())

    try:
        from .arc_agi import make_arc_agi_task
        register_task(make_arc_agi_task())
    except ImportError:
        pass  # dspy/datasets not installed — arc_agi task unavailable


def get_task(name: str) -> TaskDefinition:
    """Look up a task by name. Raises KeyError if not found."""
    _ensure_builtins()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown task: {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_tasks() -> List[str]:
    """Return sorted list of registered task names."""
    _ensure_builtins()
    return sorted(_REGISTRY.keys())
