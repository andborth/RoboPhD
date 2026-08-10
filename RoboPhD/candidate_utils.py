"""
Generic candidate <-> agent directory conversion utilities.

These are task-agnostic: they operate purely on file_mapping dicts
that map candidate keys to relative paths within agent directories.
"""

from pathlib import Path
from typing import Dict


def materialize_candidate(
    candidate: Dict[str, str],
    target_dir: Path,
    file_mapping: Dict[str, str],
    name: str = "gepa_agent",
) -> Path:
    """
    Write a candidate dict to a RoboPhD agent directory.

    For each (key, filepath) in file_mapping, writes candidate[key] to
    target_dir/filepath.

    Args:
        candidate: Dict mapping component names to text content.
        target_dir: Directory to write agent files into.
        file_mapping: Maps candidate keys to relative file paths within the agent dir.
        name: Agent name (unused, kept for API compat).

    Returns:
        target_dir (for chaining).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for key, filepath in file_mapping.items():
        if key not in candidate:
            continue
        dest = target_dir / filepath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(candidate[key])

    return target_dir


def extract_candidate(
    agent_dir: Path,
    file_mapping: Dict[str, str],
) -> Dict[str, str]:
    """
    Read a RoboPhD agent directory into a GEPA candidate dict.

    For each (key, filepath) in file_mapping, reads agent_dir/filepath
    into candidate[key], returning "" if the file is missing.

    Args:
        agent_dir: Path to agent directory.
        file_mapping: Maps candidate keys to relative file paths within the agent dir.

    Returns:
        Dict mapping component names to text content.
    """
    candidate = {}
    for key, filepath in file_mapping.items():
        src = agent_dir / filepath
        candidate[key] = src.read_text() if src.exists() else ""
    return candidate


# Directory entries that are build or tooling residue rather than agent
# artifacts. `examples/asta_ds1000/seeds/baseline/` carries a stray
# `__pycache__/agent.cpython-311.pyc` on disk (gitignored, so absent from the
# repo but present after any local run), and without this filter it would be
# handed to evolution as an editable artifact.
_IGNORED_COMPONENTS = {"__pycache__"}


def read_agent_dir(agent_dir: Path) -> Dict[str, str]:
    """
    Read every artifact in an agent directory into a candidate dict.

    The inverse of materialize_candidate for the case where the file_mapping
    is not known in advance: keys are paths relative to agent_dir, so the
    caller can derive the mapping from the result. Use extract_candidate
    instead when the mapping is already fixed.

    Skips dot-prefixed entries and build residue at any depth. Nested paths
    are preserved as keys ("lib/util.py"); materialize_candidate recreates
    the parent directories.

    Args:
        agent_dir: Path to an agent directory.

    Returns:
        Dict mapping relative path to text content, ordered by path.

    Raises:
        FileNotFoundError: agent_dir is missing or is not a directory.
        ValueError: agent_dir contains no readable artifacts, or one of them
            is not UTF-8 text (candidates are text; a binary artifact means
            the caller is pointing at the wrong directory).
    """
    agent_dir = Path(agent_dir)
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"Seed agent directory not found: {agent_dir}")

    candidate = {}
    for path in sorted(agent_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(agent_dir)
        if any(
            part.startswith(".") or part in _IGNORED_COMPONENTS
            for part in relative.parts
        ):
            continue
        try:
            candidate[str(relative)] = path.read_text()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Seed artifact {path} is not UTF-8 text. Agent artifacts are "
                f"text files; check that {agent_dir} is an agent directory."
            ) from exc

    if not candidate:
        raise ValueError(f"Seed agent directory has no artifacts: {agent_dir}")
    return candidate
