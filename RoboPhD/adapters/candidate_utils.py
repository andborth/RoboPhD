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
