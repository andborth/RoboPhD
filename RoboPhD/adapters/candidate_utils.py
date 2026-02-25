"""
Generic candidate <-> agent directory conversion utilities.

These are task-agnostic: they operate purely on file_mapping dicts
that map candidate keys to relative paths within agent directories.
"""

from pathlib import Path
from typing import Dict


def _build_agent_md(name: str, file_mapping: Dict[str, str], has_tool: bool) -> str:
    """Generate agent.md content, deriving tool paths from file_mapping."""
    if has_tool:
        tool_path = file_mapping.get("tool_code", "tools/tool.py")
        if tool_path.endswith(".py"):
            tool_command = f"python {tool_path}"
        else:
            tool_command = tool_path
        tool_output_file = "tool_output/output.txt"
        return (
            f"---\n"
            f"name: {name}\n"
            f"description: Evolved agent\n"
            f"execution_mode: tool_only\n"
            f"tool_command: {tool_command}\n"
            f"tool_output_file: {tool_output_file}\n"
            f"---\n\n"
            f"# {name}\n\n"
            f"Evolved agent.\n"
        )
    else:
        return (
            f"---\n"
            f"name: {name}\n"
            f"description: Evolved agent (no tool)\n"
            f"---\n\n"
            f"# {name}\n\n"
            f"Evolved agent (no tool component).\n"
        )


def materialize_candidate(
    candidate: Dict[str, str],
    target_dir: Path,
    file_mapping: Dict[str, str],
    name: str = "gepa_agent",
) -> Path:
    """
    Write a candidate dict to a RoboPhD agent directory.

    For each (key, filepath) in file_mapping, writes candidate[key] to
    target_dir/filepath. Generates agent.md with appropriate YAML frontmatter,
    deriving tool_command and tool_output_file from file_mapping.

    Args:
        candidate: Dict mapping component names to text content.
        target_dir: Directory to write agent files into.
        file_mapping: Maps candidate keys to relative file paths within the agent dir.
        name: Agent name for the generated agent.md.

    Returns:
        target_dir (for chaining).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    has_tool = "tool_code" in candidate and candidate["tool_code"].strip()

    for key, filepath in file_mapping.items():
        if key not in candidate:
            continue
        dest = target_dir / filepath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(candidate[key])

    agent_md_path = target_dir / "agent.md"
    agent_md_path.write_text(_build_agent_md(name, file_mapping, has_tool))

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
