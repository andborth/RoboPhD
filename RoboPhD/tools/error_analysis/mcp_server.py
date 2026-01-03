#!/usr/bin/env python3
"""
MCP server for error analysis tools.

Provides extract_error_details tool for use in Claude Code sessions.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from RoboPhD.tools.error_analysis.extract_error_details import (
    find_question_in_evaluations,
    strip_agent_prefix
)


# Create server instance
app = Server("error-analysis")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="extract_error_details",
            description=(
                "Extract detailed evaluation results for specific question IDs from iteration workspaces. "
                "Returns full evaluation data including SQL queries, predicted results, ground truth, "
                "verification attempts, and error details. Useful for deep-diving into specific failed questions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of question IDs to extract (e.g., ['1234', '5678'])"
                    },
                    "iteration_dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional: List of iteration directory paths to search. "
                            "If not provided, will auto-detect based on current working directory. "
                            "Paths can be relative (e.g., ['../../iteration_007']) or absolute."
                        )
                    },
                    "agent": {
                        "type": "string",
                        "description": "Optional: Filter results to specific agent only"
                    }
                },
                "required": ["question_ids"]
            }
        )
    ]


def auto_detect_iteration_dirs(cwd: Path) -> list[Path]:
    """
    Auto-detect iteration directories based on current working directory.

    Handles two cases:
    1. Deep Focus evolution workspace: searches for test iteration directories
    2. Normal evolution workspace: uses previous iteration
    """
    # Check for Deep Focus test round directories
    test_dirs = list(cwd.glob('round_*_test_iteration_*'))
    if test_dirs:
        # Deep Focus test round - use both previous iteration and test dir
        test_dir = sorted(test_dirs)[-1]
        prev_iter = int(test_dir.name.split('iteration_')[-1])
        return [cwd.parent.parent / f'iteration_{prev_iter:03d}', test_dir]

    # Check if we're in an evolution_output directory
    if 'evolution_output' in str(cwd):
        # Try to extract iteration number from path
        for part in cwd.parts:
            if part.startswith('iteration_'):
                try:
                    current_iter = int(part.split('_')[1])
                    prev_iter = current_iter - 1
                    # Navigate to research root and find previous iteration
                    # Typical structure: research/robophd_*/evolution_output/iteration_XXX/
                    research_root = None
                    for i, part in enumerate(cwd.parts):
                        if part == 'evolution_output':
                            research_root = Path(*cwd.parts[:i])
                            break
                    if research_root:
                        return [research_root / f'iteration_{prev_iter:03d}']
                except (ValueError, IndexError):
                    pass

    # Default: assume we're somewhere in research and try common patterns
    # Look for iteration directories in parent directories
    search_dir = cwd
    for _ in range(5):  # Search up to 5 levels up
        iteration_dirs = sorted(search_dir.glob('iteration_*'))
        if iteration_dirs:
            # Return the most recent iteration
            return [iteration_dirs[-1]]
        search_dir = search_dir.parent

    # Fallback: use current directory
    return [cwd]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    if name != "extract_error_details":
        raise ValueError(f"Unknown tool: {name}")

    # Extract arguments
    question_ids_list = arguments.get("question_ids", [])
    iteration_dirs_arg = arguments.get("iteration_dirs")
    agent_filter = arguments.get("agent")

    # Validate question IDs
    if not question_ids_list:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "No question IDs provided",
                "usage": "Provide at least one question ID in the question_ids array"
            }, indent=2)
        )]

    question_ids = set(str(qid) for qid in question_ids_list)

    # Determine iteration directories
    if iteration_dirs_arg:
        iteration_dirs = [Path(d) for d in iteration_dirs_arg]
    else:
        # Auto-detect based on current working directory
        cwd = Path.cwd()
        iteration_dirs = auto_detect_iteration_dirs(cwd)

    # Validate directories exist
    missing_dirs = [str(d) for d in iteration_dirs if not d.exists()]
    if missing_dirs:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "Iteration directory not found",
                "missing_directories": missing_dirs,
                "searched_from": str(Path.cwd())
            }, indent=2)
        )]

    # Extract details from all directories and merge
    all_results = {}
    search_summary = []

    for iteration_dir in iteration_dirs:
        results = find_question_in_evaluations(question_ids, iteration_dir)
        search_summary.append({
            "directory": str(iteration_dir),
            "questions_found": len(results),
            "agents_found": sum(len(agents) for agents in results.values())
        })

        # Merge results
        for qid, agents in results.items():
            if qid not in all_results:
                all_results[qid] = {}
            all_results[qid].update(agents)

    results = all_results

    # Filter by agent if specified
    if agent_filter:
        filtered_results = {}
        for qid, agents in results.items():
            # Try with and without agent_ prefix
            agent_key = agent_filter
            if agent_key not in agents:
                agent_key = strip_agent_prefix(agent_filter)
            if agent_key not in agents:
                agent_key = f"agent_{agent_filter}"

            if agent_key in agents:
                # Simplify structure when filtering to single agent
                filtered_results[qid] = agents[agent_key]

        results = filtered_results

        if not results:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "warning": f"No matching questions found for agent '{agent_filter}'",
                    "search_summary": search_summary,
                    "available_agents": list(set(
                        agent
                        for agents in all_results.values()
                        for agent in agents.keys()
                    ))
                }, indent=2)
            )]

    # Build summary statistics
    found_ids = set(results.keys())
    missing_ids = question_ids - found_ids

    # Check for cross-agent cases (only if not filtering by agent)
    cross_agent_cases = []
    if not agent_filter:
        cross_agent_cases = [
            {
                "question_id": qid,
                "agent_count": len(agents),
                "agents": list(agents.keys())
            }
            for qid, agents in results.items()
            if len(agents) > 1
        ]

    # Check for verification cases
    verification_cases = []
    if agent_filter:
        # Simple structure: {qid: result}
        for qid, result in results.items():
            verification_info = result.get('verification_info', {})
            attempts = verification_info.get('attempts', 1) if verification_info else 1
            if attempts > 1:
                verification_cases.append({
                    "question_id": qid,
                    "agent": agent_filter,
                    "attempts": attempts
                })
    else:
        # Multi-agent structure: {qid: {agent: result}}
        for qid, agents in results.items():
            for agent_name, result in agents.items():
                verification_info = result.get('verification_info', {})
                attempts = verification_info.get('attempts', 1) if verification_info else 1
                if attempts > 1:
                    verification_cases.append({
                        "question_id": qid,
                        "agent": agent_name,
                        "attempts": attempts
                    })

    # Build response
    response = {
        "summary": {
            "questions_requested": len(question_ids),
            "questions_found": len(found_ids),
            "missing_question_ids": sorted(missing_ids) if missing_ids else None,
            "agent_filter": agent_filter,
            "cross_agent_opportunities": len(cross_agent_cases) if cross_agent_cases else 0,
            "verification_retries": len(verification_cases) if verification_cases else 0
        },
        "search_summary": search_summary,
        "cross_agent_cases": cross_agent_cases if cross_agent_cases else None,
        "verification_cases": verification_cases if verification_cases else None,
        "results": results
    }

    # Clean up None values
    response = {k: v for k, v in response.items() if v is not None}
    response["summary"] = {k: v for k, v in response["summary"].items() if v is not None}

    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
