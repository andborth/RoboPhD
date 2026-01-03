#!/usr/bin/env python3
"""
Test script for extract_error_details MCP tool.

This script directly calls the tool logic without going through MCP protocol,
useful for quick testing from command line.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from RoboPhD.tools.error_analysis.mcp_server import call_tool


async def main():
    parser = argparse.ArgumentParser(
        description='Test extract_error_details MCP tool'
    )
    parser.add_argument(
        '--question-ids',
        type=str,
        required=True,
        help='Comma-separated question IDs (e.g., "1234,5678")'
    )
    parser.add_argument(
        '--iteration-dirs',
        type=str,
        help='Comma-separated iteration directories (optional, will auto-detect if not provided)'
    )
    parser.add_argument(
        '--agent',
        type=str,
        help='Filter to specific agent (optional)'
    )

    args = parser.parse_args()

    # Build arguments for tool
    tool_args = {
        "question_ids": [qid.strip() for qid in args.question_ids.split(',')]
    }

    if args.iteration_dirs:
        tool_args["iteration_dirs"] = [d.strip() for d in args.iteration_dirs.split(',')]

    if args.agent:
        tool_args["agent"] = args.agent

    # Call the tool
    print(f"Calling extract_error_details with:", file=sys.stderr)
    print(f"  question_ids: {tool_args['question_ids']}", file=sys.stderr)
    if "iteration_dirs" in tool_args:
        print(f"  iteration_dirs: {tool_args['iteration_dirs']}", file=sys.stderr)
    if "agent" in tool_args:
        print(f"  agent: {tool_args['agent']}", file=sys.stderr)
    print("", file=sys.stderr)

    result = await call_tool("extract_error_details", tool_args)

    # Print result
    for content in result:
        print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
