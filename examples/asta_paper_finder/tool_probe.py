#!/usr/bin/env python
"""Read-only probe for the Asta MCP corpus tools — evolution-SESSION tooling.

Calls the same eight tools the evaluation attaches to the agent, through
the same astabench wrapper stack (same snapshot date-cutoff, same field
defaults, same retry behavior), so output here is exactly what the
agent's own tool call would return.

Session use only: the agent may never import or shell out to this script
— its corpus access is ``state.tools`` (see the Standard Tools
constraint in the task documentation).

Usage::

    python tool_probe.py --list
    python tool_probe.py search_papers_by_relevance keyword="sparse attention" limit=5
    python tool_probe.py get_paper_batch ids='["CorpusId:123", "CorpusId:456"]' fields=title,year

Arguments after the tool name are ``key=value`` pairs. Values parse as
JSON when possible (``limit=5`` -> int, ``ids='["..."]'`` -> list) and
fall back to plain strings (``fields=title,year``).

Output is one JSON document per returned item — the raw payload exactly
as the agent's parse layer would see it (some tools wrap their payload
in a ``{"data": [...]}`` object — same as at eval time). Warnings on
stderr at startup (deprecations, registry entrypoint noise) come from
astabench's import machinery, are harmless, and don't touch stdout.

Requires ``ASTA_TOOL_KEY`` in the environment (present in evolution
sessions).
"""
import asyncio
import json
import os
import sys


def parse_kv(args):
    """Parse ``key=value`` CLI arguments into a kwargs dict.

    Values are JSON-decoded when possible so numbers, lists and objects
    arrive typed; anything that isn't valid JSON stays a plain string.
    """
    kwargs = {}
    for arg in args:
        if "=" not in arg:
            raise SystemExit(
                f"expected key=value, got {arg!r} "
                f"(quote values containing spaces: keyword=\"two words\")"
            )
        key, _, value = arg.partition("=")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    return kwargs


def build_tools():
    """Build the corpus tools through the evaluation's own wrapper stack."""
    from astabench.evals.paper_finder.paper_finder_utils import (
        get_inserted_before_per_dataset_type,
    )
    from astabench.tools import make_asta_mcp_tools

    # Same resolution the evaluator uses for PaperFindingBench samples
    # (any non-litqa2 dataset type maps to the PF snapshot date).
    inserted_before = get_inserted_before_per_dataset_type("paper_finder")
    return make_asta_mcp_tools(insertion_date=inserted_before)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if not os.environ.get("ASTA_TOOL_KEY"):
        print("ASTA_TOOL_KEY is not set in the environment.", file=sys.stderr)
        return 2

    from inspect_ai.tool import ToolDef

    tools = {ToolDef(t).name: t for t in build_tools()}

    if argv[0] == "--list":
        for name in sorted(tools):
            td = ToolDef(tools[name])
            params = ", ".join(td.parameters.properties) if td.parameters else ""
            print(f"{name}({params})")
        return 0

    name, kwargs = argv[0], parse_kv(argv[1:])
    tool = tools.get(name)
    if tool is None:
        print(
            f"unknown tool {name!r} — run with --list to see the eight available",
            file=sys.stderr,
        )
        return 2

    result = asyncio.run(tool(**kwargs))
    if isinstance(result, list):
        for item in result:
            print(getattr(item, "text", item))
    else:
        print(getattr(result, "text", result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
