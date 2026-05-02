"""Baseline DiscoveryBench solver.

Demonstrates the calling conventions evolution can mutate:
  - Copying Sample.files into the Docker sandbox via sandbox().write_file()
  - Calling python_session for stateful Python (pandas, etc.)
  - Calling the Inspect-tracked LLM via get_model().generate()
  - Emitting {hypothesis, workflow} JSON for the scorer

The seed is a one-shot baseline: copy data → describe → ask LLM to draft a
hypothesis from the description. It does no real analysis. Evolution is
expected to introduce statistical tests, multi-step reasoning, and
score-type-appropriate strategies. See background.md for the full surface.
"""

import json

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef
from inspect_ai.util import sandbox


def _get_tool(state: TaskState, name: str):
    """Find a tool in state.tools by its registered name."""
    for t in state.tools:
        if ToolDef(t).name == name:
            return t
    raise RuntimeError(
        f"tool {name!r} not in state.tools "
        f"(have: {[ToolDef(t).name for t in state.tools]})"
    )


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` fencing if present."""
    s = text.strip()
    if s.startswith("```"):
        # drop leading fence (and optional language tag)
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata["query"]
        datasets = state.metadata["metadata"]["datasets"]
        print(f"[{state.sample_id}] {query[:120]!r}")

        # --- 1. Copy CSVs into the sandbox (NOT auto-mounted) -------------
        for sb_path, host_path in (state.files or {}).items():
            with open(host_path, "rb") as f:
                await sandbox().write_file(f"/workspace/{sb_path}", f.read())
        print(f"  copied {len(state.files or {})} file(s) into sandbox")

        # --- 2. python_session: load + describe one CSV (stateful) -------
        py = _get_tool(state, "python_session")
        first = next(iter((state.files or {}).keys()), None)
        described = ""
        if first:
            described = await py(code=(
                f"import pandas as pd\n"
                f"df = pd.read_csv('/workspace/{first}')\n"
                f"print('shape:', df.shape)\n"
                f"print(df.describe(include='all').to_string())\n"
            ))
            print(f"  python_session described df ({len(described)} chars)")

        # --- 3. Inspect-tracked LLM call: draft hypothesis JSON ---------
        ds_summary_lines = []
        for d in datasets:
            cols = d["columns"]["raw"] if isinstance(d["columns"], dict) else d["columns"]
            ds_summary_lines.append(
                f"- {d['name']}: {d['description']}\n  columns: "
                + ", ".join(c["name"] for c in cols)
            )
        ds_summary = "\n".join(ds_summary_lines)

        prompt = (
            f"Query: {query}\n\nDatasets:\n{ds_summary}\n\n"
            f"Data summary:\n{described[:3000]}\n\n"
            f"Reply with a JSON object: "
            f'{{"hypothesis": "...", "workflow": "..."}}\n\n'
            f"The hypothesis must name the variables, their relationship, "
            f"the context/scope under which the relationship holds, and "
            f"any supporting numeric evidence. The workflow describes the "
            f"analysis steps that support the hypothesis."
        )
        resp = await get_model().generate(prompt)
        try:
            output = json.loads(_strip_code_fence(resp.completion))
            assert isinstance(output, dict)
            output.setdefault("hypothesis", "")
            output.setdefault("workflow", "")
        except (json.JSONDecodeError, AssertionError):
            # Scorer accepts raw text fallback (gen_workflow="").
            output = {"hypothesis": resp.completion.strip(), "workflow": ""}

        state.output.completion = json.dumps(output)
        print(f"  hypothesis: {output.get('hypothesis', '')[:120]!r}")
        return state

    return solve
