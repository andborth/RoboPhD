"""Baseline DiscoveryBench solver.

Demonstrates the calling conventions evolution can mutate:
  - Calling python_session for stateful Python (pandas, etc.) inside
    the Docker sandbox
  - Calling the Inspect-tracked LLM via get_model().generate()
  - Emitting {hypothesis, workflow} JSON for the scorer

Sample.files are auto-mounted into the sandbox by Inspect at evaluation
time, so the agent finds them at their declared paths under /workspace
without any explicit copy.

The seed is a one-shot baseline: discover the CSVs via python_session,
describe the first one, ask the LLM to draft a hypothesis. It does no
real analysis. Evolution is expected to introduce statistical tests,
multi-step reasoning, and score-type-appropriate strategies. See
background.md for the full surface.
"""

import json

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef


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

        py = _get_tool(state, "python_session")

        # --- 1. python_session: discover the mounted CSV(s) and describe.
        # Inspect auto-mounts Sample.files into the sandbox; we glob to
        # find them rather than hardcoding paths.
        described = await py(code=(
            "import glob, pandas as pd\n"
            "csvs = sorted(glob.glob('/workspace/**/*.csv', recursive=True))\n"
            "print('csvs:', csvs)\n"
            "df = pd.read_csv(csvs[0])\n"
            "print('shape:', df.shape)\n"
            "print(df.describe(include='all').to_string())\n"
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
        resp = await get_model().generate(prompt, config=GenerateConfig(temperature=1.0))
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
