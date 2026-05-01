"""Baseline PaperFindingBench solver.

Demonstrates the calling conventions evolution can mutate:
  - reading the query from state.metadata
  - finding a tool by name in state.tools and awaiting it
  - calling the Inspect-tracked LLM via get_model().generate()
  - writing the JSON output schema the scorer expects

The seed uses a single tool (paper_search) and a single rerank LLM call.
See background.md for the full tool inventory and the per-score-type
strategies evolution may want to introduce.
"""

import json

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef


def _get_tool(state: TaskState, name: str):
    """Find a tool in state.tools by its registered name."""
    for t in state.tools:
        if ToolDef(t).name == name:
            return t
    raise RuntimeError(f"tool {name!r} not in state.tools (have: "
                       f"{[ToolDef(t).name for t in state.tools]})")


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata["raw_query"]
        score_type = state.metadata.get("score_type", "")
        print(f"[{state.sample_id}] score_type={score_type} query={query[:80]!r}")

        # --- Tool call: retrieve candidates ---------------------------------
        # Note arg name: paper_search uses 'kquery' (snippet_search uses 'query').
        search = _get_tool(state, "paper_search")
        hits = await search(kquery=query, limit=30)
        print(f"  paper_search returned {len(hits)} hits")

        # --- LLM call: rerank via the Inspect-tracked model ----------------
        # Using get_model() ensures token usage is recorded into the .eval log,
        # which is what the leaderboard reads for cost. Do NOT bypass with a
        # direct openai/anthropic client unless you wrap the call with
        # record_model_usage_with_inspect() afterwards.
        if hits:
            candidate_lines = "\n".join(
                f"{i}: {h.get('title', '')[:120]}" for i, h in enumerate(hits)
            )
            prompt = (
                f"Query: {query}\n\nCandidates:\n{candidate_lines}\n\n"
                f"Return the indices of the most relevant candidates, "
                f"comma-separated. Up to 10 indices."
            )
            response = await get_model().generate(prompt)
            keep_idx = []
            for tok in response.completion.replace("\n", ",").split(","):
                tok = tok.strip()
                if tok.isdigit() and 0 <= int(tok) < len(hits):
                    keep_idx.append(int(tok))
            kept = [hits[i] for i in keep_idx] or hits[:10]
        else:
            kept = []

        # --- Output: write JSON in the schema the scorer expects -----------
        results = [
            {
                "paper_id": str(h.get("corpus_id") or h.get("corpusId") or ""),
                "markdown_evidence": (h.get("title", "") or "") + " — " + (h.get("abstract", "") or "")[:400],
            }
            for h in kept
        ]
        state.output.completion = json.dumps({
            "output": {"query_id": state.sample_id, "results": results}
        })
        print(f"  submitted {len(results)} papers")
        return state

    return solve
