"""Baseline PaperFindingBench solver.

Demonstrates the calling conventions evolution can mutate:
  - reading the query from state.metadata
  - finding a tool by name in state.tools and awaiting it
  - parsing the MCP tools' ContentText-JSON return shape
  - calling an LLM through a model_registry handle (Inspect-tracked)
  - writing the JSON output schema the scorer expects

The seed makes two LLM calls on the cheapest OpenAI handle — distill
the conversational query into a keyword phrase, then rerank the hits —
around a single search call. The distillation step is load-bearing:
`search_papers_by_relevance` is a literal keyword search and returns
ZERO hits for full natural-language questions. See CLAUDE.md (Domain
Background) for the full tool inventory, the nine-model menu, and the
per-score-type strategies evolution may want to introduce.
"""

import json

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

# LLM handles are imported from `model_registry`. Pick one per call,
# or mix across calls. See CLAUDE.md (Domain Background) for the
# full list of handles and their pricing.
from model_registry import GPT_5_4_MINI


def _find_tool(state: TaskState, *names: str):
    """Return (tool, name) for the first of `names` present in state.tools.

    The MCP kit (tool_source=mcp, the leaderboard tier) and the public-S2
    fallback kit (tool_source=search) expose different tool names, so the
    seed looks for both spellings.
    """
    by_name = {ToolDef(t).name: t for t in state.tools}
    for name in names:
        if name in by_name:
            return by_name[name], name
    raise RuntimeError(f"none of {names!r} in state.tools (have: {sorted(by_name)})")


def _parse_hits(raw) -> list[dict]:
    """Normalize a search tool's return into a list of paper dicts.

    The MCP tools return a list of ContentText objects whose `.text` is a
    JSON string per paper; the search-fallback tools return plain dicts.
    """
    hits = []
    for item in raw or []:
        if isinstance(item, dict):
            hits.append(item)
            continue
        text = getattr(item, "text", None) or (item if isinstance(item, str) else None)
        if text:
            try:
                hits.append(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                pass
    return hits


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata["raw_query"]
        score_type = state.metadata.get("score_type", "")
        print(f"[{state.sample_id}] score_type={score_type} query={query[:80]!r}")

        # --- LLM call 1: distill the query into search keywords ------------
        # The MCP keyword search is literal: the full conversational query
        # ("Could you suggest research that ...?") returns zero hits.
        distill = await GPT_5_4_MINI.generate(
            f"Extract a concise keyword search query (3-8 words, no "
            f"punctuation) for a scientific paper search engine from this "
            f"request. Reply with the keywords only.\n\nRequest: {query}"
        )
        keywords = distill.completion.strip().strip('"') or query
        print(f"  distilled keywords: {keywords!r}")

        # --- Tool call: retrieve candidates ---------------------------------
        # MCP name first (leaderboard tier), search-fallback name second.
        # Arg names differ per kit: keyword= (MCP) vs kquery= (fallback).
        search, tool_name = _find_tool(
            state, "search_papers_by_relevance", "paper_search"
        )
        if tool_name == "search_papers_by_relevance":
            raw = await search(
                keyword=keywords, fields="title,abstract,corpusId", limit=30
            )
        else:
            raw = await search(kquery=keywords, limit=30)
        hits = _parse_hits(raw)
        print(f"  {tool_name} returned {len(hits)} hits")

        # --- LLM call: rerank via a registry handle -------------------------
        if hits:
            candidate_lines = "\n".join(
                f"{i}: {(h.get('title') or '')[:120]}" for i, h in enumerate(hits)
            )
            prompt = (
                f"Query: {query}\n\nCandidates:\n{candidate_lines}\n\n"
                f"Return the indices of the most relevant candidates, "
                f"comma-separated. Up to 10 indices."
            )
            response = await GPT_5_4_MINI.generate(prompt)
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
                "paper_id": str(h.get("corpusId") or h.get("corpus_id") or ""),
                "markdown_evidence": (h.get("title") or "") + " — " + (h.get("abstract") or "")[:400],
            }
            for h in kept
        ]
        state.output.completion = json.dumps({
            "output": {"query_id": state.sample_id, "results": results}
        })
        print(f"  submitted {len(results)} papers")
        return state

    return solve
