"""Baseline PaperFindingBench solver.

Demonstrates the calling conventions evolution can mutate:
  - reading the query from state.metadata
  - finding a tool by name in state.tools and awaiting it
  - parsing the MCP tools' ContentText-JSON return shape
  - calling an LLM through a model_registry handle, with GenerateConfig
    and an empty-completion guard
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

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

# LLM handles are imported from `model_registry`. Pick one per call,
# or mix across calls. See CLAUDE.md (Domain Background) for the
# full list of handles and their pricing.
from model_registry import GPT_5_4_MINI

# Cap visible output. reasoning_effort is a legitimate accuracy lever,
# but NOTE the OpenAI trap: on OpenAI handles max_tokens is SHARED
# between reasoning and visible tokens, so opting into reasoning_effort
# with a tight cap yields an EMPTY completion with no error — if you
# set it, raise max_tokens several-fold to leave room for the visible
# answer. This seed omits it purely as a cost choice for two trivial
# calls. See CLAUDE.md (Domain Background) for the full warning.
_LLM_CONFIG = GenerateConfig(max_tokens=1000)


def _get_tool(state: TaskState, name: str):
    """Find a tool in state.tools by its registered name."""
    by_name = {ToolDef(t).name: t for t in state.tools}
    if name not in by_name:
        raise RuntimeError(f"{name!r} not in state.tools (have: {sorted(by_name)})")
    return by_name[name]


def _parse_hits(raw) -> list[dict]:
    """Normalize a search tool's return (list of ContentText objects
    whose `.text` is a JSON string per paper) into a list of dicts."""
    hits = []
    for item in raw or []:
        text = getattr(item, "text", None)
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
            f"request. Reply with the keywords only.\n\nRequest: {query}",
            config=_LLM_CONFIG,
        )
        # Guard every metered call against an empty completion — a silent
        # failure mode worth surfacing in stdout rather than absorbing.
        distilled = (distill.completion or "").strip().strip('"')
        keywords = distilled or query
        print(f"  distilled keywords: {keywords!r} (completion len={len(distilled)})")

        # --- Tool call: retrieve candidates ---------------------------------
        search = _get_tool(state, "search_papers_by_relevance")
        raw = await search(keyword=keywords, fields="title,abstract,corpusId", limit=30)
        hits = _parse_hits(raw)
        print(f"  search returned {len(hits)} hits")

        # --- LLM call 2: rerank via a registry handle ------------------------
        if hits:
            candidate_lines = "\n".join(
                f"{i}: {(h.get('title') or '')[:120]}" for i, h in enumerate(hits)
            )
            prompt = (
                f"Query: {query}\n\nCandidates:\n{candidate_lines}\n\n"
                f"Return the indices of the most relevant candidates, "
                f"comma-separated. Up to 10 indices."
            )
            response = await GPT_5_4_MINI.generate(prompt, config=_LLM_CONFIG)
            rerank_text = (response.completion or "").strip()
            print(f"  rerank completion len={len(rerank_text)}")
            keep_idx = []
            for tok in rerank_text.replace("\n", ",").split(","):
                tok = tok.strip()
                if tok.isdigit() and 0 <= int(tok) < len(hits):
                    keep_idx.append(int(tok))
            kept = [hits[i] for i in keep_idx] or hits[:10]
            # Never submit a near-empty semantic list: the rank term of the
            # semantic scorer degenerates on single-paper/uniform lists (see
            # CLAUDE.md), so pad best-first from the remaining hits.
            if len(kept) < 8:
                kept += [h for h in hits if h not in kept][: 8 - len(kept)]
        else:
            kept = []

        # --- Output: write JSON in the schema the scorer expects -----------
        results = [
            {
                "paper_id": str(h.get("corpusId") or ""),
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
