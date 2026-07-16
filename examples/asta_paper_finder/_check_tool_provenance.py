#!/usr/bin/env python3
"""Live provenance-capture gate for the Asta MCP tools.

Grounding silently depends on `_grounding._walk` capturing every quotable text
field the tools actually return. Two capture gaps shipped despite green unit
tests because those tests use SYNTHETIC payloads: the ContentText-list shape,
and nested `tldr.text` under a hex `paperId` (which registered a phantom cid).

This gate calls each tool once against the LIVE corpus and asserts that the
fields an agent would quote — title / abstract / tldr / snippet — are recorded
in provenance. It uses an independent oracle (`_quotable`, which does NOT go
through `_walk`) to pull those fields straight from the raw payload, then checks
each grounds. A capture gap fails loudly here instead of in a production run.

Requires ASTA_TOOL_KEY (and the usual provider keys the evaluator preflights).
Run: python _check_tool_provenance.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _grounding as g  # noqa: E402

# A known, stable paper to anchor id-based probes: ConSERT (corpusId 235187266).
CONSERT = "235187266"
FIELDS = "title,abstract,corpusId,year,venue,authors,tldr"
# get_citations rejects `tldr` ("Unrecognized or unsupported fields") — it
# returns title/abstract on the wrapped citingPaper, not a tldr.
CITATION_FIELDS = "title,abstract,corpusId,year,venue,authors"


def _cidof(d: dict) -> str | None:
    for k in ("corpusId", "corpus_id", "CorpusId"):
        if k in d:
            m = re.search(r"\d+", str(d[k]))
            if m:
                return m.group(0)
    return None


def _quotable(payload) -> dict[str, set[str]]:
    """{cid: {quotable strings}} pulled straight from the raw payload, NOT via
    _walk — this is the oracle we hold _walk to. Handles ContentText unwrapping,
    the snippet_search `{paper, snippet}` shape, and the citations `citingPaper`
    wrapper, collecting title / abstract / tldr.text / snippet.text per paper."""
    out: dict[str, set[str]] = {}

    def add(cid, s):
        if cid and isinstance(s, str) and s.strip():
            out.setdefault(cid, set()).add(s.strip())

    def fields_of(paper: dict, cid: str):
        add(cid, paper.get("title"))
        add(cid, paper.get("abstract"))
        tl = paper.get("tldr")
        add(cid, tl.get("text") if isinstance(tl, dict) else tl)

    def walk(node):
        if isinstance(node, str):
            try:
                walk(json.loads(node))
            except (ValueError, TypeError):
                pass
            return
        text = getattr(node, "text", None)  # ContentText
        if isinstance(text, str):
            walk(text)
            return
        if isinstance(node, dict):
            # snippet_search / citations wrappers: the paper is nested, and the
            # snippet text is a SIBLING of the paper block.
            paper = node.get("paper") or node.get("citingPaper")
            if isinstance(paper, dict) and _cidof(paper):
                pcid = _cidof(paper)
                fields_of(paper, pcid)
                snip = node.get("snippet")
                if isinstance(snip, dict):
                    add(pcid, snip.get("text"))
            cid = _cidof(node)
            if cid:
                fields_of(node, cid)
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(payload)
    return out


def _check(name: str, payload) -> tuple[int, int, list[str]]:
    """Record `payload`, then assert every oracle-extracted field grounds.
    Returns (n_checked, n_failed, sample_failures)."""
    g.reset()
    g.record_tool_result(payload)
    oracle = _quotable(payload)
    checked = failed = 0
    misses: list[str] = []
    for cid, texts in oracle.items():
        for t in texts:
            checked += 1
            if not g.check_evidence(cid, t)[0]:
                failed += 1
                if len(misses) < 3:
                    misses.append(f"{cid}: {t[:80]!r}")
    return checked, failed, misses


async def _run(tools: dict):
    """Yield (tool_name, coroutine) probes. Author tools feed metadata queries
    (no judged evidence) but are covered for completeness."""
    probes = []

    probes.append(("search_papers_by_relevance", tools["search_papers_by_relevance"](
        keyword="contrastive self-supervised sentence representation", fields=FIELDS, limit=5)))
    probes.append(("search_paper_by_title", tools["search_paper_by_title"](
        title="ConSERT: A Contrastive Framework for Self-Supervised Sentence "
              "Representation Transfer", fields=FIELDS)))
    probes.append(("get_paper", tools["get_paper"](
        paper_id=f"CorpusId:{CONSERT}", fields=FIELDS)))
    probes.append(("get_paper_batch", tools["get_paper_batch"](
        ids=[f"CorpusId:{CONSERT}", "CorpusId:267199943"], fields=FIELDS)))
    probes.append(("get_citations", tools["get_citations"](
        paper_id=f"CorpusId:{CONSERT}", fields=CITATION_FIELDS, limit=5)))
    probes.append(("snippet_search", tools["snippet_search"](
        query="contrastive sentence representation learning", limit=5)))

    # Author tools: chain to get a real author_id. Request authorId+papers so
    # the payload carries a chainable id and quotable paper titles.
    authors_raw = None
    try:
        authors_raw = await tools["search_authors_by_name"](
            name="Danqi Chen", fields="authorId,name,papers", limit=3)
    except Exception as e:
        print(f"  search_authors_by_name call failed (skipped): {type(e).__name__}: {e}")
    if authors_raw is not None:
        probes.append(("search_authors_by_name", authors_raw))
        aid = _first_author_id(authors_raw)
        if aid:
            try:
                probes.append(("get_author_papers", await tools["get_author_papers"](
                    author_id=str(aid), paper_fields=CITATION_FIELDS, limit=10)))
            except Exception as e:
                print(f"  get_author_papers call failed (skipped): {type(e).__name__}: {e}")

    return probes


def _first_author_id(payload):
    """Best-effort authorId extraction from a search_authors_by_name payload."""
    def walk(node):
        if isinstance(node, str):
            try:
                return walk(json.loads(node))
            except (ValueError, TypeError):
                return None
        text = getattr(node, "text", None)
        if isinstance(text, str):
            return walk(text)
        if isinstance(node, dict):
            if node.get("authorId"):
                return node["authorId"]
            for v in node.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(node, (list, tuple)):
            for v in node:
                r = walk(v)
                if r:
                    return r
        return None
    return walk(payload)


def main() -> int:
    if not os.environ.get("ASTA_TOOL_KEY"):
        print("SKIP: ASTA_TOOL_KEY not set (this is a live gate).")
        return 0
    from evaluator import _build_tools
    from inspect_ai.tool import ToolDef

    tools = {ToolDef(t).name: t for t in _build_tools("semantic_1")}

    async def _resolve():
        # Each probe is (name, coroutine); await sequentially so one tool's
        # rate-limit doesn't cascade.
        out = []
        for name, coro in await _run(tools):
            try:
                out.append((name, await coro if asyncio.iscoroutine(coro) else coro))
            except Exception as e:
                out.append((name, e))
        return out

    resolved = asyncio.run(_resolve())

    total_fail = 0
    covered = set()
    for name, payload in resolved:
        if isinstance(payload, Exception):
            print(f"  {name:28} CALL FAILED (skipped): {type(payload).__name__}: {payload}")
            continue
        checked, failed, misses = _check(name, payload)
        covered.add(name)
        if checked == 0:
            print(f"  {name:28} no quotable fields returned (skipped)")
            continue
        status = "OK" if failed == 0 else "CAPTURE GAP"
        print(f"  {name:28} {checked - failed}/{checked} fields grounded  [{status}]")
        for m in misses:
            print(f"      MISS {m}")
        total_fail += failed

    expected = {"search_papers_by_relevance", "search_paper_by_title", "get_paper",
                "get_paper_batch", "get_citations", "snippet_search"}
    missing = expected - covered
    if missing:
        print(f"\nWARNING: core paper-text tools not exercised: {sorted(missing)}")

    print("\nOK" if total_fail == 0 else f"\nFAILED: {total_fail} field(s) not captured")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
