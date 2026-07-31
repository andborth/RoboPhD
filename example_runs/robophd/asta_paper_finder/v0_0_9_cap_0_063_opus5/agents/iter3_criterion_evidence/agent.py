"""PaperFindingBench solver: score-type triage + criterion-complete evidence.

Iteration 2 established that the semantic path's `rank` term is already 0.60-0.95
while `recall` sits at 0.00-0.41, so `harmonic(rank, recall)` is entirely
recall-bound. The judge verdicts say why: on semantic_91, 70 of 100 judged papers
came back "highly relevant" (grade 2) and 0 came back "not relevant". Grade 2
earns *zero* recall — grade 3 needs `weighted > 0.99`, i.e. every weighted
criterion judged Perfectly Relevant, and the judge reads `markdown_evidence`
alone. So those 70 papers were retrieved correctly, ranked into the scored depth,
and then scored nothing because their evidence supported two criteria out of
three.

This agent therefore treats evidence as a first-class retrieval target:

  1. the query is decomposed into criteria in the same shape the gold uses, each
     carrying a short `probe` phrase;
  2. `snippet_search` is run *per criterion* over the candidate head, so returned
     passages are tagged with the criterion that found them;
  3. evidence is assembled to maximize criterion coverage under the 2500-char cap
     rather than by truncating a title/tldr/abstract concatenation;
  4. the final ordering blends the LLM grade with the measured criterion coverage
     of the assembled evidence, because K (median ~56) is smaller than the
     250-slot submission, so ordering decides which papers are judged at all.

The two exact-match paths score `2h/(n+G)` with G typically 1-3, so their fix is
arithmetic: never pad. Iteration 2 scored 0.05 on a metadata query it had *full
recall* on, because it attached 114 wrong ids to 3 right ones.
"""

import asyncio
import json
import re
import time

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

N_VARIANTS = 6             # keyword query variants for the semantic path
SEARCH_LIMIT = 100         # max allowed by search_papers_by_relevance
POOL_FOR_GRADING = 220     # candidates handed to the cheap grader
GRADE_BATCH = 45
HEAD_RERANK_N = 36         # head re-graded on the stronger model
# Measured: a 25-paper scope with limit=100 returns passages for only 13 of the
# 25 (95 s). A 20-paper scope raises per-call coverage, and each paper gets one
# shot per criterion, so a paper appears in at least one harvest ~90% of the time.
CRIT_SNIPPET_HEAD = 80     # papers given criterion-targeted snippet_search
CRIT_SNIPPET_BATCH = 20    # paper_ids per scoped snippet_search call
MAX_SUBMIT_SEMANTIC = 250  # the scorer's own cap; tail slots are free upside
MAX_SUBMIT_SPECIFIC = 3    # precision is half the score on the exact-match path
MAX_SUBMIT_METADATA = 60   # F1 = 2h/(n+G); padding is what killed metadata_4
EVIDENCE_CHARS = 2400      # under the scorer's 2500-char truncation point
MAX_PASSAGES = 8           # the scorer keeps at most 8 passages per paper
SNIPPET_CHARS = 620        # per-snippet trim; a raw snippet is ~3000 chars

PAPER_FIELDS = "title,abstract,corpusId,tldr,year,venue,authors,citationCount"
SNIPPET_TIMEOUT = 240.0    # under the 300 s per-call transport ceiling
ENRICH_DEADLINE = 1150.0   # stop optional enrichment; hard timeout is 29 min


# ---------------------------------------------------------------------------
# Tool plumbing
# ---------------------------------------------------------------------------

def _get_tool(state: TaskState, name: str):
    """Look a tool up by registered name, tolerating any single tool whose
    ToolDef cannot be built (one bad entry must not take the query down)."""
    for t in state.tools or []:
        try:
            tname = ToolDef(t).name
        except Exception:  # noqa: BLE001
            tname = getattr(t, "__name__", "")
        if tname == name:
            return t
    print(f"  [no-tool] {name} not in state.tools")
    return None


def _parse_items(raw) -> list[dict]:
    """Normalize any MCP corpus tool's return into a flat list of dicts.

    Every tool returns a list of ContentText whose `.text` is JSON; some wrap
    their payload as {"data": [...]}.
    """
    out: list[dict] = []
    for item in raw or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(doc, dict) and "data" in doc:
            out.extend(d for d in (doc.get("data") or []) if isinstance(d, dict))
        elif isinstance(doc, dict):
            out.append(doc)
        elif isinstance(doc, list):
            out.extend(d for d in doc if isinstance(d, dict))
    return out


async def _safe(coro, label: str, timeout: float | None = None):
    """Await a tool call, converting any failure into None. Breadth is built
    from many independent calls; one timing out must not sink the query."""
    try:
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro
    except Exception as exc:  # noqa: BLE001 - tool errors are expected & varied
        print(f"  [tool-fail] {label}: {type(exc).__name__}: {str(exc)[:160]}")
        return None


async def _llm(handle, prompt: str, label: str, config: GenerateConfig | None = None) -> str:
    try:
        resp = await (handle.generate(prompt, config=config) if config
                      else handle.generate(prompt))
        text = (resp.completion or "").strip()
        if not text:
            print(f"  [llm-empty] {label}")
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"  [llm-fail] {label}: {type(exc).__name__}: {str(exc)[:160]}")
        return ""


def _json_from(text: str):
    """Pull the first JSON object/array out of an LLM completion."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Paper record helpers
# ---------------------------------------------------------------------------

def _cid(paper: dict) -> str:
    """corpusId is an int on the search tools and a str on snippet_search."""
    val = paper.get("corpusId")
    return "" if val in (None, "") else str(val).strip()


def _tldr_text(paper: dict) -> str:
    tldr = paper.get("tldr")
    if isinstance(tldr, dict):
        return (tldr.get("text") or "").strip()
    return tldr.strip() if isinstance(tldr, str) else ""


def _author_names(paper: dict) -> list[str]:
    out = []
    for a in paper.get("authors") or []:
        name = a.get("name") if isinstance(a, dict) else a
        if name:
            out.append(str(name))
    return out


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _year_of(paper: dict) -> int | None:
    year = paper.get("year")
    if isinstance(year, int):
        return year
    m = re.match(r"(\d{4})", str(year or paper.get("publicationDate") or ""))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Criterion coverage (deterministic, free)
# ---------------------------------------------------------------------------

_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "must", "paper", "papers",
    "which", "such", "their", "have", "has", "are", "was", "were", "its", "into",
    "using", "used", "use", "uses", "based", "study", "studies", "work", "works",
    "approach", "method", "methods", "propose", "proposed", "discuss", "discusses",
    "about", "these", "those", "than", "then", "also", "other", "some", "more",
    "explicitly", "specifically", "focus", "focuses", "focused", "address",
    "addresses", "involve", "involves", "research", "literature", "how", "what",
    "including", "include", "includes", "well", "both", "each", "over", "when",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    return {_stem(w) for w in words if len(w) >= 4 and w not in _STOP}


def _covers(passage_terms: set[str], crit_terms: set[str]) -> bool:
    """Does a passage lexically demonstrate a criterion?

    Criterion probes are short (3-6 content words). Requiring most of them keeps
    the signal honest without demanding an exact phrase match.
    """
    if not crit_terms:
        return False
    need = max(1, int(round(0.6 * len(crit_terms))))
    return len(crit_terms & passage_terms) >= need


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _trim_snippet(snip: str, crit_terms: set[str]) -> str:
    """Cut a ~3000-char snippet down to the contiguous sentence window that best
    demonstrates the criterion. Contiguity keeps it a verbatim substring."""
    snip = re.sub(r"\s+", " ", (snip or "")).strip()
    if len(snip) <= SNIPPET_CHARS:
        return snip
    sents = _split_sentences(snip)
    if not sents:
        return snip[:SNIPPET_CHARS]
    scores = [len(_terms(s) & crit_terms) for s in sents]
    best = max(range(len(sents)), key=lambda i: (scores[i], -i))
    lo = hi = best
    out = sents[best]
    while len(out) < SNIPPET_CHARS:
        grew = False
        if hi + 1 < len(sents) and len(out) + 1 + len(sents[hi + 1]) <= SNIPPET_CHARS:
            hi += 1
            out = out + " " + sents[hi]
            grew = True
        if lo - 1 >= 0 and len(out) + 1 + len(sents[lo - 1]) <= SNIPPET_CHARS:
            lo -= 1
            out = sents[lo] + " " + out
            grew = True
        if not grew:
            break
    return out[:SNIPPET_CHARS]


def _build_evidence(paper: dict, snips: list[tuple[int, str]],
                    crit_terms: list[set[str]]) -> tuple[str, float]:
    """Assemble <=8 verbatim passages under the char cap, spending the budget on
    criteria the abstract does not already demonstrate.

    Returns (evidence, fraction of criteria lexically covered). Every emitted
    passage is a contiguous substring of text the tools returned for THIS paper,
    so it survives the scorer's grounding check.
    """
    n_crit = len(crit_terms)
    covered = [False] * n_crit
    passages: list[str] = []
    used = 0

    def mark(text: str) -> None:
        pt = _terms(text)
        for i, ct in enumerate(crit_terms):
            if not covered[i] and _covers(pt, ct):
                covered[i] = True

    def add(text: str) -> bool:
        nonlocal used
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text or len(passages) >= MAX_PASSAGES:
            return False
        cost = len(text) + (5 if passages else 0)
        if used + cost > EVIDENCE_CHARS:
            return False
        passages.append(text)
        used += cost
        mark(text)
        return True

    add((paper.get("title") or "")[:300])
    add(_tldr_text(paper)[:450])

    abstract = re.sub(r"\s+", " ", (paper.get("abstract") or "")).strip()
    # Reserve room for criterion-targeted body text when we actually have some.
    reserve = min(1100, SNIPPET_CHARS * min(2, len(snips))) if snips else 0
    if abstract:
        room = EVIDENCE_CHARS - used - reserve - 5
        if len(abstract) <= room:
            add(abstract)
        else:
            # Sentence-select: prefer sentences demonstrating criteria that are
            # still uncovered, then emit the picks as contiguous runs so each
            # stays verbatim.
            sents = _split_sentences(abstract)
            want = [i for i, c in enumerate(covered) if not c]
            scored = []
            for j, s in enumerate(sents):
                st = _terms(s)
                gain = sum(1 for i in want if _covers(st, crit_terms[i]))
                scored.append((-gain, j))
            scored.sort()
            picked, budget = set(), max(room, 400)
            spend = 0
            for _, j in scored:
                if spend + len(sents[j]) + 1 > budget:
                    continue
                picked.add(j)
                spend += len(sents[j]) + 1
            if not picked:
                picked = {0}
            runs, cur = [], []
            for j in sorted(picked):
                if cur and j == cur[-1] + 1:
                    cur.append(j)
                else:
                    if cur:
                        runs.append(cur)
                    cur = [j]
            if cur:
                runs.append(cur)
            for run in runs:
                add(" ".join(sents[j] for j in run))

    # Criterion-targeted body passages, uncovered criteria first.
    remaining = [s for s in snips]
    for i in range(n_crit):
        if covered[i]:
            continue
        for k, (ci, text) in enumerate(remaining):
            if ci != i:
                continue
            trimmed = _trim_snippet(text, crit_terms[i])
            if _covers(_terms(trimmed), crit_terms[i]) and add(trimmed):
                remaining.pop(k)
                break

    # Spend anything left over on the best remaining passages.
    for ci, text in remaining:
        if len(passages) >= MAX_PASSAGES or used >= EVIDENCE_CHARS - 200:
            break
        ct = crit_terms[ci] if 0 <= ci < n_crit else set()
        add(_trim_snippet(text, ct))

    frac = (sum(covered) / n_crit) if n_crit else 1.0
    return " ... ".join(passages), frac


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _emit(state: TaskState, results: list[dict]) -> None:
    """Write the scorer's JSON schema. `markdown_evidence` is required on every
    result even where its content is ignored (the exact-match paths)."""
    state.output.completion = json.dumps(
        {"output": {"query_id": state.sample_id, "results": results}}
    )
    lens = [len(r["markdown_evidence"]) for r in results]
    avg = sum(lens) // max(1, len(lens))
    print(f"  SUBMITTED {len(results)} papers (evidence chars avg={avg})")


def _emit_ids(state: TaskState, papers: list[dict]) -> None:
    results, seen = [], set()
    for paper in papers:
        cid = _cid(paper)
        if cid and cid not in seen:
            seen.add(cid)
            results.append({"paper_id": cid, "markdown_evidence": ""})
    _emit(state, results)


# ---------------------------------------------------------------------------
# specific_f1 — resolve a named paper to its corpus record(s)
# ---------------------------------------------------------------------------

async def solve_specific(state: TaskState, query: str) -> None:
    title_search = _get_tool(state, "search_paper_by_title")
    kw_search = _get_tool(state, "search_papers_by_relevance")

    ask = (
        "A user is looking for one specific, known scientific paper. Using your "
        "knowledge of the literature, give the paper's exact published title.\n\n"
        f'User request: "{query}"\n\n'
        "Reply with JSON only:\n"
        '{"titles": ["most likely exact title", "second guess", "third guess"]}\n'
        "Order most-confident first. Give 1-3 titles. Use the real published "
        "title, not the nickname (e.g. 'the GPT-2 paper' -> 'Language Models are "
        "Unsupervised Multitask Learners'). A citation key like 'fabri2019multinews' "
        "encodes first author, year and a title word - use it."
    )
    plan = _json_from(await _llm(GPT_5_4, ask, "specific/titles")) or {}
    titles = [t for t in (plan.get("titles") or []) if isinstance(t, str) and t.strip()][:3]
    print(f"  candidate titles: {titles}")

    calls = []
    if title_search:
        calls += [_safe(title_search(title=t, fields=PAPER_FIELDS), f"title:{t[:40]}")
                  for t in titles]
    if titles and kw_search:
        calls.append(_safe(kw_search(keyword=titles[0], fields=PAPER_FIELDS, limit=15),
                           "kw:title0"))
    if kw_search:
        # The nickname itself catches cases where the recalled title is wrong but
        # the corpus index still knows the paper.
        bare = re.sub(r"^(the|find|get)\s+", "", query.strip(), flags=re.I)
        bare = re.sub(r"\bpaper\b", "", bare, flags=re.I).strip() or query
        calls.append(_safe(kw_search(keyword=bare, fields=PAPER_FIELDS, limit=15), "kw:bare"))

    raws = await asyncio.gather(*calls) if calls else []

    candidates: list[dict] = []
    seen: set[str] = set()
    for raw in raws:
        for paper in _parse_items(raw):
            cid = _cid(paper)
            if cid and cid not in seen:
                seen.add(cid)
                candidates.append(paper)
    if not candidates:
        _emit(state, [])
        return

    lines = []
    for i, paper in enumerate(candidates[:30]):
        authors = ", ".join(_author_names(paper)[:3])
        lines.append(
            f"{i}: {(paper.get('title') or '')[:150]} | {_year_of(paper) or '?'} "
            f"| {authors[:80]} | {(paper.get('venue') or '')[:40]}"
        )
    verify = (
        f'A user asked for a specific paper: "{query}"\n\n'
        "Candidate corpus records:\n" + "\n".join(lines) + "\n\n"
        "Which single record IS that paper? Reply with the best index first. "
        "Only add a second index if it is a duplicate/alternate record of the very "
        "same paper (preprint and published version). Merely related papers must "
        "NOT be included - each wrong id directly reduces the score.\n"
        'Reply with JSON only: {"indices": [i, ...]}'
    )
    verdict = _json_from(await _llm(GPT_5_4, verify, "specific/verify")) or {}
    picked_idx = []
    for idx in verdict.get("indices") or []:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in picked_idx:
            picked_idx.append(idx)

    best = candidates[picked_idx[0]] if picked_idx else candidates[0]
    # Duplicate corpus records of ONE paper are all in gold (specific_39's gold
    # holds 5 ids for a single paper), so identical normalized titles are free
    # recall. Anything else is pure precision loss: iteration 2 scored 0.667 on
    # specific_11 for one padded id.
    target = _norm_title(best.get("title", ""))
    picks = [best]
    for paper in candidates:
        if paper is not best and _norm_title(paper.get("title", "")) == target:
            picks.append(paper)
    if len(picks) == 1 and len(picked_idx) > 1:
        picks.append(candidates[picked_idx[1]])

    print(f"  picks: {[(_cid(p), (p.get('title') or '')[:60]) for p in picks[:MAX_SUBMIT_SPECIFIC]]}")
    _emit_ids(state, picks[:MAX_SUBMIT_SPECIFIC])


# ---------------------------------------------------------------------------
# metadata_f1 — parse into a filter plan, execute against the graph
# ---------------------------------------------------------------------------

async def _verify_venues(pool_venues: list[str], wanted: list[str], query: str) -> set[str]:
    """Which of the venue strings actually present in the pool satisfy the
    request? Substring matching cannot know that Nature Communications is in the
    Nature portfolio and Communications of the ACM is not."""
    if not pool_venues or not wanted:
        return set()
    listing = "\n".join(f"{i}: {v[:80]}" for i, v in enumerate(pool_venues[:80]))
    ask = (
        f'Literature search request: "{query}"\n'
        f"It restricts results to venues described as: {', '.join(wanted)}\n\n"
        "Here are the venue strings that appear in the retrieved candidates:\n"
        f"{listing}\n\n"
        "Which indices are venues that satisfy the restriction? Be inclusive "
        "about the same publisher family (e.g. a 'Nature portfolio' restriction "
        "covers Nature and every Nature-branded journal) but exclude unrelated "
        "venues.\n"
        'Reply with JSON only: {"indices": [i, ...]}'
    )
    got = _json_from(await _llm(GPT_5_4_MINI, ask, "metadata/venues")) or {}
    out = set()
    for idx in got.get("indices") or []:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(pool_venues[:80]):
            out.add(pool_venues[idx])
    print(f"  venue verification kept {len(out)}/{len(pool_venues[:80])}: {sorted(out)[:8]}")
    return out


async def solve_metadata(state: TaskState, query: str) -> None:
    title_search = _get_tool(state, "search_paper_by_title")
    kw_search = _get_tool(state, "search_papers_by_relevance")
    citations = _get_tool(state, "get_citations")
    author_search = _get_tool(state, "search_authors_by_name")
    author_papers = _get_tool(state, "get_author_papers")

    ask = (
        "Parse this literature-search request into a structured filter plan.\n\n"
        f'Request: "{query}"\n\n'
        "Reply with JSON only, omitting or nulling anything not requested:\n"
        "{\n"
        '  "cites_titles": ["exact published title of each paper results must CITE"],\n'
        '  "authors": ["full author names results must be by"],\n'
        '  "venues": ["venue / journal / conference names or families"],\n'
        '  "year_min": null, "year_max": null,\n'
        '  "min_citations": null, "min_authors": null,\n'
        '  "topic_keywords": "short keyword phrase for the topic, or empty"\n'
        "}\n"
        "For cites_titles use the real published title (e.g. 'the T5 paper' -> "
        "'Exploring the Limits of Transfer Learning with a Unified Text-to-Text "
        "Transformer'). 'after 2022' means year_min 2023. '2019 and beyond' means "
        "year_min 2019."
    )
    plan = _json_from(await _llm(GPT_5_4, ask, "metadata/plan")) or {}
    cites = [t for t in (plan.get("cites_titles") or []) if isinstance(t, str) and t.strip()][:3]
    authors = [a for a in (plan.get("authors") or []) if isinstance(a, str) and a.strip()][:3]
    venues = [v for v in (plan.get("venues") or []) if isinstance(v, str) and v.strip()][:5]
    topic = (plan.get("topic_keywords") or "").strip()

    def _int(key):
        try:
            return int(plan.get(key))
        except (TypeError, ValueError):
            return None

    year_min, year_max = _int("year_min"), _int("year_max")
    min_cites, min_authors = _int("min_citations"), _int("min_authors")
    print(f"  plan: cites={cites} authors={authors} venues={venues} "
          f"years=[{year_min},{year_max}] min_cites={min_cites} "
          f"min_authors={min_authors} topic={topic!r}")

    pool: dict[str, dict] = {}
    intersect_sets: list[set[str]] = []
    structural = False

    if cites and title_search and citations:
        resolved = await asyncio.gather(*[
            _safe(title_search(title=t, fields="title,corpusId,year"), f"resolve:{t[:40]}")
            for t in cites
        ])
        seed_ids = []
        for raw in resolved:
            for paper in _parse_items(raw):
                if paper.get("paperId") and _cid(paper):
                    seed_ids.append(_cid(paper))
                    print(f"    cited paper -> {_cid(paper)} {(paper.get('title') or '')[:70]}")
                    break
        citing = await asyncio.gather(*[
            _safe(citations(paper_id=f"CorpusId:{sid}",
                            fields="title,abstract,corpusId,year,venue,authors,citationCount",
                            limit=1000), f"citations:{sid}")
            for sid in seed_ids
        ])
        for raw in citing:
            group: set[str] = set()
            for item in _parse_items(raw):
                paper = item.get("citingPaper") if "citingPaper" in item else item
                if isinstance(paper, dict) and _cid(paper):
                    group.add(_cid(paper))
                    pool.setdefault(_cid(paper), paper)
            if group:
                intersect_sets.append(group)
                structural = True
            print(f"    citation set size {len(group)}")

    if authors and author_search and author_papers:
        for name in authors:
            found = _parse_items(await _safe(
                author_search(name=name, fields="name,paperCount", limit=10), f"author:{name}"))
            # Same person often has several fragmentary ids; the richest is real.
            found.sort(key=lambda a: a.get("paperCount") or 0, reverse=True)
            group: set[str] = set()
            for rec in found[:2]:
                aid = rec.get("authorId")
                if not aid:
                    continue
                print(f"    author {rec.get('name')} id={aid} papers={rec.get('paperCount')}")
                raw = await _safe(author_papers(
                    author_id=str(aid),
                    paper_fields="title,abstract,corpusId,year,venue,authors,citationCount",
                    limit=500), f"author_papers:{aid}")
                for paper in _parse_items(raw):
                    if _cid(paper):
                        group.add(_cid(paper))
                        pool.setdefault(_cid(paper), paper)
            if group:
                intersect_sets.append(group)
                structural = True

    if not pool and kw_search:
        # No graph anchor: fall back to keyword search, using the venue filter
        # server-side when one was named (it genuinely filters).
        venue_arg = ",".join(venues) if venues else None
        for kw in [k for k in (topic, query) if k][:2]:
            kwargs = {"keyword": kw, "fields": PAPER_FIELDS, "limit": SEARCH_LIMIT}
            if venue_arg:
                kwargs["venues"] = venue_arg
            for paper in _parse_items(await _safe(kw_search(**kwargs), f"kw:{kw[:40]}")):
                if _cid(paper):
                    pool.setdefault(_cid(paper), paper)
            if pool:
                break
        if not pool and venue_arg:  # venue filter may have been too strict
            for paper in _parse_items(await _safe(
                    kw_search(keyword=topic or query, fields=PAPER_FIELDS,
                              limit=SEARCH_LIMIT), "kw:novenue")):
                if _cid(paper):
                    pool.setdefault(_cid(paper), paper)

    ids = set(pool)
    for group in intersect_sets:
        ids &= group
    print(f"  pool={len(pool)} after intersection={len(ids)}")

    # Venue verification: an LLM read of the venue strings actually present.
    venue_ok: set[str] | None = None
    if venues and ids:
        distinct = sorted({(pool[c].get("venue") or "").strip()
                           for c in ids if (pool[c].get("venue") or "").strip()})
        if distinct:
            venue_ok = await _verify_venues(distinct, venues, query)

    # Deterministic post-filters. Mandatory: there is no server-side year filter,
    # and get_citations output is NOT snapshot-filtered.
    kept = []
    for cid in ids:
        paper = pool[cid]
        year = _year_of(paper)
        if year_min is not None and (year is None or year < year_min):
            continue
        if year_max is not None and (year is None or year > year_max):
            continue
        if min_cites is not None:
            count = paper.get("citationCount")
            if not isinstance(count, int) or count < min_cites:
                continue
        if min_authors is not None and len(_author_names(paper)) <= min_authors:
            continue
        if venue_ok is not None:
            venue = (paper.get("venue") or "").strip()
            if venue_ok and venue not in venue_ok:
                continue
        elif venues:
            venue = (paper.get("venue") or "").lower()
            if not any(v.lower() in venue or (venue and venue in v.lower())
                       for v in venues if v):
                continue
        kept.append(paper)
    print(f"  after filters: {len(kept)}")

    if not kept:  # filters wiped everything out - fall back to the unfiltered set
        kept = [pool[c] for c in ids] or list(pool.values())
        structural = False
        print(f"  filters emptied the set; falling back to {len(kept)}")

    # A topical constraint on top of structural filters needs semantic judgement.
    if topic and len(kept) > MAX_SUBMIT_METADATA:
        crit = [{"name": "topic", "description": topic, "probe": topic}]
        ordered, _ = await _grade_and_order(kept, query, crit, GPT_5_4_MINI,
                                           limit=min(len(kept), POOL_FOR_GRADING))
        kept = ordered
    else:
        kept.sort(key=lambda p: -(p.get("citationCount") or 0))

    # F1 = 2h/(n+G) with G usually 1-3. Padding is what turned metadata_4's
    # perfect recall into a 0.05. When strict structural filters already produced
    # a small set, submit exactly it; otherwise cap hard.
    cap = len(kept) if (structural and len(kept) <= MAX_SUBMIT_METADATA) else MAX_SUBMIT_METADATA
    _emit_ids(state, kept[:cap])


# ---------------------------------------------------------------------------
# Grading — per-criterion digits, so a weak criterion is identifiable
# ---------------------------------------------------------------------------

def _grade_prompt(query: str, criteria: list[dict], batch: list[tuple[int, dict]],
                  detail: int) -> str:
    lines = []
    for idx, paper in batch:
        gist = re.sub(r"\s+", " ", _tldr_text(paper) or (paper.get("abstract") or ""))[:detail]
        lines.append(f"[{idx}] ({_year_of(paper) or '?'}) "
                     f"{(paper.get('title') or '')[:170]} :: {gist}")
    crit = "\n".join(f"  C{i + 1}. {c['name']}: {c['description']}"
                     for i, c in enumerate(criteria))
    n = len(criteria)
    return (
        "You are grading papers retrieved for a literature search.\n\n"
        f"REQUEST: {query}\n\n"
        f"CRITERIA (a paper is fully relevant only if it satisfies ALL of them):\n"
        f"{crit}\n\n"
        "CANDIDATES:\n" + "\n".join(lines) + "\n\n"
        f"For each candidate output one line `index:DDD` where DDD is exactly {n} "
        "digits, one per criterion in order:\n"
        "  2 = the paper clearly and explicitly satisfies that criterion\n"
        "  1 = partially / probably, but not stated\n"
        "  0 = it does not\n"
        f"Example for {n} criteria: `7:{'2' * (n - 1)}0`\n"
        "Output nothing but those lines. Be strict about the digit 2."
    )


def _parse_grades(text: str, valid: set[int], n_crit: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for line in (text or "").splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s*[:=\-]\s*([0-2]+)", line.strip())
        if not m:
            continue
        idx, digits = int(m.group(1)), m.group(2)
        if idx in valid and idx not in out:
            vals = [int(d) for d in digits][:n_crit]
            vals += [0] * (n_crit - len(vals))
            out[idx] = vals
    return out


async def _grade_and_order(papers: list[dict], query: str, criteria: list[dict],
                           handle, limit: int, detail: int = 200
                           ) -> tuple[list[dict], dict[str, float]]:
    """Grade `papers[:limit]` in parallel batches; return them best-first plus a
    per-cid score in [0, 1]. Ungraded papers keep their incoming order behind
    the graded ones rather than being dropped."""
    n_crit = max(1, len(criteria))
    subset = papers[:limit]
    indexed = list(enumerate(subset))
    batches = [indexed[i:i + GRADE_BATCH] for i in range(0, len(indexed), GRADE_BATCH)]
    texts = await asyncio.gather(*[
        _llm(handle, _grade_prompt(query, criteria, b, detail), f"grade[{i}]")
        for i, b in enumerate(batches)
    ])

    grades: dict[int, list[int]] = {}
    for text, batch in zip(texts, batches):
        grades.update(_parse_grades(text, {i for i, _ in batch}, n_crit))

    scores: dict[str, float] = {}
    full = 0
    for idx, vals in grades.items():
        scores[_cid(subset[idx])] = sum(vals) / (2.0 * n_crit)
        if all(v == 2 for v in vals):
            full += 1
    print(f"  graded {len(grades)}/{len(subset)}; all-criteria-clear={full}")

    ordered = [p for _, p in sorted(
        indexed, key=lambda it: (-scores.get(_cid(it[1]), -1.0), it[0]))]
    return ordered + papers[limit:], scores


# ---------------------------------------------------------------------------
# semantic_f1
# ---------------------------------------------------------------------------

async def solve_semantic(state: TaskState, query: str) -> None:
    started = time.monotonic()
    kw_search = _get_tool(state, "search_papers_by_relevance")
    snippet_search = _get_tool(state, "snippet_search")
    batch_tool = _get_tool(state, "get_paper_batch")

    # --- 1. query understanding -------------------------------------------
    # The gold `relevance_criteria` are an LLM decomposition of the query into
    # must-satisfy requirements, so mirroring that generation is the target.
    ask = (
        "You are preparing a search over a scientific-paper index.\n\n"
        f'REQUEST: "{query}"\n\n'
        "The keyword index returns ZERO results for question-shaped input, so "
        "every search query must be a bare noun phrase (no 'what', 'how', 'could "
        "you', no question mark).\n\n"
        "Reply with JSON only:\n"
        "{\n"
        f'  "queries": [{N_VARIANTS} noun-phrase search queries, 3-8 words each, '
        "covering different vocabulary, synonyms and sub-aspects; the first should "
        "be the most literal],\n"
        '  "criteria": [\n'
        '    {"name": "short label",\n'
        '     "description": "one sentence stating a requirement a paper MUST '
        'satisfy to fully answer the request",\n'
        '     "probe": "3-6 word phrase, using the words a paper would actually '
        'use, for finding a passage that demonstrates THIS requirement"}\n'
        "  ],\n"
        '  "year_min": null or an integer if the request restricts publication year\n'
        "}\n\n"
        "Give 2-4 criteria. Decompose the request into its distinct requirements: "
        "the topic, the method/approach, and any specific property, task, metric "
        "or setting it names. Each criterion must be independently checkable "
        "against a paper's text."
    )
    plan = _json_from(await _llm(GPT_5_4_MINI, ask, "semantic/plan")) or {}
    queries = [q for q in (plan.get("queries") or []) if isinstance(q, str) and q.strip()]

    criteria: list[dict] = []
    for c in plan.get("criteria") or []:
        if isinstance(c, dict) and (c.get("name") or c.get("description")):
            name = str(c.get("name") or "")[:80]
            desc = str(c.get("description") or name)[:300]
            probe = str(c.get("probe") or name or desc)[:100]
            criteria.append({"name": name or probe, "description": desc, "probe": probe})
        elif isinstance(c, str) and c.strip():
            criteria.append({"name": c[:80], "description": c[:300], "probe": c[:100]})
    criteria = criteria[:4]
    if not criteria:
        criteria = [{"name": "request", "description": query, "probe": query[:100]}]

    try:
        year_min = int(plan.get("year_min"))
    except (TypeError, ValueError):
        year_min = None
    if not queries:
        stripped = re.sub(
            r"^(what|which|how|why|who|where|are|is|can|could|do|does|show|find|"
            r"give|recommend|suggest|i am looking for|i'm looking for)\b[^,?.]*[,?.]?\s*",
            "", query.strip(), flags=re.I)
        queries = [stripped.strip(" ?.") or query]
    queries = queries[:N_VARIANTS]
    crit_terms = [_terms(c["probe"]) | _terms(c["name"]) for c in criteria]
    print(f"  queries={queries}")
    for c in criteria:
        print(f"  criterion: {c['name']!r} probe={c['probe']!r}")
    print(f"  year_min={year_min}")

    # --- 2. wide, concurrent retrieval ------------------------------------
    tasks = [_safe(kw_search(keyword=q, fields=PAPER_FIELDS, limit=SEARCH_LIMIT),
                   f"kw:{q[:40]}") for q in queries] if kw_search else []
    n_kw = len(tasks)
    if snippet_search:
        # Passage retrieval tolerates natural-language input and finds papers
        # whose abstract never states the query's vocabulary.
        tasks.append(_safe(snippet_search(query=query, limit=100), "snippet:open",
                           timeout=SNIPPET_TIMEOUT))
    raws = await asyncio.gather(*tasks) if tasks else []

    pool: dict[str, dict] = {}
    snips: dict[str, list[tuple[int, str]]] = {}
    ranked_runs: list[list[str]] = []

    for raw in raws[:n_kw]:
        run = []
        for paper in _parse_items(raw):
            cid = _cid(paper)
            if not cid:
                continue
            run.append(cid)
            if cid in pool:
                pool[cid].update({k: v for k, v in paper.items() if v})
            else:
                pool[cid] = dict(paper)
        ranked_runs.append(run)
    print(f"  keyword runs -> {[len(r) for r in ranked_runs]}")

    if len(raws) > n_kw:
        run = []
        for entry in _parse_items(raws[n_kw]):
            paper = entry.get("paper") or {}
            cid = _cid(paper)
            if not cid:
                continue
            if cid not in pool:
                pool[cid] = {"corpusId": cid, "title": paper.get("title") or ""}
            if cid not in run:
                run.append(cid)
            text = (entry.get("snippet") or {}).get("text") or ""
            if text:
                snips.setdefault(cid, []).append((0, text))
        ranked_runs.append(run)
        print(f"  snippet run -> {len(run)} papers")

    print(f"  pool: {len(pool)} unique papers")
    if not pool:
        _emit(state, [])
        return

    # --- 3. reciprocal-rank fusion ----------------------------------------
    fused: dict[str, float] = {}
    for run in ranked_runs:
        for rank, cid in enumerate(run):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60.0 + rank)
    order = sorted(pool.values(), key=lambda p: -fused.get(_cid(p), 0.0))
    order = order[:MAX_SUBMIT_SEMANTIC + 60]

    # --- 4. hydrate: no submitted paper may reach the judge abstract-less --
    # Iteration 2 shipped a 340-char evidence blob (title+tldr only) that landed
    # on grade 2. Abstracts are the single densest criterion-bearing text.
    thin = [_cid(p) for p in order if _cid(p) and not (p.get("abstract") or "").strip()]
    if thin and batch_tool:
        chunks = [thin[i:i + 100] for i in range(0, min(len(thin), 300), 100)]
        hyd = await asyncio.gather(*[
            _safe(batch_tool(ids=[f"CorpusId:{c}" for c in chunk],
                             fields="title,abstract,corpusId,tldr,year,venue,citationCount"),
                  f"hydrate[{i}]") for i, chunk in enumerate(chunks)])
        filled = 0
        for raw in hyd:
            for paper in _parse_items(raw):
                cid = _cid(paper)
                if cid in pool:
                    pool[cid].update({k: v for k, v in paper.items() if v})
                    filled += 1
        print(f"  hydrated {filled}/{len(thin)} abstract-less records")

    # --- 5. grade against the criteria ------------------------------------
    order, gscores = await _grade_and_order(order, query, criteria, GPT_5_4_MINI,
                                            limit=min(len(order), POOL_FOR_GRADING),
                                            detail=180)
    if year_min is not None:
        # An explicit year constraint is usually mirrored by a gold criterion.
        order.sort(key=lambda p: 0 if (_year_of(p) or 9999) >= year_min else 1)

    head_n = min(HEAD_RERANK_N, len(order))
    if head_n and time.monotonic() - started < ENRICH_DEADLINE:
        head, hscores = await _grade_and_order(order[:head_n], query, criteria, GPT_5_4,
                                               limit=head_n, detail=320)
        gscores.update(hscores)
        order = head + order[head_n:]

    # --- 6. criterion-targeted evidence retrieval -------------------------
    # THE lever: grade 3 needs EVERY criterion supported, and the judge reads
    # markdown_evidence alone. Probing each criterion separately, scoped to the
    # candidate head, produces passages tagged with the criterion they support.
    if snippet_search and time.monotonic() - started < ENRICH_DEADLINE:
        head_ids = [_cid(p) for p in order[:CRIT_SNIPPET_HEAD] if _cid(p)]
        batches = [head_ids[i:i + CRIT_SNIPPET_BATCH]
                   for i in range(0, len(head_ids), CRIT_SNIPPET_BATCH)]
        calls, tags = [], []
        for ci, c in enumerate(criteria):
            for chunk in batches:
                calls.append(_safe(
                    snippet_search(query=c["probe"], limit=100,
                                   paper_ids=",".join(f"CorpusId:{x}" for x in chunk)),
                    f"snip:c{ci}", timeout=SNIPPET_TIMEOUT))
                tags.append(ci)
        print(f"  criterion snippet harvest: {len(calls)} scoped calls "
              f"over {len(head_ids)} papers")
        got = await asyncio.gather(*calls) if calls else []
        added = 0
        for ci, raw in zip(tags, got):
            for entry in _parse_items(raw):
                cid = _cid(entry.get("paper") or {})
                text = (entry.get("snippet") or {}).get("text") or ""
                if not cid or not text:
                    continue
                bucket = snips.setdefault(cid, [])
                if sum(1 for t, _ in bucket if t == ci) >= 2 or len(bucket) >= 8:
                    continue
                bucket.append((ci, text))
                added += 1
        print(f"  harvested {added} criterion-tagged passages over "
              f"{len(snips)} papers ({time.monotonic() - started:.0f}s elapsed)")

    # --- 7. assemble evidence, then re-rank by predicted judge grade ------
    submit = order[:MAX_SUBMIT_SEMANTIC]
    built = []
    for paper in submit:
        cid = _cid(paper)
        if not cid:
            continue
        # Criterion-tagged passages first so uncovered criteria get first refusal.
        bucket = sorted(snips.get(cid, []), key=lambda t: t[0])
        evidence, cov = _build_evidence(paper, bucket, crit_terms)
        built.append((cid, evidence, cov, gscores.get(cid, 0.0)))

    # K (median ~56) is smaller than the 250-slot submission, so the ordering
    # decides which papers are judged at all. Evidence coverage is the proximate
    # cause of grade 3, and it is measured, not predicted.
    n = len(built)
    rank_of = {cid: i for i, (cid, _, _, _) in enumerate(built)}
    built.sort(key=lambda b: -(0.55 * b[3] + 0.30 * b[2]
                               + 0.15 * (1.0 - rank_of[b[0]] / max(1, n))))

    full_cov = sum(1 for _, _, cov, _ in built if cov >= 0.999)
    print(f"  evidence: {full_cov}/{n} papers cover every criterion")
    _emit(state, [{"paper_id": cid, "markdown_evidence": ev} for cid, ev, _, _ in built])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _infer_score_type(query: str) -> str:
    """Fallback when metadata omits score_type (never observed, but cheap)."""
    q = query.lower()
    if re.search(r"\b(citing|cites|cited by|authored by|published in|between \d{4})\b", q):
        return "metadata_f1"
    if re.search(r"^(the\s+)?[\w\-\s]{0,40}\bpaper\b", q) and len(q.split()) <= 10:
        return "specific_f1"
    return "semantic_f1"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata.get("raw_query") or state.input_text or ""
        score_type = state.metadata.get("score_type") or _infer_score_type(query)
        print(f"[{state.sample_id}] score_type={score_type} query={query[:120]!r}")

        # Whatever happens, emit a parseable payload: an unparseable completion
        # scores 0 even when the retrieval was fine.
        state.output.completion = json.dumps(
            {"output": {"query_id": state.sample_id, "results": []}}
        )
        try:
            if score_type == "specific_f1":
                await solve_specific(state, query)
            elif score_type == "metadata_f1":
                await solve_metadata(state, query)
            else:
                await solve_semantic(state, query)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FATAL] {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
        return state

    return solve
