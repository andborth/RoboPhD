"""routed-deep-recall PaperFindingBench solver.

Routes on state.metadata["score_type"]:
  - semantic_f1: multi-query fan-out retrieval (relevance + snippet search),
    cheap-LLM batch grading of a few hundred candidates, submit a long
    ranked list (recall term is /K with K up to ~134; papers past K are
    never judged, so a well-ordered tail is free).
  - metadata_f1: structured constraint parsing, then the author/citation
    tools (search_authors_by_name -> get_author_papers, or
    get_citations + intersection + snapshot verification via
    get_paper_batch), deterministic year filter, LLM venue/topic filter.
  - specific_f1: title resolution, pick ONE candidate (precision: gold is
    usually a single id; submitting 10 caps F1 at ~0.18).

Evidence on semantic queries is verbatim retrieved text (title / tldr /
abstract / snippets) joined by " ... ", capped under the 2500-char limit —
the format that already passed grounding and earned grade 3s in iter 1.
"""

import asyncio
import json
import re

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI

SEARCH_FIELDS = "title,abstract,corpusId,tldr,year,venue"
EVIDENCE_CAP = 2400
MAX_SUBMIT = 250

_TOOL_SEM = asyncio.Semaphore(6)


# --------------------------------------------------------------------------
# tool plumbing
# --------------------------------------------------------------------------

def _get_tool(state: TaskState, name: str):
    by_name = {ToolDef(t).name: t for t in state.tools}
    if name not in by_name:
        raise RuntimeError(f"{name!r} not in state.tools (have: {sorted(by_name)})")
    return by_name[name]


def _parse_items(raw) -> list:
    """Flatten the MCP ContentText-JSON return shape into a list of dicts."""
    docs = []
    for item in raw or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(doc, dict) and "data" in doc:
            docs.extend(d for d in doc["data"] if d)
        elif doc:
            docs.append(doc)
    return docs


async def _call(tool, timeout: float = 290.0, **kwargs) -> list:
    """Call an MCP tool defensively: semaphore, timeout, parse, never raise."""
    try:
        async with _TOOL_SEM:
            raw = await asyncio.wait_for(tool(**kwargs), timeout=timeout)
        return _parse_items(raw)
    except Exception as e:
        print(f"  tool call failed ({kwargs.get('keyword') or kwargs.get('query') or ''}"
              f"): {type(e).__name__}: {str(e)[:150]}")
        return []


# --------------------------------------------------------------------------
# LLM plumbing
# --------------------------------------------------------------------------

async def _llm(model, prompt: str) -> str:
    try:
        resp = await model.generate(prompt)
        return (resp.completion or "").strip()
    except Exception as e:
        print(f"  LLM call failed: {type(e).__name__}: {str(e)[:150]}")
        return ""


def _extract_json(text: str):
    """Pull the first balanced JSON object/array out of an LLM reply."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        # fall through to next bracket type
    return None


# --------------------------------------------------------------------------
# candidate records
# --------------------------------------------------------------------------

def _cid(rec) -> str:
    v = rec.get("corpusId")
    return str(v) if v is not None else ""


def _tldr_text(rec) -> str:
    t = rec.get("tldr")
    if isinstance(t, dict):
        return (t.get("text") or "").strip()
    if isinstance(t, str):
        return t.strip()
    return ""


class Cand:
    __slots__ = ("cid", "title", "abstract", "tldr", "year", "venue",
                 "snippets", "rank", "grade")

    def __init__(self, cid):
        self.cid = cid
        self.title = ""
        self.abstract = ""
        self.tldr = ""
        self.year = None
        self.venue = ""
        self.snippets = []
        self.rank = 10 ** 9
        self.grade = 0


def _absorb(cand: Cand, rec: dict, rank: int | None = None):
    cand.title = cand.title or (rec.get("title") or "")
    cand.abstract = cand.abstract or (rec.get("abstract") or "")
    cand.tldr = cand.tldr or _tldr_text(rec)
    cand.year = cand.year if cand.year is not None else rec.get("year")
    cand.venue = cand.venue or (rec.get("venue") or "")
    if rank is not None:
        cand.rank = min(cand.rank, rank)


def _evidence(cand: Cand) -> str:
    """Verbatim passages joined by ' ... ', under the truncation cap."""
    passages = []
    budget = EVIDENCE_CAP

    def add(text):
        nonlocal budget
        text = (text or "").strip()
        if not text or budget <= 60 or len(passages) >= 8:
            return
        piece = text[:budget]
        passages.append(piece)
        budget -= len(piece) + 5

    add(cand.title)
    add(cand.tldr)
    add(cand.abstract)
    for sn in cand.snippets[:3]:
        add(sn)
    return " ... ".join(passages)


def _write_output(state: TaskState, entries: list):
    """entries: list of (paper_id, evidence). Dedupe, cap, write schema."""
    seen, results = set(), []
    for pid, ev in entries:
        pid = str(pid).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        results.append({"paper_id": pid, "markdown_evidence": ev if isinstance(ev, str) else ""})
        if len(results) >= MAX_SUBMIT:
            break
    state.output.completion = json.dumps(
        {"output": {"query_id": state.sample_id, "results": results}}
    )
    print(f"  submitted {len(results)} papers")


# --------------------------------------------------------------------------
# semantic_f1
# --------------------------------------------------------------------------

PLAN_PROMPT = """You are planning literature search for this query:

{query}

Reply with ONLY a JSON object:
{{
  "keyword_queries": ["...", "..."],   // 4-6 DIVERSE keyword/noun-phrase queries (3-8 words each,
                                       // no question words) covering different phrasings, synonyms
                                       // and subtopics of the request
  "snippet_query": "...",              // one full natural-language sentence restating what is sought
  "aspects": ["...", "..."],           // the 2-4 distinct requirements a paper must satisfy to be
                                       // a perfect answer, each as a short standalone statement
  "exclusions": ["..."]                // paper types/topics explicitly excluded by the query, else []
}}"""

GRADE_PROMPT = """Literature search query: {query}

A paper is a PERFECT match only if it satisfies ALL of these requirements:
{aspects}
{exclusions}
Grade each candidate below on its title/abstract:
3 = clearly satisfies ALL requirements
2 = satisfies most requirements, one is unclear
1 = related topic but clearly misses one or more requirements
0 = not relevant, or matches an exclusion

Candidates:
{cands}

Reply with one line per candidate, "index:grade", nothing else."""


async def _solve_semantic(state: TaskState, query: str):
    rel_search = _get_tool(state, "search_papers_by_relevance")
    snip_search = _get_tool(state, "snippet_search")
    batch_tool = _get_tool(state, "get_paper_batch")

    # -- plan ---------------------------------------------------------------
    plan = _extract_json(await _llm(GPT_5_4, PLAN_PROMPT.format(query=query))) or {}
    kw_queries = [q for q in plan.get("keyword_queries", []) if isinstance(q, str) and q.strip()][:6]
    snippet_q = plan.get("snippet_query") or query
    aspects = [a for a in plan.get("aspects", []) if isinstance(a, str)][:5]
    exclusions = [x for x in plan.get("exclusions", []) if isinstance(x, str)]
    if not kw_queries:
        distilled = await _llm(GPT_5_4_MINI,
                               "Extract a concise keyword search query (3-8 words, no punctuation) "
                               "for a scientific paper search engine from this request. Reply with "
                               f"the keywords only.\n\nRequest: {query}")
        kw_queries = [distilled or query]
    print(f"  plan: {len(kw_queries)} kw queries, {len(aspects)} aspects; kw={kw_queries!r}")

    # -- retrieve -----------------------------------------------------------
    tasks = [_call(rel_search, keyword=kw, fields=SEARCH_FIELDS, limit=100)
             for kw in kw_queries]
    tasks.append(_call(snip_search, timeout=270.0, query=snippet_q, limit=40))
    result_lists = await asyncio.gather(*tasks)

    pool: dict[str, Cand] = {}
    for hits in result_lists[:-1]:
        for rank, rec in enumerate(hits):
            cid = _cid(rec)
            if not cid:
                continue
            cand = pool.setdefault(cid, Cand(cid))
            _absorb(cand, rec, rank)
    for rank, entry in enumerate(result_lists[-1]):
        paper = entry.get("paper") or {}
        cid = _cid(paper)
        if not cid:
            continue
        cand = pool.setdefault(cid, Cand(cid))
        _absorb(cand, paper, rank)
        sn = (entry.get("snippet") or {}).get("text") or ""
        if sn:
            cand.snippets.append(sn)
    print(f"  pooled {len(pool)} unique candidates")

    # -- backfill missing abstracts for retrieval-strong candidates ---------
    ordered = sorted(pool.values(), key=lambda c: c.rank)
    missing = [c for c in ordered[:250] if not c.abstract][:150]
    for i in range(0, len(missing), 50):
        chunk = missing[i:i + 50]
        recs = await _call(batch_tool,
                           ids=[f"CorpusId:{c.cid}" for c in chunk],
                           fields="title,abstract,tldr,corpusId,year,venue")
        by_cid = {_cid(r): r for r in recs if isinstance(r, dict)}
        for c in chunk:
            if c.cid in by_cid:
                _absorb(c, by_cid[c.cid])
    if missing:
        print(f"  backfilled abstracts for {len(missing)} candidates")

    # -- grade in parallel cheap-LLM batches --------------------------------
    cands = ordered[:400]
    aspect_txt = "\n".join(f"- {a}" for a in aspects) if aspects else f"- {query}"
    excl_txt = ("Excluded (grade 0): " + "; ".join(exclusions) + "\n") if exclusions else ""

    async def grade_batch(batch, offset):
        lines = []
        for j, c in enumerate(batch):
            desc = (c.abstract or c.tldr or "")[:350]
            lines.append(f"[{offset + j}] {c.title[:150]} — {desc}")
        reply = await _llm(GPT_5_4_MINI, GRADE_PROMPT.format(
            query=query, aspects=aspect_txt, exclusions=excl_txt, cands="\n".join(lines)))
        for m in re.finditer(r"(\d+)\s*[:=]\s*([0-3])", reply):
            idx, g = int(m.group(1)), int(m.group(2))
            if offset <= idx < offset + len(batch):
                cands[idx].grade = g

    await asyncio.gather(*(grade_batch(cands[i:i + 30], i)
                           for i in range(0, len(cands), 30)))
    hist = {}
    for c in cands:
        hist[c.grade] = hist.get(c.grade, 0) + 1
    print(f"  grade histogram: {dict(sorted(hist.items(), reverse=True))}")

    # -- rank & submit ------------------------------------------------------
    cands.sort(key=lambda c: (-c.grade, c.rank))
    return [(c.cid, _evidence(c)) for c in cands]


# --------------------------------------------------------------------------
# metadata_f1
# --------------------------------------------------------------------------

META_PARSE_PROMPT = """Parse this literature-search request into structured constraints.

Request: {query}

Reply with ONLY a JSON object:
{{
  "authors": ["..."],        // author names the papers must be BY (not papers they cite), else []
  "venues": ["..."],         // venue/journal/conference constraints as stated, else []
  "years": [2014, 2017],     // explicitly allowed publication years, else []
  "year_min": null,          // inclusive lower bound if a range is stated, else null
  "year_max": null,          // inclusive upper bound if a range is stated, else null
  "cited_papers": ["..."],   // if the request asks for papers CITING some paper(s), a best-guess
                             // title for each cited anchor paper, else []
  "topic": null,             // topical constraint on the papers themselves, else null
  "keyword_query": "..."     // 3-8 word keyword fallback query
}}"""

META_FILTER_PROMPT = """A user asked for: {query}

Constraints to enforce: {constraints}

Candidates (index | year | venue | title):
{rows}

Return ONLY a JSON array of the indices of candidates that satisfy ALL the constraints.
Venue constraints match the official venue name (e.g. "ACL" matches "Annual Meeting of the
Association for Computational Linguistics" but NOT workshops, TACL, or other venues).
Be strict: when a candidate clearly violates a constraint, exclude it."""


def _year_ok(year, years, ymin, ymax):
    if year is None:
        return not years and ymin is None and ymax is None
    try:
        y = int(year)
    except (TypeError, ValueError):
        return False
    if years and y not in years:
        return False
    if ymin is not None and y < ymin:
        return False
    if ymax is not None and y > ymax:
        return False
    return True


async def _resolve_anchor(state: TaskState, title: str) -> dict | None:
    by_title = _get_tool(state, "search_paper_by_title")
    hits = await _call(by_title, title=title, fields="title,corpusId,citationCount")
    for h in hits:
        if h.get("paperId") or h.get("corpusId"):
            return h
    rel = _get_tool(state, "search_papers_by_relevance")
    hits = await _call(rel, keyword=title, fields="title,corpusId,citationCount", limit=5)
    return hits[0] if hits else None


async def _solve_metadata(state: TaskState, query: str):
    parse = _extract_json(await _llm(GPT_5_4, META_PARSE_PROMPT.format(query=query))) or {}
    authors = [a for a in parse.get("authors", []) if isinstance(a, str) and a.strip()]
    venues = [v for v in parse.get("venues", []) if isinstance(v, str) and v.strip()]
    years = set()
    for y in parse.get("years", []) or []:
        try:
            years.add(int(y))
        except (TypeError, ValueError):
            pass
    ymin, ymax = parse.get("year_min"), parse.get("year_max")
    ymin = int(ymin) if isinstance(ymin, (int, float, str)) and str(ymin).isdigit() else None
    ymax = int(ymax) if isinstance(ymax, (int, float, str)) and str(ymax).isdigit() else None
    cited = [c for c in parse.get("cited_papers", []) if isinstance(c, str) and c.strip()]
    topic = parse.get("topic") if isinstance(parse.get("topic"), str) else None
    print(f"  parsed: authors={authors} venues={venues} years={sorted(years)} "
          f"range=({ymin},{ymax}) cited={cited} topic={topic!r}")

    candidates: dict[str, dict] = {}

    if cited:
        get_cit = _get_tool(state, "get_citations")
        anchors = [a for a in await asyncio.gather(
            *(_resolve_anchor(state, t) for t in cited)) if a]
        print(f"  resolved {len(anchors)}/{len(cited)} anchors: "
              f"{[a.get('title', '')[:60] for a in anchors]}")
        citer_sets = []
        for a in anchors:
            hits = await _call(get_cit, paper_id=str(a.get("corpusId")),
                               fields="title,corpusId,year", limit=1000)
            citers = {}
            for h in hits:
                rec = h.get("citingPaper") if isinstance(h.get("citingPaper"), dict) else h
                cid = _cid(rec)
                if cid:
                    citers[cid] = rec
            print(f"  anchor {str(a.get('title'))[:40]!r}: {len(citers)} citers")
            if citers:
                citer_sets.append(citers)
        if citer_sets:
            common = set(citer_sets[0])
            for s in citer_sets[1:]:
                common &= set(s)
            if not common and len(citer_sets) > 1:
                # citation windows (<=1000 newest) may not overlap; fall back to
                # the smallest citer set rather than returning nothing
                smallest = min(citer_sets, key=len)
                common = set(smallest)
                print("  empty intersection; falling back to smallest citer set")
            for cid in common:
                candidates[cid] = citer_sets[0].get(cid) or next(
                    s[cid] for s in citer_sets if cid in s)
        print(f"  {len(candidates)} candidate citing papers")

        # snapshot-verify: get_citations is NOT date-cutoff filtered, but
        # get_paper_batch is — unresolved ids are post-snapshot, drop them.
        batch_tool = _get_tool(state, "get_paper_batch")
        cids = list(candidates)
        verified: dict[str, dict] = {}
        for i in range(0, len(cids), 100):
            chunk = cids[i:i + 100]
            recs = await _call(batch_tool, ids=[f"CorpusId:{c}" for c in chunk],
                               fields="title,corpusId,year,venue,authors")
            for r in recs:
                if isinstance(r, dict) and _cid(r):
                    verified[_cid(r)] = r
        if cids and not verified:
            print("  batch verification returned nothing; keeping unverified citers")
        else:
            candidates = verified
            print(f"  {len(candidates)} survive snapshot verification")

    elif authors:
        find_auth = _get_tool(state, "search_authors_by_name")
        get_papers = _get_tool(state, "get_author_papers")
        per_author_sets = []
        for name in authors:
            recs = await _call(find_auth, name=name, limit=10)
            surname = name.split()[-1].lower()
            ids = [r for r in recs
                   if surname in (r.get("name") or "").lower() and r.get("paperCount")]
            ids.sort(key=lambda r: -(r.get("paperCount") or 0))
            ids = ids[:4]
            print(f"  author {name!r}: identities "
                  f"{[(r.get('authorId'), r.get('paperCount')) for r in ids]}")
            papers = {}
            for r in ids:
                hits = await _call(get_papers, author_id=str(r.get("authorId")),
                                   paper_fields="title,corpusId,year,venue", limit=500)
                for h in hits:
                    cid = _cid(h)
                    if cid:
                        papers[cid] = h
            per_author_sets.append(papers)
        if per_author_sets:
            common = set(per_author_sets[0])
            for s in per_author_sets[1:]:
                common &= set(s)
            if not common:
                common = set().union(*per_author_sets)
            for cid in common:
                candidates[cid] = next(s[cid] for s in per_author_sets if cid in s)
        print(f"  {len(candidates)} candidate author papers")

    from_keyword_fallback = False
    if not candidates:
        from_keyword_fallback = True
        rel = _get_tool(state, "search_papers_by_relevance")
        kw = parse.get("keyword_query") or query
        hits = await _call(rel, keyword=kw, fields="title,corpusId,year,venue,authors", limit=100)
        for h in hits:
            cid = _cid(h)
            if cid:
                candidates[cid] = h
        print(f"  keyword fallback pooled {len(candidates)}")

    # deterministic year filter
    kept = [(cid, r) for cid, r in candidates.items()
            if _year_ok(r.get("year"), years, ymin, ymax)]
    print(f"  {len(kept)} after year filter")

    # LLM filter for venue/topic (and author, on the citation/keyword paths)
    need_llm = bool(venues or topic or (cited and authors) or from_keyword_fallback)
    if kept and need_llm:
        kept = kept[:500]
        constraints = []
        if venues:
            constraints.append(f"venue must be one of {venues}")
        if topic:
            constraints.append(f"paper must be about: {topic}")
        if authors and cited:
            constraints.append(f"authored by {authors}")
        if not constraints:
            constraints.append("all constraints stated in the request")
        rows = "\n".join(
            f"{i} | {r.get('year')} | {(r.get('venue') or '')[:60]} | {(r.get('title') or '')[:110]}"
            for i, (cid, r) in enumerate(kept))
        reply = await _llm(GPT_5_4_MINI, META_FILTER_PROMPT.format(
            query=query, constraints="; ".join(constraints), rows=rows))
        idxs = _extract_json(reply)
        if isinstance(idxs, list):
            picked = [kept[i] for i in idxs
                      if isinstance(i, int) and 0 <= i < len(kept)]
            if picked:
                kept = picked
            print(f"  {len(kept)} after LLM constraint filter")

    return [(cid, "") for cid, _ in kept]


# --------------------------------------------------------------------------
# specific_f1
# --------------------------------------------------------------------------

SPECIFIC_PARSE_PROMPT = """A user is looking for one specific known paper:

{query}

Reply with ONLY a JSON object:
{{
  "title_guess": "...",    // your best guess at the paper's exact title
  "keyword_query": "..."   // 3-8 word keyword search query for it
}}"""

SPECIFIC_PICK_PROMPT = """A user is looking for this specific paper: {query}

Candidates:
{rows}

Which candidate index is THE paper the user means? Reply with the single index.
Only if two candidates genuinely could each be that exact paper, reply with both
comma-separated. Reply with indices only."""


async def _solve_specific(state: TaskState, query: str):
    parse = _extract_json(await _llm(GPT_5_4, SPECIFIC_PARSE_PROMPT.format(query=query))) or {}
    title_guess = parse.get("title_guess") or query
    kw = parse.get("keyword_query") or query

    by_title = _get_tool(state, "search_paper_by_title")
    rel = _get_tool(state, "search_papers_by_relevance")
    fields = "title,corpusId,year,venue,authors,abstract"
    title_hits, rel_hits = await asyncio.gather(
        _call(by_title, title=title_guess, fields=fields),
        _call(rel, keyword=kw, fields=fields, limit=20),
    )
    cands, seen = [], set()
    for rec in title_hits + rel_hits:
        cid = _cid(rec)
        if cid and cid not in seen and (rec.get("paperId") or rec.get("title")):
            seen.add(cid)
            cands.append(rec)
    print(f"  {len(cands)} specific candidates (title match: {bool(title_hits)})")
    if not cands:
        return []

    rows = []
    for i, r in enumerate(cands[:25]):
        auths = ", ".join((a.get("name") or "") for a in (r.get("authors") or [])[:4])
        rows.append(f"{i} | {r.get('year')} | {(r.get('venue') or '')[:40]} | "
                    f"{(r.get('title') or '')[:120]} | {auths}")
    reply = await _llm(GPT_5_4_MINI, SPECIFIC_PICK_PROMPT.format(
        query=query, rows="\n".join(rows)))
    idxs = [int(m.group(0)) for m in re.finditer(r"\d+", reply)][:2]
    picked = [cands[i] for i in idxs if 0 <= i < len(cands)]
    if not picked:
        picked = [cands[0]]
    return [(_cid(r), "") for r in picked]


# --------------------------------------------------------------------------
# fallback + solver
# --------------------------------------------------------------------------

async def _fallback(state: TaskState, query: str):
    rel = _get_tool(state, "search_papers_by_relevance")
    kw = await _llm(GPT_5_4_MINI,
                    "Extract a concise keyword search query (3-8 words, no punctuation) for a "
                    f"scientific paper search engine. Reply with keywords only.\n\nRequest: {query}")
    hits = await _call(rel, keyword=kw or query, fields=SEARCH_FIELDS, limit=50)
    entries = []
    for rec in hits:
        c = Cand(_cid(rec))
        _absorb(c, rec)
        entries.append((c.cid, _evidence(c)))
    return entries


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata.get("raw_query") or state.input_text
        score_type = state.metadata.get("score_type", "")
        print(f"[{state.sample_id}] score_type={score_type} query={query[:100]!r}")

        entries = []
        try:
            if score_type == "specific_f1":
                entries = await _solve_specific(state, query)
            elif score_type == "metadata_f1":
                entries = await _solve_metadata(state, query)
            else:
                entries = await _solve_semantic(state, query)
        except Exception as e:
            import traceback
            print(f"  solver error: {type(e).__name__}: {e}")
            traceback.print_exc()
        if not entries:
            try:
                print("  running fallback pipeline")
                entries = await _fallback(state, query)
            except Exception as e:
                print(f"  fallback error: {type(e).__name__}: {e}")
        _write_output(state, entries)
        return state

    return solve
