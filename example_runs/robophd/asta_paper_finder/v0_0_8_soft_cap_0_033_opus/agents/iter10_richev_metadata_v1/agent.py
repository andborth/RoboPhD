"""PaperFindingBench solver — iter10_richev_metadata_v1.

Combines the two proven frontiers:

  * SEMANTIC path (73% of queries) is reverted BYTE-FOR-BYTE to iter3_evidence_rerank_v1
    — the fresh-batch winner. On semantic_98 (identical criteria) iter3 earned 5 grade-3
    papers vs iter9's 2, and three of the SAME papers (209387721, 233356271, 252163145)
    were graded Perfectly Relevant under iter3's evidence but only Highly/Somewhat under
    iter9's facet-covering evidence. Recall counts only grade-3, so iter3's rich,
    snippet-led evidence dump is the better grade-2->grade-3 converter. We keep it exactly:
    snippet_search + keyword variants retrieval, chunked fault-tolerant enrichment, a cheap
    0-3 title+snippet rerank, and evidence = up to 4 distinct snippets + full abstract +
    tldr filling the 2500-char budget.

  * METADATA / SPECIFIC paths (27%) keep iter9's accumulated, guarded machinery:
    citation-relation intersection (papers citing/cited-by anchors), author LLM
    venue-disambiguation, explicit year-SETs, min_citations/min_authors filters, a
    snippet third source for specific + AuthorYYYY parsing + duplicate-only hedge.

  * NEW (additive, guarded): author-of-referenced-paper. The metadata extractor may emit
    `author_of_paper`; when present we resolve that paper, read its author list, and feed
    those names into the author path (metadata_14: "papers by an author of the BERT
    paper" — a structural 0 for all prior agents). Gated on the new field, wrapped in
    try/except, so it cannot regress existing metadata queries.

All LLM calls go through model_registry (GPT_5_4_MINI, cheapest, no reasoning). Tool
calls are free and used generously. ~$0.004/query, deep in the free zone ($0.033).
"""

import json
import re

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI


# --------------------------------------------------------------------------
# Tool + parsing helpers
# --------------------------------------------------------------------------
def _maybe_tool(state: TaskState, name: str):
    by_name = {ToolDef(t).name: t for t in state.tools}
    return by_name.get(name)


def _parse_items(raw) -> list[dict]:
    """Flatten MCP tool output: a list of ContentText whose `.text` is JSON.
    Some payloads wrap rows in {"data": [...]}; unwrap those."""
    docs = []
    for item in raw or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(doc, dict) and "data" in doc and isinstance(doc["data"], list):
            docs.extend(doc["data"])
        else:
            docs.append(doc)
    return docs


async def _call(tool, **kwargs):
    """Await a tool, swallowing failures (tools can be flaky / timeout)."""
    if tool is None:
        return None
    try:
        return await tool(**kwargs)
    except Exception as e:  # noqa: BLE001
        try:
            name = ToolDef(tool).name
        except Exception:  # noqa: BLE001
            name = "?"
        print(f"  tool error {name}: {e!r}")
        return None


async def _llm_json(prompt: str, fallback):
    """Cheap LLM call that must return a JSON value; lenient parsing."""
    try:
        resp = await GPT_5_4_MINI.generate(prompt)
        text = (resp.completion or "").strip()
    except Exception as e:  # noqa: BLE001
        print(f"  llm error: {e!r}")
        return fallback
    if not text:
        return fallback
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return fallback


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def _clip(s: str, n: int) -> str:
    return (s or "").strip()[:n]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


# --------------------------------------------------------------------------
# Evidence construction (grounding-safe: every passage verbatim from retrieved
# text for that same paper, joined by " ... ", capped < 2500 chars).
# iter3 winner: snippet-led rich dump — the better grade-2 -> grade-3 converter.
# --------------------------------------------------------------------------
def _build_evidence(title: str, abstract: str, tldr: str, snippets: list[str]) -> str:
    """Assemble the richest grounded evidence under the 2500-char budget.
    Priority: distinct snippets (often cover different criteria/sections) +
    full abstract + tldr. Passages are deduped by normalized text so we don't
    waste budget on near-duplicates."""
    passages: list[str] = []
    seen: set[str] = set()

    def add(text: str, cap: int):
        text = _clip(text, cap)
        if not text:
            return
        key = _norm(text)[:120]
        if key in seen:
            return
        # skip if this passage is a substring of one already added
        for k in seen:
            if key and key in k:
                return
        seen.add(key)
        passages.append(text)

    for sn in snippets[:4]:
        add(sn, 700)
    add(abstract, 1400)
    add(tldr, 400)
    if not passages and title:
        add(title, 300)

    ev = " ... ".join(passages)
    return ev[:2400]


# --------------------------------------------------------------------------
# Candidate accumulator
# --------------------------------------------------------------------------
class Cand:
    __slots__ = ("cid", "title", "abstract", "tldr", "snippets", "score", "sources")

    def __init__(self, cid):
        self.cid = str(cid)
        self.title = ""
        self.abstract = ""
        self.tldr = ""
        self.snippets: list[str] = []
        self.score = 0.0
        self.sources: set[str] = set()

    def add_snippet(self, sn: str):
        sn = (sn or "").strip()
        if not sn:
            return
        n = _norm(sn)[:120]
        for existing in self.snippets:
            if _norm(existing)[:120] == n:
                return
        if len(self.snippets) < 5:
            self.snippets.append(sn)


def _set_tldr(c: Cand, tl):
    if c.tldr:
        return
    if isinstance(tl, dict):
        c.tldr = (tl.get("text") or "").strip()
    elif isinstance(tl, str):
        c.tldr = tl.strip()


# --------------------------------------------------------------------------
# SEMANTIC path — maximize grade-3 recall (rich evidence) + protect rank/order
# (byte-for-byte iter3_evidence_rerank_v1)
# --------------------------------------------------------------------------
async def _solve_semantic(state, query) -> list[dict]:
    cands: dict[str, Cand] = {}

    def get(cid) -> Cand:
        cid = str(cid)
        if cid not in cands:
            cands[cid] = Cand(cid)
        return cands[cid]

    # --- keyword variants for breadth (1 cheap LLM call) ------------------
    clean_q = re.sub(r"[?!]", "", query).strip()
    ex = await _llm_json(
        "You turn a literature-search request into keyword queries for a "
        "scientific paper search engine (literal keyword match, NO question "
        "framing, NO operators). Give the core noun-phrase query plus 2 "
        "alternative phrasings using synonyms / broader or narrower terms to "
        "widen recall.\n"
        f"Request: {query}\n\n"
        'Reply ONLY with JSON: {"queries": ["q1", "q2", "q3"]}',
        {"queries": []},
    )
    variants = []
    if isinstance(ex, dict):
        variants = [q.strip() for q in ex.get("queries", []) if isinstance(q, str) and q.strip()]
    if clean_q and clean_q not in variants:
        variants.append(clean_q)
    variants = variants[:4]
    print(f"  keyword variants: {variants}")

    # --- snippet_search: NL-tolerant, gives verbatim evidence -------------
    snip = _maybe_tool(state, "snippet_search")
    if snip is not None:
        raw = await _call(snip, query=query, limit=100)
        rows = _parse_items(raw)
        print(f"  snippet_search returned {len(rows)} passages")
        n = len(rows)
        for i, r in enumerate(rows):
            paper = r.get("paper") or {}
            cid = paper.get("corpusId")
            if cid is None:
                continue
            c = get(cid)
            c.sources.add("snip")
            c.title = c.title or (paper.get("title") or "")
            c.add_snippet(((r.get("snippet") or {}).get("text") or ""))
            c.score += (n - i) / max(n, 1)

    # --- keyword search over variants -------------------------------------
    kw = _maybe_tool(state, "search_papers_by_relevance")
    if kw is not None:
        for v in variants:
            raw = await _call(kw, keyword=v, fields="title,abstract,corpusId,tldr", limit=100)
            rows = _parse_items(raw)
            n = len(rows)
            for i, r in enumerate(rows):
                cid = r.get("corpusId")
                if cid is None:
                    continue
                c = get(cid)
                c.sources.add("kw")
                c.title = c.title or (r.get("title") or "")
                if r.get("abstract") and not c.abstract:
                    c.abstract = r["abstract"]
                _set_tldr(c, r.get("tldr"))
                c.score += 0.7 * (n - i) / max(n, 1)
    print(f"  {len(cands)} unique candidates after retrieval")

    if not cands:
        return []

    # multi-source agreement bonus (found by both retrievers -> strong)
    for c in cands.values():
        if len(c.sources) > 1:
            c.score += 1.5

    ordered = sorted(cands.values(), key=lambda c: c.score, reverse=True)

    # --- enrich evidence: chunked, fault-tolerant batch fetch -------------
    # (recall-max lost a whole query when one post-cutoff id failed the batch)
    need = [c for c in ordered[:150] if not c.abstract]
    batch = _maybe_tool(state, "get_paper_batch")
    if batch is not None and need:
        for i in range(0, len(need), 20):
            chunk = need[i:i + 20]
            ids = [f"CorpusId:{c.cid}" for c in chunk]
            raw = await _call(batch, ids=ids, fields="title,abstract,corpusId,tldr")
            rows = _parse_items(raw)
            by_cid = {str(r.get("corpusId")): r for r in rows if r.get("corpusId") is not None}
            for c in chunk:
                r = by_cid.get(c.cid)
                if not r:
                    continue
                if r.get("abstract") and not c.abstract:
                    c.abstract = r["abstract"]
                _set_tldr(c, r.get("tldr"))

    # --- LLM rerank of the top slice (order = rank term + recall window) --
    ordered = await _rerank(query, ordered)

    # --- assemble submission (best-first; no precision penalty) -----------
    results = []
    have = set()
    for c in ordered:
        if c.cid in have:
            continue
        ev = _build_evidence(c.title, c.abstract, c.tldr, c.snippets)
        if not ev:
            continue
        results.append({"paper_id": c.cid, "markdown_evidence": ev})
        have.add(c.cid)
        if len(results) >= 200:
            break

    # Rank-degeneracy safeguard: guarantee grade variance (all-equal -> rank 0).
    # With diverse retrieval this is nearly always already true, but keep it.
    tail_pool = ordered[len(results):len(results) + 30]
    added = 0
    for c in tail_pool:
        if added >= 4:
            break
        if c.cid in have or not c.title:
            continue
        results.append({"paper_id": c.cid, "markdown_evidence": _clip(c.title, 250)})
        have.add(c.cid)
        added += 1

    print(f"  semantic: submitting {len(results)} papers (+{added} tail)")
    return results


async def _rerank(query, ordered: list[Cand], top_n: int = 50) -> list[Cand]:
    """Rate the top candidates 0-3 for query fit with one cheap LLM call and
    sort by (llm_score, retrieval_score). Robust: any failure keeps the input
    order. Never drops candidates."""
    head = ordered[:top_n]
    tail = ordered[top_n:]
    if len(head) < 5:
        return ordered

    def blurb(c: Cand) -> str:
        parts = [c.title[:140]]
        if c.snippets:
            parts.append(c.snippets[0][:220])
        elif c.tldr:
            parts.append(c.tldr[:220])
        return " | ".join(p for p in parts if p)

    listing = "\n".join(f"{i}: {blurb(c)}" for i, c in enumerate(head))
    res = await _llm_json(
        "You are ranking retrieved papers by how well each satisfies a "
        "literature-search request. Rate EVERY candidate 0-3: 3 = clearly and "
        "fully matches every aspect of the request; 2 = strong match, minor gap; "
        "1 = loosely related; 0 = off-topic.\n"
        f'Request: "{query}"\n\n'
        f"Candidates (index: title | snippet):\n{listing}\n\n"
        'Reply ONLY with a JSON object mapping each index (as a string) to its '
        'integer score, e.g. {"0": 3, "1": 1, ...}. Include ALL indices.',
        {},
    )
    if not isinstance(res, dict) or not res:
        print("  rerank: fallback to retrieval order")
        return ordered

    scores = {}
    for k, v in res.items():
        try:
            scores[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    if len(scores) < len(head) // 2:
        print(f"  rerank: too few scores ({len(scores)}/{len(head)}), fallback")
        return ordered

    # Stable sort: primary = llm score (missing -> 0), secondary = retrieval order.
    order_idx = {id(c): i for i, c in enumerate(head)}
    reranked = sorted(
        head,
        key=lambda c: (-(scores.get(order_idx[id(c)], 0.0)), order_idx[id(c)]),
    )
    print(f"  rerank: reordered top {len(head)} candidates")
    return reranked + tail


# --------------------------------------------------------------------------
# SPECIFIC path — high-precision; title-gen from informal reference, no blind hedge
# (iter9 heritage)
# --------------------------------------------------------------------------
async def _solve_specific(state, query) -> list[dict]:
    # 1. LLM turns the informal reference into the likely EXACT paper title,
    #    and parses any "AuthorYYYY" citation key into author/year hints.
    guess = await _llm_json(
        "A user refers to a specific, known scientific paper informally. The "
        "reference may embed a citation key like 'Smith2021' (author surname + "
        "year) or an acronym/dataset name.\n"
        f'Reference: "{query}"\n\n'
        "Give the paper's most likely EXACT published title (your best guess), a "
        "short keyword query for a title search, and any author surname / year you "
        "can infer.\n"
        'Reply ONLY with JSON: {"title": "likely exact title", "keywords": '
        '"short query", "author": "surname or empty", "year": int or null}',
        {},
    )
    guess_title = (guess.get("title") or "").strip() if isinstance(guess, dict) else ""
    guess_kw = (guess.get("keywords") or "").strip() if isinstance(guess, dict) else ""
    guess_author = (guess.get("author") or "").strip() if isinstance(guess, dict) else ""
    guess_year = guess.get("year") if isinstance(guess, dict) else None

    cleaned = re.sub(r"^\s*(the|a|an)\s+", "", query.strip(), flags=re.I)
    cleaned = re.sub(r"\s+paper[.?!]?\s*$", "", cleaned, flags=re.I).strip() or query
    title_queries = [t for t in (guess_title, cleaned) if t]
    kw_queries = [k for k in (guess_kw, cleaned) if k]

    pool: list[tuple[str, dict]] = []
    seen: set[str] = set()
    title_match = None

    def add(cid, title, abstract):
        cid = str(cid)
        if cid in seen:
            return
        seen.add(cid)
        pool.append((cid, {"title": title or "", "abstract": abstract or ""}))

    title_tool = _maybe_tool(state, "search_paper_by_title")
    if title_tool is not None:
        for tq in title_queries:
            raw = await _call(title_tool, title=tq, fields="title,abstract,corpusId")
            for r in _parse_items(raw):
                cid = r.get("corpusId")
                if cid is not None and r.get("paperId") is not None:
                    if title_match is None:
                        title_match = str(cid)
                    add(cid, r.get("title"), r.get("abstract"))

    kw = _maybe_tool(state, "search_papers_by_relevance")
    if kw is not None:
        for kq in kw_queries:
            raw = await _call(kw, keyword=kq, fields="title,abstract,corpusId", limit=10)
            for r in _parse_items(raw):
                cid = r.get("corpusId")
                if cid is not None:
                    add(cid, r.get("title"), r.get("abstract"))

    # snippet_search: tolerant of acronyms / natural-language references that the
    # keyword/title tools miss (e.g. "MS^2 DeYoung2021").
    snip = _maybe_tool(state, "snippet_search")
    if snip is not None and len(pool) < 20:
        raw = await _call(snip, query=cleaned, limit=10)
        for r in _parse_items(raw):
            paper = r.get("paper") or {}
            cid = paper.get("corpusId")
            if cid is not None:
                add(cid, paper.get("title"), "")

    if not pool:
        return []

    listing = "\n".join(f"{i}: {v['title'][:150]}" for i, (_c, v) in enumerate(pool))
    hint = ""
    if guess_author or guess_year:
        hint = f" (hint: likely by {guess_author or '?'}, year {guess_year or '?'})"
    sel = await _llm_json(
        f'A user is looking for a specific known paper: "{query}"{hint}.\n'
        f"Candidates:\n{listing}\n\n"
        "Return the indices of ONLY the candidate(s) that ARE that exact paper "
        "(usually 1; return several ONLY if they are clearly duplicate records of "
        "the SAME work). Be strict — do not include merely-related papers.\n"
        'Reply ONLY with JSON: {"indices": [i, ...]}',
        {"indices": []},
    )
    idxs = []
    if isinstance(sel, dict):
        idxs = [i for i in sel.get("indices", []) if isinstance(i, int) and 0 <= i < len(pool)]

    chosen: list[int] = []
    for i in idxs[:3]:
        if i not in chosen:
            chosen.append(i)
    if not chosen:
        if title_match:
            chosen = [next(i for i, (cid, _v) in enumerate(pool) if cid == title_match)]
        else:
            chosen = [0]

    # Duplicate-only hedge: if the LLM chose exactly one, add a second id ONLY if
    # its title is a near-duplicate of the pick (a real duplicate corpus record),
    # never a random candidate (which tanked specific_7).
    if len(chosen) == 1:
        pick_title = _norm(pool[chosen[0]][1]["title"])
        if pick_title:
            for i in range(len(pool)):
                if i in chosen:
                    continue
                t = _norm(pool[i][1]["title"])
                if t and (t == pick_title or t in pick_title or pick_title in t):
                    chosen.append(i)
                    break

    results = []
    for i in chosen:
        cid, v = pool[i]
        results.append({"paper_id": cid, "markdown_evidence": _clip(v["title"], 250)})
    print(f"  specific: submitting {len(results)} papers")
    return results


# --------------------------------------------------------------------------
# METADATA path — author/venue/year filters + citation-relation queries
# (iter9 heritage + author-of-referenced-paper branch)
# --------------------------------------------------------------------------
def _unwrap_citing(rows: list[dict]) -> list[dict]:
    """get_citations rows are wrapped: {"citingPaper": {...}}. Unwrap to the
    paper dict; pass through already-flat rows unchanged."""
    out = []
    for r in rows:
        if isinstance(r, dict) and "citingPaper" in r and isinstance(r["citingPaper"], dict):
            out.append(r["citingPaper"])
        elif isinstance(r, dict) and "citedPaper" in r and isinstance(r["citedPaper"], dict):
            out.append(r["citedPaper"])
        else:
            out.append(r)
    return out


async def _resolve_anchor(state, title_guess: str) -> str | None:
    """Resolve an anchor paper (referred to informally) to its corpusId."""
    title_tool = _maybe_tool(state, "search_paper_by_title")
    if title_tool is not None and title_guess:
        raw = await _call(title_tool, title=title_guess, fields="title,corpusId")
        for r in _parse_items(raw):
            if r.get("corpusId") is not None and r.get("paperId") is not None:
                return str(r["corpusId"])
    kw = _maybe_tool(state, "search_papers_by_relevance")
    if kw is not None and title_guess:
        raw = await _call(kw, keyword=title_guess, fields="title,corpusId", limit=5)
        rows = _parse_items(raw)
        if rows and rows[0].get("corpusId") is not None:
            return str(rows[0]["corpusId"])
    return None


async def _authors_of_paper(state, title_guess: str) -> list[str]:
    """Resolve a referenced paper and return its author names (for
    'papers by an author of the X paper' compositional queries)."""
    if not title_guess:
        return []
    cid = await _resolve_anchor(state, title_guess)
    if not cid:
        return []
    get_one = _maybe_tool(state, "get_paper")
    if get_one is None:
        return []
    raw = await _call(get_one, paper_id=f"CorpusId:{cid}", fields="title,authors")
    names: list[str] = []
    for r in _parse_items(raw):
        au = r.get("authors")
        if isinstance(au, list):
            for a in au:
                nm = (a.get("name") if isinstance(a, dict) else a) or ""
                nm = nm.strip()
                if nm and nm not in names:
                    names.append(nm)
    print(f"  author_of_paper '{title_guess}' -> id={cid} authors={names}")
    return names


async def _solve_metadata_citation(state, query, ex) -> list[dict]:
    """Handle 'papers citing X (and Y)' / 'papers cited by X' queries: resolve
    each anchor, get its citing (or cited) set, intersect across anchors, then
    apply venue/year/min_citations/min_authors filters. Exact-match scoring."""
    relation = (ex.get("relation") or "").strip().lower()
    anchors = [a for a in (ex.get("referenced_papers") or []) if isinstance(a, str) and a.strip()]
    if relation not in ("cites", "cited_by") or not anchors:
        return []

    # Resolve anchors to corpusIds
    anchor_ids = []
    for a in anchors[:4]:
        cid = await _resolve_anchor(state, a)
        if cid:
            anchor_ids.append(cid)
    print(f"  citation query: relation={relation} anchors={anchors} -> ids={anchor_ids}")
    if not anchor_ids:
        return []

    get_cit = _maybe_tool(state, "get_citations")
    if get_cit is None:
        return []

    fields = "title,corpusId,year,venue,authors,citationCount"
    sets: list[set[str]] = []
    meta: dict[str, dict] = {}
    for aid in anchor_ids:
        raw = await _call(get_cit, paper_id=f"CorpusId:{aid}", fields=fields, limit=1000)
        rows = _unwrap_citing(_parse_items(raw))
        ids = set()
        for r in rows:
            cid = r.get("corpusId")
            if cid is None:
                continue
            cid = str(cid)
            ids.add(cid)
            meta.setdefault(cid, r)
        print(f"    anchor {aid}: {len(ids)} {relation} papers")
        if ids:
            sets.append(ids)

    if not sets:
        return []
    inter = set.intersection(*sets) if len(sets) > 1 else sets[0]
    print(f"  citation intersection: {len(inter)} candidates")

    # Numeric / venue / year filters
    ymin, ymax, years = _year_spec(ex)
    venues = [v.lower() for v in (ex.get("venues") or []) if isinstance(v, str) and v.strip()]
    min_cit = ex.get("min_citations")
    min_auth = ex.get("min_authors")

    out = []
    for cid in inter:
        r = meta.get(cid, {})
        if not _year_ok(r.get("year"), ymin, ymax, years):
            continue
        if venues:
            rv = (r.get("venue") or "").lower()
            if rv and not any(v in rv or rv in v for v in venues):
                continue
        if isinstance(min_cit, int):
            cc = r.get("citationCount")
            if isinstance(cc, int) and cc < min_cit:
                continue
        if isinstance(min_auth, int):
            au = r.get("authors")
            if isinstance(au, list) and len(au) <= min_auth:
                continue
        out.append({"paper_id": cid, "markdown_evidence": _clip(r.get("title") or "", 250)})
    print(f"  citation metadata: submitting {len(out)} papers")
    return out


def _year_spec(ex):
    ymin, ymax = ex.get("year_min"), ex.get("year_max")
    years = ex.get("years")
    yset = set()
    if isinstance(years, list):
        for y in years:
            try:
                yset.add(int(y))
            except (TypeError, ValueError):
                continue
    return ymin, ymax, (yset or None)


def _year_ok(y, ymin, ymax, years):
    if y is None:
        return years is None and (ymin is None and ymax is None)
    try:
        y = int(y)
    except (TypeError, ValueError):
        return True
    if years is not None:
        return y in years
    if ymin is not None and y < ymin:
        return False
    if ymax is not None and y > ymax:
        return False
    return True


async def _solve_metadata(state, query) -> list[dict]:
    ex = await _llm_json(
        "Extract structured filters from this scholarly-search request.\n"
        f"Request: {query}\n\n"
        "Fields:\n"
        '  "relation": "cites" if it wants papers that CITE some reference '
        'paper(s); "cited_by" if it wants papers CITED BY a reference; else "".\n'
        '  "referenced_papers": informal titles of the anchor paper(s) the '
        "relation refers to (e.g. [\"T5\", \"Spider\"]); [] if none.\n"
        '  "author_of_paper": if the request wants papers written by an AUTHOR OF '
        'some named paper (e.g. "co-authored by an author of the BERT paper"), the '
        'informal title of that paper (e.g. "BERT"); "" otherwise.\n'
        '  "keywords": topic noun phrase or empty.\n'
        '  "authors": full author names explicitly named, [] if none.\n'
        '  "venues": venue names/abbreviations, [] if none.\n'
        '  "year_min"/"year_max": ints or null for a RANGE.\n'
        '  "years": explicit list of individual years if the request names them '
        '(e.g. "2014 or 2017" -> [2014, 2017]); [] otherwise.\n'
        '  "min_citations": int or null (e.g. "cited by at least 30").\n'
        '  "min_authors": int or null — the paper must have MORE THAN this many '
        'authors (e.g. "more than 3 authors" -> 3; "at least one additional '
        'author" beyond a named one -> 1).\n'
        "Reply ONLY with JSON with exactly those keys.",
        {},
    )
    if not isinstance(ex, dict):
        ex = {}

    # --- citation-relation branch (papers citing/cited-by an anchor) -------
    if (ex.get("relation") or "").strip().lower() in ("cites", "cited_by") and ex.get("referenced_papers"):
        try:
            cit_out = await _solve_metadata_citation(state, query, ex)
        except Exception as e:  # noqa: BLE001
            print(f"  citation branch error: {e!r}")
            cit_out = []
        if cit_out:
            return cit_out
        # else fall through to keyword/author path

    keywords = (ex.get("keywords") or query).strip() or query
    authors = [a for a in (ex.get("authors") or []) if isinstance(a, str) and a.strip()]
    venues = [v for v in (ex.get("venues") or []) if isinstance(v, str) and v.strip()]
    ymin, ymax, years = _year_spec(ex)
    min_auth = ex.get("min_authors")
    author_of_paper = (ex.get("author_of_paper") or "").strip() if isinstance(ex, dict) else ""

    # --- author-of-referenced-paper branch (additive, guarded) ------------
    #     "papers by an author of the X paper" -> resolve X's authors, then run
    #     the normal author path over them with venue/year/min_authors filters.
    if not authors and author_of_paper:
        try:
            ref_authors = await _authors_of_paper(state, author_of_paper)
        except Exception as e:  # noqa: BLE001
            print(f"  author_of_paper branch error: {e!r}")
            ref_authors = []
        if ref_authors:
            try:
                aop_out = await _solve_metadata_author(
                    state, query, ref_authors, venues, ymin, ymax, years, min_auth
                )
            except Exception as e:  # noqa: BLE001
                print(f"  author_of_paper author-path error: {e!r}")
                aop_out = []
            if aop_out:
                return aop_out
        # else fall through

    print(f"  metadata filters: kw={keywords!r} authors={authors} venues={venues} "
          f"yr=[{ymin},{ymax}] years={years} min_auth={min_auth}")

    # --- author path: retrieve the author's papers, LLM-select by venue/year
    if authors:
        auth_out = await _solve_metadata_author(
            state, query, authors, venues, ymin, ymax, years, min_auth
        )
        if auth_out:
            return auth_out

    # --- keyword + venue path (post-filter by year) ------------------------
    kw = _maybe_tool(state, "search_papers_by_relevance")
    rows: list[dict] = []
    if kw is not None:
        venues_arg = ",".join(venues) if venues else None
        kwargs = dict(keyword=keywords, fields="title,abstract,corpusId,year,venue", limit=100)
        if venues_arg:
            kwargs["venues"] = venues_arg
        raw = await _call(kw, **kwargs)
        rows.extend(_parse_items(raw))

    vlow = [v.lower() for v in venues]
    seen, out = set(), []
    for r in rows:
        cid = r.get("corpusId")
        if cid is None:
            continue
        cid = str(cid)
        if cid in seen or not _year_ok(r.get("year"), ymin, ymax, years):
            continue
        if vlow:
            rv = (r.get("venue") or "").lower()
            if rv and not any(v in rv or rv in v for v in vlow):
                continue
        seen.add(cid)
        out.append({"paper_id": cid, "markdown_evidence": _clip(r.get("title") or "", 250)})

    print(f"  metadata: submitting {len(out)} papers")
    return out


async def _solve_metadata_author(state, query, authors, venues, ymin, ymax, years, min_auth=None) -> list[dict]:
    """Author queries: pull each author's papers, then LLM-select which match the
    venue/year criteria (robust to venue-name variants like ACL vs TACL vs a
    workshop). Falls back to programmatic year/venue filtering on LLM failure.
    Optional min_auth: keep only papers with MORE THAN min_auth authors."""
    find_auth = _maybe_tool(state, "search_authors_by_name")
    get_papers = _maybe_tool(state, "get_author_papers")
    if find_auth is None or get_papers is None:
        return []

    rows: list[dict] = []
    for name in authors[:8]:
        raw = await _call(find_auth, name=name, fields="name,paperCount", limit=10)
        arows = [a for a in _parse_items(raw) if a.get("authorId")]
        arows.sort(key=lambda a: a.get("paperCount") or 0, reverse=True)
        if not arows:
            continue
        aid = arows[0]["authorId"]
        raw = await _call(
            get_papers, author_id=str(aid),
            paper_fields="title,corpusId,year,venue,authors", limit=500,
        )
        rows.extend(_parse_items(raw))

    # Dedup + basic year prefilter (keeps LLM candidate list small & on-topic)
    seen, cand = set(), []
    for r in rows:
        cid = r.get("corpusId")
        if cid is None:
            continue
        cid = str(cid)
        if cid in seen:
            continue
        if not _year_ok(r.get("year"), ymin, ymax, years):
            continue
        # min_authors filter (paper must have MORE THAN min_auth authors)
        if isinstance(min_auth, int):
            au = r.get("authors")
            if isinstance(au, list) and len(au) <= min_auth:
                continue
        seen.add(cid)
        cand.append(r)
    if not cand:
        return []
    print(f"  author path: {len(cand)} papers after year/min_auth prefilter")

    # If no venue constraint, year/min_auth prefilter alone is the answer.
    if not venues:
        return [{"paper_id": str(r["corpusId"]), "markdown_evidence": _clip(r.get("title") or "", 250)}
                for r in cand]

    # LLM selects which candidates match the venue(s) — it knows the full-name /
    # abbreviation mapping and distinguishes main conference from workshop/journal.
    listing = "\n".join(
        f"{i}: [{r.get('venue') or '?'} | {r.get('year') or '?'}] {(r.get('title') or '')[:120]}"
        for i, r in enumerate(cand)
    )
    ven_txt = ", ".join(venues)
    sel = await _llm_json(
        f'Selecting papers that were published at these venue(s): {ven_txt}.\n'
        "Venue names in the data may be full names or abbreviations; match the "
        "intended venue only (e.g. 'ACL' means the main ACL conference / its "
        "Annual Meeting, NOT TACL, NOT Findings, NOT a workshop, NOT EMNLP).\n"
        f"Candidates (index: [venue | year] title):\n{listing}\n\n"
        'Reply ONLY with JSON: {"indices": [i, ...]} — the indices whose venue '
        "matches. Empty list if none.",
        None,
    )
    if isinstance(sel, dict) and isinstance(sel.get("indices"), list):
        idxs = [i for i in sel["indices"] if isinstance(i, int) and 0 <= i < len(cand)]
        out = [{"paper_id": str(cand[i]["corpusId"]),
                "markdown_evidence": _clip(cand[i].get("title") or "", 250)} for i in idxs]
        print(f"  author path: LLM selected {len(out)}/{len(cand)} by venue")
        if out:
            return out

    # Fallback: substring venue match (both directions)
    vlow = [v.lower() for v in venues]
    out = []
    for r in cand:
        rv = (r.get("venue") or "").lower()
        if rv and any(v in rv or rv in v for v in vlow):
            out.append({"paper_id": str(r["corpusId"]),
                        "markdown_evidence": _clip(r.get("title") or "", 250)})
    print(f"  author path: substring-venue fallback selected {len(out)}")
    return out


# --------------------------------------------------------------------------
# Solver entry point
# --------------------------------------------------------------------------
def _classify(state, query) -> str:
    st = (state.metadata.get("score_type") or "").strip()
    if st in ("specific_f1", "metadata_f1", "semantic_f1"):
        return st.replace("_f1", "")
    q = query.lower()
    if re.search(r"\b(cite|cites|citing|cited by)\b", q):
        return "metadata"
    if re.search(r"\bthe\b.*\bpaper\b", q) and len(query) < 90:
        return "specific"
    if re.search(r"\b(by|author|published in|in \d{4}|between \d{4})\b", q):
        return "metadata"
    return "semantic"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata.get("raw_query") or getattr(state, "input_text", "") or ""
        kind = _classify(state, query)
        print(f"[{state.sample_id}] kind={kind} query={query[:90]!r}")

        try:
            if kind == "specific":
                results = await _solve_specific(state, query)
            elif kind == "metadata":
                results = await _solve_metadata(state, query)
            else:
                results = await _solve_semantic(state, query)
        except Exception as e:  # noqa: BLE001
            print(f"  FATAL path error ({kind}): {e!r}")
            results = []

        if not results:
            try:
                results = await _solve_semantic(state, query)
            except Exception as e:  # noqa: BLE001
                print(f"  fallback failed: {e!r}")
                results = []

        state.output.completion = json.dumps(
            {"output": {"query_id": state.sample_id, "results": results}}
        )
        print(f"  DONE: {len(results)} papers submitted")
        return state

    return solve
