"""broad-recall-router: PaperFindingBench solver, iteration 2.

Routes on score_type:
  - semantic_f1: multi-query broad retrieval (4 keyword variants + snippet
    search), LLM-graded candidate pool, GPT_5_4-refined head ordering,
    submit up to ~120 ranked papers with rich verbatim evidence.
  - specific_f1: LLM paper identification -> title search -> 1-3 papers.
  - metadata_f1: LLM constraint plan -> citations/author/keyword base set
    -> Python post-filters -> submit the passing set.

Design notes (see reasoning.md): the seed's failure was recall starvation
(8 papers vs K up to 222 estimated relevant); the rank term only punishes
misordering, so a long well-ordered list is nearly free upside. A short
low-relevance tail hedges the all-grades-equal => rank=0 quirk.
"""

import asyncio
import difflib
import json
import re

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI

# ---------------------------------------------------------------- helpers

MAX_SUBMIT_SEMANTIC = 120
POOL_CAP = 200
GRADE_CHUNK = 40
REFINE_TOP = 30
PAPER_FIELDS = "title,abstract,corpusId,tldr,year,venue"
META_FIELDS = "title,abstract,corpusId,year,venue,journal,authors,citationCount"


def _get_tool(state: TaskState, name: str):
    by_name = {ToolDef(t).name: t for t in state.tools}
    if name not in by_name:
        raise RuntimeError(f"{name!r} not in state.tools (have: {sorted(by_name)})")
    return by_name[name]


def _parse_items(raw) -> list[dict]:
    """Flatten MCP ContentText items; unwrap {"data": [...]} wrappers."""
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
            data = doc["data"]
            if isinstance(data, list):
                docs.extend(d for d in data if isinstance(d, dict))
        elif isinstance(doc, dict):
            docs.append(doc)
    return docs


def _cid(doc: dict) -> str:
    """Corpus id as a clean string ('' if absent)."""
    v = doc.get("corpusId")
    if v is None:
        v = doc.get("corpusid")
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower().startswith("corpusid:"):
        s = s.split(":", 1)[1]
    return s


def _tldr_text(doc: dict) -> str:
    t = doc.get("tldr")
    if isinstance(t, dict):
        return (t.get("text") or "").strip()
    if isinstance(t, str):
        return t.strip()
    return ""


def _cut(text: str, n: int) -> str:
    """Truncate at a whitespace boundary; the result stays a verbatim substring."""
    text = text or ""
    if len(text) <= n:
        return text
    cut = text[:n]
    sp = cut.rfind(" ")
    return cut[: sp if sp > n // 2 else n]


def _json_block(text: str):
    """Extract the first JSON object/array from an LLM completion."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        break
    return None


async def _gen(model, prompt: str, retries: int = 1) -> str:
    """Generate with an empty-completion retry; never raises on empty."""
    for attempt in range(retries + 1):
        try:
            resp = await model.generate(prompt)
            text = (resp.completion or "").strip()
            if text:
                return text
            print(f"  [gen] empty completion (attempt {attempt + 1})")
        except Exception as e:  # noqa: BLE001 - surface but keep going
            print(f"  [gen] error (attempt {attempt + 1}): {e!r}")
    return ""


async def _safe_tool(coro, label: str):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        print(f"  [tool:{label}] failed: {e!r}")
        return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _title_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _evidence(doc: dict) -> str:
    """Verbatim passages (title, tldr, abstract, snippets) joined by ' ... '."""
    passages = []
    title = (doc.get("title") or "").strip()
    if title:
        passages.append(title)
    tldr = _tldr_text(doc)
    if tldr:
        passages.append(tldr)
    abstract = (doc.get("abstract") or "").strip()
    if abstract:
        passages.append(_cut(abstract, 1500))
    for sn in (doc.get("_snippets") or [])[:2]:
        sn = (sn or "").strip()
        if sn:
            passages.append(_cut(sn, 800))
    return " ... ".join(passages[:8])


def _submit(state: TaskState, results: list[dict]) -> TaskState:
    state.output.completion = json.dumps(
        {"output": {"query_id": state.sample_id, "results": results[:250]}}
    )
    print(f"  submitted {min(len(results), 250)} papers")
    return state


# ------------------------------------------------------------ semantic path


async def _semantic_queries(query: str) -> tuple[list[str], str]:
    """One cheap call -> 4 keyword variants + a natural-language snippet query."""
    prompt = (
        "A user wants scholarly papers matching this request:\n"
        f"{query}\n\n"
        "Produce search inputs for a literal keyword-matching paper search "
        "engine (no operators; noun phrases only; questions return zero hits).\n"
        'Reply with JSON only: {"keyword_queries": ["...", "...", "...", "..."], '
        '"snippet_query": "..."}\n'
        "- keyword_queries: 4 DIVERSE 3-8 word noun-phrase queries covering "
        "different phrasings, synonyms, and sub-aspects of the request.\n"
        "- snippet_query: one full sentence restating what the papers should show."
    )
    obj = _json_block(await _gen(GPT_5_4_MINI, prompt))
    kws, snip = [], ""
    if isinstance(obj, dict):
        kws = [k for k in obj.get("keyword_queries") or [] if isinstance(k, str) and k.strip()]
        snip = obj.get("snippet_query") or ""
    if not kws:
        # crude fallback: strip interrogative framing
        kws = [re.sub(r"[^\w\s-]", " ", query)[:120]]
    return kws[:4], (snip or query)


async def _grade_chunk(query: str, chunk: list[tuple[int, str]]) -> dict[int, int]:
    lines = "\n".join(f"{i}. {t}" for i, t in chunk)
    prompt = (
        "You are grading candidate papers for a scholarly literature request.\n"
        f"Request: {query}\n\n"
        "Grade each candidate:\n"
        "3 = clearly satisfies EVERY requirement/aspect of the request\n"
        "2 = satisfies the main requirements but one aspect is unclear\n"
        "1 = related topic but misses a key requirement\n"
        "0 = not relevant\n"
        "Be strict: reserve 3 for papers whose text demonstrably matches every "
        "stated requirement.\n\n"
        f"Candidates:\n{lines}\n\n"
        "Output exactly one line per candidate: 'index: grade'. Nothing else."
    )
    text = await _gen(GPT_5_4_MINI, prompt)
    grades: dict[int, int] = {}
    for m in re.finditer(r"^\s*(\d+)\s*[:.\-]\s*([0-3])\b", text, re.MULTILINE):
        idx = int(m.group(1))
        grades[idx] = int(m.group(2))
    if not grades:
        # a dead chunk should not silently drop the pool's head
        print(f"  [grade] chunk of {len(chunk)} unparsed; defaulting to grade 1")
        return {i: 1 for i, _ in chunk}
    return grades


async def _refine_order(query: str, ranked: list[dict]) -> list[dict]:
    """Reorder the head of the list with the stronger model."""
    head = ranked[:REFINE_TOP]
    if len(head) < 3:
        return ranked
    lines = []
    for i, d in enumerate(head):
        body = _tldr_text(d) or (d.get("abstract") or "")
        lines.append(f"{i}. {(d.get('title') or '')[:150]} — {_cut(body, 500)}")
    prompt = (
        "Order these candidate papers for the request below, best first. "
        "Best = most completely satisfies every stated requirement of the "
        "request.\n"
        f"Request: {query}\n\n"
        "Candidates:\n" + "\n".join(lines) + "\n\n"
        "Reply with the comma-separated indices only, best first, all of them."
    )
    text = await _gen(GPT_5_4, prompt)
    order = []
    for tok in re.findall(r"\d+", text):
        i = int(tok)
        if 0 <= i < len(head) and i not in order:
            order.append(i)
    if len(order) < len(head) // 2:
        return ranked  # refinement failed; keep grade order
    order += [i for i in range(len(head)) if i not in order]
    return [head[i] for i in order] + ranked[REFINE_TOP:]


async def _solve_semantic(state: TaskState, query: str) -> TaskState:
    search = _get_tool(state, "search_papers_by_relevance")
    snippet = _get_tool(state, "snippet_search")
    batch = _get_tool(state, "get_paper_batch")

    kws, snip_q = await _semantic_queries(query)
    print(f"  keyword queries: {kws}")

    tasks = [
        _safe_tool(search(keyword=k, fields=PAPER_FIELDS, limit=60), f"rel[{k[:30]}]")
        for k in kws
    ]
    tasks.append(_safe_tool(snippet(query=snip_q, limit=20), "snippet"))
    raws = await asyncio.gather(*tasks)

    result_lists = [_parse_items(r) for r in raws[:-1]]
    snip_items = _parse_items(raws[-1])

    # snippet entries -> paper docs (in score order), snippets attached
    snip_docs: dict[str, dict] = {}
    snip_order: list[dict] = []
    for entry in snip_items:
        paper = entry.get("paper") or {}
        cid = _cid(paper)
        if not cid:
            continue
        text = ((entry.get("snippet") or {}).get("text") or "").strip()
        if cid not in snip_docs:
            doc = {"corpusId": cid, "title": paper.get("title"), "_snippets": []}
            snip_docs[cid] = doc
            snip_order.append(doc)
        if text and len(snip_docs[cid]["_snippets"]) < 2:
            snip_docs[cid]["_snippets"].append(text)

    # round-robin merge across sources, dedupe, cap pool
    pool: dict[str, dict] = {}
    ordered: list[dict] = []
    lists = [lst for lst in result_lists if lst] + ([snip_order] if snip_order else [])
    for rank in range(max((len(l) for l in lists), default=0)):
        for lst in lists:
            if rank >= len(lst) or len(ordered) >= POOL_CAP:
                continue
            doc = lst[rank]
            cid = _cid(doc)
            if not cid:
                continue
            if cid in pool:
                if doc.get("_snippets"):
                    pool[cid].setdefault("_snippets", []).extend(doc["_snippets"][:2])
                continue
            pool[cid] = doc
            ordered.append(doc)
    print(f"  candidate pool: {len(ordered)} "
          f"(per-source: {[len(l) for l in lists]})")

    if not ordered:
        return _submit(state, [])

    # enrich snippet-only docs (and any missing abstracts) via free batch fetch
    missing = [d for d in ordered if not d.get("abstract")]
    for i in range(0, len(missing), 50):
        grp = missing[i : i + 50]
        raw = await _safe_tool(
            batch(ids=[f"CorpusId:{_cid(d)}" for d in grp],
                  fields="title,abstract,corpusId,tldr"),
            "batch",
        )
        fetched = {_cid(f): f for f in _parse_items(raw or [])}
        for d in grp:
            f = fetched.get(_cid(d))
            if f:
                if not d.get("title"):
                    d["title"] = f.get("title")
                if f.get("abstract"):
                    d["abstract"] = f["abstract"]
                if f.get("tldr") and not d.get("tldr"):
                    d["tldr"] = f["tldr"]

    # grade the pool in concurrent chunks
    entries = []
    for i, d in enumerate(ordered):
        body = (d.get("abstract") or "") or _tldr_text(d) or " ".join(d.get("_snippets") or [])
        entries.append((i, f"{(d.get('title') or '')[:140]} || {_cut(body, 300)}"))
    chunks = [entries[i : i + GRADE_CHUNK] for i in range(0, len(entries), GRADE_CHUNK)]
    grade_maps = await asyncio.gather(*(_grade_chunk(query, c) for c in chunks))
    grades: dict[int, int] = {}
    for g in grade_maps:
        grades.update(g)
    dist = {g: sum(1 for v in grades.values() if v == g) for g in (3, 2, 1, 0)}
    print(f"  grade distribution: {dist}")

    # rank: grade desc, retrieval order asc
    idx_ranked = sorted(range(len(ordered)), key=lambda i: (-grades.get(i, 0), i))
    keep = [i for i in idx_ranked if grades.get(i, 0) >= 1]
    zeros = [i for i in idx_ranked if grades.get(i, 0) == 0]
    if len(keep) < 25 and zeros:  # thin pool: trust retrieval order as padding
        n_pad = 25 - len(keep)
        keep += zeros[:n_pad]
        zeros = zeros[n_pad:]
    ranked = [ordered[i] for i in keep[:MAX_SUBMIT_SEMANTIC]]

    ranked = await _refine_order(query, ranked)

    # tail hedge against the all-grades-equal => rank=0 quirk
    hedge = [ordered[i] for i in zeros[-3:] if ordered[i] not in ranked]
    ranked = ranked[: MAX_SUBMIT_SEMANTIC - len(hedge)] + hedge

    results = [
        {"paper_id": _cid(d), "markdown_evidence": _evidence(d)} for d in ranked if _cid(d)
    ]
    return _submit(state, results)


# ------------------------------------------------------------ specific path


async def _solve_specific(state: TaskState, query: str) -> TaskState:
    title_search = _get_tool(state, "search_paper_by_title")
    rel_search = _get_tool(state, "search_papers_by_relevance")

    prompt = (
        "A user refers to one specific published paper as:\n"
        f'"{query}"\n\n'
        "Identify which paper is meant. Reply with JSON only:\n"
        '{"candidates": [{"title": "<exact full paper title>", "confidence": 0.0}]}\n'
        "List 1-3 DISTINCT candidate papers, most likely first, with your "
        "confidence (0-1) that each is the one meant. Only include alternates "
        "if it is genuinely ambiguous which paper the user means."
    )
    obj = _json_block(await _gen(GPT_5_4, prompt))
    cands = []
    if isinstance(obj, dict):
        for c in obj.get("candidates") or []:
            if isinstance(c, dict) and (c.get("title") or "").strip():
                cands.append((c["title"].strip(), float(c.get("confidence") or 0)))
    if not cands:
        cands = [(query, 0.0)]
    print(f"  identification candidates: {cands}")

    hits: list[dict] = []
    for title, conf in cands[:3]:
        raw = await _safe_tool(
            title_search(title=title, fields="corpusId,title,year,authors"),
            f"title[{title[:30]}]",
        )
        for doc in _parse_items(raw or []):
            if doc.get("paperId") or _cid(doc):
                sim = _title_sim(title, doc.get("title") or "")
                print(f"    match {_cid(doc)} sim={sim:.2f}: {doc.get('title')!r}")
                if sim >= 0.5:
                    hits.append({"doc": doc, "conf": conf})
                break

    if not hits:
        # fallback: relevance search on the best-guess title, take top hit
        raw = await _safe_tool(
            rel_search(keyword=cands[0][0], fields="corpusId,title", limit=5), "rel-fb"
        )
        docs = _parse_items(raw or [])
        if docs:
            hits = [{"doc": docs[0], "conf": 0.0}]

    seen, results = set(), []
    for h in hits:
        cid = _cid(h["doc"])
        if cid and cid not in seen:
            seen.add(cid)
            results.append({"paper_id": cid, "markdown_evidence": ""})
    # confident first candidate -> submit it alone (precision: F1 vs 1-paper gold)
    if results and hits[0]["conf"] >= 0.7:
        results = results[:1]
    return _submit(state, results[:3])


# ------------------------------------------------------------ metadata path

_VENUE_ALIASES = {
    "neurips": "neural information processing systems",
    "nips": "neural information processing systems",
    "icml": "international conference on machine learning",
    "iclr": "international conference on learning representations",
    "acl": "annual meeting of the association for computational linguistics",
    "emnlp": "conference on empirical methods in natural language processing",
    "naacl": "north american chapter of the association for computational linguistics",
    "cvpr": "computer vision and pattern recognition",
    "iccv": "international conference on computer vision",
    "eccv": "european conference on computer vision",
    "aaai": "aaai conference on artificial intelligence",
    "ijcai": "international joint conference on artificial intelligence",
    "sigir": "international acm sigir conference",
    "kdd": "knowledge discovery and data mining",
    "tacl": "transactions of the association for computational linguistics",
    "jmlr": "journal of machine learning research",
    "tpami": "transactions on pattern analysis and machine intelligence",
}


def _venue_ok(doc: dict, wanted: list[str]) -> bool:
    if not wanted:
        return True
    fields = [doc.get("venue") or ""]
    j = doc.get("journal")
    if isinstance(j, dict):
        fields.append(j.get("name") or "")
    pv = _norm(" | ".join(fields))
    if not pv:
        return False
    for w in wanted:
        nw = _norm(w)
        for probe in filter(None, {nw, _VENUE_ALIASES.get(nw, "")}):
            if probe in pv or (len(pv) > 3 and pv in probe):
                return True
    return False


def _author_ok(doc: dict, wanted: list[str]) -> bool:
    if not wanted:
        return True
    names = [_norm(a.get("name") or "") for a in doc.get("authors") or []]
    for w in wanted:
        nw = _norm(w).split()
        if not nw:
            continue
        last = nw[-1]
        for n in names:
            toks = n.split()
            if toks and toks[-1] == last and (len(nw) == 1 or not nw[0] or
                                              (toks and toks[0][:1] == nw[0][:1])):
                return True
    return False


async def _solve_metadata(state: TaskState, query: str) -> TaskState:
    prompt = (
        "Parse this scholarly paper search request into JSON filters.\n"
        f"Request: {query}\n\n"
        "Reply with JSON only:\n"
        "{\n"
        '  "authors": [],            // author names the papers must be written by\n'
        '  "venues": [],             // venue names incl. BOTH abbreviation and full name, e.g. ["NeurIPS", "Neural Information Processing Systems"]\n'
        '  "year_min": null, "year_max": null,\n'
        '  "cites_paper_title": null, // if papers must CITE some paper X, the best-known exact title of X\n'
        '  "min_citations": null,     // minimum citation count required of each result\n'
        '  "min_authors": null, "max_authors": null, // bounds on number of authors per paper\n'
        '  "topic_keywords": null     // 3-6 word topical keyword phrase if the request has a topic constraint\n'
        "}\n"
        "Use null/[] for unconstrained fields."
    )
    plan = _json_block(await _gen(GPT_5_4, prompt)) or {}
    print(f"  metadata plan: {json.dumps(plan)[:300]}")

    def _num(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    authors = [a for a in plan.get("authors") or [] if isinstance(a, str)]
    venues = [v for v in plan.get("venues") or [] if isinstance(v, str)]
    y0, y1 = _num(plan.get("year_min")), _num(plan.get("year_max"))
    cites_title = plan.get("cites_paper_title")
    min_cit = _num(plan.get("min_citations"))
    min_auth, max_auth = _num(plan.get("min_authors")), _num(plan.get("max_authors"))
    topic = plan.get("topic_keywords")
    if not isinstance(cites_title, str) or not cites_title.strip():
        cites_title = None
    if not isinstance(topic, str) or not topic.strip():
        topic = None

    candidates: list[dict] = []
    author_base = False

    if cites_title:
        title_search = _get_tool(state, "search_paper_by_title")
        raw = await _safe_tool(
            title_search(title=cites_title, fields="corpusId,title"), "cite-title"
        )
        target = next((d for d in _parse_items(raw or []) if d.get("paperId") or _cid(d)), None)
        if target:
            print(f"  cited paper: {_cid(target)} {target.get('title')!r}")
            get_cit = _get_tool(state, "get_citations")
            target_id = target.get("paperId") or f"CorpusId:{_cid(target)}"
            raw = await _safe_tool(
                get_cit(paper_id=target_id, fields=META_FIELDS, limit=1000),
                "citations",
            )
            for item in _parse_items(raw or []):
                doc = item.get("citingPaper") if isinstance(item.get("citingPaper"), dict) else item
                if _cid(doc):
                    candidates.append(doc)
    elif authors:
        author_base = True
        find_auth = _get_tool(state, "search_authors_by_name")
        get_papers = _get_tool(state, "get_author_papers")
        ids: list[str] = []
        for name in authors[:3]:
            raw = await _safe_tool(
                find_auth(name=name, fields="authorId,name,paperCount", limit=20),
                f"auth[{name}]",
            )
            recs = [r for r in _parse_items(raw or []) if r.get("authorId")]
            recs.sort(key=lambda r: -(r.get("paperCount") or 0))
            # keep all plausible identities of the same person (split profiles)
            ids.extend(str(r["authorId"]) for r in recs[:4])
        for aid in ids[:8]:
            raw = await _safe_tool(
                get_papers(author_id=aid, paper_fields=META_FIELDS, limit=100),
                f"papers[{aid}]",
            )
            candidates.extend(d for d in _parse_items(raw or []) if _cid(d))
    if not candidates:
        rel = _get_tool(state, "search_papers_by_relevance")
        kw = topic or " ".join(authors) or query[:100]
        kwargs = {"keyword": kw, "fields": META_FIELDS, "limit": 100}
        if venues:
            kwargs["venues"] = ",".join(venues)
        raw = await _safe_tool(rel(**kwargs), "kw-base")
        candidates = [d for d in _parse_items(raw or []) if _cid(d)]
        if not candidates and venues:  # server-side venue name mismatch
            kwargs.pop("venues")
            raw = await _safe_tool(rel(**kwargs), "kw-base-novenue")
            candidates = [d for d in _parse_items(raw or []) if _cid(d)]

    # dedupe + hard post-filters
    seen: set[str] = set()
    kept: list[dict] = []
    for d in candidates:
        cid = _cid(d)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        year = d.get("year")
        if y0 and (not year or year < y0):
            continue
        if y1 and (not year or year > y1):
            continue
        if not _venue_ok(d, venues):
            continue
        if min_cit and (d.get("citationCount") or 0) < min_cit:
            continue
        n_auth = len(d.get("authors") or [])
        if min_auth and n_auth < min_auth:
            continue
        if max_auth and n_auth > max_auth:
            continue
        if not author_base and not _author_ok(d, authors):
            continue
        kept.append(d)
    print(f"  metadata: {len(candidates)} candidates -> {len(kept)} after filters")

    # optional topical filter (cheap LLM) when the base wasn't a topic search
    if topic and kept and (cites_title or author_base) and len(kept) <= 200:
        entries = [
            (i, f"{(d.get('title') or '')[:140]} || {_cut(d.get('abstract') or '', 200)}")
            for i, d in enumerate(kept)
        ]
        keep_idx: set[int] = set()
        for i in range(0, len(entries), GRADE_CHUNK):
            chunk = entries[i : i + GRADE_CHUNK]
            lines = "\n".join(f"{j}. {t}" for j, t in chunk)
            text = await _gen(
                GPT_5_4_MINI,
                f"Topic constraint: {topic}\nRequest: {query}\n\n"
                f"Which of these papers match the topic constraint?\n{lines}\n\n"
                "Reply with the comma-separated indices of MATCHING papers only "
                "(empty if none).",
            )
            keep_idx.update(
                int(t) for t in re.findall(r"\d+", text) if int(t) < len(kept)
            )
        if keep_idx:
            kept = [d for i, d in enumerate(kept) if i in keep_idx]
            print(f"  after topic filter: {len(kept)}")

    results = [{"paper_id": _cid(d), "markdown_evidence": ""} for d in kept[:250]]
    return _submit(state, results)


# ---------------------------------------------------------------- solver


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata["raw_query"]
        score_type = state.metadata.get("score_type", "")
        print(f"[{state.sample_id}] score_type={score_type} query={query[:100]!r}")

        try:
            if score_type == "specific_f1":
                return await _solve_specific(state, query)
            if score_type == "metadata_f1":
                return await _solve_metadata(state, query)
            return await _solve_semantic(state, query)
        except Exception as e:  # noqa: BLE001 - never crash a query to 0
            print(f"  [FALLBACK] route failed: {e!r}")
            try:
                search = _get_tool(state, "search_papers_by_relevance")
                kw = re.sub(r"[^\w\s-]", " ", query)[:100]
                raw = await _safe_tool(
                    search(keyword=kw, fields=PAPER_FIELDS, limit=30), "fallback"
                )
                docs = [d for d in _parse_items(raw or []) if _cid(d)]
                ev = "" if score_type in ("specific_f1", "metadata_f1") else None
                results = [
                    {
                        "paper_id": _cid(d),
                        "markdown_evidence": ev if ev is not None else _evidence(d),
                    }
                    for d in docs[:20]
                ]
                return _submit(state, results)
            except Exception as e2:  # noqa: BLE001
                print(f"  [FALLBACK] also failed: {e2!r}")
                return _submit(state, [])

    return solve
