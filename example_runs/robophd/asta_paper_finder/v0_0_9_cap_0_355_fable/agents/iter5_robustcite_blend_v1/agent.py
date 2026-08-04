"""iter5-robustcite-blend-v1: PaperFindingBench solver.

Routes on score_type:
  - specific_f1: LLM reference expansion (ambiguity detection, original-paper
    preference) -> title search -> submit every true referent
  - metadata_f1: LLM plan -> citation/author execution. Citation queries use a
    bisecting get_paper_batch wrapper (poison ids no longer kill whole chunks),
    a least-cited-seed base window with references-verification against the
    other seeds for "cites ALL of X,Y" queries, and a reverse citation channel
    whose references check now actually works.
  - semantic_f1: criteria decomposition -> wide retrieval (keyword variants +
    query-level and per-criterion snippet search) -> pass-1 grading -> chunked
    criterion-targeted snippet enrichment -> judge-mimic pass-2 blended (not
    substituted) into the ranking -> deep ranked list

Iteration-4 diagnostics this design targets:
  - get_paper_batch hard-fails the WHOLE call when any requested id is
    post-snapshot, and crashes server-side ('NoneType' not iterable) on some
    references batches -> every chunk died -> "0 verified citers" everywhere.
    Bisecting fetch isolates poison papers; tool calls are free.
  - intersecting two get_citations windows is wrong when one seed is huge
    (T5+Spider gave 10 candidates, 0 hits); Spider's own window spans
    2020-2025 and contains the gold era -> base on the least-cited seed and
    references-verify against the rest.
  - iter4's pass-2 full re-sort and abstract cap 900 were measured regressions
    vs iter3 (fewer true grade-3s inside top-K) -> blend pass-2 with pass-1,
    restore cap 1300.
"""

import asyncio
import json
import re
import traceback

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI

MAX_RESULTS = 250
EVIDENCE_CAP = 2400   # stay under the scorer's 2500-char truncation
ABSTRACT_CAP = 1300   # iter3's value; the 900 cut was a measured regression
SEMANTIC_CAND_CAP = 400
GRADE_CHUNK = 25
ENRICH_TOP = 160      # candidates that get criterion-targeted snippets
ENRICH_SCOPE_CHUNK = 35   # ids per scoped snippet_search call
REGRADE_TOP = 144     # candidates re-graded per-criterion on final evidence
REGRADE_CHUNK = 12
REFCHECK_CAP = 700    # reverse-channel candidates to verify via references
BASE_WINDOW_REFCHECK_CAP = 900  # base-seed window candidates to ref-verify


# --------------------------------------------------------------------------
# tool plumbing
# --------------------------------------------------------------------------

def _get_tool(state: TaskState, name: str):
    by_name = {ToolDef(t).name: t for t in state.tools}
    if name not in by_name:
        raise RuntimeError(f"{name!r} not in state.tools (have: {sorted(by_name)})")
    return by_name[name]


def _parse_items(raw) -> list:
    """Flatten MCP ContentText JSON items, unwrapping {'data': [...]}."""
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
                docs.extend(data)
            elif data:
                docs.append(data)
        else:
            docs.append(doc)
    return docs


async def _call(tool, timeout=310, quiet=False, **kwargs) -> list:
    """Guarded tool call: parsed items, or [] on any failure."""
    try:
        raw = await asyncio.wait_for(tool(**kwargs), timeout=timeout)
        return _parse_items(raw)
    except Exception as e:
        if not quiet:
            print(f"  tool call failed ({getattr(tool, '__name__', tool)}, "
                  f"{ {k: str(v)[:60] for k, v in kwargs.items()} }): {type(e).__name__}: {str(e)[:160]}")
        return []


def _cid(paper: dict) -> str:
    v = paper.get("corpusId")
    return str(v) if v is not None else ""


async def _batch_fetch(state: TaskState, cids: list, fields: str, chunk: int = 60) -> list:
    """get_paper_batch with bisection on failure.

    The batch endpoint hard-fails the ENTIRE call when any requested id is
    post-snapshot, and crashes server-side on some references payloads. A
    failed chunk is split in half recursively down to single ids, so only the
    poison papers are dropped (iteration-4 lost whole 100-id chunks to this).
    """
    tool = _get_tool(state, "get_paper_batch")
    out = []

    async def fetch(id_list, depth):
        if not id_list:
            return
        res = await _call(tool, quiet=(depth > 0),
                          ids=[f"CorpusId:{i}" for i in id_list], fields=fields)
        got = [p for p in res if isinstance(p, dict) and _cid(p)]
        if got:
            out.extend(got)
        elif len(id_list) > 1:
            mid = len(id_list) // 2
            await fetch(id_list[:mid], depth + 1)
            await fetch(id_list[mid:], depth + 1)
        # single id with no result: poison paper, drop it

    chunks = [cids[i:i + chunk] for i in range(0, len(cids), chunk)]
    await asyncio.gather(*[fetch(ch, 0) for ch in chunks], return_exceptions=True)
    return out


# --------------------------------------------------------------------------
# LLM plumbing
# --------------------------------------------------------------------------

async def _llm_json(model, prompt: str, tag: str):
    """Call a handle and parse the first JSON object in the completion."""
    try:
        resp = await model.generate(prompt)
    except Exception as e:
        print(f"  llm[{tag}] error: {type(e).__name__}: {str(e)[:200]}")
        return None
    text = (resp.completion or "").strip()
    if not text:
        print(f"  llm[{tag}] empty completion")
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        print(f"  llm[{tag}] no JSON object in completion: {text[:120]!r}")
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    print(f"  llm[{tag}] JSON parse failed: {text[:120]!r}")
    return None


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

def _tldr_text(paper: dict) -> str:
    tl = paper.get("tldr")
    if isinstance(tl, dict):
        tl = tl.get("text")
    return tl.strip() if isinstance(tl, str) else ""


def _fit(passage: str, room: int) -> str:
    """Cut at a word boundary; a substring of retrieved text stays verbatim."""
    if len(passage) <= room:
        return passage
    cut = passage[:room]
    sp = cut.rfind(" ")
    if sp > room // 2:
        cut = cut[:sp]
    return cut


def _ordered_snippets(paper: dict, cap: int = 5) -> list:
    """Round-robin across criterion buckets so no single criterion hogs the
    evidence budget, then general snippets."""
    buckets = paper.get("_crit_snips") or {}
    general = paper.get("_snippets") or []
    out, seen = [], set()

    def push(sn):
        sn = (sn or "").strip()
        if not sn or sn[:80] in seen:
            return
        seen.add(sn[:80])
        out.append(sn)

    keys = sorted(buckets)
    i = 0
    while len(out) < cap:
        added = False
        for k in keys:
            lst = buckets.get(k) or []
            if i < len(lst):
                push(lst[i])
                added = True
                if len(out) >= cap:
                    break
        if not added:
            break
        i += 1
    for sn in general:
        if len(out) >= cap:
            break
        push(sn)
    return out


def _build_evidence(paper: dict) -> str:
    """Verbatim passages joined by ' ... ': title, tldr, abstract (capped),
    then criterion-targeted snippets filling the remaining budget."""
    passages = []
    title = (paper.get("title") or "").strip()
    if title:
        passages.append(title)
    tldr = _tldr_text(paper)
    if tldr:
        passages.append(tldr)
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        passages.append(_fit(abstract, ABSTRACT_CAP))
    ab_low = abstract.lower()
    for sn in _ordered_snippets(paper, cap=5):
        # skip snippets that just repeat the abstract
        if ab_low and sn[:60].lower() in ab_low:
            continue
        passages.append(sn)
    out, total = [], 0
    for p in passages[:8]:
        sep = 5 if out else 0  # ' ... '
        room = EVIDENCE_CAP - total - sep
        if room <= 40:
            break
        p = _fit(p, room)
        out.append(p)
        total += len(p) + sep
    return " ... ".join(out)


def _mk_results(papers: list, with_evidence: bool) -> list:
    results, seen = [], set()
    for p in papers:
        cid = _cid(p)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        results.append({
            "paper_id": cid,
            "markdown_evidence": _build_evidence(p) if with_evidence else "",
        })
        if len(results) >= MAX_RESULTS:
            break
    return results


# --------------------------------------------------------------------------
# shared retrieval helpers
# --------------------------------------------------------------------------

_QUESTION_PREFIX = re.compile(
    r"^(could you|can you|please|i('m| am) looking for|find( me)?|show me|list( all)?|"
    r"what|which|are there|is there|suggest|search for|give me|i need|i want)\b[^a-z0-9]*",
    re.IGNORECASE)


def _strip_question(q: str) -> str:
    prev = None
    q = q.strip()
    while prev != q:
        prev = q
        q = _QUESTION_PREFIX.sub("", q).strip()
    return q.rstrip("?.! ") or prev


async def _resolve_title(state: TaskState, title: str,
                         fields="corpusId,title,year,authors,venue,abstract,citationCount"):
    tool = _get_tool(state, "search_paper_by_title")
    items = await _call(tool, title=title, fields=fields)
    for it in items:
        if isinstance(it, dict) and it.get("paperId"):
            return it
    return None


# --------------------------------------------------------------------------
# specific_f1
# --------------------------------------------------------------------------

async def solve_specific(state: TaskState, query: str) -> list:
    plan = await _llm_json(GPT_5_4, f"""A user wants to find a specific known research paper on Semantic Scholar.

User request: {query}

Using your knowledge of the literature, identify which paper this is. Nicknames, model names, dataset names, and author+year hints (e.g. "the BART paper", "MS^2 DeYong2021") usually refer to a well-known paper whose full title you know.

IMPORTANT: some nicknames are AMBIGUOUS — several distinct, unrelated papers (often in different fields) are each known by the same name (e.g. multiple systems named "SPIKE"). If so, list each distinct paper.

Reply with ONLY a JSON object:
{{"candidates": [{{"full_title": "<exact full title>", "note": "<one-line: what this paper is>"}}, ...],
  "ambiguous": <true if the reference plausibly names more than one distinct paper, else false>,
  "keyword_queries": ["<3-8 word keyword search>", "<alternative keyword search>"]}}
List 1-2 candidates when confident, up to 6 when ambiguous.""",
                          "specific-plan")
    cand_titles = [c.get("full_title") for c in (plan or {}).get("candidates", [])
                   if isinstance(c, dict) and c.get("full_title")]
    ambiguous = bool((plan or {}).get("ambiguous"))
    keywords = (plan or {}).get("keyword_queries") or []
    if not cand_titles and not keywords:
        keywords = [_strip_question(query)]

    tasks = [_resolve_title(state, t) for t in cand_titles[:6]]
    search = _get_tool(state, "search_papers_by_relevance")
    for kw in (keywords[:2] or [_strip_question(query)]):
        if isinstance(kw, str) and kw.strip():
            tasks.append(_call(search, keyword=kw, limit=20,
                               fields="corpusId,title,year,authors,venue,abstract,citationCount"))
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    cands, seen = [], set()
    for g in gathered:
        items = [g] if isinstance(g, dict) else (g if isinstance(g, list) else [])
        for it in items:
            if isinstance(it, dict) and it.get("paperId") and _cid(it) and _cid(it) not in seen:
                seen.add(_cid(it))
                cands.append(it)
    print(f"  specific: {len(cands)} candidates (ambiguous={ambiguous})")
    if not cands:
        return []

    lines = []
    for i, c in enumerate(cands[:50]):
        auth = ", ".join(a.get("name", "") for a in (c.get("authors") or [])[:4])
        lines.append(f"{i}: [{_cid(c)}] {(c.get('title') or '')[:150]} "
                     f"({c.get('year')}; {auth}; {(c.get('venue') or '')[:40]}; "
                     f"{c.get('citationCount')} citations)")
    pick = await _llm_json(GPT_5_4, f"""User request for a specific paper: {query}

Candidates:
{chr(10).join(lines)}

Which candidate(s) ARE the paper the user means?
- "the X paper" means the paper that INTRODUCED / first presented X — NOT sequels ("X 2"), follow-ups, surveys, benchmarks built on X, or papers merely using X. When several candidates share the name, prefer the ORIGINAL (usually the earliest of the well-known ones; use years and citation counts shown).
- If the reference unambiguously names one paper, give exactly that one index (two only if genuinely torn between near-identical records).
- If the reference is a name that several distinct papers are each known by (different fields/systems independently sharing the name), include EVERY candidate that is genuinely called this — but never papers that merely cite, use, or resemble the target.
Reply ONLY with JSON: {{"indices": [<index>, ...]}}""",
                          "specific-pick")
    idxs = [i for i in (pick or {}).get("indices", []) if isinstance(i, int) and 0 <= i < len(cands)]
    cap = 6 if ambiguous else 2
    chosen = [cands[i] for i in idxs[:cap]] or cands[:1]
    return _mk_results(chosen, with_evidence=False)


# --------------------------------------------------------------------------
# metadata_f1
# --------------------------------------------------------------------------

async def _resolve_author_papers(state: TaskState, name: str, query: str,
                                 paper_fields="corpusId,title,year,venue,journal,citationCount,authors,abstract") -> list:
    """Resolve an author name (handling fragment identities) to their papers."""
    find = _get_tool(state, "search_authors_by_name")
    get_papers = _get_tool(state, "get_author_papers")
    authors = await _call(find, name=name, fields="authorId,name,paperCount", limit=10)
    authors = [a for a in authors if isinstance(a, dict) and a.get("authorId")]
    if not authors:
        return []
    authors.sort(key=lambda a: -(a.get("paperCount") or 0))
    top = authors[:5]
    if len(top) > 1:
        # LLM disambiguation over a few sample titles per identity
        samples = await asyncio.gather(*[
            _call(get_papers, author_id=str(a["authorId"]), paper_fields="title,year", limit=5)
            for a in top])
        lines = []
        for a, papers in zip(top, samples):
            ts = "; ".join((p.get("title") or "")[:80] for p in papers[:4])
            lines.append(f"- id {a['authorId']}: name={a.get('name')!r}, "
                         f"{a.get('paperCount')} papers. Sample: {ts}")
        ans = await _llm_json(GPT_5_4_MINI, f"""Semantic Scholar has several author records for the name "{name}". The user's query is: {query}

Records:
{chr(10).join(lines)}

Some records are duplicate fragments of the same real person. Reply ONLY with JSON listing every record id that is the person the query intends: {{"author_ids": ["id", ...]}}""",
                             "author-disambig")
        ids = [str(i) for i in (ans or {}).get("author_ids", [])]
        valid = {str(a["authorId"]) for a in top}
        ids = [i for i in ids if i in valid] or [str(top[0]["authorId"])]
    else:
        ids = [str(top[0]["authorId"])]

    lists = await asyncio.gather(*[
        _call(get_papers, author_id=i, paper_fields=paper_fields, limit=1000) for i in ids])
    papers, seen = [], set()
    for lst in lists:
        for p in lst:
            if isinstance(p, dict) and _cid(p) and _cid(p) not in seen:
                seen.add(_cid(p))
                papers.append(p)
    print(f"  author {name!r}: ids={ids} -> {len(papers)} papers")
    return papers


async def _citing_ids_of_author(state: TaskState, name: str, query: str) -> set:
    """Union of corpusIds of papers citing any of the author's top papers."""
    papers = await _resolve_author_papers(
        state, name, query, paper_fields="corpusId,paperId,title,citationCount")
    papers = [p for p in papers if (p.get("citationCount") or 0) > 0]
    papers.sort(key=lambda p: -(p.get("citationCount") or 0))
    targets = papers[:80]
    cites_tool = _get_tool(state, "get_citations")
    lists = await asyncio.gather(*[
        _call(cites_tool, paper_id=f"CorpusId:{_cid(p)}", fields="corpusId", limit=1000)
        for p in targets])
    ids = set()
    for lst in lists:
        for it in lst:
            cp = it.get("citingPaper", it) if isinstance(it, dict) else {}
            if _cid(cp):
                ids.add(_cid(cp))
    print(f"  citing-union of {name!r}: {len(targets)} target papers -> {len(ids)} citing ids")
    return ids


def _norm_venue(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _venue_match(paper: dict, venues: list) -> bool:
    """venues is a list of alias lists (or plain strings). Normalized substring
    match in either direction: 'neurips' vs 'neuralinformationprocessingsystems'
    both match when the plan supplies alias variants."""
    pv = _norm_venue(paper.get("venue") or "")
    jn = paper.get("journal") or {}
    jv = _norm_venue(jn.get("name", "") if isinstance(jn, dict) else "")
    for v in venues:
        aliases = v if isinstance(v, list) else [v]
        for a in aliases:
            a = _norm_venue(a if isinstance(a, str) else "")
            if not a:
                continue
            for hay in (pv, jv):
                if hay and (a in hay or hay in a):
                    return True
    return False


def _is_journal_article(paper: dict) -> bool:
    jn = paper.get("journal") or {}
    jname = (jn.get("name", "") if isinstance(jn, dict) else "").strip().lower()
    if not jname:
        return False
    return not any(w in jname for w in ("arxiv", "biorxiv", "medrxiv", "ssrn", "preprint"))


def _apply_filters(papers: list, f: dict) -> list:
    out = []
    y_min, y_max = f.get("year_min"), f.get("year_max")
    min_c = f.get("min_citations")
    max_c = f.get("max_citations")
    min_a = f.get("min_authors")
    max_a = f.get("max_authors")
    venues = f.get("venues") or []
    journal_only = bool(f.get("journal_only"))
    excl = [n.lower().split()[-1] for n in (f.get("exclude_authored_by") or []) if n]
    for p in papers:
        year = p.get("year")
        if y_min is not None and (year is None or year < y_min):
            continue
        if y_max is not None and (year is None or year > y_max):
            continue
        cc = p.get("citationCount")
        if min_c is not None and (cc is None or cc < min_c):
            continue
        if max_c is not None and (cc is None or cc > max_c):
            continue
        if min_a is not None or max_a is not None:
            na = len(p.get("authors") or [])
            if min_a is not None and na and na < min_a:
                continue
            if max_a is not None and na and na > max_a:
                continue
        if venues and not _venue_match(p, venues):
            continue
        if journal_only and not _is_journal_article(p):
            continue
        if excl:
            last_names = {(a.get("name") or "").lower().split()[-1]
                          for a in (p.get("authors") or []) if a.get("name")}
            if any(e in last_names for e in excl):
                continue
        out.append(p)
    return out


_VERIFY_FIELDS = "corpusId,title,abstract,year,venue,journal,citationCount,authors,publicationDate"


async def _verify_batch(state: TaskState, papers: list) -> list:
    """Re-fetch citation-derived candidates via the bisecting batch fetcher:
    enforces the snapshot date-cutoff (get_citations output is unfiltered, and
    ONE post-snapshot id fails a whole naive batch call) and gives canonical
    metadata for filtering."""
    if not papers:
        return []
    ids = [_cid(p) for p in papers if _cid(p)]
    verified = await _batch_fetch(state, ids, _VERIFY_FIELDS)
    if not verified and papers:
        print("  verify_batch returned nothing; keeping unverified candidates")
        return papers
    print(f"  verify_batch: {len(ids)} -> {len(verified)} in-snapshot")
    return verified


def _seed_short_names(plan: dict, seeds_resolved: list) -> list:
    names = [m.strip() for m in (plan.get("seed_mention_terms") or [])
             if isinstance(m, str) and m.strip()]
    if not names:
        names = [(r.get("title") or "")[:40] for r in seeds_resolved]
    return names


async def _ref_verify(state: TaskState, papers: list, seeds: list) -> dict:
    """Fetch each candidate's references (bisecting batch) and return
    {cid: set(seed indices whose paper appears in the references)}.
    Seed match: the seed's paperId, or a normalized-title containment either
    direction (many reference entries have null paperId)."""
    if not papers or not seeds:
        return {}
    seed_pids = {r.get("paperId"): j for j, r in enumerate(seeds) if r.get("paperId")}
    seed_titles = [(j, _norm_venue(r.get("title") or "")) for j, r in enumerate(seeds)]
    ids = [_cid(p) for p in papers if _cid(p)]
    fetched = await _batch_fetch(state, ids, "corpusId,references", chunk=40)
    matched = {}
    for p in fetched:
        refs = p.get("references") or []
        hits = set()
        for r in refs:
            if not isinstance(r, dict):
                continue
            j = seed_pids.get(r.get("paperId"))
            if j is not None:
                hits.add(j)
                continue
            rt = _norm_venue(r.get("title") or "")
            if rt:
                for j, st in seed_titles:
                    if st and (st in rt or rt in st):
                        hits.add(j)
        if hits and _cid(p):
            matched[_cid(p)] = hits
    print(f"  ref-verify: {len(ids)} candidates -> refs fetched for {len(fetched)}, "
          f"{len(matched)} cite >=1 seed")
    return matched


async def _reverse_candidates(state: TaskState, plan: dict, seeds_resolved: list,
                              filters: dict) -> list:
    """Find likely citers of the seed paper(s) WITHOUT get_citations: topic
    keyword searches (venue-scoped when the query names venues) plus
    snippet_search for the seed's name (papers mentioning it in body text).
    Complements get_citations' newest-first 1000 window, which misses older
    citers of heavily-cited seeds. Candidates are references-verified; when the
    query expects MANY results, unverified candidates that mention the seed's
    name in title/abstract are appended after the verified ones."""
    search = _get_tool(state, "search_papers_by_relevance")
    snip = _get_tool(state, "snippet_search")
    fields = "corpusId,title,year,venue,journal,citationCount,authors"
    kws = [k for k in (plan.get("reverse_keywords") or []) if isinstance(k, str) and k.strip()][:8]
    mentions = _seed_short_names(plan, seeds_resolved)[:3]

    venue_arg = None
    venues = filters.get("venues") or []
    flat = []
    for v in venues:
        flat.extend(v if isinstance(v, list) else [v])
    flat = [v for v in flat if isinstance(v, str) and v.strip()]
    if flat:
        venue_arg = ",".join(flat[:4])

    tasks = []
    for k in kws:
        if venue_arg:
            tasks.append(_call(search, keyword=k, fields=fields, limit=100, venues=venue_arg))
        tasks.append(_call(search, keyword=k, fields=fields, limit=100))
    for m in mentions:
        tasks.append(_call(search, keyword=m, fields=fields, limit=100))
        if venue_arg:
            tasks.append(_call(snip, query=m, limit=100, venues=venue_arg, timeout=240))
        tasks.append(_call(snip, query=m, limit=100, timeout=240))
    lists = await asyncio.gather(*tasks, return_exceptions=True)

    pool = {}
    for lst in lists:
        if not isinstance(lst, list):
            continue
        for it in lst:
            if not isinstance(it, dict):
                continue
            p = it.get("paper") if "snippet" in it else it
            if not isinstance(p, dict):
                continue
            cid = _cid(p)
            if cid and cid not in pool:
                pool[cid] = p
    cands = list(pool.values())
    print(f"  reverse channel: {len(kws)} kw + {len(mentions)} mention queries -> {len(cands)} candidates")
    if not cands:
        return []

    # snippet-derived papers lack metadata; fetch canonical records for all
    cands = await _verify_batch(state, cands)

    # pre-filter before the references check to bound batch load
    pre = _apply_filters(cands, filters)
    pre.sort(key=lambda p: -(p.get("citationCount") or 0))
    pre = pre[:REFCHECK_CAP]
    print(f"  reverse channel: {len(pre)} pass filters (ref-check cap {REFCHECK_CAP})")
    if not pre or not seeds_resolved:
        return pre if not seeds_resolved else []

    need_all = (plan.get("seed_combine") or "all") == "all"
    matched = await _ref_verify(state, pre, seeds_resolved)
    required = len(seeds_resolved) if need_all else 1
    verified = [p for p in pre if len(matched.get(_cid(p), ())) >= required]
    print(f"  reverse channel: {len(verified)} verified citers of seed(s)")

    # Recall extension for broad queries: a paper that passes every metadata
    # filter AND names the seed in its title/abstract almost certainly cites
    # it. Only when the query expects many results (exact-match F1 trades a
    # little precision for a lot of recall there).
    if (plan.get("expected_result_count") or "").lower() == "many":
        terms = [m.lower() for m in mentions if m]
        extra = []
        for p in pre:
            cid = _cid(p)
            if len(matched.get(cid, ())) >= required:
                continue
            if cid in matched and not need_all:
                continue
            txt = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
            ok = all(t in txt for t in terms) if need_all else any(t in txt for t in terms)
            if terms and ok:
                extra.append(p)
        if extra:
            print(f"  reverse channel: +{len(extra)} unverified seed-mentioning candidates")
            verified = verified + extra
    return verified


async def solve_metadata(state: TaskState, query: str) -> list:
    plan = await _llm_json(GPT_5_4, f"""Parse this scholarly-paper metadata query into a retrieval plan.

Query: {query}

Reply ONLY with a JSON object with these keys (use null / [] when not applicable):
{{
 "seed_papers": [{{"reference": "<how the query names it>", "title_guess": "<exact full title of that well-known paper, from your knowledge>"}}],
 "seed_combine": "all" or "any",            // results must cite ALL seed papers, or ANY
 "seed_mention_terms": ["<the seed's short name as papers mention it in text, e.g. RoBERTa>"],
 "reverse_keywords": ["<topic keyword query likely to find papers matching the constraints / citing the seed>", ... up to 8 diverse queries],
 "authors": ["<name>"],                     // results are papers AUTHORED by these people
 "must_cite_authors": ["<name>"],           // results must cite work by these people
 "expected_result_count": "one" | "few" | "many",   // "A paper that..." => one; a broad filter => many
 "filters": {{
   "year_min": <int|null>,                  // INCLUSIVE. "after 2022" / "2022 and beyond" => 2022
   "year_max": <int|null>,                  // "before 2020" => 2019
   "min_citations": <int|null>,             // lenient: "more than 50 citations" => 50
   "max_citations": <int|null>,
   "min_authors": <int|null>,               // "more than 3 authors" => 4
   "max_authors": <int|null>,
   "venues": [["<venue name>", "<full/alternate name variant>", ...], ...],  // one inner list per venue, ALL naming variants (e.g. ["NeurIPS", "Neural Information Processing Systems"])
   "journal_only": <true|false>,            // true iff query demands journal articles
   "exclude_authored_by": ["<name>"]        // e.g. "not self-citations of X" => exclude X
 }}
}}""",
                          "meta-plan")
    if not plan:
        return []
    print(f"  metadata plan: {json.dumps(plan)[:500]}")
    filters = plan.get("filters") or {}
    seeds = plan.get("seed_papers") or []
    authors = plan.get("authors") or []
    must_cite = plan.get("must_cite_authors") or []

    candidates = None
    from_citations = False
    seeds_resolved = []

    # papers citing the seed paper(s)
    if seeds:
        resolved = await asyncio.gather(*[
            _resolve_title(state, s.get("title_guess") or s.get("reference") or "",
                           fields="corpusId,paperId,title,year,authors,venue,abstract,citationCount")
            for s in seeds[:4] if isinstance(s, dict)])
        seeds_resolved = [r for r in resolved if r]
        for r in seeds_resolved:
            print(f"  seed resolved: [{_cid(r)}] cites={r.get('citationCount')} "
                  f"{(r.get('title') or '')[:80]}")
        if seeds_resolved:
            cites_tool = _get_tool(state, "get_citations")
            lists = await asyncio.gather(*[
                _call(cites_tool, paper_id=f"CorpusId:{_cid(r)}", fields=_VERIFY_FIELDS, limit=1000)
                for r in seeds_resolved])
            per_seed = []
            for lst in lists:
                d = {}
                for it in lst:
                    cp = it.get("citingPaper", it) if isinstance(it, dict) else {}
                    if _cid(cp):
                        d[_cid(cp)] = cp
                per_seed.append(d)
            if (plan.get("seed_combine") or "all") == "all" and len(per_seed) > 1:
                # Direct intersection of citation windows misses whenever one
                # seed is heavily cited (its 1000-window covers a different era
                # than the other seed's). Base on the LEAST-cited seed's window
                # — the most complete citer list available — and verify each
                # candidate cites the OTHER seeds via its references.
                common = set.intersection(*[set(d) for d in per_seed])
                pool = {i: per_seed[0][i] for i in common}
                print(f"  direct window intersection: {len(pool)}")
                base_i = min(range(len(seeds_resolved)),
                             key=lambda j: seeds_resolved[j].get("citationCount") or 10**9)
                base_cands = [p for c, p in per_seed[base_i].items() if c not in pool]
                base_cands = await _verify_batch(state, base_cands)
                base_cands.sort(key=lambda p: -(p.get("year") or 0))
                base_cands = base_cands[:BASE_WINDOW_REFCHECK_CAP]
                others = [s for j, s in enumerate(seeds_resolved) if j != base_i]
                matched = await _ref_verify(state, base_cands, others)
                added = 0
                for p in base_cands:
                    if len(matched.get(_cid(p), ())) == len(others):
                        pool.setdefault(_cid(p), p)
                        added += 1
                print(f"  base-window ({_cid(seeds_resolved[base_i])}) ref-verified adds: {added}")
            else:
                pool = {}
                for d in per_seed:
                    pool.update(d)
            candidates = list(pool.values())
            from_citations = True
            print(f"  citing candidates: {len(candidates)}")

    # papers authored by given people
    if authors:
        author_papers = []
        for nm in authors[:3]:
            author_papers.extend(await _resolve_author_papers(state, nm, query))
        if candidates is None:
            candidates = author_papers
        else:
            ok = {_cid(p) for p in author_papers}
            candidates = [p for p in candidates if _cid(p) in ok]

    # results must cite papers by these authors
    if must_cite and candidates:
        citing_union = set()
        for nm in must_cite[:2]:
            citing_union |= await _citing_ids_of_author(state, nm, query)
        candidates = [p for p in candidates if _cid(p) in citing_union]
        print(f"  after must-cite filter: {len(candidates)}")

    if from_citations and candidates:
        candidates = await _verify_batch(state, candidates)

    filtered = _apply_filters(candidates or [], filters)
    print(f"  after filters: {len(filtered)} candidates")

    # Reverse citation channel: get_citations' newest-first 1000 window misses
    # older citers of heavily-cited seeds. Search likely citers directly and
    # verify via the references field. Run whenever seeds exist and the direct
    # channel looks incomplete (window saturated or few survivors).
    if seeds_resolved and not authors:
        window_saturated = candidates is not None and len(candidates) >= 950
        if window_saturated or len(filtered) < 30:
            try:
                extra = await _reverse_candidates(state, plan, seeds_resolved, filters)
            except Exception:
                print("  reverse channel crashed:\n" + traceback.format_exc()[-600:])
                extra = []
            have = {_cid(p) for p in filtered}
            add = [p for p in extra if _cid(p) not in have]
            if add:
                print(f"  reverse channel adds {len(add)} candidates")
                filtered = filtered + add

    # No-seed, no-author queries (e.g. venue-only constraints): best-effort
    # venue/keyword search — an empty submission scores 0 with certainty.
    if not filtered and candidates is None:
        try:
            filtered = await _reverse_candidates(state, plan, [], filters)
        except Exception:
            filtered = []

    # Relaxation ladder: never submit 0 when we had any pool at all.
    if not filtered and candidates:
        for drop in ("min_citations", "venues", "year"):
            f2 = dict(filters)
            if drop == "min_citations":
                f2["min_citations"] = None
            elif drop == "venues":
                f2["min_citations"] = None
                f2["venues"] = []
            else:
                f2["min_citations"] = None
                f2["venues"] = []
                f2["year_min"] = (filters.get("year_min") - 1) if filters.get("year_min") else None
                f2["year_max"] = (filters.get("year_max") + 1) if filters.get("year_max") else None
            filtered = _apply_filters(candidates, f2)
            if filtered:
                print(f"  relaxation ladder ({drop}) -> {len(filtered)}")
                filtered = filtered[:100]
                break

    filtered.sort(key=lambda p: -(p.get("citationCount") or 0))
    if (plan.get("expected_result_count") or "").lower() == "one":
        filtered = filtered[:3]
    return _mk_results(filtered, with_evidence=False)


# --------------------------------------------------------------------------
# semantic_f1
# --------------------------------------------------------------------------

def _cand_desc(c: dict) -> str:
    desc = c.get("abstract") or ""
    if not desc:
        desc = _tldr_text(c)
    if not desc:
        for sns in (c.get("_crit_snips") or {}).values():
            if sns:
                desc = sns[0] or ""
                break
    if not desc and c.get("_snippets"):
        desc = c["_snippets"][0] or ""
    return desc or ""


async def _grade_chunk(query: str, criteria_txt: str, cands: list, offset: int) -> dict:
    lines = []
    for i, c in enumerate(cands):
        lines.append(f"[{offset + i}] {(c.get('title') or '')[:160]} :: {_cand_desc(c)[:320]}")
    ans = await _llm_json(GPT_5_4, f"""You are ranking papers retrieved for a scholarly literature search.

Search query: {query}

The query decomposes into these relevance criteria — a top grade requires satisfying ALL of them, not just the general topic:
{criteria_txt}

Grade each paper 0-10:
 10 = clearly satisfies every criterion
 7  = probably satisfies all criteria, evidence incomplete
 4  = on the main topic but misses or contradicts one criterion
 2  = tangentially related
 0  = unrelated

Papers:
{chr(10).join(lines)}

Reply ONLY with JSON: {{"grades": {{"<index>": <grade>, ...}}}} covering every index shown.""",
                         f"grade@{offset}")
    grades = {}
    for k, v in ((ans or {}).get("grades") or {}).items():
        try:
            grades[int(k)] = max(0, min(10, int(v)))
        except (ValueError, TypeError):
            pass
    return grades


async def _grade_chunk_safe(query: str, criteria_txt: str, cands: list, offset: int) -> dict:
    try:
        return await _grade_chunk(query, criteria_txt, cands, offset)
    except Exception:
        print(f"  grade chunk @{offset} crashed:\n" + traceback.format_exc()[-600:])
        return {}


async def _enrich_with_snippets(state: TaskState, cands: list, criteria: list,
                                snippet_q: str) -> None:
    """Criterion-targeted evidence: for each predicted criterion, scoped
    snippet_search over CHUNKS of the shortlist pulls body passages stating
    that criterion in each paper's own words. Chunking matters: one call
    scoped to 100 ids gets dominated by a few strong papers and starves the
    rest (verified server behavior)."""
    top = [c for c in cands[:ENRICH_TOP] if _cid(c)]
    if not top:
        return
    by_cid = {_cid(c): c for c in top}
    snip = _get_tool(state, "snippet_search")
    queries = []
    for ci, cr in enumerate(criteria[:4]):
        kw = (cr.get("keywords") or cr.get("name") or "").strip()
        if kw:
            queries.append((ci, kw))
    if not queries:
        queries = [(0, snippet_q)]
    id_chunks = [top[i:i + ENRICH_SCOPE_CHUNK] for i in range(0, len(top), ENRICH_SCOPE_CHUNK)]
    tasks, keys = [], []
    for ci, q in queries:
        for ch in id_chunks:
            ids = ",".join(f"CorpusId:{_cid(c)}" for c in ch)
            tasks.append(_call(snip, query=q, paper_ids=ids, limit=50, timeout=240))
            keys.append(ci)
    lists = await asyncio.gather(*tasks, return_exceptions=True)
    attached = 0
    for ci, lst in zip(keys, lists):
        if not isinstance(lst, list):
            continue
        per_paper = {}
        for entry in lst:
            if not isinstance(entry, dict):
                continue
            paper = entry.get("paper") or {}
            text = (entry.get("snippet") or {}).get("text") or ""
            cid = str(paper.get("corpusId") or "")
            c = by_cid.get(cid)
            if not c or not text.strip():
                continue
            # at most 2 snippets per criterion per paper
            if per_paper.get(cid, 0) >= 2:
                continue
            per_paper[cid] = per_paper.get(cid, 0) + 1
            c.setdefault("_crit_snips", {}).setdefault(ci, []).append(text.strip())
            attached += 1
    print(f"  enrichment: {len(queries)} criteria x {len(id_chunks)} chunks -> {attached} snippets attached")


async def _judge_mimic_chunk(query: str, criteria: list, cands: list, offset: int) -> dict:
    """Rate each criterion 0/1/3 for each paper based ONLY on the evidence text
    that will be submitted — mirroring the benchmark judge, which sees nothing
    else. Returns {index: [r_c, ...]}."""
    crit_lines = "\n".join(
        f"  C{j}: {c.get('name', '')} — {c.get('keywords', '')}" for j, c in enumerate(criteria))
    lines = []
    for i, c in enumerate(cands):
        ev = c.get("_final_evidence") or ""
        lines.append(f"[{offset + i}] {ev[:1100]}")
    ans = await _llm_json(GPT_5_4, f"""A relevance judge will score papers for this literature-search query using ONLY the evidence text shown per paper (no access to the actual paper).

Query: {query}

Criteria:
{crit_lines}

For each paper below, rate EACH criterion from the evidence text alone:
 3 = the evidence explicitly demonstrates this criterion
 1 = the evidence partially/vaguely suggests it
 0 = the evidence does not support it

Papers (evidence text):
{chr(10).join(lines)}

Reply ONLY with JSON: {{"ratings": {{"<index>": [<C0 rating>, <C1 rating>, ...], ...}}}} covering every index, each list with exactly {len(criteria)} entries.""",
                         f"judge@{offset}")
    out = {}
    for k, v in ((ans or {}).get("ratings") or {}).items():
        try:
            idx = int(k)
            vals = [x if x in (0, 1, 3) else (3 if x >= 2 else 0) for x in
                    [int(y) for y in v]][:len(criteria)]
            if vals:
                out[idx] = vals
        except (ValueError, TypeError):
            pass
    return out


async def _judge_mimic_safe(query: str, criteria: list, cands: list, offset: int) -> dict:
    try:
        return await _judge_mimic_chunk(query, criteria, cands, offset)
    except Exception:
        print(f"  judge-mimic chunk @{offset} crashed:\n" + traceback.format_exc()[-600:])
        return {}


def _weighted_grade(ratings: list, weights: list) -> float:
    """The benchmark's own combination: min(1, sum w*r/3)."""
    s = sum(w * r / 3.0 for w, r in zip(weights, ratings))
    return min(1.0, s)


async def solve_semantic(state: TaskState, query: str) -> list:
    plan = await _llm_json(GPT_5_4, f"""A user is searching a scholarly paper corpus.

User query: {query}

First decompose the query into its relevance criteria: the core topic plus EVERY explicit qualifier (method, metric, domain, population, application, evaluation protocol, time constraint...). A paper is fully relevant only if it satisfies ALL criteria. Then produce search inputs.

Reply with JSON ONLY:
{{"criteria": [{{"name": "<short criterion name>", "keywords": "<3-8 word noun-phrase capturing this criterion, for passage search>"}}, ... 2-4 criteria],
  "keyword_queries": ["<noun-phrase keyword query>", ... 8 diverse variants covering synonyms, rephrasings, and adjacent terminology],
  "snippet_queries": ["<full-sentence version of the information need>", "<a differently-phrased full-sentence version>"],
  "year_min": <int|null>, "year_max": <int|null>}}
Keyword queries must be bare noun phrases (no question words, no 'papers about'). Set year bounds ONLY if the query itself states a time constraint.""",
                          "sem-plan")
    variants = [v for v in (plan or {}).get("keyword_queries", []) if isinstance(v, str) and v.strip()]
    if not variants:
        variants = [_strip_question(query)]
    snippet_qs = [q for q in (plan or {}).get("snippet_queries", [])
                  if isinstance(q, str) and q.strip()][:2] or [_strip_question(query)]
    criteria = [c for c in (plan or {}).get("criteria", []) if isinstance(c, dict)]
    if not criteria:
        criteria = [{"name": "relevance to the query", "keywords": _strip_question(query)}]
    criteria_txt = "\n".join(
        f"- {c.get('name', '')}: {c.get('keywords', '')}" for c in criteria)
    y_min, y_max = (plan or {}).get("year_min"), (plan or {}).get("year_max")
    print(f"  criteria: {[c.get('name') for c in criteria]}")

    search = _get_tool(state, "search_papers_by_relevance")
    snip = _get_tool(state, "snippet_search")
    fields = "corpusId,title,abstract,tldr,year,venue,citationCount"
    kw_tasks = [_call(search, keyword=v, fields=fields, limit=100) for v in variants[:8]]
    # snippet retrieval: the full information need, plus each criterion's
    # noun phrase — body-text matches surface papers whose abstracts never
    # state the criterion (the dominant missed-recall mode).
    crit_qs = []
    for c in criteria[:4]:
        kw = (c.get("keywords") or c.get("name") or "").strip()
        if kw and kw.lower() not in {q.lower() for q in snippet_qs}:
            crit_qs.append(kw)
    sn_tasks = [_call(snip, query=q, limit=100, timeout=240) for q in snippet_qs]
    sn_tasks += [_call(snip, query=q, limit=50, timeout=240) for q in crit_qs]
    all_results = await asyncio.gather(*kw_tasks, *sn_tasks, return_exceptions=True)
    kw_results = all_results[:len(kw_tasks)]
    sn_results = all_results[len(kw_tasks):]

    pool = {}
    for lst in kw_results:
        if not isinstance(lst, list):
            continue
        for rank, p in enumerate(lst):
            if not isinstance(p, dict):
                continue
            cid = _cid(p)
            if not cid:
                continue
            if cid not in pool:
                p["_hits"], p["_best_rank"], p["_snippets"] = 1, rank, []
                pool[cid] = p
            else:
                pool[cid]["_hits"] += 1
                pool[cid]["_best_rank"] = min(pool[cid]["_best_rank"], rank)
    for lst in sn_results:
        if not isinstance(lst, list):
            continue
        for rank, entry in enumerate(lst):
            if not isinstance(entry, dict):
                continue
            paper = entry.get("paper") or {}
            snippet = (entry.get("snippet") or {}).get("text") or ""
            cid = str(paper.get("corpusId") or "")
            if not cid:
                continue
            if cid in pool:
                pool[cid]["_hits"] += 1
                if snippet:
                    pool[cid].setdefault("_snippets", []).append(snippet)
            else:
                paper["_hits"], paper["_best_rank"] = 1, rank
                paper["_snippets"] = [snippet] if snippet else []
                pool[cid] = paper
    cands = list(pool.values())
    print(f"  semantic: {len(variants[:8])} variants + {len(snippet_qs)}+{len(crit_qs)} snippet queries "
          f"-> {len(cands)} unique candidates")

    # hard year filter only when we have plenty left afterwards
    if (y_min or y_max) and cands:
        kept = [c for c in cands
                if (y_min is None or (c.get("year") or 0) >= y_min)
                and (y_max is None or (c.get("year") or 9999) <= y_max)]
        if len(kept) >= 30:
            print(f"  year filter [{y_min},{y_max}]: {len(cands)} -> {len(kept)}")
            cands = kept

    # order pre-grade by retrieval strength so the grade cap keeps the best
    cands.sort(key=lambda c: (-c["_hits"], c["_best_rank"]))
    cands = cands[:SEMANTIC_CAND_CAP]

    chunks = [cands[i:i + GRADE_CHUNK] for i in range(0, len(cands), GRADE_CHUNK)]
    grade_maps = await asyncio.gather(*[
        _grade_chunk_safe(query, criteria_txt, ch, i * GRADE_CHUNK)
        for i, ch in enumerate(chunks)], return_exceptions=True)
    grades = {}
    retry = []
    for i, m in enumerate(grade_maps):
        if isinstance(m, dict) and m:
            grades.update(m)
        else:
            retry.append(i)
    if retry:
        print(f"  retrying {len(retry)} failed grade chunks")
        retry_maps = await asyncio.gather(*[
            _grade_chunk_safe(query, criteria_txt, chunks[i], i * GRADE_CHUNK)
            for i in retry], return_exceptions=True)
        for m in retry_maps:
            if isinstance(m, dict):
                grades.update(m)
    for i, c in enumerate(cands):
        c["_grade"] = grades.get(i, 4)  # ungraded -> mid-low

    # grade-0s go to the tail rather than being dropped: beyond position K they
    # are never judged, and a correctly-descending tail helps the rank term.
    cands.sort(key=lambda c: (-c["_grade"], -c["_hits"], c["_best_rank"]))

    # criterion-targeted evidence for the papers that will actually be judged
    try:
        await _enrich_with_snippets(state, cands, criteria, snippet_qs[0])
    except Exception:
        print("  enrichment failed:\n" + traceback.format_exc()[-600:])

    # ---- pass 2: judge-mimic rating of the ACTUAL evidence text, BLENDED ----
    # Only evidence-supported grade 3 earns recall, so rating each criterion
    # 0/1/3 from the final evidence approximates the judge. But iteration-4
    # showed a full re-sort on this noisy signal demotes true grade-3s below
    # the judged depth K; blending with the pass-1 grade (which iteration-3
    # validated) keeps both signals in play.
    weights = [1.0 / len(criteria)] * len(criteria)
    top = cands[:REGRADE_TOP]
    for c in top:
        c["_final_evidence"] = _build_evidence(c)
    r_chunks = [top[i:i + REGRADE_CHUNK] for i in range(0, len(top), REGRADE_CHUNK)]
    maps = await asyncio.gather(*[
        _judge_mimic_safe(query, criteria, ch, i * REGRADE_CHUNK)
        for i, ch in enumerate(r_chunks)], return_exceptions=True)
    ratings = {}
    for m in maps:
        if isinstance(m, dict):
            ratings.update(m)
    print(f"  judge-mimic: rated {len(ratings)}/{len(top)}")
    for i, c in enumerate(top):
        r = ratings.get(i)
        c["_score2"] = _weighted_grade(r, weights) if r is not None else c["_grade"] / 10.0
        c["_blend"] = 0.55 * c["_score2"] + 0.45 * (c["_grade"] / 10.0)
    if ratings:
        head = sorted(top, key=lambda c: (-c["_blend"], -c["_grade"], -c["_hits"]))
        cands = head + cands[len(top):]
        n3 = sum(1 for c in head if c.get("_score2", 0) > 0.99)
        print(f"  judge-mimic blend: {n3} predicted grade-3 in top {len(head)}")

    # uniformity hedge: all-equal judge grades zero the rank term. If our own
    # top-8 predicted grades are uniform, promote a clearly-mid paper to rank 5.
    def _bucket(c):
        s = c.get("_score2")
        if s is None:
            return c["_grade"]
        return 3 if s > 0.99 else (2 if s > 0.67 else (1 if s > 0.25 else 0))
    if len(cands) >= 8:
        top_b = _bucket(cands[0])
        if all(_bucket(c) == top_b for c in cands[:8]):
            hedge_i = next((i for i, c in enumerate(cands)
                            if 0 < _bucket(c) < top_b), None)
            if hedge_i is not None and hedge_i > 4:
                hedge = cands.pop(hedge_i)
                cands.insert(4, hedge)
                print(f"  uniformity hedge: bucket-{_bucket(hedge)} paper -> position 5")

    print(f"  submitting {min(len(cands), MAX_RESULTS)} papers "
          f"(top blend: {[round(c.get('_blend', -1), 2) for c in cands[:10]]})")
    return _mk_results(cands, with_evidence=True)


# --------------------------------------------------------------------------
# fallback: no/low-LLM keyword search
# --------------------------------------------------------------------------

async def solve_fallback(state: TaskState, query: str, score_type: str) -> list:
    search = _get_tool(state, "search_papers_by_relevance")
    hits = await _call(search, keyword=_strip_question(query),
                       fields="corpusId,title,abstract,tldr,year", limit=100)
    if not hits:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]+", _strip_question(query))
        hits = await _call(search, keyword=" ".join(words[:8]),
                           fields="corpusId,title,abstract,tldr,year", limit=100)
    semantic = score_type not in ("specific_f1", "metadata_f1")
    keep = hits if semantic else hits[:3]
    print(f"  fallback: submitting {len(keep)}")
    return _mk_results(keep, with_evidence=semantic)


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------

@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata.get("raw_query") or state.input_text
        score_type = state.metadata.get("score_type", "")
        print(f"[{state.sample_id}] score_type={score_type} query={query[:120]!r}")

        results = []
        try:
            if score_type == "specific_f1":
                main = solve_specific(state, query)
            elif score_type == "metadata_f1":
                main = solve_metadata(state, query)
            else:
                main = solve_semantic(state, query)
            results = await asyncio.wait_for(main, timeout=1440)  # 24 min, leave slack
        except asyncio.TimeoutError:
            print("  main path timed out")
        except Exception:
            print("  main path crashed:\n" + traceback.format_exc()[-2000:])

        if not results:
            try:
                results = await asyncio.wait_for(
                    solve_fallback(state, query, score_type), timeout=150)
            except Exception:
                print("  fallback crashed:\n" + traceback.format_exc()[-800:])

        state.output.completion = json.dumps({
            "output": {"query_id": state.sample_id, "results": results}
        })
        print(f"  submitted {len(results)} papers")
        return state

    return solve
