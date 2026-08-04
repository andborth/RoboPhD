"""iter3-recallforge-v1: PaperFindingBench solver.

Routes on score_type:
  - specific_f1: LLM reference expansion (with ambiguity detection) -> title
    search -> submit every true referent (1 when confident, up to 6 when the
    nickname names several distinct papers)
  - metadata_f1: LLM plan -> citation/author tool execution -> code filters
  - semantic_f1: criteria decomposition -> wide parallel retrieval -> LLM
    grading against explicit criteria -> criterion-targeted snippet evidence
    -> deep ranked list (up to 250)

Iteration-2 diagnostics this design targets:
  - recall (grade-3 papers in top K / K) is the bottleneck: grade 3 requires
    the judge to rate EVERY weighted criterion Perfectly Relevant from
    markdown_evidence alone, and abstract-only evidence stalls papers at
    grade <=2 (worth zero recall). Scoped snippet_search pulls body passages
    that state each criterion in the paper's own words.
  - a None-subscript crash in grading zeroed two queries -> fixed + guarded.
  - submission depth fell short of K -> wider pool, grade-0s kept at tail.
  - ambiguous specific nicknames ("the SPIKE paper") have multi-paper gold.
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
ABSTRACT_CAP = 1300   # leave evidence room for criterion snippets
SEMANTIC_CAND_CAP = 400
GRADE_CHUNK = 25
ENRICH_TOP = 100      # candidates that get criterion-targeted snippets


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


async def _call(tool, timeout=310, **kwargs) -> list:
    """Guarded tool call: parsed items, or [] on any failure."""
    try:
        raw = await asyncio.wait_for(tool(**kwargs), timeout=timeout)
        return _parse_items(raw)
    except Exception as e:
        print(f"  tool call failed ({getattr(tool, '__name__', tool)}, "
              f"{ {k: str(v)[:60] for k, v in kwargs.items()} }): {type(e).__name__}: {str(e)[:200]}")
        return []


def _cid(paper: dict) -> str:
    v = paper.get("corpusId")
    return str(v) if v is not None else ""


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
    n_snips = 0
    for sn in paper.get("_snippets", []):
        if not sn or not sn.strip():
            continue
        sn = sn.strip()
        # skip snippets that just repeat the abstract
        if ab_low and sn[:60].lower() in ab_low:
            continue
        passages.append(sn)
        n_snips += 1
        if n_snips >= 4:
            break
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
- If the reference unambiguously names one paper, give exactly that one index (two only if genuinely torn between near-identical records).
- If the reference is a name that several distinct papers are each known by, include EVERY candidate that is genuinely called this (different fields/systems sharing the name) — but never papers that merely cite, use, or resemble the target.
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


def _venue_match(paper: dict, venues: list) -> bool:
    pv = (paper.get("venue") or "")
    jn = paper.get("journal") or {}
    jname = jn.get("name", "") if isinstance(jn, dict) else ""
    hay = f"{pv} {jname}".lower()
    for v in venues:
        v = (v or "").lower().strip()
        if v and (v in hay or hay.strip() in (v,)):
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


async def _verify_batch(state: TaskState, papers: list) -> list:
    """Re-fetch citation-derived candidates via get_paper_batch: enforces the
    snapshot date-cutoff (get_citations output is unfiltered) and gives
    canonical metadata for filtering."""
    if not papers:
        return []
    batch_tool = _get_tool(state, "get_paper_batch")
    fields = "corpusId,title,abstract,year,venue,journal,citationCount,authors,publicationDate"
    verified = []
    ids = [_cid(p) for p in papers if _cid(p)]
    chunks = [ids[i:i + 100] for i in range(0, len(ids), 100)]
    lists = await asyncio.gather(*[
        _call(batch_tool, ids=[f"CorpusId:{i}" for i in ch], fields=fields) for ch in chunks])
    for lst in lists:
        for p in lst:
            if isinstance(p, dict) and _cid(p):
                verified.append(p)
    if not verified and papers:
        print("  verify_batch returned nothing; keeping unverified candidates")
        return papers
    print(f"  verify_batch: {len(ids)} -> {len(verified)} in-snapshot")
    return verified


async def solve_metadata(state: TaskState, query: str) -> list:
    plan = await _llm_json(GPT_5_4, f"""Parse this scholarly-paper metadata query into a retrieval plan.

Query: {query}

Reply ONLY with a JSON object with these keys (use null / [] when not applicable):
{{
 "seed_papers": [{{"reference": "<how the query names it>", "title_guess": "<exact full title of that well-known paper, from your knowledge>"}}],
 "seed_combine": "all" or "any",            // results must cite ALL seed papers, or ANY
 "authors": ["<name>"],                     // results are papers AUTHORED by these people
 "must_cite_authors": ["<name>"],           // results must cite work by these people
 "filters": {{
   "year_min": <int|null>,                  // "after 2022" => 2023
   "year_max": <int|null>,                  // "before 2020" => 2019
   "min_citations": <int|null>,             // "more than 50 citations" => 51
   "max_citations": <int|null>,
   "venues": ["<exact venue/journal name>"] or [],
   "journal_only": <true|false>,            // true iff query demands journal articles
   "exclude_authored_by": ["<name>"]        // e.g. "not self-citations of X" => exclude X
 }}
}}""",
                          "meta-plan")
    if not plan:
        return []
    print(f"  metadata plan: {json.dumps(plan)[:400]}")
    filters = plan.get("filters") or {}
    seeds = plan.get("seed_papers") or []
    authors = plan.get("authors") or []
    must_cite = plan.get("must_cite_authors") or []

    candidates = None
    from_citations = False

    # papers citing the seed paper(s)
    if seeds:
        resolved = await asyncio.gather(*[
            _resolve_title(state, s.get("title_guess") or s.get("reference") or "")
            for s in seeds[:4] if isinstance(s, dict)])
        resolved = [r for r in resolved if r]
        for r in resolved:
            print(f"  seed resolved: [{_cid(r)}] {(r.get('title') or '')[:80]}")
        if resolved:
            cites_tool = _get_tool(state, "get_citations")
            fields = "corpusId,title,abstract,year,venue,journal,citationCount,authors,publicationDate"
            lists = await asyncio.gather(*[
                _call(cites_tool, paper_id=f"CorpusId:{_cid(r)}", fields=fields, limit=1000)
                for r in resolved])
            per_seed = []
            for lst in lists:
                d = {}
                for it in lst:
                    cp = it.get("citingPaper", it) if isinstance(it, dict) else {}
                    if _cid(cp):
                        d[_cid(cp)] = cp
                per_seed.append(d)
            if (plan.get("seed_combine") or "all") == "all" and len(per_seed) > 1:
                common = set.intersection(*[set(d) for d in per_seed])
                pool = {i: per_seed[0][i] for i in common}
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

    if candidates is None:
        return []

    # results must cite papers by these authors
    if must_cite and candidates:
        citing_union = set()
        for nm in must_cite[:2]:
            citing_union |= await _citing_ids_of_author(state, nm, query)
        candidates = [p for p in candidates if _cid(p) in citing_union]
        print(f"  after must-cite filter: {len(candidates)}")

    if from_citations:
        candidates = await _verify_batch(state, candidates)

    candidates = _apply_filters(candidates, filters)
    print(f"  after filters: {len(candidates)} candidates")
    candidates.sort(key=lambda p: -(p.get("citationCount") or 0))
    return _mk_results(candidates, with_evidence=False)


# --------------------------------------------------------------------------
# semantic_f1
# --------------------------------------------------------------------------

def _cand_desc(c: dict) -> str:
    desc = c.get("abstract") or ""
    if not desc:
        desc = _tldr_text(c)
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
    """Criterion-targeted evidence: for each predicted criterion, one
    snippet_search scoped to the top candidates pulls body passages stating
    that criterion in each paper's own words. Attaches to c['_snippets']."""
    top = cands[:ENRICH_TOP]
    if not top:
        return
    ids = ",".join(f"CorpusId:{_cid(c)}" for c in top if _cid(c))
    by_cid = {_cid(c): c for c in top}
    snip = _get_tool(state, "snippet_search")
    queries = []
    for cr in criteria[:4]:
        kw = (cr.get("keywords") or cr.get("name") or "").strip()
        if kw:
            queries.append(kw)
    if not queries:
        queries = [snippet_q]
    lists = await asyncio.gather(*[
        _call(snip, query=q, paper_ids=ids, limit=100, timeout=240) for q in queries],
        return_exceptions=True)
    attached = 0
    for lst in lists:
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
            # at most 2 snippets per criterion-query per paper
            if per_paper.get(cid, 0) >= 2:
                continue
            per_paper[cid] = per_paper.get(cid, 0) + 1
            c.setdefault("_snippets", []).append(text.strip())
            attached += 1
    print(f"  enrichment: {len(queries)} criterion queries -> {attached} snippets attached")


async def solve_semantic(state: TaskState, query: str) -> list:
    plan = await _llm_json(GPT_5_4, f"""A user is searching a scholarly paper corpus.

User query: {query}

First decompose the query into its relevance criteria: the core topic plus EVERY explicit qualifier (method, metric, domain, population, application, evaluation protocol, time constraint...). A paper is fully relevant only if it satisfies ALL criteria. Then produce search inputs.

Reply with JSON ONLY:
{{"criteria": [{{"name": "<short criterion name>", "keywords": "<3-8 word noun-phrase capturing this criterion, for passage search>"}}, ... 2-4 criteria],
  "keyword_queries": ["<noun-phrase keyword query>", ... 6 diverse variants covering synonyms, rephrasings, and adjacent terminology],
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
    criteria_txt = "\n".join(
        f"- {c.get('name', '')}: {c.get('keywords', '')}" for c in criteria) or \
        "- (no decomposition available; infer the criteria from the query yourself)"
    y_min, y_max = (plan or {}).get("year_min"), (plan or {}).get("year_max")
    print(f"  criteria: {[c.get('name') for c in criteria]}")

    search = _get_tool(state, "search_papers_by_relevance")
    snip = _get_tool(state, "snippet_search")
    fields = "corpusId,title,abstract,tldr,year,venue,citationCount"
    kw_tasks = [_call(search, keyword=v, fields=fields, limit=100) for v in variants[:6]]
    sn_tasks = [_call(snip, query=q, limit=100, timeout=240) for q in snippet_qs]
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
                    pool[cid]["_snippets"].append(snippet)
            else:
                paper["_hits"], paper["_best_rank"] = 1, rank
                paper["_snippets"] = [snippet] if snippet else []
                pool[cid] = paper
    cands = list(pool.values())
    print(f"  semantic: {len(variants)} variants + {len(snippet_qs)} snippet queries "
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

    # uniformity hedge: all-equal judge grades zero the rank term. If our own
    # top-8 grades are uniform, promote the best clearly-mid paper to rank 5.
    if len(cands) >= 8:
        top_grade = cands[0]["_grade"]
        prefix_uniform = all(c["_grade"] == top_grade for c in cands[:8])
        if prefix_uniform:
            hedge_i = next((i for i, c in enumerate(cands)
                            if 2 <= c["_grade"] <= max(2, top_grade - 3)), None)
            if hedge_i is None:
                hedge_i = next((i for i, c in enumerate(cands)
                                if c["_grade"] < top_grade), None)
            if hedge_i is not None and hedge_i > 4:
                hedge = cands.pop(hedge_i)
                cands.insert(4, hedge)
                print(f"  uniformity hedge: grade-{hedge['_grade']} paper -> position 5")

    print(f"  submitting {min(len(cands), MAX_RESULTS)} papers "
          f"(top grades: {[c['_grade'] for c in cands[:12]]})")
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
