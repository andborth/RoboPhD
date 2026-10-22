"""PaperFindingBench agent — iteration 002 ("deepcast").

Routes on ``state.metadata["score_type"]``:

* ``semantic_f1``  — LLM query expansion -> many free keyword / snippet
  searches (hundreds of candidates) -> cheap batched LLM grading against the
  query's implicit criteria -> submit up to 250 papers ordered by grade, with
  verbatim title+abstract(+tldr/snippet) evidence.  The scorer's recall
  denominator K is often in the hundreds and only the first K entries are
  judged, so a long, well-ordered list is the main lever.
* ``metadata_f1``  — LLM parses the structured constraints (authors, venues,
  years, citation / author-count thresholds, "cites paper X", topic), then
  uses author / citation / venue-filtered search tools and strict
  post-filtering.  Exact-match scoring: precision matters, so only papers
  that pass every checkable filter are submitted.
* ``specific_f1``  — LLM guesses exact title(s) + keywords, title lookup and
  keyword search, LLM confirms which candidate(s) are the requested paper.
"""

import asyncio
import json
import re
import time
import traceback

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
MAX_SUBMIT = 250
MAX_EVIDENCE_CHARS = 2400
PAPER_FIELDS = "title,abstract,corpusId,year,tldr,venue,citationCount,authors"
GRADE_CANDIDATES = 300          # how many semantic candidates get LLM-graded
GRADE_BATCH = 40
GRADE_ABSTRACT_CHARS = 550
TOOL_TIMEOUT = 150              # seconds per tool call (wait_for)
SNIPPET_TIMEOUT = 170


# ----------------------------------------------------------------------------
# tool plumbing
# ----------------------------------------------------------------------------
def _tools_by_name(state: TaskState) -> dict:
    return {ToolDef(t).name: t for t in state.tools}


def _parse_items(raw) -> list:
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
        elif isinstance(doc, list):
            docs.extend(doc)
        else:
            docs.append(doc)
    return docs


def _err_text(e: BaseException) -> str:
    """Drill into ExceptionGroups to find the root-cause message."""
    seen = 0
    while isinstance(e, BaseExceptionGroup) and e.exceptions and seen < 6:
        e = e.exceptions[0]
        seen += 1
    return f"{type(e).__name__}: {str(e)[:300]}"


class ToolFailure(Exception):
    pass


async def _call_raw(tools: dict, tool_name: str, timeout: float = TOOL_TIMEOUT, **kwargs) -> list:
    """Call a corpus tool; raise ToolFailure with a readable message on failure."""
    tool = tools.get(tool_name)
    if tool is None:
        raise ToolFailure(f"tool {tool_name} missing")
    try:
        raw = await asyncio.wait_for(tool(**kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        raise ToolFailure(f"{tool_name} timed out after {timeout}s")
    except Exception as e:  # noqa: BLE001
        raise ToolFailure(_err_text(e))
    return _parse_items(raw)


async def _call(tools: dict, tool_name: str, timeout: float = TOOL_TIMEOUT, **kwargs) -> list:
    """Call a corpus tool; never raise — return [] on any failure."""
    try:
        return await _call_raw(tools, tool_name, timeout=timeout, **kwargs)
    except ToolFailure as e:
        print(f"  !! {tool_name} failed: {e} kwargs={str(kwargs)[:100]}")
        return []


async def _batch_meta(tools: dict, cids: list, fields: str, depth: int = 0) -> list:
    """get_paper_batch that survives a bad id (e.g. post-cutoff paper) by
    recursively splitting the batch."""
    if not cids:
        return []
    try:
        return await _call_raw(tools, "get_paper_batch", ids=[f"CorpusId:{c}" for c in cids], fields=fields)
    except ToolFailure as e:
        if len(cids) == 1 or depth > 7:
            if depth <= 1:
                print(f"  !! get_paper_batch failed ({len(cids)} ids): {e}")
            return []
        mid = len(cids) // 2
        a, b = await asyncio.gather(_batch_meta(tools, cids[:mid], fields, depth + 1),
                                    _batch_meta(tools, cids[mid:], fields, depth + 1))
        return list(a) + list(b)


def _cid(p: dict):
    v = p.get("corpusId")
    if v is None:
        return None
    return str(v)


def _tldr_text(p: dict) -> str:
    t = p.get("tldr")
    if isinstance(t, dict):
        return (t.get("text") or "").strip()
    if isinstance(t, str):
        return t.strip()
    return ""


# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------
def _extract_json(text: str):
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find the first {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        s = text.find(opener)
        e = text.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


async def _llm_json(model, prompt: str, config: GenerateConfig | None = None, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            if config is not None:
                resp = await model.generate(prompt, config=config)
            else:
                resp = await model.generate(prompt)
            text = (resp.completion or "").strip()
            if not text:
                print(f"  !! empty LLM completion (attempt {attempt})")
                continue
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
            print(f"  !! unparsable LLM JSON (attempt {attempt}): {text[:150]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  !! LLM call failed (attempt {attempt}): {str(e)[:200]}")
    return None


# ----------------------------------------------------------------------------
# evidence
# ----------------------------------------------------------------------------
def _trim_passage(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # cut at a sentence boundary if one exists reasonably late
    m = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if m > limit * 0.5:
        return cut[:m + 1]
    sp = cut.rfind(" ")
    return cut[:sp] if sp > 0 else cut


def _build_evidence(c: dict) -> str:
    """Verbatim passages (title, abstract, tldr, snippets) joined by ' ... '."""
    parts = []
    budget = MAX_EVIDENCE_CHARS
    title = re.sub(r"\s+", " ", c.get("title") or "").strip()
    if title:
        parts.append(title)
        budget -= len(title) + 5
    abstract = c.get("abstract") or ""
    if abstract and budget > 80:
        pa = _trim_passage(abstract, min(budget, 1700))
        if pa:
            parts.append(pa)
            budget -= len(pa) + 5
    tl = c.get("tldr_text") or ""
    if tl and budget > 80:
        pt = _trim_passage(tl, budget)
        if pt and pt not in parts:
            parts.append(pt)
            budget -= len(pt) + 5
    for sn in c.get("snippets") or []:
        if budget < 120 or len(parts) >= 8:
            break
        ps = _trim_passage(sn, min(budget, 700))
        if ps and ps not in parts:
            parts.append(ps)
            budget -= len(ps) + 5
    return " ... ".join(parts)


# ----------------------------------------------------------------------------
# candidate bookkeeping
# ----------------------------------------------------------------------------
class Cands:
    def __init__(self):
        self.d: dict[str, dict] = {}

    def add(self, p: dict, rank: int, weight: float = 1.0, snippet: str | None = None):
        cid = _cid(p)
        if not cid:
            return None
        c = self.d.get(cid)
        if c is None:
            c = {
                "cid": cid, "title": p.get("title") or "", "abstract": p.get("abstract") or "",
                "tldr_text": _tldr_text(p), "year": p.get("year"), "venue": p.get("venue") or "",
                "citationCount": p.get("citationCount"), "authors": p.get("authors") or [],
                "snippets": [], "score": 0.0, "hits": 0, "best_rank": rank,
            }
            self.d[cid] = c
        else:
            if not c["abstract"] and p.get("abstract"):
                c["abstract"] = p["abstract"]
            if not c["tldr_text"]:
                c["tldr_text"] = _tldr_text(p)
            if not c["title"] and p.get("title"):
                c["title"] = p["title"]
            if c["year"] is None and p.get("year") is not None:
                c["year"] = p.get("year")
        c["score"] += weight / (rank + 6.0)
        c["hits"] += 1
        c["best_rank"] = min(c["best_rank"], rank)
        if snippet and snippet not in c["snippets"] and len(c["snippets"]) < 3:
            c["snippets"].append(snippet)
        return c

    def values(self):
        return list(self.d.values())


# ----------------------------------------------------------------------------
# SEMANTIC
# ----------------------------------------------------------------------------
SEMANTIC_PLAN_PROMPT = """You are helping a scientific literature search engine find ALL papers relevant to a user's request.

User request: {query}

Produce a JSON object with these keys:
- "keyword_queries": a list of 7 short keyword search queries (2-7 words each, no punctuation, no question words). Make them DIVERSE: cover the core topic phrased different ways, standard terminology and synonyms, well-known method or model names associated with the topic, and important sub-topics. Each must be a plain noun phrase a keyword search engine understands.
- "snippet_queries": a list of 2 natural-language sentences (one declarative sentence each) that describe what a relevant paper would do or claim; these go to a passage-retrieval engine.
- "criteria": a list of 2-4 short statements of what a paper MUST satisfy to be a perfect match for the request (include any exclusions the user stated, e.g. "not a survey").

Return only the JSON object."""

GRADE_PROMPT = """You are a strict relevance judge for a literature search.

User request: {query}

A paper is relevant only if it satisfies ALL of these criteria:
{criteria}

Grade each candidate below using ONLY its title/abstract text:
3 = satisfies every criterion clearly (perfect match for the request)
2 = satisfies most criteria; one is only partially/implicitly satisfied
1 = related topic but misses a key criterion
0 = not relevant

Candidates:
{cands}

Return ONLY a JSON object mapping candidate index (as a string) to grade (integer 0-3), e.g. {{"0": 3, "1": 0}}. Include every index."""


def _default_keywords(query: str) -> list[str]:
    q = re.sub(r"[^\w\s-]", " ", query)
    q = re.sub(r"\b(could|would|can|you|please|suggest|recommend|provide|some|any|papers?|research|study|studies|work|references?|me|i|am|looking|for|are|there|what|which|how|is|that|the|a|an|of|on|in|to|about|has|been|done|key|using|used|with)\b", " ", q, flags=re.I)
    words = q.split()
    return [" ".join(words[:8])] if words else [query[:60]]


async def solve_semantic(state: TaskState, tools: dict, query: str) -> list[dict]:
    t0 = time.time()
    plan = await _llm_json(GPT_5_4_MINI, SEMANTIC_PLAN_PROMPT.format(query=query))
    kw_queries, sn_queries, criteria = [], [], []
    if isinstance(plan, dict):
        kw_queries = [str(x) for x in (plan.get("keyword_queries") or []) if str(x).strip()][:8]
        sn_queries = [str(x) for x in (plan.get("snippet_queries") or []) if str(x).strip()][:2]
        criteria = [str(x) for x in (plan.get("criteria") or []) if str(x).strip()][:5]
    if not kw_queries:
        kw_queries = _default_keywords(query)
    if not sn_queries:
        sn_queries = [query]
    if not criteria:
        criteria = [f"The paper matches the request: {query}"]
    print(f"  plan: kw={kw_queries} sn={sn_queries} criteria={criteria}")

    cands = Cands()

    async def kw_search(q, weight):
        items = await _call(tools, "search_papers_by_relevance", keyword=q, fields=PAPER_FIELDS, limit=100)
        return ("kw", q, weight, items)

    async def sn_search(q):
        items = await _call(tools, "snippet_search", timeout=SNIPPET_TIMEOUT, query=q, limit=60)
        return ("sn", q, 1.0, items)

    jobs = [kw_search(q, 1.3 if i == 0 else 1.0) for i, q in enumerate(kw_queries)]
    jobs += [sn_search(q) for q in sn_queries]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    need_meta = set()
    for r in results:
        if isinstance(r, Exception) or r is None:
            continue
        kind, q, weight, items = r
        print(f"  {kind} {q[:50]!r}: {len(items)} items")
        for rank, it in enumerate(items):
            if kind == "kw":
                cands.add(it, rank, weight)
            else:
                paper = it.get("paper") or {}
                sn = (it.get("snippet") or {}).get("text") or ""
                c = cands.add(paper, rank, 0.8, snippet=sn)
                if c is not None and not c["abstract"]:
                    need_meta.add(c["cid"])

    # fetch abstracts for snippet-only candidates
    need = [cid for cid in need_meta if not cands.d[cid]["abstract"]]
    if need:
        chunks = [need[i:i + 50] for i in range(0, len(need), 50)]
        metas = await asyncio.gather(*[_batch_meta(tools, ch, PAPER_FIELDS) for ch in chunks],
                                     return_exceptions=True)
        for m in metas:
            if isinstance(m, Exception):
                continue
            for p in m:
                cid = _cid(p)
                if cid in cands.d:
                    c = cands.d[cid]
                    if p.get("abstract"):
                        c["abstract"] = p["abstract"]
                    if not c["tldr_text"]:
                        c["tldr_text"] = _tldr_text(p)
                    if c["year"] is None:
                        c["year"] = p.get("year")

    allc = cands.values()
    allc.sort(key=lambda c: (-c["score"], c["best_rank"]))
    print(f"  {len(allc)} unique candidates after {time.time()-t0:.0f}s")
    if not allc:
        return []

    # ---- LLM grading of the top candidates -------------------------------
    to_grade = allc[:GRADE_CANDIDATES]
    crit_txt = "\n".join(f"- {c}" for c in criteria)

    async def grade_batch(batch, offset):
        lines = []
        for i, c in enumerate(batch):
            body = c["abstract"] or c["tldr_text"] or (c["snippets"][0] if c["snippets"] else "")
            body = _trim_passage(body, GRADE_ABSTRACT_CHARS)
            lines.append(f"[{i}] Title: {c['title'][:200]}\n    Abstract: {body}")
        prompt = GRADE_PROMPT.format(query=query, criteria=crit_txt, cands="\n".join(lines))
        out = await _llm_json(GPT_5_4_MINI, prompt, retries=1)
        grades = {}
        if isinstance(out, dict):
            for k, v in out.items():
                try:
                    idx = int(str(k).strip().strip("[]"))
                    g = int(v)
                except (ValueError, TypeError):
                    continue
                if 0 <= idx < len(batch):
                    grades[idx] = max(0, min(3, g))
        elif isinstance(out, list):
            for i, v in enumerate(out[:len(batch)]):
                try:
                    grades[i] = max(0, min(3, int(v)))
                except (ValueError, TypeError):
                    pass
        for i, c in enumerate(batch):
            c["grade"] = grades.get(i)
        return len(grades)

    batches = [to_grade[i:i + GRADE_BATCH] for i in range(0, len(to_grade), GRADE_BATCH)]
    graded_counts = await asyncio.gather(*[grade_batch(b, i * GRADE_BATCH) for i, b in enumerate(batches)],
                                         return_exceptions=True)
    n_graded = sum(x for x in graded_counts if isinstance(x, int))
    print(f"  graded {n_graded}/{len(to_grade)} candidates in {len(batches)} batches ({time.time()-t0:.0f}s)")

    def sort_key(c):
        g = c.get("grade")
        if g is None:
            g = 0.5  # ungraded (LLM failure): between 0 and 1
        return (-g, -c["score"], c["best_rank"])

    ordered = sorted(allc, key=sort_key)
    dist = {}
    for c in ordered:
        dist[c.get("grade")] = dist.get(c.get("grade"), 0) + 1
    print(f"  grade distribution: {dist}")

    # ---- extra evidence for the head of the list via scoped snippet search --
    head = [c for c in ordered[:40] if (c.get("grade") or 0) >= 2]
    if head:
        ids = ",".join(f"CorpusId:{c['cid']}" for c in head)
        items = await _call(tools, "snippet_search", timeout=90, query=sn_queries[0], limit=60, paper_ids=ids)
        n_added = 0
        for it in items:
            cid = _cid(it.get("paper") or {})
            sn = (it.get("snippet") or {}).get("text") or ""
            if cid in cands.d and sn and sn not in cands.d[cid]["snippets"] and len(cands.d[cid]["snippets"]) < 3:
                cands.d[cid]["snippets"].append(sn)
                n_added += 1
        print(f"  scoped snippet evidence: {n_added} passages for {len(head)} head papers")

    results = []
    for c in ordered[:MAX_SUBMIT]:
        results.append({"paper_id": c["cid"], "markdown_evidence": _build_evidence(c)})
    return results


# ----------------------------------------------------------------------------
# SPECIFIC
# ----------------------------------------------------------------------------
SPECIFIC_PLAN_PROMPT = """A user is asking for a specific, known paper (or a small known set of papers).

Request: {query}

Return a JSON object:
- "titles": list of up to 3 guesses of the EXACT title of the paper(s) requested (best guess first). If you know the paper, give its real title.
- "keywords": list of 2-3 short keyword queries (2-8 words) that would retrieve it in a keyword search (e.g. the model/method name plus distinctive words).
- "expected_count": integer, how many distinct papers the user is asking for (usually 1).
Return only JSON."""

SPECIFIC_PICK_PROMPT = """A user asked for a specific paper.

Request: {query}

Candidates:
{cands}

Which candidate(s) are the paper(s) the user is asking for? Prefer the original paper over later versions, surveys, or follow-ups; prefer the single best match unless the request clearly asks for several papers. Return ONLY a JSON object: {{"picks": [indices in order of confidence], "confidence": "high"|"medium"|"low"}}. If none match, return {{"picks": [], "confidence": "low"}}."""


def _fmt_authors(p: dict, n: int = 3) -> str:
    au = p.get("authors") or []
    names = []
    for a in au[:n]:
        if isinstance(a, dict):
            names.append(a.get("name") or "")
        else:
            names.append(str(a))
    s = ", ".join(x for x in names if x)
    if len(au) > n:
        s += " et al."
    return s


async def solve_specific(state: TaskState, tools: dict, query: str) -> list[dict]:
    plan = await _llm_json(GPT_5_4, SPECIFIC_PLAN_PROMPT.format(query=query))
    titles, keywords, expected = [], [], 1
    if isinstance(plan, dict):
        titles = [str(x) for x in (plan.get("titles") or []) if str(x).strip()][:3]
        keywords = [str(x) for x in (plan.get("keywords") or []) if str(x).strip()][:3]
        try:
            expected = max(1, int(plan.get("expected_count") or 1))
        except (ValueError, TypeError):
            expected = 1
    if not keywords:
        keywords = _default_keywords(query)
    print(f"  plan: titles={titles} keywords={keywords} expected={expected}")

    cands = Cands()
    jobs = [_call(tools, "search_paper_by_title", timeout=75, title=t, fields=PAPER_FIELDS) for t in titles]
    jobs += [_call(tools, "search_papers_by_relevance", keyword=k, fields=PAPER_FIELDS, limit=20) for k in keywords]
    res = await asyncio.gather(*jobs, return_exceptions=True)
    for j, r in enumerate(res):
        if isinstance(r, Exception):
            continue
        for rank, p in enumerate(r):
            if not p.get("paperId") and not p.get("corpusId"):
                continue
            cands.add(p, rank, 2.0 if j < len(titles) else 1.0)
    allc = cands.values()
    allc.sort(key=lambda c: (-c["score"], c["best_rank"]))
    allc = allc[:30]
    print(f"  {len(allc)} candidates")
    if not allc:
        return []

    lines = []
    for i, c in enumerate(allc):
        lines.append(f"[{i}] {c['title'][:160]} ({c['year']}; {c['venue'][:40]}; cites={c['citationCount']}; {_fmt_authors(c)})\n"
                     f"    {_trim_passage(c['abstract'] or c['tldr_text'], 250)}")
    pick = await _llm_json(GPT_5_4, SPECIFIC_PICK_PROMPT.format(query=query, cands="\n".join(lines)))
    picks = []
    if isinstance(pick, dict):
        for x in pick.get("picks") or []:
            try:
                i = int(x)
            except (ValueError, TypeError):
                continue
            if 0 <= i < len(allc) and i not in picks:
                picks.append(i)
        print(f"  picks={picks} confidence={pick.get('confidence')}")
    if not picks:
        picks = [0]
    picks = picks[:max(expected, 1) if expected > 1 else 1] if (pick or {}).get("confidence") == "high" else picks[:max(expected, 2)]
    return [{"paper_id": allc[i]["cid"], "markdown_evidence": _build_evidence(allc[i])} for i in picks]


# ----------------------------------------------------------------------------
# METADATA
# ----------------------------------------------------------------------------
METADATA_PLAN_PROMPT = """Parse this literature-search request into structured filters.

Request: {query}

Return a JSON object with exactly these keys (use null / [] when not specified):
- "authors": list of author full names the papers must be written by (people, not organisations).
- "venue_patterns": list of lowercase substrings that identify the required publication venue(s). Expand families: e.g. "Nature portfolio" -> ["nature", "npj"]; "NeurIPS" -> ["neurips", "neural information processing systems"]; "ACL" -> ["association for computational linguistics", "acl"]; "CVPR" -> ["cvpr", "computer vision and pattern recognition"]; "ICML" -> ["icml", "international conference on machine learning"]; "ICLR" -> ["iclr", "international conference on learning representations"]; "EMNLP" -> ["emnlp", "empirical methods in natural language processing"]; "arXiv" -> ["arxiv"].
- "year_min": integer or null (inclusive). "after 2020" means year_min 2021? NO — interpret loosely as inclusive: "after 2020" -> 2020. "since 2019" -> 2019. "in 2021" -> 2021 with year_max 2021.
- "year_max": integer or null (inclusive).
- "min_citations": integer or null (papers must have at least this many citations; "more than 50 citations" -> 51).
- "min_authors": integer or null ("more than 3 authors" -> 4). "max_authors": integer or null.
- "cites_paper": title (or well-known name) of a paper that results must CITE, else null. E.g. "papers citing the RoBERTa paper" -> "RoBERTa: A Robustly Optimized BERT Pretraining Approach".
- "cites_paper_keyword": a short distinctive keyword (1-3 words, e.g. "RoBERTa", "DistilBERT") for that cited paper, else null.
- "topic_keywords": list of 0-4 short keyword queries describing a required TOPIC of the papers (only if the request constrains the subject matter; author/venue/year/citation constraints are NOT topics).
- "topic_description": one sentence describing the topical requirement, or null.
Return only JSON."""

TOPIC_FILTER_PROMPT = """Decide which candidate papers are about the required topic.

Required topic: {topic}

Candidates:
{cands}

Return ONLY a JSON object: {{"relevant": [indices of candidates that clearly match the topic]}}."""


def _venue_match(c: dict, patterns: list[str]) -> bool:
    if not patterns:
        return True
    v = (c.get("venue") or "").lower()
    j = c.get("journal")
    jn = (j.get("name") or "").lower() if isinstance(j, dict) else ""
    for pat in patterns:
        pat = pat.lower().strip()
        if not pat:
            continue
        for text in (v, jn):
            if not text:
                continue
            if len(pat) <= 6:
                # short acronym — match as a whole word
                if re.search(r"(?<![a-z])" + re.escape(pat) + r"(?![a-z])", text):
                    return True
            elif pat in text:
                return True
    return False


def _author_count(c: dict):
    au = c.get("authors")
    if isinstance(au, list):
        return len(au)
    return None


def _passes(c: dict, f: dict, check_venue=True) -> bool:
    y = c.get("year")
    if f.get("year_min") is not None:
        if y is None or y < f["year_min"]:
            return False
    if f.get("year_max") is not None:
        if y is None or y > f["year_max"]:
            return False
    if f.get("min_citations") is not None:
        cc = c.get("citationCount")
        if cc is None or cc < f["min_citations"]:
            return False
    if f.get("min_authors") is not None:
        n = _author_count(c)
        if n is None or n < f["min_authors"]:
            return False
    if f.get("max_authors") is not None:
        n = _author_count(c)
        if n is None or n > f["max_authors"]:
            return False
    if check_venue and not _venue_match(c, f.get("venue_patterns") or []):
        return False
    return True


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z ]", "", (s or "").lower()).strip()


def _author_compatible(query_name: str, cand_name: str) -> bool:
    q = _norm_name(query_name).split()
    c = _norm_name(cand_name).split()
    if not q or not c:
        return False
    if q[-1] != c[-1]:
        return False
    # first initial must agree if both present
    if len(q) > 1 and len(c) > 1 and q[0][0] != c[0][0]:
        return False
    return True


async def solve_metadata(state: TaskState, tools: dict, query: str) -> list[dict]:
    t0 = time.time()
    f = await _llm_json(GPT_5_4, METADATA_PLAN_PROMPT.format(query=query))
    if not isinstance(f, dict):
        f = {}
    for k in ("year_min", "year_max", "min_citations", "min_authors", "max_authors"):
        v = f.get(k)
        try:
            f[k] = int(v) if v is not None else None
        except (ValueError, TypeError):
            f[k] = None
    authors = [str(a) for a in (f.get("authors") or []) if str(a).strip()]
    venue_patterns = [str(v) for v in (f.get("venue_patterns") or []) if str(v).strip()]
    f["venue_patterns"] = venue_patterns
    cites_paper = f.get("cites_paper")
    cites_kw = f.get("cites_paper_keyword")
    topic_keywords = [str(k) for k in (f.get("topic_keywords") or []) if str(k).strip()][:4]
    topic_desc = f.get("topic_description")
    print(f"  filters: {json.dumps(f)[:600]}")

    pool: dict[str, dict] = {}
    fields = "title,abstract,corpusId,year,venue,citationCount,authors,publicationDate"
    author_fields = fields + ",journal"

    def add_pool(p: dict):
        cid = _cid(p)
        if not cid:
            return
        if cid not in pool:
            pool[cid] = {
                "cid": cid, "title": p.get("title") or "", "abstract": p.get("abstract") or "",
                "tldr_text": _tldr_text(p), "year": p.get("year"), "venue": p.get("venue") or "",
                "journal": p.get("journal"), "citationCount": p.get("citationCount"),
                "authors": p.get("authors"), "snippets": [],
            }

    # ---- author-based retrieval -----------------------------------------
    if authors:
        for name in authors:
            recs = await _call(tools, "search_authors_by_name", name=name, fields="name,paperCount", limit=15)
            ids = [(r.get("authorId"), r.get("paperCount") or 0, r.get("name"))
                   for r in recs if r.get("authorId") and _author_compatible(name, r.get("name") or "")]
            ids.sort(key=lambda x: -x[1])
            ids = ids[:6]
            print(f"  author {name!r}: ids={ids}")
            res = await asyncio.gather(*[
                _call(tools, "get_author_papers", author_id=str(aid), paper_fields=author_fields, limit=1000)
                for aid, _, _ in ids
            ], return_exceptions=True)
            for r in res:
                if isinstance(r, Exception):
                    continue
                for p in r:
                    add_pool(p)
        print(f"  author pool: {len(pool)}")

    # ---- citation-based retrieval ----------------------------------------
    target_pid = None
    if cites_paper:
        hits = await _call(tools, "search_paper_by_title", title=str(cites_paper), fields="title,corpusId,year,citationCount")
        hits = [h for h in hits if h.get("paperId")]
        if not hits and cites_kw:
            hits = await _call(tools, "search_papers_by_relevance", keyword=str(cites_kw), fields="title,corpusId,year,citationCount", limit=5)
            hits = [h for h in hits if h.get("paperId")]
            hits.sort(key=lambda h: -(h.get("citationCount") or 0))
        if hits:
            target = hits[0]
            target_pid = target.get("paperId")
            target_cid = _cid(target)
            print(f"  cited target: {target.get('title')!r} cid={target_cid} cites={target.get('citationCount')}")
            cit = await _call(tools, "get_citations", paper_id=f"CorpusId:{target_cid}", fields=fields, limit=1000)
            n0 = len(pool)
            for it in cit:
                p = it.get("citingPaper") if isinstance(it, dict) and "citingPaper" in it else it
                if isinstance(p, dict):
                    add_pool(p)
            print(f"  get_citations: {len(cit)} -> pool +{len(pool)-n0}")

            # augment: keyword / snippet search for the cited paper's name, then verify via references
            if cites_kw:
                venues_arg = {}
                jobs = [
                    _call(tools, "search_papers_by_relevance", keyword=str(cites_kw), fields=fields, limit=100),
                    _call(tools, "search_papers_by_relevance", keyword=f"{cites_kw} {str(cites_paper)[:60]}", fields=fields, limit=100),
                    _call(tools, "snippet_search", timeout=120, query=f"{cites_kw}", limit=100),
                    _call(tools, "snippet_search", timeout=120, query=f"we use {cites_kw} as a baseline model", limit=100),
                ]
                if venue_patterns:
                    for vp in venue_patterns[:2]:
                        jobs.append(_call(tools, "search_papers_by_relevance", keyword=f"{cites_kw} {vp}", fields=fields, limit=100))
                if topic_keywords:
                    for tk in topic_keywords[:2]:
                        jobs.append(_call(tools, "search_papers_by_relevance", keyword=f"{cites_kw} {tk}", fields=fields, limit=100))
                res = await asyncio.gather(*jobs, return_exceptions=True)
                extra: dict[str, dict] = {}
                for r in res:
                    if isinstance(r, Exception):
                        continue
                    for it in r:
                        p = it.get("paper") if isinstance(it, dict) and "paper" in it else it
                        cid = _cid(p or {})
                        if cid and cid not in pool and cid not in extra:
                            extra[cid] = p
                # pre-filter by cheap constraints where metadata is present (snippet hits lack metadata)
                to_verify = [cid for cid, p in extra.items()
                             if p.get("year") is None or _passes(p, f, check_venue=False)]
                print(f"  augmentation candidates to verify: {len(to_verify)}")
                if to_verify and target_pid:
                    chunks = [to_verify[i:i + 40] for i in range(0, len(to_verify), 40)]
                    metas = await asyncio.gather(*[_batch_meta(tools, ch, fields + ",references") for ch in chunks],
                                                 return_exceptions=True)
                    n_ver = 0
                    for m in metas:
                        if isinstance(m, Exception):
                            continue
                        for p in m:
                            refs = p.get("references") or []
                            if any((r or {}).get("paperId") == target_pid for r in refs if isinstance(r, dict)):
                                add_pool(p)
                                n_ver += 1
                    print(f"  verified citers via references: {n_ver}")
        else:
            print("  !! could not resolve cited paper")

    # ---- topic / venue-only retrieval ------------------------------------
    if not authors and not cites_paper:
        kws = topic_keywords or _default_keywords(query)
        venues_csv = None
        res = await asyncio.gather(*[
            _call(tools, "search_papers_by_relevance", keyword=k, fields=fields, limit=100) for k in kws
        ], return_exceptions=True)
        for r in res:
            if isinstance(r, Exception):
                continue
            for p in r:
                add_pool(p)
        print(f"  topic pool: {len(pool)}")

    # ---- post-filter -----------------------------------------------------
    allc = list(pool.values())
    passing = [c for c in allc if _passes(c, f)]
    print(f"  pool {len(allc)} -> passing filters {len(passing)} ({time.time()-t0:.0f}s)")
    if not passing and venue_patterns and allc:
        # venue strings may be formatted differently; show what we have for debugging
        sample = sorted({(c.get('venue') or '')[:40] for c in allc})[:15]
        print(f"  venue sample: {sample}")

    # ---- topic filter (LLM) if the request constrains the subject ---------
    if passing and topic_desc and (authors or cites_paper) and topic_keywords:
        keep = []
        batches = [passing[i:i + 50] for i in range(0, len(passing), 50)]

        async def tf(batch):
            lines = [f"[{i}] {c['title'][:150]} — {_trim_passage(c['abstract'] or c['tldr_text'], 200)}" for i, c in enumerate(batch)]
            out = await _llm_json(GPT_5_4_MINI, TOPIC_FILTER_PROMPT.format(topic=topic_desc, cands="\n".join(lines)))
            sel = set()
            if isinstance(out, dict):
                for x in out.get("relevant") or []:
                    try:
                        sel.add(int(x))
                    except (ValueError, TypeError):
                        pass
            return [c for i, c in enumerate(batch) if i in sel]

        outs = await asyncio.gather(*[tf(b) for b in batches], return_exceptions=True)
        for o in outs:
            if isinstance(o, list):
                keep.extend(o)
        print(f"  topic filter: {len(passing)} -> {len(keep)}")
        if keep:
            passing = keep

    passing.sort(key=lambda c: -(c.get("citationCount") or 0))
    return [{"paper_id": c["cid"], "markdown_evidence": _build_evidence(c)} for c in passing[:MAX_SUBMIT]]


# ----------------------------------------------------------------------------
# fallback
# ----------------------------------------------------------------------------
async def fallback(tools: dict, query: str) -> list[dict]:
    kws = _default_keywords(query)
    items = await _call(tools, "search_papers_by_relevance", keyword=kws[0], fields=PAPER_FIELDS, limit=50)
    out = []
    for p in items:
        cid = _cid(p)
        if cid:
            out.append({"paper_id": cid, "markdown_evidence": _build_evidence({
                "title": p.get("title"), "abstract": p.get("abstract"), "tldr_text": _tldr_text(p), "snippets": []})})
    return out


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        t0 = time.time()
        query = state.metadata.get("raw_query") or state.input_text
        score_type = state.metadata.get("score_type", "")
        print(f"[{state.sample_id}] score_type={score_type} query={query[:120]!r}")
        tools = _tools_by_name(state)
        results: list[dict] = []
        try:
            if score_type == "specific_f1":
                results = await solve_specific(state, tools, query)
            elif score_type == "metadata_f1":
                results = await solve_metadata(state, tools, query)
            else:
                results = await solve_semantic(state, tools, query)
        except Exception:  # noqa: BLE001
            print("  !! solver crashed:\n" + traceback.format_exc()[-1500:])
        if not results:
            print("  empty result set -> fallback keyword search")
            try:
                results = await fallback(tools, query)
            except Exception:  # noqa: BLE001
                print("  !! fallback crashed:\n" + traceback.format_exc()[-800:])
                results = []

        # final sanitation: string ids, dedupe, evidence key present, cap
        seen = set()
        clean = []
        for r in results:
            pid = str(r.get("paper_id") or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            clean.append({"paper_id": pid, "markdown_evidence": str(r.get("markdown_evidence") or "")[:2500]})
            if len(clean) >= MAX_SUBMIT:
                break
        state.output.completion = json.dumps({"output": {"query_id": state.sample_id, "results": clean}})
        print(f"  submitted {len(clean)} papers in {time.time()-t0:.0f}s")
        return state

    return solve
