"""PaperFindingBench solver: plan -> multi-source retrieval -> grade -> evidence.

Three paths keyed on state.metadata["score_type"]:
  semantic_f1 : many keyword + snippet searches, per-criterion LLM grading of
                every candidate (mimicking the judge's 0/1/3 rubric), a long
                ranked list, and verbatim evidence (title + abstract + body
                snippets) for each paper.
  specific_f1 : LLM proposes real titles, title/keyword/snippet search, a
                strong LLM picks the paper(s) actually referred to.
  metadata_f1 : LLM parses the constraints into a spec; author / citation /
                venue tools build the candidate set; deterministic filters
                plus an LLM venue match produce the answer.
"""

import asyncio
import json
import re
import time

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import CLAUDE_OPUS_4_8, CLAUDE_SONNET_4_6, GPT_5_4_MINI

PLAN_MODEL = CLAUDE_OPUS_4_8
PICK_MODEL = CLAUDE_OPUS_4_8
MID_MODEL = CLAUDE_SONNET_4_6
GRADE_MODEL = GPT_5_4_MINI

PAPER_FIELDS = "title,abstract,corpusId,year,venue,citationCount,authors,tldr"
META_FIELDS = "title,corpusId,year,venue,citationCount,authors,publicationDate"
MAX_GRADE = 340          # semantic candidates sent to the grader
GRADE_BATCH = 12
MAX_SUBMIT = 250
EVIDENCE_CAP = 2450
QUERY_BUDGET_S = 20 * 60  # soft budget; hard limit is 29 min


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
class Ctx:
    def __init__(self, state):
        self.state = state
        self.tools = {ToolDef(t).name: t for t in state.tools}
        self.sem = asyncio.Semaphore(8)
        self.t0 = time.time()
        self.fail = {}

    def elapsed(self):
        return time.time() - self.t0

    def left(self):
        return QUERY_BUDGET_S - self.elapsed()


def _parse(raw):
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
            docs.extend(d for d in doc["data"] if d)
        elif isinstance(doc, list):
            docs.extend(d for d in doc if d)
        elif doc:
            docs.append(doc)
    return docs


async def call(ctx, tool_name, /, timeout=150, **kw):
    """Call a corpus tool; returns parsed list (empty on failure)."""
    if ctx.left() < 0:
        return []
    async with ctx.sem:
        try:
            raw = await asyncio.wait_for(ctx.tools[tool_name](**kw), timeout=timeout)
            return _parse(raw)
        except Exception as e:  # noqa: BLE001
            ctx.fail[tool_name] = ctx.fail.get(tool_name, 0) + 1
            if ctx.fail[tool_name] <= 3:
                inner = getattr(e, "exceptions", None)
                detail = "; ".join(f"{type(x).__name__}: {str(x)[:200]}" for x in inner) if inner else str(e)[:200]
                print(f"  tool {tool_name} failed: {type(e).__name__}: {detail}")
            return []


async def llm(handle, prompt, effort=None, max_tokens=None):
    cfg = {}
    if effort:
        cfg["reasoning_effort"] = effort
    if max_tokens:
        cfg["max_tokens"] = max_tokens
    for attempt in range(2):
        try:
            if cfg:
                resp = await handle.generate(prompt, config=GenerateConfig(**cfg))
            else:
                resp = await handle.generate(prompt)
            text = (resp.completion or "").strip()
            if text:
                return text
            print(f"  LLM empty completion (attempt {attempt})")
        except Exception as e:  # noqa: BLE001
            print(f"  LLM error: {type(e).__name__}: {str(e)[:160]}")
    return ""


def jparse(text):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def cid_of(p):
    c = p.get("corpusId")
    return str(c) if c not in (None, "") else ""


def tldr_of(p):
    t = p.get("tldr")
    if isinstance(t, dict):
        return (t.get("text") or "").strip()
    return t.strip() if isinstance(t, str) else ""


def clean_kw(q):
    q = re.sub(r"[\"'?!:;()\[\]{}]", " ", str(q))
    return re.sub(r"\s+", " ", q).strip()


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class Pool:
    """Within-query bookkeeping of retrieved papers."""

    def __init__(self):
        self.papers = {}    # cid -> merged paper dict
        self.snips = {}     # cid -> list of snippet texts (body passages)
        self.rrf = {}       # cid -> retrieval score

    def add(self, p, rank=None, weight=1.0):
        cid = cid_of(p)
        if not cid:
            return ""
        cur = self.papers.setdefault(cid, {"corpusId": cid})
        for k, v in p.items():
            if k != "corpusId" and v not in (None, "", [], {}) and not cur.get(k):
                cur[k] = v
        if rank is not None:
            self.rrf[cid] = self.rrf.get(cid, 0.0) + weight / (20.0 + rank)
        return cid

    def add_snippet(self, d, rank=None, weight=1.0):
        paper = d.get("paper") or {}
        snip = d.get("snippet") or {}
        cid = self.add({"corpusId": paper.get("corpusId"), "title": paper.get("title")},
                       rank, weight)
        text = (snip.get("text") or "").strip()
        if cid and text:
            lst = self.snips.setdefault(cid, [])
            if text not in lst:
                lst.append(text)
        return cid


async def kw_search(ctx, pool, q, limit=100, venues=None, fields=PAPER_FIELDS, weight=1.0):
    q = clean_kw(q)
    if not q:
        return []
    kw = dict(keyword=q, fields=fields, limit=limit)
    if venues:
        kw["venues"] = venues
    hits = await call(ctx, "search_papers_by_relevance", **kw)
    ids = [pool.add(h, r, weight) for r, h in enumerate(hits)]
    return [i for i in ids if i]


async def snip_search(ctx, pool, q, limit=60, paper_ids=None, timeout=150, weight=1.0):
    kw = dict(query=q, limit=limit)
    if paper_ids:
        kw["paper_ids"] = paper_ids
    hits = await call(ctx, "snippet_search", timeout=timeout, **kw)
    ids = [pool.add_snippet(h, None if paper_ids else r, weight) for r, h in enumerate(hits)]
    return [i for i in ids if i]


async def title_search(ctx, pool, title, fields=PAPER_FIELDS, weight=3.0):
    title = str(title).strip()
    if not title:
        return ""
    hits = await call(ctx, "search_paper_by_title", title=title, fields=fields)
    for h in hits:
        if h.get("paperId"):
            return pool.add(h, 0, weight)
    return ""


async def batch_get(ctx, ids, fields, chunk=25):
    out = {}

    async def one(part, depth=0):
        if not part:
            return
        before = ctx.fail.get("get_paper_batch", 0)
        docs = await call(ctx, "get_paper_batch", timeout=120,
                          ids=[f"CorpusId:{i}" for i in part], fields=fields)
        if not docs and ctx.fail.get("get_paper_batch", 0) > before and len(part) > 1 and depth < 3:
            # the call itself failed (size / timeout): retry as two smaller requests
            mid = len(part) // 2
            await asyncio.gather(one(part[:mid], depth + 1), one(part[mid:], depth + 1))
            return
        for d in docs:
            if isinstance(d, dict) and cid_of(d):
                out[cid_of(d)] = d

    ids = list(ids)
    await asyncio.gather(*[one(ids[i:i + chunk]) for i in range(0, len(ids), chunk)])
    return out


def emit(state, results):
    seen, out = set(), []
    for r in results:
        pid = str(r["paper_id"])
        if pid and pid not in seen:
            seen.add(pid)
            out.append({"paper_id": pid, "markdown_evidence": r.get("markdown_evidence") or ""})
    out = out[:MAX_SUBMIT]
    state.output.completion = json.dumps({"output": {"query_id": state.sample_id, "results": out}})
    print(f"  submitted {len(out)} papers")
    return state


# ----------------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------------
PLAN_PROMPT = """You are planning a scientific literature search over the Semantic Scholar corpus.

User request: {query}
Request category: {kind}

Return ONE JSON object (no prose) with these keys:
- "keyword_queries": {nkw} diverse keyword / noun-phrase queries for a lenient keyword search engine (3-8 words each, no question words, no punctuation, no boolean operators). Cover synonyms, alternative terminology, sub-topics, specific well-known method/system/dataset names, and narrower as well as broader phrasings.
- "snippet_queries": 3 natural-language sentences (the kind of sentence a relevant paper's abstract or body would contain) for a dense passage-retrieval engine.
- "candidate_titles": exact titles of real papers you know that match the request (up to 12; [] if none known). For a request like "the X paper", give the actual full title(s) of the paper(s) that introduced X (several if X is an ambiguous name used by different papers).
- "criteria": 2-4 relevance criteria that a paper must satisfy, each {{"name": str, "description": "The paper must ...", "weight": float}}; weights sum to 1. Decompose the request into its distinct required aspects (e.g. domain, technique, specific role/application), as a meticulous relevance judge would.
- "authors": list of author full names the request requires (else []).
- "venue_constraint": the venue requirement in the user's words, or null.
- "venue_names": if there is a venue requirement, likely exact Semantic Scholar venue strings (full official name AND common abbreviation, e.g. "Neural Information Processing Systems", "NeurIPS"); else [].
- "year_min", "year_max": inclusive integer bounds implied by the request ("after 2022" -> year_min 2023; "since 2020" -> year_min 2020), or null.
- "min_citations", "max_citations": inclusive integer bounds on the paper's own citation count ("more than 50 citations" -> min_citations 51; "at least 30" -> 30), or null.
- "min_authors", "max_authors": inclusive bounds on number of authors ("more than 3 authors" -> min_authors 4), or null.
- "cited_paper_titles": if the request asks for papers that CITE some paper, the exact full title(s) of that cited paper (else []).
- "citing_topic_queries": if cited_paper_titles is non-empty, 30 diverse keyword queries (2-6 words) spanning the research areas, tasks, and applications whose papers typically cite it (restricted to the request's topic if it has one); else [].
- "topic": the topical/content requirement of the request beyond metadata filters, as a short phrase, or null if the request is purely metadata.
"""


async def make_plan(query, kind):
    nkw = 12 if kind == "semantic_f1" else 6
    text = await llm(PLAN_MODEL, PLAN_PROMPT.format(query=query, kind=kind, nkw=nkw))
    plan = jparse(text)
    if not isinstance(plan, dict):
        print("  plan parse failed; retrying with mid model")
        plan = jparse(await llm(MID_MODEL, PLAN_PROMPT.format(query=query, kind=kind, nkw=nkw)))
    if not isinstance(plan, dict):
        plan = {}
    for k in ("keyword_queries", "snippet_queries", "candidate_titles", "criteria", "authors",
              "venue_names", "cited_paper_titles", "citing_topic_queries"):
        if not isinstance(plan.get(k), list):
            plan[k] = []
    if not plan["keyword_queries"]:
        stripped = re.sub(r"^(could you|can you|please|show me|what are|what papers|are there any|find|give me)\b",
                          "", query.strip(), flags=re.I)
        plan["keyword_queries"] = [clean_kw(stripped)[:120]]
    crit = [c for c in plan["criteria"] if isinstance(c, dict) and c.get("description")]
    if not crit:
        crit = [{"name": "Relevance", "description": f"The paper must match the request: {query}", "weight": 1.0}]
    tot = 0.0
    for c in crit:
        try:
            c["weight"] = max(0.05, float(c.get("weight") or 0))
        except (TypeError, ValueError):
            c["weight"] = 1.0
        tot += c["weight"]
    for c in crit:
        c["weight"] /= tot
    plan["criteria"] = crit
    return plan


# ----------------------------------------------------------------------------
# semantic path
# ----------------------------------------------------------------------------
GRADE_PROMPT = """You are a strict relevance judge for a literature search.

Search request: {query}

Relevance criteria:
{criteria}

For EACH paper below, rate EVERY criterion using only the text shown:
 3 = the text clearly and explicitly satisfies the criterion
 1 = partially / indirectly / only by implication
 0 = not satisfied or not mentioned

Papers:
{papers}

Reply with ONLY a JSON object mapping each paper number to its list of {nc} ratings in criteria order, e.g. {{"0": [3, 1, 0], "1": [3, 3, 3]}}. Include every paper number."""


def paper_blurb(pool, cid, abs_chars=1100, snip_chars=450):
    p = pool.papers[cid]
    parts = [f"Title: {p.get('title') or '(unknown)'}"]
    if p.get("year"):
        parts[0] += f" ({p.get('year')})"
    ab = (p.get("abstract") or "").strip() or tldr_of(p)
    if ab:
        parts.append("Abstract: " + ab[:abs_chars])
    sn = pool.snips.get(cid) or []
    if sn:
        parts.append("Passage: " + sn[0][:snip_chars if ab else 900])
    return "\n".join(parts)


async def grade_batch(query, crit_text, nc, pool, cids):
    papers = "\n\n".join(f"[{i}] {paper_blurb(pool, c)}" for i, c in enumerate(cids))
    prompt = GRADE_PROMPT.format(query=query, criteria=crit_text, papers=papers, nc=nc)
    out = {}
    for _ in range(2):
        res = jparse(await llm(GRADE_MODEL, prompt))
        if isinstance(res, dict):
            for k, v in res.items():
                m = re.search(r"\d+", str(k))
                if not m or not isinstance(v, list):
                    continue
                i = int(m.group())
                if 0 <= i < len(cids):
                    r = []
                    for x in v[:nc]:
                        x = as_int(x) or 0
                        r.append(3 if x >= 3 else (1 if x >= 1 else 0))
                    r += [0] * (nc - len(r))
                    out[cids[i]] = r
            if out:
                break
    return out


def cut_sentences(text, cap):
    """Longest prefix of text (<= cap chars) ending at a sentence boundary."""
    text = text.strip()
    if len(text) <= cap:
        return text
    chunk = text[:cap]
    j = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "), chunk.rfind(".\n"))
    if j > cap * 0.5:
        return chunk[:j + 1].strip()
    j = chunk.rfind(" ")
    return chunk[:j].strip() if j > 0 else chunk


def build_evidence(pool, cid, rich=True):
    p = pool.papers[cid]
    passages = []
    used = 0

    def push(t, cap):
        nonlocal used
        t = (t or "").strip()
        room = min(cap, EVIDENCE_CAP - used - 5)
        if not t or room < 60 and len(t) > room:
            return
        t = cut_sentences(t, room)
        if t and len(passages) < 8:
            passages.append(t)
            used += len(t) + 5

    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    snips = [s for s in (pool.snips.get(cid) or [])] if rich else []
    body = []
    for s in snips:
        key = s[:80]
        if abstract and key in abstract:
            continue
        if title and s.strip() == title:
            continue
        body.append(s)
    push(title, 300)
    if abstract:
        push(abstract, 1500 if body else 2100)
    else:
        push(tldr_of(p), 500)
    per = max(300, (EVIDENCE_CAP - used) // max(1, min(len(body), 3))) if body else 0
    for s in body[:3]:
        push(s, per)
    return " ... ".join(passages)


async def solve_semantic(ctx, state, query, plan):
    pool = Pool()
    crit = plan["criteria"]
    nc = len(crit)
    crit_text = "\n".join(f"{i + 1}. {c.get('name', '')} (weight {c['weight']:.2f}): {c['description']}"
                          for i, c in enumerate(crit))
    print("  criteria:\n" + crit_text)
    kws = [q for q in plan["keyword_queries"] if isinstance(q, str)][:14]
    sqs = [query] + [q for q in plan["snippet_queries"] if isinstance(q, str)][:3]
    venues = None
    tasks = [kw_search(ctx, pool, q, 100, venues) for q in kws]
    tasks += [snip_search(ctx, pool, q, 60, weight=1.5 if i == 0 else 1.0) for i, q in enumerate(sqs)]
    tasks += [title_search(ctx, pool, t, weight=2.0) for t in plan["candidate_titles"][:12]]
    res = await asyncio.gather(*tasks)
    print(f"  retrieval: {len(pool.papers)} candidates from {len(kws)} kw / {len(sqs)} snippet queries "
          f"(sizes {[len(r) if isinstance(r, list) else int(bool(r)) for r in res]}) t={ctx.elapsed():.0f}s")

    # hydrate snippet-only papers (no abstract) so the grader/evidence have text
    need = [c for c, p in pool.papers.items() if not p.get("abstract") and "year" not in p]
    if need:
        got = await batch_get(ctx, need[:400], PAPER_FIELDS)
        for c, d in got.items():
            pool.add(d)
        # papers the batch tool does not return may be outside the snapshot; drop them
        for c in need[:400]:
            if c not in got and not pool.snips.get(c):
                pool.papers.pop(c, None)

    # year constraint from the request (hard filter only when explicit)
    y0, y1 = as_int(plan.get("year_min")), as_int(plan.get("year_max"))
    ranked = sorted(pool.papers, key=lambda c: -pool.rrf.get(c, 0.0))
    if y0 or y1:
        def ok(c):
            y = as_int(pool.papers[c].get("year"))
            return y is None or ((not y0 or y >= y0) and (not y1 or y <= y1))
        ranked = [c for c in ranked if ok(c)]
    ranked = [c for c in ranked if pool.papers[c].get("title")]
    to_grade = ranked[:MAX_GRADE]

    batches = [to_grade[i:i + GRADE_BATCH] for i in range(0, len(to_grade), GRADE_BATCH)]
    gsem = asyncio.Semaphore(10)

    async def gb(b):
        async with gsem:
            return await grade_batch(query, crit_text, nc, pool, b)

    grades = {}
    for g in await asyncio.gather(*[gb(b) for b in batches]):
        grades.update(g)
    print(f"  graded {len(grades)}/{len(to_grade)} t={ctx.elapsed():.0f}s")

    def weighted(c):
        r = grades.get(c)
        if r is None:
            return -1.0
        return min(1.0, sum(cr["weight"] * x / 3.0 for cr, x in zip(crit, r)))

    wmap = {c: weighted(c) for c in to_grade}
    order = sorted(to_grade, key=lambda c: (-wmap[c], -pool.rrf.get(c, 0.0)))
    n3 = sum(1 for c in order if wmap[c] > 0.99)
    n2 = sum(1 for c in order if 0.67 < wmap[c] <= 0.99)
    n1 = sum(1 for c in order if 0.25 < wmap[c] <= 0.67)
    print(f"  est. grades: 3={n3} 2={n2} 1={n1} other={len(order) - n3 - n2 - n1}")

    # body-passage evidence for the papers that can plausibly reach grade 3
    top = [c for c in order if wmap[c] > 0.6][:60]
    focus = query + " " + " ".join(str(c.get("name", "")) for c in crit)

    async def ev(c):
        if ctx.left() < 240:
            return
        await snip_search(ctx, pool, focus[:400], 4, paper_ids=f"CorpusId:{c}", timeout=45)

    if top:
        # bounded phase: evidence enrichment is optional, the ranked list is not
        _, pending = await asyncio.wait([asyncio.ensure_future(ev(c)) for c in top], timeout=360)
        for t in pending:
            t.cancel()
    print(f"  evidence snippets fetched for {len(top)} papers t={ctx.elapsed():.0f}s")

    topset = set(top)
    results = []
    for c in order[:MAX_SUBMIT]:
        evd = build_evidence(pool, c, rich=True if c in topset else False)
        if not evd:
            continue
        results.append({"paper_id": c, "markdown_evidence": evd})
    for c in order[:12]:
        print(f"    {wmap[c]:.2f} {grades.get(c)} {c} {(pool.papers[c].get('title') or '')[:90]}")
    return emit(state, results)


# ----------------------------------------------------------------------------
# specific path
# ----------------------------------------------------------------------------
PICK_PROMPT = """A user asked a literature search engine for: "{query}"

This is a request for specific, known paper(s). Below are candidate papers retrieved from the corpus.

{cands}

Select the candidate(s) that ARE the paper the user is referring to.
- Usually exactly one candidate is the answer: the original paper that introduced the named method / system / dataset / result.
- If the name is genuinely ambiguous (several unrelated papers each introduce something with exactly this name, and the request does not disambiguate), include each of those papers (at most 6).
- If the same paper appears as duplicate records (e.g. preprint and published version with the same title), include all of its records.
- Never include papers that merely use, cite, extend, survey, or evaluate it.
- If no candidate fits, pick the single closest one.

Reply with ONLY JSON: {{"ids": ["<corpus id>", ...]}}"""


def cand_line(p, abs_chars=260):
    au = p.get("authors") or []
    names = ", ".join((a.get("name") if isinstance(a, dict) else str(a)) or "" for a in au[:3])
    if len(au) > 3:
        names += " et al."
    ab = ((p.get("abstract") or "") or tldr_of(p))[:abs_chars].replace("\n", " ")
    return (f"id={cid_of(p)} | {p.get('title')} | {names} | {p.get('venue') or '?'} {p.get('year') or '?'} "
            f"| cited {p.get('citationCount')} | {ab}")


async def solve_specific(ctx, state, query, plan):
    pool = Pool()
    bare = re.sub(r"^\s*the\s+", "", query.strip(), flags=re.I)
    bare = re.sub(r"\s+paper\s*$", "", bare, flags=re.I)
    titles = [t for t in plan["candidate_titles"] if isinstance(t, str)][:12]
    kws = [bare] + [q for q in plan["keyword_queries"] if isinstance(q, str)][:6] + titles[:6]
    tasks = [title_search(ctx, pool, t) for t in titles + [bare]]
    tasks += [kw_search(ctx, pool, q, 15) for q in kws]
    tasks += [snip_search(ctx, pool, query, 15, timeout=90)]
    await asyncio.gather(*tasks)
    need = [c for c, p in pool.papers.items() if "year" not in p]
    if need:
        for c, d in (await batch_get(ctx, need[:100], PAPER_FIELDS)).items():
            pool.add(d)
    ranked = [c for c in sorted(pool.papers, key=lambda c: -pool.rrf.get(c, 0.0))
              if pool.papers[c].get("title")][:70]
    print(f"  specific: {len(pool.papers)} candidates, titles tried={titles}")
    if not ranked:
        return emit(state, [])
    cands = "\n".join(cand_line(pool.papers[c]) for c in ranked)
    res = jparse(await llm(PICK_MODEL, PICK_PROMPT.format(query=query, cands=cands)))
    ids = []
    if isinstance(res, dict):
        ids = [str(i).replace("CorpusId:", "").strip() for i in (res.get("ids") or [])]
        ids = [i for i in ids if i in pool.papers]
    if not ids:
        print("  pick failed; falling back to top retrieval hit")
        ids = ranked[:1]
    # include duplicate records with an identical title
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
    chosen = {norm(pool.papers[i].get("title")) for i in ids}
    for c in ranked:
        if c not in ids and norm(pool.papers[c].get("title")) in chosen:
            ids.append(c)
    for i in ids:
        print(f"    picked {i} {pool.papers[i].get('title')}")
    return emit(state, [{"paper_id": i, "markdown_evidence": build_evidence(pool, i, False)} for i in ids[:8]])


# ----------------------------------------------------------------------------
# metadata path
# ----------------------------------------------------------------------------
def name_tokens(n):
    return [t for t in re.sub(r"[^a-z\s\-]", " ", (n or "").lower()).replace("-", " ").split() if t]


def author_match(want, got):
    w, g = name_tokens(want), name_tokens(got)
    if not w or not g or w[-1] != g[-1]:
        return False
    if len(w) == 1 or len(g) == 1:
        return len(w) == len(g)
    wf, gf = w[0], g[0]
    if wf == gf:
        return True
    if len(gf) == 1 or len(wf) == 1:
        return gf[0] == wf[0]
    return False


def paper_has_author(p, want):
    return any(author_match(want, (a.get("name") if isinstance(a, dict) else str(a)))
               for a in (p.get("authors") or []))


VENUE_PROMPT = """A literature search request has this venue requirement: "{constraint}"
(full request: "{query}")

Below are the distinct venue strings of candidate papers. Return the ones that satisfy the venue requirement (be inclusive of spelling variants, abbreviations, and — when the requirement names a family/portfolio/publisher — every member venue; exclude venues that are clearly different, e.g. workshops only if the request excludes them).

{venues}

Reply with ONLY JSON: {{"matching": ["<venue string exactly as listed>", ...]}}"""

TOPIC_PROMPT = """Search request: "{query}"
Topical requirement: "{topic}"

All candidates below already satisfy the request's metadata constraints (author / venue / year / citations). Return the ids of the candidates whose content satisfies the topical requirement. Be inclusive when a title is plausibly on topic.

{cands}

Reply with ONLY JSON: {{"ids": ["<id>", ...]}}"""


async def solve_metadata(ctx, state, query, plan):
    pool = Pool()
    authors = [a for a in plan["authors"] if isinstance(a, str) and a.strip()]
    cited_titles = [t for t in plan["cited_paper_titles"] if isinstance(t, str) and t.strip()]
    venue_names = [v for v in plan["venue_names"] if isinstance(v, str) and v.strip()]
    vcon = plan.get("venue_constraint")
    topic = plan.get("topic")
    y0, y1 = as_int(plan.get("year_min")), as_int(plan.get("year_max"))
    c0, c1 = as_int(plan.get("min_citations")), as_int(plan.get("max_citations"))
    a0, a1 = as_int(plan.get("min_authors")), as_int(plan.get("max_authors"))
    print(f"  spec: authors={authors} cites={cited_titles} venue={vcon!r}/{venue_names} years={y0}-{y1} "
          f"cit={c0}-{c1} nauth={a0}-{a1} topic={topic!r}")

    def numeric_ok(p, need_authors=True):
        y = as_int(p.get("year"))
        if (y0 or y1) and (y is None or (y0 and y < y0) or (y1 and y > y1)):
            return False
        c = as_int(p.get("citationCount"))
        if c0 is not None and (c is None or c < c0):
            return False
        if c1 is not None and (c is None or c > c1):
            return False
        if need_authors and (a0 is not None or a1 is not None):
            n = len(p.get("authors") or [])
            if (a0 is not None and n < a0) or (a1 is not None and n > a1):
                return False
        return True

    cand = None  # set of cids satisfying structural (author / citation) constraints

    # ---- author constraint ---------------------------------------------------
    if authors:
        per_author = []
        for name in authors[:4]:
            recs = await call(ctx, "search_authors_by_name", name=name, fields="name,paperCount", limit=50)
            recs = [r for r in recs if r.get("authorId") and author_match(name, r.get("name"))]
            recs.sort(key=lambda r: -(as_int(r.get("paperCount")) or 0))
            recs = recs[:20]
            print(f"  author {name!r}: {[(r.get('name'), r.get('paperCount')) for r in recs]}")
            lists = await asyncio.gather(*[
                call(ctx, "get_author_papers", author_id=str(r["authorId"]), paper_fields=META_FIELDS, limit=1000)
                for r in recs])
            ids = set()
            for lst in lists:
                for p in lst:
                    c = pool.add(p)
                    if c:
                        ids.add(c)
            per_author.append(ids)
        cand = set.intersection(*per_author) if per_author else set()
        # co-author check by name for multi-author requests (identities are fragmentary)
        if len(authors) > 1:
            union = set.union(*per_author)
            cand = {c for c in union if all(paper_has_author(pool.papers[c], a) for a in authors)}
        print(f"  author candidates: {len(cand)}")

    # ---- citation constraint ---------------------------------------------------
    if cited_titles:
        tpool = Pool()
        targets = [t for t in await asyncio.gather(*[title_search(ctx, tpool, t, META_FIELDS) for t in cited_titles[:3]]) if t]
        # also pick up duplicate records of the cited paper via keyword search
        print(f"  cited targets: {[(t, tpool.papers[t].get('title')) for t in targets]}")
        tset = set(targets)
        citing = set()
        if tset:
            lists = await asyncio.gather(*[
                call(ctx, "get_citations", timeout=280, paper_id=f"CorpusId:{t}", fields=META_FIELDS, limit=1000)
                for t in targets])
            direct = set()
            for lst in lists:
                for item in lst:
                    p = item.get("citingPaper", item) if isinstance(item, dict) else {}
                    c = pool.add(p)
                    if c and numeric_ok(pool.papers[c]):
                        direct.add(c)
            print(f"  get_citations: {len(direct)} pass numeric filters")
            # keyword expansion: the citations window is capped (newest-first), so
            # gather candidates by topic and verify through their reference lists
            expand = set()
            if cand is None:
                vn = ",".join(venue_names) if venue_names else None
                qs = [q for q in plan["citing_topic_queries"] if isinstance(q, str)][:36]
                qs += [q for q in plan["keyword_queries"] if isinstance(q, str)][:6]
                qs += [tpool.papers[t].get("title") or "" for t in targets]
                lists = await asyncio.gather(*[kw_search(ctx, pool, q, 100, vn, META_FIELDS) for q in qs])
                if vn and sum(len(x) for x in lists) == 0:
                    print("  venue-filtered expansion returned nothing; retrying unfiltered")
                    lists = await asyncio.gather(*[kw_search(ctx, pool, q, 100, None, META_FIELDS) for q in qs])
                for lst in lists:
                    expand.update(c for c in lst if numeric_ok(pool.papers[c]))
                print(f"  keyword expansion: {len(expand)} pass numeric filters")
            else:
                expand = {c for c in cand if numeric_ok(pool.papers[c])}
            to_verify = list((direct | expand) - tset)[:2500]
            refs = await batch_get(ctx, to_verify, "corpusId,year,references.corpusId", chunk=20)
            if to_verify and not refs:
                print("  reference lookup returned nothing; trusting get_citations list")
                citing |= direct
            for c in to_verify:
                d = refs.get(c)
                if d is None:
                    continue  # not returned => outside the snapshot
                rs = {str(r.get("corpusId")) for r in (d.get("references") or []) if isinstance(r, dict)}
                if c in direct or rs & tset:
                    citing.add(c)
            print(f"  verified citing: {len(citing)} of {len(to_verify)} checked")
        cand = citing if cand is None else (cand & citing)

    # ---- no structural constraint: keyword search (venue-filtered) -------------
    if cand is None:
        vn = ",".join(venue_names) if venue_names else None
        qs = [q for q in plan["keyword_queries"] if isinstance(q, str)][:8]
        if topic:
            qs.append(str(topic))
        lists = await asyncio.gather(*[kw_search(ctx, pool, q, 100, vn, META_FIELDS) for q in qs])
        if vn and sum(len(x) for x in lists) == 0:
            lists = await asyncio.gather(*[kw_search(ctx, pool, q, 100, None, META_FIELDS) for q in qs])
        cand = set().union(*[set(x) for x in lists]) if lists else set()
        print(f"  keyword candidates: {len(cand)}")

    keep = [c for c in cand if numeric_ok(pool.papers[c])]
    print(f"  after numeric filters: {len(keep)}")

    # ---- venue constraint (LLM maps the constraint onto observed venue strings) -
    if vcon and keep:
        venues = sorted({(pool.papers[c].get("venue") or "").strip() for c in keep} - {""})
        match = None
        if venues:
            res = jparse(await llm(MID_MODEL, VENUE_PROMPT.format(
                constraint=vcon, query=query, venues="\n".join(f"- {v}" for v in venues[:600]))))
            if isinstance(res, dict) and isinstance(res.get("matching"), list):
                match = {str(v).strip().lower() for v in res["matching"]}
        if match is not None:
            keep = [c for c in keep if (pool.papers[c].get("venue") or "").strip().lower() in match]
            print(f"  after venue filter ({len(match)} venues match): {len(keep)}")
        else:
            print("  venue filter skipped (LLM failure or no venue strings)")

    # ---- topical requirement ---------------------------------------------------
    if topic and keep and (authors or cited_titles or vcon):
        keep.sort(key=lambda c: -(as_int(pool.papers[c].get("citationCount")) or 0))
        sel = set()
        ok_any = False
        for i in range(0, min(len(keep), 600), 150):
            part = keep[i:i + 150]
            cands = "\n".join(f"id={c} | {pool.papers[c].get('title')} | {pool.papers[c].get('venue') or ''} "
                              f"{pool.papers[c].get('year') or ''}" for c in part)
            res = jparse(await llm(MID_MODEL, TOPIC_PROMPT.format(query=query, topic=topic, cands=cands)))
            if isinstance(res, dict) and isinstance(res.get("ids"), list):
                ok_any = True
                sel.update(str(x).replace("id=", "").strip() for x in res["ids"])
        if ok_any:
            keep = [c for c in keep if c in sel]
            print(f"  after topic filter: {len(keep)}")

    keep.sort(key=lambda c: -(as_int(pool.papers[c].get("citationCount")) or 0))
    for c in keep[:8]:
        p = pool.papers[c]
        print(f"    {c} {p.get('year')} {p.get('venue')} cit={p.get('citationCount')} {(p.get('title') or '')[:70]}")
    return emit(state, [{"paper_id": c, "markdown_evidence": ""} for c in keep])


# ----------------------------------------------------------------------------
@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata.get("raw_query") or str(state.input)
        kind = state.metadata.get("score_type", "") or "semantic_f1"
        print(f"[{state.sample_id}] score_type={kind} query={query[:200]!r}")
        ctx = Ctx(state)
        state.output.completion = json.dumps({"output": {"query_id": state.sample_id, "results": []}})
        try:
            plan = await make_plan(query, kind)
            print(f"  plan: kw={plan['keyword_queries']} titles={plan['candidate_titles'][:6]}")
            if kind == "specific_f1":
                await solve_specific(ctx, state, query, plan)
            elif kind == "metadata_f1":
                await solve_metadata(ctx, state, query, plan)
            else:
                await solve_semantic(ctx, state, query, plan)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  SOLVER ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        print(f"  done in {ctx.elapsed():.0f}s; tool failures={ctx.fail}")
        return state

    return solve
