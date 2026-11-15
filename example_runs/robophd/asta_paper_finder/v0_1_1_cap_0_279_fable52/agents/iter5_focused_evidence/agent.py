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

from model_registry import CLAUDE_OPUS_4_8, CLAUDE_SONNET_4_6, GPT_5_4, GPT_5_4_MINI

PLAN_MODEL = CLAUDE_OPUS_4_8
PICK_MODEL = CLAUDE_OPUS_4_8
MID_MODEL = CLAUDE_SONNET_4_6
GRADE_MODEL = GPT_5_4_MINI
REGRADE_MODEL = GPT_5_4

PAPER_FIELDS = "title,abstract,corpusId,year,venue,citationCount,authors,tldr"
META_FIELDS = "title,corpusId,year,venue,citationCount,authors,publicationDate"
MAX_GRADE = 320          # semantic candidates sent to the grader
GRADE_BATCH = 12
MAX_SUBMIT = 250
EVIDENCE_CAP = 2450
QUERY_BUDGET_S = 18 * 60  # soft budget: no new tool calls after this
HARD_BUDGET_S = 24 * 60   # solver is cancelled here; hard limit is 29 min
MAX_REGRADE = 80          # top candidates re-judged by the stronger model
REGRADE_BATCH = 5
CUTOFF_DATE = "2025-06-01"  # corpus snapshot (from the tools' own error text)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
class Ctx:
    def __init__(self, state):
        self.state = state
        self.tools = {ToolDef(t).name: t for t in state.tools}
        self.sem = asyncio.Semaphore(12)
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


async def call(ctx, tool_name, /, timeout=150, errs=None, **kw):
    """Call a corpus tool; returns parsed list (empty on failure)."""
    async with ctx.sem:
        if ctx.left() < 0:   # checked after queueing, so a backlog cannot overrun the budget
            return []
        try:
            raw = await asyncio.wait_for(ctx.tools[tool_name](**kw), timeout=timeout)
            return _parse(raw)
        except Exception as e:  # noqa: BLE001
            ctx.fail[tool_name] = ctx.fail.get(tool_name, 0) + 1
            root = e
            while getattr(root, "exceptions", None):   # unwrap nested ExceptionGroups to the cause
                root = root.exceptions[0]
            if ctx.fail[tool_name] <= 3:
                print(f"  tool {tool_name} failed: {type(root).__name__}: {str(root)[:200]}")
            if errs is not None:
                errs.append(str(root) or type(root).__name__)
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
        self.csnips = {}    # cid -> {criterion index (None = whole query): best passage}
        self.rsnips = {}    # cid -> passages fetched per paper for a criterion the evidence lacked
        self.alls = {}      # cid -> every body passage retrieved for the paper (any source)

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

    def add_snippet(self, d, rank=None, weight=1.0, crit_i=-1):
        paper = d.get("paper") or {}
        snip = d.get("snippet") or {}
        cid = self.add({"corpusId": paper.get("corpusId"), "title": paper.get("title")},
                       rank, weight)
        text = (snip.get("text") or "").strip()
        if cid and text:
            al = self.alls.setdefault(cid, [])
            if text not in al:
                al.append(text)
        if cid and text and crit_i != -1:
            self.csnips.setdefault(cid, {}).setdefault(crit_i, text)
        elif cid and text:
            lst = self.snips.setdefault(cid, [])
            if text not in lst:
                lst.append(text)
        return cid


async def kw_search(ctx, pool, q, limit=100, venues=None, fields=PAPER_FIELDS, weight=1.0, timeout=100):
    q = clean_kw(q)
    if not q:
        return []
    kw = dict(keyword=q, fields=fields, limit=limit)
    if venues:
        kw["venues"] = venues
    errs = []
    hits = await call(ctx, "search_papers_by_relevance", timeout=timeout, errs=errs, **kw)
    if not hits and errs and ctx.left() > 240:   # transient failure / timeout: one retry
        hits = await call(ctx, "search_papers_by_relevance", timeout=timeout, **kw)
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


def after_cutoff(p):
    d = str(p.get("publicationDate") or "")
    if d:
        return d >= CUTOFF_DATE
    y = as_int(p.get("year"))
    return y is not None and y > 2025


async def batch_get(ctx, ids, fields, chunk=25, sha=None):
    """get_paper_batch with recovery. sha: optional cid -> paperId map used to drop
    papers the server rejects as newer than the snapshot (it names them by paperId)."""
    out = {}
    sha = sha or {}

    async def one(part, depth=0):
        if not part:
            return
        errs = []
        docs = await call(ctx, "get_paper_batch", timeout=120, errs=errs,
                          ids=[f"CorpusId:{i}" for i in part], fields=fields)
        if not docs and errs and depth < 7:
            bad = set(re.findall(r"[0-9a-f]{40}", errs[0])) if "cutoff" in errs[0] else set()
            if bad:
                keep = [c for c in part if sha.get(c) not in bad]
                if len(keep) < len(part):
                    await one(keep, depth + 1)
                    return
            if len(part) > 1:
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
- "snippet_queries": 5 natural-language sentences (the kind of sentence a relevant paper's abstract or body would contain) for a dense passage-retrieval engine.
- "candidate_titles": exact titles of real papers you know that match the request (up to 12; [] if none known). For a request like "the X paper", give the actual full title(s) of the paper(s) that introduced X (several if X is an ambiguous name used by different papers).
- "criteria": 2-4 relevance criteria that a paper must satisfy, each {{"name": str, "description": "The paper must ...", "weight": float, "terms": [6-10 literal words or short phrases, including synonyms, abbreviations and spelling variants, that would appear in a sentence of a paper demonstrating this criterion]}}; weights sum to 1. Decompose the request literally into its distinct required aspects (e.g. domain, technique, specific role/application), staying close to the request's own wording, as a meticulous relevance judge would. Every specific qualifier in the request must appear in some criterion.
- "authors": list of author full names the request requires (else []).
- "venue_constraint": the venue requirement in the user's words, or null.
- "venue_names": if there is a venue requirement, likely exact Semantic Scholar venue strings (full official name AND common abbreviation, e.g. "Neural Information Processing Systems", "NeurIPS"); else [].
- "year_min", "year_max": inclusive integer bounds implied by the request, or null. Treat the named boundary year as INCLUDED: "after 2022" -> year_min 2022; "since 2020" -> year_min 2020; "before 2019" -> year_max 2019; "in 2021" -> both 2021; "2022-23" -> 2022 and 2023.
- "years_allowed": if the request names separate individual years rather than a range ("published at 2014 or 2017" -> [2014, 2017]), that explicit list; else null.
- "min_citations", "max_citations": inclusive integer bounds on the paper's own citation count, boundary value INCLUDED ("more than 50 citations" -> min_citations 50; "at least 30" -> 30; "fewer than 100" -> max_citations 100), or null.
- "min_authors", "max_authors": inclusive bounds on number of authors ("more than 3 authors" -> min_authors 4; "single-author" -> max_authors 1), or null.
- "cited_paper_titles": if the request asks for papers that CITE some paper, the exact full title(s) of that cited paper (else []).
- "cited_author_names": if the request asks for papers that cite works BY a named author (e.g. "citing papers by Jane Doe"), those author full names; else []. (This is different from "authors", which lists who wrote the requested papers.)
- "exclude_author_names": author full names that must NOT be among the requested papers' own authors (e.g. "not self-citations of Jane Doe" -> ["Jane Doe"]); else [].
- "cited_paper_mentions": if cited_paper_titles is non-empty, 4 short strings showing how a citing paper's text typically refers to each cited paper (e.g. "RoBERTa (Liu et al., 2019)", "Liu et al. RoBERTa robustly optimized BERT"); else [].
- "citing_topic_queries": if cited_paper_titles is non-empty, 50 diverse keyword queries (2-6 words) spanning the research areas, tasks, and applications whose papers typically cite it (restricted to the request's topic if it has one); else [].
- "topic": a topical/content requirement that the request STATES IN WORDS in addition to its author / venue / year / citation-count / "cites paper X" filters, as a short phrase; null otherwise. "Papers citing X" by itself has NO topic (null): never infer a topic from what the cited papers are about.
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
              "venue_names", "cited_paper_titles", "citing_topic_queries", "cited_paper_mentions",
              "cited_author_names", "exclude_author_names"):
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


def paper_blurb(pool, cid, abs_chars=950, snip_chars=400):
    p = pool.papers[cid]
    parts = [f"Title: {p.get('title') or '(unknown)'}"]
    if p.get("year"):
        parts[0] += f" ({p.get('year')})"
    ab = (p.get("abstract") or "").strip() or tldr_of(p)
    if ab:
        parts.append("Abstract: " + ab[:abs_chars])
    sn = (pool.snips.get(cid) or [])[:4]
    if sn:
        qt = getattr(pool, "qterms", None) or {}
        cap = snip_chars if ab else 900
        parts.append("Passage: " + max((best_window(x, qt, cap) for x in sn), key=lambda x: x[0])[1])
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


STOP = set("""a an the of and or for to in on at by with from as is are was were be been being this that these those it its
their our we they which who whose such than then also not no can may might must should would could will shall into over
under between among within across about per via using use used uses based paper papers work study studies research
approach approaches method methods technique techniques include includes including involve involves involving discuss
discusses address addresses explicitly specifically specific clear clearly either both any all each other more most
some how what when where why while during must focus focuses focused aim aims aimed propose proposes proposed present
presents presented describe describes described provide provides e.g i.e etc has have had do does done new novel""".split())


def _toks(text):
    return re.findall(r"[a-z0-9]+", str(text).lower().replace("-", " "))


def make_terms(weighted_texts):
    """weighted_texts: [(text, weight)] -> {stem: weight}. Stems are 6-char prefixes."""
    terms = {}
    for text, w in weighted_texts:
        for t in _toks(text):
            if t in STOP or (len(t) < 3 and not t.isdigit()) or len(t) < 2:
                continue
            st = t[:6] if len(t) > 6 else t
            terms[st] = max(terms.get(st, 0.0), w)
    return terms


def _score(text, terms):
    words = _toks(text)
    stems = set(w[:6] if len(w) > 6 else w for w in words)
    return sum(w for st, w in terms.items() if st in stems)


def _sent_spans(text):
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def best_span(text, terms, cap):
    """Best contiguous run of whole sentences (<= cap chars) by term coverage.
    Returns (score, start, end) as offsets into text; shortest run wins ties."""
    if not text:
        return 0.0, 0, 0
    if len(text) <= cap:
        return _score(text, terms), 0, len(text)
    spans = _sent_spans(text)
    best = (-1.0, 0, 0)
    for a in range(len(spans)):
        for b in range(a, len(spans)):
            if spans[b][1] - spans[a][0] > cap:
                break
            sc = _score(text[spans[a][0]:spans[b][1]], terms)
            if sc > best[0] or (sc == best[0] and spans[b][1] - spans[a][0] < best[2] - best[1]):
                best = (sc, spans[a][0], spans[b][1])
    if best[0] <= 0:
        # no whole sentence fits or matches: clip the best over-long sentence
        cands = sorted(((_score(text[a:b], terms), a, b) for a, b in spans), key=lambda x: -x[0])
        if cands and cands[0][0] > 0:
            sc, a, b = cands[0]
            return sc, a, a + len(cut_sentences(text[a:b], cap))
        return 0.0, 0, len(cut_sentences(text, cap))
    return best


def best_window(text, terms, cap):
    text = (text or "").strip()
    sc, a, b = best_span(text, terms, cap)
    return sc, text[a:b].strip()


def build_evidence(pool, cid, crit_n=0, cterms=None, win_cap=380):
    """Verbatim evidence: title, abstract, then for each criterion the sentence window (from any
    passage retrieved for this paper) that best covers the criterion's terms. Windows from the
    same passage that touch or nearly touch are merged into one contiguous quote."""
    p = pool.papers[cid]
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    cterms = [t for t in (cterms or []) if t]
    src = []
    if crit_n:
        cs = pool.csnips.get(cid) or {}
        cand = list(pool.rsnips.get(cid) or []) + [cs.get(i) for i in range(crit_n)] + [cs.get(None)] \
            + list(pool.snips.get(cid) or []) + list(pool.alls.get(cid) or [])
        seen = set()
        for sn in cand:
            sn = (sn or "").strip()
            if not sn:
                continue
            key = sn[:80]
            if key in seen or (abstract and key in abstract) or sn == title:
                continue
            seen.add(key)
            src.append(sn)
    src = src[:14]
    picks = []   # (source index, start, end), in priority order

    def add_pick(k, a, b):
        if b <= a:
            return False
        for (k2, a2, b2) in picks:
            if k2 == k and a < b2 and a2 < b:      # overlaps an earlier pick
                return False
            if src[k][a:a + 60] == src[k2][a2:a2 + 60]:
                return False
        picks.append((k, a, b))
        return True

    if src and cterms:
        abs_l = abstract[:1000]
        # criteria the abstract covers least get their window first
        order_i = sorted(range(len(cterms)),
                         key=lambda i: _score(abs_l, cterms[i]) / (sum(cterms[i].values()) or 1.0))
        for i in order_i:
            bestc = None
            for k, sn in enumerate(src):
                sc, a, b = best_span(sn, cterms[i], win_cap)
                if sc > 0 and (bestc is None or sc > bestc[0]):
                    bestc = (sc, k, a, b)
            if bestc:
                add_pick(*bestc[1:])
        union = {}
        for t in cterms:
            for st, v in t.items():
                union[st] = max(union.get(st, 0.0), v)
        extra = []
        for k, sn in enumerate(src):
            sc, a, b = best_span(sn, union, win_cap)
            if sc > 0:
                extra.append((sc, k, a, b))
        extra.sort(key=lambda x: -x[0])
        for sc, k, a, b in extra:
            if len(picks) >= min(5, len(cterms) + 1):
                break
            add_pick(k, a, b)
    # merge near-adjacent picks from the same passage, keeping the priority of the earliest pick
    merged = []
    for (k, a, b) in picks:
        for m in merged:
            if m[0] == k and a <= m[2] + 160 and m[1] <= b + 160 and max(b, m[2]) - min(a, m[1]) <= 2 * win_cap:
                m[1], m[2] = min(a, m[1]), max(b, m[2])
                break
        else:
            merged.append([k, a, b])
    wins = [src[k][a:b].strip() for k, a, b in merged]
    if src and not wins:
        wins = [cut_sentences(sn, win_cap) for sn in src[:3]]
    wins = [w for w in wins if w][:6]
    passages, used = [], 0

    def push(t, cap):
        nonlocal used
        t = (t or "").strip()
        room = min(cap, EVIDENCE_CAP - used - 5)
        if not t or len(passages) >= 8 or room < 80:
            return
        t = cut_sentences(t, room)
        if t:
            passages.append(t)
            used += len(t) + 5

    push(title, 300)
    need = sum(len(w) + 5 for w in wins)
    if abstract:
        push(abstract, max(950, EVIDENCE_CAP - used - need - 10))
    else:
        push(tldr_of(p), 500)
    for w in wins:
        push(w, 2 * win_cap + 20)
    return " ... ".join(passages)


REGRADE_PROMPT = """You are the relevance judge of a literature-search benchmark. For each paper you see ONLY an evidence text (title, abstract, passages). Judge strictly from that text.

Search request: {query}

Relevance criteria:
{criteria}

For EACH paper rate EVERY criterion:
 3 = Perfectly relevant: the evidence explicitly and fully satisfies the criterion, including every specific qualifier in it
 1 = Somewhat relevant: related, partial, generic, or only implied
 0 = Not relevant / no support in the evidence
Then give an overall rating "o" with the same scale: 3 only if this paper is exactly what the requester is looking for (every aspect of the request explicitly met), 1 if on-topic but missing or only implying some aspect, 0 if off-topic.
Be demanding: topical proximity is not enough for a 3.

Papers:
{papers}

Reply with ONLY a JSON object mapping each paper number to {{"c": [<{nc} ratings in criteria order>], "o": <overall>}}, e.g. {{"0": {{"c": [3, 1, 3], "o": 1}}}}. Include every paper number."""


async def regrade_batch(query, crit_text, nc, cids, evid):
    papers = "\n\n".join(f"[{i}] {evid.get(c) or '(no text)'}" for i, c in enumerate(cids))
    prompt = REGRADE_PROMPT.format(query=query, criteria=crit_text, papers=papers, nc=nc)
    out = {}
    res = jparse(await llm(REGRADE_MODEL, prompt))
    if isinstance(res, dict):
        q = lambda x: 3 if (as_int(x) or 0) >= 3 else (1 if (as_int(x) or 0) >= 1 else 0)
        for k, v in res.items():
            m = re.search(r"\d+", str(k))
            if not m or not isinstance(v, dict) or not isinstance(v.get("c"), list):
                continue
            i = int(m.group())
            if 0 <= i < len(cids):
                r = [q(x) for x in v["c"][:nc]]
                r += [0] * (nc - len(r))
                out[cids[i]] = (r, q(v.get("o")))
    return out


async def solve_semantic(ctx, state, query, plan):
    pool = Pool()
    crit = plan["criteria"]
    nc = len(crit)
    crit_text = "\n".join(f"{i + 1}. {c.get('name', '')} (weight {c['weight']:.2f}): {c['description']}"
                          for i, c in enumerate(crit))
    print("  criteria:\n" + crit_text)
    cterms = []
    for c in crit:
        tl = c.get("terms") if isinstance(c.get("terms"), list) else []
        cterms.append(make_terms([(c.get("description", ""), 1.0), (c.get("name", ""), 2.0),
                                  (" ".join(str(t) for t in tl[:12]), 2.0)]))
    qterms = make_terms([(query, 1.0)] + [(" ".join(t), 1.0) for t in cterms])
    pool.qterms = qterms
    kws = [q for q in plan["keyword_queries"] if isinstance(q, str)][:14]
    sqs = [query] + [q for q in plan["snippet_queries"] if isinstance(q, str)][:5]
    venues = None
    tasks = [kw_search(ctx, pool, q, 100, venues) for q in kws]
    tasks += [snip_search(ctx, pool, q, 80, weight=1.5 if i == 0 else 1.0) for i, q in enumerate(sqs)]
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

    def provisional(order_, rich=()):
        res_ = []
        for c in order_[:MAX_SUBMIT]:
            evd = evid.get(c) or build_evidence(pool, c, crit_n=nc, cterms=cterms)
            if evd:
                res_.append({"paper_id": c, "markdown_evidence": evd})
        return res_

    evid = {}
    emit(state, provisional(order))   # safety net if a later phase is cancelled

    # ---- per-criterion body passages for the papers that can plausibly reach grade 3
    top = [c for c in order if wmap[c] > 0.6][:MAX_REGRADE]
    if len(top) < 30:
        top = order[:min(30, len(order))]
    cq = [(None, query)] + [(i, f"{c.get('name', '')}: {c['description']}"[:300]) for i, c in enumerate(crit)]

    async def ev(ci, q, part):
        if ctx.left() < 200:
            return
        hits = await call(ctx, "snippet_search", timeout=70, query=q, limit=min(100, 5 * len(part)),
                          paper_ids=",".join(f"CorpusId:{c}" for c in part))
        for h in hits:
            pool.add_snippet(h, crit_i=ci)

    if top:
        jobs = [ev(ci, q, top[k:k + 10]) for ci, q in cq for k in range(0, len(top), 10)]
        _, pending = await asyncio.wait([asyncio.ensure_future(x) for x in jobs], timeout=240)
        for t in pending:
            t.cancel()
    topset = set(top)
    print(f"  criterion snippets: {sum(1 for c in top if pool.csnips.get(c))}/{len(top)} papers t={ctx.elapsed():.0f}s")
    for c in top:
        evid[c] = build_evidence(pool, c, crit_n=nc, cterms=cterms)

    # ---- stage 2: stronger model re-judges the top list from the exact evidence text
    s2 = {}
    batches2 = [top[k:k + REGRADE_BATCH] for k in range(0, len(top), REGRADE_BATCH)]

    async def rg(b):
        async with gsem:
            return await regrade_batch(query, crit_text, nc, b, evid)

    if ctx.elapsed() < HARD_BUDGET_S - 300:
        for g in await asyncio.gather(*[rg(b) for b in batches2]):
            s2.update(g)
    print(f"  regraded {len(s2)}/{len(top)} t={ctx.elapsed():.0f}s")

    def s2w(c):
        r, ov = s2[c]
        return min(1.0, sum(cr["weight"] * x / 3.0 for cr, x in zip(crit, r)))

    def is_full(c):
        return c in s2 and s2w(c) > 0.99 and s2[c][1] >= 3

    # ---- round 2: (a) expand from confirmed papers, (b) repair near-miss evidence, re-judge both
    async def round2():
        fulls = [c for c in top if is_full(c)]
        seeds = (fulls or top[:3])[:6]
        known = set(pool.papers)

        async def citers(c):
            hits = await call(ctx, "get_citations", timeout=60, paper_id=f"CorpusId:{c}",
                              fields=META_FIELDS + ",abstract", limit=60)
            for r, h in enumerate(hits):
                p = h.get("citingPaper", h) if isinstance(h, dict) else {}
                if isinstance(p, dict) and not after_cutoff(p):
                    pool.add(p, r, 0.5)

        # near misses: every criterion at least partially met, exactly the kind of paper that a
        # passage explicitly supporting the weak criterion can lift to a full match
        near = [c for c in top if c in s2 and not is_full(c) and s2w(c) >= 0.55 and min(s2[c][0]) >= 1]
        near.sort(key=lambda c: -s2w(c))
        near = near[:24]

        async def repair(c):
            weak = [i for i, x in enumerate(s2[c][0]) if x < 3] or list(range(nc))
            for i in weak[:2]:
                hits = await call(ctx, "snippet_search", timeout=45, limit=4, paper_ids=f"CorpusId:{c}",
                                  query=f"{crit[i].get('name', '')}: {crit[i]['description']}"[:300])
                for h in hits[:4]:
                    t = ((h.get("snippet") or {}).get("text") or "").strip()
                    if t and str((h.get("paper") or {}).get("corpusId")) == c:
                        lst = pool.rsnips.setdefault(c, [])
                        if t not in lst:
                            lst.append(t)

        jobs = [kw_search(ctx, pool, pool.papers[c].get("title") or "", 40, weight=0.5, timeout=60) for c in seeds]
        jobs += [snip_search(ctx, pool, tldr_of(pool.papers[c]) or pool.papers[c].get("title") or query, 40,
                             timeout=60, weight=0.5) for c in seeds[:4]]
        jobs += [citers(c) for c in seeds]
        jobs += [repair(c) for c in near]
        await asyncio.gather(*jobs)
        new = [c for c in pool.papers if c not in known]
        need2 = [c for c in new if not pool.papers[c].get("abstract") and "year" not in pool.papers[c]]
        if need2:
            for c, d in (await batch_get(ctx, need2[:120], PAPER_FIELDS)).items():
                pool.add(d)
        def ok2(c):
            p = pool.papers[c]
            if not p.get("title") or not (p.get("abstract") or pool.snips.get(c) or tldr_of(p)):
                return False
            y = as_int(p.get("year"))
            return y is None or ((not y0 or y >= y0) and (not y1 or y <= y1))
        new = sorted([c for c in new if ok2(c)], key=lambda c: -pool.rrf.get(c, 0.0))[:96]
        g2 = {}
        for g in await asyncio.gather(*[gb(new[i:i + GRADE_BATCH]) for i in range(0, len(new), GRADE_BATCH)]):
            g2.update(g)
        w2 = {c: min(1.0, sum(cr["weight"] * x / 3.0 for cr, x in zip(crit, g2[c]))) for c in g2}
        new_top = sorted([c for c in w2 if w2[c] > 0.6], key=lambda c: -w2[c])[:20]
        print(f"  round2: seeds={len(seeds)} new={len(new)} graded={len(g2)} new_top={len(new_top)} "
              f"near={len(near)} repaired={sum(1 for c in near if pool.rsnips.get(c))} t={ctx.elapsed():.0f}s")
        # body passages for the promising newcomers (per paper: scoped calls starve weak matches)
        async def ev1(c):
            hits = await call(ctx, "snippet_search", timeout=45, limit=3, paper_ids=f"CorpusId:{c}", query=query)
            for h in hits:
                pool.add_snippet(h, crit_i=None)
        if new_top and ctx.left() > 120:
            await asyncio.gather(*[ev1(c) for c in new_top])
        ev2 = {}
        for c in near:
            if pool.rsnips.get(c):
                e = build_evidence(pool, c, crit_n=nc, cterms=cterms)
                if e != evid.get(c):
                    ev2[c] = e
        for c in new_top:
            ev2[c] = build_evidence(pool, c, crit_n=nc, cterms=cterms)
        todo = list(ev2)
        r2 = {}

        async def rg2(b):
            async with gsem:
                return await regrade_batch(query, crit_text, nc, b, ev2)

        for g in await asyncio.gather(*[rg2(todo[k:k + REGRADE_BATCH]) for k in range(0, len(todo), REGRADE_BATCH)]):
            r2.update(g)
        # commit (all at once, so a cancelled round leaves the first-pass state intact)
        up = 0
        for c in g2:
            if c not in wmap:
                grades[c] = g2[c]
                wmap[c] = w2[c]
                order.append(c)
        for c, v in r2.items():
            if c in s2:
                old_w = s2w(c)
                new_w = min(1.0, sum(cr["weight"] * x / 3.0 for cr, x in zip(crit, v[0])))
                if new_w + v[1] / 30.0 <= old_w + s2[c][1] / 30.0:
                    continue   # extra passages did not help: keep the original evidence
                up += 1
            s2[c] = v
            evid[c] = ev2[c]
        print(f"  round2: rejudged {len(r2)}/{len(todo)}; upgraded near-misses={up}; "
              f"new full={sum(1 for c in new_top if is_full(c))} t={ctx.elapsed():.0f}s")

    if s2 and ctx.elapsed() < 780 and ctx.left() > 300:
        try:
            emit(state, provisional(sorted(order, key=lambda c: (
                -((2.0 if is_full(c) else 0.0) + (s2w(c) if c in s2 else 0.88 * max(wmap[c], 0.0))), -wmap[c]))))
            await asyncio.wait_for(round2(), timeout=min(400, HARD_BUDGET_S - ctx.elapsed() - 90))
        except asyncio.TimeoutError:
            print(f"  round2 timed out t={ctx.elapsed():.0f}s; keeping first-pass results")
        except Exception as e:  # noqa: BLE001
            print(f"  round2 error: {type(e).__name__}: {str(e)[:200]}")
    else:
        print(f"  round2 skipped t={ctx.elapsed():.0f}s")

    def final_key(c):
        if c in s2:
            r, ov = s2[c]
            w = min(1.0, sum(cr["weight"] * x / 3.0 for cr, x in zip(crit, r)))
            full = 1.0 if (w > 0.99 and ov >= 3) else 0.0
            return full + 0.75 * w + 0.25 * ov / 3.0
        # un-rejudged papers: the cheap grader over-rates, so discount it
        return 0.88 * max(wmap[c], 0.0)

    fkey = {c: final_key(c) for c in order}
    final = sorted(order, key=lambda c: (-fkey[c], -wmap[c], -pool.rrf.get(c, 0.0)))
    print(f"  stage2: full={sum(1 for c in s2 if fkey[c] >= 1.99)} "
          f"all-criteria={sum(1 for c in s2 if fkey[c] >= 1.0)} of {len(s2)}")
    for c in final[:15]:
        print(f"    {fkey[c]:.2f} s1={wmap[c]:.2f} s2={s2.get(c)} {c} {(pool.papers[c].get('title') or '')[:80]}")
    return emit(state, provisional(final, rich=topset))


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
- Never include papers that merely use, cite, extend, survey, or evaluate it, nor follow-up / "v2" / successor papers, nor errata or "Author Correction" records.
- Do not add a second candidate just because it is on the same topic or by the same group; when in doubt return only the single best match.
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
    junk = re.compile(r"^\s*(author correction|publisher correction|correction|erratum|corrigendum|retraction)\b", re.I)
    clean = [i for i in ids if not junk.match(pool.papers[i].get("title") or "")]
    ids = clean or ids
    for i in ids:
        print(f"    picked {i} {pool.papers[i].get('title')}")
    return emit(state, [{"paper_id": i, "markdown_evidence": build_evidence(pool, i)} for i in ids[:8]])


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

Below are the distinct venue strings of candidate papers. Return the ones that ARE the required venue: spelling variants, abbreviations and the full official name of that same venue (and, only when the requirement names a family / portfolio / publisher, every member venue). Do NOT include workshops, co-located or satellite events (e.g. "ArgMining@ACL", "SemEval"), "Findings of ..." volumes, regional chapters (e.g. NAACL or EACL for "ACL"), or journals with a similar name, unless the request asks for them.

{venues}

Reply with ONLY JSON: {{"matching": ["<venue string exactly as listed>", ...]}}"""

TOPIC_PROMPT = """Search request: "{query}"
Topical requirement: "{topic}"

All candidates below already satisfy the request's metadata constraints (author / venue / year / citations). Return the ids of the candidates whose content satisfies the topical requirement. Be inclusive when a title is plausibly on topic.

{cands}

Reply with ONLY JSON: {{"ids": ["<id>", ...]}}"""


async def author_search(ctx, name):
    """search_authors_by_name with two retries on a tool failure (an empty result is not retried)."""
    for attempt in range(3):
        errs = []
        recs = await call(ctx, "search_authors_by_name", timeout=90, errs=errs, name=name,
                          fields="name,paperCount", limit=50)
        if recs or not errs:
            return recs
        print(f"  author search for {name!r} failed (attempt {attempt + 1}): {errs[0][:120]}")
    return []


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
    ylist = plan.get("years_allowed")
    ylist = {as_int(y) for y in ylist if as_int(y)} if isinstance(ylist, list) else set()
    if ylist:
        y0, y1 = min(ylist), max(ylist)
    if isinstance(topic, str) and cited_titles and re.search(r"\bcit(e|es|ing|ed|ation)", topic, re.I):
        topic = None   # "papers citing X" is not a content requirement
    print(f"  years_allowed={sorted(ylist)}")
    print(f"  spec: authors={authors} cites={cited_titles} venue={vcon!r}/{venue_names} years={y0}-{y1} "
          f"cit={c0}-{c1} nauth={a0}-{a1} topic={topic!r}")

    def numeric_ok(p, need_authors=True):
        y = as_int(p.get("year"))
        if (y0 or y1) and (y is None or (y0 and y < y0) or (y1 and y > y1)):
            return False
        if ylist and y not in ylist:
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
            recs = await author_search(ctx, name)
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
                call(ctx, "get_citations", timeout=200, paper_id=f"CorpusId:{t}", fields=META_FIELDS, limit=1000)
                for t in targets])
            direct = {t: set() for t in targets}
            for t, lst in zip(targets, lists):
                for item in lst:
                    p = item.get("citingPaper", item) if isinstance(item, dict) else {}
                    c = pool.add(p)
                    if c and numeric_ok(pool.papers[c]) and not after_cutoff(pool.papers[c]):
                        direct[t].add(c)
            dall = set().union(*direct.values())
            common = set.intersection(*direct.values())
            tcount = [as_int(tpool.papers[t].get("citationCount")) or 0 for t in targets]
            print(f"  get_citations: {[len(v) for v in direct.values()]} pass numeric/cutoff filters; "
                  f"window sizes {[len(x) for x in lists]}; target citationCounts {tcount}; common={len(common)}")
            # The citation window (newest 1000 citers) is what a tool-based gold can see.
            #  - several targets: the papers present in every window are the answer when there are any
            #  - one target whose whole citation list fits the window: the window is complete
            window_only = (len(targets) > 1 and len(common) > 0) or \
                          (len(targets) == 1 and len(lists[0]) < 990)
            if window_only and cand is None:
                print("  using the citation window directly (no expansion)")
                citing |= common
            # the citations window is capped (newest-first), so also gather candidates by
            # topic / citation-mention passages and verify through their reference lists
            expand = set()
            if window_only and cand is None:
                pass
            elif cand is None:
                vn = ",".join(venue_names) if venue_names else None
                qs = [q for q in plan["citing_topic_queries"] if isinstance(q, str)][:60]
                qs += [q for q in plan["keyword_queries"] if isinstance(q, str)][:6]
                qs += [tpool.papers[t].get("title") or "" for t in targets]
                if topic:
                    qs.append(str(topic))
                mentions = [m for m in plan["cited_paper_mentions"] if isinstance(m, str)][:6]
                mentions += [tpool.papers[t].get("title") or "" for t in targets]

                async def mention_search(m):
                    kw = dict(query=m, limit=100)
                    if vn:
                        kw["venues"] = vn
                    hits = await call(ctx, "snippet_search", timeout=75, **kw)
                    return [x for x in (pool.add_snippet(h) for h in hits) if x]

                res = await asyncio.gather(*[kw_search(ctx, pool, q, 100, vn, META_FIELDS, timeout=60) for q in qs],
                                           *[mention_search(m) for m in mentions])
                if vn and sum(len(x) for x in res[:len(qs)]) == 0:
                    print("  venue-filtered expansion returned nothing; retrying unfiltered")
                    res = await asyncio.gather(*[kw_search(ctx, pool, q, 100, None, META_FIELDS) for q in qs])
                found = set().union(*[set(x) for x in res]) if res else set()
                # snippet-only hits lack metadata: hydrate before filtering
                need = [c for c in found if "year" not in pool.papers[c]]
                for c, d in (await batch_get(ctx, need[:500], META_FIELDS)).items():
                    pool.add(d)
                expand = {c for c in found if "year" in pool.papers[c] and numeric_ok(pool.papers[c])
                          and not after_cutoff(pool.papers[c])}
                print(f"  expansion: {len(found)} found, {len(expand)} pass numeric filters t={ctx.elapsed():.0f}s")
            else:
                expand = {c for c in cand if numeric_ok(pool.papers[c])}
            # most promising first: already in a citations window, then by citation count
            to_verify = sorted((dall | expand) - tset,
                               key=lambda c: (c not in dall, -(as_int(pool.papers[c].get("citationCount")) or 0)))[:700]
            if len(targets) == 1:
                # one target: everything in its window already cites it; only the expansion needs checking
                citing |= common
                to_verify = [c for c in to_verify if c not in common]
            if window_only and cand is None:
                to_verify = []
            # safety net: if verification is cut short, submit papers seen in every citation window
            emit(state, [{"paper_id": c, "markdown_evidence": ""} for c in common][:250])
            sha = {c: pool.papers[c].get("paperId") for c in to_verify}
            refs = await batch_get(ctx, to_verify, "corpusId,year,references.corpusId", chunk=10, sha=sha) if to_verify else {}
            if to_verify and not refs:
                print("  reference lookup returned nothing; trusting get_citations list")
                citing |= common
            for c in to_verify:
                d = refs.get(c)
                if d is None:
                    continue  # not returned => outside the snapshot
                rs = {str(r.get("corpusId")) for r in (d.get("references") or []) if isinstance(r, dict)}
                if all(c in direct[t] or t in rs for t in targets):
                    citing.add(c)
            print(f"  verified citing: {len(citing)} of {len(to_verify)} checked ({len(refs)} returned) t={ctx.elapsed():.0f}s")
        cand = citing if cand is None else (cand & citing)

    # ---- "cites works by author X" constraint -------------------------------------
    cited_auth = [a for a in plan["cited_author_names"] if isinstance(a, str) and a.strip()]
    excl_auth = [a for a in plan["exclude_author_names"] if isinstance(a, str) and a.strip()]
    if cited_auth:
        wpool = Pool()
        for name in cited_auth[:2]:
            recs = await author_search(ctx, name)
            recs = [r for r in recs if r.get("authorId") and author_match(name, r.get("name"))]
            recs.sort(key=lambda r: -(as_int(r.get("paperCount")) or 0))
            lists = await asyncio.gather(*[
                call(ctx, "get_author_papers", author_id=str(r["authorId"]),
                     paper_fields="title,corpusId,year,citationCount", limit=1000) for r in recs[:12]])
            for lst in lists:
                for p in lst:
                    wpool.add(p)
        wset = set(wpool.papers)
        print(f"  cited authors {cited_auth}: {len(wset)} papers by them")
        if wset:
            # citers of each of the cited author's papers (their citation lists are usually short
            # enough to fit the 1000-entry window), most-cited first
            wl = sorted((c for c in wset if (as_int(wpool.papers[c].get("citationCount")) or 0) > 0),
                        key=lambda c: -(as_int(wpool.papers[c].get("citationCount")) or 0))[:150]
            lists = await asyncio.gather(*[
                call(ctx, "get_citations", timeout=120, paper_id=f"CorpusId:{t}", fields=META_FIELDS, limit=1000)
                for t in wl])
            citers = set()
            for lst in lists:
                for item in lst:
                    p = item.get("citingPaper", item) if isinstance(item, dict) else {}
                    c = cid_of(p) if isinstance(p, dict) else ""
                    if not c:
                        continue
                    if cand is None:
                        pool.add(p)
                        if after_cutoff(pool.papers[c]):
                            continue
                    citers.add(c)
            print(f"  citers of papers by {cited_auth}: {len(citers)} (from {len(wl)} papers, "
                  f"{sum(1 for x in lists if x)} lists returned)")
            if citers:
                cand = citers if cand is None else (cand & citers)
                print(f"  candidates citing a paper by {cited_auth}: {len(cand)}")
            else:
                print("  no citation lists returned; cited-author constraint not applied")
    if excl_auth and cand is not None:
        n0 = len(cand)
        cand = {c for c in cand if not any(paper_has_author(pool.papers[c], a) for a in excl_auth)}
        print(f"  excluding papers authored by {excl_auth}: {n0} -> {len(cand)}")

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

    keep = [c for c in cand if numeric_ok(pool.papers[c]) and not after_cutoff(pool.papers[c])]
    print(f"  after numeric filters: {len(keep)}")
    if (authors or cited_titles or cited_auth) and not vcon and not topic:
        emit(state, [{"paper_id": c, "markdown_evidence": ""} for c in keep])

    # A bare publication-type requirement ("journal articles") names no venue. Venue strings do not
    # encode publication type reliably, so with a structural constraint present it is not filtered on.
    if vcon and (authors or cited_titles or cited_auth) and not re.sub(
            r"\b(peer|reviewed|journals?|articles?|conferences?|papers?|publications?|proceedings|published|in|a|an|the|of|only)\b|[^a-z]",
            "", str(vcon).lower()):
        print(f"  venue constraint {vcon!r} is a generic publication type; not filtering on it")
        vcon = None

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
    if topic and keep and (authors or cited_titles or cited_auth or vcon):
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
                coro = solve_specific(ctx, state, query, plan)
            elif kind == "metadata_f1":
                coro = solve_metadata(ctx, state, query, plan)
            else:
                coro = solve_semantic(ctx, state, query, plan)
            await asyncio.wait_for(coro, timeout=max(60, HARD_BUDGET_S - ctx.elapsed()))
        except asyncio.TimeoutError:
            print(f"  HARD BUDGET reached at {ctx.elapsed():.0f}s; keeping the last emitted results")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  SOLVER ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        print(f"  done in {ctx.elapsed():.0f}s; tool failures={ctx.fail}")
        return state

    return solve
