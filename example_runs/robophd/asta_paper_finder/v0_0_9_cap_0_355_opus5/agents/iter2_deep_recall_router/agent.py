"""deep-recall-router — a PaperFindingBench solver.

Three observations from iteration-001 drive the whole design:

1. `recall = grade3_in_top_K / K` and `judge_verdicts.json` reports
   `scored_depth_cap == K`. The seed submitted 8 papers against K values of
   22-304, capping recall at a few percent. Since `score = harmonic(rank,
   recall)` and rank was already 0.9-1.0, recall is the binding term by an
   order of magnitude. Semantic queries carry no precision penalty, so we
   submit *deep* (up to the scorer's 250-entry cap), best-first.

2. `rank` is 0 when every grade is equal. The seed scored 0.000 on a query
   where all 8 submitted papers were judged Perfectly Relevant. Deep
   submission makes that degenerate case impossible.

3. Grade 3 needs `weighted > 0.99`, i.e. essentially every criterion judged
   Perfectly Relevant, and only grade 3 earns recall. Evidence is the only
   text the judge sees, so we spend the full 2500-char budget on verbatim
   title + tldr + abstract + snippets rather than a 400-char abstract stub.

`specific_f1`/`metadata_f1` invert the tradeoff (exact-match F1 punishes
extra ids), so they route to precision-first paths instead.
"""

import asyncio
import json
import re
import time

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI, CLAUDE_SONNET_4_6

# ---------------------------------------------------------------- constants --

MAX_SUBMIT = 250          # the scorer reads only the first 250 entries
EVIDENCE_CAP = 2400       # judge truncates at 2500; leave margin
MAX_RERANK = 420          # stage-1 grading budget (candidates)
HEAD_SIZE = 60            # stage-2 (strong model) re-grading depth
STAGE1_BATCH = 20
STAGE2_BATCH = 12
TOOL_CONCURRENCY = 6
LLM_CONCURRENCY = 8
WALL_BUDGET = 22 * 60     # hard stop well inside the 29-minute timeout

PAPER_FIELDS = "title,abstract,corpusId,year,venue,authors,tldr,citationCount"


# ------------------------------------------------------------------- utils --

def _tool(state: TaskState, name: str):
    """Find a tool by registered name; None if the task didn't attach it."""
    for t in state.tools:
        try:
            if ToolDef(t).name == name:
                return t
        except Exception:
            continue
    return None


def _parse_items(raw) -> list[dict]:
    """Flatten the MCP return shape: a list of ContentText whose `.text` is
    JSON, sometimes wrapped in {"data": [...]}."""
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
            data = doc.get("data") or []
            if isinstance(data, list):
                out.extend(d for d in data if isinstance(d, dict))
        elif isinstance(doc, dict):
            out.append(doc)
        elif isinstance(doc, list):
            out.extend(d for d in doc if isinstance(d, dict))
    return out


def _jload(text: str):
    """Best-effort JSON object extraction from an LLM completion."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    return None


def _tldr_text(v) -> str:
    if isinstance(v, dict):
        return (v.get("text") or "").strip()
    return (v or "").strip() if isinstance(v, str) else ""


def _cid(doc: dict) -> str:
    """corpusId is an int in relevance search, a str in snippet/citation
    results; normalise to a bare digit string."""
    for key in ("corpusId", "corpusid", "CorpusId"):
        v = doc.get(key)
        if v not in (None, ""):
            s = str(v).strip()
            if s.lower().startswith("corpusid:"):
                s = s.split(":", 1)[1].strip()
            if s.isdigit():
                return s
    return ""


def _clip(text: str, limit: int) -> str:
    """Truncate at a sentence, then word, boundary — a prefix of retrieved
    text stays verbatim-derivable for the grounding check."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "; ", ", ", " "):
        idx = cut.rfind(sep)
        if idx > limit * 0.55:
            return cut[:idx + (1 if sep == ". " else 0)].strip()
    return cut.strip()


async def _gen(handle, prompt: str, *, config=None, tag: str = ""):
    """Metered LLM call with an empty-completion guard (surfaced, not hidden)."""
    try:
        resp = await handle.generate(prompt, config=config) if config else await handle.generate(prompt)
        text = (resp.completion or "").strip()
        if not text:
            print(f"  [warn] empty completion from {tag}")
        return text
    except Exception as exc:  # never let one call sink the query
        print(f"  [warn] LLM call failed ({tag}): {type(exc).__name__}: {exc}")
        return ""


# --------------------------------------------------------------- retrieval --

class Candidate:
    __slots__ = ("cid", "title", "abstract", "tldr", "year", "venue",
                 "authors", "snippets", "fusion", "g1", "g2")

    def __init__(self, cid: str):
        self.cid = cid
        self.title = ""
        self.abstract = ""
        self.tldr = ""
        self.year = None
        self.venue = ""
        self.authors = []
        self.snippets: list[str] = []
        self.fusion = 0.0
        self.g1 = None   # stage-1 grade
        self.g2 = None   # stage-2 grade

    def absorb(self, doc: dict):
        if not self.title:
            self.title = (doc.get("title") or "").strip()
        if not self.abstract:
            self.abstract = (doc.get("abstract") or "").strip()
        if not self.tldr:
            self.tldr = _tldr_text(doc.get("tldr"))
        if self.year is None:
            self.year = doc.get("year")
        if not self.venue:
            self.venue = (doc.get("venue") or "").strip()
        if not self.authors:
            au = doc.get("authors") or []
            if isinstance(au, list):
                self.authors = [a.get("name", "") for a in au if isinstance(a, dict)][:8]

    def brief(self, abs_chars: int) -> str:
        body = self.tldr or self.abstract
        if self.tldr and self.abstract:
            body = self.tldr + " " + self.abstract
        yr = f" ({self.year})" if self.year else ""
        return f"{self.title}{yr}. {_clip(body, abs_chars)}"

    def evidence(self) -> str:
        """Verbatim passages joined by ' ... ' — at most 8, under the cap."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.tldr:
            parts.append(self.tldr)
        if self.abstract:
            parts.append(self.abstract)
        for sn in self.snippets[:5]:
            parts.append(sn)

        out, used = [], 0
        for p in parts[:8]:
            if used >= EVIDENCE_CAP - 20:
                break
            room = EVIDENCE_CAP - used - (5 if out else 0)
            piece = _clip(p, room)
            if not piece:
                continue
            out.append(piece)
            used += len(piece) + 5
        return " ... ".join(out)


class Pool:
    """Deduplicated candidate store with reciprocal-rank fusion scoring."""

    def __init__(self):
        self.by_cid: dict[str, Candidate] = {}

    def add(self, doc: dict, rank: int, weight: float = 1.0) -> Candidate | None:
        cid = _cid(doc)
        if not cid:
            return None
        cand = self.by_cid.get(cid)
        if cand is None:
            cand = Candidate(cid)
            self.by_cid[cid] = cand
        cand.absorb(doc)
        cand.fusion += weight / (60.0 + rank)
        return cand

    def ranked(self) -> list[Candidate]:
        return sorted(self.by_cid.values(), key=lambda c: -c.fusion)


async def _gather(tasks, limit: int):
    """Run coroutines with bounded concurrency; exceptions become None."""
    sem = asyncio.Semaphore(limit)

    async def run(coro):
        async with sem:
            try:
                return await coro
            except Exception as exc:
                print(f"  [warn] task failed: {type(exc).__name__}: {exc}")
                return None

    return await asyncio.gather(*(run(c) for c in tasks))


# ---------------------------------------------------------- query analysis --

ANALYSIS_PROMPT = """You are planning a scientific-literature search over Semantic Scholar.

USER REQUEST: {query}
REQUEST TYPE: {stype}

Return ONLY a JSON object with these keys:

"keyword_queries": 6 short keyword/noun-phrase search queries (3-8 words, NO
  question words, NO punctuation, NO "papers about"). The search engine returns
  ZERO hits for interrogative phrasing, so use bare topical noun phrases. Make
  them DIVERSE: vary terminology, include synonyms/abbreviations the literature
  actually uses, and cover each distinct facet of the request separately.
"criteria": 2-4 short strings, each a necessary property a paper must
  explicitly demonstrate to fully satisfy the request. Decompose the request
  into its independent requirements (topic, method, application, data, setting).
"candidate_titles": if the request names a specific known paper (possibly by a
  nickname such as "the BART paper", a system name, or a one-line description),
  list 1-5 GUESSES at its exact published title, best first. Use the real title
  only, with no subtitle prefix such as "SystemName:" unless the published title
  truly begins that way. Otherwise [].
"authors": full author names the request requires, else [].
"venues": exact venue names the request requires (e.g. "Nature", "NeurIPS"), else [].
"year_min": integer or null.  "year_max": integer or null.
"""


async def analyze(query: str, stype: str) -> dict:
    text = await _gen(GPT_5_4, ANALYSIS_PROMPT.format(query=query, stype=stype),
                      tag="analysis")
    data = _jload(text)
    if not isinstance(data, dict):
        data = {}

    def _strlist(key):
        v = data.get(key) or []
        return [str(x).strip() for x in v if isinstance(x, (str, int)) and str(x).strip()] \
            if isinstance(v, list) else []

    kws = _strlist("keyword_queries")[:7]
    if not kws:
        # Fallback: strip interrogative framing ourselves.
        stripped = re.sub(
            r"^(could you|can you|please|i am looking for|i'm looking for|show me|"
            r"find me|are there|is there|has any|what are|what is|how can|how do|"
            r"do you know of|any)\b[^,]*?\b(papers?|research|studies|work|study)\b\s*"
            r"(that|which|on|about|for|using)?\s*", "", query.strip(), flags=re.I)
        stripped = stripped.strip(" ?.!,")
        kws = [stripped or query]
    return {
        "keyword_queries": kws,
        "criteria": _strlist("criteria")[:5],
        "candidate_titles": _strlist("candidate_titles")[:5],
        "authors": _strlist("authors")[:5],
        "venues": _strlist("venues")[:4],
        "year_min": data.get("year_min") if isinstance(data.get("year_min"), int) else None,
        "year_max": data.get("year_max") if isinstance(data.get("year_max"), int) else None,
    }


# ------------------------------------------------------------ broad search --

async def broad_retrieve(state: TaskState, query: str, plan: dict, pool: Pool,
                         deadline: float) -> None:
    """Fan out keyword searches + snippet searches; everything here is free."""
    search = _tool(state, "search_papers_by_relevance")
    snippet = _tool(state, "snippet_search")
    venues = ",".join(plan["venues"]) if plan["venues"] else None

    async def kw(q: str, weight: float):
        if not search or time.time() > deadline:
            return
        kwargs = {"keyword": q, "fields": PAPER_FIELDS, "limit": 100}
        if venues:
            kwargs["venues"] = venues
        docs = _parse_items(await search(**kwargs))
        for r, d in enumerate(docs):
            pool.add(d, r, weight)
        print(f"  kw {q!r} -> {len(docs)}")

    async def snip(q: str, weight: float):
        if not snippet or time.time() > deadline:
            return
        try:
            raw = await asyncio.wait_for(snippet(query=q, limit=100), timeout=280)
        except asyncio.TimeoutError:
            print(f"  snippet {q[:40]!r} timed out")
            return
        entries = _parse_items(raw)
        n = 0
        for r, e in enumerate(entries):
            paper = e.get("paper") if isinstance(e.get("paper"), dict) else e
            cand = pool.add(paper, r, weight)
            if cand is None:
                continue
            n += 1
            sn = e.get("snippet")
            txt = (sn.get("text") if isinstance(sn, dict) else None) or ""
            txt = txt.strip()
            if txt and len(cand.snippets) < 5 and txt not in cand.snippets:
                cand.snippets.append(txt)
        print(f"  snippet {q[:50]!r} -> {n}")

    tasks = [kw(q, 1.0 if i == 0 else 0.85) for i, q in enumerate(plan["keyword_queries"])]
    tasks.append(snip(query, 1.0))
    if plan["keyword_queries"]:
        tasks.append(snip(plan["keyword_queries"][0], 0.8))
    await _gather(tasks, TOOL_CONCURRENCY)


async def enrich_missing_abstracts(state: TaskState, cands: list[Candidate]) -> None:
    """snippet_search returns no abstract; batch-fetch metadata for those."""
    batch_tool = _tool(state, "get_paper_batch")
    missing = [c for c in cands if not c.abstract][:200]
    if not batch_tool or not missing:
        return

    async def fetch(chunk):
        docs = _parse_items(await batch_tool(
            ids=[f"CorpusId:{c.cid}" for c in chunk], fields=PAPER_FIELDS))
        by_cid = {}
        for d in docs:
            cid = _cid(d)
            if cid:
                by_cid[cid] = d
        for c in chunk:
            if c.cid in by_cid:
                c.absorb(by_cid[c.cid])

    chunks = [missing[i:i + 50] for i in range(0, len(missing), 50)]
    await _gather([fetch(ch) for ch in chunks], 4)
    print(f"  enriched {len(missing)} abstract-less candidates")


# ---------------------------------------------------------------- reranker --

GRADE_RULES = """Grade each candidate 0-3 on how well it satisfies EVERY requirement:
  3 = explicitly satisfies ALL of the requirements above, directly and unambiguously
  2 = satisfies most requirements but at least one is unclear or only implied
  1 = same general topic, but clearly misses a requirement
  0 = not relevant
Only give 3 when every single requirement is explicitly demonstrated by the text."""


def _grade_prompt(query: str, criteria: list[str], batch: list[Candidate],
                  abs_chars: int) -> str:
    crit = "\n".join(f"  - {c}" for c in criteria) or "  - matches the request"
    lines = "\n".join(f"[{i}] {c.brief(abs_chars)}" for i, c in enumerate(batch))
    return (
        f"USER REQUEST: {query}\n\nREQUIREMENTS a fully relevant paper must meet:\n{crit}\n\n"
        f"{GRADE_RULES}\n\nCANDIDATES:\n{lines}\n\n"
        f"Reply with exactly {len(batch)} lines, one per candidate, formatted "
        f"`index:grade` (e.g. `0:3`). No other text."
    )


def _apply_grades(text: str, batch: list[Candidate], attr: str) -> int:
    n = 0
    for m in re.finditer(r"(\d+)\s*[:=]\s*([0-3])", text or ""):
        i, g = int(m.group(1)), int(m.group(2))
        if 0 <= i < len(batch):
            setattr(batch[i], attr, g)
            n += 1
    return n


async def rerank(query: str, criteria: list[str], cands: list[Candidate],
                 deadline: float) -> None:
    """Stage 1: cheap model over everything. Stage 2: strong model over the
    head, to sharpen the ordering that `rank` and small-K recall depend on."""
    stage1 = cands[:MAX_RERANK]
    batches = [stage1[i:i + STAGE1_BATCH] for i in range(0, len(stage1), STAGE1_BATCH)]

    async def do1(batch):
        if time.time() > deadline:
            return
        txt = await _gen(GPT_5_4_MINI, _grade_prompt(query, criteria, batch, 620),
                         tag="stage1")
        _apply_grades(txt, batch, "g1")

    await _gather([do1(b) for b in batches], LLM_CONCURRENCY)
    graded = sum(1 for c in stage1 if c.g1 is not None)
    print(f"  stage1 graded {graded}/{len(stage1)}")

    for c in stage1:
        if c.g1 is None:
            c.g1 = 1 if c.abstract else 0

    head = sorted(stage1, key=lambda c: (-(c.g1 or 0), -c.fusion))[:HEAD_SIZE]
    if not head or time.time() > deadline:
        return
    hbatches = [head[i:i + STAGE2_BATCH] for i in range(0, len(head), STAGE2_BATCH)]

    async def do2(batch):
        if time.time() > deadline:
            return
        txt = await _gen(
            CLAUDE_SONNET_4_6, _grade_prompt(query, criteria, batch, 950),
            config=GenerateConfig(max_tokens=600), tag="stage2")
        _apply_grades(txt, batch, "g2")

    await _gather([do2(b) for b in hbatches], LLM_CONCURRENCY)
    print(f"  stage2 graded {sum(1 for c in head if c.g2 is not None)}/{len(head)}")


def final_order(cands: list[Candidate]) -> list[Candidate]:
    """Head ordered by the strong model (falling back to stage 1), tail by the
    cheap model then retrieval fusion. Submitted deep: on semantic_f1 there is
    no precision penalty and recall counts grade-3 papers within the first K."""
    def key(c: Candidate):
        primary = c.g2 if c.g2 is not None else (c.g1 if c.g1 is not None else 0)
        return (-(1 if c.g2 is not None else 0), -primary, -(c.g1 or 0), -c.fusion)
    return sorted(cands, key=key)


# ------------------------------------------------------- exact-match paths --

async def resolve_titles(state: TaskState, titles: list[str], pool: Pool) -> list[Candidate]:
    """Title-search each guessed title. Verified on the probe: the bare
    published title matches, a `System:` prefix does not — so try variants."""
    by_title = _tool(state, "search_paper_by_title")
    if not by_title or not titles:
        return []
    variants: list[str] = []
    for t in titles:
        for v in (t, t.split(":", 1)[1].strip() if ":" in t else None):
            if v and v not in variants:
                variants.append(v)
    found: list[Candidate] = []

    async def one(t: str, rank: int):
        docs = _parse_items(await by_title(title=t, fields=PAPER_FIELDS))
        for d in docs:
            if not d.get("paperId"):     # "no match" comes back as {"data": []}
                continue
            c = pool.add(d, rank, 3.0)
            if c is not None:
                found.append(c)
                print(f"  title {t[:60]!r} -> {c.cid} {c.title[:60]!r}")

    await _gather([one(t, i) for i, t in enumerate(variants[:6])], TOOL_CONCURRENCY)
    return found


VERIFY_PROMPT = """USER REQUEST: {query}

The request names one specific paper (or, if the nickname is genuinely ambiguous,
a small set of canonical papers). Candidates retrieved:

{cands}

Reply with ONLY a JSON list of the index numbers that ARE the paper(s) the request
means, best first. Normally return exactly ONE index. Return 2-3 ONLY if the
nickname canonically refers to several papers (e.g. an ambiguous architecture
nickname). Never pad the list: each wrong extra id directly lowers precision.
"""


async def specific_path(state: TaskState, query: str, plan: dict, pool: Pool,
                        deadline: float) -> list[Candidate]:
    await resolve_titles(state, plan["candidate_titles"], pool)
    search = _tool(state, "search_papers_by_relevance")
    if search:
        async def kw(q, w):
            docs = _parse_items(await search(keyword=q, fields=PAPER_FIELDS, limit=25))
            for r, d in enumerate(docs):
                pool.add(d, r, w)
        qs = (plan["candidate_titles"][:2] + plan["keyword_queries"][:3])
        await _gather([kw(q, 1.0) for q in qs if q], TOOL_CONCURRENCY)

    cands = pool.ranked()[:30]
    if not cands:
        return []
    lines = "\n".join(f"[{i}] {c.brief(320)}" for i, c in enumerate(cands))
    txt = await _gen(GPT_5_4, VERIFY_PROMPT.format(query=query, cands=lines),
                     tag="verify-specific")
    idxs = _jload(txt)
    picked: list[Candidate] = []
    if isinstance(idxs, list):
        for i in idxs[:3]:
            if isinstance(i, int) and 0 <= i < len(cands) and cands[i] not in picked:
                picked.append(cands[i])
    if not picked:
        picked = cands[:1]
    return picked


async def metadata_path(state: TaskState, query: str, plan: dict, pool: Pool,
                        deadline: float) -> list[Candidate]:
    """Author/venue/year filters. There is no date parameter on any tool, so
    year filtering happens here in Python."""
    find_author = _tool(state, "search_authors_by_name")
    author_papers = _tool(state, "get_author_papers")

    if find_author and author_papers and plan["authors"]:
        async def by_author(name: str):
            recs = _parse_items(await find_author(name=name,
                                                  fields="authorId,name,paperCount",
                                                  limit=10))
            # Same person often has several fragmentary ids; prefer the
            # name-matching one with the largest paperCount.
            recs = [r for r in recs if r.get("authorId")]
            exact = [r for r in recs
                     if (r.get("name") or "").lower().strip() == name.lower().strip()]
            pick = sorted(exact or recs,
                          key=lambda r: -(r.get("paperCount") or 0))[:2]
            for rec in pick:
                docs = _parse_items(await author_papers(
                    author_id=str(rec["authorId"]), paper_fields=PAPER_FIELDS, limit=500))
                for r, d in enumerate(docs):
                    pool.add(d, r, 2.0)
                print(f"  author {name!r} id={rec['authorId']} -> {len(docs)}")

        await _gather([by_author(a) for a in plan["authors"]], 3)

    await broad_retrieve(state, query, plan, pool, deadline)

    ymin, ymax = plan["year_min"], plan["year_max"]
    venues = [v.lower() for v in plan["venues"]]
    authors_l = [a.lower() for a in plan["authors"]]
    kept = []
    for c in pool.ranked():
        if ymin is not None and (c.year is None or c.year < ymin):
            continue
        if ymax is not None and (c.year is None or c.year > ymax):
            continue
        if venues and not any(v in (c.venue or "").lower() for v in venues):
            continue
        if authors_l:
            names = " ; ".join(c.authors).lower()
            if not all(any(part in names for part in a.split()[-1:]) for a in authors_l):
                continue
        kept.append(c)
    print(f"  metadata filter kept {len(kept)} of {len(pool.by_cid)}")

    if not kept:
        kept = pool.ranked()[:20]
    if len(kept) > 60:
        await rerank(query, plan["criteria"], kept[:MAX_RERANK], deadline)
        kept = [c for c in final_order(kept) if (c.g2 if c.g2 is not None else c.g1 or 0) >= 2] \
            or final_order(kept)[:40]
    return kept[:80]


# ------------------------------------------------------------------ solver --

@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        t0 = time.time()
        deadline = t0 + WALL_BUDGET
        query = state.metadata.get("raw_query") or state.input_text or ""
        stype = state.metadata.get("score_type", "") or "semantic_f1"
        print(f"[{state.sample_id}] score_type={stype} query={query[:110]!r}")

        pool = Pool()
        picked: list[Candidate] = []
        try:
            plan = await analyze(query, stype)
            print(f"  keywords={plan['keyword_queries']}")
            print(f"  criteria={plan['criteria']}")
            if plan["candidate_titles"]:
                print(f"  titles={plan['candidate_titles']}")

            if stype == "specific_f1":
                picked = await specific_path(state, query, plan, pool, deadline)
            elif stype == "metadata_f1":
                picked = await metadata_path(state, query, plan, pool, deadline)
            else:
                await broad_retrieve(state, query, plan, pool, deadline)
                cands = pool.ranked()
                print(f"  pooled {len(cands)} unique candidates")
                await enrich_missing_abstracts(state, cands[:MAX_RERANK])
                await rerank(query, plan["criteria"], cands, deadline)
                picked = final_order(cands)[:MAX_SUBMIT]
        except Exception as exc:
            print(f"  [error] pipeline failed: {type(exc).__name__}: {exc}")

        # Degrade gracefully rather than emit an empty/invalid payload: an
        # unparseable submission scores 0 for the whole query.
        if not picked:
            picked = pool.ranked()[:MAX_SUBMIT]

        seen, results = set(), []
        for c in picked:
            if not c.cid or c.cid in seen:
                continue
            seen.add(c.cid)
            results.append({
                "paper_id": c.cid,
                # Ignored on the exact-match paths; the ONLY text the judge
                # sees on semantic_f1 — so spend the full grounded budget.
                "markdown_evidence": c.evidence() if stype == "semantic_f1" else _clip(c.title, 300),
            })
            if len(results) >= MAX_SUBMIT:
                break

        state.output.completion = json.dumps(
            {"output": {"query_id": state.sample_id, "results": results}})

        top = [(c.cid, c.g2 if c.g2 is not None else c.g1) for c in picked[:10]]
        print(f"  submitted {len(results)} papers in {time.time() - t0:.0f}s; top={top}")
        return state

    return solve
