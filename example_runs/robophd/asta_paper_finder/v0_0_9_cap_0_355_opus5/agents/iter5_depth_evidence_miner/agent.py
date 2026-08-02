"""depth-evidence-miner — a PaperFindingBench solver.

Base: `iter3_criterion_evidence_miner` (44.41, best of iteration 004).

The score decomposition across iteration 004's 11 semantic problems says exactly
where the remaining points are:

    mean rank = 0.72-0.90      mean recall = 0.23

`score = harmonic(rank, recall)`, so at that operating point a point of recall is
worth ~10 points of rank. And `recall = |{i <= K : g_i == 3}| / K` — the count of
grade-3 papers inside the first K submitted. Two things gate that count:

  (a) whether a paper's `markdown_evidence` proves EVERY criterion (grade 3 needs
      weighted > 0.99, so one unproven criterion caps it at grade 2 and earns
      zero recall), and
  (b) whether that paper is inside the first K positions.

Observed K across the batch: 8, 14, 18, 20, 46, 52, 56, 114, 180, 222, 304 —
**four of eleven exceed 100**. The base agent mined evidence for only the top 96
and judge-graded only the top 96/104; every paper past that carried abstract-only
evidence and a cheap stage-1 ordering, *inside the scored window*. On
`semantic_196` (K=304) that is 150+ judged papers running on the cheap path.

Changes, all aimed at the recall term:

A. DEPTH. Evidence mining now covers the whole submitted list (240 papers, two
   tiers) instead of 96. Mining is `snippet_search` — a **free** tool call — so
   the only budget it spends is wall-clock, and iteration 004 measured 168 mining
   calls inside 780 s against a 29-minute limit. Judge-replica grading extends to
   250: Sonnet over the top 130 (where ordering is worth most), GPT-5.4-mini over
   131-250 (inside K for the large-K queries, but cheap). Stage-1 breadth 400 ->
   550. Measured base cost was $0.155/query against a $0.355 free zone; the new
   depth lands near $0.24, still inside it.

B. CRITERION-TAGGED EVIDENCE. Mined passages now remember which criterion's query
   retrieved them, and evidence assembly allocates one slot per criterion in
   weight order before spending the rest. The base agent's greedy term-overlap
   selector could spend all four snippet slots on the criterion that was already
   best covered — the exact failure that leaves a grade-2.

C. GOLD-SHAPED CRITERIA. `gold_criteria.md` across the training tree shows the
   judge's criteria are atomic concepts named in the *query's own vocabulary*,
   2-4 of them, and that a "Relation Between A and B" conjunction criterion with
   weight ~0.2 is a recurring gold pattern. The analysis prompt now teaches that
   shape from six real (query -> gold criteria) pairs instead of imposing a
   topic/method/evaluation template.

D. FINER REPLICA SCALE. The base replica rated criteria in {0,1,3} and saturated:
   28 papers tied at weighted 1.0 on `semantic_104`, where the real judge graded
   most of them 2. Ordering inside a tie block is arbitrary, and with K=56 that
   is pure lost recall. Ratings are now {0,1,2,3} with 2 = "strongly implied but
   not stated outright", which keeps the judge's arithmetic but produces a
   continuous key that separates explicit proof from near-proof.

Not done, and worth recording: a `get_citations` path for "papers that cite X"
metadata queries. Probed on RoBERTa (corpusId 198953378) at limit 1000 — the
window came back **entirely 2025** and held **0 of the 70 gold ids**. The tool is
newest-first with no paging, so on a heavily-cited landmark it is structurally
unable to reach the cohort such queries ask for.
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
MAX_RERANK = 550          # stage-1 grading budget (candidates)
HEAD_SIZE = 250           # papers carried into evidence mining + grading

# Observed K (the recall denominator, = the judged depth) across iteration 004:
# 8, 14, 18, 20, 46, 52, 56, 114, 180, 222, 304. Anything inside K is scored, so
# depth here is not "nice to have" — it is the recall term.
MINE_DEEP = 120           # tier A: one scoped call per criterion, groups of 3
MINE_DEPTH = 240          # tier B ends here; groups of 5, top-3 criteria
GRADE_DEEP = 130          # judge-replica depth on CLAUDE_SONNET_4_6
GRADE_TAIL = 250          # ... and on GPT_5_4_MINI beyond that
MINE_GROUP = 3            # paper_ids per scoped snippet_search call (tier A)
MINE_GROUP_B = 5          # ... and tier B
MINE_LIMIT = 9            # passages per scoped call
STAGE1_BATCH = 22
STAGE2_BATCH = 7
STAGE2_TAIL_BATCH = 10
TOOL_CONCURRENCY = 6
MINE_CONCURRENCY = 14
LLM_CONCURRENCY = 8
WALL_BUDGET = 23 * 60     # hard stop well inside the 29-minute timeout
MINE_BUDGET = 7 * 60      # cap on the mining phase specifically
GRADE_RESERVE = 4 * 60    # wall-clock kept back so stage 2 always runs

PAPER_FIELDS = "title,abstract,corpusId,year,venue,authors,tldr,citationCount"
# get_author_papers rejects `tldr` outright ("Unrecognized or unsupported
# fields: [tldr]") — verified on the probe, and it is what silently killed the
# author path in BOTH prior iterations (iteration 002's metadata_15 stdout shows
# the same ToolError). It needs its own, narrower field list.
AUTHOR_PAPER_FIELDS = "title,abstract,corpusId,year,venue,authors"

# Evidence assembly budgets (chars). Title+tldr+abstract must not eat the whole
# 2400 cap, or there is no room for the body passages that prove criteria 2+.
ABSTRACT_CHARS = 780
TLDR_CHARS = 250
SNIPPET_CHARS = 400
MAX_SNIPPETS = 5


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
    """Best-effort JSON extraction from an LLM completion."""
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
    """Truncate at a sentence, then word, boundary. A prefix of retrieved text
    stays verbatim-derivable, which is what the grounding check requires."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "; ", ", ", " "):
        idx = cut.rfind(sep)
        if idx > limit * 0.55:
            return cut[:idx + (1 if sep == ". " else 0)].strip()
    return cut.strip()


_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "that",
    "must", "be", "is", "are", "as", "by", "at", "from", "it", "its", "this",
    "paper", "papers", "study", "studies", "research", "work", "should",
    "explicitly", "such", "which", "their", "they", "not", "any", "using",
    "use", "used", "these", "those", "into", "than", "then", "there", "have",
    "has", "how", "what", "can", "may", "does", "do", "specifically", "e.g",
}


def _terms(text: str) -> set[str]:
    """Content words used for cheap lexical criterion-coverage scoring."""
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", (text or "").lower())
            if w not in _STOP}


async def _gen(handle, prompt: str, *, config=None, tag: str = ""):
    """Metered LLM call with an empty-completion guard (surfaced, not hidden)."""
    try:
        resp = await (handle.generate(prompt, config=config) if config
                      else handle.generate(prompt))
        text = (resp.completion or "").strip()
        if not text:
            print(f"  [warn] empty completion from {tag}")
        return text
    except Exception as exc:  # never let one call sink the query
        print(f"  [warn] LLM call failed ({tag}): {type(exc).__name__}: {exc}")
        return ""


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


# --------------------------------------------------------------- candidates --

class Candidate:
    __slots__ = ("cid", "title", "abstract", "tldr", "year", "venue",
                 "authors", "snippets", "csnips", "fusion", "g1", "weighted",
                 "crit")

    def __init__(self, cid: str):
        self.cid = cid
        self.title = ""
        self.abstract = ""
        self.tldr = ""
        self.year = None
        self.venue = ""
        self.authors: list[str] = []
        self.snippets: list[str] = []
        # Passages keyed by the index of the criterion whose scoped query
        # retrieved them. Evidence assembly spends one slot per criterion before
        # anything else, so no criterion goes unproven while another gets four
        # redundant passages.
        self.csnips: dict[int, list[str]] = {}
        self.fusion = 0.0
        self.g1 = None          # stage-1 holistic grade 0-3
        self.weighted = None    # stage-2 judge-replica weighted score in [0,1]
        self.crit: list[int] = []

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
                self.authors = [a.get("name", "") for a in au
                                if isinstance(a, dict)][:12]

    def add_snippet(self, text: str, crit_idx: int | None = None) -> bool:
        text = (text or "").strip()
        if not text or len(text) < 60:
            return False
        for existing in self.snippets:
            if text[:80] == existing[:80]:
                return False
        self.snippets.append(text)
        if crit_idx is not None:
            self.csnips.setdefault(crit_idx, []).append(text)
        return True

    def brief(self, abs_chars: int) -> str:
        body = self.tldr or self.abstract
        if self.tldr and self.abstract:
            body = self.tldr + " " + self.abstract
        yr = f" ({self.year})" if self.year else ""
        return f"{self.title}{yr}. {_clip(body, abs_chars)}"

    def evidence(self, criteria: list[dict]) -> str:
        """Verbatim passages joined by ' ... ', allocated ONE PER CRITERION.

        Every passage is a (sentence-boundary) prefix of text the tools returned
        for this same paper, so it stays verbatim-derivable for the grounding
        check. Nothing here is model-written.

        Grade 3 — the only grade that earns recall — needs `weighted > 0.99`,
        i.e. essentially every criterion rated Perfectly Relevant. So the scarce
        resource is not evidence *length*, it is evidence *coverage*: a fifth
        passage about criterion 1 is worth nothing if criterion 3 has none.
        Criteria are visited in descending weight, each taking the passage its
        own scoped query retrieved (falling back to best term overlap), before
        any leftover budget is spent.
        """
        parts: list[str] = []
        used = 0

        def push(piece: str, floor: int = 20) -> None:
            nonlocal used
            piece = _clip(piece, max(0, EVIDENCE_CAP - used - 5))
            if piece and len(piece) >= floor:
                parts.append(piece)
                used += len(piece) + 5

        # The title always goes in, however short: a paper whose evidence string
        # is empty is scored Not Relevant with no judge call at all.
        if self.title:
            push(self.title, floor=1)
        if self.tldr:
            push(_clip(self.tldr, TLDR_CHARS))
        if self.abstract:
            push(_clip(self.abstract, ABSTRACT_CHARS))

        if not self.snippets:
            return " ... ".join(parts[:8])
        if not criteria:
            for sn in self.snippets[:MAX_SNIPPETS]:
                if used >= EVIDENCE_CAP - 200:
                    break
                push(_clip(sn, SNIPPET_CHARS))
            return " ... ".join(parts[:8])

        crit_terms = [_terms(c.get("name", "") + " " + c.get("description", ""))
                      for c in criteria]
        covered = _terms(" ".join(parts))
        taken: set[str] = set()
        order = sorted(range(len(criteria)),
                       key=lambda i: -criteria[i].get("weight", 0.0))
        slots = 0

        def _take(sn: str) -> None:
            nonlocal slots
            taken.add(sn[:80])
            push(_clip(sn, SNIPPET_CHARS))
            covered.update(_terms(sn))
            slots += 1

        for ci in order:
            if slots >= MAX_SNIPPETS or used >= EVIDENCE_CAP - 220:
                break
            ct = crit_terms[ci]
            # A criterion the title/abstract already covers well does not need a
            # slot; the budget is better spent on one that is still unproven.
            if ct and len(ct & covered) / len(ct) >= 0.85:
                continue
            pool = [sn for sn in self.csnips.get(ci, []) if sn[:80] not in taken]
            if not pool:
                pool = [sn for sn in self.snippets if sn[:80] not in taken]
                if ct and pool:
                    pool = [max(pool, key=lambda sn: len(ct & _terms(sn)))]
            if pool:
                _take(pool[0])

        # Leftover slots: whichever remaining passage best serves the criterion
        # that is still least covered.
        while slots < MAX_SNIPPETS and used < EVIDENCE_CAP - 220:
            rest = [sn for sn in self.snippets if sn[:80] not in taken]
            if not rest:
                break
            gaps = [1.0 - (len(ct & covered) / len(ct)) if ct else 0.0
                    for ct in crit_terms]
            best, best_score = rest[0], -1.0
            for sn in rest:
                st = _terms(sn)
                sc = sum(g * (len(ct & st) / len(ct))
                         for g, ct in zip(gaps, crit_terms) if ct)
                if sc > best_score:
                    best, best_score = sn, sc
            _take(best)

        return " ... ".join(parts[:8])


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


# ---------------------------------------------------------- query analysis --

ANALYSIS_PROMPT = """You are planning a scientific-literature search over Semantic Scholar.

USER REQUEST: {query}
REQUEST TYPE: {stype}

Return ONLY a JSON object with these keys:

"keyword_queries": 8 short keyword/noun-phrase search queries (3-8 words, NO
  question words, NO punctuation, NO "papers about"). The keyword engine returns
  ZERO hits for interrogative phrasing, so use bare topical noun phrases. Make
  them DIVERSE: vary terminology, include synonyms/abbreviations/acronyms the
  literature actually uses, and cover each distinct facet of the request
  separately (topic alone, method alone, application alone, evaluation alone).
"snippet_queries": 3 full natural-language sentences describing what a
  qualifying paper's TEXT would say. These go to a passage-retrieval engine that
  matches body paragraphs, so phrase them the way an author would write the
  finding (e.g. "We report micro-F1 averaged over test episodes").
"criteria": 2-4 objects {{"name": short label, "description": one sentence
  stating a property the paper's text must explicitly demonstrate, "weight":
  float summing to 1.0 across the list}}.
  Name the ATOMIC CONCEPTS THE REQUEST ITSELF STATES, one per criterion, reusing
  the request's own vocabulary. Do NOT invent requirements the request never
  makes, and do NOT force a topic/method/evaluation template. When the request
  joins two concepts, it is usually right to add a final low-weight criterion
  for the CONJUNCTION of them. These are real examples of the target shape:

    "clustering-based efficient attention mechanism within Transformer models"
      -> Clustering-based Attention Mechanism 0.55 | Efficiency Improvements
         0.35 | Transformer Models 0.10
    "papers that use decoupled workers in distributed RL system"
      -> decoupled workers 0.8 | distributed RL 0.2
    "generative document retrieval models that allow quick addition of new
     documents as they become available"
      -> Generative Document Retrieval Models 0.4 | Quick Addition of New
         Documents 0.4 | Relation Between Generative Models and Quick Document
         Addition 0.2
    "vision language models that use perceivers to encode images"
      -> Vision-Language Models 0.3 | Use of Perceivers 0.4 | Perceivers for
         Image Encoding 0.3
    "common model architectures for retrieval-augmented language models"
      -> Retrieval-Augmented Language Models 0.5 | Model Architectures 0.3 |
         Commonality of Architectures 0.2
    "adaptive query expansion with LLMs, papers published on or after 2023"
      -> Adaptive Query Expansion 0.4 | Use of Large Language Models 0.4 |
         Relation Between LLMs and Adaptive Query Expansion 0.2

  Give the largest weight to whichever concept is the request's real subject —
  that is often the unusual/specific one, not the broad field name.
"candidate_titles": if the request asks for a specific known paper (possibly by
  a nickname such as "the BART paper", a system name, or a one-line
  description), list 1-5 GUESSES at its exact published title, best first. Use
  the real title only, with no subtitle prefix such as "SystemName:" unless the
  published title truly begins that way. Otherwise [].
"reference_titles": exact titles of papers the request merely REFERS TO as a
  landmark rather than asks for (e.g. "authors of the BERT paper" ->
  ["BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding"]). Otherwise [].
"authors": full author names the request requires, else [].
"venues": exact venue names the request requires (e.g. "Nature", "NeurIPS"), else [].
"venue_aliases": lowercase substrings that a matching venue string might contain,
  including the expanded form (e.g. for NAACL: ["naacl", "north american chapter"]),
  else [].
"years": explicit list of individual years the request allows (e.g. "2010 or
  2012" -> [2010, 2012]), else [].
"year_min": integer or null.  "year_max": integer or null.
"""

DEFAULT_CRITERION = {"name": "matches the request",
                     "description": "The paper matches the user request.",
                     "weight": 1.0}


async def analyze(query: str, stype: str) -> dict:
    text = await _gen(GPT_5_4, ANALYSIS_PROMPT.format(query=query, stype=stype),
                      tag="analysis")
    data = _jload(text)
    if not isinstance(data, dict):
        data = {}

    def _strlist(key, cap):
        v = data.get(key) or []
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v
                if isinstance(x, (str, int)) and str(x).strip()][:cap]

    def _intlist(key):
        v = data.get(key) or []
        out = []
        if isinstance(v, list):
            for x in v:
                try:
                    out.append(int(x))
                except (TypeError, ValueError):
                    continue
        return out

    # Criteria: normalise to {name, description, weight} with weights summing to 1.
    criteria: list[dict] = []
    raw_crit = data.get("criteria")
    if isinstance(raw_crit, list):
        for c in raw_crit[:4]:
            if isinstance(c, dict):
                name = str(c.get("name") or "").strip()
                desc = str(c.get("description") or "").strip()
                if not (name or desc):
                    continue
                try:
                    w = float(c.get("weight", 0))
                except (TypeError, ValueError):
                    w = 0.0
                criteria.append({"name": name or desc[:40],
                                 "description": desc or name,
                                 "weight": max(w, 0.0)})
            elif isinstance(c, str) and c.strip():
                criteria.append({"name": c.strip()[:40],
                                 "description": c.strip(), "weight": 0.0})
    if not criteria:
        criteria = [dict(DEFAULT_CRITERION)]
    total = sum(c["weight"] for c in criteria)
    if total <= 0:
        for c in criteria:
            c["weight"] = 1.0 / len(criteria)
    else:
        for c in criteria:
            c["weight"] = c["weight"] / total

    kws = _strlist("keyword_queries", 8)
    if not kws:
        # Fallback: strip interrogative framing ourselves — the keyword engine
        # returns zero hits on question-shaped input.
        stripped = re.sub(
            r"^(could you|can you|please|i am looking for|i'm looking for|show me|"
            r"find me|are there|is there|has any|what are|what is|how can|how do|"
            r"do you know of|any)\b[^,]*?\b(papers?|research|studies|work|study)\b\s*"
            r"(that|which|on|about|for|using)?\s*", "", query.strip(), flags=re.I)
        kws = [stripped.strip(" ?.!,") or query]

    return {
        "keyword_queries": kws,
        "snippet_queries": _strlist("snippet_queries", 3),
        "criteria": criteria,
        "candidate_titles": _strlist("candidate_titles", 5),
        "reference_titles": _strlist("reference_titles", 3),
        "authors": _strlist("authors", 6),
        "venues": _strlist("venues", 4),
        "venue_aliases": [s.lower() for s in _strlist("venue_aliases", 8)],
        "years": _intlist("years"),
        "year_min": data.get("year_min") if isinstance(data.get("year_min"), int) else None,
        "year_max": data.get("year_max") if isinstance(data.get("year_max"), int) else None,
    }


# ------------------------------------------------------------ broad search --

async def broad_retrieve(state: TaskState, query: str, plan: dict, pool: Pool,
                         deadline: float) -> None:
    """Fan out keyword searches + snippet searches. All of this is free."""
    search = _tool(state, "search_papers_by_relevance")
    snippet = _tool(state, "snippet_search")
    venues = ",".join(plan["venues"]) if plan["venues"] else None

    async def kw(q: str, weight: float):
        if not search or time.time() > deadline:
            return
        kwargs = {"keyword": q, "fields": PAPER_FIELDS, "limit": 100}
        if venues:
            kwargs["venues"] = venues
        # Keyword search is normally sub-second but has been observed to stall on
        # a cold server; without a bound one slow call can eat the whole phase.
        try:
            docs = _parse_items(await asyncio.wait_for(search(**kwargs),
                                                      timeout=150))
        except asyncio.TimeoutError:
            print(f"  kw {q!r} timed out")
            return
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
        n = 0
        for r, e in enumerate(_parse_items(raw)):
            paper = e.get("paper") if isinstance(e.get("paper"), dict) else e
            cand = pool.add(paper, r, weight)
            if cand is None:
                continue
            n += 1
            sn = e.get("snippet")
            txt = (sn.get("text") if isinstance(sn, dict) else None) or ""
            if len(cand.snippets) < 6:
                cand.add_snippet(txt)
        print(f"  snippet {q[:50]!r} -> {n}")

    tasks = [kw(q, 1.0 if i == 0 else 0.85)
             for i, q in enumerate(plan["keyword_queries"])]
    # snippet_search tolerates sentence-shaped input, so feed it the raw query
    # plus the analyzer's author-voice paraphrases.
    snip_qs = [query] + plan["snippet_queries"][:2]
    tasks += [snip(q, 1.0 if i == 0 else 0.8) for i, q in enumerate(snip_qs)]
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


# ------------------------------------------------------- evidence mining ----

async def mine_evidence(state: TaskState, head: list[Candidate], plan: dict,
                        deadline: float) -> None:
    """Pull body passages that PROVE each criterion, per paper — for the WHOLE
    submitted list, not just its head.

    The judge sees only `markdown_evidence`, and it judges every paper inside K.
    Observed K reached 304, 222 and 180 on iteration 004's batch while evidence
    mining stopped at 96 — so 150+ scored papers were carrying abstract-only
    evidence, which almost never proves a method/setting criterion and therefore
    caps them at grade 2 (zero recall). `snippet_search` is a free tool call; the
    only thing depth spends is wall-clock, and 168 mining calls measured 780 s
    against a 29-minute limit.

    Two tiers, because value per call falls with depth: groups of 3 with one call
    per criterion over the top MINE_DEEP, then groups of 5 over the top-3
    criteria to MINE_DEPTH. Groups stay small because `paper_ids` is a scope
    filter, not a per-paper allocation — probed with 8 ids at limit 24, two
    papers got zero passages.
    """
    snippet = _tool(state, "snippet_search")
    if not snippet or not head:
        return

    criteria = plan["criteria"]
    if not criteria:
        return
    queries = [(i, f"{c['name']}. {c['description']}")
               for i, c in enumerate(criteria)]
    # tier B keeps only the heaviest three criteria
    tierb_q = sorted(queries, key=lambda t: -criteria[t[0]].get("weight", 0.0))[:3]

    tier_a = head[:MINE_DEEP]
    tier_b = head[MINE_DEEP:MINE_DEPTH]
    groups_a = [tier_a[i:i + MINE_GROUP] for i in range(0, len(tier_a), MINE_GROUP)]
    groups_b = [tier_b[i:i + MINE_GROUP_B]
                for i in range(0, len(tier_b), MINE_GROUP_B)]
    # Never let mining eat the grading phase: stage 2 is what orders the
    # papers that mining just made gradeable.
    stop = min(deadline - GRADE_RESERVE, time.time() + MINE_BUDGET)
    hits = 0

    async def mine(group: list[Candidate], ci: int, q: str):
        nonlocal hits
        if time.time() > stop:
            return
        ids = ",".join(f"CorpusId:{c.cid}" for c in group)
        try:
            raw = await asyncio.wait_for(
                snippet(query=q, paper_ids=ids, limit=MINE_LIMIT), timeout=90)
        except Exception:
            return
        by_cid = {c.cid: c for c in group}
        for e in _parse_items(raw):
            paper = e.get("paper") if isinstance(e.get("paper"), dict) else {}
            cand = by_cid.get(_cid(paper))
            if cand is None or len(cand.snippets) >= 12:
                continue
            sn = e.get("snippet")
            txt = (sn.get("text") if isinstance(sn, dict) else None) or ""
            if cand.add_snippet(txt, ci):
                hits += 1

    # Interleave by criterion so a truncated run still leaves every paper's
    # heaviest criterion mined, and order groups head-first so the papers whose
    # position matters most are served before any budget runs out.
    tasks = [mine(g, ci, q) for g in groups_a for ci, q in queries]
    tasks += [mine(g, ci, q) for g in groups_b for ci, q in tierb_q]
    await _gather(tasks, MINE_CONCURRENCY)
    scope = head[:MINE_DEPTH]
    with_sn = sum(1 for c in scope if c.snippets)
    print(f"  mined {hits} passages via {len(tasks)} scoped calls; "
          f"{with_sn}/{len(scope)} papers have body evidence")


# ---------------------------------------------------------------- reranker --

def _crit_block(criteria: list[dict]) -> str:
    return "\n".join(
        f"  C{i + 1} (weight {c['weight']:.2f}) {c['name']}: {c['description']}"
        for i, c in enumerate(criteria))


STAGE1_RULES = """Grade each candidate 0-3 on how well its text satisfies EVERY requirement:
  3 = the text explicitly satisfies ALL requirements, directly and unambiguously
  2 = satisfies most requirements but at least one is unclear or only implied
  1 = same general topic, but clearly misses a requirement
  0 = not relevant
Only give 3 when every single requirement is explicitly demonstrated by the text."""


def _stage1_prompt(query: str, criteria: list[dict], batch: list[Candidate],
                   abs_chars: int) -> str:
    lines = "\n".join(f"[{i}] {c.brief(abs_chars)}" for i, c in enumerate(batch))
    return (
        f"USER REQUEST: {query}\n\nREQUIREMENTS a fully relevant paper must meet:\n"
        f"{_crit_block(criteria)}\n\n{STAGE1_RULES}\n\nCANDIDATES:\n{lines}\n\n"
        f"Reply with exactly {len(batch)} lines, one per candidate, formatted "
        f"`index:grade` (e.g. `0:3`). No other text."
    )


def _stage2_prompt(query: str, criteria: list[dict],
                   batch: list[tuple[int, Candidate, str]]) -> str:
    """Replica of the benchmark judge: rate EACH criterion separately.

    The real judge rates every criterion in {Not, Somewhat, Perfectly} = {0,1,3}
    and combines them as min(1, sum(w_c * r_c / 3)); a paper reaches grade 3 —
    the only grade that earns recall — essentially only when every criterion is
    Perfectly Relevant. Asking for the same decomposition, over the same text the
    judge will see, makes this a dry run rather than a holistic guess.
    """
    n = len(criteria)
    lines = "\n\n".join(f"[{i}] {ev}" for i, _, ev in batch)
    return (
        f"You are a strict relevance judge for a literature-search benchmark.\n\n"
        f"USER REQUEST: {query}\n\nCRITERIA:\n{_crit_block(criteria)}\n\n"
        f"For each paper below you are given ONLY its evidence text. Judge each "
        f"criterion using that text ALONE — do not use outside knowledge about "
        f"the paper, and do not give credit for a criterion the text merely "
        f"implies.\n\nRate every criterion:\n"
        f"  3 = the text states this criterion outright, in so many words\n"
        f"  2 = the text makes it clear but never quite says it\n"
        f"  1 = the text only touches on it in passing\n"
        f"  0 = the text does not support it\n\n"
        f"Most papers on the right general topic still fail at least one "
        f"criterion outright — use 3 only for text that leaves no doubt.\n\n"
        f"PAPERS:\n{lines}\n\n"
        f"Reply with exactly {len(batch)} lines, one per paper, formatted\n"
        f"`index: r1,r2,...,r{n}`  (one rating per criterion, in order; "
        f"e.g. `0: 3,1,0`)\nNo other text."
    )


def _apply_stage1(text: str, batch: list[Candidate]) -> None:
    for m in re.finditer(r"(\d+)\s*[:=]\s*([0-3])\s*(?:$|\n)", (text or "") + "\n"):
        i, g = int(m.group(1)), int(m.group(2))
        if 0 <= i < len(batch):
            batch[i].g1 = g


def _grade_from_weighted(weighted: float) -> int:
    if weighted <= 0.25:
        return 0
    if weighted <= 0.67:
        return 1
    if weighted <= 0.99:
        return 2
    return 3


def _apply_stage2(text: str, batch: list[tuple[int, Candidate, str]],
                  criteria: list[dict]) -> int:
    """Parse `index: r1,r2,r3` lines and recompute the judge's own arithmetic."""
    n_applied = 0
    for line in (text or "").splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s*[:=]\s*([0-3](?:\s*[, ]\s*[0-3])*)\s*$", line)
        if not m:
            continue
        idx = int(m.group(1))
        if not (0 <= idx < len(batch)):
            continue
        ratings = [int(x) for x in re.findall(r"[0-3]", m.group(2))]
        if not ratings:
            continue
        ratings = (ratings + [0] * len(criteria))[:len(criteria)]
        weighted = min(1.0, sum(c["weight"] * r / 3.0
                                for c, r in zip(criteria, ratings)))
        cand = batch[idx][1]
        cand.weighted = weighted
        cand.crit = ratings
        n_applied += 1
    return n_applied


async def rerank(query: str, plan: dict, cands: list[Candidate],
                 deadline: float) -> list[Candidate]:
    """Stage 1: cheap model over everything, to pick the head.
    Stage 2: judge replica over the head's actual submitted evidence."""
    criteria = plan["criteria"]
    stage1 = cands[:MAX_RERANK]
    batches = [stage1[i:i + STAGE1_BATCH]
               for i in range(0, len(stage1), STAGE1_BATCH)]

    async def do1(batch):
        if time.time() > deadline:
            return
        txt = await _gen(GPT_5_4_MINI, _stage1_prompt(query, criteria, batch, 500),
                         tag="stage1")
        _apply_stage1(txt, batch)

    await _gather([do1(b) for b in batches], LLM_CONCURRENCY)
    print(f"  stage1 graded {sum(1 for c in stage1 if c.g1 is not None)}/{len(stage1)}")
    for c in stage1:
        if c.g1 is None:
            c.g1 = 1 if c.abstract else 0

    return sorted(stage1, key=lambda c: (-(c.g1 or 0), -c.fusion))[:HEAD_SIZE]




async def judge_replica(query: str, plan: dict, head: list[Candidate],
                        deadline: float) -> None:
    """Dry-run the benchmark judge over the exact evidence that will be
    submitted, at two price points.

    Positions 1..GRADE_DEEP get CLAUDE_SONNET_4_6: this is where ordering decides
    recall on the small-K queries (K was 8, 14, 18 and 20 on four of iteration
    004's problems, so the top ~20 slots carried the entire score there).
    Positions GRADE_DEEP+1..GRADE_TAIL get GPT_5_4_MINI at a fifth of the input
    price — they are still inside K on the large-K queries (114, 180, 222, 304),
    where the base agent left them ordered by a holistic stage-1 grade alone.
    """
    criteria = plan["criteria"]
    items = [(i, c, _clip(c.evidence(criteria), 1600))
             for i, c in enumerate(head[:GRADE_TAIL])]
    items = [t for t in items if t[2]]
    deep = [t for t in items if t[0] < GRADE_DEEP]
    tail = [t for t in items if t[0] >= GRADE_DEEP]
    applied = 0

    async def do2(batch, handle, tag):
        nonlocal applied
        if time.time() > deadline:
            return
        local = [(j, c, ev) for j, (_, c, ev) in enumerate(batch)]
        txt = await _gen(handle, _stage2_prompt(query, criteria, local),
                         config=(GenerateConfig(max_tokens=900)
                                 if handle is CLAUDE_SONNET_4_6 else None),
                         tag=tag)
        applied += _apply_stage2(txt, local, criteria)

    tasks = [do2(deep[i:i + STAGE2_BATCH], CLAUDE_SONNET_4_6, "stage2")
             for i in range(0, len(deep), STAGE2_BATCH)]
    tasks += [do2(tail[i:i + STAGE2_TAIL_BATCH], GPT_5_4_MINI, "stage2tail")
              for i in range(0, len(tail), STAGE2_TAIL_BATCH)]
    await _gather(tasks, LLM_CONCURRENCY)
    graded = [c for c in head if c.weighted is not None]
    pred3 = sum(1 for c in graded if _grade_from_weighted(c.weighted) == 3)
    print(f"  stage2 judged {applied}/{len(items)} "
          f"(sonnet {len(deep)}, mini {len(tail)}); predicted grade-3: {pred3}")


def final_order(head: list[Candidate], rest: list[Candidate]) -> list[Candidate]:
    """Head by predicted weighted score (a continuous value — no mass ties in
    the top-25, where ordering is worth the most), tail by stage-1 then fusion.
    Submitted deep: semantic_f1 has no precision penalty, and anything past K is
    simply never judged."""
    ordered_head = sorted(
        head,
        key=lambda c: (-(1 if c.weighted is not None else 0),
                       -round(c.weighted or 0.0, 4), -(c.g1 or 0),
                       -len(c.snippets), -c.fusion))
    ordered_rest = sorted(rest, key=lambda c: (-(c.g1 or 0), -c.fusion))
    return ordered_head + ordered_rest


# ------------------------------------------------------- exact-match paths --

async def resolve_titles(state: TaskState, titles: list[str], pool: Pool,
                         weight: float = 3.0) -> list[Candidate]:
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
            c = pool.add(d, rank, weight)
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
    """Precision-first: exact-match F1 punishes every extra id."""
    await resolve_titles(state, plan["candidate_titles"], pool)
    search = _tool(state, "search_papers_by_relevance")
    if search:
        async def kw(q, w):
            docs = _parse_items(await search(keyword=q, fields=PAPER_FIELDS, limit=25))
            for r, d in enumerate(docs):
                pool.add(d, r, w)
        qs = plan["candidate_titles"][:2] + plan["keyword_queries"][:3]
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
    return picked or cands[:1]


async def expand_authors(state: TaskState, names: list[str], pool: Pool) -> None:
    """search_authors_by_name -> get_author_papers. The same person often has
    several fragmentary ids; prefer the name-matching one with most papers."""
    find_author = _tool(state, "search_authors_by_name")
    author_papers = _tool(state, "get_author_papers")
    if not (find_author and author_papers and names):
        return

    async def by_author(name: str):
        recs = [r for r in _parse_items(await find_author(
            name=name, fields="authorId,name,paperCount", limit=10))
            if r.get("authorId")]
        exact = [r for r in recs
                 if (r.get("name") or "").lower().strip() == name.lower().strip()]
        for rec in sorted(exact or recs, key=lambda r: -(r.get("paperCount") or 0))[:2]:
            try:
                docs = _parse_items(await author_papers(
                    author_id=str(rec["authorId"]),
                    paper_fields=AUTHOR_PAPER_FIELDS, limit=500))
            except Exception as exc:
                print(f"  [warn] get_author_papers({rec['authorId']}): {exc}")
                continue
            for r, d in enumerate(docs):
                pool.add(d, r, 2.0)
            print(f"  author {name!r} id={rec['authorId']} -> {len(docs)}")

    await _gather([by_author(a) for a in names[:8]], 3)


async def metadata_path(state: TaskState, query: str, plan: dict, pool: Pool,
                        deadline: float) -> list[Candidate]:
    """Author/venue/year filters. No tool has a date parameter, so year
    filtering happens here in Python."""
    authors = list(plan["authors"])

    # "co-authored by one of the authors of the BERT paper": the query names no
    # author at all, so resolve the referenced paper and harvest its author list.
    # This is exactly what zeroed metadata_14 last iteration.
    if plan["reference_titles"]:
        ref_pool = Pool()
        refs = await resolve_titles(state, plan["reference_titles"], ref_pool, 0.0)
        for c in refs:
            for nm in c.authors:
                if nm and nm not in authors:
                    authors.append(nm)
        if refs:
            print(f"  reference papers -> authors {authors}")

    await expand_authors(state, authors, pool)
    await broad_retrieve(state, query, plan, pool, deadline)

    years = set(plan["years"])
    ymin, ymax = plan["year_min"], plan["year_max"]
    aliases = plan["venue_aliases"] or [v.lower() for v in plan["venues"]]
    # Surnames of REQUIRED authors only. Reference-paper authors are an OR-set
    # ("one of the authors of BERT"), so they must not become an AND filter.
    required = [a.lower().split()[-1] for a in plan["authors"] if a.split()]
    any_of = [a.lower().split()[-1] for a in authors if a.split()] \
        if plan["reference_titles"] and not plan["authors"] else []

    kept = []
    for c in pool.ranked():
        if years:
            if c.year not in years:
                continue
        else:
            if ymin is not None and (c.year is None or c.year < ymin):
                continue
            if ymax is not None and (c.year is None or c.year > ymax):
                continue
        if aliases and not any(v in (c.venue or "").lower() for v in aliases):
            continue
        names = " ; ".join(c.authors).lower()
        if required and not all(s in names for s in required):
            continue
        if any_of and not any(s in names for s in any_of):
            continue
        kept.append(c)
    print(f"  metadata filter kept {len(kept)} of {len(pool.by_cid)}")

    if not kept:
        kept = pool.ranked()[:20]
    elif len(kept) > 60:
        head = await rerank(query, plan, kept, deadline)
        strong = [c for c in head if (c.g1 or 0) >= 2]
        kept = strong or head[:40]
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
        plan = {"criteria": [dict(DEFAULT_CRITERION)]}
        try:
            plan = await analyze(query, stype)
            print(f"  keywords={plan['keyword_queries']}")
            print(f"  criteria={[(c['name'], round(c['weight'], 2)) for c in plan['criteria']]}")
            if plan["candidate_titles"]:
                print(f"  titles={plan['candidate_titles']}")
            if plan["reference_titles"]:
                print(f"  refs={plan['reference_titles']}")

            if stype == "specific_f1":
                picked = await specific_path(state, query, plan, pool, deadline)
            elif stype == "metadata_f1":
                picked = await metadata_path(state, query, plan, pool, deadline)
            else:
                await broad_retrieve(state, query, plan, pool, deadline)
                cands = pool.ranked()
                print(f"  pooled {len(cands)} unique candidates")
                await enrich_missing_abstracts(state, cands[:MAX_RERANK])
                head = await rerank(query, plan, cands, deadline)
                # Mine body evidence for the head, THEN judge that evidence:
                # stage 2 must score the same text the benchmark judge scores.
                await mine_evidence(state, head, plan, deadline)
                await judge_replica(query, plan, head, deadline)
                head_ids = {c.cid for c in head}
                rest = [c for c in cands if c.cid not in head_ids]
                picked = final_order(head, rest)[:MAX_SUBMIT]
        except Exception as exc:
            print(f"  [error] pipeline failed: {type(exc).__name__}: {exc}")

        # Degrade gracefully rather than emit an empty/invalid payload: an
        # unparseable submission scores 0 for the whole query.
        if not picked:
            picked = pool.ranked()[:MAX_SUBMIT]

        criteria = plan.get("criteria") or [dict(DEFAULT_CRITERION)]
        seen, results = set(), []
        for c in picked:
            if not c.cid or c.cid in seen:
                continue
            seen.add(c.cid)
            results.append({
                "paper_id": c.cid,
                # Ignored on the exact-match paths; the ONLY text the judge sees
                # on semantic_f1 — so spend the full grounded budget there.
                "markdown_evidence": (c.evidence(criteria) if stype == "semantic_f1"
                                      else _clip(c.title, 300)),
            })
            if len(results) >= MAX_SUBMIT:
                break

        state.output.completion = json.dumps(
            {"output": {"query_id": state.sample_id, "results": results}})

        if stype == "semantic_f1" and results:
            ev_lens = [len(r["markdown_evidence"]) for r in results[:HEAD_SIZE]]
            print(f"  head evidence chars: mean={sum(ev_lens) // max(len(ev_lens), 1)} "
                  f"max={max(ev_lens)}")
            print("  top10=" + str([(c.cid, c.crit, round(c.weighted, 2)
                                     if c.weighted is not None else None)
                                    for c in picked[:10]]))
        print(f"  submitted {len(results)} papers in {time.time() - t0:.0f}s")
        return state

    return solve
