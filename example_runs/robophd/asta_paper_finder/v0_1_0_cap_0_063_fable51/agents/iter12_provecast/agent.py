"""PaperFindingBench agent — iteration 012 ("provecast"), evolved from citecast.

Changes vs citecast (semantic path only; specific/metadata paths untouched):
* Body-text evidence for the whole judged window, not just 40 head papers: the top 100 of the
  ordered list get criterion-targeted passages from scoped snippet_search, fetched in groups of
  8 papers per call (probe: a warm 10-paper scoped call returns in ~4 s and covers 9/10 papers),
  with a per-paper retry for head papers the group pass left empty.  The judge grades each
  criterion from the evidence alone; on narrow queries 14/16 judged papers were "Somewhat"
  because title+abstract never states the evaluation-protocol / property criteria that live in
  the paper body.
* Evidence is rebuilt around the criteria: abstract capped at 1150 chars (was 1700, which left
  <400 chars for body passages), then up to 3 passages chosen by overlap with the criteria the
  abstract does not cover, each focused to a ~420-char window around the best-matching
  sentence, then the tldr.  All passages remain contiguous verbatim substrings.
* Planner snippet queries are criterion-targeted (one per required aspect, phrased like a
  methods/experiments sentence), so the unscoped passage searches also return criterion
  evidence for pool papers; up to 6 passages are kept per candidate.
* Yield-adaptive bulk grading: 360 fused candidates first, then 200 more at a time while the
  last 120 graded still yielded >=5 papers graded >=2 (cap 760).  Narrow queries stop at 360
  (saves ~4 mini batches), broad ones (where 10-22 grade>=2 papers sat in the fused tail) go
  deeper; cost-neutral on the iteration-11 mix.
* Head rerank text trimmed (620-char abstract, 160-char tldr) to pay for the deeper grading.

----- citecast header follows -----
PaperFindingBench agent — iteration 011 ("citecast"), evolved from clearcast.

Changes vs clearcast (metadata path only; semantic/specific paths untouched):
* "cites papers by author Y" is now resolved from BOTH directions: the candidates' reference
  lists (as before) AND the citers of Y's own papers via get_citations, intersected with the
  pool. The task-side wrapper raises "'NoneType' object is not iterable" on get_paper_batch /
  get_paper with the `references` field for many (esp. older) papers, so on metadata_31 only
  29/210 reference records were readable and 11/16 gold papers were lost; get_citations on Y's
  99 papers reaches them from the other side.
* New "cites any paper from venue V" constraint (metadata_33: "SPLASH paper that cites any
  NeurIPS"): candidates whose reference lists are readable are kept only if a referenced paper's
  venue matches V (referenced ids looked up in batches); unreadable candidates are kept
  (verification impossible, and the gold paper on metadata_33 is itself unreadable).
* Author retrieval augmented with a free keyword search for the author's name (venue-filtered
  when a venue is given), kept only when the returned paper's author list contains a compatible
  name: a probe showed `search_papers_by_relevance keyword="Ming-Wei Chang" venues=NAACL`
  returns 10/10 papers by that author, including the two metadata_14 gold papers the author-id
  fan-out never reached.
* Reference batches shrunk 40 -> 12 ids so a single unreadable paper wastes less splitting.

----- clearcast header follows -----
PaperFindingBench agent — iteration 010 ("clearcast"), evolved from steadycast/headcast.

Changes vs steadycast (iteration 9):
* The "publication_type" journal/conference post-filter is REMOVED. It zeroed metadata_14 in
  iteration 9 (578 author-pool papers -> 110 by a "looks like a conference" regex that does not
  match "North American Chapter of the Association for Computational Linguistics" -> 0 submitted,
  where headcast scored 0.80). The generic-word stripping of venue_patterns stays.
* The metadata planner is told that venue_patterns must identify exactly the venue named and
  must not widen a specific venue to its parent family (NAACL -> not "acl"); critcast and
  steadycast both lost precision on metadata_14 when the planner emitted "acl".
* Citation expansion slightly wider (10 seeds x 150 citers, 160 graded; was 8 x 100, 120) —
  large-K semantic queries are supply-bound (K=162 with only 39 grade>=2 papers available).
* Everything else — bulk grading, planner, head rerank by the judge-rule replica,
  criterion-targeted evidence, specific path — is steadycast's (semantic-path score identical
  to headcast's in iteration 9: 0.331 vs 0.334 mean over 10 queries).

----- steadycast header follows -----
PaperFindingBench agent — iteration 009 ("steadycast"), evolved from headcast.

Changes vs headcast (which beat its derivative critcast in iteration 8):
* Semantic bulk grading and planner criteria kept exactly as headcast's (critcast's stricter
  "must" criteria template halved the number of grade>=2 papers and starved the judged window
  on large-K queries: 38 Not-relevant in the first 94 on semantic_101 vs headcast's 4).
* Head rerank widened 28 -> 36 and ordered by a replica of the judge's combination rule
  (count of criteria rated Perfect first, then weighted mean), not the plain mean.
* Per-paper body-snippet evidence for the head is targeted at the criteria the rerank rated
  below Perfect for that paper (the ones the judge is most likely to call Somewhat).
* Two deterministic keyword queries built from the raw request are always searched alongside
  the planner's 12 (reduces planner-sample retrieval variance; free).
* Specific path: tool timeouts raised to the observed latency under load (130-180 s); an
  ambiguous nickname submits up to 4 picks.
* Metadata path: generic venue words ("journal", "conference") no longer act as venue
  substring filters; new "cites papers by author Y" constraint resolved through Y's author
  papers and the candidates' reference lists, with Y's own co-authored papers excluded.

----- headcast header follows -----
PaperFindingBench agent — iteration 007 ("headcast"), evolved from surecast.

Changes vs surecast (semantic path only; specific/metadata unchanged apart from a 6-title plan):
* The head of the semantic list (top 28 after bulk grading) is re-judged by GPT-5.4 with a
  per-criterion rubric that mirrors the benchmark judge (each criterion Not/Somewhat/Perfect,
  the paper's score is the mean), and re-ordered by that score.  Only the first K submitted
  papers are judged and K is small for narrow queries (6-20 in iteration 6, both scored 0 with
  the known-good paper sitting at position 23), so head precision is where the small-K score
  lives, and the first positions carry most of the rank term's DCG weight on every query.
* Bulk grading trimmed (520 candidates x 440 chars, was 600 x 480) to pay for the head rerank
  and stay inside the cost free zone.
* Per-paper body-snippet evidence for the head: a cold scoped snippet_search takes ~120 s, so
  the old 40 s timeout failed on essentially every call; now 140 s per call, 8 concurrent,
  40 papers, guarded by the query's wall-clock budget.
* Planner criteria and grader wording aligned with the judge's per-criterion semantics.

----- surecast header follows -----
PaperFindingBench agent — iteration 006 ("surecast"), evolved from broadcast/widecast.

Changes vs broadcast: widecast's strict 0-3 grader restored (broadcast's 0-10 scorer lost
rank-term on 4 of 10 semantic queries) while keeping the 600-deep grading; the specific-paper
solver now plans with two models (GPT-5.4 + Claude Sonnet), adds snippet-search and raw-token
keyword retrieval, retries timed-out searches, and submits 2-3 candidates when confidence is
not high (F1 math: a 2-of-3 hit beats a confident miss).

Changes vs widecast: 12+4 expansion queries, 600 (was 360) fused candidates graded
with a 0-10 relevance score calibrated to the benchmark judge (finer ordering),
8 citation-expansion seeds, 60-paper body-snippet head, and a venue-only sweep
on metadata queries whose pool would otherwise be empty.

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

from model_registry import GPT_5_4, GPT_5_4_MINI, CLAUDE_SONNET_4_6

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
MAX_SUBMIT = 250
MAX_EVIDENCE_CHARS = 2450
ABSTRACT_EVIDENCE_CHARS = 1150   # leaves room for 3 criterion passages
PASSAGE_EVIDENCE_CHARS = 420
PAPER_FIELDS = "title,abstract,corpusId,year,tldr,venue,citationCount,authors"
GRADE_BATCH = 40
GRADE_ABSTRACT_CHARS = 440
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


_STOP = set("""the a an of on in to for and or with by from as at is are be been being that this these those which
what how using used use based via into than then its it their there we our they them paper papers must should
specifically specific focus focuses focusing explicitly address addresses addressing discuss discusses discussing
involve involves involving include includes including such e.g i.e well also any all about where when while whether
between across within without model models method methods approach approaches task tasks study studies research
work works results result data performance paper's can may might not non only other others over under some
more most new one two three per each both either neither however therefore thus propose proposed proposes
related relevant relevance criteria criterion require required requires requirement clear clearly connection
central component part aspect aspects context terms term defined definition analyze analyzes analysis
demonstrate demonstrates demonstrating ensure ensuring provide provides providing""".split())


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", (text or "").lower()) if len(w) > 2 and w not in _STOP}


def _split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])", text)
    return [p for p in parts if p]


def _focus_window(text: str, words: set, limit: int) -> str:
    """Contiguous run of sentences (<= limit chars) around the sentence with the most hits
    of `words`; stays a verbatim substring up to whitespace normalisation."""
    sents = _split_sentences(text)
    if not sents:
        return ""
    if sum(len(s) + 1 for s in sents) <= limit:
        return " ".join(sents)
    best, best_score = 0, -1.0
    for i, s in enumerate(sents):
        sw = _content_words(s)
        sc = len(sw & words) + 0.001 * min(len(s), 300)
        if sc > best_score:
            best, best_score = i, sc
    lo = hi = best
    total = len(sents[best])
    if total > limit:
        return _trim_passage(sents[best], limit)
    while True:
        grew = False
        if hi + 1 < len(sents) and total + 1 + len(sents[hi + 1]) <= limit:
            hi += 1
            total += 1 + len(sents[hi])
            grew = True
        if lo - 1 >= 0 and total + 1 + len(sents[lo - 1]) <= limit:
            lo -= 1
            total += 1 + len(sents[lo])
            grew = True
        if not grew:
            break
    return " ".join(sents[lo:hi + 1])


def _passage_score(sn: str, weak: set, crit: set) -> float:
    sw = _content_words(sn)
    return 2.0 * len(sw & weak) + 1.0 * len(sw & crit)


def _build_evidence(c: dict) -> str:
    """Verbatim passages (title, abstract, criterion-targeted body passages, tldr) joined by ' ... '.
    When the candidate carries `crit_words` (semantic path), the abstract is capped and the
    passages are chosen/focused by overlap with the criteria the abstract does not cover;
    otherwise the legacy title/abstract/tldr/snippets order is used."""
    parts = []
    budget = MAX_EVIDENCE_CHARS
    title = re.sub(r"\s+", " ", c.get("title") or "").strip()
    if title:
        parts.append(title)
        budget -= len(title) + 5
    abstract = c.get("abstract") or ""
    tl = c.get("tldr_text") or ""
    crit = c.get("crit_words")
    if not crit:
        if abstract and budget > 80:
            pa = _trim_passage(abstract, min(budget, 1700))
            if pa:
                parts.append(pa)
                budget -= len(pa) + 5
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

    weak = c.get("weak_words") or set()
    abs_norm = re.sub(r"\s+", " ", abstract).strip()
    if abs_norm and budget > 80:
        pa = _trim_passage(abs_norm, min(budget, ABSTRACT_EVIDENCE_CHARS))
        if pa:
            parts.append(pa)
            budget -= len(pa) + 5
    # rank body passages by how much criterion text they carry (weak criteria count double)
    scored = []
    for sn in c.get("snippets") or []:
        sn_norm = re.sub(r"\s+", " ", sn or "").strip()
        if len(sn_norm) < 40:
            continue
        if abs_norm and (sn_norm[:160] in abs_norm or abs_norm[:160] in sn_norm):
            continue  # the abstract chunk again
        scored.append((_passage_score(sn_norm, weak, crit), sn_norm))
    scored.sort(key=lambda x: -x[0])
    n_pass = 0
    for sc, sn_norm in scored:
        if budget < 150 or n_pass >= 3 or len(parts) >= 7:
            break
        if sc <= 0 and n_pass >= 1:
            break
        ps = _focus_window(sn_norm, weak or crit, min(budget, PASSAGE_EVIDENCE_CHARS))
        if ps and ps not in parts:
            parts.append(ps)
            budget -= len(ps) + 5
            n_pass += 1
    if tl and budget > 80:
        pt = _trim_passage(tl, budget)
        if pt and pt not in parts:
            parts.append(pt)
            budget -= len(pt) + 5
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
        if snippet and snippet not in c["snippets"] and len(c["snippets"]) < 6:
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
- "keyword_queries": a list of 12 short keyword search queries (2-7 words each, no punctuation, no question words). Make them DIVERSE: cover the core topic phrased different ways, standard terminology and synonyms, well-known method / model / dataset names associated with the topic, adjacent sub-topics, and the phrasing an author of such a paper would use in a title. Each must be a plain noun phrase a keyword search engine understands.
- "snippet_queries": a list of 5 natural-language declarative sentences for a passage-retrieval engine. The first describes what a relevant paper would claim, in the words its abstract would use. Each of the others pairs the core topic with ONE specific requirement of the request (the evaluation protocol, metric, data regime, property, population, time period or constraint it asks for), phrased the way the methods or experiments section of such a paper would state it (e.g. "We report the micro-F1 averaged over all test episodes for few-shot slot tagging.").
- "criteria": a list of 2-4 short statements of what a paper MUST satisfy to be a perfect match for the request. Decompose the request into its distinct required aspects (e.g. the topic/task, the specific method or property asked for, any stated constraint such as a time period, population, or exclusion like "not a survey"); when the request combines two aspects, add a criterion that the paper connects them. Mirror the request's wording.

Return only the JSON object."""

GRADE_PROMPT = """You are a strict relevance judge for a literature search.

User request: {query}

A paper is relevant only if it satisfies ALL of these criteria:
{criteria}

Grade each candidate below using ONLY its title/abstract text. Judge each criterion separately (does the text state or clearly imply it?), then combine:
3 = every criterion is clearly satisfied by the text (perfect match for the request)
2 = every criterion is at least partially satisfied and most are clearly satisfied
1 = related topic but a key criterion is missing or only weakly implied
0 = not relevant

Candidates:
{cands}

Return ONLY a JSON object mapping candidate index (as a string) to grade (integer 0-3), e.g. {{"0": 3, "1": 0}}. Include every index."""

CUTOFF_DATE = "2025-06-01"
GRADE_INITIAL = 520             # fused candidates graded in stage 1
GRADE_EXPANSION = 160           # citation-expansion candidates graded in stage 2
EXPAND_SEEDS = 10               # top graded papers whose citers are pulled
EVIDENCE_HEAD = 100             # papers that get criterion-targeted body snippets
EVIDENCE_GROUP = 8              # papers per scoped snippet_search call
EVIDENCE_GROUP_LIMIT = 60       # passages per group call
EVIDENCE_RETRY_HEAD = 40        # head papers retried one-by-one if the group pass left them empty
GRADE_ROUND1 = 360              # fused candidates graded before the first yield check
GRADE_STEP = 160
GRADE_MAX = 680
YIELD_WINDOW = 120
YIELD_MIN = 5
HEAD_RERANK = 36                # papers re-judged per-criterion by GPT-5.4
SNIPPET_EVIDENCE_TIMEOUT = 140  # a cold scoped snippet_search takes ~120 s
EVIDENCE_STAGE_DEADLINE = 1250  # skip the evidence stage if the query is already this old (s)
QUERY_SOFT_LIMIT = 1560         # evidence stage must finish by this (s); agent timeout is 29 min
SNIPPET_LIMIT = 100


def _default_keywords(query: str) -> list[str]:
    q = re.sub(r"[^\w\s-]", " ", query)
    q = re.sub(r"\b(could|would|can|you|please|suggest|recommend|provide|some|any|papers?|research|study|studies|work|references?|me|i|am|looking|for|are|there|what|which|how|is|that|the|a|an|of|on|in|to|about|has|been|done|key|using|used|with)\b", " ", q, flags=re.I)
    words = q.split()
    return [" ".join(words[:8])] if words else [query[:60]]


def _before_cutoff(p: dict) -> bool:
    """True if the paper is safely at/before the snapshot cutoff (used only for
    get_citations output, which the task-side date filter does not cover)."""
    pd = p.get("publicationDate")
    if isinstance(pd, str) and len(pd) >= 10:
        return pd[:10] <= CUTOFF_DATE
    y = p.get("year")
    return y is not None and y <= 2024


def _judge_grade(ratings: list, weights: list | None = None) -> int:
    """Replica of the benchmark judge's combination rule: per-criterion ratings in {0,1,3},
    weighted = min(1, sum(w*r/3)); grade 0 if <=0.25, 1 if <=0.67, 2 if <=0.99 else 3."""
    if not ratings:
        return 0
    n = len(ratings)
    if not weights or len(weights) != n:
        weights = [1.0 / n] * n
    w = min(1.0, sum(wi * (ri / 3.0) for wi, ri in zip(weights, ratings)))
    if w <= 0.25:
        return 0
    if w <= 0.67:
        return 1
    if w <= 0.99:
        return 2
    return 3


def _head_key(c, idx_of):
    """Order the reranked head: predicted judge grade, number of Perfect criteria, mean rating, prior position."""
    vals = c.get("head_ratings") or []
    g = _judge_grade(vals) if vals else -1
    n3 = sum(1 for v in vals if v >= 3)
    return (-g, -n3, -c.get("head_score", -1.0), idx_of[id(c)])


def _raw_keyword_queries(query: str) -> list[str]:
    """Deterministic keyword queries from the raw request (planner-independent)."""
    q = re.sub(r"[^\w\s-]", " ", query)
    q = re.sub(r"\b(could|would|can|you|please|suggest|recommend|provide|find|some|any|papers?|paper|research|study|studies|work|works|references?|literature|me|i|am|looking|for|are|there|what|which|how|is|that|the|a|an|of|on|in|to|about|has|have|been|done|key|using|used|with|and|or|into|its|it|this|these|those|do|does|explore|explores|exploring|investigate|investigates|investigating|propose|proposes|proposing|discuss|discusses|focus|focusing|focused|address|addresses|addressing)\b", " ", q, flags=re.I)
    words = [w for w in q.split() if len(w) > 1]
    out = []
    if len(words) >= 2:
        out.append(" ".join(words[:7]))
        if len(words) > 7:
            out.append(" ".join(words[-7:]))
        elif len(words) >= 4:
            out.append(" ".join(words[-4:]))
    return list(dict.fromkeys(out))


def _rel_to_grade(rel):
    """Map the 0-10 relevance score onto the judge's 0-3 scale (used for seeds,
    evidence-head selection and diagnostics)."""
    if rel is None:
        return None
    if rel >= 8:
        return 3
    if rel >= 5:
        return 2
    if rel >= 2:
        return 1
    return 0


async def _grade_candidates(query: str, criteria: list[str], to_grade: list[dict]) -> int:
    """Batched LLM grading; sets c['grade'] in place. Returns #graded."""
    crit_txt = "\n".join(f"- {c}" for c in criteria)

    async def grade_batch(batch):
        lines = []
        for i, c in enumerate(batch):
            body = c["abstract"] or c["tldr_text"] or (c["snippets"][0] if c["snippets"] else "")
            body = _trim_passage(body, GRADE_ABSTRACT_CHARS)
            lines.append(f"[{i}] Title: {c['title'][:160]}\n    Abstract: {body}")
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
            rel = grades.get(i)
            c["rel"] = rel
            c["grade"] = rel
        return len(grades)

    batches = [to_grade[i:i + GRADE_BATCH] for i in range(0, len(to_grade), GRADE_BATCH)]
    counts = await asyncio.gather(*[grade_batch(b) for b in batches], return_exceptions=True)
    return sum(x for x in counts if isinstance(x, int))


HEAD_RERANK_PROMPT = """You are the relevance judge for a scientific literature search. Judge each candidate paper against each criterion using ONLY the candidate's text.

User request: {query}

Criteria (judge each one separately):
{criteria}

For each candidate, rate every criterion: 3 = the text directly and clearly satisfies the criterion, 1 = the text only partially or implicitly satisfies it, 0 = the text does not satisfy it.

Candidates:
{cands}

Return ONLY a JSON object mapping candidate index (as a string) to a list with one rating per criterion, in the order the criteria are listed, e.g. {{"0": [3, 3, 1], "1": [0, 3, 3]}}. Include every index."""


async def _rerank_head(query: str, criteria: list[str], head: list[dict]) -> int:
    """Per-criterion judgement of the head by GPT-5.4; sets c['head_score'] in [0,1]."""
    lines = []
    for i, c in enumerate(head):
        body = c["abstract"] or c["tldr_text"] or (c["snippets"][0] if c["snippets"] else "")
        body = _trim_passage(body, 620)
        extra = ""
        if c["abstract"] and c.get("tldr_text"):
            extra = f"\n    TLDR: {_trim_passage(c['tldr_text'], 160)}"
        lines.append(f"[{i}] Title: {c['title'][:160]}\n    Abstract: {body}{extra}")
    crit_txt = "\n".join(f"{j + 1}. {c}" for j, c in enumerate(criteria))
    prompt = HEAD_RERANK_PROMPT.format(query=query, criteria=crit_txt, cands="\n".join(lines))
    out = await _llm_json(GPT_5_4, prompt, retries=1)
    n = 0
    if isinstance(out, dict):
        for k, v in out.items():
            try:
                idx = int(str(k).strip().strip("[]"))
            except (ValueError, TypeError):
                continue
            if not (0 <= idx < len(head)):
                continue
            if isinstance(v, list) and v:
                vals = []
                for x in v[:len(criteria)]:
                    try:
                        vals.append(max(0, min(3, int(x))))
                    except (ValueError, TypeError):
                        vals.append(0)
                head[idx]["head_score"] = sum(vals) / (3.0 * max(1, len(criteria)))
                head[idx]["head_ratings"] = vals
                n += 1
            elif isinstance(v, (int, float)):
                head[idx]["head_score"] = max(0, min(3, int(v))) / 3.0
                n += 1
    return n


def _sem_sort_key(c):
    r = c.get("rel")
    if r is None:
        r = 0.5  # ungraded (beyond the grading cap / LLM failure): between 0 and 1
    return (-r, -c["score"], c["best_rank"])


async def solve_semantic(state: TaskState, tools: dict, query: str) -> list[dict]:
    t0 = time.time()
    plan = await _llm_json(GPT_5_4_MINI, SEMANTIC_PLAN_PROMPT.format(query=query))
    kw_queries, sn_queries, criteria = [], [], []
    if isinstance(plan, dict):
        kw_queries = [str(x) for x in (plan.get("keyword_queries") or []) if str(x).strip()][:13]
        sn_queries = [str(x) for x in (plan.get("snippet_queries") or []) if str(x).strip()][:6]
        criteria = [str(x) for x in (plan.get("criteria") or []) if str(x).strip()][:5]
    if not kw_queries:
        kw_queries = _default_keywords(query)
    if not sn_queries:
        sn_queries = [query]
    if not criteria:
        criteria = [f"The paper matches the request: {query}"]
    for rq in _raw_keyword_queries(query):
        if rq.lower() not in {k.lower() for k in kw_queries}:
            kw_queries.append(rq)
    print(f"  plan: kw={kw_queries} sn={sn_queries} criteria={criteria}")
    # the evidence-stage query text; searched unscoped now (free candidates + passages) so that
    # the scoped group calls later reuse a warm query embedding (cold: ~120 s, warm: ~4 s)
    ev_query_all = _trim_passage(" ".join(criteria) + " " + kw_queries[0], 480)
    # the non-topic criteria alone (the planner lists the topic/task criterion first); used when
    # the window's papers already establish the topic, so passages target what they lack
    ev_query_rest = _trim_passage(" ".join(criteria[1:]) + " " + kw_queries[0], 480) if len(criteria) > 1 else ev_query_all
    for evq in (ev_query_all, ev_query_rest):
        if evq not in sn_queries:
            sn_queries.append(evq)

    cands = Cands()

    async def kw_search(q, weight):
        items = await _call(tools, "search_papers_by_relevance", keyword=q, fields=PAPER_FIELDS, limit=100)
        return ("kw", q, weight, items)

    async def sn_search(q):
        items = await _call(tools, "snippet_search", timeout=SNIPPET_TIMEOUT, query=q, limit=SNIPPET_LIMIT)
        return ("sn", q, 1.0, items)

    jobs = [kw_search(q, 1.3 if i == 0 else 1.0) for i, q in enumerate(kw_queries)]
    jobs += [sn_search(q) for q in sn_queries]
    # the raw request itself also goes to the passage engine (it tolerates sentences)
    if query not in sn_queries:
        jobs.append(sn_search(query))
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

    async def fill_meta(cids):
        need = [cid for cid in cids if cid in cands.d and not cands.d[cid]["abstract"]]
        if not need:
            return
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
                    if not c["title"] and p.get("title"):
                        c["title"] = p["title"]

    await fill_meta(list(need_meta))

    allc = cands.values()
    allc.sort(key=lambda c: (-c["score"], c["best_rank"]))
    print(f"  {len(allc)} unique candidates after {time.time()-t0:.0f}s")
    if not allc:
        return []

    # ---- stage 1: yield-adaptive LLM grading of the fused candidates --------
    for i, c in enumerate(allc):
        c["stage1_pos"] = i
    to_grade = allc[:GRADE_ROUND1]
    n_graded = await _grade_candidates(query, criteria, to_grade)
    graded_upto = len(to_grade)
    rounds = 1
    while graded_upto < min(GRADE_MAX, len(allc)) and time.time() - t0 < 900:
        window = allc[max(0, graded_upto - YIELD_WINDOW):graded_upto]
        y = sum(1 for c in window if (c.get("grade") or 0) >= 2)
        if y < YIELD_MIN:
            print(f"  stage1 yield check: {y} grade>=2 in fused positions {graded_upto - len(window)}-{graded_upto} -> stop")
            break
        nxt = allc[graded_upto:graded_upto + GRADE_STEP]
        n_graded += await _grade_candidates(query, criteria, nxt)
        graded_upto += len(nxt)
        rounds += 1
        print(f"  stage1 yield check: {y} grade>=2 in the last {len(window)} -> graded {len(nxt)} more (now {graded_upto})")
    print(f"  stage1 graded {n_graded}/{graded_upto} in {rounds} round(s) ({time.time()-t0:.0f}s)")

    # ---- stage 2: citation expansion from the best graded papers -----------
    ordered = sorted(allc, key=_sem_sort_key)
    seeds = [c for c in ordered if (c.get("grade") or 0) >= 3][:EXPAND_SEEDS]
    if len(seeds) < 3:
        seeds += [c for c in ordered if c.get("grade") == 2 and c not in seeds][:3 - len(seeds)]
    if seeds:
        known = set(cands.d.keys())
        cit_lists = await asyncio.gather(*[
            _call(tools, "get_citations", timeout=90, paper_id=f"CorpusId:{s['cid']}",
                  fields="title,abstract,corpusId,year,publicationDate,venue,citationCount", limit=150)
            for s in seeds], return_exceptions=True)
        new_c = {}
        for s, cl in zip(seeds, cit_lists):
            if isinstance(cl, Exception):
                continue
            for rank, it in enumerate(cl):
                p = it.get("citingPaper") if isinstance(it, dict) and "citingPaper" in it else it
                if not isinstance(p, dict) or not _before_cutoff(p):
                    continue
                cid = _cid(p)
                if not cid or cid in known:
                    continue
                c = cands.add(p, rank, 0.6)
                if c is not None:
                    new_c[cid] = c
        newl = list(new_c.values())
        # papers cited by several seeds first
        newl.sort(key=lambda c: (-c["hits"], -c["score"]))
        newl = newl[:GRADE_EXPANSION]
        await fill_meta([c["cid"] for c in newl])
        n2 = await _grade_candidates(query, criteria, newl) if newl else 0
        print(f"  stage2: {len(seeds)} seeds -> {len(new_c)} new citers, graded {n2}/{len(newl)} ({time.time()-t0:.0f}s)")
        # expansion papers that were not graded (beyond cap) stay ungraded (0.5)
        allc = cands.values()

    ordered = sorted(allc, key=_sem_sort_key)
    dist = {}
    for c in ordered:
        dist[c.get("grade")] = dist.get(c.get("grade"), 0) + 1
    print(f"  grade distribution: {dist}")
    tail_hi = sum(1 for c in ordered[:250] if c.get("stage1_pos", 0) >= 360 and (c.get("grade") or 0) >= 2)
    print(f"  stage1-tail (fused pos>=360) papers graded>=2 inside top250: {tail_hi}")

    # ---- head rerank: per-criterion judgement of the top of the list ----------
    head_r = ordered[:HEAD_RERANK]
    if head_r:
        try:
            n_r = await _rerank_head(query, criteria, head_r)
        except Exception as e:  # noqa: BLE001
            n_r = 0
            print(f"  !! head rerank failed: {_err_text(e)}")
        if n_r >= len(head_r) // 2:
            idx_of = {id(c): i for i, c in enumerate(head_r)}
            head_sorted = sorted(head_r, key=lambda c: _head_key(c, idx_of))
            moved = sum(1 for i, c in enumerate(head_sorted) if idx_of[id(c)] != i)
            perfect = sum(1 for c in head_sorted if c.get("head_ratings") and _judge_grade(c["head_ratings"]) >= 3)
            pg = {}
            for c in head_sorted:
                g = _judge_grade(c["head_ratings"]) if c.get("head_ratings") else None
                pg[g] = pg.get(g, 0) + 1
            print(f"  head predicted-grade distribution: {pg}")
            ordered = head_sorted + ordered[len(head_r):]
            print(f"  head rerank: {n_r}/{len(head_r)} rated, {perfect} all-criteria-perfect, {moved} moved ({time.time()-t0:.0f}s)")
            print("  head: " + " ".join(f"{c['cid']}:{c.get('head_ratings')}" for c in head_sorted[:12]))
        else:
            print(f"  head rerank skipped ({n_r}/{len(head_r)} rated)")

    # ---- criterion-targeted body passages for the judged window --------------
    crit_sets = [_content_words(cr) for cr in criteria]
    crit_all = set().union(*crit_sets) if crit_sets else set()
    topic_phrase = kw_queries[0]

    def weak_idx(c):
        """Indices of criteria the paper's own text probably does not establish."""
        vals = c.get("head_ratings") or []
        if vals and len(vals) == len(criteria):
            return [j for j, v in enumerate(vals) if v < 3]
        own = _content_words((c.get("title") or "") + " " + (c.get("abstract") or ""))
        out = []
        for j, cw in enumerate(crit_sets):
            if not cw:
                continue
            if len(cw & own) / len(cw) < 0.5:
                out.append(j)
        return out

    for c in ordered[:MAX_SUBMIT]:
        c["crit_words"] = crit_all
        wj = weak_idx(c)
        c["weak_idx"] = wj
        c["weak_words"] = set().union(*[crit_sets[j] for j in wj]) if wj else set()

    ev_pool = [c for c in ordered[:EVIDENCE_HEAD] if c.get("grade") is None or (c.get("grade") or 0) >= 1]
    elapsed = time.time() - t0
    if elapsed > EVIDENCE_STAGE_DEADLINE:
        print(f"  evidence stage skipped (elapsed {elapsed:.0f}s)")
        ev_pool = []
    if ev_pool:
        sem = asyncio.Semaphore(6)
        added = {}

        n_pool = max(1, len(ev_pool))
        topic_weak_share = sum(1 for c in ev_pool if 0 in (c.get("weak_idx") or [])) / n_pool
        ev_query = ev_query_rest if (len(criteria) > 1 and topic_weak_share < 0.3) else ev_query_all
        print(f"  evidence query: {'non-topic criteria' if ev_query is ev_query_rest else 'all criteria'} (topic weak for {topic_weak_share:.0%} of window)")

        def group_query(group):
            # one fixed, pre-warmed text for every group call: the passage engine caches the
            # query embedding, so each scoped call returns in seconds instead of ~120 s
            return ev_query

        def absorb(c, items):
            n = 0
            for it in items:
                sn = (it.get("snippet") or {}).get("text") or ""
                if sn and sn not in c["snippets"] and len(c["snippets"]) < 10:
                    c["snippets"].append(sn)
                    n += 1
            added[c["cid"]] = added.get(c["cid"], 0) + n
            return n

        async def one_group(group):
            by_cid = {c["cid"]: c for c in group}
            async with sem:
                items = await _call(tools, "snippet_search", timeout=SNIPPET_TIMEOUT,
                                    query=group_query(group), limit=EVIDENCE_GROUP_LIMIT,
                                    paper_ids=",".join(f"CorpusId:{c['cid']}" for c in group))
            buckets = {}
            for it in items:
                pid = str(((it.get("paper") or {}).get("corpusId")) or "")
                if pid in by_cid:
                    buckets.setdefault(pid, []).append(it)
            return sum(absorb(by_cid[pid], its[:6]) for pid, its in buckets.items())

        async def one_paper(c):
            async with sem:
                items = await _call(tools, "snippet_search", timeout=SNIPPET_TIMEOUT,
                                    query=ev_query, limit=5, paper_ids=f"CorpusId:{c['cid']}")
            return absorb(c, items)

        async def run_stage():
            groups = [ev_pool[i:i + EVIDENCE_GROUP] for i in range(0, len(ev_pool), EVIDENCE_GROUP)]
            r1 = await asyncio.gather(*[one_group(g) for g in groups], return_exceptions=True)
            n1 = sum(x for x in r1 if isinstance(x, int))
            print(f"  evidence group pass: {len(groups)} calls, {n1} passages, "
                  f"{sum(1 for c in ev_pool if added.get(c['cid']))}/{len(ev_pool)} papers covered ({time.time()-t0:.0f}s)")
            retry = [c for c in ev_pool[:EVIDENCE_RETRY_HEAD] if not added.get(c["cid"])]
            if retry and time.time() - t0 < QUERY_SOFT_LIMIT - 200:
                r2 = await asyncio.gather(*[one_paper(c) for c in retry], return_exceptions=True)
                n2 = sum(x for x in r2 if isinstance(x, int))
                print(f"  evidence per-paper retry: {len(retry)} calls, {n2} passages ({time.time()-t0:.0f}s)")

        try:
            await asyncio.wait_for(run_stage(), timeout=max(60, min(540, QUERY_SOFT_LIMIT - (time.time() - t0))))
        except asyncio.TimeoutError:
            print(f"  !! evidence stage hit its deadline ({time.time()-t0:.0f}s)")
        covered = sum(1 for c in ev_pool if c["snippets"])
        print(f"  evidence: {covered}/{len(ev_pool)} window papers carry body passages; "
              f"weak-criteria histogram: {dict(sorted(((j, sum(1 for c in ev_pool if j in (c.get('weak_idx') or []))) for j in range(len(criteria)))))}")

    results = []
    for c in ordered[:MAX_SUBMIT]:
        results.append({"paper_id": c["cid"], "markdown_evidence": _build_evidence(c)})
    return results


# ----------------------------------------------------------------------------
# SPECIFIC
# ----------------------------------------------------------------------------
SPECIFIC_PLAN_PROMPT = """A user is asking for a specific, known paper (or a small known set of papers).

Request: {query}

Hints: requests often use a shorthand nickname ("the BART paper"), a BibTeX-style citation key such as
"fabri2019multinews" or "DeYong2021" (first author's surname + year + a title word, possibly misspelled),
a dataset/model/benchmark name, or a vague description ("the cnn paper" = the most famous paper(s) on that topic).
Use your knowledge of the literature to identify the real paper(s).

Return a JSON object:
- "titles": list of up to 4 guesses of the EXACT title of the paper(s) requested (best guess first). If the
  request is ambiguous (e.g. "the cnn paper"), list up to 6 of the most canonical/famous candidate papers
  (the landmark papers everyone cites for that topic), most famous first.
- "keywords": list of 3-4 short keyword queries (2-8 words) that would retrieve it in a keyword search
  (model/method/dataset name plus distinctive words; for a citation key, include the surname and the topic words).
- "expected_count": integer, how many distinct papers the user is asking for (usually 1).
- "ambiguous": true if several different well-known papers could reasonably be meant, else false.
Return only JSON."""

SPECIFIC_PICK_PROMPT = """A user asked for a specific paper.

Request: {query}

Candidates:
{cands}

Which candidate(s) are the paper(s) the user is asking for? Prefer the original paper over later versions, surveys,
or follow-ups. If the request is a nickname or citation key (surname+year), match on first author surname (allow
misspellings), year, and title/topic words. If the request is vague and several famous papers could be meant,
list the plausible ones in order of likelihood. Return ONLY a JSON object:
{{"picks": [indices in order of confidence], "confidence": "high"|"medium"|"low"}}.
"high" means you are sure the first pick is exactly the requested paper. If none match, return {{"picks": [], "confidence": "low"}}."""


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


def _specific_raw_keywords(query: str) -> list[str]:
    """Keyword queries derived from the raw request: strip request words, split citation keys."""
    q = query.strip()
    q = re.sub(r"(?i)^(please\s+)?(find|get|retrieve|show|give)\s+(me\s+)?", "", q)
    q = re.sub(r"(?i)\b(the|a|an|paper|papers|article|publication|by|et al\.?|original)\b", " ", q)
    q = q.replace("^", "").replace("$", "")
    # split "fabri2019multinews" / "DeYong2021" into "fabri 2019 multinews"
    q = re.sub(r"([A-Za-z])(\d{4})", r"\1 \2", q)
    q = re.sub(r"(\d{4})([A-Za-z])", r"\1 \2", q)
    q = re.sub(r"\s+", " ", q).strip(" ,.:;")
    out = []
    if q:
        out.append(q)
        no_year = re.sub(r"\b(19|20)\d{2}\b", " ", q)
        no_year = re.sub(r"\s+", " ", no_year).strip()
        if no_year and no_year != q:
            out.append(no_year)
    return out


async def _specific_plan(model, query: str):
    plan = await _llm_json(model, SPECIFIC_PLAN_PROMPT.format(query=query))
    titles, keywords, expected, ambiguous = [], [], 1, False
    if isinstance(plan, dict):
        titles = [str(x) for x in (plan.get("titles") or []) if str(x).strip()][:6]
        keywords = [str(x) for x in (plan.get("keywords") or []) if str(x).strip()][:4]
        try:
            expected = max(1, int(plan.get("expected_count") or 1))
        except (ValueError, TypeError):
            expected = 1
        ambiguous = bool(plan.get("ambiguous"))
    return titles, keywords, expected, ambiguous


async def solve_specific(state: TaskState, tools: dict, query: str) -> list[dict]:
    plans = await asyncio.gather(_specific_plan(GPT_5_4, query), _specific_plan(CLAUDE_SONNET_4_6, query),
                                 return_exceptions=True)
    titles, keywords, expected, ambiguous = [], [], 1, False
    for pl in plans:
        if isinstance(pl, Exception):
            print(f"  !! plan failed: {str(pl)[:120]}")
            continue
        t, k, e, a = pl
        for x in t:
            if x.lower() not in {y.lower() for y in titles}:
                titles.append(x)
        for x in k:
            if x.lower() not in {y.lower() for y in keywords}:
                keywords.append(x)
        expected = max(expected, e)
        ambiguous = ambiguous or a
    titles = titles[:6]
    raw_kws = _specific_raw_keywords(query)
    for x in raw_kws:
        if x.lower() not in {y.lower() for y in keywords}:
            keywords.append(x)
    keywords = keywords[:7]
    if not keywords:
        keywords = _default_keywords(query)
    print(f"  plan: titles={titles} keywords={keywords} expected={expected} ambiguous={ambiguous}")

    cands = Cands()

    async def title_job(t):
        for to in (130, 170):
            try:
                return await _call_raw(tools, "search_paper_by_title", timeout=to, title=t, fields=PAPER_FIELDS)
            except Exception as e:  # noqa: BLE001
                print(f"  !! title search failed ({to}s): {str(e)[:100]}")
        return []

    async def kw_job(k, limit=20):
        for to in (150, 180):
            try:
                return await _call_raw(tools, "search_papers_by_relevance", timeout=to, keyword=k, fields=PAPER_FIELDS, limit=limit)
            except Exception as e:  # noqa: BLE001
                print(f"  !! keyword search failed ({to}s): {str(e)[:100]}")
        return []

    async def snip_job(q):
        try:
            raw = await _call_raw(tools, "snippet_search", timeout=160, query=q, limit=25)
        except Exception as e:  # noqa: BLE001
            print(f"  !! snippet search failed: {str(e)[:100]}")
            return []
        out, seen = [], set()
        for d in raw:
            p = (d.get("paper") or {})
            cid = p.get("corpusId")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append({"corpusId": cid, "paperId": p.get("paperId"), "title": p.get("title") or "",
                        "authors": p.get("authors") or [], "abstract": "", "snippet": (d.get("snippet") or {}).get("text") or ""})
        return out

    jobs = [title_job(t) for t in titles] + [kw_job(k) for k in keywords] + [snip_job(query), snip_job(" ".join(raw_kws[:1]) or query)]
    res = await asyncio.gather(*jobs, return_exceptions=True)
    n_titles = len(titles)
    snippet_ids = set()
    for j, r in enumerate(res):
        if isinstance(r, Exception) or not r:
            continue
        for rank, p in enumerate(r):
            if not p.get("paperId") and not p.get("corpusId"):
                continue
            if j >= n_titles + len(keywords):
                snippet_ids.add(str(p.get("corpusId")))
            cands.add(p, rank, 2.0 if j < n_titles else 1.0, snippet=p.get("snippet") or None)
    # snippet hits have no abstract: fetch metadata for them so the picker can read them
    need = [c for c in cands.values() if not c.get("abstract") and c["cid"] in snippet_ids][:40]
    if need:
        try:
            metas = await _batch_meta(tools, [c["cid"] for c in need], PAPER_FIELDS)
            by = {str(_cid(m)): m for m in metas if _cid(m) is not None}
            for c in need:
                m = by.get(c["cid"])
                if m:
                    c["abstract"] = m.get("abstract") or ""
                    c["year"] = m.get("year") or c.get("year")
                    c["venue"] = m.get("venue") or c.get("venue") or ""
                    c["citationCount"] = m.get("citationCount") or c.get("citationCount") or 0
                    c["authors"] = m.get("authors") or c.get("authors") or []
                    c["tldr_text"] = _tldr_text(m) or c.get("tldr_text", "")
        except Exception as e:  # noqa: BLE001
            print(f"  !! snippet meta failed: {str(e)[:100]}")
    allc = cands.values()
    allc.sort(key=lambda c: (-c["score"], c["best_rank"]))
    allc = allc[:40]
    print(f"  {len(allc)} candidates")
    if not allc:
        return []

    lines = []
    for i, c in enumerate(allc):
        lines.append(f"[{i}] {c['title'][:160]} ({c['year']}; {str(c.get('venue') or '')[:40]}; cites={c.get('citationCount')}; {_fmt_authors(c)})\n"
                     f"    {_trim_passage(c['abstract'] or c.get('tldr_text') or (c['snippets'][0] if c.get('snippets') else ''), 250)}")
    pick = await _llm_json(GPT_5_4, SPECIFIC_PICK_PROMPT.format(query=query, cands="\n".join(lines)))
    picks, conf = [], "low"
    if isinstance(pick, dict):
        for x in pick.get("picks") or []:
            try:
                i = int(x)
            except (ValueError, TypeError):
                continue
            if 0 <= i < len(allc) and i not in picks:
                picks.append(i)
        conf = str(pick.get("confidence") or "low").lower()
        print(f"  picks={picks} confidence={conf}")
    if not picks:
        # nothing recognised: gamble on the top-3 fused candidates (F1 0.5 if one hits, 0 otherwise)
        picks = list(range(min(3, len(allc))))
        conf = "low"
    if conf == "high" and not ambiguous:
        n = max(expected, 1)
    elif conf == "high":
        n = max(expected, 3)
    elif conf == "medium":
        n = max(expected, 4 if ambiguous else 2)
    else:
        n = max(expected, 4 if ambiguous else 3)
    picks = picks[:n]
    print(f"  submitting {len(picks)} (conf={conf}, expected={expected}, ambiguous={ambiguous})")
    return [{"paper_id": allc[i]["cid"], "markdown_evidence": _build_evidence(allc[i])} for i in picks]


# ----------------------------------------------------------------------------
# METADATA
# ----------------------------------------------------------------------------
METADATA_PLAN_PROMPT = """Parse this literature-search request into structured filters.

Request: {query}

Return a JSON object with exactly these keys (use null / [] when not specified):
- "authors": list of author full names the papers must be written by (people, not organisations).
- "venue_patterns": list of lowercase substrings that identify the required publication venue(s). Expand families: e.g. "Nature portfolio" -> ["nature", "npj"]; "NeurIPS" -> ["neurips", "neural information processing systems"]; "ACL" (the main conference only) -> ["annual meeting of the association for computational linguistics", "acl"]; "CVPR" -> ["cvpr", "computer vision and pattern recognition"]; "ICML" -> ["icml", "international conference on machine learning"]; "ICLR" -> ["iclr", "international conference on learning representations"]; "EMNLP" -> ["emnlp", "empirical methods in natural language processing"]; "arXiv" -> ["arxiv"].
- "year_min": integer or null (inclusive). "after 2020" means year_min 2021? NO — interpret loosely as inclusive: "after 2020" -> 2020. "since 2019" -> 2019. "in 2021" -> 2021 with year_max 2021.
- "year_max": integer or null (inclusive).
- "min_citations": integer or null (papers must have at least this many citations; "more than 50 citations" -> 51).
- "min_authors": integer or null ("more than 3 authors" -> 4). "max_authors": integer or null.
- "cites_paper": title (or well-known name) of a paper that results must CITE, else null. E.g. "papers citing the RoBERTa paper" -> "RoBERTa: A Robustly Optimized BERT Pretraining Approach".
- "cites_paper_keyword": a short distinctive keyword (1-3 words, e.g. "RoBERTa", "DistilBERT") for that cited paper, else null.
- "cites_author": full name of a person whose papers the results must CITE (e.g. "papers citing work by Gera Weiss" -> "Gera Weiss"), else null. Do not confuse with "authors" (who wrote the results).
- "exclude_self_citations": true if the request excludes papers co-authored by that cited person (e.g. "not self-citations"), else false.
- "cites_venue_patterns": if the results must CITE a paper published at some venue (e.g. "papers that cite any NeurIPS paper", "citing an ICML paper"), lowercase substrings identifying THAT cited venue (same expansion rules as venue_patterns, e.g. ["neurips", "neural information processing systems"]); else []. Do not confuse with venue_patterns (where the results themselves are published): in "A SPLASH 2019 and beyond paper that cites any NeurIPS", venue_patterns is ["splash"] and cites_venue_patterns is ["neurips", "neural information processing systems"]. Leave cites_paper null in that case.
- Do NOT put generic words like "journal", "conference", "article", "proceedings" into venue_patterns. venue_patterns must identify ONLY the venue(s) the request names: never widen a specific venue to its parent organisation or family (e.g. "NAACL" -> ["naacl", "north american chapter of the association for computational linguistics"], NOT "acl"; "EMNLP" -> ["emnlp", "empirical methods in natural language processing"], NOT "acl"; "Nature Machine Intelligence" -> ["nature machine intelligence"], NOT "nature").
- "topic_keywords": list of 0-4 short keyword queries describing a required TOPIC of the papers (only if the request constrains the subject matter; author/venue/year/citation constraints are NOT topics).
- "topic_description": one sentence describing the topical requirement, or null.
- "venue_search_names": list of exact venue names as a bibliographic database would spell them, usable as a venue filter (include both the acronym and the full name, e.g. ["NeurIPS", "Neural Information Processing Systems"]; ["Nature"] for the journal Nature; [] if no venue constraint).
- "cited_paper_topics": if "cites_paper" is set, a list of 10 short keyword queries (2-5 words) for research topics in which that paper is typically cited or used as a baseline/component (e.g. for RoBERTa: "text classification pretrained language model", "natural language inference", "sentiment analysis transformer", "prompt tuning", "model compression BERT", ...); else [].
Return only JSON."""

TOPIC_FILTER_PROMPT = """Decide which candidate papers are about the required topic.

Required topic: {topic}

Candidates:
{cands}

Return ONLY a JSON object: {{"relevant": [indices of candidates that clearly match the topic]}}."""


VENUE_SWEEP_TERMS = [
    "learning", "neural network", "model", "analysis", "system", "optimization",
    "language", "programming", "evaluation", "framework", "data", "algorithm",
]


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
    if check_venue and not c.get("venue_ok") and not _venue_match(c, f.get("venue_patterns") or []):
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
    _generic = {"journal", "journals", "conference", "conferences", "article", "articles", "proceedings", "workshop", "paper", "papers", "peer-reviewed", "peer reviewed"}
    venue_patterns = [str(v) for v in (f.get("venue_patterns") or []) if str(v).strip() and str(v).strip().lower() not in _generic]
    f["venue_patterns"] = venue_patterns
    cites_author = str(f.get("cites_author") or "").strip() or None
    exclude_self = bool(f.get("exclude_self_citations"))
    cites_venue_patterns = [str(v).strip().lower() for v in (f.get("cites_venue_patterns") or [])
                            if str(v).strip() and str(v).strip().lower() not in _generic]
    cites_paper = f.get("cites_paper")
    cites_kw = f.get("cites_paper_keyword")
    topic_keywords = [str(k) for k in (f.get("topic_keywords") or []) if str(k).strip()][:4]
    topic_desc = f.get("topic_description")
    venue_names = [str(v) for v in (f.get("venue_search_names") or []) if str(v).strip()][:4]
    venues_csv = ",".join(venue_names) if venue_names else None
    cited_topics = [str(k) for k in (f.get("cited_paper_topics") or []) if str(k).strip()][:12]
    print(f"  filters: {json.dumps(f)[:900]}")

    def kw_args(**kw):
        if venues_csv:
            kw["venues"] = venues_csv
        return kw

    pool: dict[str, dict] = {}
    fields = "title,abstract,corpusId,year,venue,citationCount,authors,publicationDate"
    author_fields = fields + ",journal"

    def add_pool(p: dict, venue_ok: bool = False):
        cid = _cid(p)
        if not cid:
            return
        if venue_ok and cid in pool:
            pool[cid]["venue_ok"] = True
        if cid not in pool:
            pool[cid] = {"venue_ok": venue_ok,
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
            async def author_papers(aid):
                for to in (150, 280):
                    try:
                        return await _call_raw(tools, "get_author_papers", timeout=to, author_id=str(aid),
                                               paper_fields=author_fields, limit=1000)
                    except ToolFailure as e:
                        print(f"  !! get_author_papers {aid} failed ({to}s): {e}")
                return []
            res = await asyncio.gather(*[author_papers(aid) for aid, _, _ in ids], return_exceptions=True)
            for r in res:
                if isinstance(r, Exception):
                    continue
                for p in r:
                    add_pool(p)
        print(f"  author pool: {len(pool)}")
        # free augmentation: keyword search on the author's name (venue-filtered when given),
        # kept only when the paper's author list actually contains a compatible name
        jobs = []
        for name in authors:
            jobs.append(_call(tools, "search_papers_by_relevance", timeout=120, **kw_args(keyword=name, fields=fields, limit=100)))
            if venues_csv:
                jobs.append(_call(tools, "search_papers_by_relevance", timeout=120, keyword=name, fields=fields, limit=100))
        res = await asyncio.gather(*jobs, return_exceptions=True)
        n0, n_seen = len(pool), 0
        for r in res:
            if isinstance(r, Exception):
                continue
            for p in r:
                n_seen += 1
                names = [a.get("name") if isinstance(a, dict) else str(a) for a in (p.get("authors") or [])]
                if any(_author_compatible(nm, cn or "") for nm in authors for cn in names):
                    add_pool(p)
        print(f"  author keyword augmentation: {n_seen} hits -> pool +{len(pool)-n0}")

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
            n_post = 0
            for it in cit:
                p = it.get("citingPaper") if isinstance(it, dict) and "citingPaper" in it else it
                if isinstance(p, dict):
                    if not _before_cutoff(p):
                        n_post += 1
                        continue
                    add_pool(p)
            print(f"  get_citations: {len(cit)} ({n_post} post-cutoff dropped) -> pool +{len(pool)-n0}")
            huge = (target.get("citationCount") or 0) > 900 or len(cit) >= 1000

            # augment: keyword / snippet search for the cited paper's name, then verify via references
            if cites_kw:
                sn_kw = dict(timeout=120, limit=100)
                if venues_csv:
                    sn_kw["venues"] = venues_csv
                jobs = [
                    _call(tools, "search_papers_by_relevance", **kw_args(keyword=str(cites_kw), fields=fields, limit=100)),
                    _call(tools, "search_papers_by_relevance", **kw_args(keyword=f"{cites_kw} {str(cites_paper)[:60]}", fields=fields, limit=100)),
                    _call(tools, "snippet_search", query=f"{cites_kw}", **sn_kw),
                    _call(tools, "snippet_search", query=f"we use {cites_kw} as a baseline model", **sn_kw),
                ]
                if topic_keywords:
                    for tk in topic_keywords[:2]:
                        jobs.append(_call(tools, "search_papers_by_relevance", **kw_args(keyword=f"{cites_kw} {tk}", fields=fields, limit=100)))
                if huge:
                    # the 1000-newest citer window is useless for a hugely cited target:
                    # sweep the topics where it is cited (venue-filtered when a venue is given)
                    for tk in cited_topics:
                        jobs.append(_call(tools, "search_papers_by_relevance", **kw_args(keyword=f"{cites_kw} {tk}", fields=fields, limit=100)))
                        if venues_csv:
                            jobs.append(_call(tools, "search_papers_by_relevance", **kw_args(keyword=tk, fields=fields, limit=100)))
                    for sq in (f"{cites_kw} baseline", f"fine-tuned {cites_kw}", f"{cites_kw} and BERT", f"pretrained {cites_kw} model", f"{cites_kw} large"):
                        jobs.append(_call(tools, "snippet_search", query=sq, **sn_kw))
                print(f"  augmentation: {len(jobs)} searches (huge={huge}, venues={venues_csv})")
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
                # snippet hits lack metadata: fetch it so the cheap filters can run first
                no_meta = [cid for cid, p in extra.items() if p.get("year") is None]
                if no_meta:
                    chunks = [no_meta[i:i + 50] for i in range(0, len(no_meta), 50)]
                    metas = await asyncio.gather(*[_batch_meta(tools, ch, fields) for ch in chunks], return_exceptions=True)
                    for m in metas:
                        if isinstance(m, Exception):
                            continue
                        for p in m:
                            cid = _cid(p)
                            if cid in extra:
                                extra[cid] = p
                to_verify = [cid for cid, p in extra.items()
                             if p.get("year") is not None and _passes(p, f, check_venue=False)]
                print(f"  augmentation pool: {len(extra)} unseen papers, {len(no_meta)} needed metadata")
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

    # ---- "cites papers by author Y" ----------------------------------------
    if cites_author and (time.time() - t0) < 900:
        recs = await _call(tools, "search_authors_by_name", name=cites_author, fields="name,paperCount", limit=15)
        y_ids = [(r.get("authorId"), r.get("paperCount") or 0, r.get("name"))
                 for r in recs if r.get("authorId") and _author_compatible(cites_author, r.get("name") or "")]
        y_ids.sort(key=lambda x: -x[1])
        y_ids = y_ids[:6]
        print(f"  cited author {cites_author!r}: ids={y_ids}")
        y_pids, y_cids = set(), set()
        y_papers: dict[str, dict] = {}
        res = await asyncio.gather(*[
            _call(tools, "get_author_papers", author_id=str(aid), paper_fields="title,corpusId,year,citationCount", limit=1000)
            for aid, _, _ in y_ids], return_exceptions=True)
        for r in res:
            if isinstance(r, Exception):
                continue
            for p in r:
                if p.get("paperId"):
                    y_pids.add(p["paperId"])
                if _cid(p):
                    y_cids.add(_cid(p))
                    y_papers[_cid(p)] = p
        print(f"  cited author papers: {len(y_pids)}")
        # inverted direction: who cites Y's papers? (reference lists are unreadable for many papers)
        y_citers: set[str] = set()
        if y_papers and (time.time() - t0) < 900:
            order = sorted(y_papers.values(), key=lambda p: -(p.get("citationCount") or 0))[:220]
            sem = asyncio.Semaphore(8)

            stage_t0 = time.time()

            async def citers_of(p):
                if (p.get("citationCount") or 0) == 0:
                    return []
                async with sem:
                    if time.time() - stage_t0 > 400 or time.time() - t0 > 1000:
                        return []          # stage deadline: protect the 29-minute query budget
                    return await _call(tools, "get_citations", timeout=100, paper_id=f"CorpusId:{_cid(p)}",
                                       fields="title,corpusId", limit=1000)
            res = await asyncio.gather(*[citers_of(p) for p in order], return_exceptions=True)
            n_calls = 0
            for r in res:
                if isinstance(r, Exception):
                    continue
                n_calls += 1
                for it in r:
                    q = it.get("citingPaper") if isinstance(it, dict) and "citingPaper" in it else it
                    if isinstance(q, dict) and _cid(q):
                        y_citers.add(_cid(q))
            print(f"  cites_author: get_citations on {n_calls}/{len(order)} of Y's papers -> {len(y_citers)} distinct citers ({time.time()-t0:.0f}s)")
        if not pool:
            # no author/cited-paper pool: build one from the topic keywords and the cited author's citers
            kws = topic_keywords or _default_keywords(query)
            res = await asyncio.gather(*[
                _call(tools, "search_papers_by_relevance", **kw_args(keyword=k, fields=fields, limit=100)) for k in kws
            ], return_exceptions=True)
            for r in res:
                if isinstance(r, Exception):
                    continue
                for p in r:
                    add_pool(p, venue_ok=bool(venues_csv))
            print(f"  topic pool for cites_author: {len(pool)}")
        if y_pids and pool:
            def _is_self(c):
                for a in (c.get("authors") or []):
                    nm = a.get("name") if isinstance(a, dict) else str(a)
                    if _author_compatible(cites_author, nm or ""):
                        return True
                return False
            cheap = [cid for cid, c in pool.items() if _passes(c, f, check_venue=False)]
            print(f"  cites_author: {len(cheap)} pool papers pass cheap filters; fetching references")
            chunks = [cheap[i:i + 12] for i in range(0, len(cheap), 12)]
            metas = await asyncio.gather(*[_batch_meta(tools, ch, "title,corpusId,references") for ch in chunks],
                                         return_exceptions=True)
            citing, n_self, n_meta = set(), 0, 0
            n_inv = 0
            for cid in cheap:
                if cid in y_citers:
                    if exclude_self and _is_self(pool[cid]):
                        n_self += 1
                        continue
                    citing.add(cid)
                    n_inv += 1
            print(f"  cites_author: {n_inv} pool papers found via Y's citers")
            for m in metas:
                if isinstance(m, Exception):
                    continue
                for p in m:
                    n_meta += 1
                    refs = p.get("references") or []
                    if any(isinstance(r, dict) and (r.get("paperId") in y_pids) for r in refs):
                        cid = _cid(p)
                        if cid in pool:
                            if exclude_self and _is_self(pool[cid]):
                                n_self += 1
                                continue
                            citing.add(cid)
            print(f"  cites_author: {len(citing)} pool papers cite {cites_author} ({n_self} self-citations excluded; {n_meta}/{len(cheap)} reference records read)")
            if n_meta > 0 or y_citers:
                pool = {cid: c for cid, c in pool.items() if cid in citing}
            else:
                print("  !! no reference records returned; keeping the unverified pool")
        elif y_pids:
            print("  cites_author: empty pool, nothing to verify")

    # ---- topic / venue-only retrieval ------------------------------------
    if not authors and not cites_paper and not cites_author:
        kws = topic_keywords or _default_keywords(query)
        res = await asyncio.gather(*[
            _call(tools, "search_papers_by_relevance", **kw_args(keyword=k, fields=fields, limit=100)) for k in kws
        ], return_exceptions=True)
        for r in res:
            if isinstance(r, Exception):
                continue
            for p in r:
                add_pool(p, venue_ok=bool(venues_csv))
        print(f"  topic pool: {len(pool)}")

    # ---- venue-only sweep: a venue filter with no topic/author/cited paper ----
    if venues_csv and len(pool) < 40 and not cites_author and (time.time() - t0) < 700:
        sweep = list(dict.fromkeys(venue_names + topic_keywords + VENUE_SWEEP_TERMS))[:14]
        res = await asyncio.gather(*[
            _call(tools, "search_papers_by_relevance", timeout=90, **kw_args(keyword=k, fields=fields, limit=100))
            for k in sweep
        ], return_exceptions=True)
        n0 = len(pool)
        for r in res:
            if isinstance(r, Exception):
                continue
            for p in r:
                add_pool(p, venue_ok=True)  # server-side venues= filter already applied
        print(f"  venue sweep ({len(sweep)} terms, venues={venues_csv}): pool +{len(pool)-n0}")

    # ---- "cites any paper from venue V": verify through readable reference lists --------
    cites_venue_verified = False
    if cites_venue_patterns and pool and (time.time() - t0) < 1000:
        cands = [cid for cid, c in pool.items() if _passes(c, f)][:300]
        print(f"  cites_venue {cites_venue_patterns}: {len(cands)} candidates; fetching references")
        chunks = [cands[i:i + 12] for i in range(0, len(cands), 12)]
        metas = await asyncio.gather(*[_batch_meta(tools, ch, "title,corpusId,references") for ch in chunks],
                                     return_exceptions=True)
        refs_of: dict[str, list] = {}
        for m in metas:
            if isinstance(m, Exception):
                continue
            for p in m:
                cid = _cid(p)
                if cid and isinstance(p.get("references"), list):
                    refs_of[cid] = [r.get("paperId") for r in p["references"] if isinstance(r, dict) and r.get("paperId")]
        ref_ids = list(dict.fromkeys(pid for lst in refs_of.values() for pid in lst))
        print(f"  cites_venue: {len(refs_of)}/{len(cands)} reference lists readable, {len(ref_ids)} distinct referenced papers")
        ref_venue_ok: set[str] = set()
        n_meta = 0
        if ref_ids:
            async def lookup(ch):
                try:
                    return await _call_raw(tools, "get_paper_batch", timeout=120, ids=ch, fields="title,corpusId,venue,journal,year")
                except ToolFailure as e:
                    print(f"  !! cites_venue ref lookup failed ({len(ch)} ids): {e}")
                    return []
            chunks = [ref_ids[i:i + 100] for i in range(0, len(ref_ids), 100)]
            outs = await asyncio.gather(*[lookup(ch) for ch in chunks], return_exceptions=True)
            for o in outs:
                if isinstance(o, Exception):
                    continue
                for p in o:
                    if not isinstance(p, dict) or not p.get("paperId"):
                        continue
                    n_meta += 1
                    if _venue_match(p, cites_venue_patterns):
                        ref_venue_ok.add(p["paperId"])
        print(f"  cites_venue: {n_meta} referenced papers resolved, {len(ref_venue_ok)} at the cited venue")
        if n_meta > 0:
            keep, drop = set(), 0
            for cid in cands:
                lst = refs_of.get(cid)
                if lst is None or not lst:
                    keep.add(cid)          # unverifiable: keep
                elif any(pid in ref_venue_ok for pid in lst):
                    keep.add(cid)
                else:
                    drop += 1
            print(f"  cites_venue: keep {len(keep)} (drop {drop} verified non-citers)")
            cand_set = set(cands)
            pool = {cid: c for cid, c in pool.items() if cid in keep or cid not in cand_set}
            cites_venue_verified = True

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
    cap = MAX_SUBMIT
    if not authors and not cites_paper and (cites_kw or cites_venue_patterns) and not cites_venue_verified and len(passing) > 60:
        # "cites any <venue> paper" cannot be verified with the tools; a venue-wide
        # dump only dilutes precision, so keep the most-cited slice
        cap = 60
    return [{"paper_id": c["cid"], "markdown_evidence": _build_evidence(c)} for c in passing[:cap]]


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
