"""criterion-evidence-miner — a PaperFindingBench solver.

Iteration-002 (`deep-recall-router`, 47.18) established the two structural facts
this agent builds on, and its `judge_verdicts.json` dumps revealed the failure
mode this agent attacks.

Structural facts, re-verified across all 8 semantic problems of iteration 002:

* Only the first K results are ever judged (`scored_depth_cap == k_estimate`);
  positions K+1..250 come back `beyond_scored_depth` and `grade3_at_full ==
  grade3_in_top_k` every time. Six of eight queries had K <= 46, so the top ~25
  slots carry the score. Submitting deep is free and already banked; the
  remaining lever is *what occupies those slots and what evidence they carry*.
* `recall = grade3_in_top_K / K` counts ONLY grade 3, and grade 3 needs
  `weighted > 0.99`, i.e. essentially every criterion rated Perfectly Relevant.

The failure mode: across iteration 002's batch the judge issued **65
`highly_relevant` (grade 2) verdicts against 62 `perfectly_relevant` (grade 3)**.
Grade 2 earns exactly zero recall. Those are papers we had already retrieved and
already ranked into the scored window; they lost on the last mile.

`semantic_43` is the clean proof. Criteria were (1) few-shot slot tagging w=0.4,
(2) evaluation using micro-F1 w=0.3, (3) averaged across test episodes w=0.3.
Retrieval was essentially perfect — L-TapNet, MCML, the domain-transfer meta-task
paper — and every single one was graded `somewhat_relevant`: 0.4*1 + 0.3*0 +
0.3*0 = 0.4 -> grade 1. The evidence was title+tldr+abstract, and abstracts of
few-shot slot tagging papers never say "micro-F1 averaged over test episodes".
That sentence lives in the experimental-setup section of the *body*. Final score
on that query: 0.000.

So: the bottleneck is not finding papers, it is *proving* them. Three changes:

A. CRITERION-TARGETED EVIDENCE MINING. `snippet_search` scoped by `paper_ids`
   retrieves body text. For the reranked head, fire one scoped call per criterion
   over groups of 3 papers (probed at 3.5 s/call; tool calls are free). Assemble
   evidence greedily for *criterion coverage* rather than length.
   Groups of 3, not 20: `paper_ids` is a scope filter, not a per-paper
   allocation — probed with 8 ids at limit 24, two papers got zero passages.

B. JUDGE-REPLICA RERANKING. Stage 2 asks for per-criterion ratings in {0,1,3}
   and the weighted score is computed in Python with the judge's own arithmetic,
   over *the exact evidence string that will be submitted*. It is a dry run of
   the real judge on the real input, and it demotes topically-perfect papers
   whose evidence leaves a criterion unproven.

C. METADATA REFERENCE-PAPER EXPANSION. `metadata_14` ("NAACL 2010 or 2012 papers
   co-authored by one of the authors of the BERT paper") scored 0.000 — the query
   names no author, so the author tools never fired. The analyzer now emits
   `reference_titles`, which are resolved to papers, then to their authors, then
   to those authors' papers. Year filtering uses an explicit year *set* (2010 or
   2012 is not a range) and venue matching uses analyzer-supplied aliases.
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
MAX_RERANK = 460          # stage-1 grading budget (candidates)
HEAD_SIZE = 168           # evidence-MINED depth (free: snippet_search only)
STAGE2_SIZE = 104         # judge-replica depth (this is the part that costs)
MINE_GROUP = 3            # paper_ids per scoped snippet_search call
MINE_LIMIT = 9            # passages per scoped call
STAGE1_BATCH = 20
STAGE2_BATCH = 7
TOOL_CONCURRENCY = 6
MINE_CONCURRENCY = 12
LLM_CONCURRENCY = 8
WALL_BUDGET = 22 * 60     # hard stop well inside the 29-minute timeout
MINE_BUDGET = 7 * 60      # cap on the mining phase specifically
CITE_SUBMIT = 30          # ids submitted from the citation path (see below)
REFCHECK_CAP = 300        # citers put through the `references` verification
REFCHECK_BUDGET = 5 * 60  # get_paper_batch(references) is the slowest call --
#                           measured at minutes for hundreds of ids, so it gets
#                           its own deadline and a partial result is accepted.

PAPER_FIELDS = "title,abstract,corpusId,year,venue,authors,tldr,citationCount"
# get_author_papers rejects `tldr` outright ("Unrecognized or unsupported
# fields: [tldr]") — verified on the probe, and it is what silently killed the
# author path in BOTH prior iterations (iteration 002's metadata_15 stdout shows
# the same ToolError). It needs its own, narrower field list.
AUTHOR_PAPER_FIELDS = "title,abstract,corpusId,year,venue,authors"

# Evidence assembly budgets (chars). Title+tldr+abstract must not eat the whole
# 2400 cap, or there is no room for the body passages that prove criteria 2+.
ABSTRACT_CHARS = 850
TLDR_CHARS = 260
SNIPPET_CHARS = 420
MAX_SNIPPETS = 4


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
    __slots__ = ("cid", "pid", "title", "abstract", "tldr", "year", "venue",
                 "authors", "snippets", "fusion", "g1", "weighted", "crit")

    def __init__(self, cid: str):
        self.cid = cid
        self.pid = ""
        self.title = ""
        self.abstract = ""
        self.tldr = ""
        self.year = None
        self.venue = ""
        self.authors: list[str] = []
        self.snippets: list[str] = []
        self.fusion = 0.0
        self.g1 = None          # stage-1 holistic grade 0-3
        self.weighted = None    # stage-2 judge-replica weighted score in [0,1]
        self.crit: list[int] = []

    def absorb(self, doc: dict):
        if not self.pid:
            self.pid = str(doc.get("paperId") or "")
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

    def add_snippet(self, text: str) -> None:
        text = (text or "").strip()
        if not text or len(text) < 60:
            return
        for existing in self.snippets:
            if text[:80] == existing[:80]:
                return
        self.snippets.append(text)

    def brief(self, abs_chars: int) -> str:
        body = self.tldr or self.abstract
        if self.tldr and self.abstract:
            body = self.tldr + " " + self.abstract
        yr = f" ({self.year})" if self.year else ""
        return f"{self.title}{yr}. {_clip(body, abs_chars)}"

    def evidence(self, criteria: list[dict]) -> str:
        """Verbatim passages joined by ' ... ', selected for CRITERION COVERAGE.

        Every passage is a (sentence-boundary) prefix of text the tools returned
        for this same paper, so it stays verbatim-derivable for the grounding
        check. Nothing here is model-written.
        """
        parts: list[str] = []
        used = 0

        def push(piece: str) -> None:
            nonlocal used
            piece = _clip(piece, max(0, EVIDENCE_CAP - used - 5))
            if piece and len(piece) >= 20:
                parts.append(piece)
                used += len(piece) + 5

        if self.title:
            push(self.title)
        if self.tldr:
            push(_clip(self.tldr, TLDR_CHARS))
        if self.abstract:
            push(_clip(self.abstract, ABSTRACT_CHARS))

        # Greedily add the snippet that best supports whichever criterion is
        # currently least supported by the text already selected. This is what
        # converts a grade-2 ("topic yes, method unproven") into a grade 3.
        if self.snippets and criteria:
            crit_terms = [_terms(c.get("name", "") + " " + c.get("description", ""))
                          for c in criteria]
            covered = _terms(" ".join(parts))
            pool = list(self.snippets)
            for _ in range(MAX_SNIPPETS):
                if not pool or used >= EVIDENCE_CAP - 200:
                    break
                gaps = []
                for ct in crit_terms:
                    if not ct:
                        gaps.append(0.0)
                        continue
                    gaps.append(1.0 - len(ct & covered) / len(ct))
                best, best_score = None, 0.0
                for sn in pool:
                    st = _terms(sn)
                    score = sum(gap * (len(ct & st) / len(ct))
                                for gap, ct in zip(gaps, crit_terms) if ct)
                    if score > best_score:
                        best, best_score = sn, score
                if best is None:
                    best = pool[0]
                pool.remove(best)
                push(_clip(best, SNIPPET_CHARS))
                covered |= _terms(best)
        else:
            for sn in self.snippets[:MAX_SNIPPETS]:
                if used >= EVIDENCE_CAP - 200:
                    break
                push(_clip(sn, SNIPPET_CHARS))

        return " ... ".join(parts[:8])


class Pool:
    """Deduplicated candidate store with reciprocal-rank fusion scoring."""

    def __init__(self):
        self.by_cid: dict[str, Candidate] = {}
        # Search/metadata tools apply the benchmark snapshot cutoff; corpusIds
        # are near-monotonic in ingestion date, so the largest one any of them
        # returns is a free upper bound on "inside the snapshot". get_citations
        # is the one tool the cutoff does NOT cover, so this is how its output
        # gets screened without a date parameter existing anywhere.
        self.max_filtered_cid = 0

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
        try:
            self.max_filtered_cid = max(self.max_filtered_cid, int(cid))
        except ValueError:
            pass
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
  float}}. Split the request into its ATOMIC CONCEPTS - one criterion per
  distinct concept that is literally named in the request, using the request's
  own vocabulary. Do NOT merge two concepts into one criterion, and do NOT
  invent a requirement the request never states. Example: "Video aesthetics
  score, using multimodal large models" -> three criteria: (1) "Video
  Aesthetics Scoring" (2) "Multimodal Approach" (3) "Large Models". Example:
  "long video description, long = at least several minutes" -> (1) "Long Video
  Definition" (2) "Video Description" (3) "Relation Between Long Videos and
  Description". Each description is one sentence beginning "The paper must ...".
  Weights sum to 1.0; typical splits are 0.4/0.3/0.3 or 0.4/0.4/0.2.
"candidate_titles": if the request asks for a specific known paper (possibly by
  a nickname such as "the BART paper", a system name, or a one-line
  description), list 1-5 GUESSES at its exact published title, best first. Use
  the real title only, with no subtitle prefix such as "SystemName:" unless the
  published title truly begins that way. Otherwise [].
"reference_titles": exact titles of papers the request merely REFERS TO as a
  landmark rather than asks for (e.g. "authors of the BERT paper" ->
  ["BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding"]). Otherwise [].
"cites_reference": true if the request asks for papers that CITE the paper(s)
  named in "reference_titles" (e.g. "papers citing the T5 paper"), else false.
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
        "cites_reference": bool(data.get("cites_reference")),
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

    kws = list(plan["keyword_queries"])
    # The benchmark's criteria are a CONJUNCTION of the query's atomic concepts,
    # and only a paper satisfying every one of them reaches grade 3 (the sole
    # grade that earns recall). Keyword ranking is a bag of words, so an
    # explicit all-facets query surfaces the conjunctive matches that per-facet
    # queries rank below their single-facet competitors.
    facets = [c.get("name", "").strip() for c in plan.get("criteria") or []
              if c.get("name")]
    if len(facets) >= 2:
        conj = " ".join(facets)
        if conj not in kws:
            kws.append(conj)
        for i in range(len(facets)):
            for j in range(i + 1, len(facets)):
                pair = f"{facets[i]} {facets[j]}"
                if pair not in kws:
                    kws.append(pair)
    kws = kws[:14]
    tasks = [kw(q, 1.0 if i == 0 else 0.85)
             for i, q in enumerate(kws)]
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
    """Pull body passages that PROVE each criterion, per paper.

    The judge sees only `markdown_evidence`. Abstracts establish the topical
    criterion and almost never the method/evaluation criteria, which is why 65
    papers landed on grade 2 last iteration. `snippet_search` scoped by
    `paper_ids` reaches body text and closes that gap.

    Groups of MINE_GROUP=3: `paper_ids` scopes the search, it does not allocate
    per paper — an 8-id scope at limit 24 was probed to leave 2 papers with zero
    passages, so small groups are what guarantee coverage of weak candidates.
    """
    snippet = _tool(state, "snippet_search")
    if not snippet or not head:
        return

    queries = [f"{c['name']}. {c['description']}" for c in plan["criteria"]][:3]
    if not queries:
        return

    groups = [head[i:i + MINE_GROUP] for i in range(0, len(head), MINE_GROUP)]
    stop = min(deadline, time.time() + MINE_BUDGET)
    hits = 0

    async def mine(group: list[Candidate], q: str):
        nonlocal hits
        if time.time() > stop:
            return
        ids = ",".join(f"CorpusId:{c.cid}" for c in group)
        try:
            raw = await asyncio.wait_for(
                snippet(query=q, paper_ids=ids, limit=MINE_LIMIT), timeout=90)
        except asyncio.TimeoutError:
            return
        by_cid = {c.cid: c for c in group}
        for e in _parse_items(raw):
            paper = e.get("paper") if isinstance(e.get("paper"), dict) else {}
            cand = by_cid.get(_cid(paper))
            if cand is None or len(cand.snippets) >= 10:
                continue
            sn = e.get("snippet")
            txt = (sn.get("text") if isinstance(sn, dict) else None) or ""
            before = len(cand.snippets)
            cand.add_snippet(txt)
            hits += len(cand.snippets) - before

    tasks = [mine(g, q) for g in groups for q in queries]
    await _gather(tasks, MINE_CONCURRENCY)
    with_sn = sum(1 for c in head if c.snippets)
    print(f"  mined {hits} passages via {len(tasks)} scoped calls; "
          f"{with_sn}/{len(head)} head papers have body evidence")


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
        f"  3 = the text explicitly demonstrates this criterion\n"
        f"  1 = the text partially or indirectly touches on it\n"
        f"  0 = the text does not support it\n\n"
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
        txt = await _gen(GPT_5_4_MINI, _stage1_prompt(query, criteria, batch, 620),
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
    criteria = plan["criteria"]
    head = head[:STAGE2_SIZE]
    items = [(i, c, c.evidence(criteria)) for i, c in enumerate(head)]
    # Keep prompt cost bounded: the judge reads up to 2500 chars, we show 1600.
    items = [(i, c, _clip(ev, 1600)) for i, c, ev in items if ev]
    batches = [items[i:i + STAGE2_BATCH] for i in range(0, len(items), STAGE2_BATCH)]
    applied = 0

    async def do2(batch):
        nonlocal applied
        if time.time() > deadline:
            return
        local = [(j, c, ev) for j, (_, c, ev) in enumerate(batch)]
        txt = await _gen(CLAUDE_SONNET_4_6, _stage2_prompt(query, criteria, local),
                         config=GenerateConfig(max_tokens=700), tag="stage2")
        applied += _apply_stage2(txt, local, criteria)

    await _gather([do2(b) for b in batches], LLM_CONCURRENCY)
    pred3 = sum(1 for c in head if c.weighted is not None
                and _grade_from_weighted(c.weighted) == 3)
    print(f"  stage2 judged {applied}/{len(items)}; predicted grade-3: {pred3}")


def final_order(head: list[Candidate], rest: list[Candidate]) -> list[Candidate]:
    """Head by predicted weighted score (a continuous value — no mass ties in
    the top-25, where ordering is worth the most), tail by stage-1 then fusion.
    Submitted deep: semantic_f1 has no precision penalty, and anything past K is
    simply never judged."""
    def key(c: Candidate) -> float:
        # Papers past STAGE2_SIZE were still evidence-mined, they just were not
        # judge-replicated. Fold them in on a comparable scale instead of
        # exiling them below every stage-2 paper: recall only counts grade-3
        # papers inside the first K, and K reached 228 on one training query.
        if c.weighted is not None:
            return c.weighted
        return min(0.66, (c.g1 or 0) / 3.0 * 0.85)
    ordered_head = sorted(head, key=lambda c: (-key(c), -(c.g1 or 0), -c.fusion))
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



# ------------------------------------------------- citation-metadata path --

async def _citers(gc, cid: str, limit: int = 1000) -> list[dict]:
    """Unwrap get_citations. NOTE: this is the one tool the benchmark snapshot
    cutoff does not cover, so its output must be screened by the caller."""
    try:
        raw = await asyncio.wait_for(
            gc(paper_id=cid, fields="corpusId,title,year,venue,authors", limit=limit),
            timeout=280)
    except Exception as exc:
        print(f"  [warn] get_citations({cid}) failed: {type(exc).__name__}: {exc}")
        return []
    out = []
    for d in _parse_items(raw):
        cp = d.get("citingPaper") if isinstance(d.get("citingPaper"), dict) else d
        if _cid(cp):
            out.append(cp)
    return out


async def _refs_of(gb, cids: list[str], deadline: float) -> dict[str, set]:
    """corpusId -> set of referenced paperIds, via get_paper_batch.

    get_paper_batch raises `'NoneType' object is not iterable` when ANY id in a
    chunk fails to resolve (probed), so chunks are bisected on error rather
    than dropped — otherwise one bad id silently loses 50 candidates.
    """
    out: dict[str, set] = {}

    async def chunk(ids: list[str]):
        if time.time() > deadline:
            return
        try:
            docs = _parse_items(await asyncio.wait_for(
                gb(ids=[f"CorpusId:{c}" for c in ids],
                   fields="corpusId,title,year,publicationDate,references"),
                timeout=200))
        except Exception:
            if len(ids) == 1:
                return
            half = len(ids) // 2
            await asyncio.gather(chunk(ids[:half]), chunk(ids[half:]))
            return
        for d in docs:
            cid = _cid(d)
            if not cid:
                continue
            pids = set()
            titles = set()
            for r in d.get("references") or []:
                if isinstance(r, dict):
                    if r.get("paperId"):
                        pids.add(str(r["paperId"]))
                    if r.get("title"):
                        titles.add(str(r["title"]).lower()[:48])
            out[cid] = pids | titles

    groups = [cids[i:i + 10] for i in range(0, len(cids), 10)]
    await _gather([chunk(g) for g in groups], 10)
    return out


async def citation_path(state: TaskState, query: str, plan: dict, pool: Pool,
                        deadline: float) -> list[Candidate]:
    """"papers citing X" / "papers citing X and Y".

    Both prior agents answered these with keyword search and scored 0.000 on
    every one. The tools make it near-deterministic: get_citations on the
    referenced paper IS the answer set. For a conjunction, the naive move --
    intersecting two citation lists -- fails, because get_citations caps at 1000
    with no paging and returns newest-first, so a hugely-cited paper like T5
    hands back only the last few months (probed: 0 of 10 gold ids present, while
    all 10 were in the less-cited Spider paper's list). So: enumerate the
    citers of the MOST SELECTIVE reference, then verify each one's `references`
    against the other landmark papers.
    """
    gc, gb = _tool(state, "get_citations"), _tool(state, "get_paper_batch")
    if not gc or not plan["reference_titles"]:
        return []
    refs = await resolve_titles(state, plan["reference_titles"], Pool(), 0.0)
    refs = [r for r in refs if r.cid]
    if not refs:
        return []
    print(f"  cite-refs -> {[(r.cid, r.title[:44]) for r in refs]}")

    lists = await asyncio.gather(*[_citers(gc, r.cid) for r in refs])
    for r, lst in zip(refs, lists):
        print(f"  citers of {r.title[:40]!r} -> {len(lst)}")
    order = sorted(range(len(refs)), key=lambda i: len(lists[i]) or 10 ** 9)
    base_i = order[0]
    base = lists[base_i]
    if not base:
        return []

    docs = {_cid(d): d for d in base if _cid(d)}
    keep_ids = list(docs)

    # Snapshot screen FIRST. get_citations is unfiltered, so post-snapshot
    # citers are pure precision loss on an exact-match path -- and dropping
    # them here also shrinks the expensive reference-verification pass below.
    # Every other tool IS filtered, so the largest corpusId they returned
    # bounds the snapshot from above.
    ceiling = pool.max_filtered_cid
    if ceiling:
        inside = [c for c in keep_ids if c.isdigit() and int(c) <= ceiling]
        if inside:
            print(f"  snapshot screen (cid <= {ceiling}): "
                  f"{len(inside)} of {len(keep_ids)}")
            keep_ids = inside

    others = [refs[i] for i in order[1:]]
    if others and gb:
        # get_paper_batch(references) is heavy (hundreds of refs per paper);
        # give it its own deadline so a slow corpus never eats the 29-minute
        # wall clock, and cap the fan-out.
        sub = sorted(keep_ids, key=lambda c: -int(c) if c.isdigit() else 0)[:REFCHECK_CAP]
        refmap = await _refs_of(gb, sub,
                                min(deadline, time.time() + REFCHECK_BUDGET))
        need_pid = {r.pid for r in others if r.pid}
        need_ttl = {r.title.lower()[:48] for r in others if r.title}
        verified = [c for c in keep_ids
                    if refmap.get(c) and
                    all((p in refmap[c]) for p in need_pid) and
                    all((t in refmap[c]) for t in need_ttl)]
        # Whatever the deadline left unresolved keeps its place behind the
        # verified ids rather than vanishing: a partial check is still a
        # precision win, and an empty submission scores 0.
        unresolved = [c for c in keep_ids if c not in refmap]
        # A reference list that resolved for nobody means the check is broken,
        # not that the answer is empty -- fall back rather than submit nothing.
        print(f"  cites-all-{len(refs)} check: {len(verified)} of "
              f"{len(refmap)} resolved citers")
        if verified:
            keep_ids = verified + unresolved[:max(0, 60 - len(verified))]

    # Newest-first, inside the snapshot ceiling. corpusIds are near-monotonic
    # in ingestion date, and get_citations gives no date parameter to sort on.
    keep_ids = sorted(keep_ids, key=lambda c: -int(c) if c.isdigit() else 0)

    years, ymin, ymax = set(plan["years"]), plan["year_min"], plan["year_max"]
    aliases = plan["venue_aliases"] or [v.lower() for v in plan["venues"]]
    out: list[Candidate] = []
    for cid in keep_ids:
        d = docs[cid]
        yr = d.get("year")
        if years and yr not in years:
            continue
        if not years:
            if ymin is not None and (yr is None or yr < ymin):
                continue
            if ymax is not None and (yr is None or yr > ymax):
                continue
        if aliases and not any(v in (d.get("venue") or "").lower() for v in aliases):
            continue
        cand = pool.add(d, len(out), 0.0)
        if cand is not None:
            cand.fusion += 10.0 - 0.001 * len(out)   # keep newest-first
            out.append(cand)
    print(f"  citation path -> {len(out)} candidates")
    return out


async def metadata_path(state: TaskState, query: str, plan: dict, pool: Pool,
                        deadline: float) -> list[Candidate]:
    """Author/venue/year filters. No tool has a date parameter, so year
    filtering happens here in Python."""
    authors = list(plan["authors"])

    # "papers citing the T5 paper and the spider paper" is a citation-graph
    # lookup, not a keyword search. Both prior agents scored 0.000 on every one.
    did_broad = False
    if plan.get("cites_reference") and plan["reference_titles"]:
        # Warm the pool first: the snapshot screen inside citation_path needs a
        # corpusId ceiling from the DATE-FILTERED search tools.
        await broad_retrieve(state, query, plan, pool, deadline)
        cited = await citation_path(state, query, plan, pool, deadline)
        if cited:
            if len(cited) > CITE_SUBMIT * 3:
                head = await rerank(query, plan, cited, deadline)
                strong = [c for c in head if (c.g1 or 0) >= 2]
                if len(strong) >= CITE_SUBMIT // 2:
                    cited = strong
            # Exact-match F1 punishes a long list hard: probed on
            # metadata_26, the reference check recovers 10/10 gold but 244
            # papers cite both landmarks, so submitting all of them is
            # precision 0.041 / F1 0.079. citation_path already returns
            # newest-first inside the snapshot, and the gold there was exactly
            # the newest cohort under the cutoff, so a short recent list is
            # worth far more than a complete one. Gold sizes on this path run
            # 1-10; 30 keeps recall while bounding the precision loss.
            return cited[:CITE_SUBMIT]
        print("  citation path empty -> falling back to keyword/author route")
        did_broad = True
    else:
        did_broad = False

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
    if not did_broad:
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
