"""PaperFindingBench solver: iteration 11 - a conjunctive, non-saturating,
model-ensembled ranking signal on top of iteration 10's retrieval and evidence.

Iteration 10 is the base: its semantic mean (0.372 over 11 queries) beat iter7
(0.353) and iter4 (0.294) on the same draw, and it won 8 of the 11. Retrieval,
criterion-targeted snippet harvesting, the single-paper repair pass, evidence
assembly, and the whole `specific_f1` path are inherited unchanged.

What is replaced is the signal that ORDERS the submission - which is a recall
lever, not just a `rank` lever, because only the first K results are ever judged
(`#judged == k_estimate` in every observed problem, K ran 14-180). A grade-3
paper pushed past position K earns exactly as little as one never retrieved.

The defect, visible in iteration 10's own stdout against the judge's verdicts:

  query         agent predicted-grade-3   judge grade-3   rank
  semantic_91        119 / 250               29 / 100     0.534
  semantic_203        22 / 250                3 /  24     0.331
  semantic_193         0 / 250               26 / 180     0.813

It graded on the benchmark judge's own 3-valued scale and ranked by the scorer's
own additive arithmetic `min(1, sum w_c r_c / 3)`. Over three criteria that scale
has 27 possible outputs, so it either ties 119 papers at exactly 1.0 (and their
order falls back to retrieval order - rank 0.534, the worst in the batch) or
awards no full marks at all. The additive form has a second bug: it lets a heavy
criterion buy off a missing light one, `(3,3,0)` = 0.70 outranking a balanced
`(3,1,1)` = 0.60, while the benchmark's gate needs `weighted > 0.99`, i.e.
essentially EVERY criterion perfect. The gate is conjunctive; the ranker was not.

Three changes, all in the ranking signal:

  1. 0-9 per criterion instead of 0/1/3. Same one digit per criterion, same token
     cost, ~1000x the resolution, no saturation.
  2. Weighted GEOMETRIC aggregation - the conjunctive form of the gate. Monotone
     in every criterion, never saturates, and one weak criterion drags the product
     down the way `weighted > 0.99` does.
  3. An INDEPENDENT second grader (CLAUDE_HAIKU_4_5, a different model family)
     over the top 72 - the band the judge actually reads on most queries. Votes
     are kept per grader and combined as a reliability-weighted mean, so the sort
     key can prefer a paper two graders liked over one a single lenient pass waved
     through. Confirmation count is a tiebreaker strictly below the mean, so it
     can never promote a weak paper for having been looked at more often.

Plus one guard on the metadata path: never emit an empty result set, because
F1 = 2h/(n+G) is 0 at n = 0 and any best-effort head is weakly better.

Projected cost ~$0.061 per semantic query, ~$0.052 batch mean against the $0.063
free-zone threshold. The deep and head passes run concurrently, so wall clock is
unchanged (iteration 10's worst query was 1120 s against the 1740 s limit).

--- inherited design notes from iteration 10 and earlier ---

a working cited-author filter, no survivor truncation, and a wider query pool.

Iteration 8's semantic path is the measured best of the three agents on iteration
8's draw (semantic mean 0.391 vs 0.344 for iter6 and 0.341 for iter7 over the same
eight queries), so it is kept and widened rather than rebuilt. The whole of iter8's
loss to iter7 came from one non-semantic query: `specific_9` ("the MS^2 DeYong2021
paper"), where iter7 scored 1.000 and iter8 and iter6 scored 0.000 running the SAME
resolution code. The stdout shows why - it is sampling variance in one parametric
recall call:

  iter7  candidate titles: ['MS2: Multi-Document Summarization of Medical Studies', ...]
  iter8  candidate titles: ['Multi-scale modeling of electron transfer in proteins', ...]

Four changes:

1. **The specific path stops depending on a single title draw.** Two independent
   guesses - GPT_5_4 at reasoning_effort="low" and CLAUDE_SONNET_4_6, different
   pretraining distributions - are unioned, every guess gets its own
   `search_paper_by_title`, and the request's citation key is parsed for a first
   author surname and a year. Those cues are the only signal in the loop that does
   NOT come from the model that produced the (possibly wrong) titles: they reorder
   the shortlist, they are stated to the verifier, and they gate an author-search
   fallback that fires only when nothing retrieved agrees with them. The verifier
   now echoes the chosen record's title back and is matched on that string, with
   the index as a fallback - index selection out of a numbered list is the step
   that has misfired repeatedly on this benchmark.

2. **The cited-author filter reads the edge forward.** `fields=references` errors
   out on both `get_paper` and `get_paper_batch` on this server, so iteration 8's
   "papers citing X" check silently kept 0 of 114 and voided itself
   (`[tool-fail] refs: ... 'NoneType' object is not iterable` in metadata_31's
   stdout). Replaced by `get_citations` over the author's own papers, unioned into
   a citing set and intersected with the pool - which is already snapshot-filtered,
   so the cutoff gap in `get_citations` cannot leak.

3. **Structurally filtered survivors are never truncated.** metadata_31 kept 114
   ids and submitted 100 of them, discarding 10 of the 16 gold ids for 0.103 where
   the full 114 scores 0.246. Inside a set that all passed the same filters, every
   survivor's hit rate is about h/n, and one more candidate raises F1 whenever its
   hit probability exceeds h/(n+G) = F1/2 - always true here.

4. **A second, differently-framed query planner on the semantic path.** Recall is
   grade-3 count inside K over K, the pool is its ceiling, and every keyword call
   is free; one prompt sampled once yields near-duplicate phrasings. A follow-up
   call sees the criteria and the queries already issued and returns eight more in
   other vocabulary registers (the authors\' own title wording, survey wording,
   applied wording, jargon wording), for about $0.002.

Iteration 8 notes follow.

PaperFindingBench solver: structured metadata constraints + judge-aligned ranking.

Iteration 6 keeps iteration 4's semantic and specific paths byte-identical — they
carry 88% of the test mix and iteration 5's attempt to improve the semantic path
regressed — and rebuilds `solve_metadata`, which has scored ~0.02-0.10 for every
agent in every iteration so far. Its failures were rule execution, not retrieval:
the gold papers were usually already in the pool. An index-selecting LLM venue
verifier kept `Autoimmunity` for a Nature-portfolio request and `EMNLP` for a
SPLASH one; `2010 or 2012` was read as the range 2010-2012; `min_authors=2` was
compared with `<=` so it demanded three authors; `authors of the BERT paper` had
no plan slot and got filed as a citation constraint; and when the filters emptied
the set the agent submitted the raw 60-paper pool. Venue matching is now
deterministic containment over LLM-expanded *full official* venue names, years
can be a discrete set, and co-authorship / cited-author / self-citation-exclusion
/ journal-only constraints each route to the tool that owns them. Citation sets
union `get_citations` (unfiltered, newest-first, 1000-capped) with the nested
`citations` field, which IS snapshot-filtered — metadata_26 intersected two
post-snapshot windows and returned ten ids all newer than every gold id.

Original iteration-4 design notes follow.

Iterations 2-3 established that `harmonic(rank, recall)` on the semantic path is
recall-bound: across iteration 3's eleven semantic queries `rank` was 0.82-0.91 on
eight of them while `recall` never exceeded 0.38. Recall is
`|{i <= K : grade_i = 3}| / K`, so only grade-3 papers inside the first K slots
earn anything, and a paper reaches grade 3 only when *every* weighted criterion is
judged Perfectly Relevant — from `markdown_evidence` alone. Iteration 3's verdicts
show both ways that fails: grade-2/grade-1 pile-ups where the evidence supported
two criteria out of three (semantic_189: 15 highly + 12 somewhat vs 1 perfect;
semantic_221: 85 somewhat vs 55 perfect), and low `rank` on the small-K queries
where the few convertible papers sat too deep to be judged (semantic_43 rank 0.10
at K=16, semantic_33 rank 0.29 at K=22).

The change here is where the ranking signal comes from. Iteration 3 ordered papers
by an LLM grade read off a 180-character title+tldr gist, then assembled evidence
afterwards. This agent inverts that: it assembles the evidence first and then
grades *that text* — the only text the benchmark judge will ever see — on the
judge's own per-criterion scale (r_c in {0, 1, 3}), through the judge's own
arithmetic (`weighted = min(1, sum w_c * r_c / 3)`). Predictor and scorer read the
same characters, so the ordering optimizes both score terms at once. The paid
title-gist screen is replaced by a free lexical pre-rank, which moves the whole
LLM budget onto the text that decides the score.

Retrieval also widens (8 keyword variants, 4 unscoped `snippet_search` probes, a
100-paper criterion-targeted harvest), because on the large-K queries — K reached
228 — no amount of ranking substitutes for having enough convertible papers.

The two exact-match paths score `2h/(n+G)` with G typically 1-3, so their fix is
arithmetic: never pad. Gold was a single id on three of the four observed
`specific_f1` queries, so this agent attaches an extra id only for an exact
normalized-title duplicate record and never as a second-best guess — iteration 3's
specific_15 scored 0.667 instead of 1.000 for exactly that padded guess.
"""

import asyncio
import json
import math
import re
import time

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import (CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6, GPT_5_4,
                            GPT_5_4_MINI)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

N_VARIANTS = 12            # keyword query variants from the criteria planner
N_EXTRA_VARIANTS = 8       # second-planner variants, different vocabulary register
SEARCH_LIMIT = 100         # max allowed by search_papers_by_relevance
OPEN_SNIPPET_CRIT = 4      # unscoped per-criterion snippet_search calls (discovery)
N_MORE_LIKE_THIS = 10      # top pre-ranked titles re-issued as keyword queries
GRADE_BATCH = 40
# The judge reads markdown_evidence ALONE, so the ranking signal is graded on the
# assembled evidence rather than on a title+tldr gist: the predictor and the
# scorer then read the same text.
POOL_EVIDENCE_GRADE = 250  # papers judge-graded on their assembled evidence
EV_CHARS_FOR_GRADE = 520   # evidence chars shown to the cheap grader
HEAD_RERANK_N = 28         # head re-graded on the stronger model
EV_CHARS_FOR_HEAD = 900
# A second, DIFFERENT-FAMILY grader over the band that the benchmark judge
# actually reads.  Observed K (the judge's depth) ran 14-180 with a median of
# 36, so 72 covers K outright on six of eleven observed queries and dominates
# the head on the rest.  Its purpose is not a better single verdict but an
# INDEPENDENT one:
# iteration 10's single grader marked 119/250 papers as full-coverage on
# semantic_91 where the benchmark judge found 29/100, and 119 tied scores
# collapse the ordering to retrieval order (rank 0.534, the worst in the set).
DEEP_RERANK_N = 72
EV_CHARS_FOR_DEEP = 560   # measured budget: ~$0.016/query on CLAUDE_HAIKU_4_5
# Reliability weights for the ensemble mean.  Deeper/stronger passes read more
# evidence, so they carry more mass; the pool pass is the wide, cheap prior.
VOTE_W_POOL = 0.55
VOTE_W_DEEP = 0.90
VOTE_W_HEAD = 1.00
# Measured: a 25-paper scope with limit=100 returns passages for only 13 of the
# 25 (95 s). A 20-paper scope raises per-call coverage, and each paper gets one
# shot per criterion, so a paper appears in at least one harvest ~90% of the time.
CRIT_SNIPPET_HEAD = 100    # papers given criterion-targeted snippet_search
CRIT_SNIPPET_BATCH = 20    # paper_ids per scoped snippet_search call
# `paper_ids` is a SCOPE filter, not a per-paper allocation, so a 20-paper scoped
# call hands its passages to whichever papers already match best and returns
# nothing for exactly the papers that still need the criterion demonstrated.
# The gap-driven second pass below re-probes those papers ONE at a time, which is
# the only scope under which a weak match is guaranteed its own passages.
REPAIR_HEAD = 170          # papers eligible for single-paper gap repair
REPAIR_CRIT_MAX = 2        # uncovered criteria probed per paper
REPAIR_MAX_CALLS = 420     # wall-clock guard; tool calls themselves are free
REPAIR_LIMIT = 5           # passages per single-paper probe
REPAIR_CONCURRENCY = 24    # in-flight scoped probes
REPAIR_TIMEOUT = 100.0     # a one-paper limit=5 probe is a cheap query
REPAIR_DEADLINE = 1000.0   # leave room for the grade + head passes after repair
MAX_SNIPS_PER_PAPER = 14   # raw material kept per paper for evidence assembly
MAX_SUBMIT_SEMANTIC = 250  # the scorer's own cap; tail slots are free upside
MAX_SUBMIT_SPECIFIC = 4    # precision is half the score on the exact-match path
MAX_SUBMIT_METADATA = 250  # the scorer's cap; see the note at the submit site
EVIDENCE_CHARS = 2400      # under the scorer's 2500-char truncation point
MAX_PASSAGES = 8           # the scorer keeps at most 8 passages per paper
SNIPPET_CHARS = 620        # per-snippet trim; a raw snippet is ~3000 chars

PAPER_FIELDS = "title,abstract,corpusId,tldr,year,venue,authors,citationCount"
SNIPPET_TIMEOUT = 240.0    # under the 300 s per-call transport ceiling
CITED_BY_FANOUT = 80       # seed papers expanded through get_citations
ENRICH_DEADLINE = 1150.0   # stop optional enrichment; hard timeout is 29 min


# ---------------------------------------------------------------------------
# Tool plumbing
# ---------------------------------------------------------------------------

def _get_tool(state: TaskState, name: str):
    """Look a tool up by registered name, tolerating any single tool whose
    ToolDef cannot be built (one bad entry must not take the query down)."""
    for t in state.tools or []:
        try:
            tname = ToolDef(t).name
        except Exception:  # noqa: BLE001
            tname = getattr(t, "__name__", "")
        if tname == name:
            return t
    print(f"  [no-tool] {name} not in state.tools")
    return None


def _parse_items(raw) -> list[dict]:
    """Normalize any MCP corpus tool's return into a flat list of dicts.

    Every tool returns a list of ContentText whose `.text` is JSON; some wrap
    their payload as {"data": [...]}.
    """
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
            out.extend(d for d in (doc.get("data") or []) if isinstance(d, dict))
        elif isinstance(doc, dict):
            out.append(doc)
        elif isinstance(doc, list):
            out.extend(d for d in doc if isinstance(d, dict))
    return out


async def _safe(coro, label: str, timeout: float | None = None):
    """Await a tool call, converting any failure into None. Breadth is built
    from many independent calls; one timing out must not sink the query."""
    try:
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro
    except Exception as exc:  # noqa: BLE001 - tool errors are expected & varied
        print(f"  [tool-fail] {label}: {type(exc).__name__}: {str(exc)[:160]}")
        return None


async def _llm(handle, prompt: str, label: str, config: GenerateConfig | None = None) -> str:
    try:
        resp = await (handle.generate(prompt, config=config) if config
                      else handle.generate(prompt))
        text = (resp.completion or "").strip()
        if not text:
            print(f"  [llm-empty] {label}")
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"  [llm-fail] {label}: {type(exc).__name__}: {str(exc)[:160]}")
        return ""


def _json_from(text: str):
    """Pull the first JSON object/array out of an LLM completion."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Paper record helpers
# ---------------------------------------------------------------------------

def _cid(paper: dict) -> str:
    """corpusId is an int on the search tools and a str on snippet_search."""
    val = paper.get("corpusId")
    return "" if val in (None, "") else str(val).strip()


def _tldr_text(paper: dict) -> str:
    tldr = paper.get("tldr")
    if isinstance(tldr, dict):
        return (tldr.get("text") or "").strip()
    return tldr.strip() if isinstance(tldr, str) else ""


def _author_names(paper: dict) -> list[str]:
    out = []
    for a in paper.get("authors") or []:
        name = a.get("name") if isinstance(a, dict) else a
        if name:
            out.append(str(name))
    return out


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _year_of(paper: dict) -> int | None:
    year = paper.get("year")
    if isinstance(year, int):
        return year
    m = re.match(r"(\d{4})", str(year or paper.get("publicationDate") or ""))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Criterion coverage (deterministic, free)
# ---------------------------------------------------------------------------

_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "must", "paper", "papers",
    "which", "such", "their", "have", "has", "are", "was", "were", "its", "into",
    "using", "used", "use", "uses", "based", "study", "studies", "work", "works",
    "approach", "method", "methods", "propose", "proposed", "discuss", "discusses",
    "about", "these", "those", "than", "then", "also", "other", "some", "more",
    "explicitly", "specifically", "focus", "focuses", "focused", "address",
    "addresses", "involve", "involves", "research", "literature", "how", "what",
    "including", "include", "includes", "well", "both", "each", "over", "when",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    return {_stem(w) for w in words if len(w) >= 4 and w not in _STOP}


def _covers(passage_terms: set[str], crit_terms: set[str]) -> bool:
    """Does a passage lexically demonstrate a criterion?

    Criterion probes are short (3-6 content words). Requiring most of them keeps
    the signal honest without demanding an exact phrase match.
    """
    if not crit_terms:
        return False
    need = max(1, int(round(0.6 * len(crit_terms))))
    return len(crit_terms & passage_terms) >= need


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _trim_snippet(snip: str, crit_terms: set[str]) -> str:
    """Cut a ~3000-char snippet down to the contiguous sentence window that best
    demonstrates the criterion. Contiguity keeps it a verbatim substring."""
    snip = re.sub(r"\s+", " ", (snip or "")).strip()
    if len(snip) <= SNIPPET_CHARS:
        return snip
    sents = _split_sentences(snip)
    if not sents:
        return snip[:SNIPPET_CHARS]
    scores = [len(_terms(s) & crit_terms) for s in sents]
    best = max(range(len(sents)), key=lambda i: (scores[i], -i))
    lo = hi = best
    out = sents[best]
    while len(out) < SNIPPET_CHARS:
        grew = False
        if hi + 1 < len(sents) and len(out) + 1 + len(sents[hi + 1]) <= SNIPPET_CHARS:
            hi += 1
            out = out + " " + sents[hi]
            grew = True
        if lo - 1 >= 0 and len(out) + 1 + len(sents[lo - 1]) <= SNIPPET_CHARS:
            lo -= 1
            out = sents[lo] + " " + out
            grew = True
        if not grew:
            break
    return out[:SNIPPET_CHARS]


def _build_evidence(paper: dict, snips: list[tuple[int, str]],
                    crit_terms: list[set[str]]) -> tuple[str, float, list[bool]]:
    """Assemble <=8 verbatim passages under the char cap, spending the budget on
    criteria the abstract does not already demonstrate.

    Returns (evidence, fraction of criteria lexically covered, per-criterion
    covered flags). Every emitted
    passage is a contiguous substring of text the tools returned for THIS paper,
    so it survives the scorer's grounding check.
    """
    n_crit = len(crit_terms)
    covered = [False] * n_crit
    passages: list[str] = []
    used = 0

    def mark(text: str) -> None:
        pt = _terms(text)
        for i, ct in enumerate(crit_terms):
            if not covered[i] and _covers(pt, ct):
                covered[i] = True

    def add(text: str) -> bool:
        nonlocal used
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text or len(passages) >= MAX_PASSAGES:
            return False
        cost = len(text) + (5 if passages else 0)
        if used + cost > EVIDENCE_CHARS:
            return False
        passages.append(text)
        used += cost
        mark(text)
        return True

    add((paper.get("title") or "")[:300])
    add(_tldr_text(paper)[:450])

    abstract = re.sub(r"\s+", " ", (paper.get("abstract") or "")).strip()
    # Reserve room for criterion-targeted body text when we actually have some.
    reserve = min(1100, SNIPPET_CHARS * min(2, len(snips))) if snips else 0
    if abstract:
        room = EVIDENCE_CHARS - used - reserve - 5
        if len(abstract) <= room:
            add(abstract)
        else:
            # Sentence-select: prefer sentences demonstrating criteria that are
            # still uncovered, then emit the picks as contiguous runs so each
            # stays verbatim.
            sents = _split_sentences(abstract)
            want = [i for i, c in enumerate(covered) if not c]
            scored = []
            for j, s in enumerate(sents):
                st = _terms(s)
                gain = sum(1 for i in want if _covers(st, crit_terms[i]))
                scored.append((-gain, j))
            scored.sort()
            picked, budget = set(), max(room, 400)
            spend = 0
            for _, j in scored:
                if spend + len(sents[j]) + 1 > budget:
                    continue
                picked.add(j)
                spend += len(sents[j]) + 1
            if not picked:
                picked = {0}
            runs, cur = [], []
            for j in sorted(picked):
                if cur and j == cur[-1] + 1:
                    cur.append(j)
                else:
                    if cur:
                        runs.append(cur)
                    cur = [j]
            if cur:
                runs.append(cur)
            for run in runs:
                add(" ".join(sents[j] for j in run))

    # Criterion-targeted body passages, uncovered criteria first.
    remaining = [s for s in snips]
    for i in range(n_crit):
        if covered[i]:
            continue
        hit = None
        for k, (ci, text) in enumerate(remaining):
            if ci != i:
                continue
            trimmed = _trim_snippet(text, crit_terms[i])
            if _covers(_terms(trimmed), crit_terms[i]):
                hit = (k, trimmed)
                break
        if hit is None:
            # Cross-tag rescue. A passage harvested under one criterion's probe
            # very often demonstrates a different one - the grade counts what the
            # text covers, not which query retrieved it, so re-trim every passage
            # toward THIS criterion before giving up on it.
            for k, (ci, text) in enumerate(remaining):
                trimmed = _trim_snippet(text, crit_terms[i])
                if _covers(_terms(trimmed), crit_terms[i]):
                    hit = (k, trimmed)
                    break
        if hit is not None and add(hit[1]):
            remaining.pop(hit[0])

    # Spend anything left over on the best remaining passages.
    for ci, text in remaining:
        if len(passages) >= MAX_PASSAGES or used >= EVIDENCE_CHARS - 200:
            break
        ct = crit_terms[ci] if 0 <= ci < n_crit else set()
        add(_trim_snippet(text, ct))

    frac = (sum(covered) / n_crit) if n_crit else 1.0
    return " ... ".join(passages), frac, covered


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _emit(state: TaskState, results: list[dict]) -> None:
    """Write the scorer's JSON schema. `markdown_evidence` is required on every
    result even where its content is ignored (the exact-match paths)."""
    state.output.completion = json.dumps(
        {"output": {"query_id": state.sample_id, "results": results}}
    )
    lens = [len(r["markdown_evidence"]) for r in results]
    avg = sum(lens) // max(1, len(lens))
    print(f"  SUBMITTED {len(results)} papers (evidence chars avg={avg})")


def _emit_ids(state: TaskState, papers: list[dict]) -> None:
    results, seen = [], set()
    for paper in papers:
        cid = _cid(paper)
        if cid and cid not in seen:
            seen.add(cid)
            results.append({"paper_id": cid, "markdown_evidence": ""})
    _emit(state, results)


# ---------------------------------------------------------------------------
# specific_f1 — resolve a named paper to its corpus record(s)
# ---------------------------------------------------------------------------

_CITEKEY = re.compile(r"\b([A-Za-z][A-Za-z'\-]{2,})\s*[,\s]*(?:et\s+al\.?)?[,\s(]*((?:19|20)\d{2})", re.I)
_GLUED_KEY = re.compile(r"\b([A-Za-z]{3,}?)((?:19|20)\d{2})([a-z][a-z0-9]*)?\b")
_STOP_SPECIFIC = {"the", "a", "an", "paper", "papers", "article", "study", "work",
                  "find", "get", "show", "me", "by", "et", "al", "from", "of", "on",
                  "in", "for", "and", "please", "titled", "called", "named", "with"}


def _specific_cues(query: str) -> tuple[list[str], list[int]]:
    """First-author surnames and years encoded in the request.

    Specific queries name their target with a citation key - `fabri2019multinews`,
    `DeYong2021`, `Lewis et al. 2019` - and those two fields are checkable against
    every corpus record we retrieve, so they are the cheapest disambiguator we
    have when the model's recalled title is wrong.
    """
    surnames: list[str] = []
    years: list[int] = []
    for m in _GLUED_KEY.finditer(query):
        surnames.append(m.group(1).lower())
        years.append(int(m.group(2)))
    for m in _CITEKEY.finditer(query):
        w = m.group(1).lower()
        if w not in _STOP_SPECIFIC:
            surnames.append(w)
        years.append(int(m.group(2)))
    for m in re.finditer(r"\b((?:19|20)\d{2})\b", query):
        years.append(int(m.group(1)))
    ded_s, ded_y = [], []
    for w in surnames:
        if w not in ded_s:
            ded_s.append(w)
    for y in years:
        if y not in ded_y:
            ded_y.append(y)
    return ded_s[:3], ded_y[:3]


def _surname_hit(paper: dict, surnames: list[str]) -> bool:
    """Prefix match, because citation keys mangle the spelling (`DeYong2021` for
    DeYoung) and a 4-character agreement still separates the right record from
    an unrelated one."""
    if not surnames:
        return False
    names = " ".join(_author_names(paper)).lower()
    for sn in surnames:
        if len(sn) >= 4 and (sn in names or sn[:max(4, len(sn) - 2)] in names):
            return True
    return False


async def _guess_titles(query: str, handle, label: str, config=None) -> list[str]:
    ask = (
        "A user is looking for one specific, known scientific paper. Using your "
        "knowledge of the literature, give the paper's exact published title.\n\n"
        f'User request: "{query}"\n\n'
        "Reply with JSON only:\n"
        '{"titles": ["most likely exact title", "second guess", "third guess"]}\n'
        "Order most-confident first. Give 1-4 titles. Use the real published "
        "title, not the nickname (e.g. 'the GPT-2 paper' -> 'Language Models are "
        "Unsupervised Multitask Learners'). A citation key like 'fabri2019multinews' "
        "encodes first author surname, year and a title word - decode all three and "
        "use them; the surname may be misspelled, so trust the title word and year "
        "at least as much. If the request names a system, dataset or model by an "
        "acronym, the published title usually spells the acronym out."
    )
    plan = _json_from(await _llm(handle, ask, label, config)) or {}
    return [t.strip() for t in (plan.get("titles") or [])
            if isinstance(t, str) and t.strip()][:4]


async def solve_specific(state: TaskState, query: str) -> None:
    title_search = _get_tool(state, "search_paper_by_title")
    kw_search = _get_tool(state, "search_papers_by_relevance")
    author_search = _get_tool(state, "search_authors_by_name")
    author_papers = _get_tool(state, "get_author_papers")

    surnames, years = _specific_cues(query)

    # Two model families, because this step is pure parametric recall and its
    # failure mode is sampling variance, not reasoning: on iteration 8 the same
    # code returned "MS2: Multi-Document Summarization of Medical Studies" once
    # (1.000) and "Multi-scale modeling of electron transfer in proteins" the
    # next run (0.000). One reasoning-enabled guess plus one from a different
    # pretraining distribution turns a single draw into a union.
    guesses = await asyncio.gather(
        _guess_titles(query, GPT_5_4, "specific/titles-a",
                      GenerateConfig(reasoning_effort="low")),
        _guess_titles(query, CLAUDE_SONNET_4_6, "specific/titles-b"),
    )
    titles: list[str] = []
    seen_t: set[str] = set()
    for lst in guesses:
        for t in lst:
            key = _norm_title(t)
            if key and key not in seen_t:
                seen_t.add(key)
                titles.append(t)
    titles = titles[:6]
    print(f"  candidate titles: {titles}")
    print(f"  cues: surnames={surnames} years={years}")

    bare = re.sub(r"^(the|find|get|show me|give me)\s+", "", query.strip(), flags=re.I)
    bare = re.sub(r"\bpapers?\b", "", bare, flags=re.I).strip() or query

    calls = []
    if title_search:
        calls += [_safe(title_search(title=t, fields=PAPER_FIELDS), f"title:{t[:40]}")
                  for t in titles]
        calls.append(_safe(title_search(title=bare, fields=PAPER_FIELDS), "title:bare"))
    if kw_search:
        for t in titles[:3]:
            calls.append(_safe(kw_search(keyword=t, fields=PAPER_FIELDS, limit=10),
                               f"kw:{t[:30]}"))
        # The nickname itself catches cases where the recalled title is wrong but
        # the corpus index still knows the paper.
        calls.append(_safe(kw_search(keyword=bare, fields=PAPER_FIELDS, limit=15), "kw:bare"))
        # Surname + the request's own content words: the one query that does not
        # depend on the model having recalled the title correctly.
        toks = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", query)
                if w.lower() not in _STOP_SPECIFIC and not re.fullmatch(r"(19|20)\d{2}", w)]
        if surnames and toks:
            calls.append(_safe(kw_search(keyword=" ".join([surnames[0]] + toks[:6]),
                                         fields=PAPER_FIELDS, limit=10), "kw:cue"))

    raws = await asyncio.gather(*calls) if calls else []

    candidates: list[dict] = []
    seen: set[str] = set()
    for raw in raws:
        for paper in _parse_items(raw):
            cid = _cid(paper)
            if cid and cid not in seen:
                seen.add(cid)
                candidates.append(paper)

    # Author fallback, run only when nothing retrieved so far agrees with the
    # citation key. Keeps the verifier's list clean in the common case.
    if surnames and author_search and author_papers and \
            not any(_surname_hit(p, surnames) for p in candidates):
        ids = await _safe(author_search(name=surnames[0], fields="name,paperCount",
                                        limit=5), "specific/authors")
        recs = [a for a in _parse_items(ids) if a.get("authorId")][:3]
        extra = await asyncio.gather(*[
            _safe(author_papers(author_id=a["authorId"], paper_fields=PAPER_FIELDS,
                                limit=100), f"authpapers:{a.get('name','')[:20]}")
            for a in recs]) if recs else []
        added = 0
        for raw in extra:
            for paper in _parse_items(raw):
                cid = _cid(paper)
                if not cid or cid in seen:
                    continue
                yr = _year_of(paper)
                if years and yr is not None and min(abs(yr - y) for y in years) > 1:
                    continue
                seen.add(cid)
                candidates.append(paper)
                added += 1
        print(f"  author fallback added {added} candidate(s)")

    if not candidates:
        _emit(state, [])
        return

    # Order the shortlist so cue-consistent records are seen first: an LLM
    # picking out of a 30-line list attends to the head, and the cues are the
    # only signal here that does not come from the same model that produced the
    # (possibly wrong) titles.
    guess_terms = set()
    for t in titles[:2]:
        guess_terms |= {w for w in _terms(t) if w not in _STOP_SPECIFIC}

    def _cue_score(paper: dict) -> float:
        sc = 0.0
        if _surname_hit(paper, surnames):
            sc += 2.0
        yr = _year_of(paper)
        if years and yr is not None:
            gap = min(abs(yr - y) for y in years)
            sc += 1.5 if gap == 0 else (0.6 if gap == 1 else 0.0)
        if guess_terms:
            pt = _terms(paper.get("title") or "")
            sc += 2.0 * (len(pt & guess_terms) / max(1, len(guess_terms)))
        try:
            sc += min(1.0, float(paper.get("matchScore") or 0) / 100.0)
        except (TypeError, ValueError):
            pass
        return sc

    candidates.sort(key=lambda p: -_cue_score(p))

    lines = []
    for i, paper in enumerate(candidates[:30]):
        authors = ", ".join(_author_names(paper)[:3])
        lines.append(
            f"{i}: {(paper.get('title') or '')[:150]} | {_year_of(paper) or '?'} "
            f"| {authors[:80]} | {(paper.get('venue') or '')[:40]}"
        )
    cue_note = ""
    if surnames or years:
        cue_note = ("The request's citation key indicates first author surname ~"
                    f"{surnames[0] if surnames else '?'} and year ~"
                    f"{years[0] if years else '?'} (the surname spelling may be "
                    "slightly off). A record agreeing with both is far more likely "
                    "to be the target than one that agrees with neither.\n")
    verify = (
        f'A user asked for a specific paper: "{query}"\n\n'
        "Candidate corpus records:\n" + "\n".join(lines) + "\n\n" + cue_note +
        "Which single record IS that paper? Merely related papers must NOT be "
        "included - each wrong id directly reduces the score. Only name a second "
        "record if it is a duplicate/alternate record of the very same paper "
        "(preprint and published version, identical title).\n"
        'Reply with JSON only: {"title": "<the chosen record\'s title, copied '
        'exactly from the list>", "index": <its number>, "also": [<index of a '
        'duplicate record, if any>]}'
    )
    verdict = _json_from(await _llm(GPT_5_4, verify, "specific/verify",
                                    GenerateConfig(reasoning_effort="low"))) or {}

    shown = candidates[:30]
    best = None
    # Match on the echoed title first: index selection out of a numbered list is
    # the step that has misfired before on this benchmark.
    echoed = _norm_title(str(verdict.get("title") or ""))
    if echoed:
        for paper in shown:
            if _norm_title(paper.get("title", "")) == echoed:
                best = paper
                break
    if best is None:
        try:
            idx = int(verdict.get("index"))
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(shown):
            best = shown[idx]
    if best is None:
        best = shown[0]

    # Duplicate corpus records of ONE paper are all in gold (specific_39's gold
    # holds 5 ids for a single paper), so identical normalized titles are free
    # recall. Anything else is pure precision loss: iteration 2 scored 0.667 on
    # specific_11 for one padded id.
    target = _norm_title(best.get("title", ""))
    picks = [best]
    for paper in candidates:
        if paper is not best and _norm_title(paper.get("title", "")) == target:
            picks.append(paper)
    # Do NOT fall back to the LLM's second choice when no title duplicate exists.
    # Gold is a single id in 3 of the 4 observed specific queries, so a padded
    # "alternate version" guess costs 1.000 -> 0.667 far more often than it buys
    # recall (specific_15: exactly this, submitted 2, gold 1).
    if len(picks) > 1:
        print(f"  {len(picks) - 1} exact-title duplicate record(s) attached")

    print(f"  picks: {[(_cid(p), (p.get('title') or '')[:60]) for p in picks[:MAX_SUBMIT_SPECIFIC]]}")
    _emit_ids(state, picks[:MAX_SUBMIT_SPECIFIC])


# ---------------------------------------------------------------------------
# metadata_f1 — parse into a filter plan, execute against the graph
# ---------------------------------------------------------------------------
_VENUE_SPLIT = re.compile(r"[^a-z0-9]+")


def _venue_norm(text: str) -> str:
    return " " + " ".join(_VENUE_SPLIT.split((text or "").lower())).strip() + " "


def _venue_matches(venue: str, aliases: list[str]) -> bool:
    """Deterministic venue matching against LLM-expanded aliases.

    Iteration 5's index-based LLM verifier misfired badly (it kept 'Autoimmunity'
    and 'Current Biology' for a Nature-portfolio request, and EMNLP/ICML for a
    SPLASH request), zeroing metadata_4 and metadata_33. Semantic Scholar stores
    long official names ('North American Chapter of the Association for
    Computational Linguistics', 'ACM SIGPLAN International Conference on Systems,
    Programming, Languages and Applications: Software for Humanity'), so the
    planner is asked for those full names and matching is a plain containment
    test — no cheap-model indexing step to get wrong.
    """
    vnorm = _venue_norm(venue)
    if vnorm.strip() == "":
        return False
    for alias in aliases:
        anorm = _venue_norm(alias)
        core = anorm.strip()
        if len(core) < 3:
            continue
        if len(core) <= 7 and " " not in core:
            # Short acronym: require a whole-token hit, never a substring.
            if anorm in vnorm:
                return True
            continue
        if anorm in vnorm or vnorm in anorm:
            return True
        # A family prefix ('Nature') should cover 'Nature Methods' etc.
        if vnorm.startswith(anorm.rstrip() + " ") or vnorm.strip().startswith(core + " "):
            return True
    return False


async def _verify_venues(pool_venues: list[str], wanted: list[str], query: str) -> set[str]:
    """LLM fallback used only when deterministic alias matching keeps nothing.

    Echoes the venue strings back verbatim rather than selecting indices — the
    index route is what iteration 5 got wrong.
    """
    if not pool_venues or not wanted:
        return set()
    shown = pool_venues[:120]
    listing = "\n".join(f"- {v[:90]}" for v in shown)
    ask = (
        f'Literature search request: "{query}"\n'
        f"It restricts results to venues described as: {', '.join(wanted)}\n\n"
        "Venue strings present in the retrieved candidates:\n"
        f"{listing}\n\n"
        "List the venue strings that satisfy the restriction. Copy each one back "
        "EXACTLY as written above. Be inclusive about the same publisher family "
        "(a 'Nature portfolio' restriction covers Nature, Nature Methods, Nature "
        "Communications, Scientific Reports) but exclude unrelated venues. If "
        "none qualify, return an empty list.\n"
        'Reply with JSON only: {"venues": ["...", ...]}'
    )
    got = _json_from(await _llm(GPT_5_4_MINI, ask, "metadata/venues")) or {}
    by_norm = {_venue_norm(v): v for v in shown}
    out = set()
    for name in got.get("venues") or []:
        if not isinstance(name, str):
            continue
        exact = by_norm.get(_venue_norm(name))
        if exact:
            out.add(exact)
            continue
        for v in shown:  # tolerate light paraphrase / truncation
            if _venue_norm(name).strip() and _venue_norm(name).strip() in _venue_norm(v):
                out.add(v)
    print(f"  llm venue fallback kept {len(out)}/{len(shown)}: {sorted(out)[:6]}")
    return out


_CONF_WORDS = ("conference", "symposium", "workshop", "proceedings",
               "meeting", "congress", "colloquium")


def _looks_like_journal(paper: dict) -> bool:
    venue = (paper.get("venue") or "").lower()
    journal = paper.get("journal")
    name = ""
    if isinstance(journal, dict):
        name = (journal.get("name") or "").lower()
    if any(w in venue for w in _CONF_WORDS):
        return False
    if any(w in name for w in _CONF_WORDS):
        return False
    return bool(venue or name)


async def _author_ids(author_search, name: str) -> list[tuple[str, str, int]]:
    """Resolve a person's name to the author ids worth pulling papers from.

    The same person routinely has several fragmentary ids; the richest is the
    real one but the fragments carry real papers too, so take the top few.
    """
    found = _parse_items(await _safe(
        author_search(name=name, fields="name,paperCount", limit=10), f"author:{name}"))
    want = {t for t in _venue_norm(name).split() if len(t) > 2}
    scored = []
    for rec in found:
        aid = rec.get("authorId")
        if not aid:
            continue
        got = set(_venue_norm(rec.get("name") or "").split())
        # Surname must line up; initials ('D. Harel') drop the given name.
        if want and not (want & got):
            continue
        scored.append((str(aid), rec.get("name") or "", rec.get("paperCount") or 0))
    scored.sort(key=lambda r: -r[2])
    return scored[:3]


async def _papers_of_authors(author_papers, ids: list[tuple[str, str, int]],
                             pool: dict) -> set[str]:
    group: set[str] = set()
    raws = await asyncio.gather(*[
        _safe(author_papers(
            author_id=aid,
            paper_fields="title,abstract,corpusId,year,venue,authors,citationCount,journal",
            limit=500), f"author_papers:{aid}")
        for aid, _, _ in ids
    ])
    for (aid, nm, cnt), raw in zip(ids, raws):
        items = _parse_items(raw)
        print(f"    author {nm} id={aid} declared={cnt} pulled={len(items)}")
        for paper in items:
            if _cid(paper):
                group.add(_cid(paper))
                pool.setdefault(_cid(paper), paper)
    return group


async def _citing_set(citations, get_paper, batch, seed_id: str, pool: dict) -> set[str]:
    """Everything known to cite `seed_id`.

    `get_citations` is the one tool the snapshot cutoff does not cover and it
    returns newest-first with a hard 1000 cap, so on a heavily-cited seed the
    whole window can sit *after* the snapshot — metadata_26 intersected two such
    windows and produced ten ids all newer than every gold id. The nested
    `citations` field on `get_paper` IS snapshot-filtered, so it is unioned in as
    a second, in-snapshot view of the same edge set.
    """
    group: set[str] = set()
    raw = await _safe(citations(
        paper_id=f"CorpusId:{seed_id}",
        fields="title,abstract,corpusId,year,venue,authors,citationCount,journal",
        limit=1000), f"citations:{seed_id}")
    for item in _parse_items(raw):
        paper = item.get("citingPaper") if "citingPaper" in item else item
        if isinstance(paper, dict) and _cid(paper):
            group.add(_cid(paper))
            pool.setdefault(_cid(paper), paper)
    nested = 0
    if get_paper and batch:
        raw = await _safe(get_paper(paper_id=f"CorpusId:{seed_id}", fields="citations"),
                          f"nested_cites:{seed_id}")
        pids = []
        for doc in _parse_items(raw):
            for ref in doc.get("citations") or []:
                if isinstance(ref, dict) and ref.get("paperId"):
                    pids.append(ref["paperId"])
        pids = pids[:400]
        chunks = [pids[i:i + 100] for i in range(0, len(pids), 100)]
        raws = await asyncio.gather(*[
            _safe(batch(ids=c,
                        fields="title,abstract,corpusId,year,venue,authors,citationCount,journal"),
                  f"batch_cites:{seed_id}")
            for c in chunks
        ])
        for r in raws:
            for paper in _parse_items(r):
                if isinstance(paper, dict) and _cid(paper):
                    if _cid(paper) not in group:
                        nested += 1
                    group.add(_cid(paper))
                    pool.setdefault(_cid(paper), paper)
    print(f"    citation set for {seed_id}: {len(group)} (+{nested} from the "
          f"snapshot-filtered nested field)")
    return group


async def _citers_of(citations, seeds: list[dict]) -> set[str]:
    """Corpus ids of every paper citing any of `seeds`.

    Used for the "must cite something by author X" constraint. The forward
    direction is the only one the server supports (the `references` field errors
    out on both get_paper and get_paper_batch), and the cutoff gap in
    `get_citations` is harmless here: the result is intersected with a pool that
    was already snapshot-filtered, so post-snapshot citers cannot enter.
    """
    out: set[str] = set()
    raws = await asyncio.gather(*[
        _safe(citations(paper_id=p["paperId"], fields="corpusId,year", limit=1000),
              f"citers:{(p.get('title') or '')[:24]}")
        for p in seeds])
    for raw in raws:
        for item in _parse_items(raw):
            paper = item.get("citingPaper") if "citingPaper" in item else item
            if isinstance(paper, dict) and _cid(paper):
                out.add(_cid(paper))
    return out


async def solve_metadata(state: TaskState, query: str) -> None:
    """Structured-constraint retrieval.

    Metadata queries are rule execution, not search: every one of them names
    graph constraints (authorship, citation edges, venue, year, citation count)
    that the corpus tools answer exactly. Four iterations scored ~0.02 here by
    routing them through keyword search plus a fuzzy LLM filter. This solver
    parses the constraints, resolves each through the tool that owns it, and
    intersects deterministically.
    """
    title_search = _get_tool(state, "search_paper_by_title")
    kw_search = _get_tool(state, "search_papers_by_relevance")
    citations = _get_tool(state, "get_citations")
    author_search = _get_tool(state, "search_authors_by_name")
    author_papers = _get_tool(state, "get_author_papers")
    get_paper = _get_tool(state, "get_paper")
    batch = _get_tool(state, "get_paper_batch")

    ask = (
        "Parse this literature-search request into a structured filter plan.\n\n"
        f'Request: "{query}"\n\n'
        "Reply with JSON only, omitting or nulling anything not requested:\n"
        "{\n"
        '  "cites_titles": ["exact published title of each paper the results must CITE"],\n'
        '  "coauthor_of_paper": "exact published title of a paper whose AUTHORS the '
        'results must be written by (null unless the request says so)",\n'
        '  "authors": ["full author names every result must be written by"],\n'
        '  "cites_author": "author name whose papers the results must cite, or null",\n'
        '  "exclude_authors": ["author names that must NOT appear on a result"],\n'
        '  "venue_aliases": ["FULL official venue names, plus the acronym"],\n'
        '  "years_exact": [explicit individual years the request lists],\n'
        '  "year_min": null, "year_max": null,\n'
        '  "min_citations": null, "min_authors": null, "journal_only": false,\n'
        '  "topic_keywords": "short keyword phrase for the topic, or empty"\n'
        "}\n\n"
        "Rules:\n"
        "- cites_titles / coauthor_of_paper: use the real published title. 'the T5 "
        "paper' -> 'Exploring the Limits of Transfer Learning with a Unified "
        "Text-to-Text Transformer'; 'the BERT paper' -> 'BERT: Pre-training of Deep "
        "Bidirectional Transformers for Language Understanding'.\n"
        "- venue_aliases must contain the FULL name Semantic Scholar stores, not "
        "just the acronym: NAACL -> 'North American Chapter of the Association for "
        "Computational Linguistics'; ACL -> 'Annual Meeting of the Association for "
        "Computational Linguistics'; EMNLP -> 'Conference on Empirical Methods in "
        "Natural Language Processing'; SPLASH -> 'Systems, Programming, Languages "
        "and Applications: Software for Humanity'. For a publisher family give the "
        "shared prefix and the members: 'Nature portfolio' -> ['Nature', 'Nature "
        "Methods', 'Nature Communications', 'Nature Biotechnology', 'Scientific "
        "Reports', 'npj'].\n"
        "- years_exact: '2010 or 2012' -> [2010, 2012] (NOT a range). Use year_min/"
        "year_max only for open-ended phrasing: 'after 2022' -> year_min 2023, "
        "'2019 and beyond' -> year_min 2019.\n"
        "- min_authors: 'and at least one additional author' -> 2; 'single-author' "
        "-> null with min_authors 1 and year_max unchanged.\n"
        "- 'more than 50 citations' -> min_citations 51.\n"
        "- 'journal articles' / 'journal papers' -> journal_only true."
    )
    plan = _json_from(await _llm(GPT_5_4, ask, "metadata/plan")) or {}

    def _strs(key, cap):
        return [s.strip() for s in (plan.get(key) or [])
                if isinstance(s, str) and s.strip()][:cap]

    def _int(key):
        try:
            return int(plan.get(key))
        except (TypeError, ValueError):
            return None

    cites = _strs("cites_titles", 3)
    authors = _strs("authors", 3)
    excluded = _strs("exclude_authors", 3)
    venues = _strs("venue_aliases", 12)
    coauthor_of = plan.get("coauthor_of_paper")
    coauthor_of = coauthor_of.strip() if isinstance(coauthor_of, str) else ""
    cites_author = plan.get("cites_author")
    cites_author = cites_author.strip() if isinstance(cites_author, str) else ""
    topic = (plan.get("topic_keywords") or "").strip()
    journal_only = bool(plan.get("journal_only"))
    years_exact = set()
    for y in plan.get("years_exact") or []:
        try:
            years_exact.add(int(y))
        except (TypeError, ValueError):
            pass
    year_min, year_max = _int("year_min"), _int("year_max")
    min_cites, min_authors = _int("min_citations"), _int("min_authors")
    print(f"  plan: cites={cites} coauthor_of={coauthor_of[:40]!r} authors={authors} "
          f"cites_author={cites_author!r} venues={venues} years={sorted(years_exact)} "
          f"[{year_min},{year_max}] min_cites={min_cites} min_authors={min_authors} "
          f"journal_only={journal_only} exclude={excluded} topic={topic!r}")

    pool: dict[str, dict] = {}
    intersect_sets: list[set[str]] = []
    structural = False

    async def _resolve(title: str) -> str | None:
        raw = await _safe(title_search(title=title, fields="title,corpusId,year"),
                          f"resolve:{title[:40]}")
        for paper in _parse_items(raw):
            if paper.get("paperId") and _cid(paper):
                print(f"    resolved '{title[:50]}' -> {_cid(paper)} "
                      f"{(paper.get('title') or '')[:60]}")
                return _cid(paper)
        return None

    # --- results must CITE these papers -----------------------------------
    if cites and title_search and citations:
        seeds = [s for s in await asyncio.gather(*[_resolve(t) for t in cites]) if s]
        groups = await asyncio.gather(*[
            _citing_set(citations, get_paper, batch, s, pool) for s in seeds])
        for group in groups:
            if group:
                intersect_sets.append(group)
                structural = True

    # --- results must be co-authored by an author of a named paper --------
    if coauthor_of and title_search and get_paper and author_search and author_papers:
        seed = await _resolve(coauthor_of)
        names = []
        if seed:
            raw = await _safe(get_paper(paper_id=f"CorpusId:{seed}",
                                        fields="title,authors"), f"seedauthors:{seed}")
            for doc in _parse_items(raw):
                names = _author_names(doc)[:8]
        print(f"    seed-paper authors: {names}")
        ids: list[tuple[str, str, int]] = []
        for name in names:
            ids.extend(await _author_ids(author_search, name))
        if ids:
            group = await _papers_of_authors(author_papers, ids[:12], pool)
            if group:
                intersect_sets.append(group)
                structural = True

    # --- results must be BY these authors (AND across distinct names) -----
    if authors and author_search and author_papers:
        for name in authors:
            ids = await _author_ids(author_search, name)
            if not ids:
                continue
            group = await _papers_of_authors(author_papers, ids, pool)
            if group:
                intersect_sets.append(group)
                structural = True

    # --- results must cite something written by a given author ------------
    cites_author_citers: set[str] | None = None
    if cites_author and author_search and author_papers and citations:
        ids = await _author_ids(author_search, cites_author)
        scratch: dict[str, dict] = {}
        await _papers_of_authors(author_papers, ids, scratch)
        seeds = sorted(scratch.values(), key=lambda p: -(p.get("citationCount") or 0))
        seeds = [p for p in seeds if p.get("paperId")][:CITED_BY_FANOUT]
        print(f"    cites_author '{cites_author}': {len(scratch)} papers, "
              f"expanding {len(seeds)} through get_citations")
        cites_author_citers = await _citers_of(citations, seeds)
        print(f"    ... {len(cites_author_citers)} distinct citing papers")
        if not cites_author_citers:
            cites_author_citers = None

    # --- no graph anchor: keyword search, venue-filtered server-side -------
    if not pool and kw_search:
        venue_arg = ",".join(venues[:5]) if venues else None
        for kw in [k for k in (topic, query) if k][:2]:
            kwargs = {"keyword": kw, "fields": PAPER_FIELDS + ",journal",
                      "limit": SEARCH_LIMIT}
            if venue_arg:
                kwargs["venues"] = venue_arg
            for paper in _parse_items(await _safe(kw_search(**kwargs), f"kw:{kw[:40]}")):
                if _cid(paper):
                    pool.setdefault(_cid(paper), paper)
            if pool:
                break
        if not pool and venue_arg:
            for paper in _parse_items(await _safe(
                    kw_search(keyword=topic or query, fields=PAPER_FIELDS + ",journal",
                              limit=SEARCH_LIMIT), "kw:novenue")):
                if _cid(paper):
                    pool.setdefault(_cid(paper), paper)

    ids_set = set(pool)
    for group in intersect_sets:
        ids_set &= group
    print(f"  pool={len(pool)} after intersection={len(ids_set)}")

    # --- deterministic filters --------------------------------------------
    def _passes(paper: dict, use_venue: bool) -> bool:
        year = _year_of(paper)
        if years_exact and year not in years_exact:
            return False
        if year_min is not None and (year is None or year < year_min):
            return False
        if year_max is not None and (year is None or year > year_max):
            return False
        if min_cites is not None:
            count = paper.get("citationCount")
            if not isinstance(count, int) or count < min_cites:
                return False
        if min_authors is not None and len(_author_names(paper)) < min_authors:
            return False
        if journal_only and not _looks_like_journal(paper):
            return False
        if excluded:
            have = {_venue_norm(a).strip() for a in _author_names(paper)}
            for bad in excluded:
                b = _venue_norm(bad).strip()
                if any(b == h or (b and h and (b in h or h in b)) for h in have):
                    return False
        if use_venue and venues and not _venue_matches(paper.get("venue") or "", venues):
            return False
        return True

    candidates = [pool[c] for c in ids_set]
    kept = [p for p in candidates if _passes(p, True)]
    print(f"  after filters (venue deterministic): {len(kept)}")

    if venues and not kept and candidates:
        # Alias expansion missed the stored spelling — ask the model, but only
        # about the venue strings that actually survived the other filters.
        loose = [p for p in candidates if _passes(p, False)]
        distinct = sorted({(p.get("venue") or "").strip() for p in loose
                           if (p.get("venue") or "").strip()})
        ok = await _verify_venues(distinct, venues, query) if distinct else set()
        if ok:
            kept = [p for p in loose if (p.get("venue") or "").strip() in ok]
        print(f"  after llm venue fallback: {len(kept)}")

    # --- must cite a paper by `cites_author` -------------------------------
    # Read the edge forward (who cites the author's papers) rather than backward:
    # the `references` field is rejected by both get_paper and get_paper_batch on
    # this server, which is what silently voided this filter on iteration 8's
    # metadata_31 and left 114 unfiltered ids on the submission.
    if cites_author_citers and kept:
        narrowed = [p for p in kept if _cid(p) in cites_author_citers]
        print(f"  cites_author citing-set check kept {len(narrowed)}/{len(kept)}")
        if narrowed:
            kept = narrowed

    if not kept:
        # Every filter wiped the set. Iteration 5 dumped the whole unfiltered
        # pool here (60 ids, score 0.000); a small best-effort head is strictly
        # better because F1 = 2h/(n+G) punishes n.
        relaxed = [p for p in candidates if _passes(p, False)] or candidates \
            or list(pool.values())
        relaxed.sort(key=lambda p: -(p.get("citationCount") or 0))
        kept = relaxed[:20]
        structural = False
        print(f"  filters emptied the set; best-effort fallback {len(kept)}")

    # A topical constraint on top of structural filters needs semantic judgement.
    if topic and len(kept) > 25:
        crit = [{"name": "topic", "description": topic, "probe": topic}]
        subset = kept[:200]
        items = [(_cid(p), (p.get("title") or "")[:150],
                  re.sub(r"\s+", " ", _tldr_text(p) or (p.get("abstract") or ""))[:400])
                 for p in subset if _cid(p)]
        scored = await _judge_evidence(items, query, crit, [1.0], GPT_5_4_MINI,
                                       "meta-topic")
        kept = sorted(subset, key=lambda p: -scored.get(_cid(p), -1.0)) + kept[200:]
    else:
        kept.sort(key=lambda p: -(p.get("citationCount") or 0))

    # F1 = 2h/(n+G). When the structural filters are genuinely selective their
    # whole output is the answer; otherwise cap, because unfiltered padding is
    # what has been scoring 0.00 on this path since iteration 2.
    # Truncating a homogeneous survivor set is strictly harmful: adding one more
    # candidate raises F1 whenever its hit probability exceeds h/(n+G) = F1/2, and
    # inside a set that already passed the same filters every survivor's rate is
    # about h/n > h/(n+G). Iteration 8 cut 114 survivors to 100 on metadata_31 and
    # threw away 10 of the 16 gold ids for it (0.103 where 114 ids scored 0.246).
    if structural:
        cap = min(len(kept), MAX_SUBMIT_METADATA)
    else:
        cap = min(len(kept), 20)
    if not kept:
        # F1 = 2h/(n+G) is 0 when n = 0, so an empty submission is strictly worse
        # than any best-effort head: a wrong id costs the same as no id at all,
        # and a right one is free upside. Iteration 9's metadata_33 submitted
        # nothing on all three agents.
        kept = sorted(pool.values(),
                      key=lambda p: -(p.get("citationCount") or 0))[:20]
        cap = len(kept)
        print(f"  empty survivor set; unfiltered head fallback {len(kept)}")
    print(f"  submitting {min(cap, len(kept))} of {len(kept)} (structural={structural})")
    _emit_ids(state, kept[:cap])


# ---------------------------------------------------------------------------
# Grading — per-criterion digits, so a weak criterion is identifiable
# ---------------------------------------------------------------------------

def _weights(criteria: list[dict]) -> list[float]:
    """Mirror the gold weighting shape (observed: 0.4/0.3/0.3 and 0.4/0.4/0.2 —
    the first, topical criterion carries the most mass). Uses the planner's own
    weights when they look sane, else that default."""
    raw = []
    for c in criteria:
        try:
            w = float(c.get("weight"))
        except (TypeError, ValueError):
            w = 0.0
        raw.append(max(0.0, w))
    total = sum(raw)
    n = len(criteria)
    if n and total > 0.2 and all(w > 0 for w in raw):
        return [w / total for w in raw]
    if n == 1:
        return [1.0]
    first = 0.4
    rest = (1.0 - first) / (n - 1)
    return [first] + [rest] * (n - 1)


def _judge_prompt(query: str, criteria: list[dict],
                  batch: list[tuple[int, str, str]]) -> str:
    """Grade the *evidence text* - the only thing the benchmark judge will see.

    The benchmark judge emits a 3-valued verdict per criterion, but this
    predictor is used only for ORDERING, and a 3-valued scale ties enormous
    blocks of candidates together (iteration 10: 119 of 250 papers tied at the
    top on semantic_91, so order inside that block fell back to retrieval order
    and `rank` came in at 0.534, the worst in the batch). A 0-9 scale carries
    the same information at ~1000x the resolution for the same one digit per
    criterion, and the geometric aggregation in `_weighted` turns it into a
    conjunctive estimate of P(grade 3).
    """
    lines = [f"[{idx}] {title} :: {ev}" for idx, title, ev in batch]
    crit = "\n".join(f"  C{i + 1} ({c['name']}): {c['description']}"
                     for i, c in enumerate(criteria))
    n = len(criteria)
    return (
        "You are the relevance judge for a literature search. For each candidate "
        "you see ONLY the quoted passages below - judge from that text alone, "
        "never from outside knowledge about the paper.\n\n"
        f"REQUEST: {query}\n\n"
        f"REQUIREMENTS:\n{crit}\n\n"
        "CANDIDATES:\n" + "\n".join(lines) + "\n\n"
        f"For each candidate output one line `index:DDD` with exactly {n} digits "
        "(0-9), one digit per requirement in order, rating how well the quoted "
        "text supports that requirement:\n"
        "  9 = a sentence in the quoted text explicitly states or demonstrates it\n"
        "  7-8 = the text clearly implies it without stating it outright\n"
        "  4-6 = partial: right area, but the specific requirement is not shown\n"
        "  1-3 = only a loose topical association\n"
        "  0 = the text gives no support at all\n"
        f"Example for {n} requirements: `7:{'9' * (n - 1)}4`\n"
        "Use the full 0-9 range and DISCRIMINATE: candidates that look equally "
        "on-topic almost always differ in how directly the text shows each "
        "requirement, and your job is to separate them. Reserve 9 for a "
        "requirement you could quote a sentence for. If a requirement asks the "
        "paper to CONNECT, COMPARE or RELATE two things, mentioning each "
        "separately is at most 5. If a requirement restricts the kind of paper "
        "(original research not a survey / common not niche / large-scale / a "
        "particular modality or setting), score it high only when the text shows "
        "that property. Output nothing but those lines, one per candidate, for "
        "every candidate."
    )


def _parse_judge(text: str, valid: set[int], n_crit: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for line in (text or "").splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s*[:=\-]\s*(\d+)", line.strip())
        if not m:
            continue
        idx, digits = int(m.group(1)), m.group(2)
        if idx in valid and idx not in out:
            vals = [int(d) for d in digits][:n_crit]
            vals += [0] * (n_crit - len(vals))
            out[idx] = vals
    return out


def _weighted(vals: list[int], weights: list[float]) -> float:
    """Predicted P(the benchmark judge grades this paper 3), in [0, 1].

    The scorer computes `weighted = min(1, sum(w_c * r_c / 3))` and needs
    `weighted > 0.99` for grade 3 - the only grade that earns recall. That gate
    is CONJUNCTIVE, but the linear sum is not: it saturates at exactly 1.0 for
    every paper whose criteria are all top-marked (iteration 10 tied 119 papers
    there on one query), and it lets a heavy criterion buy off a missing light
    one even though the gate would not. A weighted geometric mean is the
    conjunctive form of the same quantity - monotone in every criterion, never
    saturating, and one weak criterion drags the product down the way the gate
    does.
    """
    acc = 0.0
    tw = 0.0
    for w, v in zip(weights, vals):
        if w <= 0:
            continue
        acc += w * math.log((max(0, min(9, int(v))) + 0.5) / 9.5)
        tw += w
    if tw <= 0:
        return 0.0
    return math.exp(acc / tw)


def _mean_vote(v) -> float:
    """Reliability-weighted mean of the graders that scored this paper. Papers no
    grader could parse fall to -1.0 so they sort below every graded paper without
    being dropped - on `semantic_f1` there is no precision term, so a tail slot
    costs nothing and a withheld paper can only lose."""
    if not v:
        return -1.0
    tw = sum(w for w, _ in v)
    return sum(w * x for w, x in v) / tw if tw > 0 else -1.0


async def _judge_evidence(items: list[tuple[str, str, str]], query: str,
                          criteria: list[dict], weights: list[float], handle,
                          label: str) -> dict[str, float]:
    """items: (cid, title, evidence_excerpt). Returns cid -> predicted weighted
    relevance in [0, 1], for whichever papers came back parseable."""
    if not items:
        return {}
    n_crit = max(1, len(criteria))
    indexed = [(i, t, e) for i, (_, t, e) in enumerate(items)]
    batches = [indexed[i:i + GRADE_BATCH] for i in range(0, len(indexed), GRADE_BATCH)]
    texts = await asyncio.gather(*[
        _llm(handle, _judge_prompt(query, criteria, b), f"{label}[{i}]")
        for i, b in enumerate(batches)
    ])
    out: dict[str, float] = {}
    n_full = 0
    for text, batch in zip(texts, batches):
        for idx, vals in _parse_judge(text, {i for i, _, _ in batch}, n_crit).items():
            w = _weighted(vals, weights)
            out[items[idx][0]] = w
            if w >= 0.90:
                n_full += 1
    # Calibration line: compare this against the judge's actual grade-3 count in
    # score_meta.json. Iteration 10's single grader ran 4x high on one query and
    # 0-for-26 on another; a wide gap in either direction means the ordering
    # signal, not retrieval, is the thing to fix.
    print(f"  {label}: {len(out)}/{len(items)} graded; predicted-strong(>=.90)={n_full}")
    return out


# ---------------------------------------------------------------------------
# semantic_f1
# ---------------------------------------------------------------------------

async def solve_semantic(state: TaskState, query: str) -> None:
    started = time.monotonic()
    kw_search = _get_tool(state, "search_papers_by_relevance")
    snippet_search = _get_tool(state, "snippet_search")
    batch_tool = _get_tool(state, "get_paper_batch")

    # --- 1. query understanding -------------------------------------------
    # The gold `relevance_criteria` are an LLM decomposition of the query into
    # must-satisfy requirements, so mirroring that generation is the target.
    ask = (
        "You are preparing a search over a scientific-paper index.\n\n"
        f'REQUEST: "{query}"\n\n'
        "The keyword index returns ZERO results for question-shaped input, so "
        "every search query must be a bare noun phrase (no 'what', 'how', 'could "
        "you', no question mark).\n\n"
        "Reply with JSON only:\n"
        "{\n"
        f'  "queries": [{N_VARIANTS} noun-phrase search queries, 4-9 words each; '
        "the first the most literal],\n"
        '  "criteria": [\n'
        '    {"name": "short label",\n'
        '     "description": "one sentence stating a requirement a paper MUST '
        'satisfy to fully answer the request",\n'
        '     "probe": "3-6 word phrase, using the words a paper would actually '
        'use, for finding a passage that demonstrates THIS requirement",\n'
        '     "weight": 0.4}\n'
        "  ],\n"
        '  "conjunction_probes": [2 sentence-shaped phrasings of the request that '
        "state ALL of its requirements together, for a passage-retrieval engine "
        "that accepts natural language],\n"
        '  "year_min": null or an integer if the request restricts publication year,\n'
        '  "prefer_earliest": true only if the request asks for the earliest / '
        "first / original such work\n"
        "}\n\n"
        "RULES FOR `queries` -- this matters more than anything else. A paper only "
        "counts if it satisfies EVERY requirement of the request at once, so every "
        "query must express the WHOLE request, not one of its parts. Vary the "
        "wording, the synonyms and the field vocabulary between queries; do NOT "
        "split the request into single-aspect queries (a query naming only one "
        "requirement returns hundreds of papers that satisfy only that one and are "
        "worthless). If the request relates two things (X compared with Y, X used "
        "for Y, X based on Y), keep BOTH in every query. Include at least two "
        "queries that foreground the hardest / most specific requirement together "
        "with the main topic, and -- if the request asks for a general, common, "
        "widely-used, comparative or overview treatment -- at least two queries "
        "using survey / overview / comparison / taxonomy vocabulary.\n\n"
        "RULES FOR `criteria`. Give 2-4 criteria that a strict grader could check "
        "line by line against a paper's own text; a paper scores only if all of "
        "them hold. Cover: (a) the main topic or object of study; (b) the "
        "method, task or setting the request names; (c) if the request relates "
        "two things, a final criterion that the paper must EXPLICITLY connect or "
        "compare them (not merely mention each); (d) if the request excludes "
        "something (e.g. 'exclude survey papers') or restricts to a class "
        "(common / widely-used / earliest / large-scale), a criterion stating "
        "that restriction. Weights sum to 1.0, main topic highest -- typical "
        "splits are 0.4/0.4/0.2, 0.5/0.3/0.2 and 0.4/0.3/0.3."
    )
    plan = _json_from(await _llm(GPT_5_4_MINI, ask, "semantic/plan")) or {}
    queries = [q for q in (plan.get("queries") or []) if isinstance(q, str) and q.strip()]

    criteria: list[dict] = []
    for c in plan.get("criteria") or []:
        if isinstance(c, dict) and (c.get("name") or c.get("description")):
            name = str(c.get("name") or "")[:80]
            desc = str(c.get("description") or name)[:300]
            probe = str(c.get("probe") or name or desc)[:100]
            criteria.append({"name": name or probe, "description": desc,
                             "probe": probe, "weight": c.get("weight")})
        elif isinstance(c, str) and c.strip():
            criteria.append({"name": c[:80], "description": c[:300],
                             "probe": c[:100], "weight": None})
    criteria = criteria[:4]
    if not criteria:
        criteria = [{"name": "request", "description": query,
                     "probe": query[:100], "weight": 1.0}]
    weights = _weights(criteria)

    conj = [str(p).strip() for p in (plan.get("conjunction_probes") or [])
            if isinstance(p, str) and p.strip()][:2]
    if not conj:
        conj = [query]

    try:
        year_min = int(plan.get("year_min"))
    except (TypeError, ValueError):
        year_min = None
    prefer_earliest = bool(plan.get("prefer_earliest"))
    if not queries:
        stripped = re.sub(
            r"^(what|which|how|why|who|where|are|is|can|could|do|does|show|find|"
            r"give|recommend|suggest|i am looking for|i'm looking for)\b[^,?.]*[,?.]?\s*",
            "", query.strip(), flags=re.I)
        queries = [stripped.strip(" ?.") or query]
    queries = queries[:N_VARIANTS]

    # A second, differently-framed planner. Every keyword call is free and the
    # pool is the ceiling on recall (grade-3 count inside K, K up to 232), so the
    # binding limit on breadth is how many DISTINCT phrasings we can put to a
    # lexical index. One prompt drawn once yields near-duplicates; asking a fresh
    # call for other vocabulary registers - the authors' own title wording, the
    # survey wording, the applied wording - buys genuinely different result sets
    # for ~$0.002.
    if criteria:
        expand = (
            "You are widening a search over a scientific-paper index.\n\n"
            f'REQUEST: "{query}"\n\n'
            "Requirements a matching paper must satisfy AT ONCE:\n"
            + "\n".join(f"- {c['name']}: {c['description']}" for c in criteria)
            + "\n\nAlready issued:\n" + "\n".join(f"- {q}" for q in queries[:8])
            + "\n\nReply with JSON only: "
            f'{{"queries": [{N_EXTRA_VARIANTS} more search queries]}}\n'
            "Each must be a bare noun phrase, 4-9 words, no question words and no "
            "question mark (the index returns zero results for question-shaped "
            "input), and each must still express the WHOLE request - never a "
            "single requirement on its own. Make them lexically DIFFERENT from the "
            "already-issued ones and from each other: use the wording the paper's "
            "own authors would put in a title, the wording a survey of this area "
            "would use, the applied/downstream-task wording, the "
            "method-name/technical-jargon wording, and the wording of the "
            "sub-community that cares most about the hardest requirement."
        )
        more = _json_from(await _llm(GPT_5_4_MINI, expand, "semantic/expand")) or {}
        have = {q.lower().strip() for q in queries}
        for q in more.get("queries") or []:
            if isinstance(q, str) and q.strip() and q.lower().strip() not in have:
                have.add(q.lower().strip())
                queries.append(q.strip())
                if len(queries) >= N_VARIANTS + N_EXTRA_VARIANTS:
                    break

    crit_terms = [_terms(c["probe"]) | _terms(c["name"]) for c in criteria]
    print(f"  queries={queries}")
    for c, w in zip(criteria, weights):
        print(f"  criterion(w={w:.2f}): {c['name']!r} probe={c['probe']!r}")
    print(f"  year_min={year_min} prefer_earliest={prefer_earliest}")

    # --- 2. wide, concurrent retrieval ------------------------------------
    # Breadth is the binding constraint on the large-K queries (K observed up to
    # 228; recall = grade-3 count / K), and every retrieval call is free.
    tasks = [_safe(kw_search(keyword=q, fields=PAPER_FIELDS, limit=SEARCH_LIMIT),
                   f"kw:{q[:40]}") for q in queries] if kw_search else []
    n_kw = len(tasks)
    open_snips = []
    if snippet_search:
        # Passage retrieval tolerates natural-language input and finds papers
        # whose abstract never states the query's vocabulary. Probing each
        # criterion unscoped discovers papers the keyword runs never surface and
        # returns a criterion-tagged passage for them in the same call.
        open_snips.append((-1, query))
        # Conjunction phrasings: snippet_search tolerates sentence-shaped input and
        # ranks passages, so a phrasing naming EVERY requirement at once surfaces
        # the papers that actually satisfy the conjunction -- the only papers that
        # can reach grade 3 (weighted > 0.99 needs every criterion Perfect).
        for text in conj:
            open_snips.append((-1, text))
        for ci, c in enumerate(criteria[:OPEN_SNIPPET_CRIT]):
            open_snips.append((ci, c["probe"]))
        tasks += [_safe(snippet_search(query=text, limit=100), f"snippet:open{ci}",
                        timeout=SNIPPET_TIMEOUT) for ci, text in open_snips]
    raws = await asyncio.gather(*tasks) if tasks else []

    pool: dict[str, dict] = {}
    snips: dict[str, list[tuple[int, str]]] = {}
    ranked_runs: list[list[str]] = []

    for raw in raws[:n_kw]:
        run = []
        for paper in _parse_items(raw):
            cid = _cid(paper)
            if not cid:
                continue
            run.append(cid)
            if cid in pool:
                pool[cid].update({k: v for k, v in paper.items() if v})
            else:
                pool[cid] = dict(paper)
        ranked_runs.append(run)
    print(f"  keyword runs -> {[len(r) for r in ranked_runs]}")

    for (ci, _probe), raw in zip(open_snips, raws[n_kw:]):
        run = []
        for entry in _parse_items(raw):
            paper = entry.get("paper") or {}
            cid = _cid(paper)
            if not cid:
                continue
            if cid not in pool:
                pool[cid] = {"corpusId": cid, "title": paper.get("title") or ""}
            if cid not in run:
                run.append(cid)
            text = (entry.get("snippet") or {}).get("text") or ""
            if text:
                bucket = snips.setdefault(cid, [])
                tag = ci if ci >= 0 else 0
                if sum(1 for t, _ in bucket if t == tag) < 2 and len(bucket) < 8:
                    bucket.append((tag, text))
        ranked_runs.append(run)
        print(f"  snippet run c{ci} -> {len(run)} papers")

    print(f"  pool: {len(pool)} unique papers")
    if not pool:
        _emit(state, [])
        return


    # --- 3. reciprocal-rank fusion + free lexical prior -------------------
    n_crit = len(criteria)

    def rank_pool(keep: int) -> list[dict]:
        """RRF over every retrieval run, then the free lexical criterion-coverage
        prior. Both terms are $0, so this can be re-run after each expansion."""
        fused: dict[str, float] = {}
        for run in ranked_runs:
            for rk, cid in enumerate(run):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60.0 + rk)
        cand = sorted(pool.values(), key=lambda p: -fused.get(_cid(p), 0.0))[:keep]
        prior: dict[str, float] = {}
        for pos, paper in enumerate(cand):
            cid = _cid(paper)
            text = " ".join([paper.get("title") or "", _tldr_text(paper),
                             paper.get("abstract") or ""])
            pt = _terms(text)
            cov = sum(1 for ct in crit_terms if _covers(pt, ct)) / max(1, n_crit)
            # Coverage of ALL criteria is what grade 3 requires, so weight the
            # conjunction more heavily than fusion position.
            prior[cid] = 0.45 * (1.0 - pos / max(1, len(cand))) + 0.55 * cov
        cand.sort(key=lambda p: -prior.get(_cid(p), 0.0))
        return cand

    order = rank_pool(MAX_SUBMIT_SEMANTIC + 140)

    # --- 3b. free "more like this" expansion round ------------------------
    # Recall = grade-3 papers inside the first K, with no precision term, so extra
    # candidates are pure upside and every retrieval call is free. The papers that
    # already cover every criterion lexically are the best available description of
    # what a qualifying paper looks like; re-issuing their titles as keyword queries
    # pulls in their topical neighbourhood, which the request's own vocabulary
    # misses whenever the field words differ from the asker's.
    if kw_search and time.monotonic() - started < ENRICH_DEADLINE:
        seeds, seen_t = [], set()
        for paper in order:
            title = re.sub(r"\s+", " ", (paper.get("title") or "")).strip()
            key = _norm_title(title)
            if len(title) < 18 or key in seen_t:
                continue
            seen_t.add(key)
            seeds.append(title[:180])
            if len(seeds) >= N_MORE_LIKE_THIS:
                break
        before = len(pool)
        more = await asyncio.gather(*[
            _safe(kw_search(keyword=t, fields=PAPER_FIELDS, limit=SEARCH_LIMIT),
                  f"mlt:{t[:34]}") for t in seeds]) if seeds else []
        for raw in more:
            run = []
            for paper in _parse_items(raw):
                cid = _cid(paper)
                if not cid:
                    continue
                run.append(cid)
                if cid in pool:
                    pool[cid].update({k: v for k, v in paper.items() if v})
                else:
                    pool[cid] = dict(paper)
            ranked_runs.append(run)
        print(f"  more-like-this: {len(seeds)} seed titles -> "
              f"+{len(pool) - before} new papers (pool {len(pool)})")
        order = rank_pool(MAX_SUBMIT_SEMANTIC + 140)

    # --- 4. hydrate: no submitted paper may reach the judge abstract-less --
    # A snippet-discovered record with no abstract yields ~340 chars of evidence
    # and lands on grade 2. Abstracts are the densest criterion-bearing text.
    thin = [_cid(p) for p in order if _cid(p) and not (p.get("abstract") or "").strip()]
    if thin and batch_tool:
        chunks = [thin[i:i + 100] for i in range(0, len(thin), 100)][:4]
        hyd = await asyncio.gather(*[
            _safe(batch_tool(ids=[f"CorpusId:{c}" for c in chunk],
                             fields="title,abstract,corpusId,tldr,year,venue,citationCount"),
                  f"hydrate[{i}]") for i, chunk in enumerate(chunks)])
        filled = 0
        for raw in hyd:
            for paper in _parse_items(raw):
                cid = _cid(paper)
                if cid in pool:
                    pool[cid].update({k: v for k, v in paper.items() if v})
                    filled += 1
        print(f"  hydrated {filled}/{len(thin)} abstract-less records")

    # --- 5. re-rank now that the hydration pass filled in the abstracts ----
    order = rank_pool(MAX_SUBMIT_SEMANTIC + 140)
    print(f"  pre-ranked {len(order)} candidates "
          f"({time.monotonic() - started:.0f}s elapsed)")

    # --- 6. criterion-targeted evidence retrieval -------------------------
    # THE lever: grade 3 needs EVERY criterion supported, and the judge reads
    # markdown_evidence alone. Probing each criterion separately, scoped to the
    # candidate head, produces passages tagged with the criterion they support.
    if snippet_search and time.monotonic() - started < ENRICH_DEADLINE:
        head_ids = [_cid(p) for p in order[:CRIT_SNIPPET_HEAD] if _cid(p)]
        batches = [head_ids[i:i + CRIT_SNIPPET_BATCH]
                   for i in range(0, len(head_ids), CRIT_SNIPPET_BATCH)]
        calls, tags = [], []
        for ci, c in enumerate(criteria):
            for chunk in batches:
                calls.append(_safe(
                    snippet_search(query=c["probe"], limit=100,
                                   paper_ids=",".join(f"CorpusId:{x}" for x in chunk)),
                    f"snip:c{ci}", timeout=SNIPPET_TIMEOUT))
                tags.append(ci)
        print(f"  criterion snippet harvest: {len(calls)} scoped calls "
              f"over {len(head_ids)} papers")
        got = await asyncio.gather(*calls) if calls else []
        added = 0
        for ci, raw in zip(tags, got):
            for entry in _parse_items(raw):
                cid = _cid(entry.get("paper") or {})
                text = (entry.get("snippet") or {}).get("text") or ""
                if not cid or not text:
                    continue
                bucket = snips.setdefault(cid, [])
                if (sum(1 for t, _ in bucket if t == ci) >= 3
                        or len(bucket) >= MAX_SNIPS_PER_PAPER):
                    continue
                bucket.append((ci, text))
                added += 1
        print(f"  harvested {added} criterion-tagged passages over "
              f"{len(snips)} papers ({time.monotonic() - started:.0f}s elapsed)")

    # --- 6b. gap-driven single-paper criterion repair ---------------------
    # THE arithmetic this pass serves: score = harmonic(rank, recall), and on the
    # measured batch rank sits at 0.73-0.98 while recall sits at 0.05-0.24. With
    # r=0.85, a=0.16 the partials are d/da = 2r^2/(r+a)^2 = 1.41 against
    # d/dr = 2a^2/(r+a)^2 = 0.05 - recall is worth ~28x rank, and recall counts
    # ONLY grade-3 papers. Grade 3 needs weighted > 0.99, i.e. essentially every
    # weighted criterion judged Perfectly Relevant from the evidence alone. So the
    # binding quantity is "papers whose evidence demonstrates EVERY criterion".
    # Iteration 9's own stdout says that quantity was 4 of 250 on semantic_100,
    # against 82 grade-2 papers across the batch that were one criterion short and
    # earned nothing. This pass measures each head paper's uncovered criteria and
    # buys a passage for them one paper at a time.
    if snippet_search and n_crit > 1 and time.monotonic() - started < REPAIR_DEADLINE:
        # Query with the criterion's own surface words: `_covers` tests exactly
        # those stems, so retrieving against them is retrieving against the test.
        probes = [f'{c.get("name") or ""} {c.get("probe") or ""} '
                  f'{(c.get("description") or "")[:170]}'.strip() for c in criteria]

        jobs: list[tuple[int, int, str, int]] = []
        for pos, paper in enumerate(order[:REPAIR_HEAD]):
            cid = _cid(paper)
            if not cid:
                continue
            bucket = sorted(snips.get(cid, []), key=lambda t: t[0])
            _ev, _cv, cvd = _build_evidence(paper, bucket, crit_terms)
            miss = [i for i, ok in enumerate(cvd) if not ok]
            if not miss:
                continue
            # Heaviest uncovered criterion first: a criterion carrying w=0.4 is
            # the difference between grade 2 and grade 1 as well as grade 3.
            miss.sort(key=lambda i: -weights[i])
            for ci in miss[:REPAIR_CRIT_MAX]:
                jobs.append((len(miss), pos, cid, ci))
        # Near-misses first. A paper missing one criterion is one passage from
        # counting; a paper missing all of them is off-topic, not repairable.
        jobs.sort(key=lambda j: (j[0], j[1]))
        jobs = jobs[:REPAIR_MAX_CALLS]

        if jobs:
            gate = asyncio.Semaphore(REPAIR_CONCURRENCY)

            async def _repair(cid: str, ci: int):
                async with gate:
                    if time.monotonic() - started > REPAIR_DEADLINE:
                        return cid, ci, None
                    raw = await _safe(
                        snippet_search(query=probes[ci], limit=REPAIR_LIMIT,
                                       paper_ids=f"CorpusId:{cid}"),
                        f"repair:{cid}:c{ci}", timeout=REPAIR_TIMEOUT)
                    return cid, ci, raw

            print(f"  criterion repair: {len(jobs)} single-paper probes over "
                  f"{len({j[2] for j in jobs})} papers "
                  f"({time.monotonic() - started:.0f}s elapsed)")
            done = await asyncio.gather(*[_repair(c, i) for _, _, c, i in jobs])
            gained = 0
            for cid, ci, raw in done:
                if not raw:
                    continue
                for entry in _parse_items(raw):
                    text = (entry.get("snippet") or {}).get("text") or ""
                    if not text:
                        continue
                    bucket = snips.setdefault(cid, [])
                    if len(bucket) >= MAX_SNIPS_PER_PAPER:
                        break
                    bucket.append((ci, text))
                    gained += 1
            print(f"  repair: +{gained} passages "
                  f"({time.monotonic() - started:.0f}s elapsed)")

    # --- 7. assemble the evidence the judge will read ---------------------
    submit = order[:MAX_SUBMIT_SEMANTIC]
    built: list[tuple[str, str, float, dict]] = []
    for paper in submit:
        cid = _cid(paper)
        if not cid:
            continue
        # Criterion-tagged passages first so uncovered criteria get first refusal.
        bucket = sorted(snips.get(cid, []), key=lambda t: t[0])
        evidence, cov, _cvd = _build_evidence(paper, bucket, crit_terms)
        built.append((cid, evidence, cov, paper))
    full_cov = sum(1 for _, _, cov, _ in built if cov >= 0.999)
    head_cov = sum(1 for _, _, cov, _ in built[:60] if cov >= 0.999)
    mean_cov = sum(cov for _, _, cov, _ in built) / max(1, len(built))
    # This is the number to move next iteration: it upper-bounds grade-3 count,
    # which is the whole of `recall`.
    print(f"  evidence: {full_cov}/{len(built)} papers cover every criterion "
          f"lexically (head60={head_cov}, mean_cov={mean_cov:.2f}) "
          f"({time.monotonic() - started:.0f}s elapsed)")

    # --- 8. grade the assembled evidence on the judge's own scale ---------
    # Ranking by a prediction made from the *same text* the benchmark judge sees
    # aligns the ordering with BOTH score terms at once: `recall` counts grade-3
    # papers inside the first K, and `rank` rewards a grade-descending order.
    # Note that ordering moves recall too - only the first K submissions are ever
    # judged, so a grade-3 paper pushed past K is worth exactly as little as one
    # that was never retrieved.
    gpool = built[:POOL_EVIDENCE_GRADE]
    items = [(cid, (p.get("title") or "")[:150], re.sub(r"\s+", " ", ev)[:EV_CHARS_FOR_GRADE])
             for cid, ev, _cov, p in gpool]
    pool_pred = await _judge_evidence(items, query, criteria, weights,
                                      GPT_5_4_MINI, "grade")

    # Votes are kept per grader rather than folded into a running average, so the
    # final key can prefer a paper TWO graders liked over one a single grader
    # liked: iteration 10's blend could not express that, and its head was full of
    # papers that exactly one lenient pass had waved through.
    votes: dict[str, list[tuple[float, float]]] = {}
    for cid, w in pool_pred.items():
        votes.setdefault(cid, []).append((VOTE_W_POOL, w))

    # --- 9. two independent re-grades over the band the judge actually reads --
    # Observed K (the judge's scored depth) ran 14-180, median 36. DEEP_RERANK_N
    # covers K outright on most queries; HEAD_RERANK_N buys the strongest model
    # for the slots that decide the small-K queries outright.
    if time.monotonic() - started < ENRICH_DEADLINE:
        ranked = sorted(gpool, key=lambda b: (-_mean_vote(votes.get(b[0])), b[0]))
        deep = ranked[:DEEP_RERANK_N]
        head = deep[:HEAD_RERANK_N]
        ditems = [(cid, (p.get("title") or "")[:150],
                   re.sub(r"\s+", " ", ev)[:EV_CHARS_FOR_DEEP])
                  for cid, ev, _cov, p in deep]
        hitems = [(cid, (p.get("title") or "")[:150],
                   re.sub(r"\s+", " ", ev)[:EV_CHARS_FOR_HEAD])
                  for cid, ev, _cov, p in head]
        # A DIFFERENT model family on the deep band. The point is independence,
        # not a better single verdict: two graders agreeing on a paper is the
        # signal that separates a real grade-3 from one lenient pass, and it is
        # what breaks the saturation ties that sank `rank` in iteration 10.
        dpred, hpred = await asyncio.gather(
            _judge_evidence(ditems, query, criteria, weights,
                            CLAUDE_HAIKU_4_5, "deep"),
            _judge_evidence(hitems, query, criteria, weights, GPT_5_4, "head"),
        )
        for cid, w in dpred.items():
            votes.setdefault(cid, []).append((VOTE_W_DEEP, w))
        for cid, w in hpred.items():
            votes.setdefault(cid, []).append((VOTE_W_HEAD, w))

    # --- 10. final ordering ----------------------------------------------
    pre = {cid: i for i, (cid, _, _, _) in enumerate(built)}
    n_built = len(built)

    def sort_key(b):
        cid, _ev, cov, paper = b
        v = votes.get(cid)
        p = _mean_vote(v)
        # Confirmation count is a tiebreaker BELOW the mean, so it can only
        # reorder papers the graders scored the same; it cannot promote a weak
        # paper just for having been looked at more often.
        n_v = len(v) if v else 0
        year = _year_of(paper)
        yk = (year or 9999) if prefer_earliest else 0
        return (-p, -n_v, -cov, yk, pre[cid] / max(1, n_built))

    built.sort(key=sort_key)
    if year_min is not None:
        built.sort(key=lambda b: 0 if (_year_of(b[3]) or 9999) >= year_min else 1)

    strong = sum(1 for cid, _, _, _ in built if _mean_vote(votes.get(cid)) >= 0.90)
    top = [(cid, round(_mean_vote(votes.get(cid)), 2), len(votes.get(cid) or ()))
           for cid, _, _, _ in built[:10]]
    print(f"  ensemble: {strong}/{len(built)} papers at mean-vote >= 0.90")
    print(f"  head predictions: {top}")
    _emit(state, [{"paper_id": cid, "markdown_evidence": ev}
                  for cid, ev, _cov, _p in built])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _infer_score_type(query: str) -> str:
    """Fallback when metadata omits score_type (never observed, but cheap)."""
    q = query.lower()
    if re.search(r"\b(citing|cites|cited by|authored by|published in|between \d{4})\b", q):
        return "metadata_f1"
    if re.search(r"^(the\s+)?[\w\-\s]{0,40}\bpaper\b", q) and len(q.split()) <= 10:
        return "specific_f1"
    return "semantic_f1"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata.get("raw_query") or state.input_text or ""
        score_type = state.metadata.get("score_type") or _infer_score_type(query)
        print(f"[{state.sample_id}] score_type={score_type} query={query[:120]!r}")

        # Whatever happens, emit a parseable payload: an unparseable completion
        # scores 0 even when the retrieval was fine.
        state.output.completion = json.dumps(
            {"output": {"query_id": state.sample_id, "results": []}}
        )
        try:
            if score_type == "specific_f1":
                await solve_specific(state, query)
            elif score_type == "metadata_f1":
                await solve_metadata(state, query)
            else:
                await solve_semantic(state, query)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FATAL] {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
        return state

    return solve
