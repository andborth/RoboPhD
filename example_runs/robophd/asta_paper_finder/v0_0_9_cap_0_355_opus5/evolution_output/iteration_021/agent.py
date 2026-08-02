"""solo-mining-coverage — a PaperFindingBench solver.

Base: `iter6_conjunction_evidence_fusion`. Iteration 006's aggregate ranked it
LAST (44.42 vs 49.75 / 49.26) and the aggregate is wrong about it. On the nine
semantic queries — 73% of the benchmark, and the same share of the held-out
test set — iter6 beat both rivals:

    semantic mean F1   iter6 0.346    iter4 0.329    iter3 0.312
    grade-3 counts     170: 74v67/68   186: 24v19/17   91: 34v29/31   152: 13v12/12

Its whole deficit is `specific_44`, one all-or-nothing query it lost 1.000 →
0.000, worth 7.1 aggregate points on a 14-problem batch. The specific-path code
is byte-identical to iter4's (diffed), and iter4 got the same query right, so
that is sampling variance in one LLM call, not a defect to repair. Discarding
iter6's semantic gains over it would be discarding the signal and keeping the
noise. This agent keeps iter6 whole and attacks the two things its own
diagnostics name.

(1) GROUPED MINING STARVES MOST OF THE HEAD — the dominant, measured loss.
    iter6 mined evidence with `snippet_search` scoped to groups of 3 paper_ids
    at limit 9. Its stdout reports how that landed:

        semantic_160   97/240 head papers have body evidence   score 0.396
        semantic_120   96/240                                  score 0.000
        semantic_186  155/240                                  score 0.508
        semantic_152  186/240                                  score 0.390

    Its two worst semantic scores are its two worst coverage numbers. The cause
    is documented in CLAUDE.md and reproduced on the probe: `paper_ids` is a
    scope FILTER, not a per-paper allocation. The ranker hands all nine slots to
    whichever paper in the group matches the query best; the other two get
    nothing. 60% of the head therefore reached the judge with title + abstract
    only — and an abstract states the topic, essentially never the second and
    third criteria.

    That is exactly the grade-2 trap. `recall = |{i <= K : g_i = 3}| / K`, and
    grade 3 needs `weighted > 0.99`, i.e. EVERY weighted criterion judged
    Perfectly Relevant. One unproven criterion caps a paper at grade 2, which
    earns precisely zero recall. Read semantic_160's verdicts against its
    evidence and the mechanism is visible paper by paper: #2 (259212824) and #5
    (209901416) are squarely on topic, carry ASR evidence, carry bottleneck
    evidence, carry nothing on the gold's third criterion ("methodological
    solutions", weight 0.3) — both graded Highly Relevant, both worth nothing.
    #3 (226307012) differs only in that its evidence happens to include
    "a novel transcription workflow which combines spoken term detection and
    human-in-the-loop" — Perfectly Relevant.

    FIX: the top MINE_SOLO=72 papers are now mined SOLO — one scoped call per
    (paper, criterion), plus one on the raw request. A starved slot is
    impossible by construction when the scope holds one paper. Probed: a solo
    call keyed on "community-based transcription workflow methodology" against
    CorpusId:226307012 returns that paper's Conclusion section, which is
    precisely the passage that earns criterion 3. Tier B (papers 72-240) keeps
    grouping at 3 ids / limit 12 for large-K depth. Call count 232 -> ~440, all
    of them free; the only budget spent is wall-clock, and iteration 006's
    slowest query finished in 881 s against a 1740 s timeout. Solo tasks are
    queued first, so if MINE_BUDGET (8 -> 10 min) binds, the loss lands on the
    tail rather than on the head that decides recall.

(2) AN EMPTY CRITERION SLOT WAS BEING SILENTLY ABANDONED. Evidence assembly
    spends one passage slot per criterion, drawn from that criterion's own
    scoped passages — but when that criterion's query returned nothing for the
    paper it hit `continue` and forfeited the slot. Now it falls back to
    whichever remaining passage shares the most terms with the criterion. Same
    budget, same verbatim grounding, one more criterion argued.

Two smaller changes:

(3) THE 9th PASSAGE. Evidence joined up to 9 parts; the grounding check accepts
    up to 8 and each passage is checked independently, so the 9th was at risk of
    being dropped before the judge. Capped at 8.

(4) THE SPECIFIC PATH NOW HEDGES ON MODEL DISAGREEMENT. `specific_f1` is
    all-or-nothing on a single id and it cost iter6 an entire query. Two
    independent verifiers (GPT-5.4 and Sonnet-4.6) now rank the same candidate
    list. Agreement — the common case, and the case that was already scoring
    1.000 — submits one id and keeps the full point. Disagreement submits both:
    0.667 if either is right, against a coin flip that pays 0.500. One extra
    LLM call on ~13% of queries, inside the free cost zone either way (iter6's
    mean spend was $0.13 against a $0.355 threshold).

Everything else is iter6's and unchanged: the analyzer, facet-conjunction
keyword generation, the citation/author/metadata routing, stage-1 grading at
520, the {0,1,3} judge replica at depth 240 (Sonnet to 120, GPT-5.4-mini
beyond), and the ranking arithmetic.

ITERATION 009 (this file). Measured on iteration 008's own diagnostics and on
live tool probes:

  * Mining was starved at the CALL level, not the corpus level. 592 solo
    `snippet_search(paper_ids=<one>, limit=4)` calls per query harvested only
    126-855 passages, and fewer than half the head papers came back with any
    body text. A single probed call with ten paper_ids at limit=100 returns 100
    passages covering every group member that HAS indexed full text (37-65 s).
    So mining is now a BULK HARVEST -- groups of 10 at limit 100, once per
    criterion plus once on the raw request, over all 250 submitted papers --
    followed by a targeted solo REPAIR pass for criteria the bulk pass left
    uncovered on papers it proved have text. ~125 + ~130 calls, 12500-passage
    ceiling, all free.
  * Evidence length still tracks the grade: over 824 judged papers,
    P(grade 3) = 0.198 at 800-1199 chars and 0.451 at 2000-2399; 0.191 at three
    passages and 0.466 at seven. 243 papers were still in the short band
    because the selector had 1-3 candidate windows to fill 5 slots from.
  * The judge replica saturates: 269 of the 824 scored a perfect 1.00 and only
    61% of those were really grade 3. A second, independent GPT-5.4-mini vote
    over the same evidence (top 160) now breaks ties INSIDE an identical
    primary score, blended 0.75/0.25 with the criterion-coverage tie-break
    (which measured 0.519 -> 0.671 P(grade 3) inside that bucket). It never
    reorders across primary buckets, so its downside is bounded at zero.

Diagnostics: the `top10=` line is now `ratings=` over the full stage-2 depth —
(cid, per-criterion rating vector, criteria-covered count, weighted score,
second-vote score, retained-passage count) for every paper the judge will grade
— so the next session can join it against judge_verdicts.json and get a real
replica-calibration set for free.
"""

import asyncio
import collections
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
MAX_RERANK = 900          # stage-1 grading budget (candidates)
# Iteration 018 measurement that set this number. `recall` is the binding term
# in every semantic score (rank averaged 0.65-0.98 across the batch; recall
# 0.06-0.71), and recall counts ONLY grade-3 papers inside the first K
# submitted positions. A candidate that stage 1 never looks at cannot reach the
# head, cannot be evidence-mined, and therefore cannot be one of them. On
# `semantic_110` the pool held 969-1205 candidates against a 520 screen -- one
# agent printed "[ALERT] MAX_RERANK binding: 685 pooled candidates never
# screened" -- and of the 183 papers the judge graded Perfectly Relevant for
# SOME agent that query, the best agent submitted 152 and the worst 127. The
# screen, not the retrieval, was throwing them away. 900 costs ~$0.05 of extra
# GPT_5_4_MINI at 20 briefs per batch, against $0.13 of unused free-zone
# headroom ($0.224 spent of the $0.355 threshold).
ENRICH_CAP = 620          # abstract backfill depth: a get_paper_batch hop per
#                           100 ids is free but not fast, so the deeper tail of
#                           the screen reads whatever the search returned.
# Observed K (= the recall denominator AND the judged depth) across iterations
# 004-005: 6, 8, 14, 18, 20, 26, 34, 46, 52, 56, 94, 114, 180, 206, 222, 304.
# Anything inside K is scored, so these depths are the recall term, not polish.
HEAD_SIZE = 250           # evidence-MINED depth (free: snippet_search only)
# Iteration 008 issued 592 solo `snippet_search(paper_ids=<one>, limit=4)` calls
# per query and harvested 126-855 passages -- fewer than half its head papers
# came back with ANY body text. Probed live this session: ONE call with ten
# paper_ids at limit=100 returns 100 passages covering every paper in the group
# that has indexed full text (10/10 on top-of-list ids in 37 s, 5/10 on deep
# ids in 65 s -- the five misses have no body text in the corpus at all, so no
# amount of solo calling would have reached them). The "grouping starves
# papers" result that pushed 006->007 to solo mining was an artifact of
# limit=12 over 3 ids. Bulk harvest raises the passage ceiling from 2400 to
# 12500 for a FIFTH of the calls.
BULK_GROUP = 10           # paper_ids per bulk scoped call
BULK_LIMIT = 100          # passages per bulk call (server max)
REPAIR_DEPTH = 96         # solo follow-up depth, for criteria bulk missed
REPAIR_LIMIT = 6          # passages per solo repair call
REPAIR_BLIND = 24         # top papers retried even with no bulk hit
REPAIR_CAP = 240          # ... and a hard call cap, so pass B cannot
#                           eat the wall-clock pass A already earned
SNIPPET_CAP = 24          # retained passages per candidate (selector input)
STAGE2_SIZE = 250         # judge-replica depth
STAGE2_DEEP = 120         # ... CLAUDE_SONNET_4_6 up to here, GPT_5_4_MINI after
SECOND_DEPTH = 160        # independent second replica vote (tie-break only)
SECOND_BATCH = 10
STAGE1_BATCH = 20
STAGE2_BATCH = 7
STAGE2_TAIL_BATCH = 10
TOOL_CONCURRENCY = 6
MINE_CONCURRENCY = 28
LLM_CONCURRENCY = 8
WALL_BUDGET = 22 * 60     # hard stop well inside the 29-minute timeout
MINE_BUDGET = 12 * 60     # cap on the mining phase specifically
GRADE_RESERVE = 5 * 60    # wall-clock kept back so stage 2 always runs
CITE_SUBMIT = 30          # ids submitted from the citation path (see below)
REFCHECK_CAP = 400        # citers put through the `references` verification
REFCHECK_BUDGET = 6 * 60  # get_paper_batch(references) is the slowest call --
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
# Measured on iteration 005: the agent whose head evidence averaged 1728 chars
# earned 6 grade-3 verdicts in the top 8 of `semantic_138`; the one that averaged
# 1378 earned 3. Longer, criterion-covering evidence is strictly better as long
# as it stays under the 2500-char truncation, so the budget is spent, not saved.
ABSTRACT_CHARS = 850      # abstract budget when there is little body text
ABSTRACT_TIGHT = 620      # ... and when mined passages are competing for room
TLDR_CHARS = 260
SNIPPET_CHARS = 430       # per-window ceiling; the actual allowance is
#                           recomputed from the unspent budget each slot
MIN_WINDOW = 200
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


def _sent_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans of the sentences of `text`, covering it fully."""
    spans = [(m.start(), m.end())
             for m in re.finditer(r"[^.!?\n]+[.!?]*[ \t\r\n]*", text)
             if m.group().strip()]
    return spans or [(0, len(text))]


def _window(text: str, limit: int, terms: set[str]) -> str:
    """The best contiguous <=limit-char SENTENCE WINDOW of `text` for `terms`.

    `snippet_search` returns ~500-word body chunks -- measured 172-2247 chars on
    the probe -- and the sentence that actually proves a criterion sits wherever
    it sits. Iterations 005-007 clipped a PREFIX, so on a 2200-char chunk they
    kept the first 20% and threw away the passage the scoped query matched on.
    A contiguous substring is just as verbatim-derivable as a prefix, so this
    costs nothing against the grounding check and spends the slot on the text
    that argues the criterion instead of on the chunk's opening boilerplate.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    if not terms:
        return _clip(text, limit)
    spans = _sent_spans(text)
    sterms = [_terms(text[a:b]) for a, b in spans]
    best, best_score = None, -1.0
    for i in range(len(spans)):
        start = spans[i][0]
        acc: set[str] = set()
        for j in range(i, len(spans)):
            end = spans[j][1]
            if end - start > limit:
                break
            acc |= sterms[j]
            # hits dominate; length is a tie-break so a window that matches as
            # many criterion terms in more context wins over a bare fragment.
            score = len(terms & acc) + 0.001 * (end - start) / max(1, limit)
            if score > best_score:
                best, best_score = (start, end), score
    if best is None or best_score <= 0.999:
        # nothing in the chunk matches the criterion at all -- the prefix is as
        # good a guess as any, and it is at least coherent.
        return _clip(text, limit)
    return text[best[0]:best[1]].strip()


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
                 "csnips",
                 "authors", "snippets", "fusion", "g1", "weighted", "crit",
                 "weighted2", "ncites", "nauthors")

    def __init__(self, cid: str):
        self.cid = cid
        self.pid = ""
        self.csnips: dict[int, list[str]] = {}
        self.title = ""
        self.abstract = ""
        self.tldr = ""
        self.year = None
        self.venue = ""
        self.authors: list[str] = []
        self.ncites = None      # citationCount, for numeric metadata predicates
        self.nauthors = None    # true author count (self.authors is clipped)
        self.snippets: list[str] = []
        self.fusion = 0.0
        self.g1 = None          # stage-1 holistic grade 0-3
        self.weighted = None    # stage-2 judge-replica weighted score in [0,1]
        self.crit: list[int] = []
        self.weighted2 = None   # independent second replica vote, tie-break only

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
        if self.ncites is None and isinstance(doc.get("citationCount"), int):
            self.ncites = doc["citationCount"]
        if self.nauthors is None and isinstance(doc.get("authors"), list) \
                and doc["authors"]:
            self.nauthors = len(doc["authors"])
        if not self.venue:
            self.venue = (doc.get("venue") or "").strip()
        if not self.authors:
            au = doc.get("authors") or []
            if isinstance(au, list):
                self.authors = [a.get("name", "") for a in au
                                if isinstance(a, dict)][:12]

    def add_snippet(self, text: str, crit_idx: int | None = None) -> None:
        text = (text or "").strip()
        if not text or len(text) < 60:
            return
        for existing in self.snippets:
            if text[:80] == existing[:80]:
                return
        self.snippets.append(text)
        # Remember which criterion's scoped query retrieved this passage.
        # Evidence assembly spends one slot per criterion before anything else,
        # so no criterion is left unproven while another gets four redundant
        # passages -- that single gap is what caps a paper at grade 2.
        if crit_idx is not None:
            self.csnips.setdefault(crit_idx, []).append(text)

    def brief(self, abs_chars: int) -> str:
        body = self.tldr or self.abstract
        if self.tldr and self.abstract:
            body = self.tldr + " " + self.abstract
        yr = f" ({self.year})" if self.year else ""
        return f"{self.title}{yr}. {_clip(body, abs_chars)}"

    def evidence(self, criteria: list[dict]) -> str:
        """Verbatim passages joined by ' ... ', selected for CRITERION COVERAGE.

        Every passage is a contiguous substring of text the tools returned for
        this same paper, so it stays verbatim-derivable for the grounding check.
        Nothing here is model-written.

        Two things changed against iteration 007, both measured on its own
        output. (1) Passages are WINDOW-selected (see `_window`) rather than
        prefix-clipped -- a 2200-char body chunk used to contribute its first
        430 chars, which is rarely the sentence the scoped query matched.
        (2) The budget is now actually spent: iteration 007 averaged 1319 of
        2400 allowed chars, and across its 1130 judged papers P(grade 3) rose
        from 0.21 in the 600-1200 char band to 0.48 in the 1800-2400 band.
        """
        parts: list[str] = []
        used = 0

        def push(piece: str, floor: int = 20) -> bool:
            nonlocal used
            piece = _clip(piece, max(0, EVIDENCE_CAP - used - 5))
            if piece and len(piece) >= floor:
                parts.append(piece)
                used += len(piece) + 5
                return True
            return False

        crit_terms = [_terms(c.get("name", "") + " " + c.get("description", ""))
                      for c in (criteria or [])]

        # Body passages plus, as a last-resort source, the part of the abstract
        # that the head clip is about to drop. 60% of head papers came back from
        # mining with no body passage at all in iteration 007, and for those the
        # abstract's own tail is the only unspent verbatim text there is.
        pool: list[str] = list(self.snippets)

        # The title always goes in, however short: a paper whose evidence string
        # is empty is scored Not Relevant with no judge call at all.
        if self.title:
            push(self.title, floor=1)
        # The tldr restates the abstract's topic. When there are real body
        # passages competing for room, its 260 chars buy more as a window.
        if self.tldr and not (self.abstract and len(pool) >= 4):
            push(_clip(self.tldr, TLDR_CHARS))
        abs_head = ""
        if self.abstract:
            abs_head = _clip(self.abstract,
                             ABSTRACT_TIGHT if len(pool) >= 3 else ABSTRACT_CHARS)
            push(abs_head)
            idx = self.abstract.find(abs_head)
            tail = (self.abstract[idx + len(abs_head):].strip()
                    if idx >= 0 else "")
            if len(tail) >= 80:
                pool.append(tail)

        slots_total = min(MAX_SNIPPETS, 8 - len(parts))

        def allow() -> int:
            """Per-slot char allowance, re-derived from what is actually left."""
            left = min(slots_total - slots, len(pool))
            if left <= 0:
                return 0
            return max(MIN_WINDOW,
                       min(SNIPPET_CHARS, (EVIDENCE_CAP - used - 10) // left))

        slots = 0
        covered = _terms(" ".join(parts))
        if pool and crit_terms:
            # Pass 1 -- one window per criterion, heaviest first, taken from the
            # passages that criterion's own scoped query retrieved. Grade 3 (the
            # only grade that earns recall) needs EVERY criterion proven, so
            # coverage is bought before length.
            for ci in sorted(range(len(crit_terms)),
                             key=lambda i: -criteria[i].get("weight", 0.0)):
                if slots >= slots_total or not pool or used >= EVIDENCE_CAP - 210:
                    break
                ct = crit_terms[ci]
                own = [sn for sn in self.csnips.get(ci, []) if sn in pool]
                if not own and ct:
                    # FALLBACK: this criterion's own scoped query returned
                    # nothing for this paper -- still spend its slot, on
                    # whichever remaining passage shares the most terms with it.
                    own = [sn for sn in pool if len(ct & _terms(sn)) >= 1]
                if not own:
                    continue
                best = max(own, key=lambda sn: len(ct & _terms(sn))) if ct else own[0]
                pool.remove(best)
                if push(_window(best, allow(), ct)):
                    covered |= _terms(parts[-1])
                    slots += 1
            # Pass 2 -- fill the remaining budget with whichever passage best
            # serves the criterion that is still least covered, windowed on that
            # criterion's terms.
            while slots < slots_total and pool and used < EVIDENCE_CAP - 210:
                gaps = [(1.0 - len(ct & covered) / len(ct)) if ct else 0.0
                        for ct in crit_terms]
                best, best_ci, best_score = None, 0, 0.0
                for sn in pool:
                    st = _terms(sn)
                    for ci, (gap, ct) in enumerate(zip(gaps, crit_terms)):
                        if not ct:
                            continue
                        sc = gap * (len(ct & st) / len(ct))
                        if sc > best_score:
                            best, best_ci, best_score = sn, ci, sc
                if best is None:
                    best, best_ci = pool[0], 0
                pool.remove(best)
                tgt = crit_terms[best_ci] if crit_terms else set()
                if push(_window(best, allow(), tgt)):
                    covered |= _terms(parts[-1])
                slots += 1
        else:
            for sn in pool:
                if slots >= slots_total or used >= EVIDENCE_CAP - 210:
                    break
                push(_clip(sn, allow()))
                slots += 1

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
"criteria": objects {{"name": short label, "description": one sentence
  stating a property the paper's text must explicitly demonstrate, "weight":
  float}}. These must reproduce, as closely as you can, the rubric an
  independent annotator would write for this request. That rubric is ALWAYS
  **exactly 3 criteria** unless the request names only two separable concepts
  (then 2) or genuinely four (then 4). DEFAULT TO 3. Weights are drawn from
  0.4/0.3/0.3 (most common) or 0.5/0.3/0.2.

  Rules for splitting, in priority order:
  1. One criterion per distinct concept literally named in the request, in the
     request's own vocabulary. Never merge two concepts into one criterion.
  2. QUALIFIERS ARE THEIR OWN CRITERION. Adjectives and scope words the request
     attaches to the topic -- "common", "widely used", "large-scale", "recent",
     "state-of-the-art", "various", "real-world", "end-to-end", "open-source",
     "at least several minutes" -- each become a separate criterion, NOT a word
     folded into another criterion's label. This is the single most common way
     a rubric is under-split.
  3. NEGATIVE / EXCLUSION constraints ("please exclude survey papers", "not
     simulation-only", "excluding position papers") are ALWAYS their own
     criterion. Phrase such a criterion with the POSITIVE vocabulary a
     qualifying paper's own text would use, so the requirement is provable from
     the paper's prose: e.g. "The paper must present original research -- for
     instance proposing a new model, method or dataset and reporting
     experiments -- rather than being a survey or review of prior work."
  4. Do NOT invent a requirement the request never states.

  Each description is one sentence beginning "The paper must ...", and then
  spells out what that concept means with a defining clause, so the wording
  contains the terms a qualifying passage would contain.

  Worked examples of the target rubric:
  - "What are the common model architectures for retrieval-augmented language
    models?" -> (1) "Retrieval-Augmented Language Models" 0.5 (2) "Model
    Architectures" 0.3 (3) "Commonality of Architectures" 0.2  <- note the
    third: the qualifier "common" is its own criterion.
  - "multimodal foundation models supporting visual and audio inputs,
    pre-trained on large-scale data. Please exclude survey papers." ->
    (1) "Multimodal Foundation Models" 0.4 (2) "Pre-training on Large-Scale
    Datasets" 0.3 (3) "Exclusion of Survey Papers" 0.3.
  - "performance decline of language models like BLOOM under 4-bit integer
    columnar weight-only quantization" -> (1) "Evaluation of performance
    decline" 0.4 (2) "4-bit integer columnar weight-only quantization" 0.3
    (3) "Language models including BLOOM" 0.3.
  Weights sum to 1.0.
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
"year_min": integer or null.  "year_max": integer or null.  Note "after 2022"
  means year_min = 2023; "since 2022" / "2022 and beyond" means year_min = 2022.
"min_citations": integer or null -- the request's requirement on how many times
  the paper itself has been CITED ("with more than 50 citations" -> 51; "cited
  by at least 30 other papers" -> 30). Null when the request says nothing about
  citation counts.
"min_authors": integer or null -- requirement on the paper's OWN author count
  ("written by more than 3 authors" -> 4; "at least 5 authors" -> 5). Null
  otherwise.
"max_authors": integer or null ("single-authored" -> 1). Null otherwise.
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

    def _posint(key):
        v = data.get(key)
        if isinstance(v, bool) or v is None:
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

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
        "min_citations": _posint("min_citations"),
        "min_authors": _posint("min_authors"),
        "max_authors": _posint("max_authors"),
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
    # Raised from 14 with MAX_RERANK 520 -> 900: the screen is now wide enough
    # that extra pool depth is reachable instead of being truncated away.
    # Keyword calls are free and run at TOOL_CONCURRENCY, so this is wall-clock
    # only, and iteration 018 finished in 644-940 s of a 1320 s budget.
    kws = kws[:18]
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
                        deadline: float, query: str = "") -> None:
    """Pull body passages that PROVE each criterion, per paper.

    The judge sees only `markdown_evidence`. Abstracts establish the topical
    criterion and almost never the method/evaluation criteria, which is why 65
    papers landed on grade 2 last iteration. `snippet_search` scoped by
    `paper_ids` reaches body text and closes that gap.

    Iteration 006 measured the cost of grouping: with groups of 3 at limit 9,
    only 97 of 240 head papers on `semantic_160` and 96 of 240 on `semantic_120`
    came back with ANY body passage — and those were its two worst semantic
    scores (0.396 and 0.000). `paper_ids` is a scope FILTER, not a per-paper
    allocation: the ranker hands all nine slots to whichever paper in the group
    matches best and the other two starve. So the head is now mined SOLO, one
    call per (paper, criterion), where a starved slot is impossible by
    construction. Tier B keeps grouping (3 ids, limit 12) for depth on the
    large-K queries.
    """
    snippet = _tool(state, "snippet_search")
    if not snippet or not head:
        return

    crits = plan["criteria"]
    # (criterion index, scoped query). Index -1 is the raw user request: the
    # inferred criteria are a guess at the gold ones, and a passage retrieved by
    # the whole request hedges that inference error at the cost of one free call.
    queries = [(i, f"{c['name']}. {c['description']}")
               for i, c in enumerate(crits)][:4]
    if not queries:
        return
    queries.append((-1, query))

    stop = min(deadline, time.time() + MINE_BUDGET)
    hits = 0

    async def mine(group: list[Candidate], ci: int, q: str, limit: int):
        nonlocal hits
        if time.time() > stop:
            return
        ids = ",".join(f"CorpusId:{c.cid}" for c in group)
        try:
            raw = await asyncio.wait_for(
                snippet(query=q, paper_ids=ids, limit=limit),
                timeout=150 if limit > 20 else 90)
        except Exception:
            return
        by_cid = {c.cid: c for c in group}
        for e in _parse_items(raw):
            paper = e.get("paper") if isinstance(e.get("paper"), dict) else {}
            cand = by_cid.get(_cid(paper))
            if cand is None or len(cand.snippets) >= SNIPPET_CAP:
                continue
            sn = e.get("snippet")
            txt = (sn.get("text") if isinstance(sn, dict) else None) or ""
            before = len(cand.snippets)
            cand.add_snippet(txt, None if ci < 0 else ci)
            hits += len(cand.snippets) - before

    # ---- PASS A: bulk harvest over the whole submitted depth ----------------
    # Groups of BULK_GROUP ids at limit=BULK_LIMIT, once per criterion plus once
    # on the raw request. 25 groups x 5 queries = 125 calls with a 12500-passage
    # ceiling, against iteration 008's 592 calls with a 2400-passage ceiling for
    # the same wall-clock. Emitted head-first: heaviest criterion first and the
    # earliest groups first, so a MINE_BUDGET overrun lands on the tail, never
    # on the head that decides recall.
    body = head[:HEAD_SIZE]
    groups = [body[i:i + BULK_GROUP] for i in range(0, len(body), BULK_GROUP)]
    qh = sorted(queries[:-1], key=lambda t: -crits[t[0]].get("weight", 0.0))
    ordered_q = qh + [queries[-1]]
    tasks_a = [mine(g, ci, q, BULK_LIMIT) for g in groups for ci, q in ordered_q]
    await _gather(tasks_a, MINE_CONCURRENCY)
    bulk_hits = hits
    with_a = sum(1 for c in body if c.snippets)

    # ---- PASS B: solo repair, targeted at the gaps pass A actually left ------
    # Criterion coverage is both an evidence input and a measured RANKING
    # feature: joining iteration 008's `ratings=` stdout against its
    # `judge_verdicts.json` (824 judged papers), among the 269 papers the
    # replica scored a perfect 1.0 those with >=1 criterion-scoped body passage
    # reached grade 3 67% of the time against 52% for those with none. Solo
    # calls are only worth their latency once the gap is KNOWN, so they are
    # issued here rather than blind. Head-first, one call per missing criterion.
    # Only papers that pass A proved HAVE indexed body text are worth a solo
    # retry -- probed live, ~half the corpus returns nothing for any scoped
    # query, and re-asking cannot conjure text that is not there. The top
    # REPAIR_BLIND papers get one attempt regardless, since a bulk group can
    # in principle exhaust its 100 slots on its strongest members.
    gaps = [(c, ci, q)
            for n, c in enumerate(head[:REPAIR_DEPTH])
            if c.snippets or n < REPAIR_BLIND
            for ci, q in ordered_q
            if ci >= 0 and not c.csnips.get(ci)][:REPAIR_CAP]
    n_repair = len(gaps)
    if gaps and time.time() < stop:
        await _gather([mine([c], ci, q, REPAIR_LIMIT) for c, ci, q in gaps],
                      MINE_CONCURRENCY)
    else:
        n_repair = 0
    with_sn = sum(1 for c in body if c.snippets)
    ncov = sum(len(c.csnips) for c in head[:REPAIR_DEPTH])
    print(f"  mined {hits} passages ({bulk_hits} bulk / {hits - bulk_hits} repair) "
          f"via {len(tasks_a)} bulk + {n_repair} repair calls; "
          f"body-evidence {with_sn}/{len(body)} (bulk alone {with_a}); "
          f"{ncov / max(1, len(head[:REPAIR_DEPTH])):.2f} criteria covered "
          f"per repaired paper")


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


def _apply_second(text: str, batch: list[tuple[int, Candidate, str]],
                  criteria: list[dict]) -> int:
    """Identical parse to `_apply_stage2`, stored in `weighted2` instead."""
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
        batch[idx][1].weighted2 = min(1.0, sum(c["weight"] * r / 3.0
                                               for c, r in zip(criteria, ratings)))
        n_applied += 1
    return n_applied


async def rerank(query: str, plan: dict, cands: list[Candidate],
                 deadline: float) -> list[Candidate]:
    """Stage 1: cheap model over everything, to pick the head.
    Stage 2: judge replica over the head's actual submitted evidence."""
    criteria = plan["criteria"]
    stage1 = cands[:MAX_RERANK]
    if len(cands) > MAX_RERANK:
        print(f"  [ALERT] MAX_RERANK binding: {len(cands) - MAX_RERANK} "
              f"pooled candidates never screened")
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

    # Diagnostic for whoever tunes MAX_RERANK next: the deeper screen only buys
    # anything when the top grade bucket is SMALLER than HEAD_SIZE, or when
    # candidates past the old 520 boundary land in it. Both are printed.
    hist = collections.Counter(c.g1 for c in stage1)
    deep_top = sum(1 for c in stage1[520:] if (c.g1 or 0) >= 3)
    print(f"  stage1 grade histogram {dict(sorted(hist.items(), reverse=True))}"
          f"; g1=3 beyond old-520 boundary: {deep_top}")

    return sorted(stage1, key=lambda c: (-(c.g1 or 0), -c.fusion))[:HEAD_SIZE]


async def judge_replica(query: str, plan: dict, head: list[Candidate],
                        deadline: float) -> None:
    criteria = plan["criteria"]
    head = head[:STAGE2_SIZE]
    items = [(i, c, c.evidence(criteria)) for i, c in enumerate(head)]
    # Keep prompt cost bounded, but the replica has to see what the real judge
    # sees or its per-criterion ratings are a dry run of a different document.
    items = [(i, c, _clip(ev, 2000)) for i, c, ev in items if ev]
    # Positions 1..STAGE2_DEEP get Sonnet -- that is where ordering decides the
    # whole score on the small-K queries (K was 6, 8, 18, 20 on four of
    # iteration 005's problems). Positions beyond get GPT_5_4_MINI at a quarter
    # of the price: they only matter on the large-K queries (94, 206), where
    # the alternative is no replica grade at all.
    deep = [t for t in items if t[0] < STAGE2_DEEP]
    tail = [t for t in items if t[0] >= STAGE2_DEEP]
    batches = ([(deep[i:i + STAGE2_BATCH], CLAUDE_SONNET_4_6, "stage2")
                for i in range(0, len(deep), STAGE2_BATCH)]
               + [(tail[i:i + STAGE2_TAIL_BATCH], GPT_5_4_MINI, "stage2-tail")
                  for i in range(0, len(tail), STAGE2_TAIL_BATCH)])
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

    await _gather([do2(b, h, t) for b, h, t in batches], LLM_CONCURRENCY)
    # ---- second, INDEPENDENT vote over the same evidence ---------------------
    # The primary replica saturates: 269 of iteration 008's 824 judged papers
    # scored a perfect 1.00 and only 61% of them were really grade 3. Inside
    # that one bucket -- which carries the top of every submission and 61% of
    # all grade-3s -- the replica has no resolution left. A second opinion from
    # a different model on the same text does: agreement is evidence, and it
    # enters `final_order` ONLY as a tie-break inside an identical primary
    # score, so it can never reorder across buckets and its downside is zero.
    second = [t for t in items if t[0] < SECOND_DEPTH]
    sbatches = [second[i:i + SECOND_BATCH]
                for i in range(0, len(second), SECOND_BATCH)]
    applied2 = 0

    async def do2b(batch):
        nonlocal applied2
        if time.time() > deadline:
            return
        local = [(j, c, ev) for j, (_, c, ev) in enumerate(batch)]
        txt = await _gen(GPT_5_4_MINI, _stage2_prompt(query, criteria, local),
                         tag="stage2-vote2")
        applied2 += _apply_second(txt, local, criteria)

    if sbatches:
        await _gather([do2b(b) for b in sbatches], LLM_CONCURRENCY)
    pred3 = sum(1 for c in head if c.weighted is not None
                and _grade_from_weighted(c.weighted) == 3)
    agree = sum(1 for c in head if c.weighted is not None and c.weighted > 0.99
                and c.weighted2 is not None and c.weighted2 > 0.99)
    cov1 = sum(1 for c in head if c.csnips)
    print(f"  stage2 judged {applied}/{len(items)} (+{applied2} second votes); "
          f"predicted grade-3: {pred3} (both voters: {agree}); "
          f"criterion-covered: {cov1}/{len(head)}")


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
    # Tie-break on CRITERION COVERAGE before anything else. The replica's
    # weighted score saturates at 1.0 -- 390 of 1130 judged papers in iteration
    # 007 sat in that one bucket -- and inside it the real judge graded 70% of
    # the papers with >=1 criterion-scoped body passage a 3, against 44% of
    # those with none (consistent on every one of the 9 semantic queries). Only
    # the first K papers are ever judged, and only grade 3 earns recall, so
    # promoting the proven ones inside a tie is a direct recall gain and cannot
    # cost anything: it never reorders across different predicted scores.
    def tie(c: Candidate) -> float:
        # Blend of the two signals that resolve inside a primary tie: the
        # independent second vote (0.55 = neutral, for papers past
        # SECOND_DEPTH) and criterion coverage (measured 0.519 -> 0.671
        # P(grade 3) inside the saturated 1.00 bucket on iteration 008).
        v2 = c.weighted2 if c.weighted2 is not None else 0.55
        return 0.75 * v2 + 0.25 * (min(len(c.csnips), 3) / 3.0)

    ordered_head = sorted(head, key=lambda c: (-key(c), -tie(c),
                                               -(c.g1 or 0), -c.fusion))
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
    prompt = VERIFY_PROMPT.format(query=query, cands=lines)

    # TWO INDEPENDENT VERIFIERS, hedged on disagreement. specific_f1 is
    # all-or-nothing on one id: right = 1.000, wrong = 0.000, and iteration 006
    # lost a whole query (7 aggregate points on a 14-problem batch) to a single
    # confident wrong pick on `specific_44` -- the SAME code that got it right
    # in iteration 005, so this is sampling variance, not a bug to fix.
    # Two papers submitted with the gold one among them scores 0.667, so when
    # the two models name different papers the hedge beats the coin flip
    # (0.667 * P(gold in the union) > 0.5) while agreement keeps the full 1.000.
    a_txt, b_txt = await asyncio.gather(
        _gen(GPT_5_4, prompt, tag="verify-specific"),
        _gen(CLAUDE_SONNET_4_6, prompt, tag="verify-specific-b"),
    )

    def _picks(txt: str) -> list[Candidate]:
        idxs = _jload(txt)
        out: list[Candidate] = []
        if isinstance(idxs, list):
            for i in idxs[:3]:
                if isinstance(i, int) and 0 <= i < len(cands) and cands[i] not in out:
                    out.append(cands[i])
        return out

    a, b = _picks(a_txt), _picks(b_txt)
    picked = list(a)
    if a and b and b[0] is not a[0]:
        # Disagreement on the primary pick: carry both, best-first by the
        # retrieval fusion score (the one signal outside either model).
        if b[0] not in picked:
            picked.append(b[0])
        picked = sorted(picked, key=lambda c: -c.fusion)
        print(f"  verifiers disagree -> hedging {[c.cid for c in picked]}")
    elif not picked:
        picked = b
    print(f"  specific picks: {[(c.cid, c.title[:50]) for c in picked]}")
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

def _hard_filters(plan: dict) -> list[str]:
    """Which exact-match predicates the request actually states.

    These are checkable against real corpus fields, so they -- not an LLM
    relevance grade -- are what decides membership on `metadata_f1`. When at
    least one is present it also means the gold set is "everything that passes",
    which is what licenses submitting the full survivor list (see
    `metadata_path`).
    """
    names = []
    if plan["years"] or plan["year_min"] is not None or plan["year_max"] is not None:
        names.append("year")
    if plan["venue_aliases"] or plan["venues"]:
        names.append("venue")
    if plan.get("min_citations"):
        names.append("min_citations")
    if plan.get("min_authors") or plan.get("max_authors"):
        names.append("authors")
    return names


def _predicate_fail(d: dict, plan: dict) -> str | None:
    """Return the name of the first stated predicate `d` violates, else None.

    Operates on a raw tool doc (get_citations / search results) so it can read
    `citationCount` and `authors` directly. A field the corpus never returned is
    treated as UNKNOWN and passes -- dropping a paper for a missing field costs
    recall on a path where recall is usually the binding term.
    """
    years, ymin, ymax = set(plan["years"]), plan["year_min"], plan["year_max"]
    yr = d.get("year")
    if years:
        if yr not in years:
            return "year"
    else:
        if ymin is not None and yr is not None and yr < ymin:
            return "year"
        if ymax is not None and yr is not None and yr > ymax:
            return "year"
        # A year bound with no year at all cannot be verified; on an
        # exact-match path an unverifiable paper is a coin flip, so drop it
        # only when the request gave a bound.
        if (ymin is not None or ymax is not None) and yr is None:
            return "year"

    aliases = plan["venue_aliases"] or [v.lower() for v in plan["venues"]]
    if aliases:
        ven = (d.get("venue") or "").lower()
        jr = d.get("journal")
        if isinstance(jr, dict):
            ven += " " + str(jr.get("name") or "").lower()
        if ven.strip() and not any(v in ven for v in aliases):
            return "venue"

    need_cites = plan.get("min_citations")
    if need_cites:
        cc = d.get("citationCount")
        if isinstance(cc, int) and cc < need_cites:
            return "cites"

    lo, hi = plan.get("min_authors"), plan.get("max_authors")
    if lo or hi:
        au = d.get("authors")
        if isinstance(au, list) and au:
            n = len(au)
            if lo and n < lo:
                return "authors"
            if hi and n > hi:
                return "authors"
    return None


async def _citers(gc, cid: str, limit: int = 1000) -> list[dict]:
    """Unwrap get_citations. NOTE: this is the one tool the benchmark snapshot
    cutoff does not cover, so its output must be screened by the caller."""
    # `citationCount` and `publicationDate` are what make the numeric/date
    # predicates in a metadata request checkable at all -- without them the
    # agent can only guess "more than 50 citations" from an abstract.
    fields = "corpusId,title,year,venue,authors,citationCount,publicationDate"
    raw = None
    # A single ConnectionRefusedError on this call zeroed the whole citation
    # route on metadata_42 (score 0.045 via the keyword fallback). The tool's
    # own retry budget does not cover a refused connection, so retry here.
    for attempt in range(3):
        try:
            raw = await asyncio.wait_for(
                gc(paper_id=cid, fields=fields, limit=limit), timeout=280)
            break
        except Exception as exc:
            print(f"  [warn] get_citations({cid}) attempt {attempt + 1} failed: "
                  f"{type(exc).__name__}: {exc}")
            if attempt == 2:
                return []
            await asyncio.sleep(3.0 * (attempt + 1))
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
                   fields="corpusId,references"),
                timeout=110))
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

    # 300 ids x hundreds of references each is the slowest call in the agent:
    # on `metadata_26` it resolved 13 of 300 inside 5 minutes and the whole
    # verification silently degraded to "no answer". Smaller chunks fail-fast
    # and re-bisect less, and the fan-out is what actually buys throughput.
    groups = [cids[i:i + 5] for i in range(0, len(cids), 5)]
    await _gather([chunk(g) for g in groups], 16)
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
    seen_ref, refs = set(), [r for r in refs if r.cid]
    # resolve_titles tries title variants, so the same landmark comes back more
    # than once. A duplicate here became a self-referential AND-clause on
    # `metadata_26` ("cites-all-3 check") and could never be satisfied.
    refs = [r for r in refs if not (r.cid in seen_ref or seen_ref.add(r.cid))]
    if not refs:
        return []
    print(f"  cite-refs -> {[(r.cid, r.title[:44]) for r in refs]}")

    # Selectivity, not list length, decides which reference to enumerate.
    # get_citations caps at 1000 with no paging, so BOTH landmarks come back at
    # exactly 1000 and `len` cannot tell them apart -- probed on metadata_26,
    # where that tie picked T5 (0 of 10 gold ids inside its 1000-paper window,
    # which is entirely 2025) over Spider (10 of 10). The paper's own
    # citationCount is the signal that separates them.
    counts = {}
    if gb and len(refs) > 1:
        try:
            docs = _parse_items(await asyncio.wait_for(
                gb(ids=[f"CorpusId:{r.cid}" for r in refs],
                   fields="corpusId,title,citationCount"), timeout=90))
            for d in docs:
                if _cid(d) and isinstance(d.get("citationCount"), int):
                    counts[_cid(d)] = d["citationCount"]
        except Exception as exc:
            print(f"  [warn] ref citationCount lookup failed: {exc}")
    if counts:
        print(f"  ref citationCounts -> {counts}")

    lists = await asyncio.gather(*[_citers(gc, r.cid) for r in refs])
    for r, lst in zip(refs, lists):
        print(f"  citers of {r.title[:40]!r} -> {len(lst)}")
    order = sorted(range(len(refs)),
                   key=lambda i: (counts.get(refs[i].cid, len(lists[i]) or 10 ** 9),
                                  len(lists[i]) or 10 ** 9))
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

    drops = {"year": 0, "venue": 0, "cites": 0, "authors": 0}
    out: list[Candidate] = []
    for cid in keep_ids:
        d = docs[cid]
        why = _predicate_fail(d, plan)
        if why:
            drops[why] += 1
            continue
        cand = pool.add(d, len(out), 0.0)
        if cand is not None:
            cand.fusion += 10.0 - 0.001 * len(out)   # keep newest-first
            out.append(cand)
    print(f"  hard-predicate filter -> {len(out)} of {len(keep_ids)} "
          f"(dropped {drops})")
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
            hard = _hard_filters(plan)
            if hard:
                # The request states predicates that were CHECKED against real
                # corpus fields, so the survivor list IS the answer set and
                # precision comes from the filter, not from truncation.
                # Truncating anyway is what capped `metadata_25` (gold 172,
                # submitted 30, and its LLM-picked 30 hit 0 -- an abstract
                # cannot reveal a citationCount) and `metadata_42` (gold 70,
                # submitted 19). Recall is the binding term here: even perfect
                # precision over 30 ids caps F1 at 0.30 against a gold of 172.
                print(f"  hard filters {hard} verified -> submitting all "
                      f"{min(len(cited), MAX_SUBMIT)} survivors")
                return cited[:MAX_SUBMIT]
            # No checkable predicate: the request is a bare conjunction
            # ("papers citing the T5 paper and the spider paper"), and the gold
            # set is some unseen subset of everyone who cites both. Probed on
            # metadata_26: the reference check recovers 10/10 gold but 244
            # papers cite both landmarks, so submitting all of them is
            # precision 0.041 / F1 0.079. citation_path returns newest-first
            # inside the snapshot and the gold there was exactly the newest
            # cohort under the cutoff, so a short recent list wins.
            if len(cited) > CITE_SUBMIT * 3:
                head = await rerank(query, plan, cited, deadline)
                strong = [c for c in head if (c.g1 or 0) >= 2]
                if len(strong) >= CITE_SUBMIT // 2:
                    cited = strong
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
    need_cites = plan.get("min_citations")
    lo, hi = plan.get("min_authors"), plan.get("max_authors")

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
        # Numeric predicates, checked against real corpus fields. An unknown
        # field passes -- see `_predicate_fail`.
        if need_cites and isinstance(c.ncites, int) and c.ncites < need_cites:
            continue
        if (lo or hi) and isinstance(c.nauthors, int):
            if lo and lo < 12 and c.nauthors < lo:
                continue
            if hi and c.nauthors > hi:
                continue
        kept.append(c)
    print(f"  metadata filter kept {len(kept)} of {len(pool.by_cid)}")

    if not kept:
        kept = pool.ranked()[:20]
    elif len(kept) > 60:
        # A verified hard predicate already decided membership; re-screening
        # with an LLM only discards true positives it cannot see the fields for.
        if _hard_filters(plan):
            print(f"  hard filters {_hard_filters(plan)} verified -> "
                  f"keeping all {len(kept)}")
            return kept[:MAX_SUBMIT]
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
                await enrich_missing_abstracts(state, cands[:ENRICH_CAP])
                head = await rerank(query, plan, cands, deadline)
                # Mine body evidence for the head, THEN judge that evidence:
                # stage 2 must score the same text the benchmark judge scores.
                await mine_evidence(state, head, plan,
                                    deadline - GRADE_RESERVE, query)
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
            # Hand the NEXT evolution session a labelled calibration set: the
            # replica's per-criterion rating vector for every paper the judge
            # will actually grade (K tops out at 304), joinable against
            # judge_verdicts.json by id and position. `top10=` was capped at 10
            # and was still the single most useful artifact in the 006 tree.
            print("  ratings=" + str([(c.cid, c.crit, len(c.csnips),
                                       round(c.weighted, 2)
                                       if c.weighted is not None else None,
                                       round(c.weighted2, 2)
                                       if c.weighted2 is not None else None,
                                       len(c.snippets))
                                      for c in picked[:STAGE2_SIZE]]))
        print(f"  submitted {len(results)} papers in {time.time() - t0:.0f}s")
        return state

    return solve
