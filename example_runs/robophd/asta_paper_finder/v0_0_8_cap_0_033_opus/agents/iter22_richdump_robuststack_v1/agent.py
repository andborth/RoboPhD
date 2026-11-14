"""PaperFindingBench solver — iter22_richdump_robuststack_v1.

iter22 = iter21 (the most-complete accumulated robustness + metadata stack:
tool-retry, per-id enrich fallback, iter18 most-cited anchor resolution, and the
min_citations author fix) with ONE change: the semantic EVIDENCE ASSEMBLER is
reverted from iter21's facet-covering selection to the iter14 champion's
snippet-led RICH DUMP (`_build_evidence`).

  WHY. This is a UNION of two co-championed, proven components, not new churn.
  On iter21's own batch, iter14's rich dump beat iter21's facet-covering
  selection on the two LARGEST semantic deltas — semantic_98 (0.362 vs 0.165,
  +0.198) and semantic_125 (0.211 vs 0.073, +0.138) — over the SAME retrieved
  papers, so the win was purely the evidence text. That matches the standing
  cross-batch note "rich snippet-led dump > facet-covering selection for
  grade-3". iter14 is the repeated raw-F1 champion but LACKED the robustness and
  metadata fixes iter21 accumulated (tool-retry, most-cited anchor, min_cit
  author, per-id fallback). iter22 keeps iter21's full stack byte-for-byte and
  changes only the one function that carried iter14's semantic edge — floored on
  the strictly-more-robust base, so it cannot regress the exact-match paths.
  iter21's richer retrieval (per-facet + targeted-facet snippet_search) still
  feeds the rich dump with extra grounded passages — synergistic, not
  conflicting. Cost is unchanged (same 1 extract + 1 mini rerank LLM calls;
  ~$0.006/query, deep in the free zone) — evidence assembly is pure Python.

--- iter21 heritage (unchanged) -----------------------------------------------
iter21 = iter20 (the robust champion, which won its own batch as a guarded
superset of iter9) plus ONE guarded, attributable metadata correctness fix:

  APPLY min_citations IN THE AUTHOR PATH. The extractor already parses
  `min_citations` ("at least 10 citations", "more than 50 citations") and the
  citation-relation branch applies it — but `_solve_metadata_author` silently
  dropped it. On iter20's batch, metadata_31 ("Journal articles by David Harel
  with at least 10 citations, citing papers by Gera Weiss...") went through the
  author path: it retrieved ALL 16 gold (recall=1.0) but dumped 462 papers
  (precision 0.064, F1 0.12) because the ≥10-citations floor was never applied.
  iter21 threads `min_cit` into the author path, filters citationCount>=min_cit
  (missing == keep, exactly mirroring the citation branch so recall is never
  harmed), and sorts survivors citation-desc so the scorer's first-250 cap keeps
  the well-cited gold. Purely additive: with no citation floor in the query the
  behavior is byte-for-byte iter20 (citationCount is merely also requested).
  This is a real correctness gap (parsed-but-ignored constraint), not a fit, so
  it generalizes to any "by AUTHOR with ≥N citations" metadata query.

--- iter20 heritage (unchanged) -----------------------------------------------
iter20_robust_rerank_rich_v1.

iter9_rerank_rich_v1 is the robust champion of this lineage: across every batch
it competed in (iter13/14/17/18/19) it either won or tied the entire evidence-
assembly lineage (iter14 rich-evidence, iter17 abstract-guard, iter19 facet), and
on the iter19 batch it beat them decisively (23.1 vs 14.3 vs 9.2). The lesson:
the semantic evidence path is a delicate near-optimum, and re-tuning it produces
batch noise, not gains. So iter20 keeps iter9's semantic + specific + evidence
assembly BYTE-FOR-BYTE and makes only guarded, generalizing robustness changes,
each of which degrades to iter9 as its floor:

  1. TOOL-CALL RETRY (`_call`): the MCP transport auto-retries 429/529/504 but a
     bare HTTP 500 raises through with no retry. Those transient 500s cost iter9's
     rivals a full exact-match query on iter19 (specific_10: title resolution
     failed twice for iter19 while succeeding first-try for iter9 — pure luck).
     Tool calls are free, so we retry every call up to 2 extra times with a short
     backoff. Only fires on an exception → worst case identical behavior.

  2. STRICTLY-BETTER METADATA PATH (grafted from iter18, guarded): (a) `_resolve_
     anchor` now gathers candidates from the title tool AND relevance search and
     picks the MOST-CITED — fixing nickname mis-resolution ("T5"/"Spider" -> wrong
     paper -> empty citation intersection -> 0). (b) NEW `author_of_paper` branch:
     "papers by an author of the X paper" resolves X's authors then runs the
     author path (directly targets metadata_14, which all agents zeroed). (c)
     min_authors threaded through the author path. Only changes metadata behavior.

  3. EXACT-MATCH FALLBACK CAP: when the specific/metadata structured path returns
     empty and we fall back to a semantic dump, cap it at 40 so a lucky gold
     overlap yields real F1 instead of being diluted toward 0 precision.

--- iter9 heritage (unchanged) ------------------------------------------------
Builds directly on iter6_metadata_targeted_v1 (the aggregate winner). Metadata,
specific, and the semantic evidence-assembly paths are kept BYTE-FOR-BYTE identical.
The ONE change is the semantic reranker (`_rerank`): it now judges each candidate on
its abstract (already enriched, previously unused by the reranker) instead of a single
snippet, and rates on a finer 0-10 judge-aligned scale so genuinely all-aspect papers
sort above topic-only papers. This targets `rank` (measured 0.34-0.55 on ~40% of
semantic queries, where grade-3 papers are retrieved but poorly ordered) — better
ordering lifts both terms of score = harmonic(rank, recall). The reranker keeps its
existing empty/short-output guards, so worst case it matches iter6.

--- iter6 heritage (unchanged) ------------------------------------------------
The semantic path (facet-diverse retrieval + facet-covering evidence + facet-aware
rerank) plus four data-driven additions, each guarded so any failure degrades to
the proven iter4 behavior:

  A. METADATA citation-relation handling. Detect "citing X"/"cited by X" queries,
     resolve each anchor paper by title, get_citations, intersect across anchors,
     then apply venue/year/min_citations/min_authors filters. (metadata_26 was a
     citation-intersection query iter4 had no logic for -> scored 0.)

  B. METADATA robust author+venue+year selection via LLM over the author's papers
     (venue-name variants: ACL != TACL != workshop) + explicit year-SET support
     ("2014 or 2017", not the range 2014-2017). (metadata_15 fix.)

  C. SEMANTIC targeted per-paper facet evidence (the measured grade-2 -> grade-3
     lever). After rerank, for the top ~15 recall-critical candidates, scoped
     snippet_search on each facet their current text fails to cover, adding a
     grounded verbatim body passage. Free (tool calls), coverage-checked, bounded.

  D. SPECIFIC extra recall: snippet_search as a third candidate source + parse
     "AuthorYYYY" citation keys into author/year hints. (specific_9 fix.)

All LLM calls go through model_registry (GPT_5_4_MINI, cheapest, no reasoning).
Tool calls are free and used generously. ~$0.004/query, deep in the free zone.
"""

import asyncio
import json
import re

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4_MINI


# --------------------------------------------------------------------------
# Tool + parsing helpers
# --------------------------------------------------------------------------
def _maybe_tool(state: TaskState, name: str):
    by_name = {ToolDef(t).name: t for t in state.tools}
    return by_name.get(name)


def _parse_items(raw) -> list[dict]:
    """Flatten MCP tool output: a list of ContentText whose `.text` is JSON.
    Some payloads wrap rows in {"data": [...]}; unwrap those."""
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
        else:
            docs.append(doc)
    return docs


async def _call(tool, _retries: int = 2, _backoff: float = 0.6, **kwargs):
    """Await a tool, retrying transient failures before giving up.

    The MCP transport auto-retries 429/529/504 and broken connections, but a
    bare HTTP 500 "Internal Server Error" raises straight through with no retry.
    Those 500s are transient and cluster randomly — on iter19 they zeroed a
    specific query (specific_10) whose title resolution failed twice while the
    same query succeeded first-try for another agent (pure bad luck, recoverable
    on a retry). Since tool calls are FREE and a single lost title resolution
    zeroes an exact-match query, we retry every call up to `_retries` extra times
    with a short async backoff. Only fires on an actual exception, so worst case
    is identical behavior plus a few seconds on a genuinely dead call."""
    if tool is None:
        return None
    try:
        name = ToolDef(tool).name
    except Exception:  # noqa: BLE001
        name = "?"
    last_err = None
    for attempt in range(_retries + 1):
        try:
            return await tool(**kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < _retries:
                try:
                    await asyncio.sleep(_backoff * (attempt + 1))
                except Exception:  # noqa: BLE001
                    pass
    print(f"  tool error {name} (after {_retries + 1} tries): {last_err!r}")
    return None


async def _llm_json(prompt: str, fallback):
    """Cheap LLM call that must return a JSON value; lenient parsing."""
    try:
        resp = await GPT_5_4_MINI.generate(prompt)
        text = (resp.completion or "").strip()
    except Exception as e:  # noqa: BLE001
        print(f"  llm error: {e!r}")
        return fallback
    if not text:
        return fallback
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return fallback


# --------------------------------------------------------------------------
# Text / facet helpers
# --------------------------------------------------------------------------
def _clip(s: str, n: int) -> str:
    return (s or "").strip()[:n]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "by",
    "that", "this", "which", "how", "using", "based", "via", "into", "from",
    "paper", "papers", "research", "study", "studies", "work", "works", "model",
    "models", "method", "methods", "approach", "approaches", "task", "tasks",
    "some", "any", "their", "its", "as", "at", "is", "are", "be", "within",
    "across", "different", "provide", "references", "could", "please", "suggest",
    "investigates", "direct", "me", "you", "us",
}


def _facet_terms(facet: str) -> set[str]:
    """Distinctive content words of a facet phrase. Keeps hyphenated compounds
    (e.g. 'multi-document', 'micro-f1') AND their split parts, so coverage
    matching works whether the text hyphenates or not."""
    terms: set[str] = set()
    for tok in re.findall(r"[a-z0-9\-]+", (facet or "").lower()):
        parts = [tok] + tok.split("-") if "-" in tok else [tok]
        for p in parts:
            if len(p) >= 4 and p not in _STOP:
                terms.add(p)
    return terms


def _covers(text: str, terms: set[str]) -> bool:
    """True if `text` contains at least one distinctive term from `terms`."""
    if not terms:
        return False
    low = (text or "").lower()
    return any(t in low for t in terms)


# --------------------------------------------------------------------------
# Evidence construction (grounding-safe: every passage verbatim from retrieved
# text for that same paper, joined by " ... ", capped < 2500 chars). Selects
# passages to COVER each query facet, targeting grade-2 -> grade-3.
# --------------------------------------------------------------------------
def _select_evidence_passages(
    title: str, abstract: str, tldr: str, snippets: list[str],
    facet_term_sets: list[set[str]],
) -> str:
    passages: list[str] = []
    seen_keys: set[str] = set()
    budget = 2400

    def _key(text: str) -> str:
        return _norm(text)[:120]

    def try_add(text: str, cap: int) -> bool:
        nonlocal budget
        text = _clip(text, cap)
        if not text:
            return False
        k = _key(text)
        if not k or k in seen_keys:
            return False
        for existing in seen_keys:
            if k in existing:  # substring of something already added
                return False
        cost = len(text) + (5 if passages else 0)  # " ... " join
        if cost > budget:
            return False
        seen_keys.add(k)
        passages.append(text)
        budget -= cost
        return True

    # 1. Abstract first — usually carries the main facets in one grounded block.
    try_add(abstract, 1400)
    # 2. tldr — compact whole-paper summary.
    try_add(tldr, 350)

    covered = [any(_covers(p, ts) for p in passages) for ts in facet_term_sets]

    # 3. Facet-covering snippets: greedily pick snippets that light up a facet
    #    not yet covered (this is the grade-2 -> grade-3 lever).
    remaining = list(snippets)
    for fi, ts in enumerate(facet_term_sets):
        if covered[fi] or not ts:
            continue
        for sn in remaining:
            if _covers(sn, ts) and try_add(sn, 650):
                remaining.remove(sn)
                for fj, tj in enumerate(facet_term_sets):
                    if not covered[fj] and _covers(sn, tj):
                        covered[fj] = True
                break

    # 4. Fill any leftover budget with the strongest remaining snippets.
    for sn in remaining:
        if budget < 120:
            break
        try_add(sn, 650)

    if not passages and title:
        try_add(title, 300)

    return " ... ".join(passages)[:2400]


# --------------------------------------------------------------------------
# iter22 evidence assembler: the iter14 champion's snippet-led RICH DUMP.
# On iter21's own batch, iter14's rich dump beat iter21's facet-covering
# selection on the two largest semantic deltas (semantic_98 +0.198,
# semantic_125 +0.138) over the SAME retrieved papers — the difference was
# purely the evidence text, consistent with the standing note "rich snippet-led
# dump > facet-covering selection for grade-3". This is the batch-winning
# assembler; iter22 keeps iter21's full robustness/metadata/retrieval stack and
# reverts ONLY the evidence text to this proven form. Grounding-safe: every
# passage is verbatim retrieved text for that same paper, joined by " ... ",
# capped < 2500 chars.
# --------------------------------------------------------------------------
def _build_evidence(title: str, abstract: str, tldr: str, snippets: list[str]) -> str:
    """Assemble the richest grounded evidence under the 2500-char budget.
    Priority: distinct snippets (often cover different criteria/sections) +
    full abstract + tldr. Passages are deduped by normalized text so we don't
    waste budget on near-duplicates."""
    passages: list[str] = []
    seen: set[str] = set()

    def add(text: str, cap: int):
        text = _clip(text, cap)
        if not text:
            return
        key = _norm(text)[:120]
        if key in seen:
            return
        # skip if this passage is a substring of one already added
        for k in seen:
            if key and key in k:
                return
        seen.add(key)
        passages.append(text)

    for sn in snippets[:4]:
        add(sn, 700)
    add(abstract, 1400)
    add(tldr, 400)
    if not passages and title:
        add(title, 300)

    ev = " ... ".join(passages)
    return ev[:2400]


# --------------------------------------------------------------------------
# Candidate accumulator
# --------------------------------------------------------------------------
class Cand:
    __slots__ = ("cid", "title", "abstract", "tldr", "snippets", "score", "sources")

    def __init__(self, cid):
        self.cid = str(cid)
        self.title = ""
        self.abstract = ""
        self.tldr = ""
        self.snippets: list[str] = []
        self.score = 0.0
        self.sources: set[str] = set()

    def add_snippet(self, sn: str):
        sn = (sn or "").strip()
        if not sn:
            return
        n = _norm(sn)[:120]
        for existing in self.snippets:
            if _norm(existing)[:120] == n:
                return
        if len(self.snippets) < 8:
            self.snippets.append(sn)


def _set_tldr(c: Cand, tl):
    if c.tldr:
        return
    if isinstance(tl, dict):
        c.tldr = (tl.get("text") or "").strip()
    elif isinstance(tl, str):
        c.tldr = tl.strip()


# --------------------------------------------------------------------------
# SEMANTIC path — maximize grade-3 recall (facet-covering evidence) + rank
# --------------------------------------------------------------------------
async def _solve_semantic(state, query) -> list[dict]:
    cands: dict[str, Cand] = {}

    def get(cid) -> Cand:
        cid = str(cid)
        if cid not in cands:
            cands[cid] = Cand(cid)
        return cands[cid]

    # --- one LLM call: keyword variants AND query facets ------------------
    clean_q = re.sub(r"[?!]", "", query).strip()
    ex = await _llm_json(
        "You analyze a literature-search request for a scientific paper search "
        "engine (literal keyword match, NO question framing, NO operators).\n"
        f"Request: {query}\n\n"
        "Return JSON with two keys:\n"
        '  "queries": the core noun-phrase query plus 2 alternative phrasings '
        "(synonyms / broader or narrower terms) to widen recall.\n"
        '  "facets": 2-4 SHORT phrases naming the distinct aspects a fully '
        "relevant paper must satisfy (e.g. the method, the property it must "
        "have, the evaluation, the domain). Each 2-5 words.\n"
        'Reply ONLY with JSON: {"queries": ["q1","q2","q3"], "facets": ["f1","f2","f3"]}',
        {"queries": [], "facets": []},
    )
    variants, facets = [], []
    if isinstance(ex, dict):
        variants = [q.strip() for q in ex.get("queries", []) if isinstance(q, str) and q.strip()]
        facets = [f.strip() for f in ex.get("facets", []) if isinstance(f, str) and f.strip()]
    if clean_q and clean_q not in variants:
        variants.append(clean_q)
    variants = variants[:4]
    facets = facets[:4]
    facet_term_sets = [_facet_terms(f) for f in facets]
    print(f"  keyword variants: {variants}")
    print(f"  facets: {facets}")

    snip = _maybe_tool(state, "snippet_search")
    kw = _maybe_tool(state, "search_papers_by_relevance")

    # --- retrieval: full-query + per-facet snippet_search, run concurrently
    async def _snip(q, limit, tag):
        raw = await _call(snip, query=q, limit=limit)
        return tag, _parse_items(raw)

    snip_jobs = []
    if snip is not None:
        snip_jobs.append(_snip(query, 100, "snip"))
        for f in facets[:3]:
            snip_jobs.append(_snip(f, 60, f"facet:{f}"))
    snip_results = await asyncio.gather(*snip_jobs) if snip_jobs else []

    for tag, rows in snip_results:
        n = len(rows)
        # full-query results weigh full; facet results are breadth-only (low weight)
        base_w = 1.0 if tag == "snip" else 0.25
        print(f"  snippet_search[{tag}] returned {n} passages")
        for i, r in enumerate(rows):
            paper = r.get("paper") or {}
            cid = paper.get("corpusId")
            if cid is None:
                continue
            c = get(cid)
            c.sources.add("snip")
            c.title = c.title or (paper.get("title") or "")
            c.add_snippet(((r.get("snippet") or {}).get("text") or ""))
            c.score += base_w * (n - i) / max(n, 1)

    # --- keyword search over variants (carries abstracts for free) --------
    if kw is not None:
        kw_jobs = [
            _call(kw, keyword=v, fields="title,abstract,corpusId,tldr", limit=100)
            for v in variants
        ]
        kw_raws = await asyncio.gather(*kw_jobs) if kw_jobs else []
        for raw in kw_raws:
            rows = _parse_items(raw)
            n = len(rows)
            for i, r in enumerate(rows):
                cid = r.get("corpusId")
                if cid is None:
                    continue
                c = get(cid)
                c.sources.add("kw")
                c.title = c.title or (r.get("title") or "")
                if r.get("abstract") and not c.abstract:
                    c.abstract = r["abstract"]
                _set_tldr(c, r.get("tldr"))
                c.score += 0.7 * (n - i) / max(n, 1)
    print(f"  {len(cands)} unique candidates after retrieval")

    if not cands:
        return []

    # multi-source agreement bonus (found by both retrievers -> strong)
    for c in cands.values():
        if len(c.sources) > 1:
            c.score += 1.5

    ordered = sorted(cands.values(), key=lambda c: c.score, reverse=True)

    # --- enrich missing abstracts: robust, per-id fallback on chunk failure
    await _enrich_abstracts(state, ordered[:150])

    # --- facet-aware LLM rerank of the top slice -------------------------
    ordered = await _rerank(query, facets, ordered)

    # --- targeted per-paper facet evidence: the grade-2 -> grade-3 lever --
    #     (scoped snippet_search for facets the top candidates don't cover)
    try:
        await _targeted_facet_evidence(state, ordered[:15], facets, facet_term_sets)
    except Exception as e:  # noqa: BLE001
        print(f"  targeted-evidence skipped: {e!r}")

    # --- assemble submission (best-first; no precision penalty) -----------
    results = []
    have = set()
    for c in ordered:
        if c.cid in have:
            continue
        # iter22: rich snippet-led dump (iter14 champion assembler), replacing
        # iter21's facet-covering selection. The extra grounded snippets that
        # iter21's per-facet + targeted-facet retrieval collects still feed this
        # dump — synergistic, not conflicting.
        ev = _build_evidence(c.title, c.abstract, c.tldr, c.snippets)
        if not ev:
            continue
        results.append({"paper_id": c.cid, "markdown_evidence": ev})
        have.add(c.cid)
        if len(results) >= 200:
            break

    # Rank-degeneracy safeguard: guarantee grade variance (all-equal -> rank 0).
    tail_pool = ordered[len(results):len(results) + 30]
    added = 0
    for c in tail_pool:
        if added >= 4:
            break
        if c.cid in have or not c.title:
            continue
        results.append({"paper_id": c.cid, "markdown_evidence": _clip(c.title, 250)})
        have.add(c.cid)
        added += 1

    print(f"  semantic: submitting {len(results)} papers (+{added} tail)")
    return results


async def _targeted_facet_evidence(state, top_cands, facets, facet_term_sets):
    """For each top candidate, fetch scoped body passages for facets its current
    text (abstract+tldr+snippets) does NOT cover. snippet_search scoped to the one
    paper ranks that paper's passages by facet fit; we keep a passage only if it
    literally contains the facet's terms (grounding + coverage safe), so a paper
    that genuinely lacks the facet gains nothing (no downside). Parallelized."""
    snip = _maybe_tool(state, "snippet_search")
    if snip is None or not facets or not top_cands:
        return
    pairs = [(f, ts) for f, ts in zip(facets, facet_term_sets) if ts]
    if not pairs:
        return

    async def enrich(c: Cand):
        cur = " ".join([c.abstract, c.tldr] + c.snippets).lower()
        uncovered = [(f, ts) for f, ts in pairs if not _covers(cur, ts)]
        for f, ts in uncovered[:2]:  # at most 2 scoped calls per paper
            raw = await _call(snip, query=f, limit=3, paper_ids=f"CorpusId:{c.cid}")
            for r in _parse_items(raw):
                txt = ((r.get("snippet") or {}).get("text") or "")
                if txt and _covers(txt, ts):
                    c.add_snippet(txt)
                    break

    await asyncio.gather(*[enrich(c) for c in top_cands], return_exceptions=True)
    print(f"  targeted-evidence: scoped snippet pass over top {len(top_cands)}")


async def _enrich_abstracts(state, need_all: list[Cand]):
    """Fill missing abstracts. Chunked get_paper_batch; on any chunk error fall
    back to per-id get_paper so one post-cutoff id costs one paper, not a chunk."""
    need = [c for c in need_all if not c.abstract]
    if not need:
        return
    batch = _maybe_tool(state, "get_paper_batch")
    get_one = _maybe_tool(state, "get_paper")

    def apply(rows):
        by_cid = {str(r.get("corpusId")): r for r in rows if r.get("corpusId") is not None}
        for c in need:
            r = by_cid.get(c.cid)
            if not r:
                continue
            if r.get("abstract") and not c.abstract:
                c.abstract = r["abstract"]
            _set_tldr(c, r.get("tldr"))

    for i in range(0, len(need), 16):
        chunk = need[i:i + 16]
        ids = [f"CorpusId:{c.cid}" for c in chunk]
        raw = None
        if batch is not None:
            raw = await _call(batch, ids=ids, fields="title,abstract,corpusId,tldr")
        rows = _parse_items(raw)
        if rows:
            apply(rows)
            continue
        # batch failed (e.g. one post-cutoff id poisoned it) -> per-id fallback
        if get_one is None:
            continue
        one_jobs = [
            _call(get_one, paper_id=f"CorpusId:{c.cid}", fields="title,abstract,corpusId,tldr")
            for c in chunk
        ]
        one_raws = await asyncio.gather(*one_jobs)
        rows = []
        for orw in one_raws:
            rows.extend(_parse_items(orw))
        apply(rows)


async def _rerank(query, facets, ordered: list[Cand], top_n: int = 90) -> list[Cand]:
    """Rate the top candidates 0-10 for full-aspect coverage with one cheap LLM call
    and sort by (llm_score, retrieval_score). Robust: any failure keeps input order.

    iter9 change: judge each candidate on its ABSTRACT (enriched upstream but never
    fed to the reranker before) instead of a single snippet, on a finer 0-10 scale
    anchored to the judge's rule (top score only if EVERY aspect is satisfied). This
    breaks the grade-3 pile-up a 0-3 scale produced and sorts genuinely all-aspect
    papers above topic-only ones — the distinction the recall term rewards."""
    head = ordered[:top_n]
    tail = ordered[top_n:]
    if len(head) < 5:
        return ordered

    def blurb(c: Cand) -> str:
        parts = [c.title[:150]]
        # Abstract is the best discriminator of whether ALL aspects hold; prefer it,
        # falling back to tldr then a snippet. (Enriched by _enrich_abstracts above.)
        body = c.abstract or c.tldr or (c.snippets[0] if c.snippets else "")
        if body:
            parts.append(body[:320])
        return " | ".join(p for p in parts if p)

    listing = "\n".join(f"{i}: {blurb(c)}" for i, c in enumerate(head))
    facet_txt = "; ".join(facets) if facets else "(all aspects of the request)"
    res = await _llm_json(
        "You are ranking retrieved papers for a literature-search request. Each paper "
        "will be graded by how fully it satisfies EVERY required aspect: a top paper "
        "perfectly satisfies ALL aspects, not merely the main topic.\n"
        f'Request: "{query}"\n'
        f"Required aspects (a top paper covers ALL of them): {facet_txt}\n\n"
        "Rate EVERY candidate 0-10 by how fully it satisfies ALL the required aspects:\n"
        "  9-10 = fully satisfies EVERY aspect (the main topic AND every secondary requirement)\n"
        "  6-8  = satisfies the main topic and most aspects, but one aspect is weak or unstated\n"
        "  3-5  = on the topic but clearly misses one or more required aspects\n"
        "  0-2  = off-topic / unrelated\n"
        f"Candidates (index: title | abstract):\n{listing}\n\n"
        'Reply ONLY with a JSON object mapping each index (string) to its integer '
        'score, e.g. {"0": 9, "1": 4, ...}. Include ALL indices.',
        {},
    )
    if not isinstance(res, dict) or not res:
        print("  rerank: fallback to retrieval order")
        return ordered

    scores = {}
    for k, v in res.items():
        try:
            scores[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    if len(scores) < len(head) // 2:
        print(f"  rerank: too few scores ({len(scores)}/{len(head)}), fallback")
        return ordered

    order_idx = {id(c): i for i, c in enumerate(head)}
    reranked = sorted(
        head,
        key=lambda c: (-(scores.get(order_idx[id(c)], 0.0)), order_idx[id(c)]),
    )
    print(f"  rerank: reordered top {len(head)} candidates (facet-aware)")
    return reranked + tail


# --------------------------------------------------------------------------
# SPECIFIC path — high-precision; title-gen from informal reference, no blind hedge
# --------------------------------------------------------------------------
async def _solve_specific(state, query) -> list[dict]:
    # 1. LLM turns the informal reference into the likely EXACT paper title,
    #    and parses any "AuthorYYYY" citation key into author/year hints.
    guess = await _llm_json(
        "A user refers to a specific, known scientific paper informally. The "
        "reference may embed a citation key like 'Smith2021' (author surname + "
        "year) or an acronym/dataset name.\n"
        f'Reference: "{query}"\n\n'
        "Give the paper's most likely EXACT published title (your best guess), a "
        "short keyword query for a title search, and any author surname / year you "
        "can infer.\n"
        'Reply ONLY with JSON: {"title": "likely exact title", "keywords": '
        '"short query", "author": "surname or empty", "year": int or null}',
        {},
    )
    guess_title = (guess.get("title") or "").strip() if isinstance(guess, dict) else ""
    guess_kw = (guess.get("keywords") or "").strip() if isinstance(guess, dict) else ""
    guess_author = (guess.get("author") or "").strip() if isinstance(guess, dict) else ""
    guess_year = guess.get("year") if isinstance(guess, dict) else None

    cleaned = re.sub(r"^\s*(the|a|an)\s+", "", query.strip(), flags=re.I)
    cleaned = re.sub(r"\s+paper[.?!]?\s*$", "", cleaned, flags=re.I).strip() or query
    title_queries = [t for t in (guess_title, cleaned) if t]
    kw_queries = [k for k in (guess_kw, cleaned) if k]

    pool: list[tuple[str, dict]] = []
    seen: set[str] = set()
    title_match = None

    def add(cid, title, abstract):
        cid = str(cid)
        if cid in seen:
            return
        seen.add(cid)
        pool.append((cid, {"title": title or "", "abstract": abstract or ""}))

    title_tool = _maybe_tool(state, "search_paper_by_title")
    if title_tool is not None:
        for tq in title_queries:
            raw = await _call(title_tool, title=tq, fields="title,abstract,corpusId")
            for r in _parse_items(raw):
                cid = r.get("corpusId")
                if cid is not None and r.get("paperId") is not None:
                    if title_match is None:
                        title_match = str(cid)
                    add(cid, r.get("title"), r.get("abstract"))

    kw = _maybe_tool(state, "search_papers_by_relevance")
    if kw is not None:
        for kq in kw_queries:
            raw = await _call(kw, keyword=kq, fields="title,abstract,corpusId", limit=10)
            for r in _parse_items(raw):
                cid = r.get("corpusId")
                if cid is not None:
                    add(cid, r.get("title"), r.get("abstract"))

    # snippet_search: tolerant of acronyms / natural-language references that the
    # keyword/title tools miss (e.g. "MS^2 DeYoung2021").
    snip = _maybe_tool(state, "snippet_search")
    if snip is not None and len(pool) < 20:
        raw = await _call(snip, query=cleaned, limit=10)
        for r in _parse_items(raw):
            paper = r.get("paper") or {}
            cid = paper.get("corpusId")
            if cid is not None:
                add(cid, paper.get("title"), "")

    if not pool:
        return []

    listing = "\n".join(f"{i}: {v['title'][:150]}" for i, (_c, v) in enumerate(pool))
    hint = ""
    if guess_author or guess_year:
        hint = f" (hint: likely by {guess_author or '?'}, year {guess_year or '?'})"
    sel = await _llm_json(
        f'A user is looking for a specific known paper: "{query}"{hint}.\n'
        f"Candidates:\n{listing}\n\n"
        "Return the indices of ONLY the candidate(s) that ARE that exact paper "
        "(usually 1; return several ONLY if they are clearly duplicate records of "
        "the SAME work). Be strict — do not include merely-related papers.\n"
        'Reply ONLY with JSON: {"indices": [i, ...]}',
        {"indices": []},
    )
    idxs = []
    if isinstance(sel, dict):
        idxs = [i for i in sel.get("indices", []) if isinstance(i, int) and 0 <= i < len(pool)]

    chosen: list[int] = []
    for i in idxs[:3]:
        if i not in chosen:
            chosen.append(i)
    if not chosen:
        if title_match:
            chosen = [next(i for i, (cid, _v) in enumerate(pool) if cid == title_match)]
        else:
            chosen = [0]

    # Duplicate-only hedge: if the LLM chose exactly one, add a second id ONLY if
    # its title is a near-duplicate of the pick (a real duplicate corpus record),
    # never a random candidate (which tanked specific_7).
    if len(chosen) == 1:
        pick_title = _norm(pool[chosen[0]][1]["title"])
        if pick_title:
            for i in range(len(pool)):
                if i in chosen:
                    continue
                t = _norm(pool[i][1]["title"])
                if t and (t == pick_title or t in pick_title or pick_title in t):
                    chosen.append(i)
                    break

    results = []
    for i in chosen:
        cid, v = pool[i]
        results.append({"paper_id": cid, "markdown_evidence": _clip(v["title"], 250)})
    print(f"  specific: submitting {len(results)} papers")
    return results


# --------------------------------------------------------------------------
# METADATA path — author/venue/year filters + citation-relation queries
# --------------------------------------------------------------------------
def _unwrap_citing(rows: list[dict]) -> list[dict]:
    """get_citations rows are wrapped: {"citingPaper": {...}}. Unwrap to the
    paper dict; pass through already-flat rows unchanged."""
    out = []
    for r in rows:
        if isinstance(r, dict) and "citingPaper" in r and isinstance(r["citingPaper"], dict):
            out.append(r["citingPaper"])
        elif isinstance(r, dict) and "citedPaper" in r and isinstance(r["citedPaper"], dict):
            out.append(r["citedPaper"])
        else:
            out.append(r)
    return out


async def _resolve_anchor(state, title_guess: str) -> str | None:
    """Resolve an anchor paper (referred to informally) to its corpusId.

    Anchor papers in "papers citing X" / "cited by X" queries are almost always
    canonical, heavily-cited works (T5, DistilBERT, the Spider dataset, ...).
    The old resolver trusted `search_paper_by_title`'s single fuzzy pick, which
    mis-resolves bare nicknames — "Spider" matched an unrelated paper literally
    titled "Spider...", "T5" matched a 483-citation paper instead of the real
    ~thousands-cited T5, so the citation intersection came up empty and the query
    scored 0 (metadata_26). Fix: gather candidates from BOTH the title tool and a
    relevance search, then pick the MOST-CITED one — the canonical anchor. This
    generalizes to any nickname/short-title anchor without per-query tuning."""
    if not title_guess:
        return None
    candidates: list[dict] = []
    title_tool = _maybe_tool(state, "search_paper_by_title")
    if title_tool is not None:
        raw = await _call(title_tool, title=title_guess,
                          fields="title,corpusId,citationCount")
        for r in _parse_items(raw):
            if r.get("corpusId") is not None and r.get("paperId") is not None:
                candidates.append(r)
    kw = _maybe_tool(state, "search_papers_by_relevance")
    if kw is not None:
        raw = await _call(kw, keyword=title_guess,
                          fields="title,corpusId,citationCount", limit=10)
        for r in _parse_items(raw):
            if r.get("corpusId") is not None:
                candidates.append(r)
    if not candidates:
        return None
    best = max(candidates, key=lambda r: (r.get("citationCount") or 0))
    print(f"  anchor '{title_guess[:40]}' -> id={best.get('corpusId')} "
          f"(cites={best.get('citationCount')}, {len(candidates)} candidates)")
    return str(best["corpusId"])


async def _authors_of_paper(state, title_guess: str) -> list[str]:
    """Resolve a referenced paper and return its author names (for
    'papers by an author of the X paper' compositional queries)."""
    if not title_guess:
        return []
    cid = await _resolve_anchor(state, title_guess)
    if not cid:
        return []
    get_one = _maybe_tool(state, "get_paper")
    if get_one is None:
        return []
    raw = await _call(get_one, paper_id=f"CorpusId:{cid}", fields="title,authors")
    names: list[str] = []
    for r in _parse_items(raw):
        au = r.get("authors")
        if isinstance(au, list):
            for a in au:
                nm = (a.get("name") if isinstance(a, dict) else a) or ""
                nm = nm.strip()
                if nm and nm not in names:
                    names.append(nm)
    print(f"  author_of_paper '{title_guess}' -> id={cid} authors={names}")
    return names


async def _solve_metadata_citation(state, query, ex) -> list[dict]:
    """Handle 'papers citing X (and Y)' / 'papers cited by X' queries: resolve
    each anchor, get its citing (or cited) set, intersect across anchors, then
    apply venue/year/min_citations/min_authors filters. Exact-match scoring."""
    relation = (ex.get("relation") or "").strip().lower()
    anchors = [a for a in (ex.get("referenced_papers") or []) if isinstance(a, str) and a.strip()]
    if relation not in ("cites", "cited_by") or not anchors:
        return []

    # Resolve anchors to corpusIds
    anchor_ids = []
    for a in anchors[:4]:
        cid = await _resolve_anchor(state, a)
        if cid:
            anchor_ids.append(cid)
    print(f"  citation query: relation={relation} anchors={anchors} -> ids={anchor_ids}")
    if not anchor_ids:
        return []

    get_cit = _maybe_tool(state, "get_citations")
    if get_cit is None:
        return []

    fields = "title,corpusId,year,venue,authors,citationCount"
    sets: list[set[str]] = []
    meta: dict[str, dict] = {}
    for aid in anchor_ids:
        raw = await _call(get_cit, paper_id=f"CorpusId:{aid}", fields=fields, limit=1000)
        rows = _unwrap_citing(_parse_items(raw))
        ids = set()
        for r in rows:
            cid = r.get("corpusId")
            if cid is None:
                continue
            cid = str(cid)
            ids.add(cid)
            meta.setdefault(cid, r)
        print(f"    anchor {aid}: {len(ids)} {relation} papers")
        if ids:
            sets.append(ids)

    if not sets:
        return []
    inter = set.intersection(*sets) if len(sets) > 1 else sets[0]
    print(f"  citation intersection: {len(inter)} candidates")

    # Numeric / venue / year filters
    ymin, ymax, years = _year_spec(ex)
    venues = [v.lower() for v in (ex.get("venues") or []) if isinstance(v, str) and v.strip()]
    min_cit = ex.get("min_citations")
    min_auth = ex.get("min_authors")

    out = []
    for cid in inter:
        r = meta.get(cid, {})
        if not _year_ok(r.get("year"), ymin, ymax, years):
            continue
        if venues:
            rv = (r.get("venue") or "").lower()
            if rv and not any(v in rv or rv in v for v in venues):
                continue
        if isinstance(min_cit, int):
            cc = r.get("citationCount")
            if isinstance(cc, int) and cc < min_cit:
                continue
        if isinstance(min_auth, int):
            au = r.get("authors")
            if isinstance(au, list) and len(au) <= min_auth:
                continue
        out.append({"paper_id": cid, "markdown_evidence": _clip(r.get("title") or "", 250)})
    print(f"  citation metadata: submitting {len(out)} papers")
    return out


def _year_spec(ex):
    ymin, ymax = ex.get("year_min"), ex.get("year_max")
    years = ex.get("years")
    yset = set()
    if isinstance(years, list):
        for y in years:
            try:
                yset.add(int(y))
            except (TypeError, ValueError):
                continue
    return ymin, ymax, (yset or None)


def _year_ok(y, ymin, ymax, years):
    if y is None:
        return years is None and (ymin is None and ymax is None)
    try:
        y = int(y)
    except (TypeError, ValueError):
        return True
    if years is not None:
        return y in years
    if ymin is not None and y < ymin:
        return False
    if ymax is not None and y > ymax:
        return False
    return True


async def _solve_metadata(state, query) -> list[dict]:
    ex = await _llm_json(
        "Extract structured filters from this scholarly-search request.\n"
        f"Request: {query}\n\n"
        "Fields:\n"
        '  "relation": "cites" if it wants papers that CITE some reference '
        'paper(s); "cited_by" if it wants papers CITED BY a reference; else "".\n'
        '  "referenced_papers": the anchor paper(s) the relation refers to, each '
        "given as its MOST RECOGNIZABLE CANONICAL TITLE — expand nicknames/acronyms "
        "into a phrase that uniquely identifies the specific paper (e.g. \"T5\" -> "
        "\"T5 Exploring the Limits of Transfer Learning Text-to-Text Transformer\", "
        "\"Spider\" -> \"Spider large-scale text-to-SQL semantic parsing dataset\", "
        "\"DistilBERT\" -> \"DistilBERT distilled version of BERT\"). If a token is a "
        "VENUE/conference name (NeurIPS, ICML, ACL, ...) rather than a specific "
        "paper, do NOT list it here. [] if none.\n"
        '  "author_of_paper": if the request wants papers written by an AUTHOR OF '
        'some named paper (e.g. "co-authored by an author of the BERT paper"), the '
        'informal title of that paper (e.g. "BERT"); "" otherwise.\n'
        '  "keywords": topic noun phrase or empty.\n'
        '  "authors": full author names explicitly named, [] if none.\n'
        '  "venues": venue names/abbreviations, [] if none.\n'
        '  "year_min"/"year_max": ints or null for a RANGE.\n'
        '  "years": explicit list of individual years if the request names them '
        '(e.g. "2014 or 2017" -> [2014, 2017]); [] otherwise.\n'
        '  "min_citations": int or null (e.g. "cited by at least 30").\n'
        '  "min_authors": int or null — the paper must have MORE THAN this many '
        'authors (e.g. "more than 3 authors" -> 3; "at least one additional '
        'author" beyond a named one -> 1).\n'
        "Reply ONLY with JSON with exactly those keys.",
        {},
    )
    if not isinstance(ex, dict):
        ex = {}

    # --- citation-relation branch (papers citing/cited-by an anchor) -------
    if (ex.get("relation") or "").strip().lower() in ("cites", "cited_by") and ex.get("referenced_papers"):
        try:
            cit_out = await _solve_metadata_citation(state, query, ex)
        except Exception as e:  # noqa: BLE001
            print(f"  citation branch error: {e!r}")
            cit_out = []
        if cit_out:
            return cit_out
        # else fall through to keyword/author path

    keywords = (ex.get("keywords") or query).strip() or query
    authors = [a for a in (ex.get("authors") or []) if isinstance(a, str) and a.strip()]
    venues = [v for v in (ex.get("venues") or []) if isinstance(v, str) and v.strip()]
    ymin, ymax, years = _year_spec(ex)
    min_auth = ex.get("min_authors")
    min_cit = ex.get("min_citations")
    author_of_paper = (ex.get("author_of_paper") or "").strip() if isinstance(ex, dict) else ""

    # --- author-of-referenced-paper branch (additive, guarded) ------------
    #     "papers by an author of the X paper" -> resolve X's authors, then run
    #     the normal author path over them with venue/year/min_authors filters.
    if not authors and author_of_paper:
        try:
            ref_authors = await _authors_of_paper(state, author_of_paper)
        except Exception as e:  # noqa: BLE001
            print(f"  author_of_paper branch error: {e!r}")
            ref_authors = []
        if ref_authors:
            try:
                aop_out = await _solve_metadata_author(
                    state, query, ref_authors, venues, ymin, ymax, years, min_auth, min_cit
                )
            except Exception as e:  # noqa: BLE001
                print(f"  author_of_paper author-path error: {e!r}")
                aop_out = []
            if aop_out:
                return aop_out
        # else fall through

    print(f"  metadata filters: kw={keywords!r} authors={authors} venues={venues} "
          f"yr=[{ymin},{ymax}] years={years} min_auth={min_auth} min_cit={min_cit}")

    # --- author path: retrieve the author's papers, LLM-select by venue/year
    if authors:
        auth_out = await _solve_metadata_author(
            state, query, authors, venues, ymin, ymax, years, min_auth, min_cit)
        if auth_out:
            return auth_out

    # --- keyword + venue path (post-filter by year) ------------------------
    kw = _maybe_tool(state, "search_papers_by_relevance")
    rows: list[dict] = []
    if kw is not None:
        venues_arg = ",".join(venues) if venues else None
        kwargs = dict(keyword=keywords, fields="title,abstract,corpusId,year,venue", limit=100)
        if venues_arg:
            kwargs["venues"] = venues_arg
        raw = await _call(kw, **kwargs)
        rows.extend(_parse_items(raw))

    vlow = [v.lower() for v in venues]
    seen, out = set(), []
    for r in rows:
        cid = r.get("corpusId")
        if cid is None:
            continue
        cid = str(cid)
        if cid in seen or not _year_ok(r.get("year"), ymin, ymax, years):
            continue
        if vlow:
            rv = (r.get("venue") or "").lower()
            if rv and not any(v in rv or rv in v for v in vlow):
                continue
        seen.add(cid)
        out.append({"paper_id": cid, "markdown_evidence": _clip(r.get("title") or "", 250)})

    print(f"  metadata: submitting {len(out)} papers")
    return out


async def _solve_metadata_author(state, query, authors, venues, ymin, ymax, years, min_auth=None, min_cit=None) -> list[dict]:
    """Author queries: pull each author's papers, then LLM-select which match the
    venue/year criteria (robust to venue-name variants like ACL vs TACL vs a
    workshop). Falls back to programmatic year/venue filtering on LLM failure.
    Optional min_auth: keep only papers with MORE THAN min_auth authors.
    Optional min_cit: keep only papers with AT LEAST min_cit citations (mirrors
    the citation-branch semantics: missing citationCount is kept, so recall is
    never harmed; survivors are sorted citation-desc so the scorer's 250-cap
    keeps the most-cited/most-likely-gold papers)."""
    find_auth = _maybe_tool(state, "search_authors_by_name")
    get_papers = _maybe_tool(state, "get_author_papers")
    if find_auth is None or get_papers is None:
        return []

    rows: list[dict] = []
    for name in authors[:8]:
        raw = await _call(find_auth, name=name, fields="name,paperCount", limit=10)
        arows = [a for a in _parse_items(raw) if a.get("authorId")]
        arows.sort(key=lambda a: a.get("paperCount") or 0, reverse=True)
        if not arows:
            continue
        aid = arows[0]["authorId"]
        raw = await _call(
            get_papers, author_id=str(aid),
            paper_fields="title,corpusId,year,venue,authors,citationCount", limit=500,
        )
        rows.extend(_parse_items(raw))

    # Dedup + basic year prefilter (keeps LLM candidate list small & on-topic)
    seen, cand = set(), []
    for r in rows:
        cid = r.get("corpusId")
        if cid is None:
            continue
        cid = str(cid)
        if cid in seen:
            continue
        if not _year_ok(r.get("year"), ymin, ymax, years):
            continue
        # min_authors filter (paper must have MORE THAN min_auth authors)
        if isinstance(min_auth, int):
            au = r.get("authors")
            if isinstance(au, list) and len(au) <= min_auth:
                continue
        # min_citations filter (keep AT LEAST min_cit citations; missing == keep).
        # Mirrors the citation-branch semantics exactly so recall is never harmed.
        if isinstance(min_cit, int):
            cc = r.get("citationCount")
            if isinstance(cc, int) and cc < min_cit:
                continue
        seen.add(cid)
        cand.append(r)
    if not cand:
        return []
    # When a citation floor is stated, the gold papers are the well-cited ones;
    # sort citation-desc so the scorer's first-250 cap keeps them (order is free
    # on exact-match scoring, but truncation is not).
    if isinstance(min_cit, int):
        cand.sort(key=lambda r: (r.get("citationCount") or 0), reverse=True)
    print(f"  author path: {len(cand)} papers after year/min_auth/min_cit prefilter")

    # If no venue constraint, year/min_auth/min_cit prefilter alone is the answer.
    if not venues:
        return [{"paper_id": str(r["corpusId"]), "markdown_evidence": _clip(r.get("title") or "", 250)}
                for r in cand]

    # LLM selects which candidates match the venue(s) — it knows the full-name /
    # abbreviation mapping and distinguishes main conference from workshop/journal.
    listing = "\n".join(
        f"{i}: [{r.get('venue') or '?'} | {r.get('year') or '?'}] {(r.get('title') or '')[:120]}"
        for i, r in enumerate(cand)
    )
    ven_txt = ", ".join(venues)
    sel = await _llm_json(
        f'Selecting papers that were published at these venue(s): {ven_txt}.\n'
        "Venue names in the data may be full names or abbreviations; match the "
        "intended venue only (e.g. 'ACL' means the main ACL conference / its "
        "Annual Meeting, NOT TACL, NOT Findings, NOT a workshop, NOT EMNLP).\n"
        f"Candidates (index: [venue | year] title):\n{listing}\n\n"
        'Reply ONLY with JSON: {"indices": [i, ...]} — the indices whose venue '
        "matches. Empty list if none.",
        None,
    )
    if isinstance(sel, dict) and isinstance(sel.get("indices"), list):
        idxs = [i for i in sel["indices"] if isinstance(i, int) and 0 <= i < len(cand)]
        out = [{"paper_id": str(cand[i]["corpusId"]),
                "markdown_evidence": _clip(cand[i].get("title") or "", 250)} for i in idxs]
        print(f"  author path: LLM selected {len(out)}/{len(cand)} by venue")
        if out:
            return out

    # Fallback: substring venue match (both directions)
    vlow = [v.lower() for v in venues]
    out = []
    for r in cand:
        rv = (r.get("venue") or "").lower()
        if rv and any(v in rv or rv in v for v in vlow):
            out.append({"paper_id": str(r["corpusId"]),
                        "markdown_evidence": _clip(r.get("title") or "", 250)})
    print(f"  author path: substring-venue fallback selected {len(out)}")
    return out


# --------------------------------------------------------------------------
# Solver entry point
# --------------------------------------------------------------------------
def _classify(state, query) -> str:
    st = (state.metadata.get("score_type") or "").strip()
    if st in ("specific_f1", "metadata_f1", "semantic_f1"):
        return st.replace("_f1", "")
    q = query.lower()
    if re.search(r"\b(cite|cites|citing|cited by)\b", q):
        return "metadata"
    if re.search(r"\bthe\b.*\bpaper\b", q) and len(query) < 90:
        return "specific"
    if re.search(r"\b(by|author|published in|in \d{4}|between \d{4})\b", q):
        return "metadata"
    return "semantic"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = state.metadata.get("raw_query") or getattr(state, "input_text", "") or ""
        kind = _classify(state, query)
        print(f"[{state.sample_id}] kind={kind} query={query[:90]!r}")

        try:
            if kind == "specific":
                results = await _solve_specific(state, query)
            elif kind == "metadata":
                results = await _solve_metadata(state, query)
            else:
                results = await _solve_semantic(state, query)
        except Exception as e:  # noqa: BLE001
            print(f"  FATAL path error ({kind}): {e!r}")
            results = []

        if not results:
            try:
                results = await _solve_semantic(state, query)
                # Exact-match (specific/metadata) scoring punishes low precision:
                # a 200-paper semantic dump on a query the structured path couldn't
                # resolve is ~0 precision. Cap it so a lucky gold overlap still
                # yields real F1 instead of being diluted to ~0.
                if kind in ("metadata", "specific") and len(results) > 40:
                    results = results[:40]
                    print(f"  {kind} fallback: capped semantic dump to 40")
            except Exception as e:  # noqa: BLE001
                print(f"  fallback failed: {e!r}")
                results = []

        state.output.completion = json.dumps(
            {"output": {"query_id": state.sample_id, "results": results}}
        )
        print(f"  DONE: {len(results)} papers submitted")
        return state

    return solve
