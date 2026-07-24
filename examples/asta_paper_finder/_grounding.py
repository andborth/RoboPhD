"""Evidence grounding for PaperFindingBench.

The benchmark's relevance judge (see evaluator.py header) grades each predicted
paper solely on the agent-supplied ``markdown_evidence`` — it never fetches the
paper, and it never checks that the corpus_id is real. That makes the score
fabricable: an agent can invent evidence for a paper it never retrieved (or for
a corpus_id that does not exist) and the judge has no way to know. Our runs
optimize an agent directly against this metric, so "fabricate persuasive
evidence" is a reachable point on the reward gradient.

This module closes the hole for our runs. Every evidence passage must be
verbatim-derivable from corpus text the agent actually received from the Asta
MCP tools during the same evaluation. Evidence that is not grounded is discarded
before it reaches the judge, so fabrication earns exactly what an empty
submission earns: a Not Relevant verdict — and no judge spend.

Scope and threading: the evaluator isolates each sample in its own subprocess,
so one process handles exactly one sample. The provenance store is therefore a
plain per-process global, populated by the tool wrappers (evaluator._build_tools)
as the agent retrieves papers and read by the patched ``get_llm_relevance``
during scoring. inspect runs tool calls on a single event loop (one thread), so
plain dict/set mutation is safe. The in-process ``evaluate`` path (used by unit
tests and smoke runs) reuses one process across samples, so callers must invoke
``reset()`` at the start of each evaluation.

Grounding is only consulted on semantic queries — the only query type the judge
scores. metadata/specific queries are scored by exact corpus_id match and never
read ``markdown_evidence``.
"""

import hashlib
import os
import re
import unicodedata
from typing import Any

# When set, cap TRAINING judging to the top-`estimate` submitted papers per
# semantic query (the recall-at-estimate depth). Papers past that depth cannot
# affect adjusted-F1 — recall reads only the top-estimate, and the rank term is
# empirically unchanged — so judging them is pure cost. main.py sets this for
# training and clears it for test/pristine (which judges all, matching official
# scoring). See _grounded_get_llm_relevance.
CAP_JUDGE_ENV = "PF_JUDGE_TOP_ESTIMATE"

# Per-paper evidence character cap (training-shaping knob, 2026-07-24).
# When set to a positive int, each paper's markdown_evidence is truncated
# to this many characters BEFORE grounding/judging. Rationale: agents
# have zero cost incentive toward evidence brevity (judge cost is
# invisible to their reward), and the -005 lineage wrote uniformly
# bloated evidence (mean 3,391 chars/paper) that tripled official
# judging bills. Enforced in TRAINING only (main.py sets it there and
# clears it for test evals) so evolution adapts by selecting/densifying;
# the evolved short-evidence behavior then travels into official
# submissions where no cap exists. Truncation-before-grounding is sound:
# a truncated substring of retrieved text still grounds. Verdict caches
# need no extra namespacing — keys hash the post-truncation text.
EVIDENCE_CAP_ENV = "PF_EVIDENCE_CHAR_CAP"

# corpus_id (normalized) -> set of normalized text spans retrieved for it.
_PROVENANCE: dict[str, set[str]] = {}

# Record of the most recent get_llm_relevance call, for the diagnostics layer.
# {"query_id": str, "judgements": {pid: verdict}, "blanked": [(pid, [passages], raw)]}
_LAST: dict[str, Any] = {}

# Keys under which the numeric CORPUS id appears in Asta MCP tool payloads.
# Deliberately excludes a bare "id" (collides with authorId etc.) AND S2's
# "paperId"/"paper_id" — those are HEX HASHES, not corpus ids, so _norm_cid
# would extract a spurious digit-run from the hash and register a phantom
# second cid for the paper. That phantom disables the single-cid vacuum branch
# in _walk, silently dropping every NESTED string field (notably tldr.text)
# from provenance and wrongly failing evidence that quotes it. corpusId is
# always present alongside paperId, so nothing is lost by ignoring paperId.
_CID_KEYS = ("corpusId", "corpus_id", "CorpusId", "corpusid")

# Passage separators: the contract's canonical ` ... ` plus the spaced dashes
# current agents use to join fields (" — ", " – ", " - "). Splitting on the
# dash too means an honest `title — abstract` submission is checked as two
# verbatim passages rather than one cross-field string that would never appear
# verbatim in any single retrieved payload. Intra-word hyphens ("large-scale")
# have no surrounding whitespace and are not split. Covers the ellipsis char
# and every dash variant so splitting matches regardless of the punctuation the
# agent used.
_PASSAGE_SPLIT = re.compile(r"\s+(?:\.\.\.|…|[—–‐−-])\s+")

# A dangling separator an agent left at a passage boundary — e.g. the seed's
# "Title —" when the abstract is empty, where the trailing " —" has no text
# after it to trigger a split. The joiner is not retrieved text, so without
# this the passage fails grounding on the trailing punctuation even though its
# content is verbatim. Stripping edge whitespace / dashes / dots / ellipsis
# only shortens the passage, so it can never break a passage that was already
# grounding — it only rescues one that a boundary artifact would have failed.
_PASSAGE_EDGE = re.compile(r"^[\s.…—–‐−-]+|[\s.…—–‐−-]+$")

# Caps how many DROPPED passages check_evidence reports per paper — a
# diagnostics-noise bound only. Passage COUNT is deliberately unenforced:
# the docs advise up to 8, but the task schema doesn't require it, in-band
# separators inside verbatim text (real ellipses/dashes) make a hard count
# ambiguous, and the anti-fabrication guarantee is the per-passage verbatim
# check below, not the count.
_MAX_DROPPED_REPORTED = 8


def _passages(raw: str) -> list[str]:
    """Split evidence into passages, stripping any dangling boundary separators
    and dropping empties."""
    parts = [_PASSAGE_EDGE.sub("", p) for p in _PASSAGE_SPLIT.split(raw)]
    return [p for p in parts if p]

# Punctuation NFKC does NOT fold — smart quotes and dashes — mapped to their
# ASCII forms so a passage that differs from retrieved text only in punctuation
# style still grounds. (The ellipsis char U+2026 already NFKC-decomposes to
# "...", so it needs no entry here.)
_PUNCT_FOLD = {
    "‘": "'", "’": "'", "‛": "'", "′": "'",  # single quotes / prime
    "“": '"', "”": '"', "‟": '"', "″": '"',  # double quotes
    "‐": "-", "‑": "-", "‒": "-", "–": "-",   # hyphen / dashes
    "—": "-", "―": "-", "−": "-",                  # em / horizontal bar / minus
}
_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT_FOLD))


def _fold_punct(s: str) -> str:
    return _PUNCT_RE.sub(lambda m: _PUNCT_FOLD[m.group()], s)


def reset() -> None:
    """Clear per-evaluation state. Call at the start of each sample."""
    _PROVENANCE.clear()
    _LAST.clear()


def _norm(s: str) -> str:
    """NFKC + punctuation-fold + casefold + whitespace-collapse, so matching
    ignores unicode presentation, smart-quote/dash style, case, and reflow
    differences without being lax about content."""
    s = _fold_punct(unicodedata.normalize("NFKC", str(s))).casefold()
    return re.sub(r"\s+", " ", s).strip()


def _norm_cid(x: Any) -> str | None:
    """Normalize a corpus id to its digit run (Asta corpus ids are numeric).
    Returns None if there is no digit content (e.g. an author id slipped in)."""
    m = re.search(r"\d+", str(x))
    return m.group(0) if m else None


def _direct_cid(node: dict) -> str | None:
    for k in _CID_KEYS:
        if k in node:
            c = _norm_cid(node[k])
            if c:
                return c
    return None


def _find_cids(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _CID_KEYS:
                c = _norm_cid(v)
                if c:
                    out.add(c)
            else:
                _find_cids(v, out)
    elif isinstance(node, list):
        for v in node:
            _find_cids(v, out)


def _all_strings(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            _all_strings(v, out)
    elif isinstance(node, list):
        for v in node:
            _all_strings(v, out)


def _add(cid: str, span: str) -> None:
    n = _norm(span)
    if n:
        _PROVENANCE.setdefault(cid, set()).add(n)


def _walk(node: Any) -> None:
    """Associate retrieved text with the corpus id it describes.

    A subtree containing exactly one distinct corpus id has *all* its strings
    vacuumed to that id — this captures fields that are siblings of the id
    rather than children (e.g. snippet_search's `{paper:{corpusId}, snippet:{text}}`,
    where the quotable snippet text sits beside the paper block). A dict with a
    direct corpus-id field also has its own string fields associated up front,
    so a paper's title/abstract survive even when it carries a nested list of
    other papers (e.g. get_citations)."""
    if isinstance(node, dict):
        dc = _direct_cid(node)
        if dc is not None:
            for v in node.values():
                if isinstance(v, str):
                    _add(dc, v)
        cids: set[str] = set()
        _find_cids(node, cids)
        if len(cids) == 1:
            only = next(iter(cids))
            strings: list[str] = []
            _all_strings(node, strings)
            for s in strings:
                _add(only, s)
            return
        for v in node.values():
            _walk(v)
    elif isinstance(node, list):
        for v in node:
            _walk(v)


def _ingest(node: Any, depth: int = 0) -> None:
    """Normalize whatever a tool returned into structured nodes and _walk them.

    The Asta MCP tools return a list of inspect `ContentText` objects, each
    `.text` a JSON string for one paper; other paths return dicts, lists, or
    JSON strings directly. This unwraps all of those to the underlying
    dict/list structure the corpus-id extractor understands."""
    if node is None or depth > 6:
        return
    if isinstance(node, str):
        import json
        try:
            _walk(json.loads(node))
        except (ValueError, TypeError):
            return  # opaque text with no per-paper structure to ground against
        return
    if isinstance(node, dict):
        _walk(node)
        return
    if isinstance(node, (list, tuple)):
        for x in node:
            _ingest(x, depth + 1)
        return
    # inspect content object (ContentText and friends): unwrap .text / .content.
    text = getattr(node, "text", None)
    if isinstance(text, str):
        _ingest(text, depth + 1)
        return
    content = getattr(node, "content", None)
    if content is not None and content is not node:
        _ingest(content, depth + 1)


def record_tool_result(result: Any) -> None:
    """Record the text payload returned by one Asta MCP tool call.

    Best-effort and never raises: a recording failure must not break the
    agent's tool call."""
    try:
        _ingest(result)
    except Exception:
        pass


def scrub_evidence(corpus_id: str, evidence: str) -> tuple[str, list[str]]:
    """Keep only the passages of `evidence` that are verbatim-derivable from text
    retrieved for `corpus_id`; drop the rest. Returns (scrubbed, dropped).

    A passage is grounded iff its normalized form is a substring of the
    normalized concatenation of everything retrieved for that paper — lenient
    enough for a genuine verbatim quote from any single field, strict enough
    that invented prose (absent from every retrieved payload) fails.

    Partial rather than all-or-nothing: a single ungrounded passage drops only
    itself, so two good passages beside one bad one are not zeroed. Return
    values:
      - empty evidence           → ("", [])            nothing claimed
      - nothing retrieved         → ("", [all passages]) unverifiable by construction
      - every passage grounded    → (evidence unchanged, [])
      - some passages grounded    → (grounded passages joined by ' ... ', [dropped])
      - no passage grounded       → ("", [all passages])
    """
    raw = (evidence or "").strip()
    if not raw:
        return "", []
    passages = _passages(raw)
    if not passages:
        return "", []          # evidence was only separators/punctuation
    cid = _norm_cid(corpus_id) or str(corpus_id)
    spans = _PROVENANCE.get(cid)
    if not spans:
        # Nothing was retrieved for this paper, yet the agent submitted evidence.
        return "", passages
    # sorted(), not just join(set): set iteration order varies across processes
    # (hash randomization), and a passage that straddles a join boundary would
    # otherwise ground on one run and fail on another. A passage within a single
    # span is unaffected; sorting only makes the rare cross-boundary case
    # deterministic. (The real anti-fabrication guarantee is per-span; boundary
    # matches are a lenient bonus for field-concatenated evidence.)
    blob = " ".join(sorted(spans))
    kept, dropped = [], []
    for p in passages:
        (kept if _norm(p) in blob else dropped).append(p)
    if not dropped:
        return raw, []                      # fully grounded: exact evidence intact
    return " ... ".join(kept), dropped      # kept may be empty → "" (fully dropped)


def check_evidence(corpus_id: str, evidence: str) -> tuple[bool, list[str]]:
    """Thin predicate over scrub_evidence: (fully_grounded, dropped_passages).
    `fully_grounded` is True iff nothing was dropped. Retained for callers/tests
    that only need the all-or-nothing answer."""
    _, dropped = scrub_evidence(corpus_id, evidence)
    return (not dropped), dropped[:_MAX_DROPPED_REPORTED]


def _evidence_hash(scrubbed: str) -> str:
    return hashlib.sha256(_norm(scrubbed).encode("utf-8")).hexdigest()[:16]


def cache_key(paper_id: str, scrubbed_evidence: str) -> str:
    """Composite cache subkey: same (query, paper) with different evidence gets a
    distinct verdict, so an evidence improvement is re-judged rather than
    inheriting an earlier submission's stale verdict. \\x1f is a control char
    that cannot appear in a corpus id, so the paper id remains recoverable."""
    return f"{paper_id}\x1f{_evidence_hash(scrubbed_evidence)}"


def paper_id_of(key: str) -> str:
    """Recover the corpus id from a composite cache key (or a legacy plain key)."""
    return key.split("\x1f", 1)[0]


def install_grounded_judge() -> None:
    """Replace astabench's ``get_llm_relevance`` with a grounded, evidence-hash
    -keyed version, in this process's namespaces only. Mirrors
    ``_install_safe_judge_cache``: patch both the origin binding in eval.py and
    the from-import binding in task.py that the scorer actually calls
    (task.py:24). Idempotent via a marker attribute. Official ``astabench eval``
    submissions run stock code in a separate process and never see this."""
    from astabench.evals.paper_finder import eval as _pf_eval
    from astabench.evals.paper_finder import paper_finder_utils as _pf_utils
    from astabench.evals.paper_finder import relevance as _pf_rel
    from astabench.evals.paper_finder import task as _pf_task
    from astabench.evals.paper_finder.relevance import Document, Relevance

    if getattr(_pf_eval.get_llm_relevance, "_robophd_grounded", False):
        return

    # Resolve the mutable astabench functions through their modules at CALL
    # time, not by binding them here: get_normalizer_references, the grader
    # model behind load_relevance_judgement, and update_references are all
    # things we (or a test) may patch after install — a per-run cache redirect,
    # the training-judge override, the multiprocess-safe writer. Binding them
    # at install time would silently pin the stock versions.
    async def _grounded_get_llm_relevance(output, metric):
        normalizer, detailed_reference = _pf_eval.get_normalizer_references()
        qid = output.query_id
        # Training cost cap: judge only the top-`estimate` submitted papers.
        # recall_at_estimate reads exactly this depth and the rank term is
        # empirically unaffected, so deeper papers are pure judge cost. Off for
        # test/pristine (env unset) → judge all, matching official scoring.
        cap = normalizer.get(qid) if os.environ.get(CAP_JUDGE_ENV) else None
        try:
            ev_cap = int(os.environ.get(EVIDENCE_CAP_ENV) or 0)
        except ValueError:
            ev_cap = 0
        judgements: dict[str, str] = {}
        # (paper_id, dropped_passages, raw_evidence, kind) where kind is
        # "partial" (some passages kept) or "full" (all dropped).
        blanked: list[tuple[str, list[str], str, str]] = []
        # (paper_id, original_len, capped_len) for evidence clipped by
        # EVIDENCE_CAP_ENV — surfaced to evolution as evidence_truncation.md.
        truncated: list[tuple[str, int, int]] = []
        to_judge: list[Document] = []
        judge_keys: list[tuple[str, str]] = []  # (paper_id, composite cache key)

        for idx, result in enumerate(output.results):
            # Beyond the scored depth: not judged, not grounding-checked. Results
            # are in submission order, so break.
            if cap is not None and idx >= cap:
                break
            pid = result.paper_id
            # Gold papers are Perfect regardless of evidence — the metric never
            # judges them, so grounding is moot.
            if pid in metric.known_to_be_good:
                judgements[pid] = Relevance.PERFECT.value
                continue
            raw_ev = result.markdown_evidence or ""
            # Evidence char cap (training-only): truncate BEFORE grounding —
            # a truncated substring of retrieved text still grounds, and the
            # judge/cache operate on the capped text.
            if ev_cap > 0 and len(raw_ev) > ev_cap:
                truncated.append((pid, len(raw_ev), ev_cap))
                raw_ev = raw_ev[:ev_cap]
            # Keep only grounded passages; the judge sees `scrubbed`, never the
            # dropped ones.
            scrubbed, dropped = scrub_evidence(pid, raw_ev)
            if dropped:
                kind = "full" if not scrubbed else "partial"
                blanked.append((pid, dropped, raw_ev, kind))
            if not scrubbed:
                # Nothing grounded survived: an empty document judges to Not
                # Relevant. Short-circuit to save the judge call entirely.
                judgements[pid] = Relevance.NOT_RELEVANT.value
                continue
            ckey = cache_key(pid, scrubbed)
            cached = (detailed_reference.get(qid) or {}).get(ckey) if detailed_reference else None
            if cached is not None:
                judgements[pid] = cached
            else:
                to_judge.append(Document(corpus_id=pid, markdown=scrubbed))
                judge_keys.append((pid, ckey))

        if to_judge:
            new = await _pf_rel.load_relevance_judgement(to_judge, metric.relevance_criteria)
            cache_write: dict[str, str] = {}
            for pid, ckey in judge_keys:
                verdict = new.get(pid)
                if verdict is not None:
                    judgements[pid] = verdict
                    cache_write[ckey] = verdict
            if cache_write:
                # Look up at call time so we invoke whatever update_references
                # patch is installed (the multiprocess-safe writer), regardless
                # of import order.
                await _pf_utils.update_references(qid, cache_write)

        _LAST.clear()
        _LAST.update(query_id=qid, judgements=dict(judgements), blanked=blanked,
                     cap=cap, truncated=truncated)
        return judgements

    _grounded_get_llm_relevance._robophd_grounded = True  # type: ignore[attr-defined]
    _pf_eval.get_llm_relevance = _grounded_get_llm_relevance
    _pf_task.get_llm_relevance = _grounded_get_llm_relevance


def last_judgements() -> dict[str, str]:
    """{paper_id: verdict} from the most recent grounded judge call (this eval)."""
    return dict(_LAST.get("judgements") or {})


def last_blanked() -> list[tuple[str, list[str], str, str]]:
    """[(paper_id, dropped_passages, raw_evidence, kind)] from the most recent
    call, where kind is "partial" (some passages kept) or "full" (all dropped)."""
    return list(_LAST.get("blanked") or [])


def last_truncations() -> list[tuple[str, int, int]]:
    """[(paper_id, original_len, capped_len)] for evidence clipped by the
    per-paper char cap (EVIDENCE_CAP_ENV) in the most recent call. Empty
    when the cap is off or nothing exceeded it."""
    return list(_LAST.get("truncated") or [])


def last_cap() -> int | None:
    """The top-estimate judging depth applied in the most recent call, or None
    if uncapped (test/pristine). Submitted papers past this depth were not judged
    (they cannot affect the score)."""
    return _LAST.get("cap")
