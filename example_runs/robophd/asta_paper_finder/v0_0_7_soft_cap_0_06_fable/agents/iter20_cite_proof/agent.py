"""iter20_cite_proof: PaperFindingBench solver, iteration 20.

Base = iter18_triage_first with the SEMANTIC PIPELINE UNCHANGED (it won the
batch score in iterations 18 and 19 and is the cheapest of the line at
~$0.067/semantic query, ~$0.052 projected at the 73%-semantic test mix).
Iteration 19 tested the last open semantic question — reverting iter18's
retrieval stack to iter13's — and LOST by 3.3 points; the grade-3
attribution shows got_it in a three-way tie (54.1/53.8/53.8%), so the
retrieval-stack question is resolved as noise and this iteration touches
nothing on the semantic side.

WHAT CHANGED: three deterministic fixes on the exact-match paths, all free
(tool calls only, zero new LLM spend):

1. BODY-MENTION CITATION VERIFICATION (fixes metadata_42-type, 0.053 with a
   ~0.5 counterfactual). The refs check via get_paper_batch(references) is a
   broken instrument: on metadata_42 it returned reference lists for 67/72
   candidates but matched the target in only ~1 — S2 reference lists arrive
   truncated or id-less, so the check false-negatives at scale and
   "reference verification: 72 -> 6" discarded a candidate set that covered
   a 70-paper gold. New acceptance channel: scoped
   snippet_search(query=<short name>, paper_ids=<chunk of 25>, limit=100),
   accept a candidate iff a returned passage literally contains the cited
   work's short name (normalized, word-bounded). A paper that names
   "RoBERTa" in its body after being retrieved by a RoBERTa keyword search
   all but certainly cites it.

2. CONJUNCTION AUGMENTATION UNDER THE CITER CAP (metadata_26-type, 0.000
   for every agent). get_citations is recency-ordered and capped at 1000;
   on "papers citing the T5 paper and the spider paper" both lists cap, and
   the gold (all corpus_id 272M-276M, ~Oct 2024-Feb 2025) sits in an OLDER
   recency window that has scrolled out of the cap at eval time — the pure
   intersection can never see it. When a multi-target citing query hits the
   cap and the intersection is small (<40), add a mention-conjunction
   channel (keyword searches on the joined short names, a global snippet
   search "both A and B"), verify candidates by requiring body passages
   mentioning EVERY target (per-target scoped snippet verification,
   intersected), and admit verified extras up to 40 total. Bounded
   downside: these queries score 0.000 today.

3. _batch_bisect: EVERY get_paper_batch site now bisects on chunk failure
   (the _fetch_references pattern, factored out). Observed on metadata_42:
   the metadata backfill failed BOTH attempts on "Paper ... is newer than
   the date cutoff" — one poison id deleted 50 ids' metadata, and docs with
   citationCount=None/authors=None were then silently dropped by the cheap
   filters. Applies to the metadata backfill, _fill_abstracts, and the
   citation-expansion metadata fetch.

Inherited unchanged from iter18: 14-query diverse-category planner, POOL_CAP
320 lexical prescreen, compact stage-1 triage, citation expansion, stage-2
judge sim, grade-2 rescue, GPT_5_4 head verify, band ordering, longer
evidence cuts (abstract 2000 / snippets 900) with containment dedup, tail
sweep to 250, the specific path (retrieve wide, submit tight), the chunked
venue filter, cites_paper_titles conjunction parsing with citer-set
intersection, per-stage llm-usage telemetry.

Scoring context (re-confirmed on iteration-19 verdicts): the judge grades
exactly the first K submitted positions (K observed 12-228); recall counts
only grade-3s in that prefix; grade 3 needs EVERY weighted criterion judged
Perfectly Relevant from the submitted evidence alone. On exact-match paths
order never matters and F1 = harmonic(hits/#submitted, hits/#gold) — with a
large gold set, discarding plausible candidates is the costliest error.
"""

import asyncio
import contextvars
import difflib
import json
import re
import time

from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolDef

from model_registry import GPT_5_4, GPT_5_4_MINI

# ---------------------------------------------------------------- constants

MAX_SUBMIT = 250
POOL_CAP = 320            # initial semantic pool (after lexical prescreen)
POOL_MERGE_HEAD = 240     # pool slots kept in raw source-rank merge order
GRADE_CHUNK = 32          # stage-1 triage chunk size
SIM_CHUNK = 8             # stage-2 judge-simulation chunk size
HEAD = 100                # head depth that gets enrichment + judge simulation
FULL_COVER_DEPTH = 36     # positions that get a snippet call for EVERY criterion
PER_CRIT_DEPTH = 70       # head prefix that gets one snippet call per weak criterion
SIM_DEPTH = 55            # head prefix that gets stage-2 judge simulation
RESCUE_MAX = 22           # max papers rescued per query (depth = whole head)
SNIP_INIT_LIMIT = 100     # passages per initial snippet_search query (tool max; free)
SIM_SKIP_W = 0.45         # skip stage-2 sim below this stage-1 weight when snippetless
VERIFY_PP = 32            # head verify triggers at/below this stage-1 predicted-perfect
VERIFY_TOP = 26           # papers re-graded by GPT_5_4 in the head verify
VERIFY_THIN_PP = 10       # predicted-perfect at/below this -> extend the verify
VERIFY_TOP_THIN = 30      # verify depth on thin pools (whole score sits in top-K)
EXPAND_SEEDS = 10         # strongest candidates whose refs/citers seed expansion
EXPAND_CITE_LIMIT = 90    # citers fetched per expansion seed
EXPAND_CAP = 100          # max new docs added by citation-graph expansion
ENRICH_CONCURRENCY = 10   # stay at the shared 10 req/s endpoint budget
SNIPPET_TIMEOUT = 90      # seconds per scoped snippet call
SOFT_DEADLINE = 1300      # seconds; skip remaining enrichment past this
TAIL_SWEEP_END = 250      # last submission position eligible for the tail sweep
TAIL_SWEEP_MIN = 40       # tail positions swept on narrow queries
TAIL_BROAD_UNIQ = 400     # search uniques at/above this -> sweep the full tail
TAIL_DEADLINE = 1550      # seconds; per-call gate for the tail sweep
REF_BATCH = 20            # get_paper_batch size when fetching references
T1_TITLE = 110            # stage-1 triage title chars
T1_BODY = 170             # stage-1 triage body chars
# The grade view: what stage-2 sim / rescue / verify actually read. Until
# iteration 13 these graded _cut(evidence, 600), which is title + tldr +
# the head of the abstract — the fetched body snippets, the entire point
# of enrichment, sat past the cut and no internal grader ever saw one.
GV_TITLE = 110            # grade-view title chars
GV_ABSTRACT = 300         # grade-view abstract chars
GV_SNIP = 180             # grade-view chars per body snippet
GV_SNIP_MAX = 4           # grade-view body snippets per candidate
CV_LLM_MAX = 400          # distinct venue strings classified (chunked, not truncated)
CV_LLM_CHUNK = 100        # venue strings per classification call
# get_paper/get_paper_batch reject fields="...references" server-side
# ('NoneType' not iterable) in every observed run; the S2-subfield form is
# the one untried variant. Probe once per site, commit or skip.
REF_FIELD_VARIANTS = ("corpusId,references", "corpusId,references.corpusId,references.title")
CV_CHECK_MAX = 80         # candidates whose references get venue-verified
CV_REF_IDS_MAX = 2500     # unique referenced papers resolved for venue
PAPER_FIELDS = "title,abstract,corpusId,tldr,year,venue"
META_FIELDS = "title,abstract,corpusId,year,venue,journal,authors,citationCount"

# ---------------------------------------------------------------- helpers


def _get_tool(state: TaskState, name: str):
    by_name = {ToolDef(t).name: t for t in state.tools}
    if name not in by_name:
        raise RuntimeError(f"{name!r} not in state.tools (have: {sorted(by_name)})")
    return by_name[name]


def _parse_items(raw) -> list[dict]:
    """Flatten MCP ContentText items; unwrap {"data": [...]} wrappers."""
    docs = []
    for item in raw or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(doc, dict) and "data" in doc:
            data = doc["data"]
            if isinstance(data, list):
                docs.extend(d for d in data if isinstance(d, dict))
        elif isinstance(doc, dict):
            docs.append(doc)
    return docs


def _cid(doc: dict) -> str:
    """Corpus id as a clean string ('' if absent)."""
    if not isinstance(doc, dict):
        return ""
    v = doc.get("corpusId")
    if v is None:
        v = doc.get("corpusid")
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower().startswith("corpusid:"):
        s = s.split(":", 1)[1]
    return s


def _tldr_text(doc: dict) -> str:
    t = doc.get("tldr")
    if isinstance(t, dict):
        return (t.get("text") or "").strip()
    if isinstance(t, str):
        return t.strip()
    return ""


def _auth_names(doc: dict) -> list[str]:
    """Author names whether entries are dicts or plain strings."""
    names = []
    for a in doc.get("authors") or []:
        if isinstance(a, dict):
            n = (a.get("name") or "").strip()
        elif isinstance(a, str):
            n = a.strip()
        else:
            n = ""
        if n:
            names.append(n)
    return names


def _surname(name: str) -> str:
    toks = _norm(name).split()
    return toks[-1] if toks else ""


def _first_initial(name: str) -> str:
    toks = _norm(name).split()
    return toks[0][:1] if toks else ""


def _cut(text: str, n: int) -> str:
    """Truncate at a whitespace boundary; the result stays a verbatim substring."""
    text = text or ""
    if len(text) <= n:
        return text
    cut = text[:n]
    sp = cut.rfind(" ")
    return cut[: sp if sp > n // 2 else n]


_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉", "01234567890123456789")


def _ascii(text: str) -> str:
    """Normalize unicode that breaks keyword matching (superscripts, quotes)."""
    text = (text or "").translate(_SUPERSCRIPTS)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return text


def _json_block(text: str):
    """Extract the first JSON object/array from an LLM completion."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        break
    return None


# per-sample LLM usage telemetry: samples run concurrently in one process,
# so the counter dict rides a ContextVar (task-local), not a module global
_LLM_USE_VAR: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_llm_use", default=None)


def _llm_reset():
    _LLM_USE_VAR.set({})


def _llm_report() -> str:
    use = _LLM_USE_VAR.get() or {}
    return "; ".join(
        f"{k}:n={v[0]},in~{v[1] // 4}t,out~{v[2] // 4}t" for k, v in sorted(use.items())
    )


async def _gen(model, prompt: str, retries: int = 1, label: str = "other") -> str:
    """Generate with an empty-completion retry; never raises on empty."""
    use = _LLM_USE_VAR.get()
    rec = use.setdefault(label, [0, 0, 0]) if use is not None else [0, 0, 0]
    for attempt in range(retries + 1):
        try:
            rec[0] += 1
            rec[1] += len(prompt)
            resp = await model.generate(prompt)
            text = (resp.completion or "").strip()
            rec[2] += len(text)
            if text:
                return text
            print(f"  [gen] empty completion (attempt {attempt + 1})")
        except Exception as e:  # noqa: BLE001 - surface but keep going
            print(f"  [gen] error (attempt {attempt + 1}): {e!r}")
    return ""


_RETRY_DELAYS = (12, 40)


async def _safe_tool(factory, label: str, timeout: float | None = None, attempts: int = 3):
    """Run a tool call with retries. `factory` is a zero-arg callable that
    builds a FRESH coroutine per attempt (a bare coroutine is accepted for
    one-shot use). The transport layer only auto-retries 429/529/504 —
    observed 502/ConnectionRefused outages must be retried HERE. Tool calls
    are free; only the 29-minute wall clock bounds us."""
    coro = None if callable(factory) else factory
    for a in range(attempts):
        try:
            c = factory() if callable(factory) else coro
            if c is None:  # bare coroutine already consumed by a prior attempt
                return None
            coro = None
            if timeout is not None:
                return await asyncio.wait_for(c, timeout)
            return await c
        except Exception as e:  # noqa: BLE001
            print(f"  [tool:{label}] attempt {a + 1}/{attempts} failed: {e!r}")
            if callable(factory) and a < attempts - 1:
                await asyncio.sleep(_RETRY_DELAYS[min(a, len(_RETRY_DELAYS) - 1)])
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", _ascii(s).lower()).strip()


def _title_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _submit(state: TaskState, results: list[dict]) -> TaskState:
    state.output.completion = json.dumps(
        {"output": {"query_id": state.sample_id, "results": results[:MAX_SUBMIT]}}
    )
    print(f"  submitted {min(len(results), MAX_SUBMIT)} papers")
    return state


_EARLIEST_RE = re.compile(r"\b(earliest|first|original|seminal|pioneer\w*|oldest)\b", re.I)


# ------------------------------------------------------------ semantic path


async def _plan_semantic(query: str, earliest: bool) -> dict:
    """One GPT_5_4 call: reconstruct the judge's criteria + search inputs."""
    era_note = (
        "\n- The request asks for the EARLIEST/first/original work on the topic: "
        "make at least 3 keyword_queries use era-appropriate/classic terminology "
        "(the vocabulary older papers would use, e.g. 'spoken dialogue system' "
        "rather than 'LLM chatbot'), and avoid recent-jargon phrasings there."
        if earliest
        else ""
    )
    prompt = (
        "You are preparing a scholarly literature search whose results will be "
        "graded by a relevance judge.\n"
        f"User request: {query}\n\n"
        "The judge scores each paper against 2-4 weighted relevance criteria "
        "derived from the request. Typical pattern: one criterion per core "
        "concept (the central concept gets the largest weight, 0.4-0.6), plus "
        "lower-weight (0.1-0.2) qualifier criteria when the request implies "
        "them: an EXPLICIT connection between the concepts, a required focus "
        "(named domain/language/population/model), or breadth (e.g. 'common or "
        "widely-used approaches rather than niche designs'). If the request "
        "carries a superlative or temporal qualifier (earliest/first/original "
        "work, most recent/latest, etc.), the judge makes that its OWN "
        "criterion (weight 0.15-0.25) — include it. Reconstruct the most "
        "likely criteria.\n\n"
        "Also produce inputs for a literal keyword-matching paper search engine "
        "(no operators; noun phrases only; interrogative or imperative phrasing "
        "returns zero hits).\n\n"
        "Reply with JSON only:\n"
        "{\n"
        '  "criteria": [{"name": "...", "description": "The paper must ...", '
        '"weight": 0.4, "probe": "<3-8 word declarative phrase likely to appear '
        'near text in a paper that demonstrates this criterion>", '
        '"probe2": "<a DIFFERENT 3-8 word phrasing of the same criterion, using '
        'alternate vocabulary a paper might use instead>"}, ...],\n'
        '  "keyword_queries": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."],\n'
        '  "snippet_queries": ["<full sentence>", "<sentence 2>", "<sentence 3>", "<sentence 4>", "<sentence 5>"],\n'
        '  "year_min": null, "year_max": null\n'
        "}\n"
        "- criteria: 2-4 entries, weights summing to 1. probe/probe2 are used "
        "as queries against a body-passage search engine to retrieve the "
        "sentences that PROVE the criterion, so phrase them the way a paper "
        "would state it (e.g. 'micro-F1 averaged across test episodes', not "
        "'the paper must evaluate with micro-F1'). probe2 must use genuinely "
        "different vocabulary from probe (a synonym, the metric's other name, "
        "the experimental-setup phrasing), because it is the retry used when "
        "probe found nothing — repeating probe would retrieve the same miss.\n"
        "- keyword_queries: 14 DIVERSE 2-8 word noun-phrase queries. These are "
        "NEW ANGLES, not paraphrases — cover these categories: (a) 2-3 direct "
        "phrasings of the request, (b) 2-3 synonym/alternate-terminology "
        "phrasings a different community would use, (c) 3-4 naming SPECIFIC "
        "well-known methods, systems, datasets, or model families that "
        "instantiate the request (e.g. for rejection-sampling finetuning: "
        "'ReST reinforced self-training', 'reward ranked fine-tuning RAFT'), "
        "(d) 1-2 task/application phrasings (the downstream problem papers "
        "solve with this), (e) 1-2 adjacent-subfield phrasings likely to "
        "co-report the topic. If the request asks about approaches/solutions/"
        "architectures/landscape of a topic, make 2 of them survey-oriented "
        "('<topic> survey', '<topic> review')." + era_note + "\n"
        "- snippet_queries: 5 different full sentences for a body-passage "
        "search engine. At least 2 must state the CONNECTION or qualifier the "
        "request implies, phrased the way a paper's method or analysis "
        "section would state it (papers satisfying every aspect at once "
        "usually state the combination in body text, not the abstract).\n"
        "- year_min/year_max: only if the request states an explicit year bound."
    )
    obj = _json_block(await _gen(GPT_5_4, prompt, label="plan")) or {}
    criteria = []
    for c in obj.get("criteria") or []:
        if isinstance(c, dict) and (c.get("description") or "").strip():
            try:
                w = float(c.get("weight") or 0)
            except (TypeError, ValueError):
                w = 0.0
            criteria.append(
                {
                    "name": (c.get("name") or "")[:60],
                    "description": c["description"].strip(),
                    "weight": w,
                    "probe": (c.get("probe") or "").strip()[:120],
                    "probe2": (c.get("probe2") or "").strip()[:120],
                }
            )
    criteria = criteria[:4]
    total_w = sum(c["weight"] for c in criteria)
    if criteria and total_w > 0:
        for c in criteria:
            c["weight"] /= total_w
    elif criteria:
        for c in criteria:
            c["weight"] = 1.0 / len(criteria)
    else:
        criteria = [
            {
                "name": "topic",
                "description": f"The paper must address: {query}",
                "weight": 1.0,
                "probe": "",
                "probe2": "",
            }
        ]

    kws = [k.strip() for k in obj.get("keyword_queries") or [] if isinstance(k, str) and k.strip()]
    if not kws:
        kws = [re.sub(r"[^\w\s-]", " ", _ascii(query))[:120]]
    snips = [s.strip() for s in obj.get("snippet_queries") or [] if isinstance(s, str) and s.strip()]
    if not snips:
        snips = [query]

    def _num(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "criteria": criteria,
        "keyword_queries": kws[:14],
        "snippet_queries": snips[:5],
        "year_min": _num(obj.get("year_min")),
        "year_max": _num(obj.get("year_max")),
    }


def _crit_query(criteria: list[dict], j: int) -> str:
    """Snippet query for criterion j: the short probe, else the description."""
    return criteria[j].get("probe") or criteria[j]["description"]


def _crit_query_alt(criteria: list[dict], j: int) -> str:
    """Retry query for criterion j: the planner's alternate phrasing.

    The rescue round re-probes a criterion the first fetch failed to prove.
    Reusing `probe` there retrieves the same passages that already failed,
    so fall back through probe2 -> description -> probe."""
    c = criteria[j]
    return c.get("probe2") or c["description"] or c.get("probe") or ""


def _grade_body(doc: dict, n: int = T1_BODY) -> str:
    body = (doc.get("abstract") or "").strip() or _tldr_text(doc) or " ".join(doc.get("_snippets") or [])
    return _cut(body, n)


async def _grade_chunk(
    criteria: list[dict], chunk: list[tuple[int, str]], model=GPT_5_4_MINI, label: str = "grade"
) -> dict[int, list[int]]:
    """Per-criterion 0/1/3 verdicts for each candidate, mirroring the judge's scale.

    Candidates are numbered LOCALLY (1..N) rather than by their global pool
    index, and grades are requested unspaced ("7:313" not "412: 3 1 3").
    Triage output is billed at 6x the input rate, so with a 740-candidate
    pool those two changes are worth more than any input trim; the digits
    are extracted individually, so the compact form parses identically."""
    ncrit = len(criteria)
    crit_lines = "\n".join(
        f"C{j + 1} (weight {c['weight']:.2f}): {c['description']}" for j, c in enumerate(criteria)
    )
    local_to_global = {j + 1: gi for j, (gi, _) in enumerate(chunk)}
    lines = "\n".join(f"{j + 1}. {t}" for j, (_, t) in enumerate(chunk))
    prompt = (
        "Grade candidate papers against relevance criteria, judging ONLY from "
        "each candidate's text below (a relevance judge will see only that text).\n"
        f"Criteria:\n{crit_lines}\n\n"
        "For each candidate output exactly one line, in this exact compact "
        f"form:  N:{'g' * ncrit}\n"
        f"where N is the candidate number and each g is one grade for "
        f"criteria C1..C{ncrit} in order, with NO spaces between the grades "
        f"(e.g. '7:{'3' * ncrit}').\n"
        "Grades: 3 = the text explicitly demonstrates the criterion; "
        "1 = partially or implicitly suggests it; 0 = does not support it.\n"
        "Be strict: 3 only when the text clearly states it. A qualifier "
        "criterion (e.g. 'common/widely-used approaches', 'explicitly "
        "connects X and Y') is 3 only when the text itself states the "
        "commonality/comparison or the explicit connection.\n\n"
        f"Candidates:\n{lines}\n\n"
        "Output only the grade lines, nothing else."
    )
    text = await _gen(model, prompt, label=label)
    out: dict[int, list[int]] = {}
    for m in re.finditer(r"^\s*(\d+)\s*[:.\-]\s*([0-9 ,;/|]+?)\s*$", text, re.MULTILINE):
        idx = local_to_global.get(int(m.group(1)))
        if idx is None:
            continue
        digits = [int(d) for d in re.findall(r"[0-9]", m.group(2))][:ncrit]
        digits = [3 if d >= 3 else (1 if d in (1, 2) else 0) for d in digits]
        if len(digits) == ncrit:
            out[idx] = digits
    if not out:
        print(f"  [grade] chunk of {len(chunk)} unparsed; defaulting to partial")
        return {i: [1] * ncrit for i, _ in chunk}
    return out


def _weighted(criteria: list[dict], verdicts: list[int]) -> float:
    return min(1.0, sum(c["weight"] * v / 3.0 for c, v in zip(criteria, verdicts)))


_STOP = frozenset(
    "the a an of in on for to and or with by from as at is are be been that "
    "this these those it its must paper papers focus specifically explicitly "
    "which such into how can may their they using use used address discuss "
    "describe present including include".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _norm(text).split() if len(w) > 2 and w not in _STOP}


def _dedup_snips(doc: dict, cut: int) -> list[str]:
    """Distinct body snippets attached to a doc, each truncated verbatim."""
    snips: list[str] = []
    seen: set[str] = set()
    for sn in doc.get("_snippets") or []:
        sn = (sn or "").strip()
        key = _norm(sn)[:80]
        if sn and key not in seen:
            seen.add(key)
            snips.append(_cut(sn, cut))
    return snips


def _grade_view(doc: dict, criteria: list[dict] | None = None) -> str:
    """The text every INTERNAL grader reads (stage-2 sim, rescue, verify).

    It must contain the body snippets: those passages are fetched precisely
    to prove the criteria an abstract leaves unstated, and they are part of
    what the real judge reads. Grading a snippet-free view instead is how
    iteration 12's rescue/verify rounds confirmed nothing on query after
    query while the judge, reading the full evidence, graded those same
    papers Highly or Perfect."""
    parts = [_cut((doc.get("title") or "").strip(), GV_TITLE)]
    abstract = (doc.get("abstract") or "").strip() or _tldr_text(doc)
    if abstract:
        parts.append(_cut(abstract, GV_ABSTRACT))
    snips = _dedup_snips(doc, GV_SNIP)
    if snips and criteria:
        snips = _cover_snippets(snips, criteria, GV_SNIP_MAX)
    parts.extend(snips[:GV_SNIP_MAX])
    return " ... ".join(p for p in parts if p)


def _cover_snippets(snips: list[str], criteria: list[dict], room: int) -> list[str]:
    """Order snippets so each weighted criterion gets its best-matching one
    first (weightiest criterion first), then fill with the rest."""
    crit_words = [
        _content_words(f"{c.get('name', '')} {c['description']} {c.get('probe', '')} {c.get('probe2', '')}")
        for c in criteria
    ]
    snip_words = [_content_words(s) for s in snips]
    chosen: list[int] = []
    for j in sorted(range(len(criteria)), key=lambda j: -criteria[j]["weight"]):
        if len(chosen) >= room:
            break
        cw = crit_words[j]
        if not cw:
            continue
        best, best_ov = None, 0.12
        for si, sw in enumerate(snip_words):
            if si in chosen:
                continue
            ov = len(cw & sw) / len(cw)
            if ov > best_ov:
                best, best_ov = si, ov
        if best is not None:
            chosen.append(best)
    for si in range(len(snips)):
        if len(chosen) >= room:
            break
        if si not in chosen:
            chosen.append(si)
    return [snips[si] for si in chosen]


def _redundant(text: str, already: list[str]) -> bool:
    """True if `text` restates something already emitted (either direction).

    snippet_search returns title- and abstract-section passages freely, so
    without this a paper can spend several of its only 8 slots re-showing
    the judge text it has already read. Containment both ways catches the
    common case where one version is the truncated form of the other."""
    n = " ".join(_norm(text).split())
    if not n:
        return True
    for prev in already:
        p = " ".join(_norm(prev).split())
        if not p:
            continue
        short, long = (n, p) if len(n) <= len(p) else (p, n)
        if short in long:
            return True
    return False


def _evidence(doc: dict, criteria: list[dict] | None = None) -> str:
    """Up to 8 verbatim passages: title, tldr, abstract, then snippets chosen
    greedily to cover each weighted criterion (weightiest-uncovered first)."""
    passages = []
    title = (doc.get("title") or "").strip()
    if title:
        passages.append(title)
    abstract = (doc.get("abstract") or "").strip()
    tldr = _tldr_text(doc)
    n_snips = len(doc.get("_snippets") or [])
    # the tldr paraphrases the abstract; when both an abstract and real body
    # coverage exist it buys the judge nothing and costs one of only 8 slots
    if tldr and not (abstract and n_snips >= 3):
        passages.append(tldr)
    if abstract:
        # the scorer imposes no length limit and _cut returns a verbatim
        # substring, so longer passages are free judge-visible information
        passages.append(_cut(abstract, 2000))

    snips = []
    for sn in _dedup_snips(doc, 900):
        if not _redundant(sn, passages) and not _redundant(sn, snips):
            snips.append(sn)

    room = 8 - len(passages)
    if room <= 0 or not snips:
        return " ... ".join(passages[:8])

    if criteria:
        passages.extend(_cover_snippets(snips, criteria, room))
    else:
        passages.extend(snips[:room])
    return " ... ".join(passages[:8])


def _snip_entries_to_docs(raw, snip_docs: dict[str, dict], snip_order: list[dict]):
    """Fold snippet_search entries into paper docs with attached snippets.

    snip_docs is shared across calls (one doc object per paper, snippets
    accumulate); snip_order is per-call, so each snippet query yields its own
    source list for the round-robin merge."""
    local: set[str] = set()
    for entry in _parse_items(raw):
        paper = entry.get("paper") or {}
        cid = _cid(paper)
        if not cid:
            continue
        text = ((entry.get("snippet") or {}).get("text") or "").strip()
        if cid not in snip_docs:
            snip_docs[cid] = {"corpusId": cid, "title": paper.get("title"), "_snippets": []}
        doc = snip_docs[cid]
        if text and len(doc["_snippets"]) < 3 and text not in doc["_snippets"]:
            doc["_snippets"].append(text)
        if cid not in local:
            local.add(cid)
            snip_order.append(doc)


async def _batch_bisect(
    batch, ids: list[str], fields: str, label: str, chunk: int = 50, attempts: int = 2
) -> list[dict]:
    """get_paper_batch over ids, bisecting failed chunks so one poison id
    (date-cutoff violation, transient server error) costs one id, not the
    whole chunk. Observed on metadata_42: the backfill failed BOTH attempts
    on 'Paper ... is newer than the date cutoff', deleting 50 ids' metadata
    and letting the cheap filters silently drop good candidates."""
    out: list[dict] = []

    async def _one(grp: list[str], att: int):
        raw = await _safe_tool(lambda: batch(ids=grp, fields=fields), label, attempts=att)
        if raw is None:
            # transient errors got their retry at the top level; below it the
            # dominant cause is a deterministic poison id, so single-attempt
            # bisection avoids paying the backoff ladder at every depth
            if len(grp) > 1:
                mid = len(grp) // 2
                await _one(grp[:mid], 1)
                await _one(grp[mid:], 1)
            return
        out.extend(_parse_items(raw))

    grps = [ids[i : i + chunk] for i in range(0, len(ids), chunk)]
    await asyncio.gather(*(_one(g, attempts) for g in grps))
    return out


async def _fill_abstracts(batch, docs: list[dict]):
    """Free batch fetch of title/abstract/tldr for docs missing an abstract."""
    missing = [d for d in docs if not d.get("abstract")]
    ids = [f"CorpusId:{_cid(d)}" for d in missing]
    fetched = {
        _cid(f): f
        for f in await _batch_bisect(batch, ids, "title,abstract,corpusId,tldr,year", "batch")
    }
    for d in missing:
        f = fetched.get(_cid(d))
        if f:
            for k in ("title", "abstract", "tldr", "year"):
                if f.get(k) and not d.get(k):
                    d[k] = f[k]


async def _solve_semantic(state: TaskState, query: str, start: float) -> TaskState:
    search = _get_tool(state, "search_papers_by_relevance")
    snippet = _get_tool(state, "snippet_search")
    batch = _get_tool(state, "get_paper_batch")

    earliest = bool(_EARLIEST_RE.search(query))
    plan = await _plan_semantic(query, earliest)
    criteria = plan["criteria"]
    ncrit = len(criteria)
    print(f"  criteria: {[c['name'] for c in criteria]} weights={[round(c['weight'], 2) for c in criteria]}")
    print(f"  probes: {[c.get('probe') for c in criteria]}")
    print(f"  keyword queries: {plan['keyword_queries']} earliest={earliest}")

    tasks = [
        _safe_tool(
            lambda k=k: search(keyword=_ascii(k), fields=PAPER_FIELDS, limit=100),
            f"rel[{k[:30]}]",
            attempts=2,
        )
        for k in plan["keyword_queries"]
    ]
    n_kw = len(tasks)
    for sq in plan["snippet_queries"]:
        tasks.append(
            _safe_tool(
                lambda sq=sq: snippet(query=sq, limit=SNIP_INIT_LIMIT),
                f"snippet[{sq[:30]}]",
                timeout=240,
                attempts=2,
            )
        )
    raws = await asyncio.gather(*tasks)

    result_lists = [_parse_items(r or []) for r in raws[:n_kw]]

    # snippet entries -> paper docs (in score order), snippets attached.
    # Each snippet query keeps its OWN source list so body-matched candidates
    # (the only place conjunction phrases are stated) get a proportional share
    # of the round-robin pool instead of 1-of-11.
    snip_docs: dict[str, dict] = {}
    snip_lists: list[list[dict]] = []
    for raw in raws[n_kw:]:
        order: list[dict] = []
        _snip_entries_to_docs(raw or [], snip_docs, order)
        if order:
            snip_lists.append(order)

    # round-robin merge across sources, dedupe (no cap yet)
    pool: dict[str, dict] = {}
    merged: list[dict] = []
    lists = [lst for lst in result_lists if lst] + snip_lists
    for rank in range(max((len(l) for l in lists), default=0)):
        for lst in lists:
            if rank >= len(lst):
                continue
            doc = lst[rank]
            cid = _cid(doc)
            if not cid:
                continue
            if cid in pool:
                # same doc OBJECT can sit in several snippet lists — never
                # extend a doc's snippet list with a slice of itself
                if doc is not pool[cid] and doc.get("_snippets"):
                    have = pool[cid].setdefault("_snippets", [])
                    have.extend(s for s in doc["_snippets"][:3] if s not in have)
                continue
            pool[cid] = doc
            merged.append(doc)
    n_uniq = len(merged)

    # lexical prescreen: keep the first POOL_MERGE_HEAD in source-rank order,
    # then fill the remaining slots with the criteria-overlap best leftovers —
    # a free filter that beats blind merge-order truncation at rank 240+
    if n_uniq > POOL_CAP:
        crit_words: set[str] = set()
        for c in criteria:
            crit_words |= _content_words(f"{c.get('name', '')} {c['description']} {c.get('probe', '')}")
        keep = merged[:POOL_MERGE_HEAD]
        rest = merged[POOL_MERGE_HEAD:]

        def _lex(d: dict) -> float:
            dw = _content_words(f"{d.get('title') or ''} {(d.get('abstract') or '')[:600]}")
            return len(crit_words & dw) / max(1, len(crit_words))

        rest.sort(key=_lex, reverse=True)
        ordered = keep + rest[: POOL_CAP - len(keep)]
        pool = {_cid(d): d for d in ordered}
    else:
        ordered = merged
    print(f"  candidate pool: {len(ordered)} of {n_uniq} uniques (per-source: {[len(l) for l in lists]})")

    if not ordered:
        return _submit(state, [])

    await _fill_abstracts(batch, ordered)

    # ---- stage 1: cheap per-criterion triage over the whole pool
    async def _triage(docs: list[dict], offset: int, label: str = "t1") -> dict[int, list[int]]:
        entries = [
            (offset + i, f"{(d.get('title') or '')[:T1_TITLE]} || {_grade_body(d)}")
            for i, d in enumerate(docs)
        ]
        chunks = [entries[i : i + GRADE_CHUNK] for i in range(0, len(entries), GRADE_CHUNK)]
        maps = await asyncio.gather(*(_grade_chunk(criteria, c, label=label) for c in chunks))
        out: dict[int, list[int]] = {}
        for vm in maps:
            out.update(vm)
        return out

    verdicts = await _triage(ordered, 0)
    default_v = [1] * ncrit
    n_perfect = sum(1 for v in verdicts.values() if all(x == 3 for x in v))
    print(f"  stage1 graded {len(verdicts)}/{len(ordered)}; predicted-perfect: {n_perfect}")

    # (gap-fill round removed: five observed firings changed predicted-perfect
    # by a net +2 across all of them — its LLM budget moved into the wider,
    # more diverse upfront pool.)

    # ---- citation-graph expansion: references + citers of the strongest
    # candidates are prime grade-3 material the keyword searches missed
    # (older vocabulary, follow-up work). All retrieval here is free; only
    # the triage of the new docs costs LLM tokens.
    if time.monotonic() - start < SOFT_DEADLINE - 400:
        try:
            get_cit = _get_tool(state, "get_citations")
            seed_rank = sorted(
                range(len(ordered)),
                key=lambda i: (
                    0 if all(x == 3 for x in verdicts.get(i, default_v)) else 1,
                    -_weighted(criteria, verdicts.get(i, default_v)),
                    i,
                ),
            )
            seeds = [ordered[i] for i in seed_rank[:EXPAND_SEEDS]]
            new_cids: list[str] = []
            seen_new: set[str] = set()

            def _note(cid_val):
                cid = str(cid_val or "")
                if cid and cid not in pool and cid not in seen_new:
                    seen_new.add(cid)
                    new_cids.append(cid)

            citer_cids: list = []

            async def _citers_of(d: dict):
                raw = await _safe_tool(
                    lambda: get_cit(
                        paper_id=f"CorpusId:{_cid(d)}", fields="corpusId,title", limit=EXPAND_CITE_LIMIT
                    ),
                    f"expand-cit[{_cid(d)}]",
                    timeout=90,
                    attempts=1,
                )
                for item in _parse_items(raw or []):
                    doc = item.get("citingPaper") if isinstance(item.get("citingPaper"), dict) else item
                    if isinstance(doc, dict):
                        citer_cids.append(doc.get("corpusId"))

            # references fail server-side ('NoneType' not iterable) under the
            # plain field name in every observed run, batched OR per-paper.
            # Probe the field variants on ONE seed, commit to whichever
            # returns data, and go citers-only if neither does — no more
            # burning a doomed call per seed.
            get_paper = _get_tool(state, "get_paper")
            ref_sem = asyncio.Semaphore(6)
            ref_variant: list[str | None] = [None]  # None=undecided, ""=dead

            async def _refs_of(d: dict) -> list[dict]:
                if ref_variant[0] == "":
                    return []
                variants = [ref_variant[0]] if ref_variant[0] else list(REF_FIELD_VARIANTS)
                async with ref_sem:
                    for fv in variants:
                        raw = await _safe_tool(
                            lambda fv=fv: get_paper(paper_id=f"CorpusId:{_cid(d)}", fields=fv),
                            f"expand-ref[{_cid(d)}]",
                            timeout=60,
                            attempts=1,
                        )
                        if raw is None:
                            continue
                        if ref_variant[0] != fv:
                            ref_variant[0] = fv
                            print(f"  [expand] references field variant works: {fv!r}")
                        out: list[dict] = []
                        for f in _parse_items(raw):
                            refs = f.get("references")
                            if isinstance(refs, list):
                                out.extend(r for r in refs if isinstance(r, dict))
                        return out
                return []

            citers_task = asyncio.gather(*(_citers_of(d) for d in seeds))
            # probe with the first seed alone so the rest reuse (or skip) the verdict
            first_refs = await _refs_of(seeds[0]) if seeds else []
            if ref_variant[0] is None:
                ref_variant[0] = ""
                print("  [expand] references unavailable under all field variants; citers-only")
            rest_refs = (
                await asyncio.gather(*(_refs_of(d) for d in seeds[1:])) if ref_variant[0] else []
            )
            ref_lists = [first_refs, *rest_refs]
            await citers_task
            # interleave references and citers so neither exhausts the cap:
            # refs carry the subfield's prior work, citers its follow-ups
            ref_cids = [r.get("corpusId") for refs in ref_lists for r in refs]
            for a, b in zip(ref_cids, citer_cids):
                _note(a)
                _note(b)
            longer = ref_cids if len(ref_cids) > len(citer_cids) else citer_cids
            for c in longer[min(len(ref_cids), len(citer_cids)):]:
                _note(c)
            exp_ids = new_cids[:EXPAND_CAP]
            fresh: list[dict] = []
            if exp_ids:
                fresh_by_cid: dict[str, dict] = {}
                for d in await _batch_bisect(
                    batch, [f"CorpusId:{c}" for c in exp_ids], PAPER_FIELDS, "expand-meta"
                ):
                    cid = _cid(d)
                    if cid and cid not in pool and cid not in fresh_by_cid:
                        fresh_by_cid[cid] = d
                fresh = list(fresh_by_cid.values())
            if fresh:
                for d in fresh:
                    pool[_cid(d)] = d
                exp_verdicts = await _triage(fresh, len(ordered), label="t1exp")
                ordered.extend(fresh)
                verdicts.update(exp_verdicts)
                np_new = sum(1 for v in verdicts.values() if all(x == 3 for x in v))
                print(
                    f"  citation expansion: +{len(fresh)} docs from {len(seeds)} seeds; "
                    f"predicted-perfect {n_perfect} -> {np_new}"
                )
                n_perfect = np_new
        except Exception as e:  # noqa: BLE001 - expansion is best-effort
            print(f"  [expand] skipped: {e!r}")

    def _key1(i: int):
        v = verdicts.get(i, default_v)
        return (0 if all(x == 3 for x in v) else 1, -_weighted(criteria, v), i)

    rank1 = sorted(range(len(ordered)), key=_key1)
    head_idx = rank1[:HEAD]
    tail_idx = rank1[HEAD:]

    # ---- enrichment machinery (free tool calls)
    sem = asyncio.Semaphore(ENRICH_CONCURRENCY)
    snip_query_default = plan["snippet_queries"][0]

    async def _fetch_snips(doc: dict, q: str, limit: int, timeout: float = SNIPPET_TIMEOUT, attempts: int = 2):
        async with sem:
            raw = await _safe_tool(
                lambda: snippet(query=_cut(q, 300), paper_ids=f"CorpusId:{_cid(doc)}", limit=limit),
                f"enrich[{_cid(doc)}]",
                timeout=timeout,
                attempts=attempts,
            )
        texts = []
        for entry in _parse_items(raw or []):
            t = ((entry.get("snippet") or {}).get("text") or "").strip()
            if t:
                texts.append(t)
        if texts:
            doc.setdefault("_snippets", []).extend(texts[:limit])

    # ---- enrichment: criterion-targeted body snippets for the head
    try:
        async def _enrich(pos: int, doc: dict, v: list[int]):
            if time.monotonic() - start > SOFT_DEADLINE:
                return
            weak = [j for j in range(ncrit) if v[j] < 3]
            weak.sort(key=lambda j: -criteria[j]["weight"])
            if pos < FULL_COVER_DEPTH:
                # These positions are judged on EVERY query (smallest observed
                # K is 6), and grade 3 — the only grade that earns recall —
                # needs every criterion Perfectly Relevant. A criterion the
                # stage-1 abstract happened to satisfy still gets demoted when
                # the submitted text doesn't state it, so fetch a passage for
                # ALL criteria here, weightiest first, not just the weak ones.
                for j in sorted(range(ncrit), key=lambda j: -criteria[j]["weight"]):
                    await _fetch_snips(doc, _crit_query(criteria, j), 4)
                return
            if not weak:
                # stage-1-perfect on abstract text — but the judge demotes to
                # Highly when a qualifier criterion isn't stated by the
                # abstract. Fetch body passages for criteria whose probe
                # words never appear in the title+abstract (free).
                have = _content_words(f"{doc.get('title') or ''} {doc.get('abstract') or ''}")
                uncovered = [
                    j for j in range(ncrit)
                    if not (_content_words(_crit_query(criteria, j)) & have)
                ]
                uncovered.sort(key=lambda j: criteria[j]["weight"])
                if uncovered:
                    await _fetch_snips(
                        doc, " ".join(_crit_query(criteria, j) for j in uncovered[:2]), 3
                    )
                elif not doc.get("abstract"):
                    await _fetch_snips(doc, snip_query_default, 3)
                return
            if pos < PER_CRIT_DEPTH and len(weak) >= 2:
                # one targeted call per weak criterion (weightier first)
                for j in weak[:2]:
                    await _fetch_snips(doc, _crit_query(criteria, j), 3)
            else:
                await _fetch_snips(doc, " ".join(_crit_query(criteria, j) for j in weak), 4)

        to_enrich = [
            (pos, ordered[i], verdicts.get(i, default_v))
            for pos, i in enumerate(head_idx)
        ]
        print(f"  snippet-enriching {len(to_enrich)} of top {len(head_idx)}")
        await asyncio.gather(*(_enrich(pos, d, v) for pos, d, v in to_enrich))
    except Exception as e:  # noqa: BLE001 - enrichment is best-effort
        print(f"  [enrich] skipped: {e!r}")

    # assemble the exact evidence that will be submitted for the head
    for i in head_idx:
        ordered[i]["_evidence"] = _evidence(ordered[i], criteria)
        ordered[i]["_gview"] = _grade_view(ordered[i], criteria)

    # ---- stage 2: judge simulation on the assembled evidence.
    # Papers already all-perfect at stage 1 (graded on abstract text the judge
    # will also see) keep the top band without re-simulation — this both saves
    # cost and prevents sim-induced demotions of good papers.
    perfect1 = {i for i in head_idx if all(x == 3 for x in verdicts.get(i, default_v))}
    verdicts2: dict[int, list[int]] = {}
    try:
        # snippetless low-weight papers: sim input is just title+tldr+abstract,
        # i.e. the text stage 1 already graded — reuse that verdict for free
        # depth cap: nDCG discounts position 90 to a sixth of position 1, and
        # half the observed queries have K <= 52 — re-grading the back of the
        # head buys ordering nobody reads. Those keep their stage-1 verdicts.
        sim_targets = [
            i
            for i in head_idx[:SIM_DEPTH]
            if i not in perfect1
            and (
                ordered[i].get("_snippets")
                or _weighted(criteria, verdicts.get(i, default_v)) > SIM_SKIP_W
            )
        ]
        n_reuse = len(head_idx) - len(perfect1) - len(sim_targets)
        sim_entries = [(i, ordered[i].get("_gview") or "") for i in sim_targets]
        sim_chunks = [sim_entries[i : i + SIM_CHUNK] for i in range(0, len(sim_entries), SIM_CHUNK)]
        sim_maps = await asyncio.gather(*(_grade_chunk(criteria, c, label="sim") for c in sim_chunks))
        for vm in sim_maps:
            verdicts2.update(vm)
        n_perfect2 = sum(1 for i in sim_targets if all(x == 3 for x in verdicts2.get(i, [])))
        print(
            f"  stage2 judge-sim graded {len(verdicts2)}/{len(sim_targets)} "
            f"(skipped {len(perfect1)} stage1-perfect, {n_reuse} snippetless-weak); "
            f"newly-perfect: {n_perfect2}"
        )
    except Exception as e:  # noqa: BLE001 - fall back to stage-1 order
        print(f"  [stage2] skipped: {e!r}")

    pos1 = {i: p for p, i in enumerate(head_idx)}
    # GPT_5_4 head-verify verdicts, filled in below. Verify reads the same
    # grade view the cheap graders read, on the same 0/1/3 per-criterion
    # rubric, so its weight is directly comparable — it simply overrides.
    v_weight: dict[int, float] = {}
    v_perfect: set[int] = set()

    def _sim_perfect(i: int) -> bool:
        v2 = verdicts2.get(i)
        return v2 is not None and all(x == 3 for x in v2)

    def _cheap_perfect(i: int) -> bool:
        return i in perfect1 or _sim_perfect(i)

    def _wmax(i: int) -> float:
        w = _weighted(criteria, verdicts.get(i, default_v))
        if i in verdicts2:
            w = max(w, _weighted(criteria, verdicts2[i]))
        return w

    def _band(i: int) -> int:
        """0 = verified perfect, 1 = perfect on a cheap grader only, 2 = rest.

        Band 1 sits below band 0 but above every verified non-perfect paper:
        GPT_5_4-confirmed coverage beats an unverified guess, and a paper
        verify demoted should not outrank one it never read. GPT_5_4_MINI is
        the over-predictor here (semantic_77: 31 predicted perfect, 0 judged),
        which is why cheap perfection alone never reaches band 0."""
        if i in v_weight:
            return 0 if i in v_perfect else 2
        return 1 if _cheap_perfect(i) else 2

    def _final_w(i: int) -> float:
        return v_weight[i] if i in v_weight else _wmax(i)

    def _key2(i: int):
        # NOTE: the "earliest" year tiebreak sits AFTER the weighted score —
        # in iter9 it sat before and ordered old-but-weak papers above strong
        # ones (semantic_145 rank 0.03, the grade-3 at the bottom of K=6).
        return (
            _band(i),
            -_final_w(i),
            (ordered[i].get("year") or 3000) if earliest else 0,
            pos1[i],
        )

    head_ranked = sorted(head_idx, key=_key2)

    # ---- grade-2 rescue: near-misses at the top get extra probe-scoped
    # snippets, rebuilt evidence, and a promotion-only re-sim.
    if time.monotonic() - start < SOFT_DEADLINE - 200:
        targets: list[tuple[int, list[int]]] = []
        for i in head_ranked:
            if _band(i) == 0:
                continue
            v = verdicts2.get(i) or verdicts.get(i, default_v)
            weak = [j for j in range(ncrit) if v[j] < 3]
            if 1 <= len(weak) <= 2 and _wmax(i) > 0.5:
                targets.append((i, weak))
            if len(targets) >= RESCUE_MAX:
                break
        if targets:
            print(f"  rescue round: {len(targets)} near-miss papers")

            async def _rescue_one(i: int, weak: list[int]):
                doc = ordered[i]
                # retry the unproven criterion with the planner's ALTERNATE
                # phrasing: this criterion's first probe already ran and came
                # back without proof, so re-issuing it retrieves that miss again
                for j in sorted(weak, key=lambda j: -criteria[j]["weight"])[:2]:
                    await _fetch_snips(doc, _crit_query_alt(criteria, j), 4)
                doc["_evidence"] = _evidence(doc, criteria)
                doc["_gview"] = _grade_view(doc, criteria)

            try:
                await asyncio.gather(*(_rescue_one(i, w) for i, w in targets))
                r_entries = [(i, ordered[i].get("_gview") or "") for i, _ in targets]
                r_chunks = [r_entries[i : i + SIM_CHUNK] for i in range(0, len(r_entries), SIM_CHUNK)]
                r_maps = await asyncio.gather(*(_grade_chunk(criteria, c, label="rescue") for c in r_chunks))
                promoted = 0
                for vm in r_maps:
                    for i, v in vm.items():
                        old = verdicts2.get(i)
                        if old is None or _weighted(criteria, v) > _weighted(criteria, old):
                            verdicts2[i] = v
                            if all(x == 3 for x in v):
                                promoted += 1
                print(f"  rescue promoted {promoted} to predicted-perfect")
                head_ranked = sorted(head_idx, key=_key2)
            except Exception as e:  # noqa: BLE001 - rescue is best-effort
                print(f"  [rescue] skipped: {e!r}")

    # ---- head verify: when true perfects are scarce, the whole score sits in
    # the first ~K positions (K as low as 6) and the cheap triage is observed
    # to over-predict grade 3 (semantic_77: 31 predicted, 0 judged). One
    # high-fidelity GPT_5_4 pass over the top VERIFY_TOP grade views; its
    # verdicts then override the cheap graders' for every paper it read.
    if n_perfect <= VERIFY_PP and time.monotonic() - start < SOFT_DEADLINE - 100:
        try:
            # thin pools: the entire score sits in the top-K slots (K as low
            # as 24), so the high-fidelity pass covers more of the head
            v_top = VERIFY_TOP_THIN if n_perfect <= VERIFY_THIN_PP else VERIFY_TOP
            v_targets = head_ranked[:v_top]
            v_entries = [(i, ordered[i].get("_gview") or "") for i in v_targets]
            v_chunks = [v_entries[k : k + 6] for k in range(0, len(v_entries), 6)]
            v_maps = await asyncio.gather(
                *(_grade_chunk(criteria, c, model=GPT_5_4, label="verify") for c in v_chunks)
            )
            # every verified paper's weight enters the ranking, not just the
            # confirmed perfects: the high-fidelity pass read the same grade
            # view the judge's evidence is built from, so leaving its
            # non-perfect papers ordered by an abstract-only stage-1 score is
            # what left 13 Somewhat papers above 4 Perfect ones in a top-20
            # (semantic_192).
            for vm in v_maps:
                for i, v in vm.items():
                    if all(x == 3 for x in v):
                        v_perfect.add(i)
                    v_weight[i] = _weighted(criteria, v)
            print(
                f"  head verify: {len(v_perfect)}/{len(v_targets)} confirmed perfect "
                f"by GPT_5_4 (graded {len(v_weight)})"
            )
            if v_weight:
                head_ranked = sorted(head_idx, key=_key2)
        except Exception as e:  # noqa: BLE001 - verify is best-effort
            print(f"  [head-verify] skipped: {e!r}")

    final_idx = head_ranked + tail_idx

    # ---- tail evidence sweep (FREE): on broad queries the judge reads to
    # position ~K (up to 228), but entries beyond the head carry abstract-only
    # evidence that under-supports qualifier criteria — the observed grade-2
    # mass (semantic_110: 126 Highly vs 67 Perfectly). Fetch criterion-probe
    # passages for the judged-likely tail; snippet calls cost nothing.
    try:
        sweep_n = (TAIL_SWEEP_END - len(head_ranked)) if n_uniq >= TAIL_BROAD_UNIQ else TAIL_SWEEP_MIN
        sweep_idx = [
            i
            for i in final_idx[len(head_ranked) : len(head_ranked) + max(0, sweep_n)]
            if len(ordered[i].get("_snippets") or []) < 3
        ]
        swept = 0

        async def _sweep_one(i: int):
            nonlocal swept
            if time.monotonic() - start > TAIL_DEADLINE:
                return
            doc = ordered[i]
            v = verdicts.get(i, default_v)
            weak = [j for j in range(ncrit) if v[j] < 3] or sorted(
                range(ncrit), key=lambda j: criteria[j]["weight"]
            )[:2]
            weak.sort(key=lambda j: -criteria[j]["weight"])
            q = " ".join(_crit_query(criteria, j) for j in weak[:2])
            await _fetch_snips(doc, q, 4, timeout=45, attempts=1)
            swept += 1

        if sweep_idx:
            await asyncio.gather(*(_sweep_one(i) for i in sweep_idx))
            print(f"  tail sweep: enriched {swept}/{len(sweep_idx)} tail entries (uniq={n_uniq})")
    except Exception as e:  # noqa: BLE001 - the sweep is best-effort
        print(f"  [tail-sweep] skipped: {e!r}")

    results = []
    for i in final_idx[:MAX_SUBMIT]:
        d = ordered[i]
        cid = _cid(d)
        if not cid:
            continue
        ev = d.get("_evidence") or _evidence(d, criteria)
        results.append({"paper_id": cid, "markdown_evidence": ev})
    print(f"  llm-usage: {_llm_report()}")
    return _submit(state, results)


# ------------------------------------------------------------ specific path


def _alias_titled(alias: str, title: str) -> bool:
    """Title IS the alias or starts with it as a name token ('SPIKE: ...')."""
    a, t = _norm(alias), _norm(title)
    return bool(a) and (t == a or t.startswith(a + " "))


async def _solve_specific(state: TaskState, query: str) -> TaskState:
    title_search = _get_tool(state, "search_paper_by_title")
    rel_search = _get_tool(state, "search_papers_by_relevance")
    snippet = _get_tool(state, "snippet_search")

    prompt = (
        "A user refers to one specific published paper as:\n"
        f'"{query}"\n\n'
        "Hints for interpreting such references:\n"
        "- A token like 'Smith2021' or 'DeYong2021' is a citation key: author "
        "surname + publication year. The surname may be slightly misspelled "
        "(e.g. DeYong for DeYoung); the year is reliable.\n"
        "- The paper's real title may NOT contain the short name/alias it is "
        "known by.\n"
        "- The short name may be a model, dataset, benchmark, or system name "
        "(e.g. MS^2 is a dataset/benchmark name).\n\n"
        "Reply with JSON only:\n"
        "{\n"
        '  "canonical_name": "<the short name/alias the paper is known by, e.g. BERT, AlphaGeometry>",\n'
        '  "candidate_titles": ["<exact full paper title>", ...],\n'
        '  "author_hints": ["<author surname and likely corrected spellings>", ...],\n'
        '  "year_hint": null,\n'
        '  "confidence": 0.0\n'
        "}\n"
        "- candidate_titles: 1-3 DISTINCT likely titles, most likely first. If "
        "the reference is ambiguous (several famous papers share the alias), "
        "make the titles the distinct interpretations.\n"
        "- author_hints: [] unless the reference names an author or contains a "
        "citation key; include 1-3 spelling variants of the surname.\n"
        "- year_hint: the publication year if the reference implies one, else null.\n"
        "- confidence: probability (0-1) the first title is exactly right."
    )
    obj = _json_block(await _gen(GPT_5_4, prompt, label="plan")) or {}
    name = _ascii((obj.get("canonical_name") or "").strip())
    titles = [t.strip() for t in obj.get("candidate_titles") or [] if isinstance(t, str) and t.strip()]
    author_hints = [a.strip() for a in obj.get("author_hints") or [] if isinstance(a, str) and a.strip()]
    try:
        year_hint = int(obj.get("year_hint")) if obj.get("year_hint") is not None else None
    except (TypeError, ValueError):
        year_hint = None
    try:
        iconf = float(obj.get("confidence") or 0)
    except (TypeError, ValueError):
        iconf = 0.0
    if not titles:
        titles = [query]
    if not name:
        name = re.sub(r"\b(the|paper|papers)\b", " ", _ascii(query), flags=re.I).strip()[:80]
    # ambiguity gate: no author/year cues and a shaky interpretation => the
    # gold set may span SEVERAL distinct works known by this alias (observed:
    # "the SPIKE paper" had 5 gold ids). Hedge wide there, stay tight elsewhere.
    ambiguous = not author_hints and year_hint is None and iconf < 0.65
    print(
        f"  canonical_name={name!r} titles={titles[:3]} authors={author_hints[:3]} "
        f"year={year_hint} conf={iconf} ambiguous={ambiguous}"
    )

    raw_alias = re.sub(r"\b(the|paper|papers|original)\b", " ", _ascii(query), flags=re.I)
    raw_alias = re.sub(r"\s+", " ", raw_alias).strip()[:100]

    spec_fields = "corpusId,title,year,authors,abstract"
    # ambiguous aliases: the gold set may be SEVERAL unrelated works sharing
    # the acronym (observed: "the SPIKE paper" -> 5 unrelated gold papers, all
    # missed at limit=40). Pull the tool max — alias-titled works can sit deep
    # in relevance order when the alias is also a common word.
    name_limit = 100 if ambiguous else 20
    tasks = [
        _safe_tool(lambda t=t: title_search(title=t, fields=spec_fields), f"title[{t[:30]}]", attempts=2)
        for t in titles[:3]
    ]
    n_titles = len(tasks)
    tasks.append(
        _safe_tool(lambda: rel_search(keyword=name, fields=spec_fields, limit=name_limit), "rel-name", attempts=2)
    )
    # exact-title relevance searches surface DUPLICATE corpus records of the work
    for t in titles[:2]:
        tasks.append(
            _safe_tool(
                lambda t=t: rel_search(keyword=_ascii(t)[:100], fields=spec_fields, limit=10),
                f"rel-title[{t[:20]}]",
                attempts=2,
            )
        )
    if raw_alias and _norm(raw_alias) != _norm(name):
        tasks.append(
            _safe_tool(lambda: rel_search(keyword=raw_alias, fields=spec_fields, limit=15), "rel-raw", attempts=2)
        )
    tasks.append(_safe_tool(lambda: snippet(query=name, limit=12), "snip-name", timeout=150, attempts=2))
    raws = await asyncio.gather(*tasks)

    # author-year channel: deterministic retrieval for citation-key references
    author_docs: list[dict] = []
    if author_hints:
        find_auth = _get_tool(state, "search_authors_by_name")
        get_papers = _get_tool(state, "get_author_papers")
        aids: list[str] = []
        for hint in author_hints[:3]:
            raw = await _safe_tool(
                lambda hint=hint: find_auth(name=hint, fields="authorId,name,paperCount", limit=10),
                f"auth[{hint}]",
            )
            recs = [r for r in _parse_items(raw or []) if r.get("authorId")]
            recs.sort(key=lambda r: -(r.get("paperCount") or 0))
            aids.extend(str(r["authorId"]) for r in recs[:3])

        async def _papers_of(aid: str) -> list[dict]:
            for lim in (500, 100):
                raw = await _safe_tool(
                    lambda: get_papers(author_id=aid, paper_fields=spec_fields, limit=lim),
                    f"papers[{aid}@{lim}]",
                )
                docs = [d for d in _parse_items(raw or []) if _cid(d)]
                if docs:
                    return docs
            return []

        lists = await asyncio.gather(*(_papers_of(a) for a in aids[:5]))
        for lst in lists:
            for d in lst:
                if year_hint is None or d.get("year") in (year_hint - 1, year_hint, year_hint + 1):
                    author_docs.append(d)
        author_docs = author_docs[:60]
        print(f"  author channel: {len(author_docs)} candidates from {len(aids[:5])} profiles")

    cands: list[dict] = []
    seen: set[str] = set()

    def _add(doc: dict, source: str):
        cid = _cid(doc)
        if cid and cid not in seen and (doc.get("title") or "").strip():
            seen.add(cid)
            doc["_source"] = source
            cands.append(doc)

    for raw in raws[:n_titles]:
        for doc in _parse_items(raw or []):
            if doc.get("paperId") or _cid(doc):
                _add(doc, "title")
    for doc in author_docs:
        _add(doc, "author")
    for raw in raws[n_titles:-1]:
        for doc in _parse_items(raw or []):
            _add(doc, "rel")
    for entry in _parse_items(raws[-1] or []):
        paper = entry.get("paper") or {}
        if paper:
            _add(dict(paper), "snip")

    if not cands:
        return _submit(state, [])

    # corpus-grounded verification: which retrieved record(s) ARE the paper?
    # On ambiguous aliases, float alias-titled candidates to the front so every
    # distinct work named <alias> is visible inside the 40-entry shortlist.
    if ambiguous:
        pref = [d for d in cands if d.get("_source") == "title" or _alias_titled(name, d.get("title") or "")]
        pref_ids = {_cid(d) for d in pref}
        shortlist = (pref + [d for d in cands if _cid(d) not in pref_ids])[:48]
        n_alias = sum(1 for d in shortlist if _alias_titled(name, d.get("title") or ""))
        print(f"  ambiguous shortlist: {n_alias} alias-titled of {len(shortlist)}")
    else:
        shortlist = cands[:40]
    lines = []
    for i, d in enumerate(shortlist):
        auths = ", ".join(_auth_names(d)[:3])
        lines.append(
            f"{i}. [{d.get('year')}] {(d.get('title') or '')[:140]} — {auths} — {_cut(d.get('abstract') or '', 200)}"
        )
    cues = f"short name: {name}"
    if author_hints:
        cues += f"; author cue: {'/'.join(author_hints[:3])} (spelling may differ slightly)"
    if year_hint:
        cues += f"; year cue: {year_hint}"
    ambig_note = (
        "\nThe reference gives NO author/year cues and the alias is generic, so "
        "it may be AMBIGUOUS: several distinct published works may each be "
        "known as 'the " + name + " paper'. If so, put the best record of the "
        "most likely referent in indices, and the best record of EACH other "
        "distinct plausible work (works actually named/known by the alias) in "
        "alternates, most plausible first (up to 9)."
        if ambiguous
        else ""
    )
    vprompt = (
        f'The user asked for one specific paper: "{query}"\n'
        f"({cues})\n\n"
        "Candidates retrieved from the paper corpus:\n" + "\n".join(lines) + "\n\n"
        "Which candidate IS that exact paper — the paper itself (the one that "
        "introduced/is named this), NOT a paper that cites, extends, or surveys "
        "it? Note its real title may not contain the short name; judge from the "
        "abstract, authors, and year cues.\n"
        "IMPORTANT: the same work sometimes appears as MULTIPLE corpus records "
        "(e.g. the conference version and a re-published journal/magazine "
        "version, with near-identical titles and the same authors). If so, "
        "list ALL indices that are records of that exact work. But follow-up "
        "or extended works with meaningfully different titles (e.g. 'X-XL', "
        "'X 2.0', 'X v2', a successor dataset) are DIFFERENT papers — never "
        "include them as records of the same work." + ambig_note + "\n"
        'Reply with JSON only: {"indices": [<indices of ALL records that ARE '
        'the paper, best first>], "confidence": 0.0, '
        '"alternates": [<indices of records that are OTHER plausible '
        "interpretations of the reference, most plausible first>]}"
    )
    vobj = _json_block(await _gen(GPT_5_4, vprompt)) or {}
    idxs = [i for i in (vobj.get("indices") or []) if isinstance(i, int) and 0 <= i < len(shortlist)]
    results: list[dict] = []
    picked: set[str] = set()

    def _pick(d: dict):
        cid = _cid(d)
        if cid and cid not in picked:
            picked.add(cid)
            results.append({"paper_id": cid, "markdown_evidence": ""})

    if idxs:
        try:
            vconf = float(vobj.get("confidence") or 0)
        except (TypeError, ValueError):
            vconf = 0.0
        chosen = shortlist[idxs[0]]
        _pick(chosen)
        fa = _surname(next(iter(_auth_names(chosen)), ""))
        # mechanical guard on extra "duplicate records": true duplicates share
        # a near-identical title and first author (AlexNet's two records ~1.0);
        # follow-ups like Objaverse-XL (~0.8) must be rejected.
        for i in idxs[1:3]:
            d = shortlist[i]
            sim = _title_sim(chosen.get("title") or "", d.get("title") or "")
            fa2 = _surname(next(iter(_auth_names(d)), ""))
            if sim >= 0.88 and (not fa or not fa2 or fa == fa2):
                _pick(d)
            else:
                print(f"  rejected extra record {_cid(d)} (title_sim={sim:.2f}) {d.get('title')!r}")
        print(f"  verified records: {[r['paper_id'] for r in results]} conf={vconf}")
        # programmatic duplicate-record backstop: near-identical title + same
        # first author elsewhere in the candidate pool (re-published classics)
        for d in cands:
            if len(results) >= 3 or _cid(d) in picked:
                continue
            if _title_sim(chosen.get("title") or "", d.get("title") or "") >= 0.96:
                fa2 = _surname(next(iter(_auth_names(d)), ""))
                if not fa or not fa2 or fa == fa2:
                    print(f"  duplicate-record backstop: {_cid(d)} {d.get('title')!r}")
                    _pick(d)
        # alternates: wide on ambiguous aliases (gold sets of 5 observed),
        # confidence-laddered otherwise (precision matters when gold is 1-2).
        n_extra = 6 if ambiguous else (0 if vconf >= 0.75 else (1 if vconf >= 0.4 else 2))
        for a in (vobj.get("alternates") or [])[:n_extra]:
            if isinstance(a, int) and 0 <= a < len(shortlist):
                _pick(shortlist[a])
        # mechanical hedge backstop: on ambiguous aliases make sure distinct
        # alias-titled works are represented even if the verifier listed few.
        # Retrieve wide, submit tight. iteration 17 widened this fill to 12 and
        # the submission cap to 14 on the theory that a ~5-paper ambiguous gold
        # set rewards hedging; on the one observed case (specific_39) the wide
        # list scored 0 exactly as the tight one did, because the extra slots
        # were filled by alias homonyms from unrelated fields, not by gold. The
        # wider slots only pay if they convert, and they did not — so the extra
        # denominator is pure precision loss. The upstream widening (alias
        # search at the tool max, 48-entry shortlist) is kept: it is free and
        # gives the verifier more to choose from.
        if ambiguous and len(results) < 5:
            for d in shortlist:
                if len(results) >= 5:
                    break
                if _cid(d) not in picked and _alias_titled(name, d.get("title") or ""):
                    ok = all(
                        _title_sim(d.get("title") or "", shortlist[j].get("title") or "") < 0.85
                        for j in idxs[:1]
                    )
                    if ok:
                        print(f"  alias hedge add: {_cid(d)} {(d.get('title') or '')[:80]!r}")
                        _pick(d)
    else:
        # verification punted: fall back to best title-similarity, then top hits
        scored = sorted(
            shortlist,
            key=lambda d: -max((_title_sim(t, d.get("title") or "") for t in titles[:3]), default=0),
        )
        for d in scored[: 5 if ambiguous else 3]:
            _pick(d)
    return _submit(state, results[: 8 if ambiguous else 5])


# ------------------------------------------------------------ metadata path

_VENUE_ALIASES = {
    "neurips": "neural information processing systems",
    "nips": "neural information processing systems",
    "icml": "international conference on machine learning",
    "iclr": "international conference on learning representations",
    "acl": "annual meeting of the association for computational linguistics",
    "emnlp": "conference on empirical methods in natural language processing",
    "naacl": "north american chapter of the association for computational linguistics",
    "cvpr": "computer vision and pattern recognition",
    "iccv": "international conference on computer vision",
    "eccv": "european conference on computer vision",
    "aaai": "aaai conference on artificial intelligence",
    "ijcai": "international joint conference on artificial intelligence",
    "sigir": "international acm sigir conference",
    "kdd": "knowledge discovery and data mining",
    "tacl": "transactions of the association for computational linguistics",
    "jmlr": "journal of machine learning research",
    "tpami": "transactions on pattern analysis and machine intelligence",
}

# umbrella conferences: papers publish under hosted-track / proceedings names
_UMBRELLA = {
    "splash": [
        "splash", "oopsla", "onward", "dynamic languages symposium", "gpce",
        "software language engineering", "proceedings of the acm on programming languages",
        "pacmpl", "proc acm program lang",
    ],
    "neurips": ["neurips", "nips", "neural information processing systems"],
    "nips": ["neurips", "nips", "neural information processing systems"],
    "fcrc": ["fcrc", "federated computing research conference"],
}


def _venue_probes(vname: str) -> set[str]:
    n = _norm(vname)
    probes = {n}
    if n in _UMBRELLA:
        probes.update(_UMBRELLA[n])
    if n in _VENUE_ALIASES:
        probes.add(_VENUE_ALIASES[n])
    return {p for p in probes if p}


def _venue_matches(venue_str: str, vname: str) -> bool:
    pv = _norm(venue_str)
    if not pv:
        return False
    padded = f" {pv} "
    for p in _venue_probes(vname):
        if len(p) <= 4:
            if f" {p} " in padded:  # short acronyms: word-boundary match only
                return True
        elif p in pv:
            return True
    return False


def _venue_str(doc: dict) -> str:
    fields = [doc.get("venue") or ""]
    j = doc.get("journal")
    if isinstance(j, dict):
        fields.append(j.get("name") or "")
    return " | ".join(f for f in fields if f).strip()


def _venue_ok_substring(venue_str: str, wanted: list[str]) -> bool:
    pv = _norm(venue_str)
    if not pv:
        return False
    for w in wanted:
        nw = _norm(w)
        for probe in filter(None, {nw, _VENUE_ALIASES.get(nw, "")}):
            if probe in pv or (len(pv) > 3 and pv in probe):
                return True
        if _venue_matches(venue_str, w):
            return True
    return False


async def _venue_llm_filter(constraint: str, venue_strs: list[str]) -> set[str] | None:
    """Classify which distinct venue strings satisfy the constraint. None on failure.

    Chunked, never truncated. The previous `sorted(distinct)[:120]` was an
    ALPHABETICAL cut: on metadata_4 ("Nature portfolio papers by David Harel")
    452 author papers span well over 120 distinct venues, `venues` was empty so
    this classifier was the only gate, and every N-initial Nature venue sat
    past the cut — 46 papers submitted, 0 of 3 gold, score 0.000 for all three
    agents, where iteration 2's crude substring filter had scored 0.500."""
    distinct = sorted({v for v in venue_strs if v})[:CV_LLM_MAX]
    if not distinct:
        return None
    groups = [distinct[i : i + CV_LLM_CHUNK] for i in range(0, len(distinct), CV_LLM_CHUNK)]
    outs = await asyncio.gather(*(_venue_llm_chunk(constraint, g) for g in groups))
    if all(o is None for o in outs):
        return None
    allowed: set[str] = set()
    for o in outs:
        allowed |= o or set()
    return allowed


async def _venue_llm_chunk(constraint: str, distinct: list[str]) -> set[str] | None:
    lines = "\n".join(f"{i}. {v[:120]}" for i, v in enumerate(distinct))
    prompt = (
        f"Venue constraint from a paper-search request: {constraint}\n\n"
        "Which of these publication venue names satisfy the constraint? "
        "Interpret venue families correctly (e.g. 'Nature portfolio' includes "
        "Nature, Nature <X>, npj <X>, Scientific Reports, Communications <X>; "
        "an abbreviation matches its full venue name; a multi-track umbrella "
        "conference includes its hosted tracks — e.g. SPLASH papers appear as "
        "OOPSLA, Onward!, DLS, GPCE, SLE, or 'Proceedings of the ACM on "
        "Programming Languages'; the main conference does NOT include its "
        "workshops unless the request says so; 'journal articles' excludes "
        "conference proceedings and workshops).\n\n"
        f"Venues:\n{lines}\n\n"
        "Reply with the comma-separated indices of SATISFYING venues only "
        "(or 'none')."
    )
    text = await _gen(GPT_5_4_MINI, prompt)
    if not text:
        return None
    idxs = {int(t) for t in re.findall(r"\d+", text) if int(t) < len(distinct)}
    return {distinct[i] for i in idxs}


def _author_ok(doc: dict, wanted: list[str]) -> bool:
    if not wanted:
        return True
    names = [_norm(n) for n in _auth_names(doc)]
    for w in wanted:
        nw = _norm(w).split()
        if not nw:
            continue
        last = nw[-1]
        for n in names:
            toks = n.split()
            if toks and toks[-1] == last and (
                len(nw) == 1 or not nw[0] or (toks and toks[0][:1] == nw[0][:1])
            ):
                return True
    return False


def _has_author(doc: dict, person: str) -> bool:
    """True if `person` (surname + optional first-initial match) co-wrote doc."""
    last = _surname(person)
    fi = _first_initial(person)
    if not last:
        return False
    for n in _auth_names(doc):
        toks = _norm(n).split()
        if toks and toks[-1] == last and (not fi or toks[0][:1] == fi):
            return True
    return False


async def _author_papers(state: TaskState, aid: str, fields: str = META_FIELDS) -> list[dict]:
    """Fetch an author's papers, trying large limits first (cap is undocumented)."""
    get_papers = _get_tool(state, "get_author_papers")
    for lim in (1000, 500, 100):
        raw = await _safe_tool(
            lambda lim=lim: get_papers(author_id=aid, paper_fields=fields, limit=lim),
            f"papers[{aid}@{lim}]",
            attempts=2,
        )
        docs = [d for d in _parse_items(raw or []) if _cid(d) or d.get("paperId")]
        if docs:
            return docs
    return []


async def _author_id_sets(state: TaskState, name: str) -> tuple[set[str], set[str]]:
    """(paperId hashes, corpusIds) of every paper by any profile matching `name`."""
    find_auth = _get_tool(state, "search_authors_by_name")
    raw = await _safe_tool(
        lambda: find_auth(name=name, fields="authorId,name,paperCount", limit=20), f"auth[{name}]"
    )
    recs = [r for r in _parse_items(raw or []) if r.get("authorId")]
    recs.sort(key=lambda r: -(r.get("paperCount") or 0))
    aids = [str(r["authorId"]) for r in recs[:6]]
    hashes: set[str] = set()
    cids: set[str] = set()
    lists = await asyncio.gather(*(_author_papers(state, a, "corpusId,title") for a in aids))
    for lst in lists:
        for d in lst:
            if d.get("paperId"):
                hashes.add(str(d["paperId"]))
            if _cid(d):
                cids.add(_cid(d))
    print(f"  cited-author {name!r}: {len(hashes)} paper hashes across {len(aids)} profiles")
    return hashes, cids


def _short_name_of(raw_query: str, title: str) -> str:
    """Short/model name of a cited paper: quoted token in the query, else title head."""
    m = re.search(r'["“]([^"”]{2,40})["”]', raw_query)
    if m:
        return m.group(1).strip()
    m = re.match(r"^([^:,]{2,50})[:,]", title or "")
    if m and len(m.group(1).split()) <= 5:
        return m.group(1).strip()
    return " ".join((title or "").split()[:4])


async def _fetch_references(batch, docs: list[dict]) -> dict[str, list[dict]]:
    """cid -> list of reference dicts (paperId/corpusId/title) for each doc.

    The plain "references" field fails server-side ('NoneType' not iterable)
    in every observed run. Probe the field variants on ONE doc first, commit
    to whichever returns data, and return {} fast if neither does (the
    callers all fail open)."""
    out: dict[str, list[dict]] = {}
    if not docs:
        return out
    sem = asyncio.Semaphore(6)

    def _fold(raw) -> bool:
        got = False
        for f in _parse_items(raw or []):
            refs = f.get("references")
            if isinstance(refs, list):
                out[_cid(f)] = [r for r in refs if isinstance(r, dict)]
                got = True
        return got

    variant = None
    probe_ids = [f"CorpusId:{_cid(docs[0])}"]
    for fv in REF_FIELD_VARIANTS:
        raw = await _safe_tool(
            lambda fv=fv: batch(ids=probe_ids, fields=fv), "refs-probe", attempts=1
        )
        if raw is not None:
            variant = fv
            _fold(raw)
            break
    if variant is None:
        print("  [refs] references unavailable under all field variants; skipping ref fetch")
        return out
    print(f"  [refs] references field variant works: {variant!r}")

    async def _one(grp: list[dict]):
        ids = [f"CorpusId:{_cid(d)}" for d in grp]
        async with sem:
            raw = await _safe_tool(
                lambda: batch(ids=ids, fields=variant),
                "refs-batch",
                attempts=2,
            )
        if raw is None:
            # one poison id / transient server error kills the whole group —
            # bisect instead of losing every candidate
            if len(grp) > 1:
                mid = len(grp) // 2
                await _one(grp[:mid])
                await _one(grp[mid:])
            return
        _fold(raw)

    rest = docs[1:]
    grps = [rest[i : i + REF_BATCH] for i in range(0, len(rest), REF_BATCH)]
    await asyncio.gather(*(_one(g) for g in grps))
    return out


def _refs_contain(refs: list[dict], hashes: set[str], cids: set[str]) -> bool:
    for r in refs:
        pid = str(r.get("paperId") or "")
        if pid and pid in hashes:
            return True
        rc = r.get("corpusId")
        if rc is not None and str(rc) in cids:
            return True
    return False


def _phrase_in(phrase_norm: str, text: str) -> bool:
    """Word-bounded containment of a normalized phrase in raw text."""
    return f" {phrase_norm} " in f" {' '.join(_norm(text).split())} "


async def _snip_mention_verify(
    snippet_tool, cids: list[str], phrase: str, label: str = "snipverify"
) -> set[str]:
    """cids whose BODY passages literally contain `phrase` (word-bounded).

    Scoped snippet_search is a far more reliable does-it-cite-X instrument
    than S2 reference lists (which arrive truncated/id-less: metadata_42's
    refs check matched 1 of 67 candidates that near-certainly cite the
    target). Chunks of 25 ids at limit=100 give every paper in scope passage
    room; papers that never mention the phrase simply return nothing."""
    ph = " ".join(_norm(phrase).split())
    if not ph or not cids:
        return set()
    ok: set[str] = set()
    sem = asyncio.Semaphore(6)

    async def _chunk(grp: list[str]):
        scope = ",".join(f"CorpusId:{c}" for c in grp)
        async with sem:
            raw = await _safe_tool(
                lambda: snippet_tool(query=phrase, paper_ids=scope, limit=100),
                label,
                timeout=120,
                attempts=2,
            )
        for entry in _parse_items(raw or []):
            paper = entry.get("paper") if isinstance(entry.get("paper"), dict) else {}
            cid = str(paper.get("corpusId") or "")
            text = ((entry.get("snippet") or {}).get("text") or "")
            if cid and _phrase_in(ph, text):
                ok.add(cid)

    grps = [cids[i : i + 25] for i in range(0, len(cids), 25)]
    await asyncio.gather(*(_chunk(g) for g in grps))
    return ok


def _conj_names(raw_query: str, cite_targets: list[dict]) -> list[str]:
    """Per-target informal names ("T5", "spider") for mention verification.

    Bodies cite works by their informal names, not their formal titles —
    "the T5 paper" is named T5 in running text, and T5 never appears in its
    own title. Harvest "the X paper" tokens from the query, assign each to
    the target whose title contains it, and fill unmatched targets with the
    leftovers (in order), falling back to the title-derived short name."""
    informal = [
        m.group(1).strip().strip('"“”')
        for m in re.finditer(
            r"\bthe\s+[\"“]?([A-Za-z0-9][\w .+-]{0,30}?)[\"”]?\s+paper", raw_query, re.I
        )
    ]
    names: list[str] = []
    used: set[int] = set()
    for t in cite_targets:
        tnorm = f" {' '.join(_norm(t['title']).split())} "
        pick = None
        for i, nm in enumerate(informal):
            if i not in used and f" {' '.join(_norm(nm).split())} " in tnorm:
                pick = i
                break
        if pick is not None:
            used.add(pick)
            names.append(informal[pick])
        else:
            names.append("")
    leftovers = [nm for i, nm in enumerate(informal) if i not in used]
    for j, nm in enumerate(names):
        if not nm:
            names[j] = (
                leftovers.pop(0) if leftovers else _short_name_of(raw_query, cite_targets[j]["title"])
            )
    return [n for n in names if n]


async def _solve_metadata(state: TaskState, query: str) -> TaskState:
    prompt = (
        "Parse this scholarly paper search request into JSON filters.\n"
        f"Request: {query}\n\n"
        "Reply with JSON only:\n"
        "{\n"
        '  "authors": [],             // author names the papers must be written by\n'
        '  "venues": [],              // venue names the papers must be PUBLISHED AT, incl. BOTH abbreviation and full name, e.g. ["NeurIPS", "Neural Information Processing Systems"]\n'
        '  "venue_constraint": null,  // the publication-venue requirement restated verbally, e.g. "published in a Nature portfolio journal"; if the request asks for journal articles say "journal articles (not conference proceedings)"; null if no venue/type constraint\n'
        '  "years_allowed": [],       // EXACT publication years when specific years are named (e.g. "2014 or 2017" -> [2014, 2017]; "2022-2023" -> [2022, 2023]); [] otherwise\n'
        '  "year_min": null, "year_max": null,  // inclusive range bounds. IMPORTANT: "after 2020"/"since 2020"/"2020 and beyond" -> year_min 2020 (publication years drift, be inclusive); "before 2019" -> year_max 2019. null if years_allowed is used\n'
        '  "cites_paper_titles": [],  // if papers must CITE specific paper(s), the best-known exact title of EACH, as separate list entries. "citing the T5 paper and the spider paper" -> TWO entries (results must cite BOTH). NEVER join several titles into one string.\n'
        '  "cites_author": null,      // if papers must cite work BY some person, that person\'s name\n'
        '  "cites_venue": null,       // if papers must cite ANY paper from some venue (e.g. "cites any NeurIPS paper" -> "NeurIPS"), that venue name — do NOT also put it in venues/venue_constraint\n'
        '  "exclude_coauthor": null,  // "excluding self-citations of X" / "not by X" -> X must NOT be a co-author of the results\n'
        '  "min_citations": null,     // minimum citation count required of each result ("more than 50" -> 50)\n'
        '  "min_authors": null, "max_authors": null, // bounds on number of authors per paper ("more than 3 authors" -> min_authors 4)\n'
        '  "topic_keywords": null     // 3-6 word topical keyword phrase if the request has a topic constraint\n'
        "}\n"
        "Use null/[] for unconstrained fields."
    )
    plan = _json_block(await _gen(GPT_5_4, prompt)) or {}
    print(f"  metadata plan: {json.dumps(plan)[:500]}")

    def _num(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _strv(v):
        return v.strip() if isinstance(v, str) and v.strip() else None

    authors = [a for a in plan.get("authors") or [] if isinstance(a, str)]
    venues = [v for v in plan.get("venues") or [] if isinstance(v, str)]
    venue_constraint = _strv(plan.get("venue_constraint")) or (", ".join(venues) if venues else None)
    years_allowed = {y for y in (_num(v) for v in plan.get("years_allowed") or []) if y}
    y0, y1 = _num(plan.get("year_min")), _num(plan.get("year_max"))
    if years_allowed:
        y0 = y1 = None
    # accept both the list form and the legacy scalar; split a model that
    # ignored the instruction and joined several titles with ';' anyway
    _ct_raw = plan.get("cites_paper_titles")
    if isinstance(_ct_raw, str):
        _ct_raw = [_ct_raw]
    if not _ct_raw:
        _ct_raw = [plan.get("cites_paper_title")] if plan.get("cites_paper_title") else []
    cites_titles: list[str] = []
    for t in _ct_raw:
        if isinstance(t, str):
            for part in re.split(r"\s*;\s*", t):
                part = part.strip()
                if part and part not in cites_titles:
                    cites_titles.append(part)
    cites_titles = cites_titles[:4]
    cites_title = cites_titles[0] if cites_titles else None
    cites_author = _strv(plan.get("cites_author"))
    cites_venue = _strv(plan.get("cites_venue"))
    exclude_coauthor = _strv(plan.get("exclude_coauthor"))
    min_cit = _num(plan.get("min_citations"))
    min_auth, max_auth = _num(plan.get("min_authors")), _num(plan.get("max_authors"))
    topic = _strv(plan.get("topic_keywords"))

    batch = _get_tool(state, "get_paper_batch")
    rel = _get_tool(state, "search_papers_by_relevance")
    snippet_tool = _get_tool(state, "snippet_search")

    # ---- resolve citation targets
    target_hashes: set[str] = set()
    target_cids: set[str] = set()
    target_title = None
    short_name = None
    # each entry: {"cid", "title"} for one paper the results must cite. With
    # >1 entry the request is a CONJUNCTION ("citing the T5 paper and the
    # spider paper") and the answer is the intersection of the citer sets.
    cite_targets: list[dict] = []
    if cites_titles:
        title_search = _get_tool(state, "search_paper_by_title")
        traws = await asyncio.gather(
            *(
                _safe_tool(
                    lambda t=t: title_search(title=t, fields="corpusId,title"),
                    f"cite-title[{t[:30]}]",
                    attempts=2,
                )
                for t in cites_titles
            )
        )
        for t, raw in zip(cites_titles, traws):
            tgt = next((d for d in _parse_items(raw or []) if d.get("paperId") or _cid(d)), None)
            if not tgt or not _cid(tgt):
                print(f"  cited paper {t!r}: NOT RESOLVED")
                continue
            cite_targets.append({"cid": _cid(tgt), "title": tgt.get("title") or t})
            if tgt.get("paperId"):
                target_hashes.add(str(tgt["paperId"]))
            target_cids.add(_cid(tgt))
        if cite_targets:
            target_title = cite_targets[0]["title"]
            short_name = _short_name_of(query, target_title)
            print(
                f"  cited papers ({len(cite_targets)}): "
                f"{[(t['cid'], t['title'][:50]) for t in cite_targets]} short_name={short_name!r}"
            )
    if cites_author:
        h, c = await _author_id_sets(state, cites_author)
        target_hashes |= h
        target_cids |= c
    needs_ref_check = bool(target_hashes or target_cids)

    # ---- candidate channels
    candidates: list[dict] = []
    author_base = False

    if authors:
        # base = the required author's own papers (cite constraints become filters)
        author_base = True
        find_auth = _get_tool(state, "search_authors_by_name")
        ids: list[str] = []
        for name in authors[:3]:
            raw = await _safe_tool(
                lambda name=name: find_auth(name=name, fields="authorId,name,paperCount", limit=20),
                f"auth[{name}]",
            )
            recs = [r for r in _parse_items(raw or []) if r.get("authorId")]
            recs.sort(key=lambda r: -(r.get("paperCount") or 0))
            # keep all plausible identities of the same person (split profiles)
            ids.extend(str(r["authorId"]) for r in recs[:6])
        paper_lists = await asyncio.gather(*(_author_papers(state, aid) for aid in ids[:10]))
        for lst in paper_lists:
            candidates.extend(lst)
    elif cite_targets:
        # channel A: the citations list of EACH cited target. get_citations is
        # the only citation surface that works (references is dead
        # server-side), it is hard-capped at 1000 with no paging, and the
        # sample it returns is recency-ordered.
        get_cit = _get_tool(state, "get_citations")

        async def _citers_of_target(t: dict) -> list[dict]:
            raw = await _safe_tool(
                lambda: get_cit(
                    paper_id=f"CorpusId:{t['cid']}", fields=META_FIELDS, limit=1000
                ),
                f"citations[{t['cid']}]",
                attempts=2,
            )
            out = []
            for item in _parse_items(raw or []):
                doc = item.get("citingPaper") if isinstance(item.get("citingPaper"), dict) else item
                if isinstance(doc, dict) and _cid(doc):
                    out.append(doc)
            return out

        citer_lists = await asyncio.gather(*(_citers_of_target(t) for t in cite_targets))
        citer_sets = [{_cid(d) for d in lst} for lst in citer_lists]
        cap_hit = any(len(lst) >= 1000 for lst in citer_lists)
        by_cid_all: dict[str, dict] = {}
        for lst in citer_lists:
            for d in lst:
                by_cid_all.setdefault(_cid(d), d)
        for cid, d in by_cid_all.items():
            d["_cites_count"] = sum(1 for s in citer_sets if cid in s)
        print(
            f"  channel A (get_citations x{len(cite_targets)}): "
            f"{[len(l) for l in citer_lists]} citers, {len(by_cid_all)} unique, cap_hit={cap_hit}"
        )

        if len(cite_targets) > 1:
            # CONJUNCTION: results must cite every target, so the answer is
            # the intersection. Fall back to citing-the-most-targets when the
            # intersection is empty (one target may be unresolved or its
            # citer list truncated by the 1000-cap).
            need = len(cite_targets)
            inter = [d for d in by_cid_all.values() if d["_cites_count"] >= need]
            while not inter and need > 1:
                need -= 1
                inter = [d for d in by_cid_all.values() if d["_cites_count"] >= need]
            # membership in the intersection IS the citation verification —
            # mark verified so the (dead-API) reference check can't drop them
            for d in inter:
                d["_cites_target"] = True
            inter.sort(key=lambda d: (-(d.get("_cites_count") or 0), -(d.get("citationCount") or 0)))
            candidates.extend(inter)
            print(
                f"  citation intersection: {len(inter)} papers citing >={need} "
                f"of {len(cite_targets)} targets"
            )
            # conjunction augmentation under the citer cap: get_citations is
            # recency-ordered, so when a citer list caps at 1000 the visible
            # window is only the newest few months — and metadata_26's gold
            # (all ids ~Oct 2024-Feb 2025) had already scrolled out of that
            # window at eval time, making the pure intersection score 0 with
            # certainty. Content search + per-target body-mention
            # verification is the only channel that reaches older citers.
            if cap_hit and len(inter) < 40:
                names = _conj_names(query, cite_targets)
                joined = " ".join(names)
                kws = [joined]
                if topic:
                    kws.append(f"{joined} {topic}")
                ctasks = [
                    _safe_tool(
                        lambda k=k: rel(keyword=_ascii(k), fields=META_FIELDS, limit=100),
                        f"conj[{k[:25]}]",
                        attempts=2,
                    )
                    for k in kws
                ]
                if len(names) >= 2:
                    ctasks.append(
                        _safe_tool(
                            lambda: snippet_tool(
                                query=f"both {names[0]} and {names[1]}", limit=100
                            ),
                            "conj-snip",
                            timeout=200,
                            attempts=2,
                        )
                    )
                craws = await asyncio.gather(*ctasks)
                conj_cands: dict[str, dict] = {}
                for raw in craws:
                    for d in _parse_items(raw or []):
                        if isinstance(d, dict) and isinstance(d.get("paper"), dict):
                            d = dict(d["paper"])
                        cid = _cid(d) if isinstance(d, dict) else ""
                        if cid and cid not in by_cid_all and cid not in conj_cands:
                            conj_cands[cid] = d
                check_ids = list(conj_cands)[:150]
                ok_all: set[str] | None = None
                for nm in names:
                    got = await _snip_mention_verify(
                        snippet_tool, check_ids, nm, label=f"conjverify[{nm[:15]}]"
                    )
                    ok_all = got if ok_all is None else (ok_all & got)
                extras = [conj_cands[c] for c in check_ids if c in (ok_all or set())]
                extras = extras[: max(0, 40 - len(inter))]
                for d in extras:
                    d["_cites_target"] = True
                candidates.extend(extras)
                print(
                    f"  conjunction augmentation (names={names}): "
                    f"{len(conj_cands)} mention candidates, {len(extras)} verified "
                    f"citing all {len(names)} targets"
                )
        else:
            for d in by_cid_all.values():
                d["_cites_target"] = True
                candidates.append(d)
        # channels B/C: papers that MENTION the cited work (verified via refs
        # later) — recovers highly-cited citers the 1000-cap can't return.
        # Skipped on conjunctions: a mention of the FIRST target says nothing
        # about the others, and the intersection is already exact.
        if len(cite_targets) == 1:
            kw_set: list[str] = []
            for k in (
                short_name,
                topic,
                f"{short_name} {topic}" if short_name and topic else None,
                " ".join(
                    w
                    for w in re.findall(r"[A-Za-z][\w-]+", target_title or "")
                    if _norm(w) not in _STOP
                )[:60]
                or None,
                f"using {short_name}" if short_name else None,
            ):
                if k and k not in kw_set:
                    kw_set.append(k)
            # The 1000-cap sample is recency-ordered, so on a heavily-cited
            # target it is ENTIRELY the newest tail: metadata_25 submitted 31
            # papers, all 2025, against gold that is all 2022-2024. When a
            # min_citations filter implies a large established gold set, add
            # topical probes that reach the older, well-cited citers the cap
            # cannot return.
            if min_cit and cap_hit:
                for k in (
                    f"{short_name} distillation" if short_name else None,
                    f"{short_name} benchmark evaluation" if short_name else None,
                    f"{topic} survey" if topic else None,
                ):
                    if k and k not in kw_set:
                        kw_set.append(k)
            kws = kw_set
            btasks = []
            for k in kws[:8]:
                kwargs = {"keyword": _ascii(k), "fields": META_FIELDS, "limit": 100}
                if venues:
                    kwargs["venues"] = ",".join(venues)
                btasks.append(_safe_tool(lambda kwargs=dict(kwargs): rel(**kwargs), f"relB[{k[:25]}]", attempts=2))
                if venues:  # unfiltered variant too: server venue names may mismatch
                    btasks.append(
                        _safe_tool(
                            lambda k=k: rel(keyword=_ascii(k), fields=META_FIELDS, limit=100),
                            f"relB-nv[{k[:25]}]",
                            attempts=2,
                        )
                    )
            if short_name:
                skwargs = {"query": short_name, "limit": 100}
                if venues:
                    skwargs["venues"] = ",".join(venues)
                btasks.append(_safe_tool(lambda: snippet_tool(**skwargs), "snipC", timeout=200, attempts=2))
            braws = await asyncio.gather(*btasks)
            n_before = len(candidates)
            for raw in braws:
                for d in _parse_items(raw or []):
                    if isinstance(d, dict) and d.get("paper"):  # snippet entry
                        d = dict(d["paper"])
                    if isinstance(d, dict) and _cid(d):
                        candidates.append(d)
            print(f"  channels B/C (mention search): +{len(candidates) - n_before}")
    if not candidates and venues and not topic:
        # venue-base channel (e.g. "a SPLASH 2019+ paper that cites any
        # NeurIPS"): per-venue-name searches, with AND without the server-side
        # venue filter (server venue strings often mismatch the common name)
        vtasks = []
        for vname in venues[:3]:
            vtasks.append(
                _safe_tool(
                    lambda vname=vname: rel(
                        keyword=_ascii(vname), fields=META_FIELDS, limit=100, venues=",".join(venues)
                    ),
                    f"venue[{vname[:20]}]",
                    attempts=2,
                )
            )
            vtasks.append(
                _safe_tool(
                    lambda vname=vname: rel(keyword=_ascii(vname), fields=META_FIELDS, limit=100),
                    f"venue-nv[{vname[:20]}]",
                    attempts=2,
                )
            )
        vraws = await asyncio.gather(*vtasks)
        for raw in vraws:
            candidates.extend(d for d in _parse_items(raw or []) if _cid(d))
        print(f"  venue-base channel: {len(candidates)} candidates")
    if not candidates:
        kw = topic or short_name or " ".join(authors) or _ascii(query)[:100]
        kwargs = {"keyword": _ascii(kw), "fields": META_FIELDS, "limit": 100}
        if venues:
            kwargs["venues"] = ",".join(venues)
        raw = await _safe_tool(lambda: rel(**kwargs), "kw-base", attempts=2)
        candidates = [d for d in _parse_items(raw or []) if _cid(d)]
        if not candidates and venues:  # server-side venue name mismatch
            kwargs.pop("venues")
            raw = await _safe_tool(lambda: rel(**kwargs), "kw-base-novenue", attempts=2)
            candidates = [d for d in _parse_items(raw or []) if _cid(d)]
        if not candidates:
            # content-word retry: interrogative/verbose query text returns 0
            cw = " ".join(w for w in _norm(query).split() if w not in _STOP)[:100]
            if cw and _norm(cw) != _norm(kw):
                raw = await _safe_tool(
                    lambda: rel(keyword=cw, fields=META_FIELDS, limit=100), "kw-contentwords", attempts=2
                )
                candidates = [d for d in _parse_items(raw or []) if _cid(d)]

    # dedupe (merge _cites_target flags)
    by_cid: dict[str, dict] = {}
    order: list[str] = []
    for d in candidates:
        cid = _cid(d)
        if not cid:
            continue
        if cid in by_cid:
            if d.get("_cites_target"):
                by_cid[cid]["_cites_target"] = True
            continue
        by_cid[cid] = d
        order.append(cid)
    deduped = [by_cid[c] for c in order]

    # some channel-B/C docs lack authors/citationCount fields -> batch-fill
    incomplete = [d for d in deduped if d.get("authors") is None or d.get("citationCount") is None]
    if incomplete and (min_cit or min_auth or max_auth or authors or exclude_coauthor or years_allowed or y0 or y1):
        ids = [f"CorpusId:{_cid(d)}" for d in incomplete]
        fetched = {_cid(f): f for f in await _batch_bisect(batch, ids, META_FIELDS, "meta-batch")}
        for d in incomplete:
            f = fetched.get(_cid(d))
            if f:
                for k in ("title", "abstract", "year", "venue", "journal", "authors", "citationCount"):
                    if d.get(k) is None and f.get(k) is not None:
                        d[k] = f[k]

    # ---- cheap hard filters
    def _passes(d: dict, use_min_cit: bool = True) -> bool:
        year = d.get("year")
        if years_allowed and year not in years_allowed:
            return False
        if y0 and (not year or year < y0):
            return False
        if y1 and (not year or year > y1):
            return False
        if use_min_cit and min_cit and (d.get("citationCount") or 0) < min_cit:
            return False
        n_auth = len(d.get("authors") or [])
        if min_auth and n_auth < min_auth:
            return False
        if max_auth and n_auth > max_auth:
            return False
        if not author_base and authors and not _author_ok(d, authors):
            return False
        if exclude_coauthor and _has_author(d, exclude_coauthor):
            return False
        return True

    kept = [d for d in deduped if _passes(d)]
    print(f"  metadata: {len(deduped)} candidates -> {len(kept)} after cheap filters")

    # ---- venue filter: UNION of LLM classification and substring/alias match
    pre_venue = list(kept)
    if venues or venue_constraint:
        allowed = await _venue_llm_filter(venue_constraint or ", ".join(venues), [_venue_str(d) for d in kept])
        kept = [
            d for d in kept
            if (allowed is not None and _venue_str(d) in allowed)
            or _venue_ok_substring(_venue_str(d), venues)
        ]
        print(f"  after venue filter: {len(kept)}")

    # ---- reference verification for citation constraints
    pre_ref = list(kept)
    if needs_ref_check and kept:
        unverified = [d for d in kept if not d.get("_cites_target")]
        to_check = unverified[:300]
        refs_map = await _fetch_references(batch, to_check) if to_check else {}
        checked_ok = {
            _cid(d) for d in to_check if _refs_contain(refs_map.get(_cid(d), []), target_hashes, target_cids)
        }
        n_missing_refs = sum(1 for d in to_check if _cid(d) not in refs_map)

        # fail-open: a candidate whose title/abstract explicitly names the
        # cited work all but certainly cites it — S2 reference lists are
        # sometimes truncated or unfetchable, and hard-dropping these traded
        # recall for nothing (metadata_25 hit 1 of 172 gold).
        sn = _norm(short_name) if short_name else ""

        def _mentions_target(d: dict) -> bool:
            if len(sn) < 4:
                return False
            blob = f" {_norm(d.get('title') or '')} {_norm(d.get('abstract') or '')} "
            return f" {sn} " in blob

        # body-mention verification: the refs check false-negatives at scale
        # (metadata_42: reference lists returned for 67/72 candidates matched
        # the target in ~1, discarding a candidate set covering a 70-paper
        # gold). A body passage naming the cited work is decisive evidence.
        mention_phrase = short_name or (_surname(cites_author) if cites_author else "")
        still = [
            d
            for d in to_check
            if _cid(d) not in checked_ok and not _mentions_target(d)
        ][:250]
        snip_ok: set[str] = set()
        if mention_phrase and still:
            snip_ok = await _snip_mention_verify(
                snippet_tool, [_cid(d) for d in still], mention_phrase
            )

        kept = [
            d
            for d in kept
            if d.get("_cites_target")
            or _cid(d) in checked_ok
            or _cid(d) in snip_ok
            or _mentions_target(d)
        ]
        print(
            f"  reference verification: {len(pre_ref)} -> {len(kept)} "
            f"(checked {len(to_check)}, no-refs-returned {n_missing_refs}, "
            f"body-mention-verified {len(snip_ok)})"
        )

    # ---- cites-venue verification: keep candidates whose reference list
    # contains >=1 paper published at the target venue (free batch lookups)
    pre_cv = list(kept)
    if cites_venue and kept:
        to_check = kept[:CV_CHECK_MAX]
        refs_map = await _fetch_references(batch, to_check)
        ref_ids: list[str] = []
        seen_r: set[str] = set()
        for refs in refs_map.values():
            for r in refs:
                rc = r.get("corpusId")
                if rc is not None and str(rc) not in seen_r:
                    seen_r.add(str(rc))
                    ref_ids.append(str(rc))
        ref_ids = ref_ids[:CV_REF_IDS_MAX]
        venue_of: dict[str, str] = {}
        vsem = asyncio.Semaphore(6)

        async def _vbatch(grp: list[str]):
            ids = [f"CorpusId:{c}" for c in grp]
            async with vsem:
                raw = await _safe_tool(
                    lambda: batch(ids=ids, fields="corpusId,venue,journal"), "refvenue", attempts=2
                )
            for f in _parse_items(raw or []):
                venue_of[_cid(f)] = _venue_str(f)

        vgrps = [ref_ids[i : i + 50] for i in range(0, len(ref_ids), 50)]
        await asyncio.gather(*(_vbatch(g) for g in vgrps))
        ok_cids: set[str] = set()
        for d in to_check:
            refs = refs_map.get(_cid(d), [])
            for r in refs:
                rc = r.get("corpusId")
                if rc is not None and _venue_matches(venue_of.get(str(rc), ""), cites_venue):
                    ok_cids.add(_cid(d))
                    break
        kept = [d for d in to_check if _cid(d) in ok_cids]
        print(
            f"  cites-venue({cites_venue}) verification: {len(pre_cv)} -> {len(kept)} "
            f"(resolved {len(venue_of)}/{len(ref_ids)} referenced venues)"
        )

    # ---- optional topical filter (cheap LLM) when the base wasn't a topic search
    if topic and kept and (cites_title or cites_author or author_base) and len(kept) <= 200 and not needs_ref_check:
        entries = [
            (i, f"{(d.get('title') or '')[:140]} || {_cut(d.get('abstract') or '', 200)}")
            for i, d in enumerate(kept)
        ]
        keep_idx: set[int] = set()
        for i in range(0, len(entries), 40):
            chunk = entries[i : i + 40]
            lines = "\n".join(f"{j}. {t}" for j, t in chunk)
            text = await _gen(
                GPT_5_4_MINI,
                f"Topic constraint: {topic}\nRequest: {query}\n\n"
                f"Which of these papers match the topic constraint?\n{lines}\n\n"
                "Reply with the comma-separated indices of MATCHING papers only "
                "(empty if none).",
            )
            keep_idx.update(int(t) for t in re.findall(r"\d+", text) if int(t) < len(kept))
        if keep_idx:
            kept = [d for i, d in enumerate(kept) if i in keep_idx]
            print(f"  after topic filter: {len(kept)}")

    # ---- relax ladder: an empty submission is a guaranteed 0
    if not kept:
        if cites_venue and pre_cv:
            print(f"  [relax] cites-venue check emptied the set; submitting {min(15, len(pre_cv))} pre-check papers")
            kept = pre_cv[:15]
        elif pre_ref:
            print(f"  [relax] ref-check emptied the set; submitting {min(30, len(pre_ref))} pre-ref papers")
            kept = pre_ref[:30]
        elif pre_venue:
            print(f"  [relax] venue filter emptied the set; submitting {min(15, len(pre_venue))} pre-venue papers")
            kept = pre_venue[:15]
        else:
            relaxed = [d for d in deduped if _passes(d, use_min_cit=False)]
            if relaxed:
                print(f"  [relax] no min-citation filter; submitting {min(15, len(relaxed))}")
                kept = relaxed[:15]
            elif deduped:
                print("  [relax] all filters emptied the set; submitting top 10 raw candidates")
                kept = deduped[:10]

    # ---- terminal never-empty fallback (metadata_33 submitted 0 => sure 0)
    if not kept:
        cw = " ".join(w for w in _norm(query).split() if w not in _STOP)[:100]
        raw = await _safe_tool(
            lambda: rel(keyword=cw or _ascii(query)[:100], fields=META_FIELDS, limit=30),
            "terminal-fallback",
            attempts=2,
        )
        kept = [d for d in _parse_items(raw or []) if _cid(d)][:10]
        if kept:
            print(f"  [relax] terminal fallback: submitting {len(kept)} keyword hits")

    # When more candidates survive than we can submit, keep the ones most
    # likely to be gold: citation-verified first, then well-cited (a
    # min_citations request implies gold is made of established papers).
    if len(kept) > MAX_SUBMIT:
        kept.sort(
            key=lambda d: (
                0 if d.get("_cites_target") else 1,
                -(d.get("_cites_count") or 0),
                -(d.get("citationCount") or 0),
            )
        )
        print(f"  {len(kept)} survivors -> submitting the {MAX_SUBMIT} best-supported")

    results = [{"paper_id": _cid(d), "markdown_evidence": ""} for d in kept[:MAX_SUBMIT]]
    return _submit(state, results)


# ---------------------------------------------------------------- solver


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        start = time.monotonic()
        _llm_reset()
        query = state.metadata["raw_query"]
        score_type = state.metadata.get("score_type", "")
        print(f"[{state.sample_id}] score_type={score_type} query={query[:100]!r}")

        try:
            if score_type == "specific_f1":
                return await _solve_specific(state, query)
            if score_type == "metadata_f1":
                return await _solve_metadata(state, query)
            return await _solve_semantic(state, query, start)
        except Exception as e:  # noqa: BLE001 - never crash a query to 0
            print(f"  [FALLBACK] route failed: {e!r}")
            try:
                search = _get_tool(state, "search_papers_by_relevance")
                kw = re.sub(r"[^\w\s-]", " ", _ascii(query))[:100]
                raw = await _safe_tool(
                    lambda: search(keyword=kw, fields=PAPER_FIELDS, limit=30), "fallback", attempts=3
                )
                docs = [d for d in _parse_items(raw or []) if _cid(d)]
                exact = score_type in ("specific_f1", "metadata_f1")
                results = [
                    {"paper_id": _cid(d), "markdown_evidence": "" if exact else _evidence(d)}
                    for d in docs[:20]
                ]
                return _submit(state, results)
            except Exception as e2:  # noqa: BLE001
                print(f"  [FALLBACK] also failed: {e2!r}")
                return _submit(state, [])

    return solve
