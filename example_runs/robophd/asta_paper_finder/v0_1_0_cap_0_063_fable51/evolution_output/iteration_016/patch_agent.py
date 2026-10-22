import ast, re
src = open("agent.py").read()

def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)

# ---- 1. cost trims (semantic path) ----
rep("GRADE_ABSTRACT_CHARS = 440\n", "GRADE_ABSTRACT_CHARS = 400\n")
rep("GRADE_EXPANSION = 160           # citation-expansion candidates graded in stage 2\n",
    "GRADE_EXPANSION = 120           # expansion candidates (citers + title-search neighbours) graded in stage 2\n"
    "PRF_SEEDS = 6                   # top graded papers whose titles are re-searched as keyword queries\n"
    "PRF_LIMIT = 40                  # hits kept per title search\n")
rep("GRADE_MAX = 680\n", "GRADE_MAX = 600\n")
rep("HEAD_RERANK = 36                # papers re-judged per-criterion by GPT-5.4\n",
    "HEAD_RERANK = 32                # papers re-judged per-criterion by GPT-5.4\n")

# ---- 2. stage 2: pseudo-relevance feedback via title searches on the seeds ----
rep("""        known = set(cands.d.keys())
        cit_lists = await asyncio.gather(*[
            _call(tools, "get_citations", timeout=90, paper_id=f"CorpusId:{s['cid']}",
                  fields="title,abstract,corpusId,year,publicationDate,venue,citationCount", limit=150)
            for s in seeds], return_exceptions=True)
        new_c = {}
""", """        known = set(cands.d.keys())
        # the titles of the best graded papers are the most precise topical queries we have:
        # lexical neighbours of a known-relevant title (pseudo-relevance feedback) reach the
        # relevant papers whose wording the planner's queries missed
        prf_seeds = [s for s in seeds[:PRF_SEEDS] if (s.get("title") or "").strip()]
        cit_lists, prf_lists = await asyncio.gather(
            asyncio.gather(*[
                _call(tools, "get_citations", timeout=90, paper_id=f"CorpusId:{s['cid']}",
                      fields="title,abstract,corpusId,year,publicationDate,venue,citationCount", limit=150)
                for s in seeds], return_exceptions=True),
            asyncio.gather(*[
                _call(tools, "search_papers_by_relevance", timeout=60,
                      keyword=_trim_passage(re.sub(r"[^\\w\\s-]", " ", s["title"]), 200),
                      fields=PAPER_FIELDS, limit=PRF_LIMIT)
                for s in prf_seeds], return_exceptions=True))
        new_c = {}
        n_prf = 0
        for s, pl in zip(prf_seeds, prf_lists):
            if isinstance(pl, Exception):
                print(f"  !! stage2 title search failed: {_err_text(pl)[:120]}")
                continue
            for rank, p in enumerate(pl):
                if not isinstance(p, dict):
                    continue
                cid = _cid(p)
                if not cid or cid in known:
                    continue
                c = cands.add(p, rank, 0.6)
                if c is not None:
                    if cid not in new_c:
                        n_prf += 1
                    new_c[cid] = c
        print(f"  stage2: {len(prf_seeds)} title searches -> {n_prf} new neighbours")
""")
rep("""        print(f"  stage2: {len(seeds)} seeds -> {len(new_c)} new citers, graded {n2}/{len(newl)} ({time.time()-t0:.0f}s)")""",
    """        print(f"  stage2: {len(seeds)} seeds -> {len(new_c)} new citers+neighbours, graded {n2}/{len(newl)} ({time.time()-t0:.0f}s)")
        print(f"  stage2 graded>=2: {sum(1 for c in newl if (c.get('grade') or 0) >= 2)}/{len(newl)} (neighbour-only: {sum(1 for c in newl if (c.get('grade') or 0) >= 2 and c.get('prf_only'))})")""")
ast.parse(src)
open("agent.py", "w").write(src)
print("patched OK")
# (applied separately in-session) patch 2: planner criteria exclude date/venue/author constraints;
# patch 1b: prf_only marker on stage-2 candidates.
# patch 3: GRADE_ROUND1 360->320 (cost)
# patch 4: title-search timeout 60->90 s (runs in parallel with get_citations), per-seed hit diagnostics
