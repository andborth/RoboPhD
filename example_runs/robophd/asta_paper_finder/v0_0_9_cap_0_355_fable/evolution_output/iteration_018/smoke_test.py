"""Smoke tests for iter18-cocite-largegold-v1.

Tests exec the SHIPPED block text (sliced out of agent.py by marker),
not a copy, per the lineage's exec-extracted-block pattern.
"""
import ast, asyncio, re, sys

SRC = open("agent.py").read()

# 1. whole file parses
ast.parse(SRC)
print("PASS ast.parse")

# 2. extract the co-cite channel block and exec it with stubs
start = SRC.index("# --- co-cite channel start ---")
end = SRC.index("# --- co-cite channel end ---")
block = SRC[start:end]

calls = {"ref_fetch_ids": None, "batch_ids": None, "batch_raw": None,
         "verify_papers": None}

def _cid(p):
    v = p.get("corpusId")
    return str(v) if v is not None else ""

async def _fetch_references(state, ids, deadline=None):
    calls["ref_fetch_ids"] = list(ids)
    # citer c1 refs: seed sha + two candidates; c2 refs: cand A again + null
    return [
        {"corpusId": "1", "references": [
            {"paperId": "SEED_SHA", "title": "seed"},
            {"paperId": "shaA", "title": "cand A"},
            {"paperId": "shaB", "title": "cand B"}]},
        {"corpusId": "2", "references": [
            {"paperId": "shaA", "title": "cand A"},
            {"paperId": None, "title": "null-id ref"},
            "not-a-dict-is-impossible-here-but-refs-are-dicts"][:2]},
    ]

async def _batch_fetch(state, ids, fields, chunk=60, deadline=None,
                       call_timeout=310, raw_ids=False):
    calls["batch_ids"] = list(ids)
    calls["batch_raw"] = raw_ids
    return [
        {"paperId": "shaA", "corpusId": 100, "year": 2023, "citationCount": 90},
        {"paperId": "shaB", "corpusId": 101, "year": 2023, "citationCount": 80},
        {"paperId": "shaC", "corpusId": 999, "year": 2023, "citationCount": 70},  # = seed cid
        {"paperId": "shaD", "corpusId": 102, "year": 2020, "citationCount": 60},  # fails year filter
    ]

def _apply_filters(papers, f):
    return [p for p in papers if (p.get("year") or 0) >= f["year_min"]]

async def _ref_verify(state, papers, seeds, deadline=None):
    calls["verify_papers"] = [_cid(p) for p in papers]
    return {"100": {0}}  # only cid 100 verified as citing the seed

g = {"_cid": _cid, "_fetch_references": _fetch_references,
     "_batch_fetch": _batch_fetch, "_apply_filters": _apply_filters,
     "_ref_verify": _ref_verify, "_VERIFY_FIELDS": "f",
     "COCITE_CITER_CAP": 500, "COCITE_SHA_CAP": 1000,
     "COCITE_VERIFY_CAP": 400, "TaskState": object, "asyncio": asyncio,
     "print": print}
exec(compile(block, "cocite_block", "exec"), g)
coc = g["_cocite_candidates"]

seeds = [{"corpusId": 999, "paperId": "SEED_SHA", "title": "DistilBERT"}]
window = [{"corpusId": 10, "year": 2025}, {"corpusId": 11, "year": 2024},
          {"corpusId": 12, "year": 2024}]
verified, unverified = asyncio.run(coc(
    None, window, seeds, {"seed_combine": "all"}, {"year_min": 2022}))

# oldest-first mining order
assert calls["ref_fetch_ids"] == ["11", "12", "10"], calls["ref_fetch_ids"]
# seed sha excluded; shaA counted twice ranks first; null paperId skipped
assert calls["batch_ids"][0] == "shaA" and "SEED_SHA" not in calls["batch_ids"]
assert set(calls["batch_ids"]) == {"shaA", "shaB"}
assert calls["batch_raw"] is True
# seed cid 999 dropped before filters; 102 dropped by year filter
assert set(calls["verify_papers"]) == {"100", "101"}
# cid 100 verified; 101 unverified with provenance flag
assert [_cid(p) for p in verified] == ["100"]
assert [_cid(p) for p in unverified] == ["101"] and unverified[0]["_cocite_src"]
print("PASS co-cite channel fixtures")

# degenerate: no citers / no seeds
assert asyncio.run(coc(None, [], seeds, {}, {})) == ([], [])
assert asyncio.run(coc(None, window, [], {}, {})) == ([], [])
print("PASS co-cite degenerate cases")

# ranked-empty: citers whose refs are all the seed itself
async def _fr_empty(state, ids, deadline=None):
    return [{"corpusId": "1", "references": [{"paperId": "SEED_SHA", "title": "s"}]}]
g2 = dict(g); g2["_fetch_references"] = _fr_empty
exec(compile(block, "cocite_block", "exec"), g2)
assert asyncio.run(g2["_cocite_candidates"](
    None, window, seeds, {}, {"year_min": 2022})) == ([], [])
print("PASS co-cite ranked-empty")

# 3. _batch_fetch raw_ids flag: extract the shipped function and check the
# ids it sends with and without the flag
m = re.search(r"async def _batch_fetch\(.*?\n(?=async def|def |# ---)", SRC, re.S)
bf_block = m.group(0)
sent = []
class _Tool: pass
async def _call(tool, quiet=False, timeout=None, **kw):
    sent.append(kw["ids"])
    return [{"corpusId": int(i.split(":")[-1]) if ":" in i else 7} for i in kw["ids"]]
g3 = {"_get_tool": lambda s, n: _Tool(), "_call": _call, "_cid": _cid,
      "asyncio": asyncio, "time": __import__("time"), "TaskState": object}
exec(compile(bf_block, "bf_block", "exec"), g3)
asyncio.run(g3["_batch_fetch"](None, ["123"], "f"))
assert sent[-1] == ["CorpusId:123"], sent
asyncio.run(g3["_batch_fetch"](None, ["deadbeef"], "f", raw_ids=True))
assert sent[-1] == ["deadbeef"], sent
print("PASS _batch_fetch raw_ids")

# 4. integration block: gate + crash handling. Extract from the channel
# comment to the no-seed fallback comment.
istart = SRC.index("    # Reverse citation channel:")
iend = SRC.index("    # No-seed, no-author queries")
iblock = "async def _integ(scenario):\n" + \
    "    state=plan_holder['plan'];plan=plan_holder['plan'];filters=plan_holder['filters']\n" + \
    "    candidates=plan_holder['candidates'];filtered=list(plan_holder['filtered'])\n" + \
    "    seeds_resolved=plan_holder['seeds'];authors=[];progress={}\n" + \
    "    deadline=time.monotonic()+600\n" + \
    "    from_citations=True\n" + \
    SRC[istart:iend] + \
    "    return filtered, cocite_unverified\n"

async def _rev_stub(state, plan, seeds_resolved, filters, progress=None, deadline=None):
    return [{"corpusId": 300, "citationCount": 5}]

scen = {"mode": "ok"}
async def _coc_stub(state, cands, seeds, plan, filters, deadline=None):
    if scen["mode"] == "crash":
        raise RuntimeError("boom")
    return ([{"corpusId": 400, "citationCount": 9}],
            [{"corpusId": 401, "_cocite_src": True}])

class _WF:
    pass
plan_holder = {}
g4 = {"_reverse_candidates": _rev_stub, "_cocite_candidates": _coc_stub,
      "_cid": _cid, "asyncio": asyncio, "time": __import__("time"),
      "plan_holder": plan_holder, "print": print}
exec(compile(iblock, "integ_block", "exec"), g4)

def run_integ(exp="many", venues=None, mode="ok", ncand=960):
    scen["mode"] = mode
    plan_holder.update({
        "plan": {"expected_result_count": exp, "seed_combine": "all"},
        "filters": {"venues": venues or []},
        "candidates": [{"corpusId": i} for i in range(ncand)],
        "filtered": [{"corpusId": 1, "citationCount": 99}],
        "seeds": [{"corpusId": 999, "paperId": "S", "citationCount": 10229}],
    })
    return asyncio.run(g4["_integ"](None))

filtered, cu = run_integ()
cids = [_cid(p) for p in filtered]
assert "300" in cids and "400" in cids and [_cid(p) for p in cu] == ["401"]
# venue-bearing query: co-cite must NOT fire (gate), reverse still adds
filtered, cu = run_integ(venues=[["NeurIPS"]])
cids = [_cid(p) for p in filtered]
assert "300" in cids and "400" not in cids and cu == []
# "one" query with small window: neither channel fires unless filtered<30 —
# here filtered has 1 entry so reverse fires, co-cite gated off by exp
filtered, cu = run_integ(exp="one")
assert "400" not in [_cid(p) for p in filtered] and cu == []
# co-cite crash: reverse results survive, no co-cite additions
filtered, cu = run_integ(mode="crash")
cids = [_cid(p) for p in filtered]
assert "300" in cids and "400" not in cids and cu == []
print("PASS integration gate + crash fixtures")

print("ALL SMOKE TESTS PASS")
