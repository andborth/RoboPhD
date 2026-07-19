"""Smoke test for iter10_cite_expand: import with stubbed model_registry,
exercise _fetch_references bisect, _norm mention matching, and key helpers."""
import asyncio
import sys
import types

stub = types.ModuleType("model_registry")
stub.GPT_5_4 = object()
stub.GPT_5_4_MINI = object()
sys.modules["model_registry"] = stub

import agent  # noqa: E402

# ---- _fetch_references bisect: one poison id kills its batch; bisect recovers the rest
calls = []


async def fake_batch(ids, fields):
    calls.append(list(ids))
    if "CorpusId:666" in ids and len(ids) > 1:
        raise RuntimeError("ToolError: 'NoneType' object is not iterable")

    class Item:
        def __init__(self, cid):
            import json

            self.text = json.dumps(
                {"corpusId": cid, "references": [{"corpusId": 1000 + cid, "title": "r"}]}
            )

    return [Item(int(i.split(":")[1])) for i in ids if i != "CorpusId:666"]


async def run_refs():
    docs = [{"corpusId": c} for c in [1, 2, 3, 666, 5, 6, 7, 8]]
    # patch retry delays to keep the test fast
    agent._RETRY_DELAYS = (0, 0)
    out = await agent._fetch_references(fake_batch, docs)
    got = sorted(out)
    assert "1" in got and "8" in got, got
    assert "666" not in got
    assert len(got) >= 6, got  # recovered all but (at most) the poison id's tiny group
    print(f"bisect ok: recovered {len(got)}/7 non-poison docs, {len(calls)} batch calls")


asyncio.run(run_refs())

# ---- mention matching logic (mirrors _mentions_target)
sn = agent._norm("DistilBERT")
blob = f" {agent._norm('TinyStories: using DistilBERT distillation')} "
assert f" {sn} " in blob
blob2 = f" {agent._norm('BERT for sentiment')} "
assert f" {sn} " not in blob2
print("mention matching ok")

# ---- _key ordering sanity: year tiebreak must come after weighted score
import inspect

src = inspect.getsource(agent._solve_semantic)
i_w = src.index("-_wmax(i)")
i_y = src.index('(ordered[i].get("year") or 3000) if earliest')
assert i_w < i_y, "year tiebreak must sit after weighted score in _key2"
print("_key2 ordering ok")

# ---- expansion constants present and consistent
assert agent.EXPAND_SEEDS == 8 and agent.EXPAND_CAP == 140
assert agent.HEAD == 120 and agent.VERIFY_TOP == 16 and agent.SOFT_DEADLINE == 1300
assert agent.RESCUE_DEPTH <= agent.HEAD and agent.PER_CRIT_DEPTH <= agent.HEAD
print("constants ok")

# ---- solver factory builds
s = agent.make_solver()
assert callable(s)
print("solver builds ok")
print("ALL SMOKE TESTS PASSED")
