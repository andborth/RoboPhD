"""Per-sample module isolation gate for the generated submission wrapper.

The unit tests parse WRAPPER_TEMPLATE and assert its SHAPE (agent_inner is
loaded via _isolated, model_registry stays a module-scope import, the warm-up
sits above the @solver). None of that proves the emitted wrapper actually
works, because the failure modes are all runtime and all invisible to AST:

  - inspect's chdir_python() puts the submission dir on sys.path only while
    agent.py itself is loading, and __exit__ restores a snapshot
    (inspect_ai/_util/path.py:60). Per-sample loads happen long after. If the
    wrapper stops pinning model_registry into sys.modules during that window,
    every sample dies with ModuleNotFoundError -- a full run scoring ~0 at
    full judge cost.
  - agent_inner's own @solver registers under the bare name "make_solver" and
    clobbers the wrapper's registry entry, which inspect's eval-set resume
    path resolves by.
  - The whole point: module-level state must be per-sample. A regression
    reintroducing a shared namespace is silent -- the eval completes, scores
    plausibly, and only a paired comparison against internal reveals it.

This gate builds a synthetic submission tree whose agent_inner keeps a
deadline clock and a semaphore at module scope (the exact v0_0_9 pattern),
loads the REAL WRAPPER_TEMPLATE through inspect's REAL solver_from_spec path,
and then drives concurrent per-sample loads the way a live eval would.

Synthetic rather than a staged submission on purpose: it must run without any
submission present, and it must fail if the WRAPPER contract regresses rather
than if some particular agent changed.

No API keys required: the fixture's model_registry is a sentinel object.

Usage:  python examples/asta_paper_finder/_check_isolation.py
        python examples/asta_paper_finder/_check_isolation.py --template ds1000
"""

import argparse
import asyncio
import gc
import resource
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

N_SAMPLES = 6            # matches --max-samples 6 in both submit scripts
N_CYCLES = 300           # load/drop cycles for the leak check
RSS_BUDGET_MB = 150      # peak growth allowed across N_CYCLES
LOAD_BUDGET_MS = 500     # per-load budget once warm

SOLVE_BUDGET = 1560.0

# Mirrors agent_inner.py:168-188 of v0_0_9_cap_0_063_fable: a deadline clock
# and a tool semaphore at MODULE scope. Under a shared namespace every
# starting sample re-stamps the clock and _remaining() never counts down.
_FIXTURE_INNER = '''
import asyncio, time
from inspect_ai.solver import solver
from model_registry import SHARED_HANDLE

SOLVE_BUDGET = {budget}
_TOOL_SEM = asyncio.Semaphore(10)
_START = [time.monotonic()]
_DEADLINE = [time.monotonic() + 10 ** 9]


def _stamp_clock():
    _START[0] = time.monotonic()
    _DEADLINE[0] = _START[0] + SOLVE_BUDGET


def _remaining() -> float:
    return _DEADLINE[0] - time.monotonic()


@solver
def make_solver():
    async def solve(state, generate):
        return state
    return solve
'''

_FIXTURE_SEED = '''
from inspect_ai.solver import solver
from model_registry import SHARED_HANDLE


@solver
def make_solver():
    async def solve(state, generate):
        return state
    return solve
'''

# A sentinel stands in for the real model handles: the contract is that every
# isolated agent_inner resolves to the SAME object (one connection pool).
_FIXTURE_REGISTRY = '''
class _Handle:
    pass


SHARED_HANDLE = _Handle()
'''


def _template(which: str) -> str:
    import importlib.util

    script = {
        "paper_finder": REPO / "scripts" / "asta_paper_finder_submit.py",
        "ds1000": REPO / "scripts" / "asta_ds1000_submit.py",
    }[which]
    spec = importlib.util.spec_from_file_location(f"_submit_{which}", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.WRAPPER_TEMPLATE


def _build_tree(root: Path, which: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent_inner.py").write_text(
        _FIXTURE_INNER.replace("{budget}", repr(SOLVE_BUDGET)))
    (root / "seed_agent.py").write_text(_FIXTURE_SEED)
    (root / "model_registry.py").write_text(_FIXTURE_REGISTRY)
    # The paper_finder wrapper imports tool_pacer; harmless for ds1000.
    shutil.copy(HERE / "tool_pacer.py", root / "tool_pacer.py")
    (root / "agent.py").write_text(_template(which))
    return root


def _rss_mb() -> float:
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", choices=("paper_finder", "ds1000"),
                    default="paper_finder")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from inspect_ai._eval.loader import solver_from_spec
    from inspect_ai._util.registry import registry_lookup
    from inspect_ai.solver._solver import SolverSpec

    # Running `python examples/asta_paper_finder/_check_isolation.py` puts THIS
    # directory on sys.path[0] — and the real model_registry.py lives here. The
    # wrapper's bare `import model_registry` would then resolve to the
    # example's module rather than the fixture's, making the shared-handle
    # assertion vacuous (it failed loudly the first time, which is the good
    # case; a fixture that happened to match the real API would have passed
    # while testing nothing). Drop the dir and evict any name the fixture tree
    # provides, so only the staged copies can satisfy these imports.
    for entry in (str(HERE), ""):
        while entry in sys.path:
            sys.path.remove(entry)
    for name in ("model_registry", "tool_pacer", "agent_inner", "seed_agent"):
        sys.modules.pop(name, None)

    tmp = Path(tempfile.mkdtemp(prefix="pf_isolation_"))
    try:
        tree = _build_tree(tmp / "submission", args.template)

        # Load through inspect's REAL path: chdir_python opens, puts the dir on
        # sys.path, then __exit__ restores it. Everything after this point runs
        # in the same sys.path state a live sample would.
        solver = solver_from_spec(SolverSpec(solver=str(tree / "agent.py")))
        if solver is None:
            print("FAIL: solver_from_spec returned None")
            return 1
        if str(tree) in sys.path:
            print("FAIL: submission dir leaked into sys.path after load "
                  "(the gate's premise no longer holds; rewrite it)")
            return 1
        isolated = solver.__globals__.get("_isolated")
        if isolated is None:
            print("FAIL: wrapper exposes no _isolated(); isolation removed?")
            return 1

        # 1. A per-sample load must work AFTER the chdir window closed. This is
        #    the ModuleNotFoundError trap.
        try:
            a = isolated("agent_inner")
            b = isolated("agent_inner")
            seed = isolated("seed_agent")
        except Exception as e:
            print(f"FAIL: per-sample load after chdir window: "
                  f"{type(e).__name__}: {e}")
            return 1

        # 2. Module-level state must be per-sample.
        if a._DEADLINE is b._DEADLINE or a._START is b._START:
            print("FAIL: deadline clock is SHARED across loads")
            return 1
        if a._TOOL_SEM is b._TOOL_SEM:
            print("FAIL: tool semaphore is SHARED across loads")
            return 1
        if not hasattr(seed, "make_solver"):
            print("FAIL: seed_agent did not load")
            return 1

        # 3. ...but the registry and pacer must NOT be. One connection pool.
        if a.SHARED_HANDLE is not b.SHARED_HANDLE:
            print("FAIL: model_registry was isolated too — every sample would "
                  "get its own connection pool")
            return 1

        # 4. The wrapper must still own the bare "make_solver" registry entry
        #    that inspect's resume path resolves by.
        if registry_lookup("solver", "make_solver") is None:
            print("FAIL: solver registry entry lost after isolated loads")
            return 1

        # 5. The behaviour the whole change exists for: concurrent samples each
        #    counting down their OWN elapsed time.
        async def sample(i):
            mod = isolated("agent_inner")
            mod._stamp_clock()
            await asyncio.sleep(0.15 * (i + 1))
            return SOLVE_BUDGET - mod._remaining()

        async def staggered():
            async def one(i):
                await asyncio.sleep(0.1 * i)
                return await sample(i)
            return await asyncio.gather(*(one(i) for i in range(N_SAMPLES)))

        observed = asyncio.run(staggered())
        expected = [0.15 * (i + 1) for i in range(N_SAMPLES)]
        drift = [abs(o - e) for o, e in zip(observed, expected)]
        if max(drift) > 0.10:
            print("FAIL: samples did not observe their own elapsed time")
            print(f"  expected {[round(e, 2) for e in expected]}")
            print(f"  observed {[round(o, 2) for o in observed]}")
            print("  (a shared clock shows near-zero countdown for early "
                  "starters — the v0_0_9 bug)")
            return 1

        # 6. Loads must be cheap and must not leak: 267 samples x N loads runs
        #    inside one 12h process that is also buffering a 200 MB eval log.
        gc.collect()
        base = _rss_mb()
        times = []
        for _ in range(N_CYCLES):
            t0 = time.perf_counter()
            mod = isolated("agent_inner")
            times.append(time.perf_counter() - t0)
            del mod
        gc.collect()
        growth = _rss_mb() - base
        median_ms = statistics.median(times) * 1000

        print(f"template={args.template} samples={N_SAMPLES} "
              f"cycles={N_CYCLES} median_load={median_ms:.1f}ms "
              f"rss_growth={growth:.1f}MB "
              f"max_clock_drift={max(drift) * 1000:.0f}ms")

        if median_ms > LOAD_BUDGET_MS:
            print(f"FAIL: median load {median_ms:.0f}ms over "
                  f"{LOAD_BUDGET_MS}ms budget — an expensive module-scope "
                  f"import now runs once per sample")
            return 1
        if growth > RSS_BUDGET_MB:
            print(f"FAIL: peak RSS grew {growth:.0f}MB over {N_CYCLES} "
                  f"cycles — isolated modules are being retained")
            return 1

        print("PASS: per-sample module isolation holds")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
