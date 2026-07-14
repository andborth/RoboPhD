"""Multi-process stress gate for the safe judge-cache writer.

The unit tests exercise the flock path with THREADS (fast, but the
protection being claimed is cross-PROCESS); this gate is the process-
level evidence: 8 subprocesses hammer the patched writer on a shared
tmp cache and every update must survive into a valid file. A regression
that is process-visible but thread-invisible (e.g. a per-process cache
layer added above the lock) fails here and nowhere else.

For calibration, the stock-writer leg is run first and reported (not
asserted): at the original incident it produced either a torn file or
massive lost updates (13/400 survived on first measurement). It is
informational because its failure is probabilistic.

No API keys required: the safe writer installs at evaluator import and
the workers only exercise file I/O.

Usage:  python examples/asta_paper_finder/_check_cache_stress.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

N_PROCS = 8
N_UPDATES = 50

_WORKER = r'''
import asyncio, sys
mode, path, wid, here, repo = sys.argv[1:6]
sys.path.insert(0, here)
sys.path.insert(0, repo)
from astabench.evals.paper_finder import paper_finder_utils as pfu
if mode == "patched":
    import evaluator  # installs the safe writer on both bindings
pfu.detailed_reference_path = path
update = pfu.update_references
async def main():
    for i in range(int(sys.argv[6])):
        await update(f"q_{wid}_{i}", {"paper": "perfectly_relevant_papers"})
asyncio.run(main())
'''


def _run(mode: str) -> tuple[bool, int, list[str]]:
    d = tempfile.mkdtemp(prefix=f"cache_stress_{mode}_")
    path = os.path.join(d, "detailed_reference.json")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, mode, path, str(w),
             str(HERE), str(HERE.parent.parent), str(N_UPDATES)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        for w in range(N_PROCS)
    ]
    crashes = []
    for p in procs:
        _, err = p.communicate()
        if p.returncode != 0:
            crashes.append(err.decode()[-200:])
    try:
        data = json.load(open(path))
        return True, len(data), crashes
    except (json.JSONDecodeError, FileNotFoundError):
        return False, 0, crashes


def main() -> int:
    expected = N_PROCS * N_UPDATES

    valid, n, crashes = _run("stock")
    print(f"stock:   valid={valid} entries={n}/{expected} crashes={len(crashes)} "
          f"(informational — failure is probabilistic)")

    valid, n, crashes = _run("patched")
    print(f"patched: valid={valid} entries={n}/{expected} crashes={len(crashes)}")
    if not valid or n != expected or crashes:
        print("\nFAIL: the safe writer lost updates, corrupted the file, or "
              "crashed workers under multi-process contention.")
        for c in crashes[:2]:
            print("  worker stderr:", c)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
