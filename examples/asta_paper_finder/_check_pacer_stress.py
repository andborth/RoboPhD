"""Multi-process stress gate for the global tool-call launch pacer.

The unit tests exercise the file-backed slot reservation from ONE process
(many asyncio tasks); the protection being claimed is cross-PROCESS: N
subprocesses sharing one slot-state file must collectively respect the
configured per-endpoint launch rate. This gate is that evidence: each
worker records the wall-clock time at which each of its paced launches
was released, and the merged timeline must never exceed the rate (with a
small tolerance for clock skew between processes on one host).

A regression that is process-visible but task-invisible (e.g. state kept
in a per-process cache above the flock) fails here and nowhere else.

No API keys required: workers exercise only tool_pacer's file I/O.

Usage:  python examples/asta_paper_finder/_check_pacer_stress.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

N_PROCS = 8
N_LAUNCHES = 12          # per process
RATE = 40.0              # launches/second per endpoint (fast gate: ~2.4s)
TOLERANCE = 1.25         # allowed short-window overshoot factor (clock skew)

_WORKER = r'''
import asyncio, json, os, sys, time
path, out_path, here, rate, n = sys.argv[1:6]
sys.path.insert(0, here)
os.environ["PF_TOOL_PACER_PATH"] = path
import tool_pacer

async def main():
    stamps = []
    async def one():
        await tool_pacer.pace("snippet_search", float(rate))
        stamps.append(time.time())
    await asyncio.gather(*(one() for _ in range(int(n))))
    with open(out_path, "w") as f:
        json.dump(stamps, f)

asyncio.run(main())
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        state = os.path.join(td, "launch_slots.json")
        outs = []
        procs = []
        for i in range(N_PROCS):
            out = os.path.join(td, f"stamps_{i}.json")
            outs.append(out)
            procs.append(subprocess.Popen(
                [sys.executable, "-c", _WORKER, state, out, str(HERE),
                 str(RATE), str(N_LAUNCHES)],
            ))
        for p in procs:
            rc = p.wait(timeout=180)
            if rc != 0:
                print(f"FAIL: worker exited {rc}")
                return 1

        stamps = sorted(
            t for out in outs for t in json.load(open(out))
        )
        total = len(stamps)
        expected = N_PROCS * N_LAUNCHES
        if total != expected:
            print(f"FAIL: {total} launches recorded, expected {expected}")
            return 1

        span = stamps[-1] - stamps[0]
        overall = (total - 1) / span if span > 0 else float("inf")
        # Sliding-window check: any RATE consecutive launches must span
        # >= ~1s (i.e. no 1-second window exceeds RATE * TOLERANCE).
        win = int(RATE)
        worst = float("inf")
        for i in range(total - win):
            worst = min(worst, stamps[i + win] - stamps[i])
        window_rate = win / worst if worst not in (0, float("inf")) else 0.0

        print(f"launches={total} span={span:.2f}s "
              f"overall={overall:.1f}/s worst-window={window_rate:.1f}/s "
              f"(configured {RATE}/s, tolerance x{TOLERANCE})")

        if overall > RATE * TOLERANCE:
            print("FAIL: overall launch rate exceeds configured rate")
            return 1
        if window_rate > RATE * TOLERANCE:
            print("FAIL: burst window exceeds configured rate")
            return 1
        print("PASS: cross-process launch pacing holds")
        return 0


if __name__ == "__main__":
    sys.exit(main())
