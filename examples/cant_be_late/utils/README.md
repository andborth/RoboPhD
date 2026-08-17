# Third-party code: Can't Be Late spot simulator

**This subtree is not covered by the repository's MIT `LICENSE`.** Everything else in
RoboPhD is MIT (see the root `LICENSE`); the files listed under "Vendored" below arrive from
upstream projects and carry their licenses with them. Full texts are in `LICENSES/`.

## Provenance

Vendored on **2026-03-11** from
[`gepa-ai/gepa`](https://github.com/gepa-ai/gepa) (**MIT**, © 2025 Lakshya A Agrawal), path
`examples/adrs/can_be_late/utils/simulator/`.

gepa in turn carries the simulator from
[`UCB-ADRS/ADRS`](https://github.com/UCB-ADRS/ADRS) (**Apache-2.0**, © 2025 ADRS Team), whose
copy identifies itself as the *SkyPilot Spot Simulator* — the artifact behind
[Can't Be Late: Optimizing Spot Instance Savings under Deadlines](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)
(Wu et al., NSDI 2024).

**Both license texts are included** because the chain crosses a license boundary and gepa's
copy carries no Apache-2.0 notice. We received this code from gepa under MIT and comply with
MIT by reproducing its notice; the Apache-2.0 text is preserved alongside it so the original
licensor's terms travel with the code regardless of how the relicensing upstream is read.
Whether gepa could redistribute Apache-2.0 code under MIT is a question about gepa, not about
this repository, and we take no position on it.

## What is vendored, and what is ours

**Vendored, byte-identical to gepa** (9 files) — do not edit; a change here silently forks
from upstream:

```
simulator/simulator/sky_spot/__init__.py            strategies/__init__.py
simulator/simulator/sky_spot/env.py                 strategies/strategy.py
simulator/simulator/sky_spot/migration_model.py     task.py
simulator/simulator/sky_spot/multi_region_types.py  trace.py
simulator/simulator/sky_spot/utils.py
```

**Vendored with local changes** (2 files) — each carries a header describing what was changed
and why:

| File | Change |
|---|---|
| `simulator/simulator/main.py` | `wandb` stub injects into `sys.modules`; `--silent` also suppresses logging and `print()` |
| `simulator/simulator/sky_spot/simulate.py` | cost summary uses `print()` so it survives `--silent` |

**Written for RoboPhD** (5 files), covered by the repository's MIT license:

```
__init__.py   constants.py   simulator/__init__.py
simulator/dataset.py   simulator/simulation.py
```

## Trace data is not vendored

The AWS spot availability traces are **not** in this repository. `download_traces.sh` fetches
them from ADRS at run time; they are covered by ADRS's terms, not ours.

## Verifying against upstream

The nine identical files must stay identical. To check:

```bash
G=https://raw.githubusercontent.com/gepa-ai/gepa/main/examples/adrs/can_be_late/utils/simulator
curl -sfL $G/sky_spot/env.py | diff - simulator/simulator/sky_spot/env.py && echo "in sync"
```
