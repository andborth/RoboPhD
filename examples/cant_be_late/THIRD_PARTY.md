# Third-party notices — Can't Be Late example

Covers what `examples/cant_be_late/requirements.txt` and `download_traces.sh` bring in. Core
framework dependencies are in the repository's [`NOTICE.md`](../../NOTICE.md).

## Vendored code

**This example contains the only vendored third-party source in the repository.** The spot
simulator under `utils/` is not ours and is not covered by the repository's MIT license.

Provenance, the full chain, and both upstream license texts are in
[`utils/README.md`](utils/README.md) and `utils/LICENSES/`. In short: vendored 2026-03-11
from [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa) (MIT, © 2025 Lakshya A Agrawal), which
carries it from [`UCB-ADRS/ADRS`](https://github.com/UCB-ADRS/ADRS) (Apache-2.0, © 2025 ADRS
Team) — the SkyPilot spot simulator behind
[Can't Be Late](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)
(Wu et al., NSDI 2024). Nine files are byte-identical to gepa, two carry local changes
documented in their headers, and five are ours.

## Python packages

All exist for the vendored simulator; nothing in RoboPhD's own code imports them.

| Package | License | Project |
| --- | --- | --- |
| `ConfigArgParse` | MIT | https://github.com/bw2/ConfigArgParse |
| `colorama` | BSD-3-Clause | https://github.com/tartley/colorama |
| `PyYAML` | MIT | https://github.com/yaml/pyyaml |
| `numpy` | BSD-3-Clause | https://github.com/numpy/numpy |

The simulator also imports `tqdm`, which is a core dependency (see the root
[`NOTICE.md`](../../NOTICE.md)), and `wandb`, which is never installed — `simulator/main.py`
injects a stub into `sys.modules`, one of this copy's documented local changes.

## Services

**None.** This example makes no LLM calls — the strategy is pure Python evaluated against a
simulator. Only the evolution loop uses the Claude Code CLI, which is covered in the root
[`NOTICE.md`](../../NOTICE.md).

## Data

AWS spot availability traces (~151 MB), fetched by `download_traces.sh` from the ADRS
repository. **Not redistributed here** — the download is direct from
[`UCB-ADRS/ADRS`](https://github.com/UCB-ADRS/ADRS) (Apache-2.0) at run time, and the traces
are covered by that project's terms.
