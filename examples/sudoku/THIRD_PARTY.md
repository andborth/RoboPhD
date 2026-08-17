# Third-party notices — Sudoku example

Covers what `examples/sudoku/requirements.txt` brings in. Core framework dependencies are in
the repository's [`NOTICE.md`](../../NOTICE.md).

## Python packages

| Package | License | Project |
| --- | --- | --- |
| `huggingface_hub` | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| `pandas` | BSD-3-Clause | https://github.com/pandas-dev/pandas |

## Services

**None.** This example makes no LLM calls — the agent is a pure-Python solver. Only the
evolution loop uses the Claude Code CLI, which is covered in the root
[`NOTICE.md`](../../NOTICE.md).

## Datasets

**`sapientinc/sudoku-extreme`**, fetched from HuggingFace at run time by `evaluator.py`.
**Not redistributed here.**

| | |
| --- | --- |
| Source | https://huggingface.co/datasets/sapientinc/sudoku-extreme |
| License | **None stated** |
| Upstream sources | [tdoku benchmarks](https://github.com/t-dillon/tdoku) (BSD-2-Clause), [enjoysudoku forum](http://forum.enjoysudoku.com) (no stated terms) |

The dataset card declares no license — its `cardData` carries only `task_categories`, and
there is no license tag or LICENSE file. Absent an explicit grant, ordinary copyright applies
and reuse rights are undetermined; this is a stronger caveat than "permissive by default".
The publisher's associated code project ([`sapientinc/HRM`](https://github.com/sapientinc/HRM),
the Hierarchical Reasoning Model) is Apache-2.0, but a code license does not extend to a
separately published dataset.

The card describes the puzzles as **"collected from the Sudoku community"**, naming two
upstream sources: the [tdoku benchmark suite](https://github.com/t-dillon/tdoku)
(BSD-2-Clause, though its benchmark README states no terms for the data specifically) and a
thread on the [enjoysudoku forum](http://forum.enjoysudoku.com) (user postings, no stated
terms). So sapientinc neither authored nor licensed this data — the publication is an
aggregation of other people's collections, and anyone needing to establish rights would be
tracing those two lineages rather than asking sapientinc.

Using it to evaluate solvers, as this example does, is unproblematic. Redistributing the
puzzles, or building something that embeds them, is where you would want the publisher to
clarify.
