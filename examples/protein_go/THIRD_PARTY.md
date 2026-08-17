# Third-party notices — Protein GO example

Covers what `examples/protein_go/requirements.txt` and `setup.sh` bring in. Core framework
dependencies are in the repository's [`NOTICE.md`](../../NOTICE.md).

**This example is the only one in the repository that involves GPL-3.0 components.** If you
are redistributing work built on it, read the copyleft section below rather than assuming the
repository's MIT license covers everything you have installed.

## Python packages

| Package | License | Project |
| --- | --- | --- |
| `biopython` | Biopython License Agreement (MIT-style, custom) | https://biopython.org |
| `numpy` | BSD-3-Clause | https://github.com/numpy/numpy |
| `cafaeval` | **GPL-3.0** | https://github.com/BioComputingUP/CAFA-evaluator |
| `tfrecord` | MIT | https://github.com/vahidk/tfrecord |
| `torch` | BSD-3-Clause | https://github.com/pytorch/pytorch |
| `fair-esm` | MIT | https://github.com/facebookresearch/esm |

`biopython`'s license is its own agreement, not MIT — similar in effect, but read it rather
than assuming.

## Copyleft notice

Two GPL-3.0 components are involved, in **different ways**, and the difference is what
determines your obligations.

**`cafaeval` — GPL-3.0, imported as a library.** `evaluator.compute_cafa_fmax_batch` calls it
directly to compute the canonical CAFA Fmax metric. This repository does not redistribute
cafaeval; `pip install cafaeval` obtains it from PyPI under its own terms, and the code we
ship merely imports it. Anyone **redistributing a combined work** that links cafaeval should
review GPL-3.0 §5 and reach their own conclusion. We use the canonical implementation
deliberately: a reimplementation would risk silently diverging from the metric the CAFA
literature reports.

**`diamond` — GPL-3.0, invoked as a separate program.** DIAMOND is an external binary run as
a subprocess for BLAST searches. Running a GPL program as a separate process does not create
a derived work, so this imposes no obligation on RoboPhD's own code. You do need to install
it yourself, and it remains GPL-3.0.

Nothing here is legal advice.

## External tools

| Tool | License | Notes |
| --- | --- | --- |
| DIAMOND | GPL-3.0 | https://github.com/bbuchfink/diamond — separate binary, subprocess only |

## Services

- **OpenRouter → Gemini 3.1 Flash Lite** — the solver model, via the agent's `llm()` and
  `embed()` callables. Your own API key; governed by
  [OpenRouter's terms](https://openrouter.ai/terms) and Google's model terms.

## Datasets

`setup.sh` downloads roughly 2.5 GB. **This repository redistributes none of it** — each item
comes from its own publisher under that publisher's terms.

| Data | Source | License |
| --- | --- | --- |
| SwissProt 2022_01 (annotations + BLAST reference) | UniProt Consortium | CC BY 4.0 |
| ProteInfer clustered-split TFRecords | [`google-research/proteinfer`](https://github.com/google-research/proteinfer) (Sanderson et al., 2023) | Apache-2.0 |
| Price-149 table | CLEAN paper repository | see that repository |
| Gene Ontology | [Gene Ontology Consortium](http://geneontology.org) | CC BY 4.0 |

The derived files `setup.sh` builds locally — `swissprot_train.dmnd`, the three JSONL splits,
`esm_train_embeddings.npy` — are outputs over the above data and remain subject to those
sources' terms. They are not committed to this repository.
