# DiscoveryBench (AstaBench)

Each example is a data-driven discovery task: a research **query** (a question about a phenomenon), one or more **CSV data files**, dataset descriptions, and a hidden **gold hypothesis + analysis workflow**. The agent loads the data, analyzes it, and emits a hypothesis JSON. The scorer judges the hypothesis along three dimensions and produces a single Hierarchical Matching Score (HMS).

Example query (illustrative — fictional):

> *"Do songbirds with longer beaks consume larger seeds in arid regions?"*

with one CSV (`bird_observations.csv`, ~2000 rows of foraging records: `beak_length_mm`, `seed_size_g`, `region`, `species`, `observation_date`, ...) and a hidden gold hypothesis along the lines of *"In arid biomes, beak length correlates positively with mean seed size consumed among ground-feeding species; the relationship weakens in temperate regions."*

## Where the inputs live in the solver state

| Field | Type | Contents |
| --- | --- | --- |
| `state.metadata["query"]` | str | The natural-language research question |
| `state.metadata["metadata"]["datasets"]` | list[dict] | Per-CSV: `{name, description, columns: {raw: [{name, description}, ...]}}` |
| `state.files` | dict[str, str] | `{sandbox_relative_path: host_absolute_path}` — **NOT auto-mounted into the sandbox; you must copy them in** |

`state.metadata` is intentionally limited to those two keys regardless of which split the example came from. The training mixture (real vs synth) is hidden from the runtime solver — branching on the source distribution would overfit to the training mixture rather than the underlying task. Evolution can still see split/domain/difficulty in post-hoc per-problem diagnostics (used for failure-pattern analysis), but the agent itself should produce hypotheses from query + datasets + the data alone.

Gold (hidden from the agent, surfaced to the scorer): `state.target == [gold_hypothesis_str, gold_workflow_str]`. Synth has `gold_workflow=""`.

## Required output schema

Write a JSON string to `state.output.completion`. Example (illustrative — fictional):

```json
{
  "hypothesis": "In arid-region observations of ground-feeding species (n≈800), mean consumed seed size increases with beak length (β≈0.04 g/mm, p<0.01); the relationship is absent in temperate samples after controlling for species and observation year.",
  "workflow": "1. Load bird_observations.csv. 2. Filter to ground-feeding species in arid regions. 3. OLS of seed_size_g ~ beak_length_mm + species + observation_year. 4. Repeat in temperate subset for comparison."
}
```

The hypothesis must name the variables, the relationship form, the context/scope, and supporting numeric evidence. The workflow describes the analysis steps that support the hypothesis.

The scorer accepts a raw-text fallback (no JSON → `gen_hypo=text`, `gen_workflow=""`) but emit JSON.

## The Docker sandbox

`python_session` runs Python inside a Docker container (`python:3.11-bookworm` base) with AstaBench's curated package set: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, transformers, torch, tensorflow, spacy, nltk, sympy, mlflow, hyperopt, gensim, and ~50 others. Each sample gets a fresh container; variables persist *within* a sample across multiple `python_session` calls (Jupyter-kernel-like). Default cell timeout: 5 minutes. Working directory: `/workspace/`.

## API surfaces

### `python_session` — the only `state.tools` entry

```python
from inspect_ai.tool import ToolDef
py = next(t for t in state.tools if ToolDef(t).name == "python_session")
result_str = await py(code="import pandas as pd\nprint(pd.__version__)")
```

Stateful within a sample. Returns stdout + the final-expression value as a single string.

### `sandbox()` — file ops

```python
from inspect_ai.util import sandbox
# Copy host file into the container:
await sandbox().write_file("/workspace/data.csv", open(host_path, "rb").read())
# Read a file out of the container:
content = await sandbox().read_file("/workspace/result.txt")
```

Always copy files from `state.files` into `/workspace/` before reading them in `python_session`. The host paths in `state.files` are NOT visible inside the sandbox.

## LLM calls

Use Inspect's tracked model API so usage flows into the `.eval` log:

```python
from inspect_ai.model import GenerateConfig, get_model
resp = await get_model().generate("Your prompt here", config=GenerateConfig(temperature=1.0))
text = resp.completion
```

`config` is optional; pass a `GenerateConfig` to set sampling parameters such as `temperature`. See `inspect_ai.model.GenerateConfig` for the full set.

Default model is **GPT-5 Mini** (`openai/gpt-5-mini`). Evolution may switch via the model string. **Do not** import `openai` / `anthropic` / `litellm` directly. If you must (e.g., to use a model not in Inspect's registry), wrap the call with `record_model_usage_with_inspect(model_name, ModelUsage(...))` afterward, or you silently underreport cost.

## Scoring (Hierarchical Matching Score)

```
HMS = context_score × var_f1 × rel_score
```

Each factor comes from a separate judge LLM call comparing the agent's hypothesis to the gold:

- **`context_score ∈ {0, 1}`** — does the hypothesis carry the same scope/boundary conditions as the gold? (E.g., "for adults" matters.) Multiplicative gate: 0 here zeros HMS regardless of var/rel.
- **`var_f1 ∈ [0, 1]`** — F1 over the set of dependent and independent variables, fuzzy-matched.
- **`rel_score ∈ {0, 0.5, 1.0}`** — 1.0 if the form of the relationship matches very well, 0.5 if similar but more general, 0.0 if different.

**Worked example** (illustrative — fictional): the gold says *"In arid regions, beak length correlates with seed size in ground-feeding species"*. An agent emits *"Beak length correlates with seed size in ground-feeding species"* — names the right variables and relationship, but drops the "in arid regions" scope. → `context_score=0` → `HMS=0` regardless of var/rel quality. Specifying scope is as important as naming variables.

## Per-example cost cap

The agent has a **$0.10 per-example budget** for its own LLM and tool spend. If the agent's `agent_cost_usd` exceeds the cap, the example score is multiplied by 0.9 (same penalty as docfinqa / protein_go / ARC-AGI). Cost is computed from `get_model().generate()` token usage and any wrapped out-of-band calls.

**Judge cost is excluded from the cap.** The DiscoveryBench scorer runs ~5 LLM judge calls per evaluation (≈$0.015–0.020/sample at gpt-4o-2024-08-06 prices). Those calls are evaluator-side overhead the agent has no way to influence, so they don't count against the $0.10 budget. The judge spend is reported separately as `other_cost` in result.json and gets its own column in `cost_report.md` / `interim_report.md` / `final_report.md`. Evolution and meta-evolution see only agent-side spend (`eval_cost`); judge overhead is captured for accounting but doesn't pollute the optimization signal.

Practical implication: at GPT-5 Mini rates, $0.10 covers many tool-aided reasoning rounds. Runaway loops (e.g. 30+ rounds of `python_session` + LLM critique with long context) will breach.

## Standard Tools constraint

Allowed in evolved code:
- `python_session` (the entry in `state.tools`)
- The `sandbox()` API for file ops
- `get_model()` for LLM calls
- The Python standard library

Disallowed:
- Importing additional Inspect tools or third-party search/analysis backends
- Reading files outside `state.files`
- Bypassing Inspect's model tracker without `record_model_usage_with_inspect`

## Diagnostics

`print()` output from the solver is captured into `agent_stdout`. Use brief, structured prints (`f"[step=load] rows={n}"`) to track which path the agent took on each example — useful when reading later iterations.

The per-dimension HMS pieces (`context_score`, `var_score`, `rel_score`) are surfaced in the diagnostics dict alongside the headline score, so when an evolved agent gains 0.05 HMS you can see whether the win came from better scope-matching or better variable-naming.

## A note on score distributions

HMS values land on coarse points (0, 0.5, 1.0 per dimension), so per-sample scores cluster — you'll see 0.0 / 0.3 / 0.5 / 0.6 / 1.0 more often than smooth distributions.
