# DiscoveryBench (AstaBench)

Each example is a data-driven discovery task: a research **query** (a question about a phenomenon), one or more **CSV data files**, dataset descriptions, and a hidden **gold hypothesis + analysis workflow**. The agent loads the data, analyzes it, and emits a hypothesis JSON. The scorer judges the hypothesis along three dimensions and produces a single Hierarchical Matching Score (HMS).

Example query (from real/train):

> *"Does increased time preference lead to higher BMI?"*

with one CSV (`nls_raw.csv`, ~6000 rows × 61 columns of NLSY79 survey data) and a gold hypothesis along the lines of *"Higher time preference associated with higher BMI for 1989 data."*

## Where the inputs live in the solver state

| Field | Type | Contents |
| --- | --- | --- |
| `state.metadata["query"]` | str | The natural-language research question |
| `state.metadata["metadata"]["datasets"]` | list[dict] | Per-CSV: `{name, description, columns: {raw: [{name, description}, ...]}}` |
| `state.files` | dict[str, str] | `{sandbox_relative_path: host_absolute_path}` — **NOT auto-mounted into the sandbox; you must copy them in** |

`state.metadata` is intentionally limited to those two keys regardless of which split the example came from. The training mixture (real vs synth) is hidden from the runtime solver — branching on the source distribution would overfit to the training mixture rather than the underlying task. Evolution can still see split/domain/difficulty in post-hoc per-problem diagnostics (used for failure-pattern analysis), but the agent itself should produce hypotheses from query + datasets + the data alone.

Gold (hidden from the agent, surfaced to the scorer): `state.target == [gold_hypothesis_str, gold_workflow_str]`. Synth has `gold_workflow=""`.

## Required output schema

Write a JSON string to `state.output.completion`:

```json
{
  "hypothesis": "Time preference is positively associated with BMI in adults from the NLSY79 cohort. The relationship is moderate (β≈0.12, p<0.01) and persists after controlling for age, sex, and education.",
  "workflow": "1. Load nls_raw.csv. 2. Filter to age ≥ 18 (n≈4500). 3. OLS regression of BMI ~ time_preference + controls. 4. Inspect coefficient + 95% CI."
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
from inspect_ai.model import get_model
resp = await get_model().generate("Your prompt here")
text = resp.completion
```

Default model is **GPT-5 Mini** (`openai/gpt-5-mini`). Evolution may switch via the model string. **Do not** import `openai` / `anthropic` / `litellm` directly. If you must (e.g., to use a model not in Inspect's registry), wrap the call with `record_model_usage_with_inspect(model_name, ModelUsage(...))` afterward, or you silently underreport cost.

## Scoring (Hierarchical Matching Score)

```
HMS = context_score × var_f1 × rel_score
```

Each factor comes from a separate judge LLM call comparing the agent's hypothesis to the gold:

- **`context_score ∈ {0, 1}`** — does the hypothesis carry the same scope/boundary conditions as the gold? (E.g., "for adults" matters.) Multiplicative gate: 0 here zeros HMS regardless of var/rel.
- **`var_f1 ∈ [0, 1]`** — F1 over the set of dependent and independent variables, fuzzy-matched.
- **`rel_score ∈ {0, 0.5, 1.0}`** — 1.0 if the form of the relationship matches very well, 0.5 if similar but more general, 0.0 if different.

**Worked example**: the gold says *"Higher time preference is associated with higher BMI in adults"*. An agent emits *"Higher time preference is associated with higher BMI"* — names the right variables and relationship, but misses the "in adults" scope. → `context_score=0` → `HMS=0` regardless of var/rel quality. Specifying scope is as important as naming variables.

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

## Headline benchmark target

| Tier | Reference agent | HMS | Notes |
| --- | --- | --- | --- |
| Various | Reflexion (Oracle) + GPT-4o | **0.245** | Best in DiscoveryBench paper |
| Various | ReAct + GPT-4o | 0.154 | Standard ReAct ceiling per paper |
| Various | CodeGen + GPT-4o | 0.155 | Code-first agent in paper |
| Various | Llama-3 baselines | 0.11–0.13 | Open-weight reference |

The leaderboard shows different points; treat 0.20–0.22 HMS as a reasonable target for an evolved Standard-tools agent on real/test (paper ceiling 0.245, and we shouldn't expect to match Reflexion's oracle setup). HMS values land on coarse points (0, 0.5, 1.0 per dimension), so per-sample scores cluster — you'll see 0.0 / 0.3 / 0.5 / 0.6 / 1.0 more often than smooth distributions.
