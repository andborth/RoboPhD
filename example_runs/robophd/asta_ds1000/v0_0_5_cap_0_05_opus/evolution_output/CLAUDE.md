# Domain Background

# DS-1000 (AstaBench)

Each example is a Python data-science problem. The agent receives a natural-language question with an embedded code skeleton and must emit Python code that, when appended to the program, makes a variable called `result` hold the correct value.

Example prompt (illustrative — fictional, not a real DS-1000 problem):

> Problem: I have a numpy array `a = np.array([0, 3, 0, 5, 7, 0])`. I want to drop the zeros. What's the cleanest way?
>
> ```
> A:
> <code>
> import numpy as np
> a = np.array([0, 3, 0, 5, 7, 0])
> </code>
> BEGIN SOLUTION
> <code>
> [insert]
> </code>
> END SOLUTION
> <code>
> print(result)
> </code>
> ```

A correct completion writes `<code>result = a[a != 0]</code>` (any equivalent NumPy expression works).

## Where the inputs live in the solver state

| Field | Type | Contents |
| --- | --- | --- |
| `state.input` | str | The full problem prompt, including the code skeleton and the format instruction "Put your answer inside `<code>` and `</code>` tags." |
| `state.metadata["library"]` | str | One of `"Numpy"`, `"Pandas"`, `"Matplotlib"`, `"Scipy"`, `"Sklearn"`, `"Pytorch"`, `"Tensorflow"`. The same library is implied in the prompt. |

## Required output

Write a single `<code>...</code>` block to `state.output.completion`. The opening `<code>` and closing `</code>` tags **are required** — the scorer uses them to extract the answer. Everything between them is appended to a hidden test program that exercises `result` against test inputs.

```
<code>
result = a[a != 0]
</code>
```

Inside the tags: executable Python only. No prose, no markdown fences (` ```python `), no `BEGIN SOLUTION` / `END SOLUTION` markers. Python `#` comments are optional.

Outside the tags (i.e., in `state.output.completion` before `<code>` or after `</code>`): nothing. Don't preface the answer with a chain-of-thought summary — the scorer doesn't see it and it just adds tokens.

## The Docker sandbox

`python_session` runs Python inside a Docker container with a curated data-science package set: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, gensim, torch, tensorflow-cpu, xgboost (versions pinned to AstaBench's compose). Each sample gets a fresh container; variables persist within a sample across multiple `python_session` calls (Jupyter-kernel-like). Default cell timeout: 5 minutes. Working directory: `/workspace/`.

## API surfaces

The solver code (agent.py) should only use these Inspect entries — don't import additional Inspect tools or third-party search/analysis backends.

### `python_session` — the only `state.tools` entry

```python
from inspect_ai.tool import ToolDef
py = next(t for t in state.tools if ToolDef(t).name == "python_session")
result_str = await py(code="import numpy as np\nprint(np.__version__)")
```

Stateful within a sample. Returns stdout + the final-expression value as a single string. Useful for self-checking a candidate solution against the prompt's example values before committing to a final answer.

### `sandbox()` — file ops

```python
from inspect_ai.util import sandbox
content = await sandbox().read_file("/workspace/scratch.txt")
```

Most DS-1000 problems don't need files, but the API is available.

## LLM calls

The following model handles are available, imported from `model_registry`:

| Handle | Input ($/M tok) | Output ($/M tok) | Default `reasoning_effort` | Available overrides |
| --- | --- | --- | --- | --- |
| `GPT_5_4_MINI` | 0.75 | 4.50 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `GPT_5_4` | 2.50 | 15.00 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `GPT_5_5` | 5.00 | 30.00 | model-managed | `"low"`, `"medium"`, `"high"` |
| `CLAUDE_HAIKU_4_5` | 1.00 | 5.00 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `CLAUDE_SONNET_4_6` | 3.00 | 15.00 | `"none"` | `"low"`, `"medium"`, `"high"` |
| `CLAUDE_OPUS_4_8` | 5.00 | 25.00 | model-managed | `"low"`, `"medium"`, `"high"` |
| `GEMINI_3_1_FLASH_LITE` | 0.25 | 1.50 | `"low"` | `"low"`, `"high"` |
| `GEMINI_3_5_FLASH` | 1.50 | 9.00 | `"low"` | `"low"`, `"high"` |
| `GEMINI_3_1_PRO_PREVIEW` | 2.00 | 12.00 | `"low"` | `"low"`, `"high"` |

Setting `reasoning_effort` to any value in the "available overrides" column adds reasoning tokens above what the default already costs. For handles whose default is `"none"`, picking `"low"` is the cheapest opt-in step but it's still strictly more expensive than omitting `reasoning_effort` entirely. For the Gemini handles whose default is already `"low"`, the only opt-up is `"high"`. To stay at the cheapest path on any handle, omit the `reasoning_effort` field from `GenerateConfig`.

`max_tokens` is a universal output-budget cap accepted on every handle (an integer; no provider rejects or strips it). Pass it via `GenerateConfig(max_tokens=N)`. On Anthropic and Gemini handles, the cap applies to the visible completion only — reasoning tokens (when `reasoning_effort` is set) come on top of it. On OpenAI handles, the cap is shared between reasoning and visible tokens, so set it generously when combined with `reasoning_effort` or you may get an empty completion.

```python
from inspect_ai.model import GenerateConfig
from model_registry import GPT_5_4_MINI, CLAUDE_SONNET_4_6

# Default call (cheapest, no extra reasoning):
resp = await GPT_5_4_MINI.generate("Your prompt here")

# Opt into reasoning for a hard problem and cap the output:
resp = await CLAUDE_SONNET_4_6.generate(
    "Your prompt here",
    config=GenerateConfig(reasoning_effort="low", max_tokens=2048),
)
text = resp.completion
```

`config` is optional. The two knobs to use are `reasoning_effort` (trades cost for quality on hard problems; see the per-handle table above for default and available values) and `max_tokens` (caps the output budget). All LLM calls must go through one of the handles above.

## Scoring

The agent's `<code>` block is extracted, concatenated with hidden setup and test code, and run inside the sandbox. The score for the sample is **1.0** if the appended code makes `result` match the reference under all hidden test inputs, else **0.0**. No partial credit.

A subset of problems additionally enforce **style/idiom constraints on the submitted code itself**. Two flavors appear: (1) forbidding Python control-flow constructs like `for`/`while` to push toward library calls, and (2) requiring a specific library function name to appear in the solution, ruling out manual reimplementations. The constraint is sometimes flagged in the prompt ("without using X", "the efficient way", "not one by one") but is more often implicit in the spirit of the question: asking *"how do I do X with NumPy"* invites a NumPy-idiomatic answer, and a workaround that bypasses the library can fail even when the output is correct. When this happens, the per-problem `test_result.md` shows an assertion raised from a `test_string` function (versus correctness failures, which raise from `test_execution`). Both outcomes score 0.0; the traceback tells the agent whether to fix the *answer* or the *form*.

## Iteration-aggregate score

Per-example scoring is binary correctness (1.0 or 0.0). At the end of each iteration, your batch is combined into a single score: your accuracy (on a 0–100 scale) minus a cost penalty when your mean batch spend exceeds the threshold. The penalty is expressed in wrong-answer units — each $0.01 of mean spend over $0.05 subtracts one error-equivalent from your score. Only `get_model()` calls are metered — `python_session` and `sandbox()` don't count.

| Mean cost | Effect on score |
|---|---|
| ≤ $0.05 | No effect on score — two free-zone agents with the same raw accuracy score identically, regardless of their actual spend |
| $0.05–$0.06 | Tiebreaker — lose tied accuracy to a cheaper agent; **need 1+ more correct** to win |
| $0.06–$0.07 | **Need 2+ more correct** than a free-zone agent to win |
| $0.07–$0.08 | **Need 3+ more correct** than a free-zone agent to win (in practice, a decisive penalty) |
| … | Each additional $0.01 of mean spend adds 1 to the breakeven correct-count |

## Time budget

Your agent times out and the problem scores 0 if a single problem takes more than **29 minutes** of wall-clock. This is a generous budget and is unlikely to be the binding constraint. Per-problem wall-clock is recorded as `eval_wall_clock_seconds` in each problem's `result.json`.

## Diagnostics

`print()` output from the solver is captured into `agent_stdout`. The extracted `<code>` block and the test program's stdout/stderr are surfaced as per-example diagnostic files (`extracted_code.md`, `test_result.md`) so failures can be inspected without re-running.

# Domain Objective

Evolve a DS-1000 agent that, given a Python data-science problem prompt, produces a `<code>...</code>` block whose contents make the hidden test program's `result` variable match the reference under all hidden test inputs.

Your primary goal is simple: maximize the score on held-out problems. The scoring function (described in Domain Background in CLAUDE.md) encodes this directly — correctness is the dominant signal, and cost acts as a tiebreaker close to threshold but starts to actively trade off against correctness farther out. The free zone is the batch *average*, not per-problem: you can spend more on some problems and less on others. Above $0.05 per problem on average, every $0.01 of extra spend costs you one error-equivalent of score; see the cost-penalty table in Domain Background for the breakeven math.

Each iteration draws a different sample of problems, and the final agent is evaluated on a held-out test set it has never seen. After you construct your agent, it will be tested on entirely new batches of examples in future iterations. So, to rephrase your goal, your objective is to build an agent that generalizes to unseen problems — the visible batch is a training signal, not the target.

# Evolution Environment

## Available Data

Use your available tools to explore the experiment directory. Key artifacts:

- `../../agents/<name>/` — agent source code (one directory per agent)
- `../../iteration_NNN/` — evaluation results per iteration:
  - `error_analysis_report.md` — cross-agent score comparison and failure summary
  - `error_index.json` — machine-readable score data (error analysis report was derived from this)
  - `cost_report.md` — per-agent cost breakdown. Useful if you are instructed to pay attention to cost
  - `agent_<name>/problems/<id>/` — per-problem results and diagnostics

During testing and refinement rounds (Rounds 2+), your results appear in `./iteration_NNN_test/` (in your working directory, not the experiment root).

## Strategy Tools

If a `strategy_tools/` directory exists in your working directory, it contains Python helper scripts provided by your evolution strategy. **Run them** — they analyze prior iteration data and produce structured output to guide your work. Use `/opt/anaconda3/envs/robophd_demo/bin/python strategy_tools/<script>.py --help` to discover usage.

## Scratch space

Your working directory is your iteration's evolution dir. If you need to drop a small test harness or scratch script while debugging, write it there directly — `/tmp` is outside the write scope, and there's no need to clean up after yourself afterwards. Leftover debugging artifacts in the iteration dir are useful for retrospective analysis.



## CLI Tools

Available: `jq`, `tree`