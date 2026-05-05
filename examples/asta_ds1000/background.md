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
| `state.metadata["library"]` | str | One of `"Numpy"`, `"Pandas"`, `"Matplotlib"`, `"Scipy"`, `"Scikit-learn"`, `"Pytorch"`, `"Tensorflow"`. The same library is implied in the prompt. |

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

These are the only Inspect entries the solver code (agent.py) should touch — don't import additional Inspect tools or third-party search/analysis backends.

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

Use Inspect's tracked model API so usage flows into the `.eval` log:

```python
from inspect_ai.model import GenerateConfig, get_model
resp = await get_model().generate("Your prompt here", config=GenerateConfig(temperature=0.0))
text = resp.completion
```

`config` is optional; pass a `GenerateConfig` to set sampling parameters such as `temperature`. The configured model (currently **GPT-5.4 Mini**) is the one returned by a bare `get_model()` call — no arguments. Don't pass a model string to `get_model()` and don't import `openai` / `anthropic` / `litellm` directly. All LLM calls must go through `get_model()` so usage flows into the Inspect tracker and cost is reported correctly.

## Scoring

The agent's `<code>` block is extracted, concatenated with hidden setup and test code, and run inside the sandbox. The score for the sample is **1.0** if the appended code makes `result` match the reference under all hidden test inputs, else **0.0**. No partial credit.

A subset of problems additionally enforce **style/idiom constraints on the submitted code itself**. Two flavors appear: (1) forbidding Python control-flow constructs like `for`/`while` to push toward library calls, and (2) requiring a specific library function name to appear in the solution, ruling out manual reimplementations. The constraint is sometimes flagged in the prompt ("without using X", "the efficient way", "not one by one") but is more often implicit in the spirit of the question: asking *"how do I do X with NumPy"* invites a NumPy-idiomatic answer, and a workaround that bypasses the library can fail even when the output is correct. When this happens, the per-problem `test_result.md` shows an assertion raised from a `test_string` function (versus correctness failures, which raise from `test_execution`). Both outcomes score 0.0; the traceback tells the agent whether to fix the *answer* or the *form*.

## Per-example cost cap

The agent's LLM spend is capped at **$0.06 per example** (only `get_model()` calls are metered — `python_session` and `sandbox()` don't count). Exceeding the cap multiplies the example score by 0.9.

## Diagnostics

`print()` output from the solver is captured into `agent_stdout`. The extracted `<code>` block and the test program's stdout/stderr are surfaced as per-example diagnostic files (`extracted_code.md`, `test_result.md`) so failures can be inspected without re-running.
