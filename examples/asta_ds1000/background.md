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

Six model handles are available, imported from `model_registry`. They pair by family into a cheap/fast tier and a stronger/slower tier:

| Handle | Input ($/M tok) | Output ($/M tok) |
| --- | --- | --- |
| `GPT_5_4_MINI` | 0.75 | 4.50 |
| `GPT_5_4` | 2.50 | 15.00 |
| `CLAUDE_HAIKU_4_5` | 1.00 | 5.00 |
| `CLAUDE_SONNET_4_6` | 3.00 | 15.00 |
| `GEMINI_3_1_FLASH_LITE_PREVIEW` | 0.25 | 1.50 |
| `GEMINI_3_FLASH_PREVIEW` | 0.50 | 3.00 |

Use these to estimate per-problem spend against the $0.04 free-zone threshold without having to look up rates externally. As a rough order-of-magnitude check: a single 2K-input / 500-output call to `CLAUDE_SONNET_4_6` is ~$0.014, well under the threshold; the same call to `GPT_5_4_MINI` is ~$0.004.

```python
from inspect_ai.model import GenerateConfig
from model_registry import GPT_5_4_MINI, CLAUDE_SONNET_4_6, GEMINI_3_FLASH_PREVIEW

resp = await GPT_5_4_MINI.generate(
    "Your prompt here", config=GenerateConfig(temperature=0.0)
)
text = resp.completion
```

Use one of these when you want to make an LLM call. You can decide to use only one model, or mix them across calls — for example, route easy problems to a mini-tier handle and hard problems to a stronger-tier handle. `config` is optional; pass a `GenerateConfig` to set sampling parameters such as `temperature`. See `inspect_ai.model.GenerateConfig` for the full set. All LLM calls must go through one of the six handles above.

## Scoring

The agent's `<code>` block is extracted, concatenated with hidden setup and test code, and run inside the sandbox. The score for the sample is **1.0** if the appended code makes `result` match the reference under all hidden test inputs, else **0.0**. No partial credit.

A subset of problems additionally enforce **style/idiom constraints on the submitted code itself**. Two flavors appear: (1) forbidding Python control-flow constructs like `for`/`while` to push toward library calls, and (2) requiring a specific library function name to appear in the solution, ruling out manual reimplementations. The constraint is sometimes flagged in the prompt ("without using X", "the efficient way", "not one by one") but is more often implicit in the spirit of the question: asking *"how do I do X with NumPy"* invites a NumPy-idiomatic answer, and a workaround that bypasses the library can fail even when the output is correct. When this happens, the per-problem `test_result.md` shows an assertion raised from a `test_string` function (versus correctness failures, which raise from `test_execution`). Both outcomes score 0.0; the traceback tells the agent whether to fix the *answer* or the *form*.

## Per-example score

During training, the per-example score is `100 × (0 or 1) − cost_penalty`, where the binary part is correctness and `cost_penalty ∈ [0, 1]` is a bounded function of agent LLM spend (only `get_model()` calls are metered — `python_session` and `sandbox()` don't count). Spend below **$0.04** incurs zero penalty (free zone). Above $0.04 the penalty ramps linearly, hitting its maximum of 1.0 at **$1.00** of per-example agent spend; further spend incurs no extra penalty. The penalty's [0, 1] range is two orders of magnitude smaller than the score scale, so it acts as a tiebreaker between correctness-tied agents — it never overrides a real correctness gap.

## Diagnostics

`print()` output from the solver is captured into `agent_stdout`. The extracted `<code>` block and the test program's stdout/stderr are surfaced as per-example diagnostic files (`extracted_code.md`, `test_result.md`) so failures can be inspected without re-running.

## Scratch space

Your working directory is your iteration's evolution dir. If you need to drop a small test harness or scratch script while debugging, write it there directly — `/tmp` is outside the write scope, and there's no need to clean up after yourself afterwards. Leftover debugging artifacts in the iteration dir are useful for retrospective analysis.
