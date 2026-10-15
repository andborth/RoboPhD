"""Baseline DS-1000 solver.

One-shot baseline: send the problem prompt to the LLM, take the response,
and emit the required `<code>...</code>` block.
"""

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

# LLM handles are imported from `model_registry`. Pick one per call,
# or mix across calls. See CLAUDE.md (Domain Background) for the
# full list of handles and their pricing.
from model_registry import GPT_5_4_MINI


def _wrap_in_code_tags(text: str) -> str:
    """Ensure the response is wrapped in `<code>...</code>` tags.

    The DS-1000 scorer's `postprocess` strips a few common envelopes
    (```python fences, <code> tags, END SOLUTION markers). This helper
    is conservative: if the model already produced `<code>...</code>`
    or a markdown fence, leave it alone; otherwise wrap raw code.
    """
    s = text.strip()
    if "<code>" in s and "</code>" in s:
        return s
    if s.startswith("```"):
        return s
    return f"<code>\n{s}\n</code>"


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Demonstration print statement — captured in `agent_stdout`
        # alongside the per-problem diagnostics, so anything you print
        # here is available for retrospective analysis.
        print(f"[{state.sample_id}] library={state.metadata.get('library', '?')}")

        resp = await GPT_5_4_MINI.generate(state.input)
        completion = resp.completion or ""
        state.output.completion = _wrap_in_code_tags(completion)
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
