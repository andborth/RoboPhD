"""Baseline DS-1000 solver.

One-shot baseline: send the problem prompt to the LLM, take the response,
and emit a `<code>...</code>` block. No self-check, no python_session
calls, no library-specific scaffolding. The scorer extracts the code
between `<code>` and `</code>` tags and runs it inside the sandbox.

Evolution is expected to add things like:
  - python_session self-check (run the candidate against the example
    inputs in the prompt, retry on failure)
  - library-specific prompt scaffolding (different system prompts for
    NumPy vs Pandas vs Matplotlib problems)
  - few-shot retrieval or chain-of-thought prompting
  - syntactic post-processing (strip markdown fences, balance brackets)
"""

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

# Pre-resolved Model handles. Pick one per call, or mix across calls.
# The cost penalty applied during training is computed against the
# total agent_cost_usd across whichever models you use. See
# model_registry.py for the full list (CLAUDE_HAIKU_4_5,
# GEMINI_3_1_FLASH_LITE_PREVIEW).
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
        print(f"[{state.sample_id}] library={state.metadata.get('library', '?')}")

        resp = await GPT_5_4_MINI.generate(
            state.input,
            config=GenerateConfig(temperature=0.0),
        )
        completion = resp.completion or ""
        state.output.completion = _wrap_in_code_tags(completion)
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
