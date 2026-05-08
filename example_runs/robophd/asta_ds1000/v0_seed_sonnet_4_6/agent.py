"""Baseline DS-1000 solver — Sonnet 4.6 variant.

Identical to the canonical seed (`seed_tl2zapfl`) except the solver model
is `CLAUDE_SONNET_4_6` instead of `GPT_5_4_MINI`. Used as a fair baseline
for evolved agents whose ensembles are dominated by Sonnet 4.6 (e.g. the
4-candidate-jury pattern where Sonnet is one of the candidates AND the
judge AND the rescue writer). Comparing such an agent's test score to a
GPT-5.4-mini seed conflates model uplift with architecture uplift; this
seed isolates the architecture component.

One-shot: send the problem prompt to Sonnet 4.6, take the response, and
emit the required `<code>...</code>` block. No self-check, no
python_session calls, no library-specific scaffolding.
"""

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, solver

from model_registry import CLAUDE_SONNET_4_6


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

        resp = await CLAUDE_SONNET_4_6.generate(
            state.input,
            config=GenerateConfig(temperature=0.0),
        )
        completion = resp.completion or ""
        state.output.completion = _wrap_in_code_tags(completion)
        print(f"  emitted {len(state.output.completion)} chars")
        return state

    return solve
