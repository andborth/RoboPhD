"""Model registry for DiscoveryBench evolved agents.

Exports three pre-resolved Inspect-AI Model handles. Evolved agents
import these handles and call `.generate(...)` on them; the underlying
provider/model strings live here, OUTSIDE the evolvable artifact
(agent.py is the only file in a candidate's file_mapping). This keeps
evolution from substituting an arbitrary provider/model.

Usage from agent.py:

    from model_registry import GPT_5_4_MINI, CLAUDE_HAIKU_4_5
    resp = await GPT_5_4_MINI.generate("...", config=GenerateConfig(...))

Pick one per call, or mix across calls. The per-example cost penalty
applied during training (see background.md) is computed against the
sum of agent_cost_usd across whichever models you use.
"""

from inspect_ai.model import get_model

# Internal — strings stay private so agent.py doesn't import them.
_GPT_5_4_MINI_ID = "openai/gpt-5.4-mini"
_CLAUDE_HAIKU_4_5_ID = "anthropic/claude-haiku-4-5-20251001"
_GEMINI_3_1_FLASH_LITE_PREVIEW_ID = "google/gemini-3.1-flash-lite-preview"

# Public handles — pre-resolved Model objects. Provider key validation
# is asymmetric: OpenAI and Google initialize their clients lazily on
# first .generate() call, but the Anthropic provider validates
# ANTHROPIC_API_KEY at get_model() construction time. Importing this
# module therefore requires all three keys to be set in the
# environment, even if you only intend to call one of the three
# handles. See README.md "Credentials" for the user-facing version.
GPT_5_4_MINI = get_model(_GPT_5_4_MINI_ID)
CLAUDE_HAIKU_4_5 = get_model(_CLAUDE_HAIKU_4_5_ID)
GEMINI_3_1_FLASH_LITE_PREVIEW = get_model(_GEMINI_3_1_FLASH_LITE_PREVIEW_ID)
