"""Model registry for DiscoveryBench evolved agents.

Exports three pre-resolved Inspect-AI Model handles. Evolved agents
import these handles and call `.generate(...)` on them; the underlying
provider/model strings live here, OUTSIDE the evolvable artifact
(agent.py is the only file in a candidate's file_mapping). This keeps
evolution from substituting an arbitrary provider/model.

Usage from agent.py:

    from model_registry import GPT_5_4_MINI, CLAUDE_HAIKU_4_5
    resp = await GPT_5_4_MINI.generate("...", config=GenerateConfig(...))

Pick one per call, or mix across calls (e.g., a cheap model for fan-out
followed by a stronger model for synthesis). The $0.10 per-example
agent budget is shared across whichever models you use.
"""

from inspect_ai.model import get_model

# Internal — strings stay private so agent.py doesn't import them.
_GPT_5_4_MINI_ID = "openai/gpt-5.4-mini"
_CLAUDE_HAIKU_4_5_ID = "anthropic/claude-haiku-4-5-20251001"
_GEMINI_3_1_FLASH_LITE_PREVIEW_ID = "google/gemini-3.1-flash-lite-preview"

# Public handles — pre-resolved Model objects. Provider clients are
# initialized lazily on first .generate() call, so import is cheap.
GPT_5_4_MINI = get_model(_GPT_5_4_MINI_ID)
CLAUDE_HAIKU_4_5 = get_model(_CLAUDE_HAIKU_4_5_ID)
GEMINI_3_1_FLASH_LITE_PREVIEW = get_model(_GEMINI_3_1_FLASH_LITE_PREVIEW_ID)
