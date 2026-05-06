"""Model registry for DS-1000 evolved agents.

Exports six pre-resolved Inspect-AI Model handles, paired by family
into a cheap/fast tier and a stronger/slower tier:

  OpenAI:    GPT_5_4_MINI                  / GPT_5_4
  Anthropic: CLAUDE_HAIKU_4_5              / CLAUDE_SONNET_4_6
  Google:    GEMINI_3_1_FLASH_LITE_PREVIEW / GEMINI_3_FLASH_PREVIEW

Evolved agents import these handles and call `.generate(...)` on them;
the underlying provider/model strings live here, OUTSIDE the evolvable
artifact (agent.py is the only file in a candidate's file_mapping).
This keeps evolution from substituting an arbitrary provider/model.

Usage from agent.py:

    from model_registry import GPT_5_4_MINI, CLAUDE_SONNET_4_6
    resp = await GPT_5_4_MINI.generate("...", config=GenerateConfig(...))

Pick one per call, or mix across calls. The per-example cost penalty
applied during training (see background.md) is computed against the
sum of agent_cost_usd across whichever models you use.
"""

import os

from inspect_ai.model import get_model

# Internal — strings stay private so agent.py doesn't import them.
_GPT_5_4_MINI_ID = "openai/gpt-5.4-mini"
_GPT_5_4_ID = "openai/gpt-5.4-2026-03-05"
_CLAUDE_HAIKU_4_5_ID = "anthropic/claude-haiku-4-5-20251001"
_CLAUDE_SONNET_4_6_ID = "anthropic/claude-sonnet-4-6"
# Note: the Anthropic IDs are asymmetric in date-stamping (Haiku
# is stamped, Sonnet is not). This mirrors the actual upstream
# model-ID conventions — DO NOT "normalize" them by adding a
# date stamp to Sonnet or removing it from Haiku without
# verifying the resulting strings still resolve. Inspect-AI
# routes to whatever the provider's canonical id is.
_GEMINI_3_1_FLASH_LITE_PREVIEW_ID = "google/gemini-3.1-flash-lite-preview"
_GEMINI_3_FLASH_PREVIEW_ID = "google/gemini-3-flash-preview"

# RoboPhD convention: prefer ANTHROPIC_API_KEY_FOR_ROBOPHD over
# ANTHROPIC_API_KEY so the user's Claude Code CLI sessions (which read
# ANTHROPIC_API_KEY) keep using their normal subscription credentials,
# while RoboPhD evaluations use a separate API key. We pass the
# resolved key to inspect_ai's get_model(api_key=...) explicitly rather
# than mutating os.environ, since the env mutation would leak to any
# Claude Code subprocesses spawned later in the run. Same pattern as
# examples/text2sql/evaluator.py:208-210.
_ANTHROPIC_API_KEY = (
    os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")
)

# Public handles — pre-resolved Model objects. Provider key validation
# is asymmetric: OpenAI and Google initialize their clients lazily on
# first .generate() call, but the Anthropic provider validates its key
# at get_model() construction time. Importing this module therefore
# requires all three provider keys to be available (Anthropic via
# either ANTHROPIC_API_KEY or ANTHROPIC_API_KEY_FOR_ROBOPHD), even if
# you only intend to call one of the six handles. See README.md
# "Credentials" for the user-facing version.
GPT_5_4_MINI = get_model(_GPT_5_4_MINI_ID)
GPT_5_4 = get_model(_GPT_5_4_ID)
CLAUDE_HAIKU_4_5 = get_model(_CLAUDE_HAIKU_4_5_ID, api_key=_ANTHROPIC_API_KEY)
CLAUDE_SONNET_4_6 = get_model(_CLAUDE_SONNET_4_6_ID, api_key=_ANTHROPIC_API_KEY)
GEMINI_3_1_FLASH_LITE_PREVIEW = get_model(_GEMINI_3_1_FLASH_LITE_PREVIEW_ID)
GEMINI_3_FLASH_PREVIEW = get_model(_GEMINI_3_FLASH_PREVIEW_ID)
