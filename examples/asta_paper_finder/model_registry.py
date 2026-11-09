"""Model registry for PaperFindingBench evolved agents.

Exports nine pre-resolved Inspect-AI Model handles, grouped by family
into a cheap/fast tier, a standard tier, and a strong/slow tier:

  OpenAI:    GPT_5_4_MINI              GPT_5_4               GPT_5_5
  Anthropic: CLAUDE_HAIKU_4_5          CLAUDE_SONNET_4_6     CLAUDE_OPUS_4_8
  Google:    GEMINI_3_1_FLASH_LITE     GEMINI_3_5_FLASH      GEMINI_3_1_PRO_PREVIEW

Evolved agents import these handles and call `.generate(...)` on them;
the underlying provider/model strings live here, OUTSIDE the evolvable
artifact (agent.py is the only file in a candidate's file_mapping).
This keeps evolution from substituting an arbitrary provider/model.
The cost-penalty in the training scorer (see background.md) disciplines
overuse of the strong tier without requiring an explicit gate.

The benchmark's relevance judge (the LLM that scores semantic queries)
is NOT a handle here and is never billed to the agent — its spend is
tracked separately as other_cost_usd. Keeping the judge's model ID out
of this registry is what makes that split sound; do not add a handle
with the judge's model ID (see evaluator.py JUDGE_MODEL_IDS).

Usage from agent.py:

    from model_registry import GPT_5_4_MINI, CLAUDE_SONNET_4_6
    resp = await GPT_5_4_MINI.generate("...", config=GenerateConfig(...))

Pick one per call, or mix across calls. The cost penalty applied during
training (see background.md) is computed against the sum of
agent_cost_usd across whichever models you use.
"""

import os

from inspect_ai.model import GenerateConfig, get_model

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
_GEMINI_3_1_FLASH_LITE_ID = "google/gemini-3.1-flash-lite"
_GEMINI_3_5_FLASH_ID = "google/gemini-3.5-flash"

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
# you only intend to call one of the nine handles. See README.md
# "Credentials" for the user-facing version.
GPT_5_4_MINI = get_model(_GPT_5_4_MINI_ID)
GPT_5_4 = get_model(_GPT_5_4_ID)
CLAUDE_HAIKU_4_5 = get_model(_CLAUDE_HAIKU_4_5_ID, api_key=_ANTHROPIC_API_KEY)
CLAUDE_SONNET_4_6 = get_model(_CLAUDE_SONNET_4_6_ID, api_key=_ANTHROPIC_API_KEY)
# Gemini 3.5 Flash and 3.1 Flash Lite are thinking models that default
# to "high" reasoning at the provider — too aggressive for the short
# query-reformulation and rerank prompts a paper-finding pipeline makes
# (deep reasoning inflates latency and bills as output tokens). Pin all
# Gemini handles to "low" for cheap throughput; the step-up path is
# per-call `reasoning_effort="high"` from an evolved agent. Only "low"
# and "high" are sanctioned override values: Inspect's Google provider
# silently upgrades "medium" to HIGH thinking on non-flash Gemini
# handles, so a "medium" default would be misleading. Per-call configs
# from evolved agents merge field-by-field over these defaults via
# Inspect's base_config.merge mechanism.
#
# Why only Gemini? GPT-5 family defaults to reasoning_effort="none" at
# OpenAI's API, and Claude 4.x defaults to extended-thinking-off — both
# already-conservative baselines. If a timeout pattern emerges on a
# Sonnet or GPT-5 *opt-in* reasoning path, this is the place to add an
# analogous pin.
GEMINI_3_1_FLASH_LITE = get_model(
    _GEMINI_3_1_FLASH_LITE_ID,
    config=GenerateConfig(reasoning_effort="low"),
)
GEMINI_3_5_FLASH = get_model(
    _GEMINI_3_5_FLASH_ID,
    config=GenerateConfig(reasoning_effort="low"),
)

# Strong-tier handles. Cost rates are ~5-40× the cheap tier; the
# cost-penalty in the training scorer disciplines overuse (see
# background.md). Raise --cost-threshold for a more generous free zone
# when relying on these, or raise --cost-per-error to soften the
# per-dollar penalty.
_GPT_5_5_ID = "openai/gpt-5.5"
_CLAUDE_OPUS_4_8_ID = "anthropic/claude-opus-4-8"
_GEMINI_3_1_PRO_PREVIEW_ID = "google/gemini-3.1-pro-preview"

GPT_5_5 = get_model(_GPT_5_5_ID)
CLAUDE_OPUS_4_8 = get_model(_CLAUDE_OPUS_4_8_ID, api_key=_ANTHROPIC_API_KEY)
# Pro Preview thinking cannot be disabled at all; "low" is the cheapest
# legal floor (and keeps the Gemini family default uniform). Evolution
# opts up to "high" per-call when it needs more.
GEMINI_3_1_PRO_PREVIEW = get_model(
    _GEMINI_3_1_PRO_PREVIEW_ID,
    config=GenerateConfig(reasoning_effort="low"),
)


# ---------------------------------------------------------------------------
# Model menu (main.py --model-menu)
# ---------------------------------------------------------------------------
# A run may narrow its menu to a subset of the nine handles. main.py's
# evaluator sets ROBOPHD_MODEL_MENU before importing this module (worker
# subprocesses inherit it); when set, every handle NOT on the menu is deleted
# from this module below, so importing it fails. The menu can only narrow:
# an unknown name raises at import rather than being ignored.
#
# Self-contained on purpose — no import of model_menu.py — because the submit
# scripts copy this file alone into submission directories. With the env var
# unset (every submission, every unrestricted run) nothing below changes what
# this module exports. The handle list and env-var name duplicate
# model_menu.MENU / MODEL_MENU_ENV; unit_tests/test_model_menu.py
# (test_registry_menu_copy_matches_model_menu) pins the copies together.
_MODEL_MENU_ENV = "ROBOPHD_MODEL_MENU"
_MENU = (
    "GPT_5_4_MINI",
    "GPT_5_4",
    "GPT_5_5",
    "CLAUDE_HAIKU_4_5",
    "CLAUDE_SONNET_4_6",
    "CLAUDE_OPUS_4_8",
    "GEMINI_3_1_FLASH_LITE",
    "GEMINI_3_5_FLASH",
    "GEMINI_3_1_PRO_PREVIEW",
)
# Every exported handle -> the model id Inspect records it under in
# model_usage (the evaluator's off-menu check matches on these ids).
_HANDLE_IDS = {
    "GPT_5_4_MINI": _GPT_5_4_MINI_ID,
    "GPT_5_4": _GPT_5_4_ID,
    "GPT_5_5": _GPT_5_5_ID,
    "CLAUDE_HAIKU_4_5": _CLAUDE_HAIKU_4_5_ID,
    "CLAUDE_SONNET_4_6": _CLAUDE_SONNET_4_6_ID,
    "CLAUDE_OPUS_4_8": _CLAUDE_OPUS_4_8_ID,
    "GEMINI_3_1_FLASH_LITE": _GEMINI_3_1_FLASH_LITE_ID,
    "GEMINI_3_5_FLASH": _GEMINI_3_5_FLASH_ID,
    "GEMINI_3_1_PRO_PREVIEW": _GEMINI_3_1_PRO_PREVIEW_ID,
}


def _read_menu_env():
    raw = os.environ.get(_MODEL_MENU_ENV, "")
    names = []
    for n in (s.strip() for s in raw.split(",")):
        if n and n not in names:
            names.append(n)
    if not names:
        return None
    unknown = [n for n in names if n not in _MENU]
    if unknown:
        raise ImportError(
            f"{_MODEL_MENU_ENV} names handles not on the model menu: "
            f"{', '.join(unknown)}. Valid handles: {', '.join(_MENU)}"
        )
    return tuple(names)


# None = unrestricted (all handles exported).
ACTIVE_MENU = _read_menu_env()


def allowed_model_ids(menu=None):
    """Model ids a run may record usage under: the menu's, or (menu None)
    every id this registry can produce, archive-only handles included.

    Public: this is the registry's contract with the evaluator's off-menu
    check. Harmless for an agent to import -- it grants no model access."""
    return frozenset(_HANDLE_IDS[h] for h in (menu or _HANDLE_IDS))


if ACTIVE_MENU is not None:
    for _name in _HANDLE_IDS:
        if _name not in ACTIVE_MENU:
            del globals()[_name]
    del _name
