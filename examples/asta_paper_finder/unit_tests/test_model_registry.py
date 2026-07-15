"""Pin the wired-in defaults on the asta_paper_finder model registry.

Three concerns this test covers:

  1. The Gemini handles have `reasoning_effort` baked in via
     `get_model(config=...)`. Without this assertion, a future
     Inspect-AI release that renames the kwarg (or stops honoring it on
     Model.config) would silently no-op and Gemini calls would run at
     the provider's expensive "high" default.

  2. All nine handles are exported — evolved agents that import them at
     eval time would otherwise ImportError inside subprocess workers.

  3. **Judge-split soundness**: no registry model ID may equal the
     benchmark's relevance-judge model ID. The evaluator's
     agent-vs-judge cost split is an exact match on model ID
     (evaluator.JUDGE_MODEL_IDS); a registry handle resolving to the
     judge's ID would silently misroute that agent's spend into the
     never-penalized other_cost_usd bucket.

Import-requiring tests skip when the Anthropic key isn't set, since the
registry validates it at import time. The judge-ID guard parses the
source instead, so it runs everywhere.
"""
import ast
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PFB_DIR = REPO_ROOT / "examples" / "asta_paper_finder"

_HAS_ANTHROPIC_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")
)

requires_anthropic = pytest.mark.skipif(
    not _HAS_ANTHROPIC_KEY,
    reason="model_registry import validates Anthropic key at construction time",
)


def _registry_ids_from_source() -> dict[str, str]:
    """Handle name -> "provider/model" from _<HANDLE>_ID assignments,
    parsed from source (no import → no key requirement)."""
    src = (PFB_DIR / "model_registry.py").read_text()
    out = {}
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("_")
            and node.targets[0].id.endswith("_ID")
            and isinstance(node.value, ast.Constant)
        ):
            out[node.targets[0].id[1:-3]] = node.value.value
    return out


@pytest.fixture(scope="module")
def model_registry():
    """Import asta_paper_finder's model_registry once per module."""
    sys.path.insert(0, str(PFB_DIR))
    try:
        import model_registry as mr  # noqa: E402
        return mr
    finally:
        sys.path.remove(str(PFB_DIR))


EXPECTED_HANDLES = (
    "GPT_5_4_MINI", "GPT_5_4", "GPT_5_5",
    "CLAUDE_HAIKU_4_5", "CLAUDE_SONNET_4_6", "CLAUDE_OPUS_4_8",
    "GEMINI_3_1_FLASH_LITE", "GEMINI_3_5_FLASH", "GEMINI_3_1_PRO_PREVIEW",
)


def test_nine_id_constants_in_source():
    ids = _registry_ids_from_source()
    missing = [h for h in EXPECTED_HANDLES if h not in ids]
    assert not missing, f"missing _<HANDLE>_ID constants: {missing}"


def test_no_registry_id_is_the_judge_model():
    """The soundness condition for the evaluator's exact-match cost
    split. Checked against the evaluator's full JUDGE_MODEL_IDS set (GPT-4o
    plus the opt-in training grader gpt-5.4-nano) AND astabench's live
    GRADER_MODEL_NAME, so no agent-selectable handle can ever collide with a
    judge model — which would misroute its spend to the agent bucket."""
    from astabench.evals.paper_finder.relevance import GRADER_MODEL_NAME
    from evaluator import JUDGE_MODEL_IDS

    judge_ids = set(JUDGE_MODEL_IDS) | {GRADER_MODEL_NAME}
    assert "openai/gpt-4o-2024-11-20" in judge_ids
    assert "openai/gpt-5.4-nano" in judge_ids  # opt-in training grader
    ids = _registry_ids_from_source()
    offenders = {h: mid for h, mid in ids.items() if mid in judge_ids}
    assert not offenders, (
        f"registry handle(s) resolve to the relevance-judge model ID — "
        f"the agent-vs-judge cost split would misroute their spend: "
        f"{offenders}"
    )


def test_no_backcompat_aliases():
    """PFB has no pre-migration evolved agents; the ds1000 _PREVIEW
    aliases were deliberately not ported. If one reappears, it was
    probably copied over by accident."""
    ids_src = (PFB_DIR / "model_registry.py").read_text()
    assert "GEMINI_3_1_FLASH_LITE_PREVIEW" not in ids_src
    assert "GEMINI_3_FLASH_PREVIEW" not in ids_src


@requires_anthropic
def test_all_nine_handles_exported(model_registry):
    missing = [n for n in EXPECTED_HANDLES if not hasattr(model_registry, n)]
    assert not missing, f"missing handles: {missing}"


@requires_anthropic
@pytest.mark.parametrize("handle", [
    "GEMINI_3_1_FLASH_LITE", "GEMINI_3_5_FLASH", "GEMINI_3_1_PRO_PREVIEW",
])
def test_gemini_handles_pinned_to_low(model_registry, handle):
    assert getattr(model_registry, handle).config.reasoning_effort == "low"


def test_get_model_still_accepts_pinned_kwargs():
    """Guard against a future Inspect release renaming the kwargs we use.

    If `config` ever becomes `default_config` (or `api_key` becomes
    `key` etc.), our get_model(..., config=..., api_key=...) calls would
    silently drop the override. The fallout differs by kwarg:
      - config drop → Gemini reasoning_effort pins no-op (slow/expensive
        but functional)
      - api_key drop → Anthropic models 401 mid-run instead of failing
        at startup, which is a worse failure mode
    Signature check catches both on the first test run after upgrade.
    """
    import inspect as py_inspect

    from inspect_ai.model import get_model

    sig = py_inspect.signature(get_model)
    for kwarg in ("config", "api_key"):
        assert kwarg in sig.parameters, (
            f"inspect_ai.model.get_model no longer accepts a `{kwarg}` kwarg — "
            f"the asta_paper_finder model_registry.py wiring may have silently no-op'd"
        )
