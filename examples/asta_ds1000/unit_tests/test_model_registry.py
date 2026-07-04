"""Pin the wired-in defaults on the asta_ds1000 model registry.

Two concerns this test covers:

  1. The Gemini handles in `examples/asta_ds1000/model_registry.py`
     have `reasoning_effort` baked in via `get_model(config=...)`.
     Without this assertion, a future Inspect-AI release that renames
     the kwarg (or stops honoring it on Model.config) would silently
     no-op and the timeout fix would regress invisibly.

  2. The asymmetry vs the non-Gemini handles is load-bearing:
     OpenAI/Anthropic are intentionally NOT pinned because their
     defaults are already conservative. We don't assert positively on
     those (their defaults are "no config set"), but we do verify the
     Gemini pins are present so the asymmetry is enforced where it's
     supposed to be.

Skip when the Anthropic key isn't set, since the registry validates it
at import time (per the comment in model_registry.py) and a missing key
would make every test in this module ImportError rather than skip.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ASTA_DS1000_DIR = REPO_ROOT / "examples" / "asta_ds1000"


pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")
    ),
    reason="model_registry import validates Anthropic key at construction time",
)


@pytest.fixture(scope="module")
def model_registry():
    """Import asta_ds1000's model_registry once per module.

    sys.path manipulation rather than a package install because
    examples/asta_ds1000/ isn't an installable package.
    """
    sys.path.insert(0, str(ASTA_DS1000_DIR))
    try:
        import model_registry as mr  # noqa: E402
        return mr
    finally:
        sys.path.remove(str(ASTA_DS1000_DIR))


def test_flash_lite_defaults_to_low(model_registry):
    assert (
        model_registry.GEMINI_3_1_FLASH_LITE.config.reasoning_effort == "low"
    )


def test_flash_lite_preview_alias_points_to_ga(model_registry):
    """Backwards-compat: evolved agents from before the preview→GA
    migration still import GEMINI_3_1_FLASH_LITE_PREVIEW. The alias
    must resolve to the same Model handle as the canonical GA name."""
    assert (
        model_registry.GEMINI_3_1_FLASH_LITE_PREVIEW
        is model_registry.GEMINI_3_1_FLASH_LITE
    )


def test_flash_defaults_to_low(model_registry):
    assert (
        model_registry.GEMINI_3_5_FLASH.config.reasoning_effort == "low"
    )


def test_flash_preview_alias_points_to_3_5(model_registry):
    """Backwards-compat: evolved agents from before the 3-flash-preview →
    3.5-flash migration still import GEMINI_3_FLASH_PREVIEW. The alias
    must resolve to the same Model handle as the canonical new name."""
    assert (
        model_registry.GEMINI_3_FLASH_PREVIEW
        is model_registry.GEMINI_3_5_FLASH
    )


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
            f"the asta_ds1000 model_registry.py wiring may have silently no-op'd"
        )


def test_all_nine_handles_exported(model_registry):
    """All nine handles must be exported unconditionally — the gate
    was removed in favor of the cost-penalty disciplining overuse.
    If any of these go missing, evolved agents that import them at
    eval time will ImportError silently in subprocess workers.
    (CLAUDE_FABLE_5 is intentionally excluded to keep runs comparable
    with the existing experimental campaign — no evolved agent ever
    selected it as a solver during its brief availability.)"""
    expected = (
        "GPT_5_4_MINI", "GPT_5_4", "GPT_5_5",
        "CLAUDE_HAIKU_4_5", "CLAUDE_SONNET_4_6", "CLAUDE_OPUS_4_8",
        "GEMINI_3_1_FLASH_LITE", "GEMINI_3_5_FLASH",
        "GEMINI_3_1_PRO_PREVIEW",
    )
    missing = [n for n in expected if not hasattr(model_registry, n)]
    assert not missing, f"missing handles: {missing}"


def test_pro_preview_reasoning_effort_pinned_to_low(model_registry):
    """The Gemini 3.1 Pro Preview handle should default to "low"
    reasoning, parallel to GEMINI_3_5_FLASH and Flash Lite — all
    three Gemini handles share the same default. Note: Inspect's google
    provider would silently UPGRADE "medium" to ThinkingLevel.HIGH on
    non-flash Gemini (_providers/google.py:778-780), so any value above
    "low" here would be misleading about what's actually sent."""
    assert (
        model_registry.GEMINI_3_1_PRO_PREVIEW.config.reasoning_effort == "low"
    )
