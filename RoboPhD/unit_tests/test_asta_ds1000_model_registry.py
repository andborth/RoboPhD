"""Pin the wired-in defaults on the asta_ds1000 model registry.

Two concerns this test covers:

  1. The Gemini handles in `examples/asta_ds1000/model_registry.py`
     have `reasoning_effort` baked in via `get_model(config=...)`.
     Without this assertion, a future Inspect-AI release that renames
     the kwarg (or stops honoring it on Model.config) would silently
     no-op and the timeout fix would regress invisibly.

  2. The asymmetry vs the four non-Gemini handles is load-bearing:
     OpenAI/Anthropic are intentionally NOT pinned because their
     defaults are already conservative. We don't assert positively on
     those (their defaults are "no config set"), but we do verify the
     two Gemini pins are present so the asymmetry is enforced where
     it's supposed to be.

Skip when the Anthropic key isn't set, since the registry validates it
at import time (per the comment in model_registry.py) and a missing key
would make every test in this module ImportError rather than skip.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
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
        model_registry.GEMINI_3_1_FLASH_LITE_PREVIEW.config.reasoning_effort == "low"
    )


def test_flash_defaults_to_medium(model_registry):
    assert (
        model_registry.GEMINI_3_FLASH_PREVIEW.config.reasoning_effort == "medium"
    )


def test_get_model_still_accepts_config_kwarg():
    """Guard against a future Inspect release renaming the kwarg.

    If `config` ever becomes `default_config` or similar, our
    get_model(..., config=GenerateConfig(...)) calls would silently
    drop the override and the Gemini pins would no-op. This signature
    check catches that on the first test run after upgrade.
    """
    import inspect as py_inspect

    from inspect_ai.model import get_model

    sig = py_inspect.signature(get_model)
    assert "config" in sig.parameters, (
        "inspect_ai.model.get_model no longer accepts a `config` kwarg — "
        "the Gemini reasoning_effort pins in examples/asta_ds1000/model_registry.py "
        "may have silently no-op'd"
    )
