"""Pin the wired-in defaults on the asta_ds1000 model registry.

Three concerns this test covers:

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

  3. The stronger-models gate (env var ASTA_DS1000_ALLOW_STRONGER_MODELS)
     correctly hides three additional handles by default and exposes
     them when set. Cost asymmetry is severe (~5-10× cheap tier), so
     accidental leakage is a real concern.

Skip when the Anthropic key isn't set, since the registry validates it
at import time (per the comment in model_registry.py) and a missing key
would make every test in this module ImportError rather than skip.
"""
import os
import subprocess
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


# --- Stronger-models gate (ASTA_DS1000_ALLOW_STRONGER_MODELS) ----------------
# Subprocess-based to keep import-time semantics clean. The default
# `model_registry` fixture above imports without the env var, so the
# fixture-using test catches accidental leakage; the env-var-set tests
# spawn fresh subprocesses so the writeable env doesn't pollute other
# tests in the suite.
#
# The set of gated handles lives on `model_registry.GATED_HANDLE_NAMES`
# (the single source of truth, also consumed by main.py's resume-time
# auto-detect). These tests pull the names from that constant rather
# than hardcoding a local tuple, so a future fourth gated handle gets
# tested automatically AND drift between the constant and the
# if-block body fails this file loud (the constant says X, but the
# `if` block doesn't define X → hasattr check fails).


def test_stronger_models_absent_by_default(model_registry):
    """Without ASTA_DS1000_ALLOW_STRONGER_MODELS=1, the gated handles
    listed in `GATED_HANDLE_NAMES` must not be exported — accidental
    presence inflates per-example spend by ~5-10× the cheap tier."""
    for name in model_registry.GATED_HANDLE_NAMES:
        assert not hasattr(model_registry, name), (
            f"{name} leaked into the default registry"
        )


@pytest.fixture(scope="module")
def with_strong_env(model_registry):
    """Import model_registry in a subprocess with the env var set, and
    parse handle availability + the Pro Preview's reasoning_effort.

    Iterates over GATED_HANDLE_NAMES (read from the parent fixture's
    no-env-var import — the constant is defined at module level
    BEFORE the env-gated `if` block, so it's available without the
    flag). This is what makes the test fail-loud on
    constant↔if-block-body drift: if a name appears in the constant
    but isn't created inside the if block, hasattr returns False and
    `have_strong` is False.

    Subprocess (rather than fixture-based reload) for two reasons:
    (1) avoids polluting the env var into the rest of the test suite,
    (2) guarantees a clean import that exercises the real
    `if os.environ.get(...) == "1":` gate at module load. Module-scoped
    so multiple env-gated tests can share one subprocess invocation.
    """
    gated_names = list(model_registry.GATED_HANDLE_NAMES)
    code = f"""
import sys
sys.path.insert(0, {str(ASTA_DS1000_DIR)!r})
import model_registry as mr
gated = {gated_names!r}
have = all(hasattr(mr, n) for n in gated)
missing = [n for n in gated if not hasattr(mr, n)]
print("HAVE_STRONG:", have)
print("MISSING:", missing)
if have:
    print("PRO_EFFORT:", mr.GEMINI_3_1_PRO_PREVIEW.config.reasoning_effort)
"""
    env = dict(os.environ)
    env["ASTA_DS1000_ALLOW_STRONGER_MODELS"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"subprocess failed (returncode={r.returncode}):\n"
        f"STDOUT: {r.stdout}\nSTDERR: {r.stderr[-1000:]}"
    )

    state: dict = {"raw": r.stdout}
    for line in r.stdout.splitlines():
        if line.startswith("HAVE_STRONG:"):
            state["have_strong"] = line.endswith("True")
        elif line.startswith("MISSING:"):
            state["missing"] = line.split(":", 1)[1].strip()
        elif line.startswith("PRO_EFFORT:"):
            state["pro_effort"] = line.split(":", 1)[1].strip()
    return state


def test_stronger_models_present_when_flag_set(with_strong_env):
    """With ASTA_DS1000_ALLOW_STRONGER_MODELS=1, every name in
    `GATED_HANDLE_NAMES` must actually be exported by the registry —
    otherwise the constant has drifted from the `if` block body and
    main.py's auto-detect would set the env var for handles that
    don't exist."""
    assert with_strong_env.get("have_strong") is True, (
        f"GATED_HANDLE_NAMES claims handles that the registry "
        f"doesn't actually create when the flag is set. Missing: "
        f"{with_strong_env.get('missing')}\nRaw:\n{with_strong_env['raw']}"
    )


def test_pro_preview_reasoning_effort_pinned_to_medium(with_strong_env):
    """The Gemini 3.1 Pro Preview handle should default to medium
    reasoning, parallel to GEMINI_3_FLASH_PREVIEW. Conservative starting
    point given the timeouts we saw on Flash Lite at high reasoning."""
    assert with_strong_env.get("pro_effort") == "medium", (
        f"GEMINI_3_1_PRO_PREVIEW.config.reasoning_effort != 'medium':\n"
        f"{with_strong_env['raw']}"
    )
