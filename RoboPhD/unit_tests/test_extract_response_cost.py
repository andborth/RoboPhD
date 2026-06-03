"""Unit tests for RoboPhD.eval_utils.extract_response_cost.

Focus: cost extraction must not silently report $0 under OpenRouter BYOK
(bring-your-own-key), where OpenRouter's own ``usage.cost`` is a legitimate 0
and the real spend lives in ``usage.cost_details.upstream_inference_cost``.
Also covers the zero-with-tokens fall-through and the genuine-free-call path.

Responses are stubbed with SimpleNamespace; the pricing-DB fallback
(litellm.completion_cost) is monkeypatched so no network/litellm pricing data
is required.
"""

from types import SimpleNamespace

import pytest

from RoboPhD.eval_utils import extract_response_cost


def _resp(usage=None, hidden=None):
    r = SimpleNamespace()
    r.usage = usage
    r._hidden_params = hidden or {}
    return r


def _usage(cost=None, cost_details=None, prompt_tokens=0, completion_tokens=0):
    return SimpleNamespace(
        cost=cost,
        cost_details=cost_details,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


@pytest.fixture(autouse=True)
def _stub_completion_cost(monkeypatch):
    """Make the pricing-DB fallback deterministic and observable."""
    calls = {"n": 0}

    def fake(completion_response=None, model=None):
        calls["n"] += 1
        return 0.000625

    import litellm
    monkeypatch.setattr(litellm, "completion_cost", fake)
    return calls


def test_byok_reads_upstream_inference_cost(_stub_completion_cost):
    # The bug: cost=0 + is_byok, real spend only in cost_details.
    usage = _usage(
        cost=0,
        cost_details={"upstream_inference_cost": 6e-06},
        prompt_tokens=6, completion_tokens=3,
    )
    assert extract_response_cost(_resp(usage), "m") == pytest.approx(6e-06)
    assert _stub_completion_cost["n"] == 0  # short-circuited, no DB lookup


def test_byok_cost_details_as_attr_object(_stub_completion_cost):
    cd = SimpleNamespace(upstream_inference_cost=2.5e-05)
    usage = _usage(cost=0, cost_details=cd, prompt_tokens=100, completion_tokens=50)
    assert extract_response_cost(_resp(usage), "m") == pytest.approx(2.5e-05)


def test_zero_upstream_with_tokens_falls_through(_stub_completion_cost):
    # cost_details present but upstream is 0 → not trusted → DB fallback.
    usage = _usage(
        cost=0,
        cost_details={"upstream_inference_cost": 0},
        prompt_tokens=10, completion_tokens=5,
    )
    assert extract_response_cost(_resp(usage), "m") == pytest.approx(0.000625)
    assert _stub_completion_cost["n"] == 1


def test_positive_usage_cost_trusted(_stub_completion_cost):
    usage = _usage(cost=0.0123, prompt_tokens=10, completion_tokens=5)
    assert extract_response_cost(_resp(usage), "m") == pytest.approx(0.0123)
    assert _stub_completion_cost["n"] == 0


def test_zero_cost_with_tokens_falls_through(_stub_completion_cost):
    # Non-BYOK provider that omits cost: 0 with real tokens → DB fallback.
    usage = _usage(cost=0, prompt_tokens=8, completion_tokens=2)
    assert extract_response_cost(_resp(usage), "m") == pytest.approx(0.000625)
    assert _stub_completion_cost["n"] == 1


def test_zero_cost_zero_tokens_is_real_free_call(_stub_completion_cost):
    # Genuine free / zero-token call: trust the 0, don't hit the DB.
    usage = _usage(cost=0, prompt_tokens=0, completion_tokens=0)
    assert extract_response_cost(_resp(usage), "m") == 0.0
    assert _stub_completion_cost["n"] == 0


def test_hidden_response_cost_used_when_usage_cost_absent(_stub_completion_cost):
    usage = _usage(cost=None, prompt_tokens=5, completion_tokens=5)
    resp = _resp(usage, hidden={"response_cost": 0.009})
    assert extract_response_cost(resp, "m") == pytest.approx(0.009)
    assert _stub_completion_cost["n"] == 0


def test_fallback_raises_returns_zero(monkeypatch):
    import litellm

    def boom(completion_response=None, model=None):
        raise ValueError("This model isn't mapped yet")

    monkeypatch.setattr(litellm, "completion_cost", boom)
    usage = _usage(cost=None, prompt_tokens=5, completion_tokens=5)
    assert extract_response_cost(_resp(usage), "some-unmapped-model") == 0.0
