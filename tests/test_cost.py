import httpx
import pytest
import respx

from ragmeter.metrics.cost import MODELS_URL, compute_cost, fetch_prices

CATALOG = {
    "data": [
        {"id": "openai/gpt-4o-mini",
         "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}},
        {"id": "nvidia/nemotron-3-ultra-550b-a55b:free",
         "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "broken/no-pricing"},
        {"id": "broken/variable-pricing",
         "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}


@respx.mock
def test_fetch_prices_parses_and_skips_unusable_entries():
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
    prices = fetch_prices()
    assert prices["openai/gpt-4o-mini"] == (0.00000015, 0.0000006)
    assert prices["nvidia/nemotron-3-ultra-550b-a55b:free"] == (0.0, 0.0)
    # Missing pricing and negative "variable" pricing are both dropped rather
    # than turned into a number that would silently understate real spend.
    assert "broken/no-pricing" not in prices
    assert "broken/variable-pricing" not in prices


@respx.mock
def test_fetch_prices_raises_on_http_error():
    respx.get(MODELS_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_prices()


def test_supplied_cost_wins_over_computed():
    prices = {"m": (1.0, 1.0)}
    assert compute_cost("m", 10, 10, prices, supplied=0.5) == 0.5


def test_computed_from_tokens():
    prices = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}
    # 1000 * 0.00000015 + 500 * 0.0000006 = 0.00015 + 0.0003 = 0.00045
    assert compute_cost("openai/gpt-4o-mini", 1000, 500, prices) == pytest.approx(0.00045)


def test_unknown_model_is_none_not_zero():
    # A zero here would make an unpriced model look free and quietly break any
    # cost regression threshold built on top of it.
    assert compute_cost("who/knows", 1000, 500, {}) is None


def test_missing_tokens_is_none():
    prices = {"m": (1.0, 1.0)}
    assert compute_cost("m", None, 500, prices) is None
    assert compute_cost("m", 1000, None, prices) is None


def test_missing_model_is_none():
    assert compute_cost(None, 1000, 500, {"m": (1.0, 1.0)}) is None
