"""Token counts to USD, priced from OpenRouter's public catalog.

The catalog endpoint needs no API key, so prices stay current without the user
maintaining a hand-written table that silently goes stale.
"""

import httpx

MODELS_URL = "https://openrouter.ai/api/v1/models"

__all__ = ["MODELS_URL", "fetch_prices", "compute_cost"]


def fetch_prices(client: httpx.Client | None = None) -> dict[str, tuple[float, float]]:
    """Return {model_id: (prompt_usd_per_token, completion_usd_per_token)}.

    Entries without usable pricing are dropped, never defaulted to zero.
    """
    owned = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.get(MODELS_URL)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owned:
            client.close()

    prices: dict[str, tuple[float, float]] = {}
    for model in payload.get("data", []):
        pricing = model.get("pricing") or {}
        try:
            prompt = float(pricing["prompt"])
            completion = float(pricing["completion"])
        except (KeyError, TypeError, ValueError):
            continue
        # OpenRouter uses -1 for models whose price is not fixed.
        if prompt < 0 or completion < 0:
            continue
        prices[model["id"]] = (prompt, completion)
    return prices


def compute_cost(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prices: dict[str, tuple[float, float]],
    supplied: float | None = None,
) -> float | None:
    """Client-supplied cost wins. Anything unpriceable is None, never 0.0."""
    if supplied is not None:
        return supplied
    if model is None or prompt_tokens is None or completion_tokens is None:
        return None
    if model not in prices:
        return None
    prompt_price, completion_price = prices[model]
    return prompt_tokens * prompt_price + completion_tokens * completion_price
