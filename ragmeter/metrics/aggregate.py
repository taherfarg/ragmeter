"""Aggregation over values that may be unmeasurable.

Every summary reports n_null alongside the mean. An average over 12 of 200
questions is not the same claim as an average over 200, and the reader must be
able to tell them apart.
"""

import math
from statistics import fmean

__all__ = ["percentile", "summarize"]


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. No interpolation, no dependencies."""
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    if not values:
        return None
    ordered = sorted(values)
    index = max(1, math.ceil(p / 100 * len(ordered))) - 1
    return ordered[index]


def summarize(values: list[float | None]) -> dict[str, float | int | None]:
    """Mean/p50/p95 over the measurable values, plus how many were not."""
    measured = [v for v in values if v is not None]
    return {
        "n": len(values),
        "n_measured": len(measured),
        "n_null": len(values) - len(measured),
        "mean": fmean(measured) if measured else None,
        "p50": percentile(measured, 50),
        "p95": percentile(measured, 95),
    }
