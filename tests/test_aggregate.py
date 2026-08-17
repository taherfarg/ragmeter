import pytest

from ragmeter.metrics.aggregate import percentile, summarize


def test_summarize_excludes_nulls_and_reports_them():
    # mean of [1.0, 3.0] is 2.0. If the None were counted as 0.0 the mean would
    # be 1.333 -- the exact silent corruption this function exists to prevent.
    out = summarize([1.0, None, 3.0])
    assert out["n"] == 3
    assert out["n_measured"] == 2
    assert out["n_null"] == 1
    assert out["mean"] == 2.0


def test_summarize_all_null():
    out = summarize([None, None])
    assert out["n_measured"] == 0
    assert out["n_null"] == 2
    assert out["mean"] is None
    assert out["p50"] is None
    assert out["p95"] is None


def test_summarize_empty():
    out = summarize([])
    assert out == {"n": 0, "n_measured": 0, "n_null": 0,
                   "mean": None, "p50": None, "p95": None}


def test_percentile_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5    # ceil(0.50 * 10) = 5 -> values[4]
    assert percentile(values, 95) == 10   # ceil(0.95 * 10) = 10 -> values[9]
    assert percentile(values, 100) == 10


def test_percentile_single_value():
    assert percentile([42], 95) == 42


def test_percentile_empty_is_none():
    assert percentile([], 50) is None


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError, match="p must be in"):
        percentile([1, 2], 101)
