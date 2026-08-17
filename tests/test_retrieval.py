"""Every expected value here is computed by hand, not by running the code.

A test that asserts what the implementation happens to produce tests nothing.
"""
import pytest

from ragmeter.metrics.retrieval import (
    evaluate_retrieval,
    metric_names,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_perfect_retrieval():
    r = ["a", "b", "c"]
    g = ["a", "b", "c"]
    assert recall_at_k(r, g, 3) == 1.0
    assert precision_at_k(r, g, 3) == 1.0
    assert mrr_at_k(r, g, 3) == 1.0
    assert ndcg_at_k(r, g, 3) == 1.0


def test_nothing_relevant_retrieved():
    r = ["x", "y", "z"]
    g = ["a"]
    assert recall_at_k(r, g, 3) == 0.0
    assert precision_at_k(r, g, 3) == 0.0
    assert mrr_at_k(r, g, 3) == 0.0
    assert ndcg_at_k(r, g, 3) == 0.0


def test_partial_hit_at_rank_two():
    # retrieved: x(1) a(2) y(3); relevant: {a, b}
    # recall    = |{a}| / |{a,b}|          = 0.5
    # precision = |{a}| / 3                = 0.333...
    # mrr       = 1/2                      = 0.5
    # dcg       = 1/log2(3)                = 0.6309297535714574
    # idcg      = 1/log2(2) + 1/log2(3)    = 1.6309297535714574
    # ndcg      = 0.6309297535714574 / 1.6309297535714574 = 0.38685280723454163
    r = ["x", "a", "y"]
    g = ["a", "b"]
    assert recall_at_k(r, g, 3) == 0.5
    assert precision_at_k(r, g, 3) == pytest.approx(1 / 3)
    assert mrr_at_k(r, g, 3) == 0.5
    assert ndcg_at_k(r, g, 3) == pytest.approx(0.38685280723454163)


def test_duplicate_chunk_ids_are_deduplicated():
    # Without dedup, [a, a] at k=2 scores recall 0.5. With dedup, [a, b] scores 1.0.
    r = ["a", "a", "b"]
    g = ["a", "b"]
    assert recall_at_k(r, g, 2) == 1.0
    assert precision_at_k(r, g, 2) == 1.0


def test_fewer_retrieved_than_k():
    # retrieved: a(1); relevant: {a, b}; k=5
    # precision divides by 1 actually retrieved, not by k -- otherwise a system
    # that returns one perfect chunk is punished for its restraint.
    # dcg  = 1/log2(2)               = 1.0
    # idcg = 1/log2(2) + 1/log2(3)   = 1.6309297535714574
    # ndcg = 0.6131471927654584
    r = ["a"]
    g = ["a", "b"]
    assert recall_at_k(r, g, 5) == 0.5
    assert precision_at_k(r, g, 5) == 1.0
    assert mrr_at_k(r, g, 5) == 1.0
    assert ndcg_at_k(r, g, 5) == pytest.approx(0.6131471927654584)


def test_empty_retrieval_is_zero_recall_but_undefined_precision():
    # Recall is 0.0: it genuinely found none of the relevant chunks.
    # Precision is None: there is no denominator, so there is nothing to measure.
    r = []
    g = ["a"]
    assert recall_at_k(r, g, 5) == 0.0
    assert precision_at_k(r, g, 5) is None
    assert mrr_at_k(r, g, 5) == 0.0
    assert ndcg_at_k(r, g, 5) == 0.0


def test_empty_relevant_set_is_unmeasurable_not_zero():
    r = ["a", "b"]
    g = []
    assert recall_at_k(r, g, 2) is None
    assert precision_at_k(r, g, 2) is None
    assert mrr_at_k(r, g, 2) is None
    assert ndcg_at_k(r, g, 2) is None


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(["a"], ["a"], 0)


def test_evaluate_retrieval_returns_all_four_keyed_by_k():
    out = evaluate_retrieval(["x", "a", "y"], ["a", "b"], 3)
    assert out == {
        "recall@3": 0.5,
        "precision@3": pytest.approx(1 / 3),
        "mrr@3": 0.5,
        "ndcg@3": pytest.approx(0.38685280723454163),
    }


def test_metric_names_matches_evaluate_retrieval_keys():
    # If these ever drift, unmatched traces get metric keys that no aggregate
    # can line up with matched ones.
    assert metric_names(7) == list(evaluate_retrieval(["a"], ["a"], 7).keys())
