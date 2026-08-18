"""Cross-check our retrieval metrics against RAGAS as a reference implementation.

RAGAS is a reference, not a dependency. Installing it pulls 60+ packages --
numpy, scipy, pandas, pyarrow and the whole langchain tree -- which is exactly
why it lives in an optional extra and why this file skips unless it is present.

Only the *deterministic* RAGAS metrics are used here. Their LLM-backed metrics
would make this test depend on a model's mood, and a flaky reference is not a
reference.

To run it, in a venv kept separate so the main one stays free of numpy/scipy:

    python -m venv .venv-ragas
    .venv-ragas/Scripts/python -m pip install -e ".[ragas]"
    .venv-ragas/Scripts/python -m pytest tests/test_ragas_parity.py
"""

import asyncio

import pytest

from ragmeter.metrics.retrieval import precision_at_k, recall_at_k

ragas_metrics = pytest.importorskip(
    "ragas.metrics", reason="the ragas extra is not installed"
)
from ragas.dataset_schema import SingleTurnSample  # noqa: E402

pytestmark = pytest.mark.ragas

# Distinct sentences, not chunk ids. RAGAS matches contexts by string
# similarity, so short ids like "c1" and "c2" would fuzzily match each other
# and quietly invent agreement.
CORPUS = {
    "c1": "Items may be returned within 30 days of delivery.",
    "c2": "Returned items must be unused and in original packaging.",
    "c3": "Our warehouse operates Sunday through Thursday.",
    "c5": "Standard shipping takes 3 to 5 business days.",
}


def _ragas_score(metric, retrieved: list[str], relevant: list[str]) -> float:
    sample = SingleTurnSample(
        retrieved_contexts=[CORPUS[c] for c in retrieved],
        reference_contexts=[CORPUS[c] for c in relevant],
    )
    return asyncio.run(metric.single_turn_ascore(sample))


@pytest.mark.parametrize(
    "retrieved,relevant",
    [
        (["c1", "c2"], ["c1", "c2"]),          # everything found
        (["c3", "c1"], ["c1", "c2"]),          # half found
        (["c3"], ["c1", "c2"]),                # nothing found
        (["c1", "c2", "c3"], ["c1"]),          # extra noise retrieved
        (["c1", "c1", "c2"], ["c1", "c2"]),    # duplicate retrieval
        (["c5", "c1"], ["c5"]),                # single relevant, found late
    ],
)
def test_recall_matches_ragas(retrieved, relevant):
    """Our recall@k and RAGAS NonLLMContextRecall are the same definition."""
    # k covers everything retrieved, since RAGAS does not truncate.
    ours = recall_at_k(retrieved, relevant, k=len(retrieved))
    theirs = _ragas_score(ragas_metrics.NonLLMContextRecall(), retrieved, relevant)
    assert ours == pytest.approx(theirs)


def test_precision_deliberately_diverges_from_ragas_on_rank():
    """RAGAS context precision is rank-aware; ours is not. That is a choice.

    RAGAS computes average precision, which rewards putting the relevant chunk
    first. `precision_at_k` answers a different question -- what fraction of what
    you retrieved was useful -- and is order-independent by design. MRR and NDCG
    are where we account for rank.

    This test exists so nobody "fixes" our precision to match RAGAS without
    first deciding they want a different metric.
    """
    metric = ragas_metrics.NonLLMContextPrecisionWithReference()
    relevant = ["c1"]

    # Relevant chunk first: RAGAS rewards the ranking, we do not.
    assert precision_at_k(["c1", "c3"], relevant, k=2) == 0.5
    assert _ragas_score(metric, ["c1", "c3"], relevant) == pytest.approx(1.0)

    # Same two chunks, relevant one last: now the two agree.
    assert precision_at_k(["c3", "c1"], relevant, k=2) == 0.5
    assert _ragas_score(metric, ["c3", "c1"], relevant) == pytest.approx(0.5)


def test_precision_agrees_when_rank_cannot_matter():
    """With every retrieved chunk relevant, or none, ordering is irrelevant."""
    metric = ragas_metrics.NonLLMContextPrecisionWithReference()

    assert precision_at_k(["c1", "c2"], ["c1", "c2"], k=2) == 1.0
    assert _ragas_score(metric, ["c1", "c2"], ["c1", "c2"]) == pytest.approx(1.0)

    assert precision_at_k(["c3"], ["c1"], k=1) == 0.0
    assert _ragas_score(metric, ["c3"], ["c1"]) == pytest.approx(0.0)
