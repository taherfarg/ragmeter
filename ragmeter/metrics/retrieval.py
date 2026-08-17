"""Retrieval quality metrics. Pure functions: no I/O, no package imports.

None means "not measurable". 0.0 means "measured, scored zero". These are not
interchangeable: a None excluded from an average is honest, a 0.0 silently
included is a lie that makes a broken pipeline look mediocre instead of broken.
"""

from math import log2

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "evaluate_retrieval",
    "metric_names",
]


def metric_names(k: int) -> list[str]:
    """The metric keys produced at this k.

    Callers that need the key names without having data to evaluate (an
    unmatched trace, for instance) use this instead of re-deriving the naming
    convention and drifting out of sync with evaluate_retrieval.
    """
    return [f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}"]


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _top_k(retrieved: list[str], k: int) -> list[str]:
    """Deduplicate keeping the first occurrence, then truncate to k.

    A retriever that returns the same chunk three times has retrieved one chunk.
    Counting it three times inflates every metric.
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id in retrieved:
        if chunk_id not in seen:
            seen.add(chunk_id)
            out.append(chunk_id)
        if len(out) == k:
            break
    return out


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Fraction of relevant chunks that made it into the top k."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    return len(rel & set(_top_k(retrieved, k))) / len(rel)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Fraction of the top k that is relevant.

    Divides by however many were actually retrieved, not by k, so a retriever
    returning one perfect chunk scores 1.0 rather than 1/k.
    """
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    top = _top_k(retrieved, k)
    if not top:
        return None
    return len(rel & set(top)) / len(top)


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Reciprocal rank of the first relevant chunk. 0.0 if none in the top k."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    for rank, chunk_id in enumerate(_top_k(retrieved, k), start=1):
        if chunk_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Normalized discounted cumulative gain with binary relevance."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    top = _top_k(retrieved, k)
    dcg = sum(1.0 / log2(i + 1) for i, c in enumerate(top, start=1) if c in rel)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, min(k, len(rel)) + 1))
    if idcg == 0:
        return None
    return dcg / idcg


def evaluate_retrieval(
    retrieved: list[str], relevant: list[str], k: int
) -> dict[str, float | None]:
    """All four metrics, keyed with the k baked into the name.

    The k is in the key because comparing recall@5 against recall@10 across runs
    is a silent apples-to-oranges bug the regression gate must not be able to make.
    """
    return {
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"mrr@{k}": mrr_at_k(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }
