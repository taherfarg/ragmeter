"""BM25 ranking. Pure Python, no dependencies.

Chunking is the variable under test, so the retriever is deliberately the
boring, deterministic option: no model weights, no randomness, no GPU, and a
rerun always produces the identical ranking.
"""

import math
import re
from collections import Counter

__all__ = ["tokenize", "BM25"]

_WORD = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    def __init__(self, documents: dict[str, str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = list(documents)
        self.tokens = {i: Counter(tokenize(t)) for i, t in documents.items()}
        self.lengths = {i: sum(c.values()) for i, c in self.tokens.items()}
        self.average_length = (
            sum(self.lengths.values()) / len(self.lengths) if self.lengths else 0.0
        )

        self.document_frequency: Counter = Counter()
        self.postings: dict[str, list[str]] = {}
        for doc_id, counts in self.tokens.items():
            self.document_frequency.update(counts.keys())
            for term in counts:
                self.postings.setdefault(term, []).append(doc_id)

        # Precomputed so tie-breaking is O(1); list.index() inside a sort key
        # would be O(n) per comparison.
        self.position = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.document_frequency.get(term, 0)
        # Standard BM25 idf with the +0.5 smoothing that keeps it positive.
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, doc_id: str) -> float:
        counts = self.tokens[doc_id]
        length = self.lengths[doc_id]
        total = 0.0
        for term in tokenize(query):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / (self.average_length or 1.0)
            )
            total += self._idf(term) * frequency * (self.k1 + 1) / denominator
        return total

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not self.doc_ids or not terms:
            return []

        # Only documents containing at least one query term can score above
        # zero, so scoring the whole corpus is wasted work.
        candidates: set[str] = set()
        for term in terms:
            candidates.update(self.postings.get(term, ()))
        if not candidates:
            return []

        scored = [(i, self.score(query, i)) for i in candidates]
        # Insertion order breaks ties, so a rerun cannot look like a regression.
        ranked = sorted(
            [(i, s) for i, s in scored if s > 0],
            key=lambda pair: (-pair[1], self.position[pair[0]]),
        )
        return ranked[:k]
