"""Chunking strategies. Pure functions over text, no I/O.

Every strategy returns chunks that tile the document without gaps, because a
gap could swallow an answer span and show up as a retrieval failure that never
happened.

chunk_id is `{doc_id}:{start}-{end}` -- position-derived, so it is stable for a
given strategy, debuggable by eye, and guaranteed different across strategies
that cut in different places. That last part is fine: the golden set is derived
per strategy from answer offsets, never carried between them.
"""

import re
from dataclasses import dataclass

__all__ = ["Chunk", "STRATEGIES", "chunk_document"]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    start: int
    end: int


def _make(doc_id: str, text: str, start: int, end: int) -> Chunk:
    return Chunk(f"{doc_id}:{start}-{end}", text[start:end], start, end)


def _fixed(doc_id: str, text: str, size: int, overlap: int = 0) -> list[Chunk]:
    # max(1, ...) so an overlap >= size can never freeze the loop.
    step = max(1, size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(_make(doc_id, text, start, end))
        if end == len(text):
            break
        start += step
    return chunks


def _spans_between(boundaries: list[int], length: int) -> list[tuple[int, int]]:
    """Turn split points into contiguous [start, end) spans covering the text."""
    edges = [0, *boundaries, length]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    boundaries = [m.end() for m in _SENTENCE_END.finditer(text)]
    return _spans_between(boundaries, len(text))


def _paragraphs(doc_id: str, text: str) -> list[Chunk]:
    boundaries = [m.end() for m in _PARAGRAPH_BREAK.finditer(text)]
    return [_make(doc_id, text, a, b) for a, b in _spans_between(boundaries, len(text))]


def _sentences(doc_id: str, text: str, per_chunk: int) -> list[Chunk]:
    spans = _sentence_spans(text)
    chunks = []
    for i in range(0, len(spans), per_chunk):
        group = spans[i:i + per_chunk]
        chunks.append(_make(doc_id, text, group[0][0], group[-1][1]))
    return chunks


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _lexical_cohesion(doc_id: str, text: str, threshold: float = 0.15) -> list[Chunk]:
    """Start a new chunk where adjacent sentences stop sharing vocabulary.

    An embedding-free approximation of semantic chunking: Jaccard overlap of
    adjacent sentence token sets. Named for what it measures, not for what it
    approximates.
    """
    spans = _sentence_spans(text)
    if not spans:
        return []

    chunks = []
    group_start = spans[0][0]
    previous = _tokens(text[spans[0][0]:spans[0][1]])

    for start, end in spans[1:]:
        current = _tokens(text[start:end])
        union = previous | current
        similarity = len(previous & current) / len(union) if union else 0.0
        if similarity < threshold:
            chunks.append(_make(doc_id, text, group_start, start))
            group_start = start
        previous = current

    chunks.append(_make(doc_id, text, group_start, spans[-1][1]))
    return chunks


STRATEGIES = {
    "fixed-100": lambda d, t: _fixed(d, t, 100),
    "fixed-100-overlap-50": lambda d, t: _fixed(d, t, 100, 50),
    "fixed-400": lambda d, t: _fixed(d, t, 400),
    "fixed-400-overlap-100": lambda d, t: _fixed(d, t, 400, 100),
    "paragraph": _paragraphs,
    "sentence-2": lambda d, t: _sentences(d, t, 2),
    "sentence-4": lambda d, t: _sentences(d, t, 4),
    "lexical-cohesion": _lexical_cohesion,
}


def chunk_document(doc_id: str, text: str, strategy: str) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; have {sorted(STRATEGIES)}")
    if not text:
        return []
    return STRATEGIES[strategy](doc_id, text)
