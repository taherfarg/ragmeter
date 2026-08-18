"""Deriving relevant chunk ids from answer spans.

This is what makes comparing chunking strategies possible without relabelling:
the ground truth is anchored to character offsets in the source document, so it
can be recomputed for any set of chunk boundaries.
"""

from example_rag.chunking import Chunk

__all__ = ["relevant_chunk_ids"]


def relevant_chunk_ids(spans: list[tuple[int, int]], chunks: list[Chunk]) -> list[str]:
    """Chunk ids overlapping any answer span, in document order, deduplicated.

    Spans and chunks are half-open [start, end), so an answer ending exactly at
    a boundary belongs only to the chunk before it.
    """
    out: list[str] = []
    for chunk in chunks:
        for start, end in spans:
            if end <= start:
                continue  # a zero-length span points at nothing
            if start < chunk.end and end > chunk.start:
                if chunk.chunk_id not in out:
                    out.append(chunk.chunk_id)
                break
    return out
