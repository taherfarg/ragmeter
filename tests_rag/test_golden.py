from example_rag.chunking import Chunk
from example_rag.golden import relevant_chunk_ids

CHUNKS = [
    Chunk("d:0-10", "0123456789", 0, 10),
    Chunk("d:10-20", "abcdefghij", 10, 20),
    Chunk("d:20-30", "ABCDEFGHIJ", 20, 30),
]


def test_span_inside_one_chunk():
    assert relevant_chunk_ids([(12, 15)], CHUNKS) == ["d:10-20"]


def test_span_crossing_a_boundary_marks_both():
    # An answer split across a boundary is genuinely in both chunks; retrieving
    # either one gives the reader part of the answer.
    assert relevant_chunk_ids([(8, 12)], CHUNKS) == ["d:0-10", "d:10-20"]


def test_multiple_spans_union_their_chunks():
    assert relevant_chunk_ids([(2, 4), (22, 24)], CHUNKS) == ["d:0-10", "d:20-30"]


def test_result_is_deduplicated_and_ordered():
    assert relevant_chunk_ids([(1, 2), (3, 4)], CHUNKS) == ["d:0-10"]


def test_touching_the_boundary_exactly_does_not_include_the_next_chunk():
    # A span ending exactly at 10 lies wholly in the first chunk. Half-open
    # intervals: [0,10) and [10,20) do not overlap.
    assert relevant_chunk_ids([(5, 10)], CHUNKS) == ["d:0-10"]


def test_zero_length_span_is_ignored():
    assert relevant_chunk_ids([(10, 10)], CHUNKS) == []


def test_no_spans_yields_nothing():
    assert relevant_chunk_ids([], CHUNKS) == []


def test_overlapping_chunks_can_both_match():
    overlapping = [Chunk("d:0-10", "x", 0, 10), Chunk("d:5-15", "y", 5, 15)]
    assert relevant_chunk_ids([(6, 8)], overlapping) == ["d:0-10", "d:5-15"]
