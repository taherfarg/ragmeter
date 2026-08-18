import pytest

from example_rag.bm25 import BM25, tokenize

DOCS = {
    "a": "the cat sat on the mat",
    "b": "the dog sat on the log",
    "c": "volcanoes erupt molten lava",
}


@pytest.fixture()
def index():
    return BM25(DOCS)


def test_tokenize_lowercases_and_splits_on_non_words():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_exact_term_ranks_first(index):
    assert [doc for doc, _ in index.search("volcanoes", k=1)] == ["c"]


def test_search_returns_at_most_k(index):
    assert len(index.search("the sat", k=2)) == 2


def test_distinguishing_term_beats_a_common_one(index):
    # "cat" appears once, "the" appears everywhere. IDF must prefer "cat".
    assert index.search("cat", k=1)[0][0] == "a"


def test_unmatched_query_returns_nothing(index):
    assert index.search("helicopter", k=3) == []


def test_empty_query_returns_nothing(index):
    assert index.search("", k=3) == []


def test_results_are_sorted_by_score_descending(index):
    scores = [score for _, score in index.search("sat log", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_deterministically():
    # Two identical documents must not swap order between runs, or a rerun
    # would look like a retrieval regression.
    index = BM25({"x": "same words here", "y": "same words here"})
    assert [d for d, _ in index.search("same words", k=2)] == ["x", "y"]


def test_empty_index_is_safe():
    assert BM25({}).search("anything", k=3) == []
