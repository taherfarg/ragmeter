import pytest

from example_rag.chunking import STRATEGIES, chunk_document

TEXT = (
    "Alpha one two. Beta three four. Gamma five six.\n\n"
    "Delta seven eight. Epsilon nine ten. Zeta eleven twelve.\n\n"
    "Eta thirteen. Theta fourteen."
)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_covers_the_text_without_gaps(name):
    # A gap means an answer could fall between chunks and be unreachable,
    # which would look like a retrieval failure that never happened.
    chunks = chunk_document("d1", TEXT, name)
    assert chunks
    assert chunks[0].start == 0
    assert chunks[-1].end == len(TEXT)
    for previous, following in zip(chunks, chunks[1:]):
        assert following.start <= previous.end, "gap between chunks"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_chunk_text_matches_its_offsets(name):
    for chunk in chunk_document("d1", TEXT, name):
        assert chunk.text == TEXT[chunk.start:chunk.end]


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_chunk_ids_are_unique_and_carry_the_document(name):
    chunks = chunk_document("d1", TEXT, name)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id.startswith("d1:") for c in chunks)


def test_fixed_respects_its_size():
    chunks = chunk_document("d1", "x" * 250, "fixed-100")
    assert [c.end - c.start for c in chunks] == [100, 100, 50]


def test_overlap_strategy_actually_overlaps():
    chunks = chunk_document("d1", "x" * 250, "fixed-100-overlap-50")
    assert chunks[1].start < chunks[0].end
    assert chunks[1].start == 50


def test_paragraph_strategy_splits_on_blank_lines():
    chunks = chunk_document("d1", TEXT, "paragraph")
    assert len(chunks) == 3
    assert chunks[0].text.startswith("Alpha")
    assert chunks[2].text.startswith("Eta")


def test_sentence_strategy_groups_sentences():
    chunks = chunk_document("d1", TEXT, "sentence-2")
    # 8 sentences grouped two at a time.
    assert len(chunks) == 4


def test_lexical_cohesion_keeps_related_sentences_together():
    # Two sentences sharing vocabulary, then an unrelated one.
    text = "Cats like fish. Cats eat fish daily. Volcanoes erupt lava."
    chunks = chunk_document("d1", text, "lexical-cohesion")
    assert len(chunks) == 2
    assert "Cats eat fish daily." in chunks[0].text
    assert "Volcanoes" in chunks[1].text


def test_empty_text_yields_no_chunks():
    assert chunk_document("d1", "", "paragraph") == []


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_document("d1", TEXT, "nope")
