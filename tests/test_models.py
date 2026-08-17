import pytest
from pydantic import ValidationError

from ragmeter.models import Chunk, GoldenItemIn, TraceIn


def test_trace_minimal_fields():
    t = TraceIn(trace_id="t1", question="why?")
    assert t.trace_id == "t1"
    assert t.retrieved == []
    assert t.answer == ""
    assert t.question_id is None
    assert t.metadata == {}


def test_trace_with_chunks():
    t = TraceIn(
        trace_id="t1",
        question_id="q1",
        question="why?",
        retrieved=[{"chunk_id": "c1", "text": "because", "score": 0.9, "rank": 1}],
        answer="because",
        model="openai/gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=350,
    )
    assert t.retrieved[0] == Chunk(chunk_id="c1", text="because", score=0.9, rank=1)
    assert t.chunk_ids() == ["c1"]


def test_trace_requires_id_and_question():
    with pytest.raises(ValidationError):
        TraceIn(question="why?")
    with pytest.raises(ValidationError):
        TraceIn(trace_id="t1")


def test_golden_item_valid():
    g = GoldenItemIn(question_id="q1", question="why?", relevant_chunk_ids=["c1", "c2"])
    assert g.relevant_chunk_ids == ["c1", "c2"]
    assert g.reference_answer is None


def test_golden_item_rejects_empty_relevant_ids():
    # An unlabeled golden item cannot produce retrieval metrics. Rejecting it at
    # load time beats discovering a run of silent Nones after evaluation.
    with pytest.raises(ValidationError, match="relevant_chunk_ids must not be empty"):
        GoldenItemIn(question_id="q1", question="why?", relevant_chunk_ids=[])


def test_golden_item_deduplicates_relevant_ids():
    g = GoldenItemIn(question_id="q1", question="why?", relevant_chunk_ids=["c1", "c1", "c2"])
    assert g.relevant_chunk_ids == ["c1", "c2"]
