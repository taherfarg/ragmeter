import pytest

from ragmeter.judge.client import JudgeError
from ragmeter.judge.scoring import (
    score_answer_relevance,
    score_chunk_relevance,
    score_faithfulness,
)

CHUNKS = [{"chunk_id": "c1", "text": "Returns accepted within 30 days."},
          {"chunk_id": "c2", "text": "Items must be unused."}]


class FakeJudge:
    """Stands in for JudgeClient. Records prompts, returns canned JSON."""

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_faithfulness_all_supported():
    judge = FakeJudge({"claims": [
        {"claim": "a", "supported": True, "chunk_ids": ["c1"]},
        {"claim": "b", "supported": True, "chunk_ids": ["c2"]},
    ]})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    assert out["score"] == 1.0
    assert len(out["claims"]) == 2


def test_faithfulness_partial():
    judge = FakeJudge({"claims": [
        {"claim": "a", "supported": True, "chunk_ids": ["c1"]},
        {"claim": "b", "supported": False, "chunk_ids": []},
        {"claim": "c", "supported": False, "chunk_ids": []},
    ]})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    assert out["score"] == pytest.approx(1 / 3)


def test_faithfulness_with_no_chunks_is_unmeasurable():
    # There is nothing to be faithful to. Zero would read as "the model
    # hallucinated", which is a different and unproven claim.
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", [], "answer")
    assert out["score"] is None
    assert judge.prompts == []


def test_faithfulness_with_empty_answer_is_unmeasurable():
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", CHUNKS, "   ")
    assert out["score"] is None
    assert judge.prompts == []


def test_faithfulness_with_no_claims_returned_is_unmeasurable():
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    assert out["score"] is None


def test_faithfulness_rejects_missing_claims_key():
    judge = FakeJudge({"wrong": []})
    with pytest.raises(JudgeError, match="claims"):
        score_faithfulness(judge, "q", CHUNKS, "answer")


def test_faithfulness_treats_non_true_supported_as_false():
    judge = FakeJudge({"claims": [{"claim": "a", "supported": "yes"},
                                  {"claim": "b", "supported": True}]})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    # Only a literal true counts. A string "yes" is the model ignoring the
    # schema, and reading it as support would inflate the score.
    assert out["score"] == 0.5


def test_answer_relevance_normalizes_to_unit_interval():
    assert score_answer_relevance(FakeJudge({"score": 5}), "q", "a")["score"] == 1.0
    assert score_answer_relevance(FakeJudge({"score": 1}), "q", "a")["score"] == 0.0
    assert score_answer_relevance(FakeJudge({"score": 4}), "q", "a")["score"] == 0.75


def test_answer_relevance_keeps_the_reason():
    out = score_answer_relevance(FakeJudge({"score": 3, "reason": "vague"}), "q", "a")
    assert out["reason"] == "vague"


def test_answer_relevance_rejects_out_of_range():
    with pytest.raises(JudgeError, match="1..5"):
        score_answer_relevance(FakeJudge({"score": 9}), "q", "a")


def test_answer_relevance_rejects_non_numeric():
    with pytest.raises(JudgeError, match="1..5"):
        score_answer_relevance(FakeJudge({"score": "high"}), "q", "a")


def test_answer_relevance_empty_answer_is_unmeasurable():
    judge = FakeJudge({"score": 1})
    out = score_answer_relevance(judge, "q", "")
    assert out["score"] is None
    assert judge.prompts == []


def test_chunk_relevance_returns_judgments_and_precision():
    judge = FakeJudge({"judgments": [{"chunk_id": "c1", "relevant": True},
                                     {"chunk_id": "c2", "relevant": False}]})
    out = score_chunk_relevance(judge, "q", CHUNKS)
    assert out["precision"] == 0.5
    assert len(out["judgments"]) == 2


def test_chunk_relevance_ignores_ids_that_were_not_retrieved():
    # A model that invents a chunk id must not be able to change the denominator.
    judge = FakeJudge({"judgments": [{"chunk_id": "c1", "relevant": True},
                                     {"chunk_id": "ghost", "relevant": True}]})
    out = score_chunk_relevance(judge, "q", CHUNKS)
    assert out["precision"] == 0.5


def test_chunk_relevance_with_no_chunks_is_unmeasurable():
    judge = FakeJudge({"judgments": []})
    out = score_chunk_relevance(judge, "q", [])
    assert out["precision"] is None
    assert judge.prompts == []


def test_judge_errors_propagate():
    judge = FakeJudge(JudgeError("rate limited"))
    with pytest.raises(JudgeError, match="rate limited"):
        score_faithfulness(judge, "q", CHUNKS, "answer")


def test_refusal_yielding_no_claims_is_unmeasurable():
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", CHUNKS, "I don't know.")
    assert out["score"] is None
    assert "no factual claims" in out["reason"]
