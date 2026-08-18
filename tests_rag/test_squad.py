import json

import pytest

from example_rag.squad import load_squad

RAW = {
    "version": "1.1",
    "data": [
        {
            "title": "Test_Article",
            "paragraphs": [
                {
                    "context": "Alpha beta gamma.",
                    "qas": [{"id": "q1", "question": "Which letter?",
                             "answers": [{"answer_start": 6, "text": "beta"}]}],
                },
                {
                    "context": "Delta epsilon zeta.",
                    "qas": [{"id": "q2", "question": "Which second?",
                             "answers": [{"answer_start": 6, "text": "epsilon"}]}],
                },
            ],
        }
    ],
}


@pytest.fixture()
def squad_file(tmp_path):
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    return path


def test_one_document_per_article(squad_file):
    docs = load_squad(squad_file)
    assert len(docs) == 1
    assert docs[0].doc_id == "Test_Article"


def test_paragraphs_are_joined_into_one_document(squad_file):
    doc = load_squad(squad_file)[0]
    assert "Alpha beta gamma." in doc.text
    assert "Delta epsilon zeta." in doc.text


def test_answer_offsets_are_lifted_to_document_level(squad_file):
    # SQuAD offsets are relative to their paragraph. Joining paragraphs shifts
    # every answer after the first, and a stale offset would silently label the
    # wrong chunk as relevant.
    doc = load_squad(squad_file)[0]
    q2 = next(q for q in doc.questions if q.question_id == "q2")
    start, end = q2.spans[0]
    assert doc.text[start:end] == "epsilon"


def test_first_paragraph_offsets_are_unchanged(squad_file):
    doc = load_squad(squad_file)[0]
    q1 = next(q for q in doc.questions if q.question_id == "q1")
    start, end = q1.spans[0]
    assert doc.text[start:end] == "beta"


def test_duplicate_answers_are_deduplicated(tmp_path):
    raw = json.loads(json.dumps(RAW))
    raw["data"][0]["paragraphs"][0]["qas"][0]["answers"] = [
        {"answer_start": 6, "text": "beta"},
        {"answer_start": 6, "text": "beta"},
    ]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    doc = load_squad(path)[0]
    # SQuAD dev carries three human annotations; identical ones must not
    # inflate the span list.
    assert len(doc.questions[0].spans) == 1


def test_limit_articles(squad_file):
    assert load_squad(squad_file, max_articles=0) == []


def test_every_span_resolves_to_its_answer_text(squad_file):
    for doc in load_squad(squad_file):
        for question in doc.questions:
            for index, (start, end) in enumerate(question.spans):
                assert doc.text[start:end] == question.answer_texts[index]
