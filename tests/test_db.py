import pytest
from sqlalchemy.exc import IntegrityError

from ragmeter.db import Evaluation, GoldenItem, Run, Trace, init_db, make_engine, make_session


@pytest.fixture()
def session():
    engine = make_engine("sqlite://")  # in-memory
    init_db(engine)
    with make_session(engine)() as s:
        yield s


def test_run_and_trace_roundtrip(session):
    run = Run(name="semantic-v2", git_sha="abc123", config={"chunker": "semantic"})
    session.add(run)
    session.flush()
    session.add(Trace(
        trace_id="t1", run_id=run.run_id, question_id="q1", question="why?",
        retrieved=[{"chunk_id": "c1", "rank": 1}], answer="because",
        model="openai/gpt-4o-mini", prompt_tokens=10, completion_tokens=2,
        latency_ms=120, meta={"env": "dev"},
    ))
    session.commit()

    loaded = session.get(Trace, "t1")
    assert loaded.question_id == "q1"
    assert loaded.retrieved == [{"chunk_id": "c1", "rank": 1}]
    assert loaded.meta == {"env": "dev"}
    assert loaded.run_id == run.run_id


def test_run_name_is_unique(session):
    session.add(Run(name="dup"))
    session.commit()
    session.add(Run(name="dup"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_golden_item_composite_key(session):
    # Same question_id in two dataset versions must coexist -- that is how you
    # relabel a golden set without destroying the old one the baseline used.
    session.add(GoldenItem(dataset="docs", version="v1", question_id="q1",
                           question="why?", relevant_chunk_ids=["c1"]))
    session.add(GoldenItem(dataset="docs", version="v2", question_id="q1",
                           question="why?", relevant_chunk_ids=["c9"]))
    session.commit()
    assert session.get(GoldenItem, ("docs", "v1", "q1")).relevant_chunk_ids == ["c1"]
    assert session.get(GoldenItem, ("docs", "v2", "q1")).relevant_chunk_ids == ["c9"]


def test_evaluation_defaults_to_skipped_judge(session):
    run = Run(name="r")
    session.add(run)
    session.flush()
    session.add(Trace(trace_id="t1", run_id=run.run_id, question="why?"))
    session.flush()
    ev = Evaluation(trace_id="t1", k=5, metrics={"recall@5": 1.0})
    session.add(ev)
    session.commit()
    assert ev.judge_status == "skipped"
    assert ev.evaluation_id is not None
