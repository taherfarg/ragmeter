from pathlib import Path

import pytest

from ragmeter.db import GoldenItem, Run, Trace, init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with make_session(engine)() as s:
        yield s


def test_load_golden(session):
    n = load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    session.commit()
    assert n == 5
    assert session.get(GoldenItem, ("docs", "v1", "q1")).relevant_chunk_ids == ["c1", "c2"]


def test_load_golden_is_idempotent(session):
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    session.commit()
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    session.commit()
    assert session.query(GoldenItem).count() == 5


def test_load_golden_rejects_empty_relevant_ids(session, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- question_id: q1\n  question: why?\n  relevant_chunk_ids: []\n")
    with pytest.raises(ValueError, match="relevant_chunk_ids must not be empty"):
        load_golden(session, bad, dataset="docs", version="v1")


def test_load_traces(session):
    run = get_or_create_run(session, "baseline")
    result = load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    assert result == {"ingested": 6, "skipped": 0}
    assert session.query(Trace).count() == 6
    assert session.get(Trace, "t6").question_id is None


def test_load_traces_is_idempotent_on_trace_id(session):
    run = get_or_create_run(session, "baseline")
    load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    result = load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    # Re-ingesting is a no-op, not an error: reruns of a CI job must be safe.
    assert result == {"ingested": 0, "skipped": 6}
    assert session.query(Trace).count() == 6


def test_load_traces_reports_bad_line_number(session, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"trace_id": "ok", "question": "fine"}\n{"question": "no id"}\n')
    run = get_or_create_run(session, "r")
    with pytest.raises(ValueError, match="line 2"):
        load_traces(session, bad, run)


def test_get_or_create_run_reuses_by_name(session):
    a = get_or_create_run(session, "same")
    session.commit()
    b = get_or_create_run(session, "same")
    assert a.run_id == b.run_id
    assert session.query(Run).count() == 1
