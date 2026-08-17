from pathlib import Path

import pytest

from ragmeter.db import Evaluation, Run, Trace, init_db, make_engine, make_session
from ragmeter.gate.collect import collect_run_metrics
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


@pytest.fixture()
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    s = make_session(engine)()
    load_golden(s, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    for name, path in (("baseline", "traces.jsonl"), ("candidate", "traces_v2.jsonl")):
        run = get_or_create_run(s, name)
        load_traces(s, FIXTURES / path, run)
    s.commit()
    evaluate_run(s, "baseline", "docs", "v1", k=3, prices=PRICES)
    evaluate_run(s, "candidate", "docs", "v1", k=3, prices=PRICES)
    yield s
    s.close()


def test_by_question_holds_labeled_traces_only(session):
    rm = collect_run_metrics(session, "baseline", k=3)
    assert sorted(rm.by_question) == ["q1", "q2", "q3", "q4", "q5"]
    assert rm.by_question["q1"]["recall@3"] == 1.0


def test_all_values_includes_unlabeled_traces(session):
    rm = collect_run_metrics(session, "baseline", k=3)
    # by_question drops the trace with no question_id; all_values keeps it,
    # because cost and latency need no ground truth.
    assert len(rm.all_values["latency_ms"]) == 6
    assert len(rm.by_question) == 5
    assert rm.n_traces == 6


def test_candidate_recall_matches_hand_computed_values(session):
    rm = collect_run_metrics(session, "candidate", k=3)
    assert rm.by_question["q1"]["recall@3"] == 0.0
    assert rm.by_question["q2"]["recall@3"] == 0.0
    assert rm.by_question["q3"]["recall@3"] == 1.0
    assert rm.by_question["q4"]["recall@3"] == 1.0
    assert rm.by_question["q5"]["recall@3"] == 0.0


def test_judge_failures_are_counted(session):
    ev = session.query(Evaluation).filter_by(trace_id="t1", k=3).one()
    ev.judge_status = "failed"
    session.commit()
    assert collect_run_metrics(session, "baseline", k=3).judge_failures == 1


def test_repeated_question_ids_are_averaged(session):
    # Running the same golden question twice in one run is legitimate; averaging
    # keeps a single question from counting twice in the pairing.
    baseline = session.query(Run).filter_by(name="baseline").one()
    session.add(Trace(trace_id="t1b", run_id=baseline.run_id, question_id="q1",
                      question="dup", retrieved=[], answer=""))
    session.flush()
    session.add(Evaluation(trace_id="t1b", k=3, metrics={"recall@3": 0.0}))
    session.commit()

    rm = collect_run_metrics(session, "baseline", k=3)
    # t1 measured 1.0, t1b measured 0.0 for the same question.
    assert rm.by_question["q1"]["recall@3"] == 0.5


def test_unknown_run_raises(session):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        collect_run_metrics(session, "nope", k=3)


def test_run_without_evaluations_raises(session):
    get_or_create_run(session, "empty")
    session.commit()
    with pytest.raises(ValueError, match="no evaluations"):
        collect_run_metrics(session, "empty", k=3)


def test_wrong_k_raises(session):
    with pytest.raises(ValueError, match="no evaluations"):
        collect_run_metrics(session, "baseline", k=99)
