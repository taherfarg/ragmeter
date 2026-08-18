from pathlib import Path

import pytest

from ragmeter.calibration import collect_labeled_pairs, unlabeled_traces
from ragmeter.db import Evaluation, HumanLabel, init_db, make_engine, make_session
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
    run = get_or_create_run(s, "baseline")
    load_traces(s, FIXTURES / "traces.jsonl", run)
    s.commit()
    evaluate_run(s, "baseline", "docs", "v1", k=3, prices=PRICES)
    # Give three traces a judge faithfulness score; leave the rest unmeasured.
    for trace_id, score in (("t1", 1.0), ("t2", 0.5), ("t3", 0.0)):
        ev = s.query(Evaluation).filter_by(trace_id=trace_id, k=3).one()
        ev.metrics = dict(ev.metrics, faithfulness=score)
    s.commit()
    yield s
    s.close()


def test_unlabeled_lists_only_traces_with_a_judge_score(session):
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="me", limit=10)
    assert [t.trace_id for t, _ in pending] == ["t1", "t2", "t3"]


def test_unlabeled_respects_the_limit(session):
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="me", limit=2)
    assert len(pending) == 2


def test_unlabeled_excludes_what_this_person_already_labelled(session):
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=1.0, labeler="me"))
    session.commit()
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="me", limit=10)
    assert [t.trace_id for t, _ in pending] == ["t2", "t3"]


def test_unlabeled_is_per_labeler(session):
    # Two people labelling the same trace is the point of inter-rater work,
    # so one person's label must not hide the trace from another.
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=1.0, labeler="me"))
    session.commit()
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="someone-else", limit=10)
    assert [t.trace_id for t, _ in pending] == ["t1", "t2", "t3"]


def test_collect_pairs_matches_judge_scores_to_labels(session):
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=1.0, labeler="me"))
    session.add(HumanLabel(trace_id="t3", metric="faithfulness", value=0.0, labeler="me"))
    session.commit()
    pairs = collect_labeled_pairs(session, "baseline", "faithfulness", k=3)
    assert sorted(pairs) == [(0.0, 0.0), (1.0, 1.0)]


def test_collect_pairs_skips_labels_without_a_judge_score(session):
    # t4 has no faithfulness score; a human label on it cannot be compared.
    session.add(HumanLabel(trace_id="t4", metric="faithfulness", value=1.0, labeler="me"))
    session.commit()
    assert collect_labeled_pairs(session, "baseline", "faithfulness", k=3) == []


def test_collect_pairs_ignores_other_metrics(session):
    session.add(HumanLabel(trace_id="t1", metric="answer_relevance",
                           value=1.0, labeler="me"))
    session.commit()
    assert collect_labeled_pairs(session, "baseline", "faithfulness", k=3) == []


def test_unknown_run_raises(session):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        collect_labeled_pairs(session, "nope", "faithfulness", k=3)
    with pytest.raises(ValueError, match="no run named 'nope'"):
        unlabeled_traces(session, "nope", "faithfulness", k=3, labeler="me", limit=5)
