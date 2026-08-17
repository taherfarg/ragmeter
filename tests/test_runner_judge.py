from pathlib import Path

import pytest

from ragmeter.db import Evaluation, init_db, make_engine, make_session
from ragmeter.judge.client import JudgeError
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


class FakeJudge:
    """Returns a canned response shaped by which prompt it was handed."""

    model = "fake/judge"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def ask(self, prompt):
        self.calls += 1
        if self.fail:
            raise JudgeError("rate limited")
        if "atomic factual claims" in prompt:
            return {"claims": [{"claim": "x", "supported": True, "chunk_ids": ["c1"]},
                               {"claim": "y", "supported": False, "chunk_ids": []}]}
        if "Rate on this scale" in prompt:
            return {"score": 5, "reason": "direct"}
        return {"judgments": [{"chunk_id": "c1", "relevant": True}]}


@pytest.fixture()
def loaded():
    engine = make_engine("sqlite://")
    init_db(engine)
    session = make_session(engine)()
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    run = get_or_create_run(session, "baseline")
    load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    yield session
    session.close()


def test_no_judge_leaves_status_skipped(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.judge_status == "skipped"
    assert "faithfulness" not in ev.metrics


def test_judge_adds_metrics_and_claims(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=FakeJudge())
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.judge_status == "ok"
    assert ev.metrics["faithfulness"] == 0.5
    assert ev.metrics["answer_relevance"] == 1.0
    assert len(ev.claims) == 2
    assert ev.judge_model == "fake/judge"


def test_trace_without_chunks_gets_none_faithfulness(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=FakeJudge())
    ev = loaded.query(Evaluation).filter_by(trace_id="t4").one()
    assert ev.metrics["faithfulness"] is None
    # Still 'ok': the judge was asked nothing because there was nothing to ask.
    assert ev.judge_status == "ok"


def test_unmatched_trace_gets_chunk_relevance_precision(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=FakeJudge())
    ev = loaded.query(Evaluation).filter_by(trace_id="t6").one()
    # t6 has no golden item, so precision comes from the judge instead.
    assert ev.metrics["precision@3"] == 1.0
    assert ev.chunk_judgments == [{"chunk_id": "c1", "relevant": True}]
    # Recall stays unmeasurable: nothing can see what was never retrieved.
    assert ev.metrics["recall@3"] is None


def test_matched_trace_does_not_call_chunk_relevance(loaded):
    judge = FakeJudge()
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=judge)
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    # Golden labels beat a judge's opinion, so precision@3 stays the computed 2/3.
    assert ev.metrics["precision@3"] == pytest.approx(2 / 3)
    assert ev.chunk_judgments is None


def test_judge_failure_is_recorded_not_swallowed(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES,
                 judge=FakeJudge(fail=True))
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.judge_status == "failed"
    assert "rate limited" in ev.judge_error
    # No number is invented for a metric that was never measured.
    assert ev.metrics["faithfulness"] is None
    assert ev.metrics["answer_relevance"] is None


def test_judge_failure_does_not_stop_retrieval_metrics(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES,
                 judge=FakeJudge(fail=True))
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.metrics["recall@3"] == 1.0
