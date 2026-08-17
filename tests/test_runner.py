from pathlib import Path

import pytest

from ragmeter.db import Evaluation, init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


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


def test_evaluate_writes_one_evaluation_per_trace(loaded):
    summary = evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    assert loaded.query(Evaluation).count() == 6
    assert summary["n_traces"] == 6
    assert summary["n_matched"] == 5
    assert summary["n_unmatched"] == 1


def test_metrics_for_a_partial_hit(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    # retrieved c1,c2,c3; relevant c1,c2 -> recall 1.0, precision 2/3, mrr 1.0
    assert ev.metrics["recall@3"] == 1.0
    assert ev.metrics["precision@3"] == pytest.approx(2 / 3)
    assert ev.metrics["mrr@3"] == 1.0
    assert ev.dataset == "docs"
    assert ev.judge_status == "skipped"


def test_duplicate_chunks_do_not_inflate(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t3").one()
    # retrieved c7,c7,c8 dedupes to c7,c8; relevant c7,c8 -> recall 1.0
    assert ev.metrics["recall@3"] == 1.0
    assert ev.metrics["precision@3"] == 1.0


def test_empty_retrieval_is_zero_recall_none_precision(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t4").one()
    assert ev.metrics["recall@3"] == 0.0
    assert ev.metrics["precision@3"] is None


def test_unmatched_trace_gets_no_retrieval_metrics(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t6").one()
    assert ev.metrics["recall@3"] is None
    assert ev.metrics["precision@3"] is None
    assert ev.dataset is None
    # Cost and latency still apply: they need no ground truth.
    assert ev.metrics["cost_usd"] == pytest.approx(500 * 0.00000015 + 25 * 0.0000006)
    assert ev.metrics["latency_ms"] == 300


def test_unpriced_model_yields_none_cost(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t5").one()
    assert ev.metrics["cost_usd"] is None


def test_reevaluating_replaces_rather_than_duplicates(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    assert loaded.query(Evaluation).filter_by(trace_id="t1").count() == 1


def test_different_k_coexist(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    evaluate_run(loaded, "baseline", "docs", "v1", k=1, prices=PRICES)
    # k=1 and k=3 are different measurements of the same trace, both worth keeping.
    assert loaded.query(Evaluation).filter_by(trace_id="t1").count() == 2


def test_unknown_run_raises(loaded):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        evaluate_run(loaded, "nope", "docs", "v1", k=3, prices=PRICES)


def test_unknown_dataset_raises(loaded):
    with pytest.raises(ValueError, match="no golden items"):
        evaluate_run(loaded, "baseline", "missing", "v1", k=3, prices=PRICES)
