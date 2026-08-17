from pathlib import Path

import pytest

from ragmeter.db import init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.report import render_summary, summarize_run
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


@pytest.fixture()
def evaluated():
    engine = make_engine("sqlite://")
    init_db(engine)
    session = make_session(engine)()
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    run = get_or_create_run(session, "baseline")
    load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    evaluate_run(session, "baseline", "docs", "v1", k=3, prices=PRICES)
    yield session
    session.close()


def test_summarize_run_reports_null_counts(evaluated):
    out = summarize_run(evaluated, "baseline", k=3)
    # 6 traces; only 5 have a golden match, so one recall value is unmeasurable.
    assert out["recall@3"]["n"] == 6
    assert out["recall@3"]["n_measured"] == 5
    assert out["recall@3"]["n_null"] == 1
    # t4 retrieved nothing, so precision is unmeasurable there too.
    assert out["precision@3"]["n_null"] == 2


def test_summarize_run_mean_excludes_nulls(evaluated):
    out = summarize_run(evaluated, "baseline", k=3)
    # t1=1.0, t2=1.0, t3=1.0, t4=0.0, t5=0.0 -> mean 0.6 over 5 measured values
    assert out["recall@3"]["mean"] == pytest.approx(0.6)


def test_render_summary_shows_measured_counts(evaluated):
    text = render_summary(summarize_run(evaluated, "baseline", k=3), "baseline", k=3)
    assert "baseline" in text
    assert "recall@3" in text
    # The count must be visible: a mean over 5 of 6 is a different claim than
    # a mean over 6, and the reader has to be able to see which one this is.
    assert "5/6" in text


def test_summarize_unknown_run_raises(evaluated):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        summarize_run(evaluated, "nope", k=3)
