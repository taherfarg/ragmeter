from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ragmeter.cli import app
from ragmeter.db import Evaluation, HumanLabel, init_db, make_engine, make_session
from ragmeter.metrics.cost import MODELS_URL

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = {"data": [{"id": "openai/gpt-4o-mini",
                     "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]}

runner = CliRunner()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'cal.db'}"
    monkeypatch.setenv("RAGMETER_DB_URL", url)
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                            "--version", "v1", "--k", "3"])

    session = make_session(make_engine(url))()
    for trace_id, score in (("t1", 1.0), ("t2", 1.0), ("t3", 0.0)):
        ev = session.query(Evaluation).filter_by(trace_id=trace_id, k=3).one()
        ev.metrics = dict(ev.metrics, faithfulness=score)
    session.commit()
    session.close()
    return url


def session_for(url):
    engine = make_engine(url)
    init_db(engine)
    return make_session(engine)()


def test_label_stores_answers_in_order(db):
    result = runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                                 "--k", "3", "--labeler", "me"], input="y\nn\ny\n")
    assert result.exit_code == 0, result.output

    session = session_for(db)
    labels = {row.trace_id: row.value
              for row in session.query(HumanLabel).filter_by(labeler="me")}
    assert labels == {"t1": 1.0, "t2": 0.0, "t3": 1.0}
    session.close()


def test_label_hides_the_judge_score_by_default(db):
    result = runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                                 "--k", "3", "--limit", "1"], input="y\n")
    # Showing the judge's verdict first would anchor the human and make the
    # agreement number circular.
    assert "judge faithfulness" not in result.output.lower()


def test_label_can_reveal_the_judge_score_on_request(db):
    result = runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                                 "--k", "3", "--limit", "1", "--show-judge"], input="y\n")
    assert "judge faithfulness" in result.output.lower()


def test_label_skip_stores_nothing(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="s\ns\ns\n")
    session = session_for(db)
    assert session.query(HumanLabel).count() == 0
    session.close()


def test_label_quit_stops_early(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="y\nq\n")
    session = session_for(db)
    assert session.query(HumanLabel).count() == 1
    session.close()


def test_label_is_resumable(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me", "--limit", "1"], input="y\n")
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me", "--limit", "1"], input="n\n")
    session = session_for(db)
    labels = {row.trace_id: row.value for row in session.query(HumanLabel)}
    # The second pass moves on to the next unlabelled trace rather than repeating.
    assert labels == {"t1": 1.0, "t2": 0.0}
    session.close()


def test_label_reports_when_nothing_is_pending(db):
    result = runner.invoke(app, ["label", "--run", "baseline",
                                 "--metric", "answer_relevance", "--k", "3"])
    assert result.exit_code == 0
    assert "nothing to label" in result.output.lower()


def test_calibration_reports_agreement_and_kappa(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="y\ny\nn\n")
    result = runner.invoke(app, ["calibration", "--run", "baseline",
                                 "--metric", "faithfulness", "--k", "3"])
    assert result.exit_code == 0, result.output
    # judge 1,1,0 vs human 1,1,0 -> perfect agreement, kappa 1.0
    assert "1.0000" in result.output
    assert "kappa" in result.output.lower()


def test_calibration_warns_when_kappa_lags_agreement(db):
    # judge says yes to t1 and t2, no to t3; the human says yes to all three.
    # a=2 b=0 c=1 d=0, po=2/3; P(j=1)=2/3, P(h=1)=1 -> pe=2/3; kappa=0.0
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="y\ny\ny\n")
    result = runner.invoke(app, ["calibration", "--run", "baseline",
                                 "--metric", "faithfulness", "--k", "3"])
    assert "0.6667" in result.output
    assert "chance" in result.output.lower()


def test_calibration_without_labels_exits_two(db):
    result = runner.invoke(app, ["calibration", "--run", "baseline",
                                 "--metric", "faithfulness", "--k", "3"])
    assert result.exit_code == 2
    assert "no labelled pairs" in result.output.lower()


def test_calibration_unknown_run_exits_two(db):
    result = runner.invoke(app, ["calibration", "--run", "nope",
                                 "--metric", "faithfulness", "--k", "3"])
    assert result.exit_code == 2
