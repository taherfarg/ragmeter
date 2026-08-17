from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ragmeter.cli import app
from ragmeter.metrics.cost import MODELS_URL

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = {"data": [{"id": "openai/gpt-4o-mini",
                     "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]}

runner = CliRunner()


@pytest.fixture()
def db_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("RAGMETER_DB_URL", url)
    return url


def test_full_workflow(db_url):
    result = runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                                 "--name", "docs", "--version", "v1"])
    assert result.exit_code == 0, result.output
    assert "5" in result.output

    result = runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    assert result.exit_code == 0, result.output
    assert "6" in result.output

    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        result = runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                                     "--version", "v1", "--k", "3"])
    assert result.exit_code == 0, result.output
    assert "recall@3" in result.output
    assert "5/6" in result.output


def test_ingest_twice_reports_skips(db_url):
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    result = runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    assert result.exit_code == 0, result.output
    assert "skipped 6" in result.output


def test_eval_unknown_run_exits_nonzero(db_url):
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        result = runner.invoke(app, ["eval", "--run", "nope", "--dataset", "docs",
                                     "--version", "v1"])
    assert result.exit_code == 2
    assert "no run named" in result.output


def test_eval_survives_price_fetch_failure(db_url):
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(503))
        result = runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                                     "--version", "v1", "--k", "3"])
    # Retrieval quality does not depend on pricing. Losing the price catalog must
    # cost you the cost column, not the whole evaluation.
    assert result.exit_code == 0, result.output
    assert "could not fetch model prices" in result.output
    assert "recall@3" in result.output


def test_judge_without_api_key_fails_before_evaluating(db_url, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        result = runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                                     "--version", "v1", "--judge"])
    assert result.exit_code == 2
    assert "OPENROUTER_API_KEY" in result.output
