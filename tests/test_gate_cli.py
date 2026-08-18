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
def evaluated(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGMETER_DB_URL", f"sqlite:///{tmp_path / 'gate.db'}")
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    for name, path in (("baseline", "traces.jsonl"), ("candidate", "traces_v2.jsonl")):
        runner.invoke(app, ["ingest", str(FIXTURES / path), "--run", name])
        with respx.mock:
            respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
            runner.invoke(app, ["eval", "--run", name, "--dataset", "docs",
                                "--version", "v1", "--k", "3"])
    return tmp_path


def gate_file(tmp_path, text):
    path = tmp_path / "g.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_gate_fails_on_the_known_regression(evaluated):
    config = gate_file(evaluated, "min_samples: 3\nmetrics:\n  recall@3:\n    max_drop: 0.02\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    # Exit 1 means "worse", distinct from exit 2 which means "broken".
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    assert "recall@3" in result.output


def test_gate_passes_with_a_loose_threshold(evaluated):
    config = gate_file(evaluated, "min_samples: 3\nmetrics:\n  recall@3:\n    max_drop: 0.5\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_gate_fails_on_doubled_cost(evaluated):
    # traces_v2 uses exactly double the tokens, so the mean cost rises 100%.
    config = gate_file(evaluated,
                       "metrics:\n  cost_usd:\n    stat: mean\n    max_increase_pct: 20\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 1, result.output
    assert "100" in result.output


def test_gate_reports_min_samples_shortfall(evaluated):
    config = gate_file(evaluated, "min_samples: 99\nmetrics:\n  recall@3:\n    max_drop: 0.9\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 1, result.output
    assert "min_samples" in result.output


def test_bad_config_exits_two_not_one(evaluated):
    config = gate_file(evaluated, "metrics: {}\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    # A broken config is not a regression. CI must be able to tell them apart.
    assert result.exit_code == 2, result.output


def test_unknown_run_exits_two(evaluated):
    config = gate_file(evaluated, "metrics:\n  recall@3:\n    max_drop: 0.5\n")
    result = runner.invoke(app, ["gate", "--run", "nope", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 2, result.output
    assert "no run named" in result.output


def test_gate_against_itself_always_passes(evaluated):
    config = gate_file(evaluated, "min_samples: 3\nmetrics:\n  recall@3:\n    max_drop: 0.0\n")
    result = runner.invoke(app, ["gate", "--run", "baseline", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 0, result.output


def test_compare_shows_the_diff_without_failing(evaluated):
    result = runner.invoke(app, ["compare", "--run", "candidate",
                                 "--baseline", "baseline", "--k", "3"])
    # compare reports; it never blocks. That is what gate is for.
    assert result.exit_code == 0, result.output
    assert "recall@3" in result.output


def test_export_writes_a_snapshot(evaluated):
    out = evaluated / "baseline.json"
    result = runner.invoke(app, ["export", "--run", "baseline", "--k", "3",
                                 "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()

    from ragmeter.gate.snapshot import load_snapshot
    snapshot = load_snapshot(out)
    assert snapshot.by_question["q1"]["recall@3"] == 1.0


def test_export_unknown_run_exits_two(evaluated):
    result = runner.invoke(app, ["export", "--run", "nope", "--k", "3",
                                 "--out", str(evaluated / "x.json")])
    assert result.exit_code == 2


def test_gate_against_a_snapshot_matches_the_database(evaluated):
    out = evaluated / "baseline.json"
    runner.invoke(app, ["export", "--run", "baseline", "--k", "3", "--out", str(out)])
    config = gate_file(evaluated, "min_samples: 3\nmetrics:\n  recall@3:\n    max_drop: 0.02\n")

    from_db = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                  "--config", config, "--k", "3"])
    from_file = runner.invoke(app, ["gate", "--run", "candidate",
                                    "--baseline-file", str(out),
                                    "--config", config, "--k", "3"])
    # A snapshot must be indistinguishable from the live run it came from.
    assert from_db.exit_code == from_file.exit_code == 1
    assert "-0.2000" in from_file.output


def test_gate_needs_exactly_one_baseline_source(evaluated):
    config = gate_file(evaluated, "metrics:\n  recall@3:\n    max_drop: 0.5\n")
    neither = runner.invoke(app, ["gate", "--run", "candidate",
                                  "--config", config, "--k", "3"])
    assert neither.exit_code == 2
    assert "--baseline" in neither.output

    both = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                               "--baseline-file", str(evaluated / "b.json"),
                               "--config", config, "--k", "3"])
    assert both.exit_code == 2


def test_missing_snapshot_exits_two_and_says_how_to_make_one(evaluated):
    config = gate_file(evaluated, "metrics:\n  recall@3:\n    max_drop: 0.5\n")
    result = runner.invoke(app, ["gate", "--run", "candidate",
                                 "--baseline-file", str(evaluated / "absent.json"),
                                 "--config", config, "--k", "3"])
    # Exit 2, not 1: a missing baseline is a data problem, not a regression.
    # The first CI run must not look like the model got worse.
    assert result.exit_code == 2
    assert "ragmeter export" in result.output
