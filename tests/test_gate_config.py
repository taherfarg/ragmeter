from pathlib import Path

import pytest

from ragmeter.gate.config import GateConfigError, load_gate_config

FIXTURES = Path(__file__).parent / "fixtures"


def write(tmp_path, text):
    path = tmp_path / "gate.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_the_sample_config():
    cfg = load_gate_config(FIXTURES / "gate.yaml")
    assert cfg.min_samples == 3
    assert cfg.fail_on_missing is True
    assert [r.name for r in cfg.metrics] == ["recall@3", "cost_usd", "latency_ms"]


def test_paired_and_aggregate_rules_are_distinguished():
    cfg = load_gate_config(FIXTURES / "gate.yaml")
    by_name = {r.name: r for r in cfg.metrics}
    assert by_name["recall@3"].is_paired is True
    assert by_name["cost_usd"].is_paired is False
    assert by_name["cost_usd"].stat == "mean"
    assert by_name["latency_ms"].stat == "p95"


def test_defaults(tmp_path):
    cfg = load_gate_config(write(tmp_path, "metrics:\n  recall@3:\n    max_drop: 0.1\n"))
    assert cfg.min_samples == 1
    # Fail-closed is the default. A gate you have to opt into is a gate that
    # silently passes on the day it matters.
    assert cfg.fail_on_missing is True


def test_aggregate_stat_defaults_to_mean(tmp_path):
    cfg = load_gate_config(
        write(tmp_path, "metrics:\n  cost_usd:\n    max_increase_pct: 10\n"))
    assert cfg.metrics[0].stat == "mean"


def test_rejects_missing_metrics_section(tmp_path):
    with pytest.raises(GateConfigError, match="at least one metric"):
        load_gate_config(write(tmp_path, "min_samples: 5\n"))


def test_rejects_empty_metrics(tmp_path):
    with pytest.raises(GateConfigError, match="at least one metric"):
        load_gate_config(write(tmp_path, "metrics: {}\n"))


def test_rejects_rule_with_neither_threshold(tmp_path):
    with pytest.raises(GateConfigError, match="exactly one of"):
        load_gate_config(write(tmp_path, "metrics:\n  recall@3:\n    stat: mean\n"))


def test_rejects_rule_with_both_thresholds(tmp_path):
    # Ambiguous direction: is higher better or worse? Refuse to guess.
    text = "metrics:\n  recall@3:\n    max_drop: 0.1\n    max_increase_pct: 5\n"
    with pytest.raises(GateConfigError, match="exactly one of"):
        load_gate_config(write(tmp_path, text))


def test_rejects_negative_threshold(tmp_path):
    with pytest.raises(GateConfigError, match="must not be negative"):
        load_gate_config(write(tmp_path, "metrics:\n  recall@3:\n    max_drop: -0.1\n"))


def test_rejects_unknown_stat(tmp_path):
    text = "metrics:\n  cost_usd:\n    stat: median\n    max_increase_pct: 5\n"
    with pytest.raises(GateConfigError, match="stat must be"):
        load_gate_config(write(tmp_path, text))


def test_rejects_stat_on_a_paired_rule(tmp_path):
    # A paired rule compares per-question deltas; there is no aggregate to pick.
    text = "metrics:\n  recall@3:\n    max_drop: 0.1\n    stat: p95\n"
    with pytest.raises(GateConfigError, match="stat is only valid"):
        load_gate_config(write(tmp_path, text))


def test_rejects_negative_min_samples(tmp_path):
    text = "min_samples: -1\nmetrics:\n  recall@3:\n    max_drop: 0.1\n"
    with pytest.raises(GateConfigError, match="min_samples"):
        load_gate_config(write(tmp_path, text))


def test_missing_file_raises(tmp_path):
    with pytest.raises(GateConfigError, match="not found"):
        load_gate_config(tmp_path / "nope.yaml")


def test_diff_config_never_fails_any_metric():
    from ragmeter.gate.config import diff_config

    cfg = diff_config(["recall@3", "cost_usd", "faithfulness"], k=3)
    by_name = {r.name: r for r in cfg.metrics}
    # Quality metrics compare per question; everything else on an aggregate.
    assert by_name["recall@3"].is_paired is True
    assert by_name["faithfulness"].is_paired is True
    assert by_name["cost_usd"].is_paired is False
    # Limits are infinite: a diff reports, it never blocks.
    assert all(r.limit == float("inf") for r in cfg.metrics)
    assert cfg.fail_on_missing is False
    assert cfg.min_samples == 0
