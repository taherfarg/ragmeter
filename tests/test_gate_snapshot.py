import json

import pytest

from ragmeter.gate.compare import RunMetrics, compare
from ragmeter.gate.config import GateConfig, MetricRule
from ragmeter.gate.snapshot import SnapshotError, dump_snapshot, load_snapshot

METRICS = RunMetrics(
    by_question={"q1": {"recall@3": 1.0, "cost_usd": None},
                 "q2": {"recall@3": 0.5, "cost_usd": 0.001}},
    all_values={"recall@3": [1.0, 0.5], "cost_usd": [None, 0.001]},
    judge_failures=2,
    n_traces=2,
)


def test_round_trip_preserves_everything(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    loaded = load_snapshot(path)
    assert loaded.by_question == METRICS.by_question
    assert loaded.all_values == METRICS.all_values
    assert loaded.judge_failures == 2
    assert loaded.n_traces == 2


def test_none_survives_the_round_trip(tmp_path):
    # None means "not measurable". If JSON turned it into 0.0 the baseline
    # would claim a measurement that never happened.
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    loaded = load_snapshot(path)
    assert loaded.by_question["q1"]["cost_usd"] is None
    assert loaded.all_values["cost_usd"][0] is None


def test_snapshot_is_readable_and_diffable(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    text = path.read_text(encoding="utf-8")
    # Indented and key-sorted so a pull request diff is reviewable and a rerun
    # with unchanged numbers produces no diff at all.
    assert "\n  " in text
    assert text.index('"q1"') < text.index('"q2"')


def test_snapshot_records_its_provenance(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="semantic-v2", k=7)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["run"] == "semantic-v2"
    assert raw["k"] == 7


def test_a_snapshot_compares_exactly_like_a_live_run(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    config = GateConfig(metrics=(MetricRule("recall@3", max_drop=0.01),),
                        min_samples=1, fail_on_missing=False)

    live = compare(METRICS, METRICS, config)
    from_file = compare(load_snapshot(path), METRICS, config)
    assert from_file.outcomes[0].delta == live.outcomes[0].delta
    assert from_file.n_paired == live.n_paired


def test_missing_file_raises_with_the_fix_in_the_message(tmp_path):
    with pytest.raises(SnapshotError, match="ragmeter export"):
        load_snapshot(tmp_path / "nope.json")


def test_malformed_snapshot_raises(tmp_path):
    path = tmp_path / "b.json"
    path.write_text('{"not": "a snapshot"}', encoding="utf-8")
    with pytest.raises(SnapshotError, match="missing"):
        load_snapshot(path)


def test_unparseable_json_raises(tmp_path):
    path = tmp_path / "b.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(SnapshotError, match="not valid JSON"):
        load_snapshot(path)
