import pytest

from ragmeter.gate.compare import RunMetrics, compare
from ragmeter.gate.config import GateConfig, MetricRule


def metrics(by_question=None, all_values=None, judge_failures=0, n_traces=0):
    return RunMetrics(
        by_question=by_question or {},
        all_values=all_values or {},
        judge_failures=judge_failures,
        n_traces=n_traces or len(by_question or {}),
    )


def paired_config(limit=0.02, min_samples=1, fail_on_missing=True):
    return GateConfig(metrics=(MetricRule("recall@3", max_drop=limit),),
                      min_samples=min_samples, fail_on_missing=fail_on_missing)


def aggregate_config(limit=20.0, stat="mean", fail_on_missing=True):
    return GateConfig(
        metrics=(MetricRule("cost_usd", max_increase_pct=limit, stat=stat),),
        min_samples=1, fail_on_missing=fail_on_missing)


def test_identical_runs_pass():
    values = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 0.5}}
    result = compare(metrics(values), metrics(values), paired_config())
    assert result.passed is True
    assert result.outcomes[0].delta == 0.0


def test_clear_regression_fails():
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 0.0}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config(limit=0.02))
    assert result.passed is False
    assert result.outcomes[0].delta == -0.5
    assert result.outcomes[0].n_regressed == 1
    assert result.outcomes[0].n_improved == 0


def test_regression_within_tolerance_passes():
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 0.98}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config(limit=0.02))
    assert result.passed is True


def test_improvement_passes():
    base = {"q1": {"recall@3": 0.5}}
    cand = {"q1": {"recall@3": 0.9}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.passed is True
    assert result.outcomes[0].delta == pytest.approx(0.4)


def test_counts_expose_a_redistribution_the_mean_hides():
    # Mean delta is exactly zero, but one question collapsed and another jumped.
    # The counts are the only thing that reveals it.
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 0.0}}
    cand = {"q1": {"recall@3": 0.0}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    outcome = result.outcomes[0]
    assert outcome.delta == 0.0
    assert outcome.n_improved == 1
    assert outcome.n_regressed == 1
    assert result.passed is True


def test_only_questions_in_both_runs_are_paired():
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 1.0}, "q3": {"recall@3": 0.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.outcomes[0].n_paired == 1
    assert result.n_paired == 1


def test_unmeasurable_values_are_excluded_from_pairing():
    # A None on either side means that question was never measured. Treating it
    # as 0.0 would manufacture a regression that did not happen.
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": None}}
    cand = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.outcomes[0].n_paired == 1
    assert result.passed is True


def test_too_few_pairs_fails():
    base = {"q1": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config(min_samples=10))
    assert result.passed is False
    assert "min_samples" in result.outcomes[0].reason


def test_metric_absent_everywhere_fails_closed():
    base = {"q1": {"other": 1.0}}
    cand = {"q1": {"other": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.passed is False
    assert "no paired measurements" in result.outcomes[0].reason


def test_metric_absent_passes_when_fail_on_missing_is_off():
    base = {"q1": {"other": 1.0}}
    cand = {"q1": {"other": 1.0}}
    result = compare(metrics(base), metrics(cand),
                     paired_config(fail_on_missing=False))
    assert result.passed is True


def test_judge_failure_blocks_the_gate():
    values = {"q1": {"recall@3": 1.0}}
    result = compare(metrics(values), metrics(values, judge_failures=2),
                     paired_config())
    assert result.passed is False
    assert any("judge" in r for r in result.blocking_reasons)


def test_judge_failure_ignored_when_fail_on_missing_is_off():
    values = {"q1": {"recall@3": 1.0}}
    result = compare(metrics(values), metrics(values, judge_failures=2),
                     paired_config(fail_on_missing=False))
    assert result.passed is True


def test_aggregate_increase_within_limit_passes():
    base = metrics(all_values={"cost_usd": [1.0, 1.0]})
    cand = metrics(all_values={"cost_usd": [1.1, 1.1]})
    result = compare(base, cand, aggregate_config(limit=20.0))
    assert result.passed is True
    assert result.outcomes[0].delta == pytest.approx(10.0)


def test_aggregate_increase_past_limit_fails():
    base = metrics(all_values={"cost_usd": [1.0, 1.0]})
    cand = metrics(all_values={"cost_usd": [2.0, 2.0]})
    result = compare(base, cand, aggregate_config(limit=20.0))
    assert result.passed is False
    assert result.outcomes[0].delta == pytest.approx(100.0)


def test_aggregate_decrease_always_passes():
    base = metrics(all_values={"cost_usd": [2.0]})
    cand = metrics(all_values={"cost_usd": [1.0]})
    result = compare(base, cand, aggregate_config(limit=0.0))
    assert result.passed is True


def test_aggregate_uses_the_requested_stat():
    base = metrics(all_values={"cost_usd": [1.0, 1.0, 1.0, 100.0]})
    cand = metrics(all_values={"cost_usd": [1.0, 1.0, 1.0, 100.0]})
    result = compare(base, cand, aggregate_config(stat="p95"))
    assert result.outcomes[0].baseline == 100.0


def test_aggregate_ignores_unmeasurable_values():
    base = metrics(all_values={"cost_usd": [1.0, None]})
    cand = metrics(all_values={"cost_usd": [1.0, None]})
    result = compare(base, cand, aggregate_config())
    assert result.outcomes[0].baseline == 1.0
    assert result.passed is True


def test_aggregate_growth_from_zero_fails():
    # Percent change from zero is undefined; a rise from nothing to something
    # is a real increase and must not be silently treated as 0%.
    base = metrics(all_values={"cost_usd": [0.0]})
    cand = metrics(all_values={"cost_usd": [1.0]})
    result = compare(base, cand, aggregate_config())
    assert result.passed is False
    assert "zero" in result.outcomes[0].reason


def test_aggregate_zero_to_zero_passes():
    base = metrics(all_values={"cost_usd": [0.0]})
    cand = metrics(all_values={"cost_usd": [0.0]})
    result = compare(base, cand, aggregate_config())
    assert result.passed is True


def test_aggregate_missing_fails_closed():
    base = metrics(all_values={"cost_usd": [None]})
    cand = metrics(all_values={"cost_usd": [None]})
    result = compare(base, cand, aggregate_config())
    assert result.passed is False


def test_one_failing_metric_fails_the_whole_gate():
    config = GateConfig(metrics=(MetricRule("recall@3", max_drop=0.01),
                                 MetricRule("ndcg@3", max_drop=0.01)),
                        min_samples=1)
    base = metrics({"q1": {"recall@3": 1.0, "ndcg@3": 1.0}})
    cand = metrics({"q1": {"recall@3": 1.0, "ndcg@3": 0.0}})
    result = compare(base, cand, config)
    assert result.passed is False
    assert [o.passed for o in result.outcomes] == [True, False]
