from ragmeter.gate.compare import GateResult, MetricOutcome
from ragmeter.gate.render import render_gate


def outcome(**kwargs):
    base = dict(name="recall@3", kind="paired", baseline=1.0, candidate=0.8,
                delta=-0.2, limit=0.02, passed=False, reason="dropped 0.2000",
                n_paired=5, n_improved=1, n_regressed=2)
    base.update(kwargs)
    return MetricOutcome(**base)


def test_shows_verdict_and_numbers():
    text = render_gate(GateResult(outcomes=[outcome()], n_paired=5), "cand", "base", 3)
    assert "FAIL" in text
    assert "recall@3" in text
    assert "cand" in text and "base" in text


def test_shows_improved_and_regressed_counts():
    text = render_gate(GateResult(outcomes=[outcome()], n_paired=5), "cand", "base", 3)
    # These counts are the whole point of a paired gate: they show the spread
    # that a mean delta flattens away.
    assert "+1/-2" in text


def test_passing_result_says_pass():
    passing = outcome(passed=True, delta=0.0, reason="")
    text = render_gate(GateResult(outcomes=[passing], n_paired=5), "cand", "base", 3)
    assert "PASS" in text
    assert "FAIL" not in text


def test_blocking_reasons_are_shown():
    result = GateResult(outcomes=[outcome(passed=True, reason="")], n_paired=5,
                        blocking_reasons=["candidate has 2 judge failure(s)"])
    text = render_gate(result, "cand", "base", 3)
    assert "FAIL" in text
    assert "judge failure" in text


def test_unmeasurable_values_render_as_dashes():
    blank = outcome(baseline=None, candidate=None, delta=None,
                    passed=False, reason="no paired measurements")
    text = render_gate(GateResult(outcomes=[blank]), "cand", "base", 3)
    assert "-" in text
    assert "no paired measurements" in text
