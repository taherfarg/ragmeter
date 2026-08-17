import pytest

from ragmeter.calibration import calibrate

# Every expected value below is computed by hand from the kappa formula.


def test_perfect_agreement():
    # judge 1,1,0,0 vs human 1,1,0,0
    # po = 1.0; P(j=1)=0.5, P(h=1)=0.5 -> pe = 0.25 + 0.25 = 0.5
    # kappa = (1.0 - 0.5) / (1 - 0.5) = 1.0
    result = calibrate([(1.0, 1.0), (0.9, 1.0), (0.1, 0.0), (0.0, 0.0)])
    assert result.n == 4
    assert result.agreement == 1.0
    assert result.kappa == 1.0


def test_a_lazy_judge_scores_high_agreement_and_zero_kappa():
    # THE point of this module. The judge says "good" every time. Nine of ten
    # answers really are good, so it agrees 90% of the time -- and is useless.
    # po = 0.9; P(j=1)=1.0, P(h=1)=0.9 -> pe = 1.0*0.9 + 0.0*0.1 = 0.9
    # kappa = (0.9 - 0.9) / (1 - 0.9) = 0.0
    pairs = [(1.0, 1.0)] * 9 + [(1.0, 0.0)]
    result = calibrate(pairs)
    assert result.agreement == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.0)


def test_worked_example():
    # judge 1,1,1,1,0,0,0,0,0,0  human 1,1,0,0,1,0,0,0,0,0
    # a=2 b=2 c=1 d=5, n=10
    # po = (2+5)/10 = 0.7
    # P(j=1)=0.4, P(h=1)=0.3 -> pe = 0.4*0.3 + 0.6*0.7 = 0.12 + 0.42 = 0.54
    # kappa = (0.7 - 0.54) / (1 - 0.54) = 0.16/0.46 = 0.34782608695652173
    judge = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    human = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    result = calibrate([(float(j), float(h)) for j, h in zip(judge, human)])
    assert result.agreement == pytest.approx(0.7)
    assert result.kappa == pytest.approx(0.34782608695652173)
    assert result.confusion == {"both_yes": 2, "judge_yes_human_no": 2,
                                "judge_no_human_yes": 1, "both_no": 5}


def test_complete_disagreement_is_negative():
    # a=0 b=1 c=1 d=0; po=0; P(j=1)=0.5, P(h=1)=0.5 -> pe=0.5
    # kappa = (0 - 0.5)/(1 - 0.5) = -1.0
    result = calibrate([(1.0, 0.0), (0.0, 1.0)])
    assert result.agreement == 0.0
    assert result.kappa == -1.0


def test_kappa_undefined_when_everything_is_one_class():
    # pe == 1, so the formula divides by zero. That is "cannot compute", which
    # is a different claim from "no better than chance".
    result = calibrate([(1.0, 1.0), (1.0, 1.0), (0.9, 1.0)])
    assert result.agreement == 1.0
    assert result.kappa is None
    assert "undefined" in result.kappa_note


def test_no_pairs_is_all_none():
    result = calibrate([])
    assert result.n == 0
    assert result.agreement is None
    assert result.kappa is None


def test_threshold_binarizes_the_judge_score():
    # 0.6 is above the default 0.5 and below a 0.8 threshold.
    assert calibrate([(0.6, 1.0)], threshold=0.5).confusion["both_yes"] == 1
    assert calibrate([(0.6, 1.0)], threshold=0.8).confusion["judge_no_human_yes"] == 1


def test_threshold_boundary_counts_as_yes():
    # >= not >. Half the claims supported reads as "supported" at 0.5.
    assert calibrate([(0.5, 1.0)], threshold=0.5).confusion["both_yes"] == 1


def test_human_values_are_binarized_too():
    result = calibrate([(1.0, 1.0), (0.0, 0.0)])
    assert result.confusion["both_yes"] == 1
    assert result.confusion["both_no"] == 1


def test_agreement_and_kappa_can_diverge_sharply():
    # 8 of 10 agree, but the judge barely tracks the human.
    # judge 1,1,1,1,1,1,1,1,1,0  human 1,1,1,1,1,1,1,1,0,1
    # a=8 b=1 c=1 d=0; po=0.8
    # P(j=1)=0.9, P(h=1)=0.9 -> pe = 0.81 + 0.01 = 0.82
    # kappa = (0.8 - 0.82)/(1 - 0.82) = -0.02/0.18 = -0.1111...
    judge = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    human = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    result = calibrate([(float(j), float(h)) for j, h in zip(judge, human)])
    assert result.agreement == pytest.approx(0.8)
    assert result.kappa == pytest.approx(-0.1111111111111111)
    # 80% agreement, worse than chance. Quoting agreement alone would be a lie.
    assert result.kappa < 0


def test_result_carries_the_threshold_used():
    assert calibrate([(1.0, 1.0)], threshold=0.7).threshold == 0.7
