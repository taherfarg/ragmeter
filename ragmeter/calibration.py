"""How well does the judge agree with a human?

Reports Cohen's kappa next to the raw agreement rate, because agreement alone
is flattering nonsense: on a corpus where 90% of answers are good, a judge that
always says "good" agrees 90% of the time and has learned nothing.

Pure arithmetic. No numpy, no scipy, no sklearn.
"""

from dataclasses import dataclass, field

__all__ = ["Calibration", "calibrate"]


@dataclass
class Calibration:
    n: int
    agreement: float | None
    kappa: float | None
    threshold: float
    kappa_note: str = ""
    confusion: dict[str, int] = field(default_factory=dict)


def calibrate(pairs: list[tuple[float, float]], threshold: float = 0.5) -> Calibration:
    """Compare judge scores against human labels.

    `pairs` is (judge_score, human_label). Both are binarized at `threshold`
    using >=, so a faithfulness of exactly 0.5 reads as supported.
    """
    if not pairs:
        return Calibration(n=0, agreement=None, kappa=None, threshold=threshold,
                           kappa_note="no labelled pairs")

    counts = {"both_yes": 0, "judge_yes_human_no": 0,
              "judge_no_human_yes": 0, "both_no": 0}
    for judge_score, human_value in pairs:
        judge_yes = judge_score >= threshold
        human_yes = human_value >= threshold
        if judge_yes and human_yes:
            counts["both_yes"] += 1
        elif judge_yes:
            counts["judge_yes_human_no"] += 1
        elif human_yes:
            counts["judge_no_human_yes"] += 1
        else:
            counts["both_no"] += 1

    n = len(pairs)
    agreement = (counts["both_yes"] + counts["both_no"]) / n

    judge_yes_rate = (counts["both_yes"] + counts["judge_yes_human_no"]) / n
    human_yes_rate = (counts["both_yes"] + counts["judge_no_human_yes"]) / n
    expected = (judge_yes_rate * human_yes_rate
                + (1 - judge_yes_rate) * (1 - human_yes_rate))

    if expected >= 1.0:
        # Every label on both sides fell in one class, so chance alone predicts
        # perfect agreement and kappa has no denominator. Undefined is not zero.
        return Calibration(n=n, agreement=agreement, kappa=None, threshold=threshold,
                           confusion=counts,
                           kappa_note="undefined: all labels fall in one class, "
                                      "so chance already predicts full agreement")

    return Calibration(n=n, agreement=agreement,
                       kappa=(agreement - expected) / (1 - expected),
                       threshold=threshold, confusion=counts)
