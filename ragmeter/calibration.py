"""How well does the judge agree with a human?

Reports Cohen's kappa next to the raw agreement rate, because agreement alone
is flattering nonsense: on a corpus where 90% of answers are good, a judge that
always says "good" agrees 90% of the time and has learned nothing.

Pure arithmetic. No numpy, no scipy, no sklearn.
"""

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ragmeter.db import Evaluation, HumanLabel, Run, Trace

__all__ = ["Calibration", "calibrate", "collect_labeled_pairs", "unlabeled_traces"]


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


def collect_labeled_pairs(
    session: Session, run_name: str, metric: str, k: int
) -> list[tuple[float, float]]:
    """(judge_score, human_label) for every trace in the run that has both."""
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    rows = (
        session.query(Evaluation, HumanLabel)
        .join(Trace, Trace.trace_id == Evaluation.trace_id)
        .join(HumanLabel, HumanLabel.trace_id == Evaluation.trace_id)
        .filter(Trace.run_id == run.run_id, Evaluation.k == k,
                HumanLabel.metric == metric)
        .all()
    )

    pairs = []
    for evaluation, label in rows:
        score = evaluation.metrics.get(metric)
        # A judge score of None was never measured, so there is nothing to
        # compare the human against.
        if score is not None:
            pairs.append((float(score), float(label.value)))
    return pairs


def unlabeled_traces(
    session: Session, run_name: str, metric: str, k: int, labeler: str, limit: int
) -> list[tuple[Trace, Evaluation]]:
    """Traces with a judge score for this metric and no label from this person yet."""
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    already = {
        row.trace_id
        for row in session.query(HumanLabel)
        .filter_by(metric=metric, labeler=labeler)
        .all()
    }

    rows = (
        session.query(Trace, Evaluation)
        .join(Evaluation, Evaluation.trace_id == Trace.trace_id)
        .filter(Trace.run_id == run.run_id, Evaluation.k == k)
        .order_by(Trace.trace_id)
        .all()
    )

    out = []
    for trace, evaluation in rows:
        if trace.trace_id in already:
            continue
        if evaluation.metrics.get(metric) is None:
            continue
        out.append((trace, evaluation))
        if len(out) == limit:
            break
    return out
