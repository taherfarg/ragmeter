"""Database to RunMetrics. The only file in the gate package that sees a session."""

from statistics import fmean

from sqlalchemy.orm import Session

from ragmeter.db import Evaluation, Run, Trace
from ragmeter.gate.compare import RunMetrics

__all__ = ["collect_run_metrics"]


def collect_run_metrics(session: Session, run_name: str, k: int) -> RunMetrics:
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    rows = (
        session.query(Evaluation, Trace)
        .join(Trace, Trace.trace_id == Evaluation.trace_id)
        .filter(Trace.run_id == run.run_id, Evaluation.k == k)
        .all()
    )
    if not rows:
        raise ValueError(
            f"run {run_name!r} has no evaluations at k={k}; run `ragmeter eval` first"
        )

    all_values: dict[str, list[float | None]] = {}
    grouped: dict[str, dict[str, list[float]]] = {}

    for evaluation, trace in rows:
        for name, value in evaluation.metrics.items():
            all_values.setdefault(name, []).append(value)
            if trace.question_id is None or value is None:
                continue
            grouped.setdefault(trace.question_id, {}).setdefault(name, []).append(value)

    # A run may legitimately answer the same golden question more than once.
    # Averaging keeps one question from carrying extra weight in the pairing.
    by_question = {
        question_id: {name: fmean(values) for name, values in metrics.items()}
        for question_id, metrics in grouped.items()
    }

    return RunMetrics(
        by_question=by_question,
        all_values=all_values,
        judge_failures=sum(1 for ev, _ in rows if ev.judge_status == "failed"),
        n_traces=len(rows),
    )
