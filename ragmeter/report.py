"""Run summaries as plain text.

Every row shows measured/total. A mean is meaningless without knowing how many
values went into it.
"""

from sqlalchemy.orm import Session

from ragmeter.db import Evaluation, Run, Trace
from ragmeter.metrics.aggregate import summarize

__all__ = ["summarize_run", "render_summary"]


def summarize_run(session: Session, run_name: str, k: int) -> dict[str, dict]:
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    evaluations = (
        session.query(Evaluation)
        .join(Trace, Trace.trace_id == Evaluation.trace_id)
        .filter(Trace.run_id == run.run_id, Evaluation.k == k)
        .all()
    )

    names: list[str] = []
    for ev in evaluations:
        for name in ev.metrics:
            if name not in names:
                names.append(name)

    return {
        name: summarize([ev.metrics.get(name) for ev in evaluations])
        for name in names
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}"


def render_summary(summary: dict[str, dict], run_name: str, k: int) -> str:
    lines = [f"run: {run_name}   k={k}", ""]
    lines.append(f"{'metric':<16}{'mean':>12}{'p50':>12}{'p95':>12}{'measured':>12}")
    lines.append("-" * 64)
    for name, stats in summary.items():
        measured = f"{stats['n_measured']}/{stats['n']}"
        lines.append(
            f"{name:<16}{_fmt(stats['mean']):>12}{_fmt(stats['p50']):>12}"
            f"{_fmt(stats['p95']):>12}{measured:>12}"
        )
    return "\n".join(lines)
