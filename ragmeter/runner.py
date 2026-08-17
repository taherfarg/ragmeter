"""Joins traces to golden items and writes Evaluation rows.

The only place that knows both about the database and about metrics. Metrics
stay pure; the database stays dumb.
"""

from sqlalchemy.orm import Session

from ragmeter.db import Evaluation, GoldenItem, Run, Trace
from ragmeter.metrics.cost import compute_cost
from ragmeter.metrics.retrieval import evaluate_retrieval, metric_names

__all__ = ["evaluate_run"]


def evaluate_run(
    session: Session,
    run_name: str,
    dataset: str,
    version: str,
    k: int,
    prices: dict[str, tuple[float, float]],
) -> dict[str, int]:
    """Evaluate every trace in a run. Re-running replaces results for the same k."""
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    golden = {
        item.question_id: item
        for item in session.query(GoldenItem).filter_by(dataset=dataset, version=version)
    }
    if not golden:
        raise ValueError(f"no golden items for dataset {dataset!r} version {version!r}")

    traces = session.query(Trace).filter_by(run_id=run.run_id).all()
    matched = 0

    for trace in traces:
        item = golden.get(trace.question_id) if trace.question_id else None
        chunk_ids = [c["chunk_id"] for c in (trace.retrieved or [])]

        if item is not None:
            metrics = evaluate_retrieval(chunk_ids, item.relevant_chunk_ids, k)
            matched += 1
        else:
            # No ground truth: report the retrieval metrics as unmeasurable rather
            # than omitting the keys, so aggregates keep a consistent shape.
            metrics = {name: None for name in metric_names(k)}

        metrics["cost_usd"] = compute_cost(
            trace.model, trace.prompt_tokens, trace.completion_tokens,
            prices, supplied=trace.cost_usd,
        )
        metrics["latency_ms"] = trace.latency_ms

        existing = session.query(Evaluation).filter_by(trace_id=trace.trace_id, k=k).one_or_none()
        if existing is not None:
            session.delete(existing)
            session.flush()

        session.add(Evaluation(
            trace_id=trace.trace_id,
            k=k,
            dataset=dataset if item is not None else None,
            dataset_version=version if item is not None else None,
            metrics=metrics,
            judge_status="skipped",
        ))

    session.commit()
    return {
        "n_traces": len(traces),
        "n_matched": matched,
        "n_unmatched": len(traces) - matched,
    }
