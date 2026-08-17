"""Joins traces to golden items and writes Evaluation rows.

The only place that knows both about the database and about metrics. Metrics
stay pure; the database stays dumb.
"""

from sqlalchemy.orm import Session

from ragmeter.db import Evaluation, GoldenItem, Run, Trace
from ragmeter.judge.client import JudgeError
from ragmeter.judge.scoring import (
    score_answer_relevance,
    score_chunk_relevance,
    score_faithfulness,
)
from ragmeter.metrics.cost import compute_cost
from ragmeter.metrics.retrieval import evaluate_retrieval, metric_names

__all__ = ["evaluate_run"]


def _judge_trace(judge, trace: Trace, chunks: list[dict], k: int, labeled: bool) -> dict:
    """Run the judge over one trace. Raises JudgeError; the caller records it."""
    faithfulness = score_faithfulness(judge, trace.question, chunks, trace.answer)
    relevance = score_answer_relevance(judge, trace.question, trace.answer)

    result = {
        "metrics": {
            "faithfulness": faithfulness["score"],
            "answer_relevance": relevance["score"],
        },
        "claims": faithfulness["claims"] or None,
        "chunk_judgments": None,
    }

    if not labeled:
        # Without golden labels the judge is the only source of precision.
        # It can never supply recall -- nothing can see what was not retrieved.
        chunk = score_chunk_relevance(judge, trace.question, chunks)
        result["metrics"][f"precision@{k}"] = chunk["precision"]
        result["chunk_judgments"] = chunk["judgments"] or None

    return result


def evaluate_run(
    session: Session,
    run_name: str,
    dataset: str,
    version: str,
    k: int,
    prices: dict[str, tuple[float, float]],
    judge=None,
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
    judge_failures = 0

    for trace in traces:
        item = golden.get(trace.question_id) if trace.question_id else None
        chunks = list(trace.retrieved or [])
        chunk_ids = [c["chunk_id"] for c in chunks]

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

        claims = None
        chunk_judgments = None
        judge_status = "skipped"
        judge_error = None

        if judge is not None:
            try:
                judged = _judge_trace(judge, trace, chunks, k, labeled=item is not None)
            except JudgeError as exc:
                # Record the failure. Never substitute a number for a measurement
                # that did not happen -- the gate must be able to see this.
                judge_status = "failed"
                judge_error = str(exc)
                judge_failures += 1
                metrics["faithfulness"] = None
                metrics["answer_relevance"] = None
            else:
                judge_status = "ok"
                metrics.update(judged["metrics"])
                claims = judged["claims"]
                chunk_judgments = judged["chunk_judgments"]

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
            claims=claims,
            chunk_judgments=chunk_judgments,
            judge_model=getattr(judge, "model", None) if judge is not None else None,
            judge_status=judge_status,
            judge_error=judge_error,
        ))

    session.commit()
    return {
        "n_traces": len(traces),
        "n_matched": matched,
        "n_unmatched": len(traces) - matched,
        "n_judge_failures": judge_failures,
    }
