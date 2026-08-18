"""HTTP API. Stores and serves; computes nothing.

Every calculation lives in the library, so the regression gate keeps working
with no server running. If a handler here starts doing arithmetic, that
arithmetic belongs somewhere else.
"""

import logging
import os
from dataclasses import asdict
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ragmeter.db import (
    DEFAULT_DB_URL, GoldenItem, Run, Trace, init_db, make_engine, make_session,
)
from ragmeter.gate.collect import collect_run_metrics
from ragmeter.gate.compare import compare
from ragmeter.gate.config import diff_config
from ragmeter.loaders import get_or_create_run, ingest_golden, ingest_traces
from ragmeter.metrics.cost import fetch_prices
from ragmeter.models import GoldenItemIn, TraceIn
from ragmeter.report import summarize_run
from ragmeter.runner import evaluate_run

log = logging.getLogger("ragmeter.api")

app = FastAPI(title="ragmeter", description="Measure any RAG system.")

# One engine per database URL, rebuilt only when the URL changes. Reading the
# env var per request keeps tests able to point at a temporary database.
_state: dict[str, Any] = {"url": None, "sessionmaker": None}


def _sessionmaker():
    url = os.environ.get("RAGMETER_DB_URL", DEFAULT_DB_URL)
    if _state["url"] != url:
        engine = make_engine(url)
        init_db(engine)
        _state.update(url=url, sessionmaker=make_session(engine))
    return _state["sessionmaker"]


def get_session():
    session = _sessionmaker()()
    try:
        yield session
    finally:
        session.close()


class RunIn(BaseModel):
    name: str
    git_sha: str | None = None
    config: dict = Field(default_factory=dict)


class DatasetIn(BaseModel):
    name: str
    version: str = "v1"
    items: list[GoldenItemIn]


class TraceBatch(BaseModel):
    # Raw dicts, not TraceIn: validating per item is what lets one bad record
    # be reported without rejecting the rest of the batch.
    traces: list[dict]


class EvaluateIn(BaseModel):
    dataset: str
    version: str = "v1"
    k: int = Field(default=5, ge=1)


def _run_or_404(session: Session, run_id: str) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run with id {run_id!r}")
    return run


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/runs", status_code=201)
def create_run(body: RunIn, session: Session = Depends(get_session)) -> dict:
    run = get_or_create_run(session, body.name, git_sha=body.git_sha, config=body.config)
    session.commit()
    return {"run_id": run.run_id, "name": run.name}


@app.post("/v1/datasets", status_code=201)
def upload_dataset(body: DatasetIn, session: Session = Depends(get_session)) -> dict:
    count = ingest_golden(session, body.items, dataset=body.name, version=body.version)
    session.commit()
    return {"dataset": body.name, "version": body.version, "items": count}


@app.post("/v1/runs/{run_id}/traces")
def add_traces(
    run_id: str, body: TraceBatch, response: Response,
    session: Session = Depends(get_session),
) -> dict:
    run = _run_or_404(session, run_id)

    records = []
    errors = []
    for index, raw in enumerate(body.traces):
        try:
            records.append(TraceIn.model_validate(raw))
        except Exception as exc:
            errors.append({"index": index, "error": str(exc)})

    result = ingest_traces(session, records, run)
    session.commit()

    if errors and not records:
        # Nothing usable arrived: that is a request problem, not a partial success.
        response.status_code = 422
    return {**result, "errors": errors}


def _evaluate_in_background(run_name: str, dataset: str, version: str, k: int) -> None:
    """Runs after the response is sent, so it needs its own session."""
    session = _sessionmaker()()
    try:
        prices = fetch_prices()
    except Exception as exc:
        log.warning("could not fetch model prices: %s; cost will be blank", exc)
        prices = {}
    try:
        evaluate_run(session, run_name, dataset, version, k=k, prices=prices)
    except Exception:
        # ponytail: the failure is logged and lost. There is no job status
        # endpoint -- add a jobs table if callers need to see why it failed.
        log.exception("background evaluation of run %r failed", run_name)
    finally:
        session.close()


@app.post("/v1/runs/{run_id}/evaluate", status_code=202)
def start_evaluation(
    run_id: str, body: EvaluateIn, background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> dict:
    run = _run_or_404(session, run_id)

    exists = (
        session.query(GoldenItem)
        .filter_by(dataset=body.dataset, version=body.version)
        .first()
    )
    if exists is None:
        raise HTTPException(
            status_code=404,
            detail=f"no golden items for dataset {body.dataset!r} version {body.version!r}",
        )

    n_traces = session.query(Trace).filter_by(run_id=run.run_id).count()
    background.add_task(_evaluate_in_background, run.name, body.dataset, body.version, body.k)
    return {"run": run.name, "k": body.k, "traces": n_traces, "status": "accepted"}


@app.get("/v1/runs/{run_id}/metrics")
def read_metrics(run_id: str, k: int = 5, session: Session = Depends(get_session)) -> dict:
    run = _run_or_404(session, run_id)
    summary = summarize_run(session, run.name, k=k)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"run {run.name!r} has no evaluations at k={k}; evaluate it first",
        )
    return {"run": run.name, "k": k, "metrics": summary}


@app.get("/v1/compare")
def compare_runs(
    run: str, baseline: str, k: int = 5, session: Session = Depends(get_session),
) -> dict:
    try:
        base = collect_run_metrics(session, baseline, k=k)
        cand = collect_run_metrics(session, run, k=k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    names = set(base.all_values) | set(cand.all_values)
    result = compare(base, cand, diff_config(names, k))
    return {"run": run, "baseline": baseline, "k": k, "passed": result.passed,
            "n_paired": result.n_paired,
            "outcomes": [asdict(o) for o in result.outcomes]}
