# ragmeter Phase 5 — HTTP API and the Refusal Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thin FastAPI wrapper so a running RAG can post traces over HTTP, backed by Postgres — plus a correctness fix so an honest refusal is scored as *unmeasurable* rather than as a hallucination.

**Architecture:** The API stores and serves; it computes nothing. Every calculation already lives in the library, which is why this phase is much smaller than the four before it. The one genuinely new concern is per-item batch validation, because a single bad trace in a batch of 200 must not reject the other 199.

**Tech Stack:** FastAPI, uvicorn, psycopg, docker-compose. All already declared as extras in `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`

---

## The Refusal Fix

Observed live in Phase 2:

| trace | answer | retrieved | faithfulness |
|---|---|---|---|
| t4 | "I don't know." | nothing | `None` |
| t5 | "No idea." | one irrelevant chunk | **0.0** |

Two equivalent honest refusals, scored differently, purely because retrieval
returned something for one of them. Worse, `0.0` is the *same score a confident
fabrication gets*. A gate tuned on faithfulness would therefore punish a model
for admitting it does not know and reward one that invents an answer — exactly
backwards.

The cause: the judge dutifully turns "No idea." into the claim *"the speaker
does not know whether an order can be cancelled"*, finds no source for it, and
marks it unsupported. But that is a statement about the speaker, not a factual
claim about the world, and faithfulness is not defined over it.

**The fix is in the prompt**: a refusal or non-answer yields `claims: []`, which
`score_faithfulness` already reports as `None`. `PROMPT_VERSION` bumps to `2` so
the cache does not keep serving verdicts produced by the old wording.

## Design Rules

**A bad item in a batch must not reject the good ones.** Per-item validation
with an errors array, not all-or-nothing.

**The API computes nothing.** If a handler needs arithmetic, that arithmetic
belongs in the library and the handler calls it.

## File Structure

| File | Responsibility |
|---|---|
| `ragmeter/judge/prompts.py` (modify) | refusal instruction, `PROMPT_VERSION` -> 2 |
| `ragmeter/judge/scoring.py` (modify) | clearer reason text for the empty-claims case |
| `ragmeter/loaders.py` (modify) | extract `ingest_traces` / `ingest_golden` from the file readers |
| `ragmeter/gate/config.py` (modify) | extract `diff_config` shared by CLI compare and the API |
| `ragmeter/api.py` | FastAPI app. Stores and serves only |
| `ragmeter/cli.py` (modify) | `serve` command |
| `docker-compose.yml` | Postgres |

---

### Task 1: The refusal fix

**Files:**
- Modify: `ragmeter/judge/prompts.py`
- Modify: `ragmeter/judge/scoring.py`
- Test: `tests/test_judge_prompts.py`, `tests/test_judge_scoring.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_judge_prompts.py`:

```python
def test_faithfulness_prompt_tells_the_judge_how_to_treat_a_refusal():
    # "I don't know" asserts nothing about the world, so faithfulness is not
    # defined over it. Without this instruction the judge invents a claim about
    # the speaker and scores the refusal 0.0 -- the same score a confident
    # fabrication gets.
    p = faithfulness_prompt("q", [{"chunk_id": "c1", "text": "t"}], "I don't know.")
    lowered = p.lower()
    assert "refus" in lowered or "does not answer" in lowered
    assert "empty" in lowered or "[]" in p


def test_prompt_version_bumped_past_one():
    # The cache key embeds this. Editing the wording without bumping it would
    # keep serving verdicts produced by the previous prompt.
    assert PROMPT_VERSION != "1"
```

Append to `tests/test_judge_scoring.py`:

```python
def test_refusal_yielding_no_claims_is_unmeasurable():
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", CHUNKS, "I don't know.")
    assert out["score"] is None
    assert "no factual claims" in out["reason"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_judge_prompts.py tests/test_judge_scoring.py -q`
Expected: 3 failures.

- [ ] **Step 3: Update the prompt**

In `ragmeter/judge/prompts.py`, set `PROMPT_VERSION = "2"` and replace the
instruction paragraph of `faithfulness_prompt` with:

```
Break the ANSWER into atomic factual claims. For each claim, decide whether the
SOURCES support it. A claim is supported only if a source states it or directly
implies it. Correct-sounding general knowledge that does not appear in the
SOURCES is NOT supported. Cite the chunk ids in square brackets above.

If the ANSWER asserts nothing about the world -- a refusal, "I don't know", an
apology, or a request for clarification -- return an empty claims list. Do NOT
turn the refusal itself into a claim about the speaker. An answer that declines
to answer is not unfaithful; it simply cannot be scored for faithfulness.

Return only a JSON object of this shape and nothing else:
{"claims": [{"claim": "...", "supported": true, "chunk_ids": ["c1"], "reason": "..."}]}
```

- [ ] **Step 4: Update the reason text in `ragmeter/judge/scoring.py`**

Replace the empty-claims return with:

```python
    if not claims:
        # A refusal or non-answer. Not a zero: it asserted nothing to check.
        return {"score": None, "claims": [],
                "reason": "answer makes no factual claims (refusal or non-answer)"}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_judge_prompts.py tests/test_judge_scoring.py -q`
Expected: `25 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/judge tests/test_judge_prompts.py tests/test_judge_scoring.py
git commit -m "fix: an honest refusal is unmeasurable, not unfaithful"
```

---

### Task 2: Extract reusable ingest functions

The API receives parsed JSON, not files. Rather than duplicating the insert
logic, split the existing loaders into "parse a file" and "ingest records".

**Files:**
- Modify: `ragmeter/loaders.py`
- Test: `tests/test_loaders.py` (append)

- [ ] **Step 1: Write the failing tests**

Add to the imports at the top of `tests/test_loaders.py`:

```python
from ragmeter.db import GoldenItem, Run, Trace, init_db, make_engine, make_session
from ragmeter.loaders import (
    get_or_create_run, ingest_golden, ingest_traces, load_golden, load_traces,
)
from ragmeter.models import GoldenItemIn, TraceIn
```

Append to `tests/test_loaders.py`:

```python
def test_ingest_traces_accepts_parsed_records(session):
    run = get_or_create_run(session, "api")
    result = ingest_traces(session, [TraceIn(trace_id="a1", question="why?")], run)
    session.commit()
    assert result == {"ingested": 1, "skipped": 0}
    assert session.get(Trace, "a1").question == "why?"


def test_ingest_traces_is_idempotent(session):
    run = get_or_create_run(session, "api")
    records = [TraceIn(trace_id="a1", question="why?")]
    ingest_traces(session, records, run)
    session.commit()
    assert ingest_traces(session, records, run) == {"ingested": 0, "skipped": 1}


def test_ingest_golden_accepts_parsed_records(session):
    items = [GoldenItemIn(question_id="q9", question="why?", relevant_chunk_ids=["c1"])]
    assert ingest_golden(session, items, dataset="d", version="v1") == 1
    session.commit()
    assert session.get(GoldenItem, ("d", "v1", "q9")).relevant_chunk_ids == ["c1"]


def test_file_loaders_still_work_after_the_split(session):
    # The CLI path must be unchanged: the file readers now delegate, and their
    # fail-fast behaviour on a malformed line is deliberate.
    run = get_or_create_run(session, "baseline")
    assert load_traces(session, FIXTURES / "traces.jsonl", run)["ingested"] == 6
    assert load_golden(session, FIXTURES / "golden.yaml", dataset="d", version="v1") == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_loaders.py -q`
Expected: `ImportError: cannot import name 'ingest_traces'`

- [ ] **Step 3: Refactor `ragmeter/loaders.py`**

Add `"ingest_traces"` and `"ingest_golden"` to `__all__`. Add these two
functions, and rewrite the file readers to delegate to them:

```python
def ingest_golden(
    session: Session, items: list[GoldenItemIn], dataset: str, version: str
) -> int:
    """Upsert already-validated golden items. Re-loading overwrites in place."""
    for item in items:
        session.merge(GoldenItem(
            dataset=dataset,
            version=version,
            question_id=item.question_id,
            question=item.question,
            relevant_chunk_ids=item.relevant_chunk_ids,
            reference_answer=item.reference_answer,
        ))
    return len(items)


def ingest_traces(session: Session, records: list[TraceIn], run: Run) -> dict[str, int]:
    """Insert already-validated traces. Idempotent on trace_id."""
    ingested = 0
    skipped = 0
    for trace_in in records:
        if session.get(Trace, trace_in.trace_id) is not None:
            skipped += 1
            continue
        session.add(Trace(
            trace_id=trace_in.trace_id,
            run_id=run.run_id,
            question_id=trace_in.question_id,
            question=trace_in.question,
            retrieved=[c.model_dump() for c in trace_in.retrieved],
            answer=trace_in.answer,
            model=trace_in.model,
            prompt_tokens=trace_in.prompt_tokens,
            completion_tokens=trace_in.completion_tokens,
            cost_usd=trace_in.cost_usd,
            latency_ms=trace_in.latency_ms,
            meta=trace_in.metadata,
        ))
        session.flush()
        ingested += 1
    return {"ingested": ingested, "skipped": skipped}
```

Replace the body of `load_golden` after the `raw` type check with:

```python
    items = []
    for index, entry in enumerate(raw, start=1):
        try:
            items.append(GoldenItemIn.model_validate(entry))
        except Exception as exc:
            raise ValueError(f"{path}: item {index}: {exc}") from exc
    return ingest_golden(session, items, dataset, version)
```

Replace the body of `load_traces` with:

```python
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TraceIn.model_validate(json.loads(line)))
            except Exception as exc:
                # Fail fast on a malformed file: a corrupt input is a problem you
                # want to hear about before half of it lands in the database. The
                # HTTP batch path deliberately behaves differently.
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    return ingest_traces(session, records, run)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_loaders.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/loaders.py tests/test_loaders.py
git commit -m "refactor: split record ingestion out of the file loaders"
```

---

### Task 3: Shared diff config

`compare` in the CLI builds a never-failing GateConfig. The API needs the same
thing. Extract it once.

**Files:**
- Modify: `ragmeter/gate/config.py`
- Modify: `ragmeter/cli.py`
- Test: `tests/test_gate_config.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate_config.py`:

```python
def test_diff_config_never_fails_any_metric():
    from ragmeter.gate.config import diff_config

    cfg = diff_config(["recall@3", "cost_usd", "faithfulness"], k=3)
    by_name = {r.name: r for r in cfg.metrics}
    # Quality metrics compare per question; everything else on an aggregate.
    assert by_name["recall@3"].is_paired is True
    assert by_name["faithfulness"].is_paired is True
    assert by_name["cost_usd"].is_paired is False
    # Limits are infinite: a diff reports, it never blocks.
    assert all(r.limit == float("inf") for r in cfg.metrics)
    assert cfg.fail_on_missing is False
    assert cfg.min_samples == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_gate_config.py -q`
Expected: `ImportError: cannot import name 'diff_config'`

- [ ] **Step 3: Add `diff_config` to `ragmeter/gate/config.py`**

Add `"diff_config"` to `__all__` and append:

```python
def diff_config(names, k: int) -> GateConfig:
    """A config that reports every metric and fails none of them.

    Used by `ragmeter compare` and by GET /v1/compare: both show the same paired
    diff the gate uses, without passing judgement.
    """
    from ragmeter.metrics.retrieval import metric_names

    paired = set(metric_names(k)) | {"faithfulness", "answer_relevance"}
    rules = tuple(
        MetricRule(name, max_drop=float("inf")) if name in paired
        else MetricRule(name, max_increase_pct=float("inf"))
        for name in sorted(names)
    )
    return GateConfig(metrics=rules, min_samples=0, fail_on_missing=False)
```

- [ ] **Step 4: Use it in `ragmeter/cli.py`**

In `compare_runs`, replace the rule-building block with:

```python
    names = set(base.all_values) | set(cand.all_values)
    result = compare(base, cand, diff_config(names, k))
```

Update the gate config import line to:

```python
from ragmeter.gate.config import GateConfigError, diff_config, load_gate_config
```

and delete the now-unused `from ragmeter.metrics.retrieval import metric_names` line.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_gate_config.py tests/test_gate_cli.py -q`
Expected: `22 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/gate/config.py ragmeter/cli.py tests/test_gate_config.py
git commit -m "refactor: share the diff config between CLI compare and the API"
```

---

### Task 4: The API

**Files:**
- Create: `ragmeter/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Install the API extra**

```bash
.venv/Scripts/python -m pip install -q -e ".[dev,api]"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_api.py`:

```python
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from ragmeter.metrics.cost import MODELS_URL

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = {"data": [{"id": "openai/gpt-4o-mini",
                     "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]}

GOLDEN = [
    {"question_id": "q1", "question": "return policy?", "relevant_chunk_ids": ["c1", "c2"]},
    {"question_id": "q2", "question": "shipping?", "relevant_chunk_ids": ["c5"]},
]

TRACES = [
    {"trace_id": "t1", "question_id": "q1", "question": "return policy?",
     "retrieved": [{"chunk_id": "c1", "rank": 1}, {"chunk_id": "c2", "rank": 2}],
     "answer": "30 days", "model": "openai/gpt-4o-mini",
     "prompt_tokens": 100, "completion_tokens": 10, "latency_ms": 200},
    {"trace_id": "t2", "question_id": "q2", "question": "shipping?",
     "retrieved": [{"chunk_id": "c9", "rank": 1}],
     "answer": "no idea", "model": "openai/gpt-4o-mini",
     "prompt_tokens": 100, "completion_tokens": 10, "latency_ms": 400},
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGMETER_DB_URL", f"sqlite:///{tmp_path / 'api.db'}")
    from ragmeter.api import app
    with TestClient(app) as c:
        yield c


def seed(client, run="baseline", traces=None):
    client.post("/v1/datasets", json={"name": "docs", "version": "v1", "items": GOLDEN})
    run_id = client.post("/v1/runs", json={"name": run}).json()["run_id"]
    client.post(f"/v1/runs/{run_id}/traces", json={"traces": traces or TRACES})
    return run_id


def evaluate(client, run_id, k=2):
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        return client.post(f"/v1/runs/{run_id}/evaluate",
                           json={"dataset": "docs", "version": "v1", "k": k})


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_create_run_returns_an_id(client):
    response = client.post("/v1/runs", json={"name": "r1", "git_sha": "abc"})
    assert response.status_code == 201
    assert response.json()["run_id"]


def test_creating_the_same_run_twice_returns_the_same_id(client):
    first = client.post("/v1/runs", json={"name": "r1"}).json()["run_id"]
    second = client.post("/v1/runs", json={"name": "r1"}).json()["run_id"]
    assert first == second


def test_ingest_traces(client):
    run_id = client.post("/v1/runs", json={"name": "r1"}).json()["run_id"]
    response = client.post(f"/v1/runs/{run_id}/traces", json={"traces": TRACES})
    assert response.status_code == 200
    assert response.json() == {"ingested": 2, "skipped": 0, "errors": []}


def test_reingesting_skips_rather_than_failing(client):
    run_id = client.post("/v1/runs", json={"name": "r1"}).json()["run_id"]
    client.post(f"/v1/runs/{run_id}/traces", json={"traces": TRACES})
    body = client.post(f"/v1/runs/{run_id}/traces", json={"traces": TRACES}).json()
    assert body["ingested"] == 0 and body["skipped"] == 2


def test_one_bad_trace_does_not_reject_the_good_ones(client):
    run_id = client.post("/v1/runs", json={"name": "r1"}).json()["run_id"]
    payload = {"traces": [TRACES[0], {"question": "no trace_id"}]}
    response = client.post(f"/v1/runs/{run_id}/traces", json=payload)
    # A single malformed record in a batch of 200 must not cost you the other 199.
    assert response.status_code == 200
    body = response.json()
    assert body["ingested"] == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["index"] == 1


def test_a_batch_where_everything_fails_is_a_422(client):
    run_id = client.post("/v1/runs", json={"name": "r1"}).json()["run_id"]
    response = client.post(f"/v1/runs/{run_id}/traces",
                           json={"traces": [{"question": "no id"}]})
    assert response.status_code == 422


def test_traces_for_an_unknown_run_is_404(client):
    response = client.post("/v1/runs/does-not-exist/traces", json={"traces": TRACES})
    assert response.status_code == 404


def test_upload_dataset(client):
    response = client.post("/v1/datasets",
                           json={"name": "docs", "version": "v1", "items": GOLDEN})
    assert response.status_code == 201
    assert response.json()["items"] == 2


def test_dataset_with_an_unlabelled_item_is_422(client):
    bad = [{"question_id": "q1", "question": "why?", "relevant_chunk_ids": []}]
    response = client.post("/v1/datasets",
                           json={"name": "docs", "version": "v1", "items": bad})
    assert response.status_code == 422


def test_evaluate_then_read_metrics(client):
    run_id = seed(client)
    assert evaluate(client, run_id).status_code == 202

    body = client.get(f"/v1/runs/{run_id}/metrics", params={"k": 2}).json()
    # t1 retrieved both relevant chunks, t2 retrieved none of its one.
    assert body["metrics"]["recall@2"]["mean"] == 0.5
    assert body["metrics"]["recall@2"]["n_measured"] == 2
    assert body["run"] == "baseline"


def test_metrics_reports_unmeasured_counts(client):
    run_id = seed(client)
    evaluate(client, run_id)
    body = client.get(f"/v1/runs/{run_id}/metrics", params={"k": 2}).json()
    assert "n_null" in body["metrics"]["recall@2"]


def test_metrics_before_evaluating_is_404(client):
    run_id = seed(client)
    assert client.get(f"/v1/runs/{run_id}/metrics", params={"k": 2}).status_code == 404


def test_evaluate_unknown_run_is_404(client):
    assert evaluate(client, "does-not-exist").status_code == 404


def test_evaluate_unknown_dataset_is_404(client):
    run_id = seed(client)
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        response = client.post(f"/v1/runs/{run_id}/evaluate",
                               json={"dataset": "nope", "version": "v1", "k": 2})
    assert response.status_code == 404


def test_compare_two_runs(client):
    base_id = seed(client, "baseline")
    worse = [dict(TRACES[0], trace_id="w1", retrieved=[{"chunk_id": "zz", "rank": 1}]),
             dict(TRACES[1], trace_id="w2")]
    cand_id = seed(client, "candidate", traces=worse)
    evaluate(client, base_id)
    evaluate(client, cand_id)

    body = client.get("/v1/compare",
                      params={"run": "candidate", "baseline": "baseline", "k": 2}).json()
    assert body["n_paired"] == 2
    recall = next(o for o in body["outcomes"] if o["name"] == "recall@2")
    # q1 went from 1.0 to 0.0, q2 stayed at 0.0 -> mean delta -0.5.
    assert recall["delta"] == -0.5
    assert recall["n_regressed"] == 1


def test_compare_never_fails(client):
    base_id = seed(client, "baseline")
    evaluate(client, base_id)
    body = client.get("/v1/compare",
                      params={"run": "baseline", "baseline": "baseline", "k": 2}).json()
    # A diff reports; only the gate blocks.
    assert body["passed"] is True


def test_compare_unknown_run_is_404(client):
    assert client.get("/v1/compare",
                      params={"run": "nope", "baseline": "nope", "k": 2}).status_code == 404
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.api'`

- [ ] **Step 4: Write the implementation**

Create `ragmeter/api.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -q`
Expected: `18 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/api.py tests/test_api.py
git commit -m "feat: thin FastAPI wrapper over the library"
```

---

### Task 5: `serve` command and Postgres

**Files:**
- Modify: `ragmeter/cli.py`
- Create: `docker-compose.yml`

- [ ] **Step 1: Add the `serve` command to `ragmeter/cli.py`**

Add before the `if __name__` block:

```python
@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the HTTP API."""
    try:
        import uvicorn
    except ImportError:
        typer.echo('error: the API extra is not installed; run: pip install -e ".[api]"',
                   err=True)
        raise typer.Exit(2)
    uvicorn.run("ragmeter.api:app", host=host, port=port, reload=reload)
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ragmeter
      POSTGRES_PASSWORD: ragmeter
      POSTGRES_DB: ragmeter
    ports:
      - "5433:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ragmeter"]
      interval: 2s
      timeout: 3s
      retries: 20
    volumes:
      - ragmeter_pgdata:/var/lib/postgresql/data

volumes:
  ragmeter_pgdata:
```

Port 5433 on the host so this never collides with a local Postgres on 5432.

- [ ] **Step 3: Verify against real Postgres**

SQLite has proven the logic; only Postgres can prove the schema — in particular
the `metadata` column name and the JSON columns.

```bash
docker compose up -d
.venv/Scripts/python -m pip install -q -e ".[postgres]"
export RAGMETER_DB_URL="postgresql+psycopg://ragmeter:ragmeter@localhost:5433/ragmeter"
.venv/Scripts/ragmeter.exe dataset load tests/fixtures/golden.yaml --name docs --version v1
.venv/Scripts/ragmeter.exe ingest tests/fixtures/traces.jsonl --run baseline
.venv/Scripts/ragmeter.exe ingest tests/fixtures/traces_v2.jsonl --run candidate
.venv/Scripts/ragmeter.exe eval --run baseline --dataset docs --version v1 --k 3
.venv/Scripts/ragmeter.exe eval --run candidate --dataset docs --version v1 --k 3
.venv/Scripts/ragmeter.exe gate --run candidate --baseline baseline --config gate.yaml --k 3
echo "exit: $?"
```

Expected: identical numbers to the SQLite run — `recall@3` 0.6000 to 0.4000,
`+1/-2`, exit code 1.

- [ ] **Step 4: Tear down**

```bash
docker compose down -v
```

- [ ] **Step 5: Commit**

```bash
git add ragmeter/cli.py docker-compose.yml
git commit -m "feat: serve command and Postgres compose file"
```

---

### Task 6: Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add an API section to `README.md`**

Insert before `## Trace format`:

````markdown
## HTTP API

```bash
pip install -e ".[api]"
ragmeter serve --port 8000
```

| method | path | |
|---|---|---|
| POST | `/v1/runs` | create or fetch a run by name |
| POST | `/v1/datasets` | upload a golden dataset |
| POST | `/v1/runs/{id}/traces` | batch ingest, idempotent on `trace_id` |
| POST | `/v1/runs/{id}/evaluate` | 202; runs in the background |
| GET | `/v1/runs/{id}/metrics` | aggregates with unmeasured counts |
| GET | `/v1/compare` | the paired diff, no verdict |
| GET | `/healthz` | |

A malformed trace in a batch does not reject the rest: valid records ingest and
the response carries an `errors` array naming the offending indexes. Only a
batch where nothing was usable returns 422.

The API stores and serves; it computes nothing. Every calculation lives in the
library, which is why `ragmeter gate` runs in CI with no server at all.

### Postgres

```bash
docker compose up -d
export RAGMETER_DB_URL="postgresql+psycopg://ragmeter:ragmeter@localhost:5433/ragmeter"
```
````

Change Status to:

```markdown
All five phases complete: retrieval metrics, LLM judge, regression gate, judge
calibration, HTTP API. Deferred by choice: the Next.js dashboard and
OpenTelemetry ingestion.
```

- [ ] **Step 2: Note the refusal behaviour under the judge section**

After the paragraph about judge failures, add:

```markdown
An answer that declines to answer ("I don't know") scores `faithfulness` as
blank, not zero. It asserts nothing, so there is nothing to check — and scoring
it zero would give an honest refusal the same mark as a confident fabrication,
teaching the gate to prefer the fabrication.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: HTTP API and refusal scoring"
```

---

## Definition of Done

- [ ] `.venv/Scripts/python -m pytest -q` reports 234 passed
- [ ] A refusal yields `faithfulness = None`, never `0.0`
- [ ] `PROMPT_VERSION` is `2`, so cached pre-fix verdicts are not reused
- [ ] One bad trace in a batch still ingests the good ones
- [ ] The full CLI workflow produces identical numbers on Postgres and SQLite
- [ ] `ragmeter/api.py` performs no arithmetic

## Out of Scope

The Next.js dashboard and OpenTelemetry ingestion stay deferred, as they have
been since the spec. No auth: this is a single-user tool.
