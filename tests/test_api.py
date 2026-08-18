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
