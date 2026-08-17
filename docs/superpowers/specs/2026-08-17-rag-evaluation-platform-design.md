# ragmeter — RAG Evaluation & Observability Platform

**Date:** 2026-08-17
**Status:** Approved design

## Purpose

A tool that measures any RAG system. It does not build RAG. It receives traces
(question, retrieved chunks, final answer) and computes retrieval quality,
faithfulness, answer relevance, cost, and latency — then blocks deploys that
make things worse.

The user has no RAG system yet. The tool is built against a documented trace
contract and tested with fixture traces.

## Goals

- Retrieval recall / precision / MRR / NDCG @ k against a labeled golden dataset
- Faithfulness: is the answer actually supported by the retrieved chunks
- Answer relevance: does the answer address the question
- Cost and latency per query
- Regression gate: fail CI when a new run is worse than its baseline
- Judge calibration: measure judge/human agreement honestly (kappa, not raw rate)

## Non-Goals

- Next.js dashboard (separate spec, later)
- OpenTelemetry ingestion (nothing emits spans yet; added later as an adapter)
- Multi-tenancy, auth, billing
- Building or hosting a RAG pipeline

## Shape

Library core + thin FastAPI wrapper + CLI.

All computation lives in a plain Python package. FastAPI stores and serves.
The CLI runs the regression gate in CI. **The gate must work with no server
running** — this constraint drives the split.

## Data Model

Five entities.

### Run
The unit of comparison. One version/configuration of a RAG system.

| field | type | notes |
|---|---|---|
| `run_id` | uuid | PK |
| `name` | str | unique, e.g. `semantic-chunking-v2` |
| `git_sha` | str? | |
| `config` | json | arbitrary; whatever distinguishes this version |
| `created_at` | timestamp | |

### Trace
One end-to-end RAG query.

| field | type | notes |
|---|---|---|
| `trace_id` | str | PK, client-supplied. Ingestion is idempotent on this |
| `run_id` | uuid | FK |
| `question_id` | str? | links to `GoldenItem`. Absent for production traces |
| `question` | str | |
| `retrieved` | json | `[{chunk_id, text, score?, rank}]`, ordered |
| `answer` | str | |
| `model` | str | the model the RAG used, e.g. `openai/gpt-4o-mini` |
| `prompt_tokens` | int? | |
| `completion_tokens` | int? | |
| `cost_usd` | float? | client-supplied; wins over computed cost |
| `latency_ms` | int? | |
| `metadata` | json | free-form |
| `created_at` | timestamp | |

### GoldenItem
Ground truth. Belongs to a named, versioned dataset.

| field | type | notes |
|---|---|---|
| `dataset` | str | PK part |
| `version` | str | PK part |
| `question_id` | str | PK part |
| `question` | str | |
| `relevant_chunk_ids` | json | `[str]` |
| `reference_answer` | str? | not used by any metric in this spec; stored for later |

### Evaluation
Metrics for one trace. **Separate from Trace** so re-evaluation does not
require re-ingestion.

| field | type | notes |
|---|---|---|
| `evaluation_id` | uuid | PK |
| `trace_id` | str | FK |
| `k` | int | the k these metrics were computed at |
| `dataset`, `dataset_version` | str? | null when no golden match |
| `metrics` | json | `{metric_name: float\|null}` |
| `claims` | json? | faithfulness audit trail |
| `chunk_judgments` | json? | per-chunk relevance from the judge |
| `judge_model` | str? | |
| `judge_status` | enum | `ok` \| `skipped` \| `failed` |
| `judge_error` | str? | |
| `created_at` | timestamp | |

### HumanLabel
Fuel for calibration.

| field | type | notes |
|---|---|---|
| `label_id` | uuid | PK |
| `trace_id` | str | FK |
| `metric` | str | e.g. `faithfulness` |
| `value` | float | 0.0 or 1.0 — the CLI collects binary judgments only |
| `labeler` | str | |
| `created_at` | timestamp | |

Plus two internal tables: `judge_cache` and `model_price`.

## Metrics

### Retrieval — pure functions, no I/O, no LLM

Let `R` = set of `relevant_chunk_ids` from the golden item.
Let `D` = retrieved `chunk_id`s in rank order, **deduplicated keeping the first
occurrence**.

- `recall@k = |R ∩ D[:k]| / |R|` — **`None` when `|R| == 0`**, never 0.0
- `precision@k = |R ∩ D[:k]| / min(k, |D|)` — `None` when `|D| == 0`
- `mrr@k = 1 / rank` of the first relevant item in `D[:k]`, 1-indexed;
  `0.0` when none found; `None` when `|R| == 0`
- `ndcg@k` with binary gains:
  - `DCG = Σ(i=1..min(k,|D|)) rel_i / log2(i+1)`
  - `IDCG = Σ(i=1..min(k,|R|)) 1 / log2(i+1)`
  - `ndcg = DCG / IDCG`, `None` when `IDCG == 0`

`None` propagates: a `None` metric is excluded from aggregates, and its count
is reported alongside every aggregate.

These four edge cases — empty relevant set, empty retrieval, fewer than k
retrieved, duplicate chunk ids — are the required unit test cases.

### Cost & latency

`cost_usd = prompt_tokens × price_prompt + completion_tokens × price_completion`

Prices come from `GET https://openrouter.ai/api/v1/models` (public, no auth),
cached in `model_price(model_id, prompt_price, completion_price, fetched_at)`.

Precedence: client-supplied `cost_usd` > computed > `None`.
Unknown model → `None` plus a warning; never a silent zero.

Latency aggregates: mean, p50, p95.

### Judge — one LLM call per metric

**`faithfulness`** — decompose the answer into atomic claims, judge each against
the retrieved chunks.

Response schema:
```json
{"claims": [{"claim": "...", "supported": true, "chunk_ids": ["c1"], "reason": "..."}]}
```
`score = supported_count / total_claims`; `None` when the claim list is empty.
**The claim list is persisted** — it is the audit trail that makes the number
trustworthy.

**`answer_relevance`** — direct rubric.
```json
{"score": 4, "reason": "..."}
```
Integer 1–5, normalized to `(score - 1) / 4`.

RAGAS computes this by generating questions from the answer and comparing
embeddings. That requires an embedding model and a second dependency tree for
a weaker signal. Rejected.

**`chunk_relevance`** — the fallback for traces with no golden item.
```json
{"judgments": [{"chunk_id": "c1", "relevant": true}]}
```
Yields `precision@k` without ground truth. **It cannot yield recall** — you
cannot measure what was never retrieved. Recall stays `None` on this path, and
this limitation is stated in the output, not hidden.

### Judge client

- `POST https://openrouter.ai/api/v1/chat/completions`
- Auth: `OPENROUTER_API_KEY`
- Model: `RAGMETER_JUDGE_MODEL`, default `nvidia/nemotron-3-ultra-550b-a55b:free`
  (verified present in the OpenRouter catalog on 2026-08-17: 1M context, $0/$0)
- `temperature = 0`
- Structured output via `response_format: json_schema` when the model supports
  it; otherwise a tolerant extractor (strip fences, find the outermost JSON
  object) plus one retry with a "return only JSON" nudge
- **Cache**: `judge_cache(key PK, response_json, created_at)` where
  `key = sha256(model | prompt_version | prompt_text)`. `prompt_version` is a
  constant bumped by hand when a prompt changes, so edits invalidate the cache.
- **Retry**: on 429/5xx, exponential backoff 1/2/4/8/16s, max 5 attempts,
  honoring `Retry-After`. The free tier has daily rate limits; this is the
  main source of flakiness.
- On final failure: raise `JudgeError`. The Evaluation row records
  `judge_status='failed'` with the error. **Never a silent zero.**

## Calibration

Raw agreement rate is misleading: if 90% of cases are "good", a judge that
always says "good" scores 90%. So report both:

- `agreement_rate` — judge (binarized at 0.5, configurable) vs human label
- `cohens_kappa = (po - pe) / (1 - pe)` — agreement above chance. **This is the
  honest number.** Hand-implemented, ~10 lines, no dependency.

Edge case: `pe == 1` (all labels identical) → kappa is undefined, return `None`
with an explanatory message. Do not return 0.

Human labels are collected by `ragmeter label`: prints question, chunks, and
answer to the terminal, takes y/n. Binary only. Graded labels and Spearman
correlation are deliberately deferred — they would pull in scipy for a signal
we do not yet have.

## Regression Gate

```bash
ragmeter gate --run new --baseline prev --config gate.yaml
```

```yaml
min_samples: 50
fail_on_missing: true
metrics:
  recall@5:       {max_drop: 0.02}
  precision@5:    {max_drop: 0.05}
  faithfulness:   {max_drop: 0.03}
  cost_usd_mean:  {max_increase_pct: 20}
  latency_p95_ms: {max_increase_pct: 25}
```

**Paired comparison.** Only `question_id`s present in *both* runs are compared.
Reports `n_paired`, `n_improved`, `n_regressed`, `mean_delta` per metric — not
just an aggregate average, because an average hides a redistribution where half
the questions got much worse.

Fails when: `mean_delta < -max_drop`, or `n_paired < min_samples`, or
(when `fail_on_missing`) a configured metric is absent or its judge failed.

**The gate fails closed.** A missing metric is a failure, not a pass. A gate
that passes when it cannot measure is worse than no gate.

Exit codes: `0` pass, `1` regression, `2` config/data error.

```
# ponytail: threshold comparison, no significance test. If metric noise causes
# false failures, upgrade to a paired bootstrap CI over per-question deltas.
```

## HTTP API

Thin. It stores and serves; it does not compute.

| method | path | notes |
|---|---|---|
| POST | `/v1/runs` | → `{run_id}` |
| POST | `/v1/runs/{run_id}/traces` | batch; idempotent on `trace_id`; returns `{ingested, skipped, errors[]}` with per-item errors |
| POST | `/v1/datasets` | upload a golden dataset |
| POST | `/v1/runs/{run_id}/evaluate` | `BackgroundTasks`; returns 202 |
| GET | `/v1/runs/{run_id}/metrics` | aggregates + null counts |
| GET | `/v1/compare?run=X&baseline=Y` | the same paired diff the gate uses |
| GET | `/healthz` | |

```
# ponytail: BackgroundTasks means single-process evaluation. If runs get large
# or need to survive restarts, move to a real queue.
```

## CLI

```
ragmeter dataset load golden.yaml --name docs --version v1
ragmeter ingest traces.jsonl --run semantic-v2
ragmeter eval --run semantic-v2 --dataset docs --k 5 [--judge/--no-judge]
ragmeter label --run semantic-v2 --metric faithfulness --limit 50
ragmeter gate --run semantic-v2 --baseline fixed-v1 --config gate.yaml
ragmeter compare --run semantic-v2 --baseline fixed-v1
ragmeter calibration --run semantic-v2 --metric faithfulness
ragmeter serve
```

## Layout

```
ragmeter/
  models.py              pydantic schemas
  db.py                  SQLAlchemy 2.0 tables + session
  metrics/retrieval.py   pure functions — recall, precision, mrr, ndcg
  metrics/cost.py        token → usd, OpenRouter price table
  metrics/aggregate.py   mean/p50/p95, null counting
  judge/client.py        OpenRouter, cache, retry
  judge/prompts.py       prompt templates + PROMPT_VERSION
  judge/scoring.py       faithfulness, answer_relevance, chunk_relevance
  calibration.py         agreement rate, Cohen's kappa
  gate.py                paired comparison, thresholds, exit codes
  api.py                 FastAPI
  cli.py                 Typer
tests/
  fixtures/traces.jsonl, golden.yaml
docker-compose.yml       Postgres
```

## Configuration

| env var | default |
|---|---|
| `RAGMETER_DB_URL` | `sqlite:///ragmeter.db` |
| `OPENROUTER_API_KEY` | — (required for `--judge`) |
| `RAGMETER_JUDGE_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` |

SQLite for development and tests, Postgres for running. Same SQLAlchemy code.
JSON columns use `JSON` (portable across both).

## Dependencies

Runtime: `fastapi`, `uvicorn`, `sqlalchemy>=2`, `pydantic>=2`, `httpx`,
`typer`, `pyyaml`. Postgres extra: `psycopg[binary]`.

Dev: `pytest`, `respx`.

**No numpy, no scipy** — every computation here is `math.log2` and arithmetic.

**RAGAS is not a dependency.** It pulls langchain and datasets. It goes in a
`[ragas]` extra used by exactly one comparison test that verifies our
faithfulness and retrieval numbers match RAGAS on a fixture. Reference, not
runtime.

**Typer, not argparse.** Eight subcommands with options is roughly 60 lines of
argparse boilerplate against 15 with Typer. That exceeds "a few lines".

## Error Handling

| failure | behavior |
|---|---|
| Malformed trace in a batch | 422 with per-item errors; valid items still ingest |
| Duplicate `trace_id` | skipped, counted, not an error |
| Judge HTTP failure after retries | `judge_status='failed'`, error stored, gate fails closed |
| Judge returns unparseable JSON | one retry with a stricter nudge, then `failed` |
| Unknown model for pricing | `cost_usd = None` + warning; never 0.0 |
| No golden item for a `question_id` | retrieval metrics `None`; judge path still runs |
| `OPENROUTER_API_KEY` missing with `--judge` | fail immediately with a clear message, before any work |

## Testing

| file | covers |
|---|---|
| `test_retrieval.py` | parametrized, hand-computed expected values; the four edge cases above |
| `test_cost.py` | precedence rules, unknown model, missing tokens |
| `test_judge.py` | mocked httpx: cache hit, retry/backoff, malformed JSON, final failure |
| `test_calibration.py` | kappa against worked examples incl. the `pe == 1` case |
| `test_gate.py` | pass, fail-on-drop, fail-on-missing, below-min-samples |
| `test_api.py` | TestClient: ingestion idempotency, batch partial failure |
| `test_ragas_parity.py` | marked `ragas`, skipped unless the extra is installed |

## Build Order

Each phase is independently useful.

1. **models + db + retrieval metrics + `eval` CLI** — working value with zero LLM calls
2. **judge** — client, cache, retry, faithfulness, answer_relevance
3. **gate** — paired comparison, config, CI exit codes
4. **calibration** — `label` CLI, agreement, kappa
5. **API** — FastAPI, docker-compose, Postgres

## Deferred

| item | add when |
|---|---|
| Next.js dashboard | the CLI output stops being enough |
| OpenTelemetry ingestion | something actually emits GenAI spans |
| Alembic migrations | the first schema change against real stored data |
| Celery/Redis queue | evaluation runs outgrow one process |
| Spearman / graded human labels | binary labels prove insufficient |
| Significance testing in the gate | threshold comparison produces false failures |
| Auth / multi-tenancy | someone other than the author uses it |

## Assumptions

- The trace contract is defined by this tool, not by an existing producer.
  The RAG system will be written to emit it.
- `chunk_id` is stable across runs for a given corpus. If a chunking strategy
  change alters chunk ids, golden `relevant_chunk_ids` must be relabeled —
  this is inherent to comparing chunking strategies, not a tool defect.
- Human labels are binary. Graded labels are a later change.
- The package name `ragmeter` is provisional.
