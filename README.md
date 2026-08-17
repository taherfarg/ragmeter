# ragmeter

Measures any RAG system. Does not build one.

Feed it traces (question, retrieved chunks, answer) and a labeled golden
dataset; it reports retrieval quality, cost, and latency. Phase 1 makes no LLM
calls at all.

## Install

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
```

## Use

```bash
ragmeter dataset load golden.yaml --name docs --version v1
ragmeter ingest traces.jsonl --run semantic-v2
ragmeter eval --run semantic-v2 --dataset docs --version v1 --k 5
```

```
run: semantic-v2   k=3

metric                  mean         p50         p95    measured
----------------------------------------------------------------
recall@3              0.6000      1.0000      1.0000         5/6
precision@3           0.5417      0.5000      1.0000         4/6
mrr@3                 0.5000      0.5000      1.0000         5/6
ndcg@3                0.5262      0.6309      1.0000         5/6
cost_usd            9.90e-05    1.08e-04    1.44e-04         5/6
latency_ms          441.6667    380.0000    900.0000         6/6
```

## Trace format

One JSON object per line:

```json
{"trace_id": "t1", "question_id": "q1", "question": "What is the return policy?",
 "retrieved": [{"chunk_id": "c1", "rank": 1}], "answer": "30 days.",
 "model": "openai/gpt-4o-mini", "prompt_tokens": 800, "completion_tokens": 40,
 "latency_ms": 420}
```

`question_id` is optional. Without it a trace still gets cost and latency, but
no retrieval metrics — there is nothing to compare against.

## Golden format

```yaml
- question_id: q1
  question: What is the return policy?
  relevant_chunk_ids: [c1, c2]
```

`relevant_chunk_ids` must not be empty. An unlabeled item cannot produce
retrieval metrics, so it is rejected at load time rather than silently
producing blanks.

## Reading the output

`measured` shows how many traces produced a value for that metric. A mean over
`5/6` is a different claim than a mean over `6/6`.

A blank (`-`) means **not measurable**, not zero. Retrieval metrics are blank
for traces with no golden match; `cost_usd` is blank when the model is absent
from OpenRouter's price catalog.

Prices are fetched live from OpenRouter's public catalog, which needs no API
key. If that fetch fails, evaluation still runs and the cost column goes blank.

## Chunk id stability

`chunk_id` must be stable across the runs you compare. Changing a chunking
strategy usually changes chunk ids, which means the golden set has to be
relabeled before the comparison means anything. This is inherent to comparing
chunking strategies, not a limitation of the tool — but it is the thing that
bites first.

## Configuration

| variable | default |
|---|---|
| `RAGMETER_DB_URL` | `sqlite:///ragmeter.db` |

SQLite for development, Postgres for running. Same code either way.

## Development

```bash
.venv/Scripts/python -m pytest -q
```

## Status

Phase 1 of 5. Next: LLM judge (faithfulness, answer relevance), then the
regression gate, judge calibration, and the HTTP API. See
`docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`.
