# ragmeter

Measures any RAG system. Does not build one.

Feed it traces (question, retrieved chunks, answer) and a labeled golden
dataset; it reports retrieval quality, faithfulness, cost, and latency, then
blocks a deploy that made any of them worse.

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

With the LLM judge:

```bash
export OPENROUTER_API_KEY=sk-or-...
ragmeter eval --run semantic-v2 --dataset docs --version v1 --k 5 --judge
```

`--judge` adds `faithfulness` (what fraction of the answer's claims the
retrieved chunks actually support) and `answer_relevance` (whether the answer
addresses the question at all). Responses are cached by prompt hash, so
re-running an evaluation costs nothing.

When a trace has no golden match, the judge also grades each retrieved chunk,
which yields `precision@k` without labels. It never yields recall — nothing can
measure a relevant chunk that was never retrieved.

If the judge fails, its metrics stay blank and the run reports the failure
count. They are never filled in with zero.

```
run: semantic-v2   k=3

metric                  mean         p50         p95    measured
----------------------------------------------------------------
recall@3              0.6000      1.0000      1.0000         5/6
precision@3           0.4333      0.5000      1.0000         5/6
mrr@3                 0.5000      0.5000      1.0000         5/6
ndcg@3                0.5262      0.6309      1.0000         5/6
cost_usd            9.90e-05    1.08e-04    1.44e-04         5/6
latency_ms          441.6667    380.0000    900.0000         6/6
faithfulness          0.5333      0.6667      1.0000         5/6
answer_relevance      0.5833      0.2500      1.0000         6/6
```

## Regression gate

```bash
ragmeter gate --run candidate --baseline baseline --config gate.yaml --k 3
```

```
gate: FAIL
  run=candidate  baseline=baseline  k=3  paired questions=5

metric                    baseline   candidate       delta     limit       +/-  verdict
------------------------------------------------------------------------------------------
recall@3                    0.6000      0.4000     -0.2000    0.0200     +1/-2  FAIL  (dropped 0.2000, limit is 0.0200)
cost_usd (mean)           9.90e-05    1.98e-04    100.0000   20.0000            FAIL  (rose 100.00%, limit is 20.00%)
```

Exit codes: `0` pass, `1` regression, `2` config or data error. CI needs to
tell "the model got worse" from "the tool broke".

```yaml
min_samples: 3
fail_on_missing: true
metrics:
  recall@3:    {max_drop: 0.02}                     # higher is better, per question
  cost_usd:    {stat: mean, max_increase_pct: 20}   # lower is better, aggregate
  latency_ms:  {stat: p95, max_increase_pct: 25}
```

Quality metrics are compared **per question**, using only the questions present
in both runs. The output shows `+improved/-regressed` counts next to the mean
delta, because a mean near zero can hide half the questions collapsing while
the other half improve — which is the change you most need to catch.

The gate **fails closed**: a missing metric, an unmeasurable one, or a judge
failure in either run blocks the deploy. Set `fail_on_missing: false` to
override, understanding that you are asking it to pass on things it could not
measure.

`ragmeter compare` shows the same diff with no verdict and no non-zero exit.

## Judge calibration

An LLM judge is only worth what its agreement with a human is worth. Measure it:

```bash
ragmeter label --run baseline --metric faithfulness --k 3 --labeler you
ragmeter calibration --run baseline --metric faithfulness --k 3
```

```
calibration: faithfulness   run=baseline   k=3   threshold=0.5
  pairs                 5
  agreement rate        0.6000
  Cohen's kappa         0.0000

  both yes              3
  judge yes / human no  0
  judge no / human yes  2
  both no               0

NOTE: agreement 0.60 but kappa 0.00. Most of that agreement is chance,
because the labels are lopsided. Quote the kappa.
```

**Agreement rate alone is a trap.** If 90% of your answers are genuinely
faithful, a judge that blindly says "faithful" every time scores 90% agreement
and is worthless. Cohen's kappa subtracts the agreement chance already explains
and gives that judge the 0.0 it deserves. Kappa is the number to publish.

Reading kappa: `< 0` worse than chance, `0.0-0.2` negligible, `0.2-0.4` fair,
`0.4-0.6` moderate, `0.6-0.8` substantial, `> 0.8` near-perfect.

`label` **hides the judge's score by default**. Seeing it first would anchor
your answer and make the agreement number circular. Pass `--show-judge` only
when reviewing after the fact.

Labelling is resumable: each run offers traces you have not judged yet, and
labels are per-labeler, so two people can rate the same traces.

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
| `OPENROUTER_API_KEY` | — (required for `--judge`) |
| `RAGMETER_JUDGE_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` |

SQLite for development, Postgres for running. Same code either way.

## Development

```bash
.venv/Scripts/python -m pytest -q
```

## Status

Phase 4 of 5. Next: the HTTP API.
See `docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`.
