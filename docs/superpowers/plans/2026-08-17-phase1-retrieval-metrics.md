# ragmeter Phase 1 — Retrieval Metrics Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working CLI that ingests RAG traces, loads a labeled golden dataset, and reports retrieval quality (recall/precision/MRR/NDCG@k) plus cost and latency — with zero LLM calls.

**Architecture:** Pure metric functions with no I/O at the bottom, SQLAlchemy persistence above them, a Typer CLI on top. Metrics never touch the database and the database never computes. This split is what lets Phase 3's regression gate run in CI without a server.

**Tech Stack:** Python 3.11, pydantic 2, SQLAlchemy 2, Typer, PyYAML, httpx, pytest, respx. SQLite for dev/test, Postgres for running. No numpy, no scipy.

**Spec:** `docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`

---

## Design Rule That Governs Every Task

`None` means **"not measurable"**. `0.0` means **"measured, scored zero"**.

Confusing them poisons every aggregate downstream: a `None` excluded from an
average is honest, a `0.0` silently included is a lie that makes a broken
pipeline look mediocre instead of broken. Every metric function returns `None`
rather than guessing, and every aggregate reports how many values it could not
measure.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | packaging, deps, pytest config, `ragmeter` entry point |
| `ragmeter/metrics/retrieval.py` | recall/precision/MRR/NDCG. Pure, no I/O, no imports from ragmeter |
| `ragmeter/metrics/aggregate.py` | mean/p50/p95 over lists containing `None`; null counting |
| `ragmeter/metrics/cost.py` | OpenRouter price catalog fetch, token→USD |
| `ragmeter/models.py` | pydantic input schemas + validation |
| `ragmeter/db.py` | SQLAlchemy tables, engine, session |
| `ragmeter/loaders.py` | golden YAML → DB, traces JSONL → DB |
| `ragmeter/runner.py` | joins traces to golden, computes metrics, writes Evaluation rows |
| `ragmeter/report.py` | renders a run summary as a text table |
| `ragmeter/cli.py` | Typer commands: `dataset load`, `ingest`, `eval` |
| `tests/fixtures/golden.yaml` | 6 labeled questions |
| `tests/fixtures/traces.jsonl` | 6 traces covering every edge case |

Dependency direction is strictly downward: `cli` → `runner` → `loaders`/`db`/`metrics`.
`metrics/retrieval.py` imports nothing from the package — it is the most-tested
and least-coupled file on purpose.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `ragmeter/__init__.py`
- Create: `ragmeter/metrics/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "ragmeter"
version = "0.1.0"
description = "Measure any RAG system: retrieval quality, faithfulness, cost, regressions."
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2",
    "sqlalchemy>=2",
    "httpx>=0.27",
    "typer>=0.12",
    "pyyaml>=6",
]

[project.optional-dependencies]
api = ["fastapi>=0.115", "uvicorn>=0.30"]
postgres = ["psycopg[binary]>=3.2"]
dev = ["pytest>=8", "respx>=0.21"]
ragas = ["ragas>=0.2"]

[project.scripts]
ragmeter = "ragmeter.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["ragmeter*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["ragas: cross-check against RAGAS (requires the ragas extra)"]
```

- [ ] **Step 2: Create the empty package files**

```bash
mkdir -p ragmeter/metrics tests/fixtures
touch ragmeter/__init__.py ragmeter/metrics/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create the virtualenv and install**

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -q -e ".[dev]"
```

Expected: installs without error, ending in `Successfully installed ... ragmeter-0.1.0 ...`

- [ ] **Step 4: Verify pytest runs**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `no tests ran` (exit code 5). This confirms discovery works before any test exists.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml ragmeter tests
git commit -m "chore: scaffold ragmeter package"
```

---

### Task 2: Retrieval metrics

The highest-value file in the project and the only one with zero dependencies.
Written test-first with hand-computed expected values.

**Files:**
- Create: `ragmeter/metrics/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval.py`:

```python
"""Every expected value here is computed by hand, not by running the code.

A test that asserts what the implementation happens to produce tests nothing.
"""
import pytest

from ragmeter.metrics.retrieval import (
    evaluate_retrieval,
    metric_names,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_perfect_retrieval():
    r = ["a", "b", "c"]
    g = ["a", "b", "c"]
    assert recall_at_k(r, g, 3) == 1.0
    assert precision_at_k(r, g, 3) == 1.0
    assert mrr_at_k(r, g, 3) == 1.0
    assert ndcg_at_k(r, g, 3) == 1.0


def test_nothing_relevant_retrieved():
    r = ["x", "y", "z"]
    g = ["a"]
    assert recall_at_k(r, g, 3) == 0.0
    assert precision_at_k(r, g, 3) == 0.0
    assert mrr_at_k(r, g, 3) == 0.0
    assert ndcg_at_k(r, g, 3) == 0.0


def test_partial_hit_at_rank_two():
    # retrieved: x(1) a(2) y(3); relevant: {a, b}
    # recall    = |{a}| / |{a,b}|          = 0.5
    # precision = |{a}| / 3                = 0.333...
    # mrr       = 1/2                      = 0.5
    # dcg       = 1/log2(3)                = 0.6309297535714574
    # idcg      = 1/log2(2) + 1/log2(3)    = 1.6309297535714574
    # ndcg      = 0.6309297535714574 / 1.6309297535714574 = 0.38685280723454163
    r = ["x", "a", "y"]
    g = ["a", "b"]
    assert recall_at_k(r, g, 3) == 0.5
    assert precision_at_k(r, g, 3) == pytest.approx(1 / 3)
    assert mrr_at_k(r, g, 3) == 0.5
    assert ndcg_at_k(r, g, 3) == pytest.approx(0.38685280723454163)


def test_duplicate_chunk_ids_are_deduplicated():
    # Without dedup, [a, a] at k=2 scores recall 0.5. With dedup, [a, b] scores 1.0.
    r = ["a", "a", "b"]
    g = ["a", "b"]
    assert recall_at_k(r, g, 2) == 1.0
    assert precision_at_k(r, g, 2) == 1.0


def test_fewer_retrieved_than_k():
    # retrieved: a(1); relevant: {a, b}; k=5
    # precision divides by 1 actually retrieved, not by k -- otherwise a system
    # that returns one perfect chunk is punished for its restraint.
    # dcg  = 1/log2(2)               = 1.0
    # idcg = 1/log2(2) + 1/log2(3)   = 1.6309297535714574
    # ndcg = 0.6131471927654584
    r = ["a"]
    g = ["a", "b"]
    assert recall_at_k(r, g, 5) == 0.5
    assert precision_at_k(r, g, 5) == 1.0
    assert mrr_at_k(r, g, 5) == 1.0
    assert ndcg_at_k(r, g, 5) == pytest.approx(0.6131471927654584)


def test_empty_retrieval_is_zero_recall_but_undefined_precision():
    # Recall is 0.0: it genuinely found none of the relevant chunks.
    # Precision is None: there is no denominator, so there is nothing to measure.
    r = []
    g = ["a"]
    assert recall_at_k(r, g, 5) == 0.0
    assert precision_at_k(r, g, 5) is None
    assert mrr_at_k(r, g, 5) == 0.0
    assert ndcg_at_k(r, g, 5) == 0.0


def test_empty_relevant_set_is_unmeasurable_not_zero():
    r = ["a", "b"]
    g = []
    assert recall_at_k(r, g, 2) is None
    assert precision_at_k(r, g, 2) is None
    assert mrr_at_k(r, g, 2) is None
    assert ndcg_at_k(r, g, 2) is None


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(["a"], ["a"], 0)


def test_evaluate_retrieval_returns_all_four_keyed_by_k():
    out = evaluate_retrieval(["x", "a", "y"], ["a", "b"], 3)
    assert out == {
        "recall@3": 0.5,
        "precision@3": pytest.approx(1 / 3),
        "mrr@3": 0.5,
        "ndcg@3": pytest.approx(0.38685280723454163),
    }


def test_metric_names_matches_evaluate_retrieval_keys():
    # If these ever drift, unmatched traces get metric keys that no aggregate
    # can line up with matched ones.
    assert metric_names(7) == list(evaluate_retrieval(["a"], ["a"], 7).keys())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_retrieval.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.metrics.retrieval'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/metrics/retrieval.py`:

```python
"""Retrieval quality metrics. Pure functions: no I/O, no package imports.

None means "not measurable". 0.0 means "measured, scored zero". These are not
interchangeable -- see the design rule at the top of the plan.
"""

from math import log2

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "evaluate_retrieval",
    "metric_names",
]


def metric_names(k: int) -> list[str]:
    """The metric keys produced at this k.

    Callers that need the key names without having data to evaluate (an
    unmatched trace, for instance) use this instead of re-deriving the naming
    convention and drifting out of sync with evaluate_retrieval.
    """
    return [f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}"]


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _top_k(retrieved: list[str], k: int) -> list[str]:
    """Deduplicate keeping the first occurrence, then truncate to k.

    A retriever that returns the same chunk three times has retrieved one chunk.
    Counting it three times inflates every metric.
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id in retrieved:
        if chunk_id not in seen:
            seen.add(chunk_id)
            out.append(chunk_id)
        if len(out) == k:
            break
    return out


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Fraction of relevant chunks that made it into the top k."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    return len(rel & set(_top_k(retrieved, k))) / len(rel)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Fraction of the top k that is relevant.

    Divides by however many were actually retrieved, not by k, so a retriever
    returning one perfect chunk scores 1.0 rather than 1/k.
    """
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    top = _top_k(retrieved, k)
    if not top:
        return None
    return len(rel & set(top)) / len(top)


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Reciprocal rank of the first relevant chunk. 0.0 if none in the top k."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    for rank, chunk_id in enumerate(_top_k(retrieved, k), start=1):
        if chunk_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Normalized discounted cumulative gain with binary relevance."""
    _check_k(k)
    rel = set(relevant)
    if not rel:
        return None
    top = _top_k(retrieved, k)
    dcg = sum(1.0 / log2(i + 1) for i, c in enumerate(top, start=1) if c in rel)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, min(k, len(rel)) + 1))
    if idcg == 0:
        return None
    return dcg / idcg


def evaluate_retrieval(
    retrieved: list[str], relevant: list[str], k: int
) -> dict[str, float | None]:
    """All four metrics, keyed with the k baked into the name.

    The k is in the key because comparing recall@5 against recall@10 across runs
    is a silent apples-to-oranges bug the regression gate must not be able to make.
    """
    return {
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"mrr@{k}": mrr_at_k(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_retrieval.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/metrics/retrieval.py tests/test_retrieval.py
git commit -m "feat: retrieval metrics with None for unmeasurable cases"
```

---

### Task 3: Aggregation that counts what it could not measure

**Files:**
- Create: `ragmeter/metrics/aggregate.py`
- Test: `tests/test_aggregate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_aggregate.py`:

```python
import pytest

from ragmeter.metrics.aggregate import percentile, summarize


def test_summarize_excludes_nulls_and_reports_them():
    # mean of [1.0, 3.0] is 2.0. If the None were counted as 0.0 the mean would
    # be 1.333 -- the exact silent corruption this function exists to prevent.
    out = summarize([1.0, None, 3.0])
    assert out["n"] == 3
    assert out["n_measured"] == 2
    assert out["n_null"] == 1
    assert out["mean"] == 2.0


def test_summarize_all_null():
    out = summarize([None, None])
    assert out["n_measured"] == 0
    assert out["n_null"] == 2
    assert out["mean"] is None
    assert out["p50"] is None
    assert out["p95"] is None


def test_summarize_empty():
    out = summarize([])
    assert out == {"n": 0, "n_measured": 0, "n_null": 0,
                   "mean": None, "p50": None, "p95": None}


def test_percentile_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5    # ceil(0.50 * 10) = 5 -> values[4]
    assert percentile(values, 95) == 10   # ceil(0.95 * 10) = 10 -> values[9]
    assert percentile(values, 100) == 10


def test_percentile_single_value():
    assert percentile([42], 95) == 42


def test_percentile_empty_is_none():
    assert percentile([], 50) is None


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError, match="p must be in"):
        percentile([1, 2], 101)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_aggregate.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.metrics.aggregate'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/metrics/aggregate.py`:

```python
"""Aggregation over values that may be unmeasurable.

Every summary reports n_null alongside the mean. An average over 12 of 200
questions is not the same claim as an average over 200, and the reader must be
able to tell them apart.
"""

import math
from statistics import fmean

__all__ = ["percentile", "summarize"]


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. No interpolation, no dependencies."""
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    if not values:
        return None
    ordered = sorted(values)
    index = max(1, math.ceil(p / 100 * len(ordered))) - 1
    return ordered[index]


def summarize(values: list[float | None]) -> dict[str, float | int | None]:
    """Mean/p50/p95 over the measurable values, plus how many were not."""
    measured = [v for v in values if v is not None]
    return {
        "n": len(values),
        "n_measured": len(measured),
        "n_null": len(values) - len(measured),
        "mean": fmean(measured) if measured else None,
        "p50": percentile(measured, 50),
        "p95": percentile(measured, 95),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_aggregate.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/metrics/aggregate.py tests/test_aggregate.py
git commit -m "feat: aggregation that reports unmeasurable counts"
```

---

### Task 4: Cost from the OpenRouter price catalog

**Files:**
- Create: `ragmeter/metrics/cost.py`
- Test: `tests/test_cost.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cost.py`:

```python
import httpx
import pytest
import respx

from ragmeter.metrics.cost import MODELS_URL, compute_cost, fetch_prices

CATALOG = {
    "data": [
        {"id": "openai/gpt-4o-mini",
         "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}},
        {"id": "nvidia/nemotron-3-ultra-550b-a55b:free",
         "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "broken/no-pricing"},
        {"id": "broken/variable-pricing",
         "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}


@respx.mock
def test_fetch_prices_parses_and_skips_unusable_entries():
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
    prices = fetch_prices()
    assert prices["openai/gpt-4o-mini"] == (0.00000015, 0.0000006)
    assert prices["nvidia/nemotron-3-ultra-550b-a55b:free"] == (0.0, 0.0)
    # Missing pricing and negative "variable" pricing are both dropped rather
    # than turned into a number that would silently understate real spend.
    assert "broken/no-pricing" not in prices
    assert "broken/variable-pricing" not in prices


@respx.mock
def test_fetch_prices_raises_on_http_error():
    respx.get(MODELS_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_prices()


def test_supplied_cost_wins_over_computed():
    prices = {"m": (1.0, 1.0)}
    assert compute_cost("m", 10, 10, prices, supplied=0.5) == 0.5


def test_computed_from_tokens():
    prices = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}
    # 1000 * 0.00000015 + 500 * 0.0000006 = 0.00015 + 0.0003 = 0.00045
    assert compute_cost("openai/gpt-4o-mini", 1000, 500, prices) == pytest.approx(0.00045)


def test_unknown_model_is_none_not_zero():
    # A zero here would make an unpriced model look free and quietly break any
    # cost regression threshold built on top of it.
    assert compute_cost("who/knows", 1000, 500, {}) is None


def test_missing_tokens_is_none():
    prices = {"m": (1.0, 1.0)}
    assert compute_cost("m", None, 500, prices) is None
    assert compute_cost("m", 1000, None, prices) is None


def test_missing_model_is_none():
    assert compute_cost(None, 1000, 500, {"m": (1.0, 1.0)}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_cost.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.metrics.cost'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/metrics/cost.py`:

```python
"""Token counts to USD, priced from OpenRouter's public catalog.

The catalog endpoint needs no API key, so prices stay current without the user
maintaining a hand-written table that silently goes stale.
"""

import httpx

MODELS_URL = "https://openrouter.ai/api/v1/models"

__all__ = ["MODELS_URL", "fetch_prices", "compute_cost"]


def fetch_prices(client: httpx.Client | None = None) -> dict[str, tuple[float, float]]:
    """Return {model_id: (prompt_usd_per_token, completion_usd_per_token)}.

    Entries without usable pricing are dropped, never defaulted to zero.
    """
    owned = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.get(MODELS_URL)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owned:
            client.close()

    prices: dict[str, tuple[float, float]] = {}
    for model in payload.get("data", []):
        pricing = model.get("pricing") or {}
        try:
            prompt = float(pricing["prompt"])
            completion = float(pricing["completion"])
        except (KeyError, TypeError, ValueError):
            continue
        # OpenRouter uses -1 for models whose price is not fixed.
        if prompt < 0 or completion < 0:
            continue
        prices[model["id"]] = (prompt, completion)
    return prices


def compute_cost(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prices: dict[str, tuple[float, float]],
    supplied: float | None = None,
) -> float | None:
    """Client-supplied cost wins. Anything unpriceable is None, never 0.0."""
    if supplied is not None:
        return supplied
    if model is None or prompt_tokens is None or completion_tokens is None:
        return None
    if model not in prices:
        return None
    prompt_price, completion_price = prices[model]
    return prompt_tokens * prompt_price + completion_tokens * completion_price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cost.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/metrics/cost.py tests/test_cost.py
git commit -m "feat: cost from OpenRouter catalog, None for unpriceable models"
```

---

### Task 5: Pydantic input schemas

**Files:**
- Create: `ragmeter/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from ragmeter.models import Chunk, GoldenItemIn, TraceIn


def test_trace_minimal_fields():
    t = TraceIn(trace_id="t1", question="why?")
    assert t.trace_id == "t1"
    assert t.retrieved == []
    assert t.answer == ""
    assert t.question_id is None
    assert t.metadata == {}


def test_trace_with_chunks():
    t = TraceIn(
        trace_id="t1",
        question_id="q1",
        question="why?",
        retrieved=[{"chunk_id": "c1", "text": "because", "score": 0.9, "rank": 1}],
        answer="because",
        model="openai/gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=350,
    )
    assert t.retrieved[0] == Chunk(chunk_id="c1", text="because", score=0.9, rank=1)
    assert t.chunk_ids() == ["c1"]


def test_trace_requires_id_and_question():
    with pytest.raises(ValidationError):
        TraceIn(question="why?")
    with pytest.raises(ValidationError):
        TraceIn(trace_id="t1")


def test_golden_item_valid():
    g = GoldenItemIn(question_id="q1", question="why?", relevant_chunk_ids=["c1", "c2"])
    assert g.relevant_chunk_ids == ["c1", "c2"]
    assert g.reference_answer is None


def test_golden_item_rejects_empty_relevant_ids():
    # An unlabeled golden item cannot produce retrieval metrics. Rejecting it at
    # load time beats discovering a run of silent Nones after evaluation.
    with pytest.raises(ValidationError, match="relevant_chunk_ids must not be empty"):
        GoldenItemIn(question_id="q1", question="why?", relevant_chunk_ids=[])


def test_golden_item_deduplicates_relevant_ids():
    g = GoldenItemIn(question_id="q1", question="why?", relevant_chunk_ids=["c1", "c1", "c2"])
    assert g.relevant_chunk_ids == ["c1", "c2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.models'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/models.py`:

```python
"""Input schemas. This is the trace contract any RAG system must emit."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = ["Chunk", "TraceIn", "GoldenItemIn"]


class Chunk(BaseModel):
    chunk_id: str
    text: str = ""
    score: float | None = None
    rank: int | None = None


class TraceIn(BaseModel):
    """One end-to-end RAG query.

    question_id links the trace to a golden item. Traces without one are
    production traces: they get cost and latency, but no retrieval metrics.
    """

    trace_id: str
    question: str
    question_id: str | None = None
    retrieved: list[Chunk] = Field(default_factory=list)
    answer: str = ""
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def chunk_ids(self) -> list[str]:
        """Retrieved chunk ids in rank order, as the metric functions want them."""
        return [c.chunk_id for c in self.retrieved]


class GoldenItemIn(BaseModel):
    question_id: str
    question: str
    relevant_chunk_ids: list[str]
    reference_answer: str | None = None

    @field_validator("relevant_chunk_ids")
    @classmethod
    def _non_empty_and_unique(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError(
                "relevant_chunk_ids must not be empty; an unlabeled item "
                "cannot produce retrieval metrics"
            )
        return list(dict.fromkeys(value))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/models.py tests/test_models.py
git commit -m "feat: trace and golden item schemas"
```

---

### Task 6: Database layer

**Files:**
- Create: `ragmeter/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db.py`:

```python
import pytest
from sqlalchemy.exc import IntegrityError

from ragmeter.db import Evaluation, GoldenItem, Run, Trace, init_db, make_engine, make_session


@pytest.fixture()
def session():
    engine = make_engine("sqlite://")  # in-memory
    init_db(engine)
    with make_session(engine)() as s:
        yield s


def test_run_and_trace_roundtrip(session):
    run = Run(name="semantic-v2", git_sha="abc123", config={"chunker": "semantic"})
    session.add(run)
    session.flush()
    session.add(Trace(
        trace_id="t1", run_id=run.run_id, question_id="q1", question="why?",
        retrieved=[{"chunk_id": "c1", "rank": 1}], answer="because",
        model="openai/gpt-4o-mini", prompt_tokens=10, completion_tokens=2,
        latency_ms=120, meta={"env": "dev"},
    ))
    session.commit()

    loaded = session.get(Trace, "t1")
    assert loaded.question_id == "q1"
    assert loaded.retrieved == [{"chunk_id": "c1", "rank": 1}]
    assert loaded.meta == {"env": "dev"}
    assert loaded.run_id == run.run_id


def test_run_name_is_unique(session):
    session.add(Run(name="dup"))
    session.commit()
    session.add(Run(name="dup"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_golden_item_composite_key(session):
    # Same question_id in two dataset versions must coexist -- that is how you
    # relabel a golden set without destroying the old one the baseline used.
    session.add(GoldenItem(dataset="docs", version="v1", question_id="q1",
                           question="why?", relevant_chunk_ids=["c1"]))
    session.add(GoldenItem(dataset="docs", version="v2", question_id="q1",
                           question="why?", relevant_chunk_ids=["c9"]))
    session.commit()
    assert session.get(GoldenItem, ("docs", "v1", "q1")).relevant_chunk_ids == ["c1"]
    assert session.get(GoldenItem, ("docs", "v2", "q1")).relevant_chunk_ids == ["c9"]


def test_evaluation_defaults_to_skipped_judge(session):
    run = Run(name="r")
    session.add(run)
    session.flush()
    session.add(Trace(trace_id="t1", run_id=run.run_id, question="why?"))
    session.flush()
    ev = Evaluation(trace_id="t1", k=5, metrics={"recall@5": 1.0})
    session.add(ev)
    session.commit()
    assert ev.judge_status == "skipped"
    assert ev.evaluation_id is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.db'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/db.py`:

```python
"""Persistence. SQLite for development and tests, Postgres for running.

Schema is created with create_all. Alembic arrives with the first schema change
against data worth keeping -- see the spec's deferred list.
"""

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

__all__ = [
    "Base", "Run", "Trace", "GoldenItem", "Evaluation", "ModelPrice",
    "make_engine", "init_db", "make_session",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    git_sha: Mapped[str | None] = mapped_column(String(40), default=None)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Trace(Base):
    __tablename__ = "traces"

    trace_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    question_id: Mapped[str | None] = mapped_column(String(200), index=True, default=None)
    question: Mapped[str] = mapped_column(Text)
    retrieved: Mapped[list] = mapped_column(JSON, default=list)
    answer: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str | None] = mapped_column(String(200), default=None)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    cost_usd: Mapped[float | None] = mapped_column(Float, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    # Attribute is `meta` because `metadata` is reserved by SQLAlchemy's declarative base.
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GoldenItem(Base):
    __tablename__ = "golden_items"

    dataset: Mapped[str] = mapped_column(String(200), primary_key=True)
    version: Mapped[str] = mapped_column(String(50), primary_key=True)
    question_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    relevant_chunk_ids: Mapped[list] = mapped_column(JSON)
    reference_answer: Mapped[str | None] = mapped_column(Text, default=None)


class Evaluation(Base):
    __tablename__ = "evaluations"

    evaluation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.trace_id"), index=True)
    k: Mapped[int] = mapped_column(Integer)
    dataset: Mapped[str | None] = mapped_column(String(200), default=None)
    dataset_version: Mapped[str | None] = mapped_column(String(50), default=None)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    claims: Mapped[list | None] = mapped_column(JSON, default=None)
    chunk_judgments: Mapped[list | None] = mapped_column(JSON, default=None)
    judge_model: Mapped[str | None] = mapped_column(String(200), default=None)
    judge_status: Mapped[str] = mapped_column(String(20), default="skipped")
    judge_error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ModelPrice(Base):
    __tablename__ = "model_prices"

    model_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    prompt_price: Mapped[float] = mapped_column(Float)
    completion_price: Mapped[float] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


DEFAULT_DB_URL = "sqlite:///ragmeter.db"


def make_engine(url: str | None = None):
    return create_engine(url or os.environ.get("RAGMETER_DB_URL", DEFAULT_DB_URL))


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def make_session(engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/db.py tests/test_db.py
git commit -m "feat: SQLAlchemy schema for runs, traces, golden items, evaluations"
```

---

### Task 7: Loaders and fixtures

**Files:**
- Create: `ragmeter/loaders.py`
- Create: `tests/fixtures/golden.yaml`
- Create: `tests/fixtures/traces.jsonl`
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Create the fixtures**

Create `tests/fixtures/golden.yaml`:

```yaml
- question_id: q1
  question: What is the return policy?
  relevant_chunk_ids: [c1, c2]
- question_id: q2
  question: How long does shipping take?
  relevant_chunk_ids: [c5]
- question_id: q3
  question: Do you ship to the UAE?
  relevant_chunk_ids: [c7, c8]
- question_id: q4
  question: What payment methods are accepted?
  relevant_chunk_ids: [c10]
- question_id: q5
  question: Can I cancel an order?
  relevant_chunk_ids: [c12, c13, c14]
```

Create `tests/fixtures/traces.jsonl` (one JSON object per line, no trailing commas):

```jsonl
{"trace_id": "t1", "question_id": "q1", "question": "What is the return policy?", "retrieved": [{"chunk_id": "c1", "rank": 1}, {"chunk_id": "c2", "rank": 2}, {"chunk_id": "c3", "rank": 3}], "answer": "30 days.", "model": "openai/gpt-4o-mini", "prompt_tokens": 800, "completion_tokens": 40, "latency_ms": 420}
{"trace_id": "t2", "question_id": "q2", "question": "How long does shipping take?", "retrieved": [{"chunk_id": "c9", "rank": 1}, {"chunk_id": "c5", "rank": 2}], "answer": "3-5 days.", "model": "openai/gpt-4o-mini", "prompt_tokens": 600, "completion_tokens": 30, "latency_ms": 380}
{"trace_id": "t3", "question_id": "q3", "question": "Do you ship to the UAE?", "retrieved": [{"chunk_id": "c7", "rank": 1}, {"chunk_id": "c7", "rank": 2}, {"chunk_id": "c8", "rank": 3}], "answer": "Yes.", "model": "openai/gpt-4o-mini", "prompt_tokens": 700, "completion_tokens": 20, "latency_ms": 500}
{"trace_id": "t4", "question_id": "q4", "question": "What payment methods are accepted?", "retrieved": [], "answer": "I don't know.", "model": "openai/gpt-4o-mini", "prompt_tokens": 200, "completion_tokens": 10, "latency_ms": 150}
{"trace_id": "t5", "question_id": "q5", "question": "Can I cancel an order?", "retrieved": [{"chunk_id": "c99", "rank": 1}], "answer": "No idea.", "model": "unknown/model", "prompt_tokens": 300, "completion_tokens": 15, "latency_ms": 900}
{"trace_id": "t6", "question": "Untracked production question", "retrieved": [{"chunk_id": "c1", "rank": 1}], "answer": "Something.", "model": "openai/gpt-4o-mini", "prompt_tokens": 500, "completion_tokens": 25, "latency_ms": 300}
```

These six traces cover, in order: a clean partial hit, a hit at rank 2,
duplicate chunk ids, empty retrieval, a total miss with an unpriced model, and a
production trace with no `question_id`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_loaders.py`:

```python
from pathlib import Path

import pytest

from ragmeter.db import GoldenItem, Run, Trace, init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with make_session(engine)() as s:
        yield s


def test_load_golden(session):
    n = load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    session.commit()
    assert n == 5
    assert session.get(GoldenItem, ("docs", "v1", "q1")).relevant_chunk_ids == ["c1", "c2"]


def test_load_golden_is_idempotent(session):
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    session.commit()
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    session.commit()
    assert session.query(GoldenItem).count() == 5


def test_load_golden_rejects_empty_relevant_ids(session, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- question_id: q1\n  question: why?\n  relevant_chunk_ids: []\n")
    with pytest.raises(ValueError, match="relevant_chunk_ids must not be empty"):
        load_golden(session, bad, dataset="docs", version="v1")


def test_load_traces(session):
    run = get_or_create_run(session, "baseline")
    result = load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    assert result == {"ingested": 6, "skipped": 0}
    assert session.query(Trace).count() == 6
    assert session.get(Trace, "t6").question_id is None


def test_load_traces_is_idempotent_on_trace_id(session):
    run = get_or_create_run(session, "baseline")
    load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    result = load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    # Re-ingesting is a no-op, not an error: reruns of a CI job must be safe.
    assert result == {"ingested": 0, "skipped": 6}
    assert session.query(Trace).count() == 6


def test_load_traces_reports_bad_line_number(session, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"trace_id": "ok", "question": "fine"}\n{"question": "no id"}\n')
    run = get_or_create_run(session, "r")
    with pytest.raises(ValueError, match="line 2"):
        load_traces(session, bad, run)


def test_get_or_create_run_reuses_by_name(session):
    a = get_or_create_run(session, "same")
    session.commit()
    b = get_or_create_run(session, "same")
    assert a.run_id == b.run_id
    assert session.query(Run).count() == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_loaders.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.loaders'`

- [ ] **Step 4: Write the implementation**

Create `ragmeter/loaders.py`:

```python
"""Reading golden datasets and traces off disk into the database.

Validation happens here, at the trust boundary. Everything downstream may
assume the data is well-formed.
"""

import json
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from ragmeter.db import GoldenItem, Run, Trace
from ragmeter.models import GoldenItemIn, TraceIn

__all__ = ["get_or_create_run", "load_golden", "load_traces"]


def get_or_create_run(
    session: Session, name: str, git_sha: str | None = None, config: dict | None = None
) -> Run:
    run = session.query(Run).filter_by(name=name).one_or_none()
    if run is None:
        run = Run(name=name, git_sha=git_sha, config=config or {})
        session.add(run)
        session.flush()
    return run


def load_golden(session: Session, path: Path, dataset: str, version: str) -> int:
    """Load a golden YAML file. Re-loading the same file overwrites in place."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of golden items, got {type(raw).__name__}")

    count = 0
    for index, entry in enumerate(raw, start=1):
        try:
            item = GoldenItemIn.model_validate(entry)
        except Exception as exc:
            raise ValueError(f"{path}: item {index}: {exc}") from exc
        session.merge(GoldenItem(
            dataset=dataset,
            version=version,
            question_id=item.question_id,
            question=item.question,
            relevant_chunk_ids=item.relevant_chunk_ids,
            reference_answer=item.reference_answer,
        ))
        count += 1
    return count


def load_traces(session: Session, path: Path, run: Run) -> dict[str, int]:
    """Load a JSONL trace file. Idempotent on trace_id: duplicates are skipped."""
    ingested = 0
    skipped = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                trace_in = TraceIn.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc

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

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_loaders.py -q`
Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/loaders.py tests/test_loaders.py tests/fixtures
git commit -m "feat: golden YAML and trace JSONL loaders with fixtures"
```

---

### Task 8: Evaluation runner

**Files:**
- Create: `ragmeter/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runner.py`:

```python
from pathlib import Path

import pytest

from ragmeter.db import Evaluation, init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


@pytest.fixture()
def loaded():
    engine = make_engine("sqlite://")
    init_db(engine)
    session = make_session(engine)()
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    run = get_or_create_run(session, "baseline")
    load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    yield session
    session.close()


def test_evaluate_writes_one_evaluation_per_trace(loaded):
    summary = evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    assert loaded.query(Evaluation).count() == 6
    assert summary["n_traces"] == 6
    assert summary["n_matched"] == 5
    assert summary["n_unmatched"] == 1


def test_metrics_for_a_partial_hit(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    # retrieved c1,c2,c3; relevant c1,c2 -> recall 1.0, precision 2/3, mrr 1.0
    assert ev.metrics["recall@3"] == 1.0
    assert ev.metrics["precision@3"] == pytest.approx(2 / 3)
    assert ev.metrics["mrr@3"] == 1.0
    assert ev.dataset == "docs"
    assert ev.judge_status == "skipped"


def test_duplicate_chunks_do_not_inflate(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t3").one()
    # retrieved c7,c7,c8 dedupes to c7,c8; relevant c7,c8 -> recall 1.0
    assert ev.metrics["recall@3"] == 1.0
    assert ev.metrics["precision@3"] == 1.0


def test_empty_retrieval_is_zero_recall_none_precision(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t4").one()
    assert ev.metrics["recall@3"] == 0.0
    assert ev.metrics["precision@3"] is None


def test_unmatched_trace_gets_no_retrieval_metrics(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t6").one()
    assert ev.metrics["recall@3"] is None
    assert ev.metrics["precision@3"] is None
    assert ev.dataset is None
    # Cost and latency still apply: they need no ground truth.
    assert ev.metrics["cost_usd"] == pytest.approx(500 * 0.00000015 + 25 * 0.0000006)
    assert ev.metrics["latency_ms"] == 300


def test_unpriced_model_yields_none_cost(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t5").one()
    assert ev.metrics["cost_usd"] is None


def test_reevaluating_replaces_rather_than_duplicates(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    assert loaded.query(Evaluation).filter_by(trace_id="t1").count() == 1


def test_different_k_coexist(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    evaluate_run(loaded, "baseline", "docs", "v1", k=1, prices=PRICES)
    # k=1 and k=3 are different measurements of the same trace, both worth keeping.
    assert loaded.query(Evaluation).filter_by(trace_id="t1").count() == 2


def test_unknown_run_raises(loaded):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        evaluate_run(loaded, "nope", "docs", "v1", k=3, prices=PRICES)


def test_unknown_dataset_raises(loaded):
    with pytest.raises(ValueError, match="no golden items"):
        evaluate_run(loaded, "baseline", "missing", "v1", k=3, prices=PRICES)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_runner.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.runner'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/runner.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_runner.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/runner.py tests/test_runner.py
git commit -m "feat: evaluation runner joining traces to golden items"
```

---

### Task 9: Report rendering

**Files:**
- Create: `ragmeter/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
from pathlib import Path

import pytest

from ragmeter.db import init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.report import render_summary, summarize_run
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


@pytest.fixture()
def evaluated():
    engine = make_engine("sqlite://")
    init_db(engine)
    session = make_session(engine)()
    load_golden(session, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    run = get_or_create_run(session, "baseline")
    load_traces(session, FIXTURES / "traces.jsonl", run)
    session.commit()
    evaluate_run(session, "baseline", "docs", "v1", k=3, prices=PRICES)
    yield session
    session.close()


def test_summarize_run_reports_null_counts(evaluated):
    out = summarize_run(evaluated, "baseline", k=3)
    # 6 traces; only 5 have a golden match, so one recall value is unmeasurable.
    assert out["recall@3"]["n"] == 6
    assert out["recall@3"]["n_measured"] == 5
    assert out["recall@3"]["n_null"] == 1
    # t4 retrieved nothing, so precision is unmeasurable there too.
    assert out["precision@3"]["n_null"] == 2


def test_summarize_run_mean_excludes_nulls(evaluated):
    out = summarize_run(evaluated, "baseline", k=3)
    # t1=1.0, t2=1.0, t3=1.0, t4=0.0, t5=0.0 -> mean 0.6 over 5 measured values
    assert out["recall@3"]["mean"] == pytest.approx(0.6)


def test_render_summary_shows_measured_counts(evaluated):
    text = render_summary(summarize_run(evaluated, "baseline", k=3), "baseline", k=3)
    assert "baseline" in text
    assert "recall@3" in text
    # The count must be visible: a mean over 5 of 6 is a different claim than
    # a mean over 6, and the reader has to be able to see which one this is.
    assert "5/6" in text


def test_summarize_unknown_run_raises(evaluated):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        summarize_run(evaluated, "nope", k=3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_report.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.report'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/report.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_report.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/report.py tests/test_report.py
git commit -m "feat: run summary reporting with measured counts"
```

---

### Task 10: CLI

**Files:**
- Create: `ragmeter/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ragmeter.cli import app
from ragmeter.metrics.cost import MODELS_URL

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = {"data": [{"id": "openai/gpt-4o-mini",
                     "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]}

runner = CliRunner()


@pytest.fixture()
def db_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("RAGMETER_DB_URL", url)
    return url


def test_full_workflow(db_url):
    result = runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                                 "--name", "docs", "--version", "v1"])
    assert result.exit_code == 0, result.output
    assert "5" in result.output

    result = runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    assert result.exit_code == 0, result.output
    assert "6" in result.output

    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        result = runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                                     "--version", "v1", "--k", "3"])
    assert result.exit_code == 0, result.output
    assert "recall@3" in result.output
    assert "5/6" in result.output


def test_ingest_twice_reports_skips(db_url):
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    result = runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    assert result.exit_code == 0, result.output
    assert "skipped 6" in result.output


def test_eval_unknown_run_exits_nonzero(db_url):
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        result = runner.invoke(app, ["eval", "--run", "nope", "--dataset", "docs",
                                     "--version", "v1"])
    assert result.exit_code == 2
    assert "no run named" in result.output


def test_eval_survives_price_fetch_failure(db_url):
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(503))
        result = runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                                     "--version", "v1", "--k", "3"])
    # Retrieval quality does not depend on pricing. Losing the price catalog must
    # cost you the cost column, not the whole evaluation.
    assert result.exit_code == 0, result.output
    assert "could not fetch model prices" in result.output
    assert "recall@3" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.cli'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/cli.py`:

```python
"""Command line interface.

Exit codes: 0 success, 1 reserved for the Phase 3 regression gate, 2 usage or
data error. The gate needs 1 to itself so CI can tell "worse" from "broken".
"""

from pathlib import Path

import typer

from ragmeter.db import init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.metrics.cost import fetch_prices
from ragmeter.report import render_summary, summarize_run
from ragmeter.runner import evaluate_run

app = typer.Typer(no_args_is_help=True, help="Measure any RAG system.")
dataset_app = typer.Typer(no_args_is_help=True, help="Manage golden datasets.")
app.add_typer(dataset_app, name="dataset")


def _session():
    engine = make_engine()
    init_db(engine)
    return make_session(engine)()


@dataset_app.command("load")
def dataset_load(
    path: Path = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Option(..., "--name", help="Dataset name."),
    version: str = typer.Option("v1", "--version", help="Dataset version."),
) -> None:
    """Load a golden YAML file."""
    session = _session()
    try:
        count = load_golden(session, path, dataset=name, version=version)
        session.commit()
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()
    typer.echo(f"loaded {count} golden items into {name}@{version}")


@app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True),
    run: str = typer.Option(..., "--run", help="Run name. Created if absent."),
    git_sha: str | None = typer.Option(None, "--git-sha"),
) -> None:
    """Ingest a JSONL trace file into a run."""
    session = _session()
    try:
        run_row = get_or_create_run(session, run, git_sha=git_sha)
        result = load_traces(session, path, run_row)
        session.commit()
    except ValueError as exc:
        session.rollback()
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()
    typer.echo(f"ingested {result['ingested']}, skipped {result['skipped']} into run {run!r}")


@app.command("eval")
def evaluate(
    run: str = typer.Option(..., "--run"),
    dataset: str = typer.Option(..., "--dataset"),
    version: str = typer.Option("v1", "--version"),
    k: int = typer.Option(5, "--k", min=1),
) -> None:
    """Compute retrieval, cost, and latency metrics for a run."""
    try:
        prices = fetch_prices()
    except Exception as exc:
        # Pricing is an enrichment, not a prerequisite. Say so loudly and continue.
        typer.echo(f"warning: could not fetch model prices ({exc}); cost will be blank", err=True)
        prices = {}

    session = _session()
    try:
        result = evaluate_run(session, run, dataset, version, k=k, prices=prices)
        summary = summarize_run(session, run, k=k)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()

    typer.echo(render_summary(summary, run, k))
    typer.echo("")
    typer.echo(
        f"{result['n_traces']} traces, {result['n_matched']} matched to golden, "
        f"{result['n_unmatched']} unmatched"
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -q`
Expected: `4 passed`

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `59 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/cli.py tests/test_cli.py
git commit -m "feat: CLI for dataset load, ingest, and eval"
```

---

### Task 11: End-to-end verification and README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the real CLI against the fixtures**

```bash
RAGMETER_DB_URL="sqlite:///smoke.db" .venv/Scripts/ragmeter dataset load tests/fixtures/golden.yaml --name docs --version v1
```

Expected: `loaded 5 golden items into docs@v1`

- [ ] **Step 2: Ingest and evaluate**

```bash
RAGMETER_DB_URL="sqlite:///smoke.db" .venv/Scripts/ragmeter ingest tests/fixtures/traces.jsonl --run baseline
```

Expected: `ingested 6, skipped 0 into run 'baseline'`

```bash
RAGMETER_DB_URL="sqlite:///smoke.db" .venv/Scripts/ragmeter eval --run baseline --dataset docs --version v1 --k 3
```

Expected: a table containing `recall@3` with `mean 0.6000` and `measured 5/6`,
followed by `6 traces, 5 matched to golden, 1 unmatched`. This hits the live
OpenRouter catalog, so `cost_usd` should show a real number rather than `-`.

- [ ] **Step 3: Delete the smoke database**

```bash
rm -f smoke.db
```

- [ ] **Step 4: Write `README.md`**

````markdown
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

## Configuration

| variable | default |
|---|---|
| `RAGMETER_DB_URL` | `sqlite:///ragmeter.db` |

## Status

Phase 1 of 5. Next: LLM judge (faithfulness, answer relevance), then the
regression gate, judge calibration, and the HTTP API. See
`docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`.
````

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README for phase 1"
```

---

## Definition of Done

- [ ] `.venv/Scripts/python -m pytest -q` reports 59 passed (retrieval 10, aggregate 7, cost 7, models 6, db 4, loaders 7, runner 10, report 4, cli 4)
- [ ] The Task 11 smoke run prints `recall@3` mean `0.6000` over `5/6` measured
- [ ] `ragmeter --help` lists `dataset`, `ingest`, and `eval`
- [ ] No metric ever returns `0.0` where it means "could not measure"
- [ ] `ragmeter/metrics/retrieval.py` imports nothing from `ragmeter`

## Out of Scope for This Plan

LLM judge, regression gate, calibration, HTTP API, Postgres deployment, and the
dashboard. Each gets its own plan. See the spec's build order.
