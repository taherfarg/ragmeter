# ragmeter Phase 2 — LLM Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add faithfulness and answer-relevance scoring via an OpenRouter LLM judge, with caching, retry, and honest failure reporting — wired into `ragmeter eval --judge`.

**Architecture:** A transport layer (`judge/client.py`) that knows nothing about RAG, prompt templates in their own module, and scoring functions that turn parsed JSON into numbers. The runner calls scoring; scoring calls the client; the client handles HTTP and cache. A judge failure marks the evaluation failed rather than producing a number.

**Tech Stack:** httpx, the existing SQLAlchemy schema, pytest + respx. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`

---

## Findings That Shape This Plan

Checked against the live OpenRouter catalog on 2026-08-17:

`nvidia/nemotron-3-ultra-550b-a55b:free` reports
`supported_parameters: include_reasoning, max_tokens, reasoning, reasoning_effort, seed, temperature, tool_choice, tools, top_p`.

**No `response_format`, no `structured_outputs`.** The spec's "structured output via
`json_schema` when the model supports it" therefore resolves to the fallback
path for this model. JSON is constrained by prompt and recovered by a tolerant
extractor. Tool-calling *is* supported and is the documented upgrade path, but
it is model-specific and this judge must stay model-agnostic.

It is also a reasoning model. `include_reasoning` stays off so reasoning traces
do not land in the content we parse, and `max_tokens` is set high enough that a
long claim list is not truncated mid-JSON.

## Design Rules

**A judge that fails is not a judge that scored zero.** Every failure path
records `judge_status='failed'` with the error text. The Phase 3 gate treats
that as a gate failure. Nothing in this plan may return a number it did not
measure.

**Faithfulness with no sources is unmeasurable, not zero.** An answer retrieved
from nothing cannot be graded against nothing. Score is `None`.

## File Structure

| File | Responsibility |
|---|---|
| `ragmeter/db.py` (modify) | add the `judge_cache` table |
| `ragmeter/judge/__init__.py` | package marker |
| `ragmeter/judge/parsing.py` | recover a JSON object from messy model output |
| `ragmeter/judge/prompts.py` | templates + `PROMPT_VERSION` |
| `ragmeter/judge/client.py` | HTTP, retry/backoff, cache. Knows nothing about RAG |
| `ragmeter/judge/scoring.py` | faithfulness, answer relevance, chunk relevance |
| `ragmeter/runner.py` (modify) | optional judge pass |
| `ragmeter/cli.py` (modify) | `--judge` flag |
| `tests/fixtures/traces.jsonl` (modify) | add `text` to chunks — faithfulness needs source text |

---

### Task 1: Add chunk text to the fixtures

Faithfulness grades an answer against source *text*. The Phase 1 fixtures carry
only chunk ids, so nothing can be judged against them.

**Files:**
- Modify: `tests/fixtures/traces.jsonl`

- [ ] **Step 1: Rewrite the fixture with chunk text**

Replace the whole file with (one JSON object per line):

```jsonl
{"trace_id": "t1", "question_id": "q1", "question": "What is the return policy?", "retrieved": [{"chunk_id": "c1", "text": "Items may be returned within 30 days of delivery.", "rank": 1}, {"chunk_id": "c2", "text": "Returned items must be unused and in original packaging.", "rank": 2}, {"chunk_id": "c3", "text": "Our warehouse operates Sunday through Thursday.", "rank": 3}], "answer": "You can return items within 30 days if unused.", "model": "openai/gpt-4o-mini", "prompt_tokens": 800, "completion_tokens": 40, "latency_ms": 420}
{"trace_id": "t2", "question_id": "q2", "question": "How long does shipping take?", "retrieved": [{"chunk_id": "c9", "text": "Gift wrapping is available at checkout.", "rank": 1}, {"chunk_id": "c5", "text": "Standard shipping takes 3 to 5 business days.", "rank": 2}], "answer": "Shipping takes 3-5 business days.", "model": "openai/gpt-4o-mini", "prompt_tokens": 600, "completion_tokens": 30, "latency_ms": 380}
{"trace_id": "t3", "question_id": "q3", "question": "Do you ship to the UAE?", "retrieved": [{"chunk_id": "c7", "text": "We ship to all GCC countries including the UAE.", "rank": 1}, {"chunk_id": "c7", "text": "We ship to all GCC countries including the UAE.", "rank": 2}, {"chunk_id": "c8", "text": "Deliveries to Dubai and Abu Dhabi arrive within 2 days.", "rank": 3}], "answer": "Yes, we ship to the UAE, and delivery to Dubai takes 2 days. Shipping is free.", "model": "openai/gpt-4o-mini", "prompt_tokens": 700, "completion_tokens": 20, "latency_ms": 500}
{"trace_id": "t4", "question_id": "q4", "question": "What payment methods are accepted?", "retrieved": [], "answer": "I don't know.", "model": "openai/gpt-4o-mini", "prompt_tokens": 200, "completion_tokens": 10, "latency_ms": 150}
{"trace_id": "t5", "question_id": "q5", "question": "Can I cancel an order?", "retrieved": [{"chunk_id": "c99", "text": "Our head office is located in Sharjah.", "rank": 1}], "answer": "No idea.", "model": "unknown/model", "prompt_tokens": 300, "completion_tokens": 15, "latency_ms": 900}
{"trace_id": "t6", "question": "Untracked production question", "retrieved": [{"chunk_id": "c1", "text": "Items may be returned within 30 days of delivery.", "rank": 1}], "answer": "Something.", "model": "openai/gpt-4o-mini", "prompt_tokens": 500, "completion_tokens": 25, "latency_ms": 300}
```

Note `t3`: its answer ends with "Shipping is free", which no chunk supports.
That is the deliberate unfaithful claim the live smoke test in Task 8 looks for.

- [ ] **Step 2: Confirm Phase 1 tests still pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `59 passed`. No Phase 1 test asserts on chunk text, so adding it must
change nothing.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/traces.jsonl
git commit -m "test: add chunk text to fixtures for faithfulness judging"
```

---

### Task 2: JSON recovery from messy model output

**Files:**
- Create: `ragmeter/judge/__init__.py`
- Create: `ragmeter/judge/parsing.py`
- Test: `tests/test_judge_parsing.py`

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p ragmeter/judge && touch ragmeter/judge/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_judge_parsing.py`:

```python
import pytest

from ragmeter.judge.parsing import ParseError, extract_json


def test_plain_json():
    assert extract_json('{"score": 4}') == {"score": 4}


def test_fenced_json():
    assert extract_json('```json\n{"score": 4}\n```') == {"score": 4}


def test_fenced_without_language_tag():
    assert extract_json('```\n{"score": 4}\n```') == {"score": 4}


def test_prose_before_and_after():
    text = 'Here is my assessment:\n{"score": 4}\nHope that helps!'
    assert extract_json(text) == {"score": 4}


def test_nested_objects_survive():
    text = 'blah {"claims": [{"claim": "x", "supported": true}]} blah'
    assert extract_json(text) == {"claims": [{"claim": "x", "supported": True}]}


def test_leading_whitespace_and_newlines():
    assert extract_json('\n\n  {"score": 1}  \n') == {"score": 1}


def test_no_json_at_all_raises():
    with pytest.raises(ParseError, match="no JSON object"):
        extract_json("I refuse to answer.")


def test_malformed_json_raises():
    with pytest.raises(ParseError, match="not valid JSON"):
        extract_json('{"score": }')


def test_empty_string_raises():
    with pytest.raises(ParseError, match="no JSON object"):
        extract_json("")


def test_top_level_array_is_rejected():
    # Every judge prompt asks for an object. An array means the model ignored
    # the shape, and guessing which key it meant is worse than failing.
    with pytest.raises(ParseError, match="no JSON object"):
        extract_json("[1, 2, 3]")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_judge_parsing.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.judge.parsing'`

- [ ] **Step 4: Write the implementation**

Create `ragmeter/judge/parsing.py`:

```python
"""Recover a JSON object from model output that may be wrapped in prose or fences.

The judge model does not support response_format, so the shape is requested in
the prompt and enforced here. Recovery is deliberately narrow: it finds one
top-level object or it fails. Guessing is how a judge silently starts scoring
something other than what you asked it.
"""

import json

__all__ = ["ParseError", "extract_json"]


class ParseError(ValueError):
    """The model's output did not contain a usable JSON object."""


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ParseError(f"no JSON object in model output: {text[:200]!r}")

    # ponytail: outermost-braces slice, not a real brace-matching scan. A model
    # that emits two sibling objects would produce garbage here -- switch to a
    # depth counter if that ever shows up in judge_error.
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ParseError(f"not valid JSON: {exc}; got {candidate[:200]!r}") from exc

    if not isinstance(parsed, dict):
        raise ParseError(f"no JSON object, got {type(parsed).__name__}")
    return parsed
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_judge_parsing.py -q`
Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/judge tests/test_judge_parsing.py
git commit -m "feat: tolerant JSON extraction from judge output"
```

---

### Task 3: Prompt templates

**Files:**
- Create: `ragmeter/judge/prompts.py`
- Test: `tests/test_judge_prompts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge_prompts.py`:

```python
from ragmeter.judge.prompts import (
    PROMPT_VERSION,
    answer_relevance_prompt,
    chunk_relevance_prompt,
    faithfulness_prompt,
)

CHUNKS = [{"chunk_id": "c1", "text": "Returns accepted within 30 days."},
          {"chunk_id": "c2", "text": "Items must be unused."}]


def test_faithfulness_includes_question_chunks_and_answer():
    p = faithfulness_prompt("What is the return policy?", CHUNKS, "30 days.")
    assert "What is the return policy?" in p
    assert "Returns accepted within 30 days." in p
    assert "[c1]" in p and "[c2]" in p
    assert "30 days." in p
    assert '"claims"' in p


def test_faithfulness_labels_chunks_by_id_not_position():
    # The model must cite the real chunk id, otherwise the stored audit trail
    # cannot be traced back to a source.
    p = faithfulness_prompt("q", [{"chunk_id": "xyz-9", "text": "t"}], "a")
    assert "[xyz-9]" in p


def test_faithfulness_handles_chunks_without_text():
    p = faithfulness_prompt("q", [{"chunk_id": "c1"}], "a")
    assert "[c1]" in p


def test_answer_relevance_excludes_correctness():
    p = answer_relevance_prompt("Why?", "Because.")
    assert "Why?" in p
    assert "Because." in p
    # Relevance and faithfulness must measure different things, or running both
    # tells you the same thing twice.
    assert "correctness" in p.lower()
    assert '"score"' in p


def test_chunk_relevance_lists_every_chunk():
    p = chunk_relevance_prompt("q", CHUNKS)
    assert "[c1]" in p and "[c2]" in p
    assert '"judgments"' in p


def test_prompt_version_is_a_nonempty_string():
    # The cache key includes this. Editing a prompt without bumping it serves
    # stale scores from before the edit.
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_judge_prompts.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.judge.prompts'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/judge/prompts.py`:

```python
"""Judge prompt templates.

PROMPT_VERSION is part of every cache key. Bump it by hand whenever a template
below changes, or the cache will keep serving scores produced by the old wording.
"""

PROMPT_VERSION = "1"

__all__ = [
    "PROMPT_VERSION",
    "faithfulness_prompt",
    "answer_relevance_prompt",
    "chunk_relevance_prompt",
]


def _render_chunks(chunks: list[dict]) -> str:
    return "\n".join(
        f"[{c['chunk_id']}] {c.get('text') or '(no text captured)'}" for c in chunks
    )


def faithfulness_prompt(question: str, chunks: list[dict], answer: str) -> str:
    return f"""You are auditing whether an answer is supported by its sources.

QUESTION:
{question}

SOURCES:
{_render_chunks(chunks)}

ANSWER:
{answer}

Break the ANSWER into atomic factual claims. For each claim, decide whether the
SOURCES support it. A claim is supported only if a source states it or directly
implies it. Correct-sounding general knowledge that does not appear in the
SOURCES is NOT supported. Cite the chunk ids in square brackets above.

Return only a JSON object of this shape and nothing else:
{{"claims": [{{"claim": "...", "supported": true, "chunk_ids": ["c1"], "reason": "..."}}]}}
"""


def answer_relevance_prompt(question: str, answer: str) -> str:
    return f"""You are rating how well an answer addresses a question.

QUESTION:
{question}

ANSWER:
{answer}

Rate on this scale:
5 - fully addresses the question
4 - addresses it with minor gaps
3 - partially addresses it
2 - barely related
1 - does not address it at all

Judge relevance only. Do NOT judge factual correctness or whether the answer is
supported by any source; those are measured separately. An answer that is wrong
but on-topic still scores high here.

Return only a JSON object of this shape and nothing else:
{{"score": 4, "reason": "..."}}
"""


def chunk_relevance_prompt(question: str, chunks: list[dict]) -> str:
    return f"""You are judging whether retrieved passages are useful for a question.

QUESTION:
{question}

PASSAGES:
{_render_chunks(chunks)}

For each passage, decide whether it contains information useful for answering
the QUESTION. Judge each passage independently.

Return only a JSON object of this shape and nothing else, with one entry per
passage above:
{{"judgments": [{{"chunk_id": "c1", "relevant": true}}]}}
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_judge_prompts.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/judge/prompts.py tests/test_judge_prompts.py
git commit -m "feat: judge prompt templates with versioning"
```

---

### Task 4: Judge cache table

**Files:**
- Modify: `ragmeter/db.py`
- Test: `tests/test_db.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_db.py`:

```python
def test_judge_cache_roundtrip(session):
    from ragmeter.db import JudgeCache

    session.add(JudgeCache(key="abc123", response_json={"score": 4}))
    session.commit()
    assert session.get(JudgeCache, "abc123").response_json == {"score": 4}
```

Add `JudgeCache` to the import at the top of the file:

```python
from ragmeter.db import (
    Evaluation, GoldenItem, JudgeCache, Run, Trace,
    init_db, make_engine, make_session,
)
```

and delete the now-duplicated `from ragmeter.db import JudgeCache` inside the
test body.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -q`
Expected: `ImportError: cannot import name 'JudgeCache'`

- [ ] **Step 3: Add the table**

In `ragmeter/db.py`, add `"JudgeCache"` to `__all__`, and add this class after
`ModelPrice`:

```python
class JudgeCache(Base):
    """Judge responses keyed by sha256(model | prompt_version | prompt).

    The judge model runs at temperature 0, so an identical prompt has an
    identical answer. Caching keeps a re-run of a 200-question eval free rather
    than burning a free-tier daily quota to recompute what did not change.
    """

    __tablename__ = "judge_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    response_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/db.py tests/test_db.py
git commit -m "feat: judge_cache table"
```

---

### Task 5: Judge client with retry and caching

**Files:**
- Create: `ragmeter/judge/client.py`
- Test: `tests/test_judge_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge_client.py`:

```python
import json

import httpx
import pytest
import respx

from ragmeter.db import JudgeCache, init_db, make_engine, make_session
from ragmeter.judge.client import (
    CHAT_URL,
    DbJudgeCache,
    JudgeClient,
    JudgeError,
    MemoryCache,
    cache_key,
)


def reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def make_client(**kwargs) -> JudgeClient:
    # sleep is injected so retry tests do not actually wait.
    kwargs.setdefault("sleep", lambda _seconds: None)
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("model", "test/model")
    return JudgeClient(**kwargs)


@respx.mock
def test_ask_returns_parsed_json():
    respx.post(CHAT_URL).mock(return_value=reply('{"score": 4}'))
    assert make_client().ask("prompt") == {"score": 4}


@respx.mock
def test_ask_sends_auth_and_zero_temperature():
    route = respx.post(CHAT_URL).mock(return_value=reply('{"score": 4}'))
    make_client().ask("prompt")
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer test-key"
    # Parse rather than substring-match: httpx serializes JSON compactly, so
    # '"temperature": 0' with a space would never appear in the body.
    body = json.loads(request.content)
    assert body["temperature"] == 0
    assert body["model"] == "test/model"


@respx.mock
def test_ask_recovers_fenced_json():
    respx.post(CHAT_URL).mock(return_value=reply('```json\n{"score": 5}\n```'))
    assert make_client().ask("prompt") == {"score": 5}


@respx.mock
def test_unparseable_output_retries_once_then_fails():
    route = respx.post(CHAT_URL).mock(return_value=reply("I refuse."))
    with pytest.raises(JudgeError, match="could not parse"):
        make_client().ask("prompt")
    # One original attempt plus exactly one strict-nudge retry.
    assert route.call_count == 2


@respx.mock
def test_unparseable_then_parseable_succeeds():
    route = respx.post(CHAT_URL).mock(
        side_effect=[reply("nonsense"), reply('{"score": 3}')])
    assert make_client().ask("prompt") == {"score": 3}
    assert route.call_count == 2


@respx.mock
def test_retries_on_429_then_succeeds():
    route = respx.post(CHAT_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(429), reply('{"score": 2}')])
    assert make_client().ask("prompt") == {"score": 2}
    assert route.call_count == 3


@respx.mock
def test_gives_up_after_max_attempts():
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(JudgeError, match="429"):
        make_client(max_attempts=3).ask("prompt")
    assert route.call_count == 3


@respx.mock
def test_honors_retry_after_header():
    slept = []
    respx.post(CHAT_URL).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "7"}),
                     reply('{"score": 1}')])
    make_client(sleep=slept.append).ask("prompt")
    assert slept == [7.0]


@respx.mock
def test_backoff_is_exponential_without_retry_after():
    slept = []
    respx.post(CHAT_URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(503), reply('{"score": 1}')])
    make_client(sleep=slept.append).ask("prompt")
    assert slept == [1.0, 2.0]


@respx.mock
def test_client_error_is_not_retried():
    # A 401 will never succeed on retry; burning five attempts on it just makes
    # the real problem take longer to surface.
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(401))
    with pytest.raises(JudgeError, match="401"):
        make_client().ask("prompt")
    assert route.call_count == 1


@respx.mock
def test_cache_hit_skips_the_network():
    cache = MemoryCache()
    route = respx.post(CHAT_URL).mock(return_value=reply('{"score": 4}'))
    client = make_client(cache=cache)
    assert client.ask("same prompt") == {"score": 4}
    assert client.ask("same prompt") == {"score": 4}
    assert route.call_count == 1


@respx.mock
def test_different_prompts_do_not_share_a_cache_entry():
    cache = MemoryCache()
    route = respx.post(CHAT_URL).mock(return_value=reply('{"score": 4}'))
    client = make_client(cache=cache)
    client.ask("prompt a")
    client.ask("prompt b")
    assert route.call_count == 2


def test_cache_key_depends_on_model_prompt_and_version():
    a = cache_key("m1", "v1", "p")
    assert a != cache_key("m2", "v1", "p")
    assert a != cache_key("m1", "v2", "p")
    assert a != cache_key("m1", "v1", "other")
    assert a == cache_key("m1", "v1", "p")


@respx.mock
def test_db_cache_persists_across_clients():
    engine = make_engine("sqlite://")
    init_db(engine)
    session = make_session(engine)()
    route = respx.post(CHAT_URL).mock(return_value=reply('{"score": 4}'))

    make_client(cache=DbJudgeCache(session)).ask("prompt")
    make_client(cache=DbJudgeCache(session)).ask("prompt")

    assert route.call_count == 1
    assert session.query(JudgeCache).count() == 1
    session.close()


def test_missing_api_key_fails_immediately(monkeypatch):
    # delenv so the test still means something on a machine where the key is set.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Fail before any work rather than after ingesting and evaluating.
    with pytest.raises(JudgeError, match="OPENROUTER_API_KEY"):
        JudgeClient(api_key=None, model="test/model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_judge_client.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.judge.client'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/judge/client.py`:

```python
"""OpenRouter transport for the judge: HTTP, retry, and caching.

Knows nothing about RAG. It takes a prompt string and returns a parsed JSON
object, or raises. Every scoring decision lives in scoring.py.
"""

import hashlib
import os
import time
from collections.abc import Callable

import httpx
from sqlalchemy.orm import Session

from ragmeter.db import JudgeCache
from ragmeter.judge.parsing import ParseError, extract_json
from ragmeter.judge.prompts import PROMPT_VERSION

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# 4xx codes worth retrying: everything else in that range is a request problem
# that will fail identically on the next attempt.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

STRICT_NUDGE = (
    "\n\nYour previous reply could not be parsed. Reply with the JSON object "
    "only: no prose, no markdown fences, no explanation."
)

__all__ = [
    "CHAT_URL", "DEFAULT_MODEL", "JudgeError", "JudgeClient",
    "MemoryCache", "DbJudgeCache", "cache_key",
]


class JudgeError(RuntimeError):
    """The judge could not produce a usable answer."""


def cache_key(model: str, prompt_version: str, prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model}|{prompt_version}|{prompt}".encode())
    return digest.hexdigest()


class MemoryCache:
    """In-process cache. Used by tests and by anything running without a DB."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        return self._store.get(key)

    def set(self, key: str, value: dict) -> None:
        self._store[key] = value


class DbJudgeCache:
    """Cache backed by the judge_cache table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, key: str) -> dict | None:
        row = self._session.get(JudgeCache, key)
        return row.response_json if row is not None else None

    def set(self, key: str, value: dict) -> None:
        # Commits immediately, mid-evaluation, on purpose: a crash halfway
        # through a 200-question run must not throw away judge answers that
        # already cost real free-tier quota. The price is that a crashed run
        # leaves partial Evaluation rows committed too.
        self._session.merge(JudgeCache(key=key, response_json=value))
        self._session.commit()


class JudgeClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cache=None,
        http: httpx.Client | None = None,
        max_attempts: int = 5,
        max_tokens: int = 4096,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise JudgeError(
                "OPENROUTER_API_KEY is not set; the judge cannot run without it"
            )
        self.api_key = api_key
        self.model = model or os.environ.get("RAGMETER_JUDGE_MODEL", DEFAULT_MODEL)
        self.cache = cache
        self.http = http or httpx.Client(timeout=180)
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens
        self.sleep = sleep

    def ask(self, prompt: str) -> dict:
        """Return the parsed JSON object for this prompt, from cache or the API."""
        key = cache_key(self.model, PROMPT_VERSION, prompt)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        content = self._complete(prompt)
        try:
            parsed = extract_json(content)
        except ParseError:
            # One strict retry. A model that ignores the shape twice at
            # temperature 0 will not comply on a third ask.
            content = self._complete(prompt + STRICT_NUDGE)
            try:
                parsed = extract_json(content)
            except ParseError as exc:
                raise JudgeError(f"could not parse judge output: {exc}") from exc

        if self.cache is not None:
            self.cache.set(key, parsed)
        return parsed

    def _complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            retry_after: str | None = None
            try:
                response = self.http.post(CHAT_URL, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code == 200:
                    return self._content_of(response.json())
                last_error = f"{response.status_code}: {response.text[:200]}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise JudgeError(f"OpenRouter returned {last_error}")
                retry_after = response.headers.get("Retry-After")

            if attempt == self.max_attempts:
                break
            self.sleep(float(retry_after) if retry_after else float(2 ** (attempt - 1)))

        raise JudgeError(
            f"judge failed after {self.max_attempts} attempts; last error {last_error}"
        )

    @staticmethod
    def _content_of(body: dict) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError(f"unexpected OpenRouter response shape: {body}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_judge_client.py -q`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/judge/client.py tests/test_judge_client.py
git commit -m "feat: judge client with retry, backoff, and caching"
```

---

### Task 6: Scoring functions

**Files:**
- Create: `ragmeter/judge/scoring.py`
- Test: `tests/test_judge_scoring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge_scoring.py`:

```python
import pytest

from ragmeter.judge.client import JudgeError
from ragmeter.judge.scoring import (
    score_answer_relevance,
    score_chunk_relevance,
    score_faithfulness,
)

CHUNKS = [{"chunk_id": "c1", "text": "Returns accepted within 30 days."},
          {"chunk_id": "c2", "text": "Items must be unused."}]


class FakeJudge:
    """Stands in for JudgeClient. Records prompts, returns canned JSON."""

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_faithfulness_all_supported():
    judge = FakeJudge({"claims": [
        {"claim": "a", "supported": True, "chunk_ids": ["c1"]},
        {"claim": "b", "supported": True, "chunk_ids": ["c2"]},
    ]})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    assert out["score"] == 1.0
    assert len(out["claims"]) == 2


def test_faithfulness_partial():
    judge = FakeJudge({"claims": [
        {"claim": "a", "supported": True, "chunk_ids": ["c1"]},
        {"claim": "b", "supported": False, "chunk_ids": []},
        {"claim": "c", "supported": False, "chunk_ids": []},
    ]})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    assert out["score"] == pytest.approx(1 / 3)


def test_faithfulness_with_no_chunks_is_unmeasurable():
    # There is nothing to be faithful to. Zero would read as "the model
    # hallucinated", which is a different and unproven claim.
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", [], "answer")
    assert out["score"] is None
    assert judge.prompts == []


def test_faithfulness_with_empty_answer_is_unmeasurable():
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", CHUNKS, "   ")
    assert out["score"] is None
    assert judge.prompts == []


def test_faithfulness_with_no_claims_returned_is_unmeasurable():
    judge = FakeJudge({"claims": []})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    assert out["score"] is None


def test_faithfulness_rejects_missing_claims_key():
    judge = FakeJudge({"wrong": []})
    with pytest.raises(JudgeError, match="claims"):
        score_faithfulness(judge, "q", CHUNKS, "answer")


def test_faithfulness_treats_non_true_supported_as_false():
    judge = FakeJudge({"claims": [{"claim": "a", "supported": "yes"},
                                  {"claim": "b", "supported": True}]})
    out = score_faithfulness(judge, "q", CHUNKS, "answer")
    # Only a literal true counts. A string "yes" is the model ignoring the
    # schema, and reading it as support would inflate the score.
    assert out["score"] == 0.5


def test_answer_relevance_normalizes_to_unit_interval():
    assert score_answer_relevance(FakeJudge({"score": 5}), "q", "a")["score"] == 1.0
    assert score_answer_relevance(FakeJudge({"score": 1}), "q", "a")["score"] == 0.0
    assert score_answer_relevance(FakeJudge({"score": 4}), "q", "a")["score"] == 0.75


def test_answer_relevance_keeps_the_reason():
    out = score_answer_relevance(FakeJudge({"score": 3, "reason": "vague"}), "q", "a")
    assert out["reason"] == "vague"


def test_answer_relevance_rejects_out_of_range():
    with pytest.raises(JudgeError, match="1..5"):
        score_answer_relevance(FakeJudge({"score": 9}), "q", "a")


def test_answer_relevance_rejects_non_numeric():
    with pytest.raises(JudgeError, match="1..5"):
        score_answer_relevance(FakeJudge({"score": "high"}), "q", "a")


def test_answer_relevance_empty_answer_is_unmeasurable():
    judge = FakeJudge({"score": 1})
    out = score_answer_relevance(judge, "q", "")
    assert out["score"] is None
    assert judge.prompts == []


def test_chunk_relevance_returns_judgments_and_precision():
    judge = FakeJudge({"judgments": [{"chunk_id": "c1", "relevant": True},
                                     {"chunk_id": "c2", "relevant": False}]})
    out = score_chunk_relevance(judge, "q", CHUNKS)
    assert out["precision"] == 0.5
    assert len(out["judgments"]) == 2


def test_chunk_relevance_ignores_ids_that_were_not_retrieved():
    # A model that invents a chunk id must not be able to change the denominator.
    judge = FakeJudge({"judgments": [{"chunk_id": "c1", "relevant": True},
                                     {"chunk_id": "ghost", "relevant": True}]})
    out = score_chunk_relevance(judge, "q", CHUNKS)
    assert out["precision"] == 0.5


def test_chunk_relevance_with_no_chunks_is_unmeasurable():
    judge = FakeJudge({"judgments": []})
    out = score_chunk_relevance(judge, "q", [])
    assert out["precision"] is None
    assert judge.prompts == []


def test_judge_errors_propagate():
    judge = FakeJudge(JudgeError("rate limited"))
    with pytest.raises(JudgeError, match="rate limited"):
        score_faithfulness(judge, "q", CHUNKS, "answer")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_judge_scoring.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.judge.scoring'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/judge/scoring.py`:

```python
"""Turning judge output into numbers.

Every function returns a dict with a `score` that is either a float or None.
None means the question could not be asked or answered meaningfully -- never a
stand-in for a bad score.
"""

from ragmeter.judge.client import JudgeError
from ragmeter.judge.prompts import (
    answer_relevance_prompt,
    chunk_relevance_prompt,
    faithfulness_prompt,
)

__all__ = ["score_faithfulness", "score_answer_relevance", "score_chunk_relevance"]


def score_faithfulness(judge, question: str, chunks: list[dict], answer: str) -> dict:
    """Fraction of the answer's claims that the retrieved chunks support."""
    if not chunks or not answer.strip():
        # Nothing to be faithful to, or nothing to check. Not a zero.
        return {"score": None, "claims": [],
                "reason": "no retrieved chunks" if not chunks else "empty answer"}

    data = judge.ask(faithfulness_prompt(question, chunks, answer))
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise JudgeError(f"judge response missing a 'claims' list: {data}")
    if not claims:
        return {"score": None, "claims": [], "reason": "judge extracted no claims"}

    # Only a literal True counts as support. Anything else is the model
    # departing from the schema, and reading it as support inflates the score.
    supported = sum(1 for c in claims if c.get("supported") is True)
    return {"score": supported / len(claims), "claims": claims, "reason": None}


def score_answer_relevance(judge, question: str, answer: str) -> dict:
    """How well the answer addresses the question, normalized to 0..1."""
    if not answer.strip():
        return {"score": None, "reason": "empty answer"}

    data = judge.ask(answer_relevance_prompt(question, answer))
    raw = data.get("score")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not 1 <= raw <= 5:
        raise JudgeError(f"judge score must be a number in 1..5, got {raw!r}")
    return {"score": (raw - 1) / 4, "reason": data.get("reason")}


def score_chunk_relevance(judge, question: str, chunks: list[dict]) -> dict:
    """Per-chunk usefulness, yielding precision when no golden labels exist.

    This cannot yield recall: nothing here can see a relevant chunk that was
    never retrieved.
    """
    if not chunks:
        return {"precision": None, "judgments": [], "reason": "no retrieved chunks"}

    data = judge.ask(chunk_relevance_prompt(question, chunks))
    judgments = data.get("judgments")
    if not isinstance(judgments, list):
        raise JudgeError(f"judge response missing a 'judgments' list: {data}")

    retrieved_ids = {c["chunk_id"] for c in chunks}
    # Invented chunk ids must not move the denominator.
    known = [j for j in judgments if j.get("chunk_id") in retrieved_ids]
    if not known:
        return {"precision": None, "judgments": judgments,
                "reason": "judge returned no judgments for retrieved chunks"}

    relevant = sum(1 for j in known if j.get("relevant") is True)
    return {"precision": relevant / len(retrieved_ids), "judgments": known, "reason": None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_judge_scoring.py -q`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/judge/scoring.py tests/test_judge_scoring.py
git commit -m "feat: faithfulness, answer relevance, and chunk relevance scoring"
```

---

### Task 7: Wire the judge into the runner and CLI

**Files:**
- Modify: `ragmeter/runner.py`
- Modify: `ragmeter/cli.py`
- Test: `tests/test_runner_judge.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runner_judge.py`:

```python
from pathlib import Path

import pytest

from ragmeter.db import Evaluation, init_db, make_engine, make_session
from ragmeter.judge.client import JudgeError
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


class FakeJudge:
    """Returns a canned response shaped by which prompt it was handed."""

    model = "fake/judge"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def ask(self, prompt):
        self.calls += 1
        if self.fail:
            raise JudgeError("rate limited")
        if "atomic factual claims" in prompt:
            return {"claims": [{"claim": "x", "supported": True, "chunk_ids": ["c1"]},
                               {"claim": "y", "supported": False, "chunk_ids": []}]}
        if "Rate on this scale" in prompt:
            return {"score": 5, "reason": "direct"}
        return {"judgments": [{"chunk_id": "c1", "relevant": True}]}


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


def test_no_judge_leaves_status_skipped(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES)
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.judge_status == "skipped"
    assert "faithfulness" not in ev.metrics


def test_judge_adds_metrics_and_claims(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=FakeJudge())
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.judge_status == "ok"
    assert ev.metrics["faithfulness"] == 0.5
    assert ev.metrics["answer_relevance"] == 1.0
    assert len(ev.claims) == 2
    assert ev.judge_model == "fake/judge"


def test_trace_without_chunks_gets_none_faithfulness(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=FakeJudge())
    ev = loaded.query(Evaluation).filter_by(trace_id="t4").one()
    assert ev.metrics["faithfulness"] is None
    # Still 'ok': the judge was asked nothing because there was nothing to ask.
    assert ev.judge_status == "ok"


def test_unmatched_trace_gets_chunk_relevance_precision(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=FakeJudge())
    ev = loaded.query(Evaluation).filter_by(trace_id="t6").one()
    # t6 has no golden item, so precision comes from the judge instead.
    assert ev.metrics["precision@3"] == 1.0
    assert ev.chunk_judgments == [{"chunk_id": "c1", "relevant": True}]
    # Recall stays unmeasurable: nothing can see what was never retrieved.
    assert ev.metrics["recall@3"] is None


def test_matched_trace_does_not_call_chunk_relevance(loaded):
    judge = FakeJudge()
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES, judge=judge)
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    # Golden labels beat a judge's opinion, so precision@3 stays the computed 2/3.
    assert ev.metrics["precision@3"] == pytest.approx(2 / 3)
    assert ev.chunk_judgments is None


def test_judge_failure_is_recorded_not_swallowed(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES,
                 judge=FakeJudge(fail=True))
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.judge_status == "failed"
    assert "rate limited" in ev.judge_error
    # No number is invented for a metric that was never measured.
    assert ev.metrics["faithfulness"] is None
    assert ev.metrics["answer_relevance"] is None


def test_judge_failure_does_not_stop_retrieval_metrics(loaded):
    evaluate_run(loaded, "baseline", "docs", "v1", k=3, prices=PRICES,
                 judge=FakeJudge(fail=True))
    ev = loaded.query(Evaluation).filter_by(trace_id="t1").one()
    assert ev.metrics["recall@3"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_runner_judge.py -q`
Expected: `TypeError: evaluate_run() got an unexpected keyword argument 'judge'`

- [ ] **Step 3: Update `ragmeter/runner.py`**

Replace the whole file with:

```python
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
```

- [ ] **Step 4: Run the runner tests**

Run: `.venv/Scripts/python -m pytest tests/test_runner_judge.py tests/test_runner.py -q`
Expected: `17 passed`

- [ ] **Step 5: Add the `--judge` flag to `ragmeter/cli.py`**

Add these imports below the existing ones:

```python
from ragmeter.judge.client import DbJudgeCache, JudgeClient, JudgeError
```

Replace the `evaluate` command with:

```python
@app.command("eval")
def evaluate(
    run: str = typer.Option(..., "--run"),
    dataset: str = typer.Option(..., "--dataset"),
    version: str = typer.Option("v1", "--version"),
    k: int = typer.Option(5, "--k", min=1),
    judge: bool = typer.Option(False, "--judge/--no-judge",
                               help="Score faithfulness and answer relevance via OpenRouter."),
    judge_model: str | None = typer.Option(None, "--judge-model"),
) -> None:
    """Compute retrieval, cost, and latency metrics for a run."""
    try:
        prices = fetch_prices()
    except Exception as exc:
        # Pricing is an enrichment, not a prerequisite. Say so loudly and continue.
        typer.echo(f"warning: could not fetch model prices ({exc}); cost will be blank", err=True)
        prices = {}

    session = _session()
    judge_client = None
    if judge:
        try:
            judge_client = JudgeClient(model=judge_model, cache=DbJudgeCache(session))
        except JudgeError as exc:
            # Fail before evaluating rather than after, so the user is not left
            # wondering why every judge column is blank.
            session.close()
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2)

    try:
        result = evaluate_run(session, run, dataset, version, k=k, prices=prices,
                              judge=judge_client)
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
    if result["n_judge_failures"]:
        typer.echo(
            f"WARNING: the judge failed on {result['n_judge_failures']} trace(s); "
            f"those metrics are blank, not zero",
            err=True,
        )
```

- [ ] **Step 6: Add a CLI test**

Append to `tests/test_cli.py`:

```python
def test_judge_without_api_key_fails_before_evaluating(db_url, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        result = runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                                     "--version", "v1", "--judge"])
    assert result.exit_code == 2
    assert "OPENROUTER_API_KEY" in result.output
```

- [ ] **Step 7: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `115 passed`

- [ ] **Step 8: Commit**

```bash
git add ragmeter/runner.py ragmeter/cli.py tests/test_runner_judge.py tests/test_cli.py
git commit -m "feat: wire the LLM judge into the runner and CLI"
```

---

### Task 8: Live smoke test and docs

**Requires `OPENROUTER_API_KEY`.** If it is not available, stop here, report
that Tasks 1-7 are complete and verified against mocks, and ask for the key
before running this task. Do not fake this step.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the judge against the real API**

```bash
$env:RAGMETER_DB_URL = "sqlite:///smoke.db"
.venv/Scripts/ragmeter.exe dataset load tests/fixtures/golden.yaml --name docs --version v1
.venv/Scripts/ragmeter.exe ingest tests/fixtures/traces.jsonl --run baseline
.venv/Scripts/ragmeter.exe eval --run baseline --dataset docs --version v1 --k 3 --judge
```

Expected: the summary table gains `faithfulness` and `answer_relevance` rows.

- [ ] **Step 2: Verify the judge caught the planted unfaithful claim**

`t3`'s answer ends with "Shipping is free", which no chunk supports.

```bash
.venv/Scripts/python.exe -c "from ragmeter.db import *; e=make_engine('sqlite:///smoke.db'); s=make_session(e)(); ev=s.query(Evaluation).filter_by(trace_id='t3').one(); print(ev.judge_status, ev.metrics['faithfulness']); [print(c['supported'], '|', c['claim']) for c in ev.claims]"
```

Expected: `faithfulness` below 1.0, with the "shipping is free" claim marked
`False`. If the judge marks it supported, the prompt needs work — report that
rather than accepting the number.

- [ ] **Step 3: Verify caching works**

Re-run the same eval command and confirm it returns quickly with identical
scores, and that the cache table is populated:

```bash
.venv/Scripts/python.exe -c "from ragmeter.db import *; e=make_engine('sqlite:///smoke.db'); s=make_session(e)(); print('cached judge responses:', s.query(JudgeCache).count())"
```

Expected: a non-zero count, and the second run visibly faster.

- [ ] **Step 4: Delete the smoke database**

```bash
Remove-Item -Force smoke.db
```

- [ ] **Step 5: Update `README.md`**

Change the Use section to add:

````markdown
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
````

Change the Configuration table to:

```markdown
| variable | default |
|---|---|
| `RAGMETER_DB_URL` | `sqlite:///ragmeter.db` |
| `OPENROUTER_API_KEY` | — (required for `--judge`) |
| `RAGMETER_JUDGE_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` |
```

Change Status to: `Phase 2 of 5. Next: the regression gate, judge calibration,
and the HTTP API.`

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: README for phase 2"
```

---

## Definition of Done

- [ ] `.venv/Scripts/python -m pytest -q` reports 115 passed (59 from phase 1, plus parsing 10, prompts 6, db 1, client 15, scoring 16, runner-judge 7, cli 1)
- [ ] A judge failure produces `judge_status='failed'` with blank metrics, never a zero
- [ ] Faithfulness with no retrieved chunks is `None`, not `0.0`
- [ ] Re-running an eval with `--judge` hits the cache and makes no HTTP calls
- [ ] The live judge marks `t3`'s unsupported "shipping is free" claim as unsupported
- [ ] `ragmeter/judge/client.py` contains no RAG-specific logic

## Out of Scope

Regression gate, calibration, HTTP API, dashboard. Judge token cost tracking is
not included: the default judge model is free, and adding it now would be
speculation.
