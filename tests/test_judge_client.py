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
