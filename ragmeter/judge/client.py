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
