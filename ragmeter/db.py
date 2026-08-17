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
    "Base", "Run", "Trace", "GoldenItem", "Evaluation", "ModelPrice", "JudgeCache",
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


DEFAULT_DB_URL = "sqlite:///ragmeter.db"


def make_engine(url: str | None = None):
    return create_engine(url or os.environ.get("RAGMETER_DB_URL", DEFAULT_DB_URL))


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def make_session(engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)
