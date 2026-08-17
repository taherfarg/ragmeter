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
