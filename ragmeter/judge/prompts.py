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
