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
