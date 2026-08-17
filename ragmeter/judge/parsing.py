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
