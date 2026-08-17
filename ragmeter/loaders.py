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
