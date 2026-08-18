"""Run one chunking strategy end to end and emit ragmeter's contract.

Deliberately does not import ragmeter. It writes JSONL and YAML that the tool
ingests, which is what makes "measures any RAG" a claim rather than a wish.
"""

import json
import time
from pathlib import Path

import yaml

from example_rag.bm25 import BM25
from example_rag.chunking import chunk_document
from example_rag.golden import relevant_chunk_ids
from example_rag.squad import Document

__all__ = ["run_strategy"]


def _answer_from(chunks: list[dict]) -> str:
    """A stand-in extractive answer: the top chunk's first sentence.

    Generation is not the variable under test, and a real LLM call per question
    would make a multi-strategy sweep slow and rate-limited for no gain in what
    retrieval metrics measure.
    """
    if not chunks:
        return ""
    head = chunks[0]["text"].strip()
    for stop in (". ", "! ", "? "):
        if stop in head:
            return head[:head.index(stop) + 1]
    return head[:300]


def run_strategy(
    documents: list[Document], strategy: str, k: int, out_dir: Path,
    limit: int | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_doc = {d.doc_id: chunk_document(d.doc_id, d.text, strategy) for d in documents}
    corpus = {c.chunk_id: c.text for chunks in chunks_by_doc.values() for c in chunks}
    index = BM25(corpus)

    traces_path = out_dir / f"{strategy}.traces.jsonl"
    golden_path = out_dir / f"{strategy}.golden.yaml"

    golden: list[dict] = []
    question_ids: list[str] = []
    written = 0
    unlabelled = 0

    with traces_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            if limit is not None and written >= limit:
                break  # the inner break alone would let the next document past the limit
            for question in document.questions:
                if limit is not None and written >= limit:
                    break

                relevant = relevant_chunk_ids(question.spans, chunks_by_doc[document.doc_id])
                if not relevant:
                    # No chunk contains the answer, so recall is undefined here.
                    # ragmeter rejects an empty label at load time, and skipping
                    # is honest where inventing a label would not be.
                    unlabelled += 1
                    continue

                started = time.perf_counter()
                hits = index.search(question.text, k=k)
                latency_ms = int((time.perf_counter() - started) * 1000)

                retrieved = [
                    {"chunk_id": chunk_id, "text": corpus[chunk_id],
                     "score": round(score, 6), "rank": rank}
                    for rank, (chunk_id, score) in enumerate(hits, start=1)
                ]

                handle.write(json.dumps({
                    # Namespaced: trace_id is a global primary key in ragmeter,
                    # so a bare question_id would collide across strategies.
                    "trace_id": f"{strategy}:{question.question_id}",
                    "question_id": question.question_id,
                    "question": question.text,
                    "retrieved": retrieved,
                    "answer": _answer_from(retrieved),
                    "model": "bm25/extractive",
                    "latency_ms": latency_ms,
                    "metadata": {"strategy": strategy, "doc_id": document.doc_id},
                }, ensure_ascii=False) + "\n")

                golden.append({
                    "question_id": question.question_id,
                    "question": question.text,
                    "relevant_chunk_ids": relevant,
                })
                question_ids.append(question.question_id)
                written += 1

    golden_path.write_text(
        yaml.safe_dump(golden, allow_unicode=True, sort_keys=False), encoding="utf-8")

    return {
        "strategy": strategy,
        "questions": written,
        "unlabelled": unlabelled,
        "chunks": len(corpus),
        "question_ids": question_ids,
        "traces": str(traces_path),
        "golden": str(golden_path),
    }
