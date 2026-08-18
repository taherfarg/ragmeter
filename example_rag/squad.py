"""SQuAD JSON to documents whose answer offsets are document-relative.

SQuAD stores an answer offset relative to its own paragraph. This module joins
an article's paragraphs into one document and shifts every offset accordingly,
because the whole experiment depends on being able to ask "which chunk contains
this answer" for arbitrary chunk boundaries.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

PARAGRAPH_SEPARATOR = "\n\n"

__all__ = ["Question", "Document", "load_squad"]


@dataclass
class Question:
    question_id: str
    text: str
    spans: list[tuple[int, int]] = field(default_factory=list)
    answer_texts: list[str] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    text: str
    questions: list[Question] = field(default_factory=list)


def load_squad(path: Path, max_articles: int | None = None) -> list[Document]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    articles = raw["data"]
    if max_articles is not None:
        articles = articles[:max_articles]

    documents = []
    for article in articles:
        parts: list[str] = []
        questions: list[Question] = []
        offset = 0

        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                spans: list[tuple[int, int]] = []
                texts: list[str] = []
                for answer in qa["answers"]:
                    start = offset + answer["answer_start"]
                    span = (start, start + len(answer["text"]))
                    # Dev SQuAD carries three annotations per question and they
                    # are usually identical; counting them thrice would weight
                    # one question three times in the golden set.
                    if span in spans:
                        continue
                    spans.append(span)
                    texts.append(answer["text"])
                if spans:
                    questions.append(Question(qa["id"], qa["question"], spans, texts))

            parts.append(context)
            offset += len(context) + len(PARAGRAPH_SEPARATOR)

        documents.append(
            Document(article["title"], PARAGRAPH_SEPARATOR.join(parts), questions)
        )
    return documents
