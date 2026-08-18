# Example RAG — Chunking Strategy Experiment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small BM25 RAG over SQuAD that emits ragmeter's trace contract, so five chunking strategies can be measured against each other on the same questions.

**Architecture:** Chunking is the independent variable, so everything else is held deliberately boring: BM25 rather than embeddings, and the golden dataset is *derived* per strategy from SQuAD's answer offsets rather than labelled by hand.

**Tech Stack:** Python stdlib plus PyYAML. **No torch, no embeddings, no vector store.** The example RAG does not import ragmeter — it writes the trace contract as plain JSON, which is the whole point of a tool that measures *any* RAG.

---

## The Idea That Makes This Work

The known hard problem: comparing chunking strategies changes every `chunk_id`,
so a hand-labelled golden set is invalidated by the very change you want to
measure.

SQuAD gives an answer's **character offset inside its paragraph**. Lift that to
a document-level offset and the golden set can be *generated* for any chunking:

```
relevant_chunk_ids(question, chunks) =
    [c.chunk_id for c in chunks if c overlaps any answer span of question]
```

The labels are anchored to the source text, not to chunk boundaries. Each
strategy gets its own correct golden set for free, and because `question_id`
stays the SQuAD id across all of them, ragmeter's paired comparison lines the
strategies up question by question.

**This is the reason the experiment is possible at all.** Without it, five
strategies would need five rounds of manual labelling.

## Design Rules

**Hold everything constant except chunking.** Same questions, same retriever,
same k. A difference in recall must be attributable to chunk boundaries alone.

**Name things by what they do.** The fifth strategy measures vocabulary overlap
between adjacent sentences. It is called `lexical-cohesion`, not `semantic`,
because it uses no embeddings and claiming otherwise would overstate it.

**The RAG must not import ragmeter.** It writes JSONL and YAML. If it needed
the library, the claim that ragmeter measures *any* RAG would be untested.

## File Structure

| File | Responsibility |
|---|---|
| `example_rag/squad.py` | SQuAD JSON to documents with document-level answer spans |
| `example_rag/chunking.py` | the five strategies. Pure functions over text |
| `example_rag/golden.py` | derive relevant chunk ids from answer spans |
| `example_rag/bm25.py` | BM25 index and search. No dependencies |
| `example_rag/pipeline.py` | build, retrieve, emit traces and golden |
| `example_rag/cli.py` | `build` and `run` |
| `data/` | the downloaded corpus, gitignored |

---

### Task 1: SQuAD loader

**Files:**
- Create: `example_rag/__init__.py`, `example_rag/squad.py`
- Test: `tests_rag/test_squad.py`

- [ ] **Step 1: Scaffold**

```bash
mkdir -p example_rag tests_rag
touch example_rag/__init__.py tests_rag/__init__.py
```

Add `tests_rag` to `testpaths` in `pyproject.toml`:

```toml
testpaths = ["tests", "tests_rag"]
```

Add `data/` to `.gitignore`.

- [ ] **Step 2: Write the failing tests**

Create `tests_rag/test_squad.py`:

```python
import json

import pytest

from example_rag.squad import Document, Question, load_squad

RAW = {
    "version": "1.1",
    "data": [
        {
            "title": "Test_Article",
            "paragraphs": [
                {
                    "context": "Alpha beta gamma.",
                    "qas": [{"id": "q1", "question": "Which letter?",
                             "answers": [{"answer_start": 6, "text": "beta"}]}],
                },
                {
                    "context": "Delta epsilon zeta.",
                    "qas": [{"id": "q2", "question": "Which second?",
                             "answers": [{"answer_start": 6, "text": "epsilon"}]}],
                },
            ],
        }
    ],
}


@pytest.fixture()
def squad_file(tmp_path):
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    return path


def test_one_document_per_article(squad_file):
    docs = load_squad(squad_file)
    assert len(docs) == 1
    assert docs[0].doc_id == "Test_Article"


def test_paragraphs_are_joined_into_one_document(squad_file):
    doc = load_squad(squad_file)[0]
    assert "Alpha beta gamma." in doc.text
    assert "Delta epsilon zeta." in doc.text


def test_answer_offsets_are_lifted_to_document_level(squad_file):
    # SQuAD offsets are relative to their paragraph. Joining paragraphs shifts
    # every answer after the first, and a stale offset would silently label the
    # wrong chunk as relevant.
    doc = load_squad(squad_file)[0]
    q2 = next(q for q in doc.questions if q.question_id == "q2")
    start, end = q2.spans[0]
    assert doc.text[start:end] == "epsilon"


def test_first_paragraph_offsets_are_unchanged(squad_file):
    doc = load_squad(squad_file)[0]
    q1 = next(q for q in doc.questions if q.question_id == "q1")
    start, end = q1.spans[0]
    assert doc.text[start:end] == "beta"


def test_duplicate_answers_are_deduplicated(tmp_path):
    raw = json.loads(json.dumps(RAW))
    raw["data"][0]["paragraphs"][0]["qas"][0]["answers"] = [
        {"answer_start": 6, "text": "beta"},
        {"answer_start": 6, "text": "beta"},
    ]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    doc = load_squad(path)[0]
    # SQuAD dev carries three human annotations; identical ones must not
    # inflate the span list.
    assert len(doc.questions[0].spans) == 1


def test_limit_articles(squad_file):
    assert load_squad(squad_file, max_articles=0) == []


def test_every_span_resolves_to_its_answer_text(squad_file):
    for doc in load_squad(squad_file):
        for question in doc.questions:
            for start, end in question.spans:
                assert doc.text[start:end] == question.answer_texts[
                    question.spans.index((start, end))]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests_rag/test_squad.py -q`
Expected: `ModuleNotFoundError: No module named 'example_rag.squad'`

- [ ] **Step 4: Write the implementation**

Create `example_rag/squad.py`:

```python
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

        documents.append(Document(article["title"], PARAGRAPH_SEPARATOR.join(parts),
                                  questions))
    return documents
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests_rag/test_squad.py -q`
Expected: `7 passed`

- [ ] **Step 6: Verify against the real corpus**

```bash
.venv/Scripts/python -c "
from example_rag.squad import load_squad
docs = load_squad('data/squad-dev-v1.1.json')
bad = [(d.doc_id, q.question_id) for d in docs for q in d.questions
       for i,(s,e) in enumerate(q.spans) if d.text[s:e] != q.answer_texts[i]]
print(f'{len(docs)} docs, {sum(len(d.questions) for d in docs)} questions')
print('offset mismatches:', len(bad))
assert not bad, bad[:5]
"
```

Expected: 48 docs, ~10.5k questions, **0 offset mismatches**. This is the check
that the whole golden-generation idea rests on.

- [ ] **Step 7: Commit**

```bash
git add example_rag tests_rag pyproject.toml .gitignore
git commit -m "feat: SQuAD loader with document-level answer offsets"
```

---

### Task 2: Chunking strategies

**Files:**
- Create: `example_rag/chunking.py`
- Test: `tests_rag/test_chunking.py`

- [ ] **Step 1: Write the failing tests**

Create `tests_rag/test_chunking.py`:

```python
import pytest

from example_rag.chunking import STRATEGIES, Chunk, chunk_document

TEXT = (
    "Alpha one two. Beta three four. Gamma five six.\n\n"
    "Delta seven eight. Epsilon nine ten. Zeta eleven twelve.\n\n"
    "Eta thirteen. Theta fourteen."
)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_covers_the_text_without_gaps(name):
    # A gap means an answer could fall between chunks and be unreachable,
    # which would look like a retrieval failure that never happened.
    chunks = chunk_document("d1", TEXT, name)
    assert chunks
    assert chunks[0].start == 0
    assert chunks[-1].end == len(TEXT)
    for previous, following in zip(chunks, chunks[1:]):
        assert following.start <= previous.end, "gap between chunks"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_chunk_text_matches_its_offsets(name):
    for chunk in chunk_document("d1", TEXT, name):
        assert chunk.text == TEXT[chunk.start:chunk.end]


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_chunk_ids_are_unique_and_carry_the_document(name):
    chunks = chunk_document("d1", TEXT, name)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id.startswith("d1:") for c in chunks)


def test_fixed_respects_its_size():
    chunks = chunk_document("d1", "x" * 250, "fixed-100")
    assert [c.end - c.start for c in chunks] == [100, 100, 50]


def test_overlap_strategy_actually_overlaps():
    chunks = chunk_document("d1", "x" * 250, "fixed-100-overlap-50")
    assert chunks[1].start < chunks[0].end
    assert chunks[1].start == 50


def test_paragraph_strategy_splits_on_blank_lines():
    chunks = chunk_document("d1", TEXT, "paragraph")
    assert len(chunks) == 3
    assert chunks[0].text.startswith("Alpha")
    assert chunks[2].text.startswith("Eta")


def test_sentence_strategy_groups_sentences():
    chunks = chunk_document("d1", TEXT, "sentence-2")
    # 8 sentences grouped two at a time.
    assert len(chunks) == 4


def test_lexical_cohesion_keeps_related_sentences_together():
    # Two sentences sharing vocabulary, then an unrelated one.
    text = "Cats like fish. Cats eat fish daily. Volcanoes erupt lava."
    chunks = chunk_document("d1", text, "lexical-cohesion")
    assert len(chunks) == 2
    assert "Cats eat fish daily." in chunks[0].text
    assert "Volcanoes" in chunks[1].text


def test_empty_text_yields_no_chunks():
    assert chunk_document("d1", "", "paragraph") == []


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_document("d1", TEXT, "nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests_rag/test_chunking.py -q`
Expected: `ModuleNotFoundError: No module named 'example_rag.chunking'`

- [ ] **Step 3: Write the implementation**

Create `example_rag/chunking.py`:

```python
"""Chunking strategies. Pure functions over text, no I/O.

Every strategy returns chunks that tile the document without gaps, because a
gap could swallow an answer span and show up as a retrieval failure that never
happened.

chunk_id is `{doc_id}:{start}-{end}` -- position-derived, so it is stable for a
given strategy, debuggable by eye, and guaranteed different across strategies
that cut in different places. That last part is fine: the golden set is derived
per strategy from answer offsets, never carried between them.
"""

import re
from dataclasses import dataclass

__all__ = ["Chunk", "STRATEGIES", "chunk_document"]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    start: int
    end: int


def _make(doc_id: str, text: str, start: int, end: int) -> Chunk:
    return Chunk(f"{doc_id}:{start}-{end}", text[start:end], start, end)


def _fixed(doc_id: str, text: str, size: int, overlap: int = 0) -> list[Chunk]:
    # max(1, ...) so an overlap >= size can never freeze the loop.
    step = max(1, size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(_make(doc_id, text, start, end))
        if end == len(text):
            break
        start += step
    return chunks


def _spans_between(boundaries: list[int], length: int) -> list[tuple[int, int]]:
    """Turn split points into contiguous [start, end) spans covering the text."""
    edges = [0, *boundaries, length]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    boundaries = [m.end() for m in _SENTENCE_END.finditer(text)]
    return _spans_between(boundaries, len(text))


def _paragraphs(doc_id: str, text: str) -> list[Chunk]:
    boundaries = [m.end() for m in _PARAGRAPH_BREAK.finditer(text)]
    return [_make(doc_id, text, a, b) for a, b in _spans_between(boundaries, len(text))]


def _sentences(doc_id: str, text: str, per_chunk: int) -> list[Chunk]:
    spans = _sentence_spans(text)
    chunks = []
    for i in range(0, len(spans), per_chunk):
        group = spans[i:i + per_chunk]
        chunks.append(_make(doc_id, text, group[0][0], group[-1][1]))
    return chunks


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _lexical_cohesion(doc_id: str, text: str, threshold: float = 0.15) -> list[Chunk]:
    """Start a new chunk where adjacent sentences stop sharing vocabulary.

    An embedding-free approximation of semantic chunking: Jaccard overlap of
    adjacent sentence token sets. Named for what it measures, not for what it
    approximates.
    """
    spans = _sentence_spans(text)
    if not spans:
        return []

    chunks = []
    group_start = spans[0][0]
    previous = _tokens(text[spans[0][0]:spans[0][1]])

    for start, end in spans[1:]:
        current = _tokens(text[start:end])
        union = previous | current
        similarity = len(previous & current) / len(union) if union else 0.0
        if similarity < threshold:
            chunks.append(_make(doc_id, text, group_start, start))
            group_start = start
        previous = current

    chunks.append(_make(doc_id, text, group_start, spans[-1][1]))
    return chunks


STRATEGIES = {
    "fixed-100": lambda d, t: _fixed(d, t, 100),
    "fixed-100-overlap-50": lambda d, t: _fixed(d, t, 100, 50),
    "fixed-400": lambda d, t: _fixed(d, t, 400),
    "fixed-400-overlap-100": lambda d, t: _fixed(d, t, 400, 100),
    "paragraph": _paragraphs,
    "sentence-2": lambda d, t: _sentences(d, t, 2),
    "sentence-4": lambda d, t: _sentences(d, t, 4),
    "lexical-cohesion": _lexical_cohesion,
}


def chunk_document(doc_id: str, text: str, strategy: str) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; have {sorted(STRATEGIES)}")
    if not text:
        return []
    return STRATEGIES[strategy](doc_id, text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests_rag/test_chunking.py -q`
Expected: `31 passed`

- [ ] **Step 5: Commit**

```bash
git add example_rag/chunking.py tests_rag/test_chunking.py
git commit -m "feat: eight chunking strategies that tile text without gaps"
```

---

### Task 3: Deriving the golden set

**Files:**
- Create: `example_rag/golden.py`
- Test: `tests_rag/test_golden.py`

- [ ] **Step 1: Write the failing tests**

Create `tests_rag/test_golden.py`:

```python
import pytest

from example_rag.chunking import Chunk
from example_rag.golden import relevant_chunk_ids

CHUNKS = [
    Chunk("d:0-10", "0123456789", 0, 10),
    Chunk("d:10-20", "abcdefghij", 10, 20),
    Chunk("d:20-30", "ABCDEFGHIJ", 20, 30),
]


def test_span_inside_one_chunk():
    assert relevant_chunk_ids([(12, 15)], CHUNKS) == ["d:10-20"]


def test_span_crossing_a_boundary_marks_both():
    # An answer split across a boundary is genuinely in both chunks; retrieving
    # either one gives the reader part of the answer.
    assert relevant_chunk_ids([(8, 12)], CHUNKS) == ["d:0-10", "d:10-20"]


def test_multiple_spans_union_their_chunks():
    assert relevant_chunk_ids([(2, 4), (22, 24)], CHUNKS) == ["d:0-10", "d:20-30"]


def test_result_is_deduplicated_and_ordered():
    assert relevant_chunk_ids([(1, 2), (3, 4)], CHUNKS) == ["d:0-10"]


def test_touching_the_boundary_exactly_does_not_include_the_next_chunk():
    # A span ending exactly at 10 lies wholly in the first chunk. Half-open
    # intervals: [0,10) and [10,20) do not overlap.
    assert relevant_chunk_ids([(5, 10)], CHUNKS) == ["d:0-10"]


def test_zero_length_span_is_ignored():
    assert relevant_chunk_ids([(10, 10)], CHUNKS) == []


def test_no_spans_yields_nothing():
    assert relevant_chunk_ids([], CHUNKS) == []


def test_overlapping_chunks_can_both_match():
    overlapping = [Chunk("d:0-10", "x", 0, 10), Chunk("d:5-15", "y", 5, 15)]
    assert relevant_chunk_ids([(6, 8)], overlapping) == ["d:0-10", "d:5-15"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests_rag/test_golden.py -q`
Expected: `ModuleNotFoundError: No module named 'example_rag.golden'`

- [ ] **Step 3: Write the implementation**

Create `example_rag/golden.py`:

```python
"""Deriving relevant chunk ids from answer spans.

This is what makes comparing chunking strategies possible without relabelling:
the ground truth is anchored to character offsets in the source document, so it
can be recomputed for any set of chunk boundaries.
"""

from example_rag.chunking import Chunk

__all__ = ["relevant_chunk_ids"]


def relevant_chunk_ids(spans: list[tuple[int, int]], chunks: list[Chunk]) -> list[str]:
    """Chunk ids overlapping any answer span, in document order, deduplicated.

    Spans and chunks are half-open [start, end), so an answer ending exactly at
    a boundary belongs only to the chunk before it.
    """
    out: list[str] = []
    for chunk in chunks:
        for start, end in spans:
            if end <= start:
                continue  # a zero-length span points at nothing
            if start < chunk.end and end > chunk.start:
                if chunk.chunk_id not in out:
                    out.append(chunk.chunk_id)
                break
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests_rag/test_golden.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add example_rag/golden.py tests_rag/test_golden.py
git commit -m "feat: derive golden chunk ids from answer spans"
```

---

### Task 4: BM25

**Files:**
- Create: `example_rag/bm25.py`
- Test: `tests_rag/test_bm25.py`

- [ ] **Step 1: Write the failing tests**

Create `tests_rag/test_bm25.py`:

```python
import pytest

from example_rag.bm25 import BM25, tokenize

DOCS = {
    "a": "the cat sat on the mat",
    "b": "the dog sat on the log",
    "c": "volcanoes erupt molten lava",
}


@pytest.fixture()
def index():
    return BM25(DOCS)


def test_tokenize_lowercases_and_splits_on_non_words():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_exact_term_ranks_first(index):
    assert [doc for doc, _ in index.search("volcanoes", k=1)] == ["c"]


def test_search_returns_at_most_k(index):
    assert len(index.search("the sat", k=2)) == 2


def test_distinguishing_term_beats_a_common_one(index):
    # "cat" appears once, "the" appears everywhere. IDF must prefer "cat".
    top = index.search("cat", k=1)[0][0]
    assert top == "a"


def test_unmatched_query_returns_nothing(index):
    assert index.search("helicopter", k=3) == []


def test_empty_query_returns_nothing(index):
    assert index.search("", k=3) == []


def test_results_are_sorted_by_score_descending(index):
    scores = [score for _, score in index.search("sat log", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_deterministically():
    # Two identical documents must not swap order between runs, or a rerun
    # would look like a retrieval regression.
    index = BM25({"x": "same words here", "y": "same words here"})
    assert [d for d, _ in index.search("same words", k=2)] == ["x", "y"]


def test_empty_index_is_safe():
    assert BM25({}).search("anything", k=3) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests_rag/test_bm25.py -q`
Expected: `ModuleNotFoundError: No module named 'example_rag.bm25'`

- [ ] **Step 3: Write the implementation**

Create `example_rag/bm25.py`:

```python
"""BM25 ranking. Pure Python, no dependencies.

Chunking is the variable under test, so the retriever is deliberately the
boring, deterministic option: no model weights, no randomness, no GPU, and a
rerun always produces the identical ranking.
"""

import math
import re
from collections import Counter

__all__ = ["tokenize", "BM25"]

_WORD = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    def __init__(self, documents: dict[str, str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = list(documents)
        self.tokens = {i: Counter(tokenize(t)) for i, t in documents.items()}
        self.lengths = {i: sum(c.values()) for i, c in self.tokens.items()}
        self.average_length = (
            sum(self.lengths.values()) / len(self.lengths) if self.lengths else 0.0
        )

        self.document_frequency: Counter = Counter()
        self.postings: dict[str, list[str]] = {}
        for doc_id, counts in self.tokens.items():
            self.document_frequency.update(counts.keys())
            for term in counts:
                self.postings.setdefault(term, []).append(doc_id)

        # Precomputed so tie-breaking is O(1); list.index() inside a sort key
        # would be O(n) per comparison.
        self.position = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.document_frequency.get(term, 0)
        # Standard BM25 idf with the +0.5 smoothing that keeps it positive.
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, doc_id: str) -> float:
        counts = self.tokens[doc_id]
        length = self.lengths[doc_id]
        total = 0.0
        for term in tokenize(query):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / (self.average_length or 1.0)
            )
            total += self._idf(term) * frequency * (self.k1 + 1) / denominator
        return total

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not self.doc_ids or not terms:
            return []

        # Only documents containing at least one query term can score above
        # zero, so scoring the whole corpus is wasted work.
        candidates: set[str] = set()
        for term in terms:
            candidates.update(self.postings.get(term, ()))
        if not candidates:
            return []

        scored = [(i, self.score(query, i)) for i in candidates]
        # Insertion order breaks ties, so a rerun cannot look like a regression.
        ranked = sorted(
            [(i, s) for i, s in scored if s > 0],
            key=lambda pair: (-pair[1], self.position[pair[0]]),
        )
        return ranked[:k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests_rag/test_bm25.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add example_rag/bm25.py tests_rag/test_bm25.py
git commit -m "feat: dependency-free BM25 with deterministic tie-breaking"
```

---

### Task 5: Pipeline and CLI

**Files:**
- Create: `example_rag/pipeline.py`, `example_rag/cli.py`
- Test: `tests_rag/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests_rag/test_pipeline.py`:

```python
import json

import pytest
import yaml

from example_rag.pipeline import run_strategy
from example_rag.squad import Document, Question

TEXT = "Cats like fish. Cats eat fish daily.\n\nVolcanoes erupt molten lava."
# Derived, not hand-counted. A wrong offset here would silently point at the
# wrong chunk and the suite would still look green.
_FISH = TEXT.index("fish", TEXT.index("Cats eat"))
_MOLTEN = TEXT.index("molten")

DOCS = [
    Document(
        doc_id="d1",
        text=TEXT,
        questions=[
            Question("q1", "What do cats eat?", [(_FISH, _FISH + 4)], ["fish"]),
            Question("q2", "What do volcanoes erupt?", [(_MOLTEN, _MOLTEN + 6)], ["molten"]),
        ],
    )
]


def test_writes_traces_and_golden(tmp_path):
    result = run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=None)
    assert (tmp_path / "paragraph.traces.jsonl").exists()
    assert (tmp_path / "paragraph.golden.yaml").exists()
    assert result["questions"] == 2


def test_traces_follow_the_ragmeter_contract(tmp_path):
    run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=None)
    lines = (tmp_path / "paragraph.traces.jsonl").read_text(encoding="utf-8").splitlines()
    trace = json.loads(lines[0])
    for field in ("trace_id", "question_id", "question", "retrieved", "answer", "latency_ms"):
        assert field in trace
    assert all("chunk_id" in c and "text" in c and "rank" in c for c in trace["retrieved"])


def test_golden_ids_come_from_the_same_chunking(tmp_path):
    run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=None)
    golden = yaml.safe_load((tmp_path / "paragraph.golden.yaml").read_text(encoding="utf-8"))
    ids = {c for item in golden for c in item["relevant_chunk_ids"]}

    # Golden ids must come from the same chunk space the retriever searched, or
    # recall is measured against chunks that could never have been returned.
    from example_rag.chunking import chunk_document
    available = {c.chunk_id for c in chunk_document("d1", DOCS[0].text, "paragraph")}
    assert ids
    assert ids <= available


def test_question_ids_are_stable_across_strategies(tmp_path):
    a = run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=None)
    b = run_strategy(DOCS, "sentence-2", k=2, out_dir=tmp_path, limit=None)
    assert a["question_ids"] == b["question_ids"]


def test_chunk_ids_differ_across_strategies(tmp_path):
    run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=None)
    run_strategy(DOCS, "fixed-100", k=2, out_dir=tmp_path, limit=None)

    def ids(name):
        text = (tmp_path / f"{name}.golden.yaml").read_text(encoding="utf-8")
        return {c for item in yaml.safe_load(text) for c in item["relevant_chunk_ids"]}

    # Different boundaries, different ids. This is exactly why the golden set is
    # regenerated per strategy instead of being reused.
    assert ids("paragraph") != ids("fixed-100")


def test_golden_never_contains_an_unlabelled_item(tmp_path):
    run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=None)
    golden = yaml.safe_load((tmp_path / "paragraph.golden.yaml").read_text(encoding="utf-8"))
    # ragmeter rejects an empty relevant_chunk_ids at load time, so emitting one
    # would break ingestion rather than quietly skew a number.
    assert all(item["relevant_chunk_ids"] for item in golden)


def test_limit_caps_the_question_count(tmp_path):
    result = run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=1)
    assert result["questions"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests_rag/test_pipeline.py -q`
Expected: `ModuleNotFoundError: No module named 'example_rag.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `example_rag/pipeline.py`:

```python
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
    would make a 200-question sweep slow and rate-limited for no gain in what
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

    with traces_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            if limit is not None and written >= limit:
                break  # the inner break alone would let the next document past the limit
            for question in document.questions:
                if limit is not None and written >= limit:
                    break

                relevant = relevant_chunk_ids(question.spans, chunks_by_doc[document.doc_id])
                if not relevant:
                    # No chunk contains the answer, so recall is undefined for this
                    # question. ragmeter rejects an empty label at load time, and
                    # skipping is honest where inventing a label would not be.
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
        "chunks": len(corpus),
        "question_ids": question_ids,
        "traces": str(traces_path),
        "golden": str(golden_path),
    }
```

Create `example_rag/cli.py`:

```python
"""Build trace and golden files for one or more chunking strategies."""

import argparse
from pathlib import Path

from example_rag.chunking import STRATEGIES
from example_rag.pipeline import run_strategy
from example_rag.squad import load_squad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Example BM25 RAG over SQuAD.")
    parser.add_argument("--squad", default="data/squad-dev-v1.1.json")
    parser.add_argument("--out", default="data/runs")
    parser.add_argument("--articles", type=int, default=5)
    parser.add_argument("--limit", type=int, default=200,
                        help="Questions per strategy.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategies", nargs="*", default=sorted(STRATEGIES))
    args = parser.parse_args(argv)

    documents = load_squad(args.squad, max_articles=args.articles)
    print(f"{len(documents)} documents, "
          f"{sum(len(d.questions) for d in documents)} questions available")

    for strategy in args.strategies:
        result = run_strategy(documents, strategy, k=args.k,
                              out_dir=Path(args.out), limit=args.limit)
        print(f"  {strategy:<24} {result['questions']:>4} questions  "
              f"{result['chunks']:>6} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests_rag/test_pipeline.py -q`
Expected: `7 passed`

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `296 passed, 1 skipped`

- [ ] **Step 6: Commit**

```bash
git add example_rag/pipeline.py example_rag/cli.py tests_rag/test_pipeline.py
git commit -m "feat: pipeline emitting ragmeter traces and derived golden sets"
```

---

### Task 6: The experiment

**Files:**
- Create: `docs/chunking-experiment.md`

- [ ] **Step 1: Generate the runs**

```bash
.venv/Scripts/python -m example_rag.cli --articles 5 --limit 200 --k 5
```

Expected: one traces/golden pair per strategy under `data/runs`, 200 questions each.

- [ ] **Step 2: Measure every strategy with ragmeter**

```bash
export RAGMETER_DB_URL="sqlite:///data/experiment.db"
for s in fixed-100 fixed-100-overlap-50 fixed-400 fixed-400-overlap-100 \
         paragraph sentence-2 sentence-4 lexical-cohesion; do
  .venv/Scripts/ragmeter.exe dataset load "data/runs/$s.golden.yaml" --name "$s" --version v1 > /dev/null
  .venv/Scripts/ragmeter.exe ingest "data/runs/$s.traces.jsonl" --run "$s" > /dev/null
  .venv/Scripts/ragmeter.exe eval --run "$s" --dataset "$s" --version v1 --k 5 | \
    grep -E "recall@5|ndcg@5"
done
```

- [ ] **Step 3: Compare the best against the worst**

Use `ragmeter compare` between the highest and lowest recall strategies. Because
`question_id` is the SQuAD id in every run, the comparison is paired per
question and the `+improved/-regressed` counts are meaningful.

- [ ] **Step 4: Write `docs/chunking-experiment.md`**

Record the actual table produced, the winner, and — importantly — the paired
counts, not just the means. Note the number of questions dropped because no
chunk contained the answer, per strategy: that is itself a finding about a
chunking strategy, not a defect.

- [ ] **Step 5: Commit**

```bash
git add docs/chunking-experiment.md
git commit -m "docs: chunking strategy experiment results"
```

---

## Definition of Done

- [ ] Every SQuAD answer span resolves to its answer text at document level (0 mismatches over all 48 articles)
- [ ] Every strategy tiles its document with no gaps
- [ ] `question_id` is identical across strategies; `chunk_id` is not
- [ ] The golden set is regenerated per strategy, never carried between them
- [ ] `example_rag` imports nothing from `ragmeter`
- [ ] The experiment produces a real recall comparison with paired counts

## Out of Scope

LLM generation per question: the answer is extractive, because generation is not
the variable under test and 200 LLM calls per strategy would be slow and
rate-limited for no gain in what retrieval metrics measure. Embeddings and
hybrid retrieval are a possible second experiment, not this one.
