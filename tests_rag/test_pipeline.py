import json

import yaml

from example_rag.chunking import chunk_document
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


def test_trace_ids_are_namespaced_by_strategy(tmp_path):
    run_strategy(DOCS, "paragraph", k=2, out_dir=tmp_path, limit=None)
    run_strategy(DOCS, "fixed-100", k=2, out_dir=tmp_path, limit=None)

    def trace_ids(name):
        lines = (tmp_path / f"{name}.traces.jsonl").read_text(encoding="utf-8").splitlines()
        return {json.loads(line)["trace_id"] for line in lines}

    # trace_id is a global primary key in ragmeter. Reusing question_id alone
    # would make the second strategy ingest as duplicates and silently vanish.
    assert not trace_ids("paragraph") & trace_ids("fixed-100")
