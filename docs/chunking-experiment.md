# Chunking Strategy Experiment

Eight chunking strategies, 200 SQuAD questions each, measured with ragmeter.

**Setup.** 5 SQuAD dev articles, BM25 retrieval, k=5, extractive answers. The
retriever, the questions and k are identical across every run, so a difference
in recall is attributable to chunk boundaries alone.

**How the ground truth was built.** SQuAD gives each answer's character offset.
Lifting those to document level lets the golden set be *derived* per strategy:
any chunk overlapping an answer span is relevant. No manual labelling, and no
stale labels when the boundaries move — which is the thing that normally makes
this comparison impractical.

Every strategy tiles its document without gaps, so no question was ever dropped
for lack of a containing chunk: **200 questions labelled, 0 unlabelled, in all
eight runs.**

## Results

| strategy | chunks | recall@5 | precision@5 | mrr@5 | ndcg@5 |
|---|---:|---:|---:|---:|---:|
| `sentence-4` | 312 | **0.8650** | 0.1730 | 0.6072 | 0.6718 |
| `paragraph` | 288 | 0.8600 | 0.1720 | **0.6322** | **0.6895** |
| `sentence-2` | 620 | 0.7475 | 0.1540 | 0.5765 | 0.6141 |
| `fixed-400` | 450 | 0.7150 | 0.1520 | 0.5503 | 0.5803 |
| `fixed-400-overlap-100` | 597 | 0.7117 | **0.1960** | 0.5833 | 0.5883 |
| `lexical-cohesion` | 1000 | 0.6725 | 0.1390 | 0.5396 | 0.5661 |
| `fixed-100` | 1792 | 0.4492 | 0.1120 | 0.3713 | 0.3677 |
| `fixed-100-overlap-50` | 3575 | 0.3573 | 0.1700 | 0.4181 | 0.3243 |

## Headline

Switching from `fixed-100` to `sentence-4` raised recall@5 from **44.9% to
86.5%** and cut mean retrieval latency from 3.14ms to 0.36ms.

Paired per question, that is **98 questions improved and 4 regressed** out of
200 — not an average that hides a redistribution, but a near-uniform gain.

```
metric          baseline   candidate      delta       +/-
recall@5          0.4492      0.8650     0.4158    +98/-4
ndcg@5            0.3677      0.6718     0.3041   +122/-22
mrr@5             0.3713      0.6072     0.2359   +103/-24
```

## Chunk size dominates everything else

The ordering tracks chunk size far more than cleverness. ~600-char chunks
(`paragraph`, `sentence-4`) win; 100-char chunks lose badly. At k=5 a 100-char
chunk simply cannot carry enough context to contain an answer and its
supporting sentence.

## Overlap is not free, and at small sizes it backfires

Conventional advice says add overlap so answers are not cut in half. At 100
chars it made things **worse**:

```
fixed-100 -> fixed-100-overlap-50
recall@5    0.4492 -> 0.3573   (-0.0919)   +33/-64
ndcg@5      0.3677 -> 0.3243   (-0.0434)   +39/-76
latency     3.14ms -> 7.05ms   (+124%)
```

64 questions got worse against 33 better, at double the latency and double the
index size. The reason is visible in the chunk count: overlap turned 1,792
chunks into 3,575 near-duplicates, so the five result slots fill with
overlapping windows of the same text instead of five distinct regions.

At 400 chars overlap was roughly neutral for recall (0.7150 → 0.7117) while
lifting precision (0.1520 → **0.1960**, the best in the table) and mrr. So
overlap buys ranking quality, not coverage — and only once chunks are large
enough that the duplicates are not crowding out the results.

## The clever strategy lost

`lexical-cohesion` — splitting where adjacent sentences stop sharing vocabulary,
an embedding-free stand-in for semantic chunking — scored **0.6725**, below
plain `paragraph` at 0.8600.

It produced 1,000 chunks averaging 178 characters. Having optimised for
topical coherence, it made chunks too small, and size mattered more. A
cheap heuristic aimed at the wrong variable loses to splitting on blank lines.

Whether real embedding-based semantic chunking would beat `paragraph` is an
open question this experiment does not answer. It would need chunks in the
600-character range to be a fair test.

## What to take from this

1. **Tune chunk size first.** It moved recall by 42 points; nothing else came close.
2. **`paragraph` is a strong default** when documents have real paragraph structure. It beat every fixed-size variant and cost the least to compute.
3. **Do not add overlap reflexively.** It helped precision at 400 chars and actively hurt at 100.
4. **Read the paired counts, not the mean.** `fixed-100-overlap-50` has higher *precision* than `fixed-100` while being worse overall — the mean of any single metric would have told a partial story.

## Caveats

- 5 articles, 200 questions, one corpus. SQuAD paragraphs are unusually clean and well-formed; a corpus of messy PDFs would likely reorder these.
- BM25 only. A dense retriever has different sensitivity to chunk length, and these rankings may not transfer.
- Answers are extractive, so no faithfulness or answer-relevance numbers here. Retrieval quality is what this measures.

## Reproducing

```bash
python -m example_rag.cli --articles 5 --limit 200 --k 5

export RAGMETER_DB_URL="sqlite:///data/experiment.db"
for s in fixed-100 fixed-400 paragraph sentence-4 lexical-cohesion; do
  ragmeter dataset load "data/runs/$s.golden.yaml" --name "$s" --version v1
  ragmeter ingest "data/runs/$s.traces.jsonl" --run "$s"
  ragmeter eval --run "$s" --dataset "$s" --version v1 --k 5
done

ragmeter compare --run sentence-4 --baseline fixed-100 --k 5
```
