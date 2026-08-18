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
    available = sum(len(d.questions) for d in documents)
    print(f"{len(documents)} documents, {available} questions available")
    print(f"{'strategy':<24}{'questions':>10}{'chunks':>9}{'unlabelled':>12}")
    print("-" * 55)

    for strategy in args.strategies:
        result = run_strategy(documents, strategy, k=args.k,
                              out_dir=Path(args.out), limit=args.limit)
        print(f"{strategy:<24}{result['questions']:>10}{result['chunks']:>9}"
              f"{result['unlabelled']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
