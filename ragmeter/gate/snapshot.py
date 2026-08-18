"""Baseline snapshots: RunMetrics to and from a committable JSON file.

CI has no database that survives between runs, and the gate compares per
question, so an aggregate summary is not enough to pair against. A snapshot
travels with the repository the way a snapshot test does.
"""

import json
from pathlib import Path

from ragmeter.gate.compare import RunMetrics

__all__ = ["SnapshotError", "dump_snapshot", "load_snapshot"]

REQUIRED_KEYS = ("by_question", "all_values")


class SnapshotError(ValueError):
    """The baseline snapshot is missing or unusable."""


def dump_snapshot(
    metrics: RunMetrics, path: Path, run: str, k: int,
    metrics_filter: list[str] | None = None,
) -> None:
    """Write a baseline snapshot.

    Indented and key-sorted on purpose: a baseline nobody can read in a pull
    request is a baseline nobody will ever question, and stable ordering means
    a rerun with unchanged numbers produces no diff.

    `metrics_filter` restricts which metrics are stored. Leaving a timing metric
    in would make the file differ on every run -- latency is noise, and a
    baseline that always diffs is one nobody reviews.
    """
    keep = None if metrics_filter is None else set(metrics_filter)

    def prune(values: dict) -> dict:
        return values if keep is None else {n: v for n, v in values.items() if n in keep}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run": run,
                "k": k,
                "n_traces": metrics.n_traces,
                "judge_failures": metrics.judge_failures,
                "by_question": {q: prune(v) for q, v in metrics.by_question.items()},
                "all_values": prune(metrics.all_values),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_snapshot(path: Path) -> RunMetrics:
    path = Path(path)
    if not path.is_file():
        raise SnapshotError(
            f"baseline snapshot not found: {path}. Create one from a known-good "
            f"run with:  ragmeter export --run <name> --k <k> --out {path}"
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"{path}: not valid JSON: {exc}") from exc

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise SnapshotError(f"{path}: missing {', '.join(missing)}")

    return RunMetrics(
        by_question=raw["by_question"],
        all_values=raw["all_values"],
        judge_failures=raw.get("judge_failures", 0),
        n_traces=raw.get("n_traces", 0),
    )
