"""Gate configuration: load and validate gate.yaml. Pure, no I/O beyond the read.

Every validation here exists because the alternative is a gate that runs and
reports a verdict that does not mean what the author thought it meant.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_STATS = ("mean", "p50", "p95")

__all__ = ["GateConfigError", "MetricRule", "GateConfig", "load_gate_config",
           "diff_config"]


class GateConfigError(ValueError):
    """The gate configuration cannot be used as written."""


@dataclass(frozen=True)
class MetricRule:
    name: str
    max_drop: float | None = None
    max_increase_pct: float | None = None
    stat: str = "mean"

    @property
    def is_paired(self) -> bool:
        """Paired rules compare per-question deltas; aggregate rules compare a stat."""
        return self.max_drop is not None

    @property
    def limit(self) -> float:
        return self.max_drop if self.is_paired else self.max_increase_pct


@dataclass(frozen=True)
class GateConfig:
    metrics: tuple[MetricRule, ...]
    min_samples: int = 1
    fail_on_missing: bool = True


def _rule(name: str, body) -> MetricRule:
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise GateConfigError(f"metric {name!r}: expected a mapping, got {body!r}")

    max_drop = body.get("max_drop")
    max_increase_pct = body.get("max_increase_pct")
    if (max_drop is None) == (max_increase_pct is None):
        raise GateConfigError(
            f"metric {name!r}: needs exactly one of max_drop (higher is better) "
            f"or max_increase_pct (lower is better)"
        )

    threshold = max_drop if max_drop is not None else max_increase_pct
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise GateConfigError(f"metric {name!r}: threshold must be a number, got {threshold!r}")
    if threshold < 0:
        raise GateConfigError(f"metric {name!r}: threshold must not be negative")

    stat = body.get("stat", "mean")
    if stat not in VALID_STATS:
        raise GateConfigError(f"metric {name!r}: stat must be one of {VALID_STATS}, got {stat!r}")
    if "stat" in body and max_drop is not None:
        raise GateConfigError(
            f"metric {name!r}: stat is only valid with max_increase_pct; a paired "
            f"rule compares per-question deltas, so there is no aggregate to pick"
        )

    return MetricRule(name=name, max_drop=max_drop,
                      max_increase_pct=max_increase_pct, stat=stat)


def load_gate_config(path: Path) -> GateConfig:
    path = Path(path)
    if not path.is_file():
        raise GateConfigError(f"gate config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise GateConfigError(f"{path}: expected a mapping at the top level")

    metrics = raw.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        raise GateConfigError(f"{path}: needs at least one metric under 'metrics'")

    min_samples = raw.get("min_samples", 1)
    if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples < 0:
        raise GateConfigError(f"{path}: min_samples must be a non-negative integer")

    return GateConfig(
        metrics=tuple(_rule(name, body) for name, body in metrics.items()),
        min_samples=min_samples,
        fail_on_missing=bool(raw.get("fail_on_missing", True)),
    )


def diff_config(names, k: int) -> GateConfig:
    """A config that reports every metric and fails none of them.

    Used by `ragmeter compare` and by GET /v1/compare: both show the same paired
    diff the gate uses, without passing judgement.
    """
    from ragmeter.metrics.retrieval import metric_names

    paired = set(metric_names(k)) | {"faithfulness", "answer_relevance"}
    rules = tuple(
        MetricRule(name, max_drop=float("inf")) if name in paired
        else MetricRule(name, max_increase_pct=float("inf"))
        for name in sorted(names)
    )
    return GateConfig(metrics=rules, min_samples=0, fail_on_missing=False)
