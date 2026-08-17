"""The gate verdict. A pure function: two RunMetrics and a config in, a result out.

No database, no files, no clock. Everything interesting about the gate --
pairing, thresholds, and the fail-closed rules -- is decided here and can be
tested with dictionaries.
"""

from dataclasses import dataclass, field
from statistics import fmean

from ragmeter.gate.config import GateConfig, MetricRule
from ragmeter.metrics.aggregate import percentile

__all__ = ["RunMetrics", "MetricOutcome", "GateResult", "compare"]


@dataclass
class RunMetrics:
    """Everything the gate needs to know about one run."""

    by_question: dict[str, dict[str, float | None]]
    all_values: dict[str, list[float | None]]
    judge_failures: int = 0
    n_traces: int = 0


@dataclass
class MetricOutcome:
    name: str
    kind: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    limit: float
    passed: bool
    reason: str = ""
    n_paired: int = 0
    n_improved: int = 0
    n_regressed: int = 0


@dataclass
class GateResult:
    outcomes: list[MetricOutcome]
    n_paired: int = 0
    baseline_judge_failures: int = 0
    candidate_judge_failures: int = 0
    blocking_reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.blocking_reasons and all(o.passed for o in self.outcomes)


def _stat(values: list[float | None], stat: str) -> float | None:
    measured = [v for v in values if v is not None]
    if not measured:
        return None
    if stat == "mean":
        return fmean(measured)
    return percentile(measured, 50 if stat == "p50" else 95)


def _compare_paired(
    baseline: RunMetrics, candidate: RunMetrics, rule: MetricRule, config: GateConfig
) -> MetricOutcome:
    pairs: list[tuple[float, float]] = []
    for question_id, base_metrics in baseline.by_question.items():
        cand_metrics = candidate.by_question.get(question_id)
        if cand_metrics is None:
            continue
        before = base_metrics.get(rule.name)
        after = cand_metrics.get(rule.name)
        # A None on either side means that question was never measured. Reading
        # it as 0.0 would manufacture a regression that never happened.
        if before is None or after is None:
            continue
        pairs.append((before, after))

    out = MetricOutcome(name=rule.name, kind="paired", baseline=None, candidate=None,
                        delta=None, limit=rule.limit, passed=True, n_paired=len(pairs))

    if not pairs:
        out.passed = not config.fail_on_missing
        out.reason = f"no paired measurements for {rule.name!r}"
        return out

    deltas = [after - before for before, after in pairs]
    out.baseline = fmean(before for before, _ in pairs)
    out.candidate = fmean(after for _, after in pairs)
    out.delta = fmean(deltas)
    out.n_improved = sum(1 for d in deltas if d > 0)
    out.n_regressed = sum(1 for d in deltas if d < 0)

    if len(pairs) < config.min_samples:
        out.passed = False
        out.reason = f"only {len(pairs)} paired, min_samples is {config.min_samples}"
    elif out.delta < -rule.limit:
        out.passed = False
        out.reason = f"dropped {-out.delta:.4f}, limit is {rule.limit:.4f}"
    return out


def _compare_aggregate(
    baseline: RunMetrics, candidate: RunMetrics, rule: MetricRule, config: GateConfig
) -> MetricOutcome:
    before = _stat(baseline.all_values.get(rule.name, []), rule.stat)
    after = _stat(candidate.all_values.get(rule.name, []), rule.stat)

    out = MetricOutcome(name=f"{rule.name} ({rule.stat})", kind="aggregate",
                        baseline=before, candidate=after, delta=None,
                        limit=rule.limit, passed=True)

    if before is None or after is None:
        out.passed = not config.fail_on_missing
        out.reason = f"no measurements for {rule.name!r}"
        return out

    if before == 0:
        # Percent change from zero is undefined. Going from nothing to something
        # is a real increase and must not be reported as 0%.
        if after == 0:
            out.delta = 0.0
            return out
        out.passed = False
        out.delta = None
        out.reason = f"baseline is zero; rose to {after:.6g}, percent change undefined"
        return out

    out.delta = (after - before) / before * 100
    if out.delta > rule.limit:
        out.passed = False
        out.reason = f"rose {out.delta:.2f}%, limit is {rule.limit:.2f}%"
    return out


def compare(baseline: RunMetrics, candidate: RunMetrics, config: GateConfig) -> GateResult:
    """Decide whether the candidate run may ship."""
    shared = set(baseline.by_question) & set(candidate.by_question)
    result = GateResult(
        outcomes=[], n_paired=len(shared),
        baseline_judge_failures=baseline.judge_failures,
        candidate_judge_failures=candidate.judge_failures,
    )

    if config.fail_on_missing:
        # Fail closed: a run whose judge fell over has not been measured, and an
        # unmeasured run must never be allowed to read as an unchanged one.
        for label, run in (("baseline", baseline), ("candidate", candidate)):
            if run.judge_failures:
                result.blocking_reasons.append(
                    f"{label} has {run.judge_failures} judge failure(s); "
                    f"re-run the evaluation (cached responses make this cheap)"
                )

    for rule in config.metrics:
        if rule.is_paired:
            result.outcomes.append(_compare_paired(baseline, candidate, rule, config))
        else:
            result.outcomes.append(_compare_aggregate(baseline, candidate, rule, config))

    return result
