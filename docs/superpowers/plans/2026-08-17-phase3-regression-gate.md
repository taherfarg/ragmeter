# ragmeter Phase 3 — Regression Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ragmeter gate` compares a run against a baseline and exits non-zero when quality dropped or cost rose past configured limits, so CI can block a worse deploy.

**Architecture:** The comparison is a **pure function** over two plain data structures. The database layer produces those structures; the comparison never touches a session. This is what makes the interesting logic — pairing, thresholds, fail-closed rules — testable without fixtures, and it is why this phase splits differently from the previous two.

**Tech Stack:** PyYAML, the existing schema, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`

---

## Design Rules

**Paired, not averaged.** Only `question_id`s present in *both* runs are
compared, and the verdict is built from per-question deltas. An overall mean
hides a redistribution where half the questions got much worse and half got
better — which is exactly the change you most need to catch.

**The gate fails closed.** A missing metric, an unmeasurable metric, or a judge
failure in either run is a gate failure, not a pass. A gate that says "fine"
when it could not measure is worse than no gate, because it converts absence of
evidence into evidence of absence.

**Exit codes carry meaning.** `0` pass, `1` regression, `2` config or data
error. CI must be able to tell "the model got worse" from "the tool broke".

## Two Kinds of Rule

| config | direction | comparison |
|---|---|---|
| `max_drop: 0.02` | higher is better | **paired** per-question deltas |
| `max_increase_pct: 20` | lower is better | **aggregate** (`stat`, default `mean`) |

Quality metrics are paired because the same question in both runs removes most
of the variance. Cost and latency are aggregate because what matters is total
spend and tail latency, not whether question 47 individually got cheaper.

## File Structure

| File | Responsibility |
|---|---|
| `ragmeter/gate/__init__.py` | package marker |
| `ragmeter/gate/config.py` | load and validate `gate.yaml`. Pure |
| `ragmeter/gate/collect.py` | database → `RunMetrics`. The only file here that sees a session |
| `ragmeter/gate/compare.py` | `RunMetrics` × config → `GateResult`. **Pure**, no I/O |
| `ragmeter/gate/render.py` | `GateResult` → text table. Pure |
| `ragmeter/cli.py` (modify) | `gate` and `compare` commands |
| `tests/fixtures/traces_v2.jsonl` | a candidate run with a known mixed regression |
| `tests/fixtures/gate.yaml` | a sample gate config |

---

### Task 1: Candidate-run fixture

**Files:**
- Create: `tests/fixtures/traces_v2.jsonl`

- [ ] **Step 1: Create the fixture**

Hand-computed against `golden.yaml` at k=3, versus the `traces.jsonl` baseline:

| question | baseline recall | candidate recall | delta |
|---|---|---|---|
| q1 | 1.0 | 0.0 | −1.0 regression |
| q2 | 1.0 | 0.0 | −1.0 regression |
| q3 | 1.0 | 1.0 | 0 |
| q4 | 0.0 | 1.0 | +1.0 improvement |
| q5 | 0.0 | 0.0 | 0 |

`mean_delta = −0.2`, `n_paired = 5`, `n_improved = 1`, `n_regressed = 2`.
Token counts are exactly double the baseline, so `cost_usd` mean and
`latency_ms` p95 both rise exactly 100%.

```jsonl
{"trace_id": "v1", "question_id": "q1", "question": "What is the return policy?", "retrieved": [{"chunk_id": "c3", "text": "Our warehouse operates Sunday through Thursday.", "rank": 1}], "answer": "We are open Sunday to Thursday.", "model": "openai/gpt-4o-mini", "prompt_tokens": 1600, "completion_tokens": 80, "latency_ms": 840}
{"trace_id": "v2", "question_id": "q2", "question": "How long does shipping take?", "retrieved": [{"chunk_id": "c9", "text": "Gift wrapping is available at checkout.", "rank": 1}, {"chunk_id": "c4", "text": "We accept returns by post.", "rank": 2}], "answer": "You can gift wrap at checkout.", "model": "openai/gpt-4o-mini", "prompt_tokens": 1200, "completion_tokens": 60, "latency_ms": 760}
{"trace_id": "v3", "question_id": "q3", "question": "Do you ship to the UAE?", "retrieved": [{"chunk_id": "c7", "text": "We ship to all GCC countries including the UAE.", "rank": 1}, {"chunk_id": "c8", "text": "Deliveries to Dubai and Abu Dhabi arrive within 2 days.", "rank": 2}], "answer": "Yes, we ship to the UAE.", "model": "openai/gpt-4o-mini", "prompt_tokens": 1400, "completion_tokens": 40, "latency_ms": 1000}
{"trace_id": "v4", "question_id": "q4", "question": "What payment methods are accepted?", "retrieved": [{"chunk_id": "c10", "text": "We accept Visa, Mastercard, and Apple Pay.", "rank": 1}], "answer": "Visa, Mastercard, and Apple Pay.", "model": "openai/gpt-4o-mini", "prompt_tokens": 400, "completion_tokens": 20, "latency_ms": 300}
{"trace_id": "v5", "question_id": "q5", "question": "Can I cancel an order?", "retrieved": [{"chunk_id": "c99", "text": "Our head office is located in Sharjah.", "rank": 1}], "answer": "Our office is in Sharjah.", "model": "unknown/model", "prompt_tokens": 600, "completion_tokens": 30, "latency_ms": 1800}
{"trace_id": "v6", "question": "Untracked production question", "retrieved": [{"chunk_id": "c1", "text": "Items may be returned within 30 days of delivery.", "rank": 1}], "answer": "Something else.", "model": "openai/gpt-4o-mini", "prompt_tokens": 1000, "completion_tokens": 50, "latency_ms": 600}
```

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/traces_v2.jsonl
git commit -m "test: candidate run fixture with a known mixed regression"
```

---

### Task 2: Gate config loading and validation

**Files:**
- Create: `ragmeter/gate/__init__.py`
- Create: `ragmeter/gate/config.py`
- Create: `tests/fixtures/gate.yaml`
- Test: `tests/test_gate_config.py`

- [ ] **Step 1: Create the package marker and sample config**

```bash
mkdir -p ragmeter/gate && touch ragmeter/gate/__init__.py
```

Create `tests/fixtures/gate.yaml`:

```yaml
min_samples: 3
fail_on_missing: true
metrics:
  recall@3:
    max_drop: 0.02
  cost_usd:
    stat: mean
    max_increase_pct: 20
  latency_ms:
    stat: p95
    max_increase_pct: 25
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_gate_config.py`:

```python
from pathlib import Path

import pytest

from ragmeter.gate.config import GateConfigError, load_gate_config

FIXTURES = Path(__file__).parent / "fixtures"


def write(tmp_path, text):
    path = tmp_path / "gate.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_the_sample_config():
    cfg = load_gate_config(FIXTURES / "gate.yaml")
    assert cfg.min_samples == 3
    assert cfg.fail_on_missing is True
    assert [r.name for r in cfg.metrics] == ["recall@3", "cost_usd", "latency_ms"]


def test_paired_and_aggregate_rules_are_distinguished():
    cfg = load_gate_config(FIXTURES / "gate.yaml")
    by_name = {r.name: r for r in cfg.metrics}
    assert by_name["recall@3"].is_paired is True
    assert by_name["cost_usd"].is_paired is False
    assert by_name["cost_usd"].stat == "mean"
    assert by_name["latency_ms"].stat == "p95"


def test_defaults(tmp_path):
    cfg = load_gate_config(write(tmp_path, "metrics:\n  recall@3:\n    max_drop: 0.1\n"))
    assert cfg.min_samples == 1
    # Fail-closed is the default. A gate you have to opt into is a gate that
    # silently passes on the day it matters.
    assert cfg.fail_on_missing is True


def test_aggregate_stat_defaults_to_mean(tmp_path):
    cfg = load_gate_config(
        write(tmp_path, "metrics:\n  cost_usd:\n    max_increase_pct: 10\n"))
    assert cfg.metrics[0].stat == "mean"


def test_rejects_missing_metrics_section(tmp_path):
    with pytest.raises(GateConfigError, match="at least one metric"):
        load_gate_config(write(tmp_path, "min_samples: 5\n"))


def test_rejects_empty_metrics(tmp_path):
    with pytest.raises(GateConfigError, match="at least one metric"):
        load_gate_config(write(tmp_path, "metrics: {}\n"))


def test_rejects_rule_with_neither_threshold(tmp_path):
    with pytest.raises(GateConfigError, match="exactly one of"):
        load_gate_config(write(tmp_path, "metrics:\n  recall@3:\n    stat: mean\n"))


def test_rejects_rule_with_both_thresholds(tmp_path):
    # Ambiguous direction: is higher better or worse? Refuse to guess.
    text = "metrics:\n  recall@3:\n    max_drop: 0.1\n    max_increase_pct: 5\n"
    with pytest.raises(GateConfigError, match="exactly one of"):
        load_gate_config(write(tmp_path, text))


def test_rejects_negative_threshold(tmp_path):
    with pytest.raises(GateConfigError, match="must not be negative"):
        load_gate_config(write(tmp_path, "metrics:\n  recall@3:\n    max_drop: -0.1\n"))


def test_rejects_unknown_stat(tmp_path):
    text = "metrics:\n  cost_usd:\n    stat: median\n    max_increase_pct: 5\n"
    with pytest.raises(GateConfigError, match="stat must be"):
        load_gate_config(write(tmp_path, text))


def test_rejects_stat_on_a_paired_rule(tmp_path):
    # A paired rule compares per-question deltas; there is no aggregate to pick.
    text = "metrics:\n  recall@3:\n    max_drop: 0.1\n    stat: p95\n"
    with pytest.raises(GateConfigError, match="stat is only valid"):
        load_gate_config(write(tmp_path, text))


def test_rejects_negative_min_samples(tmp_path):
    text = "min_samples: -1\nmetrics:\n  recall@3:\n    max_drop: 0.1\n"
    with pytest.raises(GateConfigError, match="min_samples"):
        load_gate_config(write(tmp_path, text))


def test_missing_file_raises(tmp_path):
    with pytest.raises(GateConfigError, match="not found"):
        load_gate_config(tmp_path / "nope.yaml")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_gate_config.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.gate.config'`

- [ ] **Step 4: Write the implementation**

Create `ragmeter/gate/config.py`:

```python
"""Gate configuration: load and validate gate.yaml. Pure, no I/O beyond the read.

Every validation here exists because the alternative is a gate that runs and
reports a verdict that does not mean what the author thought it meant.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_STATS = ("mean", "p50", "p95")

__all__ = ["GateConfigError", "MetricRule", "GateConfig", "load_gate_config"]


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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_gate_config.py -q`
Expected: `13 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/gate tests/test_gate_config.py tests/fixtures/gate.yaml
git commit -m "feat: gate config loading and validation"
```

---

### Task 3: The pure comparison

The heart of this phase. No database, no files — two data structures in, a
verdict out.

**Files:**
- Create: `ragmeter/gate/compare.py`
- Test: `tests/test_gate_compare.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_compare.py`:

```python
import pytest

from ragmeter.gate.compare import RunMetrics, compare
from ragmeter.gate.config import GateConfig, MetricRule


def metrics(by_question=None, all_values=None, judge_failures=0, n_traces=0):
    return RunMetrics(
        by_question=by_question or {},
        all_values=all_values or {},
        judge_failures=judge_failures,
        n_traces=n_traces or len(by_question or {}),
    )


def paired_config(limit=0.02, min_samples=1, fail_on_missing=True):
    return GateConfig(metrics=(MetricRule("recall@3", max_drop=limit),),
                      min_samples=min_samples, fail_on_missing=fail_on_missing)


def aggregate_config(limit=20.0, stat="mean", fail_on_missing=True):
    return GateConfig(
        metrics=(MetricRule("cost_usd", max_increase_pct=limit, stat=stat),),
        min_samples=1, fail_on_missing=fail_on_missing)


def test_identical_runs_pass():
    values = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 0.5}}
    result = compare(metrics(values), metrics(values), paired_config())
    assert result.passed is True
    assert result.outcomes[0].delta == 0.0


def test_clear_regression_fails():
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 0.0}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config(limit=0.02))
    assert result.passed is False
    assert result.outcomes[0].delta == -0.5
    assert result.outcomes[0].n_regressed == 1
    assert result.outcomes[0].n_improved == 0


def test_regression_within_tolerance_passes():
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 0.98}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config(limit=0.02))
    assert result.passed is True


def test_improvement_passes():
    base = {"q1": {"recall@3": 0.5}}
    cand = {"q1": {"recall@3": 0.9}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.passed is True
    assert result.outcomes[0].delta == pytest.approx(0.4)


def test_counts_expose_a_redistribution_the_mean_hides():
    # Mean delta is exactly zero, but one question collapsed and another jumped.
    # The counts are the only thing that reveals it.
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 0.0}}
    cand = {"q1": {"recall@3": 0.0}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    outcome = result.outcomes[0]
    assert outcome.delta == 0.0
    assert outcome.n_improved == 1
    assert outcome.n_regressed == 1
    assert result.passed is True


def test_only_questions_in_both_runs_are_paired():
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 1.0}, "q3": {"recall@3": 0.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.outcomes[0].n_paired == 1
    assert result.n_paired == 1


def test_unmeasurable_values_are_excluded_from_pairing():
    # A None on either side means that question was never measured. Treating it
    # as 0.0 would manufacture a regression that did not happen.
    base = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": None}}
    cand = {"q1": {"recall@3": 1.0}, "q2": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.outcomes[0].n_paired == 1
    assert result.passed is True


def test_too_few_pairs_fails():
    base = {"q1": {"recall@3": 1.0}}
    cand = {"q1": {"recall@3": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config(min_samples=10))
    assert result.passed is False
    assert "min_samples" in result.outcomes[0].reason


def test_metric_absent_everywhere_fails_closed():
    base = {"q1": {"other": 1.0}}
    cand = {"q1": {"other": 1.0}}
    result = compare(metrics(base), metrics(cand), paired_config())
    assert result.passed is False
    assert "no paired measurements" in result.outcomes[0].reason


def test_metric_absent_passes_when_fail_on_missing_is_off():
    base = {"q1": {"other": 1.0}}
    cand = {"q1": {"other": 1.0}}
    result = compare(metrics(base), metrics(cand),
                     paired_config(fail_on_missing=False))
    assert result.passed is True


def test_judge_failure_blocks_the_gate():
    values = {"q1": {"recall@3": 1.0}}
    result = compare(metrics(values), metrics(values, judge_failures=2),
                     paired_config())
    assert result.passed is False
    assert any("judge" in r for r in result.blocking_reasons)


def test_judge_failure_ignored_when_fail_on_missing_is_off():
    values = {"q1": {"recall@3": 1.0}}
    result = compare(metrics(values), metrics(values, judge_failures=2),
                     paired_config(fail_on_missing=False))
    assert result.passed is True


def test_aggregate_increase_within_limit_passes():
    base = metrics(all_values={"cost_usd": [1.0, 1.0]})
    cand = metrics(all_values={"cost_usd": [1.1, 1.1]})
    result = compare(base, cand, aggregate_config(limit=20.0))
    assert result.passed is True
    assert result.outcomes[0].delta == pytest.approx(10.0)


def test_aggregate_increase_past_limit_fails():
    base = metrics(all_values={"cost_usd": [1.0, 1.0]})
    cand = metrics(all_values={"cost_usd": [2.0, 2.0]})
    result = compare(base, cand, aggregate_config(limit=20.0))
    assert result.passed is False
    assert result.outcomes[0].delta == pytest.approx(100.0)


def test_aggregate_decrease_always_passes():
    base = metrics(all_values={"cost_usd": [2.0]})
    cand = metrics(all_values={"cost_usd": [1.0]})
    result = compare(base, cand, aggregate_config(limit=0.0))
    assert result.passed is True


def test_aggregate_uses_the_requested_stat():
    base = metrics(all_values={"cost_usd": [1.0, 1.0, 1.0, 100.0]})
    cand = metrics(all_values={"cost_usd": [1.0, 1.0, 1.0, 100.0]})
    result = compare(base, cand, aggregate_config(stat="p95"))
    assert result.outcomes[0].baseline == 100.0


def test_aggregate_ignores_unmeasurable_values():
    base = metrics(all_values={"cost_usd": [1.0, None]})
    cand = metrics(all_values={"cost_usd": [1.0, None]})
    result = compare(base, cand, aggregate_config())
    assert result.outcomes[0].baseline == 1.0
    assert result.passed is True


def test_aggregate_growth_from_zero_fails():
    # Percent change from zero is undefined; a rise from nothing to something
    # is a real increase and must not be silently treated as 0%.
    base = metrics(all_values={"cost_usd": [0.0]})
    cand = metrics(all_values={"cost_usd": [1.0]})
    result = compare(base, cand, aggregate_config())
    assert result.passed is False
    assert "zero" in result.outcomes[0].reason


def test_aggregate_zero_to_zero_passes():
    base = metrics(all_values={"cost_usd": [0.0]})
    cand = metrics(all_values={"cost_usd": [0.0]})
    result = compare(base, cand, aggregate_config())
    assert result.passed is True


def test_aggregate_missing_fails_closed():
    base = metrics(all_values={"cost_usd": [None]})
    cand = metrics(all_values={"cost_usd": [None]})
    result = compare(base, cand, aggregate_config())
    assert result.passed is False


def test_one_failing_metric_fails_the_whole_gate():
    config = GateConfig(metrics=(MetricRule("recall@3", max_drop=0.01),
                                 MetricRule("ndcg@3", max_drop=0.01)),
                        min_samples=1)
    base = metrics({"q1": {"recall@3": 1.0, "ndcg@3": 1.0}})
    cand = metrics({"q1": {"recall@3": 1.0, "ndcg@3": 0.0}})
    result = compare(base, cand, config)
    assert result.passed is False
    assert [o.passed for o in result.outcomes] == [True, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_gate_compare.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.gate.compare'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/gate/compare.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_gate_compare.py -q`
Expected: `21 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/gate/compare.py tests/test_gate_compare.py
git commit -m "feat: pure gate comparison with paired and aggregate rules"
```

---

### Task 4: Collecting run metrics from the database

**Files:**
- Create: `ragmeter/gate/collect.py`
- Test: `tests/test_gate_collect.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_collect.py`:

```python
from pathlib import Path

import pytest

from ragmeter.db import Evaluation, init_db, make_engine, make_session
from ragmeter.gate.collect import collect_run_metrics
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.runner import evaluate_run

FIXTURES = Path(__file__).parent / "fixtures"
PRICES = {"openai/gpt-4o-mini": (0.00000015, 0.0000006)}


@pytest.fixture()
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    s = make_session(engine)()
    load_golden(s, FIXTURES / "golden.yaml", dataset="docs", version="v1")
    for name, path in (("baseline", "traces.jsonl"), ("candidate", "traces_v2.jsonl")):
        run = get_or_create_run(s, name)
        load_traces(s, FIXTURES / path, run)
    s.commit()
    evaluate_run(s, "baseline", "docs", "v1", k=3, prices=PRICES)
    evaluate_run(s, "candidate", "docs", "v1", k=3, prices=PRICES)
    yield s
    s.close()


def test_by_question_holds_labeled_traces_only(session):
    rm = collect_run_metrics(session, "baseline", k=3)
    assert sorted(rm.by_question) == ["q1", "q2", "q3", "q4", "q5"]
    assert rm.by_question["q1"]["recall@3"] == 1.0


def test_all_values_includes_unlabeled_traces(session):
    rm = collect_run_metrics(session, "baseline", k=3)
    # by_question drops the trace with no question_id; all_values keeps it,
    # because cost and latency need no ground truth.
    assert len(rm.all_values["latency_ms"]) == 6
    assert len(rm.by_question) == 5
    assert rm.n_traces == 6


def test_candidate_recall_matches_hand_computed_values(session):
    rm = collect_run_metrics(session, "candidate", k=3)
    assert rm.by_question["q1"]["recall@3"] == 0.0
    assert rm.by_question["q2"]["recall@3"] == 0.0
    assert rm.by_question["q3"]["recall@3"] == 1.0
    assert rm.by_question["q4"]["recall@3"] == 1.0
    assert rm.by_question["q5"]["recall@3"] == 0.0


def test_judge_failures_are_counted(session):
    ev = session.query(Evaluation).filter_by(trace_id="t1", k=3).one()
    ev.judge_status = "failed"
    session.commit()
    assert collect_run_metrics(session, "baseline", k=3).judge_failures == 1


def test_repeated_question_ids_are_averaged(session):
    # Running the same golden question twice in one run is legitimate; averaging
    # keeps a single question from counting twice in the pairing.
    run = session.query(Evaluation).filter_by(trace_id="t1", k=3).one()
    run.metrics = dict(run.metrics, **{"recall@3": 1.0})
    session.commit()
    from ragmeter.db import Run, Trace
    baseline = session.query(Run).filter_by(name="baseline").one()
    session.add(Trace(trace_id="t1b", run_id=baseline.run_id, question_id="q1",
                      question="dup", retrieved=[], answer=""))
    session.flush()
    session.add(Evaluation(trace_id="t1b", k=3, metrics={"recall@3": 0.0}))
    session.commit()

    rm = collect_run_metrics(session, "baseline", k=3)
    assert rm.by_question["q1"]["recall@3"] == 0.5


def test_unknown_run_raises(session):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        collect_run_metrics(session, "nope", k=3)


def test_run_without_evaluations_raises(session):
    get_or_create_run(session, "empty")
    session.commit()
    with pytest.raises(ValueError, match="no evaluations"):
        collect_run_metrics(session, "empty", k=3)


def test_wrong_k_raises(session):
    with pytest.raises(ValueError, match="no evaluations"):
        collect_run_metrics(session, "baseline", k=99)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_gate_collect.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.gate.collect'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/gate/collect.py`:

```python
"""Database to RunMetrics. The only file in the gate package that sees a session."""

from statistics import fmean

from sqlalchemy.orm import Session

from ragmeter.db import Evaluation, Run, Trace
from ragmeter.gate.compare import RunMetrics

__all__ = ["collect_run_metrics"]


def collect_run_metrics(session: Session, run_name: str, k: int) -> RunMetrics:
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    rows = (
        session.query(Evaluation, Trace)
        .join(Trace, Trace.trace_id == Evaluation.trace_id)
        .filter(Trace.run_id == run.run_id, Evaluation.k == k)
        .all()
    )
    if not rows:
        raise ValueError(
            f"run {run_name!r} has no evaluations at k={k}; run `ragmeter eval` first"
        )

    all_values: dict[str, list[float | None]] = {}
    grouped: dict[str, dict[str, list[float]]] = {}

    for evaluation, trace in rows:
        for name, value in evaluation.metrics.items():
            all_values.setdefault(name, []).append(value)
            if trace.question_id is None or value is None:
                continue
            grouped.setdefault(trace.question_id, {}).setdefault(name, []).append(value)

    # A run may legitimately answer the same golden question more than once.
    # Averaging keeps one question from carrying extra weight in the pairing.
    by_question = {
        question_id: {name: fmean(values) for name, values in metrics.items()}
        for question_id, metrics in grouped.items()
    }

    return RunMetrics(
        by_question=by_question,
        all_values=all_values,
        judge_failures=sum(1 for ev, _ in rows if ev.judge_status == "failed"),
        n_traces=len(rows),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_gate_collect.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/gate/collect.py tests/test_gate_collect.py
git commit -m "feat: collect run metrics for gate comparison"
```

---

### Task 5: Rendering the verdict

**Files:**
- Create: `ragmeter/gate/render.py`
- Test: `tests/test_gate_render.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_render.py`:

```python
from ragmeter.gate.compare import GateResult, MetricOutcome
from ragmeter.gate.render import render_gate


def outcome(**kwargs):
    base = dict(name="recall@3", kind="paired", baseline=1.0, candidate=0.8,
                delta=-0.2, limit=0.02, passed=False, reason="dropped 0.2000",
                n_paired=5, n_improved=1, n_regressed=2)
    base.update(kwargs)
    return MetricOutcome(**base)


def test_shows_verdict_and_numbers():
    text = render_gate(GateResult(outcomes=[outcome()], n_paired=5), "cand", "base", 3)
    assert "FAIL" in text
    assert "recall@3" in text
    assert "cand" in text and "base" in text


def test_shows_improved_and_regressed_counts():
    text = render_gate(GateResult(outcomes=[outcome()], n_paired=5), "cand", "base", 3)
    # These counts are the whole point of a paired gate: they show the spread
    # that a mean delta flattens away.
    assert "1" in text and "2" in text
    assert "improved" in text.lower() or "+1" in text


def test_passing_result_says_pass():
    passing = outcome(passed=True, delta=0.0, reason="")
    text = render_gate(GateResult(outcomes=[passing], n_paired=5), "cand", "base", 3)
    assert "PASS" in text
    assert "FAIL" not in text


def test_blocking_reasons_are_shown():
    result = GateResult(outcomes=[outcome(passed=True, reason="")], n_paired=5,
                        blocking_reasons=["candidate has 2 judge failure(s)"])
    text = render_gate(result, "cand", "base", 3)
    assert "FAIL" in text
    assert "judge failure" in text


def test_unmeasurable_values_render_as_dashes():
    blank = outcome(baseline=None, candidate=None, delta=None,
                    passed=False, reason="no paired measurements")
    text = render_gate(GateResult(outcomes=[blank]), "cand", "base", 3)
    assert "-" in text
    assert "no paired measurements" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_gate_render.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.gate.render'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/gate/render.py`:

```python
"""Gate verdict as plain text, for a CI log."""

from ragmeter.gate.compare import GateResult

__all__ = ["render_gate"]


def _num(value: float | None) -> str:
    if value is None:
        return "-"
    if value != 0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def render_gate(result: GateResult, run: str, baseline: str, k: int) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        f"gate: {verdict}",
        f"  run={run}  baseline={baseline}  k={k}  paired questions={result.n_paired}",
        "",
        f"{'metric':<22}{'baseline':>12}{'candidate':>12}{'delta':>12}"
        f"{'limit':>10}{'+/-':>10}  verdict",
        "-" * 90,
    ]

    for out in result.outcomes:
        spread = f"+{out.n_improved}/-{out.n_regressed}" if out.kind == "paired" else ""
        lines.append(
            f"{out.name:<22}{_num(out.baseline):>12}{_num(out.candidate):>12}"
            f"{_num(out.delta):>12}{out.limit:>10.4f}{spread:>10}  "
            f"{'ok' if out.passed else 'FAIL'}"
            + (f"  ({out.reason})" if out.reason else "")
        )

    if result.blocking_reasons:
        lines.append("")
        lines.append("blocking:")
        lines.extend(f"  - {reason}" for reason in result.blocking_reasons)

    lines.append("")
    lines.append("+/- counts improved/regressed questions. A mean delta near zero "
                 "with a large spread is a redistribution, not a no-op.")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_gate_render.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/gate/render.py tests/test_gate_render.py
git commit -m "feat: gate verdict rendering"
```

---

### Task 6: CLI gate and compare commands

**Files:**
- Modify: `ragmeter/cli.py`
- Test: `tests/test_gate_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_cli.py`:

```python
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ragmeter.cli import app
from ragmeter.metrics.cost import MODELS_URL

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = {"data": [{"id": "openai/gpt-4o-mini",
                     "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]}

runner = CliRunner()


@pytest.fixture()
def evaluated(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGMETER_DB_URL", f"sqlite:///{tmp_path / 'gate.db'}")
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    for name, path in (("baseline", "traces.jsonl"), ("candidate", "traces_v2.jsonl")):
        runner.invoke(app, ["ingest", str(FIXTURES / path), "--run", name])
        with respx.mock:
            respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
            runner.invoke(app, ["eval", "--run", name, "--dataset", "docs",
                                "--version", "v1", "--k", "3"])
    return tmp_path


def gate_file(tmp_path, text):
    path = tmp_path / "g.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_gate_fails_on_the_known_regression(evaluated):
    config = gate_file(evaluated, "min_samples: 3\nmetrics:\n  recall@3:\n    max_drop: 0.02\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    # Exit 1 means "worse", distinct from exit 2 which means "broken".
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    assert "recall@3" in result.output


def test_gate_passes_with_a_loose_threshold(evaluated):
    config = gate_file(evaluated, "min_samples: 3\nmetrics:\n  recall@3:\n    max_drop: 0.5\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_gate_fails_on_doubled_cost(evaluated):
    # traces_v2 uses exactly double the tokens, so the mean cost rises 100%.
    config = gate_file(evaluated,
                       "metrics:\n  cost_usd:\n    stat: mean\n    max_increase_pct: 20\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 1, result.output
    assert "100" in result.output


def test_gate_reports_min_samples_shortfall(evaluated):
    config = gate_file(evaluated, "min_samples: 99\nmetrics:\n  recall@3:\n    max_drop: 0.9\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 1, result.output
    assert "min_samples" in result.output


def test_bad_config_exits_two_not_one(evaluated):
    config = gate_file(evaluated, "metrics: {}\n")
    result = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    # A broken config is not a regression. CI must be able to tell them apart.
    assert result.exit_code == 2, result.output


def test_unknown_run_exits_two(evaluated):
    config = gate_file(evaluated, "metrics:\n  recall@3:\n    max_drop: 0.5\n")
    result = runner.invoke(app, ["gate", "--run", "nope", "--baseline", "baseline",
                                 "--config", config, "--k", "3"])
    assert result.exit_code == 2, result.output
    assert "no run named" in result.output


def test_compare_shows_the_diff_without_failing(evaluated):
    result = runner.invoke(app, ["compare", "--run", "candidate",
                                 "--baseline", "baseline", "--k", "3"])
    # compare reports; it never blocks. That is what gate is for.
    assert result.exit_code == 0, result.output
    assert "recall@3" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_gate_cli.py -q`
Expected: failures with `No such command 'gate'`

- [ ] **Step 3: Add the commands to `ragmeter/cli.py`**

Add these imports below the existing ones:

```python
from ragmeter.gate.collect import collect_run_metrics
from ragmeter.gate.compare import compare
from ragmeter.gate.config import GateConfig, GateConfigError, MetricRule, load_gate_config
from ragmeter.gate.render import render_gate
from ragmeter.metrics.retrieval import metric_names
```

Append these commands to the end of the file, before the `if __name__` block:

```python
@app.command("gate")
def gate(
    run: str = typer.Option(..., "--run", help="The candidate run."),
    baseline: str = typer.Option(..., "--baseline", help="The run to compare against."),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    k: int = typer.Option(5, "--k", min=1),
) -> None:
    """Fail when a run is worse than its baseline. Exit 1 = regression, 2 = error."""
    try:
        gate_config = load_gate_config(config)
    except GateConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)

    session = _session()
    try:
        base = collect_run_metrics(session, baseline, k=k)
        cand = collect_run_metrics(session, run, k=k)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()

    result = compare(base, cand, gate_config)
    typer.echo(render_gate(result, run, baseline, k))
    # 1 is reserved for "the run got worse". A broken config or missing data
    # exits 2 above, so CI can tell a regression from a tooling failure.
    raise typer.Exit(0 if result.passed else 1)


@app.command("compare")
def compare_runs(
    run: str = typer.Option(..., "--run"),
    baseline: str = typer.Option(..., "--baseline"),
    k: int = typer.Option(5, "--k", min=1),
) -> None:
    """Show the paired diff between two runs without passing judgement."""
    session = _session()
    try:
        base = collect_run_metrics(session, baseline, k=k)
        cand = collect_run_metrics(session, run, k=k)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()

    # Every metric present in either run, with limits wide enough never to fail:
    # compare reports, gate decides.
    names = sorted(set(base.all_values) | set(cand.all_values))
    paired = set(metric_names(k)) | {"faithfulness", "answer_relevance"}
    rules = tuple(
        MetricRule(name, max_drop=float("inf")) if name in paired
        else MetricRule(name, max_increase_pct=float("inf"))
        for name in names
    )
    result = compare(base, cand, GateConfig(metrics=rules, min_samples=0,
                                            fail_on_missing=False))
    typer.echo(render_gate(result, run, baseline, k))
```

- [ ] **Step 4: Run the gate CLI tests**

Run: `.venv/Scripts/python -m pytest tests/test_gate_cli.py -q`
Expected: `7 passed`

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `175 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/cli.py tests/test_gate_cli.py
git commit -m "feat: gate and compare CLI commands"
```

---

### Task 7: End-to-end verification and docs

**Files:**
- Create: `gate.yaml`
- Modify: `README.md`

- [ ] **Step 1: Create a real gate config at the repo root**

Create `gate.yaml`:

```yaml
# Thresholds a run must clear to be allowed to ship.
# Quality rules use max_drop and are compared per question; cost rules use
# max_increase_pct and are compared on an aggregate.
min_samples: 3
fail_on_missing: true

metrics:
  recall@3:
    max_drop: 0.02
  ndcg@3:
    max_drop: 0.03
  cost_usd:
    stat: mean
    max_increase_pct: 20
  latency_ms:
    stat: p95
    max_increase_pct: 25
```

- [ ] **Step 2: Run the gate end to end**

```bash
$env:RAGMETER_DB_URL = "sqlite:///smoke.db"
.venv/Scripts/ragmeter.exe dataset load tests/fixtures/golden.yaml --name docs --version v1
.venv/Scripts/ragmeter.exe ingest tests/fixtures/traces.jsonl --run baseline
.venv/Scripts/ragmeter.exe ingest tests/fixtures/traces_v2.jsonl --run candidate
.venv/Scripts/ragmeter.exe eval --run baseline --dataset docs --version v1 --k 3
.venv/Scripts/ragmeter.exe eval --run candidate --dataset docs --version v1 --k 3
.venv/Scripts/ragmeter.exe gate --run candidate --baseline baseline --config gate.yaml --k 3
echo "exit code: $LASTEXITCODE"
```

Expected: `gate: FAIL`, `recall@3` showing delta `-0.2000` with `+1/-2`,
`cost_usd (mean)` showing `100.00%`, and **exit code 1**.

- [ ] **Step 3: Confirm the passing path**

```bash
.venv/Scripts/ragmeter.exe gate --run baseline --baseline baseline --config gate.yaml --k 3
echo "exit code: $LASTEXITCODE"
```

Expected: `gate: PASS`, exit code 0. A run compared against itself must always pass.

- [ ] **Step 4: Delete the smoke database**

```bash
Remove-Item -Force smoke.db
```

- [ ] **Step 5: Update `README.md`**

Add after the judge section:

````markdown
## Regression gate

```bash
ragmeter gate --run candidate --baseline baseline --config gate.yaml --k 3
```

Exit codes: `0` pass, `1` regression, `2` config or data error. CI needs to
tell "the model got worse" from "the tool broke".

```yaml
min_samples: 3
fail_on_missing: true
metrics:
  recall@3:    {max_drop: 0.02}          # higher is better, compared per question
  cost_usd:    {stat: mean, max_increase_pct: 20}   # lower is better, aggregate
  latency_ms:  {stat: p95, max_increase_pct: 25}
```

Quality metrics are compared **per question**, using only the questions present
in both runs. The output shows `+improved/-regressed` counts alongside the mean
delta, because a mean near zero can hide half the questions collapsing while
the other half improve.

The gate **fails closed**: a missing metric, an unmeasurable one, or a judge
failure in either run blocks the deploy. Set `fail_on_missing: false` to
override, understanding that you are asking it to pass on things it could not
measure.

`ragmeter compare` shows the same diff without a verdict or a non-zero exit.
````

Change Status to: `Phase 3 of 5. Next: judge calibration and the HTTP API.`

- [ ] **Step 6: Commit**

```bash
git add gate.yaml README.md
git commit -m "docs: regression gate usage and a sample config"
```

---

## Definition of Done

- [ ] `.venv/Scripts/python -m pytest -q` reports 175 passed
- [ ] The gate exits **1** on the fixture regression and **0** on a run against itself
- [ ] A bad config exits **2**, never 1
- [ ] A judge failure in either run blocks the gate when `fail_on_missing`
- [ ] `ragmeter/gate/compare.py` imports no session and touches no file
- [ ] The output shows improved/regressed counts, not just a mean

## Out of Scope

Calibration, HTTP API, dashboard. No statistical significance testing: the
threshold comparison is deliberate, and the upgrade path is recorded as a
`ponytail:` note in the spec if metric noise ever causes false failures.
