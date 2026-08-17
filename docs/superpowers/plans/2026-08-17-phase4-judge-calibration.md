# ragmeter Phase 4 — Judge Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect human judgements and measure how well the LLM judge agrees with them — reporting Cohen's kappa alongside the raw agreement rate, so a judge that is right by luck cannot look trustworthy.

**Architecture:** The statistics are a pure function over a list of `(judge_score, human_label)` pairs. The database supplies the pairs; an interactive CLI collects the human side. Same split as Phase 3, for the same reason: the arithmetic that matters gets tested without fixtures.

**Tech Stack:** The existing schema, Typer's prompt, pytest. **No scipy, no sklearn** — Cohen's kappa is ten lines of arithmetic.

**Spec:** `docs/superpowers/specs/2026-08-17-rag-evaluation-platform-design.md`

---

## Why Kappa

Agreement rate alone is a trap. If 90% of answers are genuinely faithful, a
judge that blindly answers "faithful" every single time scores **90% agreement**
and is worth nothing.

Cohen's kappa subtracts the agreement you would expect from chance:

```
po = observed agreement
pe = agreement expected by chance
kappa = (po - pe) / (1 - pe)
```

That same useless judge scores **kappa 0.0**. This is the number to publish, and
this phase exists to make it impossible to quote the flattering one alone.

Rough reading: `< 0` worse than chance, `0.0-0.2` negligible, `0.2-0.4` fair,
`0.4-0.6` moderate, `0.6-0.8` substantial, `> 0.8` near-perfect.

## Design Rules

**Never show the human the judge's answer before they decide.** Anchoring
would inflate agreement and make the whole measurement circular. `--show-judge`
exists for reviewing afterwards and defaults to off.

**Kappa is `None`, never `0.0`, when undefined.** When every label on both
sides is identical, `pe == 1` and the formula divides by zero. That is "cannot
be computed", which is a different statement from "no better than chance".

## File Structure

| File | Responsibility |
|---|---|
| `ragmeter/db.py` (modify) | add the `human_labels` table |
| `ragmeter/calibration.py` | pure `calibrate()`, plus the database queries |
| `ragmeter/cli.py` (modify) | `label` and `calibration` commands |

---

### Task 1: Human label table

**Files:**
- Modify: `ragmeter/db.py`
- Test: `tests/test_db.py` (append)

- [ ] **Step 1: Write the failing test**

Update the import at the top of `tests/test_db.py`:

```python
from ragmeter.db import (
    Evaluation, GoldenItem, HumanLabel, JudgeCache, Run, Trace,
    init_db, make_engine, make_session,
)
```

Append to `tests/test_db.py`:

```python
def test_human_label_roundtrip(session):
    run = Run(name="r")
    session.add(run)
    session.flush()
    session.add(Trace(trace_id="t1", run_id=run.run_id, question="why?"))
    session.flush()
    session.add(HumanLabel(trace_id="t1", metric="faithfulness",
                           value=1.0, labeler="taher"))
    session.commit()

    loaded = session.query(HumanLabel).filter_by(trace_id="t1").one()
    assert loaded.value == 1.0
    assert loaded.metric == "faithfulness"
    assert loaded.label_id is not None


def test_one_label_per_trace_metric_labeler(session):
    run = Run(name="r")
    session.add(run)
    session.flush()
    session.add(Trace(trace_id="t1", run_id=run.run_id, question="why?"))
    session.flush()
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=1.0, labeler="a"))
    session.commit()
    # Relabelling must replace, not accumulate: two contradictory labels from
    # the same person would silently double-count in the kappa.
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=0.0, labeler="a"))
    with pytest.raises(IntegrityError):
        session.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -q`
Expected: `ImportError: cannot import name 'HumanLabel'`

- [ ] **Step 3: Add the table**

In `ragmeter/db.py`, add `"HumanLabel"` to `__all__`, add `UniqueConstraint` to
the `sqlalchemy` import line, and add this class after `JudgeCache`:

```python
class HumanLabel(Base):
    """A person's binary verdict on one trace, for calibrating the judge.

    Binary only: the label CLI asks yes/no. Graded labels and rank correlation
    are deferred until binary proves insufficient -- see the spec.
    """

    __tablename__ = "human_labels"
    __table_args__ = (
        # Relabelling replaces. Two contradictory labels from one person on one
        # trace would silently double-count in the kappa.
        UniqueConstraint("trace_id", "metric", "labeler", name="uq_label_once"),
    )

    label_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.trace_id"), index=True)
    metric: Mapped[str] = mapped_column(String(50), index=True)
    value: Mapped[float] = mapped_column(Float)
    labeler: Mapped[str] = mapped_column(String(100), default="human")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/db.py tests/test_db.py
git commit -m "feat: human_labels table for judge calibration"
```

---

### Task 2: The calibration statistics

Pure arithmetic. No database in this task at all.

**Files:**
- Create: `ragmeter/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration.py`:

```python
import pytest

from ragmeter.calibration import calibrate

# Every expected value below is computed by hand from the kappa formula.


def test_perfect_agreement():
    # judge 1,1,0,0 vs human 1,1,0,0
    # po = 1.0; P(j=1)=0.5, P(h=1)=0.5 -> pe = 0.25 + 0.25 = 0.5
    # kappa = (1.0 - 0.5) / (1 - 0.5) = 1.0
    result = calibrate([(1.0, 1.0), (0.9, 1.0), (0.1, 0.0), (0.0, 0.0)])
    assert result.n == 4
    assert result.agreement == 1.0
    assert result.kappa == 1.0


def test_a_lazy_judge_scores_high_agreement_and_zero_kappa():
    # THE point of this module. The judge says "good" every time. Nine of ten
    # answers really are good, so it agrees 90% of the time -- and is useless.
    # po = 0.9; P(j=1)=1.0, P(h=1)=0.9 -> pe = 1.0*0.9 + 0.0*0.1 = 0.9
    # kappa = (0.9 - 0.9) / (1 - 0.9) = 0.0
    pairs = [(1.0, 1.0)] * 9 + [(1.0, 0.0)]
    result = calibrate(pairs)
    assert result.agreement == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.0)


def test_worked_example():
    # judge 1,1,1,1,0,0,0,0,0,0  human 1,1,0,0,1,0,0,0,0,0
    # a=2 b=2 c=1 d=5, n=10
    # po = (2+5)/10 = 0.7
    # P(j=1)=0.4, P(h=1)=0.3 -> pe = 0.4*0.3 + 0.6*0.7 = 0.12 + 0.42 = 0.54
    # kappa = (0.7 - 0.54) / (1 - 0.54) = 0.16/0.46 = 0.34782608695652173
    judge = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    human = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    result = calibrate([(float(j), float(h)) for j, h in zip(judge, human)])
    assert result.agreement == pytest.approx(0.7)
    assert result.kappa == pytest.approx(0.34782608695652173)
    assert result.confusion == {"both_yes": 2, "judge_yes_human_no": 2,
                                "judge_no_human_yes": 1, "both_no": 5}


def test_complete_disagreement_is_negative():
    # a=0 b=1 c=1 d=0; po=0; P(j=1)=0.5, P(h=1)=0.5 -> pe=0.5
    # kappa = (0 - 0.5)/(1 - 0.5) = -1.0
    result = calibrate([(1.0, 0.0), (0.0, 1.0)])
    assert result.agreement == 0.0
    assert result.kappa == -1.0


def test_kappa_undefined_when_everything_is_one_class():
    # pe == 1, so the formula divides by zero. That is "cannot compute", which
    # is a different claim from "no better than chance".
    result = calibrate([(1.0, 1.0), (1.0, 1.0), (0.9, 1.0)])
    assert result.agreement == 1.0
    assert result.kappa is None
    assert "undefined" in result.kappa_note


def test_no_pairs_is_all_none():
    result = calibrate([])
    assert result.n == 0
    assert result.agreement is None
    assert result.kappa is None


def test_threshold_binarizes_the_judge_score():
    # 0.6 is above the default 0.5 and below a 0.8 threshold.
    assert calibrate([(0.6, 1.0)], threshold=0.5).confusion["both_yes"] == 1
    assert calibrate([(0.6, 1.0)], threshold=0.8).confusion["judge_no_human_yes"] == 1


def test_threshold_boundary_counts_as_yes():
    # >= not >. Half the claims supported reads as "supported" at 0.5.
    assert calibrate([(0.5, 1.0)], threshold=0.5).confusion["both_yes"] == 1


def test_human_values_are_binarized_too():
    result = calibrate([(1.0, 1.0), (0.0, 0.0)])
    assert result.confusion["both_yes"] == 1
    assert result.confusion["both_no"] == 1


def test_agreement_and_kappa_can_diverge_sharply():
    # 8 of 10 agree, but the judge barely tracks the human.
    # judge 1,1,1,1,1,1,1,1,1,0  human 1,1,1,1,1,1,1,1,0,1
    # a=8 b=1 c=1 d=0; po=0.8
    # P(j=1)=0.9, P(h=1)=0.9 -> pe = 0.81 + 0.01 = 0.82
    # kappa = (0.8 - 0.82)/(1 - 0.82) = -0.02/0.18 = -0.1111...
    judge = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    human = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    result = calibrate([(float(j), float(h)) for j, h in zip(judge, human)])
    assert result.agreement == pytest.approx(0.8)
    assert result.kappa == pytest.approx(-0.1111111111111111)
    # 80% agreement, worse than chance. Quoting agreement alone would be a lie.
    assert result.kappa < 0


def test_result_carries_the_threshold_used():
    assert calibrate([(1.0, 1.0)], threshold=0.7).threshold == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_calibration.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.calibration'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/calibration.py`:

```python
"""How well does the judge agree with a human?

Reports Cohen's kappa next to the raw agreement rate, because agreement alone
is flattering nonsense: on a corpus where 90% of answers are good, a judge that
always says "good" agrees 90% of the time and has learned nothing.

Pure arithmetic. No numpy, no scipy, no sklearn.
"""

from dataclasses import dataclass, field

__all__ = ["Calibration", "calibrate"]


@dataclass
class Calibration:
    n: int
    agreement: float | None
    kappa: float | None
    threshold: float
    kappa_note: str = ""
    confusion: dict[str, int] = field(default_factory=dict)


def calibrate(pairs: list[tuple[float, float]], threshold: float = 0.5) -> Calibration:
    """Compare judge scores against human labels.

    `pairs` is (judge_score, human_label). Both are binarized at `threshold`
    using >=, so a faithfulness of exactly 0.5 reads as supported.
    """
    if not pairs:
        return Calibration(n=0, agreement=None, kappa=None, threshold=threshold,
                           kappa_note="no labelled pairs")

    counts = {"both_yes": 0, "judge_yes_human_no": 0,
              "judge_no_human_yes": 0, "both_no": 0}
    for judge_score, human_value in pairs:
        judge_yes = judge_score >= threshold
        human_yes = human_value >= threshold
        if judge_yes and human_yes:
            counts["both_yes"] += 1
        elif judge_yes:
            counts["judge_yes_human_no"] += 1
        elif human_yes:
            counts["judge_no_human_yes"] += 1
        else:
            counts["both_no"] += 1

    n = len(pairs)
    agreement = (counts["both_yes"] + counts["both_no"]) / n

    judge_yes_rate = (counts["both_yes"] + counts["judge_yes_human_no"]) / n
    human_yes_rate = (counts["both_yes"] + counts["judge_no_human_yes"]) / n
    expected = (judge_yes_rate * human_yes_rate
                + (1 - judge_yes_rate) * (1 - human_yes_rate))

    if expected >= 1.0:
        # Every label on both sides fell in one class, so chance alone predicts
        # perfect agreement and kappa has no denominator. Undefined is not zero.
        return Calibration(n=n, agreement=agreement, kappa=None, threshold=threshold,
                           confusion=counts,
                           kappa_note="undefined: all labels fall in one class, "
                                      "so chance already predicts full agreement")

    return Calibration(n=n, agreement=agreement,
                       kappa=(agreement - expected) / (1 - expected),
                       threshold=threshold, confusion=counts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_calibration.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/calibration.py tests/test_calibration.py
git commit -m "feat: agreement rate and Cohen kappa, no numpy or scipy"
```

---

### Task 3: Database queries for calibration

**Files:**
- Modify: `ragmeter/calibration.py`
- Test: `tests/test_calibration_db.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration_db.py`:

```python
from pathlib import Path

import pytest

from ragmeter.calibration import collect_labeled_pairs, unlabeled_traces
from ragmeter.db import Evaluation, HumanLabel, init_db, make_engine, make_session
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
    run = get_or_create_run(s, "baseline")
    load_traces(s, FIXTURES / "traces.jsonl", run)
    s.commit()
    evaluate_run(s, "baseline", "docs", "v1", k=3, prices=PRICES)
    # Give three traces a judge faithfulness score; leave the rest unmeasured.
    for trace_id, score in (("t1", 1.0), ("t2", 0.5), ("t3", 0.0)):
        ev = s.query(Evaluation).filter_by(trace_id=trace_id, k=3).one()
        ev.metrics = dict(ev.metrics, faithfulness=score)
    s.commit()
    yield s
    s.close()


def test_unlabeled_lists_only_traces_with_a_judge_score(session):
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="me", limit=10)
    assert [t.trace_id for t, _ in pending] == ["t1", "t2", "t3"]


def test_unlabeled_respects_the_limit(session):
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="me", limit=2)
    assert len(pending) == 2


def test_unlabeled_excludes_what_this_person_already_labelled(session):
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=1.0, labeler="me"))
    session.commit()
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="me", limit=10)
    assert [t.trace_id for t, _ in pending] == ["t2", "t3"]


def test_unlabeled_is_per_labeler(session):
    # Two people labelling the same trace is the point of inter-rater work,
    # so one person's label must not hide the trace from another.
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=1.0, labeler="me"))
    session.commit()
    pending = unlabeled_traces(session, "baseline", "faithfulness", k=3,
                               labeler="someone-else", limit=10)
    assert [t.trace_id for t, _ in pending] == ["t1", "t2", "t3"]


def test_collect_pairs_matches_judge_scores_to_labels(session):
    session.add(HumanLabel(trace_id="t1", metric="faithfulness", value=1.0, labeler="me"))
    session.add(HumanLabel(trace_id="t3", metric="faithfulness", value=0.0, labeler="me"))
    session.commit()
    pairs = collect_labeled_pairs(session, "baseline", "faithfulness", k=3)
    assert sorted(pairs) == [(0.0, 0.0), (1.0, 1.0)]


def test_collect_pairs_skips_labels_without_a_judge_score(session):
    # t4 has no faithfulness score; a human label on it cannot be compared.
    session.add(HumanLabel(trace_id="t4", metric="faithfulness", value=1.0, labeler="me"))
    session.commit()
    assert collect_labeled_pairs(session, "baseline", "faithfulness", k=3) == []


def test_collect_pairs_ignores_other_metrics(session):
    session.add(HumanLabel(trace_id="t1", metric="answer_relevance",
                           value=1.0, labeler="me"))
    session.commit()
    assert collect_labeled_pairs(session, "baseline", "faithfulness", k=3) == []


def test_unknown_run_raises(session):
    with pytest.raises(ValueError, match="no run named 'nope'"):
        collect_labeled_pairs(session, "nope", "faithfulness", k=3)
    with pytest.raises(ValueError, match="no run named 'nope'"):
        unlabeled_traces(session, "nope", "faithfulness", k=3, labeler="me", limit=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_calibration_db.py -q`
Expected: `ImportError: cannot import name 'collect_labeled_pairs'`

- [ ] **Step 3: Append the query functions to `ragmeter/calibration.py`**

Add these imports below the `dataclasses` import:

```python
from sqlalchemy.orm import Session

from ragmeter.db import Evaluation, HumanLabel, Run, Trace
```

Change `__all__` to:

```python
__all__ = ["Calibration", "calibrate", "collect_labeled_pairs", "unlabeled_traces"]
```

Append to the end of the file:

```python
def collect_labeled_pairs(
    session: Session, run_name: str, metric: str, k: int
) -> list[tuple[float, float]]:
    """(judge_score, human_label) for every trace in the run that has both."""
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    rows = (
        session.query(Evaluation, HumanLabel)
        .join(Trace, Trace.trace_id == Evaluation.trace_id)
        .join(HumanLabel, HumanLabel.trace_id == Evaluation.trace_id)
        .filter(Trace.run_id == run.run_id, Evaluation.k == k,
                HumanLabel.metric == metric)
        .all()
    )

    pairs = []
    for evaluation, label in rows:
        score = evaluation.metrics.get(metric)
        # A judge score of None was never measured, so there is nothing to
        # compare the human against.
        if score is not None:
            pairs.append((float(score), float(label.value)))
    return pairs


def unlabeled_traces(
    session: Session, run_name: str, metric: str, k: int, labeler: str, limit: int
) -> list[tuple[Trace, Evaluation]]:
    """Traces with a judge score for this metric and no label from this person yet."""
    run = session.query(Run).filter_by(name=run_name).one_or_none()
    if run is None:
        raise ValueError(f"no run named {run_name!r}")

    already = {
        row.trace_id
        for row in session.query(HumanLabel)
        .filter_by(metric=metric, labeler=labeler)
        .all()
    }

    rows = (
        session.query(Trace, Evaluation)
        .join(Evaluation, Evaluation.trace_id == Trace.trace_id)
        .filter(Trace.run_id == run.run_id, Evaluation.k == k)
        .order_by(Trace.trace_id)
        .all()
    )

    out = []
    for trace, evaluation in rows:
        if trace.trace_id in already:
            continue
        if evaluation.metrics.get(metric) is None:
            continue
        out.append((trace, evaluation))
        if len(out) == limit:
            break
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_calibration_db.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/calibration.py tests/test_calibration_db.py
git commit -m "feat: calibration database queries"
```

---

### Task 4: The `label` and `calibration` CLI commands

**Files:**
- Modify: `ragmeter/cli.py`
- Test: `tests/test_calibration_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibration_cli.py`:

```python
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ragmeter.cli import app
from ragmeter.db import Evaluation, HumanLabel, init_db, make_engine, make_session
from ragmeter.metrics.cost import MODELS_URL

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = {"data": [{"id": "openai/gpt-4o-mini",
                     "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}]}

runner = CliRunner()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'cal.db'}"
    monkeypatch.setenv("RAGMETER_DB_URL", url)
    runner.invoke(app, ["dataset", "load", str(FIXTURES / "golden.yaml"),
                        "--name", "docs", "--version", "v1"])
    runner.invoke(app, ["ingest", str(FIXTURES / "traces.jsonl"), "--run", "baseline"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=CATALOG))
        runner.invoke(app, ["eval", "--run", "baseline", "--dataset", "docs",
                            "--version", "v1", "--k", "3"])

    session = make_session(make_engine(url))()
    for trace_id, score in (("t1", 1.0), ("t2", 1.0), ("t3", 0.0)):
        ev = session.query(Evaluation).filter_by(trace_id=trace_id, k=3).one()
        ev.metrics = dict(ev.metrics, faithfulness=score)
    session.commit()
    session.close()
    return url


def session_for(url):
    engine = make_engine(url)
    init_db(engine)
    return make_session(engine)()


def test_label_stores_answers_in_order(db):
    result = runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                                 "--k", "3", "--labeler", "me"], input="y\nn\ny\n")
    assert result.exit_code == 0, result.output

    session = session_for(db)
    labels = {row.trace_id: row.value
              for row in session.query(HumanLabel).filter_by(labeler="me")}
    assert labels == {"t1": 1.0, "t2": 0.0, "t3": 1.0}
    session.close()


def test_label_hides_the_judge_score_by_default(db):
    result = runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                                 "--k", "3", "--limit", "1"], input="y\n")
    # Showing the judge's verdict first would anchor the human and make the
    # agreement number circular.
    assert "judge faithfulness" not in result.output.lower()


def test_label_can_reveal_the_judge_score_on_request(db):
    result = runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                                 "--k", "3", "--limit", "1", "--show-judge"], input="y\n")
    assert "judge faithfulness" in result.output.lower()


def test_label_skip_stores_nothing(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="s\ns\ns\n")
    session = session_for(db)
    assert session.query(HumanLabel).count() == 0
    session.close()


def test_label_quit_stops_early(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="y\nq\n")
    session = session_for(db)
    assert session.query(HumanLabel).count() == 1
    session.close()


def test_label_is_resumable(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me", "--limit", "1"], input="y\n")
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me", "--limit", "1"], input="n\n")
    session = session_for(db)
    labels = {row.trace_id: row.value for row in session.query(HumanLabel)}
    # The second pass moves on to the next unlabelled trace rather than repeating.
    assert labels == {"t1": 1.0, "t2": 0.0}
    session.close()


def test_label_reports_when_nothing_is_pending(db):
    result = runner.invoke(app, ["label", "--run", "baseline",
                                 "--metric", "answer_relevance", "--k", "3"])
    assert result.exit_code == 0
    assert "nothing to label" in result.output.lower()


def test_calibration_reports_agreement_and_kappa(db):
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="y\ny\nn\n")
    result = runner.invoke(app, ["calibration", "--run", "baseline",
                                 "--metric", "faithfulness", "--k", "3"])
    assert result.exit_code == 0, result.output
    # judge 1,1,0 vs human 1,1,0 -> perfect agreement, kappa 1.0
    assert "1.0000" in result.output
    assert "kappa" in result.output.lower()


def test_calibration_warns_when_kappa_lags_agreement(db):
    # judge says yes to t1 and t2, no to t3; the human says yes to all three.
    # a=2 b=0 c=1 d=0, po=2/3; P(j=1)=2/3, P(h=1)=1 -> pe=2/3; kappa=0.0
    runner.invoke(app, ["label", "--run", "baseline", "--metric", "faithfulness",
                        "--k", "3", "--labeler", "me"], input="y\ny\ny\n")
    result = runner.invoke(app, ["calibration", "--run", "baseline",
                                 "--metric", "faithfulness", "--k", "3"])
    assert "0.6667" in result.output
    assert "chance" in result.output.lower()


def test_calibration_without_labels_exits_two(db):
    result = runner.invoke(app, ["calibration", "--run", "baseline",
                                 "--metric", "faithfulness", "--k", "3"])
    assert result.exit_code == 2
    assert "no labelled pairs" in result.output.lower()


def test_calibration_unknown_run_exits_two(db):
    result = runner.invoke(app, ["calibration", "--run", "nope",
                                 "--metric", "faithfulness", "--k", "3"])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_calibration_cli.py -q`
Expected: failures with `No such command 'label'`

- [ ] **Step 3: Add the commands to `ragmeter/cli.py`**

Add to the imports:

```python
from ragmeter.calibration import calibrate, collect_labeled_pairs, unlabeled_traces
```

and change the existing db import line to:

```python
from ragmeter.db import HumanLabel, init_db, make_engine, make_session
```

Add these constants after the `app.add_typer(...)` line:

```python
# What to actually ask a person, per metric. Phrasing matters: the question must
# be answerable from the screen alone, without the judge's opinion.
LABEL_QUESTIONS = {
    "faithfulness": "Is every claim in the answer supported by the sources above?",
    "answer_relevance": "Does the answer address the question?",
}

YES = {"y", "yes"}
NO = {"n", "no"}
QUIT = {"q", "quit"}
```

Add these commands before the `if __name__` block:

```python
@app.command("label")
def label(
    run: str = typer.Option(..., "--run"),
    metric: str = typer.Option("faithfulness", "--metric"),
    k: int = typer.Option(5, "--k", min=1),
    limit: int = typer.Option(20, "--limit", min=1),
    labeler: str = typer.Option("human", "--labeler"),
    show_judge: bool = typer.Option(
        False, "--show-judge/--hide-judge",
        help="Reveal the judge's score. Off by default: seeing it first anchors "
             "your answer and makes the agreement number circular."),
) -> None:
    """Collect human yes/no judgements to calibrate the LLM judge against."""
    question = LABEL_QUESTIONS.get(metric, f"Is {metric} satisfactory?")

    session = _session()
    try:
        pending = unlabeled_traces(session, run, metric, k=k, labeler=labeler, limit=limit)
    except ValueError as exc:
        session.close()
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)

    if not pending:
        session.close()
        typer.echo(f"nothing to label for {metric!r} in run {run!r} as {labeler!r}")
        return

    stored = 0
    skipped = 0
    try:
        for index, (trace, evaluation) in enumerate(pending, start=1):
            typer.echo("=" * 72)
            typer.echo(f"[{index}/{len(pending)}]  trace {trace.trace_id}")
            typer.echo(f"\nQUESTION: {trace.question}")
            typer.echo("\nSOURCES:")
            for chunk in trace.retrieved or []:
                text = chunk.get("text") or "(no text captured)"
                typer.echo(f"  [{chunk['chunk_id']}] {text}")
            typer.echo(f"\nANSWER: {trace.answer}")
            if show_judge:
                typer.echo(f"\njudge {metric}: {evaluation.metrics.get(metric)}")
            typer.echo("")

            answer = typer.prompt(f"{question} [y/n/s=skip/q=quit]").strip().lower()
            if answer in QUIT:
                break
            if answer not in YES and answer not in NO:
                # Anything unrecognised is a skip rather than a re-prompt. The
                # trace stays pending, so the next run offers it again.
                typer.echo("  not recognised, skipping")
                skipped += 1
                continue

            session.add(HumanLabel(trace_id=trace.trace_id, metric=metric,
                                   value=1.0 if answer in YES else 0.0,
                                   labeler=labeler))
            session.commit()
            stored += 1
    finally:
        session.close()

    typer.echo("")
    typer.echo(f"labelled {stored}, skipped {skipped} as {labeler!r}")


@app.command("calibration")
def calibration(
    run: str = typer.Option(..., "--run"),
    metric: str = typer.Option("faithfulness", "--metric"),
    k: int = typer.Option(5, "--k", min=1),
    threshold: float = typer.Option(0.5, "--threshold", min=0.0, max=1.0),
) -> None:
    """Measure how well the judge agrees with human labels."""
    session = _session()
    try:
        pairs = collect_labeled_pairs(session, run, metric, k=k)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()

    result = calibrate(pairs, threshold=threshold)
    if result.n == 0:
        typer.echo(f"error: no labelled pairs for {metric!r} in run {run!r}; "
                   f"run `ragmeter label` first", err=True)
        raise typer.Exit(2)

    typer.echo(f"calibration: {metric}   run={run}   k={k}   threshold={threshold}")
    typer.echo(f"  pairs                 {result.n}")
    typer.echo(f"  agreement rate        {result.agreement:.4f}")
    if result.kappa is None:
        typer.echo(f"  Cohen's kappa         -  ({result.kappa_note})")
    else:
        typer.echo(f"  Cohen's kappa         {result.kappa:.4f}")
    typer.echo("")
    typer.echo(f"  both yes              {result.confusion['both_yes']}")
    typer.echo(f"  judge yes / human no  {result.confusion['judge_yes_human_no']}")
    typer.echo(f"  judge no / human yes  {result.confusion['judge_no_human_yes']}")
    typer.echo(f"  both no               {result.confusion['both_no']}")

    if result.kappa is not None and result.agreement - result.kappa > 0.2:
        typer.echo("")
        typer.echo(
            f"NOTE: agreement {result.agreement:.2f} but kappa {result.kappa:.2f}. "
            f"Most of that agreement is chance, because the labels are lopsided. "
            f"Quote the kappa."
        )
```

- [ ] **Step 4: Run the CLI tests**

Run: `.venv/Scripts/python -m pytest tests/test_calibration_cli.py -q`
Expected: `11 passed`

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `208 passed`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/cli.py tests/test_calibration_cli.py
git commit -m "feat: label and calibration CLI commands"
```

---

### Task 5: End-to-end verification and docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the labelling flow against real judge output**

```bash
$env:RAGMETER_DB_URL = "sqlite:///smoke.db"
.venv/Scripts/ragmeter.exe dataset load tests/fixtures/golden.yaml --name docs --version v1
.venv/Scripts/ragmeter.exe ingest tests/fixtures/traces.jsonl --run baseline
.venv/Scripts/ragmeter.exe eval --run baseline --dataset docs --version v1 --k 3 --judge
"y`ny`ny`ny`n" | .venv/Scripts/ragmeter.exe label --run baseline --metric faithfulness --k 3 --labeler taher
.venv/Scripts/ragmeter.exe calibration --run baseline --metric faithfulness --k 3
```

Expected: the label screens show question, sources, and answer but **not** the
judge score; calibration then prints agreement, kappa, and the confusion counts.
Because every human answer here is "yes" while the judge said no to some, kappa
should sit visibly below agreement and the NOTE should appear.

- [ ] **Step 2: Delete the smoke database**

```bash
Remove-Item -Force smoke.db
```

- [ ] **Step 3: Update `README.md`**

Add before the Trace format section:

````markdown
## Judge calibration

An LLM judge is only worth what its agreement with a human is worth. Measure it:

```bash
ragmeter label --run baseline --metric faithfulness --k 3 --labeler you
ragmeter calibration --run baseline --metric faithfulness --k 3
```

```
calibration: faithfulness   run=baseline   k=3   threshold=0.5
  pairs                 10
  agreement rate        0.9000
  Cohen's kappa         0.0000

NOTE: agreement 0.90 but kappa 0.00. Most of that agreement is chance,
because the labels are lopsided. Quote the kappa.
```

**Agreement rate alone is a trap.** If 90% of your answers are genuinely
faithful, a judge that blindly says "faithful" every time scores 90% agreement
and is worthless. Cohen's kappa subtracts chance agreement and gives that judge
the 0.0 it deserves. Kappa is the number to publish.

Reading kappa: `< 0` worse than chance, `0.0-0.2` negligible, `0.2-0.4` fair,
`0.4-0.6` moderate, `0.6-0.8` substantial, `> 0.8` near-perfect.

`label` **hides the judge's score by default**. Seeing it first would anchor
your answer and make the agreement number circular. Pass `--show-judge` only
when reviewing after the fact.

Labelling is resumable: each run offers traces you have not judged yet, and
labels are per-labeler so two people can rate the same traces.
````

Change Status to: `Phase 4 of 5. Next: the HTTP API.`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: judge calibration"
```

---

## Definition of Done

- [ ] `.venv/Scripts/python -m pytest -q` reports 208 passed
- [ ] A judge that always says yes scores high agreement and **kappa 0.0**
- [ ] Kappa is `None` with an explanation when undefined, never `0.0`
- [ ] `label` does not print the judge's score unless `--show-judge` is passed
- [ ] Labelling is resumable and per-labeler
- [ ] `calibration.py` needs no numpy, scipy, or sklearn

## Out of Scope

The HTTP API and the dashboard. Graded (non-binary) human labels and Spearman
correlation stay deferred: they would pull in scipy for a signal binary labels
have not yet proven insufficient.
