# Wiring the Regression Gate into CI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ragmeter gate` runnable in CI as one command against a baseline committed to the repository, so a change that makes retrieval worse fails the build.

**Architecture:** The gate's comparison is already a pure function over `RunMetrics`. CI has no database that survives between runs, so the missing piece is getting a `RunMetrics` in and out of a file. That is an export command and a `--baseline-file` flag — not a new comparison path.

**Tech Stack:** stdlib `json`, the existing gate package, GitHub Actions.

---

## Why a File and Not a Database

The gate compares **per question**, so a baseline of aggregate numbers is not
enough — pairing needs each question's value. That rules out the obvious
"store the summary" approach.

Regenerating the baseline in CI by checking out the main branch and rerunning
is the alternative. It doubles CI time and quietly redefines the baseline on
every push, which is the opposite of what a regression gate is for.

So: a committed snapshot, the way snapshot tests work. `RunMetrics` is already
a plain dataclass of JSON-friendly types, so export is `asdict` and import is
the constructor.

## Design Rules

**A missing baseline is exit 2, not exit 1.** No baseline means the gate could
not measure, which is a data problem, not a regression. It must also print the
exact command that creates one, or the first CI run becomes a puzzle.

**The exported file is human-readable and diffable.** A baseline you cannot
read in a pull request is a baseline nobody will ever question.

## File Structure

| File | Responsibility |
|---|---|
| `ragmeter/gate/snapshot.py` | `RunMetrics` to and from JSON |
| `ragmeter/cli.py` (modify) | `export` command; `gate --baseline-file` |
| `.github/workflows/ci.yml` | test job, then gate job |
| `baselines/` | committed baseline snapshots |

---

### Task 1: Baseline snapshots

**Files:**
- Create: `ragmeter/gate/snapshot.py`
- Test: `tests/test_gate_snapshot.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_snapshot.py`:

```python
import json

import pytest

from ragmeter.gate.compare import RunMetrics, compare
from ragmeter.gate.config import GateConfig, MetricRule
from ragmeter.gate.snapshot import SnapshotError, dump_snapshot, load_snapshot

METRICS = RunMetrics(
    by_question={"q1": {"recall@3": 1.0, "cost_usd": None},
                 "q2": {"recall@3": 0.5, "cost_usd": 0.001}},
    all_values={"recall@3": [1.0, 0.5], "cost_usd": [None, 0.001]},
    judge_failures=2,
    n_traces=2,
)


def test_round_trip_preserves_everything(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    loaded = load_snapshot(path)
    assert loaded.by_question == METRICS.by_question
    assert loaded.all_values == METRICS.all_values
    assert loaded.judge_failures == 2
    assert loaded.n_traces == 2


def test_none_survives_the_round_trip(tmp_path):
    # None means "not measurable". If JSON turned it into 0.0 the baseline
    # would claim a measurement that never happened.
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    loaded = load_snapshot(path)
    assert loaded.by_question["q1"]["cost_usd"] is None
    assert loaded.all_values["cost_usd"][0] is None


def test_snapshot_is_readable_and_diffable(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    text = path.read_text(encoding="utf-8")
    # Indented and key-sorted so a pull request diff is reviewable and a rerun
    # with unchanged numbers produces no diff at all.
    assert "\n  " in text
    assert text.index('"q1"') < text.index('"q2"')


def test_snapshot_records_its_provenance(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="semantic-v2", k=7)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["run"] == "semantic-v2"
    assert raw["k"] == 7


def test_a_snapshot_compares_exactly_like_a_live_run(tmp_path):
    path = tmp_path / "b.json"
    dump_snapshot(METRICS, path, run="baseline", k=3)
    config = GateConfig(metrics=(MetricRule("recall@3", max_drop=0.01),),
                        min_samples=1, fail_on_missing=False)

    live = compare(METRICS, METRICS, config)
    from_file = compare(load_snapshot(path), METRICS, config)
    assert from_file.outcomes[0].delta == live.outcomes[0].delta
    assert from_file.n_paired == live.n_paired


def test_missing_file_raises_with_the_fix_in_the_message(tmp_path):
    with pytest.raises(SnapshotError, match="ragmeter export"):
        load_snapshot(tmp_path / "nope.json")


def test_malformed_snapshot_raises(tmp_path):
    path = tmp_path / "b.json"
    path.write_text('{"not": "a snapshot"}', encoding="utf-8")
    with pytest.raises(SnapshotError, match="missing"):
        load_snapshot(path)


def test_unparseable_json_raises(tmp_path):
    path = tmp_path / "b.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(SnapshotError, match="not valid JSON"):
        load_snapshot(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_gate_snapshot.py -q`
Expected: `ModuleNotFoundError: No module named 'ragmeter.gate.snapshot'`

- [ ] **Step 3: Write the implementation**

Create `ragmeter/gate/snapshot.py`:

```python
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


def dump_snapshot(metrics: RunMetrics, path: Path, run: str, k: int) -> None:
    """Write a baseline snapshot.

    Indented and key-sorted on purpose: a baseline nobody can read in a pull
    request is a baseline nobody will ever question, and stable ordering means
    a rerun with unchanged numbers produces no diff.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run": run,
                "k": k,
                "n_traces": metrics.n_traces,
                "judge_failures": metrics.judge_failures,
                "by_question": metrics.by_question,
                "all_values": metrics.all_values,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_gate_snapshot.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add ragmeter/gate/snapshot.py tests/test_gate_snapshot.py
git commit -m "feat: baseline snapshots so the gate can run without a database"
```

---

### Task 2: `export` command and `gate --baseline-file`

**Files:**
- Modify: `ragmeter/cli.py`
- Test: `tests/test_gate_cli.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gate_cli.py`:

```python
def test_export_writes_a_snapshot(evaluated):
    out = evaluated / "baseline.json"
    result = runner.invoke(app, ["export", "--run", "baseline", "--k", "3",
                                 "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()

    from ragmeter.gate.snapshot import load_snapshot
    snapshot = load_snapshot(out)
    assert snapshot.by_question["q1"]["recall@3"] == 1.0


def test_export_unknown_run_exits_two(evaluated):
    result = runner.invoke(app, ["export", "--run", "nope", "--k", "3",
                                 "--out", str(evaluated / "x.json")])
    assert result.exit_code == 2


def test_gate_against_a_snapshot_matches_the_database(evaluated):
    out = evaluated / "baseline.json"
    runner.invoke(app, ["export", "--run", "baseline", "--k", "3", "--out", str(out)])
    config = gate_file(evaluated, "min_samples: 3\nmetrics:\n  recall@3:\n    max_drop: 0.02\n")

    from_db = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                                  "--config", config, "--k", "3"])
    from_file = runner.invoke(app, ["gate", "--run", "candidate",
                                    "--baseline-file", str(out),
                                    "--config", config, "--k", "3"])
    # A snapshot must be indistinguishable from the live run it came from.
    assert from_db.exit_code == from_file.exit_code == 1
    assert "-0.2000" in from_file.output


def test_gate_needs_exactly_one_baseline_source(evaluated):
    config = gate_file(evaluated, "metrics:\n  recall@3:\n    max_drop: 0.5\n")
    neither = runner.invoke(app, ["gate", "--run", "candidate",
                                  "--config", config, "--k", "3"])
    assert neither.exit_code == 2
    assert "--baseline" in neither.output

    both = runner.invoke(app, ["gate", "--run", "candidate", "--baseline", "baseline",
                               "--baseline-file", str(evaluated / "b.json"),
                               "--config", config, "--k", "3"])
    assert both.exit_code == 2


def test_missing_snapshot_exits_two_and_says_how_to_make_one(evaluated):
    config = gate_file(evaluated, "metrics:\n  recall@3:\n    max_drop: 0.5\n")
    result = runner.invoke(app, ["gate", "--run", "candidate",
                                 "--baseline-file", str(evaluated / "absent.json"),
                                 "--config", config, "--k", "3"])
    # Exit 2, not 1: a missing baseline is a data problem, not a regression.
    # The first CI run must not look like the model got worse.
    assert result.exit_code == 2
    assert "ragmeter export" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_gate_cli.py -q`
Expected: failures with `No such command 'export'`

- [ ] **Step 3: Update `ragmeter/cli.py`**

Add to the imports:

```python
from ragmeter.gate.snapshot import SnapshotError, dump_snapshot, load_snapshot
```

Add the `export` command before the `if __name__` block:

```python
@app.command("export")
def export(
    run: str = typer.Option(..., "--run"),
    k: int = typer.Option(5, "--k", min=1),
    out: Path = typer.Option(..., "--out", help="Where to write the snapshot."),
) -> None:
    """Write a run's per-question metrics to a committable baseline file."""
    session = _session()
    try:
        metrics = collect_run_metrics(session, run, k=k)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()

    dump_snapshot(metrics, out, run=run, k=k)
    typer.echo(f"wrote {out} ({len(metrics.by_question)} questions, k={k})")
```

Replace the `gate` command with:

```python
@app.command("gate")
def gate(
    run: str = typer.Option(..., "--run", help="The candidate run."),
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    baseline: str | None = typer.Option(None, "--baseline",
                                        help="A run in the database."),
    baseline_file: Path | None = typer.Option(None, "--baseline-file",
                                              help="A snapshot from `ragmeter export`."),
    k: int = typer.Option(5, "--k", min=1),
) -> None:
    """Fail when a run is worse than its baseline. Exit 1 = regression, 2 = error."""
    if (baseline is None) == (baseline_file is None):
        typer.echo("error: pass exactly one of --baseline or --baseline-file", err=True)
        raise typer.Exit(2)

    try:
        gate_config = load_gate_config(config)
    except GateConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)

    session = _session()
    try:
        if baseline_file is not None:
            base = load_snapshot(baseline_file)
            baseline_label = str(baseline_file)
        else:
            base = collect_run_metrics(session, baseline, k=k)
            baseline_label = baseline
        cand = collect_run_metrics(session, run, k=k)
    except (ValueError, SnapshotError) as exc:
        # Exit 2, not 1. Missing or unreadable data is a tooling problem, and a
        # first run with no baseline must not look like a regression.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()

    result = compare(base, cand, gate_config)
    typer.echo(render_gate(result, run, baseline_label, k))
    # 1 is reserved for "the run got worse".
    raise typer.Exit(0 if result.passed else 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_gate_cli.py -q`
Expected: `13 passed`

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `310 passed, 1 skipped`

- [ ] **Step 6: Commit**

```bash
git add ragmeter/cli.py tests/test_gate_cli.py
git commit -m "feat: export baselines and gate against a snapshot file"
```

---

### Task 3: The CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/gate.sh`

- [ ] **Step 1: Write the one-command gate script**

Create `scripts/gate.sh`:

```bash
#!/usr/bin/env bash
# Regenerate the example RAG, evaluate it, and gate against the committed
# baseline. Exit 1 means retrieval got worse; exit 2 means the check could not
# run at all.
set -euo pipefail

STRATEGY="${STRATEGY:-paragraph}"
K="${K:-5}"
ARTICLES="${ARTICLES:-5}"
LIMIT="${LIMIT:-200}"
BASELINE="baselines/${STRATEGY}.json"

export RAGMETER_DB_URL="sqlite:///ci-gate.db"
rm -f ci-gate.db

python -m example_rag.cli --articles "$ARTICLES" --limit "$LIMIT" --k "$K" \
    --strategies "$STRATEGY"

ragmeter dataset load "data/runs/${STRATEGY}.golden.yaml" --name "$STRATEGY" --version v1
ragmeter ingest "data/runs/${STRATEGY}.traces.jsonl" --run candidate
ragmeter eval --run candidate --dataset "$STRATEGY" --version v1 --k "$K"

ragmeter gate --run candidate --baseline-file "$BASELINE" --config gate.yaml --k "$K"
```

- [ ] **Step 2: Create the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev,api]"
      - run: pytest -q

  gate:
    # Separate job so a retrieval regression is distinguishable at a glance
    # from a broken test, and so it still runs when tests are unrelated.
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e .

      - name: Fetch the corpus
        run: |
          mkdir -p data
          curl -sSL -o data/squad-dev-v1.1.json \
            https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

      - name: Regression gate
        run: bash scripts/gate.sh
```

- [ ] **Step 3: Verify the script locally**

```bash
bash scripts/gate.sh; echo "exit: $?"
```

Expected on a first run: exit 2, with the message naming `ragmeter export`.
That is the correct first-run behaviour and proves a missing baseline is not
reported as a regression.

- [ ] **Step 4: Create the baseline and rerun**

```bash
export RAGMETER_DB_URL="sqlite:///ci-gate.db"
ragmeter export --run candidate --k 5 --out baselines/paragraph.json
bash scripts/gate.sh; echo "exit: $?"
```

Expected: exit 0, `gate: PASS`, all deltas zero. Nothing changed, so nothing regressed.

- [ ] **Step 5: Prove the gate actually catches a regression**

Point the script at a strategy known to be worse and confirm it fails against
the `paragraph` baseline:

```bash
STRATEGY=fixed-100 bash scripts/gate.sh; echo "exit: $?"
```

This will exit 2, because `baselines/fixed-100.json` does not exist. To test a
true regression, gate the worse run against the good baseline directly:

```bash
export RAGMETER_DB_URL="sqlite:///ci-gate.db"
python -m example_rag.cli --articles 5 --limit 200 --k 5 --strategies fixed-100
ragmeter dataset load data/runs/fixed-100.golden.yaml --name fixed-100 --version v1
ragmeter ingest data/runs/fixed-100.traces.jsonl --run worse
ragmeter eval --run worse --dataset fixed-100 --version v1 --k 5
ragmeter gate --run worse --baseline-file baselines/paragraph.json --config gate.yaml --k 5
echo "exit: $?"
```

Expected: exit **1**, with `recall@5` dropping around 0.41 and a large
`-regressed` count. This is the check that the gate does its job.

- [ ] **Step 6: Clean up and commit**

```bash
rm -f ci-gate.db
git add .github scripts baselines
git commit -m "ci: gate the example RAG against a committed baseline"
```

---

### Task 4: Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a CI section before `## Trace format`**

````markdown
## Running the gate in CI

```bash
ragmeter export --run known-good --k 5 --out baselines/paragraph.json   # once
ragmeter gate --run candidate --baseline-file baselines/paragraph.json --config gate.yaml --k 5
```

The baseline is a **committed JSON snapshot** of per-question metrics, not a
database. CI keeps no state between runs, and the gate compares per question,
so an aggregate summary would not be enough to pair against.

The snapshot is indented and key-sorted, so it diffs readably in a pull request
and an unchanged rerun produces no diff at all.

Exit codes matter here: a missing or unreadable baseline is **2**, not 1, and
the error names the `ragmeter export` command that creates one. A first run
with no baseline must not look like the model got worse.

See `.github/workflows/ci.yml` and `scripts/gate.sh`.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: running the gate in CI"
```

---

## Definition of Done

- [ ] `.venv/Scripts/python -m pytest -q` reports 310 passed, 1 skipped
- [ ] A snapshot compares identically to the live run it was exported from
- [ ] `None` survives the round trip and never becomes `0.0`
- [ ] A missing baseline exits **2** and names the command that fixes it
- [ ] Gating a known-worse strategy against the good baseline exits **1**
- [ ] The workflow file exists and the gate script runs green locally

## Out of Scope

Pushing to GitHub, which sends this code to an external service and needs
explicit permission first. The workflow file is written and the script verified
locally either way.
