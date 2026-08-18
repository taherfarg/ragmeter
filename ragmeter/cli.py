"""Command line interface.

Exit codes: 0 success, 1 reserved for the Phase 3 regression gate, 2 usage or
data error. The gate needs 1 to itself so CI can tell "worse" from "broken".
"""

from pathlib import Path

import typer

from ragmeter.calibration import calibrate, collect_labeled_pairs, unlabeled_traces
from ragmeter.db import HumanLabel, init_db, make_engine, make_session
from ragmeter.gate.collect import collect_run_metrics
from ragmeter.gate.compare import compare
from ragmeter.gate.config import GateConfig, GateConfigError, MetricRule, load_gate_config
from ragmeter.gate.render import render_gate
from ragmeter.judge.client import DbJudgeCache, JudgeClient, JudgeError
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.metrics.cost import fetch_prices
from ragmeter.metrics.retrieval import metric_names
from ragmeter.report import render_summary, summarize_run
from ragmeter.runner import evaluate_run

app = typer.Typer(no_args_is_help=True, help="Measure any RAG system.")
dataset_app = typer.Typer(no_args_is_help=True, help="Manage golden datasets.")
app.add_typer(dataset_app, name="dataset")

# What to actually ask a person, per metric. Phrasing matters: the question must
# be answerable from the screen alone, without the judge's opinion.
LABEL_QUESTIONS = {
    "faithfulness": "Is every claim in the answer supported by the sources above?",
    "answer_relevance": "Does the answer address the question?",
}

YES = {"y", "yes"}
NO = {"n", "no"}
QUIT = {"q", "quit"}


def _session():
    engine = make_engine()
    init_db(engine)
    return make_session(engine)()


@dataset_app.command("load")
def dataset_load(
    path: Path = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Option(..., "--name", help="Dataset name."),
    version: str = typer.Option("v1", "--version", help="Dataset version."),
) -> None:
    """Load a golden YAML file."""
    session = _session()
    try:
        count = load_golden(session, path, dataset=name, version=version)
        session.commit()
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()
    typer.echo(f"loaded {count} golden items into {name}@{version}")


@app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True),
    run: str = typer.Option(..., "--run", help="Run name. Created if absent."),
    git_sha: str | None = typer.Option(None, "--git-sha"),
) -> None:
    """Ingest a JSONL trace file into a run."""
    session = _session()
    try:
        run_row = get_or_create_run(session, run, git_sha=git_sha)
        result = load_traces(session, path, run_row)
        session.commit()
    except ValueError as exc:
        session.rollback()
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()
    typer.echo(f"ingested {result['ingested']}, skipped {result['skipped']} into run {run!r}")


@app.command("eval")
def evaluate(
    run: str = typer.Option(..., "--run"),
    dataset: str = typer.Option(..., "--dataset"),
    version: str = typer.Option("v1", "--version"),
    k: int = typer.Option(5, "--k", min=1),
    judge: bool = typer.Option(False, "--judge/--no-judge",
                               help="Score faithfulness and answer relevance via OpenRouter."),
    judge_model: str | None = typer.Option(None, "--judge-model"),
) -> None:
    """Compute retrieval, cost, and latency metrics for a run."""
    try:
        prices = fetch_prices()
    except Exception as exc:
        # Pricing is an enrichment, not a prerequisite. Say so loudly and continue.
        typer.echo(f"warning: could not fetch model prices ({exc}); cost will be blank", err=True)
        prices = {}

    session = _session()
    judge_client = None
    if judge:
        try:
            judge_client = JudgeClient(model=judge_model, cache=DbJudgeCache(session))
        except JudgeError as exc:
            # Fail before evaluating rather than after, so the user is not left
            # wondering why every judge column is blank.
            session.close()
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2)

    try:
        result = evaluate_run(session, run, dataset, version, k=k, prices=prices,
                              judge=judge_client)
        summary = summarize_run(session, run, k=k)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    finally:
        session.close()

    typer.echo(render_summary(summary, run, k))
    typer.echo("")
    typer.echo(
        f"{result['n_traces']} traces, {result['n_matched']} matched to golden, "
        f"{result['n_unmatched']} unmatched"
    )
    if result["n_judge_failures"]:
        typer.echo(
            f"WARNING: the judge failed on {result['n_judge_failures']} trace(s); "
            f"those metrics are blank, not zero",
            err=True,
        )


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


if __name__ == "__main__":
    app()
