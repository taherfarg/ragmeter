"""Command line interface.

Exit codes: 0 success, 1 reserved for the Phase 3 regression gate, 2 usage or
data error. The gate needs 1 to itself so CI can tell "worse" from "broken".
"""

from pathlib import Path

import typer

from ragmeter.db import init_db, make_engine, make_session
from ragmeter.loaders import get_or_create_run, load_golden, load_traces
from ragmeter.metrics.cost import fetch_prices
from ragmeter.report import render_summary, summarize_run
from ragmeter.runner import evaluate_run

app = typer.Typer(no_args_is_help=True, help="Measure any RAG system.")
dataset_app = typer.Typer(no_args_is_help=True, help="Manage golden datasets.")
app.add_typer(dataset_app, name="dataset")


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
) -> None:
    """Compute retrieval, cost, and latency metrics for a run."""
    try:
        prices = fetch_prices()
    except Exception as exc:
        # Pricing is an enrichment, not a prerequisite. Say so loudly and continue.
        typer.echo(f"warning: could not fetch model prices ({exc}); cost will be blank", err=True)
        prices = {}

    session = _session()
    try:
        result = evaluate_run(session, run, dataset, version, k=k, prices=prices)
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


if __name__ == "__main__":
    app()
