#!/usr/bin/env bash
# Regenerate the example RAG, evaluate it, and gate against the committed
# baseline. Exit 1 means retrieval got worse; exit 2 means the check could not
# run at all.
set -euo pipefail

STRATEGY="${STRATEGY:-fixed-100}"
K="${K:-5}"
ARTICLES="${ARTICLES:-5}"
LIMIT="${LIMIT:-200}"
BASELINE="baselines/paragraph.json"
# Must match scripts/gate.yaml. Timing metrics are excluded on purpose:
# BM25 latency rounds to 0 or 1 ms, so it is noise, not signal.
GATED_METRICS="${GATED_METRICS:-recall@5,ndcg@5,mrr@5}"

export RAGMETER_DB_URL="sqlite:///ci-gate.db"
rm -f ci-gate.db

# `python -m` rather than the console script, so this runs identically in CI
# and from a shell where the venv is not on PATH.
PY_BIN="${PY_BIN:-python}"
RAGMETER="$PY_BIN -m ragmeter.cli"

$PY_BIN -m example_rag.cli --articles "$ARTICLES" --limit "$LIMIT" --k "$K" \
    --strategies "$STRATEGY"

$RAGMETER dataset load "data/runs/${STRATEGY}.golden.yaml" --name "$STRATEGY" --version v1
$RAGMETER ingest "data/runs/${STRATEGY}.traces.jsonl" --run candidate
$RAGMETER eval --run candidate --dataset "$STRATEGY" --version v1 --k "$K"

$RAGMETER gate --run candidate --baseline-file "$BASELINE" --config "${GATE_CONFIG:-scripts/gate.yaml}" --k "$K"
