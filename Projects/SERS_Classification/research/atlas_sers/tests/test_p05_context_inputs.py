from __future__ import annotations

import csv
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation.p05_outer_inputs import P05OuterInputsError, load_context_rows

RUN_ID = "p04plan-run"
FIELDS = (
    "context_id",
    "experiment_id",
    "phase_gate",
    "outer_repeat",
    "outer_fold",
    "station",
    "domain",
    "held_instrument",
    "partition_id",
)


def _row(
    context_id,
    *,
    phase_gate="development",
    outer_repeat="1",
    outer_fold="0",
    held_instrument="",
):
    return {
        "context_id": context_id,
        "experiment_id": "exp",
        "phase_gate": phase_gate,
        "outer_repeat": outer_repeat,
        "outer_fold": outer_fold,
        "station": "S",
        "domain": "D",
        "held_instrument": held_instrument,
        "partition_id": "P",
    }


def _registry(tmp_path):
    directory = Path(tmp_path) / "p04plan" / "runs" / RUN_ID
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "context_registry.csv"


def _write(tmp_path, rows, header=FIELDS):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    raw = buffer.getvalue().encode("utf-8")
    _registry(tmp_path).write_bytes(raw)
    return raw


def _bundle(tmp_path, rows=None, *, support=None, raw=None, header=FIELDS, digest=None):
    if raw is None:
        raw = _write(tmp_path, rows or [], header)
    if support is None:
        support = SimpleNamespace(contexts=tuple(dict(r) for r in rows or []))
    pin = core._canon().sha256_bytes(raw) if digest is None else digest
    contract = {"input_pins": {"p04plan_run_id": RUN_ID, "contexts_sha256": pin}}
    return {"artifact_root": Path(tmp_path), "contract": contract, "support": support}


def _matches(code):
    return pytest.raises(P05OuterInputsError, match=code)


def test_success_blank_held_instrument_preserved(tmp_path):
    rows = [_row("c1"), _row("c2", held_instrument="")]
    bundle = _bundle(tmp_path, rows)
    before = _registry(tmp_path).read_bytes()
    loaded = load_context_rows(bundle)
    assert [r["context_id"] for r in loaded] == ["c1", "c2"]
    assert loaded[1]["held_instrument"] == ""
    assert _registry(tmp_path).read_bytes() == before


def test_hash_mismatch(tmp_path):
    bundle = _bundle(tmp_path, [_row("c1")], digest="0" * 64)
    with _matches("context_registry_digest_mismatch"):
        load_context_rows(bundle)


def test_duplicate_context(tmp_path):
    bundle = _bundle(tmp_path, [_row("c1"), _row("c1")])
    with _matches("context_registry_context_duplicate"):
        load_context_rows(bundle)


def test_missing_support_context(tmp_path):
    support = SimpleNamespace(contexts=({"context_id": "other"},))
    bundle = _bundle(tmp_path, [_row("c1")], support=support)
    with _matches("context_registry_context_mismatch"):
        load_context_rows(bundle)


def test_unknown_phase(tmp_path):
    bundle = _bundle(tmp_path, [_row("c1", phase_gate="banana")])
    with _matches("context_phase_gate_unknown"):
        load_context_rows(bundle)


def test_noncanonical_repeat(tmp_path):
    bundle = _bundle(tmp_path, [_row("c1", outer_repeat="01")])
    with _matches("context_outer_repeat_malformed"):
        load_context_rows(bundle)


def test_noncanonical_fold(tmp_path):
    bundle = _bundle(tmp_path, [_row("c1", outer_fold="1.0")])
    with _matches("context_outer_fold_malformed"):
        load_context_rows(bundle)


def test_missing_csv_cell_none(tmp_path):
    raw = (",".join(FIELDS) + "\nc1,exp,development,0,0,S,D\n").encode("utf-8")
    _registry(tmp_path).write_bytes(raw)
    bundle = _bundle(tmp_path, raw=raw, support=SimpleNamespace(contexts=()))
    with _matches("context_registry_row_malformed"):
        load_context_rows(bundle)


def test_duplicate_header(tmp_path):
    raw = _write(tmp_path, [], header=("context_id", "context_id"))
    bundle = _bundle(tmp_path, raw=raw, support=SimpleNamespace(contexts=()))
    with _matches("context_registry_header_invalid"):
        load_context_rows(bundle)
