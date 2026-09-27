"""Synthetic tests for the read-only P05 publication gate.

The real comprehensive-reporting producer persists artificial aggregate tables,
receipts, manifests and figure files using the existing reporting-test harness.
Upstream authority, input/provenance checks, aggregate builders and the PDF/PNG
compiler are fixtures. The gate's upstream ``verify_reporting_sources`` call is
also replaced and its arguments captured; the reporting-inputs suite separately
tests that helper. The publication gate checks the actual stored output bytes.
"""

from __future__ import annotations

# ruff: noqa: E402
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_comprehensive_reporting as reporting
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_reporting_inputs
from atlas_sers.governance import p05_publication_gate as gate
from atlas_sers.visualization import p05_benchmark_figures as benchmark_figures
from tests import test_p05_comprehensive_reporting as reporting_fixture

BINDING_KEYS = p05_reporting_inputs.BINDING_KEYS
TOTAL_TABLES = reporting.TOTAL_TABLE_COUNT
SOURCE_FITS = 14904
REUSED_FITS = 36
EVIDENCE_FITS = 14940
ALIASES = 2880
UNIQUE_REFITS = 3
SOURCE_UPDATES = 100
REFIT_UPDATES = 7
PLAN_ID = "d" * 64


def _source_costs(prior: float) -> dict[str, object]:
    return {
        "new_source_fits": SOURCE_FITS,
        "reused_pilot_fits": REUSED_FITS,
        "source_evidence_fits": EVIDENCE_FITS,
        "unique_refitted_models": UNIQUE_REFITS,
        "unique_scalar_calibrations": UNIQUE_REFITS,
        "new_neural_fits_total": SOURCE_FITS + UNIQUE_REFITS,
        "strategy_alias_count": ALIASES,
        "new_source_optimizer_updates": SOURCE_UPDATES,
        "refit_optimizer_updates": REFIT_UPDATES,
        "combined_new_optimizer_updates": SOURCE_UPDATES + REFIT_UPDATES,
        "source_scientific_seconds": 10.0,
        "refit_scientific_seconds": 2.0,
        "scientific_seconds_cumulative_bound_through_comparison": float(prior),
    }


def _paths(artifact_root: Path) -> SimpleNamespace:
    run_root = (
        artifact_root
        / reporting.COMPREHENSIVE_DIR
        / reporting.RUNS_DIR
        / inputs.COMPREHENSIVE_PERMIT_SHA256
    )
    stage = run_root / reporting.STAGE_NAME
    return SimpleNamespace(
        artifact_root=artifact_root,
        run_root=run_root,
        stage=stage,
        public_root=stage / reporting.PUBLIC_ROOT_NAME,
        receipt=json.loads((run_root / reporting.RECEIPT_NAME).read_text(encoding="utf-8")),
        receipt_sha256=core._canon().sha256_file(run_root / reporting.RECEIPT_NAME),
        deadline=time.perf_counter() + 3600.0,
        calls=[],
        ctx=None,
    )


def _sealed(monkeypatch: pytest.MonkeyPatch, base: Path) -> SimpleNamespace:
    ctx = reporting_fixture._setup(monkeypatch, base)
    run_root = ctx["run_root"]
    comparison_receipt_path = run_root / reporting.COMPARISON_RECEIPT_NAME
    comparison_manifest_path = (
        run_root / reporting.COMPARISON_STAGE_NAME / reporting.COMPARISON_MANIFEST_NAME
    )
    bindings = {key: "a" * 64 for key in BINDING_KEYS}
    bindings["comparison_receipt_sha256"] = core._canon().sha256_file(comparison_receipt_path)
    bindings["comparison_manifest_sha256"] = core._canon().sha256_file(comparison_manifest_path)
    monkeypatch.setattr(reporting_fixture, "BINDINGS", bindings)
    monkeypatch.setattr(reporting_fixture, "STAGE_COSTS", _source_costs(ctx["prior"]))
    monkeypatch.setattr(inputs, "COMPREHENSIVE_PERMIT_SHA256", ctx["bundle"]["permit_sha256"])
    monkeypatch.setattr(inputs, "CORE_CONTRACT_SHA256", ctx["bundle"]["contract_sha256"])
    monkeypatch.setattr(inputs, "CORE_PLAN_ID", ctx["bundle"]["core_plan_id"])
    monkeypatch.setattr(inputs, "LEDGER_ID", ctx["bundle"]["ledger"]["ledger_id"])
    comparison_receipt = json.loads(comparison_receipt_path.read_text(encoding="utf-8"))

    def _authenticate(bundle, *, deadline):
        return {
            "prior_seconds": ctx["prior"],
            "comparison_receipt": dict(comparison_receipt),
            "plan": {"plan_id": PLAN_ID},
            "comparison_tables": {},
            "aggregation_tables": {"ensemble_predictions": pd.DataFrame({"value": [1.0]})},
            "source_optimizer_steps": SOURCE_UPDATES,
            "refit_optimizer_steps": REFIT_UPDATES,
        }

    monkeypatch.setattr(ctx["authority"], "authenticate_comparison", _authenticate)
    calls: list[dict[str, str]] = []

    def _verify(bundle, *, bindings, deadline):
        freeze._check_deadline(deadline)
        calls.append(dict(bindings))
        return {"verified": True}

    monkeypatch.setattr(gate.reporting_inputs, "verify_reporting_sources", _verify)
    reporting_fixture.mod.run_reporting(**ctx["kwargs"])
    paths = _paths(ctx["artifact_root"])
    paths.calls = calls
    paths.ctx = ctx
    return paths


@pytest.fixture(scope="module")
def sealed_base(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("publication_gate")
    try:
        yield _sealed(mp, base)
    finally:
        mp.undo()


@pytest.fixture
def sealed(tmp_path, sealed_base):
    dest = tmp_path / "artifact"
    shutil.copytree(sealed_base.artifact_root, dest)
    paths = _paths(dest)
    paths.calls = sealed_base.calls
    paths.calls.clear()
    paths.ctx = sealed_base.ctx
    return paths


def _load(sealed: SimpleNamespace, *, deadline: object = None):
    return gate.load_public_bundle(
        sealed.artifact_root,
        expected_reporting_receipt_sha256=sealed.receipt_sha256,
        deadline=sealed.deadline if deadline is None else deadline,
    )


def _receipt_path(sealed: SimpleNamespace) -> Path:
    return sealed.run_root / reporting.RECEIPT_NAME


def _reseal_manifest_and_receipt(sealed: SimpleNamespace) -> None:
    core._write_manifest(sealed.stage)
    path = _receipt_path(sealed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stage_manifest_sha256"] = core._canon().sha256_file(
        sealed.stage / reporting.MANIFEST_NAME
    )
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    sealed.receipt = payload
    sealed.receipt_sha256 = core._canon().sha256_file(path)


def _mutate_receipt(sealed: SimpleNamespace, mutator) -> None:
    path = _receipt_path(sealed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    sealed.receipt = payload
    sealed.receipt_sha256 = core._canon().sha256_file(path)


def _mutate_summary(sealed: SimpleNamespace, mutator) -> None:
    path = sealed.stage / reporting.SUMMARY_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    _reseal_manifest_and_receipt(sealed)


def _mutate_costs(sealed: SimpleNamespace, mutator) -> None:
    path = sealed.public_root / reporting.TABLES_DIR_NAME / reporting.COSTS_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    _reseal_manifest_and_receipt(sealed)


def _set(key: str, value):
    def _apply(payload):
        payload[key] = value

    return _apply


def _set_counter(key: str, value):
    def _apply(payload):
        payload["counters"][key] = value

    return _apply


def _snapshot(root: Path) -> dict[str, tuple[str, object]]:
    entries: dict[str, tuple[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries[relative] = ("link", os.readlink(path))
        elif path.is_dir():
            entries[relative] = ("dir", None)
        else:
            entries[relative] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
    return entries


def test_success_bundle_and_file_hashes(sealed):
    bundle = gate.verify_public_bundle(
        sealed.artifact_root,
        expected_reporting_receipt_sha256=sealed.receipt_sha256,
        deadline=sealed.deadline,
    )
    assert isinstance(bundle, gate.PublicBundle)
    assert bundle.public_root == sealed.public_root
    assert bundle.receipt_sha256 == sealed.receipt_sha256
    assert len(bundle.tables) == TOTAL_TABLES
    assert set(bundle.figure_manifests) == {
        reporting.PAIRED_UNIT_NAME,
        reporting.DIAGNOSTIC_UNIT_NAME,
    }
    assert bundle.cumulative_seconds > 0.0
    table_files = {
        name
        for name in bundle.files
        if name.startswith(f"{reporting.TABLES_DIR_NAME}/") and name.endswith(".csv")
    }
    assert len(table_files) == TOTAL_TABLES
    assert f"{reporting.TABLES_DIR_NAME}/{reporting.COSTS_NAME}" in bundle.files
    assert len(sealed.calls) == 2
    for name, digest in bundle.files.items():
        assert core._canon().sha256_file(bundle.public_root / name) == digest


def test_receipt_source_and_refit_counts(sealed):
    bundle = _load(sealed)
    assert sealed.receipt["source_optimizer_steps"] == SOURCE_UPDATES
    assert sealed.receipt["refit_optimizer_steps"] == REFIT_UPDATES
    assert sealed.receipt["selection_plan_id"] == PLAN_ID
    assert bundle.cumulative_seconds == pytest.approx(
        sealed.receipt["scientific_seconds_cumulative_bound"]
    )


def test_costs_identity_and_bounds(sealed):
    bundle = _load(sealed)
    costs = bundle.costs
    assert costs["new_source_fits"] == SOURCE_FITS
    assert costs["reused_pilot_fits"] == REUSED_FITS
    assert costs["source_evidence_fits"] == EVIDENCE_FITS
    assert costs["strategy_alias_count"] == ALIASES
    assert costs["unique_refitted_models"] == UNIQUE_REFITS
    assert costs["new_neural_fits_total"] == SOURCE_FITS + UNIQUE_REFITS
    assert costs["combined_new_optimizer_updates"] == SOURCE_UPDATES + REFIT_UPDATES
    assert set(costs) <= set(p05_reporting_inputs.PUBLIC_COST_KEYS)
    assert set(p05_reporting_inputs.REQUIRED_PUBLIC_COST_KEYS) <= set(costs)


def test_read_only_inventory_and_bytes(sealed):
    before = _snapshot(sealed.artifact_root)
    _load(sealed)
    assert _snapshot(sealed.artifact_root) == before


def test_read_tables_preserves_lexical_values(tmp_path):
    names = tuple(f"table_{index}" for index in range(TOTAL_TABLES))
    tables = tmp_path / reporting.TABLES_DIR_NAME
    tables.mkdir(parents=True)
    payload = "code,value\n00123,1.0\nNA,2.0\n,3.0\n"
    for name in names:
        (tables / f"{name}.csv").write_text(payload, encoding="utf-8")
    frames = gate._read_tables(tmp_path, names, time.perf_counter() + 3600.0)
    assert list(frames) == list(names)
    assert frames[names[0]]["code"].tolist() == ["00123", "NA", ""]


def test_wrong_expected_digest(sealed):
    with pytest.raises(gate.P05PublicationGateError) as exc:
        gate.load_public_bundle(
            sealed.artifact_root,
            expected_reporting_receipt_sha256="e" * 64,
            deadline=sealed.deadline,
        )
    assert exc.value.reason_code == "reporting_receipt_digest_mismatch"


def test_malformed_expected_digest(sealed):
    with pytest.raises(gate.P05PublicationGateError) as exc:
        gate.load_public_bundle(
            sealed.artifact_root,
            expected_reporting_receipt_sha256="not-a-digest",
            deadline=sealed.deadline,
        )
    assert exc.value.reason_code == "expected_receipt_sha_malformed"


def test_missing_receipt(sealed):
    (sealed.run_root / reporting.RECEIPT_NAME).unlink()
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "reporting_receipt_missing"


def test_malformed_deadline(sealed):
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed, deadline="soon")
    assert exc.value.reason_code == "deadline_malformed"


def test_expired_deadline(sealed):
    with pytest.raises(core.P05CoreError):
        _load(sealed, deadline=0.0)


IDENTITY_CASES = [
    ("permit", _set("permit_sha256", "e" * 64), "receipt_permit_sha256_mismatch"),
    ("stage", _set("stage", "other"), "receipt_stage_mismatch"),
    ("status", _set("status", "running"), "receipt_status_incomplete"),
    ("command", _set("command", "other"), "receipt_command_mismatch"),
    ("schema", _set("schema_version", "other"), "receipt_schema_mismatch"),
    ("protocol", _set("protocol_version", "other"), "receipt_protocol_mismatch"),
    ("core_plan", _set("core_plan_id", "other"), "receipt_core_plan_id_mismatch"),
    (
        "core_contract",
        _set("core_contract_sha256", "e" * 64),
        "receipt_core_contract_sha256_mismatch",
    ),
    ("ledger", _set("ledger_id", "other"), "receipt_ledger_id_mismatch"),
    ("reporting_complete", _set("reporting_complete", False), "receipt_reporting_incomplete"),
]


@pytest.mark.parametrize(
    "name,mutator,code", IDENTITY_CASES, ids=[case[0] for case in IDENTITY_CASES]
)
def test_receipt_identity_mismatch(sealed, name, mutator, code):
    _mutate_receipt(sealed, mutator)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == code


COUNTER_CASES = [
    ("bool", _set_counter("public_tables", True), "counter_public_tables_invalid"),
    ("float", _set_counter("public_tables", 7.0), "counter_public_tables_invalid"),
    ("updates", _set_counter("updates", 1), "counter_updates_nonzero"),
    ("paired", _set_counter("paired_figures", 0), "counter_paired_figures_missing"),
]


@pytest.mark.parametrize(
    "name,mutator,code", COUNTER_CASES, ids=[case[0] for case in COUNTER_CASES]
)
def test_counter_mismatch(sealed, name, mutator, code):
    _mutate_summary(sealed, mutator)
    _mutate_receipt(sealed, mutator)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == code


ELAPSED_CASES = [
    ("large", _set("scientific_seconds_this_stage", 1.0e9), "summary_elapsed_out_of_range"),
    ("negative", _set("scientific_seconds_this_stage", -1.0), "summary_elapsed_out_of_range"),
    (
        "malformed",
        _set("scientific_seconds_this_stage", "soon"),
        "counter_elapsed_seconds_malformed",
    ),
]


@pytest.mark.parametrize(
    "name,mutator,code", ELAPSED_CASES, ids=[case[0] for case in ELAPSED_CASES]
)
def test_summary_elapsed(sealed, name, mutator, code):
    def _change(payload):
        mutator(payload)
        payload["counters"]["elapsed_seconds"] = payload["scientific_seconds_this_stage"]

    _mutate_summary(sealed, _change)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == code


def test_receipt_prior_below_reserve(sealed):
    mutator = _set("prior_scientific_seconds_cumulative_bound", 0.0)
    _mutate_summary(sealed, mutator)
    _mutate_receipt(sealed, mutator)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "receipt_prior_out_of_range"


def test_receipt_limit_mismatch(sealed):
    mutator = _set("maximum_total_seconds", 1.0)
    _mutate_summary(sealed, mutator)
    _mutate_receipt(sealed, mutator)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "receipt_limit_mismatch"


def _negate_cost(payload):
    payload["new_source_fits"] = -1


def _bool_cost(payload):
    payload["new_source_fits"] = True


def _drop_cost(payload):
    del payload["new_source_fits"]


def _extra_cost(payload):
    payload["delta"] = 1


COST_CASES = [
    ("negative", _negate_cost, "public_cost_value_negative"),
    ("bool", _bool_cost, "public_cost_value_malformed"),
    ("missing", _drop_cost, "public_cost_key_missing"),
    ("unknown", _extra_cost, "public_cost_key_not_allowed"),
]


@pytest.mark.parametrize("name,mutator,code", COST_CASES, ids=[case[0] for case in COST_CASES])
def test_costs_violation(sealed, name, mutator, code):
    _mutate_costs(sealed, mutator)
    with pytest.raises(reporting.P05ComprehensiveReportingError) as exc:
        _load(sealed)
    assert exc.value.reason_code == code


def test_missing_public_table(sealed):
    table = sealed.public_root / reporting.TABLES_DIR_NAME / "strategy_contexts.csv"
    table.unlink()
    _reseal_manifest_and_receipt(sealed)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "public_table_missing"


def test_extra_public_file(sealed):
    (sealed.public_root / "secret.txt").write_text("x", encoding="utf-8")
    _reseal_manifest_and_receipt(sealed)
    with pytest.raises(reporting.P05ComprehensiveReportingError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "public_file_inventory_mismatch"


def test_symlinked_table_rejected(tmp_path):
    names = tuple(f"table_{index}" for index in range(TOTAL_TABLES))
    tables = tmp_path / reporting.TABLES_DIR_NAME
    tables.mkdir(parents=True)
    for name in names:
        (tables / f"{name}.csv").write_text("a\n1\n", encoding="utf-8")
    target = tmp_path / "real.csv"
    target.write_text("a\n1\n", encoding="utf-8")
    link = tables / f"{names[0]}.csv"
    link.unlink()
    link.symlink_to(target)
    with pytest.raises(core.P05CoreError):
        gate._read_tables(tmp_path, names, time.perf_counter() + 3600.0)


def test_mutated_table_rejected(sealed):
    table = sealed.public_root / reporting.TABLES_DIR_NAME / "strategy_contexts.csv"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises((core.P05CoreError, gate.P05PublicationGateError)):
        _load(sealed)


def test_mutated_figure_rejected(sealed):
    render = (
        sealed.public_root
        / reporting.FIGURES_DIR_NAME
        / reporting.PAIRED_UNIT_NAME
        / reporting.RENDER_DIR_NAME
    )
    manifest = json.loads((render / benchmark_figures.MANIFEST_NAME).read_text(encoding="utf-8"))
    name = manifest["files"][0]["path"]
    (render / name).write_bytes(b"tampered")
    with pytest.raises((core.P05CoreError, gate.P05PublicationGateError)):
        _load(sealed)


def test_mutated_bindings_rejected(sealed):
    path = sealed.stage / reporting.BINDINGS_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selector_sha256"] = "e" * 64
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    _reseal_manifest_and_receipt(sealed)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "reporting_binding_digest_mismatch"


def test_mutated_comparison_binding_rejected(sealed):
    path = sealed.stage / reporting.COMPARISON_BINDING_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comparison_receipt_sha256"] = "e" * 64
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    _reseal_manifest_and_receipt(sealed)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "comparison_binding_mismatch"


def test_final_receipt_mutation_during_scan(sealed, monkeypatch):
    real = reporting._verify_public_inventory

    def _hook(public_root, frames, figures):
        real(public_root, frames, figures)
        path = _receipt_path(sealed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "tampered"
        core._atomic_write(path, core._canon().canonical_json_bytes(payload))

    monkeypatch.setattr(reporting, "_verify_public_inventory", _hook)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "reporting_receipt_changed"


def test_final_manifest_mutation_during_scan(sealed, monkeypatch):
    real = reporting._verify_public_inventory

    def _hook(public_root, frames, figures):
        real(public_root, frames, figures)
        path = sealed.stage / reporting.MANIFEST_NAME
        core._atomic_write(path, path.read_bytes() + b"\n")

    monkeypatch.setattr(reporting, "_verify_public_inventory", _hook)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "reporting_manifest_changed"


def test_source_verify_captures_bindings(sealed):
    _load(sealed)
    assert len(sealed.calls) == 2
    captured = sealed.calls[-1]
    assert set(captured) == set(BINDING_KEYS)
    assert captured["comparison_receipt_sha256"] == core._canon().sha256_file(
        sealed.run_root / reporting.COMPARISON_RECEIPT_NAME
    )
    assert captured["comparison_manifest_sha256"] == core._canon().sha256_file(
        sealed.run_root / reporting.COMPARISON_STAGE_NAME / reporting.COMPARISON_MANIFEST_NAME
    )


def test_source_verify_failure_propagates(sealed, monkeypatch):
    def _boom(bundle, *, bindings, deadline):
        raise gate.P05PublicationGateError("reporting_source_missing")

    monkeypatch.setattr(gate.reporting_inputs, "verify_reporting_sources", _boom)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "reporting_source_missing"


# --- added adversarial coverage ---------------------------------------------
_SOURCE_MAX = gate.evaluation_authority.SOURCE_MAXIMUM_UPDATES
_REFIT_MAX = (
    gate.evaluation_authority.MAXIMUM_REFITS * gate.evaluation_authority.MAXIMUM_UPDATES_PER_REFIT
)


def _set_cost(key: str, value: object):
    def _apply(payload):
        payload[key] = value

    return _apply


def _bump_counter(key: str, delta: float):
    def _apply(payload):
        payload["counters"][key] = payload["counters"][key] + delta

    return _apply


def _bump_cumulative(payload):
    payload["scientific_seconds_cumulative_bound_through_comparison"] += 1.0


def test_malformed_expected_digest_types(sealed):
    for value in (None, 7, True):
        with pytest.raises(gate.P05PublicationGateError) as exc:
            gate.load_public_bundle(
                sealed.artifact_root,
                expected_reporting_receipt_sha256=value,
                deadline=sealed.deadline,
            )
        assert exc.value.reason_code == "expected_receipt_sha_malformed"


def test_invalid_selection_plan_id(sealed):
    _mutate_receipt(sealed, _set("selection_plan_id", "not-a-digest"))
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "receipt_selection_plan_id_malformed"


STEP_CASES = [
    (
        "source_nonint",
        _set("source_optimizer_steps", "x"),
        "receipt_source_optimizer_steps_malformed",
    ),
    (
        "source_negative",
        _set("source_optimizer_steps", -1),
        "receipt_source_optimizer_steps_out_of_range",
    ),
    (
        "source_oversized",
        _set("source_optimizer_steps", _SOURCE_MAX + 1),
        "receipt_source_optimizer_steps_out_of_range",
    ),
    (
        "refit_nonint",
        _set("refit_optimizer_steps", None),
        "receipt_refit_optimizer_steps_malformed",
    ),
    (
        "refit_negative",
        _set("refit_optimizer_steps", -1),
        "receipt_refit_optimizer_steps_out_of_range",
    ),
    (
        "refit_oversized",
        _set("refit_optimizer_steps", _REFIT_MAX + 1),
        "receipt_refit_optimizer_steps_out_of_range",
    ),
]


@pytest.mark.parametrize("name,mutator,code", STEP_CASES, ids=[c[0] for c in STEP_CASES])
def test_receipt_optimizer_step_bounds(sealed, name, mutator, code):
    _mutate_receipt(sealed, mutator)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == code


def test_summary_only_boolean_zero_counter(sealed):
    name = reporting.ZERO_COUNTERS[0]
    _mutate_summary(sealed, _set_counter(name, False))
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == f"counter_{name}_invalid"


@pytest.mark.parametrize(
    "target,code",
    [("summary", "summary_elapsed_mismatch"), ("receipt", "receipt_elapsed_mismatch")],
)
def test_counter_elapsed_mismatch(sealed, target, code):
    mutate = _mutate_summary if target == "summary" else _mutate_receipt
    mutate(sealed, _bump_counter("elapsed_seconds", 1.0))
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == code


COST_AGREEMENT_CASES = [
    (
        "no_refits",
        _set_cost("unique_refitted_models", 0),
        "public_cost_unique_refitted_models_out_of_range",
    ),
    (
        "too_many_refits",
        _set_cost("unique_refitted_models", ALIASES + 1),
        "public_cost_unique_refitted_models_out_of_range",
    ),
    (
        "calibrations",
        _set_cost("unique_scalar_calibrations", UNIQUE_REFITS - 1),
        "public_cost_unique_scalar_calibrations_mismatch",
    ),
    (
        "total",
        _set_cost("new_neural_fits_total", SOURCE_FITS + UNIQUE_REFITS + 1),
        "public_cost_new_neural_fits_total_mismatch",
    ),
    (
        "aliases",
        _set_cost("strategy_alias_count", ALIASES + 1),
        "public_cost_strategy_alias_count_mismatch",
    ),
    (
        "reused",
        _set_cost("reused_pilot_fits", REUSED_FITS + 1),
        "public_cost_reused_pilot_fits_mismatch",
    ),
    ("new", _set_cost("new_source_fits", SOURCE_FITS + 1), "public_cost_new_source_fits_mismatch"),
    (
        "evidence",
        _set_cost("source_evidence_fits", EVIDENCE_FITS + 1),
        "public_cost_source_evidence_fits_mismatch",
    ),
    (
        "source_updates",
        _set_cost("new_source_optimizer_updates", SOURCE_UPDATES + 1),
        "public_cost_source_updates_mismatch",
    ),
    (
        "refit_updates",
        _set_cost("refit_optimizer_updates", REFIT_UPDATES + 1),
        "public_cost_refit_updates_mismatch",
    ),
    (
        "combined_updates",
        _set_cost("combined_new_optimizer_updates", SOURCE_UPDATES + REFIT_UPDATES + 1),
        "public_cost_combined_updates_mismatch",
    ),
    ("time_prior", _bump_cumulative, "public_cost_cumulative_mismatch"),
    (
        "peak",
        _set_cost(
            "refit_peak_allocated_gpu_bytes",
            gate.evaluation_authority.MAXIMUM_CUDA_ALLOCATED_BYTES + 1,
        ),
        "public_cost_refit_peak_allocated_gpu_bytes_out_of_range",
    ),
]


@pytest.mark.parametrize(
    "name,mutator,code", COST_AGREEMENT_CASES, ids=[c[0] for c in COST_AGREEMENT_CASES]
)
def test_cost_agreement_mismatch(sealed, name, mutator, code):
    _mutate_costs(sealed, mutator)
    with pytest.raises(
        (gate.P05PublicationGateError, reporting.P05ComprehensiveReportingError)
    ) as exc:
        _load(sealed)
    assert exc.value.reason_code == code


def test_second_source_verify_failure(sealed, monkeypatch):
    calls = {"n": 0}

    def _verify(bundle, *, bindings, deadline):
        freeze._check_deadline(deadline)
        calls["n"] += 1
        if calls["n"] == 2:
            raise gate.P05PublicationGateError("reporting_source_missing")
        return {"verified": True}

    monkeypatch.setattr(gate.reporting_inputs, "verify_reporting_sources", _verify)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "reporting_source_missing"
    assert calls["n"] == 2


def test_receipt_mutated_after_final_manifest_scan(sealed, monkeypatch):
    real = gate.pilot._verify_manifest
    scans = {"n": 0}

    def _verify(stage):
        real(stage)
        scans["n"] += 1
        if scans["n"] == 2:
            path = _receipt_path(sealed)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["status"] = "tampered"
            core._atomic_write(path, core._canon().canonical_json_bytes(payload))

    monkeypatch.setattr(gate.pilot, "_verify_manifest", _verify)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert exc.value.reason_code == "reporting_receipt_changed"


def test_receipt_mutated_during_second_source_verification(sealed, monkeypatch):
    calls = []

    def _verify(bundle, *, bindings, deadline):
        calls.append(dict(bindings))
        if len(calls) == 2:
            path = _receipt_path(sealed)
            core._atomic_write(path, path.read_bytes() + b"\n")
        return {"verified": True}

    monkeypatch.setattr(gate.reporting_inputs, "verify_reporting_sources", _verify)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load(sealed)
    assert len(calls) == 2
    assert exc.value.reason_code == "reporting_receipt_changed"


def test_late_deadline_expires_after_final_scan(sealed, monkeypatch):
    real_verify = gate.pilot._verify_manifest
    real_check = freeze._check_deadline
    state = {"scans": 0, "expired": False}

    def _verify(stage):
        real_verify(stage)
        state["scans"] += 1
        if state["scans"] == 2:
            state["expired"] = True

    def _check(deadline):
        if state["expired"]:
            raise core.P05CoreError("deadline_expired")
        return real_check(deadline)

    monkeypatch.setattr(gate.pilot, "_verify_manifest", _verify)
    monkeypatch.setattr(gate.freeze, "_check_deadline", _check)
    with pytest.raises(core.P05CoreError):
        _load(sealed)


def test_entry_deadline_expired_skips_receipt_read(sealed, monkeypatch):
    reads = []
    real_read = core._read_json

    def _read(path, code):
        reads.append(Path(path))
        return real_read(path, code)

    monkeypatch.setattr(gate.core, "_read_json", _read)
    with pytest.raises(core.P05CoreError):
        _load(sealed, deadline=0.0)
    assert reads == []
