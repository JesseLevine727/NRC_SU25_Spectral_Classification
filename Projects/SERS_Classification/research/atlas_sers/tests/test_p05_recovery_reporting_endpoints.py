"""End-to-end synthetic tests for the recovered P05 reporting endpoints.

Endpoint one drives the read-only reporting-inputs loader across a recovered
source layout: a real recovery summary/receipt pair, a real recovered manifest,
recovery-aware accounting and regenerated selection bindings.  Upstream source
acceptance and proof checks stay mocked by the existing reporting-inputs fixture,
but resolver paths, manifests, accounting and bindings are checked for real.

Endpoint two drives the comprehensive reporting producer against a recovered
comparison authority and consumes the sealed output through the real read-only
publication gate.  Only upstream ``verify_reporting_sources`` is stubbed, exactly
as the existing publication-gate fixture does; every stored file, manifest, cost
and accounting value is checked from real bytes.
"""

from __future__ import annotations

import json
import shutil
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as comprehensive_inputs
from atlas_sers.evaluation import p05_comprehensive_reporting as reporting
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_recovery_authority as recovery_authority
from atlas_sers.evaluation import p05_recovery_receipt as recovery_receipt
from atlas_sers.evaluation import p05_recovery_source as recovery_source
from atlas_sers.evaluation import p05_refit_authority as refit_authority
from atlas_sers.evaluation import p05_reporting_inputs
from atlas_sers.governance import p05_publication_gate as gate
from tests import test_p05_comprehensive_reporting as reporting_fixture
from tests import test_p05_recovery_source as recovery_fixture
from tests import test_p05_reporting_inputs as reporting_inputs_fixture

RECOVERY_PERMIT = recovery_authority.RECOVERY_PERMIT_SHA256

_RECOVERY_REFIT_STEPS = 240
_RECOVERY_COMPARISON_PRIOR = 40650.0
_RECOVERY_PLAN_ID = "d" * 64
_RECOVERY_UNIQUE_REFITS = 2
_RECOVERY_DEV_SECONDS = 1000.0
_RECOVERY_REFIT_SECONDS = 20.0


def _future() -> float:
    return time.perf_counter() + 3600.0


def _recovered_accounting() -> dict:
    summary, receipt_record = recovery_fixture._recovered_pair()
    return recovery_source.accounting_from_recovered(summary, receipt_record)


# ---------------------------------------------------------------------------
# Endpoint one: recovered reporting-input load against the resolver.
# ---------------------------------------------------------------------------


def _build_recovered_reporting_inputs(tmp_path, monkeypatch) -> SimpleNamespace:
    env = reporting_inputs_fixture._build(tmp_path, monkeypatch)
    summary, receipt_record = recovery_fixture._recovered_pair()
    accounting = recovery_source.accounting_from_recovered(summary, receipt_record)
    source_steps = accounting["source_optimizer_steps_successful_exact"]

    run_root = env.run_root
    approved = run_root / "recoveries" / RECOVERY_PERMIT
    approved.mkdir(parents=True)
    recovered_develop = approved / "develop"
    shutil.copytree(run_root / "develop", recovered_develop)
    (approved / "replay_lease.json").write_text("{}", encoding="utf-8")

    reporting_inputs_fixture._write_json(
        recovered_develop / p05_reporting_inputs.SUMMARY_NAME, summary
    )
    recovered_manifest_sha = reporting_inputs_fixture._seal(
        recovered_develop, p05_reporting_inputs.DEVELOP_MANIFEST_NAME
    )
    recovered_receipt = recovery_receipt.build_receipt(
        summary=summary,
        stage_manifest_sha256=recovered_manifest_sha,
        scientific_seconds=summary["scientific_seconds_this_stage"],
    )
    reporting_inputs_fixture._write_json(
        run_root / recovery_receipt.RECEIPT_NAME, recovered_receipt
    )
    (run_root / "development_receipt.json").unlink()

    env.paths = recovery_source.resolve_paths(env.bundle)

    evaluation_authority = reporting_inputs_fixture.evaluation_authority
    refit_receipt_path = run_root / evaluation_authority.RECEIPT_NAME
    refit_receipt = json.loads(refit_receipt_path.read_text(encoding="utf-8"))
    refit_receipt["source_optimizer_steps"] = source_steps
    refit_receipt["source_execution_accounting"] = dict(accounting)
    refit_receipt["total_new_optimizer_steps"] = source_steps + reporting_inputs_fixture.REFIT_STEPS
    refit_receipt["total_new_optimizer_steps_charged_upper_bound"] = (
        accounting["source_optimizer_steps_charged_upper_bound"]
        + reporting_inputs_fixture.REFIT_STEPS
    )
    refit_receipt["total_new_optimizer_steps_all_attempts_exact"] = False
    reporting_inputs_fixture._write_json(refit_receipt_path, refit_receipt)

    comparison_receipt_path = run_root / reporting_inputs_fixture.comparison_authority.RECEIPT_NAME
    comparison_receipt = json.loads(comparison_receipt_path.read_text(encoding="utf-8"))
    comparison_receipt["scientific_seconds_cumulative_bound"] = 40600.0
    reporting_inputs_fixture._write_json(comparison_receipt_path, comparison_receipt)

    env.authenticated["source_optimizer_steps"] = source_steps
    env.authenticated["source_execution_accounting"] = dict(accounting)
    env.authenticated["comparison_receipt"] = dict(comparison_receipt)
    env.authenticated["prior_seconds"] = 40600.0

    local_bundle = {**env.bundle, "source_execution_accounting": dict(accounting)}
    recovered_bindings = freeze._source_bindings(
        local_bundle,
        recovered_manifest_sha,
        core._canon().sha256_file(recovered_develop / p05_reporting_inputs.SOURCE_LEDGER_NAME),
        core._canon().sha256_file(recovered_develop / p05_reporting_inputs.SELECTOR_NAME),
    )
    selection_stage = run_root / "selection"
    reporting_inputs_fixture._write_json(
        selection_stage / refit_authority.SELECTION_BINDINGS_NAME, recovered_bindings
    )
    selection_manifest_sha = reporting_inputs_fixture._seal(
        selection_stage, refit_authority.SELECTION_MANIFEST_NAME
    )
    selection_receipt_path = run_root / "selection_receipt.json"
    selection_receipt = json.loads(selection_receipt_path.read_text(encoding="utf-8"))
    selection_receipt["selection_manifest_sha256"] = selection_manifest_sha
    reporting_inputs_fixture._write_json(selection_receipt_path, selection_receipt)

    env.accounting = accounting
    env.source_steps = source_steps
    env.recovered_develop = recovered_develop
    env.recovered_manifest_sha = recovered_manifest_sha
    env.local_bundle = local_bundle
    return env


@pytest.fixture
def recovered_env(tmp_path, monkeypatch):
    return _build_recovered_reporting_inputs(tmp_path, monkeypatch)


def _load_recovered(env: SimpleNamespace) -> dict:
    return p05_reporting_inputs.load_reporting_sources(
        env.bundle,
        authenticated=env.authenticated,
        deadline=_future(),
    )


def _mutate_recovered_refit_receipt(env: SimpleNamespace, mutator) -> None:
    path = env.run_root / reporting_inputs_fixture.evaluation_authority.RECEIPT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    reporting_inputs_fixture._write_json(path, payload)


def test_recovered_reporting_inputs_load_success(recovered_env):
    result = _load_recovered(recovered_env)
    assert len(result["selector_records"]) == p05_reporting_inputs.SOURCE_EVIDENCE_FITS

    accounting = recovered_env.accounting
    source_steps = recovered_env.source_steps
    refit_steps = reporting_inputs_fixture.REFIT_STEPS
    costs = result["public_costs"]

    assert costs["new_source_attempts"] == accounting["source_attempts"] == 14905
    assert costs["original_interrupted_source_attempts"] == 1
    assert costs["replayed_source_attempts"] == 1
    assert costs["new_neural_attempts_total"] == 14905 + 1
    assert costs["source_optimizer_updates_observed_lower_bound"] == source_steps + 68
    assert costs["source_optimizer_updates_charged_upper_bound"] == source_steps + 800
    assert (
        costs["combined_optimizer_updates_observed_lower_bound"] == source_steps + refit_steps + 68
    )
    assert (
        costs["combined_optimizer_updates_charged_upper_bound"] == source_steps + refit_steps + 800
    )
    assert costs["prior_source_scientific_seconds_charged_upper_bound"] == 36000.0
    assert costs["recovery_source_scientific_seconds_charged_upper_bound"] == 1000.0
    assert costs["source_scientific_seconds"] == 37000.0
    assert set(result["bindings"]) == p05_reporting_inputs.BINDING_KEYS

    verified = p05_reporting_inputs.verify_reporting_sources(
        recovered_env.bundle, bindings=result["bindings"], deadline=_future()
    )
    assert verified["verified"] is True
    assert verified["bindings"] == result["bindings"]


def test_recovered_reporting_inputs_reject_stripped_authenticated_accounting(recovered_env):
    recovered_env.authenticated.pop("source_execution_accounting")
    with pytest.raises(
        p05_reporting_inputs.ReportingInputsError, match="source_recovery_mode_mismatch"
    ):
        _load_recovered(recovered_env)


def test_recovered_reporting_inputs_reject_missing_receipt_accounting(recovered_env):
    def _strip(payload):
        payload.pop("source_execution_accounting")

    _mutate_recovered_refit_receipt(recovered_env, _strip)
    with pytest.raises(recovery_source.RecoverySourceError, match="source_accounting_malformed"):
        _load_recovered(recovered_env)


def test_recovered_reporting_inputs_reject_tampered_charged_bound(recovered_env):
    def _bump(payload):
        nested = dict(payload["source_execution_accounting"])
        nested["source_optimizer_steps_charged_upper_bound"] += 4
        payload["source_execution_accounting"] = nested

    _mutate_recovered_refit_receipt(recovered_env, _bump)
    with pytest.raises(
        recovery_source.RecoverySourceError, match="source_accounting_upper_bound_mismatch"
    ):
        _load_recovered(recovered_env)


# ---------------------------------------------------------------------------
# Endpoint two: recovered reporting producer and publication-gate round-trip.
# ---------------------------------------------------------------------------


def _reseal_reporting_stage(copy: SimpleNamespace) -> None:
    core._write_manifest(copy.stage)
    manifest_path = copy.stage / reporting.MANIFEST_NAME
    receipt_path = copy.run_root / reporting.RECEIPT_NAME
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["stage_manifest_sha256"] = core._canon().sha256_file(manifest_path)
    core._atomic_write(receipt_path, core._canon().canonical_json_bytes(payload))
    copy.receipt = payload
    copy.receipt_sha256 = core._canon().sha256_file(receipt_path)


def _mutate_reporting_receipt(copy: SimpleNamespace, mutator) -> None:
    receipt_path = copy.run_root / reporting.RECEIPT_NAME
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutator(payload)
    core._atomic_write(receipt_path, core._canon().canonical_json_bytes(payload))
    copy.receipt = payload
    copy.receipt_sha256 = core._canon().sha256_file(receipt_path)


def _mutate_reporting_costs(copy: SimpleNamespace, mutator) -> None:
    costs_path = copy.public_root / reporting.TABLES_DIR_NAME / reporting.COSTS_NAME
    payload = json.loads(costs_path.read_text(encoding="utf-8"))
    mutator(payload)
    core._atomic_write(costs_path, core._canon().canonical_json_bytes(payload))
    _reseal_reporting_stage(copy)


def _seal_recovered_reporting(mp: pytest.MonkeyPatch, base) -> SimpleNamespace:
    ctx = reporting_fixture._setup(mp, base)
    accounting = _recovered_accounting()
    source_steps = accounting["source_optimizer_steps_successful_exact"]

    run_root = ctx["run_root"]
    comparison_receipt_path = run_root / reporting.COMPARISON_RECEIPT_NAME
    comparison_manifest_path = (
        run_root / reporting.COMPARISON_STAGE_NAME / reporting.COMPARISON_MANIFEST_NAME
    )
    comparison_manifest_sha = core._canon().sha256_file(comparison_manifest_path)
    comparison_receipt = {
        reporting.PRIOR_FIELD: _RECOVERY_COMPARISON_PRIOR,
        "stage_manifest_sha256": comparison_manifest_sha,
    }
    core._atomic_write(
        comparison_receipt_path, core._canon().canonical_json_bytes(comparison_receipt)
    )

    def _authenticate(bundle, *, deadline):
        return {
            "prior_seconds": _RECOVERY_COMPARISON_PRIOR,
            "comparison_receipt": dict(comparison_receipt),
            "plan": {"plan_id": _RECOVERY_PLAN_ID},
            "comparison_tables": {},
            "aggregation_tables": {"ensemble_predictions": pd.DataFrame({"value": [1.0]})},
            "source_optimizer_steps": source_steps,
            "refit_optimizer_steps": _RECOVERY_REFIT_STEPS,
            "source_execution_accounting": dict(accounting),
        }

    mp.setattr(ctx["authority"], "authenticate_comparison", _authenticate)

    costs = p05_reporting_inputs._assemble_costs(
        {"scientific_seconds_this_stage": _RECOVERY_DEV_SECONDS},
        {"scientific_seconds_this_stage": _RECOVERY_REFIT_SECONDS},
        {},
        comparison_receipt,
        _RECOVERY_UNIQUE_REFITS,
        source_steps,
        _RECOVERY_REFIT_STEPS,
        source_accounting=accounting,
    )
    mp.setattr(reporting_fixture, "STAGE_COSTS", costs)

    bindings = {key: "a" * 64 for key in p05_reporting_inputs.BINDING_KEYS}
    bindings["comparison_receipt_sha256"] = core._canon().sha256_file(comparison_receipt_path)
    bindings["comparison_manifest_sha256"] = core._canon().sha256_file(comparison_manifest_path)
    mp.setattr(reporting_fixture, "BINDINGS", bindings)

    mp.setattr(comprehensive_inputs, "COMPREHENSIVE_PERMIT_SHA256", ctx["bundle"]["permit_sha256"])
    mp.setattr(comprehensive_inputs, "CORE_CONTRACT_SHA256", ctx["bundle"]["contract_sha256"])
    mp.setattr(comprehensive_inputs, "CORE_PLAN_ID", ctx["bundle"]["core_plan_id"])
    mp.setattr(comprehensive_inputs, "LEDGER_ID", ctx["bundle"]["ledger"]["ledger_id"])

    calls: list[dict[str, str]] = []

    def _stub_verify_sources(bundle, *, bindings, deadline):
        freeze._check_deadline(deadline)
        calls.append(dict(bindings))
        return {"verified": True}

    mp.setattr(gate.reporting_inputs, "verify_reporting_sources", _stub_verify_sources)

    receipt = reporting.run_reporting(**ctx["kwargs"])
    ctx["receipt"] = receipt
    return SimpleNamespace(
        artifact_root=ctx["artifact_root"],
        bundle=ctx["bundle"],
        run_root=run_root,
        ctx=ctx,
        receipt=receipt,
        accounting=accounting,
        source_steps=source_steps,
        costs=costs,
        calls=calls,
    )


@pytest.fixture(scope="module")
def reporting_sealed(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("recovered_reporting")
    try:
        yield _seal_recovered_reporting(mp, base)
    finally:
        mp.undo()


@pytest.fixture
def reporting_copy(tmp_path, reporting_sealed):
    dest = tmp_path / "artifact"
    shutil.copytree(reporting_sealed.artifact_root, dest)
    run_root = (
        dest
        / reporting.COMPREHENSIVE_DIR
        / reporting.RUNS_DIR
        / reporting_sealed.bundle["permit_sha256"]
    )
    stage = run_root / reporting.STAGE_NAME
    receipt_path = run_root / reporting.RECEIPT_NAME
    return SimpleNamespace(
        artifact_root=dest,
        bundle=dict(reporting_sealed.bundle),
        run_root=run_root,
        stage=stage,
        public_root=stage / reporting.PUBLIC_ROOT_NAME,
        receipt=json.loads(receipt_path.read_text(encoding="utf-8")),
        receipt_sha256=core._canon().sha256_file(receipt_path),
        deadline=_future(),
        accounting=dict(reporting_sealed.accounting),
        source_steps=reporting_sealed.source_steps,
    )


def _load_gate(copy: SimpleNamespace):
    return gate.load_public_bundle(
        copy.artifact_root,
        expected_reporting_receipt_sha256=copy.receipt_sha256,
        deadline=copy.deadline,
    )


def test_recovered_reporting_producer_propagates_accounting(reporting_copy):
    receipt = reporting_copy.receipt
    summary = json.loads(
        (reporting_copy.stage / reporting.SUMMARY_NAME).read_text(encoding="utf-8")
    )
    accounting = reporting_copy.accounting

    assert receipt["status"] == "complete"
    assert receipt["reporting_complete"] is True
    assert receipt["source_execution_accounting"] == accounting
    assert summary["source_execution_accounting"] == accounting
    assert (
        receipt["source_optimizer_steps"] == accounting["source_optimizer_steps_successful_exact"]
    )
    assert receipt["refit_optimizer_steps"] == _RECOVERY_REFIT_STEPS
    assert receipt["prior_scientific_seconds_cumulative_bound"] == _RECOVERY_COMPARISON_PRIOR


def test_recovered_reporting_publication_roundtrip(reporting_copy):
    bundle = _load_gate(reporting_copy)
    assert isinstance(bundle, gate.PublicBundle)
    assert bundle.receipt_sha256 == reporting_copy.receipt_sha256
    assert len(bundle.tables) == reporting.TOTAL_TABLE_COUNT
    assert set(bundle.figure_manifests) == {
        reporting.PAIRED_UNIT_NAME,
        reporting.DIAGNOSTIC_UNIT_NAME,
    }

    source_steps = reporting_copy.source_steps
    costs = bundle.costs
    assert set(p05_reporting_inputs.RECOVERY_PUBLIC_COST_KEYS) <= set(costs)
    assert costs["new_source_attempts"] == 14905
    assert costs["original_interrupted_source_attempts"] == 1
    assert costs["replayed_source_attempts"] == 1
    assert costs["new_neural_attempts_total"] == 14905 + _RECOVERY_UNIQUE_REFITS
    assert costs["source_optimizer_updates_observed_lower_bound"] == source_steps + 68
    assert costs["source_optimizer_updates_charged_upper_bound"] == source_steps + 800
    assert (
        costs["combined_optimizer_updates_observed_lower_bound"]
        == source_steps + _RECOVERY_REFIT_STEPS + 68
    )
    assert (
        costs["combined_optimizer_updates_charged_upper_bound"]
        == source_steps + _RECOVERY_REFIT_STEPS + 800
    )
    assert costs["prior_source_scientific_seconds_charged_upper_bound"] == 36000.0
    assert costs["recovery_source_scientific_seconds_charged_upper_bound"] == _RECOVERY_DEV_SECONDS
    assert (
        bundle.cumulative_seconds == reporting_copy.receipt["scientific_seconds_cumulative_bound"]
    )


def test_recovered_reporting_gate_rejects_malformed_accounting(reporting_copy):
    def _break_interrupted(payload):
        nested = dict(payload["source_execution_accounting"])
        nested["source_interrupted_attempts"] = True
        payload["source_execution_accounting"] = nested

    _mutate_reporting_receipt(reporting_copy, _break_interrupted)
    with pytest.raises(gate.P05PublicationGateError) as exc:
        _load_gate(reporting_copy)
    assert exc.value.reason_code == "receipt_source_accounting_malformed"


def test_recovered_reporting_gate_rejects_stripped_recovery_costs(reporting_copy):
    def _drop_attempts(payload):
        del payload["new_source_attempts"]

    _mutate_reporting_costs(reporting_copy, _drop_attempts)
    with pytest.raises(core.P05CoreError) as exc:
        _load_gate(reporting_copy)
    assert exc.value.reason_code == "recovery_public_cost_key_missing"
