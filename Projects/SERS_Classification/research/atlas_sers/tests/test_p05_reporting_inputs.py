"""Boundary tests for the P05 reporting-inputs read-only helper.

Only the reporting-input boundary is exercised: the already-authenticated
comparison authority result is fed in, the frozen stage directories are
rebuilt from real manifests and SHA256 receipts, and every mutation must
fail closed instead of exporting reporting inputs.
"""

from __future__ import annotations

# ruff: noqa: E402
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from atlas_sers.evaluation import p05_comparison_authority as comparison_authority
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as evaluation_authority
from atlas_sers.evaluation import p05_refit_authority as refit_authority
from atlas_sers.evaluation import p05_reporting_inputs as reporting

PLAN_ID = "a" * 64
PERMIT = "b" * 64
CONTRACT = "c" * 64
CORE_PLAN = "d" * 64
LEDGER_ID = "e" * 64
SOURCE_STEPS = 1
REFIT_STEPS = 1


def _future() -> float:
    return time.perf_counter() + 3600.0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _seal(stage: Path, manifest_name: str) -> str:
    core._write_manifest(stage)
    manifest_path = stage / manifest_name
    assert manifest_path.is_file(), "core._write_manifest produced no manifest"
    return core._canon().sha256_file(manifest_path)


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    artifact_root = tmp_path / "artifacts"
    bundle = {
        "artifact_root": str(artifact_root),
        "permit_sha256": PERMIT,
        "contract_sha256": CONTRACT,
        "core_plan_id": CORE_PLAN,
        "ledger": {"ledger_id": LEDGER_ID},
    }
    run_root = (
        artifact_root
        / comparison_authority.COMPREHENSIVE_DIR
        / comparison_authority.RUNS_DIR
        / PERMIT
    )
    develop = run_root / "develop"
    selection = run_root / "selection"
    refit_stage = run_root / evaluation_authority.STAGE_NAME
    comparison_stage = run_root / comparison_authority.STAGE_NAME
    for directory in (develop, selection, refit_stage, comparison_stage):
        directory.mkdir(parents=True, exist_ok=True)

    paths = freeze._paths(bundle)
    monkeypatch.setattr(freeze, "_check_receipt", lambda receipt, expected: None)
    monkeypatch.setattr(freeze, "_expected_new_units", lambda bundle: None)

    unique_refits = {"refit_0000": {"seed": 0}}
    aliases = [{"refit_id": "refit_0000"} for _ in range(reporting.STRATEGY_ALIASES)]
    plan = {"plan_id": PLAN_ID, "unique_refits": unique_refits, "strategy_aliases": aliases}

    selector_path = develop / reporting.SELECTOR_NAME
    with selector_path.open("w", encoding="utf-8") as handle:
        for index in range(reporting.SOURCE_EVIDENCE_FITS):
            handle.write(json.dumps({"index": index, "status": "complete"}) + "\n")
    _write_json(develop / reporting.SOURCE_LEDGER_NAME, {"ledger_id": LEDGER_ID})
    develop_manifest_sha = _seal(develop, reporting.DEVELOP_MANIFEST_NAME)

    development_receipt = {
        "new_completions": reporting.NEW_SOURCE_FITS,
        "selector_records": reporting.SOURCE_EVIDENCE_FITS,
        "optimizer_steps": SOURCE_STEPS,
        "scientific_seconds_this_stage": 0.0,
        "stage_manifest_sha256": develop_manifest_sha,
    }
    _write_json(paths["receipt"], development_receipt)

    _write_json(refit_stage / "payload.json", {"ok": True})
    refit_manifest_sha = _seal(refit_stage, evaluation_authority.MANIFEST_NAME)
    counters = {key: 0 for key in evaluation_authority.COUNTER_KEYS}
    counters.update(
        {
            "calibration_started": 1,
            "calibration_completed": 1,
            "calibration_failed": 0,
            "neural_started": 1,
            "neural_completed": 1,
            "neural_failed": 0,
            "optimizer_steps": REFIT_STEPS,
            "optimizer_steps_exact": True,
        }
    )
    refit_receipt = {
        "schema_version": evaluation_authority.SCHEMA,
        "stage": evaluation_authority.STAGE_NAME,
        "command": evaluation_authority.COMMAND,
        "claim": evaluation_authority.CLAIM,
        "status": "complete",
        "refits_complete": True,
        "calibrations_complete": True,
        "outer_predictions_started": 0,
        "permit_sha256": PERMIT,
        "core_contract_sha256": CONTRACT,
        "core_plan_id": CORE_PLAN,
        "ledger_id": LEDGER_ID,
        "selection_plan_id": PLAN_ID,
        "source_fit_count": reporting.NEW_SOURCE_FITS,
        "reused_pilot_fit_count": reporting.REUSED_PILOT_FITS,
        "unique_refit_count": 1,
        "strategy_alias_count": reporting.STRATEGY_ALIASES,
        "source_optimizer_steps": SOURCE_STEPS,
        "counters": counters,
        "total_new_optimizer_steps": SOURCE_STEPS + REFIT_STEPS,
        "scientific_seconds_this_stage": 0.0,
        "stage_manifest_sha256": refit_manifest_sha,
    }
    _write_json(run_root / evaluation_authority.RECEIPT_NAME, refit_receipt)

    _write_json(comparison_stage / "payload.json", {"ok": True})
    comparison_manifest_sha = _seal(comparison_stage, comparison_authority.MANIFEST_NAME)
    cumulative = development.PRELAUNCH_AUDIT_RESERVE_SECONDS + 1.0
    comparison_receipt = {
        "status": "complete",
        "comparison_complete": True,
        "permit_sha256": PERMIT,
        "selection_plan_id": PLAN_ID,
        "stage_manifest_sha256": comparison_manifest_sha,
        "scientific_seconds_cumulative_bound": cumulative,
    }
    _write_json(run_root / comparison_authority.RECEIPT_NAME, comparison_receipt)

    _write_json(selection / refit_authority.SELECTION_PLAN_NAME, plan)
    _write_json(
        selection / refit_authority.SELECTION_BINDINGS_NAME,
        freeze._source_bindings(
            bundle,
            develop_manifest_sha,
            core._canon().sha256_file(develop / reporting.SOURCE_LEDGER_NAME),
            core._canon().sha256_file(selector_path),
        ),
    )
    selection_manifest_sha = _seal(selection, refit_authority.SELECTION_MANIFEST_NAME)
    selection_receipt = {
        "stage": freeze.SELECTION_STAGE_NAME,
        "selection_plan_id": PLAN_ID,
        "selection_manifest_sha256": selection_manifest_sha,
    }
    _write_json(paths["selection_receipt"], selection_receipt)

    authenticated = {
        "plan": plan,
        "comparison_receipt": comparison_receipt,
        "source_optimizer_steps": SOURCE_STEPS,
        "refit_optimizer_steps": REFIT_STEPS,
        "prior_seconds": cumulative,
    }
    return SimpleNamespace(
        run_root=run_root,
        paths=paths,
        bundle=bundle,
        authenticated=authenticated,
        development_receipt=development_receipt,
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch)


def _load(env, deadline=None):
    return reporting.load_reporting_sources(
        env.bundle,
        authenticated=env.authenticated,
        deadline=_future() if deadline is None else deadline,
    )


def test_happy_path_returns_full_inputs(env):
    result = _load(env)
    assert len(result["selector_records"]) == 14940
    costs = result["public_costs"]
    assert costs["new_source_fits"] == 14904
    assert costs["reused_pilot_fits"] == 36
    assert costs["source_evidence_fits"] == 14940
    assert costs["strategy_alias_count"] == 2880
    assert costs["unique_refitted_models"] == 1
    assert costs["unique_scalar_calibrations"] == 1
    assert costs["new_neural_fits_total"] == 14905
    assert costs["combined_new_optimizer_updates"] == SOURCE_STEPS + REFIT_STEPS
    for value in costs.values():
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert set(result["bindings"]) == reporting.BINDING_KEYS
    verified = reporting.verify_reporting_sources(
        env.bundle, bindings=result["bindings"], deadline=_future()
    )
    assert verified["verified"] is True


def _mutate_count(env):
    env.development_receipt["new_completions"] = reporting.NEW_SOURCE_FITS - 1
    _write_json(env.paths["receipt"], env.development_receipt)


def _mutate_nan_time(env):
    env.development_receipt["scientific_seconds_this_stage"] = float("nan")
    _write_json(env.paths["receipt"], env.development_receipt)


def _mutate_bool_time(env):
    env.development_receipt["scientific_seconds_this_stage"] = True
    _write_json(env.paths["receipt"], env.development_receipt)


def _mutate_incomplete_status(env):
    path = env.paths["develop"] / reporting.SELECTOR_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps({"index": 0, "status": "failed"})
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env.development_receipt["stage_manifest_sha256"] = _seal(
        env.paths["develop"], reporting.DEVELOP_MANIFEST_NAME
    )
    _write_json(env.paths["receipt"], env.development_receipt)


@pytest.mark.parametrize(
    "mutate", [_mutate_count, _mutate_nan_time, _mutate_bool_time, _mutate_incomplete_status]
)
def test_pre_load_mutations_fail_closed(env, mutate):
    mutate(env)
    with pytest.raises(core.P05CoreError):
        _load(env)


def test_verify_detects_selector_change(env):
    bindings = _load(env)["bindings"]
    path = env.paths["develop"] / reporting.SELECTOR_NAME
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(core.P05CoreError):
        reporting.verify_reporting_sources(env.bundle, bindings=bindings, deadline=_future())


def test_verify_detects_receipt_change(env):
    bindings = _load(env)["bindings"]
    path = env.run_root / comparison_authority.RECEIPT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    _write_json(path, payload)
    with pytest.raises(core.P05CoreError):
        reporting.verify_reporting_sources(env.bundle, bindings=bindings, deadline=_future())


def test_verify_detects_extra_file(env):
    bindings = _load(env)["bindings"]
    _write_json(env.paths["develop"] / "extra.json", {"unexpected": True})
    with pytest.raises(core.P05CoreError):
        reporting.verify_reporting_sources(env.bundle, bindings=bindings, deadline=_future())


def test_verify_detects_symlink(env):
    bindings = _load(env)["bindings"]
    target = env.run_root / comparison_authority.RECEIPT_NAME
    real = target.with_name(target.name + ".real")
    target.rename(real)
    target.symlink_to(real)
    with pytest.raises(core.P05CoreError):
        reporting.verify_reporting_sources(env.bundle, bindings=bindings, deadline=_future())


def test_expired_deadline_fails_closed(env):
    with pytest.raises(core.P05CoreError):
        _load(env, deadline=time.perf_counter() - 1.0)
    bindings = _load(env)["bindings"]
    with pytest.raises(core.P05CoreError):
        reporting.verify_reporting_sources(
            env.bundle, bindings=bindings, deadline=time.perf_counter() - 1.0
        )


@pytest.mark.parametrize("status", ["failed", "skipped", "unknown", "", True])
def test_resealed_selector_still_requires_complete_status(env, status):
    selector = env.paths["develop"] / reporting.SELECTOR_NAME
    records = selector.read_text().splitlines()
    records[0] = json.dumps({"index": 0, "status": status})
    selector.write_text("\n".join(records) + "\n")
    stage_hash = _seal(env.paths["develop"], reporting.DEVELOP_MANIFEST_NAME)
    env.development_receipt["stage_manifest_sha256"] = stage_hash
    _write_json(env.paths["receipt"], env.development_receipt)
    _write_json(
        env.paths["selection"] / refit_authority.SELECTION_BINDINGS_NAME,
        freeze._source_bindings(
            env.bundle,
            stage_hash,
            core._canon().sha256_file(env.paths["develop"] / reporting.SOURCE_LEDGER_NAME),
            core._canon().sha256_file(selector),
        ),
    )
    selection_hash = _seal(env.paths["selection"], refit_authority.SELECTION_MANIFEST_NAME)
    receipt = json.loads(env.paths["selection_receipt"].read_text())
    receipt["selection_manifest_sha256"] = selection_hash
    _write_json(env.paths["selection_receipt"], receipt)
    with pytest.raises(
        reporting.ReportingInputsError, match="selector_(record_incomplete|status_malformed)"
    ):
        _load(env)
