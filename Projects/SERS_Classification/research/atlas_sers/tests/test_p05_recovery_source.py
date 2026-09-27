"""Read-only tests for the P05 recovery source resolver/accounting bridge."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_receipt as recovery_receipt
from atlas_sers.evaluation import p05_recovery_source as source

BASE_PERMIT = authority.BASECOMPREHENSIVE_PERMIT_SHA256
RECOVERY_PERMIT = authority.RECOVERY_PERMIT_SHA256
NAMESPACE = source.COMPREHENSIVE_NAMESPACE


def _require_builders() -> None:
    for name in ("build_summary", "build_receipt"):
        assert hasattr(recovery_receipt, name), f"recovery receipt builder missing: {name}"


def _base_bundle() -> dict:
    return {
        "permit_sha256": inputs.COMPREHENSIVE_PERMIT_SHA256,
        "core_contract_sha256": inputs.CORE_CONTRACT_SHA256,
        "core_plan_id": inputs.CORE_PLAN_ID,
        "ledger_id": inputs.LEDGER_ID,
        "pilot_permit_sha256": inputs.PILOT_PERMIT_SHA256,
        "pilot_plan_id": inputs.PILOT_PLAN_ID,
        "pilot_manifest_sha256": inputs.PILOT_MANIFEST_SHA256,
    }


def _counters() -> dict:
    return {
        "new_started": 6184,
        "new_completed": 6184,
        "new_failed": 0,
        "new_optimizer_steps": 2000000,
        "new_optimizer_steps_exact": True,
        "new_elapsed_seconds": 100.0,
        "new_peak_cuda_bytes": 1024,
        "reused_completed": 8720,
        "reused_optimizer_steps": 1669388,
        "replay_started": 1,
        "unstarted_started": 6183,
    }


def _recovered_pair() -> tuple[dict, dict]:
    _require_builders()
    summary = recovery_receipt.build_summary(
        base_bundle=_base_bundle(),
        recovery_plan_id="a" * 64,
        counters=_counters(),
        units_completed=1242,
        selector_records=14940,
        scientific_seconds=1000.0,
        live_bytes=1,
    )
    receipt_value = recovery_receipt.build_receipt(
        summary=summary, stage_manifest_sha256="b" * 64, scientific_seconds=1000.0
    )
    return summary, receipt_value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def _run_root(tmp_path: Path) -> Path:
    return Path(tmp_path) / NAMESPACE / "runs" / BASE_PERMIT


def _written_receipt(tmp_path: Path) -> dict:
    _summary, receipt_value = _recovered_pair()
    return receipt_value


def _make_recovered_tree(tmp_path: Path, *, with_receipt: bool = True) -> tuple[Path, Path]:
    run_root = _run_root(tmp_path)
    (run_root / "develop").mkdir(parents=True)
    (run_root / "develop" / "dummy.json").write_text("{}", encoding="utf-8")
    approved = run_root / "recoveries" / RECOVERY_PERMIT
    (approved / "develop").mkdir(parents=True)
    (approved / "develop" / "dummy.json").write_text("{}", encoding="utf-8")
    (approved / "replay_lease.json").write_text("{}", encoding="utf-8")
    if with_receipt:
        _write_json(run_root / recovery_receipt.RECEIPT_NAME, _written_receipt(tmp_path))
    return run_root, approved


def _bundle(tmp_path: Path) -> dict:
    return {"artifact_root": tmp_path, "permit_sha256": BASE_PERMIT}


def _snapshot(root: Path) -> dict:
    result = {}
    for path in sorted(Path(root).rglob("*")):
        result[str(path.relative_to(root))] = None if path.is_dir() else path.read_bytes()
    return result


def test_resolve_paths_legacy_no_writes(tmp_path):
    bundle = _bundle(tmp_path)
    before = _snapshot(tmp_path)
    resolved = source.resolve_paths(bundle)
    after = _snapshot(tmp_path)
    assert before == after
    run_root = _run_root(tmp_path)
    assert set(resolved) == {
        "run_root",
        "develop",
        "receipt",
        "selection",
        "selection_receipt",
    }
    assert resolved["run_root"] == run_root
    assert resolved["develop"] == run_root / "develop"
    assert resolved["receipt"] == run_root / "development_receipt.json"
    assert resolved["selection"] == run_root / "selection"
    assert resolved["selection_receipt"] == run_root / "selection_receipt.json"


def test_resolve_paths_recovered_returns_recovery_develop(tmp_path):
    run_root, approved = _make_recovered_tree(tmp_path)
    bundle = _bundle(tmp_path)
    before = _snapshot(tmp_path)
    resolved = source.resolve_paths(bundle)
    after = _snapshot(tmp_path)
    assert before == after
    assert resolved["run_root"] == run_root
    assert resolved["develop"] == approved / "develop"
    assert resolved["receipt"] == run_root / recovery_receipt.RECEIPT_NAME
    assert resolved["selection"] == run_root / "selection"
    assert resolved["selection_receipt"] == run_root / "selection_receipt.json"


def test_resolve_paths_receipt_without_recoveries_rejected(tmp_path):
    run_root, _approved = _make_recovered_tree(tmp_path)
    shutil.rmtree(run_root / "recoveries")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_recoveries_without_receipt_rejected(tmp_path):
    run_root, _approved = _make_recovered_tree(tmp_path, with_receipt=False)
    assert not (run_root / recovery_receipt.RECEIPT_NAME).exists()
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_wrong_child_rejected(tmp_path):
    run_root, _approved = _make_recovered_tree(tmp_path)
    (run_root / "recoveries" / ("0" * 64)).mkdir()
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_wrong_schema_rejected(tmp_path):
    run_root, _approved = _make_recovered_tree(tmp_path)
    _write_json(
        run_root / recovery_receipt.RECEIPT_NAME,
        {"schema_version": "not-the-recovery-schema"},
    )
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_dual_authority_rejected(tmp_path):
    run_root, _approved = _make_recovered_tree(tmp_path)
    (run_root / "development_receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_symlink_recoveries_rejected(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlink unavailable")
    run_root, _approved = _make_recovered_tree(tmp_path)
    shutil.rmtree(run_root / "recoveries")
    target = tmp_path / "elsewhere"
    target.mkdir()
    try:
        os.symlink(target, run_root / "recoveries")
    except OSError as error:  # pragma: no cover - platform dependent
        pytest.skip(f"symlink unavailable: {error}")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_symlink_receipt_rejected(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlink unavailable")
    run_root, _approved = _make_recovered_tree(tmp_path)
    receipt_path = run_root / recovery_receipt.RECEIPT_NAME
    stored = tmp_path / "stored_receipt.json"
    shutil.move(str(receipt_path), str(stored))
    try:
        os.symlink(stored, receipt_path)
    except OSError as error:  # pragma: no cover - platform dependent
        pytest.skip(f"symlink unavailable: {error}")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_replay_lease_fifo_rejected(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo unavailable")
    _run_root_dir, approved = _make_recovered_tree(tmp_path)
    lease = approved / "replay_lease.json"
    lease.unlink()
    try:
        os.mkfifo(lease)
    except OSError as error:  # pragma: no cover - platform dependent
        pytest.skip(f"mkfifo unavailable: {error}")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_duplicate_receipt_keys_rejected(tmp_path):
    run_root, _approved = _make_recovered_tree(tmp_path)
    text = '{"schema_version": "x", "schema_version": "y"}'
    (run_root / recovery_receipt.RECEIPT_NAME).write_text(text, encoding="utf-8")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_nonfinite_receipt_rejected(tmp_path):
    run_root, _approved = _make_recovered_tree(tmp_path)
    text = '{"schema_version": "x", "value": NaN}'
    (run_root / recovery_receipt.RECEIPT_NAME).write_text(text, encoding="utf-8")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_resolve_paths_bundle_permit_mismatch(tmp_path):
    bundle = {"artifact_root": tmp_path, "permit_sha256": "0" * 64}
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(bundle)


def test_resolve_paths_symlink_ancestor_rejected(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlink unavailable")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(real, link)
    except OSError as error:  # pragma: no cover - platform dependent
        pytest.skip(f"symlink unavailable: {error}")
    bundle = {"artifact_root": link, "permit_sha256": BASE_PERMIT}
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(bundle)


def test_is_recovered_detection():
    assert source.is_recovered({"schema_version": recovery_receipt.SCHEMA_VERSION}) is True
    assert source.is_recovered({"recovery_permit_sha256": RECOVERY_PERMIT}) is True
    assert source.is_recovered({"schema_version": "nato-sers-p05-comprehensive-v1"}) is False
    assert source.is_recovered(None) is False
    assert source.is_recovered("text") is False


def test_accounting_from_recovered_normalized():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    assert accounting["schema_version"] == source.ACCOUNTING_SCHEMA_VERSION
    assert accounting["mode"] == "recovered"
    steps = accounting["source_optimizer_steps_successful_exact"]
    assert isinstance(steps, int) and not isinstance(steps, bool)
    assert accounting["source_optimizer_steps_observed_lower_bound"] == steps + 68
    assert accounting["source_optimizer_steps_charged_upper_bound"] == steps + 800
    assert accounting["source_optimizer_steps_all_attempts_exact"] is False
    assert accounting["source_successful_fits"] == 14904
    assert accounting["source_attempts"] == 14905
    assert accounting["source_interrupted_attempts"] == 1
    assert accounting["recovery_permit_sha256"] == RECOVERY_PERMIT
    assert accounting["maximum_source_optimizer_steps"] == 11924000
    assert accounting["maximum_new_neural_executions"] == 17785
    assert accounting["maximum_new_optimizer_steps"] == 14228000


def test_validate_accounting_roundtrip():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    steps = accounting["source_optimizer_steps_successful_exact"]
    copy = source.validate_accounting(accounting, source_optimizer_steps=steps)
    assert copy == accounting
    assert copy is not accounting


def test_clean_accounting_bounds_and_roundtrip():
    zero = source.clean_accounting(0)
    assert zero["mode"] == "clean"
    assert zero["source_optimizer_steps_observed_lower_bound"] == 0
    assert zero["source_optimizer_steps_charged_upper_bound"] == 0
    assert zero["source_optimizer_steps_all_attempts_exact"] is True
    assert zero["recovery_permit_sha256"] is None
    assert source.validate_accounting(zero, source_optimizer_steps=0) == zero

    top = source.clean_accounting(source.CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS)
    assert (
        source.validate_accounting(
            top, source_optimizer_steps=source.CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS
        )
        == top
    )

    with pytest.raises(source.RecoverySourceError):
        source.clean_accounting(source.CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS + 1)
    with pytest.raises(source.RecoverySourceError):
        source.clean_accounting(True)
    with pytest.raises(source.RecoverySourceError):
        source.clean_accounting(1.5)


def test_validate_rejects_forged_recovery_permit():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    steps = accounting["source_optimizer_steps_successful_exact"]
    bad = dict(accounting)
    bad["recovery_permit_sha256"] = "0" * 64
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(bad, source_optimizer_steps=steps)


def test_validate_rejects_forged_lower_bound():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    steps = accounting["source_optimizer_steps_successful_exact"]
    bad = dict(accounting)
    bad["source_optimizer_steps_observed_lower_bound"] = steps
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(bad, source_optimizer_steps=steps)


def test_validate_rejects_forged_upper_bound():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    steps = accounting["source_optimizer_steps_successful_exact"]
    bad = dict(accounting)
    bad["source_optimizer_steps_charged_upper_bound"] = steps
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(bad, source_optimizer_steps=steps)


def test_validate_rejects_clean_flag_forgery():
    clean = source.clean_accounting(100)
    bad = dict(clean)
    bad["source_optimizer_steps_all_attempts_exact"] = False
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(bad, source_optimizer_steps=100)


def test_validate_rejects_boolean_and_float_steps():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    steps = accounting["source_optimizer_steps_successful_exact"]

    boolean = dict(accounting)
    boolean["source_optimizer_steps_successful_exact"] = True
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(boolean, source_optimizer_steps=steps)

    floating = dict(accounting)
    floating["source_optimizer_steps_successful_exact"] = float(steps)
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(floating, source_optimizer_steps=steps)


def test_validate_rejects_wrong_successful_matching():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    steps = accounting["source_optimizer_steps_successful_exact"]
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(accounting, source_optimizer_steps=steps + 4)


def test_validate_rejects_out_of_range_clean():
    clean = source.clean_accounting(0)
    over = source.CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS + 4
    bad = dict(clean)
    bad["source_optimizer_steps_successful_exact"] = over
    bad["source_optimizer_steps_observed_lower_bound"] = over
    bad["source_optimizer_steps_charged_upper_bound"] = over
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(bad, source_optimizer_steps=over)


def test_mode_cost_constants():
    assert source.CLEAN_SOURCE_ATTEMPTS == source.RECOVERED_SOURCE_ATTEMPTS - 1
    assert source.CLEAN_SOURCE_INTERRUPTED_ATTEMPTS == 0
    assert source.RECOVERED_SOURCE_INTERRUPTED_ATTEMPTS == 1
    assert (
        source.CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS
        == source.RECOVERED_MAXIMUM_SOURCE_OPTIMIZER_STEPS - 800
    )
    assert (
        source.CLEAN_MAXIMUM_NEW_OPTIMIZER_STEPS
        == source.RECOVERED_MAXIMUM_NEW_OPTIMIZER_STEPS - 800
    )
    assert source.CLEAN_MAXIMUM_NEW_NEURAL_EXECUTIONS == 17784
    assert source.RECOVERED_MAXIMUM_NEW_NEURAL_EXECUTIONS == 17785


def test_from_authenticated_absent_key_is_clean():
    result = source.from_authenticated({"source_optimizer_steps": 3669388})
    assert result["mode"] == "clean"
    assert result["source_optimizer_steps_successful_exact"] == 3669388
    assert result["source_optimizer_steps_observed_lower_bound"] == 3669388
    assert result["source_optimizer_steps_charged_upper_bound"] == 3669388
    assert result["source_optimizer_steps_all_attempts_exact"] is True


def test_from_authenticated_valid_recovered():
    _require_builders()
    summary, receipt_value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt_value)
    steps = accounting["source_optimizer_steps_successful_exact"]
    result = source.from_authenticated(
        {
            "source_optimizer_steps": steps,
            "source_execution_accounting": accounting,
        }
    )
    assert result["mode"] == "recovered"
    assert result == accounting


def test_from_authenticated_present_none_rejected():
    with pytest.raises(source.RecoverySourceError):
        source.from_authenticated(
            {"source_optimizer_steps": 0, "source_execution_accounting": None}
        )


def test_from_authenticated_malformed_not_fallback():
    with pytest.raises(source.RecoverySourceError):
        source.from_authenticated(
            {
                "source_optimizer_steps": 0,
                "source_execution_accounting": {"mode": "clean"},
            }
        )


def test_completed_receipt_cannot_override_failed_stage(tmp_path):
    _root, approved = _make_recovered_tree(tmp_path)
    (approved / "develop" / "failure.json").write_text("{}")
    with pytest.raises(source.RecoverySourceError, match="recovery_stage_failed"):
        source.resolve_paths(_bundle(tmp_path))


@pytest.mark.parametrize("name", ["develop", "development_receipt.json", "selection"])
def test_legacy_paths_reject_symlinks_even_without_recovery(tmp_path, name):
    run_root = _run_root(tmp_path)
    run_root.mkdir(parents=True)
    (run_root / name).symlink_to(tmp_path / "missing")
    with pytest.raises(source.RecoverySourceError):
        source.resolve_paths(_bundle(tmp_path))


def test_bad_validator_return_never_falls_back_to_unvalidated_summary(monkeypatch):
    summary, value = _recovered_pair()
    monkeypatch.setattr(recovery_receipt, "validate_pair", lambda *a, **k: None)
    with pytest.raises(source.RecoverySourceError, match="recovered_optimizer_steps_missing"):
        source.accounting_from_recovered(summary, value)


@pytest.mark.parametrize(
    "key",
    [
        "source_successful_fits",
        "source_attempts",
        "source_interrupted_attempts",
        "maximum_source_optimizer_steps",
        "maximum_new_neural_executions",
        "maximum_new_optimizer_steps",
        "source_optimizer_steps_observed_lower_bound",
        "source_optimizer_steps_charged_upper_bound",
    ],
)
def test_recovery_accounting_cannot_silently_widen_any_count_or_bound(key):
    summary, value = _recovered_pair()
    accounting = source.accounting_from_recovered(summary, value)
    steps = accounting["source_optimizer_steps_successful_exact"]
    accounting[key] += 1
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(accounting, source_optimizer_steps=steps)


@pytest.mark.parametrize("steps", [0, 1669388 + 6184 * 120 - 4, 1669388 + 6184 * 800 + 4])
def test_recovery_bounds_do_not_accept_arbitrary_successful_steps(steps):
    accounting = source._recovered_accounting(steps)
    with pytest.raises(source.RecoverySourceError):
        source.validate_accounting(accounting, source_optimizer_steps=steps)
