"""Torch-free recovery selection-boundary tests for the P05 freeze stage.

These tests exercise only the recovery source-flavor boundary and the source
accounting plumbing added for the approved recovered source run.  No science
input, array, checkpoint, fit or dataset is touched: the completed-source proof
and the freeze path/receipt helpers are monkeypatched at the proof boundary,
while the pure recovery-receipt builders are exercised directly.
"""

from __future__ import annotations

import pytest

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_recovery_acceptance as acceptance
from atlas_sers.evaluation import p05_recovery_receipt as recovery_receipt
from atlas_sers.evaluation import p05_recovery_source as source
from atlas_sers.evaluation import p05_refit_authority as authority

PERMIT = inputs.COMPREHENSIVE_PERMIT_SHA256
IDENTITY_BUNDLE = {
    "permit_sha256": "1" * 64,
    "contract_sha256": "2" * 64,
    "core_plan_id": "3" * 64,
    "ledger": {"ledger_id": "4" * 64},
}
RECOVERY_BASE_BUNDLE = {
    "permit_sha256": recovery_receipt.BASE_PERMIT_SHA256,
    "contract_sha256": recovery_receipt.CORE_CONTRACT_SHA256,
    "core_plan_id": recovery_receipt.CORE_PLAN_ID,
    "ledger_id": recovery_receipt.LEDGER_ID,
}
RECOVERY_PLAN_ID = "a" * 64
RECOVERY_STAGE_SECONDS = 10.0
RECOVERY_RECEIPT_SECONDS = 20.0
NEW_RECOVERY_STEPS = (
    recovery_receipt.REQUIRED_RECOVERY_COMPLETED * recovery_receipt.MINIMUM_UPDATES_PER_RECOVERY_FIT
)


def _counters():
    return {
        "new_started": recovery_receipt.REQUIRED_RECOVERY_STARTED,
        "new_completed": recovery_receipt.REQUIRED_RECOVERY_COMPLETED,
        "new_failed": recovery_receipt.REQUIRED_RECOVERY_FAILED,
        "new_optimizer_steps": NEW_RECOVERY_STEPS,
        "new_optimizer_steps_exact": True,
        "new_elapsed_seconds": 5.0,
        "new_peak_cuda_bytes": 0,
        "reused_completed": recovery_receipt.REQUIRED_REUSED_ORIGINAL_COMPLETIONS,
        "reused_optimizer_steps": recovery_receipt.REUSED_ORIGINAL_OPTIMIZER_STEPS,
        "replay_started": recovery_receipt.REQUIRED_REPLAY_ATTEMPTS,
        "unstarted_started": recovery_receipt.REQUIRED_ORIGINALLY_UNSTARTED_ATTEMPTS,
    }


def _recovered_pair():
    summary = recovery_receipt.build_summary(
        base_bundle=RECOVERY_BASE_BUNDLE,
        recovery_plan_id=RECOVERY_PLAN_ID,
        counters=_counters(),
        units_completed=recovery_receipt.DEFAULT_EXPECTED_UNITS,
        selector_records=recovery_receipt.REQUIRED_SELECTOR_RECORDS,
        scientific_seconds=RECOVERY_STAGE_SECONDS,
        live_bytes=0,
    )
    receipt = recovery_receipt.build_receipt(
        summary=summary,
        stage_manifest_sha256="b" * 64,
        scientific_seconds=RECOVERY_RECEIPT_SECONDS,
    )
    return summary, receipt


def _legacy_receipt():
    stage = 120.0
    return {
        "scientific_seconds_this_stage": stage,
        "scientific_seconds_cumulative_bound": stage + freeze.PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "maximum_total_seconds": freeze.MAXIMUM_TOTAL_SECONDS,
    }


def _forbidden(label):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{label} must not be called")

    return _fail


def _wire_proof_boundary(monkeypatch, tmp_path, *, receipt, summary, proof):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    run_root = artifact / freeze.NAMESPACE / "runs" / PERMIT
    paths = {
        "run_root": run_root,
        "develop": run_root / "develop",
        "receipt": run_root / "development_receipt.json",
        "selection": run_root / "selection",
        "selection_receipt": run_root / "selection_receipt.json",
    }
    bundle = {
        "project_root": tmp_path,
        "artifact_root": artifact,
        "repository_root": tmp_path,
        "permit_sha256": PERMIT,
    }
    payloads = {
        paths["receipt"]: receipt,
        paths["develop"] / "summary.json": summary,
    }
    monkeypatch.setattr(inputs, "prepare", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(freeze, "_paths", lambda bundle: paths)
    monkeypatch.setattr(freeze, "_expected_new_units", lambda bundle: 1242)
    monkeypatch.setattr(freeze, "_read_mapping", lambda path, code: payloads[path])
    monkeypatch.setattr(freeze, "_check_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze, "_check_develop_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze, "_check_prior_bound", lambda receipt: 39610.0)
    monkeypatch.setattr(acceptance, "authenticate_completed_source", proof)
    return bundle, paths


def _invoke_freeze(tmp_path, artifact_root):
    return freeze.freeze_selection(
        project_root=tmp_path,
        artifact_root=artifact_root,
        contract_path=tmp_path / "contract.json",
        permit_path=tmp_path / "permit.json",
    )


def test_base_payload_legacy_keys_unchanged():
    payload = freeze._base_payload(IDENTITY_BUNDLE)
    assert "source_execution_accounting" not in payload
    assert payload["fits_started"] == 0


def test_base_payload_includes_validated_accounting():
    accounting = source.clean_accounting(0)
    bundle = {**IDENTITY_BUNDLE, "source_execution_accounting": accounting}
    payload = freeze._base_payload(bundle)
    assert payload["source_execution_accounting"] == accounting
    assert set(payload) == set(freeze._base_payload(IDENTITY_BUNDLE)) | {
        "source_execution_accounting"
    }


def test_base_payload_rejects_malformed_accounting():
    with pytest.raises(freeze.FreezeSelectionError) as exc:
        freeze._base_payload({**IDENTITY_BUNDLE, "source_execution_accounting": 5})
    assert exc.value.reason_code == "source_accounting_malformed"


def test_base_payload_rejects_invalid_accounting():
    accounting = dict(source.clean_accounting(0), mode="bogus")
    with pytest.raises(freeze.FreezeSelectionError) as exc:
        freeze._base_payload({**IDENTITY_BUNDLE, "source_execution_accounting": accounting})
    assert exc.value.reason_code == "source_accounting_invalid"


def test_selection_summary_carries_accounting():
    accounting = source.clean_accounting(0)
    bundle = {**IDENTITY_BUNDLE, "source_execution_accounting": accounting}
    plan = {
        "decisions": [{}],
        "counts": {"strategy_alias_count": 1, "unique_refit_count": 1},
        "plan_id": "a" * 64,
    }
    summary = freeze._selection_summary(bundle, plan, 10.0, 1.0)
    assert summary["source_execution_accounting"] == accounting


def test_check_receipt_accepts_recovered_receipt():
    _summary, receipt = _recovered_pair()
    freeze._check_receipt(receipt, recovery_receipt.DEFAULT_EXPECTED_UNITS)


def test_check_develop_summary_accepts_recovered_summary():
    summary, _receipt = _recovered_pair()
    freeze._check_develop_summary(summary, recovery_receipt.DEFAULT_EXPECTED_UNITS)


def test_check_prior_bound_recovered_includes_prior_bound():
    _summary, receipt = _recovered_pair()
    prior = freeze._check_prior_bound(receipt)
    expected = (
        float(
            recovery_receipt.PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND
            + recovery_receipt.PRELAUNCH_AUDIT_RESERVE_SECONDS
        )
        + RECOVERY_RECEIPT_SECONDS
    )
    assert prior == pytest.approx(expected)
    assert prior > float(recovery_receipt.PRELAUNCH_AUDIT_RESERVE_SECONDS)


def test_check_prior_bound_legacy_unchanged():
    prior = freeze._check_prior_bound(_legacy_receipt())
    assert prior == pytest.approx(120.0 + freeze.PRELAUNCH_AUDIT_RESERVE_SECONDS)


def test_check_source_optimizer_steps_recovered_accounting():
    summary, receipt = _recovered_pair()
    steps = authority._check_source_optimizer_steps(summary, receipt)
    accounting = source.accounting_from_recovered(summary, receipt)
    assert steps == accounting["source_optimizer_steps_successful_exact"]
    assert steps == recovery_receipt.REUSED_ORIGINAL_OPTIMIZER_STEPS + NEW_RECOVERY_STEPS


def test_check_source_optimizer_steps_rejects_mixed_flavor():
    summary, _receipt = _recovered_pair()
    with pytest.raises(authority.RefitAuthorityError) as exc:
        authority._check_source_optimizer_steps(summary, {"optimizer_steps": 5})
    assert exc.value.reason_code == "develop_recovery_flavor_mismatch"


def test_check_source_optimizer_steps_rejects_malformed_pair():
    summary, receipt = _recovered_pair()
    broken = dict(receipt, unexpected=1)
    with pytest.raises(authority.RefitAuthorityError) as exc:
        authority._check_source_optimizer_steps(summary, broken)
    assert exc.value.reason_code == "develop_recovery_pair_invalid"


def test_check_source_optimizer_steps_legacy_unchanged():
    summary = {
        "optimizer_steps": 5,
        "maximum_optimizer_steps": development.MAXIMUM_FIT_STEPS,
    }
    assert authority._check_source_optimizer_steps(summary, {"optimizer_steps": 5}) == 5


def test_freeze_calls_completed_proof_before_selection_write(monkeypatch, tmp_path):
    recovered_receipt = {"recovery_permit_sha256": "f" * 64}
    recovered_summary = {"recovery_permit_sha256": "f" * 64}
    calls: dict[str, object] = {}

    def proof(*args, **kwargs):
        calls["bundle"] = args[0]
        calls["kwargs"] = kwargs
        calls["prewrite"] = (
            not kwargs["paths"]["selection"].exists()
            and not kwargs["paths"]["selection_receipt"].exists()
        )
        raise RuntimeError("stop-before-full-verify")

    bundle, paths = _wire_proof_boundary(
        monkeypatch,
        tmp_path,
        receipt=recovered_receipt,
        summary=recovered_summary,
        proof=proof,
    )
    with pytest.raises(freeze.FreezeSelectionError) as exc:
        _invoke_freeze(tmp_path, bundle["artifact_root"])
    assert exc.value.reason_code == "source_acceptance_failed"
    assert calls["prewrite"] is True
    assert calls["bundle"] is bundle
    assert "source_execution_accounting" not in calls["bundle"]
    assert calls["kwargs"]["paths"] is paths
    assert calls["kwargs"]["summary"] is recovered_summary
    assert calls["kwargs"]["receipt_record"] is recovered_receipt
    assert isinstance(calls["kwargs"]["deadline"], float)


def test_freeze_rejects_mixed_source_flavors(monkeypatch, tmp_path):
    recovered_receipt = {"recovery_permit_sha256": "f" * 64}
    legacy_summary = {"status": "complete"}
    bundle, paths = _wire_proof_boundary(
        monkeypatch,
        tmp_path,
        receipt=recovered_receipt,
        summary=legacy_summary,
        proof=_forbidden("acceptance.authenticate_completed_source"),
    )
    with pytest.raises(freeze.FreezeSelectionError) as exc:
        _invoke_freeze(tmp_path, bundle["artifact_root"])
    assert exc.value.reason_code == "source_flavor_mismatch"
    assert not paths["selection"].exists()
