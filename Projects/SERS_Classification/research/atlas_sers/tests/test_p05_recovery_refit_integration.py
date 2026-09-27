"""Cross-stage recovery/refit integration tests for P05.

CPU-only. They reuse the synthetic refit world from
``test_p05_comprehensive_refits`` and the synthetic evaluation stage from
``test_p05_evaluation_authority``, then thread the recovered
source-execution accounting through the refit runner, the evidence logit
loader and the evaluation authority. No scientific model is fitted.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
# Optional torch is needed by the reused fixtures below.
# ruff: noqa: E402, F811

from atlas_sers.evaluation import p05_comprehensive_refits as runner
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_recovery_authority as recovery_authority
from atlas_sers.evaluation import p05_recovery_source as source
from atlas_sers.evaluation import p05_recovery_unit as recovery_unit
from atlas_sers.evaluation import p05_refit_authority as authority
from atlas_sers.evaluation import p05_refit_evidence as evidence
from tests.test_p05_comprehensive_refits import world  # noqa: F401
from tests.test_p05_evaluation_authority import world as evaluation_world  # noqa: F401

RECOVERED_STEPS = (
    source.RECOVERED_REUSED_SUCCESSFUL_STEPS
    + source.RECOVERED_NEW_FIT_EXECUTIONS * source.RECOVERY_MINIMUM_STEPS_PER_NEW_FIT
)


def _recovered_accounting() -> dict:
    return source.validate_accounting(
        {
            "schema_version": source.ACCOUNTING_SCHEMA_VERSION,
            "mode": source.RECOVERY_ACCOUNTING_MODE,
            "source_successful_fits": source.RECOVERED_SOURCE_SUCCESSFUL_FITS,
            "source_attempts": source.RECOVERED_SOURCE_ATTEMPTS,
            "source_interrupted_attempts": source.RECOVERED_SOURCE_INTERRUPTED_ATTEMPTS,
            "source_optimizer_steps_successful_exact": RECOVERED_STEPS,
            "source_optimizer_steps_observed_lower_bound": (
                RECOVERED_STEPS + source.INTERRUPTED_OBSERVED_STEPS
            ),
            "source_optimizer_steps_charged_upper_bound": (
                RECOVERED_STEPS + source.INTERRUPTED_CHARGED_STEPS
            ),
            "source_optimizer_steps_all_attempts_exact": False,
            "maximum_source_optimizer_steps": source.RECOVERED_MAXIMUM_SOURCE_OPTIMIZER_STEPS,
            "maximum_new_neural_executions": source.RECOVERED_MAXIMUM_NEW_NEURAL_EXECUTIONS,
            "maximum_new_optimizer_steps": source.RECOVERED_MAXIMUM_NEW_OPTIMIZER_STEPS,
            "recovery_permit_sha256": recovery_authority.RECOVERY_PERMIT_SHA256,
        },
        source_optimizer_steps=RECOVERED_STEPS,
    )


def _stage_path(case) -> Path:
    run_root = runner._run_root(case.artifact_root, case.bundle["permit_sha256"])
    return run_root / runner.STAGE_NAME


@pytest.fixture
def recovered_world(world, monkeypatch):
    accounting = _recovered_accounting()
    phases: list[str] = []
    recovery_prior = 39620.0
    world.set_prior(recovery_prior)

    def recovered_auth(bundle, deadline):
        return {
            "prior_seconds": recovery_prior,
            "plan": world.plan,
            "source_optimizer_steps": RECOVERED_STEPS,
            "source_execution_accounting": dict(accounting),
        }

    monkeypatch.setattr(authority, "authenticate_selection", recovered_auth)
    monkeypatch.setattr(
        recovery_authority,
        "check_resources",
        lambda torch_module, *, phase: phases.append(phase),
    )
    world.accounting = accounting
    world.phases = phases
    return world


def test_recovered_receipt_carries_honest_accounting(recovered_world):
    receipt = runner.run_refits(**recovered_world.run_kwargs())
    accounting = recovered_world.accounting
    refit_steps = receipt["counters"]["optimizer_steps"]

    assert receipt["status"] == "complete"
    assert receipt["source_execution_accounting"] == accounting
    assert receipt["source_optimizer_steps"] == RECOVERED_STEPS
    assert receipt["total_new_optimizer_steps"] == RECOVERED_STEPS + refit_steps
    assert (
        receipt["total_new_optimizer_steps_charged_upper_bound"]
        == RECOVERED_STEPS + source.INTERRUPTED_CHARGED_STEPS + refit_steps
    )
    assert receipt["total_new_optimizer_steps_all_attempts_exact"] is False

    summary = core._read_json(_stage_path(recovered_world) / "summary.json", "summary")
    assert summary["source_execution_accounting"] == accounting
    assert (
        summary["total_new_optimizer_steps_charged_upper_bound"]
        == RECOVERED_STEPS + source.INTERRUPTED_CHARGED_STEPS + refit_steps
    )
    assert summary["total_new_optimizer_steps_all_attempts_exact"] is False


def test_recovered_guards_run_before_units_and_each_epoch(recovered_world):
    runner.run_refits(**recovered_world.run_kwargs())
    phases = recovered_world.phases
    epochs = int(recovered_world.specs[0]["epochs"])
    fits = len(recovered_world.specs)

    assert phases[0] == "launch"
    assert phases.count("epoch") >= fits * epochs
    assert phases.index("fit") < phases.index("epoch")
    assert phases[-1] == "fit"


def test_recovered_stops_when_source_attempt_quota_exhausted(recovered_world, monkeypatch):
    real = source.from_authenticated

    def tightened(auth):
        account = dict(real(auth))
        account["maximum_new_neural_executions"] = account["source_attempts"] + 2
        return account

    monkeypatch.setattr(source, "from_authenticated", tightened)
    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**recovered_world.run_kwargs())

    assert info.value.reason_code == "source_attempts_exceeded"
    assert len(recovered_world.record["train"]) == 2


def test_legacy_refit_receipt_has_no_recovery_fields(world):
    receipt = runner.run_refits(**world.run_kwargs())
    assert "source_execution_accounting" not in receipt
    assert "total_new_optimizer_steps_charged_upper_bound" not in receipt
    assert "total_new_optimizer_steps_all_attempts_exact" not in receipt
    assert receipt["total_new_optimizer_steps"] == receipt["source_optimizer_steps"] + 720


def _recovered_evaluation_auth(case, accounting):
    def authenticate(bundle, *, deadline):
        return {
            "plan": case.plan,
            "prior_seconds": 39620.0,
            "source_optimizer_steps": RECOVERED_STEPS,
            "selection_receipt": {"receipt_id": "selection"},
            "source_execution_accounting": dict(accounting),
        }

    return authenticate


def _rewrite(path, payload):
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))


def test_evaluation_authority_rejects_missing_recovery_accounting(evaluation_world, monkeypatch):
    accounting = _recovered_accounting()
    monkeypatch.setattr(
        authority,
        "authenticate_selection",
        _recovered_evaluation_auth(evaluation_world, accounting),
    )
    receipt = core._read_json(evaluation_world.receipt_path, "receipt")
    receipt["source_optimizer_steps"] = RECOVERED_STEPS
    _rewrite(evaluation_world.receipt_path, receipt)

    with pytest.raises(core.P05CoreError) as info:
        evaluation_world.call()
    assert info.value.reason_code == "receipt_source_execution_accounting_mismatch"


def test_evaluation_authority_rejects_forged_legacy_accounting(evaluation_world):
    receipt = core._read_json(evaluation_world.receipt_path, "receipt")
    receipt["source_execution_accounting"] = _recovered_accounting()
    _rewrite(evaluation_world.receipt_path, receipt)

    with pytest.raises(core.P05CoreError) as info:
        evaluation_world.call()
    assert info.value.reason_code == "receipt_recovery_accounting_forbidden"


def _loader_bundle(root) -> dict:
    unit = {
        "unit_id": "u1",
        "context_id": "c1",
        "selection_unit_id": "su1",
        "fitting_role_id": "fit",
        "validation_role_id": "val",
    }
    slots = [{**unit, "slot_id": slot_id} for slot_id in ("p1", "d1")]
    return {
        "artifact_root": str(root),
        "permit_sha256": "a" * 64,
        "ledger": {"units": [unit], "slots": slots},
    }


def test_logit_loader_uses_resolved_develop_and_pilot_root(tmp_path, monkeypatch):
    develop = tmp_path / "run" / "recoveries" / "develop"
    pilot_root = tmp_path / "p05pilot" / "runs" / "pilot"
    loaded: list[Path] = []

    inputs_module = types.ModuleType("atlas_sers.evaluation.p05_comprehensive_inputs")
    inputs_module.pilot_slot_ids = lambda bundle: ["p1"]
    inputs_module._pilot_run_dir = lambda artifact_root: pilot_root
    inputs_module._load_logits = lambda numpy, path: (
        loaded.append(Path(path)) or {"path": str(path)}
    )
    pilot_module = types.ModuleType("atlas_sers.evaluation.p05_pilot")
    pilot_module.execution_id = lambda unit, slot: f"exec-{unit['unit_id']}-{slot['slot_id']}"
    source_module = types.ModuleType("atlas_sers.evaluation.p05_recovery_source")
    source_module.resolve_paths = lambda bundle: {"develop": develop}
    source_module.RecoverySourceError = source.RecoverySourceError

    monkeypatch.setitem(
        sys.modules, "atlas_sers.evaluation.p05_comprehensive_inputs", inputs_module
    )
    monkeypatch.setitem(sys.modules, "atlas_sers.evaluation.p05_pilot", pilot_module)
    monkeypatch.setitem(sys.modules, "atlas_sers.evaluation.p05_recovery_source", source_module)

    bundle = _loader_bundle(tmp_path)
    loader = evidence.make_logit_loader(bundle)
    loader(bundle["ledger"]["slots"][0])
    loader(bundle["ledger"]["slots"][1])

    assert loaded[0] == pilot_root / "executions" / "exec-u1-p1" / "validation_logits.npz"
    assert (
        loaded[1]
        == develop / "units" / "u1" / "executions" / "exec-u1-d1" / "validation_logits.npz"
    )


def _install_complete_recovered_evaluation(case, monkeypatch):
    accounting = _recovered_accounting()
    monkeypatch.setattr(
        authority, "authenticate_selection", _recovered_evaluation_auth(case, accounting)
    )
    for path in (case.receipt_path, case.summary_path):
        payload = core._read_json(path, "synthetic")
        payload["source_optimizer_steps"] = RECOVERED_STEPS
        payload["source_execution_accounting"] = accounting
        total = RECOVERED_STEPS + payload["counters"]["optimizer_steps"]
        payload["total_new_optimizer_steps"] = total
        payload["total_new_optimizer_steps_charged_upper_bound"] = total + 800
        payload["total_new_optimizer_steps_all_attempts_exact"] = False
        payload["prior_scientific_seconds_cumulative_bound"] = 39620.0
        payload["scientific_seconds_cumulative_bound"] = (
            39620.0 + payload["scientific_seconds_this_stage"]
        )
        _rewrite(path, payload)
    case.refresh()
    return accounting


def test_recovered_evaluation_reauthenticates_and_propagates_accounting(
    evaluation_world, monkeypatch
):
    accounting = _install_complete_recovered_evaluation(evaluation_world, monkeypatch)
    result = evaluation_world.call()
    assert result["source_execution_accounting"] == accounting
    assert result["source_optimizer_steps"] == RECOVERED_STEPS


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_interrupted_attempts", True),
        ("source_optimizer_steps_charged_upper_bound", float(RECOVERED_STEPS + 800)),
    ],
)
def test_recovered_accounting_numeric_equality_does_not_override_strict_types(
    evaluation_world, monkeypatch, field, value
):
    _install_complete_recovered_evaluation(evaluation_world, monkeypatch)
    for path in (evaluation_world.receipt_path, evaluation_world.summary_path):
        payload = core._read_json(path, "synthetic")
        payload["source_execution_accounting"][field] = value
        _rewrite(path, payload)
    evaluation_world.refresh()
    with pytest.raises(core.P05CoreError, match="receipt_source_accounting_invalid"):
        evaluation_world.call()


def test_recovery_host_guard_failure_prevents_stage_creation(recovered_world, monkeypatch):
    def fail(torch, *, phase):
        raise recovery_authority.RecoveryAuthorityError("insufficient_host_memory")

    monkeypatch.setattr(recovery_authority, "check_resources", fail)
    with pytest.raises(recovery_authority.RecoveryAuthorityError):
        runner.run_refits(**recovered_world.run_kwargs())
    assert not _stage_path(recovered_world).exists()
    assert recovered_world.record["train"] == []


def test_recovery_epoch_guard_preserves_partial_history_and_stops(recovered_world, monkeypatch):
    def fail_epoch(torch, *, phase):
        if phase == "epoch":
            raise recovery_authority.RecoveryAuthorityError("insufficient_host_memory")

    monkeypatch.setattr(recovery_authority, "check_resources", fail_epoch)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        runner.run_refits(**recovered_world.run_kwargs())
    assert len(recovered_world.record["train"]) == 1
    summary = core._read_json(_stage_path(recovered_world) / "summary.json", "synthetic")
    assert summary["status"] == "fail"
    assert summary["counters"]["optimizer_steps"] == 4
    assert summary["counters"]["optimizer_steps_exact"] is False
