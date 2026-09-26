"""Tests for the P05 comprehensive input and pilot-import boundary."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_pilot

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PERMIT_SHA256 = "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8"


def _torch():
    return pytest.importorskip("torch")


def _runtime():
    return pytest.importorskip("atlas_sers.evaluation.p04_runtime")


def _authorization(**overrides):
    auth = {
        "schema_version": inputs.COMPREHENSIVE_SCHEMA_VERSION,
        "core_contract_sha256": inputs.CORE_CONTRACT_SHA256,
        "core_plan_id": inputs.CORE_PLAN_ID,
        "ledger_id": inputs.LEDGER_ID,
        "pilot_permit_sha256": inputs.PILOT_PERMIT_SHA256,
        "pilot_plan_id": inputs.PILOT_PLAN_ID,
        "pilot_manifest_sha256": inputs.PILOT_MANIFEST_SHA256,
        "frozen_numerical_files": {},
    }
    auth.update(overrides)
    return auth


# --------------------------------------------------------------------------- #
# Permit and authorization pins
# --------------------------------------------------------------------------- #


def test_comprehensive_permit_pin_is_canonical():
    assert inputs.COMPREHENSIVE_PERMIT_SHA256 == _PERMIT_SHA256
    path = _REPO_ROOT / "plan/contracts/p05_comprehensive.json"
    assert inputs._canon().sha256_value(json.loads(path.read_text())) == _PERMIT_SHA256


def test_load_permit_accepts_canonical_and_rejects_mutation(monkeypatch, tmp_path):
    permit = _authorization()
    digest = inputs._canon().sha256_value(permit)
    monkeypatch.setattr(inputs, "COMPREHENSIVE_PERMIT_SHA256", digest)
    path = tmp_path / "permit.json"
    path.write_bytes(inputs._canon().canonical_json_bytes(permit))
    loaded, observed = inputs._load_permit(path)
    assert observed == digest and loaded == permit
    mutated = dict(permit)
    mutated["context_count"] = 321
    bad = tmp_path / "mutated.json"
    bad.write_bytes(inputs._canon().canonical_json_bytes(mutated))
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._load_permit(bad)
    assert exc.value.reason_code == "permit_digest_mismatch"


def test_authorization_pins_and_schema_are_enforced():
    inputs._check_authorization_pins(_authorization())
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_authorization_pins(_authorization(schema_version="other"))
    assert exc.value.reason_code == "authorization_schema_mismatch"
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_authorization_pins(_authorization(core_plan_id="0" * 64))
    assert exc.value.reason_code == "authorization_pin_mismatch"


# --------------------------------------------------------------------------- #
# Frozen numerical file pins and safety
# --------------------------------------------------------------------------- #


def test_frozen_numerical_file_pins_accept_then_reject_mutation(tmp_path):
    relative = "src/atlas_sers/models/deep.py"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"frozen\n")
    auth = _authorization(frozen_numerical_files={relative: inputs._canon().sha256_file(target)})
    inputs._check_frozen_files(tmp_path, auth)
    target.write_bytes(b"tampered\n")
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_frozen_files(tmp_path, auth)
    assert exc.value.reason_code == "frozen_numerical_file_mismatch"


def test_frozen_numerical_file_missing_and_empty_mapping(tmp_path):
    auth = _authorization(frozen_numerical_files={"src/missing.py": "0" * 64})
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_frozen_files(tmp_path, auth)
    assert exc.value.reason_code == "frozen_numerical_file_missing"
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_frozen_files(tmp_path, _authorization(frozen_numerical_files={}))
    assert exc.value.reason_code == "frozen_numerical_files_malformed"


def test_frozen_numerical_file_symlink_is_refused(tmp_path):
    real = tmp_path / "real.py"
    real.write_bytes(b"data\n")
    link = tmp_path / "link.py"
    link.symlink_to(real)
    auth = _authorization(frozen_numerical_files={"link.py": inputs._canon().sha256_file(real)})
    with pytest.raises(inputs.core.P05CoreError) as exc:
        inputs._check_frozen_files(tmp_path, auth)
    assert exc.value.reason_code == "symlink_path_rejected"


# --------------------------------------------------------------------------- #
# Selector records
# --------------------------------------------------------------------------- #


def _selector_unit():
    return {
        "unit_id": "unit-1",
        "context_id": "ctx-1",
        "selection_unit_id": "sel-1",
        "fitting_role_id": "role-fit",
        "validation_role_id": "role-val",
    }


def _selector_slot():
    return {
        "slot_id": "slot-1",
        "slot_kind": "inherited_selection_fit",
        "recipe_id": "D0-M",
        "seed": 20260805,
    }


def _selector_summary(status="complete", **overrides):
    summary = {
        "status": status,
        "slot_id": "slot-1",
        "context_id": "ctx-1",
        "selection_unit_id": "sel-1",
        "slot_kind": "inherited_selection_fit",
        "fitting_role_id": "role-fit",
        "validation_role_id": "role-val",
        "recipe_id": "D0-M",
        "seed": 20260805,
    }
    if status == "complete":
        summary.update(
            {
                "best_epoch": 1,
                "best_validation_balanced_accuracy": 0.5,
                "best_validation_nll": 1.0,
                "best_validation_macro_f1": 0.5,
                "best_validation_predicted_class_count": 3,
            }
        )
    summary.update(overrides)
    return summary


def test_selector_record_complete_preserves_identity_and_metrics():
    record = inputs.selector_record(_selector_unit(), _selector_slot(), _selector_summary())
    assert record["status"] == "complete"
    assert record["context_id"] == "ctx-1" and record["recipe_id"] == "D0-M"
    assert record["best_validation_balanced_accuracy"] == 0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("slot_id", "slot-other"),
        ("context_id", "ctx-other"),
        ("selection_unit_id", "sel-other"),
        ("slot_kind", "guard_selection_fit"),
        ("fitting_role_id", "role-other"),
        ("validation_role_id", "role-other"),
        ("recipe_id", "D1"),
        ("seed", 20260807),
    ],
)
def test_selector_identity_mismatch_is_rejected(field, value):
    summary = _selector_summary(**{field: value})
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs.selector_record(_selector_unit(), _selector_slot(), summary)
    assert exc.value.reason_code == "selector_identity_mismatch"


def test_selector_slot_unit_identity_mismatch_is_rejected():
    slot = _selector_slot()
    slot["context_id"] = "ctx-other"
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs.selector_record(_selector_unit(), slot, _selector_summary())
    assert exc.value.reason_code == "selector_slot_unit_mismatch"


def test_selector_complete_metric_omission_is_rejected():
    summary = _selector_summary()
    del summary["best_validation_nll"]
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs.selector_record(_selector_unit(), _selector_slot(), summary)
    assert exc.value.reason_code == "selector_metric_missing"


def test_selector_preserves_collapsed_complete_record():
    summary = _selector_summary(best_validation_predicted_class_count=1)
    record = inputs.selector_record(_selector_unit(), _selector_slot(), summary)
    assert record["status"] == "complete"
    assert record["best_validation_predicted_class_count"] == 1


# --------------------------------------------------------------------------- #
# Persisted development-result round trip
# --------------------------------------------------------------------------- #


def _round_trip_result(runtime, torch):
    module = pytest.importorskip("atlas_sers.evaluation.p05_development")
    names = [field.name for field in dataclasses.fields(module.DevelopmentFitResult)]
    fake = dataclasses.make_dataclass(
        "_RoundTripResult",
        [(name, object, dataclasses.field(default=None)) for name in names],
    )()
    best = {"backbone.0.weight": torch.zeros(2, dtype=torch.float32)}
    terminal = {"backbone.0.weight": torch.ones(2, dtype=torch.float32)}
    history = [
        {
            "epoch": epoch,
            "epoch_optimizer_steps": 4,
            "total_optimizer_steps": epoch * 4,
            "train_balanced_accuracy": 0.6,
            "validation_balanced_accuracy": 0.5,
            "validation_nll": 1.0,
            "validation_macro_f1": 0.5,
            "validation_predicted_class_count": 3,
            "sampling_digest": "s" * 64,
            "augmentation_digest": "a" * 64,
            "pair_digest": "p" * 64,
        }
        for epoch in range(1, 4)
    ]
    values = {
        "status": "complete",
        "reason_code": None,
        "history": history,
        "epochs_completed": 3,
        "parameter_count": 100,
        "optimizer_steps": 12,
        "best_epoch": 1,
        "best_validation_balanced_accuracy": 0.5,
        "best_validation_nll": 1.0,
        "best_validation_macro_f1": 0.5,
        "best_validation_predicted_class_count": 3,
        "best_training_balanced_accuracy": 0.6,
        "initial_state_digest": runtime._state_hash(best),
        "best_state_digest": runtime._state_hash(best),
        "terminal_state_digest": runtime._state_hash(terminal),
        "initial_backbone_digest": "i" * 64,
        "best_state_dict": best,
        "terminal_state_dict": terminal,
        "state_dict": best,
        "classes": ("A", "B", "C"),
        "validation_uids": ("v-a", "v-b", "v-c"),
        "validation_logits": np.zeros((3, 3), dtype=np.float64),
        "sampling_digest": "s" * 64,
        "augmentation_digest": "a" * 64,
        "pair_digest": "p" * 64,
        "finite_gradient_batches": 12,
        "paired_support": {"enabled": 1, "available_batches": 0, "pairs": 0},
        "role_id": "role-1",
        "recipe": "D0-M",
        "seed": 20260805,
        "elapsed_seconds": 0.5,
        "peak_cuda_bytes": 0,
    }
    for name, value in values.items():
        if hasattr(fake, name):
            setattr(fake, name, value)
    return fake


_UNIT = {"unit_id": "unit-1", "station": "cwa"}
_SLOT = {"slot_id": "slot-1", "recipe_id": "D0-M", "seed": 20260805}


def _persist(tmp_path, torch, runtime):
    result = _round_trip_result(runtime, torch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    p05_pilot.persist_result(torch, run_dir, _UNIT, _SLOT, result)
    return result, run_dir


def test_load_development_result_round_trips_state_npz_and_history(tmp_path):
    torch = _torch()
    result, run_dir = _persist(tmp_path, torch, _runtime())
    reloaded = inputs.load_development_result(run_dir, _UNIT, _SLOT)
    assert np.array_equal(reloaded.validation_logits, result.validation_logits)
    assert tuple(reloaded.validation_uids) == tuple(result.validation_uids)
    assert tuple(reloaded.classes) == tuple(result.classes)
    assert reloaded.history == result.history
    for name in ("best_state_dict", "terminal_state_dict"):
        original, observed = getattr(result, name), getattr(reloaded, name)
        assert set(observed) == set(original)
        assert all(torch.equal(observed[key], original[key]) for key in original)


def test_history_jsonl_must_match_reloaded_history(tmp_path):
    torch = _torch()
    result, run_dir = _persist(tmp_path, torch, _runtime())
    identifier = p05_pilot.execution_id(_UNIT, _SLOT)
    history_dir = run_dir / "histories"
    history_dir.mkdir(parents=True)
    path = history_dir / f"{identifier}.jsonl"
    path.write_bytes(
        b"".join(inputs._canon().canonical_json_bytes(rec) + b"\n" for rec in result.history)
    )
    inputs._check_history_matches(run_dir, _UNIT, _SLOT, result)
    path.write_bytes(b'{"epoch": 99}\n')
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_history_matches(run_dir, _UNIT, _SLOT, result)
    assert exc.value.reason_code == "pilot_history_length_mismatch"


def test_damaged_terminal_checkpoint_rejected_even_with_best_logits(tmp_path):
    torch = _torch()
    result, run_dir = _persist(tmp_path, torch, _runtime())
    directory = run_dir / "executions" / p05_pilot.execution_id(_UNIT, _SLOT)
    damaged = {name: value + 5.0 for name, value in result.terminal_state_dict.items()}
    p05_pilot.core._save_state(torch, damaged, directory / "terminal.pt")
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs.load_development_result(run_dir, _UNIT, _SLOT)
    assert exc.value.reason_code == "pilot_checkpoint_digest_mismatch"


# --------------------------------------------------------------------------- #
# Pilot run manifest, summary and leases
# --------------------------------------------------------------------------- #


def _write_manifest(run_dir, payload):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_bytes(inputs._canon().canonical_json_bytes(payload))
    return inputs._canon().sha256_file(path)


def test_pilot_manifest_missing_and_raw_digest_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_pilot_run_manifest(run_dir, _authorization())
    assert exc.value.reason_code == "pilot_manifest_missing"
    _write_manifest(run_dir, {"files": {}})
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_pilot_run_manifest(run_dir, _authorization())
    assert exc.value.reason_code == "pilot_manifest_digest_mismatch"


def test_pilot_manifest_requires_new_permit_pin_not_core_contract(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    digest = _write_manifest(run_dir, {"files": {}})
    monkeypatch.setattr(inputs, "PILOT_MANIFEST_SHA256", digest)
    inputs._check_pilot_run_manifest(run_dir, {"pilot_manifest_sha256": digest})
    core_contract = {"core_contract_sha256": inputs.CORE_CONTRACT_SHA256}
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_pilot_run_manifest(run_dir, core_contract)
    assert exc.value.reason_code == "pilot_manifest_pin_mismatch"


def test_pilot_manifest_inventory_mismatch_is_rejected(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    digest = _write_manifest(run_dir, {"files": {"missing.bin": "0" * 64}})
    monkeypatch.setattr(inputs, "PILOT_MANIFEST_SHA256", digest)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        inputs._check_pilot_run_manifest(run_dir, {"pilot_manifest_sha256": digest})
    assert exc.value.reason_code == "manifest_integrity_mismatch"


def _lease_bundle(tmp_path, slots):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    return {"artifact_root": artifact, "pilot_bundle": {"slots": slots}}


def test_pilot_slot_lease_identity_validated(tmp_path):
    slot = {"slot_id": "slot-a", "unit_id": "unit-a", "recipe_id": "D0-M", "seed": 20260805}
    bundle = _lease_bundle(tmp_path, [slot])
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_pilot_slot_leases(bundle)
    assert exc.value.reason_code == "pilot_slot_lease_missing"
    p05_pilot._reserve_slot_lease(
        bundle["artifact_root"], inputs.CORE_CONTRACT_SHA256, inputs.CORE_PLAN_ID, slot
    )
    inputs._check_pilot_slot_leases(bundle)
    lease_path = inputs._slot_lease_root(bundle["artifact_root"]) / "slot-a" / "lease.json"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["unit_id"] = "unit-other"
    lease_path.write_bytes(inputs._canon().canonical_json_bytes(lease))
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._check_pilot_slot_leases(bundle)
    assert exc.value.reason_code == "pilot_slot_lease_mismatch"


def test_foreign_slot_lease_rejected_without_writes(tmp_path):
    bundle = _lease_bundle(tmp_path, [{"slot_id": "slot-a"}])
    root = inputs._slot_lease_root(bundle["artifact_root"])
    root.mkdir(parents=True)
    (root / "foreign-slot").mkdir()
    before = sorted(entry.name for entry in root.iterdir())
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs._assert_no_foreign_leases(bundle)
    assert exc.value.reason_code == "foreign_slot_lease_exists"
    assert sorted(entry.name for entry in root.iterdir()) == before
    assert inputs.pilot_slot_ids(bundle) == {"slot-a"}


class _StubCanon:
    def canonical_json_bytes(self, value):
        return b"{}"

    def sha256_bytes(self, value):
        return inputs.CORE_PLAN_ID

    def sha256_value(self, value):
        return inputs.COMPREHENSIVE_PERMIT_SHA256


def _stub_prepare(monkeypatch, ledger, pilot_bundle):
    monkeypatch.setattr(inputs, "_canon", lambda: _StubCanon())
    monkeypatch.setattr(inputs, "_load_permit", lambda p: ({}, inputs.COMPREHENSIVE_PERMIT_SHA256))
    monkeypatch.setattr(inputs, "_authorization", lambda permit, project: {})
    monkeypatch.setattr(inputs, "_check_authorization_pins", lambda auth: None)
    monkeypatch.setattr(inputs, "_check_frozen_files", lambda project, auth: None)
    monkeypatch.setattr(
        inputs.core,
        "_load_contract",
        lambda path, pin: ({"population": {"rows": 0}}, inputs.CORE_CONTRACT_SHA256),
    )
    monkeypatch.setattr(
        inputs.core, "_authenticate", lambda artifact, contract: (object(), None, None)
    )
    monkeypatch.setattr(inputs.core, "_manifest_uids", lambda support, rows: [])
    monkeypatch.setattr(inputs.core, "_build_plan", lambda support, contract, project: {})
    monkeypatch.setattr(inputs.core, "_minimal_plan_checks", lambda plan, contract: None)
    monkeypatch.setattr(inputs, "_build_ledger", lambda plan, support, contract: ledger)
    monkeypatch.setattr(inputs, "_check_ledger", lambda ledger, auth: None)
    monkeypatch.setattr(inputs.pilot, "prepare", lambda *a, **k: pilot_bundle)
    monkeypatch.setattr(inputs, "_check_pilot_bundle", lambda bundle, auth: None)
    monkeypatch.setattr(inputs.pilot, "_resolve_paths", lambda p, a: (Path(p), Path(a), Path(p)))


def test_prepare_require_unstarted_gates_foreign_leases(monkeypatch, tmp_path):
    ledger = {"units": [], "slots": [], "ledger_id": inputs.LEDGER_ID}
    pilot_bundle = {"slots": [{"slot_id": "slot-a"}]}
    _stub_prepare(monkeypatch, ledger, pilot_bundle)
    artifact = tmp_path / "artifacts"
    root = inputs._slot_lease_root(artifact)
    root.mkdir(parents=True)
    (root / "foreign").mkdir()
    with pytest.raises(inputs.ComprehensiveInputsError) as exc:
        inputs.prepare(tmp_path, artifact, tmp_path / "c.json", tmp_path / "p.json")
    assert exc.value.reason_code == "foreign_slot_lease_exists"
    bundle = inputs.prepare(
        tmp_path,
        artifact,
        tmp_path / "c.json",
        tmp_path / "p.json",
        require_unstarted=False,
    )
    assert bundle["pilot_bundle"] is pilot_bundle


def test_module_import_is_torch_lazy_in_fresh_subprocess():
    code = (
        "import sys\n"
        "import atlas_sers.evaluation.p05_comprehensive_inputs\n"
        "assert 'torch' not in sys.modules, 'torch imported eagerly'\n"
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT / "src") + (os.pathsep + existing if existing else "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
