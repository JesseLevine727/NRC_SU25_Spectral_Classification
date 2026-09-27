"""Independent regressions for the recovered-source read-only proof boundary."""

import time

import pytest

from atlas_sers.evaluation import p05_recovery_acceptance as accept
from atlas_sers.evaluation import p05_recovery_inputs as inputs
from tests.test_p05_recovery_inputs import _build
from tests.test_p05_recovery_source import _recovered_pair


def test_receipt_validation_reaches_path_gate_with_real_api(monkeypatch):
    summary, receipt = _recovered_pair()

    def reached(*args):
        raise accept.RecoveryAcceptanceError("paths_reached")

    monkeypatch.setattr(accept, "_check_paths", reached)
    with pytest.raises(accept.RecoveryAcceptanceError, match="paths_reached"):
        accept.authenticate_completed_source(
            {}, paths={}, summary=summary, receipt_record=receipt, deadline=time.perf_counter() + 30
        )


def test_compact_anchor_indirectly_binds_sealed_unit_files(tmp_path, monkeypatch):
    fx = _build(tmp_path, monkeypatch)
    proof = inputs.authenticate_original_stage(fx["bundle"], fx["permit"], time.perf_counter() + 30)
    assert len(proof["original_inventory"]) > len(proof["original_anchor"]["files"])
    accept._check_original_inventory(
        fx["bundle"],
        fx["plan"],
        fx["stage"],
        proof["original_inventory"],
        proof["original_anchor"],
        {"A"},
        time.perf_counter() + 30,
    )


def test_unstarted_units_are_valid_completed_recovery_units(tmp_path):
    unit = {"unit_id": "synthetic-new", "station": "cwa"}
    slots = [
        {
            "unit_id": unit["unit_id"],
            "slot_id": f"synthetic-{recipe}-{seed}",
            "recipe_id": recipe,
            "seed": seed,
        }
        for recipe in accept.recovery_plan.RECIPES
        for seed in accept.recovery_plan.SEEDS
    ]
    bundle = {"ledger": {"units": [unit], "slots": slots}, "pilot_bundle": {"slots": []}}
    stage = tmp_path / "develop"
    unit_dir = stage / "units" / unit["unit_id"]
    for slot in slots:
        for relative in inputs._slot_files(unit, slot):
            path = unit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque synthetic bytes")
    accept.core._write_manifest(unit_dir)
    inventory = {
        unit["unit_id"]: inputs._hash_file_record(
            unit_dir / "manifest.json", time.perf_counter() + 30
        )
    }
    plan = {
        "sealed_unit_ids": [],
        "incomplete_unit_id": "synthetic-partial",
        "reused_original_slot_ids": [],
    }
    accept._check_recovered_units(stage, bundle, plan, {}, inventory, time.perf_counter() + 30)


def test_completed_acceptance_must_use_existing_source_ledger_helper():
    assert hasattr(accept.recovery_inputs, "_expected_source_ledger")
    assert not hasattr(accept.base_inputs, "_expected_source_ledger")


def test_ledger_duplicate_cannot_silently_collapse():
    with pytest.raises(accept.RecoveryAcceptanceError, match="ledger_unit_duplicate"):
        accept._index({"ledger": {"units": [{"unit_id": "u"}, {"unit_id": "u"}], "slots": []}})


def test_ledger_parent_path_rejected_before_any_read():
    with pytest.raises(accept.RecoveryAcceptanceError, match="unit_identity_invalid"):
        accept._index({"ledger": {"units": [{"unit_id": "../u"}], "slots": []}})
