"""Persistence tests for the P05 recovery prefix.

Synthetic temporary fixtures only: no real private data, models or training is
used.  Source artifacts are opaque bytes and the recovery permit is a
fixture-only synthetic authority object.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_inputs as inputs
from atlas_sers.evaluation import p05_recovery_persistence as persistence
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan
from atlas_sers.evaluation.p05_comprehensive_storage import (
    P05StorageError,
    StorageBudget,
)

RECIPES = ("D0-M", "D1", "D2", "D3")
SEEDS = (20260805, 20260817, 20260829)
UNIT_ID = "unit-0001"
FUTURE = float(2**31)
APPROVED_CEILING = 100 * 1024**3


def _canon_bytes(value):
    return core._canon().canonical_json_bytes(value)


def _canon_sha(value):
    return core._canon().sha256_value(value)


def _record_bytes(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _record(path):
    return _record_bytes(Path(path).read_bytes())


def _execution_id(unit, slot):
    return pilot.execution_id(unit, slot)


def _slot_files(unit, slot):
    return set(inputs._slot_files(unit, slot))


def _content(relative):
    return f"{UNIT_ID}|{relative}\n".encode()


def _tree_map(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def _build_env(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(authority, "validate_recovery_permit", lambda permit: None)

    artifact_root = tmp_path / "artifact"
    run_root = (
        artifact_root
        / inputs.COMPREHENSIVE_NAMESPACE
        / "runs"
        / authority.BASECOMPREHENSIVE_PERMIT_SHA256
    )
    stage = run_root / inputs.DEVELOP_STAGE_NAME
    unit_dir = stage / "units" / UNIT_ID
    unit = {
        "unit_id": UNIT_ID,
        "station": "STN-01",
        "instrument": "instrument-a",
        "auxiliary_support": {},
    }
    slots = [
        {
            "slot_id": f"{UNIT_ID}-{recipe}-{seed}",
            "unit_id": UNIT_ID,
            "recipe_id": recipe,
            "seed": seed,
            "station": "STN-01",
        }
        for recipe in RECIPES
        for seed in SEEDS
    ]
    if kind == "sealed":
        completed, interrupted = list(slots), None
    else:
        completed, interrupted = slots[:8], slots[8]

    for slot in completed:
        for relative in _slot_files(unit, slot):
            path = unit_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_content(relative))
    if interrupted is not None:
        relative = f"histories/{_execution_id(unit, interrupted)}.jsonl"
        path = unit_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_content(relative))
    if kind == "sealed":
        manifest_files = {
            str(path.relative_to(unit_dir)): _record(path)
            for path in sorted(unit_dir.rglob("*"))
            if path.is_file()
        }
        (unit_dir / "manifest.json").write_bytes(_canon_bytes({"files": manifest_files}))

    inventory = {
        f"units/{UNIT_ID}/{path.relative_to(unit_dir)}": _record(path)
        for path in sorted(unit_dir.rglob("*"))
        if path.is_file()
    }
    if kind == "sealed":
        anchor = {
            "files": {f"units/{UNIT_ID}/manifest.json": inventory[f"units/{UNIT_ID}/manifest.json"]}
        }
    else:
        anchor = {"files": dict(inventory)}

    lease_inventory = {}
    if interrupted is not None:
        lease_dir = base_inputs._slot_lease_root(artifact_root) / interrupted["slot_id"]
        lease_dir.mkdir(parents=True, exist_ok=True)
        lease_payload = {
            "slot_id": interrupted["slot_id"],
            "unit_id": UNIT_ID,
            "recipe_id": interrupted["recipe_id"],
            "seed": interrupted["seed"],
            "contract_sha256": base_inputs.CORE_CONTRACT_SHA256,
            "core_plan_id": base_inputs.CORE_PLAN_ID,
            "permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        }
        lease_bytes = _canon_bytes(lease_payload)
        (lease_dir / "lease.json").write_bytes(lease_bytes)
        lease_inventory[f"{interrupted['slot_id']}/lease.json"] = _record_bytes(lease_bytes)

    recovery_root = run_root / "recoveries" / authority.RECOVERY_PERMIT_SHA256
    new_develop = recovery_root / inputs.DEVELOP_STAGE_NAME
    new_develop.mkdir(parents=True, exist_ok=True)

    base_bundle = {
        "permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "artifact_root": str(artifact_root),
        "ledger": {"units": [dict(unit)], "slots": [dict(slot) for slot in slots]},
        "contract_sha256": base_inputs.CORE_CONTRACT_SHA256,
        "core_plan_id": base_inputs.CORE_PLAN_ID,
    }
    if kind == "sealed":
        plan = {
            "sealed_unit_ids": [UNIT_ID],
            "incomplete_unit_id": None,
            "reused_original_slot_ids": [slot["slot_id"] for slot in completed],
            "interrupted_slot_id": "another-unit-interrupted-slot",
        }
    else:
        plan = {
            "sealed_unit_ids": [],
            "incomplete_unit_id": UNIT_ID,
            "reused_original_slot_ids": [slot["slot_id"] for slot in completed],
            "interrupted_slot_id": interrupted["slot_id"],
        }
    anchor_sha = _canon_sha(anchor)
    monkeypatch.setattr(authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", anchor_sha)
    recovery_bundle = {
        "recovery_permit": {"synthetic_fixture": True},
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "base_bundle": base_bundle,
        "plan": plan,
        "original_run_root": str(run_root),
        "original_stage": str(stage),
        "original_inventory": inventory,
        "original_anchor": anchor,
        "original_anchor_sha256": anchor_sha,
        "original_lease_inventory": lease_inventory,
    }
    budget = StorageBudget(artifact_root, run_root)
    return {
        "artifact_root": artifact_root,
        "run_root": run_root,
        "stage": stage,
        "unit_dir": unit_dir,
        "recovery_root": recovery_root,
        "new_develop": new_develop,
        "unit": unit,
        "slots": slots,
        "interrupted": interrupted,
        "base_bundle": base_bundle,
        "plan": plan,
        "recovery_bundle": recovery_bundle,
        "budget": budget,
    }


def _copy(env, budget=None, deadline=FUTURE, unit_id=UNIT_ID, bundle=None):
    return persistence.copy_original_unit(
        recovery_bundle=env["recovery_bundle"] if bundle is None else bundle,
        unit_id=unit_id,
        budget=env["budget"] if budget is None else budget,
        deadline=deadline,
    )


def _reserve(env, slot=None, budget=None, deadline=FUTURE):
    return persistence.reserve_replay_lease(
        recovery_bundle=env["recovery_bundle"],
        slot=env["interrupted"] if slot is None else slot,
        budget=env["budget"] if budget is None else budget,
        deadline=deadline,
    )


def test_fixture_matches_recovery_plan():
    assert len(recovery_plan.RECIPES) == 4
    assert len(recovery_plan.SEEDS) == 3
    assert recovery_plan.SLOTS_PER_UNIT == 12
    assert recovery_plan.PARTIAL_COMPLETED == 8


def test_sealed_copy_preserves_every_file(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    source_before = _tree_map(env["unit_dir"])
    result = _copy(env)
    assert result["unit_active"] is True
    assert result["unit_kind"] == "sealed"
    assert result["omitted_interrupted_history"] is False
    assert len(result["copied_files"]) == 61
    destination = result["destination_unit_dir"]
    assert destination == env["new_develop"] / "units" / UNIT_ID
    assert _tree_map(destination) == source_before
    assert result["copied_bytes"] == sum(len(data) for data in source_before.values())
    assert _tree_map(env["unit_dir"]) == source_before
    env["budget"].close_unit()


def test_partial_copy_omits_interrupted_history(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    interrupted_relative = f"histories/{_execution_id(env['unit'], env['interrupted'])}.jsonl"
    source_before = _tree_map(env["unit_dir"])
    result = _copy(env)
    assert result["unit_kind"] == "partial"
    assert result["omitted_interrupted_history"] is True
    assert len(result["copied_files"]) == 40
    copied = _tree_map(result["destination_unit_dir"])
    assert len(copied) == 40
    assert interrupted_relative not in copied
    assert interrupted_relative in source_before
    assert _tree_map(env["unit_dir"]) == source_before
    env["budget"].close_unit()


def test_copied_files_are_independent_inodes(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    result = _copy(env)
    destination = result["destination_unit_dir"]
    for relative in result["copied_files"]:
        source = env["unit_dir"] / relative
        target = destination / relative
        assert source.read_bytes() == target.read_bytes()
        assert source.stat().st_ino != target.stat().st_ino
        assert not target.is_symlink()
    env["budget"].close_unit()


def test_destination_confined_to_recovery_root(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    result = _copy(env)
    assert env["recovery_root"] in result["destination_unit_dir"].parents
    env["budget"].close_unit()
    assert sorted(p.name for p in env["new_develop"].iterdir()) == ["units"]
    assert sorted(p.name for p in env["recovery_root"].iterdir()) == ["develop"]


def test_budget_accounts_unit_once_on_close(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    budget = env["budget"]
    base_before = budget._base
    result = _copy(env)
    assert budget._unit == result["destination_unit_dir"]
    assert budget.check() > base_before
    size = budget.close_unit()
    assert size == result["copied_bytes"]
    assert budget._base == base_before + size
    assert result["destination_unit_dir"] in budget._closed_units
    assert budget._unit is None
    with pytest.raises(P05StorageError):
        budget.close_unit()


def test_second_copy_refused_without_change(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    result = _copy(env)
    env["budget"].close_unit()
    before = _tree_map(result["destination_unit_dir"])
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env)
    assert _tree_map(result["destination_unit_dir"]) == before


def test_preoccupied_unit_refused(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    destination = env["new_develop"] / "units" / UNIT_ID
    destination.mkdir(parents=True)
    (destination / "keep.bin").write_bytes(b"keep")
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env)
    assert sorted(p.name for p in destination.iterdir()) == ["keep.bin"]


def test_expired_deadline_writes_nothing(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env, deadline=0.0)
    assert not (env["new_develop"] / "units" / UNIT_ID).exists()


def test_headroom_failure_precedes_activation(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    tight = StorageBudget(env["artifact_root"], env["run_root"], ceiling=env["budget"]._base + 1)
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env, budget=tight)
    assert tight._unit is None
    assert not (env["new_develop"] / "units" / UNIT_ID).exists()


def test_wrong_recovery_path_forbidden(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    bundle = dict(env["recovery_bundle"])
    bundle["original_run_root"] = str(
        env["artifact_root"] / inputs.COMPREHENSIVE_NAMESPACE / "runs" / ("0" * 64)
    )
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env, bundle=bundle)


def test_record_size_requires_exact_keys_and_hex():
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._record_size({"sha256": "0" * 64, "size_bytes": 1, "extra": 1})
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._record_size({"sha256": "z" * 64, "size_bytes": 1})
    for size in (True, -1):
        with pytest.raises(persistence.RecoveryPersistenceError):
            persistence._record_size({"sha256": "0" * 64, "size_bytes": size})


def test_abs_rejects_parent_segments():
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._abs("units/../units")


def test_budget_ceiling_must_not_exceed_approved(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    oversized = StorageBudget(env["artifact_root"], env["run_root"], ceiling=2 * APPROVED_CEILING)
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env, budget=oversized)


def test_budget_artifact_root_must_match(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    monkeypatch.setattr(env["budget"], "_artifact_root", tmp_path / "elsewhere")
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env)


def test_sealed_copy_plan_requires_exact_product(monkeypatch):
    unit = {"unit_id": UNIT_ID}
    slots = [
        {"slot_id": f"s{index}", "unit_id": UNIT_ID, "recipe_id": recipe, "seed": seed}
        for index, (recipe, seed) in enumerate(
            [(recipe, seed) for recipe in RECIPES for seed in SEEDS]
        )
    ]
    slots[-1] = dict(slots[0])
    plan = {"sealed_unit_ids": [UNIT_ID], "incomplete_unit_id": None}
    with pytest.raises(persistence.RecoveryPersistenceError):
        persistence._unit_copy_plan(plan, unit, slots, "sealed")


def test_duplicate_unit_ids_rejected(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    ledger = env["base_bundle"]["ledger"]
    ledger["units"] = [dict(ledger["units"][0]), dict(ledger["units"][0])]
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env)


def test_inventory_must_be_anchor_bound(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    env["recovery_bundle"]["original_anchor"] = {"files": {}}
    env["recovery_bundle"]["original_anchor_sha256"] = "0" * 64
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env)


def test_source_growth_cannot_exceed_inventory_size(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    inventory = env["recovery_bundle"]["original_inventory"]
    prefix = f"units/{UNIT_ID}/"
    relative = sorted(
        key[len(prefix) :] for key in inventory if key.startswith(prefix + "executions/")
    )[0]
    recorded = inventory[prefix + relative]["size_bytes"]
    target_source = env["unit_dir"] / relative
    target_source.write_bytes(target_source.read_bytes() + b"x" * 32)
    with pytest.raises(persistence.RecoveryPersistenceError):
        _copy(env)
    target = env["new_develop"] / "units" / UNIT_ID / relative
    assert not target.exists() or target.stat().st_size <= recorded


def test_replay_lease_payload_is_exact_and_single(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    interrupted = env["interrupted"]
    old_lease = (
        base_inputs._slot_lease_root(env["artifact_root"]) / interrupted["slot_id"] / "lease.json"
    )
    old_bytes = old_lease.read_bytes()
    lease_path = _reserve(env)
    assert lease_path == env["recovery_root"] / "replay_lease.json"
    expected = {
        "schema_version": persistence.REPLAY_LEASE_SCHEMA_VERSION,
        "attempt_kind": persistence.REPLAY_LEASE_ATTEMPT_KIND,
        "base_comprehensive_permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "core_contract_sha256": str(env["base_bundle"]["contract_sha256"]),
        "core_plan_id": str(env["base_bundle"]["core_plan_id"]),
        "slot_id": interrupted["slot_id"],
        "unit_id": UNIT_ID,
        "recipe_id": interrupted["recipe_id"],
        "seed": interrupted["seed"],
        "original_lease_sha256": hashlib.sha256(old_bytes).hexdigest(),
    }
    assert json.loads(lease_path.read_bytes()) == expected
    assert old_lease.read_bytes() == old_bytes
    assert lease_path in env["budget"]._charged
    assert env["budget"]._unit is None
    with pytest.raises(persistence.RecoveryPersistenceError):
        _reserve(env)
    assert json.loads(lease_path.read_bytes()) == expected


def test_replay_rejects_non_interrupted_slot(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    with pytest.raises(persistence.RecoveryPersistenceError):
        _reserve(env, slot=env["slots"][0])
    assert not (env["recovery_root"] / "replay_lease.json").exists()


def test_replay_rejects_changed_or_float_seed(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    for seed in (1, float(env["interrupted"]["seed"])):
        altered = dict(env["interrupted"])
        altered["seed"] = seed
        with pytest.raises(persistence.RecoveryPersistenceError):
            _reserve(env, slot=altered)
    assert not (env["recovery_root"] / "replay_lease.json").exists()


def test_replay_rejects_wrong_old_lease(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    old_lease = (
        base_inputs._slot_lease_root(env["artifact_root"])
        / env["interrupted"]["slot_id"]
        / "lease.json"
    )
    old_lease.write_bytes(old_lease.read_bytes() + b" ")
    with pytest.raises(persistence.RecoveryPersistenceError):
        _reserve(env)
    assert not (env["recovery_root"] / "replay_lease.json").exists()


def test_replay_rejects_bad_lease_inventory_digest(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    key = f"{env['interrupted']['slot_id']}/lease.json"
    env["recovery_bundle"]["original_lease_inventory"][key]["sha256"] = "0" * 64
    with pytest.raises(persistence.RecoveryPersistenceError):
        _reserve(env)
    assert not (env["recovery_root"] / "replay_lease.json").exists()


def test_replay_rejects_occupied_lease(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    occupied = env["recovery_root"] / "replay_lease.json"
    occupied.write_bytes(b"occupied")
    with pytest.raises(persistence.RecoveryPersistenceError):
        _reserve(env)
    assert occupied.read_bytes() == b"occupied"


def test_failure_during_copy_preserves_both_original_and_partial_destination(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    before = _tree_map(env["unit_dir"])
    original = persistence._copy_evidence
    copied = []

    def fail_second(source, destination, record, budget, deadline):
        if copied:
            raise persistence.RecoveryPersistenceError("injected_copy_failure")
        result = original(source, destination, record, budget, deadline)
        copied.append(destination)
        return result

    monkeypatch.setattr(persistence, "_copy_evidence", fail_second)
    with pytest.raises(persistence.RecoveryPersistenceError, match="injected_copy_failure"):
        _copy(env)
    assert len(copied) == 1 and copied[0].is_file()
    assert _tree_map(env["unit_dir"]) == before
    assert env["budget"]._unit is not None
    with pytest.raises(persistence.RecoveryPersistenceError, match="destination_unit_exists"):
        _copy(env)


def test_replay_lease_accounting_outside_active_unit(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    copied = _copy(env)
    before = env["budget"].check()
    lease = _reserve(env)
    assert env["budget"]._unit == copied["destination_unit_dir"]
    assert env["budget"].check() == before + lease.stat().st_size
    env["budget"].close_unit()
    assert env["budget"].check() == before + lease.stat().st_size


@pytest.mark.parametrize("kind", ["sealed", "partial"])
def test_changed_bytes_and_matching_replaced_inventory_still_fail_anchor(
    tmp_path, monkeypatch, kind
):
    env = _build_env(tmp_path, monkeypatch, kind)
    original_keys = sorted(env["recovery_bundle"]["original_inventory"])
    key = next(key for key in original_keys if key.endswith("best.pt"))
    path = env["stage"] / key
    path.write_bytes(b"replacement-weights")
    env["recovery_bundle"]["original_inventory"][key] = _record(path)
    with pytest.raises(
        persistence.RecoveryPersistenceError, match="unit_inventory_anchor_mismatch"
    ):
        _copy(env)
    assert env["budget"]._unit is None
    assert not (env["new_develop"] / "units" / UNIT_ID).exists()
