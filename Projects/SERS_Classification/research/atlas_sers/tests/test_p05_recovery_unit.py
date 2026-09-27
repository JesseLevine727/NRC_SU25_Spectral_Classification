"""Synthetic unit tests for the P05 recovery single-unit boundary.

Temporary synthetic fixtures only.  No private data, no real torch and no
``p05_development`` import: numerical boundaries, training and the authority
loader are monkeypatched.  The storage budget, journal appends and path layout
are the real implementations.  Several regressions intentionally assert the
required behaviour and therefore fail until the reviewed defects are fixed.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_evidence as evidence
from atlas_sers.evaluation import p05_recovery_inputs as inputs
from atlas_sers.evaluation import p05_recovery_persistence as persistence
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan
from atlas_sers.evaluation import p05_recovery_unit as recovery_unit
from atlas_sers.evaluation.p05_comprehensive_storage import StorageBudget

RECIPES = ("D0-M", "D1", "D2", "D3")
SEEDS = (20260805, 20260817, 20260829)
UNIT_ID = "unit-0001"
FUTURE = float(2**31)
EPOCHS = 30
FIT_STEPS = EPOCHS * 4


def _canon_bytes(value):
    return core._canon().canonical_json_bytes(value)


def _canon_sha(value):
    return core._canon().sha256_value(value)


@dataclasses.dataclass
class _FakeResult:
    optimizer_steps: int = FIT_STEPS
    elapsed_seconds: float = 0.5
    peak_cuda_bytes: int = 1024
    status: str = "complete"
    history: list = dataclasses.field(default_factory=list)


class _Recorder:
    def __init__(self, path):
        self.records = []
        self.closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("xb")

    def __call__(self, record):
        self.records.append(dict(record))
        self.stream.write(_canon_bytes(record) + b"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.closed = True
        self.stream.close()


def _ordered_slots():
    return [
        {
            "slot_id": f"{UNIT_ID}-{recipe}-{seed}",
            "unit_id": UNIT_ID,
            "recipe_id": recipe,
            "seed": seed,
        }
        for recipe in RECIPES
        for seed in SEEDS
    ]


def _build_env(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(authority, "validate_recovery_permit", lambda permit: None)
    monkeypatch.setattr(authority, "check_resources", lambda torch, phase=None, **kw: None)

    artifact_root = tmp_path / "artifact"
    run_root = (
        artifact_root
        / inputs.COMPREHENSIVE_NAMESPACE
        / "runs"
        / authority.BASECOMPREHENSIVE_PERMIT_SHA256
    )
    stage = run_root / inputs.DEVELOP_STAGE_NAME
    unit = {"unit_id": UNIT_ID, "station": "STN-01"}
    slots = _ordered_slots()
    ordered = [slot["slot_id"] for slot in slots]

    if kind == "sealed":
        reused, unstarted = list(ordered), []
        interrupted = "other-unit-interrupted-slot"
        sealed, incomplete = [UNIT_ID], None
    elif kind == "partial":
        reused, interrupted, unstarted = ordered[:8], ordered[8], ordered[9:]
        sealed, incomplete = [], UNIT_ID
    else:
        reused, unstarted = [], list(ordered)
        interrupted = "other-unit-interrupted-slot"
        sealed, incomplete = [], None

    plan = {
        "sealed_unit_ids": sealed,
        "incomplete_unit_id": incomplete,
        "reused_original_slot_ids": reused,
        "unstarted_slot_ids": unstarted,
        "interrupted_slot_id": interrupted,
    }
    base_bundle = {
        "permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "artifact_root": str(artifact_root),
        "contract": {"synthetic": True},
        "contract_sha256": base_inputs.CORE_CONTRACT_SHA256,
        "core_plan_id": base_inputs.CORE_PLAN_ID,
        "ledger": {"units": [dict(unit)], "slots": [dict(slot) for slot in slots]},
    }
    anchor = {"files": {}}
    anchor_sha = _canon_sha(anchor)
    monkeypatch.setattr(authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", anchor_sha)
    recovery_bundle = {
        "recovery_permit": {"synthetic": True},
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "base_bundle": base_bundle,
        "plan": plan,
        "original_run_root": str(run_root),
        "original_stage": str(stage),
        "original_inventory": {},
        "original_anchor": anchor,
        "original_anchor_sha256": anchor_sha,
        "original_lease_inventory": {},
    }
    recovery_root = run_root / recovery_unit.RECOVERIES_DIRNAME / authority.RECOVERY_PERMIT_SHA256
    rec_stage = recovery_root / recovery_unit.RECOVERY_STAGE_NAME
    rec_stage.mkdir(parents=True, exist_ok=True)
    (rec_stage / "units").mkdir()
    events_path = rec_stage / "events.jsonl"
    selector_path = rec_stage / "selector.jsonl"
    base_inputs._slot_lease_root(artifact_root).mkdir(parents=True, exist_ok=True)
    counters = {
        "new_started": 0,
        "new_completed": 0,
        "new_failed": 0,
        "new_optimizer_steps": 0,
        "new_optimizer_steps_exact": True,
        "new_elapsed_seconds": 0.0,
        "new_peak_cuda_bytes": 0,
        "reused_completed": 0,
        "reused_optimizer_steps": 0,
        "replay_started": 0,
        "unstarted_started": 0,
    }

    completed_slots = [slot for slot in slots if slot["slot_id"] in set(reused)]

    def fake_verified(**kwargs):
        items, selectors, updates = [], [], 0
        for slot in completed_slots:
            items.append(
                {
                    "slot": dict(slot),
                    "unit": dict(unit),
                    "unit_id": UNIT_ID,
                    "seed": int(slot["seed"]),
                    "result": _FakeResult(),
                }
            )
            selectors.append({"slot_id": slot["slot_id"], "status": "complete"})
            updates += FIT_STEPS
        return {
            "items": items,
            "selector_records": selectors,
            "unit_id": UNIT_ID,
            "completed_slots": [slot["slot_id"] for slot in completed_slots],
            "optimizer_updates_exact": updates,
            "complete_unit_cross_recipe_checks": kind == "sealed",
            "deferred_until_replay": kind != "sealed",
            "fits_started": 0,
            "files_written": 0,
        }

    monkeypatch.setattr(evidence, "load_verified_original_unit", fake_verified)

    def fake_copy(*, recovery_bundle, unit_id, budget, deadline):
        destination = rec_stage / "units" / unit_id
        budget.activate_unit(destination)
        (destination / "manifest.json").write_bytes(b"{}\n")
        (destination / "history.jsonl").write_bytes(b"synthetic\n")
        return {
            "schema_version": "synthetic",
            "unit_id": unit_id,
            "unit_kind": kind if kind != "new" else "sealed",
            "destination_unit_dir": destination,
            "copied_files": ["manifest.json", "history.jsonl"],
            "copied_bytes": 13,
            "omitted_interrupted_history": kind == "partial",
            "unit_active": True,
        }

    monkeypatch.setattr(persistence, "copy_original_unit", fake_copy)

    def fake_reserve(*, recovery_bundle, slot, budget, deadline):
        path = recovery_root / "replay_lease.json"
        path.write_bytes(_canon_bytes({"slot_id": str(slot["slot_id"])}))
        budget.account_new_file(path)
        return path

    monkeypatch.setattr(persistence, "reserve_replay_lease", fake_reserve)
    monkeypatch.setattr(
        pilot,
        "prepare_role_inputs",
        lambda bundle: {
            str(entry["unit_id"]): {"unit": str(entry["unit_id"])} for entry in bundle["units"]
        },
    )
    recorders = []

    def fake_open(unit_dir, identifier):
        recorder = _Recorder(unit_dir / "histories" / f"{identifier}.jsonl")
        recorders.append((identifier, recorder))
        return recorder

    monkeypatch.setattr(pilot, "open_history_recorder", fake_open)

    def fake_train(unit_inputs, unit, slot, device, deadline, recorder):
        history = []
        for index in range(EPOCHS):
            record = {"epoch": index + 1, "sampling_digest": "a" * 64}
            recorder(record)
            history.append(record)
        return _FakeResult(history=history)

    monkeypatch.setattr(pilot, "train_fit", fake_train)
    monkeypatch.setattr(pilot, "persist_result", lambda torch, unit_dir, unit, slot, result: None)
    monkeypatch.setattr(pilot, "check_completed_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot, "check_sparse_support", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot, "_verify_manifest", lambda path: None)
    monkeypatch.setattr(
        base_inputs,
        "selector_record",
        lambda unit, slot, summary: {"slot_id": str(slot["slot_id"]), "status": "complete"},
    )
    monkeypatch.setattr(development, "_source_identity_adapter", lambda result, unit, slot: {})
    original_history = [
        {"epoch": index + 1, "sampling_digest": "a" * 64}
        for index in range(recovery_plan.INTERRUPTED_EPOCHS)
    ]
    if kind == "partial":
        identifier = pilot.execution_id(unit, slots[8])
        relative = f"units/{UNIT_ID}/histories/{identifier}.jsonl"
        history_path = stage / relative
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_bytes(b"".join(_canon_bytes(row) + b"\n" for row in original_history))
        record = inputs._hash_file_record(history_path, FUTURE)
        anchor["files"][relative] = record
        recovery_bundle["original_inventory"][relative] = dict(record)
        anchor_sha = _canon_sha(anchor)
        recovery_bundle["original_anchor_sha256"] = anchor_sha
        monkeypatch.setattr(authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", anchor_sha)
    budget = StorageBudget(artifact_root, run_root)
    budget.register_growing(events_path)
    budget.register_growing(selector_path)
    shared_calls, sparse_calls, manifest_calls = [], [], []
    monkeypatch.setattr(
        pilot, "check_shared_prefixes", lambda items: shared_calls.append(len(items))
    )
    monkeypatch.setattr(
        pilot,
        "check_sparse_equivalences",
        lambda items, units, contract: sparse_calls.append(len(items)),
    )
    monkeypatch.setattr(core, "_write_manifest", lambda path: manifest_calls.append(Path(path)))

    return {
        "artifact_root": artifact_root,
        "run_root": run_root,
        "stage": stage,
        "rec_stage": rec_stage,
        "recovery_root": recovery_root,
        "units_root": rec_stage / "units",
        "events_path": events_path,
        "selector_path": selector_path,
        "unit": unit,
        "slots": slots,
        "interrupted_slot_id": interrupted,
        "base_bundle": base_bundle,
        "recovery_bundle": recovery_bundle,
        "budget": budget,
        "counters": counters,
        "torch": object(),
        "recorders": recorders,
        "shared_calls": shared_calls,
        "sparse_calls": sparse_calls,
        "manifest_calls": manifest_calls,
    }


def _run(env, **overrides):
    kwargs = {
        "recovery_bundle": env["recovery_bundle"],
        "unit": env["unit"],
        "expected_selectors": {},
        "torch": env["torch"],
        "device": "cuda",
        "budget": env["budget"],
        "deadline": FUTURE,
        "events_path": env["events_path"],
        "selector_path": env["selector_path"],
        "counters": env["counters"],
    }
    kwargs.update(overrides)
    return recovery_unit.run_unit(**kwargs)


def test_fixture_matches_recovery_plan():
    assert recovery_plan.SLOTS_PER_UNIT == 12
    assert recovery_plan.PARTIAL_COMPLETED == 8
    assert len(RECIPES) * len(SEEDS) == 12


def test_sealed_unit_reuses_twelve_without_fits(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    result = _run(env)
    assert result["unit_kind"] == "sealed"
    assert result["new_started"] == 0 and result["new_completed"] == 0
    assert result["optimizer_updates_reused"] == 12 * FIT_STEPS
    assert env["counters"]["reused_completed"] == 12
    assert env["counters"]["new_started"] == 0
    assert env["manifest_calls"] == []
    assert env["shared_calls"] == []


def test_partial_unit_replays_one_and_runs_three(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    result = _run(env)
    assert result["unit_kind"] == "partial"
    assert result["replay_slot_id"] == env["interrupted_slot_id"]
    assert result["new_started"] == 4 and result["new_completed"] == 4
    assert env["counters"]["reused_completed"] == 8
    assert env["counters"]["replay_started"] == 1
    assert env["counters"]["unstarted_started"] == 3
    assert env["shared_calls"] == [12]
    assert env["sparse_calls"] == [12]
    assert len(env["manifest_calls"]) == 1
    assert (env["recovery_root"] / "replay_lease.json").is_file()


def test_new_unit_runs_all_twelve(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    result = _run(env)
    assert result["unit_kind"] == "new"
    assert result["new_started"] == 12 and result["new_completed"] == 12
    assert env["counters"]["unstarted_started"] == 12
    assert env["shared_calls"] == [12] and env["sparse_calls"] == [12]


def test_cumulative_counters_match_result(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    result = _run(env)
    assert env["counters"]["new_started"] == result["new_started"] == 12
    assert env["counters"]["new_completed"] == result["new_completed"] == 12
    assert env["counters"]["new_optimizer_steps"] == result["optimizer_updates_new"]


def test_event_journal_ordering(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    _run(env)
    kinds = [
        json.loads(line)["event"]
        for line in env["events_path"].read_text().splitlines()
        if line.strip()
    ]
    assert kinds.count("started") == 12
    assert kinds.count("completed") == 12
    for index, kind in enumerate(kinds):
        if kind == "completed":
            assert kinds[index - 1] == "started"


def test_original_stage_files_unchanged(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    original_dir = env["stage"] / "units" / UNIT_ID
    original_dir.mkdir(parents=True, exist_ok=True)
    marker = original_dir / "marker.bin"
    marker.write_bytes(b"original")
    _run(env)
    assert marker.read_bytes() == b"original"


def test_deadline_failure_writes_nothing(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env, deadline=0.0)
    assert not env["events_path"].exists()
    assert env["counters"]["new_started"] == 0


def test_epoch_guard_failure_persists_epoch_first(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")

    def fake_check(torch, phase=None, **kwargs):
        if phase == "epoch":
            raise authority.RecoveryAuthorityError("host_guard")
        return None

    monkeypatch.setattr(authority, "check_resources", fake_check)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert env["recorders"], "recorder should have been opened"
    _identifier, recorder = env["recorders"][0]
    assert recorder.records, "epoch must be persisted before the guard failure"
    assert recorder.closed is True


def test_recorder_closed_when_fit_raises(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")

    def boom(*args, **kwargs):
        raise RuntimeError("model boom")

    monkeypatch.setattr(pilot, "train_fit", boom)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert env["recorders"][0][1].closed is True


def test_model_exception_sets_inexact_lower_bound(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")

    def boom(unit_inputs, unit, slot, device, deadline, recorder):
        recorder({"epoch": 1})
        raise RuntimeError("model boom")

    monkeypatch.setattr(pilot, "train_fit", boom)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert env["counters"]["new_failed"] == 1
    assert env["counters"]["new_optimizer_steps"] == 4
    assert env["counters"]["new_optimizer_steps_exact"] is False


def test_result_acceptance_failure_counts(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")

    def reject(*args, **kwargs):
        raise RuntimeError("reject")

    monkeypatch.setattr(pilot, "check_completed_result", reject)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert env["counters"]["new_started"] == 1
    assert env["counters"]["new_completed"] == 0
    assert env["counters"]["new_failed"] == 1


def test_wrong_journal_paths_rejected(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env, events_path=env["rec_stage"] / "other.jsonl")
    assert not (env["rec_stage"] / "other.jsonl").exists()


def test_unknown_unit_rejected(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env, unit={"unit_id": "unit-9999", "station": "STN-01"})


def test_unit_identity_mismatch(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env, unit={"unit_id": UNIT_ID, "station": "STN-01", "extra": 1})


def test_slot_product_mismatch(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    env["base_bundle"]["ledger"]["slots"][0]["seed"] = 99999999
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)


@pytest.mark.parametrize("bad", [True, -1, float("nan")])
def test_counters_reject_bool_negative_nonfinite(tmp_path, monkeypatch, bad):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    env["counters"]["new_started"] = bad
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)


def test_fit_ceiling_checked_before_reserving_lease(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    env["counters"].update(
        new_started=recovery_plan.RECOVERY_FITS,
        new_completed=recovery_plan.RECOVERY_FITS,
        new_optimizer_steps=recovery_plan.RECOVERY_FITS * 120,
        replay_started=1,
        unstarted_started=recovery_plan.ORIGINAL_UNSTARTED,
    )
    with pytest.raises(recovery_unit.RecoveryUnitError, match="recovery_fit_ceiling_exceeded"):
        _run(env)
    leases_root = base_inputs._slot_lease_root(env["artifact_root"])
    assert not list(leases_root.rglob("lease.json"))


def test_unstarted_lease_uses_base_permit(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")
    _run(env)
    lease_files = list(base_inputs._slot_lease_root(env["artifact_root"]).rglob("lease.json"))
    assert lease_files
    for path in lease_files:
        assert json.loads(path.read_bytes())["permit_sha256"] == (
            authority.BASECOMPREHENSIVE_PERMIT_SHA256
        )


def test_aggregate_reports_real_fit_count_without_invented_file_count(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    result = _run(env)
    assert result["new_completed"] == 4
    assert result["fits_started"] == 4
    assert "files_written" not in result or result["files_written"] > 0


def test_reused_selectors_not_written_before_copy(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    (env["units_root"] / UNIT_ID).mkdir(parents=True, exist_ok=True)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert not env["selector_path"].exists() or env["selector_path"].read_bytes() == b""


def test_duplicate_sealed_unit_rejected_before_journal(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    first = _run(env)
    assert first["unit_kind"] == "sealed"
    events_before = env["events_path"].read_bytes()
    selectors_before = env["selector_path"].read_bytes()
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert env["events_path"].read_bytes() == events_before
    assert env["selector_path"].read_bytes() == selectors_before


def test_unit_classification_requires_plan_membership(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    env["recovery_bundle"]["plan"]["sealed_unit_ids"] = []
    env["recovery_bundle"]["plan"]["incomplete_unit_id"] = "other-unit"
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)


def test_symlinked_journal_cannot_write_outside(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "sealed")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"")
    env["events_path"].unlink(missing_ok=True)
    env["events_path"].symlink_to(outside)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert outside.read_bytes() == b""


@pytest.mark.parametrize("defect", ["ceiling", "root", "journal"])
def test_bad_budget_binding_prevents_every_attempt(tmp_path, monkeypatch, defect):
    env = _build_env(tmp_path, monkeypatch, "new")
    if defect == "ceiling":
        env["budget"]._ceiling = authority.PRIVATE_STORAGE_CEILING_BYTES + 1
    elif defect == "root":
        env["budget"]._run_dir = env["rec_stage"]
    else:
        env["budget"]._growing.clear()
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert env["counters"]["new_started"] == 0
    assert env["recorders"] == []
    assert not env["events_path"].exists()


def test_changed_original_prefix_cannot_become_replay_evidence(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    train = pilot.train_fit

    def tampering_train(*args):
        result = train(*args)
        identifier = pilot.execution_id(env["unit"], env["slots"][8])
        path = env["stage"] / "units" / UNIT_ID / "histories" / f"{identifier}.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]["sampling_digest"] = "b" * 64
        result.history[0]["sampling_digest"] = "b" * 64
        path.write_bytes(b"".join(_canon_bytes(row) + b"\n" for row in rows))
        return result

    monkeypatch.setattr(pilot, "train_fit", tampering_train)
    with pytest.raises(
        recovery_unit.RecoveryUnitError, match="interrupted_history_digest_mismatch"
    ):
        _run(env)
    assert env["counters"]["replay_started"] == 1
    assert env["counters"]["new_completed"] == 0
    assert env["counters"]["new_failed"] == 1
    assert len(env["selector_path"].read_bytes().splitlines()) == 8


def test_replay_mismatch_preserves_single_attempt_and_prevents_more_fits(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")
    train = pilot.train_fit

    def changed_train(*args):
        result = train(*args)
        result.history[0]["sampling_digest"] = "b" * 64
        return result

    monkeypatch.setattr(pilot, "train_fit", changed_train)
    with pytest.raises(recovery_unit.RecoveryUnitError, match="replay_prefix_mismatch"):
        _run(env)
    assert env["counters"]["new_started"] == 1
    assert env["counters"]["new_failed"] == 1
    assert env["counters"]["new_optimizer_steps_exact"] is True
    assert len(env["recorders"]) == 1
    assert (env["recovery_root"] / "replay_lease.json").exists()
    before = env["events_path"].read_bytes()
    with pytest.raises(recovery_unit.RecoveryUnitError, match="prior_recovery_failure"):
        _run(env)
    assert env["events_path"].read_bytes() == before


def test_lease_accounting_failure_cannot_restore_attempt_credit(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")

    def fail_after_creation(path):
        assert path.exists()
        raise RuntimeError("synthetic accounting failure")

    monkeypatch.setattr(env["budget"], "account_new_file", fail_after_creation)
    with pytest.raises(recovery_unit.RecoveryUnitError):
        _run(env)
    assert env["counters"]["new_started"] == 1
    assert env["counters"]["new_failed"] == 1
    assert env["counters"]["new_optimizer_steps_exact"] is False
    assert len(list(base_inputs._slot_lease_root(env["artifact_root"]).rglob("lease.json"))) == 1
    assert env["recorders"] == []


def test_guard_failure_after_fitting_keeps_exact_returned_update_count(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "new")

    def guard(torch, phase, **kwargs):
        if phase == "fit" and env["recorders"] and env["recorders"][0][1].closed:
            raise authority.RecoveryAuthorityError("host_low_after_fit")

    monkeypatch.setattr(authority, "check_resources", guard)
    with pytest.raises(recovery_unit.RecoveryUnitError, match="host_low_after_fit"):
        _run(env)
    assert env["counters"]["new_started"] == 1
    assert env["counters"]["new_failed"] == 1
    assert env["counters"]["new_optimizer_steps"] == FIT_STEPS
    assert env["counters"]["new_optimizer_steps_exact"] is True


def test_copy_failure_does_not_append_reused_selectors(tmp_path, monkeypatch):
    env = _build_env(tmp_path, monkeypatch, "partial")

    def failed_copy(**kwargs):
        path = env["budget"].activate_unit(env["units_root"] / UNIT_ID)
        (path / "first.bin").write_bytes(b"preserved partial copy")
        raise RuntimeError("synthetic copy failure")

    monkeypatch.setattr(persistence, "copy_original_unit", failed_copy)
    with pytest.raises(recovery_unit.RecoveryUnitError, match="original_unit_copy_failed"):
        _run(env)
    assert (env["units_root"] / UNIT_ID / "first.bin").read_bytes() == b"preserved partial copy"
    assert not env["selector_path"].exists()
    assert env["counters"]["reused_completed"] == 0
