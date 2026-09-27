"""Synthetic integration tests for the P05 recovery-development boundary.

These tests exercise the real serial control flow of
``p05_recovery_development`` against a tiny three-unit fixture.  The real
filesystem, :class:`StorageBudget`, exclusive containers, ``_write_new_file``,
growing journals, counters and ``_validate_unit_result`` are used unchanged.
Only the outer seams (input bundle preparation, torch import, numerical
helpers, provenance, the unit worker, the original-stage/lease closure and the
receipt builders) are replaced with narrow synthetic stand-ins, and the outer
programme-size constants are scaled down for the fixture.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_recovery_development as rd

UPDATES_PER_FIT = 120
REUSED_PILOT_SLOTS = 36
NONPILOT_UNITS = 3
SLOTS_PER_UNIT = 12
NEW_UNIT_COUNT = 3
RECOVERY_FITS = 16
REUSED_COMPLETED = 20
REPLAY_STARTED = 1
UNSTARTED_STARTED = 15
TOTAL_SELECTORS = REUSED_PILOT_SLOTS + NONPILOT_UNITS * SLOTS_PER_UNIT
PRIOR_CHARGE_SECONDS = 39600.0
MAXIMUM_TOTAL_SECONDS = 172800.0

_UNIT_KINDS = ("sealed", "partial", "new")
_SPECS = {
    "sealed": {"reused": 12, "replay": False},
    "partial": {"reused": 8, "replay": True},
    "new": {"reused": 0, "replay": False},
}


class _Reason(Exception):
    """Synthetic failure carrying a path-free reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class _FakeTorch:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    def __init__(self) -> None:
        self.cuda = _FakeTorch._Cuda()

    @staticmethod
    def set_num_threads(_count: int) -> None:
        return None


class _ImportShim:
    """Forward every import to the stdlib except a synthetic ``torch``."""

    def __init__(self, torch_module: _FakeTorch) -> None:
        self._torch = torch_module
        self._importlib = importlib

    def __getattr__(self, name: str):
        return getattr(self._importlib, name)

    def import_module(self, name: str, *args, **kwargs):
        if name == "torch":
            return self._torch
        return self._importlib.import_module(name, *args, **kwargs)


def _unit_records():
    pilot_ids = [f"pilot-slot-{index:03d}" for index in range(REUSED_PILOT_SLOTS)]
    units = []
    for index, kind in enumerate(_UNIT_KINDS):
        unit_id = f"unit-{kind}-{index:02d}"
        slot_ids = [f"{unit_id}-slot-{position:02d}" for position in range(SLOTS_PER_UNIT)]
        units.append({"unit_id": unit_id, "kind": kind, "slot_ids": slot_ids})
    return pilot_ids, units


def _stage_path(env):
    return (
        env["original_run_root"]
        / rd.persistence.RECOVERIES_DIRNAME
        / rd.authority.RECOVERY_PERMIT_SHA256
        / rd.persistence.DEVELOP_STAGE_NAME
    )


def _install(monkeypatch, tmp_path, **options):
    behavior = options.get("behavior", "success")
    closure_raises = bool(options.get("closure_raises", False))
    launch_raises = bool(options.get("launch_resources_raises", False))
    preexisting = bool(options.get("preexisting_recoveries", False))
    deadline_raises = bool(options.get("deadline_raises", False))
    receipt_raises = bool(options.get("receipt_write_raises", False))

    artifact = tmp_path / "artifact"
    (artifact / "p05development").mkdir(parents=True, exist_ok=True)
    (artifact / "p05comprehensive").mkdir(parents=True, exist_ok=True)
    project_root = tmp_path / "project"
    project_root.mkdir()
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    original_run_root = (
        artifact / "p05comprehensive" / "runs" / rd.authority.BASECOMPREHENSIVE_PERMIT_SHA256
    )
    original_stage = original_run_root / "develop"
    original_stage.mkdir(parents=True, exist_ok=True)
    (original_stage / "preserved-original.bin").write_bytes(b"original-evidence-sentinel")

    pilot_ids, units = _unit_records()
    unit_by_id: dict = {}
    ordered: list = []
    slots: list = []
    all_slot_ids = set(pilot_ids)
    for unit in units:
        unit_id = unit["unit_id"]
        slot_ids = list(unit["slot_ids"])
        unit_by_id[unit_id] = {
            "unit_id": unit_id,
            "station": "station-0",
            "fitting_uid_set_sha256": "a" * 64,
            "validation_uid_set_sha256": "b" * 64,
            "kind": unit["kind"],
            "slot_ids": slot_ids,
        }
        unit_slots = []
        for position, slot_id in enumerate(slot_ids):
            slot = {
                "slot_id": slot_id,
                "unit_id": unit_id,
                "recipe_id": "recipe-0",
                "seed": position,
            }
            unit_slots.append(slot)
            slots.append(dict(slot))
        ordered.append((unit_id, unit_slots))
        all_slot_ids.update(slot_ids)

    ledger = {
        "ledger_id": rd.recovery_plan.LEDGER_ID,
        "units": [dict(value) for value in unit_by_id.values()],
        "slots": slots,
    }
    base = {
        "artifact_root": artifact,
        "repository_root": repository_root,
        "project_root": project_root,
        "contract": {"contract": "base"},
        "support": {"support": "base"},
        "permit_sha256": rd.authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "contract_sha256": "d" * 64,
        "core_plan_id": "e" * 64,
        "ledger": ledger,
    }

    partial_slots = unit_by_id["unit-partial-01"]["slot_ids"]
    unstarted_slot_ids = partial_slots[9:] + unit_by_id["unit-new-02"]["slot_ids"]
    plan = {
        "ledger_id": rd.recovery_plan.LEDGER_ID,
        "base_permit_sha256": rd.authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "execution_authorized": False,
        "fits_started": 0,
        "unstarted_slot_ids": unstarted_slot_ids,
        "interrupted_slot_id": partial_slots[8],
    }
    plan["plan_id"] = rd.core._canon().sha256_value(
        {key: value for key, value in plan.items() if key != "plan_id"}
    )

    recovery_bundle = {
        "base_bundle": base,
        "plan": plan,
        "recovery_permit": {"permit": "recovery"},
        "original_run_root": original_run_root,
        "original_stage": original_stage,
        "original_inventory": {"develop/selector.jsonl": {"sha256": "c" * 64, "size": 1}},
        "original_anchor": {"anchor": "original"},
        "original_anchor_sha256": "f" * 64,
        "original_lease_inventory": {"slot-000/lease.json": {"sha256": "0" * 64, "size": 1}},
    }

    monkeypatch.setattr(rd, "MAXIMUM_TOTAL_SECONDS", MAXIMUM_TOTAL_SECONDS)
    monkeypatch.setattr(rd, "PRIOR_CHARGE_SECONDS", PRIOR_CHARGE_SECONDS)
    monkeypatch.setattr(rd.authority, "MAXIMUM_TOTAL_SECONDS", MAXIMUM_TOTAL_SECONDS)
    monkeypatch.setattr(rd.development, "REUSED_PILOT_SLOTS", REUSED_PILOT_SLOTS)
    monkeypatch.setattr(rd.development, "MAXIMUM_NEW_FITS", NONPILOT_UNITS * SLOTS_PER_UNIT)
    monkeypatch.setattr(rd.recovery_plan, "NEW_UNIT_COUNT", NEW_UNIT_COUNT)
    monkeypatch.setattr(rd.recovery_plan, "RECOVERY_FITS", RECOVERY_FITS)
    monkeypatch.setattr(rd.recovery_plan, "ORIGINAL_COMPLETED", REUSED_COMPLETED)
    monkeypatch.setattr(rd.recovery_plan, "ORIGINAL_INTERRUPTED", REPLAY_STARTED)
    monkeypatch.setattr(rd.recovery_plan, "ORIGINAL_UNSTARTED", UNSTARTED_STARTED)

    monkeypatch.setattr(rd, "importlib", _ImportShim(_FakeTorch()))
    monkeypatch.setattr(rd.core, "_configure_environment", lambda: None)
    monkeypatch.setattr(
        rd.core, "_capture_provenance", lambda *args, **kwargs: {"provenance": "before"}
    )
    monkeypatch.setattr(rd.pilot, "_development_kernel", lambda: None)
    monkeypatch.setattr(rd.pilot, "_checkpoint_preflight", lambda *args, **kwargs: None)

    calls = {"closure": [], "post_reauth": []}

    def _post_run_reauth(*args, **kwargs):
        calls["post_reauth"].append(True)
        return {"provenance": "after"}

    monkeypatch.setattr(rd.pilot, "_post_run_reauth", _post_run_reauth)

    def _check_resources(_torch, phase="launch"):
        if launch_raises and phase == "launch":
            raise _Reason("insufficient_launch_resources")

    monkeypatch.setattr(rd.authority, "check_resources", _check_resources)

    pilot_records = [{"slot_id": slot_id, "status": "complete"} for slot_id in pilot_ids]
    monkeypatch.setattr(rd.base_inputs, "prepare", lambda *args, **kwargs: {"prepared": True})
    monkeypatch.setattr(
        rd.base_inputs, "import_pilot", lambda base_bundle, device="cuda": list(pilot_records)
    )
    monkeypatch.setattr(rd.base_inputs, "pilot_slot_ids", lambda base_bundle: set(pilot_ids))
    monkeypatch.setattr(rd.recovery_inputs, "prepare_recovery", lambda **kwargs: recovery_bundle)
    monkeypatch.setattr(
        rd.development,
        "_plan_units",
        lambda base_bundle: (unit_by_id, ordered, all_slot_ids),
    )

    monkeypatch.setattr(rd, "_read_original_selectors", lambda *args, **kwargs: {})
    monkeypatch.setattr(rd, "_check_pilot_selectors", lambda *args, **kwargs: None)

    def _check_selector_coverage(records, covered_slot_ids, *args, **kwargs):
        observed = {str(record.get("slot_id")) for record in records}
        assert len(records) == TOTAL_SELECTORS
        assert observed == set(covered_slot_ids)

    monkeypatch.setattr(rd, "_check_selector_coverage", _check_selector_coverage)

    def _reauthenticate_original(*args, **kwargs):
        calls["closure"].append("original")
        if closure_raises:
            raise rd.RecoveryDevelopmentError("original_stage_evidence_changed")

    def _reauthenticate_leases(*args, **kwargs):
        calls["closure"].append("leases")

    monkeypatch.setattr(rd, "_reauthenticate_original", _reauthenticate_original)
    monkeypatch.setattr(rd, "_reauthenticate_leases", _reauthenticate_leases)
    monkeypatch.setattr(rd, "_check_replay_lease", lambda *args, **kwargs: None)

    if deadline_raises:

        def _deadline(_deadline):
            raise rd.RecoveryDevelopmentError("global_deadline_exceeded")

        monkeypatch.setattr(rd, "_check_deadline", _deadline)

    unit_calls: list = []

    def fake_run_unit(**kwargs):
        unit_calls.append(kwargs)
        if behavior == "fail_second" and len(unit_calls) == 2:
            raise _Reason("unit_worker_failed")
        unit = kwargs["unit"]
        unit_id = str(unit["unit_id"])
        slot_ids = [str(value) for value in unit["slot_ids"]]
        kind = unit["kind"]
        spec = _SPECS[kind]
        if behavior == "wrong_result" and len(unit_calls) == 1:
            return {"unit_id": "wrong-unit"}

        stage = Path(kwargs["events_path"]).parent
        unit_dir = stage / "units" / unit_id
        unit_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = unit_dir / "manifest.json"
        if not manifest_path.exists():
            payload = unit_dir / "opaque-synthetic.bin"
            payload.write_bytes(b"opaque synthetic checkpoint stand-in")
            kwargs["budget"].account_new_file(payload)
            rd.core._write_manifest(unit_dir)
            kwargs["budget"].account_new_file(manifest_path)

        for slot_id in slot_ids:
            rd.development._append_jsonl(
                kwargs["selector_path"], {"slot_id": slot_id, "status": "complete"}
            )

        counters = kwargs["counters"]
        reused = slot_ids[: spec["reused"]]
        new = slot_ids[spec["reused"] :]
        replay_ids = [new[0]] if spec["replay"] else []
        updates_new = len(new) * UPDATES_PER_FIT
        updates_reused = len(reused) * UPDATES_PER_FIT
        counters["new_started"] += len(new)
        counters["new_completed"] += len(new)
        counters["new_optimizer_steps"] += updates_new
        counters["new_elapsed_seconds"] += 0.5 * len(new)
        counters["new_peak_cuda_bytes"] += 1024 * len(new)
        counters["reused_completed"] += len(reused)
        counters["reused_optimizer_steps"] += updates_reused
        counters["replay_started"] += len(replay_ids)
        counters["unstarted_started"] += len(new) - len(replay_ids)
        return {
            "unit_id": unit_id,
            "completed_slots": list(slot_ids),
            "reused_slots": list(reused),
            "new_slots": list(new),
            "replay_slot_id": replay_ids[0] if replay_ids else None,
            "unit_kind": kind,
            "new_started": len(new),
            "new_completed": len(new),
            "new_failed": 0,
            "fits_started": len(new),
            "optimizer_updates_reused": updates_reused,
            "optimizer_updates_new": updates_new,
        }

    monkeypatch.setattr(rd.unit_runner, "run_unit", fake_run_unit)

    summary_calls: list = []
    receipt_calls: list = []

    def build_summary(**kwargs):
        summary_calls.append(kwargs)
        seconds = float(kwargs["scientific_seconds"])
        counters = kwargs["counters"]
        return {
            "status": "complete",
            "command": "run_recovery_development",
            "recovery_plan_id": kwargs["recovery_plan_id"],
            "units_completed": kwargs["units_completed"],
            "selector_records": kwargs["selector_records"],
            "new_started": counters["new_started"],
            "new_completed": counters["new_completed"],
            "reused_completed": counters["reused_completed"],
            "replay_started": counters["replay_started"],
            "unstarted_started": counters["unstarted_started"],
            "live_bytes": kwargs["live_bytes"],
            "scientific_seconds_this_stage": seconds,
            "scientific_seconds_cumulative_bound": PRIOR_CHARGE_SECONDS + seconds,
        }

    def build_receipt(**kwargs):
        receipt_calls.append(kwargs)
        seconds = float(kwargs["scientific_seconds"])
        return {
            "status": "complete",
            "stage_manifest_sha256": kwargs["stage_manifest_sha256"],
            "scientific_seconds_this_stage": seconds,
            "scientific_seconds_cumulative_bound": PRIOR_CHARGE_SECONDS + seconds,
        }

    monkeypatch.setattr(rd.receipt, "build_summary", build_summary)
    monkeypatch.setattr(rd.receipt, "build_receipt", build_receipt)
    monkeypatch.setattr(
        rd.receipt,
        "validate_pair",
        lambda summary, recovery_receipt: {"status": "complete", "validated": True},
    )

    if receipt_raises:
        original_write = rd._write_new_file

        def failing_write(path, payload, budget):
            if Path(path).name == rd.receipt.RECEIPT_NAME:
                raise rd.RecoveryDevelopmentError("receipt_write_failed")
            return original_write(path, payload, budget)

        monkeypatch.setattr(rd, "_write_new_file", failing_write)

    if preexisting:
        (original_run_root / rd.persistence.RECOVERIES_DIRNAME).mkdir(parents=True)

    return {
        "artifact": artifact,
        "project_root": project_root,
        "contract_path": tmp_path / "contract.json",
        "base_permit_path": tmp_path / "base_permit.json",
        "recovery_permit_path": tmp_path / "recovery_permit.json",
        "original_run_root": original_run_root,
        "original_stage": original_stage,
        "unit_calls": unit_calls,
        "summary_calls": summary_calls,
        "receipt_calls": receipt_calls,
        "calls": calls,
        "unit_order": [unit_id for unit_id, _slots in ordered],
        "plan": plan,
        "base": base,
    }


def _run(env):
    return rd.run_recovery_development(
        project_root=env["project_root"],
        artifact_root=env["artifact"],
        contract_path=env["contract_path"],
        base_permit_path=env["base_permit_path"],
        recovery_permit_path=env["recovery_permit_path"],
        device="cuda",
    )


def test_success_orders_units_and_writes_private_receipt(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path)

    result = _run(env)

    assert result == {"status": "complete", "validated": True}
    assert [str(call["unit"]["unit_id"]) for call in env["unit_calls"]] == env["unit_order"]
    assert env["calls"]["closure"] == ["original", "leases"]
    assert env["calls"]["post_reauth"] == [True]
    assert len(env["summary_calls"]) == 1
    assert len(env["receipt_calls"]) == 1
    summary_call = env["summary_calls"][0]
    assert summary_call["units_completed"] == NEW_UNIT_COUNT
    assert summary_call["selector_records"] == TOTAL_SELECTORS
    counters = summary_call["counters"]
    assert counters["new_started"] == RECOVERY_FITS
    assert counters["new_completed"] == RECOVERY_FITS
    assert counters["reused_completed"] == REUSED_COMPLETED
    assert counters["replay_started"] == REPLAY_STARTED
    assert counters["unstarted_started"] == UNSTARTED_STARTED
    assert counters["new_optimizer_steps"] == RECOVERY_FITS * UPDATES_PER_FIT
    assert counters["reused_optimizer_steps"] == REUSED_COMPLETED * UPDATES_PER_FIT
    assert env["receipt_calls"][0]["scientific_seconds"] >= 120.0

    stage = _stage_path(env)
    receipt_path = env["original_run_root"] / rd.receipt.RECEIPT_NAME
    assert (stage / "summary.json").is_file()
    assert receipt_path.is_file()
    assert receipt_path.parent == env["original_run_root"]
    assert not str(receipt_path).startswith(str(stage))
    assert (
        env["original_stage"] / "preserved-original.bin"
    ).read_bytes() == b"original-evidence-sentinel"


def test_failed_unit_stops_without_retry_and_marks_failure(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, behavior="fail_second")

    with pytest.raises(rd.RecoveryDevelopmentError) as excinfo:
        _run(env)

    assert excinfo.value.reason_code == "unit_worker_failed"
    assert len(env["unit_calls"]) == 2
    marker = json.loads((_stage_path(env) / "failure.json").read_text())
    assert marker["reason_code"] == "unit_worker_failed"
    assert marker["status"] == "fail"
    assert (
        env["original_stage"] / "preserved-original.bin"
    ).read_bytes() == b"original-evidence-sentinel"


def test_postclosure_reauth_failure_marks_failure(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, closure_raises=True)

    with pytest.raises(rd.RecoveryDevelopmentError) as excinfo:
        _run(env)

    assert excinfo.value.reason_code == "original_stage_evidence_changed"
    stage = _stage_path(env)
    assert (stage / "failure.json").is_file()
    assert not (stage / "summary.json").exists()


def test_receipt_write_failure_preserves_summary_and_marks_failure(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, receipt_write_raises=True)

    with pytest.raises(rd.RecoveryDevelopmentError) as excinfo:
        _run(env)

    assert excinfo.value.reason_code == "receipt_write_failed"
    stage = _stage_path(env)
    assert (stage / "summary.json").is_file()
    marker = json.loads((stage / "failure.json").read_text())
    assert marker["reason_code"] == "receipt_write_failed"
    assert not (env["original_run_root"] / rd.receipt.RECEIPT_NAME).exists()


def test_preexisting_recoveries_fails_before_worker(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, preexisting_recoveries=True)

    with pytest.raises(rd.RecoveryDevelopmentError) as excinfo:
        _run(env)

    assert excinfo.value.reason_code == "recovery_container_exists"
    assert env["unit_calls"] == []
    recovery_root = (
        env["original_run_root"]
        / rd.persistence.RECOVERIES_DIRNAME
        / rd.authority.RECOVERY_PERMIT_SHA256
    )
    assert not recovery_root.exists()


def test_low_launch_resources_fail_before_writes(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, launch_resources_raises=True)

    with pytest.raises(rd.RecoveryDevelopmentError) as excinfo:
        _run(env)

    assert excinfo.value.reason_code == "insufficient_launch_resources"
    assert env["unit_calls"] == []
    assert not (env["original_run_root"] / rd.persistence.RECOVERIES_DIRNAME).exists()


def test_deadline_exceeded(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, deadline_raises=True)

    with pytest.raises(rd.RecoveryDevelopmentError) as excinfo:
        _run(env)

    assert excinfo.value.reason_code == "global_deadline_exceeded"
    assert env["unit_calls"] == []


def test_changed_unit_result_fails_with_marker(monkeypatch, tmp_path):
    env = _install(monkeypatch, tmp_path, behavior="wrong_result")

    with pytest.raises(rd.RecoveryDevelopmentError) as excinfo:
        _run(env)

    assert excinfo.value.reason_code == "unit_result_identity_mismatch"
    marker = json.loads((_stage_path(env) / "failure.json").read_text())
    assert marker["reason_code"] == "unit_result_identity_mismatch"
