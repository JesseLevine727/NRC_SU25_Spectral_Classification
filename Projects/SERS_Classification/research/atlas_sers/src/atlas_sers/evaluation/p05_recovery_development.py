"""P05 recovery source-only development runner (outer serial boundary).

``run_recovery_development`` is the single serial recovery boundary for the
interrupted P05 comprehensive source benchmark.  It authenticates the original
interruption prefix and the recovery permit in place, re-authenticates the 36
reused pilot fits, then processes the 1,242 canonical non-pilot units in
original order: the 726 sealed units are independently copied, the single
partial unit replays exactly one interrupted fit, and the remaining unstarted
slots are fit once.  It performs no selection, refit, calibration or outer-test
evaluation, never fabricates a clean-run receipt and stops immediately on any
resource, deadline or storage breach.

The module imports only the standard library and the existing stdlib-only P05
boundary modules at load time; torch and numpy are imported lazily inside the
boundary and never at module scope.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_inputs as recovery_inputs
from atlas_sers.evaluation import p05_recovery_persistence as persistence
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan
from atlas_sers.evaluation import p05_recovery_receipt as receipt
from atlas_sers.evaluation import p05_recovery_unit as unit_runner
from atlas_sers.evaluation.p05_comprehensive_storage import (
    StorageBudget,
)

__all__ = [
    "RecoveryDevelopmentError",
    "SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "CLAIM",
    "STORAGE_CEILING_BYTES",
    "PRIOR_CHARGE_SECONDS",
    "run_recovery_development",
]

SCHEMA_VERSION = "nato-sers-p05-recovery-development-v1"
PROTOCOL_VERSION = authority.RECOVERY_PROTOCOL_VERSION
CLAIM = receipt.CLAIM

STORAGE_CEILING_BYTES = authority.PRIVATE_STORAGE_CEILING_BYTES
MAXIMUM_TOTAL_SECONDS = authority.MAXIMUM_TOTAL_SECONDS
PRIOR_CHARGE_SECONDS = (
    authority.PRIOR_SCIENTIFIC_SECONDS_CHARGED + authority.PRELAUNCH_AUDIT_RESERVE_SECONDS
)

UNIT_BUDGET_HEADROOM_BYTES = 8 * 1024 * 1024
_MAX_LEASE_BYTES = 64 * 1024
RECEIPT_FINALIZATION_ALLOWANCE_SECONDS = 120.0

_EXPECTED_COUNTER_KEYS = frozenset(
    {
        "new_started",
        "new_completed",
        "new_failed",
        "new_optimizer_steps",
        "new_optimizer_steps_exact",
        "new_elapsed_seconds",
        "new_peak_cuda_bytes",
        "reused_completed",
        "reused_optimizer_steps",
        "replay_started",
        "unstarted_started",
    }
)

_LEASE_KEYS = frozenset(
    {
        "slot_id",
        "unit_id",
        "recipe_id",
        "seed",
        "contract_sha256",
        "core_plan_id",
        "permit_sha256",
    }
)


class RecoveryDevelopmentError(ValueError):
    """Stable, path-free recovery-development failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _initial_counters() -> dict[str, Any]:
    return {
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


def _check_counters(counters: Any) -> dict[str, Any]:
    if not isinstance(counters, dict) or set(counters) != _EXPECTED_COUNTER_KEYS:
        raise RecoveryDevelopmentError("counters_malformed")
    try:
        unit_runner._validate_counters(counters)
    except unit_runner.RecoveryUnitError as error:
        raise RecoveryDevelopmentError(error.reason_code) from error
    return counters


def _check_deadline(deadline: Any) -> None:
    try:
        recovery_inputs._check_deadline(deadline)
    except recovery_inputs.RecoveryInputsError as error:
        raise RecoveryDevelopmentError(error.reason_code) from error


def _write_new_file(path: Path, payload: bytes, budget: StorageBudget) -> None:
    recovery_inputs._reject_symlink_chain(path)
    budget._require_inside_run(path)
    budget.check(headroom_bytes=len(payload))
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as error:
        raise RecoveryDevelopmentError("recovery_file_exists") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    budget.account_new_file(path)


def _write_progress(
    path: Path, counters: Mapping[str, Any], units_completed: int, units_total: int
) -> None:
    core._atomic_write(
        path,
        core._canon().canonical_json_bytes(
            {
                "new_started": counters["new_started"],
                "new_completed": counters["new_completed"],
                "new_failed": counters["new_failed"],
                "new_optimizer_steps": counters["new_optimizer_steps"],
                "new_optimizer_steps_exact": counters["new_optimizer_steps_exact"],
                "new_elapsed_seconds": counters["new_elapsed_seconds"],
                "new_peak_cuda_bytes": counters["new_peak_cuda_bytes"],
                "reused_completed": counters["reused_completed"],
                "reused_optimizer_steps": counters["reused_optimizer_steps"],
                "replay_started": counters["replay_started"],
                "unstarted_started": counters["unstarted_started"],
                "units_completed": int(units_completed),
                "units_total": int(units_total),
            }
        ),
    )


def _create_exclusive(*paths: Path) -> None:
    for path in paths:
        core._reject_symlink_chain(path)
        if path.exists() or path.is_symlink():
            raise RecoveryDevelopmentError("recovery_container_exists")
        core._mkdir_exclusive(path, "recovery_container_exists")


def _check_plan(plan: Any) -> None:
    if not isinstance(plan, Mapping):
        raise RecoveryDevelopmentError("recovery_plan_malformed")
    if str(plan.get("ledger_id")) != recovery_plan.LEDGER_ID:
        raise RecoveryDevelopmentError("recovery_plan_identity_mismatch")
    if str(plan.get("base_permit_sha256")) != authority.BASECOMPREHENSIVE_PERMIT_SHA256:
        raise RecoveryDevelopmentError("recovery_plan_identity_mismatch")
    if plan.get("execution_authorized") is not False:
        raise RecoveryDevelopmentError("recovery_plan_authorization_invalid")
    if type(plan.get("fits_started")) is not int or plan.get("fits_started") != 0:
        raise RecoveryDevelopmentError("recovery_plan_authorization_invalid")
    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not core._is_hex64(plan_id):
        raise RecoveryDevelopmentError("recovery_plan_identity_mismatch")
    if (
        core._canon().sha256_value({key: value for key, value in plan.items() if key != "plan_id"})
        != plan_id
    ):
        raise RecoveryDevelopmentError("recovery_plan_digest_mismatch")


def _read_original_selectors(
    stage: Path, permit: Mapping[str, Any], deadline: Any
) -> dict[str, Mapping[str, Any]]:
    raw = recovery_inputs._read_bytes_bounded(
        stage / "selector.jsonl",
        recovery_inputs._MAX_JSON_BYTES,
        "original_selector",
        deadline,
    )
    if hashlib.sha256(raw).hexdigest() != str(permit.get("original_selector_sha256")):
        raise RecoveryDevelopmentError("original_selector_digest_mismatch")
    records = recovery_inputs._parse_jsonl(raw, "original_selector")
    by_slot: dict[str, Mapping[str, Any]] = {}
    for record in records:
        slot_id = str(record.get("slot_id"))
        if slot_id in by_slot:
            raise RecoveryDevelopmentError("original_selector_duplicate")
        by_slot[slot_id] = record
    return by_slot


def _check_pilot_selectors(
    pilot_records: Any, original_by_slot: Mapping[str, Mapping[str, Any]]
) -> None:
    canon = core._canon()
    for record in pilot_records:
        slot_id = str(record["slot_id"])
        original = original_by_slot.get(slot_id)
        if original is None or canon.sha256_value(record) != canon.sha256_value(original):
            raise RecoveryDevelopmentError("pilot_selector_mismatch")


def _input_manifest(
    base: Mapping[str, Any], plan: Mapping[str, Any], recovery_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "base_comprehensive_permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "core_contract_sha256": str(base["contract_sha256"]),
        "core_plan_id": str(base["core_plan_id"]),
        "ledger_id": str(base["ledger"]["ledger_id"]),
        "recovery_plan_id": str(plan["plan_id"]),
        "original_evidence_anchor_sha256": str(recovery_bundle["original_anchor_sha256"]),
        "reused_pilot_slots": development.REUSED_PILOT_SLOTS,
        "nonpilot_units": recovery_plan.NEW_UNIT_COUNT,
        "new_fits": development.MAXIMUM_NEW_FITS,
        "recovery_fits": recovery_plan.RECOVERY_FITS,
        "receipt_finalization_allowance_seconds": RECEIPT_FINALIZATION_ALLOWANCE_SECONDS,
        "receipt_stage_seconds_are_charged_upper_bound": True,
        "total_selector_records": (development.REUSED_PILOT_SLOTS + development.MAXIMUM_NEW_FITS),
    }


def _validate_unit_result(
    result: Any,
    unit_id: str,
    unit_slot_ids: set[str],
    prior: Mapping[str, Any],
    counters: Mapping[str, Any],
) -> None:
    if not isinstance(result, Mapping):
        raise RecoveryDevelopmentError("unit_result_malformed")
    if str(result.get("unit_id")) != unit_id:
        raise RecoveryDevelopmentError("unit_result_identity_mismatch")
    completed = [str(value) for value in result.get("completed_slots", ())]
    if len(set(completed)) != len(completed):
        raise RecoveryDevelopmentError("unit_duplicate_slots")
    if set(completed) != set(unit_slot_ids):
        raise RecoveryDevelopmentError("unit_slot_coverage_mismatch")
    reused = [str(value) for value in result.get("reused_slots", ())]
    new = [str(value) for value in result.get("new_slots", ())]
    replay = result.get("replay_slot_id")
    replay_ids = [] if replay is None else [str(replay)]
    if len(set(reused)) != len(reused) or len(set(new)) != len(new):
        raise RecoveryDevelopmentError("unit_duplicate_slots")
    if set(reused) & set(new) or not set(replay_ids) <= set(new):
        raise RecoveryDevelopmentError("unit_slot_kind_overlap")
    if set(reused) | set(new) != set(completed):
        raise RecoveryDevelopmentError("unit_slot_kind_mismatch")
    kind = result.get("unit_kind")
    expected_counts = {"sealed": (12, 0, 0), "partial": (8, 4, 1), "new": (0, 12, 0)}
    if (
        kind not in expected_counts
        or (len(reused), len(new), len(replay_ids)) != expected_counts[kind]
    ):
        raise RecoveryDevelopmentError("unit_kind_mismatch")
    for key in ("new_started", "new_completed", "new_failed", "fits_started"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecoveryDevelopmentError("unit_result_malformed")
    for key in ("optimizer_updates_reused", "optimizer_updates_new"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecoveryDevelopmentError("unit_result_malformed")
    if result["new_started"] != result["new_completed"] + result["new_failed"]:
        raise RecoveryDevelopmentError("unit_result_inconsistent")
    if result["new_failed"] != 0 or result["fits_started"] != result["new_started"]:
        raise RecoveryDevelopmentError("unit_result_inconsistent")
    if counters["new_started"] - prior["new_started"] != result["new_started"]:
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if counters["new_completed"] - prior["new_completed"] != result["new_completed"]:
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if counters["new_failed"] - prior["new_failed"] != result["new_failed"]:
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if counters["reused_completed"] - prior["reused_completed"] != len(reused):
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if (
        counters["reused_optimizer_steps"] - prior["reused_optimizer_steps"]
        != result["optimizer_updates_reused"]
    ):
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if (
        counters["new_optimizer_steps"] - prior["new_optimizer_steps"]
        != result["optimizer_updates_new"]
    ):
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if counters["new_started"] - prior["new_started"] != len(new):
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if counters["replay_started"] - prior["replay_started"] != len(replay_ids):
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    if counters["unstarted_started"] - prior["unstarted_started"] != len(new) - len(replay_ids):
        raise RecoveryDevelopmentError("unit_counter_mismatch")
    _check_counters(dict(counters))


def _check_final_counts(
    counters: Mapping[str, Any],
    units_completed: int,
    selector_seen: set[str],
    all_slot_ids: set[str],
    reused_slot_ids: set[str],
) -> None:
    _check_counters(dict(counters))
    if units_completed != recovery_plan.NEW_UNIT_COUNT:
        raise RecoveryDevelopmentError("recovery_unit_count_mismatch")
    if counters["new_started"] != recovery_plan.RECOVERY_FITS:
        raise RecoveryDevelopmentError("recovery_started_mismatch")
    if counters["new_completed"] != recovery_plan.RECOVERY_FITS:
        raise RecoveryDevelopmentError("recovery_completed_mismatch")
    if counters["new_failed"] != 0:
        raise RecoveryDevelopmentError("recovery_failed_present")
    if counters["reused_completed"] != recovery_plan.ORIGINAL_COMPLETED:
        raise RecoveryDevelopmentError("reused_completed_mismatch")
    if counters["replay_started"] != recovery_plan.ORIGINAL_INTERRUPTED:
        raise RecoveryDevelopmentError("replay_count_mismatch")
    if counters["unstarted_started"] != recovery_plan.ORIGINAL_UNSTARTED:
        raise RecoveryDevelopmentError("unstarted_count_mismatch")
    if counters["replay_started"] + counters["unstarted_started"] != counters["new_started"]:
        raise RecoveryDevelopmentError("recovery_counter_inconsistent")
    if counters["new_optimizer_steps_exact"] is not True:
        raise RecoveryDevelopmentError("optimizer_steps_inexact")
    if len(reused_slot_ids) != recovery_plan.ORIGINAL_COMPLETED:
        raise RecoveryDevelopmentError("reused_slot_count_mismatch")
    expected_selector_count = development.REUSED_PILOT_SLOTS + development.MAXIMUM_NEW_FITS
    if len(selector_seen) != expected_selector_count:
        raise RecoveryDevelopmentError("selector_count_mismatch")
    if selector_seen != set(all_slot_ids):
        raise RecoveryDevelopmentError("selector_coverage_mismatch")


def _read_recovery_selectors(selector_path: Path, deadline: Any) -> list[dict[str, Any]]:
    raw = recovery_inputs._read_bytes_bounded(
        selector_path,
        recovery_inputs._MAX_JSON_BYTES,
        "recovery_selector",
        deadline,
    )
    return recovery_inputs._parse_jsonl(raw, "recovery_selector")


def _check_selector_coverage(
    records: Any,
    all_slot_ids: set[str],
    slot_by_id: Mapping[str, Mapping[str, Any]],
    unit_by_id: Mapping[str, Mapping[str, Any]],
    pilot_records: Any,
    original_by_slot: Mapping[str, Mapping[str, Any]],
    reused_slot_ids: set[str],
    *,
    stage: Path,
) -> None:
    canon = core._canon()
    expected_total = development.REUSED_PILOT_SLOTS + development.MAXIMUM_NEW_FITS
    if len(records) != expected_total:
        raise RecoveryDevelopmentError("recovery_selector_count_mismatch")
    by_slot: dict[str, Mapping[str, Any]] = {}
    for record in records:
        slot_id = str(record.get("slot_id"))
        if slot_id in by_slot:
            raise RecoveryDevelopmentError("recovery_selector_duplicate")
        by_slot[slot_id] = record
    if set(by_slot) != set(all_slot_ids):
        raise RecoveryDevelopmentError("recovery_selector_coverage_mismatch")
    for record in pilot_records:
        slot_id = str(record["slot_id"])
        observed = by_slot.get(slot_id)
        if observed is None or canon.sha256_value(record) != canon.sha256_value(observed):
            raise RecoveryDevelopmentError("recovery_pilot_selector_mismatch")
    for slot_id in reused_slot_ids:
        original = original_by_slot.get(slot_id)
        observed = by_slot.get(slot_id)
        if original is None or observed is None:
            raise RecoveryDevelopmentError("recovery_reused_selector_mismatch")
        if canon.sha256_value(original) != canon.sha256_value(observed):
            raise RecoveryDevelopmentError("recovery_reused_selector_mismatch")
    pilot_slot_ids = {str(row["slot_id"]) for row in pilot_records}
    for slot_id, record in by_slot.items():
        slot = slot_by_id.get(slot_id)
        if slot is None:
            raise RecoveryDevelopmentError("recovery_selector_slot_unknown")
        unit = unit_by_id.get(str(slot["unit_id"]))
        if unit is None:
            raise RecoveryDevelopmentError("recovery_selector_unit_unknown")
        if record.get("status") != "complete":
            raise RecoveryDevelopmentError("recovery_selector_not_complete")
        if slot_id in pilot_slot_ids:
            continue
        summary = base_inputs._read_pilot_summary(
            stage / "units" / str(unit["unit_id"]), unit, slot
        )
        expected = base_inputs.selector_record(unit, slot, summary)
        if canon.canonical_json_bytes(expected) != canon.canonical_json_bytes(record):
            raise RecoveryDevelopmentError("recovery_selector_summary_mismatch")


def _verify_unit_manifests(stage: Path, inventory: Mapping[str, Any], deadline: float) -> None:
    """Bind each unit's accepted manifest, then verify its referenced bytes."""

    for unit_id, record in inventory.items():
        recovery_inputs._check_component(unit_id)
        _check_deadline(deadline)
        unit_dir = stage / "units" / unit_id
        if recovery_inputs._hash_file_record(unit_dir / "manifest.json", deadline) != record:
            raise RecoveryDevelopmentError("unit_manifest_changed")
        pilot._verify_manifest(unit_dir)
        _check_deadline(deadline)


def _reauthenticate_original(
    original_stage: Path,
    base: Mapping[str, Any],
    plan: Mapping[str, Any],
    recovery_bundle: Mapping[str, Any],
    deadline: Any,
) -> None:
    expected_files, expected_dirs, _sm, _suf, _uf = recovery_inputs._expected_layout(base, plan)
    observed_files, observed_dirs = recovery_inputs._collect_tree(original_stage, deadline)
    if set(observed_files) != set(expected_files):
        raise RecoveryDevelopmentError("original_stage_inventory_changed")
    if set(observed_dirs) != set(expected_dirs):
        raise RecoveryDevelopmentError("original_stage_directory_changed")
    inventory = recovery_bundle["original_inventory"]
    if set(inventory) != set(expected_files):
        raise RecoveryDevelopmentError("original_inventory_incomplete")
    for relative in sorted(inventory):
        observed = recovery_inputs._hash_file_record(observed_files[relative], deadline)
        if observed != inventory[relative]:
            raise RecoveryDevelopmentError("original_stage_evidence_changed")


def _check_unstarted_lease(
    lease: Any,
    slot_id: str,
    slot_by_id: Mapping[str, Mapping[str, Any]],
    base: Mapping[str, Any],
) -> None:
    if not isinstance(lease, Mapping) or set(lease) != _LEASE_KEYS:
        raise RecoveryDevelopmentError("recovery_lease_malformed")
    slot = slot_by_id.get(slot_id)
    if slot is None:
        raise RecoveryDevelopmentError("recovery_lease_slot_unknown")
    if str(lease.get("slot_id")) != slot_id:
        raise RecoveryDevelopmentError("recovery_lease_identity_mismatch")
    if str(lease.get("unit_id")) != str(slot["unit_id"]):
        raise RecoveryDevelopmentError("recovery_lease_identity_mismatch")
    if str(lease.get("recipe_id")) != str(slot["recipe_id"]):
        raise RecoveryDevelopmentError("recovery_lease_identity_mismatch")
    seed = lease.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != int(slot["seed"]):
        raise RecoveryDevelopmentError("recovery_lease_identity_mismatch")
    if str(lease.get("contract_sha256")) != str(base["contract_sha256"]):
        raise RecoveryDevelopmentError("recovery_lease_contract_mismatch")
    if str(lease.get("core_plan_id")) != str(base["core_plan_id"]):
        raise RecoveryDevelopmentError("recovery_lease_plan_mismatch")
    if str(lease.get("permit_sha256")) != authority.BASECOMPREHENSIVE_PERMIT_SHA256:
        raise RecoveryDevelopmentError("recovery_lease_permit_mismatch")


def _reauthenticate_leases(
    base: Mapping[str, Any],
    plan: Mapping[str, Any],
    recovery_bundle: Mapping[str, Any],
    slot_by_id: Mapping[str, Mapping[str, Any]],
    deadline: Any,
) -> None:
    lease_root = base_inputs._slot_lease_root(base["artifact_root"])
    original_inventory = recovery_bundle["original_lease_inventory"]
    for relative, record in original_inventory.items():
        observed = recovery_inputs._hash_file_record(lease_root / relative, deadline)
        if observed != record:
            raise RecoveryDevelopmentError("original_lease_changed")
    lease_files, lease_dirs = recovery_inputs._collect_tree(lease_root, deadline)
    unstarted = [str(value) for value in plan["unstarted_slot_ids"]]
    expected_paths = set(original_inventory)
    for slot_id in unstarted:
        expected_paths.add(f"{slot_id}/lease.json")
    if set(lease_files) != expected_paths:
        raise RecoveryDevelopmentError("recovery_lease_inventory_mismatch")
    if lease_dirs != {str(Path(relative).parent) for relative in expected_paths}:
        raise RecoveryDevelopmentError("recovery_lease_directory_mismatch")
    if len(lease_files) != recovery_plan.LEASE_COUNT + recovery_plan.ORIGINAL_UNSTARTED:
        raise RecoveryDevelopmentError("recovery_lease_count_mismatch")
    for slot_id in unstarted:
        relative = f"{slot_id}/lease.json"
        raw = recovery_inputs._read_bytes_bounded(
            lease_files[relative], _MAX_LEASE_BYTES, "recovery_lease", deadline
        )
        lease = recovery_inputs._parse_json(raw, "recovery_lease")
        _check_unstarted_lease(lease, slot_id, slot_by_id, base)


def _check_replay_lease(
    recovery_root: Path,
    base: Mapping[str, Any],
    plan: Mapping[str, Any],
    recovery_bundle: Mapping[str, Any],
    slot_by_id: Mapping[str, Mapping[str, Any]],
    deadline: Any,
) -> None:
    path = recovery_root / persistence.REPLAY_LEASE_NAME
    raw = recovery_inputs._read_bytes_bounded(path, _MAX_LEASE_BYTES, "replay_lease", deadline)
    lease = recovery_inputs._parse_json(raw, "replay_lease")
    interrupted = str(plan["interrupted_slot_id"])
    slot = slot_by_id.get(interrupted)
    if slot is None:
        raise RecoveryDevelopmentError("replay_lease_slot_unknown")
    original_record = recovery_bundle["original_lease_inventory"].get(f"{interrupted}/lease.json")
    if not isinstance(original_record, Mapping):
        raise RecoveryDevelopmentError("replay_lease_missing_original")
    expected = {
        "schema_version": persistence.REPLAY_LEASE_SCHEMA_VERSION,
        "attempt_kind": persistence.REPLAY_LEASE_ATTEMPT_KIND,
        "base_comprehensive_permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "core_contract_sha256": str(base["contract_sha256"]),
        "core_plan_id": str(base["core_plan_id"]),
        "slot_id": interrupted,
        "unit_id": str(slot["unit_id"]),
        "recipe_id": str(slot["recipe_id"]),
        "seed": int(slot["seed"]),
        "original_lease_sha256": str(original_record["sha256"]),
    }
    canon = core._canon()
    if canon.canonical_json_bytes(lease) != canon.canonical_json_bytes(expected):
        raise RecoveryDevelopmentError("replay_lease_mismatch")


def _write_failure_summary(
    stage: Path,
    base: Mapping[str, Any],
    plan: Mapping[str, Any],
    counters: Mapping[str, Any],
    units_completed: int,
    error: BaseException,
    wall_start: float,
    budget: StorageBudget,
) -> None:
    elapsed = time.perf_counter() - wall_start
    ledger = base.get("ledger")
    ledger_id = str(ledger["ledger_id"]) if isinstance(ledger, Mapping) else None
    payload = {
        "status": "fail",
        "command": "run_recovery_development",
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "base_comprehensive_permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "core_contract_sha256": str(base.get("contract_sha256")),
        "core_plan_id": str(base.get("core_plan_id")),
        "ledger_id": ledger_id,
        "recovery_plan_id": str(plan.get("plan_id")),
        "reason_code": str(getattr(error, "reason_code", type(error).__name__)),
        "original_started": recovery_plan.ORIGINAL_STARTED,
        "original_completed": recovery_plan.ORIGINAL_COMPLETED,
        "original_interrupted": recovery_plan.ORIGINAL_INTERRUPTED,
        "original_unstarted": recovery_plan.ORIGINAL_UNSTARTED,
        "recovery_started": counters.get("new_started"),
        "recovery_completed": counters.get("new_completed"),
        "recovery_failed": counters.get("new_failed"),
        "reused_completed": counters.get("reused_completed"),
        "replay_started": counters.get("replay_started"),
        "unstarted_started": counters.get("unstarted_started"),
        "new_optimizer_steps": counters.get("new_optimizer_steps"),
        "new_optimizer_steps_exact": counters.get("new_optimizer_steps_exact"),
        "reused_optimizer_steps": counters.get("reused_optimizer_steps"),
        "interrupted_observed_lower_bound": recovery_plan.INTERRUPTED_UPDATES,
        "interrupted_charged_upper_bound": recovery_plan.INTERRUPTED_CHARGE,
        "original_completed_updates": recovery_plan.ORIGINAL_COMPLETED_UPDATES,
        "original_charged_upper_bound": recovery_plan.ORIGINAL_CHARGED_UPPER_BOUND,
        "prior_scientific_seconds_charged": authority.PRIOR_SCIENTIFIC_SECONDS_CHARGED,
        "prelaunch_audit_reserve_seconds": authority.PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "elapsed_seconds": elapsed,
        "scientific_seconds_cumulative_bound": elapsed + PRIOR_CHARGE_SECONDS,
        "units_completed": int(units_completed),
        "units_total": recovery_plan.NEW_UNIT_COUNT,
        "claim": CLAIM,
        "source_fits_only": True,
        "selection_authorized": False,
        "refit_authorized": False,
        "calibration_authorized": False,
        "outer_evaluation_authorized": False,
    }
    # A new marker preserves any already-written summary and invalidates a
    # prematurely completed stage manifest if final receipt persistence fails.
    _write_new_file(stage / "failure.json", core._canon().canonical_json_bytes(payload), budget)


def _best_effort_failure(ctx: Mapping[str, Any], error: BaseException, wall_start: float) -> None:
    stage = ctx.get("stage")
    base = ctx.get("base")
    plan = ctx.get("plan")
    budget = ctx.get("budget")
    if stage is None or base is None or plan is None or budget is None:
        return
    try:
        _write_failure_summary(
            stage,
            base,
            plan,
            ctx["counters"],
            ctx["units_completed"],
            error,
            wall_start,
            budget,
        )
    except Exception:
        pass


def _run_recovery(
    *,
    project_root: Path,
    artifact_root: Path,
    contract_path: Path,
    base_permit_path: Path,
    recovery_permit_path: Path,
    device: str,
    wall_start: float,
    deadline: float,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    try:
        recovery_bundle = recovery_inputs.prepare_recovery(
            project_root=project_root,
            artifact_root=artifact_root,
            contract_path=contract_path,
            base_permit_path=base_permit_path,
            recovery_permit_path=recovery_permit_path,
            deadline=deadline,
        )
    except recovery_inputs.RecoveryInputsError as error:
        raise RecoveryDevelopmentError(error.reason_code) from error

    base = recovery_bundle["base_bundle"]
    plan = recovery_bundle["plan"]
    permit = recovery_bundle["recovery_permit"]
    artifact = Path(base["artifact_root"])
    original_run_root = Path(recovery_bundle["original_run_root"])
    original_stage = Path(recovery_bundle["original_stage"])
    ctx["base"] = base
    ctx["plan"] = plan
    counters = _check_counters(ctx["counters"])

    _check_plan(plan)
    _check_deadline(deadline)

    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    pilot._development_kernel()
    if not bool(torch.cuda.is_available()):
        raise RecoveryDevelopmentError("cuda_unavailable")
    authority.check_resources(torch, phase="launch")
    pilot._checkpoint_preflight(torch, artifact)
    provenance_before = core._capture_provenance(
        base["repository_root"], base["project_root"], artifact
    )
    _check_deadline(deadline)

    pilot_records = base_inputs.import_pilot(base, device=device)
    if len(pilot_records) != development.REUSED_PILOT_SLOTS:
        raise RecoveryDevelopmentError("pilot_record_count_mismatch")
    if {str(record["slot_id"]) for record in pilot_records} != base_inputs.pilot_slot_ids(base):
        raise RecoveryDevelopmentError("pilot_slot_coverage_mismatch")
    original_by_slot = _read_original_selectors(original_stage, permit, deadline)
    _check_pilot_selectors(pilot_records, original_by_slot)

    unit_by_id, ordered, all_slot_ids = development._plan_units(base)
    slot_by_id = {str(slot["slot_id"]): slot for slot in base["ledger"]["slots"]}

    receipt_path = original_run_root / receipt.RECEIPT_NAME
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RecoveryDevelopmentError("recovery_receipt_exists")

    recoveries_dir = original_run_root / persistence.RECOVERIES_DIRNAME
    recovery_root = recoveries_dir / authority.RECOVERY_PERMIT_SHA256
    stage = recovery_root / persistence.DEVELOP_STAGE_NAME
    units_root = stage / persistence.UNITS_DIRNAME
    _check_deadline(deadline)
    authority.check_resources(torch, phase="fit")
    _create_exclusive(recoveries_dir, recovery_root, stage)
    ctx["stage"] = stage
    _create_exclusive(units_root)

    budget = StorageBudget(artifact, original_run_root, ceiling=STORAGE_CEILING_BYTES)
    ctx["budget"] = budget
    events_path = stage / "events.jsonl"
    selector_path = stage / "selector.jsonl"
    progress_path = stage / "progress.json"
    budget.register_growing(events_path)
    budget.register_growing(selector_path)
    budget.register_growing(progress_path)

    development._write_ledger(stage, base, budget)
    development._write_source_ledger(stage, base, ordered, unit_by_id, budget)
    _write_new_file(stage / "plan.json", core._canon().canonical_json_bytes(plan), budget)
    _write_new_file(
        stage / "original_inventory.json",
        core._canon().canonical_json_bytes(recovery_bundle["original_inventory"]),
        budget,
    )
    _write_new_file(
        stage / "original_anchor.json",
        core._canon().canonical_json_bytes(recovery_bundle["original_anchor"]),
        budget,
    )
    _write_new_file(
        stage / "original_lease_inventory.json",
        core._canon().canonical_json_bytes(recovery_bundle["original_lease_inventory"]),
        budget,
    )
    _write_new_file(
        stage / "recovery_permit.json",
        core._canon().canonical_json_bytes(permit),
        budget,
    )
    _write_new_file(
        stage / "input_manifest.json",
        core._canon().canonical_json_bytes(_input_manifest(base, plan, recovery_bundle)),
        budget,
    )
    _write_new_file(
        stage / "provenance_before.json",
        core._canon().canonical_json_bytes(provenance_before),
        budget,
    )

    selector_seen: set[str] = set()
    reused_slot_ids: set[str] = set()
    unit_manifest_inventory: dict[str, Any] = {}
    for record in pilot_records:
        development._append_jsonl(selector_path, record)
        selector_seen.add(str(record["slot_id"]))

    units_total = len(ordered)
    for unit_id, unit_slots in ordered:
        _check_deadline(deadline)
        budget.check(headroom_bytes=UNIT_BUDGET_HEADROOM_BYTES)
        authority.check_resources(torch, phase="fit")
        unit = unit_by_id[unit_id]
        unit_slot_ids = {str(slot["slot_id"]) for slot in unit_slots}
        expected_selectors = {
            slot_id: original_by_slot[slot_id]
            for slot_id in unit_slot_ids
            if slot_id in original_by_slot
        }
        development._append_jsonl(
            events_path,
            {
                "event": "unit_started",
                "unit_id": unit_id,
                "slot_count": len(unit_slots),
            },
        )
        prior = dict(counters)
        result = unit_runner.run_unit(
            recovery_bundle=recovery_bundle,
            unit=unit,
            expected_selectors=expected_selectors,
            torch=torch,
            device=device,
            budget=budget,
            deadline=deadline,
            events_path=events_path,
            selector_path=selector_path,
            counters=counters,
        )
        _validate_unit_result(result, unit_id, unit_slot_ids, prior, counters)
        unit_manifest_inventory[unit_id] = recovery_inputs._hash_file_record(
            units_root / unit_id / "manifest.json", deadline
        )
        for slot_id in result["completed_slots"]:
            slot_id = str(slot_id)
            if slot_id in selector_seen:
                raise RecoveryDevelopmentError("selector_duplicate")
            selector_seen.add(slot_id)
        reused_slot_ids.update(str(value) for value in result["reused_slots"])
        ctx["units_completed"] += 1
        development._append_jsonl(
            events_path,
            {
                "event": "unit_completed",
                "unit_id": unit_id,
                "unit_kind": str(result["unit_kind"]),
                "counters": dict(counters),
                "units_completed": ctx["units_completed"],
            },
        )
        _write_progress(progress_path, counters, ctx["units_completed"], units_total)
        _check_deadline(deadline)
        budget.check(headroom_bytes=UNIT_BUDGET_HEADROOM_BYTES)
        authority.check_resources(torch, phase="fit")

    units_completed = ctx["units_completed"]
    _check_final_counts(counters, units_completed, selector_seen, all_slot_ids, reused_slot_ids)

    new_selector_records = _read_recovery_selectors(selector_path, deadline)
    _check_selector_coverage(
        new_selector_records,
        all_slot_ids,
        slot_by_id,
        unit_by_id,
        pilot_records,
        original_by_slot,
        reused_slot_ids,
        stage=stage,
    )
    if set(unit_manifest_inventory) != {unit_id for unit_id, _slots in ordered}:
        raise RecoveryDevelopmentError("unit_manifest_coverage_mismatch")
    _verify_unit_manifests(stage, unit_manifest_inventory, deadline)
    _write_new_file(
        stage / "unit_manifest_inventory.json",
        core._canon().canonical_json_bytes(unit_manifest_inventory),
        budget,
    )
    _reauthenticate_original(original_stage, base, plan, recovery_bundle, deadline)
    _reauthenticate_leases(base, plan, recovery_bundle, slot_by_id, deadline)
    _check_replay_lease(recovery_root, base, plan, recovery_bundle, slot_by_id, deadline)

    base_inputs.prepare(
        project_root=project_root,
        artifact_root=artifact,
        contract_path=contract_path,
        permit_path=base_permit_path,
        require_unstarted=False,
    )
    _check_deadline(deadline)
    provenance_after = pilot._post_run_reauth(
        artifact,
        base["contract"],
        base["support"],
        provenance_before,
        base["repository_root"],
        base["project_root"],
    )
    _check_deadline(deadline)
    authority.check_resources(torch, phase="fit")
    _write_new_file(
        stage / "provenance_after.json",
        core._canon().canonical_json_bytes(provenance_after),
        budget,
    )

    scientific_seconds = time.perf_counter() - wall_start
    if scientific_seconds + PRIOR_CHARGE_SECONDS > authority.MAXIMUM_TOTAL_SECONDS:
        raise RecoveryDevelopmentError("total_wall_exceeded")
    live_bytes = budget.check()
    summary = receipt.build_summary(
        base_bundle=base,
        recovery_plan_id=str(plan["plan_id"]),
        counters=counters,
        units_completed=units_completed,
        selector_records=len(new_selector_records),
        scientific_seconds=scientific_seconds,
        live_bytes=live_bytes,
    )
    _write_new_file(
        stage / "summary.json",
        core._canon().canonical_json_bytes(summary),
        budget,
    )
    core._write_manifest(stage)
    budget.account_new_file(stage / "manifest.json")
    pilot._verify_manifest(stage)
    budget.reconcile()

    stage_manifest_sha256 = core._canon().sha256_file(stage / "manifest.json")
    current_elapsed_seconds = time.perf_counter() - wall_start
    charged_stage_seconds = current_elapsed_seconds + RECEIPT_FINALIZATION_ALLOWANCE_SECONDS
    if PRIOR_CHARGE_SECONDS + charged_stage_seconds > MAXIMUM_TOTAL_SECONDS:
        raise RecoveryDevelopmentError("charged_total_wall_exceeded")
    recovery_receipt = receipt.build_receipt(
        summary=summary,
        stage_manifest_sha256=stage_manifest_sha256,
        scientific_seconds=charged_stage_seconds,
    )
    _check_deadline(deadline)
    authority.check_resources(torch, phase="fit")
    _write_new_file(
        receipt_path,
        core._canon().canonical_json_bytes(recovery_receipt),
        budget,
    )
    budget.check(headroom_bytes=0)
    _check_deadline(deadline)
    authority.check_resources(torch, phase="fit")
    final_elapsed_seconds = time.perf_counter() - wall_start
    if final_elapsed_seconds > charged_stage_seconds:
        raise RecoveryDevelopmentError("stage_finalization_allowance_exceeded")
    return receipt.validate_pair(summary, recovery_receipt)


def run_recovery_development(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    base_permit_path: Path | str,
    recovery_permit_path: Path | str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the single serial P05 recovery source-development boundary."""

    core._configure_environment()
    wall_start = time.perf_counter()
    deadline = wall_start + (MAXIMUM_TOTAL_SECONDS - PRIOR_CHARGE_SECONDS)
    ctx: dict[str, Any] = {
        "counters": _initial_counters(),
        "units_completed": 0,
        "units_total": recovery_plan.NEW_UNIT_COUNT,
        "stage": None,
        "base": None,
        "plan": None,
    }
    try:
        for raw_path in (
            project_root,
            artifact_root,
            contract_path,
            base_permit_path,
            recovery_permit_path,
        ):
            recovery_inputs._reject_symlink_chain(raw_path)
        if device != "cuda":
            raise RecoveryDevelopmentError("device_invalid")
        return _run_recovery(
            project_root=Path(project_root),
            artifact_root=Path(artifact_root),
            contract_path=Path(contract_path),
            base_permit_path=Path(base_permit_path),
            recovery_permit_path=Path(recovery_permit_path),
            device=str(device),
            wall_start=wall_start,
            deadline=deadline,
            ctx=ctx,
        )
    except BaseException as error:
        _best_effort_failure(ctx, error, wall_start)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, RecoveryDevelopmentError):
            raise
        reason_code = getattr(error, "reason_code", None)
        if not isinstance(reason_code, str) or not reason_code:
            reason_code = "recovery_development_failed"
        raise RecoveryDevelopmentError(reason_code) from error
