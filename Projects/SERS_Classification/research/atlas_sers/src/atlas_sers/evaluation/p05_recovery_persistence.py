"""Exact evidence copy and exclusive replay lease for the P05 recovery prefix.

This module copies the already-authenticated original develop evidence into the
private recovery stage byte-for-byte and reserves the single authorized
interruption replay lease.  It never trains, instantiates a model, loads a
checkpoint or logits payload, selects, exports or writes the original stage.
Source artifacts are treated as opaque bytes.  It imports only the standard
library and the existing stdlib-only P05 boundary modules; torch and numpy are
never imported here.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_evidence as evidence
from atlas_sers.evaluation import p05_recovery_inputs as inputs
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan
from atlas_sers.evaluation.p05_comprehensive_storage import (
    P05StorageError,
    StorageBudget,
)

__all__ = [
    "RecoveryPersistenceError",
    "RECOVERY_PERSISTENCE_SCHEMA_VERSION",
    "REPLAY_LEASE_SCHEMA_VERSION",
    "REPLAY_LEASE_ATTEMPT_KIND",
    "copy_original_unit",
    "reserve_replay_lease",
]

RECOVERY_PERSISTENCE_SCHEMA_VERSION = "nato-sers-p05-recovery-persistence-v1"
REPLAY_LEASE_SCHEMA_VERSION = "nato-sers-p05-replay-lease-v1"
REPLAY_LEASE_ATTEMPT_KIND = "single_authorized_interruption_replay"
RECOVERIES_DIRNAME = "recoveries"
DEVELOP_STAGE_NAME = "develop"
UNITS_DIRNAME = "units"
REPLAY_LEASE_NAME = "replay_lease.json"
UNIT_MANIFEST_NAME = "manifest.json"

_COPY_CHUNK_BYTES = 1024 * 1024
_COPY_HEADROOM_BYTES = 8 * 1024 * 1024
_MAX_LEASE_BYTES = 64 * 1024


class RecoveryPersistenceError(ValueError):
    """Stable, path-free recovery persistence failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _run(operation: Any) -> Any:
    """Map every boundary failure onto a stable, path-free reason code."""

    try:
        return operation()
    except RecoveryPersistenceError:
        raise
    except inputs.RecoveryInputsError as error:
        raise RecoveryPersistenceError(error.reason_code) from error
    except P05StorageError as error:
        raise RecoveryPersistenceError(error.reason_code) from error
    except authority.RecoveryAuthorityError as error:
        raise RecoveryPersistenceError("authority_rejected") from error
    except evidence.RecoveryEvidenceError as error:
        raise RecoveryPersistenceError(error.reason_code) from error
    except Exception as error:
        raise RecoveryPersistenceError("persistence_failed") from error


def _abs(path: Path | str) -> Path:
    try:
        inputs._reject_symlink_chain(path)
        return Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError) as error:
        raise RecoveryPersistenceError("path_rejected") from error


def _record_size(record: Any) -> tuple[int, str]:
    if not isinstance(record, Mapping) or set(record) != {"sha256", "size_bytes"}:
        raise RecoveryPersistenceError("inventory_record_malformed")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if type(size) is not int or size < 0:
        raise RecoveryPersistenceError("inventory_record_malformed")
    if not isinstance(digest, str) or not core._is_hex64(digest):
        raise RecoveryPersistenceError("inventory_record_malformed")
    return size, digest


def _check_budget_binding(budget: Any, run_root: Path) -> None:
    if not isinstance(budget, StorageBudget):
        raise RecoveryPersistenceError("budget_invalid")
    observed = getattr(budget, "_run_dir", None)
    try:
        observed_path = _abs(observed)
    except (TypeError, ValueError):
        raise RecoveryPersistenceError("budget_run_root_mismatch") from None
    if observed_path != run_root:
        raise RecoveryPersistenceError("budget_run_root_mismatch")
    ceiling = getattr(budget, "_ceiling", None)
    if type(ceiling) is not int or not 0 < ceiling <= authority.PRIVATE_STORAGE_CEILING_BYTES:
        raise RecoveryPersistenceError("budget_ceiling_mismatch")
    if _abs(getattr(budget, "_artifact_root", None)) != run_root.parents[2]:
        raise RecoveryPersistenceError("budget_artifact_root_mismatch")


def _validated_context(recovery_bundle: Any) -> dict[str, Any]:
    evidence._base_bundle(recovery_bundle)
    if not isinstance(recovery_bundle, Mapping):
        raise RecoveryPersistenceError("recovery_bundle_malformed")
    authority.validate_recovery_permit(recovery_bundle.get("recovery_permit"))
    if recovery_bundle.get("recovery_permit_sha256") != authority.RECOVERY_PERMIT_SHA256:
        raise RecoveryPersistenceError("recovery_permit_hash_mismatch")
    base = recovery_bundle.get("base_bundle")
    if (
        not isinstance(base, Mapping)
        or base.get("permit_sha256") != authority.BASECOMPREHENSIVE_PERMIT_SHA256
    ):
        raise RecoveryPersistenceError("base_bundle_rejected")
    plan = recovery_bundle.get("plan")
    inventory = recovery_bundle.get("original_inventory")
    if not isinstance(plan, Mapping) or not isinstance(inventory, Mapping):
        raise RecoveryPersistenceError("recovery_bundle_malformed")
    artifact_root = _abs(base["artifact_root"])
    run_root = _abs(recovery_bundle["original_run_root"])
    expected_run_root = (
        artifact_root
        / inputs.COMPREHENSIVE_NAMESPACE
        / "runs"
        / authority.BASECOMPREHENSIVE_PERMIT_SHA256
    )
    stage = _abs(recovery_bundle["original_stage"])
    if run_root != expected_run_root or stage != expected_run_root / inputs.DEVELOP_STAGE_NAME:
        raise RecoveryPersistenceError("original_stage_mismatch")
    inputs._reject_symlink_chain(stage)
    if not stage.is_dir():
        raise RecoveryPersistenceError("original_stage_missing")
    new_develop = (
        expected_run_root
        / RECOVERIES_DIRNAME
        / authority.RECOVERY_PERMIT_SHA256
        / DEVELOP_STAGE_NAME
    )
    inputs._reject_symlink_chain(new_develop)
    if not new_develop.parent.is_dir():
        raise RecoveryPersistenceError("recovery_container_missing")
    if not new_develop.is_dir():
        raise RecoveryPersistenceError("recovery_develop_missing")
    return {
        "base_bundle": base,
        "plan": plan,
        "run_root": run_root,
        "stage": stage,
        "original_inventory": inventory,
        "new_develop": new_develop,
    }


def _ledger_index(base: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list]]:
    ledger = base.get("ledger")
    if not isinstance(ledger, Mapping):
        raise RecoveryPersistenceError("base_ledger_malformed")
    units = ledger.get("units")
    slots = ledger.get("slots")
    for group in (units, slots):
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise RecoveryPersistenceError("base_ledger_malformed")
        if any(not isinstance(entry, Mapping) for entry in group):
            raise RecoveryPersistenceError("base_ledger_malformed")
    unit_by_id = {}
    for unit in units:
        unit_id = inputs._check_component(unit.get("unit_id"))
        if unit_id in unit_by_id:
            raise RecoveryPersistenceError("ledger_unit_duplicate")
        unit_by_id[unit_id] = unit
    slots_by_unit: dict[str, list] = {}
    slot_ids = set()
    for slot in slots:
        slot_id = inputs._check_component(slot.get("slot_id"))
        unit_id = inputs._check_component(slot.get("unit_id"))
        if slot_id in slot_ids or unit_id not in unit_by_id:
            raise RecoveryPersistenceError("ledger_slot_identity_mismatch")
        slot_ids.add(slot_id)
        slots_by_unit.setdefault(unit_id, []).append(slot)
    return unit_by_id, slots_by_unit


def _classify_unit(plan: Mapping[str, Any], unit_id: str) -> str:
    if unit_id in {str(value) for value in plan.get("sealed_unit_ids", ())}:
        return "sealed"
    if unit_id == str(plan.get("incomplete_unit_id")):
        return "partial"
    raise RecoveryPersistenceError("unit_not_copyable")


def _unit_copy_plan(
    plan: Mapping[str, Any], unit: Mapping[str, Any], slots: list, kind: str
) -> tuple[set[str], set[str], bool]:
    copy_relatives: set[str] = set()
    original_relatives: set[str] = set()
    slots = _run(lambda: evidence._unit_slots({"slots": slots}, str(unit["unit_id"])))
    _run(lambda: evidence._completed_slots(slots, plan, kind == "sealed"))
    if kind == "sealed":
        copy_relatives.add(UNIT_MANIFEST_NAME)
        original_relatives.add(UNIT_MANIFEST_NAME)
        for slot in slots:
            original_relatives.update(inputs._slot_files(unit, slot))
        copy_relatives.update(original_relatives)
        return copy_relatives, original_relatives, False
    completed = {str(value) for value in plan.get("reused_original_slot_ids", ())}
    interrupted = str(plan.get("interrupted_slot_id"))
    completed_count = 0
    interrupted_seen = False
    for slot in slots:
        slot_id = str(slot["slot_id"])
        if slot_id in completed:
            completed_count += 1
            copy_relatives.update(inputs._slot_files(unit, slot))
        elif slot_id == interrupted:
            interrupted_seen = True
            identifier = inputs._check_component(pilot.execution_id(unit, slot))
            original_relatives.add(f"histories/{identifier}.jsonl")
    if completed_count != recovery_plan.PARTIAL_COMPLETED or not interrupted_seen:
        raise RecoveryPersistenceError("original_partial_completion_mismatch")
    original_relatives.update(copy_relatives)
    return copy_relatives, original_relatives, True


def _unit_inventory(inventory: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    prefix = f"{UNITS_DIRNAME}/{unit_id}/"
    return {
        relative[len(prefix) :]: record
        for relative, record in inventory.items()
        if isinstance(relative, str) and relative.startswith(prefix)
    }


def _copy_evidence(
    source: Path,
    destination: Path,
    record: Any,
    budget: StorageBudget,
    deadline: Any,
) -> int:
    size, digest = _record_size(record)
    inputs._check_deadline(deadline)
    inputs._reject_symlink_chain(source)
    inputs._reject_symlink_chain(destination)
    os.makedirs(os.fspath(destination.parent), mode=0o700, exist_ok=True)
    inputs._reject_symlink_chain(destination.parent)
    source_handle = inputs._open_regular(source, "source_unreadable")
    hasher = hashlib.sha256()
    written = 0
    try:
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError as error:
            raise RecoveryPersistenceError("destination_exists") from error
        try:
            output = os.fdopen(descriptor, "wb")
        except (OSError, ValueError):
            os.close(descriptor)
            raise
        with output:
            while True:
                inputs._check_deadline(deadline)
                chunk = source_handle.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if written + len(chunk) > size:
                    raise RecoveryPersistenceError("source_size_exceeded")
                budget.check(headroom_bytes=len(chunk))
                hasher.update(chunk)
                written += len(chunk)
                output.write(chunk)
                budget.check()
            output.flush()
            os.fsync(output.fileno())
    finally:
        source_handle.close()
    if written != size or hasher.hexdigest() != digest:
        raise RecoveryPersistenceError("source_digest_mismatch")
    observed = inputs._hash_file_record(destination, deadline)
    if observed.get("sha256") != digest or observed.get("size_bytes") != size:
        raise RecoveryPersistenceError("destination_digest_mismatch")
    budget.check()
    inputs._check_deadline(deadline)
    return written


def _copy_original_unit(
    recovery_bundle: Mapping[str, Any],
    unit_id: str,
    budget: StorageBudget,
    deadline: float,
) -> dict[str, Any]:
    inputs._check_deadline(deadline)
    context = _validated_context(recovery_bundle)
    _check_budget_binding(budget, context["run_root"])
    unit_id = inputs._check_component(unit_id)
    plan = context["plan"]
    unit_by_id, slots_by_unit = _ledger_index(context["base_bundle"])
    unit = unit_by_id.get(unit_id)
    slots = slots_by_unit.get(unit_id)
    if unit is None or not isinstance(slots, list) or len(slots) != recovery_plan.SLOTS_PER_UNIT:
        raise RecoveryPersistenceError("unit_not_copyable")
    kind = _classify_unit(plan, unit_id)
    copy_relatives, original_relatives, omitted = _unit_copy_plan(plan, unit, slots, kind)
    inventory = _unit_inventory(context["original_inventory"], unit_id)
    if set(inventory) != original_relatives:
        raise RecoveryPersistenceError("original_unit_inventory_mismatch")
    source_unit = context["stage"] / UNITS_DIRNAME / unit_id
    evidence._check_inventory(source_unit, inventory, deadline)
    evidence._hash_unit(source_unit, inventory, deadline)
    evidence._bind_unit_inventory(
        source_unit,
        unit_id,
        kind == "sealed",
        recovery_bundle["original_anchor"],
        inventory,
        deadline,
    )
    expected_bytes = 0
    for relative in copy_relatives:
        size, _digest = _record_size(inventory[relative])
        expected_bytes += size
    destination = context["new_develop"] / UNITS_DIRNAME / unit_id
    inputs._reject_symlink_chain(destination.parent)
    if destination.is_symlink() or destination.exists():
        raise RecoveryPersistenceError("destination_unit_exists")
    budget.check(headroom_bytes=expected_bytes + _COPY_HEADROOM_BYTES)
    inputs._check_deadline(deadline)
    budget.activate_unit(destination)
    copied_files: list[str] = []
    copied_bytes = 0
    for relative in sorted(copy_relatives):
        inputs._check_deadline(deadline)
        copied_bytes += _copy_evidence(
            context["stage"] / UNITS_DIRNAME / unit_id / relative,
            destination / relative,
            inventory[relative],
            budget,
            deadline,
        )
        copied_files.append(relative)
        budget.check()
    budget.check()
    inputs._check_deadline(deadline)
    return {
        "schema_version": RECOVERY_PERSISTENCE_SCHEMA_VERSION,
        "unit_id": unit_id,
        "unit_kind": kind,
        "destination_unit_dir": destination,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "omitted_interrupted_history": omitted,
        "unit_active": True,
    }


def copy_original_unit(
    *,
    recovery_bundle: Mapping[str, Any],
    unit_id: str,
    budget: StorageBudget,
    deadline: float,
) -> dict[str, Any]:
    """Copy one authenticated original unit into the private recovery stage."""

    return _run(lambda: _copy_original_unit(recovery_bundle, unit_id, budget, deadline))


def _reserve_replay_lease(
    recovery_bundle: Mapping[str, Any],
    slot: Mapping[str, Any],
    budget: StorageBudget,
    deadline: float,
) -> Path:
    inputs._check_deadline(deadline)
    context = _validated_context(recovery_bundle)
    _check_budget_binding(budget, context["run_root"])
    if not isinstance(slot, Mapping):
        raise RecoveryPersistenceError("slot_malformed")
    slot_id = slot.get("slot_id")
    if not isinstance(slot_id, str):
        raise RecoveryPersistenceError("slot_identity_mismatch")
    slot_id = inputs._check_component(slot_id)
    plan = context["plan"]
    if slot_id != str(plan.get("interrupted_slot_id")):
        raise RecoveryPersistenceError("slot_not_interrupted")
    base = context["base_bundle"]
    _ledger_index(base)
    registered = None
    for candidate in base["ledger"]["slots"]:
        if str(candidate["slot_id"]) == slot_id:
            registered = candidate
            break
    if registered is None:
        raise RecoveryPersistenceError("slot_not_registered")
    if core._canon().sha256_value(dict(slot)) != core._canon().sha256_value(dict(registered)):
        raise RecoveryPersistenceError("slot_identity_mismatch")
    unit_id = str(registered["unit_id"])
    if unit_id != str(plan.get("incomplete_unit_id")):
        raise RecoveryPersistenceError("slot_not_interrupted")
    evidence._unit_slots(base["ledger"], unit_id)
    recipe_id = str(registered["recipe_id"])
    seed = registered["seed"]
    if type(seed) is not int:
        raise RecoveryPersistenceError("slot_identity_mismatch")
    record = recovery_bundle["original_lease_inventory"].get(f"{slot_id}/lease.json")
    size, digest = _record_size(record)
    lease_path = base_inputs._slot_lease_root(base["artifact_root"]) / slot_id / "lease.json"
    inputs._reject_symlink_chain(lease_path)
    raw = inputs._read_bytes_bounded(lease_path, _MAX_LEASE_BYTES, "original_lease", deadline)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise RecoveryPersistenceError("original_lease_mismatch")
    lease = inputs._parse_json(raw, "original_lease")
    if not isinstance(lease, Mapping) or (
        str(lease.get("slot_id")) != slot_id
        or str(lease.get("unit_id")) != unit_id
        or str(lease.get("recipe_id")) != recipe_id
        or type(lease.get("seed")) is not int
        or lease.get("seed") != seed
        or str(lease.get("contract_sha256")) != str(base["contract_sha256"])
        or str(lease.get("core_plan_id")) != str(base["core_plan_id"])
        or str(lease.get("permit_sha256")) != authority.BASECOMPREHENSIVE_PERMIT_SHA256
    ):
        raise RecoveryPersistenceError("original_lease_mismatch")
    recovery_root = context["new_develop"].parent
    inputs._reject_symlink_chain(recovery_root)
    lease_out = recovery_root / REPLAY_LEASE_NAME
    inputs._reject_symlink_chain(lease_out)
    if lease_out.is_symlink() or lease_out.exists():
        raise RecoveryPersistenceError("replay_lease_exists")
    payload = {
        "schema_version": REPLAY_LEASE_SCHEMA_VERSION,
        "attempt_kind": REPLAY_LEASE_ATTEMPT_KIND,
        "base_comprehensive_permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "core_contract_sha256": str(base["contract_sha256"]),
        "core_plan_id": str(base["core_plan_id"]),
        "slot_id": slot_id,
        "unit_id": unit_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "original_lease_sha256": digest,
    }
    raw_out = core._canon().canonical_json_bytes(payload)
    budget.check(headroom_bytes=len(raw_out))
    inputs._check_deadline(deadline)
    try:
        descriptor = os.open(
            lease_out,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except FileExistsError as error:
        raise RecoveryPersistenceError("replay_lease_exists") from error
    try:
        output = os.fdopen(descriptor, "wb")
    except (OSError, ValueError):
        os.close(descriptor)
        raise
    with output:
        output.write(raw_out)
        output.flush()
        os.fsync(output.fileno())
    budget.account_new_file(lease_out)
    budget.check()
    inputs._check_deadline(deadline)
    return lease_out


def reserve_replay_lease(
    *,
    recovery_bundle: Mapping[str, Any],
    slot: Mapping[str, Any],
    budget: StorageBudget,
    deadline: float,
) -> Path:
    """Create the exclusive single-interruption replay lease exactly once."""

    return _run(lambda: _reserve_replay_lease(recovery_bundle, slot, budget, deadline))
