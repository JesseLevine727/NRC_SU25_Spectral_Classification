"""Read-only post-completion authentication of the recovered P05 source stage.

``authenticate_completed_source`` re-proves, from the recovered stage bytes on
disk, that the single serial recovery boundary produced exactly the approved
source-only development evidence before any selection stage may consume it.
It is only ever called on a recovered run: the caller MUST already hold an
independently resolved ``paths`` mapping equal to ``source.resolve_paths(bundle)``.

The module never trains, never imports ``torch`` or ``numpy``, never loads a
checkpoint or logits payload, never creates a lease and never writes.  It hashes
artifact files as opaque serial streams only.  It reuses the existing stdlib-only
P05 boundaries, including the outer runner's private re-authentication helpers,
so guard logic stays consistent.  It imports nothing from the freeze module.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_development as outer
from atlas_sers.evaluation import p05_recovery_evidence as evidence
from atlas_sers.evaluation import p05_recovery_inputs as recovery_inputs
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan
from atlas_sers.evaluation import p05_recovery_receipt as receipt
from atlas_sers.evaluation import p05_recovery_source as source

__all__ = ["RecoveryAcceptanceError", "authenticate_completed_source"]

SCHEMA_VERSION = "nato-sers-p05-recovery-acceptance-v1"

ANCHOR_FILE_COUNT = 774
SLOT_FILES_PER_SLOT = len(recovery_inputs.SLOT_EXECUTION_FILES) + 1

RECOVERY_TOP_FILES = (
    "events.jsonl",
    "selector.jsonl",
    "progress.json",
    "ledger.json",
    "source_ledger.json",
    "plan.json",
    "original_inventory.json",
    "original_anchor.json",
    "original_lease_inventory.json",
    "recovery_permit.json",
    "input_manifest.json",
    "provenance_before.json",
    "provenance_after.json",
    "unit_manifest_inventory.json",
    "summary.json",
    "manifest.json",
)
RECOVERY_ALLOWED_TOP_ENTRIES = frozenset(RECOVERY_TOP_FILES) | {"units"}


class RecoveryAcceptanceError(ValueError):
    """Stable, path-free recovery source-authentication failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = str(reason_code)


def _call(code: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except RecoveryAcceptanceError:
        raise
    except Exception as error:
        raise RecoveryAcceptanceError(code) from error


def _deadline(deadline: Any) -> None:
    try:
        recovery_inputs._check_deadline(deadline)
    except recovery_inputs.RecoveryInputsError as error:
        raise RecoveryAcceptanceError(error.reason_code) from error


def _canon() -> Any:
    return core._canon()


def _read_mapping(path: Path, code: str, deadline: Any) -> dict[str, Any]:
    value = _call(code, recovery_inputs._read_json_mapping, path, code, deadline)
    if not isinstance(value, Mapping):
        raise RecoveryAcceptanceError(f"{code}_malformed")
    return dict(value)


def _read_raw(path: Path, code: str, deadline: Any) -> bytes:
    return _call(
        code,
        recovery_inputs._read_bytes_bounded,
        path,
        recovery_inputs._MAX_JSON_BYTES,
        code,
        deadline,
    )


def _sha256_file(path: Path) -> str:
    return _call("file_unreadable", _canon().sha256_file, path)


def _same_bytes(left: Any, right: Any, code: str) -> None:
    if _call(code, _canon().canonical_json_bytes, left) != _call(
        code, _canon().canonical_json_bytes, right
    ):
        raise RecoveryAcceptanceError(code)


def _check_paths(bundle: Mapping[str, Any], paths: Any) -> dict[str, Path]:
    if not isinstance(paths, Mapping):
        raise RecoveryAcceptanceError("paths_not_mapping")
    resolved = _call("paths_unresolved", source.resolve_paths, bundle)
    if not isinstance(resolved, Mapping) or set(resolved) != set(paths):
        raise RecoveryAcceptanceError("paths_keys_mismatch")
    normalized: dict[str, Path] = {}
    try:
        for key, value in resolved.items():
            candidate = Path(value)
            if candidate != Path(paths[key]):
                raise RecoveryAcceptanceError("paths_mismatch")
            normalized[str(key)] = candidate
    except RecoveryAcceptanceError:
        raise
    except (TypeError, ValueError) as error:
        raise RecoveryAcceptanceError("paths_mismatch") from error
    return normalized


def _index(bundle: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    ledger = bundle.get("ledger")
    if not isinstance(ledger, Mapping):
        raise RecoveryAcceptanceError("base_bundle_malformed")
    units = ledger.get("units")
    slots = ledger.get("slots")
    if not isinstance(units, (list, tuple)) or not isinstance(slots, (list, tuple)):
        raise RecoveryAcceptanceError("base_bundle_malformed")
    unit_by_id = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            raise RecoveryAcceptanceError("ledger_unit_malformed")
        unit_id = _call(
            "unit_identity_invalid", recovery_inputs._check_component, unit.get("unit_id")
        )
        if unit_id in unit_by_id:
            raise RecoveryAcceptanceError("ledger_unit_duplicate")
        unit_by_id[unit_id] = unit
    slots_by_unit: dict[str, list[Any]] = {}
    slot_ids: set[str] = set()
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise RecoveryAcceptanceError("ledger_slot_malformed")
        slot_id = _call(
            "slot_identity_invalid", recovery_inputs._check_component, slot.get("slot_id")
        )
        unit_id = slot.get("unit_id")
        if not isinstance(unit_id, str) or unit_id not in unit_by_id:
            raise RecoveryAcceptanceError("slot_unit_unknown")
        if slot_id in slot_ids:
            raise RecoveryAcceptanceError("ledger_slot_duplicate")
        slot_ids.add(slot_id)
        slots_by_unit.setdefault(unit_id, []).append(slot)
    for group in slots_by_unit.values():
        group.sort(key=lambda slot: (str(slot["recipe_id"]), int(slot["seed"])))
    return unit_by_id, slots_by_unit


def _pilot_ids(bundle: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    pilot_bundle = bundle.get("pilot_bundle")
    if not isinstance(pilot_bundle, Mapping):
        raise RecoveryAcceptanceError("base_bundle_malformed")
    slots = pilot_bundle.get("slots")
    if not isinstance(slots, (list, tuple)):
        raise RecoveryAcceptanceError("base_bundle_malformed")
    slot_ids = {str(slot["slot_id"]) for slot in slots}
    unit_ids = {str(slot["unit_id"]) for slot in slots}
    return unit_ids, slot_ids


def _read_top(stage: Path, deadline: Any) -> dict[str, bytes]:
    raw: dict[str, bytes] = {}
    for name in RECOVERY_TOP_FILES:
        raw[name] = _read_raw(stage / name, f"stage_{name}", deadline)
        _deadline(deadline)
    return raw


def _check_top_layout(stage: Path) -> None:
    entries = _call("stage_unreadable", recovery_inputs._list_entries, stage, "stage_unreadable")
    if set(entries) != set(RECOVERY_ALLOWED_TOP_ENTRIES):
        raise RecoveryAcceptanceError("stage_layout_rejected")
    for name in RECOVERY_TOP_FILES:
        if entries.get(name) != "file":
            raise RecoveryAcceptanceError("stage_layout_rejected")
    if entries.get("units") != "dir":
        raise RecoveryAcceptanceError("stage_layout_rejected")


def _check_anchor(stored_anchor: Any) -> Mapping[str, Any]:
    if not isinstance(stored_anchor, Mapping):
        raise RecoveryAcceptanceError("original_anchor_malformed")
    if stored_anchor.get("schema_version") != recovery_inputs.INTERRUPTION_ANCHOR_SCHEMA_VERSION:
        raise RecoveryAcceptanceError("original_anchor_malformed")
    files = stored_anchor.get("files")
    if not isinstance(files, Mapping) or len(files) != ANCHOR_FILE_COUNT:
        raise RecoveryAcceptanceError("original_anchor_malformed")
    digest = _call("original_anchor_malformed", _canon().sha256_value, dict(stored_anchor))
    if digest != authority.ORIGINAL_EVIDENCE_ANCHOR_SHA256:
        raise RecoveryAcceptanceError("original_anchor_digest_mismatch")
    return stored_anchor


def _check_original_inventory(
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
    original_stage: Path,
    stored_inventory: Any,
    stored_anchor: Mapping[str, Any],
    sealed_unit_ids: set[str],
    deadline: Any,
) -> tuple[set[str], set[str]]:
    if not isinstance(stored_inventory, Mapping):
        raise RecoveryAcceptanceError("original_inventory_malformed")
    layout = _call(
        "original_inventory_malformed",
        recovery_inputs._expected_layout,
        bundle,
        plan,
    )
    expected_files, expected_dirs, _sealed, _sealed_files, _unsealed = layout
    if set(stored_inventory) != set(expected_files):
        raise RecoveryAcceptanceError("original_inventory_mismatch")
    anchored = stored_anchor["files"]
    for record in stored_inventory.values():
        _call("original_inventory_malformed", evidence._inventory_record, record)
    expected_anchors = set(recovery_inputs.ORIGINAL_TOP_FILES) | set(_sealed) | set(_unsealed)
    if set(anchored) != expected_anchors:
        raise RecoveryAcceptanceError("original_anchor_inventory_mismatch")
    for relative, record in anchored.items():
        if not recovery_inputs._record_equal(stored_inventory.get(relative), record):
            raise RecoveryAcceptanceError("original_inventory_anchor_mismatch")
    unit_ids = [str(unit_id) for unit_id in plan["sealed_unit_ids"]]
    unit_ids.append(str(plan["incomplete_unit_id"]))
    for unit_id in unit_ids:
        _call("unit_identity_invalid", recovery_inputs._check_component, unit_id)
        expected_unit = _call(
            "original_inventory_malformed",
            evidence._expected_unit_files,
            stored_inventory,
            unit_id,
        )
        _call(
            "unit_inventory_anchor_mismatch",
            evidence._bind_unit_inventory,
            original_stage / "units" / unit_id,
            unit_id,
            unit_id in sealed_unit_ids,
            stored_anchor,
            expected_unit,
            deadline,
        )
        _deadline(deadline)
    return expected_files, expected_dirs


def _old_lease_paths(bundle: Mapping[str, Any], plan: Mapping[str, Any]) -> set[str]:
    _pilot_units, pilot_slots = _pilot_ids(bundle)
    reused = {str(value) for value in plan["reused_original_slot_ids"]}
    interrupted = str(plan["interrupted_slot_id"])
    slot_ids = set(pilot_slots) | reused | {interrupted}
    return {f"{slot_id}/lease.json" for slot_id in slot_ids}


def _read_old_leases(
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
    stored_lease_inventory: Any,
    deadline: Any,
) -> list[dict[str, Any]]:
    if not isinstance(stored_lease_inventory, Mapping):
        raise RecoveryAcceptanceError("original_lease_inventory_malformed")
    expected = _old_lease_paths(bundle, plan)
    if set(stored_lease_inventory) != expected:
        raise RecoveryAcceptanceError("original_lease_inventory_mismatch")
    if len(stored_lease_inventory) != recovery_plan.LEASE_COUNT:
        raise RecoveryAcceptanceError("original_lease_count_mismatch")
    lease_root = base_inputs._slot_lease_root(bundle["artifact_root"])
    records: list[dict[str, Any]] = []
    for relative in sorted(stored_lease_inventory):
        _deadline(deadline)
        slot_id = _call(
            "original_lease_identity_mismatch",
            recovery_inputs._check_component,
            relative.split("/", 1)[0],
        )
        record = _call(
            "original_lease_inventory_malformed",
            evidence._inventory_record,
            stored_lease_inventory[relative],
        )
        observed = _call(
            "original_lease_unreadable",
            recovery_inputs._hash_file_record,
            lease_root / relative,
            deadline,
        )
        if observed != record:
            raise RecoveryAcceptanceError("original_lease_changed")
        lease = _read_mapping(lease_root / relative, "original_slot_lease", deadline)
        if str(lease.get("slot_id")) != slot_id:
            raise RecoveryAcceptanceError("original_lease_identity_mismatch")
        records.append(lease)
    return records


def _anchored_original_evidence(
    bundle: Mapping[str, Any],
    original_stage: Path,
    stored_inventory: Mapping[str, Any],
    deadline: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events_raw = _read_raw(original_stage / "events.jsonl", "original_events", deadline)
    if _call("original_events_mismatch", hashlib.sha256, events_raw).hexdigest() != str(
        stored_inventory["events.jsonl"]["sha256"]
    ):
        raise RecoveryAcceptanceError("original_events_mismatch")
    events = _call(
        "original_events_malformed", recovery_inputs._parse_jsonl, events_raw, "original_events"
    )
    selector_raw = _read_raw(original_stage / "selector.jsonl", "original_selector", deadline)
    if _call("original_selector_mismatch", hashlib.sha256, selector_raw).hexdigest() != str(
        stored_inventory["selector.jsonl"]["sha256"]
    ):
        raise RecoveryAcceptanceError("original_selector_mismatch")
    selectors = _call(
        "original_selector_malformed",
        recovery_inputs._parse_jsonl,
        selector_raw,
        "original_selector",
    )
    if not events:
        raise RecoveryAcceptanceError("original_events_empty")
    last = events[-1]
    if last.get("event") != "started":
        raise RecoveryAcceptanceError("original_interruption_missing")
    unit_by_id, _slots = _index(bundle)
    unit_id = str(last.get("unit_id"))
    slot_id = str(last.get("slot_id"))
    slot = next(
        (
            candidate
            for candidate in bundle["ledger"]["slots"]
            if str(candidate["slot_id"]) == slot_id
        ),
        None,
    )
    unit = unit_by_id.get(unit_id)
    if slot is None or unit is None or str(slot["unit_id"]) != unit_id:
        raise RecoveryAcceptanceError("original_interruption_identity_mismatch")
    identifier = _call("original_interruption_identity_mismatch", pilot.execution_id, unit, slot)
    relative = f"units/{unit_id}/histories/{identifier}.jsonl"
    if str(last.get("execution_id")) != identifier or relative not in stored_inventory:
        raise RecoveryAcceptanceError("original_interruption_identity_mismatch")
    history_raw = _read_raw(original_stage / relative, "original_interrupted_history", deadline)
    if _call(
        "original_interrupted_history_mismatch", hashlib.sha256, history_raw
    ).hexdigest() != str(stored_inventory[relative]["sha256"]):
        raise RecoveryAcceptanceError("original_interrupted_history_mismatch")
    history = _call(
        "original_interrupted_history_malformed",
        recovery_inputs._parse_jsonl,
        history_raw,
        "original_interrupted_history",
    )
    return events, selectors, history


def _rebuild_plan(
    bundle: Mapping[str, Any],
    stored_plan: Mapping[str, Any],
    events: list[dict[str, Any]],
    selectors: list[dict[str, Any]],
    leases: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> Mapping[str, Any]:
    try:
        rebuilt = recovery_plan.build_recovery_plan(
            ledger=bundle["ledger"],
            pilot_slots=bundle["pilot_bundle"]["slots"],
            events=events,
            selector_records=selectors,
            leases=leases,
            interrupted_history=history,
        )
    except recovery_plan.RecoveryPlanError as error:
        raise RecoveryAcceptanceError(f"recovery_plan_{error.reason_code}") from error
    _same_bytes(rebuilt, stored_plan, "recovery_plan_mismatch")
    return rebuilt


def _check_recovered_units(
    stage: Path,
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
    stored_inventory: Mapping[str, Any],
    unit_manifest_inventory: Any,
    deadline: Any,
) -> None:
    if not isinstance(unit_manifest_inventory, Mapping):
        raise RecoveryAcceptanceError("unit_manifest_inventory_malformed")
    unit_by_id, slots_by_unit = _index(bundle)
    pilot_units, _pilot_slots = _pilot_ids(bundle)
    nonpilot = {unit_id for unit_id in unit_by_id if unit_id not in pilot_units}
    if set(unit_manifest_inventory) != nonpilot:
        raise RecoveryAcceptanceError("unit_manifest_inventory_mismatch")
    units_root = stage / "units"
    entries = _call(
        "units_unreadable", recovery_inputs._list_entries, units_root, "units_unreadable"
    )
    if set(entries) != nonpilot or any(kind != "dir" for kind in entries.values()):
        raise RecoveryAcceptanceError("unit_directory_mismatch")
    sealed = {str(unit_id) for unit_id in plan["sealed_unit_ids"]}
    partial = str(plan["incomplete_unit_id"])
    reused = {str(unit_id) for unit_id in plan["reused_original_slot_ids"]}
    for unit_id in sorted(unit_manifest_inventory):
        _deadline(deadline)
        _call("unit_identity_invalid", recovery_inputs._check_component, unit_id)
        record = _call(
            "unit_manifest_inventory_malformed",
            evidence._inventory_record,
            unit_manifest_inventory[unit_id],
        )
        unit_dir = units_root / unit_id
        observed = _call(
            "unit_manifest_changed",
            recovery_inputs._hash_file_record,
            unit_dir / "manifest.json",
            deadline,
        )
        if observed != record:
            raise RecoveryAcceptanceError("unit_manifest_changed")
        unit = unit_by_id[unit_id]
        expected_relative = set()
        for slot in slots_by_unit[unit_id]:
            expected_relative.update(recovery_inputs._slot_files(unit, slot))
        if len(expected_relative) != recovery_plan.SLOTS_PER_UNIT * SLOT_FILES_PER_SLOT:
            raise RecoveryAcceptanceError("unit_manifest_inventory_mismatch")
        actual_files, actual_dirs = recovery_inputs._collect_tree(unit_dir, deadline)
        if set(actual_files) != expected_relative | {"manifest.json"}:
            raise RecoveryAcceptanceError("unit_file_inventory_mismatch")
        if actual_dirs != evidence._expected_dirs({name: None for name in actual_files}):
            raise RecoveryAcceptanceError("unit_directory_inventory_mismatch")
        manifest = _read_mapping(unit_dir / "manifest.json", "unit_manifest", deadline)
        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, Mapping) or set(manifest_files) != expected_relative:
            raise RecoveryAcceptanceError("unit_manifest_inventory_mismatch")
        if unit_id in sealed:
            if observed != stored_inventory.get(f"units/{unit_id}/manifest.json"):
                raise RecoveryAcceptanceError("sealed_unit_manifest_changed")
        elif unit_id == partial:
            for slot in slots_by_unit[unit_id]:
                if str(slot["slot_id"]) not in reused:
                    continue
                for relative in recovery_inputs._slot_files(unit, slot):
                    key = f"units/{unit_id}/{relative}"
                    current = _call(
                        "reused_slot_unreadable",
                        recovery_inputs._hash_file_record,
                        unit_dir / relative,
                        deadline,
                    )
                    if current != _call(
                        "original_inventory_malformed",
                        evidence._inventory_record,
                        stored_inventory[key],
                    ):
                        raise RecoveryAcceptanceError("reused_slot_changed")
    _deadline(deadline)


def _check_replay_prefix(
    stage: Path,
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
    anchored_history: list[dict[str, Any]],
    deadline: Any,
) -> None:
    unit_by_id, _slots = _index(bundle)
    interrupted = str(plan["interrupted_slot_id"])
    slot = next(
        candidate
        for candidate in bundle["ledger"]["slots"]
        if str(candidate["slot_id"]) == interrupted
    )
    unit = unit_by_id[str(slot["unit_id"])]
    identifier = _call("replay_history_unreadable", pilot.execution_id, unit, slot)
    recovered = _call(
        "replay_history_malformed",
        recovery_inputs._read_jsonl,
        stage / "units" / str(unit["unit_id"]) / "histories" / f"{identifier}.jsonl",
        "replay_history",
        deadline,
    )
    if len(recovered) < len(anchored_history):
        raise RecoveryAcceptanceError("replay_history_truncated")
    for observed, expected in zip(recovered, anchored_history, strict=False):
        _same_bytes(observed, expected, "replay_history_prefix_mismatch")


def _stream_optimizer_steps(
    stage: Path,
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    deadline: Any,
) -> None:
    unit_by_id, slots_by_unit = _index(bundle)
    pilot_units, _pilot_slots = _pilot_ids(bundle)
    reused = {str(value) for value in plan["reused_original_slot_ids"]}
    total = 0
    reused_total = 0
    new_total = 0
    new_count = 0
    for unit_id in sorted(unit_id for unit_id in unit_by_id if unit_id not in pilot_units):
        unit = unit_by_id[unit_id]
        unit_dir = stage / "units" / unit_id
        for slot in slots_by_unit[unit_id]:
            _deadline(deadline)
            record = _call(
                "summary_unreadable", base_inputs._read_pilot_summary, unit_dir, unit, slot
            )
            if str(record.get("status")) != "complete":
                raise RecoveryAcceptanceError("summary_status_invalid")
            steps = record.get("optimizer_steps")
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise RecoveryAcceptanceError("optimizer_steps_invalid")
            if steps % recovery_plan.BATCH_DRAWS_PER_EPOCH != 0:
                raise RecoveryAcceptanceError("optimizer_steps_invalid")
            if not (
                recovery_plan.MINIMUM_FIT_UPDATES <= steps <= recovery_plan.MAXIMUM_FIT_UPDATES
            ):
                raise RecoveryAcceptanceError("optimizer_steps_invalid")
            total += steps
            if str(slot["slot_id"]) in reused:
                reused_total += steps
            else:
                new_total += steps
                new_count += 1
    if total != summary["optimizer_steps"]:
        raise RecoveryAcceptanceError("optimizer_steps_total_mismatch")
    if reused_total != receipt.REUSED_ORIGINAL_OPTIMIZER_STEPS:
        raise RecoveryAcceptanceError("reused_optimizer_steps_mismatch")
    if new_total != summary["recovery_optimizer_steps_exact"]:
        raise RecoveryAcceptanceError("recovery_optimizer_steps_mismatch")
    if new_count != summary["recovery_completed"]:
        raise RecoveryAcceptanceError("recovery_fit_count_mismatch")
    _deadline(deadline)


def authenticate_completed_source(
    bundle: Mapping[str, Any],
    *,
    paths: Mapping[str, Any],
    summary: Mapping[str, Any],
    receipt_record: Mapping[str, Any],
    deadline: float,
) -> dict[str, Any]:
    """Authenticate the completed recovered source stage and return accounting."""

    _deadline(deadline)
    if (
        not isinstance(bundle, Mapping)
        or not isinstance(summary, Mapping)
        or not isinstance(receipt_record, Mapping)
    ):
        raise RecoveryAcceptanceError("inputs_malformed")

    validated = _call(
        "recovery_receipt_invalid",
        receipt.validate_pair,
        summary,
        receipt_record,
        source.RECOVERED_UNIT_COUNT,
    )
    resolved = _check_paths(bundle, paths)
    run_root = resolved["run_root"]
    stage = resolved["develop"]
    if resolved["receipt"] != run_root / receipt.RECEIPT_NAME:
        raise RecoveryAcceptanceError("paths_mismatch")
    _same_bytes(
        _read_mapping(resolved["receipt"], "recovery_receipt", deadline),
        receipt_record,
        "receipt_changed",
    )
    original_stage = run_root / source.DEVELOP_STAGE_NAME
    _call("original_stage_unreadable", recovery_inputs._reject_symlink_chain, original_stage)
    _call("recovery_stage_unreadable", recovery_inputs._reject_symlink_chain, stage)
    _deadline(deadline)

    _check_top_layout(stage)
    top = _read_top(stage, deadline)

    stored_permit = _call(
        "recovery_permit_invalid",
        authority.validate_recovery_permit,
        _call(
            "recovery_permit_malformed",
            recovery_inputs._parse_json,
            top["recovery_permit.json"],
            "recovery_permit",
        ),
    )
    if not isinstance(stored_permit, Mapping):
        raise RecoveryAcceptanceError("recovery_permit_malformed")
    if _call("recovery_permit_invalid", _canon().sha256_value, dict(stored_permit)) != (
        authority.RECOVERY_PERMIT_SHA256
    ):
        raise RecoveryAcceptanceError("recovery_permit_mismatch")
    _deadline(deadline)

    _call("summary_malformed", recovery_inputs._parse_json, top["summary.json"], "summary")
    if top["summary.json"] != _call(
        "summary_malformed", _canon().canonical_json_bytes, dict(summary)
    ):
        raise RecoveryAcceptanceError("summary_mismatch")

    stored_plan = _call(
        "recovery_plan_malformed", recovery_inputs._parse_json, top["plan.json"], "plan"
    )
    if not isinstance(stored_plan, Mapping):
        raise RecoveryAcceptanceError("recovery_plan_malformed")
    _call("recovery_plan_malformed", outer._check_plan, stored_plan)
    if str(validated.get("recovery_plan_id")) != str(stored_plan.get("plan_id")):
        raise RecoveryAcceptanceError("recovery_plan_id_mismatch")
    _call(
        "recovery_plan_permit_mismatch",
        recovery_inputs._check_plan_against_permit,
        stored_plan,
        stored_permit,
    )
    _deadline(deadline)

    stored_anchor = _check_anchor(
        _call(
            "original_anchor_malformed",
            recovery_inputs._parse_json,
            top["original_anchor.json"],
            "original_anchor",
        )
    )
    if str(validated.get("original_evidence_anchor_sha256")) != (
        authority.ORIGINAL_EVIDENCE_ANCHOR_SHA256
    ):
        raise RecoveryAcceptanceError("original_anchor_digest_mismatch")

    stored_inventory = _call(
        "original_inventory_malformed",
        recovery_inputs._parse_json,
        top["original_inventory.json"],
        "original_inventory",
    )
    sealed_unit_ids = {str(unit_id) for unit_id in stored_plan["sealed_unit_ids"]}
    expected_files, expected_dirs = _check_original_inventory(
        bundle,
        stored_plan,
        original_stage,
        stored_inventory,
        stored_anchor,
        sealed_unit_ids,
        deadline,
    )
    view = {
        "original_inventory": dict(stored_inventory),
        "original_anchor": dict(stored_anchor),
        "original_anchor_sha256": authority.ORIGINAL_EVIDENCE_ANCHOR_SHA256,
    }
    _call(
        "original_stage_authentication_failed",
        outer._reauthenticate_original,
        original_stage,
        bundle,
        stored_plan,
        view,
        deadline,
    )
    _deadline(deadline)

    stored_lease_inventory = _call(
        "original_lease_inventory_malformed",
        recovery_inputs._parse_json,
        top["original_lease_inventory.json"],
        "original_lease_inventory",
    )
    old_leases = _read_old_leases(bundle, stored_plan, stored_lease_inventory, deadline)
    events, selectors, history = _anchored_original_evidence(
        bundle, original_stage, stored_inventory, deadline
    )
    _rebuild_plan(bundle, stored_plan, events, selectors, old_leases, history)
    _deadline(deadline)

    slot_by_id = {str(slot["slot_id"]): slot for slot in bundle["ledger"]["slots"]}
    view["original_lease_inventory"] = dict(stored_lease_inventory)
    _call(
        "original_lease_changed",
        outer._reauthenticate_leases,
        bundle,
        stored_plan,
        view,
        slot_by_id,
        deadline,
    )
    _call(
        "replay_lease_mismatch",
        outer._check_replay_lease,
        stage.parent,
        bundle,
        stored_plan,
        view,
        slot_by_id,
        deadline,
    )
    _deadline(deadline)

    _call("ledger_malformed", recovery_inputs._parse_json, top["ledger.json"], "ledger")
    if top["ledger.json"] != _call(
        "ledger_malformed", _canon().canonical_json_bytes, bundle["ledger"]
    ):
        raise RecoveryAcceptanceError("ledger_mismatch")
    _call(
        "source_ledger_malformed",
        recovery_inputs._parse_json,
        top["source_ledger.json"],
        "source_ledger",
    )
    if top["source_ledger.json"] != _call(
        "source_ledger_malformed",
        _canon().canonical_json_bytes,
        recovery_inputs._expected_source_ledger(bundle),
    ):
        raise RecoveryAcceptanceError("source_ledger_mismatch")
    for name in (
        "events.jsonl",
        "selector.jsonl",
        "progress.json",
        "provenance_before.json",
        "provenance_after.json",
    ):
        _call(
            f"{name}_malformed",
            recovery_inputs._parse_jsonl
            if name.endswith(".jsonl")
            else recovery_inputs._parse_json,
            top[name],
            name.replace(".jsonl", "").replace(".json", ""),
        )
    _deadline(deadline)

    _call(
        "input_manifest_malformed",
        recovery_inputs._parse_json,
        top["input_manifest.json"],
        "input_manifest",
    )
    expected_input_manifest = _call(
        "input_manifest_malformed", outer._input_manifest, bundle, stored_plan, view
    )
    if top["input_manifest.json"] != _call(
        "input_manifest_malformed", _canon().canonical_json_bytes, expected_input_manifest
    ):
        raise RecoveryAcceptanceError("input_manifest_mismatch")

    manifest_digest = _call("stage_manifest_mismatch", _sha256_file, stage / "manifest.json")
    if manifest_digest != str(validated.get("stage_manifest_sha256")):
        raise RecoveryAcceptanceError("stage_manifest_mismatch")
    _call("stage_manifest_invalid", pilot._verify_manifest, stage)
    _deadline(deadline)

    unit_manifest_inventory = _call(
        "unit_manifest_inventory_malformed",
        recovery_inputs._parse_json,
        top["unit_manifest_inventory.json"],
        "unit_manifest_inventory",
    )
    _check_recovered_units(
        stage, bundle, stored_plan, stored_inventory, unit_manifest_inventory, deadline
    )
    _call(
        "unit_manifest_changed",
        outer._verify_unit_manifests,
        stage,
        unit_manifest_inventory,
        deadline,
    )
    _deadline(deadline)

    _call(
        "replay_history_mismatch",
        _check_replay_prefix,
        stage,
        stored_plan,
        bundle,
        history,
        deadline,
    )
    _call(
        "original_stage_inventory_changed",
        _collect_and_compare,
        original_stage,
        expected_files,
        expected_dirs,
        deadline,
    )
    _call("recovery_stage_unreadable", recovery_inputs._collect_tree, stage, deadline)
    _deadline(deadline)

    _stream_optimizer_steps(stage, bundle, stored_plan, validated, deadline)
    # Bind the exact bytes parsed above, not a later independently valid file.
    for name, raw in top.items():
        if recovery_inputs._hash_file_record(stage / name, deadline) != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }:
            raise RecoveryAcceptanceError("recovery_evidence_changed")
    _same_bytes(
        _read_mapping(resolved["receipt"], "recovery_receipt", deadline),
        receipt_record,
        "receipt_changed",
    )
    _deadline(deadline)
    if time.perf_counter() > deadline:
        raise RecoveryAcceptanceError("deadline_exceeded")
    accounting = _call(
        "accounting_failed", source.accounting_from_recovered, summary, receipt_record
    )
    if not isinstance(accounting, Mapping):
        raise RecoveryAcceptanceError("accounting_failed")
    return dict(accounting)


def _collect_and_compare(
    original_stage: Path,
    expected_files: set[str],
    expected_dirs: set[str],
    deadline: Any,
) -> None:
    files, directories = recovery_inputs._collect_tree(original_stage, deadline)
    if set(files) != set(expected_files):
        raise RecoveryAcceptanceError("original_stage_inventory_changed")
    if set(directories) != set(expected_dirs):
        raise RecoveryAcceptanceError("original_stage_directory_changed")
