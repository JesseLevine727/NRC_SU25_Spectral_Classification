"""Single-unit execution boundary for the approved P05 recovery.

This module executes exactly one canonical non-pilot ledger unit (12
recipe/seed slots) against an already authenticated recovery bundle, the
existing exclusive recovery stage, the original selector mapping, a fixed
torch CUDA object, a bounded storage budget and a global deadline.  It is the
unit-granular worker a later outer runner drives: it never consumes
full-stage authority, never writes a full-stage receipt, never selects or
exports and never reads outer-test data.  On any failure it preserves the
partially written recovery evidence, marks the in-flight attempt honestly and
raises a stable, path-free :class:`RecoveryUnitError`; it never retries and
never overwrites original evidence.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_evidence as evidence
from atlas_sers.evaluation import p05_recovery_inputs as inputs
from atlas_sers.evaluation import p05_recovery_persistence as persistence
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan

__all__ = ["RecoveryUnitError", "run_unit"]

RECOVERIES_DIRNAME = "recoveries"
RECOVERY_STAGE_NAME = "develop"

_ATTEMPT_REPLAY = "replay"
_ATTEMPT_UNSTARTED = "unstarted"

_COUNTER_KEYS = (
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
)


class RecoveryUnitError(ValueError):
    """Stable, path-free single-unit recovery failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = str(reason_code)
        message = self.reason_code if not detail else f"{self.reason_code}: {detail}"
        super().__init__(message)


def _call(code: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except RecoveryUnitError:
        raise
    except Exception as error:
        detail = getattr(error, "reason_code", None)
        if isinstance(detail, str) and detail:
            raise RecoveryUnitError(code, detail) from error
        raise RecoveryUnitError(code) from error


def _check_deadline(deadline: Any) -> None:
    try:
        inputs._check_deadline(deadline)
    except inputs.RecoveryInputsError as error:
        raise RecoveryUnitError(error.reason_code) from error


def _guard_resources(torch: Any, phase: str) -> None:
    try:
        authority.check_resources(torch, phase=phase)
    except RecoveryUnitError:
        raise
    except authority.RecoveryAuthorityError as error:
        raise RecoveryUnitError(f"resources_{error.reason_code}") from error
    except Exception as error:
        raise RecoveryUnitError("resources_check_failed") from error


def _check_storage(budget: Any, headroom_bytes: int = 0) -> None:
    try:
        budget.check(headroom_bytes=headroom_bytes)
    except RecoveryUnitError:
        raise
    except Exception as error:
        raise RecoveryUnitError("storage_check_failed") from error


def _strict_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryUnitError(code)
    return value


def _validate_counters(counters: Any) -> None:
    if not isinstance(counters, MutableMapping):
        raise RecoveryUnitError("counters_malformed")
    if set(counters) != set(_COUNTER_KEYS):
        raise RecoveryUnitError("counters_malformed")
    for key in _COUNTER_KEYS:
        if key in ("new_optimizer_steps_exact", "new_elapsed_seconds"):
            continue
        if type(counters[key]) is not int or counters[key] < 0:
            raise RecoveryUnitError("counters_malformed")
    elapsed = counters["new_elapsed_seconds"]
    try:
        valid_elapsed = (
            type(elapsed) in (int, float) and math.isfinite(float(elapsed)) and elapsed >= 0
        )
    except (OverflowError, ValueError):
        valid_elapsed = False
    if not valid_elapsed:
        raise RecoveryUnitError("counters_elapsed_invalid")
    if counters["new_optimizer_steps_exact"] is not True or counters["new_failed"] != 0:
        raise RecoveryUnitError("prior_recovery_failure")
    if (
        counters["new_started"] != counters["new_completed"]
        or counters["new_started"] != counters["replay_started"] + counters["unstarted_started"]
        or counters["new_started"] > recovery_plan.RECOVERY_FITS
        or counters["replay_started"] > 1
        or counters["unstarted_started"] > recovery_plan.ORIGINAL_UNSTARTED
        or counters["reused_completed"] > recovery_plan.ORIGINAL_COMPLETED
        or counters["new_peak_cuda_bytes"] > authority.FROZEN_SOURCE_FIT_CUDA_CAP_BYTES
    ):
        raise RecoveryUnitError("counters_inconsistent")
    for count, updates in (
        (counters["new_completed"], counters["new_optimizer_steps"]),
        (counters["reused_completed"], counters["reused_optimizer_steps"]),
    ):
        if not 120 * count <= updates <= 800 * count or updates % 4:
            raise RecoveryUnitError("counters_updates_invalid")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return core._canon().canonical_json_bytes(dict(value))


def _recovery_stage(recovery_bundle: Mapping[str, Any], base: Mapping[str, Any]) -> Path:
    if recovery_bundle.get("recovery_permit_sha256") != authority.RECOVERY_PERMIT_SHA256:
        raise RecoveryUnitError("recovery_permit_identity_mismatch")
    artifact_root = base.get("artifact_root")
    if not isinstance(artifact_root, (str, Path)):
        raise RecoveryUnitError("base_bundle_malformed")
    run_root = inputs._original_run_root(artifact_root)
    declared = recovery_bundle.get("original_run_root")
    if not isinstance(declared, (str, Path)) or Path(declared) != run_root:
        raise RecoveryUnitError("recovery_run_root_mismatch")
    return run_root / RECOVERIES_DIRNAME / authority.RECOVERY_PERMIT_SHA256 / RECOVERY_STAGE_NAME


def _validate_journal_paths(stage: Path, events_path: Any, selector_path: Any) -> tuple[Path, Path]:
    expected_events = stage / "events.jsonl"
    expected_selector = stage / "selector.jsonl"
    if not isinstance(events_path, (str, Path)) or not isinstance(selector_path, (str, Path)):
        raise RecoveryUnitError("recovery_journal_path_invalid")
    if Path(events_path) != expected_events:
        raise RecoveryUnitError("recovery_events_path_mismatch")
    if Path(selector_path) != expected_selector:
        raise RecoveryUnitError("recovery_selector_path_mismatch")
    for path in (stage, expected_events, expected_selector):
        _call("recovery_journal_path_invalid", inputs._reject_symlink_chain, path)
    return expected_events, expected_selector


def _ledger_index(
    ledger: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    units = ledger.get("units")
    slots = ledger.get("slots")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise RecoveryUnitError("ledger_units_malformed")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise RecoveryUnitError("ledger_slots_malformed")
    unit_by_id: dict[str, dict[str, Any]] = {}
    for entry in units:
        if not isinstance(entry, Mapping):
            raise RecoveryUnitError("ledger_units_malformed")
        key = _call("ledger_unit_invalid", inputs._check_component, entry.get("unit_id"))
        if key in unit_by_id:
            raise RecoveryUnitError("ledger_unit_duplicate")
        unit_by_id[key] = dict(entry)
    slot_by_id: dict[str, dict[str, Any]] = {}
    for entry in slots:
        if not isinstance(entry, Mapping):
            raise RecoveryUnitError("ledger_slots_malformed")
        key = _call("ledger_slot_invalid", inputs._check_component, entry.get("slot_id"))
        if entry.get("unit_id") not in unit_by_id:
            raise RecoveryUnitError("ledger_slot_unit_unknown")
        if key in slot_by_id:
            raise RecoveryUnitError("ledger_slot_duplicate")
        slot_by_id[key] = dict(entry)
    return unit_by_id, slot_by_id


def _reused_event(unit: Mapping[str, Any], slot: Mapping[str, Any], result: Any) -> dict[str, Any]:
    return {
        "event": "reused",
        "execution_id": pilot.execution_id(unit, slot),
        "slot_id": str(slot["slot_id"]),
        "unit_id": str(unit["unit_id"]),
        "recipe_id": str(slot["recipe_id"]),
        "seed": int(slot["seed"]),
        "optimizer_steps": int(result.optimizer_steps),
    }


def _append_event(path: Path, entry: Mapping[str, Any]) -> None:
    development._append_jsonl(path, entry)


def _check_replay_prefix(
    result: Any,
    *,
    recovery_bundle: Mapping[str, Any],
    unit_by_id: Mapping[str, Mapping[str, Any]],
    slot_by_id: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    deadline: Any,
) -> None:
    interrupted_slot = slot_by_id.get(str(plan.get("interrupted_slot_id")))
    if interrupted_slot is None:
        raise RecoveryUnitError("interrupted_slot_unknown")
    interrupted_unit = unit_by_id.get(str(interrupted_slot.get("unit_id")))
    if interrupted_unit is None:
        raise RecoveryUnitError("interrupted_unit_unknown")
    identifier = pilot.execution_id(interrupted_unit, interrupted_slot)
    original_stage = recovery_bundle.get("original_stage")
    if not isinstance(original_stage, (str, Path)):
        raise RecoveryUnitError("original_stage_missing")
    history_path = (
        Path(original_stage)
        / "units"
        / str(interrupted_unit["unit_id"])
        / "histories"
        / f"{identifier}.jsonl"
    )
    relative = str(history_path.relative_to(Path(original_stage)))
    anchored = recovery_bundle["original_anchor"]["files"].get(relative)
    inventory = recovery_bundle["original_inventory"].get(relative)
    if anchored != inventory:
        raise RecoveryUnitError("interrupted_history_anchor_mismatch")
    size, digest = _call("interrupted_history_record_invalid", persistence._record_size, anchored)
    raw = _call(
        "interrupted_history_unreadable",
        inputs._read_bytes_bounded,
        history_path,
        size,
        "original_interrupted_history",
        deadline,
    )
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise RecoveryUnitError("interrupted_history_digest_mismatch")
    original = _call(
        "interrupted_history_invalid", inputs._parse_jsonl, raw, "original_interrupted_history"
    )
    if len(original) != recovery_plan.INTERRUPTED_EPOCHS:
        raise RecoveryUnitError("interrupted_history_length_mismatch")
    history = list(result.history)
    if len(history) < recovery_plan.INTERRUPTED_EPOCHS:
        raise RecoveryUnitError("replay_history_short")
    for index in range(recovery_plan.INTERRUPTED_EPOCHS):
        if _canonical(history[index]) != _canonical(original[index]):
            raise RecoveryUnitError("replay_prefix_mismatch")


class _GuardedEpochRecorder:
    """Persist an epoch, then enforce deadline, storage and epoch resources."""

    __slots__ = ("_budget", "_deadline", "_recorder", "_torch")

    def __init__(self, recorder: Any, budget: Any, deadline: Any, torch: Any) -> None:
        self._recorder = recorder
        self._budget = budget
        self._deadline = deadline
        self._torch = torch

    def __call__(self, record: Mapping[str, Any]) -> None:
        self._recorder(record)
        _check_deadline(self._deadline)
        _check_storage(self._budget, development.UNIT_BUDGET_HEADROOM_BYTES)
        _guard_resources(self._torch, "epoch")

    def close(self) -> None:
        self._recorder.close()


def _aggregate(
    *,
    unit_id: str,
    unit_kind: str,
    group: Sequence[Mapping[str, Any]],
    completed_ids: Any,
    interrupted_slot_id: str,
    reused_updates: int,
    new_updates: int,
    unit_new_started: int,
    unit_new_completed: int,
    unit_new_failed: int,
    started: float,
) -> dict[str, Any]:
    completed = {str(item) for item in completed_ids}
    return {
        "unit_id": unit_id,
        "unit_kind": unit_kind,
        "completed_slots": [str(slot["slot_id"]) for slot in group],
        "reused_slots": [
            str(slot["slot_id"]) for slot in group if str(slot["slot_id"]) in completed
        ],
        "new_slots": [
            str(slot["slot_id"]) for slot in group if str(slot["slot_id"]) not in completed
        ],
        "replay_slot_id": interrupted_slot_id if unit_kind == "partial" else None,
        "optimizer_updates_reused": int(reused_updates),
        "optimizer_updates_new": int(new_updates),
        "new_started": int(unit_new_started),
        "new_completed": int(unit_new_completed),
        "new_failed": int(unit_new_failed),
        "fits_started": int(unit_new_started),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_unit(
    *,
    recovery_bundle: Mapping[str, Any],
    unit: Mapping[str, Any],
    expected_selectors: Mapping[str, Mapping[str, Any]],
    torch: Any,
    device: str,
    budget: Any,
    deadline: Any,
    events_path: Path | str,
    selector_path: Path | str,
    counters: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Execute or authenticate exactly one canonical non-pilot ledger unit."""

    started = time.perf_counter()
    _check_deadline(deadline)
    if str(device) != "cuda":
        raise RecoveryUnitError("device_invalid")
    if not isinstance(expected_selectors, Mapping):
        raise RecoveryUnitError("expected_selectors_malformed")
    _validate_counters(counters)
    _guard_resources(torch, "fit")

    base = _call("recovery_bundle_malformed", evidence._base_bundle, recovery_bundle)
    if not isinstance(base, Mapping):
        raise RecoveryUnitError("base_bundle_malformed")
    ledger = base.get("ledger")
    contract = base.get("contract")
    if not isinstance(ledger, Mapping) or not isinstance(contract, Mapping):
        raise RecoveryUnitError("base_bundle_malformed")
    plan = recovery_bundle.get("plan")
    if not isinstance(plan, Mapping):
        raise RecoveryUnitError("recovery_plan_malformed")

    stage = _recovery_stage(recovery_bundle, base)
    _call("recovery_stage_unreadable", inputs._reject_symlink_chain, stage)
    journal_events, journal_selector = _validate_journal_paths(stage, events_path, selector_path)
    if not stage.is_dir() or not (stage / "units").is_dir():
        raise RecoveryUnitError("recovery_stage_missing")
    _call(
        "budget_binding_invalid",
        persistence._check_budget_binding,
        budget,
        inputs._original_run_root(base["artifact_root"]),
    )
    if budget._unit is not None or not {journal_events, journal_selector} <= set(budget._growing):
        raise RecoveryUnitError("budget_tracking_invalid")

    unit_by_id, slot_by_id = _ledger_index(ledger)
    if not isinstance(unit, Mapping):
        raise RecoveryUnitError("unit_malformed")
    unit_id = _call("unit_id_invalid", inputs._check_component, unit.get("unit_id"))
    ledger_unit = unit_by_id.get(unit_id)
    if ledger_unit is None:
        raise RecoveryUnitError("unit_unknown")
    if _canonical(unit) != _canonical(ledger_unit):
        raise RecoveryUnitError("unit_identity_mismatch")

    group = _call("unit_slots_failed", evidence._unit_slots, ledger, unit_id)
    group_ids = {str(slot["slot_id"]) for slot in group}
    reused_ids = {str(item) for item in plan.get("reused_original_slot_ids", ())}
    unstarted_ids = {str(item) for item in plan.get("unstarted_slot_ids", ())}
    interrupted_slot_id = str(plan.get("interrupted_slot_id"))
    if group_ids & {str(item) for item in plan.get("reused_pilot_slot_ids", ())}:
        raise RecoveryUnitError("pilot_unit_forbidden")
    if reused_ids & unstarted_ids or interrupted_slot_id in reused_ids | unstarted_ids:
        raise RecoveryUnitError("plan_slot_overlap")
    if group_ids <= reused_ids:
        unit_kind = "sealed"
    elif interrupted_slot_id in group_ids:
        unit_kind = "partial"
    elif group_ids <= unstarted_ids:
        unit_kind = "new"
    else:
        raise RecoveryUnitError("unit_not_authenticated")
    if (unit_kind == "sealed") != (unit_id in plan.get("sealed_unit_ids", ())):
        raise RecoveryUnitError("unit_kind_mismatch")
    if (unit_kind == "partial") != (unit_id == plan.get("incomplete_unit_id")):
        raise RecoveryUnitError("unit_kind_mismatch")
    expected_destination = stage / "units" / unit_id
    _call("unit_destination_invalid", inputs._reject_symlink_chain, expected_destination)
    if expected_destination.exists() or expected_destination.is_symlink():
        raise RecoveryUnitError("destination_unit_exists")
    _check_storage(budget, development.UNIT_BUDGET_HEADROOM_BYTES)

    if unit_kind == "sealed":
        completed_slots = _call(
            "unit_completion_failed", evidence._completed_slots, group, plan, True
        )
    elif unit_kind == "partial":
        completed_slots = _call(
            "unit_completion_failed", evidence._completed_slots, group, plan, False
        )
    else:
        completed_slots = []
    completed_ids = [str(slot["slot_id"]) for slot in completed_slots]
    completed_id_set = set(completed_ids)

    reused_items: list[dict[str, Any]] = []
    new_items: list[dict[str, Any]] = []
    reused_updates = 0
    new_updates = 0
    unit_new_started = 0
    unit_new_completed = 0
    unit_new_failed = 0
    selector_slot_ids: set[str] = set()
    in_flight = False
    result_available = False
    current_identifier: str | None = None
    unit_dir: Path | None = None
    unit_inputs: Mapping[str, Any] | None = None

    try:
        if unit_kind in ("sealed", "partial"):
            verified = _call(
                "original_unit_authentication_failed",
                evidence.load_verified_original_unit,
                recovery_bundle=recovery_bundle,
                unit=unit,
                expected_selectors=expected_selectors,
                torch=torch,
                device=device,
                deadline=deadline,
            )
            reused_items = list(verified["items"])
            reused_selectors = list(verified["selector_records"])
            reused_updates = _strict_int(
                verified["optimizer_updates_exact"], "optimizer_updates_invalid"
            )
            verified_completed = {str(item) for item in verified["completed_slots"]}
            if verified_completed != completed_id_set:
                raise RecoveryUnitError("completed_slot_mismatch")
            if len(reused_items) != len(completed_ids):
                raise RecoveryUnitError("reused_record_count_mismatch")
            if len(reused_selectors) != len(completed_ids):
                raise RecoveryUnitError("reused_record_count_mismatch")
            if counters["reused_completed"] + len(completed_ids) > recovery_plan.ORIGINAL_COMPLETED:
                raise RecoveryUnitError("reused_ceiling_exceeded")
            copied = _call(
                "original_unit_copy_failed",
                persistence.copy_original_unit,
                recovery_bundle=recovery_bundle,
                unit_id=unit_id,
                budget=budget,
                deadline=deadline,
            )
            if copied.get("unit_active") is not True:
                raise RecoveryUnitError("copied_unit_not_active")
            if copied.get("unit_kind") != unit_kind:
                raise RecoveryUnitError("copied_unit_kind_mismatch")
            unit_dir = Path(copied["destination_unit_dir"])
            if unit_dir != expected_destination:
                raise RecoveryUnitError("copied_unit_destination_mismatch")
            for record in reused_selectors:
                slot_id = str(record.get("slot_id"))
                if slot_id not in completed_id_set:
                    raise RecoveryUnitError("selector_slot_not_completed")
                if slot_id in selector_slot_ids:
                    raise RecoveryUnitError("selector_duplicate")
                selector_slot_ids.add(slot_id)
                _append_event(journal_selector, record)
            by_slot = {str(item["slot"]["slot_id"]): item for item in reused_items}
            for slot in group:
                slot_id = str(slot["slot_id"])
                if slot_id not in completed_id_set:
                    continue
                item = by_slot.get(slot_id)
                if item is None:
                    raise RecoveryUnitError("reused_item_missing")
                _append_event(journal_events, _reused_event(unit, slot, item["result"]))
            counters["reused_completed"] += len(completed_ids)
            counters["reused_optimizer_steps"] += reused_updates

            if unit_kind == "sealed":
                _call("sealed_manifest_verify_failed", pilot._verify_manifest, unit_dir)
                budget.close_unit()
                _check_deadline(deadline)
                _guard_resources(torch, "fit")
                _check_storage(budget)
                _validate_counters(counters)
                return _aggregate(
                    unit_id=unit_id,
                    unit_kind="sealed",
                    group=group,
                    completed_ids=completed_ids,
                    interrupted_slot_id=interrupted_slot_id,
                    reused_updates=reused_updates,
                    new_updates=0,
                    unit_new_started=0,
                    unit_new_completed=0,
                    unit_new_failed=0,
                    started=started,
                )

        if unit_kind == "new":
            _check_storage(budget, development.UNIT_BUDGET_HEADROOM_BYTES)
            unit_dir = _call(
                "unit_activation_failed", budget.activate_unit, stage / "units" / unit_id
            )
        if unit_dir is None:
            raise RecoveryUnitError("unit_directory_missing")

        prepared = _call(
            "unit_inputs_failed",
            pilot.prepare_role_inputs,
            {**base, "units": [unit]},
        )
        if not isinstance(prepared, Mapping) or unit_id not in prepared:
            raise RecoveryUnitError("unit_inputs_missing")
        unit_inputs = prepared[unit_id]

        for slot in group:
            slot_id = str(slot["slot_id"])
            if slot_id in completed_id_set:
                continue
            if slot_id == interrupted_slot_id:
                attempt_kind = _ATTEMPT_REPLAY
            elif slot_id in unstarted_ids:
                attempt_kind = _ATTEMPT_UNSTARTED
            else:
                raise RecoveryUnitError("slot_not_recoverable")

            _check_deadline(deadline)
            _guard_resources(torch, "fit")
            _check_storage(budget, development.UNIT_BUDGET_HEADROOM_BYTES)

            if counters["new_started"] >= recovery_plan.RECOVERY_FITS:
                raise RecoveryUnitError("recovery_fit_ceiling_exceeded")
            if counters["new_optimizer_steps"] + 800 > recovery_plan.RECOVERY_FITS * 800:
                raise RecoveryUnitError("recovery_updates_ceiling_exceeded")

            if attempt_kind == _ATTEMPT_REPLAY:
                if counters["replay_started"] >= 1:
                    raise RecoveryUnitError("replay_already_started")
            else:
                if counters["unstarted_started"] >= recovery_plan.ORIGINAL_UNSTARTED:
                    raise RecoveryUnitError("unstarted_ceiling_exceeded")

            # Charge attempt entry before the exclusive lease operation: a
            # lease or its accounting may fail after it has already persisted.
            # Such failures stop the stage and never restore an attempt credit.
            counters["new_started"] += 1
            unit_new_started += 1
            if attempt_kind == _ATTEMPT_REPLAY:
                counters["replay_started"] += 1
            else:
                counters["unstarted_started"] += 1

            identifier = pilot.execution_id(unit, slot)
            current_identifier = identifier
            in_flight = True
            result_available = False
            _append_event(
                journal_events,
                {
                    "event": "started",
                    "execution_id": identifier,
                    "slot_id": slot_id,
                    "unit_id": unit_id,
                    "recipe_id": str(slot["recipe_id"]),
                    "seed": int(slot["seed"]),
                    "used_fit_count": int(counters["new_started"]),
                    "attempt_kind": attempt_kind,
                    "attempt_entry": "before_exclusive_lease_reservation",
                },
            )
            if attempt_kind == _ATTEMPT_REPLAY:
                _call(
                    "replay_lease_failed",
                    persistence.reserve_replay_lease,
                    recovery_bundle=recovery_bundle,
                    slot=slot,
                    budget=budget,
                    deadline=deadline,
                )
            else:
                lease_path = _call(
                    "unstarted_lease_failed",
                    development._reserve_slot_lease,
                    Path(base["artifact_root"]),
                    base["contract_sha256"],
                    base["core_plan_id"],
                    slot,
                    authority.BASECOMPREHENSIVE_PERMIT_SHA256,
                )
                budget.account_new_file(lease_path)
            recorder = _call(
                "history_recorder_failed",
                pilot.open_history_recorder,
                unit_dir,
                identifier,
            )
            guarded = _GuardedEpochRecorder(recorder, budget, deadline, torch)
            try:
                result = _call(
                    "fit_failed",
                    pilot.train_fit,
                    unit_inputs,
                    unit,
                    slot,
                    device,
                    deadline,
                    guarded,
                )
            finally:
                guarded.close()
            result_available = True
            new_updates += _strict_int(result.optimizer_steps, "optimizer_steps_invalid")
            counters["new_optimizer_steps"] += int(result.optimizer_steps)
            counters["new_elapsed_seconds"] += float(result.elapsed_seconds)
            counters["new_peak_cuda_bytes"] = max(
                int(counters["new_peak_cuda_bytes"]), int(result.peak_cuda_bytes)
            )
            _check_deadline(deadline)
            _guard_resources(torch, "fit")
            _check_storage(budget, development.UNIT_BUDGET_HEADROOM_BYTES)

            _call(
                "result_persist_failed",
                pilot.persist_result,
                torch,
                unit_dir,
                unit,
                slot,
                result,
            )
            if attempt_kind == _ATTEMPT_REPLAY:
                _check_replay_prefix(
                    result,
                    recovery_bundle=recovery_bundle,
                    unit_by_id=unit_by_id,
                    slot_by_id=slot_by_id,
                    plan=plan,
                    deadline=deadline,
                )
            _call(
                "result_acceptance_failed",
                pilot.check_completed_result,
                result,
                unit_dir,
                unit,
                slot,
                contract,
                unit_inputs,
                torch,
                device,
            )
            _call(
                "sparse_support_failed",
                pilot.check_sparse_support,
                unit,
                slot,
                contract,
                result,
            )
            _check_deadline(deadline)
            _guard_resources(torch, "fit")
            _check_storage(budget, development.UNIT_BUDGET_HEADROOM_BYTES)
            record = _call(
                "selector_record_failed",
                base_inputs.selector_record,
                unit,
                slot,
                development._source_identity_adapter(result, unit, slot),
            )
            if record.get("status") != "complete":
                raise RecoveryUnitError("selector_not_complete")
            if slot_id in selector_slot_ids:
                raise RecoveryUnitError("selector_duplicate")
            selector_slot_ids.add(slot_id)
            _append_event(journal_selector, record)
            new_items.append(
                {
                    "slot": slot,
                    "unit": unit,
                    "unit_id": unit_id,
                    "seed": int(slot["seed"]),
                    "result": result,
                }
            )
            counters["new_completed"] += 1
            unit_new_completed += 1
            in_flight = False
            current_identifier = None
            _append_event(
                journal_events,
                {
                    "event": "completed",
                    "execution_id": identifier,
                    "slot_id": slot_id,
                    "status": str(result.status),
                    "optimizer_steps": int(result.optimizer_steps),
                },
            )

        all_items = list(reused_items) + list(new_items)
        if len(all_items) != recovery_plan.SLOTS_PER_UNIT:
            raise RecoveryUnitError("unit_completion_count_mismatch")
        if selector_slot_ids != group_ids:
            raise RecoveryUnitError("unit_selector_coverage_mismatch")
        _call("shared_prefix_check_failed", pilot.check_shared_prefixes, all_items)
        _call(
            "sparse_equivalence_check_failed",
            pilot.check_sparse_equivalences,
            all_items,
            [unit],
            contract,
        )
        _call("manifest_write_failed", core._write_manifest, unit_dir)
        _call("manifest_verify_failed", pilot._verify_manifest, unit_dir)
        budget.close_unit()
        _check_deadline(deadline)
        _guard_resources(torch, "fit")
        _check_storage(budget)
        _validate_counters(counters)
        return _aggregate(
            unit_id=unit_id,
            unit_kind=unit_kind,
            group=group,
            completed_ids=completed_ids,
            interrupted_slot_id=interrupted_slot_id,
            reused_updates=reused_updates,
            new_updates=new_updates,
            unit_new_started=unit_new_started,
            unit_new_completed=unit_new_completed,
            unit_new_failed=unit_new_failed,
            started=started,
        )
    except BaseException as error:
        if in_flight:
            counters["new_failed"] += 1
            unit_new_failed += 1
            if not result_available and current_identifier is not None and unit_dir is not None:
                counters["new_optimizer_steps"] += development._lower_bound_updates(
                    unit_dir, current_identifier
                )
                counters["new_optimizer_steps_exact"] = False
        if isinstance(error, (RecoveryUnitError, KeyboardInterrupt, SystemExit)):
            raise
        raise RecoveryUnitError("unit_execution_failed") from error
    finally:
        development._drop_unit_states(reused_items)
        development._drop_unit_states(new_items)
        unit_inputs = None
