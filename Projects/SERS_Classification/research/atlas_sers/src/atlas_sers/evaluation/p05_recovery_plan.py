"""P05 recovery interruption-prefix planner (pure stdlib, no IO).

This module builds a deterministic, private recovery plan from already-saved
metadata: the frozen development ledger, the original pilot slots, the original
comprehensive event journal, the saved selector evidence, the original
per-slot leases and the durable history of the single interrupted fit.

It authenticates nothing and executes nothing.  It validates that the supplied
records describe exactly one consistent interruption prefix and returns the
identity of the one interrupted slot plus the canonically unstarted slots, so a
later boundary can perform exactly one replay plus the unstarted fits after
authenticating the private files, permits, frozen ledger and checkpoints.  It
performs no ranking, selection, calibration or outer-test evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["RecoveryPlanError", "build_recovery_plan"]

SCHEMA_VERSION = "nato-sers-p05-recovery-plan-v1"

BASE_PERMIT_SHA256 = "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8"
CORE_CONTRACT_SHA256 = "60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae"
CORE_PLAN_ID = "a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37"
LEDGER_ID = "P05DEV-8bf60eeca36d4b4663441eda"
PILOT_PERMIT_SHA256 = "652f5c07a1076a907778a9dd80394203ded9084298a95ce791cb5ee2814e576d"

RECIPES = ("D0-M", "D1", "D2", "D3")
SEEDS = (20260805, 20260817, 20260829)
_RECIPE_SET = frozenset(RECIPES)
_SEED_SET = frozenset(SEEDS)
_PRODUCT = frozenset((recipe, seed) for recipe in RECIPES for seed in SEEDS)

BATCH_DRAWS_PER_EPOCH = 4
SLOTS_PER_UNIT = 12
UNIT_COUNT = 1245
SLOT_COUNT = 14940
PILOT_UNIT_COUNT = 3
PILOT_SLOT_COUNT = 36
NEW_UNIT_COUNT = 1242
NEW_FIT_COUNT = 14904

SEALED_UNIT_COUNT = 726
PARTIAL_COMPLETED = 8
ORIGINAL_STARTED = 8721
ORIGINAL_COMPLETED = 8720
ORIGINAL_INTERRUPTED = 1
ORIGINAL_UNSTARTED = 6183
RECOVERY_FITS = 6184

ORIGINAL_COMPLETED_UPDATES = 1669388
INTERRUPTED_EPOCHS = 17
INTERRUPTED_UPDATES = INTERRUPTED_EPOCHS * BATCH_DRAWS_PER_EPOCH
INTERRUPTED_CHARGE = 800
ORIGINAL_CHARGED_UPPER_BOUND = ORIGINAL_COMPLETED_UPDATES + INTERRUPTED_CHARGE

SELECTOR_COUNT = 8756
LEASE_COUNT = 8757

MINIMUM_FIT_UPDATES = 120
MAXIMUM_FIT_UPDATES = 800
MINIMUM_EPOCH = 1
MAXIMUM_EPOCH = 200
MINIMUM_PREDICTED_CLASS_COUNT = 1
MAXIMUM_PREDICTED_CLASS_COUNT = 3

_MAX_IDENTIFIER_LENGTH = 512

_RUN_STARTED_KEYS = frozenset(
    {
        "event",
        "permit_sha256",
        "core_plan_id",
        "ledger_id",
        "device",
        "new_units",
        "new_fits",
    }
)
_UNIT_STARTED_KEYS = frozenset({"event", "unit_id", "slot_count"})
_STARTED_KEYS = frozenset(
    {
        "event",
        "execution_id",
        "slot_id",
        "unit_id",
        "recipe_id",
        "seed",
        "used_fit_count",
    }
)
_COMPLETED_KEYS = frozenset({"event", "execution_id", "slot_id", "status", "optimizer_steps"})
_UNIT_COMPLETED_KEYS = frozenset({"event", "unit_id", "completed", "started", "optimizer_steps"})
_SELECTOR_KEYS = frozenset(
    {
        "slot_id",
        "context_id",
        "selection_unit_id",
        "slot_kind",
        "fitting_role_id",
        "validation_role_id",
        "recipe_id",
        "seed",
        "status",
        "best_epoch",
        "best_validation_balanced_accuracy",
        "best_validation_nll",
        "best_validation_macro_f1",
        "best_validation_predicted_class_count",
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


class RecoveryPlanError(ValueError):
    """Stable recovery-plan failure with a path-free reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = str(reason_code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise RecoveryPlanError("non_canonical_input") from error


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_nonfinite(value: Any, *, depth: int = 0) -> None:
    if depth > 128:
        raise RecoveryPlanError("input_nesting_exceeded")
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecoveryPlanError("non_finite_number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecoveryPlanError("mapping_key_invalid")
            _reject_nonfinite(item, depth=depth + 1)
        return
    if isinstance(value, (bytes, bytearray)):
        raise RecoveryPlanError("unsupported_input_type")
    if isinstance(value, Sequence):
        for item in value:
            _reject_nonfinite(item, depth=depth + 1)
        return
    raise RecoveryPlanError("unsupported_input_type")


def _identifier(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise RecoveryPlanError("identifier_invalid")
    if not value or len(value) > _MAX_IDENTIFIER_LENGTH:
        raise RecoveryPlanError("identifier_invalid")
    if (
        ".." in value
        or "/" in value
        or "\\" in value
        or not value.isprintable()
        or any(character.isspace() for character in value)
    ):
        raise RecoveryPlanError("identifier_invalid")
    for character in value:
        code = ord(character)
        if code < 0x20 or code == 0x7F:
            raise RecoveryPlanError("identifier_invalid")
    return value


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryPlanError("integer_invalid")
    return value


def _finite_metric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryPlanError("selector_metric_invalid")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise RecoveryPlanError("selector_metric_invalid") from error
    if not math.isfinite(result):
        raise RecoveryPlanError("selector_metric_invalid")
    return result


def _as_sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RecoveryPlanError(code)
    return value


def _require_keys(record: Any, expected: frozenset[str], code: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise RecoveryPlanError(code)
    if frozenset(record.keys()) != expected:
        raise RecoveryPlanError(code)
    return record


def _execution_id(unit: Mapping[str, Any], slot: Mapping[str, Any]) -> str:
    return f"{unit['station']}-{slot['recipe_id']}-{slot['seed']}-{slot['slot_id'][:16]}"


def _validate_started(
    record: Mapping[str, Any],
    unit: Mapping[str, Any],
    slot: Mapping[str, Any],
    expected_used: int,
) -> None:
    if record.get("unit_id") != unit["unit_id"]:
        raise RecoveryPlanError("started_unit_mismatch")
    if record.get("slot_id") != slot["slot_id"]:
        raise RecoveryPlanError("started_slot_mismatch")
    if record.get("recipe_id") != slot["recipe_id"]:
        raise RecoveryPlanError("started_recipe_mismatch")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != slot["seed"]:
        raise RecoveryPlanError("started_seed_mismatch")
    if record.get("execution_id") != _execution_id(unit, slot):
        raise RecoveryPlanError("started_execution_mismatch")
    used = record.get("used_fit_count")
    if isinstance(used, bool) or not isinstance(used, int) or used != expected_used:
        raise RecoveryPlanError("started_used_fit_mismatch")


def _validate_completed(
    record: Mapping[str, Any],
    unit: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> int:
    if record.get("slot_id") != slot["slot_id"]:
        raise RecoveryPlanError("completed_slot_mismatch")
    if record.get("execution_id") != _execution_id(unit, slot):
        raise RecoveryPlanError("completed_execution_mismatch")
    if record.get("status") != "complete":
        raise RecoveryPlanError("completed_status_invalid")
    steps = _strict_int(record.get("optimizer_steps"))
    if steps % BATCH_DRAWS_PER_EPOCH != 0 or not (
        MINIMUM_FIT_UPDATES <= steps <= MAXIMUM_FIT_UPDATES
    ):
        raise RecoveryPlanError("completed_steps_invalid")
    return steps


def _validate_selector(
    record: Mapping[str, Any],
    slot: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> None:
    if record.get("slot_id") != slot["slot_id"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    if record.get("context_id") != unit["context_id"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    if record.get("selection_unit_id") != unit["selection_unit_id"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    if record.get("slot_kind") != slot["slot_kind"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    if record.get("fitting_role_id") != unit["fitting_role_id"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    if record.get("validation_role_id") != unit["validation_role_id"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    if record.get("recipe_id") != slot["recipe_id"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != slot["seed"]:
        raise RecoveryPlanError("selector_identity_mismatch")
    if record.get("status") != "complete":
        raise RecoveryPlanError("selector_status_invalid")
    best_epoch = record.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise RecoveryPlanError("selector_metric_invalid")
    if not (MINIMUM_EPOCH <= best_epoch <= MAXIMUM_EPOCH):
        raise RecoveryPlanError("selector_metric_invalid")
    for field in ("best_validation_balanced_accuracy", "best_validation_macro_f1"):
        numeric = _finite_metric(record.get(field))
        if not (0.0 <= numeric <= 1.0):
            raise RecoveryPlanError("selector_metric_invalid")
    if _finite_metric(record.get("best_validation_nll")) < 0.0:
        raise RecoveryPlanError("selector_metric_invalid")
    predicted = record.get("best_validation_predicted_class_count")
    if isinstance(predicted, bool) or not isinstance(predicted, int):
        raise RecoveryPlanError("selector_metric_invalid")
    if not (MINIMUM_PREDICTED_CLASS_COUNT <= predicted <= MAXIMUM_PREDICTED_CLASS_COUNT):
        raise RecoveryPlanError("selector_metric_invalid")


def build_recovery_plan(
    *,
    ledger: Mapping[str, Any],
    pilot_slots: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    selector_records: Sequence[Mapping[str, Any]],
    leases: Sequence[Mapping[str, Any]],
    interrupted_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one interruption prefix and return a deterministic private plan."""

    if not isinstance(ledger, Mapping):
        raise RecoveryPlanError("ledger_malformed")
    pilot_slots = _as_sequence(pilot_slots, "pilot_slots_malformed")
    events = _as_sequence(events, "events_malformed")
    selector_records = _as_sequence(selector_records, "selector_records_malformed")
    leases = _as_sequence(leases, "leases_malformed")
    interrupted_history = _as_sequence(interrupted_history, "history_malformed")
    for value in (ledger, pilot_slots, events, selector_records, leases, interrupted_history):
        _reject_nonfinite(value)
    evidence_sha256 = {
        "ledger": _sha256_value(ledger),
        "pilot_slots": _sha256_value(list(pilot_slots)),
        "events": _sha256_value(list(events)),
        "selector_records": _sha256_value(list(selector_records)),
        "leases": _sha256_value(list(leases)),
        "interrupted_history": _sha256_value(list(interrupted_history)),
    }

    if ledger.get("ledger_id") != LEDGER_ID:
        raise RecoveryPlanError("ledger_identity_mismatch")
    if ledger.get("execution_authorized") is not False:
        raise RecoveryPlanError("ledger_authorization_invalid")
    if ledger.get("arrays_loaded") is not False:
        raise RecoveryPlanError("ledger_arrays_invalid")
    fits_started = ledger.get("fits_started")
    if isinstance(fits_started, bool) or not isinstance(fits_started, int) or fits_started != 0:
        raise RecoveryPlanError("ledger_fits_invalid")

    units_raw = _as_sequence(ledger.get("units"), "ledger_units_malformed")
    slots_raw = _as_sequence(ledger.get("slots"), "ledger_slots_malformed")
    if len(units_raw) != UNIT_COUNT:
        raise RecoveryPlanError("ledger_unit_count_mismatch")
    if len(slots_raw) != SLOT_COUNT:
        raise RecoveryPlanError("ledger_slot_count_mismatch")

    unit_by_id: dict[str, dict[str, Any]] = {}
    unit_order: list[str] = []
    for raw in units_raw:
        if not isinstance(raw, Mapping):
            raise RecoveryPlanError("ledger_unit_malformed")
        unit_id = _identifier(raw.get("unit_id"))
        if unit_id in unit_by_id:
            raise RecoveryPlanError("ledger_unit_duplicate")
        unit_by_id[unit_id] = {
            "unit_id": unit_id,
            "context_id": _identifier(raw.get("context_id")),
            "selection_unit_id": _identifier(raw.get("selection_unit_id")),
            "fitting_role_id": _identifier(raw.get("fitting_role_id")),
            "validation_role_id": _identifier(raw.get("validation_role_id")),
            "station": _identifier(raw.get("station")),
        }
        unit_order.append(unit_id)

    slots_by_unit: dict[str, list[dict[str, Any]]] = {}
    slot_by_id: dict[str, dict[str, Any]] = {}
    registered_slot_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in slots_raw:
        if not isinstance(raw, Mapping):
            raise RecoveryPlanError("ledger_slot_malformed")
        slot_id = _identifier(raw.get("slot_id"))
        if slot_id in slot_by_id:
            raise RecoveryPlanError("ledger_slot_duplicate")
        unit_id = _identifier(raw.get("unit_id"))
        unit = unit_by_id.get(unit_id)
        if unit is None:
            raise RecoveryPlanError("ledger_slot_unit_unknown")
        recipe_id = raw.get("recipe_id")
        if not isinstance(recipe_id, str) or recipe_id not in _RECIPE_SET:
            raise RecoveryPlanError("ledger_slot_recipe_invalid")
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in _SEED_SET:
            raise RecoveryPlanError("ledger_slot_seed_invalid")
        fitting_role_id = _identifier(raw.get("fitting_role_id"))
        validation_role_id = _identifier(raw.get("validation_role_id"))
        if fitting_role_id != unit["fitting_role_id"]:
            raise RecoveryPlanError("ledger_slot_fitting_role_mismatch")
        if validation_role_id != unit["validation_role_id"]:
            raise RecoveryPlanError("ledger_slot_validation_role_mismatch")
        for field in ("context_id", "selection_unit_id"):
            if field in raw and raw[field] != unit[field]:
                raise RecoveryPlanError("ledger_slot_context_mismatch")
        if raw.get("excluded_by_protocol") is not False:
            raise RecoveryPlanError("ledger_slot_excluded")
        slot = {
            "slot_id": slot_id,
            "unit_id": unit_id,
            "recipe_id": recipe_id,
            "seed": seed,
            "slot_kind": _identifier(raw.get("slot_kind")),
            "fitting_role_id": fitting_role_id,
            "validation_role_id": validation_role_id,
        }
        slot_by_id[slot_id] = slot
        registered_slot_by_id[slot_id] = raw
        slots_by_unit.setdefault(unit_id, []).append(slot)

    if len(slot_by_id) != SLOT_COUNT or len(slots_by_unit) != UNIT_COUNT:
        raise RecoveryPlanError("ledger_coverage_mismatch")
    for unit_id in unit_order:
        group = slots_by_unit.get(unit_id)
        if group is None or len(group) != SLOTS_PER_UNIT:
            raise RecoveryPlanError("ledger_unit_slot_count_mismatch")
        if {(slot["recipe_id"], slot["seed"]) for slot in group} != _PRODUCT:
            raise RecoveryPlanError("ledger_unit_slot_product_mismatch")

    if len(pilot_slots) != PILOT_SLOT_COUNT:
        raise RecoveryPlanError("pilot_slot_count_mismatch")
    pilot_ids: list[str] = []
    pilot_seen: set[str] = set()
    pilot_by_unit: dict[str, list[str]] = {}
    for raw in pilot_slots:
        if not isinstance(raw, Mapping):
            raise RecoveryPlanError("pilot_slot_malformed")
        slot_id = _identifier(raw.get("slot_id"))
        if slot_id in pilot_seen:
            raise RecoveryPlanError("pilot_slot_duplicate")
        ledger_slot = slot_by_id.get(slot_id)
        if ledger_slot is None:
            raise RecoveryPlanError("pilot_slot_unknown")
        if _canonical_bytes(raw) != _canonical_bytes(registered_slot_by_id[slot_id]):
            raise RecoveryPlanError("pilot_registered_slot_mismatch")
        for field in (
            "unit_id",
            "recipe_id",
            "seed",
            "slot_kind",
            "fitting_role_id",
            "validation_role_id",
        ):
            if raw.get(field) != ledger_slot[field]:
                raise RecoveryPlanError("pilot_slot_mismatch")
        if raw.get("excluded_by_protocol") is not False:
            raise RecoveryPlanError("pilot_slot_mismatch")
        pilot_seen.add(slot_id)
        pilot_ids.append(slot_id)
        pilot_by_unit.setdefault(ledger_slot["unit_id"], []).append(slot_id)
    if len(pilot_by_unit) != PILOT_UNIT_COUNT:
        raise RecoveryPlanError("pilot_unit_count_mismatch")
    pilot_unit_ids: list[str] = []
    for unit_id, ids in pilot_by_unit.items():
        group = slots_by_unit[unit_id]
        if len(ids) != SLOTS_PER_UNIT or set(ids) != {slot["slot_id"] for slot in group}:
            raise RecoveryPlanError("pilot_unit_coverage_mismatch")
        pilot_unit_ids.append(unit_id)

    pilot_unit_set = set(pilot_unit_ids)
    new_schedule: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for unit_id in unit_order:
        if unit_id in pilot_unit_set:
            continue
        group = sorted(slots_by_unit[unit_id], key=lambda slot: (slot["recipe_id"], slot["seed"]))
        new_schedule.append((unit_by_id[unit_id], group))
    if len(new_schedule) != NEW_UNIT_COUNT:
        raise RecoveryPlanError("new_unit_count_mismatch")
    new_slot_order = [slot for _unit, group in new_schedule for slot in group]
    if len(new_slot_order) != NEW_FIT_COUNT:
        raise RecoveryPlanError("new_slot_count_mismatch")

    if len(events) == 0:
        raise RecoveryPlanError("event_journal_empty")
    run_started = _require_keys(events[0], _RUN_STARTED_KEYS, "run_started_malformed")
    if run_started.get("event") != "run_started":
        raise RecoveryPlanError("event_sequence_invalid")
    if run_started.get("permit_sha256") != BASE_PERMIT_SHA256:
        raise RecoveryPlanError("run_started_permit_mismatch")
    if run_started.get("core_plan_id") != CORE_PLAN_ID:
        raise RecoveryPlanError("run_started_plan_mismatch")
    if run_started.get("ledger_id") != LEDGER_ID:
        raise RecoveryPlanError("run_started_ledger_mismatch")
    if run_started.get("device") != "cuda":
        raise RecoveryPlanError("run_started_device_mismatch")
    if _strict_int(run_started.get("new_units")) != NEW_UNIT_COUNT:
        raise RecoveryPlanError("run_started_unit_count_mismatch")
    if _strict_int(run_started.get("new_fits")) != NEW_FIT_COUNT:
        raise RecoveryPlanError("run_started_fit_count_mismatch")

    position = 1
    sealed_units: list[str] = []
    completed_ids: list[str] = []
    started_count = 0
    completed_count = 0
    cumulative_steps = 0
    unit_index = 0
    partial_unit_id: str | None = None
    interrupted_slot_id: str | None = None
    partial_completed = 0

    while position < len(events):
        if unit_index >= len(new_schedule):
            raise RecoveryPlanError("event_unit_overflow")
        unit, unit_slots = new_schedule[unit_index]
        unit_id = unit["unit_id"]

        unit_started = _require_keys(events[position], _UNIT_STARTED_KEYS, "unit_started_malformed")
        if unit_started.get("event") != "unit_started":
            raise RecoveryPlanError("event_sequence_invalid")
        if unit_started.get("unit_id") != unit_id:
            raise RecoveryPlanError("event_unit_order_mismatch")
        if _strict_int(unit_started.get("slot_count")) != SLOTS_PER_UNIT:
            raise RecoveryPlanError("event_slot_count_mismatch")
        position += 1

        interrupted = False
        for slot_index in range(SLOTS_PER_UNIT):
            if position >= len(events):
                raise RecoveryPlanError("event_journal_truncated")
            slot = unit_slots[slot_index]
            started = _require_keys(events[position], _STARTED_KEYS, "started_malformed")
            if started.get("event") != "started":
                raise RecoveryPlanError("event_sequence_invalid")
            _validate_started(started, unit, slot, started_count + 1)
            started_count += 1
            position += 1
            next_record = events[position] if position < len(events) else None
            if isinstance(next_record, Mapping) and next_record.get("event") == "completed":
                completed = _require_keys(next_record, _COMPLETED_KEYS, "completed_malformed")
                cumulative_steps += _validate_completed(completed, unit, slot)
                completed_count += 1
                completed_ids.append(slot["slot_id"])
                position += 1
            else:
                interrupted = True
                partial_unit_id = unit_id
                interrupted_slot_id = slot["slot_id"]
                partial_completed = slot_index
                break

        if interrupted:
            if position != len(events):
                raise RecoveryPlanError("event_after_interruption")
            break

        if position >= len(events):
            raise RecoveryPlanError("event_journal_truncated")
        unit_completed = _require_keys(
            events[position], _UNIT_COMPLETED_KEYS, "unit_completed_malformed"
        )
        if unit_completed.get("event") != "unit_completed":
            raise RecoveryPlanError("event_sequence_invalid")
        if unit_completed.get("unit_id") != unit_id:
            raise RecoveryPlanError("event_unit_order_mismatch")
        if _strict_int(unit_completed.get("completed")) != completed_count:
            raise RecoveryPlanError("event_counter_mismatch")
        if _strict_int(unit_completed.get("started")) != started_count:
            raise RecoveryPlanError("event_counter_mismatch")
        if _strict_int(unit_completed.get("optimizer_steps")) != cumulative_steps:
            raise RecoveryPlanError("event_counter_mismatch")
        position += 1
        sealed_units.append(unit_id)
        unit_index += 1

    if interrupted_slot_id is None:
        raise RecoveryPlanError("event_interruption_missing")
    if unit_index != SEALED_UNIT_COUNT:
        raise RecoveryPlanError("event_partial_unit_position_mismatch")
    if partial_completed != PARTIAL_COMPLETED:
        raise RecoveryPlanError("event_partial_unit_count_mismatch")
    if started_count != ORIGINAL_STARTED:
        raise RecoveryPlanError("event_started_count_mismatch")
    if completed_count != ORIGINAL_COMPLETED:
        raise RecoveryPlanError("event_completed_count_mismatch")
    if len(sealed_units) != SEALED_UNIT_COUNT:
        raise RecoveryPlanError("event_sealed_unit_count_mismatch")
    if cumulative_steps != ORIGINAL_COMPLETED_UPDATES:
        raise RecoveryPlanError("event_optimizer_updates_mismatch")

    canonical_completed_ids = [slot["slot_id"] for slot in new_slot_order[:ORIGINAL_COMPLETED]]
    if completed_ids != canonical_completed_ids:
        raise RecoveryPlanError("completed_order_mismatch")
    if interrupted_slot_id != new_slot_order[ORIGINAL_COMPLETED]["slot_id"]:
        raise RecoveryPlanError("interrupted_order_mismatch")
    unstarted_slot_ids = [slot["slot_id"] for slot in new_slot_order[ORIGINAL_STARTED:]]
    if len(unstarted_slot_ids) != ORIGINAL_UNSTARTED:
        raise RecoveryPlanError("unstarted_count_mismatch")
    canonical_sealed_units = [unit["unit_id"] for unit, _group in new_schedule[:SEALED_UNIT_COUNT]]
    if sealed_units != canonical_sealed_units:
        raise RecoveryPlanError("sealed_unit_order_mismatch")
    if partial_unit_id != new_schedule[SEALED_UNIT_COUNT][0]["unit_id"]:
        raise RecoveryPlanError("incomplete_unit_mismatch")

    if len(selector_records) != SELECTOR_COUNT:
        raise RecoveryPlanError("selector_count_mismatch")
    for index, raw in enumerate(selector_records):
        if not isinstance(raw, Mapping):
            raise RecoveryPlanError("selector_malformed")
        _require_keys(raw, _SELECTOR_KEYS, "selector_malformed")
        if index < PILOT_SLOT_COUNT:
            slot = slot_by_id[pilot_ids[index]]
        else:
            slot = slot_by_id[completed_ids[index - PILOT_SLOT_COUNT]]
        _validate_selector(raw, slot, unit_by_id[slot["unit_id"]])

    if len(leases) != LEASE_COUNT:
        raise RecoveryPlanError("lease_count_mismatch")
    lease_ids: set[str] = set()
    pilot_lease_ids: set[str] = set()
    base_lease_ids: set[str] = set()
    for raw in leases:
        if not isinstance(raw, Mapping):
            raise RecoveryPlanError("lease_malformed")
        _require_keys(raw, _LEASE_KEYS, "lease_malformed")
        slot_id = _identifier(raw.get("slot_id"))
        if slot_id in lease_ids:
            raise RecoveryPlanError("lease_duplicate")
        slot = slot_by_id.get(slot_id)
        if slot is None:
            raise RecoveryPlanError("lease_slot_unknown")
        if raw.get("unit_id") != slot["unit_id"]:
            raise RecoveryPlanError("lease_identity_mismatch")
        if raw.get("recipe_id") != slot["recipe_id"]:
            raise RecoveryPlanError("lease_identity_mismatch")
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed != slot["seed"]:
            raise RecoveryPlanError("lease_identity_mismatch")
        if raw.get("contract_sha256") != CORE_CONTRACT_SHA256:
            raise RecoveryPlanError("lease_contract_mismatch")
        if raw.get("core_plan_id") != CORE_PLAN_ID:
            raise RecoveryPlanError("lease_plan_mismatch")
        permit = raw.get("permit_sha256")
        if permit == PILOT_PERMIT_SHA256:
            pilot_lease_ids.add(slot_id)
        elif permit == BASE_PERMIT_SHA256:
            base_lease_ids.add(slot_id)
        else:
            raise RecoveryPlanError("lease_permit_mismatch")
        lease_ids.add(slot_id)
    if pilot_lease_ids != pilot_seen:
        raise RecoveryPlanError("pilot_lease_coverage_mismatch")
    if base_lease_ids != (set(completed_ids) | {interrupted_slot_id}):
        raise RecoveryPlanError("base_lease_coverage_mismatch")

    if len(interrupted_history) != INTERRUPTED_EPOCHS:
        raise RecoveryPlanError("history_length_mismatch")
    for index, raw in enumerate(interrupted_history, start=1):
        if not isinstance(raw, Mapping):
            raise RecoveryPlanError("history_record_malformed")
        for field in ("epoch", "epoch_optimizer_steps", "total_optimizer_steps"):
            if field not in raw:
                raise RecoveryPlanError("history_field_missing")
        if _strict_int(raw.get("epoch")) != index:
            raise RecoveryPlanError("history_epoch_mismatch")
        if _strict_int(raw.get("epoch_optimizer_steps")) != BATCH_DRAWS_PER_EPOCH:
            raise RecoveryPlanError("history_epoch_steps_mismatch")
        if _strict_int(raw.get("total_optimizer_steps")) != BATCH_DRAWS_PER_EPOCH * index:
            raise RecoveryPlanError("history_total_steps_mismatch")

    counts = {
        "source_slots": SLOT_COUNT,
        "pilot_reused": PILOT_SLOT_COUNT,
        "original_started": ORIGINAL_STARTED,
        "original_completed": ORIGINAL_COMPLETED,
        "original_interrupted": ORIGINAL_INTERRUPTED,
        "unstarted": ORIGINAL_UNSTARTED,
        "recovery_fits": RECOVERY_FITS,
        "final_new_source_successes": NEW_FIT_COUNT,
        "final_new_source_attempts": NEW_FIT_COUNT + ORIGINAL_INTERRUPTED,
    }
    optimizer_updates = {
        "original_completed_exact": ORIGINAL_COMPLETED_UPDATES,
        "interrupted_observed_lower_bound": INTERRUPTED_UPDATES,
        "interrupted_charged_upper_bound": INTERRUPTED_CHARGE,
        "original_charged_upper_bound": ORIGINAL_CHARGED_UPPER_BOUND,
    }
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "base_permit_sha256": BASE_PERMIT_SHA256,
        "core_contract_sha256": CORE_CONTRACT_SHA256,
        "core_plan_id": CORE_PLAN_ID,
        "ledger_id": LEDGER_ID,
        "pilot_permit_sha256": PILOT_PERMIT_SHA256,
        "reused_pilot_slot_ids": list(pilot_ids),
        "reused_original_slot_ids": list(completed_ids),
        "interrupted_slot_id": interrupted_slot_id,
        "unstarted_slot_ids": list(unstarted_slot_ids),
        "recovery_slot_ids": [interrupted_slot_id] + list(unstarted_slot_ids),
        "incomplete_unit_id": partial_unit_id,
        "sealed_unit_ids": list(sealed_units),
        "counts": counts,
        "optimizer_updates": optimizer_updates,
        "execution_authorized": False,
        "fits_started": 0,
        "evidence_sha256": evidence_sha256,
    }
    plan["plan_id"] = _sha256_value(plan)
    return plan
