"""Pure, file-IO-free P05 calibration adapter for one refit specification.

Validates the specification, the development ledger, the manifest metadata and
the calibration-slot allowlist before it invokes the injected, already
authorized logits loader. Every calibration slot resolves to its globally
unique ledger unit by ``unit_id``; the slot's inner fitting/validation roles
must match that unit, and the unit's validation UIDs must be a subset of the
specification's outer fitting UIDs. The scalar master-equal temperature fit is
delegated unchanged to the frozen P04 runtime helper, and the calibration plus
a private semantics audit are returned.

It never reads a file, never trains or refits a neural model, never reads an
outer-test score or outcome and never authorizes execution. The scalar
temperature fit is the only fit performed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.classical import TemperatureCalibration
from atlas_sers.evaluation.p04_runtime import _master_equal_calibration
from atlas_sers.evaluation.p05_selection import INHERITED_SLOT_KIND
from atlas_sers.governance.canonical import sha256_value

SCHEMA_VERSION = "nato-sers-p05-calibration-v1"
CLASS_COUNT = 3


class CalibrationError(ValueError):
    """Raised when the specification, ledger, manifest or loaded logits are invalid."""


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CalibrationError(code)
    return value


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CalibrationError(code)
    return value


def _integer(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError(code)
    return value


def _text_tuple(value: Any, code: str) -> list[str]:
    return [_text(raw, code) for raw in _sequence(value, code)]


def _sorted_unique(values: list[str], code: str) -> list[str]:
    if not values:
        raise CalibrationError(f"{code}_empty")
    if values != sorted(values) or len(set(values)) != len(values):
        raise CalibrationError(f"{code}_not_sorted_unique")
    return values


def _ledger_index(ledger: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    slots: dict[str, Any] = {}
    for raw in _sequence(ledger.get("slots"), "ledger_slots_malformed"):
        slot = _mapping(raw, "ledger_slot_malformed")
        slot_id = _text(slot.get("slot_id"), "ledger_slot_id_malformed")
        if slot_id in slots:
            raise CalibrationError("ledger_slot_duplicate")
        slots[slot_id] = slot
    units: dict[str, Any] = {}
    for raw in _sequence(ledger.get("units"), "ledger_units_malformed"):
        unit = _mapping(raw, "ledger_unit_malformed")
        unit_id = _text(unit.get("unit_id"), "ledger_unit_id_malformed")
        if unit_id in units:
            raise CalibrationError("ledger_unit_duplicate")
        units[unit_id] = unit
    if not slots or not units:
        raise CalibrationError("ledger_empty")
    return slots, units


def _manifest_index(manifest: Any) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    master_target: dict[str, str] = {}
    master_station: dict[str, str] = {}
    for raw in _sequence(manifest, "manifest_malformed"):
        row = _mapping(raw, "manifest_row_malformed")
        uid = _text(row.get("observation_uid"), "manifest_uid_malformed")
        master = _text(row.get("master_sample_id"), "manifest_master_malformed")
        target = _text(row.get("target_analyte"), "manifest_target_malformed")
        station = _text(row.get("station"), "manifest_station_malformed")
        _text(row.get("instrument"), "manifest_instrument_malformed")
        if uid in index:
            raise CalibrationError("manifest_uid_duplicate")
        if master_target.get(master, target) != target:
            raise CalibrationError("manifest_master_target_conflict")
        if master_station.get(master, station) != station:
            raise CalibrationError("manifest_master_station_conflict")
        master_target[master] = target
        master_station[master] = station
        index[uid] = {"master": master, "target": target, "station": station}
    if not index:
        raise CalibrationError("manifest_empty")
    return index


def calibrate_spec(
    *, spec: Any, ledger: Any, manifest: Any, load_logits: Any
) -> tuple[TemperatureCalibration, dict[str, Any]]:
    """Fit the frozen master-equal temperature for one refit specification."""

    spec = _mapping(spec, "spec_malformed")
    context_id = _text(spec.get("context_id"), "spec_context_malformed")
    recipe_id = _text(spec.get("recipe_id"), "spec_recipe_malformed")
    seed = _integer(spec.get("seed"), "spec_seed_malformed")
    refit_id = _text(spec.get("refit_id"), "spec_refit_malformed")
    fitting_uids = _sorted_unique(
        _text_tuple(spec.get("fitting_uids"), "spec_fitting_uids_malformed"),
        "spec_fitting_uids",
    )
    classes = _sorted_unique(
        _text_tuple(spec.get("classes"), "spec_classes_malformed"), "spec_classes"
    )
    if len(classes) != CLASS_COUNT:
        raise CalibrationError("spec_classes_not_three")
    calibration_slot_ids = _sorted_unique(
        _text_tuple(spec.get("calibration_slot_ids"), "spec_calibration_slots_malformed"),
        "spec_calibration_slots",
    )

    ledger = _mapping(ledger, "ledger_malformed")
    slots, units = _ledger_index(ledger)
    manifest_index = _manifest_index(manifest)

    fitting_set = set(fitting_uids)
    if any(uid not in manifest_index for uid in fitting_uids):
        raise CalibrationError("spec_fitting_uid_unknown")
    fitting_rows = [manifest_index[uid] for uid in fitting_uids]
    if len({row["station"] for row in fitting_rows}) != 1:
        raise CalibrationError("spec_fitting_station_mismatch")
    if {row["target"] for row in fitting_rows} != set(classes):
        raise CalibrationError("spec_fitting_classes_mismatch")

    registered = {
        slot_id
        for slot_id, slot in slots.items()
        if str(slot.get("context_id")) == context_id
        and str(slot.get("recipe_id")) == recipe_id
        and slot.get("seed") == seed
        and slot.get("slot_kind") == INHERITED_SLOT_KIND
        and slot.get("excluded_by_protocol") is not True
    }
    if not registered:
        raise CalibrationError("recipe_seed_unregistered")

    plan: list[tuple[str, Mapping[str, Any], list[str]]] = []
    for slot_id in calibration_slot_ids:
        slot = slots.get(slot_id)
        if slot is None:
            raise CalibrationError("calibration_slot_missing")
        if str(slot.get("context_id")) != context_id:
            raise CalibrationError("calibration_slot_cross_context")
        if str(slot.get("recipe_id")) != recipe_id:
            raise CalibrationError("calibration_slot_other_recipe")
        if slot.get("seed") != seed:
            raise CalibrationError("calibration_slot_other_seed")
        if slot.get("slot_kind") != INHERITED_SLOT_KIND:
            raise CalibrationError("calibration_slot_kind_guard")
        if slot.get("unit_kind") != "inherited":
            raise CalibrationError("calibration_slot_unit_kind_guard")
        if slot.get("excluded_by_protocol") is True:
            raise CalibrationError("calibration_slot_excluded_guard")
        unit = units.get(_text(slot.get("unit_id"), "ledger_slot_unit_malformed"))
        if unit is None:
            raise CalibrationError("calibration_slot_unit_missing")
        if str(unit.get("context_id")) != context_id:
            raise CalibrationError("calibration_unit_cross_context")
        if str(unit.get("selection_unit_id")) != str(slot.get("selection_unit_id")):
            raise CalibrationError("calibration_unit_selection_mismatch")
        if str(unit.get("fitting_role_id")) != str(slot.get("fitting_role_id")):
            raise CalibrationError("calibration_unit_fitting_role_mismatch")
        if str(unit.get("validation_role_id")) != str(slot.get("validation_role_id")):
            raise CalibrationError("calibration_unit_validation_mismatch")
        if unit.get("unit_kind") != "inherited":
            raise CalibrationError("calibration_unit_kind_guard")
        expected_uids = _sorted_unique(
            _text_tuple(unit.get("validation_uids"), "ledger_unit_validation_uids_malformed"),
            "ledger_unit_validation_uids",
        )
        if not set(expected_uids) <= fitting_set:
            raise CalibrationError("calibration_validation_uid_outside_source")
        plan.append((slot_id, slot, expected_uids))
    if registered != set(calibration_slot_ids):
        raise CalibrationError("calibration_slot_set_incomplete")

    rows: list[dict[str, Any]] = []
    ordered_uids: list[str] = []
    ordered_masters: list[str] = []
    ordered_labels: list[str] = []
    ordered_logits: list[list[float]] = []
    class_set = set(classes)
    for _slot_id, slot, expected_uids in plan:
        payload = load_logits(slot)
        if not isinstance(payload, tuple) or len(payload) != 3:
            raise CalibrationError("slot_logits_payload_malformed")
        raw_logits, raw_classes, raw_uids = payload
        if tuple(str(value) for value in _sequence(raw_classes, "slot_classes_malformed")) != tuple(
            classes
        ):
            raise CalibrationError("slot_class_order_mismatch")
        logits = np.asarray(raw_logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != CLASS_COUNT or not np.isfinite(logits).all():
            raise CalibrationError("slot_logits_malformed")
        uids = tuple(str(value) for value in _sequence(raw_uids, "slot_uids_malformed"))
        if len(uids) != logits.shape[0]:
            raise CalibrationError("slot_logits_uid_length_mismatch")
        if uids != tuple(expected_uids):
            raise CalibrationError("slot_validation_uid_order_mismatch")
        for index, uid in enumerate(uids):
            metadata = manifest_index.get(uid)
            if metadata is None:
                raise CalibrationError("calibration_uid_unknown")
            if metadata["target"] not in class_set:
                raise CalibrationError("calibration_label_outside_classes")
            rows.append(
                {
                    "logit_0": float(logits[index, 0]),
                    "logit_1": float(logits[index, 1]),
                    "logit_2": float(logits[index, 2]),
                    "true_label": metadata["target"],
                    "master_sample_id": metadata["master"],
                }
            )
            ordered_uids.append(uid)
            ordered_masters.append(metadata["master"])
            ordered_labels.append(metadata["target"])
            ordered_logits.append([float(value) for value in logits[index]])
    if not rows:
        raise CalibrationError("calibration_rows_empty")

    frame = pd.DataFrame.from_records(
        rows, columns=["logit_0", "logit_1", "logit_2", "true_label", "master_sample_id"]
    )
    calibration = _master_equal_calibration(frame, tuple(classes))
    if (
        not calibration.optimizer_success
        or not np.isfinite(calibration.temperature)
        or calibration.temperature <= 0
        or not np.isfinite(calibration.optimizer_objective)
    ):
        raise CalibrationError("calibration_fit_invalid")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "context_id": context_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "refit_id": refit_id,
        "calibration_slot_ids": list(calibration_slot_ids),
        "slot_count": len(calibration_slot_ids),
        "row_count": len(frame),
        "unique_observation_uid_count": len(set(ordered_uids)),
        "unique_master_sample_id_count": len(set(ordered_masters)),
        "temperature": calibration.temperature,
        "optimizer_success": calibration.optimizer_success,
        "optimizer_objective": calibration.optimizer_objective,
        "calibration_state_sha256": calibration.state_sha256,
        "semantics_input_sha256": sha256_value(
            {
                "calibration_slot_ids": list(calibration_slot_ids),
                "observation_uids": ordered_uids,
                "classes": list(classes),
                "logits": ordered_logits,
                "master_sample_ids": ordered_masters,
                "true_labels": ordered_labels,
            }
        ),
    }
    return calibration, audit
