"""Synthetic tests for the pure, file-IO-free P05 calibration adapter."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import pandas as pd  # noqa: E402

from atlas_sers.evaluation import p05_calibration  # noqa: E402
from atlas_sers.evaluation.classical import TemperatureCalibration  # noqa: E402
from atlas_sers.evaluation.p04_runtime import _master_equal_calibration  # noqa: E402
from atlas_sers.evaluation.p05_calibration import (  # noqa: E402
    CalibrationError,
    calibrate_spec,
)
from atlas_sers.evaluation.p05_selection import INHERITED_SLOT_KIND  # noqa: E402

CLASSES = ("cls-1", "cls-2", "cls-3")
CONTEXT = "ctx-A"
RECIPE = "R0"
SEED = 11
OUTER_ROLE = "outer_fit"
GUARD_SLOT_KIND = "guard_selection_fit"


def _manifest_rows():
    return [
        {
            "observation_uid": "u1",
            "master_sample_id": "m1",
            "target_analyte": "cls-1",
            "station": "st1",
            "instrument": "inst1",
        },
        {
            "observation_uid": "u2",
            "master_sample_id": "m1",
            "target_analyte": "cls-1",
            "station": "st1",
            "instrument": "inst2",
        },
        {
            "observation_uid": "u3",
            "master_sample_id": "m2",
            "target_analyte": "cls-2",
            "station": "st1",
            "instrument": "inst1",
        },
        {
            "observation_uid": "u4",
            "master_sample_id": "m2",
            "target_analyte": "cls-2",
            "station": "st1",
            "instrument": "inst2",
        },
        {
            "observation_uid": "u5",
            "master_sample_id": "m3",
            "target_analyte": "cls-3",
            "station": "st1",
            "instrument": "inst1",
        },
        {
            "observation_uid": "u6",
            "master_sample_id": "m3",
            "target_analyte": "cls-3",
            "station": "st1",
            "instrument": "inst2",
        },
    ]


def _logits_a1():
    return np.asarray([[2.0, 0.5, -1.0], [0.1, 1.5, -0.2]], dtype=np.float64)


def _logits_a2():
    return np.asarray([[1.2, -0.3, 0.7], [-0.5, 2.0, 0.4]], dtype=np.float64)


def _build_case():
    manifest = _manifest_rows()
    units = [
        {
            "unit_id": "unit-A1",
            "unit_kind": "inherited",
            "context_id": CONTEXT,
            "selection_unit_id": "sel-A1",
            "fitting_role_id": "inner_fit_A1",
            "validation_role_id": "inner_val_A1",
            "validation_uids": ["u1", "u2"],
            "excluded_by_protocol": False,
        },
        {
            "unit_id": "unit-A2",
            "unit_kind": "inherited",
            "context_id": CONTEXT,
            "selection_unit_id": "sel-A2",
            "fitting_role_id": "inner_fit_A2",
            "validation_role_id": "inner_val_A2",
            "validation_uids": ["u1", "u3"],
            "excluded_by_protocol": False,
        },
    ]
    slots = [
        {
            "slot_id": "slot-A1",
            "slot_kind": INHERITED_SLOT_KIND,
            "unit_id": "unit-A1",
            "unit_kind": "inherited",
            "context_id": CONTEXT,
            "selection_unit_id": "sel-A1",
            "fitting_role_id": "inner_fit_A1",
            "validation_role_id": "inner_val_A1",
            "recipe_id": RECIPE,
            "seed": SEED,
            "excluded_by_protocol": False,
        },
        {
            "slot_id": "slot-A2",
            "slot_kind": INHERITED_SLOT_KIND,
            "unit_id": "unit-A2",
            "unit_kind": "inherited",
            "context_id": CONTEXT,
            "selection_unit_id": "sel-A2",
            "fitting_role_id": "inner_fit_A2",
            "validation_role_id": "inner_val_A2",
            "recipe_id": RECIPE,
            "seed": SEED,
            "excluded_by_protocol": False,
        },
    ]
    ledger = {"slots": slots, "units": units}
    spec = {
        "context_id": CONTEXT,
        "recipe_id": RECIPE,
        "seed": SEED,
        "refit_id": "refit-1",
        "fitting_role_id": OUTER_ROLE,
        "fitting_uids": ["u1", "u2", "u3", "u4", "u5", "u6"],
        "classes": list(CLASSES),
        "calibration_slot_ids": ["slot-A1", "slot-A2"],
    }
    payloads = {
        "slot-A1": (_logits_a1(), CLASSES, ("u1", "u2")),
        "slot-A2": (_logits_a2(), CLASSES, ("u1", "u3")),
    }
    return spec, ledger, manifest, payloads


def _loader(payloads):
    calls = []

    def load_logits(slot):
        calls.append(slot["slot_id"])
        return payloads[slot["slot_id"]]

    return load_logits, calls


def _slot(ledger, slot_id):
    return next(slot for slot in ledger["slots"] if slot["slot_id"] == slot_id)


def _unit(ledger, unit_id):
    return next(unit for unit in ledger["units"] if unit["unit_id"] == unit_id)


def test_matches_frozen_master_equal_calibration_exactly():
    spec, ledger, manifest, payloads = _build_case()
    load_logits, calls = _loader(payloads)

    calibration, audit = calibrate_spec(
        spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits
    )

    reference_frame = pd.DataFrame.from_records(
        [
            {
                "logit_0": 2.0,
                "logit_1": 0.5,
                "logit_2": -1.0,
                "true_label": "cls-1",
                "master_sample_id": "m1",
            },
            {
                "logit_0": 0.1,
                "logit_1": 1.5,
                "logit_2": -0.2,
                "true_label": "cls-1",
                "master_sample_id": "m1",
            },
            {
                "logit_0": 1.2,
                "logit_1": -0.3,
                "logit_2": 0.7,
                "true_label": "cls-1",
                "master_sample_id": "m1",
            },
            {
                "logit_0": -0.5,
                "logit_1": 2.0,
                "logit_2": 0.4,
                "true_label": "cls-2",
                "master_sample_id": "m2",
            },
        ],
        columns=["logit_0", "logit_1", "logit_2", "true_label", "master_sample_id"],
    )
    reference = _master_equal_calibration(reference_frame, CLASSES)

    assert isinstance(calibration, TemperatureCalibration)
    assert calibration.temperature == reference.temperature
    assert calibration.optimizer_objective == reference.optimizer_objective
    assert calibration.optimizer_success == reference.optimizer_success
    assert calibration.state_sha256 == reference.state_sha256
    assert calls == ["slot-A1", "slot-A2"]
    assert audit["slot_count"] == 2
    assert audit["row_count"] == 4
    assert audit["unique_observation_uid_count"] == 3
    assert audit["unique_master_sample_id_count"] == 2
    assert audit["calibration_slot_ids"] == ["slot-A1", "slot-A2"]
    assert audit["calibration_state_sha256"] == calibration.state_sha256
    assert len(audit["semantics_input_sha256"]) == 64


def test_inner_source_roles_may_differ_from_outer_refit_role():
    spec, ledger, manifest, payloads = _build_case()
    assert spec["fitting_role_id"] != _unit(ledger, "unit-A1")["fitting_role_id"]

    load_logits, calls = _loader(payloads)
    calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-A1", "slot-A2"]


def test_same_selection_unit_id_across_contexts_resolves_by_unit_id():
    manifest = _manifest_rows()
    shared = "sel-shared"
    units = [
        {
            "unit_id": "unit-A",
            "unit_kind": "inherited",
            "context_id": "ctx-A",
            "selection_unit_id": shared,
            "fitting_role_id": "inner_fit_A",
            "validation_role_id": "inner_val_A",
            "validation_uids": ["u1", "u2"],
            "excluded_by_protocol": False,
        },
        {
            "unit_id": "unit-B",
            "unit_kind": "inherited",
            "context_id": "ctx-B",
            "selection_unit_id": shared,
            "fitting_role_id": "inner_fit_B",
            "validation_role_id": "inner_val_B",
            "validation_uids": ["u3", "u4"],
            "excluded_by_protocol": False,
        },
    ]
    slots = [
        {
            "slot_id": "slot-A",
            "slot_kind": INHERITED_SLOT_KIND,
            "unit_id": "unit-A",
            "unit_kind": "inherited",
            "context_id": "ctx-A",
            "selection_unit_id": shared,
            "fitting_role_id": "inner_fit_A",
            "validation_role_id": "inner_val_A",
            "recipe_id": RECIPE,
            "seed": SEED,
            "excluded_by_protocol": False,
        },
        {
            "slot_id": "slot-B",
            "slot_kind": INHERITED_SLOT_KIND,
            "unit_id": "unit-B",
            "unit_kind": "inherited",
            "context_id": "ctx-B",
            "selection_unit_id": shared,
            "fitting_role_id": "inner_fit_B",
            "validation_role_id": "inner_val_B",
            "recipe_id": RECIPE,
            "seed": SEED,
            "excluded_by_protocol": False,
        },
    ]
    ledger = {"slots": slots, "units": units}
    spec = {
        "context_id": "ctx-B",
        "recipe_id": RECIPE,
        "seed": SEED,
        "refit_id": "refit-B",
        "fitting_role_id": OUTER_ROLE,
        "fitting_uids": ["u1", "u2", "u3", "u4", "u5", "u6"],
        "classes": list(CLASSES),
        "calibration_slot_ids": ["slot-B"],
    }
    payloads = {"slot-B": (_logits_a2(), CLASSES, ("u3", "u4"))}
    load_logits, calls = _loader(payloads)

    _, audit = calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-B"]
    assert audit["context_id"] == "ctx-B"
    assert audit["calibration_slot_ids"] == ["slot-B"]


def test_rejects_missing_calibration_slot():
    spec, ledger, manifest, payloads = _build_case()
    spec["calibration_slot_ids"] = ["slot-A1", "slot-A2", "slot-missing"]
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_slot_missing"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_incomplete_calibration_slot_set():
    spec, ledger, manifest, payloads = _build_case()
    spec["calibration_slot_ids"] = ["slot-A1"]
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_slot_set_incomplete"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_duplicate_calibration_slot():
    spec, ledger, manifest, payloads = _build_case()
    spec["calibration_slot_ids"] = ["slot-A1", "slot-A1"]
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="spec_calibration_slots_not_sorted_unique"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_unsorted_calibration_slot():
    spec, ledger, manifest, payloads = _build_case()
    spec["calibration_slot_ids"] = ["slot-A2", "slot-A1"]
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="spec_calibration_slots_not_sorted_unique"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_guard_slot():
    spec, ledger, manifest, payloads = _build_case()
    _slot(ledger, "slot-A2")["slot_kind"] = GUARD_SLOT_KIND
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_slot_kind_guard"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_slot_from_other_context():
    spec, ledger, manifest, payloads = _build_case()
    _slot(ledger, "slot-A2")["context_id"] = "ctx-Z"
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_slot_cross_context"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_unit_from_other_context():
    spec, ledger, manifest, payloads = _build_case()
    _unit(ledger, "unit-A2")["context_id"] = "ctx-Z"
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_unit_cross_context"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_slot_with_other_recipe():
    spec, ledger, manifest, payloads = _build_case()
    _slot(ledger, "slot-A2")["recipe_id"] = "R9"
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_slot_other_recipe"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_slot_with_other_seed():
    spec, ledger, manifest, payloads = _build_case()
    _slot(ledger, "slot-A2")["seed"] = 99
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_slot_other_seed"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_unauthorized_validation_uid_before_loading():
    spec, ledger, manifest, payloads = _build_case()
    _unit(ledger, "unit-A2")["validation_uids"] = ["u1", "u9"]
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="calibration_validation_uid_outside_source"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


def test_rejects_wrong_validation_uid_order():
    spec, ledger, manifest, payloads = _build_case()
    payloads["slot-A1"] = (_logits_a1(), CLASSES, ("u2", "u1"))
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="slot_validation_uid_order_mismatch"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-A1"]


def test_rejects_wrong_class_order():
    spec, ledger, manifest, payloads = _build_case()
    payloads["slot-A1"] = (_logits_a1(), ("cls-3", "cls-2", "cls-1"), ("u1", "u2"))
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="slot_class_order_mismatch"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-A1"]


def test_rejects_wrong_logit_shape():
    spec, ledger, manifest, payloads = _build_case()
    payloads["slot-A1"] = (np.zeros((2, 4)), CLASSES, ("u1", "u2"))
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="slot_logits_malformed"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-A1"]


def test_rejects_nonfinite_logits():
    spec, ledger, manifest, payloads = _build_case()
    payloads["slot-A1"] = (
        np.asarray([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        CLASSES,
        ("u1", "u2"),
    )
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="slot_logits_malformed"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-A1"]


def test_rejects_uid_length_mismatch():
    spec, ledger, manifest, payloads = _build_case()
    payloads["slot-A1"] = (_logits_a1(), CLASSES, ("u1",))
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="slot_logits_uid_length_mismatch"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-A1"]


def test_rejects_conflicting_master_labels():
    spec, ledger, manifest, payloads = _build_case()
    manifest[1]["target_analyte"] = "cls-2"
    load_logits, calls = _loader(payloads)

    with pytest.raises(CalibrationError, match="manifest_master_target_conflict"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == []


@pytest.mark.parametrize(
    "success,temperature",
    [(False, 1.0), (True, float("nan")), (True, 0.0)],
)
def test_rejects_invalid_calibration(monkeypatch, success, temperature):
    spec, ledger, manifest, payloads = _build_case()
    load_logits, calls = _loader(payloads)

    def fake_calibration(frame, classes):
        return TemperatureCalibration(
            temperature=temperature,
            class_vocabulary=classes,
            observations=len(frame),
            masters=int(frame.master_sample_id.nunique()),
            fit_observation_uid_sha256="0" * 64,
            fit_master_uid_sha256="0" * 64,
            optimizer_success=success,
            optimizer_objective=0.5,
        )

    monkeypatch.setattr(p05_calibration, "_master_equal_calibration", fake_calibration)
    with pytest.raises(CalibrationError, match="calibration_fit_invalid"):
        calibrate_spec(spec=spec, ledger=ledger, manifest=manifest, load_logits=load_logits)
    assert calls == ["slot-A1", "slot-A2"]
