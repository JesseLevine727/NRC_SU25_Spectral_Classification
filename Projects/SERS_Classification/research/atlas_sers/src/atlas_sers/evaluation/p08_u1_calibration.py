"""P08-U1 universal scalar calibration and seed-aggregation adapter.

This is a pure numerical adapter over previously authenticated P08 graph
records and artifacts.  It performs no file IO, no training, no authority
grant, no new optimizer and no new hyperparameter search.  It delegates the
scalar temperature fit unchanged to the frozen P03 cross-fitted kernel and
the frozen P04 master-equal kernel, and delegates the classical seed
ensemble unchanged to the frozen P03 aggregation kernel.

The controller remains responsible for verifying persisted receipts and exact
artifact bytes.  This module only checks in-memory job hashes, schema, policy,
stage, model, resolution, exact dependency sets, metadata agreement and the
finite/ordered structure of the score and probability frames it receives.
Every dependency mapping is keyed by ``job_id`` and carries
``{"job": <graph record>, "predictions": <pd.DataFrame>}``.  For a classical
seed ensemble the single scalar dependency additionally carries
``{"job": <scalar job>, "calibration": <TemperatureCalibration>}``.

No held/outer-test row may enter calibration.  Calibration frames must be
scored on subsets of the caller-supplied source-only outer-fitting metadata.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p03_runtime, p04_runtime, p08_plan
from atlas_sers.evaluation.classical import TemperatureCalibration
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.classical import STOCHASTIC_MODELS

__all__ = [
    "CalibrationAdapterError",
    "SCHEMA_VERSION",
    "calibrate",
    "ensemble",
]

SCHEMA_VERSION = "nato-sers-p08-u1-calibration-v1"

UNIVERSAL_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
CLASSICAL_MODELS = frozenset(("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES"))
NEURAL_RECIPES = frozenset(("D0-M", "D1", "D2", "D3"))
MODEL_IDS = CLASSICAL_MODELS | NEURAL_RECIPES
STOCHASTIC = frozenset(STOCHASTIC_MODELS)

SOURCE_VALIDATION_PREDICTION = "source_validation_prediction"
CALIBRATION_VALIDATION_PREDICTION = "calibration_validation_prediction"
CALIBRATION_PREDICTION_ALIAS = p08_plan.CALIBRATION_ALIAS_STAGE
HELD_PREDICTION = "held_prediction"
SEED_ENSEMBLE_PREDICTION = "seed_ensemble_prediction"
SCALAR_CALIBRATION = p08_plan.SCALAR_STAGE

_RESOLUTION_UNCALIBRATED = p08_plan.RESOLUTION_UNCALIBRATED
_RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE = (
    p08_plan.RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE
)
_RESOLUTION_CALIBRATE_SEED_AVERAGED = p08_plan.RESOLUTION_CALIBRATE_SEED_AVERAGED
_RESOLVE_SELECTED_CANDIDATE = p08_plan.RESOLVE_SELECTED_CANDIDATE
_SOURCE_SELECTION_DEPENDENT = p08_plan.SOURCE_SELECTION_DEPENDENT
_SOURCE_EPOCH_DEPENDENT = p08_plan.SOURCE_EPOCH_DEPENDENT
_FIXED_SPEC = p08_plan.FIXED_SPEC

_PROBABILITY_CLIP = (1e-7, 1.0 - 1e-7)
_STANDARD_COLUMNS = (
    "observation_uid",
    "master_sample_id",
    "instrument",
    "station",
    "true_label",
    "fit_id",
    "predicted_label",
    "class_vocabulary",
    "scores",
    "probabilities",
    "probability_status",
)
_KEY_COLUMNS = (
    "observation_uid",
    "master_sample_id",
    "instrument",
    "station",
    "true_label",
)


class CalibrationAdapterError(ValueError):
    """Raised when a job, dependency, frame or calibration value is invalid."""


def _fail(code: str) -> None:
    raise CalibrationAdapterError(code)


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)
    return value


def _as_classes(classes: Any) -> tuple[str, ...]:
    if isinstance(classes, (str, bytes)) or not isinstance(classes, Sequence):
        _fail("classes_invalid")
    values = list(classes)
    if len(values) != 3:
        _fail("classes_invalid")
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            _fail("classes_invalid")
    if len(set(values)) != 3 or values != sorted(values):
        _fail("classes_invalid")
    return tuple(values)


def _graph_job(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("job_malformed")
    record = dict(value)
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not job_id.startswith("P08JOB-"):
        _fail("job_id_invalid")
    allowed = set(p08_plan.JOB_FIELDS) | {"job_id", "dependencies"}
    extras = [field for field in record if field not in allowed]
    if extras:
        _fail("job_unknown_field")
    missing = [field for field in p08_plan.JOB_FIELDS if field not in record]
    if missing:
        _fail("job_fields_missing")
    payload = {key: record[key] for key in record if key != "job_id"}
    if "P08JOB-" + p08_plan._hash(payload) != job_id:
        _fail("job_hash_invalid")
    return record


def _require_job_identity(job_record: Mapping[str, Any]) -> None:
    policy = job_record.get("policy_id")
    if policy not in UNIVERSAL_POLICIES:
        _fail("job_policy_invalid")
    if job_record.get("representation_id") != p08_plan.POLICY_REPRESENTATION[policy]:
        _fail("job_representation_invalid")
    if job_record.get("model_id") not in MODEL_IDS:
        _fail("job_model_invalid")


def _require_resolution(job_record: Mapping[str, Any], expected: str) -> None:
    if job_record.get("resolution") != expected:
        _fail("job_resolution_invalid")


def _dependency_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("dependencies_must_be_mapping")
    return value


def _require_dependency_set(
    job_record: Mapping[str, Any], dependencies: Mapping[str, Any]
) -> None:
    declared = job_record.get("dependencies")
    if not isinstance(declared, list):
        _fail("job_dependencies_malformed")
    if len(declared) != len(set(declared)):
        _fail("job_dependencies_duplicate")
    if declared != sorted(declared):
        _fail("job_dependencies_not_sorted")
    if set(declared) != set(dependencies.keys()):
        _fail("dependency_set_mismatch")
    for key in dependencies:
        if not isinstance(key, str):
            _fail("dependency_key_invalid")


def _entry(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("dependency_entry_malformed")
    if "job" not in value:
        _fail("dependency_entry_missing_job")
    return value


def _entry_predictions(entry: Mapping[str, Any]) -> pd.DataFrame:
    if "predictions" not in entry:
        _fail("dependency_entry_missing_predictions")
    frame = entry["predictions"]
    if not isinstance(frame, pd.DataFrame):
        _fail("dependency_predictions_invalid")
    return frame


def _entry_calibration(entry: Mapping[str, Any]) -> TemperatureCalibration:
    calibration = entry.get("calibration")
    if not isinstance(calibration, TemperatureCalibration):
        _fail("dependency_calibration_invalid")
    return calibration


def _require_dependency(
    job_record: Mapping[str, Any],
    dep_job: Mapping[str, Any],
    allowed_stages: Sequence[str],
) -> None:
    if dep_job["policy_id"] != job_record["policy_id"]:
        _fail("dependency_policy_mismatch")
    if dep_job["context_id"] != job_record["context_id"]:
        _fail("dependency_context_mismatch")
    if dep_job["model_id"] != job_record["model_id"]:
        _fail("dependency_model_mismatch")
    if dep_job["representation_id"] != job_record["representation_id"]:
        _fail("dependency_representation_mismatch")
    if dep_job["model_spec_sha256"] != job_record["model_spec_sha256"]:
        _fail("dependency_model_spec_mismatch")
    if dep_job["array_sha256"] != job_record["array_sha256"]:
        _fail("dependency_array_mismatch")
    if dep_job["stage"] not in allowed_stages:
        _fail("dependency_stage_invalid")
    # Scalar calibration consumes source jobs with the same reserved test hash.
    # An ensemble's scalar dependency has no held-test role of its own.
    if (job_record["stage"] == SCALAR_CALIBRATION or dep_job["stage"] == HELD_PREDICTION) and (
        dep_job["test_uid_sha256"] != job_record["test_uid_sha256"]
    ):
        _fail("dependency_test_uid_mismatch")


def _uid_hash(values: Any) -> str:
    return sha256_value(sorted(str(value) for value in values))


def _require_uid_hash(frame: pd.DataFrame, expected: Any, code: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        _fail("dependency_uid_hash_invalid")
    if _uid_hash(frame.observation_uid) != expected:
        _fail(code)


def _standard_frame(
    frame: Any, classes: tuple[str, ...], code: str
) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        _fail(f"{code}_frame_invalid")
    missing = [column for column in _STANDARD_COLUMNS if column not in frame.columns]
    if missing:
        _fail(f"{code}_frame_columns")
    uids = frame.observation_uid.astype(str)
    if uids.duplicated().any():
        _fail(f"{code}_frame_uid_duplicate")
    parsed_vocabulary = {
        tuple(json.loads(value)) for value in frame.class_vocabulary.astype(str).unique()
    }
    if parsed_vocabulary != {tuple(classes)}:
        _fail(f"{code}_class_order_mismatch")
    scores = np.asarray(
        [json.loads(value) for value in frame.scores.astype(str)], dtype=np.float64
    )
    if scores.shape != (len(frame), len(classes)) or not np.isfinite(scores).all():
        _fail(f"{code}_scores_invalid")
    labels = frame.true_label.astype(str)
    if not set(labels).issubset(set(classes)):
        _fail(f"{code}_true_label_outside_classes")
    return scores


def _metadata_index(
    source_metadata: Any, classes: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    if not isinstance(source_metadata, pd.DataFrame) or source_metadata.empty:
        _fail("source_metadata_invalid")
    required = {
        "observation_uid",
        "master_sample_id",
        "target_analyte",
        "instrument",
        "station",
    }
    if not required <= set(source_metadata.columns):
        _fail("source_metadata_columns")
    index: dict[str, dict[str, str]] = {}
    master_target: dict[str, str] = {}
    master_station: dict[str, str] = {}
    seen_targets: set[str] = set()
    seen_stations: set[str] = set()
    for row in source_metadata.itertuples(index=False):
        uid = str(row.observation_uid)
        master = str(row.master_sample_id)
        target = str(row.target_analyte)
        instrument = str(row.instrument)
        station = str(row.station)
        if uid in index:
            _fail("source_metadata_uid_duplicate")
        if target not in classes:
            _fail("source_metadata_label_outside_classes")
        if master_target.get(master, target) != target:
            _fail("source_metadata_master_target_conflict")
        if master_station.get(master, station) != station:
            _fail("source_metadata_master_station_conflict")
        master_target[master] = target
        master_station[master] = station
        seen_targets.add(target)
        seen_stations.add(station)
        index[uid] = {
            "master": master,
            "target": target,
            "instrument": instrument,
            "station": station,
        }
    if seen_targets != set(classes):
        _fail("source_metadata_class_vocabulary_incomplete")
    if len(seen_stations) != 1:
        _fail("source_metadata_station_invalid")
    return index


def _check_frame_source(
    frame: pd.DataFrame, index: Mapping[str, Mapping[str, str]], code: str
) -> None:
    for uid, master, label, instrument, station in zip(
        frame.observation_uid.astype(str),
        frame.master_sample_id.astype(str),
        frame.true_label.astype(str),
        frame.instrument.astype(str),
        frame.station.astype(str),
        strict=True,
    ):
        metadata = index.get(uid)
        if metadata is None:
            _fail(f"{code}_uid_not_in_source_metadata")
        if (
            metadata["master"] != master
            or metadata["target"] != label
            or metadata["instrument"] != instrument
            or metadata["station"] != station
        ):
            _fail(f"{code}_source_metadata_mismatch")


def _require_valid_temperature(calibration: Any) -> TemperatureCalibration:
    if not isinstance(calibration, TemperatureCalibration):
        _fail("calibration_type_invalid")
    if not calibration.optimizer_success:
        _fail("calibration_optimizer_failed")
    if not np.isfinite(calibration.temperature) or calibration.temperature <= 0:
        _fail("calibration_temperature_invalid")
    if not np.isfinite(calibration.optimizer_objective):
        _fail("calibration_objective_invalid")
    return calibration


def _score_sha256(scores: np.ndarray) -> str:
    return sha256_value([[float(value) for value in row] for row in scores])


def _require_seed_structure(
    cross: pd.DataFrame, expected_seeds: Sequence[Any]
) -> None:
    expected = set(expected_seeds)
    for _unit_id, group in cross.groupby("selection_unit_id", sort=True):
        seeds = set(group.seed.tolist())
        if seeds != expected:
            _fail("calibration_seed_structure_invalid")
        reference: pd.DataFrame | None = None
        for _seed, sub in group.groupby("seed", sort=True):
            ordered = sub.sort_values("observation_uid", kind="stable").reset_index(
                drop=True
            )
            key = ordered[list(_KEY_COLUMNS)]
            if reference is None:
                reference = key
            elif not reference.equals(key):
                _fail("calibration_unit_rows_mismatch")


def _require_uncalibrated(frame: pd.DataFrame, scope: str = "held") -> None:
    if not frame.probability_status.astype(str).eq("uncalibrated").all():
        _fail(f"{scope}_predictions_calibrated")
    if frame.probabilities.notna().any():
        _fail(f"{scope}_probabilities_present")


def _require_frame_seed(frame: pd.DataFrame, seed: Any) -> None:
    if "seed" in frame.columns:
        values = set(frame.seed.tolist())
        if values != {seed}:
            _fail("frame_seed_mismatch")


def _require_calibrated_probabilities(
    frame: pd.DataFrame, classes: tuple[str, ...]
) -> np.ndarray:
    if frame.probabilities.isna().any():
        _fail("held_probabilities_missing")
    matrix = np.asarray(
        [json.loads(value) for value in frame.probabilities.astype(str)],
        dtype=np.float64,
    )
    if matrix.shape != (len(frame), len(classes)) or not np.isfinite(matrix).all():
        _fail("held_probabilities_invalid")
    if (matrix < 0).any() or (matrix > 1).any():
        _fail("held_probabilities_out_of_range")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        _fail("held_probabilities_unnormalized")
    return matrix


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("observation_uid", kind="stable").reset_index(drop=True)


def _classical_scalar(
    job_record: Mapping[str, Any],
    dep_map: Mapping[str, Any],
    classes: tuple[str, ...],
    index: Mapping[str, Mapping[str, str]],
) -> tuple[TemperatureCalibration, dict[str, Any]]:
    expected_seeds = (
        tuple(p08_plan.SEEDS)
        if job_record["model_id"] in STOCHASTIC
        else (p08_plan.SVM_SEED,)
    )
    frames: list[pd.DataFrame] = []
    dependency_ids: list[str] = []
    dependency_stages: dict[str, str] = {}
    score_hashes: dict[str, str] = {}
    uid_hashes: dict[str, str] = {}
    for dep_id in sorted(dep_map):
        entry = _entry(dep_map[dep_id])
        dep_job = _graph_job(entry["job"])
        if dep_job["job_id"] != dep_id:
            _fail("dependency_key_job_mismatch")
        _require_dependency(
            job_record,
            dep_job,
            (CALIBRATION_VALIDATION_PREDICTION, CALIBRATION_PREDICTION_ALIAS),
        )
        if dep_job["stage"] == CALIBRATION_VALIDATION_PREDICTION:
            _require_resolution(dep_job, _SOURCE_SELECTION_DEPENDENT)
        else:
            _require_resolution(dep_job, _RESOLVE_SELECTED_CANDIDATE)
        frame = _entry_predictions(entry)
        scores = _standard_frame(frame, classes, "calibration")
        _require_uid_hash(
            frame, dep_job["validation_uid_sha256"], "calibration_uid_hash_mismatch"
        )
        _check_frame_source(frame, index, "calibration")
        _require_uncalibrated(frame, "calibration")
        _require_frame_seed(frame, dep_job["seed"])
        prepared = frame.copy()
        prepared["seed"] = dep_job["seed"]
        prepared["selection_unit_id"] = str(dep_job["unit_id"])
        frames.append(prepared)
        dependency_ids.append(dep_id)
        dependency_stages[dep_id] = dep_job["stage"]
        score_hashes[dep_id] = _score_sha256(scores)
        uid_hashes[dep_id] = _uid_hash(frame.observation_uid)
    if not frames:
        _fail("calibration_dependencies_empty")
    cross = pd.concat(frames, ignore_index=True)
    _require_seed_structure(cross, expected_seeds)
    result = p03_runtime.fit_cross_fitted_temperature(
        cross,
        model_id=job_record["model_id"],
        class_vocabulary=classes,
    )
    calibration = _require_valid_temperature(result.calibration)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "adapter": "scalar_calibration",
        "calibration_kind": "classical_cross_fitted",
        "policy_id": job_record["policy_id"],
        "context_id": job_record["context_id"],
        "model_id": job_record["model_id"],
        "stage": job_record["stage"],
        "job_id": job_record["job_id"],
        "representation_id": job_record["representation_id"],
        "array_sha256": job_record["array_sha256"],
        "model_spec_sha256": job_record["model_spec_sha256"],
        "dependency_job_ids": dependency_ids,
        "dependency_stages": dependency_stages,
        "dependency_scores_sha256": score_hashes,
        "dependency_validation_uid_sha256": uid_hashes,
        "source_metadata_observation_count": len(index),
        "classes": list(classes),
        "temperature": float(calibration.temperature),
        "optimizer_success": bool(calibration.optimizer_success),
        "optimizer_objective": float(calibration.optimizer_objective),
        "calibration_state_sha256": calibration.state_sha256,
        "calibration_observations": int(calibration.observations),
        "calibration_masters": int(calibration.masters),
        "selection_unit_count": int(result.selection_unit_count),
        "cross_fitted_evidence_fit_id_sha256": result.evidence_fit_id_sha256,
        "cross_fitted_observation_count": int(len(result.cross_fitted_predictions)),
        "execution_authorized": False,
    }
    return calibration, audit


def _neural_scalar(
    job_record: Mapping[str, Any],
    dep_map: Mapping[str, Any],
    classes: tuple[str, ...],
    index: Mapping[str, Mapping[str, str]],
) -> tuple[TemperatureCalibration, dict[str, Any]]:
    if job_record["seed"] == p08_plan.NOT_APPLICABLE:
        _fail("scalar_seed_missing")
    if job_record["seed"] not in p08_plan.SEEDS:
        _fail("scalar_seed_invalid")
    records: list[dict[str, Any]] = []
    units: list[str] = []
    observation_uids: list[str] = []
    dependency_ids: list[str] = []
    score_hashes: dict[str, str] = {}
    uid_hashes: dict[str, str] = {}
    for dep_id in sorted(dep_map):
        entry = _entry(dep_map[dep_id])
        dep_job = _graph_job(entry["job"])
        if dep_job["job_id"] != dep_id:
            _fail("dependency_key_job_mismatch")
        _require_dependency(job_record, dep_job, (SOURCE_VALIDATION_PREDICTION,))
        _require_resolution(dep_job, _FIXED_SPEC)
        if dep_job["seed"] != job_record["seed"]:
            _fail("dependency_seed_mismatch")
        frame = _entry_predictions(entry)
        scores = _standard_frame(frame, classes, "calibration")
        _require_uid_hash(
            frame, dep_job["validation_uid_sha256"], "calibration_uid_hash_mismatch"
        )
        _check_frame_source(frame, index, "calibration")
        _require_uncalibrated(frame, "calibration")
        _require_frame_seed(frame, dep_job["seed"])
        observation_uids.extend(frame.observation_uid.astype(str).tolist())
        units.append(str(dep_job["unit_id"]))
        dependency_ids.append(dep_id)
        score_hashes[dep_id] = _score_sha256(scores)
        uid_hashes[dep_id] = _uid_hash(frame.observation_uid)
        for position in range(len(frame)):
            records.append(
                {
                    "logit_0": float(scores[position, 0]),
                    "logit_1": float(scores[position, 1]),
                    "logit_2": float(scores[position, 2]),
                    "true_label": str(frame.true_label.iloc[position]),
                    "master_sample_id": str(frame.master_sample_id.iloc[position]),
                }
            )
    if len(set(units)) != len(units):
        _fail("calibration_unit_duplicate")
    if not records:
        _fail("calibration_rows_empty")
    ordered = pd.DataFrame.from_records(
        records,
        columns=["logit_0", "logit_1", "logit_2", "true_label", "master_sample_id"],
    )
    calibration = _require_valid_temperature(
        p04_runtime._master_equal_calibration(ordered, classes)
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "adapter": "scalar_calibration",
        "calibration_kind": "neural_master_equal",
        "policy_id": job_record["policy_id"],
        "context_id": job_record["context_id"],
        "model_id": job_record["model_id"],
        "stage": job_record["stage"],
        "job_id": job_record["job_id"],
        "seed": job_record["seed"],
        "representation_id": job_record["representation_id"],
        "array_sha256": job_record["array_sha256"],
        "model_spec_sha256": job_record["model_spec_sha256"],
        "dependency_job_ids": dependency_ids,
        "dependency_scores_sha256": score_hashes,
        "dependency_validation_uid_sha256": uid_hashes,
        "source_metadata_observation_count": len(index),
        "source_unit_count": len(set(units)),
        "classes": list(classes),
        "row_count": len(ordered),
        "unique_observation_uid_count": len(set(observation_uids)),
        "unique_master_sample_id_count": int(ordered.master_sample_id.nunique()),
        "temperature": float(calibration.temperature),
        "optimizer_success": bool(calibration.optimizer_success),
        "optimizer_objective": float(calibration.optimizer_objective),
        "calibration_state_sha256": calibration.state_sha256,
        "calibration_observations": int(calibration.observations),
        "calibration_masters": int(calibration.masters),
        "execution_authorized": False,
    }
    return calibration, audit


def _classical_ensemble(
    job_record: Mapping[str, Any],
    dep_map: Mapping[str, Any],
    classes: tuple[str, ...],
    calibration: TemperatureCalibration | None,
) -> pd.DataFrame:
    expected = (
        tuple(p08_plan.SEEDS)
        if job_record["model_id"] in STOCHASTIC
        else (p08_plan.SVM_SEED,)
    )
    held: list[tuple[Any, pd.DataFrame]] = []
    scalar_entry: Mapping[str, Any] | None = None
    for dep_id in sorted(dep_map):
        entry = _entry(dep_map[dep_id])
        dep_job = _graph_job(entry["job"])
        if dep_job["job_id"] != dep_id:
            _fail("dependency_key_job_mismatch")
        _require_dependency(job_record, dep_job, (HELD_PREDICTION, SCALAR_CALIBRATION))
        if dep_job["stage"] == HELD_PREDICTION:
            _require_resolution(dep_job, _RESOLUTION_UNCALIBRATED)
            frame = _entry_predictions(entry)
            _standard_frame(frame, classes, "held")
            _require_uncalibrated(frame)
            _require_uid_hash(frame, dep_job["test_uid_sha256"], "held_uid_hash_mismatch")
            _require_frame_seed(frame, dep_job["seed"])
            held.append((dep_job["seed"], frame))
        else:
            if scalar_entry is not None:
                _fail("ensemble_scalar_duplicate")
            _require_resolution(dep_job, _RESOLUTION_CALIBRATE_SEED_AVERAGED)
            scalar_entry = entry
    if scalar_entry is None:
        _fail("ensemble_scalar_missing")
    seeds = [seed for seed, _ in held]
    if len(seeds) != len(expected) or set(seeds) != set(expected):
        _fail("ensemble_seed_structure_invalid")
    dependency_calibration = _require_valid_temperature(
        _entry_calibration(scalar_entry)
    )
    if tuple(dependency_calibration.class_vocabulary) != tuple(classes):
        _fail("calibration_class_vocabulary_mismatch")
    if calibration is not None and calibration != dependency_calibration:
        _fail("calibration_object_mismatch")
    order = {seed: position for position, seed in enumerate(expected)}
    ordered = [
        frame for _, frame in sorted(held, key=lambda item: order[item[0]])
    ]
    return p03_runtime.aggregate_seed_prediction_frames(
        ordered,
        model_id=job_record["model_id"],
        aggregate_fit_id=job_record["job_id"],
        class_vocabulary=classes,
        calibration=dependency_calibration,
    )


def _neural_ensemble(
    job_record: Mapping[str, Any],
    dep_map: Mapping[str, Any],
    classes: tuple[str, ...],
) -> pd.DataFrame:
    expected = tuple(p08_plan.SEEDS)
    collected: list[tuple[Any, pd.DataFrame]] = []
    for dep_id in sorted(dep_map):
        entry = _entry(dep_map[dep_id])
        dep_job = _graph_job(entry["job"])
        if dep_job["job_id"] != dep_id:
            _fail("dependency_key_job_mismatch")
        _require_dependency(job_record, dep_job, (HELD_PREDICTION,))
        _require_resolution(dep_job, _SOURCE_EPOCH_DEPENDENT)
        frame = _entry_predictions(entry)
        _standard_frame(frame, classes, "held")
        _require_uid_hash(frame, dep_job["test_uid_sha256"], "held_uid_hash_mismatch")
        _require_frame_seed(frame, dep_job["seed"])
        statuses = frame.probability_status.astype(str)
        if statuses.eq("uncalibrated").any():
            _fail("held_probabilities_uncalibrated")
        if not statuses.eq("cross_fitted_temperature").all():
            _fail("held_probability_status_invalid")
        collected.append((dep_job["seed"], frame))
    seeds = [seed for seed, _ in collected]
    if len(seeds) != len(expected) or set(seeds) != set(expected):
        _fail("ensemble_seed_structure_invalid")
    order = {seed: position for position, seed in enumerate(expected)}
    collected.sort(key=lambda item: order[item[0]])
    ordered = [(seed, _sorted_frame(frame)) for seed, frame in collected]
    reference = ordered[0][1]
    reference_key = reference[list(_KEY_COLUMNS)]
    for _seed, frame in ordered[1:]:
        if not reference_key.equals(frame[list(_KEY_COLUMNS)]):
            _fail("ensemble_metadata_mismatch")
    probability_stack: list[np.ndarray] = []
    probability_hashes: dict[str, str] = {}
    for seed, frame in ordered:
        probabilities = _require_calibrated_probabilities(frame, classes)
        probability_stack.append(probabilities)
        probability_hashes[str(seed)] = sha256_value(
            [[float(value) for value in row] for row in probabilities]
        )
    mean = np.stack(probability_stack).mean(axis=0, dtype=np.float64)
    scores = np.log(np.clip(mean, *_PROBABILITY_CLIP))
    metadata = reference.rename(columns={"true_label": "target_analyte"})
    result = p03_runtime._prediction_frame(
        metadata=metadata,
        scores=scores,
        class_vocabulary=classes,
        fit_id=job_record["job_id"],
        calibrated_probabilities=mean,
    )
    result["probability_status"] = "seedwise_temperature_ensemble"
    result["technical_seed_count"] = len(ordered)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": "seedwise_temperature_ensemble",
        "job_id": job_record["job_id"],
        "model_id": job_record["model_id"],
        "policy_id": job_record["policy_id"],
        "context_id": job_record["context_id"],
        "dependency_job_ids": sorted(dep_map),
        "seeds": [seed for seed, _ in ordered],
        "class_vocabulary": list(classes),
        "held_probability_sha256": probability_hashes,
        "mean_probability_sha256": sha256_value(
            [[float(value) for value in row] for row in mean]
        ),
    }
    result["evidence_sha256"] = sha256_value(evidence)
    return result


def calibrate(
    *,
    job: Any,
    dependencies: Any,
    classes: Any,
    source_metadata: Any,
) -> tuple[TemperatureCalibration, dict[str, Any]]:
    """Fit one frozen scalar temperature for a universal calibration slot.

    Every value is derived from explicit, previously authenticated dependency
    artifacts.  Nothing is invented, defaulted or silently dropped.
    """

    job_record = _graph_job(job)
    _require_job_identity(job_record)
    if job_record["stage"] != SCALAR_CALIBRATION:
        _fail("job_stage_invalid")
    vocabulary = _as_classes(classes)
    dep_map = _dependency_mapping(dependencies)
    _require_dependency_set(job_record, dep_map)
    index = _metadata_index(source_metadata, vocabulary)
    model_id = job_record["model_id"]
    if model_id in CLASSICAL_MODELS:
        _require_resolution(job_record, _RESOLUTION_CALIBRATE_SEED_AVERAGED)
        return _classical_scalar(job_record, dep_map, vocabulary, index)
    if model_id in NEURAL_RECIPES:
        _require_resolution(job_record, _SOURCE_EPOCH_DEPENDENT)
        return _neural_scalar(job_record, dep_map, vocabulary, index)
    _fail("job_model_invalid")


def ensemble(
    *,
    job: Any,
    dependencies: Any,
    classes: Any,
    calibration: TemperatureCalibration | None = None,
) -> pd.DataFrame:
    """Aggregate declared technical seeds for a universal ensemble slot.

    Classical forests average uncalibrated probabilities before the single
    temperature; neural recipes average already calibrated seed
    probabilities with no second temperature.
    """

    job_record = _graph_job(job)
    _require_job_identity(job_record)
    if job_record["stage"] != SEED_ENSEMBLE_PREDICTION:
        _fail("job_stage_invalid")
    vocabulary = _as_classes(classes)
    dep_map = _dependency_mapping(dependencies)
    _require_dependency_set(job_record, dep_map)
    model_id = job_record["model_id"]
    if model_id in CLASSICAL_MODELS:
        _require_resolution(job_record, _RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE)
        return _classical_ensemble(job_record, dep_map, vocabulary, calibration)
    if model_id in NEURAL_RECIPES:
        _require_resolution(job_record, _SOURCE_EPOCH_DEPENDENT)
        if calibration is not None:
            _fail("neural_ensemble_calibration_forbidden")
        return _neural_ensemble(job_record, dep_map, vocabulary)
    _fail("job_model_invalid")
