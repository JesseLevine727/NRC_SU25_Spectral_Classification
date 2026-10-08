"""Unit tests for the P08-U1 calibration and ensemble adapter.

All jobs, scores, probabilities and calibrations are invented in-memory
values.  No experiment is executed, no artifact is read and no authority is
granted.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p03_runtime, p04_runtime, p08_plan
from atlas_sers.evaluation.classical import TemperatureCalibration, apply_temperature, softmax
from atlas_sers.evaluation.p08_u1_calibration import (
    CalibrationAdapterError,
    calibrate,
    ensemble,
)
from atlas_sers.governance.canonical import sha256_value

CLASSES = ("A", "B", "C")
POLICY = "PP-U-SG"
HASH64 = "a" * 64


def _uids_hash(uids) -> str:
    return sha256_value(sorted(str(value) for value in uids))


def _fields(
    model_id,
    stage,
    unit_id,
    seed,
    resolution,
    *,
    policy=POLICY,
    validation_uid_sha256=HASH64,
    test_uid_sha256=HASH64,
    fit_uid_sha256=HASH64,
):
    return {
        "policy_id": policy,
        "representation_id": p08_plan.POLICY_REPRESENTATION[policy],
        "array_sha256": HASH64,
        "context_id": "CTX-1",
        "model_id": model_id,
        "model_spec_sha256": HASH64,
        "stage": stage,
        "unit_id": unit_id,
        "seed": seed,
        "candidate_id": "candidate-x",
        "hyperparameter_sha256": HASH64,
        "fit_uid_sha256": fit_uid_sha256,
        "validation_uid_sha256": validation_uid_sha256,
        "test_uid_sha256": test_uid_sha256,
        "resolution": resolution,
        "evidence_status": p08_plan.EVIDENCE_FUTURE,
    }


def _job(**fields):
    dependencies = fields.pop("dependencies", ())
    return p08_plan._new_job(_fields(**fields), list(dependencies))


def _with_dependencies(job, dependencies):
    payload = {key: value for key, value in job.items() if key != "job_id"}
    payload["dependencies"] = list(dependencies)
    return {"job_id": "P08JOB-" + p08_plan._hash(payload), **payload}


def _metadata(unit):
    return pd.DataFrame(
        [
            {
                "observation_uid": f"{unit}-uid-0",
                "master_sample_id": f"{unit}-master-0",
                "instrument": f"{unit}-inst-0",
                "station": "ST-1",
                "target_analyte": "A",
            },
            {
                "observation_uid": f"{unit}-uid-1",
                "master_sample_id": f"{unit}-master-1",
                "instrument": f"{unit}-inst-1",
                "station": "ST-1",
                "target_analyte": "B",
            },
            {
                "observation_uid": f"{unit}-uid-2",
                "master_sample_id": f"{unit}-master-2",
                "instrument": f"{unit}-inst-0",
                "station": "ST-1",
                "target_analyte": "C",
            },
        ]
    )


def _frame(unit, scores, fit_id):
    return p03_runtime._prediction_frame(
        metadata=_metadata(unit),
        scores=np.asarray(scores, dtype=float),
        class_vocabulary=CLASSES,
        fit_id=fit_id,
    )


def _single_frame(uid, master, label, score_row, fit_id):
    metadata = pd.DataFrame(
        [
            {
                "observation_uid": uid,
                "master_sample_id": master,
                "instrument": "I0",
                "station": "ST-1",
                "target_analyte": label,
            }
        ]
    )
    return p03_runtime._prediction_frame(
        metadata=metadata,
        scores=np.asarray([score_row], dtype=float),
        class_vocabulary=CLASSES,
        fit_id=fit_id,
    )


def _source_metadata(frames):
    rows = []
    for frame in frames:
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "observation_uid": str(row.observation_uid),
                    "master_sample_id": str(row.master_sample_id),
                    "target_analyte": str(row.true_label),
                    "instrument": str(row.instrument),
                    "station": str(row.station),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        subset="observation_uid", keep="first"
    ).reset_index(drop=True)


def _dependency(job, frame=None, calibration=None):
    entry = {"job": job}
    if frame is not None:
        entry["predictions"] = frame
    if calibration is not None:
        entry["calibration"] = calibration
    return entry


def _temperature(value):
    return TemperatureCalibration(
        temperature=value,
        class_vocabulary=CLASSES,
        observations=9,
        masters=9,
        fit_observation_uid_sha256=HASH64,
        fit_master_uid_sha256=HASH64,
        optimizer_success=True,
        optimizer_objective=0.5,
    )


def _scores_of(frame):
    return np.asarray(
        [json.loads(value) for value in frame.scores.astype(str)], dtype=float
    )


def _probs_of(frame):
    return np.asarray(
        [json.loads(value) for value in frame.probabilities.astype(str)], dtype=float
    )


def _classical_calibration_case(
    model_id,
    seeds,
    *,
    dep_stage="calibration_validation_prediction",
    dep_resolution=p08_plan.SOURCE_SELECTION_DEPENDENT,
):
    units = ("U0", "U1")
    dep_jobs = []
    frames = []
    for unit_index, unit in enumerate(units):
        unit_rng = np.random.default_rng(1000 + unit_index)
        for seed_index, seed in enumerate(seeds):
            scores = unit_rng.normal(size=(3, 3))
            frame = _frame(unit, scores, f"FIT-{unit}-{seed_index}")
            job = _job(
                model_id=model_id,
                stage=dep_stage,
                unit_id=unit,
                seed=seed,
                resolution=dep_resolution,
                validation_uid_sha256=_uids_hash(frame.observation_uid),
            )
            dep_jobs.append(job)
            frames.append(frame)
    scalar = _job(
        model_id=model_id,
        stage=p08_plan.SCALAR_STAGE,
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.RESOLUTION_CALIBRATE_SEED_AVERAGED,
        dependencies=[job["job_id"] for job in dep_jobs],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(dep_jobs, frames, strict=True)
    }
    return scalar, dependencies, dep_jobs, frames


def _cross_frame(jobs, frames):
    prepared = []
    for job, frame in zip(jobs, frames, strict=True):
        copy_frame = frame.copy()
        copy_frame["seed"] = job["seed"]
        copy_frame["selection_unit_id"] = str(job["unit_id"])
        prepared.append(copy_frame)
    return pd.concat(prepared, ignore_index=True)


def _neural_calibration_case(recipe="D0-M", seed=p08_plan.SEEDS[0]):
    dep_jobs = []
    frames = []
    for unit_index, unit in enumerate(("U0", "U1")):
        unit_rng = np.random.default_rng(2000 + unit_index)
        frame = _frame(unit, unit_rng.normal(size=(3, 3)), f"FIT-N-{unit}")
        job = _job(
            model_id=recipe,
            stage="source_validation_prediction",
            unit_id=unit,
            seed=seed,
            resolution=p08_plan.FIXED_SPEC,
            validation_uid_sha256=_uids_hash(frame.observation_uid),
        )
        dep_jobs.append(job)
        frames.append(frame)
    scalar = _job(
        model_id=recipe,
        stage=p08_plan.SCALAR_STAGE,
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=seed,
        resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
        dependencies=[job["job_id"] for job in dep_jobs],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(dep_jobs, frames, strict=True)
    }
    return scalar, dependencies, frames


@pytest.mark.parametrize("neural", [False, True])
def test_calibration_rejects_dependency_bound_to_another_test_set(neural):
    if neural:
        scalar, dependencies, frames = _neural_calibration_case()
    else:
        scalar, dependencies, _jobs, frames = _classical_calibration_case(
            "C-RBF-SVM", [p08_plan.SVM_SEED]
        )
    old_id = next(iter(dependencies))
    entry = dependencies.pop(old_id)
    changed = dict(entry["job"], test_uid_sha256="b" * 64)
    changed = _with_dependencies(changed, changed["dependencies"])
    dependencies[changed["job_id"]] = dict(entry, job=changed)
    scalar = _with_dependencies(scalar, sorted(dependencies))
    with pytest.raises(CalibrationAdapterError, match="dependency_test_uid_mismatch"):
        calibrate(
            job=scalar,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_classical_calibrate_parity_trees():
    model_id = "C-RANDOM-FOREST"
    scalar, dependencies, dep_jobs, frames = _classical_calibration_case(
        model_id, p08_plan.SEEDS
    )
    metadata = _source_metadata(frames)
    calibration, audit = calibrate(
        job=scalar,
        dependencies=dependencies,
        classes=CLASSES,
        source_metadata=metadata,
    )
    expected = p03_runtime.fit_cross_fitted_temperature(
        _cross_frame(dep_jobs, frames),
        model_id=model_id,
        class_vocabulary=CLASSES,
    )
    assert calibration.temperature == expected.calibration.temperature
    assert calibration.state_sha256 == expected.calibration.state_sha256
    assert audit["selection_unit_count"] == 2
    assert audit["calibration_kind"] == "classical_cross_fitted"
    json.dumps(audit)


def test_classical_calibrate_svm_single_seed():
    model_id = "C-RBF-SVM"
    scalar, dependencies, dep_jobs, frames = _classical_calibration_case(
        model_id, (p08_plan.SVM_SEED,)
    )
    calibration, audit = calibrate(
        job=scalar,
        dependencies=dependencies,
        classes=CLASSES,
        source_metadata=_source_metadata(frames),
    )
    expected = p03_runtime.fit_cross_fitted_temperature(
        _cross_frame(dep_jobs, frames),
        model_id=model_id,
        class_vocabulary=CLASSES,
    )
    assert calibration.temperature == expected.calibration.temperature
    assert audit["dependency_stages"]


def test_neural_calibrate_parity():
    recipe = "D0-M"
    scalar, dependencies, frames = _neural_calibration_case(recipe)
    calibration, audit = calibrate(
        job=scalar,
        dependencies=dependencies,
        classes=CLASSES,
        source_metadata=_source_metadata(frames),
    )
    records = []
    for frame in frames:
        scores = _scores_of(frame)
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
    ordered = pd.DataFrame.from_records(
        records,
        columns=["logit_0", "logit_1", "logit_2", "true_label", "master_sample_id"],
    )
    expected = p04_runtime._master_equal_calibration(ordered, CLASSES)
    assert calibration.temperature == pytest.approx(expected.temperature)
    assert audit["calibration_kind"] == "neural_master_equal"
    json.dumps(audit)


def _classical_ensemble_case(model_id, seeds):
    held_jobs = []
    held_frames = []
    for seed_index, seed in enumerate(seeds):
        rng = np.random.default_rng(3000 + seed_index)
        frame = _frame("U0", rng.normal(size=(3, 3)) * 1.5, f"HELD-{seed_index}")
        job = _job(
            model_id=model_id,
            stage="held_prediction",
            unit_id=p08_plan.NOT_APPLICABLE,
            seed=seed,
            resolution=p08_plan.RESOLUTION_UNCALIBRATED,
            test_uid_sha256=_uids_hash(frame.observation_uid),
        )
        held_jobs.append(job)
        held_frames.append(frame)
    calibration = _temperature(0.35)
    scalar_job = _job(
        model_id=model_id,
        stage=p08_plan.SCALAR_STAGE,
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.RESOLUTION_CALIBRATE_SEED_AVERAGED,
    )
    ensemble_job = _job(
        model_id=model_id,
        stage="seed_ensemble_prediction",
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE,
        test_uid_sha256=_uids_hash(held_frames[0].observation_uid),
        dependencies=[job["job_id"] for job in held_jobs] + [scalar_job["job_id"]],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(held_jobs, held_frames, strict=True)
    }
    dependencies[scalar_job["job_id"]] = _dependency(
        scalar_job, calibration=calibration
    )
    return ensemble_job, dependencies, held_jobs, held_frames, calibration


def test_classical_ensemble_matches_frozen_helper():
    job, dependencies, held_jobs, held_frames, calibration = _classical_ensemble_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    result = ensemble(
        job=job, dependencies=dependencies, classes=CLASSES, calibration=calibration
    )
    ordered = [
        frame
        for _, frame in sorted(
            zip(p08_plan.SEEDS, held_frames, strict=True), key=lambda item: item[0]
        )
    ]
    expected = p03_runtime.aggregate_seed_prediction_frames(
        ordered,
        model_id="C-RANDOM-FOREST",
        aggregate_fit_id=job["job_id"],
        class_vocabulary=CLASSES,
        calibration=calibration,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_classical_ensemble_order_differs_from_seedwise_calibration():
    score_rows = [
        [5.0, 0.0, 0.0],
        [0.0, 0.5, 4.0],
        [1.0, 3.0, 0.0],
    ]
    held_jobs = []
    held_frames = []
    for seed_index, seed in enumerate(p08_plan.SEEDS):
        frame = _single_frame(
            "OBS-1", "MASTER-1", "A", score_rows[seed_index], f"HELD-{seed_index}"
        )
        job = _job(
            model_id="C-RANDOM-FOREST",
            stage="held_prediction",
            unit_id=p08_plan.NOT_APPLICABLE,
            seed=seed,
            resolution=p08_plan.RESOLUTION_UNCALIBRATED,
            test_uid_sha256=_uids_hash(frame.observation_uid),
        )
        held_jobs.append(job)
        held_frames.append(frame)
    calibration = _temperature(0.3)
    scalar_job = _job(
        model_id="C-RANDOM-FOREST",
        stage=p08_plan.SCALAR_STAGE,
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.RESOLUTION_CALIBRATE_SEED_AVERAGED,
    )
    ensemble_job = _job(
        model_id="C-RANDOM-FOREST",
        stage="seed_ensemble_prediction",
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.RESOLUTION_SEED_AVERAGE_SINGLE_TEMPERATURE,
        test_uid_sha256=_uids_hash(held_frames[0].observation_uid),
        dependencies=[job["job_id"] for job in held_jobs] + [scalar_job["job_id"]],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(held_jobs, held_frames, strict=True)
    }
    dependencies[scalar_job["job_id"]] = _dependency(scalar_job, calibration=calibration)
    result = ensemble(job=ensemble_job, dependencies=dependencies, classes=CLASSES)
    alternative = np.mean(
        [
            apply_temperature(np.asarray([row], dtype=float), calibration)
            for row in score_rows
        ],
        axis=0,
    )
    got = _probs_of(result)
    assert np.abs(got - alternative).max() > 1e-4


def test_neural_ensemble_averages_calibrated_probabilities():
    held_jobs = []
    held_frames = []
    for seed_index, seed in enumerate(p08_plan.SEEDS):
        scores = np.asarray([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]) * (
            1.0 + 0.1 * seed_index
        )
        probabilities = softmax(scores, temperature=0.5 * (seed_index + 1))
        frame = p03_runtime._prediction_frame(
            metadata=_metadata("U0"),
            scores=scores,
            class_vocabulary=CLASSES,
            fit_id=f"HELD-N-{seed_index}",
            calibrated_probabilities=probabilities,
        )
        job = _job(
            model_id="D1",
            stage="held_prediction",
            unit_id=p08_plan.NOT_APPLICABLE,
            seed=seed,
            resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
            test_uid_sha256=_uids_hash(frame.observation_uid),
        )
        held_jobs.append(job)
        held_frames.append(frame)
    ensemble_job = _job(
        model_id="D1",
        stage="seed_ensemble_prediction",
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
        test_uid_sha256=_uids_hash(held_frames[0].observation_uid),
        dependencies=[job["job_id"] for job in held_jobs],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(held_jobs, held_frames, strict=True)
    }
    result = ensemble(job=ensemble_job, dependencies=dependencies, classes=CLASSES)
    expected = np.mean([_probs_of(frame) for frame in held_frames], axis=0)
    got = _probs_of(result)
    assert np.allclose(got, expected, atol=1e-12)
    assert result.probability_status.eq("seedwise_temperature_ensemble").all()
    assert result.technical_seed_count.eq(3).all()
    assert result.evidence_sha256.nunique() == 1


def test_job_hash_tamper_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    tampered = dict(scalar)
    tampered["policy_id"] = "PP-U-ARPLS"
    with pytest.raises(CalibrationAdapterError, match="job_hash_invalid"):
        calibrate(
            job=tampered,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_non_universal_policy_is_rejected():
    job = _job(
        policy="PP-U-MIN",
        model_id="C-RBF-SVM",
        stage=p08_plan.SCALAR_STAGE,
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.RESOLUTION_CALIBRATE_SEED_AVERAGED,
        dependencies=[],
    )
    with pytest.raises(CalibrationAdapterError, match="job_policy_invalid"):
        calibrate(job=job, dependencies={}, classes=CLASSES, source_metadata=pd.DataFrame() )


def test_calibration_rejects_held_dependency_stage():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST",
        p08_plan.SEEDS,
        dep_stage="held_prediction",
        dep_resolution=p08_plan.RESOLUTION_UNCALIBRATED,
    )
    with pytest.raises(CalibrationAdapterError, match="dependency_stage_invalid"):
        calibrate(
            job=scalar,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_uid_hash_tamper_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    tampered = {
        dep_id: dict(entry) for dep_id, entry in dependencies.items()
    }
    first_id = sorted(tampered)[0]
    frame = tampered[first_id]["predictions"].copy()
    frame.loc[0, "observation_uid"] = "TAMPERED-UID"
    tampered[first_id]["predictions"] = frame
    with pytest.raises(CalibrationAdapterError, match="calibration_uid_hash_mismatch"):
        calibrate(
            job=scalar,
            dependencies=tampered,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_source_metadata_mismatch_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    metadata = _source_metadata(frames).iloc[1:].reset_index(drop=True)
    with pytest.raises(
        CalibrationAdapterError, match="calibration_uid_not_in_source_metadata"
    ):
        calibrate(
            job=scalar,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=metadata,
        )


def test_class_order_mismatch_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    tampered = {dep_id: dict(entry) for dep_id, entry in dependencies.items()}
    first_id = sorted(tampered)[0]
    frame = tampered[first_id]["predictions"].copy()
    frame["class_vocabulary"] = json.dumps(list(reversed(CLASSES)))
    tampered[first_id]["predictions"] = frame
    with pytest.raises(CalibrationAdapterError, match="calibration_class_order_mismatch"):
        calibrate(
            job=scalar,
            dependencies=tampered,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_missing_tree_seed_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS[:2]
    )
    with pytest.raises(
        CalibrationAdapterError, match="calibration_seed_structure_invalid"
    ):
        calibrate(
            job=scalar,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_dependency_set_mismatch_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    trimmed = dict(dependencies)
    trimmed.pop(sorted(trimmed)[0])
    with pytest.raises(CalibrationAdapterError, match="dependency_set_mismatch"):
        calibrate(
            job=scalar,
            dependencies=trimmed,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_invalid_temperature_is_rejected_by_ensemble():
    job, dependencies, _held_jobs, _frames, _calibration = _classical_ensemble_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    invalid = TemperatureCalibration(
        temperature=1.0,
        class_vocabulary=CLASSES,
        observations=9,
        masters=9,
        fit_observation_uid_sha256=HASH64,
        fit_master_uid_sha256=HASH64,
        optimizer_success=False,
        optimizer_objective=0.5,
    )
    scalar_id = sorted(
        dep_id
        for dep_id, entry in dependencies.items()
        if "calibration" in entry
    )[0]
    dependencies = dict(dependencies)
    dependencies[scalar_id] = _dependency(dependencies[scalar_id]["job"], calibration=invalid)
    with pytest.raises(CalibrationAdapterError, match="calibration_optimizer_failed"):
        ensemble(job=job, dependencies=dependencies, classes=CLASSES, calibration=invalid)


def test_neural_ensemble_requires_calibrated_probabilities():
    held_jobs = []
    held_frames = []
    for seed_index, seed in enumerate(p08_plan.SEEDS):
        frame = _frame("U0", np.eye(3), f"HELD-MISSING-{seed_index}")
        job = _job(
            model_id="D0-M",
            stage="held_prediction",
            unit_id=p08_plan.NOT_APPLICABLE,
            seed=seed,
            resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
            test_uid_sha256=_uids_hash(frame.observation_uid),
        )
        held_jobs.append(job)
        held_frames.append(frame)
    ensemble_job = _job(
        model_id="D0-M",
        stage="seed_ensemble_prediction",
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
        test_uid_sha256=_uids_hash(held_frames[0].observation_uid),
        dependencies=[job["job_id"] for job in held_jobs],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(held_jobs, held_frames, strict=True)
    }
    with pytest.raises(
        CalibrationAdapterError, match="held_probabilities_uncalibrated"
    ):
        ensemble(job=ensemble_job, dependencies=dependencies, classes=CLASSES)


def test_held_resolution_guard():
    job, dependencies, _held_jobs, _frames, calibration = _classical_ensemble_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    held_id = sorted(
        dep_id for dep_id, entry in dependencies.items() if "predictions" in entry
    )[0]
    tampered_entry = dict(dependencies[held_id])
    held_job = dict(tampered_entry["job"])
    held_job["resolution"] = p08_plan.SOURCE_EPOCH_DEPENDENT
    held_job = p08_plan._new_job(
        {key: value for key, value in held_job.items() if key != "job_id"},
        held_job["dependencies"],
    )
    tampered = dict(dependencies)
    tampered.pop(held_id)
    tampered[held_job["job_id"]] = _dependency(
        held_job, tampered_entry.get("predictions")
    )
    # Rebuild the ensemble job so the dependency set matches the tampered key.
    ensemble_job = dict(job)
    ensemble_job["dependencies"] = sorted(tampered)
    ensemble_job = p08_plan._new_job(
        {key: value for key, value in ensemble_job.items() if key != "job_id"},
        ensemble_job["dependencies"],
    )
    with pytest.raises(CalibrationAdapterError, match="job_resolution_invalid"):
        ensemble(job=ensemble_job, dependencies=tampered, classes=CLASSES)


def test_inputs_are_not_mutated():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    job_snapshot = copy.deepcopy(scalar)
    dependency_snapshot = copy.deepcopy(dependencies)
    frame_snapshots = [copy.deepcopy(frame) for frame in frames]
    calibrate(
        job=scalar,
        dependencies=dependencies,
        classes=CLASSES,
        source_metadata=_source_metadata(frames),
    )
    assert scalar == job_snapshot
    for frame, snapshot in zip(frames, frame_snapshots, strict=True):
        pd.testing.assert_frame_equal(frame, snapshot)
    for dep_id, entry in dependency_snapshot.items():
        if "predictions" in entry:
            pd.testing.assert_frame_equal(
                dependencies[dep_id]["predictions"], entry["predictions"]
            )


def test_frame_station_mismatch_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    tampered = {dep_id: dict(entry) for dep_id, entry in dependencies.items()}
    first_id = sorted(tampered)[0]
    frame = tampered[first_id]["predictions"].copy()
    frame["station"] = "ST-OTHER"
    tampered[first_id]["predictions"] = frame
    with pytest.raises(
        CalibrationAdapterError, match="calibration_source_metadata_mismatch"
    ):
        calibrate(
            job=scalar,
            dependencies=tampered,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_source_metadata_requires_single_station():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    metadata = _source_metadata(frames)
    metadata.loc[0, "station"] = "ST-2"
    with pytest.raises(
        CalibrationAdapterError, match="source_metadata_station_invalid"
    ):
        calibrate(
            job=scalar,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=metadata,
        )


def test_calibration_rejects_contradictory_frame_seed():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    tampered = {dep_id: dict(entry) for dep_id, entry in dependencies.items()}
    first_id = sorted(tampered)[0]
    frame = tampered[first_id]["predictions"].copy()
    frame["seed"] = 987654
    tampered[first_id]["predictions"] = frame
    with pytest.raises(CalibrationAdapterError, match="frame_seed_mismatch"):
        calibrate(
            job=scalar,
            dependencies=tampered,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_neural_scalar_seed_must_be_declared_seed():
    scalar, dependencies, frames = _neural_calibration_case(seed="SEED-X")
    with pytest.raises(CalibrationAdapterError, match="scalar_seed_invalid"):
        calibrate(
            job=scalar,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_neural_ensemble_rejects_unknown_probability_status():
    held_jobs = []
    held_frames = []
    for seed_index, seed in enumerate(p08_plan.SEEDS):
        scores = np.asarray(
            [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
        ) * (1.0 + 0.1 * seed_index)
        probabilities = softmax(scores, temperature=0.5 * (seed_index + 1))
        frame = p03_runtime._prediction_frame(
            metadata=_metadata("U0"),
            scores=scores,
            class_vocabulary=CLASSES,
            fit_id=f"HELD-S-{seed_index}",
            calibrated_probabilities=probabilities,
        )
        frame["probability_status"] = "calibrated"
        job = _job(
            model_id="D1",
            stage="held_prediction",
            unit_id=p08_plan.NOT_APPLICABLE,
            seed=seed,
            resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
            test_uid_sha256=_uids_hash(frame.observation_uid),
        )
        held_jobs.append(job)
        held_frames.append(frame)
    ensemble_job = _job(
        model_id="D1",
        stage="seed_ensemble_prediction",
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=p08_plan.NOT_APPLICABLE,
        resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
        test_uid_sha256=_uids_hash(held_frames[0].observation_uid),
        dependencies=[job["job_id"] for job in held_jobs],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(held_jobs, held_frames, strict=True)
    }
    with pytest.raises(
        CalibrationAdapterError, match="held_probability_status_invalid"
    ):
        ensemble(job=ensemble_job, dependencies=dependencies, classes=CLASSES)


def test_duplicate_dependency_declaration_is_rejected():
    scalar, dependencies, _jobs, frames = _classical_calibration_case(
        "C-RANDOM-FOREST", p08_plan.SEEDS
    )
    declared = list(scalar["dependencies"])
    job = _with_dependencies(scalar, declared + [declared[0]])
    with pytest.raises(CalibrationAdapterError, match="job_dependencies_duplicate"):
        calibrate(
            job=job,
            dependencies=dependencies,
            classes=CLASSES,
            source_metadata=_source_metadata(frames),
        )


def test_neural_audit_counts_unique_observation_uids():
    recipe = "D0-M"
    seed = p08_plan.SEEDS[0]
    dep_jobs = []
    frames = []
    for unit_index, unit in enumerate(("U0", "U1")):
        unit_rng = np.random.default_rng(4000 + unit_index)
        frame = p03_runtime._prediction_frame(
            metadata=_metadata("U0").copy(),
            scores=unit_rng.normal(size=(3, 3)),
            class_vocabulary=CLASSES,
            fit_id=f"FIT-OVERLAP-{unit}",
        )
        job = _job(
            model_id=recipe,
            stage="source_validation_prediction",
            unit_id=unit,
            seed=seed,
            resolution=p08_plan.FIXED_SPEC,
            validation_uid_sha256=_uids_hash(frame.observation_uid),
        )
        dep_jobs.append(job)
        frames.append(frame)
    scalar = _job(
        model_id=recipe,
        stage=p08_plan.SCALAR_STAGE,
        unit_id=p08_plan.NOT_APPLICABLE,
        seed=seed,
        resolution=p08_plan.SOURCE_EPOCH_DEPENDENT,
        dependencies=[job["job_id"] for job in dep_jobs],
    )
    dependencies = {
        job["job_id"]: _dependency(job, frame)
        for job, frame in zip(dep_jobs, frames, strict=True)
    }
    _calibration, audit = calibrate(
        job=scalar,
        dependencies=dependencies,
        classes=CLASSES,
        source_metadata=_source_metadata(frames),
    )
    assert audit["row_count"] == 6
    assert audit["unique_observation_uid_count"] == 3
