"""Compose unchanged calibration-parity and held-inference kernels; no fitting or IO."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import torch

from atlas_sers.evaluation import p03_runtime, p04_runtime, p05_development, p08_plan
from atlas_sers.evaluation.classical import TemperatureCalibration, apply_temperature
from atlas_sers.evaluation.p05_smoke import RECIPE_SPECIFICATIONS
from atlas_sers.evaluation.p08_u0_stage_backend import _metrics_equal
from atlas_sers.evaluation.p08_u1_calibration import _graph_job, _require_job_identity
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.acquisition import AcquisitionClassifier


class PredictionError(ValueError):
    """Graph, artifact or numerical evidence disagrees with the requested operation."""


def _require(condition, reason):
    if not condition:
        raise PredictionError(reason)


def _job(record, stage):
    record = _graph_job(record)
    _require_job_identity(record)
    _require(record["stage"] == stage, "stage_mismatch")
    expected = (p08_plan.SVM_SEED,) if record["model_id"] == p08_plan.SVM_MODEL else p08_plan.SEEDS
    _require(record["seed"] in expected, "seed_mismatch")
    return record


def _parent(child, parent):
    for key in p08_plan.JOB_FIELDS:
        if key not in {"stage", "dependencies", "resolution"}:
            _require(child[key] == parent[key], "parent_identity_mismatch")


def _classes(values):
    result = tuple(values)
    _require(len(result) == 3 and result == tuple(sorted(set(result))), "class_order_invalid")
    return result


def _scores(frame, classes):
    _require(
        all(tuple(json.loads(v)) == classes for v in frame.class_vocabulary), "class_order_mismatch"
    )
    values = np.asarray([json.loads(v) for v in frame.scores], dtype=np.float64)
    _require(values.shape == (len(frame), 3) and np.isfinite(values).all(), "scores_invalid")
    return values


def _canonical_frame(frame):
    result = frame.sort_values("observation_uid", kind="stable").reset_index(drop=True)
    return result.fillna("").astype(str)


def verify_calibration_prediction(*, job, fit_job, fit_bundle, fit_kwargs, saved_artifact_bytes):
    """Recheck the already-fitted estimator on its recorded source validation role."""
    job = _job(job, "calibration_validation_prediction")
    fit_job = _job(fit_job, "calibration_model_fit")
    _parent(job, fit_job)
    _require(job["model_id"] in p08_plan.CLASSICAL_MODELS, "classical_model_required")
    _require(job["dependencies"] == [fit_job["job_id"]], "dependency_mismatch")
    _require(fit_bundle.job == fit_job and fit_bundle.status == "complete", "bundle_mismatch")
    result = fit_bundle.result
    _require(isinstance(result, p03_runtime.CandidateFitOutcome), "outcome_type_invalid")
    _require(result.status == "complete" and result.estimator is not None, "outcome_incomplete")
    for name, expected in (
        ("fit_id", fit_job["job_id"]),
        ("model_id", job["model_id"]),
        ("seed", job["seed"]),
        ("candidate_id", fit_kwargs["candidate_id"]),
        ("fit_uid_sha256", job["fit_uid_sha256"]),
        ("validation_uid_sha256", job["validation_uid_sha256"]),
    ):
        _require(getattr(result, name) == expected, "outcome_identity_mismatch")
    _require(saved_artifact_bytes == fit_bundle.artifact_bytes(), "artifact_bytes_mismatch")
    _require(
        set(saved_artifact_bytes) == {"summary.json", "predictions.csv"}, "artifact_set_invalid"
    )
    frame = pd.read_csv(
        io.BytesIO(saved_artifact_bytes["predictions.csv"]), dtype=str, keep_default_na=False
    )
    frame = frame.sort_values("observation_uid", kind="stable").reset_index(drop=True)
    _require(
        _canonical_frame(frame).equals(_canonical_frame(result.validation_predictions)),
        "saved_predictions_mismatch",
    )
    uids = frame.observation_uid.tolist()
    _require(
        len(uids) == len(set(uids)) and uids == sorted(fit_kwargs["validation_uids"]),
        "validation_uid_mismatch",
    )
    _require(sha256_value(uids) == job["validation_uid_sha256"], "validation_hash_mismatch")
    values, metadata = fit_kwargs["dataset"].subset(uids)
    for output, source in (
        ("observation_uid", "observation_uid"),
        ("true_label", "target_analyte"),
        ("master_sample_id", "master_sample_id"),
        ("instrument", "instrument"),
        ("station", "station"),
    ):
        _require(
            frame[output].tolist() == metadata[source].astype(str).tolist(), "metadata_mismatch"
        )
    _require(frame.fit_id.eq(fit_job["job_id"]).all(), "prediction_fit_mismatch")
    _require(
        frame.probability_status.eq("uncalibrated").all() and frame.probabilities.eq("").all(),
        "predictions_already_calibrated",
    )
    classes = _classes(fit_kwargs["class_vocabulary"])
    scores = _scores(frame, classes)
    recomputed = p03_runtime._aligned_scores(result.estimator, values, classes)
    _require(np.array_equal(scores, recomputed), "score_parity_mismatch")
    _require(
        _metrics_equal(
            p03_runtime._master_metrics(frame, scores, classes), result.validation_metrics
        ),
        "metric_parity_mismatch",
    )
    frame["probabilities"] = None
    frame["seed"] = job["seed"]
    frame["selection_unit_id"] = job["unit_id"]
    return frame


def predict_held(
    *,
    job,
    fit_job,
    inputs,
    estimator=None,
    terminal_state=None,
    fit_summary,
    calibration_job=None,
    calibration=None,
    device="cpu",
):
    """Predict only after final source-fit and required calibration evidence is bound."""
    job = _job(job, "held_prediction")
    fit_job = _job(fit_job, "final_refit")
    _parent(job, fit_job)
    _require(fit_summary.get("status") == "complete", "fit_incomplete")
    neural = job["model_id"] in p08_plan.NEURAL_RECIPES
    if not neural:
        _require(
            estimator is not None
            and terminal_state is None
            and calibration is None
            and calibration_job is None,
            "classical_inputs_invalid",
        )
        _require(job["dependencies"] == [fit_job["job_id"]], "dependency_mismatch")
        for key, expected in (
            ("fit_id", fit_job["job_id"]),
            ("model_id", job["model_id"]),
            ("seed", job["seed"]),
            ("fit_uid_sha256", job["fit_uid_sha256"]),
        ):
            _require(fit_summary.get(key) == expected, "fit_summary_mismatch")
    else:
        _require(estimator is None, "neural_estimator_invalid")
        for key, expected in (
            ("fit_job_id", fit_job["job_id"]),
            ("recipe", job["model_id"]),
            ("seed", job["seed"]),
        ):
            _require(fit_summary.get(key) == expected, "fit_summary_mismatch")
        classes = _classes(fit_summary["classes"])
        _require(isinstance(terminal_state, dict) and bool(terminal_state), "state_missing")
        _require(
            all(
                isinstance(t, torch.Tensor) and torch.isfinite(t).all().item()
                for t in terminal_state.values()
            ),
            "state_nonfinite",
        )
        _require(
            p04_runtime._state_hash(terminal_state) == fit_summary.get("terminal_state_digest"),
            "state_digest_mismatch",
        )
        calibration_job = _job(calibration_job, p08_plan.SCALAR_STAGE)
        for key in (
            "policy_id",
            "representation_id",
            "array_sha256",
            "context_id",
            "model_id",
            "model_spec_sha256",
            "seed",
            "test_uid_sha256",
        ):
            _require(calibration_job[key] == job[key], "calibration_identity_mismatch")
        _require(
            job["dependencies"] == sorted([fit_job["job_id"], calibration_job["job_id"]]),
            "dependency_mismatch",
        )
        _require(isinstance(calibration, TemperatureCalibration), "calibration_type_invalid")
        _require(
            calibration.optimizer_success
            and np.isfinite(calibration.temperature)
            and calibration.temperature > 0
            and np.isfinite(calibration.optimizer_objective),
            "calibration_invalid",
        )
        _require(tuple(calibration.class_vocabulary) == classes, "calibration_class_mismatch")
    # Held arrays are acquired only after the eligible source evidence checks above.
    values, metadata, supplied_classes, forbidden = inputs.held_inputs(job)
    supplied_classes = _classes(supplied_classes)
    if neural:
        _require(supplied_classes == classes, "held_class_order_mismatch")
    classes = supplied_classes
    values = np.asarray(values)
    _require(
        values.shape == (len(metadata), 1401)
        and values.dtype == np.float32
        and np.isfinite(values).all(),
        "held_values_invalid",
    )
    uids = metadata.observation_uid.astype(str).tolist()
    _require(
        len(uids) > 0 and len(uids) == len(set(uids)) and not set(uids) & set(forbidden),
        "held_uids_invalid",
    )
    _require(sha256_value(sorted(uids)) == job["test_uid_sha256"], "test_hash_mismatch")
    _require(
        metadata.station.nunique() == 1 and set(metadata.target_analyte) <= set(classes),
        "held_metadata_invalid",
    )
    if not neural:
        dataset = p03_runtime.P03Dataset.from_frozen_representation(
            intensity=values, representation_uids=np.asarray(uids), metadata=metadata
        )
        frame = p03_runtime.run_final_prediction(
            dataset=dataset,
            estimator=estimator,
            fit_id=fit_job["job_id"],
            test_uids=uids,
            forbidden_fit_uids=forbidden,
            class_vocabulary=classes,
            calibration=None,
        )
    else:
        torch_device = torch.device(device)
        cuda = torch_device.type == "cuda"
        if cuda:
            _require(torch.cuda.memory_allocated(torch_device) <= 4 * 2**30, "cuda_limit")
        with torch.random.fork_rng(devices=[torch_device.index or 0] if cuda else []):
            model = AcquisitionClassifier(
                class_count=3, use_projection=bool(RECIPE_SPECIFICATIONS[job["model_id"]][2])
            )
            model.load_state_dict(terminal_state, strict=True)
            model.to(torch_device)
            tensor = torch.from_numpy(np.ascontiguousarray(values[:, None, :]))
            logits = p05_development._predict_logits(model, tensor, torch_device)
            if cuda:
                _require(torch.cuda.memory_allocated(torch_device) <= 4 * 2**30, "cuda_limit")
        frame = p03_runtime._prediction_frame(
            metadata=metadata,
            scores=logits,
            class_vocabulary=classes,
            fit_id=fit_job["job_id"],
            calibrated_probabilities=apply_temperature(logits, calibration),
        )
    _scores(frame, classes)
    frame["seed"] = job["seed"]
    return frame
