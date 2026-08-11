from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.classical import fit_temperature
from atlas_sers.evaluation.p03_controls import permute_master_labels
from atlas_sers.evaluation.p03_runtime import (
    P03Dataset,
    aggregate_seed_prediction_frames,
    fit_cross_fitted_temperature,
    run_candidate_fit,
    run_final_fit,
    run_final_prediction,
)
from atlas_sers.governance.canonical import sha256_value


def _dataset() -> P03Dataset:
    rng = np.random.default_rng(20260805)
    records: list[dict[str, str]] = []
    rows: list[np.ndarray] = []
    centers = {"a": -1.0, "b": 0.0, "c": 1.0}
    index = 0
    for label in ("a", "b", "c"):
        for master_index in range(3):
            master = f"{label}-master-{master_index}"
            for instrument in ("unit-1", "unit-2"):
                uid = f"row-{index:03d}"
                records.append(
                    {
                        "observation_uid": uid,
                        "master_sample_id": master,
                        "target_analyte": label,
                        "instrument": instrument,
                        "station": "cwa",
                        "sensor_family": "sensor-a",
                        "sensor_variant": "variant-a",
                        "source_format": "text",
                        "team": "team-a",
                        "averages": 3,
                        "laser_power": 10,
                        "n_points": 1401,
                        "axis_min_cm1": 400,
                        "axis_max_cm1": 1800,
                        "axis_step_median_cm1": 1.0,
                        "leading_constant_points": 0,
                        "trailing_constant_points": 0,
                        "finite_fraction": 1.0,
                    }
                )
                rows.append(rng.normal(centers[label], 0.1, 12))
                index += 1
    metadata = pd.DataFrame(records)
    uids = metadata.observation_uid.to_numpy()
    return P03Dataset.from_frozen_representation(
        intensity=np.vstack(rows),
        representation_uids=uids,
        metadata=metadata,
    )


def _role_uids(dataset: P03Dataset) -> tuple[list[str], list[str]]:
    train_masters = {
        "a-master-0",
        "a-master-1",
        "b-master-0",
        "b-master-1",
        "c-master-0",
        "c-master-1",
    }
    fit = dataset.metadata[dataset.metadata.master_sample_id.isin(train_masters)]
    validation = dataset.metadata[~dataset.metadata.master_sample_id.isin(train_masters)]
    return fit.observation_uid.tolist(), validation.observation_uid.tolist()


def test_runtime_fit_audits_roles_and_emits_master_metrics() -> None:
    dataset = _dataset()
    fit_uids, validation_uids = _role_uids(dataset)
    outcome = run_candidate_fit(
        dataset=dataset,
        fit_id="fit-1",
        model_id="C-PCA-LDA",
        candidate_id="C-PCA-LDA-000",
        parameters={"pca_components": 5, "lda": "svd"},
        seed="deterministic",
        fit_uids=fit_uids,
        validation_uids=validation_uids,
        class_vocabulary=["a", "b", "c"],
        expected_fit_uid_sha256=sha256_value(sorted(fit_uids)),
        expected_validation_uid_sha256=sha256_value(sorted(validation_uids)),
    )
    assert outcome.status == "complete"
    assert outcome.estimator is not None
    assert len(outcome.validation_predictions) == 6
    assert outcome.validation_metrics is not None
    assert set(outcome.validation_metrics["per_class_recall"]) == {"a", "b", "c"}
    assert outcome.status_record()["status"] == "complete"


def test_runtime_preserves_rank_failure_as_terminal_record() -> None:
    dataset = _dataset()
    fit_uids, validation_uids = _role_uids(dataset)
    outcome = run_candidate_fit(
        dataset=dataset,
        fit_id="fit-rank",
        model_id="C-PCA-LDA",
        candidate_id="C-PCA-LDA-oversized",
        parameters={"pca_components": 40, "lda": "svd"},
        seed="deterministic",
        fit_uids=fit_uids,
        validation_uids=validation_uids,
        class_vocabulary=["a", "b", "c"],
    )
    assert outcome.status == "rank_failure"
    assert outcome.validation_predictions.empty
    assert outcome.estimator is None
    assert "pca_components_exceed_rank" in str(outcome.reason_code)


def test_runtime_role_hash_and_master_overlap_fail_closed() -> None:
    dataset = _dataset()
    fit_uids, validation_uids = _role_uids(dataset)
    with pytest.raises(ValueError, match="Fit UID hash"):
        run_candidate_fit(
            dataset=dataset,
            fit_id="fit-hash",
            model_id="C-PRIOR",
            candidate_id="C-PRIOR-000",
            parameters={"prior": "uniform"},
            seed="deterministic",
            fit_uids=fit_uids,
            validation_uids=validation_uids,
            class_vocabulary=["a", "b", "c"],
            expected_fit_uid_sha256="0" * 64,
        )
    overlapping_validation = [fit_uids[0], *validation_uids]
    with pytest.raises(ValueError, match="observation UIDs overlap"):
        run_candidate_fit(
            dataset=dataset,
            fit_id="fit-overlap",
            model_id="C-PRIOR",
            candidate_id="C-PRIOR-000",
            parameters={"prior": "uniform"},
            seed="deterministic",
            fit_uids=fit_uids,
            validation_uids=overlapping_validation,
            class_vocabulary=["a", "b", "c"],
        )


def test_outer_prediction_requires_disjoint_frozen_calibration() -> None:
    dataset = _dataset()
    fit_uids, validation_uids = _role_uids(dataset)
    outcome = run_candidate_fit(
        dataset=dataset,
        fit_id="fit-final",
        model_id="C-RBF-SVM",
        candidate_id="C-RBF-SVM-000",
        parameters={"C": 1.0, "gamma": "scale", "class_weight": "balanced"},
        seed="deterministic",
        fit_uids=fit_uids,
        validation_uids=validation_uids,
        class_vocabulary=["a", "b", "c"],
    )
    assert outcome.estimator is not None
    scores = np.vstack(
        [np.asarray(json.loads(value)) for value in outcome.validation_predictions.scores]
    )
    calibration = fit_temperature(
        scores,
        outcome.validation_predictions.true_label.to_numpy(),
        class_vocabulary=["a", "b", "c"],
        observation_uids=outcome.validation_predictions.observation_uid.to_numpy(),
        master_ids=outcome.validation_predictions.master_sample_id.to_numpy(),
    )
    predictions = run_final_prediction(
        dataset=dataset,
        estimator=outcome.estimator,
        fit_id="fit-final",
        test_uids=validation_uids,
        forbidden_fit_uids=fit_uids,
        class_vocabulary=["a", "b", "c"],
        calibration=calibration,
    )
    assert predictions.probability_status.eq("cross_fitted_temperature").all()
    with pytest.raises(ValueError, match="Outer-test UID"):
        run_final_prediction(
            dataset=dataset,
            estimator=outcome.estimator,
            fit_id="fit-final",
            test_uids=validation_uids,
            forbidden_fit_uids=[*fit_uids, validation_uids[0]],
            class_vocabulary=["a", "b", "c"],
            calibration=calibration,
        )


def test_final_refit_never_requires_validation_or_test_data() -> None:
    dataset = _dataset()
    fit_uids, test_uids = _role_uids(dataset)
    outcome = run_final_fit(
        dataset=dataset,
        fit_id="final-refit",
        model_id="C-SPECTRAL-MATCH",
        candidate_id="C-SPECTRAL-MATCH-000",
        parameters={"metric": "cosine"},
        seed="deterministic",
        fit_uids=fit_uids,
        expected_fit_uid_sha256=sha256_value(sorted(fit_uids)),
    )
    assert outcome.status == "complete"
    assert outcome.estimator is not None
    predictions = run_final_prediction(
        dataset=dataset,
        estimator=outcome.estimator,
        fit_id=outcome.fit_id,
        test_uids=test_uids,
        forbidden_fit_uids=fit_uids,
        class_vocabulary=["a", "b", "c"],
        calibration=None,
    )
    assert len(predictions) == len(test_uids)
    assert predictions.probability_status.eq("uncalibrated").all()


def test_metadata_runtime_never_uses_spectral_matrix_for_fit_or_prediction() -> None:
    dataset = _dataset()
    fit_uids, validation_uids = _role_uids(dataset)
    parameters = {
        "base_parameters": {"C": 1.0, "l1_ratio": 0.5},
        "categorical_features": [
            "instrument",
            "instrument_family",
            "sensor_family",
            "sensor_variant",
            "source_format",
            "team",
        ],
        "numeric_features": [
            "averages",
            "laser_power",
            "n_points",
            "axis_min_cm1",
            "axis_max_cm1",
            "axis_step_median_cm1",
            "leading_constant_points",
            "trailing_constant_points",
            "finite_fraction",
        ],
    }
    outcome = run_candidate_fit(
        dataset=dataset,
        fit_id="metadata-control",
        model_id="C-METADATA-LOGREG",
        candidate_id="CTRL-META-test",
        parameters=parameters,
        seed="deterministic",
        fit_uids=fit_uids,
        validation_uids=validation_uids,
        class_vocabulary=["a", "b", "c"],
    )
    assert outcome.status == "complete"
    assert outcome.estimator is not None
    assert outcome.estimator.input_kind == "acquisition_metadata"
    state = outcome.estimator.source_state_sha256_
    predictions = run_final_prediction(
        dataset=dataset,
        estimator=outcome.estimator,
        fit_id="metadata-control",
        test_uids=validation_uids,
        forbidden_fit_uids=fit_uids,
        class_vocabulary=["a", "b", "c"],
        calibration=None,
    )
    assert len(predictions) == len(validation_uids)
    assert outcome.estimator.source_state_sha256_ == state


def test_final_refit_accepts_only_a_hashed_master_safe_label_override() -> None:
    dataset = _dataset()
    fit_uids, _ = _role_uids(dataset)
    fit_metadata = dataset.subset(fit_uids)[1]
    permutation = permute_master_labels(fit_metadata, seed=20261001)
    outcome = run_final_fit(
        dataset=dataset,
        fit_id="permutation-control",
        model_id="C-SPECTRAL-MATCH",
        candidate_id="frozen-real-selection",
        parameters={"metric": "cosine"},
        seed="deterministic",
        fit_uids=fit_uids,
        fit_labels=permutation.labels,
        expected_fit_label_sha256=permutation.label_sha256,
    )
    assert outcome.status == "complete"
    assert outcome.fit_label_sha256 == permutation.label_sha256
    with pytest.raises(ValueError, match="Fit-label hash"):
        run_final_fit(
            dataset=dataset,
            fit_id="permutation-control-bad-hash",
            model_id="C-SPECTRAL-MATCH",
            candidate_id="frozen-real-selection",
            parameters={"metric": "cosine"},
            seed="deterministic",
            fit_uids=fit_uids,
            fit_labels=permutation.labels,
            expected_fit_label_sha256="0" * 64,
        )


def test_forest_seed_aggregation_requires_three_identical_test_roles() -> None:
    dataset = _dataset()
    fit_uids, test_uids = _role_uids(dataset)
    outcome = run_final_fit(
        dataset=dataset,
        fit_id="forest-1",
        model_id="C-RANDOM-FOREST",
        candidate_id="C-RANDOM-FOREST-test",
        parameters={
            "n_estimators": 10,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        },
        seed=20260805,
        fit_uids=fit_uids,
    )
    assert outcome.estimator is not None
    first = run_final_prediction(
        dataset=dataset,
        estimator=outcome.estimator,
        fit_id="forest-1",
        test_uids=test_uids,
        forbidden_fit_uids=fit_uids,
        class_vocabulary=["a", "b", "c"],
        calibration=None,
    )
    frames = []
    for index in range(3):
        frame = first.copy()
        frame["fit_id"] = f"forest-{index}"
        frames.append(frame)
    aggregate = aggregate_seed_prediction_frames(
        frames,
        model_id="C-RANDOM-FOREST",
        aggregate_fit_id="forest-aggregate",
        class_vocabulary=["a", "b", "c"],
        calibration=None,
    )
    assert aggregate.technical_seed_count.eq(3).all()
    assert aggregate.fit_id.eq("forest-aggregate").all()
    with pytest.raises(ValueError, match="requires 3"):
        aggregate_seed_prediction_frames(
            frames[:2],
            model_id="C-RANDOM-FOREST",
            aggregate_fit_id="incomplete",
            class_vocabulary=["a", "b", "c"],
            calibration=None,
        )


def test_temperature_fit_uses_unique_master_grouped_crossfit_rows() -> None:
    dataset = _dataset()
    fit_uids, validation_uids = _role_uids(dataset)
    outcome = run_candidate_fit(
        dataset=dataset,
        fit_id="crossfit-1",
        model_id="C-SPECTRAL-MATCH",
        candidate_id="C-SPECTRAL-MATCH-000",
        parameters={"metric": "cosine"},
        seed="deterministic",
        fit_uids=fit_uids,
        validation_uids=validation_uids,
        class_vocabulary=["a", "b", "c"],
    )
    frame = outcome.validation_predictions.copy()
    frame["seed"] = "deterministic"
    frame["selection_unit_id"] = "fold-0"
    result = fit_cross_fitted_temperature(
        frame,
        model_id="C-SPECTRAL-MATCH",
        class_vocabulary=["a", "b", "c"],
    )
    assert result.selection_unit_count == 1
    assert result.calibration.observations == len(validation_uids)
    assert result.calibration.masters == 3
    duplicate = pd.concat([frame, frame.assign(selection_unit_id="fold-1")])
    with pytest.raises(ValueError, match="multiple cross-fit folds"):
        fit_cross_fitted_temperature(
            duplicate,
            model_id="C-SPECTRAL-MATCH",
            class_vocabulary=["a", "b", "c"],
        )
