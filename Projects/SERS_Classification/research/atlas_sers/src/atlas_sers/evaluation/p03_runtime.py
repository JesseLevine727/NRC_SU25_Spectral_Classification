"""Leakage-audited runtime primitives for outcome-bearing P03 fits."""

from __future__ import annotations

import json
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from atlas_sers.evaluation.classical import (
    TemperatureCalibration,
    apply_temperature,
    classification_metrics,
    fit_temperature,
    instrument_balanced_master_probabilities,
    softmax,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.classical import (
    STOCHASTIC_MODELS,
    UnsupportedCandidate,
    build_classical_estimator,
)

TERMINAL_STATUSES = {
    "complete",
    "unsupported_candidate",
    "rank_failure",
    "convergence_failure",
    "numerical_failure",
    "resource_failure",
    "data_failure",
    "fit_failure",
    "excluded_by_protocol",
}


def _uid_hash(values: np.ndarray | list[str]) -> str:
    return sha256_value(sorted(str(value) for value in values))


@dataclass(frozen=True)
class P03Dataset:
    intensity: np.ndarray
    metadata: pd.DataFrame
    uid_to_index: dict[str, int]

    @classmethod
    def from_frozen_representation(
        cls,
        *,
        intensity: np.ndarray,
        representation_uids: np.ndarray,
        metadata: pd.DataFrame,
    ) -> P03Dataset:
        matrix = np.asarray(intensity, dtype=np.float64)
        uids = np.asarray(representation_uids).astype(str)
        frame = metadata.copy().reset_index(drop=True)
        required = {
            "observation_uid",
            "master_sample_id",
            "target_analyte",
            "instrument",
            "station",
        }
        if not required <= set(frame):
            raise ValueError(f"Frozen metadata misses fields: {sorted(required - set(frame))}")
        manifest_uids = frame.observation_uid.astype(str).to_numpy()
        if matrix.ndim != 2 or len(matrix) != len(uids) or len(uids) != len(frame):
            raise ValueError("Representation, UID order, and manifest lengths disagree.")
        if len(set(uids)) != len(uids):
            raise ValueError("Representation observation UIDs are not unique.")
        if not np.array_equal(uids, manifest_uids):
            raise ValueError("Representation UID order differs from the immutable manifest.")
        if not np.isfinite(matrix).all():
            raise ValueError("Frozen representation contains a nonfinite value.")
        return cls(matrix, frame, {uid: index for index, uid in enumerate(uids)})

    def indices(self, observation_uids: list[str] | np.ndarray) -> np.ndarray:
        requested = [str(value) for value in observation_uids]
        if len(requested) != len(set(requested)):
            raise ValueError("A role requested duplicate observation UIDs.")
        missing = [uid for uid in requested if uid not in self.uid_to_index]
        if missing:
            raise ValueError(f"A role requested {len(missing)} unknown observation UIDs.")
        return np.asarray([self.uid_to_index[uid] for uid in requested], dtype=int)

    def subset(self, observation_uids: list[str] | np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
        indices = self.indices(observation_uids)
        return self.intensity[indices], self.metadata.iloc[indices].reset_index(drop=True)


@dataclass
class CandidateFitOutcome:
    fit_id: str
    status: str
    reason_code: str | None
    model_id: str
    candidate_id: str
    seed: int | str
    fit_uid_sha256: str
    validation_uid_sha256: str
    fit_master_sha256: str
    elapsed_seconds: float
    inference_seconds: float | None
    serialized_model_bytes: int | None
    warnings: list[str]
    traceback_digest: str | None
    validation_predictions: pd.DataFrame
    validation_metrics: dict[str, Any] | None
    estimator: Any | None

    def status_record(self) -> dict[str, Any]:
        if self.status not in TERMINAL_STATUSES:
            raise ValueError(f"Unknown P03 terminal status: {self.status}")
        return {
            "fit_id": self.fit_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "model_id": self.model_id,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "fit_uid_sha256": self.fit_uid_sha256,
            "validation_uid_sha256": self.validation_uid_sha256,
            "fit_master_sha256": self.fit_master_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "inference_seconds": self.inference_seconds,
            "serialized_model_bytes": self.serialized_model_bytes,
            "warnings": "|".join(self.warnings) if self.warnings else "none",
            "traceback_digest": self.traceback_digest,
        }


@dataclass
class FinalFitOutcome:
    fit_id: str
    status: str
    reason_code: str | None
    model_id: str
    candidate_id: str
    seed: int | str
    fit_uid_sha256: str
    fit_master_sha256: str
    elapsed_seconds: float
    serialized_model_bytes: int | None
    warnings: list[str]
    traceback_digest: str | None
    estimator: Any | None
    fit_label_sha256: str | None = None

    def status_record(self) -> dict[str, Any]:
        if self.status not in TERMINAL_STATUSES:
            raise ValueError(f"Unknown P03 terminal status: {self.status}")
        return {
            "fit_id": self.fit_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "model_id": self.model_id,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "fit_uid_sha256": self.fit_uid_sha256,
            "fit_master_sha256": self.fit_master_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "serialized_model_bytes": self.serialized_model_bytes,
            "warnings": "|".join(self.warnings) if self.warnings else "none",
            "traceback_digest": self.traceback_digest,
            "fit_label_sha256": self.fit_label_sha256,
        }


@dataclass(frozen=True)
class CrossFittedCalibrationResult:
    calibration: TemperatureCalibration
    cross_fitted_predictions: pd.DataFrame
    selection_unit_count: int
    evidence_fit_id_sha256: str


def _aligned_scores(
    model: Any, x: Any, class_vocabulary: tuple[str, ...]
) -> np.ndarray:
    local_scores = model.scores(x)
    local_classes = [str(value) for value in model.classes_]
    if local_scores.shape != (len(x), len(local_classes)):
        raise ValueError("Estimator score columns do not match its classes.")
    aligned = np.full((len(x), len(class_vocabulary)), -1e12, dtype=np.float64)
    vocabulary_index = {label: index for index, label in enumerate(class_vocabulary)}
    unknown = [label for label in local_classes if label not in vocabulary_index]
    if unknown:
        raise ValueError("Estimator produced a class outside the frozen vocabulary.")
    for local_index, label in enumerate(local_classes):
        aligned[:, vocabulary_index[label]] = local_scores[:, local_index]
    return aligned


def _prediction_frame(
    *,
    metadata: pd.DataFrame,
    scores: np.ndarray,
    class_vocabulary: tuple[str, ...],
    fit_id: str,
    calibrated_probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    predicted = np.asarray(class_vocabulary)[np.argmax(scores, axis=1)]
    frame = metadata[
        [
            "observation_uid",
            "master_sample_id",
            "instrument",
            "station",
            "target_analyte",
        ]
    ].copy()
    frame = frame.rename(columns={"target_analyte": "true_label"})
    frame["fit_id"] = fit_id
    frame["predicted_label"] = predicted
    frame["class_vocabulary"] = json.dumps(class_vocabulary, separators=(",", ":"))
    frame["scores"] = [json.dumps(row.tolist(), separators=(",", ":")) for row in scores]
    if calibrated_probabilities is None:
        frame["probabilities"] = None
        frame["probability_status"] = "uncalibrated"
    else:
        frame["probabilities"] = [
            json.dumps(row.tolist(), separators=(",", ":")) for row in calibrated_probabilities
        ]
        frame["probability_status"] = "cross_fitted_temperature"
    return frame


def _master_metrics(
    predictions: pd.DataFrame,
    scores: np.ndarray,
    class_vocabulary: tuple[str, ...],
) -> dict[str, Any]:
    # Score softmax is used only to aggregate a class decision during inner
    # selection. It is never reported as calibrated probability evidence.
    master = instrument_balanced_master_probabilities(
        probabilities=softmax(scores),
        true_labels=predictions.true_label.to_numpy(),
        master_ids=predictions.master_sample_id.to_numpy(),
        instruments=predictions.instrument.to_numpy(),
        class_vocabulary=class_vocabulary,
    )
    return classification_metrics(
        master.true_label.to_numpy(),
        master.predicted_label.to_numpy(),
        class_vocabulary=class_vocabulary,
        probabilities=None,
    )


def run_candidate_fit(
    *,
    dataset: P03Dataset,
    fit_id: str,
    model_id: str,
    candidate_id: str,
    parameters: dict[str, Any],
    seed: int | str,
    fit_uids: list[str] | np.ndarray,
    validation_uids: list[str] | np.ndarray,
    class_vocabulary: list[str] | tuple[str, ...],
    expected_fit_uid_sha256: str | None = None,
    expected_validation_uid_sha256: str | None = None,
) -> CandidateFitOutcome:
    """Fit one candidate using only explicit authorized role UIDs."""

    fit_uid_sha256 = _uid_hash(list(fit_uids))
    validation_uid_sha256 = _uid_hash(list(validation_uids))
    if expected_fit_uid_sha256 and fit_uid_sha256 != expected_fit_uid_sha256:
        raise ValueError("Fit UID hash differs from the no-fit P03 manifest.")
    if expected_validation_uid_sha256 and (validation_uid_sha256 != expected_validation_uid_sha256):
        raise ValueError("Validation UID hash differs from the no-fit P03 manifest.")
    if set(str(value) for value in fit_uids) & set(str(value) for value in validation_uids):
        raise ValueError("Fit and validation observation UIDs overlap.")
    x_fit, fit_metadata = dataset.subset(fit_uids)
    x_validation, validation_metadata = dataset.subset(validation_uids)
    fit_masters = fit_metadata.master_sample_id.astype(str).to_numpy()
    validation_masters = validation_metadata.master_sample_id.astype(str).to_numpy()
    if set(fit_masters) & set(validation_masters):
        raise ValueError("Fit and validation physical masters overlap.")
    vocabulary = tuple(str(value) for value in class_vocabulary)
    start = time.perf_counter()
    captured: list[str] = []
    model: Any | None = None
    try:
        estimator_seed = None if seed == "deterministic" else int(seed)
        if model_id == "C-METADATA-LOGREG":
            from atlas_sers.evaluation.p03_controls import build_metadata_only_classifier

            model = build_metadata_only_classifier(parameters, seed=estimator_seed)
        else:
            model = build_classical_estimator(
                model_id,
                parameters,
                fit_samples=len(x_fit),
                features=x_fit.shape[1],
                seed=estimator_seed,
            )
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            if model_id == "C-METADATA-LOGREG":
                model.fit(
                    fit_metadata,
                    fit_metadata.target_analyte.astype(str).to_numpy(),
                    observation_uids=fit_metadata.observation_uid.astype(str).to_numpy(),
                    master_ids=fit_masters,
                )
            else:
                model.fit(
                    x_fit,
                    fit_metadata.target_analyte.astype(str).to_numpy(),
                    observation_uids=fit_metadata.observation_uid.astype(str).to_numpy(),
                    master_ids=fit_masters,
                    domain_ids=fit_metadata.instrument.astype(str).to_numpy(),
                )
        captured = [
            f"{record.category.__name__}:{str(record.message)}" for record in warning_records
        ]
        convergence = [
            record for record in warning_records if issubclass(record.category, ConvergenceWarning)
        ]
        if convergence:
            return _failure_outcome(
                fit_id=fit_id,
                status="convergence_failure",
                reason="estimator_emitted_convergence_warning",
                model_id=model_id,
                candidate_id=candidate_id,
                seed=seed,
                fit_uid_sha256=fit_uid_sha256,
                validation_uid_sha256=validation_uid_sha256,
                fit_masters=fit_masters,
                start=start,
                warnings_list=captured,
            )
        if model.fit_audit is None or model.fit_audit.observation_uid_sha256 != fit_uid_sha256:
            raise RuntimeError("Estimator fit audit does not match the authorized UID set.")
        inference_start = time.perf_counter()
        validation_input = validation_metadata if model_id == "C-METADATA-LOGREG" else x_validation
        scores = _aligned_scores(model, validation_input, vocabulary)
        inference_seconds = time.perf_counter() - inference_start
        if not np.isfinite(scores).all():
            raise FloatingPointError("candidate_scores_nonfinite")
        predictions = _prediction_frame(
            metadata=validation_metadata,
            scores=scores,
            class_vocabulary=vocabulary,
            fit_id=fit_id,
        )
        metrics = _master_metrics(predictions, scores, vocabulary)
        return CandidateFitOutcome(
            fit_id=fit_id,
            status="complete",
            reason_code=None,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            validation_uid_sha256=validation_uid_sha256,
            fit_master_sha256=_uid_hash(fit_masters),
            elapsed_seconds=time.perf_counter() - start,
            inference_seconds=inference_seconds,
            serialized_model_bytes=model.serialized_size_bytes(),
            warnings=captured,
            traceback_digest=None,
            validation_predictions=predictions,
            validation_metrics=metrics,
            estimator=model,
        )
    except UnsupportedCandidate as error:
        reason = str(error)
        status = (
            "rank_failure"
            if "rank" in reason or "components" in reason
            else "unsupported_candidate"
        )
        return _failure_outcome(
            fit_id=fit_id,
            status=status,
            reason=reason,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            validation_uid_sha256=validation_uid_sha256,
            fit_masters=fit_masters,
            start=start,
            warnings_list=captured,
        )
    except MemoryError as error:
        return _exception_outcome(
            error=error,
            status="resource_failure",
            fit_id=fit_id,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            validation_uid_sha256=validation_uid_sha256,
            fit_masters=fit_masters,
            start=start,
            warnings_list=captured,
        )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        return _exception_outcome(
            error=error,
            status="numerical_failure",
            fit_id=fit_id,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            validation_uid_sha256=validation_uid_sha256,
            fit_masters=fit_masters,
            start=start,
            warnings_list=captured,
        )
    except (KeyError, ValueError) as error:
        return _exception_outcome(
            error=error,
            status="data_failure",
            fit_id=fit_id,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            validation_uid_sha256=validation_uid_sha256,
            fit_masters=fit_masters,
            start=start,
            warnings_list=captured,
        )
    except Exception as error:  # pragma: no cover - terminal containment boundary
        return _exception_outcome(
            error=error,
            status="fit_failure",
            fit_id=fit_id,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            validation_uid_sha256=validation_uid_sha256,
            fit_masters=fit_masters,
            start=start,
            warnings_list=captured,
        )


def _failure_outcome(
    *,
    fit_id: str,
    status: str,
    reason: str,
    model_id: str,
    candidate_id: str,
    seed: int | str,
    fit_uid_sha256: str,
    validation_uid_sha256: str,
    fit_masters: np.ndarray,
    start: float,
    warnings_list: list[str],
    traceback_digest: str | None = None,
) -> CandidateFitOutcome:
    return CandidateFitOutcome(
        fit_id=fit_id,
        status=status,
        reason_code=reason,
        model_id=model_id,
        candidate_id=candidate_id,
        seed=seed,
        fit_uid_sha256=fit_uid_sha256,
        validation_uid_sha256=validation_uid_sha256,
        fit_master_sha256=_uid_hash(fit_masters),
        elapsed_seconds=time.perf_counter() - start,
        inference_seconds=None,
        serialized_model_bytes=None,
        warnings=warnings_list,
        traceback_digest=traceback_digest,
        validation_predictions=pd.DataFrame(),
        validation_metrics=None,
        estimator=None,
    )


def _exception_outcome(error: Exception, *, status: str, **kwargs: Any) -> CandidateFitOutcome:
    sanitized_traceback = "\n".join(
        line for line in traceback.format_exception_only(type(error), error)
    ).strip()
    return _failure_outcome(
        status=status,
        reason=f"{type(error).__name__}:{error}",
        traceback_digest=sha256_value({"exception": sanitized_traceback}),
        **kwargs,
    )


def _final_failure_outcome(
    *,
    fit_id: str,
    status: str,
    reason: str,
    model_id: str,
    candidate_id: str,
    seed: int | str,
    fit_uid_sha256: str,
    fit_masters: np.ndarray,
    start: float,
    warnings_list: list[str],
    traceback_digest: str | None = None,
) -> FinalFitOutcome:
    return FinalFitOutcome(
        fit_id=fit_id,
        status=status,
        reason_code=reason,
        model_id=model_id,
        candidate_id=candidate_id,
        seed=seed,
        fit_uid_sha256=fit_uid_sha256,
        fit_master_sha256=_uid_hash(fit_masters),
        elapsed_seconds=time.perf_counter() - start,
        serialized_model_bytes=None,
        warnings=warnings_list,
        traceback_digest=traceback_digest,
        estimator=None,
    )


def _final_exception_outcome(
    error: Exception, *, status: str, **kwargs: Any
) -> FinalFitOutcome:
    sanitized = "\n".join(
        line for line in traceback.format_exception_only(type(error), error)
    ).strip()
    return _final_failure_outcome(
        status=status,
        reason=f"{type(error).__name__}:{error}",
        traceback_digest=sha256_value({"exception": sanitized}),
        **kwargs,
    )


def run_final_fit(
    *,
    dataset: P03Dataset,
    fit_id: str,
    model_id: str,
    candidate_id: str,
    parameters: dict[str, Any],
    seed: int | str,
    fit_uids: list[str] | np.ndarray,
    expected_fit_uid_sha256: str | None = None,
    fit_labels: np.ndarray | list[str] | None = None,
    expected_fit_label_sha256: str | None = None,
) -> FinalFitOutcome:
    """Refit a frozen selected candidate without exposing an outer-test row."""

    fit_uid_sha256 = _uid_hash(list(fit_uids))
    if expected_fit_uid_sha256 and fit_uid_sha256 != expected_fit_uid_sha256:
        raise ValueError("Fit UID hash differs from the no-fit P03 manifest.")
    x_fit, metadata = dataset.subset(fit_uids)
    fit_masters = metadata.master_sample_id.astype(str).to_numpy()
    effective_labels = (
        metadata.target_analyte.astype(str).to_numpy()
        if fit_labels is None
        else np.asarray(fit_labels).astype(str)
    )
    if len(effective_labels) != len(metadata):
        raise ValueError("Fit-label override length differs from the authorized fit role.")
    if len(np.unique(effective_labels)) < 2:
        raise ValueError("Fit labels contain fewer than two classes.")
    if (
        pd.DataFrame({"master": fit_masters, "label": effective_labels})
        .groupby("master")
        .label.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Fit-label override assigns multiple labels to one physical master.")
    fit_label_sha256 = sha256_value(
        [
            {"observation_uid": str(uid), "fit_label": str(label)}
            for uid, label in zip(metadata.observation_uid, effective_labels, strict=True)
        ]
    )
    if expected_fit_label_sha256 and fit_label_sha256 != expected_fit_label_sha256:
        raise ValueError("Fit-label hash differs from the frozen control mapping.")
    start = time.perf_counter()
    captured: list[str] = []

    def captured_exception(error: Exception, status: str) -> FinalFitOutcome:
        return _final_exception_outcome(
            error,
            status=status,
            fit_id=fit_id,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            fit_masters=fit_masters,
            start=start,
            warnings_list=captured,
        )

    try:
        estimator_seed = None if seed == "deterministic" else int(seed)
        if model_id == "C-METADATA-LOGREG":
            from atlas_sers.evaluation.p03_controls import build_metadata_only_classifier

            model = build_metadata_only_classifier(parameters, seed=estimator_seed)
        else:
            model = build_classical_estimator(
                model_id,
                parameters,
                fit_samples=len(x_fit),
                features=x_fit.shape[1],
                seed=estimator_seed,
            )
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            if model_id == "C-METADATA-LOGREG":
                model.fit(
                    metadata,
                    effective_labels,
                    observation_uids=metadata.observation_uid.astype(str).to_numpy(),
                    master_ids=fit_masters,
                )
            else:
                model.fit(
                    x_fit,
                    effective_labels,
                    observation_uids=metadata.observation_uid.astype(str).to_numpy(),
                    master_ids=fit_masters,
                    domain_ids=metadata.instrument.astype(str).to_numpy(),
                )
        captured = [
            f"{record.category.__name__}:{str(record.message)}" for record in warning_records
        ]
        if any(issubclass(record.category, ConvergenceWarning) for record in warning_records):
            return _final_failure_outcome(
                fit_id=fit_id,
                status="convergence_failure",
                reason="estimator_emitted_convergence_warning",
                model_id=model_id,
                candidate_id=candidate_id,
                seed=seed,
                fit_uid_sha256=fit_uid_sha256,
                fit_masters=fit_masters,
                start=start,
                warnings_list=captured,
            )
        if model.fit_audit is None or model.fit_audit.observation_uid_sha256 != fit_uid_sha256:
            raise RuntimeError("Estimator fit audit does not match the authorized UID set.")
        return FinalFitOutcome(
            fit_id=fit_id,
            status="complete",
            reason_code=None,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            fit_master_sha256=_uid_hash(fit_masters),
            elapsed_seconds=time.perf_counter() - start,
            serialized_model_bytes=model.serialized_size_bytes(),
            warnings=captured,
            traceback_digest=None,
            estimator=model,
            fit_label_sha256=fit_label_sha256,
        )
    except UnsupportedCandidate as error:
        reason = str(error)
        status = (
            "rank_failure"
            if "rank" in reason or "components" in reason
            else "unsupported_candidate"
        )
        return _final_failure_outcome(
            fit_id=fit_id,
            status=status,
            reason=reason,
            model_id=model_id,
            candidate_id=candidate_id,
            seed=seed,
            fit_uid_sha256=fit_uid_sha256,
            fit_masters=fit_masters,
            start=start,
            warnings_list=captured,
        )
    except MemoryError as error:
        return captured_exception(error, "resource_failure")
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        return captured_exception(error, "numerical_failure")
    except (KeyError, ValueError) as error:
        return captured_exception(error, "data_failure")
    except Exception as error:  # pragma: no cover - terminal containment boundary
        return captured_exception(error, "fit_failure")


def run_final_prediction(
    *,
    dataset: P03Dataset,
    estimator: Any,
    fit_id: str,
    test_uids: list[str] | np.ndarray,
    forbidden_fit_uids: list[str] | np.ndarray,
    class_vocabulary: list[str] | tuple[str, ...],
    calibration: TemperatureCalibration | None,
) -> pd.DataFrame:
    """Predict an outer test role only after the estimator and calibration freeze."""

    requested_test = {str(value) for value in test_uids}
    forbidden = {str(value) for value in forbidden_fit_uids}
    if requested_test & forbidden:
        raise ValueError("Outer-test UID appears in a fitting/selection/calibration set.")
    x_test, metadata = dataset.subset(test_uids)
    vocabulary = tuple(str(value) for value in class_vocabulary)
    prediction_input = metadata if getattr(estimator, "input_kind", None) == (
        "acquisition_metadata"
    ) else x_test
    scores = _aligned_scores(estimator, prediction_input, vocabulary)
    probabilities = apply_temperature(scores, calibration) if calibration else None
    return _prediction_frame(
        metadata=metadata,
        scores=scores,
        class_vocabulary=vocabulary,
        fit_id=fit_id,
        calibrated_probabilities=probabilities,
    )


def aggregate_seed_prediction_frames(
    prediction_frames: list[pd.DataFrame],
    *,
    model_id: str,
    aggregate_fit_id: str,
    class_vocabulary: list[str] | tuple[str, ...],
    calibration: TemperatureCalibration | None,
) -> pd.DataFrame:
    """Average declared forest seeds before calibration and class decisions."""

    if not prediction_frames:
        raise ValueError("Seed aggregation requires at least one prediction frame.")
    expected = 3 if model_id in STOCHASTIC_MODELS else 1
    if len(prediction_frames) != expected:
        raise ValueError(
            f"Model {model_id} requires {expected} completed technical-seed predictions."
        )
    keys = [
        "observation_uid",
        "master_sample_id",
        "instrument",
        "station",
        "true_label",
        "class_vocabulary",
    ]
    ordered = [
        frame.sort_values("observation_uid", kind="stable").reset_index(drop=True)
        for frame in prediction_frames
    ]
    reference = ordered[0][keys]
    if any(not reference.equals(frame[keys]) for frame in ordered[1:]):
        raise ValueError("Technical-seed prediction rows or metadata differ.")
    score_stack = np.stack(
        [
            np.vstack([np.asarray(json.loads(value), dtype=float) for value in frame.scores])
            for frame in ordered
        ]
    )
    vocabulary = tuple(str(value) for value in class_vocabulary)
    if score_stack.shape[2] != len(vocabulary) or not np.isfinite(score_stack).all():
        raise ValueError("Technical-seed scores differ from the frozen class vocabulary.")
    if model_id in STOCHASTIC_MODELS:
        mean_probability = np.mean(
            np.stack([softmax(seed_scores) for seed_scores in score_stack]), axis=0
        )
        aggregate_scores = np.log(np.clip(mean_probability, 1e-7, 1 - 1e-7))
    else:
        aggregate_scores = score_stack[0]
    probabilities = apply_temperature(aggregate_scores, calibration) if calibration else None
    metadata = reference.rename(columns={"true_label": "target_analyte"})
    result = _prediction_frame(
        metadata=metadata,
        scores=aggregate_scores,
        class_vocabulary=vocabulary,
        fit_id=aggregate_fit_id,
        calibrated_probabilities=probabilities,
    )
    result["technical_seed_count"] = len(prediction_frames)
    result["technical_seed_fit_id_sha256"] = sha256_value(
        sorted(str(frame.fit_id.iloc[0]) for frame in ordered)
    )
    return result


def fit_cross_fitted_temperature(
    validation_predictions: pd.DataFrame,
    *,
    model_id: str,
    class_vocabulary: list[str] | tuple[str, ...],
) -> CrossFittedCalibrationResult:
    """Aggregate technical seeds, then fit temperature to master-grouped OOF scores."""

    required = {
        "fit_id",
        "seed",
        "selection_unit_id",
        "observation_uid",
        "master_sample_id",
        "scores",
        "true_label",
    }
    if not required <= set(validation_predictions):
        raise ValueError(
            f"Calibration predictions miss fields: {sorted(required - set(validation_predictions))}"
        )
    if validation_predictions.empty:
        raise ValueError("Calibration predictions are empty.")
    expected_seeds = 3 if model_id in STOCHASTIC_MODELS else 1
    unit_frames: list[pd.DataFrame] = []
    evidence_fit_ids: set[str] = set()
    for unit_id, unit in validation_predictions.groupby("selection_unit_id", sort=True):
        seed_frames: list[pd.DataFrame] = []
        fit_groups = list(unit.groupby("fit_id", sort=True))
        if len(fit_groups) != expected_seeds:
            raise ValueError(
                f"Calibration unit {unit_id} lacks declared technical-seed predictions."
            )
        for fit_id, frame in fit_groups:
            if frame.seed.nunique(dropna=False) != 1:
                raise ValueError("One calibration fit ID contains multiple seed identities.")
            evidence_fit_ids.add(str(fit_id))
            seed_frames.append(frame)
        aggregate = aggregate_seed_prediction_frames(
            seed_frames,
            model_id=model_id,
            aggregate_fit_id=f"calibration:{unit_id}",
            class_vocabulary=class_vocabulary,
            calibration=None,
        )
        aggregate["selection_unit_id"] = unit_id
        unit_frames.append(aggregate)
    cross_fitted = pd.concat(unit_frames, ignore_index=True)
    if not cross_fitted.observation_uid.astype(str).is_unique:
        raise ValueError("A calibration observation appears in multiple cross-fit folds.")
    master_units = cross_fitted.groupby("master_sample_id").selection_unit_id.nunique()
    if not master_units.eq(1).all():
        raise ValueError("A calibration master appears in multiple cross-fit folds.")
    scores = np.vstack(
        [np.asarray(json.loads(value), dtype=float) for value in cross_fitted.scores]
    )
    calibration = fit_temperature(
        scores,
        cross_fitted.true_label.astype(str).to_numpy(),
        class_vocabulary=class_vocabulary,
        observation_uids=cross_fitted.observation_uid.astype(str).to_numpy(),
        master_ids=cross_fitted.master_sample_id.astype(str).to_numpy(),
    )
    return CrossFittedCalibrationResult(
        calibration=calibration,
        cross_fitted_predictions=cross_fitted,
        selection_unit_count=len(unit_frames),
        evidence_fit_id_sha256=sha256_value(sorted(evidence_fit_ids)),
    )
