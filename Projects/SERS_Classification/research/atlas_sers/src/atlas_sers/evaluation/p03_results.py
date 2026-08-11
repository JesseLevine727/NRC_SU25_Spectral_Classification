"""Normalize and validate protected P03 predictions against the public result schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

EMPTY_P03_PREDICTION_COLUMNS = [
    "protocol_version",
    "code_sha256",
    "config_sha256",
    "input_sha256",
    "experiment_id",
    "run_id",
    "research_question_id",
    "scope",
    "task_id",
    "information_regime",
    "preprocessing_information_regime",
    "target_access_role",
    "population_id",
    "preprocessing_policy_id",
    "representation_id",
    "preprocessing_policy_sha256",
    "preprocessing_action_representation_id",
    "preprocessing_fallback_reason",
    "preprocessing_policy_support_status",
    "preprocessing_gate_candidate_id",
    "model_id",
    "candidate_id",
    "procedure_id",
    "hyperparameter_sha256",
    "station",
    "domain",
    "outer_run_id",
    "outer_repeat",
    "outer_fold",
    "seed",
    "split_role",
    "observation_uid",
    "master_sample_id",
    "instrument",
    "instrument_family",
    "sensor_family",
    "target_analyte",
    "true_label",
    "predicted_label",
    "class_vocabulary",
    "scores",
    "logits",
    "probabilities",
    "probability_status",
    "run_status",
    "failure_reason",
]


@dataclass(frozen=True)
class P03ResultIdentity:
    run_id: str
    code_sha256: str
    config_sha256: str
    input_sha256: str
    preprocessing_policy_sha256: str


def _vector(value: Any) -> list[float] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parsed = json.loads(value) if isinstance(value, str) else value
    vector = np.asarray(parsed, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("A P03 prediction vector is invalid.")
    return vector.tolist()


def _vocabulary(value: Any) -> list[str]:
    parsed = json.loads(value) if isinstance(value, str) else value
    labels = [str(label) for label in parsed]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("A P03 prediction class vocabulary is invalid.")
    return labels


def _information_regime(task_id: str) -> str:
    if task_id.startswith("T1-"):
        return "closed_set"
    if task_id.startswith("T2-"):
        return "station_transfer"
    if task_id == "T3-ZS":
        return "zero_shot_source_only"
    raise ValueError(f"Unknown P03 task information regime: {task_id}")


def _single_research_question(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("A P03 result experiment lacks a research-question reference.")
    references = [item.strip() for item in value.split("|") if item.strip()]
    if len(references) != 1:
        raise ValueError(
            "Each P03 result experiment must reference exactly one research question."
        )
    return references[0]


def normalize_p03_predictions(
    predictions: pd.DataFrame,
    *,
    primary_manifest: pd.DataFrame,
    experiment_registry: pd.DataFrame,
    identity: P03ResultIdentity,
) -> pd.DataFrame:
    """Add the fixed scientific context and JSON-schema-compatible value types."""

    if predictions.empty:
        return pd.DataFrame(columns=EMPTY_P03_PREDICTION_COLUMNS)
    required = {
        "experiment_id",
        "task_id",
        "outer_run_id",
        "domain",
        "station",
        "outer_repeat",
        "outer_fold",
        "observation_uid",
        "master_sample_id",
        "instrument",
        "true_label",
        "predicted_label",
        "class_vocabulary",
        "scores",
        "probabilities",
        "model_id",
        "procedure_id",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"P03 final predictions miss schema fields: {sorted(missing)}")
    metadata_fields = [
        field
        for field in (
            "observation_uid",
            "instrument_family",
            "sensor_family",
            "sensor_variant",
            "quality_tier",
            "source_scan_id",
        )
        if field in primary_manifest
    ]
    metadata = primary_manifest[metadata_fields].copy()
    if "observation_uid" not in metadata or not metadata.observation_uid.astype(str).is_unique:
        raise ValueError("Primary manifest cannot uniquely enrich P03 prediction rows.")
    frame = predictions.copy()
    frame["observation_uid"] = frame.observation_uid.astype(str)
    metadata["observation_uid"] = metadata.observation_uid.astype(str)
    frame = frame.merge(metadata, on="observation_uid", how="left", validate="many_to_one")
    question_columns = [
        column
        for column in ("research_question_id", "research_question_ids")
        if column in experiment_registry
    ]
    if len(question_columns) != 1:
        raise ValueError(
            "Experiment registry must expose exactly one research-question column."
        )
    question_column = question_columns[0]
    registered_experiment_ids = set(frame.experiment_id.astype(str))
    registry = experiment_registry.loc[
        experiment_registry.experiment_id.astype(str).isin(registered_experiment_ids),
        ["experiment_id", "scope", question_column],
    ].copy()
    registry["research_question_id"] = registry[question_column].map(
        _single_research_question
    )
    if question_column != "research_question_id":
        registry = registry.drop(columns=question_column)
    if registry.experiment_id.astype(str).duplicated().any():
        raise ValueError("Experiment registry has duplicate experiment IDs.")
    frame = frame.merge(
        registry,
        on="experiment_id",
        how="left",
        validate="many_to_one",
    )
    if frame[["scope", "research_question_id"]].isna().any(axis=None):
        raise ValueError("A P03 prediction references an unregistered experiment.")
    frame["protocol_version"] = "atlas-sers-research-master-plan-v1"
    frame["run_id"] = identity.run_id
    frame["code_sha256"] = identity.code_sha256
    frame["config_sha256"] = identity.config_sha256
    frame["input_sha256"] = identity.input_sha256
    frame["information_regime"] = frame.task_id.astype(str).map(_information_regime)
    frame["preprocessing_information_regime"] = "fixed_source_only"
    frame["target_access_role"] = "evaluation_only"
    frame["population_id"] = "primary_598"
    frame["preprocessing_policy_id"] = "PP-U-MIN"
    frame["representation_id"] = "R_MIN_400_1800"
    frame["preprocessing_policy_sha256"] = identity.preprocessing_policy_sha256
    frame["preprocessing_action_representation_id"] = "R_MIN_400_1800"
    frame["preprocessing_fallback_reason"] = None
    frame["preprocessing_policy_support_status"] = "fixed_universal_primary"
    frame["preprocessing_gate_candidate_id"] = None
    frame["split_role"] = "test"
    frame["seed"] = None
    frame["run_status"] = "complete"
    frame["failure_reason"] = None
    frame["target_analyte"] = frame.true_label.astype(str)
    for field in ("instrument_family", "sensor_family"):
        if field not in frame:
            frame[field] = "unknown"
        frame[field] = frame[field].fillna("unknown").astype(str)
    if "sensor_variant" in frame:
        frame["sensor_variant"] = pd.Series(
            [None if pd.isna(value) else value for value in frame.sensor_variant],
            index=frame.index,
            dtype=object,
        )
    if "source_scan_id" in frame:
        frame["source_scan_id"] = pd.Series(
            [None if pd.isna(value) else value for value in frame.source_scan_id],
            index=frame.index,
            dtype=object,
        )
    if "quality_tier" in frame:
        frame["quality_tier"] = frame.quality_tier.fillna("unknown").astype(str)
    hyperparameter = frame.get(
        "hyperparameter_sha256", pd.Series(index=frame.index, dtype=object)
    )
    if "base_hyperparameter_sha256" in frame:
        hyperparameter = hyperparameter.fillna(frame.base_hyperparameter_sha256)
    frame["hyperparameter_sha256"] = hyperparameter
    if frame.hyperparameter_sha256.isna().any():
        raise ValueError("A P03 prediction lacks its selected hyperparameter hash.")
    frame["class_vocabulary"] = frame.class_vocabulary.map(_vocabulary)
    frame["logits"] = frame.scores.map(_vector)
    frame["scores"] = frame.logits
    frame["probabilities"] = frame.probabilities.map(_vector)
    for row in frame.itertuples(index=False):
        vocabulary = row.class_vocabulary
        if len(row.logits) != len(vocabulary):
            raise ValueError("P03 logits disagree with the class vocabulary.")
        if row.probabilities is not None and len(row.probabilities) != len(vocabulary):
            raise ValueError("P03 probabilities disagree with the class vocabulary.")
    return frame


def validate_p03_prediction_schema(
    predictions: pd.DataFrame, *, result_schema: dict[str, Any]
) -> dict[str, Any]:
    """Validate every protected prediction row, not a convenience sample."""

    validator = Draft202012Validator(result_schema)
    error_count = 0
    first_errors: list[str] = []
    for index, record in enumerate(predictions.to_dict(orient="records")):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        error_count += len(errors)
        for error in errors[: max(0, 20 - len(first_errors))]:
            path = ".".join(str(value) for value in error.path)
            first_errors.append(f"row={index} field={path or '<record>'}: {error.message}")
    report = {
        "schema_version": "p03-result-schema-validation-v1",
        "prediction_rows": len(predictions),
        "validated_rows": len(predictions),
        "schema_error_count": error_count,
        "first_errors": first_errors,
        "status": "pass" if error_count == 0 else "fail",
    }
    if error_count:
        raise ValueError(
            "P03 prediction result-schema validation failed: " + "; ".join(first_errors)
        )
    return report
