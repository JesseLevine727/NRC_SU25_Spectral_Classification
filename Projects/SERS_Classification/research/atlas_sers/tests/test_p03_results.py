from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p03_metrics import build_p03_metric_tables
from atlas_sers.evaluation.p03_results import (
    P03ResultIdentity,
    normalize_p03_predictions,
    validate_p03_prediction_schema,
)
from tests.test_p03_metrics import _predictions

PROJECT = Path(__file__).resolve().parents[1]


def _normalized(experiments: pd.DataFrame | None = None) -> pd.DataFrame:
    predictions = _predictions().copy()
    predictions["model_id"] = "C-PCA-LDA"
    predictions["candidate_id"] = "C-PCA-LDA-000"
    predictions["hyperparameter_sha256"] = "e" * 64
    predictions["domain"] = "cwa:within"
    manifest = pd.DataFrame(
        {
            "observation_uid": predictions.observation_uid.unique(),
            "instrument_family": "family-1",
            "sensor_family": "sensor-1",
        }
    )
    if experiments is None:
        experiments = pd.DataFrame(
            [
                {
                    "experiment_id": "EXP-C03-T1",
                    "scope": "P",
                    "research_question_id": "RQ-P01",
                }
            ]
        )
    return normalize_p03_predictions(
        predictions,
        primary_manifest=manifest,
        experiment_registry=experiments,
        identity=P03ResultIdentity(
            run_id="P03-test",
            code_sha256="a" * 64,
            config_sha256="b" * 64,
            input_sha256="c" * 64,
            preprocessing_policy_sha256="d" * 64,
        ),
    )


def test_normalization_accepts_versioned_experiment_registry_schema() -> None:
    experiments = pd.read_csv(PROJECT / "plan" / "registries" / "experiment_registry.csv")

    assert "research_question_ids" in experiments
    normalized = _normalized(experiments)

    assert normalized.research_question_id.eq("RQ-P01").all()


def test_normalization_rejects_ambiguous_research_question_mapping() -> None:
    experiments = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-C03-T1",
                "scope": "P",
                "research_question_ids": "RQ-P01|RQ-E01",
            }
        ]
    )

    with pytest.raises(ValueError, match="exactly one research question"):
        _normalized(experiments)


def test_normalized_predictions_validate_every_row_against_result_schema() -> None:
    normalized = _normalized()
    schema = json.loads(
        (PROJECT / "plan" / "contracts" / "result_schema.json").read_text()
    )
    report = validate_p03_prediction_schema(normalized, result_schema=schema)
    assert report["status"] == "pass"
    assert report["validated_rows"] == len(normalized)
    assert isinstance(normalized.class_vocabulary.iloc[0], list)
    assert isinstance(normalized.probabilities.iloc[0], list)
    assert normalized.instrument_family.eq("family-1").all()


def test_normalized_list_vectors_still_reconstruct_identical_metrics() -> None:
    original = build_p03_metric_tables(_predictions()).pooled_metrics
    normalized = build_p03_metric_tables(_normalized()).pooled_metrics
    columns = ["aggregation_level", "balanced_accuracy", "macro_f1", "ece"]
    pd.testing.assert_frame_equal(
        original[columns].reset_index(drop=True),
        normalized[columns].reset_index(drop=True),
    )


def test_prediction_schema_rejects_a_missing_provenance_hash() -> None:
    normalized = _normalized()
    normalized.loc[0, "code_sha256"] = "bad"
    schema = json.loads(
        (PROJECT / "plan" / "contracts" / "result_schema.json").read_text()
    )
    with pytest.raises(ValueError, match="result-schema validation failed"):
        validate_p03_prediction_schema(normalized, result_schema=schema)


def test_optional_manifest_nulls_are_json_schema_compatible() -> None:
    predictions = _predictions().copy()
    predictions["model_id"] = "C-PCA-LDA"
    predictions["candidate_id"] = "C-PCA-LDA-000"
    predictions["hyperparameter_sha256"] = "e" * 64
    predictions["domain"] = "cwa:within"
    observation_uids = predictions.observation_uid.unique()
    manifest = pd.DataFrame(
        {
            "observation_uid": observation_uids,
            "instrument_family": "family-1",
            "sensor_family": "sensor-1",
            "sensor_variant": [np.nan, *("variant-a" for _ in observation_uids[1:])],
            "source_scan_id": [np.nan, *(7 for _ in observation_uids[1:])],
            "quality_tier": [np.nan, *("clear" for _ in observation_uids[1:])],
        }
    )
    experiments = pd.DataFrame(
        [{"experiment_id": "EXP-C03-T1", "scope": "P", "research_question_id": "RQ-P01"}]
    )
    normalized = normalize_p03_predictions(
        predictions,
        primary_manifest=manifest,
        experiment_registry=experiments,
        identity=P03ResultIdentity(
            run_id="P03-test",
            code_sha256="a" * 64,
            config_sha256="b" * 64,
            input_sha256="c" * 64,
            preprocessing_policy_sha256="d" * 64,
        ),
    )
    schema = json.loads(
        (PROJECT / "plan" / "contracts" / "result_schema.json").read_text()
    )

    assert normalized.loc[0, "sensor_variant"] is None
    assert normalized.loc[0, "source_scan_id"] is None
    assert normalized.loc[0, "quality_tier"] == "unknown"
    assert validate_p03_prediction_schema(normalized, result_schema=schema)[
        "status"
    ] == "pass"
