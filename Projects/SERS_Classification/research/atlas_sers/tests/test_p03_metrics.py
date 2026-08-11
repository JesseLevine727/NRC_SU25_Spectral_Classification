from __future__ import annotations

import json

import pandas as pd
import pytest

from atlas_sers.evaluation.p03_metrics import (
    build_expected_endpoint_registry,
    build_p03_metric_tables,
)
from atlas_sers.governance.canonical import sha256_value


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(4):
        for label, probabilities in (("a", [0.9, 0.1]), ("b", [0.1, 0.9])):
            rows.append(
                {
                    "experiment_id": "EXP-C03-T1",
                    "task_id": "T1-CWA",
                    "domain": "cwa:within",
                    "station": "cwa",
                    "procedure_id": "C-PCA-LDA-000",
                    "outer_repeat": 1,
                    "outer_fold": fold,
                    "outer_run_id": f"outer-{fold}",
                    "observation_uid": f"{label}-{fold}",
                    "master_sample_id": f"{label}-master-{fold}",
                    "instrument": f"unit-{fold % 2}",
                    "true_label": label,
                    "predicted_label": label,
                    "class_vocabulary": json.dumps(["a", "b"], separators=(",", ":")),
                    "scores": json.dumps(probabilities, separators=(",", ":")),
                    "probabilities": json.dumps(probabilities, separators=(",", ":")),
                    "probability_status": "cross_fitted_temperature",
                }
            )
    return pd.DataFrame(rows)


def test_metrics_pool_four_folds_before_spectrum_and_master_scoring() -> None:
    tables = build_p03_metric_tables(_predictions())
    assert len(tables.pooled_metrics) == 2
    assert set(tables.pooled_metrics.aggregation_level) == {
        "spectrum",
        "instrument_balanced_master",
    }
    assert tables.pooled_metrics.balanced_accuracy.eq(1.0).all()
    assert tables.pooled_metrics.fold_count.eq(4).all()
    assert len(tables.master_predictions) == 8
    assert tables.domain_summaries.empty


def test_metrics_refuse_sparse_or_duplicate_pooled_endpoint() -> None:
    predictions = _predictions()
    with pytest.raises(ValueError, match="all four folds"):
        build_p03_metric_tables(predictions[predictions.outer_fold != 3])
    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repeats an observation UID"):
        build_p03_metric_tables(duplicate)


def test_expected_endpoint_registry_preserves_missing_fold_as_unavailable() -> None:
    predictions = _predictions()
    expected = predictions[
        [
            "experiment_id",
            "task_id",
            "domain",
            "station",
            "procedure_id",
            "outer_repeat",
            "outer_fold",
            "outer_run_id",
        ]
    ].drop_duplicates()
    expected["expected_test_rows"] = 2
    expected["expected_test_masters"] = 2
    expected["expected_test_uid_sha256"] = expected.outer_fold.map(
        lambda fold: sha256_value(sorted([f"a-{fold}", f"b-{fold}"]))
    )
    incomplete = predictions[predictions.outer_fold != 3]
    tables = build_p03_metric_tables(
        incomplete,
        expected_endpoints=expected,
    )
    assert len(tables.pooled_metrics) == 2
    assert tables.pooled_metrics.endpoint_status.eq("unavailable").all()
    assert tables.pooled_metrics.expected_fold_count.eq(4).all()
    assert tables.pooled_metrics.fold_count.eq(3).all()
    assert tables.master_predictions.empty


def test_expected_endpoint_registry_expands_both_fixed_prior_procedures() -> None:
    expected_runs = pd.DataFrame(
        [
            {
                "outer_run_id": "prior-outer",
                "experiment_id": "EXP-C00-T1",
                "task_id": "T1-CWA",
                "domain": "cwa:within",
                "station": "cwa",
                "model_id": "C-PRIOR",
                "outer_repeat": 1,
                "outer_fold": 0,
                "execution_status": "planned",
            },
            {
                "outer_run_id": "selected-outer",
                "experiment_id": "EXP-C09-T3",
                "task_id": "T3-ZS",
                "domain": "cwa:held-unit",
                "station": "cwa",
                "model_id": "C-SELECTED",
                "outer_repeat": 1,
                "outer_fold": 0,
                "execution_status": "planned",
            },
        ]
    )
    fit_manifest = pd.DataFrame(
        [
            {
                "outer_run_id": outer_run_id,
                "test_rows": 2,
                "test_masters": 2,
                "test_uid_sha256": sha256_value([f"{outer_run_id}-a", f"{outer_run_id}-b"]),
            }
            for outer_run_id in ("prior-outer", "selected-outer")
        ]
    )
    candidate_registry = pd.DataFrame(
        [
            {"model_id": "C-PRIOR", "candidate_id": "C-PRIOR-000"},
            {"model_id": "C-PRIOR", "candidate_id": "C-PRIOR-001"},
            {"model_id": "C-PCA-LDA", "candidate_id": "C-PCA-LDA-000"},
        ]
    )
    endpoints = build_expected_endpoint_registry(
        expected_run_registry=expected_runs,
        fit_manifest=fit_manifest,
        candidate_registry=candidate_registry,
    )
    assert len(endpoints) == 3
    assert set(endpoints.procedure_id) == {
        "C-PRIOR:C-PRIOR-000",
        "C-PRIOR:C-PRIOR-001",
        "C-SELECTED",
    }


def test_expected_registry_can_make_all_failed_endpoints_explicit() -> None:
    expected = _predictions()[
        [
            "experiment_id",
            "task_id",
            "domain",
            "station",
            "procedure_id",
            "outer_repeat",
            "outer_fold",
            "outer_run_id",
        ]
    ].drop_duplicates()
    expected["expected_test_rows"] = 2
    expected["expected_test_masters"] = 2
    expected["expected_test_uid_sha256"] = expected.outer_fold.map(
        lambda fold: sha256_value(sorted([f"a-{fold}", f"b-{fold}"]))
    )
    tables = build_p03_metric_tables(pd.DataFrame(), expected_endpoints=expected)
    assert len(tables.pooled_metrics) == 2
    assert tables.pooled_metrics.endpoint_status.eq("unavailable").all()
    assert tables.pooled_metrics.fold_count.eq(0).all()
