"""Reconstructable spectrum/master metrics and pooled P03 endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.classical import (
    classification_metrics,
    instrument_balanced_master_probabilities,
    softmax,
)
from atlas_sers.governance.canonical import sha256_value

IDENTITY_COLUMNS = (
    "experiment_id",
    "task_id",
    "domain",
    "station",
    "procedure_id",
    "outer_repeat",
)


@dataclass(frozen=True)
class P03MetricTables:
    pooled_metrics: pd.DataFrame
    master_predictions: pd.DataFrame
    domain_summaries: pd.DataFrame


def build_expected_endpoint_registry(
    *,
    expected_run_registry: pd.DataFrame,
    fit_manifest: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    class_vocabulary: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Enumerate every final procedure/fold endpoint before inspecting predictions."""

    runs = expected_run_registry[
        ~expected_run_registry.execution_status.astype(str).eq(
            "manifest_only_exploratory"
        )
    ]
    prior_candidates = candidate_registry[
        candidate_registry.model_id.astype(str).eq("C-PRIOR")
    ].candidate_id.astype(str).tolist()
    planned_by_outer = {
        str(outer_run_id): group
        for outer_run_id, group in fit_manifest.groupby(
            "outer_run_id", sort=False, dropna=False
        )
    }
    rows: list[dict[str, Any]] = []
    for run in runs.itertuples(index=False):
        planned = planned_by_outer.get(str(run.outer_run_id), pd.DataFrame())
        if planned.empty or planned.test_uid_sha256.astype(str).nunique() != 1:
            raise ValueError("Expected outer endpoint has no unique planned test role.")
        procedures: list[str]
        if str(run.experiment_id) == "EXP-C00-T1":
            procedures = [f"C-PRIOR:{candidate}" for candidate in prior_candidates]
        elif str(run.experiment_id) == "EXP-C09-CONTROL-PERM":
            procedures = [
                f"C-PERMUTED-SELECTED:{str(run.control_candidate_id)}"
            ]
        elif str(run.experiment_id) == "EXP-C09-CONTROL-PRIOR":
            procedures = [f"C-PRIOR:{str(run.control_candidate_id)}"]
        else:
            procedures = [str(run.model_id)]
        for procedure_id in procedures:
            vocabulary_key = (
                str(run.task_id)
                if str(run.task_id).startswith("T2-")
                else str(run.station)
            )
            rows.append(
                {
                    "experiment_id": str(run.experiment_id),
                    "task_id": str(run.task_id),
                    "domain": str(run.domain),
                    "station": str(run.station),
                    "procedure_id": procedure_id,
                    "outer_repeat": int(run.outer_repeat),
                    "outer_fold": int(run.outer_fold),
                    "outer_run_id": str(run.outer_run_id),
                    "expected_test_rows": int(planned.test_rows.iloc[0]),
                    "expected_test_masters": int(planned.test_masters.iloc[0]),
                    "expected_test_uid_sha256": str(planned.test_uid_sha256.iloc[0]),
                    "class_vocabulary_json": (
                        json.dumps(
                            class_vocabulary[vocabulary_key], separators=(",", ":")
                        )
                        if class_vocabulary is not None
                        else None
                    ),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        [*IDENTITY_COLUMNS, "outer_fold"], kind="stable"
    ).reset_index(drop=True)
    if frame.empty or frame.duplicated(["outer_run_id", "procedure_id"]).any():
        raise RuntimeError("P03 expected final-endpoint registry is incomplete.")
    return frame


def _json_matrix(values: pd.Series, *, field: str) -> np.ndarray:
    try:
        matrix = np.vstack(
            [
                np.asarray(
                    json.loads(value) if isinstance(value, str) else value,
                    dtype=float,
                )
                for value in values
            ]
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Prediction {field} cannot be parsed.") from error
    if not np.isfinite(matrix).all():
        raise ValueError(f"Prediction {field} contains a nonfinite value.")
    return matrix


def _vocabulary(frame: pd.DataFrame) -> tuple[str, ...]:
    values = [
        tuple(
            str(label)
            for label in (
                json.loads(value) if isinstance(value, str) else value
            )
        )
        for value in frame.class_vocabulary
    ]
    unique = set(values)
    if len(unique) != 1:
        raise ValueError("A pooled prediction group contains multiple class vocabularies.")
    vocabulary = next(iter(unique))
    if len(vocabulary) < 2 or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("A pooled prediction group has an invalid class vocabulary.")
    return vocabulary


def _probabilities(frame: pd.DataFrame) -> np.ndarray | None:
    calibrated = frame.probability_status.eq("cross_fitted_temperature")
    if calibrated.all():
        return _json_matrix(frame.probabilities, field="probabilities")
    if calibrated.any():
        raise ValueError("A pooled group mixes calibrated and uncalibrated rows.")
    return None


def _metric_record(
    metrics: dict[str, Any],
    *,
    aggregation_level: str,
    observations: int,
    masters: int,
    fold_count: int,
    uid_sha256: str,
    class_vocabulary: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "endpoint_status": "complete",
        "aggregation_level": aggregation_level,
        "observations": observations,
        "masters": masters,
        "fold_count": fold_count,
        "observation_uid_sha256": uid_sha256,
        "class_vocabulary_json": json.dumps(
            class_vocabulary, separators=(",", ":")
        ),
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "negative_log_likelihood": metrics["negative_log_likelihood"],
        "brier_score": metrics["brier_score"],
        "ece": metrics["ece"],
        "per_class_recall_json": json.dumps(
            metrics["per_class_recall"], separators=(",", ":"), sort_keys=True
        ),
        "confusion_matrix_json": json.dumps(
            metrics["confusion_matrix"], separators=(",", ":")
        ),
        "support_json": json.dumps(metrics["support"], separators=(",", ":"), sort_keys=True),
    }


def _unavailable_metric_record(
    *,
    aggregation_level: str,
    expected_observations: int,
    expected_masters: int,
    expected_fold_count: int,
    observed_fold_count: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "endpoint_status": "unavailable",
        "endpoint_reason": reason,
        "aggregation_level": aggregation_level,
        "observations": 0,
        "masters": 0,
        "expected_observations": expected_observations,
        "expected_masters": expected_masters,
        "fold_count": observed_fold_count,
        "expected_fold_count": expected_fold_count,
        "observation_uid_sha256": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "negative_log_likelihood": None,
        "brier_score": None,
        "ece": None,
        "per_class_recall_json": None,
        "confusion_matrix_json": None,
        "support_json": None,
    }


def _pooling_keys(frame: pd.DataFrame) -> list[str]:
    missing = set(IDENTITY_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Final predictions miss identity fields: {sorted(missing)}")
    return list(IDENTITY_COLUMNS)


def build_p03_metric_tables(
    final_predictions: pd.DataFrame,
    *,
    expected_endpoints: pd.DataFrame | None = None,
) -> P03MetricTables:
    """Pool outer folds first, then compute spectrum and instrument-balanced master metrics."""

    required = {
        "outer_fold",
        "observation_uid",
        "master_sample_id",
        "instrument",
        "true_label",
        "predicted_label",
        "class_vocabulary",
        "scores",
        "probabilities",
        "probability_status",
    }
    identity_and_required = {*IDENTITY_COLUMNS, "outer_run_id", *required}
    if final_predictions.empty and expected_endpoints is not None:
        final_predictions = pd.DataFrame(columns=sorted(identity_and_required))
    if not required <= set(final_predictions):
        raise ValueError(
            f"Final predictions miss fields: {sorted(required - set(final_predictions))}"
        )
    if final_predictions.empty and expected_endpoints is None:
        raise ValueError("Final predictions are empty.")
    metric_rows: list[dict[str, Any]] = []
    master_frames: list[pd.DataFrame] = []
    keys = list(IDENTITY_COLUMNS) if expected_endpoints is not None else _pooling_keys(
        final_predictions
    )
    if expected_endpoints is not None:
        missing_expected = set([*keys, "outer_fold", "outer_run_id"]) - set(
            expected_endpoints
        )
        if missing_expected:
            raise ValueError(
                f"Expected endpoints miss fields: {sorted(missing_expected)}"
            )
        def grouped_by_identity(frame: pd.DataFrame) -> dict[tuple[Any, ...], pd.DataFrame]:
            return {
                identity if isinstance(identity, tuple) else (identity,): group
                for identity, group in frame.groupby(keys, sort=True, dropna=False)
            }

        expected_groups = grouped_by_identity(expected_endpoints)
        observed_groups = grouped_by_identity(final_predictions)
        if set(observed_groups) - set(expected_groups):
            raise ValueError("Final predictions contain an unplanned endpoint identity.")
        grouped = [
            (
                identity,
                observed_groups.get(
                    identity, pd.DataFrame(columns=final_predictions.columns)
                ),
                expected_group,
            )
            for identity, expected_group in expected_groups.items()
        ]
    else:
        grouped = [
            (identity, pooled, None)
            for identity, pooled in final_predictions.groupby(
                keys, sort=True, dropna=False
            )
        ]
    for identity, pooled, expected_group in grouped:
        if not isinstance(identity, tuple):
            identity = (identity,)
        context = dict(zip(keys, identity, strict=True))
        task_id = str(context["task_id"])
        folds = sorted(pooled.outer_fold.astype(int).unique())
        expected_folds = [0, 1, 2, 3] if (
            task_id.startswith("T1-") or task_id == "T3-ZS"
        ) else [-1]
        unavailable_reason = None
        if expected_group is not None:
            observed_outer_ids = set(pooled.outer_run_id.astype(str))
            expected_outer_ids = set(expected_group.outer_run_id.astype(str))
            if observed_outer_ids - expected_outer_ids:
                raise ValueError("An endpoint prediction references an unexpected outer run.")
            for expected in expected_group.itertuples(index=False):
                outer = pooled[
                    pooled.outer_run_id.astype(str).eq(str(expected.outer_run_id))
                ]
                if outer.empty:
                    continue
                if len(outer) != int(expected.expected_test_rows):
                    raise ValueError("An endpoint outer prediction has the wrong row count.")
                if sha256_value(sorted(outer.observation_uid.astype(str))) != str(
                    expected.expected_test_uid_sha256
                ):
                    raise ValueError("An endpoint outer prediction has the wrong UID set.")
            if not pooled.observation_uid.astype(str).is_unique:
                raise ValueError(f"Pooled endpoint {context} repeats an observation UID.")
            if observed_outer_ids != expected_outer_ids or folds != expected_folds:
                unavailable_reason = "one_or_more_outer_folds_have_no_final_predictions"
        if unavailable_reason is not None:
            expected_observations = int(expected_group.expected_test_rows.sum())
            expected_masters = int(expected_group.expected_test_masters.sum())
            expected_vocabulary = None
            if "class_vocabulary_json" in expected_group:
                vocabularies = expected_group.class_vocabulary_json.dropna().astype(str).unique()
                if len(vocabularies) > 1:
                    raise ValueError("Expected endpoint group has multiple class vocabularies.")
                if len(vocabularies) == 1:
                    expected_vocabulary = vocabularies[0]
            for level in ("spectrum", "instrument_balanced_master"):
                metric_rows.append(
                    {
                        **context,
                        "class_vocabulary_json": expected_vocabulary,
                        **_unavailable_metric_record(
                            aggregation_level=level,
                            expected_observations=expected_observations,
                            expected_masters=expected_masters,
                            expected_fold_count=len(expected_folds),
                            observed_fold_count=len(folds),
                            reason=unavailable_reason,
                        ),
                    }
                )
            continue
        if task_id.startswith("T1-") or task_id == "T3-ZS":
            if folds != [0, 1, 2, 3]:
                raise ValueError(f"Pooled endpoint {context} does not contain all four folds.")
        if not pooled.observation_uid.astype(str).is_unique:
            raise ValueError(f"Pooled endpoint {context} repeats an observation UID.")
        vocabulary = _vocabulary(pooled)
        probabilities = _probabilities(pooled)
        spectrum_metrics = classification_metrics(
            pooled.true_label.astype(str).to_numpy(),
            pooled.predicted_label.astype(str).to_numpy(),
            class_vocabulary=vocabulary,
            probabilities=probabilities,
        )
        uid_hash = sha256_value(sorted(pooled.observation_uid.astype(str)))
        metric_rows.append(
            {
                **context,
                **_metric_record(
                    spectrum_metrics,
                    aggregation_level="spectrum",
                    observations=len(pooled),
                    masters=pooled.master_sample_id.astype(str).nunique(),
                    fold_count=len(folds),
                    uid_sha256=uid_hash,
                    class_vocabulary=vocabulary,
                ),
                "expected_observations": len(pooled),
                "expected_masters": pooled.master_sample_id.astype(str).nunique(),
                "expected_fold_count": len(expected_folds),
                "endpoint_reason": None,
            }
        )
        aggregation_values = probabilities
        if aggregation_values is None:
            aggregation_values = softmax(_json_matrix(pooled.scores, field="scores"))
        master = instrument_balanced_master_probabilities(
            probabilities=aggregation_values,
            true_labels=pooled.true_label.astype(str).to_numpy(),
            master_ids=pooled.master_sample_id.astype(str).to_numpy(),
            instruments=pooled.instrument.astype(str).to_numpy(),
            class_vocabulary=vocabulary,
        )
        master_probability = None
        if probabilities is not None:
            master_probability = np.vstack(master.probabilities)
        master_metrics = classification_metrics(
            master.true_label.astype(str).to_numpy(),
            master.predicted_label.astype(str).to_numpy(),
            class_vocabulary=vocabulary,
            probabilities=master_probability,
        )
        metric_rows.append(
            {
                **context,
                **_metric_record(
                    master_metrics,
                    aggregation_level="instrument_balanced_master",
                    observations=len(pooled),
                    masters=len(master),
                    fold_count=len(folds),
                    uid_sha256=uid_hash,
                    class_vocabulary=vocabulary,
                ),
                "expected_observations": len(pooled),
                "expected_masters": len(master),
                "expected_fold_count": len(expected_folds),
                "endpoint_reason": None,
            }
        )
        for field, value in context.items():
            master[field] = value
        master["class_vocabulary"] = json.dumps(vocabulary, separators=(",", ":"))
        master["probability_status"] = (
            "cross_fitted_temperature" if probabilities is not None else "uncalibrated"
        )
        master_frames.append(master)
    metrics = pd.DataFrame(metric_rows)
    masters = (
        pd.concat(master_frames, ignore_index=True)
        if master_frames
        else pd.DataFrame()
    )
    t3 = metrics[metrics.task_id.eq("T3-ZS")]
    summaries: list[dict[str, Any]] = []
    summary_keys = ["experiment_id", "procedure_id", "outer_repeat", "aggregation_level"]
    for identity, group in t3.groupby(summary_keys, sort=True, dropna=False):
        if group.domain.nunique() != 13:
            raise ValueError("A T3 domain summary does not contain all 13 primary domains.")
        complete = group[group.endpoint_status.eq("complete")]
        summaries.append(
            {
                **dict(zip(summary_keys, identity, strict=True)),
                "domain_count": group.domain.nunique(),
                "complete_domain_count": complete.domain.nunique(),
                "summary_status": "complete"
                if complete.domain.nunique() == 13
                else "incomplete_terminal_cells",
                "mean_domain_balanced_accuracy": complete.balanced_accuracy.mean(),
                "worst_domain_balanced_accuracy": complete.balanced_accuracy.min(),
                "mean_domain_macro_f1": complete.macro_f1.mean(),
            }
        )
    return P03MetricTables(metrics, masters, pd.DataFrame(summaries))
