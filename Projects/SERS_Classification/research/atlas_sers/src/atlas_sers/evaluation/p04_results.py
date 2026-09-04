"""Aggregation, reconciliation, and safe summaries for P04 D0 predictions."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from atlas_sers.evaluation.classical import (
    classification_metrics,
    instrument_balanced_master_probabilities,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.splits.p02 import instrument_family


def normalize_p04_predictions(
    predictions: pd.DataFrame,
    *,
    contexts: pd.DataFrame,
    manifest: pd.DataFrame,
    fit_status: pd.DataFrame,
    identity: dict[str, str],
    ensemble: bool,
) -> pd.DataFrame:
    """Add the frozen result-schema provenance to private P04 prediction rows."""

    if predictions.empty:
        return predictions.copy()
    metadata = manifest.copy()
    if "instrument_family" not in metadata:
        metadata["instrument_family"] = metadata.instrument.astype(str).map(instrument_family)
    context_fields = contexts[["context_id", "task_id", "selection_mode", "outer_fit_uid_sha256"]]
    frame = predictions.merge(
        context_fields,
        on="context_id",
        how="left",
        validate="many_to_one",
    )
    metadata_fields = [
        field
        for field in (
            "observation_uid",
            "instrument_family",
            "sensor_family",
            "sensor_variant",
            "source_scan_id",
            "quality_tier",
        )
        if field in metadata
    ]
    frame = frame.merge(
        metadata[metadata_fields],
        on="observation_uid",
        how="left",
        validate="many_to_one",
    )
    final = fit_status[fit_status.stage.eq("final_selected_refit")][
        ["context_id", "seed", "hyperparameter_sha256", "best_epoch", "status"]
    ].rename(columns={"best_epoch": "selected_epoch", "status": "fit_run_status"})
    if ensemble:
        procedure = final.groupby("context_id", as_index=False).agg(
            hyperparameter_sha256=("hyperparameter_sha256", "first"),
            selected_epoch=("selected_epoch", "median"),
            complete_seed_count=("fit_run_status", lambda values: values.eq("complete").sum()),
        )
        procedure["selected_epoch"] = procedure.selected_epoch.round().astype(int)
        frame = frame.merge(procedure, on="context_id", validate="many_to_one")
        frame["seed"] = None
        frame["logits"] = None
    else:
        frame = frame.merge(
            final,
            on=["context_id", "seed"],
            validate="many_to_one",
        )
        frame["selected_epoch"] = frame.selected_epoch.astype(int)
        frame["logits"] = frame.apply(
            lambda row: [float(row[f"logit_{index}"]) for index in range(3)], axis=1
        )
    frame["class_vocabulary"] = frame.class_vocabulary.map(
        lambda value: json.loads(value) if isinstance(value, str) else list(value)
    )
    frame["probabilities"] = frame.apply(
        lambda row: [float(row[f"probability_{index}"]) for index in range(3)], axis=1
    )
    frame["protocol_version"] = "atlas-sers-research-master-plan-v1"
    frame["code_sha256"] = identity["code_sha256"]
    frame["config_sha256"] = identity["config_sha256"]
    frame["input_sha256"] = identity["input_sha256"]
    frame["run_id"] = identity["run_id"]
    frame["research_question_id"] = "RQ-P01"
    frame["scope"] = "P"
    frame["information_regime"] = frame.experiment_id.map(
        {"EXP-N00-DEV": "closed_set", "EXP-N00-T3": "zero_shot_source_only"}
    )
    frame["preprocessing_information_regime"] = "fixed_source_only"
    frame["target_access_role"] = "evaluation_only"
    frame["population_id"] = "primary_598"
    frame["preprocessing_policy_id"] = "PP-U-MIN"
    frame["representation_id"] = "R_MIN_400_1800"
    frame["preprocessing_policy_sha256"] = identity["preprocessing_policy_sha256"]
    frame["preprocessing_action_representation_id"] = "R_MIN_400_1800"
    frame["preprocessing_fallback_reason"] = None
    frame["preprocessing_policy_support_status"] = "fixed_universal_primary"
    frame["preprocessing_gate_candidate_id"] = None
    frame["model_id"] = "D0-ERM"
    frame["split_role"] = "test"
    frame["target_analyte"] = frame.true_label.astype(str)
    frame["run_status"] = "complete"
    frame["failure_reason"] = None
    return frame


def validate_prediction_schema(
    predictions: pd.DataFrame, result_schema: dict[str, Any]
) -> dict[str, Any]:
    validator = Draft202012Validator(result_schema)
    errors = []
    error_count = 0
    required = list(result_schema["required"])
    for row_index, row in enumerate(predictions.to_dict("records")):
        record = {field: row.get(field) for field in required}
        record["selected_epoch"] = row.get("selected_epoch")
        record["logits"] = row.get("logits")
        for key, value in list(record.items()):
            if isinstance(value, np.generic):
                record[key] = value.item()
            elif pd.isna(value) if not isinstance(value, (list, dict)) else False:
                record[key] = None
        for error in validator.iter_errors(record):
            error_count += 1
            if len(errors) < 10:
                field = ".".join(str(part) for part in error.absolute_path)
                errors.append(f"row {row_index}, field {field}: {error.message}")
    return {
        "status": "pass" if not errors else "fail",
        "rows_validated": len(predictions),
        "error_count": error_count,
        "first_errors": errors,
    }


def _classes(value: str | list[str]) -> tuple[str, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    classes = tuple(str(item) for item in parsed)
    if len(classes) != 3 or len(set(classes)) != 3:
        raise ValueError("P04 predictions require three unique station-local classes.")
    return classes


def ensemble_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average three calibrated neural seeds before any endpoint metric."""

    if predictions.empty:
        return predictions.copy()
    keys = [
        "context_id",
        "experiment_id",
        "domain",
        "held_instrument",
        "outer_repeat",
        "outer_fold",
        "observation_uid",
        "master_sample_id",
        "instrument",
        "station",
        "true_label",
        "class_vocabulary",
        "candidate_id",
    ]
    probability_columns = ["probability_0", "probability_1", "probability_2"]
    if not set(keys + probability_columns + ["seed"]) <= set(predictions):
        raise ValueError("P04 test prediction table is incomplete.")
    if predictions.duplicated(["context_id", "observation_uid", "seed"]).any():
        raise ValueError("P04 predictions contain a duplicate observation/seed row.")
    probability_values = predictions[probability_columns].to_numpy(dtype=float)
    if (
        not np.isfinite(probability_values).all()
        or (probability_values < 0).any()
        or (probability_values > 1).any()
        or not np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("P04 seed probabilities must be finite and normalized.")
    counts = predictions.groupby(keys, dropna=False).seed.nunique()
    if not counts.eq(3).all():
        raise ValueError("P04 ensemble rows do not all contain three distinct neural seeds.")
    ensemble = (
        predictions.groupby(keys, as_index=False, dropna=False)[probability_columns]
        .mean()
        .sort_values(
            ["experiment_id", "domain", "outer_repeat", "outer_fold", "observation_uid"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    labels = []
    for row in ensemble.itertuples(index=False):
        classes = _classes(row.class_vocabulary)
        probabilities = np.asarray(
            [row.probability_0, row.probability_1, row.probability_2], dtype=float
        )
        labels.append(classes[int(probabilities.argmax())])
    ensemble["predicted_label"] = labels
    ensemble["seed_count"] = 3
    return ensemble


def endpoint_metrics(ensemble: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectrum_rows: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []
    group_columns = [
        "context_id",
        "experiment_id",
        "domain",
        "held_instrument",
        "outer_repeat",
        "outer_fold",
        "station",
        "candidate_id",
    ]
    for keys, cell in ensemble.groupby(group_columns, dropna=False, sort=True):
        identity = dict(zip(group_columns, keys, strict=True))
        classes = _classes(cell.class_vocabulary.iloc[0])
        probabilities = cell[["probability_0", "probability_1", "probability_2"]].to_numpy()
        metrics = classification_metrics(
            cell.true_label.astype(str).to_numpy(),
            cell.predicted_label.astype(str).to_numpy(),
            class_vocabulary=classes,
            probabilities=probabilities,
        )
        spectrum_rows.append(
            {
                **identity,
                "aggregation_id": "M01",
                "observations": len(cell),
                "physical_masters": cell.master_sample_id.nunique(),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "negative_log_likelihood": metrics["negative_log_likelihood"],
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
                "per_class_recall": json.dumps(
                    metrics["per_class_recall"], sort_keys=True, separators=(",", ":")
                ),
                "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
                "test_uid_sha256": sha256_value(sorted(cell.observation_uid.astype(str).tolist())),
            }
        )
        master = instrument_balanced_master_probabilities(
            probabilities=probabilities,
            true_labels=cell.true_label.astype(str).to_numpy(),
            master_ids=cell.master_sample_id.astype(str).to_numpy(),
            instruments=cell.instrument.astype(str).to_numpy(),
            class_vocabulary=classes,
        )
        master_probabilities = np.asarray(master.probabilities.tolist(), dtype=float)
        master_metrics = classification_metrics(
            master.true_label.astype(str).to_numpy(),
            master.predicted_label.astype(str).to_numpy(),
            class_vocabulary=classes,
            probabilities=master_probabilities,
        )
        master_rows.append(
            {
                **identity,
                "aggregation_id": "M06",
                "observations": len(master),
                "physical_masters": len(master),
                "balanced_accuracy": master_metrics["balanced_accuracy"],
                "macro_f1": master_metrics["macro_f1"],
                "negative_log_likelihood": master_metrics["negative_log_likelihood"],
                "brier_score": master_metrics["brier_score"],
                "ece": master_metrics["ece"],
                "per_class_recall": json.dumps(
                    master_metrics["per_class_recall"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "confusion_matrix": json.dumps(
                    master_metrics["confusion_matrix"], separators=(",", ":")
                ),
                "test_uid_sha256": sha256_value(sorted(cell.observation_uid.astype(str).tolist())),
            }
        )
    return pd.DataFrame(spectrum_rows), pd.DataFrame(master_rows)


def endpoint_coverage(
    expected: pd.DataFrame,
    fit_status: pd.DataFrame,
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    finals = fit_status[fit_status.stage.eq("final_selected_refit")]
    records = []
    for row in expected.itertuples(index=False):
        status = finals[finals.context_id.astype(str).eq(str(row.context_id))]
        predictions = ensemble[ensemble.context_id.astype(str).eq(str(row.context_id))]
        complete_seeds = int(status.status.eq("complete").sum())
        observed_uid_hash = (
            sha256_value(sorted(predictions.observation_uid.astype(str).tolist()))
            if len(predictions)
            else sha256_value([])
        )
        complete = (
            complete_seeds == int(row.expected_seed_count)
            and len(predictions) == int(row.expected_test_rows)
            and observed_uid_hash == str(row.expected_test_uid_sha256)
        )
        records.append(
            {
                "context_id": str(row.context_id),
                "experiment_id": str(row.experiment_id),
                "domain": str(row.domain),
                "station": str(row.station),
                "held_instrument": str(row.held_instrument),
                "outer_repeat": int(row.outer_repeat),
                "outer_fold": int(row.outer_fold),
                "expected_test_rows": int(row.expected_test_rows),
                "observed_test_rows": len(predictions),
                "expected_seed_count": int(row.expected_seed_count),
                "complete_seed_count": complete_seeds,
                "status": "complete" if complete else "terminal_failure",
                "reason_code": "complete" if complete else "missing_seed_or_prediction",
                "expected_test_uid_sha256": str(row.expected_test_uid_sha256),
                "observed_test_uid_sha256": observed_uid_hash,
            }
        )
    return pd.DataFrame(records)


def summarize_endpoints(metrics: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    joined = coverage.merge(
        metrics[
            [
                "context_id",
                "aggregation_id",
                "balanced_accuracy",
                "macro_f1",
                "negative_log_likelihood",
                "brier_score",
                "ece",
            ]
        ],
        on="context_id",
        how="left",
        validate="one_to_many",
    )
    rows = []
    groups = ["experiment_id", "aggregation_id", "station"]
    for keys, cell in joined.groupby(groups, dropna=False, sort=True):
        complete = cell[cell.status.eq("complete")]
        records = {
            "experiment_id": keys[0],
            "aggregation_id": keys[1],
            "station": keys[2],
            "planned_endpoints": len(cell),
            "complete_endpoints": len(complete),
            "endpoint_coverage": len(complete) / len(cell),
            "mean_balanced_accuracy_success": float(complete.balanced_accuracy.mean()),
            "worst_balanced_accuracy_success": float(complete.balanced_accuracy.min()),
            "failure_sensitive_mean_balanced_accuracy": float(
                cell.balanced_accuracy.fillna(0.0).mean()
            ),
            "mean_macro_f1_success": float(complete.macro_f1.mean()),
            "mean_negative_log_likelihood_success": float(complete.negative_log_likelihood.mean()),
            "mean_ece_success": float(complete.ece.mean()),
        }
        rows.append(records)
    return pd.DataFrame(rows)


def learning_curve_summary(
    histories: pd.DataFrame,
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    winners = selections[selections.selected][["context_id", "candidate_id"]].drop_duplicates()
    selected = histories[histories.stage.eq("inner_selection")].merge(
        winners,
        on=["context_id", "candidate_id"],
        how="inner",
        validate="many_to_one",
    )
    grouped = selected.groupby(["experiment_id", "epoch"], as_index=False).agg(
        run_count=("fit_id", "nunique"),
        training_balanced_accuracy_mean=("training_balanced_accuracy", "mean"),
        training_balanced_accuracy_q25=(
            "training_balanced_accuracy",
            lambda values: values.quantile(0.25),
        ),
        training_balanced_accuracy_q75=(
            "training_balanced_accuracy",
            lambda values: values.quantile(0.75),
        ),
        validation_balanced_accuracy_mean=("validation_balanced_accuracy", "mean"),
        validation_balanced_accuracy_q25=(
            "validation_balanced_accuracy",
            lambda values: values.quantile(0.25),
        ),
        validation_balanced_accuracy_q75=(
            "validation_balanced_accuracy",
            lambda values: values.quantile(0.75),
        ),
    )
    selected_epochs = (
        selected.merge(
            selected.groupby("fit_id", as_index=False)
            .validation_balanced_accuracy.max()
            .rename(columns={"validation_balanced_accuracy": "maximum_validation_ba"}),
            on="fit_id",
            validate="many_to_one",
        )
        .query("validation_balanced_accuracy == maximum_validation_ba")
        .sort_values(["fit_id", "validation_negative_log_likelihood", "epoch"])
        .drop_duplicates("fit_id")
    )
    epoch_summary = selected_epochs.groupby("experiment_id", as_index=False).agg(
        fits=("fit_id", "nunique"),
        selected_epoch_median=("epoch", "median"),
        selected_epoch_q25=("epoch", lambda values: values.quantile(0.25)),
        selected_epoch_q75=("epoch", lambda values: values.quantile(0.75)),
        selected_epoch_minimum=("epoch", "min"),
        selected_epoch_maximum=("epoch", "max"),
    )
    return grouped, epoch_summary
