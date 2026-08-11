"""Fail-closed, resumable execution of independent P03 selection fits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from atlas_sers.evaluation.p03_controls import metadata_control_candidate_registry
from atlas_sers.evaluation.p03_roles import resolve_fit_roles
from atlas_sers.evaluation.p03_runtime import P03Dataset, run_candidate_fit
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore

SELECTION_STAGES = {"inner_selection", "training_station_inner_selection"}
CONTEXT_COLUMNS = (
    "experiment_id",
    "task_id",
    "outer_run_id",
    "domain",
    "station",
    "held_instrument",
    "outer_repeat",
    "outer_fold",
    "selection_mode",
    "stage",
    "selection_unit_id",
)


@dataclass(frozen=True)
class SelectionShardResult:
    fit_status: pd.DataFrame
    validation_predictions: pd.DataFrame
    selection_unit_metrics: pd.DataFrame


def _candidate_lookup(candidate_registry: pd.DataFrame) -> dict[str, pd.Series]:
    required = {
        "candidate_id",
        "model_id",
        "parameters_json",
        "hyperparameter_sha256",
    }
    if not required <= set(candidate_registry):
        raise ValueError(
            f"Candidate registry misses fields: {sorted(required - set(candidate_registry))}"
        )
    if not candidate_registry.candidate_id.astype(str).is_unique:
        raise ValueError("Candidate registry IDs are not unique.")
    return {
        str(row.candidate_id): row for _, row in candidate_registry.iterrows()
    }


def _vocabulary(fit_row: pd.Series, contract: dict[str, Any]) -> tuple[str, ...]:
    key = str(fit_row.task_id) if str(fit_row.task_id).startswith("T2-") else str(fit_row.station)
    try:
        classes = tuple(str(value) for value in contract["class_vocabulary"][key])
    except KeyError as error:
        raise ValueError(f"No frozen class vocabulary exists for {key}.") from error
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise ValueError(f"Frozen class vocabulary is invalid for {key}.")
    return classes


def _context(row: pd.Series) -> dict[str, Any]:
    return {field: row[field] for field in CONTEXT_COLUMNS}


def _metric_record(row: pd.Series, status: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    record = {
        **_context(row),
        "fit_id": str(row.fit_id),
        "model_id": str(row.model_id),
        "candidate_id": str(row.candidate_id),
        "seed": row.seed,
        "status": status,
        "balanced_accuracy": None,
        "macro_f1": None,
        "negative_log_likelihood": None,
        "brier_score": None,
        "ece": None,
        "per_class_recall_json": None,
        "confusion_matrix_json": None,
        "support_json": None,
    }
    if metrics is not None:
        for field in (
            "balanced_accuracy",
            "macro_f1",
            "negative_log_likelihood",
            "brier_score",
            "ece",
        ):
            record[field] = metrics[field]
        for field in ("per_class_recall", "confusion_matrix", "support"):
            record[f"{field}_json"] = json.dumps(
                metrics[field], allow_nan=False, separators=(",", ":"), sort_keys=True
            )
    return record


def execute_selection_rows(
    *,
    dataset: P03Dataset,
    fit_rows: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
    allowed_stages: set[str] | None = None,
) -> SelectionShardResult:
    """Execute concrete inner candidates only after the plan gate authorizes fitting."""

    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if fit_rows.empty:
        raise ValueError("Selection shard contains no fit rows.")
    stages = SELECTION_STAGES if allowed_stages is None else allowed_stages
    if not fit_rows.stage.astype(str).isin(stages).all():
        invalid = sorted(set(fit_rows.stage.astype(str)) - stages)
        raise ValueError(f"Selection shard contains dependency-bound stages: {invalid}")
    if not fit_rows.accounting.astype(str).eq("new_fit").all():
        raise ValueError("Selection shard contains a non-fitting cache record.")
    if not fit_rows.fit_id.astype(str).is_unique:
        raise ValueError("Selection shard contains duplicate fit IDs.")
    candidates = _candidate_lookup(candidate_registry)
    statuses: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for _, row in fit_rows.iterrows():
        candidate_id = str(row.candidate_id)
        if candidate_id not in candidates:
            raise ValueError(f"Fit row references an unregistered candidate: {candidate_id}")
        candidate = candidates[candidate_id]
        if str(candidate.model_id) != str(row.model_id):
            raise ValueError("Fit model differs from its frozen candidate registry row.")
        if str(candidate.hyperparameter_sha256) != str(row.hyperparameter_sha256):
            raise ValueError("Fit hyperparameters differ from the frozen candidate registry row.")
        parameters = json.loads(str(candidate.parameters_json))
        if sha256_value(parameters) != str(row.hyperparameter_sha256):
            raise ValueError("Parsed candidate parameters fail their frozen hash.")
        roles = resolve_fit_roles(row, manifest=manifest, p02_tables=p02_tables)
        vocabulary = _vocabulary(row, p03_contract)
        labels = set(roles.fit.target_analyte.astype(str)) | set(
            roles.validation.target_analyte.astype(str)
        )
        if not labels <= set(vocabulary):
            raise ValueError("A fitting or validation label lies outside the frozen vocabulary.")
        outcome = run_candidate_fit(
            dataset=dataset,
            fit_id=str(row.fit_id),
            model_id=str(row.model_id),
            candidate_id=candidate_id,
            parameters=parameters,
            seed=row.seed,
            fit_uids=roles.fit_uids,
            validation_uids=roles.validation_uids,
            class_vocabulary=vocabulary,
            expected_fit_uid_sha256=str(row.fit_uid_sha256),
            expected_validation_uid_sha256=str(row.validation_uid_sha256),
        )
        statuses.append({**_context(row), **outcome.status_record()})
        metrics.append(_metric_record(row, outcome.status, outcome.validation_metrics))
        if not outcome.validation_predictions.empty:
            prediction = outcome.validation_predictions.copy()
            for field, value in _context(row).items():
                prediction[field] = value
            prediction["model_id"] = str(row.model_id)
            prediction["candidate_id"] = str(row.candidate_id)
            prediction["hyperparameter_sha256"] = str(row.hyperparameter_sha256)
            prediction["seed"] = row.seed
            predictions.append(prediction)
    status_frame = pd.DataFrame(statuses)
    if len(status_frame) != len(fit_rows) or not status_frame.status.isin(
        {
            "complete",
            "unsupported_candidate",
            "rank_failure",
            "convergence_failure",
            "numerical_failure",
            "resource_failure",
            "data_failure",
            "fit_failure",
        }
    ).all():
        raise RuntimeError("Selection shard failed terminal fit accounting.")
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return SelectionShardResult(status_frame, prediction_frame, pd.DataFrame(metrics))


def execute_source_covariance_selection_rows(
    *,
    dataset: P03Dataset,
    fit_rows: pd.DataFrame,
    coral_candidate_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
) -> SelectionShardResult:
    """Execute C12 only after its versioned source-only method is authorized."""

    if p03_contract["coral"]["status"] != (
        "resolved_source_to_source_covariance_augmentation_v1"
    ):
        raise PermissionError("P03 C12 source-only covariance method remains unresolved.")
    return execute_selection_rows(
        dataset=dataset,
        fit_rows=fit_rows,
        candidate_registry=coral_candidate_registry,
        manifest=manifest,
        p02_tables=p02_tables,
        p03_contract=p03_contract,
        scientific_fitting_authorized=scientific_fitting_authorized,
        allowed_stages={"inner_source_coral_selection"},
    )


def execute_metadata_control_selection_rows(
    *,
    dataset: P03Dataset,
    fit_rows: pd.DataFrame,
    control_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
) -> SelectionShardResult:
    """Execute metadata-only selection only after the exact controls are authorized."""

    if p03_contract["negative_controls"]["status"] != "resolved_p03_negative_controls_v1":
        raise PermissionError("P03 negative-control scope remains unresolved.")
    candidates = metadata_control_candidate_registry(control_registry)
    return execute_selection_rows(
        dataset=dataset,
        fit_rows=fit_rows,
        candidate_registry=candidates,
        manifest=manifest,
        p02_tables=p02_tables,
        p03_contract=p03_contract,
        scientific_fitting_authorized=scientific_fitting_authorized,
        allowed_stages={"metadata_inner_selection"},
    )


def execute_selection_shard(
    *,
    store: P03ShardStore,
    shard_id: int,
    protected_state_sha256: str,
    dataset: P03Dataset,
    fit_rows: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
    allowed_stages: set[str] | None = None,
    native_thread_limit: int = 1,
) -> tuple[Path, str]:
    """Execute and atomically commit one bounded selection shard."""

    # The authorization check intentionally precedes lease creation as well as
    # estimator construction, so a blocked production plan leaves no run trace.
    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if native_thread_limit != 1:
        raise ValueError("P03 requires exactly one native math thread per worker.")
    lease = store.begin(
        shard_id=shard_id,
        protected_state_sha256=protected_state_sha256,
    )
    if lease.action == "verified_skip":
        return lease.final_dir, lease.action
    if lease.temporary_dir is None:
        raise RuntimeError("New P03 shard lease has no temporary directory.")
    try:
        with threadpool_limits(limits=native_thread_limit):
            result = execute_selection_rows(
                dataset=dataset,
                fit_rows=fit_rows,
                candidate_registry=candidate_registry,
                manifest=manifest,
                p02_tables=p02_tables,
                p03_contract=p03_contract,
                scientific_fitting_authorized=True,
                allowed_stages=allowed_stages,
            )
        result.fit_status.to_csv(lease.temporary_dir / "fit_status.csv", index=False)
        result.validation_predictions.to_parquet(
            lease.temporary_dir / "validation_predictions.parquet",
            index=False,
            compression="zstd",
        )
        result.selection_unit_metrics.to_csv(
            lease.temporary_dir / "selection_unit_metrics.csv", index=False
        )
        descriptor = {
            "schema_version": "p03-selection-shard-v1",
            "shard_id": shard_id,
            "protected_state_sha256": protected_state_sha256,
            "fit_count": len(fit_rows),
            "fit_id_sha256": sha256_value(sorted(fit_rows.fit_id.astype(str))),
            "terminal_status_count": len(result.fit_status),
            "prediction_rows": len(result.validation_predictions),
            "metric_rows": len(result.selection_unit_metrics),
            "native_math_threads_per_worker": native_thread_limit,
        }
        (lease.temporary_dir / "shard_descriptor.json").write_bytes(
            canonical_json_bytes(descriptor, pretty=True)
        )
        return store.commit(lease), lease.action
    except Exception:
        store.abort(lease, reason="selection_shard_execution_failed")
        raise
