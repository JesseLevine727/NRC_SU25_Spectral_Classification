"""Dependency-aware execution of non-selection P03 outer tasks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from atlas_sers.evaluation.p03_finalize import (
    CALIBRATION_STAGES,
    FINAL_STAGES,
    execute_selected_calibration_refits,
    execute_selected_outer_refit,
    fit_cached_selected_calibration,
)
from atlas_sers.evaluation.p03_roles import resolve_fit_roles
from atlas_sers.evaluation.p03_runtime import (
    P03Dataset,
    run_final_fit,
    run_final_prediction,
)
from atlas_sers.governance.canonical import sha256_value


@dataclass(frozen=True)
class OuterExecutionResult:
    fit_status: pd.DataFrame
    calibration: dict[str, Any] | None
    calibration_predictions: pd.DataFrame
    final_predictions: pd.DataFrame


def _cache_statuses(rows: pd.DataFrame, *, evidence_sha256: str) -> list[dict[str, Any]]:
    return [
        {
            "fit_id": str(row.fit_id),
            "outer_run_id": str(row.outer_run_id),
            "status": "complete",
            "reason_code": "cache_reuse_verified",
            "model_id": str(row.model_id),
            "candidate_id": str(row.candidate_id),
            "seed": row.seed,
            "fit_uid_sha256": str(row.fit_uid_sha256),
            "cache_evidence_sha256": evidence_sha256,
        }
        for row in rows.itertuples(index=False)
    ]


def _dependency_failure_statuses(
    rows: pd.DataFrame, *, reason_code: str
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fit_id": str(row.fit_id),
                "outer_run_id": str(row.outer_run_id),
                "status": "excluded_by_protocol",
                "reason_code": reason_code,
                "model_id": str(row.model_id),
                "candidate_id": str(row.candidate_id),
                "seed": row.seed,
                "fit_uid_sha256": str(row.fit_uid_sha256),
            }
            for row in rows.itertuples(index=False)
        ]
    )


def execute_selected_procedure_outer(
    *,
    dataset: P03Dataset,
    outer_fit_rows: pd.DataFrame,
    selection: pd.Series,
    candidate_registry: pd.DataFrame,
    selection_fit_manifest: pd.DataFrame,
    cached_selection_predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
    t2_first_repeat_only: bool = False,
) -> OuterExecutionResult:
    """Calibrate and refit one selected procedure after source-only selection freezes."""

    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if outer_fit_rows.empty or outer_fit_rows.outer_run_id.astype(str).nunique() != 1:
        raise ValueError("Selected outer execution requires exactly one outer run.")
    dependency_rows = outer_fit_rows[
        ~outer_fit_rows.stage.astype(str).isin(
            {
                "inner_selection",
                "training_station_inner_selection",
                "inner_source_coral_selection",
                "metadata_inner_selection",
            }
        )
    ]
    if str(selection.status) != "complete":
        return OuterExecutionResult(
            _dependency_failure_statuses(
                dependency_rows,
                reason_code="selection_dependency_not_complete",
            ),
            None,
            pd.DataFrame(),
            pd.DataFrame(),
        )
    final_rows = dependency_rows[dependency_rows.stage.astype(str).isin(FINAL_STAGES)]
    if final_rows.empty:
        raise ValueError("Selected outer execution has no final refit rows.")
    calibration_rows = dependency_rows[
        dependency_rows.stage.astype(str).isin(
            {*CALIBRATION_STAGES, "calibration_crossfit_reuse"}
        )
    ]
    bookkeeping_cache = dependency_rows[
        dependency_rows.accounting.astype(str).eq("cache_reuse")
        & ~dependency_rows.index.isin(calibration_rows.index)
    ]
    statuses: list[dict[str, Any]] = []
    calibration_predictions = pd.DataFrame()
    calibration_result = None
    needs_cached_calibration = calibration_rows.empty or calibration_rows.accounting.astype(
        str
    ).eq("cache_reuse").all()
    if needs_cached_calibration:
        if cached_selection_predictions.empty:
            raise ValueError("Cached calibration requires selected source predictions.")
        calibration_result = fit_cached_selected_calibration(
            selection_fit_manifest=selection_fit_manifest,
            selection_predictions=cached_selection_predictions,
            selection=selection,
            candidate_registry=candidate_registry,
            t2_first_repeat_only=t2_first_repeat_only,
        )
        calibration_predictions = calibration_result.cross_fitted_predictions
        statuses.extend(
            _cache_statuses(
                calibration_rows,
                evidence_sha256=calibration_result.evidence_fit_id_sha256,
            )
        )
    else:
        executable = calibration_rows[
            ~calibration_rows.accounting.astype(str).eq("cache_reuse")
        ]
        calibration_outcome = execute_selected_calibration_refits(
            dataset=dataset,
            calibration_fit_rows=executable,
            selection=selection,
            candidate_registry=candidate_registry,
            manifest=manifest,
            p02_tables=p02_tables,
            p03_contract=p03_contract,
            scientific_fitting_authorized=True,
        )
        statuses.extend(calibration_outcome.fit_status.to_dict(orient="records"))
        calibration_predictions = calibration_outcome.validation_predictions
        calibration_result = calibration_outcome.calibration_result
    if calibration_result is None:
        statuses.extend(
            _dependency_failure_statuses(
                final_rows,
                reason_code="calibration_dependency_not_complete",
            ).to_dict(orient="records")
        )
        return OuterExecutionResult(
            pd.DataFrame(statuses), None, pd.DataFrame(), pd.DataFrame()
        )
    statuses.extend(
        _cache_statuses(
            bookkeeping_cache,
            evidence_sha256=calibration_result.evidence_fit_id_sha256,
        )
    )
    final = execute_selected_outer_refit(
        dataset=dataset,
        final_fit_rows=final_rows,
        selection=selection,
        candidate_registry=candidate_registry,
        manifest=manifest,
        p02_tables=p02_tables,
        p03_contract=p03_contract,
        calibration=calibration_result.calibration,
        calibration_observation_uids=calibration_predictions.observation_uid.astype(
            str
        ).tolist(),
        scientific_fitting_authorized=True,
    )
    statuses.extend(final.fit_status.to_dict(orient="records"))
    calibration_record = {
        **asdict(calibration_result.calibration),
        "state_sha256": calibration_result.calibration.state_sha256,
        "selection_unit_count": calibration_result.selection_unit_count,
        "evidence_fit_id_sha256": calibration_result.evidence_fit_id_sha256,
    }
    planned_ids = set(dependency_rows.fit_id.astype(str))
    observed_ids = {str(row["fit_id"]) for row in statuses}
    if planned_ids != observed_ids:
        raise RuntimeError(
            "Outer dependency status coverage differs: "
            f"missing={len(planned_ids - observed_ids)} extra={len(observed_ids - planned_ids)}"
        )
    return OuterExecutionResult(
        pd.DataFrame(statuses),
        calibration_record,
        calibration_predictions,
        final.final_predictions,
    )


def execute_fixed_prior_t1_outer(
    *,
    dataset: P03Dataset,
    final_fit_rows: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
) -> OuterExecutionResult:
    """Execute both frozen C00 priors without selection or learned calibration."""

    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if final_fit_rows.empty or not final_fit_rows.stage.astype(str).eq("final_fixed").all():
        raise ValueError("Fixed-prior outer rows are empty or contain another stage.")
    statuses: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for _, row in final_fit_rows.iterrows():
        candidate = candidate_registry[
            candidate_registry.candidate_id.astype(str).eq(str(row.candidate_id))
        ]
        if len(candidate) != 1 or str(candidate.model_id.iloc[0]) != "C-PRIOR":
            raise ValueError("C00 prior row differs from the frozen candidate registry.")
        parameters = json.loads(str(candidate.parameters_json.iloc[0]))
        if sha256_value(parameters) != str(row.hyperparameter_sha256):
            raise ValueError("C00 prior parameters fail their frozen hash.")
        roles = resolve_fit_roles(row, manifest=manifest, p02_tables=p02_tables)
        outcome = run_final_fit(
            dataset=dataset,
            fit_id=str(row.fit_id),
            model_id="C-PRIOR",
            candidate_id=str(row.candidate_id),
            parameters=parameters,
            seed="deterministic",
            fit_uids=roles.fit_uids,
            expected_fit_uid_sha256=str(row.fit_uid_sha256),
        )
        status = outcome.status_record()
        status["outer_run_id"] = row.outer_run_id
        status["inference_seconds"] = None
        status["prediction_rows"] = 0
        if outcome.estimator is None:
            statuses.append(status)
            continue
        inference_start = time.perf_counter()
        prediction = run_final_prediction(
            dataset=dataset,
            estimator=outcome.estimator,
            fit_id=str(row.fit_id),
            test_uids=roles.test_uids,
            forbidden_fit_uids=roles.fit_uids,
            class_vocabulary=p03_contract["class_vocabulary"][row.station],
            calibration=None,
        )
        status["inference_seconds"] = time.perf_counter() - inference_start
        status["prediction_rows"] = len(prediction)
        statuses.append(status)
        for field in (
            "experiment_id",
            "task_id",
            "outer_run_id",
            "domain",
            "station",
            "held_instrument",
            "outer_repeat",
            "outer_fold",
        ):
            prediction[field] = row[field]
        prediction["model_id"] = "C-PRIOR"
        prediction["candidate_id"] = str(row.candidate_id)
        prediction["hyperparameter_sha256"] = str(row.hyperparameter_sha256)
        prediction["procedure_id"] = f"C-PRIOR:{str(row.candidate_id)}"
        predictions.append(prediction)
    return OuterExecutionResult(
        pd.DataFrame(statuses),
        None,
        pd.DataFrame(),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
    )
