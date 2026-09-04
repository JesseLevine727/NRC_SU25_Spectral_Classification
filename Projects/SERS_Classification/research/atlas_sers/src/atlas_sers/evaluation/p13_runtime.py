"""Leakage-audited execution of one locked P13 prediction shard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from atlas_sers.evaluation.p03_runtime import (
    P03Dataset,
    aggregate_seed_prediction_frames,
    fit_cross_fitted_temperature,
    run_candidate_fit,
    run_final_fit,
    run_final_prediction,
)
from atlas_sers.governance.canonical import sha256_value


@dataclass(frozen=True)
class P13ShardResult:
    fit_status: pd.DataFrame
    calibration_status: pd.DataFrame
    fold_endpoint_status: pd.DataFrame
    predictions: pd.DataFrame


def _uids(role: pd.Series) -> list[str]:
    values = json.loads(str(role.observation_uids_json))
    if not isinstance(values, list):
        raise ValueError("P13 role UIDs are not a JSON list.")
    return [str(value) for value in values]


def _role_lookup(role_registry: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    if role_registry.duplicated(["role_context_id", "role_name"]).any():
        raise ValueError("P13 role registry contains duplicate keys.")
    return {
        (str(row.role_context_id), str(row.role_name)): row
        for _, row in role_registry.iterrows()
    }


def _fit_status(
    row: pd.Series,
    status: str,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "context_id": str(row.context_id),
        "fit_id": str(row.fit_id),
        "procedure_id": str(row.procedure_id),
        "candidate_id": str(row.candidate_id),
        "model_id": str(row.model_id),
        "stage": str(row.stage),
        "inner_fold": int(row.inner_fold),
        "seed": str(row.seed),
        "status": status,
        "reason_code": reason_code,
        "fit_uid_sha256": str(row.fit_uid_sha256),
        "validation_uid_sha256": row.validation_uid_sha256,
        "fit_master_sha256": None,
        "elapsed_seconds": 0.0,
        "inference_seconds": None,
        "serialized_model_bytes": None,
        "warnings": "none",
        "traceback_digest": None,
    }


def _decorate_status(
    context: pd.Series, procedure_id: str, stage: str, record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "context_id": str(context.context_id),
        "domain_id": str(context.domain_id),
        "policy_id": str(context.policy_id),
        "outer_repeat": int(context.outer_repeat),
        "outer_fold": int(context.outer_fold),
        "procedure_id": procedure_id,
        "stage": stage,
        **record,
    }


def _endpoint_record(
    context: pd.Series,
    procedure: pd.Series,
    *,
    status: str,
    reason_code: str | None,
    held_rows: int,
    source_rows: int,
) -> dict[str, Any]:
    return {
        "context_id": str(context.context_id),
        "domain_id": str(context.domain_id),
        "station": str(context.station),
        "substrate_family": str(context.substrate_family),
        "held_instrument": str(context.held_instrument),
        "support_tier": str(context.support_tier),
        "policy_id": str(context.policy_id),
        "procedure_id": str(procedure.procedure_id),
        "candidate_id": procedure.candidate_id,
        "model_id": procedure.model_id,
        "outer_repeat": int(context.outer_repeat),
        "outer_fold": int(context.outer_fold),
        "status": status,
        "reason_code": reason_code,
        "held_prediction_rows": held_rows,
        "matched_source_prediction_rows": source_rows,
    }


def _decorate_predictions(
    frame: pd.DataFrame,
    *,
    context: pd.Series,
    procedure: pd.Series,
    prediction_role: str,
    calibration_sha256: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["context_id"] = str(context.context_id)
    result["domain_id"] = str(context.domain_id)
    result["substrate_family"] = str(context.substrate_family)
    result["held_instrument"] = str(context.held_instrument)
    result["support_tier"] = str(context.support_tier)
    result["policy_id"] = str(context.policy_id)
    result["procedure_id"] = str(procedure.procedure_id)
    result["candidate_id"] = str(procedure.candidate_id)
    result["model_id"] = str(procedure.model_id)
    result["outer_repeat"] = int(context.outer_repeat)
    result["outer_fold"] = int(context.outer_fold)
    result["prediction_role"] = prediction_role
    result["calibration_sha256"] = calibration_sha256
    return result


def execute_p13_shard_rows(
    *,
    dataset: P03Dataset,
    contexts: pd.DataFrame,
    role_registry: pd.DataFrame,
    procedure_registry: pd.DataFrame,
    fit_manifest: pd.DataFrame,
    scientific_fitting_authorized: bool,
) -> P13ShardResult:
    """Execute four outer folds for one domain/policy/repeat shard."""

    if not scientific_fitting_authorized:
        raise PermissionError("P13 fitting is not authorized by the validated plan.")
    if len(contexts) != 4 or contexts.outer_fold.nunique() != 4:
        raise ValueError("One P13 shard must contain exactly four outer folds.")
    if contexts.policy_id.nunique() != 1 or contexts.domain_id.nunique() != 1:
        raise ValueError("One P13 shard must contain one domain and one policy.")
    roles = _role_lookup(role_registry)
    fit_statuses: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for _, context in contexts.sort_values("outer_fold", kind="stable").iterrows():
        procedures = procedure_registry[
            procedure_registry.context_id.astype(str).eq(str(context.context_id))
        ].sort_values("procedure_order", kind="stable")
        if len(procedures) != 7:
            raise ValueError("P13 context does not have seven declared procedures.")
        vocabulary = tuple(
            sorted(
                dataset.metadata.loc[
                    dataset.metadata.station.astype(str).eq(str(context.station)),
                    "target_analyte",
                ]
                .astype(str)
                .unique()
            )
        )
        if len(vocabulary) != 3:
            raise ValueError("P13 evaluation requires an exact three-class vocabulary.")
        role_context_id = str(context.role_context_id)
        outer_fit_role = roles[(role_context_id, "outer_source_fit")]
        held_role = roles[(role_context_id, "outer_held_test")]
        source_role = roles[(role_context_id, "outer_matched_source_test")]
        outer_fit_uids = _uids(outer_fit_role)
        held_uids = _uids(held_role)
        source_uids = _uids(source_role)
        for _, procedure in procedures.iterrows():
            procedure_id = str(procedure.procedure_id)
            if str(procedure.planned_status) == "unavailable":
                endpoints.append(
                    _endpoint_record(
                        context,
                        procedure,
                        status="unavailable_terminal_failure",
                        reason_code=str(procedure.reason_code),
                        held_rows=0,
                        source_rows=0,
                    )
                )
                continue
            context_fits = fit_manifest[
                fit_manifest.context_id.astype(str).eq(str(context.context_id))
                & fit_manifest.procedure_id.astype(str).eq(procedure_id)
            ].copy()
            parameters = json.loads(str(procedure.parameters_json))
            calibration_rows = context_fits[
                context_fits.stage.astype(str).eq("calibration_crossfit")
            ].sort_values(["inner_fold", "seed"], kind="stable")
            validation_predictions: list[pd.DataFrame] = []
            calibration_complete = True
            for _, fit in calibration_rows.iterrows():
                inner_fold = int(fit.inner_fold)
                fit_role = roles[(role_context_id, f"calibration_fit_{inner_fold}")]
                validation_role = roles[
                    (role_context_id, f"calibration_validation_{inner_fold}")
                ]
                outcome = run_candidate_fit(
                    dataset=dataset,
                    fit_id=str(fit.fit_id),
                    model_id=str(fit.model_id),
                    candidate_id=str(fit.candidate_id),
                    parameters=parameters,
                    seed=fit.seed,
                    fit_uids=_uids(fit_role),
                    validation_uids=_uids(validation_role),
                    class_vocabulary=vocabulary,
                    expected_fit_uid_sha256=str(fit.fit_uid_sha256),
                    expected_validation_uid_sha256=str(
                        fit.validation_uid_sha256
                    ),
                )
                fit_statuses.append(
                    _decorate_status(
                        context,
                        procedure_id,
                        "calibration_crossfit",
                        outcome.status_record(),
                    )
                )
                calibration_complete &= outcome.status == "complete"
                if outcome.status == "complete":
                    prediction = outcome.validation_predictions.copy()
                    prediction["selection_unit_id"] = f"inner-{inner_fold}"
                    prediction["seed"] = str(fit.seed)
                    validation_predictions.append(prediction)
            calibration = None
            calibration_reason: str | None = None
            if calibration_complete:
                try:
                    calibration_result = fit_cross_fitted_temperature(
                        pd.concat(validation_predictions, ignore_index=True),
                        model_id=str(procedure.model_id),
                        class_vocabulary=vocabulary,
                    )
                    calibration = calibration_result.calibration
                except Exception as error:  # fail-closed aggregation boundary
                    calibration_reason = f"{type(error).__name__}:{error}"
            else:
                calibration_reason = "one_or_more_calibration_fits_failed"
            calibration_sha256 = calibration.state_sha256 if calibration else "unavailable"
            calibrations.append(
                {
                    "context_id": str(context.context_id),
                    "domain_id": str(context.domain_id),
                    "policy_id": str(context.policy_id),
                    "procedure_id": procedure_id,
                    "outer_repeat": int(context.outer_repeat),
                    "outer_fold": int(context.outer_fold),
                    "status": "complete" if calibration else "terminal_failure",
                    "reason_code": calibration_reason,
                    "calibration_sha256": calibration_sha256,
                    "calibration_json": json.dumps(
                        asdict(calibration) if calibration else {},
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                }
            )
            final_rows = context_fits[
                context_fits.stage.astype(str).eq("outer_final")
            ].sort_values("seed", kind="stable")
            if calibration is None:
                for _, fit in final_rows.iterrows():
                    fit_statuses.append(
                        _decorate_status(
                            context,
                            procedure_id,
                            "outer_final",
                            _fit_status(
                                fit,
                                "excluded_by_protocol",
                                "calibration_terminal_failure",
                            ),
                        )
                    )
                endpoints.append(
                    _endpoint_record(
                        context,
                        procedure,
                        status="unavailable_terminal_failure",
                        reason_code=calibration_reason,
                        held_rows=0,
                        source_rows=0,
                    )
                )
                continue
            held_seed_predictions: list[pd.DataFrame] = []
            source_seed_predictions: list[pd.DataFrame] = []
            final_complete = True
            for _, fit in final_rows.iterrows():
                outcome = run_final_fit(
                    dataset=dataset,
                    fit_id=str(fit.fit_id),
                    model_id=str(fit.model_id),
                    candidate_id=str(fit.candidate_id),
                    parameters=parameters,
                    seed=fit.seed,
                    fit_uids=outer_fit_uids,
                    expected_fit_uid_sha256=str(fit.fit_uid_sha256),
                )
                fit_statuses.append(
                    _decorate_status(
                        context,
                        procedure_id,
                        "outer_final",
                        outcome.status_record(),
                    )
                )
                final_complete &= outcome.status == "complete"
                if outcome.status != "complete" or outcome.estimator is None:
                    continue
                if held_uids:
                    held_prediction = run_final_prediction(
                        dataset=dataset,
                        estimator=outcome.estimator,
                        fit_id=str(fit.fit_id),
                        test_uids=held_uids,
                        forbidden_fit_uids=outer_fit_uids,
                        class_vocabulary=vocabulary,
                        calibration=None,
                    )
                    held_prediction["seed"] = str(fit.seed)
                    held_seed_predictions.append(held_prediction)
                if source_uids:
                    source_prediction = run_final_prediction(
                        dataset=dataset,
                        estimator=outcome.estimator,
                        fit_id=str(fit.fit_id),
                        test_uids=source_uids,
                        forbidden_fit_uids=outer_fit_uids,
                        class_vocabulary=vocabulary,
                        calibration=None,
                    )
                    source_prediction["seed"] = str(fit.seed)
                    source_seed_predictions.append(source_prediction)
            if not final_complete:
                endpoints.append(
                    _endpoint_record(
                        context,
                        procedure,
                        status="unavailable_terminal_failure",
                        reason_code="one_or_more_outer_final_fits_failed",
                        held_rows=0,
                        source_rows=0,
                    )
                )
                continue
            if not held_uids:
                endpoints.append(
                    _endpoint_record(
                        context,
                        procedure,
                        status="complete_empty_outer_fold",
                        reason_code="no_held_view_in_this_outer_fold_by_design",
                        held_rows=0,
                        source_rows=0,
                    )
                )
                continue
            aggregate_id = "P13PRED-" + sha256_value(
                {"context_id": str(context.context_id), "procedure_id": procedure_id}
            )[:24]
            held_prediction = aggregate_seed_prediction_frames(
                held_seed_predictions,
                model_id=str(procedure.model_id),
                aggregate_fit_id=aggregate_id,
                class_vocabulary=vocabulary,
                calibration=calibration,
            )
            held_prediction = _decorate_predictions(
                held_prediction,
                context=context,
                procedure=procedure,
                prediction_role="held_test",
                calibration_sha256=calibration_sha256,
            )
            predictions.append(held_prediction)
            source_rows = 0
            if source_seed_predictions:
                source_prediction = aggregate_seed_prediction_frames(
                    source_seed_predictions,
                    model_id=str(procedure.model_id),
                    aggregate_fit_id=aggregate_id,
                    class_vocabulary=vocabulary,
                    calibration=calibration,
                )
                source_prediction = _decorate_predictions(
                    source_prediction,
                    context=context,
                    procedure=procedure,
                    prediction_role="matched_source_test",
                    calibration_sha256=calibration_sha256,
                )
                source_rows = len(source_prediction)
                predictions.append(source_prediction)
            endpoints.append(
                _endpoint_record(
                    context,
                    procedure,
                    status="complete",
                    reason_code=None
                    if source_rows
                    else "matched_source_view_unavailable_by_design",
                    held_rows=len(held_prediction),
                    source_rows=source_rows,
                )
            )
    result = P13ShardResult(
        fit_status=pd.DataFrame(fit_statuses),
        calibration_status=pd.DataFrame(calibrations),
        fold_endpoint_status=pd.DataFrame(endpoints),
        predictions=pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(),
    )
    expected_fits = len(
        fit_manifest[fit_manifest.context_id.astype(str).isin(contexts.context_id)]
    )
    if len(result.fit_status) != expected_fits:
        raise RuntimeError("P13 shard did not retain one terminal row per planned fit.")
    if len(result.fold_endpoint_status) != len(contexts) * 7:
        raise RuntimeError("P13 shard did not retain every fold endpoint.")
    return result
