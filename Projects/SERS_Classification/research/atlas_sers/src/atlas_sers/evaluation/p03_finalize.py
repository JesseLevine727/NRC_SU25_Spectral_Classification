"""Dependency-aware selected refits and outer predictions for P03."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from atlas_sers.evaluation.classical import TemperatureCalibration
from atlas_sers.evaluation.p03_controls import permute_master_labels
from atlas_sers.evaluation.p03_roles import resolve_fit_roles
from atlas_sers.evaluation.p03_runtime import (
    CrossFittedCalibrationResult,
    P03Dataset,
    aggregate_seed_prediction_frames,
    fit_cross_fitted_temperature,
    run_candidate_fit,
    run_final_fit,
    run_final_prediction,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.classical import STOCHASTIC_MODELS

FINAL_STAGES = {
    "final_selected_refit",
    "final_family_refit",
    "final_source_coral_refit",
    "metadata_final_refit",
}
CALIBRATION_STAGES = {"calibration_crossfit", "metadata_calibration_crossfit"}


@dataclass(frozen=True)
class FinalOuterResult:
    fit_status: pd.DataFrame
    final_predictions: pd.DataFrame


@dataclass(frozen=True)
class CalibrationOuterResult:
    fit_status: pd.DataFrame
    validation_predictions: pd.DataFrame
    calibration_result: CrossFittedCalibrationResult | None


def _selected_candidate(
    selection: pd.Series, candidate_registry: pd.DataFrame
) -> tuple[str, str, dict[str, Any], str]:
    if str(selection.status) != "complete":
        raise ValueError("A terminally failed selection cannot be refit.")
    candidate_id = str(selection.selected_candidate_id)
    rows = candidate_registry[candidate_registry.candidate_id.astype(str) == candidate_id]
    if len(rows) != 1:
        raise ValueError("Selected candidate is absent or non-unique in the frozen registry.")
    row = rows.iloc[0]
    model_id = str(row.model_id)
    hyperparameter_sha256 = str(row.hyperparameter_sha256)
    if hyperparameter_sha256 != str(selection.selected_hyperparameter_sha256):
        raise ValueError("Selected hyperparameter hash differs from the candidate registry.")
    parameters = json.loads(str(row.parameters_json))
    if sha256_value(parameters) != hyperparameter_sha256:
        raise ValueError("Selected candidate parameters fail their frozen hash.")
    return candidate_id, model_id, parameters, hyperparameter_sha256


def materialize_conditional_seed_rows(
    fit_rows: pd.DataFrame, *, selected_model_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Activate exactly one deterministic slot or all three forest slots."""

    if fit_rows.empty:
        raise ValueError("Final fit rows are empty.")
    stochastic = selected_model_id in STOCHASTIC_MODELS
    active_condition = (
        "selected_model_is_stochastic"
        if stochastic
        else "selected_model_is_deterministic"
    )
    conditional = fit_rows.accounting.eq("conditional_fit")
    active = fit_rows[(~conditional) | fit_rows.condition.eq(active_condition)].copy()
    excluded = fit_rows[conditional & ~fit_rows.condition.eq(active_condition)].copy()
    expected = 3 if stochastic else 1
    if len(active) != expected:
        raise ValueError(
            f"Selected model {selected_model_id} resolves to {len(active)}, not {expected}, fits."
        )
    if stochastic and not active.seed.astype(str).str.fullmatch(r"\d+").all():
        raise ValueError("A stochastic final fit lacks an integer technical seed.")
    if not stochastic and active.seed.astype(str).tolist() != ["deterministic"]:
        raise ValueError("A deterministic final fit lacks its deterministic seed slot.")
    return active, excluded


def fit_cached_selected_calibration(
    *,
    selection_fit_manifest: pd.DataFrame,
    selection_predictions: pd.DataFrame,
    selection: pd.Series,
    candidate_registry: pd.DataFrame,
    t2_first_repeat_only: bool = False,
) -> CrossFittedCalibrationResult:
    """Recover exact selected OOF scores already produced during inner selection."""

    candidate_id, model_id, _, _ = _selected_candidate(selection, candidate_registry)
    expected = selection_fit_manifest[
        selection_fit_manifest.candidate_id.astype(str) == candidate_id
    ].copy()
    if t2_first_repeat_only:
        expected = expected[
            expected.selection_unit_id.astype(str).str.startswith("repeat:1:")
        ]
    expected_fit_ids = set(expected.fit_id.astype(str))
    observed = selection_predictions[
        selection_predictions.fit_id.astype(str).isin(expected_fit_ids)
    ].copy()
    if set(observed.fit_id.astype(str)) != expected_fit_ids:
        raise ValueError("Cached calibration predictions do not cover selected fit IDs.")
    if set(observed.candidate_id.astype(str)) != {candidate_id}:
        raise ValueError("Cached calibration evidence contains another candidate.")
    key = str(expected.task_id.iloc[0]) if str(expected.task_id.iloc[0]).startswith("T2-") else str(
        expected.station.iloc[0]
    )
    vocabulary_json = observed.class_vocabulary.astype(str).unique()
    if len(vocabulary_json) != 1:
        raise ValueError(f"Cached calibration class vocabulary is inconsistent for {key}.")
    vocabulary = json.loads(vocabulary_json[0])
    return fit_cross_fitted_temperature(
        observed,
        model_id=model_id,
        class_vocabulary=vocabulary,
    )


def execute_selected_calibration_refits(
    *,
    dataset: P03Dataset,
    calibration_fit_rows: pd.DataFrame,
    selection: pd.Series,
    candidate_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
) -> CalibrationOuterResult:
    """Run only source-master calibration folds that were not reusable selection fits."""

    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if calibration_fit_rows.empty or not calibration_fit_rows.stage.isin(
        CALIBRATION_STAGES
    ).all():
        raise ValueError("Calibration refit rows are empty or contain another stage.")
    if calibration_fit_rows.accounting.eq("cache_reuse").any():
        raise ValueError("Reusable calibration rows must use cached selection evidence.")
    if calibration_fit_rows.outer_run_id.astype(str).nunique() != 1:
        raise ValueError("Calibration refits span multiple outer runs.")
    if str(calibration_fit_rows.outer_run_id.iloc[0]) != str(selection.outer_run_id):
        raise ValueError("Selection and calibration rows reference different outer runs.")
    candidate_id, model_id, parameters, _ = _selected_candidate(
        selection, candidate_registry
    )
    statuses: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    vocabulary_row = calibration_fit_rows.iloc[0]
    vocabulary = p03_contract["class_vocabulary"][
        vocabulary_row.task_id
        if str(vocabulary_row.task_id).startswith("T2-")
        else vocabulary_row.station
    ]
    expected_active = 0
    for _, unit_rows in calibration_fit_rows.groupby("selection_unit_id", sort=True):
        active, excluded = materialize_conditional_seed_rows(
            unit_rows, selected_model_id=model_id
        )
        expected_active += len(active)
        statuses.extend(
            {
                "fit_id": row.fit_id,
                "outer_run_id": row.outer_run_id,
                "status": "excluded_by_protocol",
                "reason_code": "inactive_mutually_exclusive_selected_seed_branch",
                "model_id": model_id,
                "candidate_id": candidate_id,
                "seed": row.seed,
                "fit_uid_sha256": row.fit_uid_sha256,
            }
            for row in excluded.itertuples(index=False)
        )
        for _, row in active.iterrows():
            roles = resolve_fit_roles(row, manifest=manifest, p02_tables=p02_tables)
            effective_seed: int | str = (
                int(row.seed) if model_id in STOCHASTIC_MODELS else "deterministic"
            )
            outcome = run_candidate_fit(
                dataset=dataset,
                fit_id=str(row.fit_id),
                model_id=model_id,
                candidate_id=candidate_id,
                parameters=parameters,
                seed=effective_seed,
                fit_uids=roles.fit_uids,
                validation_uids=roles.validation_uids,
                class_vocabulary=vocabulary,
                expected_fit_uid_sha256=str(row.fit_uid_sha256),
                expected_validation_uid_sha256=str(row.validation_uid_sha256),
            )
            status = outcome.status_record()
            status["outer_run_id"] = row.outer_run_id
            statuses.append(status)
            if not outcome.validation_predictions.empty:
                prediction = outcome.validation_predictions.copy()
                prediction["seed"] = effective_seed
                prediction["selection_unit_id"] = row.selection_unit_id
                prediction["candidate_id"] = candidate_id
                predictions.append(prediction)
    completed = sum(row["status"] == "complete" for row in statuses)
    if completed != expected_active:
        return CalibrationOuterResult(pd.DataFrame(statuses), pd.DataFrame(), None)
    validation = pd.concat(predictions, ignore_index=True)
    calibration_result = fit_cross_fitted_temperature(
        validation,
        model_id=model_id,
        class_vocabulary=vocabulary,
    )
    return CalibrationOuterResult(
        pd.DataFrame(statuses), validation, calibration_result
    )


def execute_selected_outer_refit(
    *,
    dataset: P03Dataset,
    final_fit_rows: pd.DataFrame,
    selection: pd.Series,
    candidate_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    calibration: TemperatureCalibration,
    calibration_observation_uids: list[str],
    scientific_fitting_authorized: bool,
) -> FinalOuterResult:
    """Refit a frozen selection and reveal its exact outer test only afterward."""

    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if not final_fit_rows.stage.astype(str).isin(FINAL_STAGES).all():
        raise ValueError("Selected outer refit received a non-final stage.")
    if final_fit_rows.outer_run_id.astype(str).nunique() != 1:
        raise ValueError("Selected outer refit spans multiple outer runs.")
    if str(final_fit_rows.outer_run_id.iloc[0]) != str(selection.outer_run_id):
        raise ValueError("Selection and final fit rows reference different outer runs.")
    candidate_id, model_id, parameters, hyperparameter_sha256 = _selected_candidate(
        selection, candidate_registry
    )
    active, excluded = materialize_conditional_seed_rows(
        final_fit_rows, selected_model_id=model_id
    )
    statuses: list[dict[str, Any]] = []
    for row in excluded.itertuples(index=False):
        statuses.append(
            {
                "fit_id": row.fit_id,
                "outer_run_id": row.outer_run_id,
                "status": "excluded_by_protocol",
                "reason_code": "inactive_mutually_exclusive_selected_seed_branch",
                "model_id": model_id,
                "candidate_id": candidate_id,
                "seed": row.seed,
                "fit_uid_sha256": row.fit_uid_sha256,
            }
        )
    seed_predictions: list[pd.DataFrame] = []
    test_uid_hashes: set[str] = set()
    for _, row in active.iterrows():
        roles = resolve_fit_roles(row, manifest=manifest, p02_tables=p02_tables)
        forbidden = set(roles.fit_uids) | {str(value) for value in calibration_observation_uids}
        if forbidden & set(roles.test_uids):
            raise ValueError("Calibration or fitting evidence overlaps the outer-test role.")
        effective_seed: int | str = (
            int(row.seed) if model_id in STOCHASTIC_MODELS else "deterministic"
        )
        outcome = run_final_fit(
            dataset=dataset,
            fit_id=str(row.fit_id),
            model_id=model_id,
            candidate_id=candidate_id,
            parameters=parameters,
            seed=effective_seed,
            fit_uids=roles.fit_uids,
            expected_fit_uid_sha256=str(row.fit_uid_sha256),
        )
        status = outcome.status_record()
        status["outer_run_id"] = row.outer_run_id
        status["selected_hyperparameter_sha256"] = hyperparameter_sha256
        status["inference_seconds"] = None
        status["prediction_rows"] = 0
        if outcome.estimator is not None:
            inference_start = time.perf_counter()
            prediction = run_final_prediction(
                dataset=dataset,
                estimator=outcome.estimator,
                fit_id=str(row.fit_id),
                test_uids=roles.test_uids,
                forbidden_fit_uids=sorted(forbidden),
                class_vocabulary=p03_contract["class_vocabulary"][
                    row.task_id if str(row.task_id).startswith("T2-") else row.station
                ],
                calibration=None,
            )
            status["inference_seconds"] = time.perf_counter() - inference_start
            status["prediction_rows"] = len(prediction)
            prediction["seed"] = effective_seed
            seed_predictions.append(prediction)
            test_uid_hashes.add(sha256_value(sorted(prediction.observation_uid.astype(str))))
        statuses.append(status)
    completed = [row for row in statuses if row["status"] == "complete"]
    expected_completed = 3 if model_id in STOCHASTIC_MODELS else 1
    if len(completed) != expected_completed:
        return FinalOuterResult(pd.DataFrame(statuses), pd.DataFrame())
    if len(test_uid_hashes) != 1:
        raise ValueError("Technical-seed final predictions use different outer-test UIDs.")
    vocabulary = p03_contract["class_vocabulary"][
        final_fit_rows.task_id.iloc[0]
        if str(final_fit_rows.task_id.iloc[0]).startswith("T2-")
        else final_fit_rows.station.iloc[0]
    ]
    aggregate = aggregate_seed_prediction_frames(
        seed_predictions,
        model_id=model_id,
        aggregate_fit_id=f"aggregate:{selection.outer_run_id}",
        class_vocabulary=vocabulary,
        calibration=calibration,
    )
    context_fields = [
        "experiment_id",
        "task_id",
        "outer_run_id",
        "domain",
        "station",
        "held_instrument",
        "outer_repeat",
        "outer_fold",
    ]
    for field in context_fields:
        aggregate[field] = final_fit_rows.iloc[0][field]
    aggregate["model_id"] = model_id
    aggregate["candidate_id"] = candidate_id
    aggregate["hyperparameter_sha256"] = hyperparameter_sha256
    aggregate["selection_state_sha256"] = selection.selection_state_sha256
    aggregate["calibration_state_sha256"] = calibration.state_sha256
    # Pool folds under the declared algorithmic procedure. The selected
    # concrete model/candidate remain separate audit columns and may differ by
    # fold without fragmenting the endpoint.
    aggregate["procedure_id"] = str(final_fit_rows.model_id.iloc[0])
    return FinalOuterResult(pd.DataFrame(statuses), aggregate)


def execute_permutation_outer_refit(
    *,
    dataset: P03Dataset,
    permutation_fit_rows: pd.DataFrame,
    real_selection: pd.Series,
    candidate_registry: pd.DataFrame,
    control_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
) -> FinalOuterResult:
    """Refit the frozen real-label C09 model under one master-label permutation."""

    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if p03_contract["negative_controls"]["status"] != "resolved_p03_negative_controls_v1":
        raise PermissionError("P03 negative-control scope remains unresolved.")
    if permutation_fit_rows.empty or not permutation_fit_rows.stage.eq(
        "permutation_selected_refit"
    ).all():
        raise ValueError("Permutation refit rows are empty or contain another stage.")
    if permutation_fit_rows.outer_run_id.astype(str).nunique() != 1:
        raise ValueError("Permutation refit spans multiple control outer runs.")
    source_outer_ids = set(permutation_fit_rows.source_outer_run_id.astype(str))
    if source_outer_ids != {str(real_selection.outer_run_id)}:
        raise ValueError("Permutation control does not reference its real C09 selection.")
    control_ids = set(permutation_fit_rows.control_candidate_id.astype(str))
    if len(control_ids) != 1:
        raise ValueError("Permutation refit contains multiple control configurations.")
    control_rows = control_registry[
        control_registry.control_candidate_id.astype(str).isin(control_ids)
        & control_registry.control_type.astype(str).eq("master_label_permutation")
    ]
    if len(control_rows) != 1:
        raise ValueError("Permutation configuration is absent or non-unique.")
    control = control_rows.iloc[0]
    control_parameters = json.loads(str(control.parameters_json))
    if sha256_value(control_parameters) != str(control.configuration_sha256):
        raise ValueError("Permutation configuration fails its frozen hash.")
    candidate_id, model_id, parameters, hyperparameter_sha256 = _selected_candidate(
        real_selection, candidate_registry
    )
    active, excluded = materialize_conditional_seed_rows(
        permutation_fit_rows, selected_model_id=model_id
    )
    statuses: list[dict[str, Any]] = [
        {
            "fit_id": row.fit_id,
            "outer_run_id": row.outer_run_id,
            "status": "excluded_by_protocol",
            "reason_code": "inactive_mutually_exclusive_selected_seed_branch",
            "model_id": model_id,
            "candidate_id": candidate_id,
            "seed": row.seed,
            "fit_uid_sha256": row.fit_uid_sha256,
            "control_candidate_id": control.control_candidate_id,
        }
        for row in excluded.itertuples(index=False)
    ]
    seed_predictions: list[pd.DataFrame] = []
    mapping_hashes: set[str] = set()
    label_hashes: set[str] = set()
    fixed_points: set[int] = set()
    for _, row in active.iterrows():
        roles = resolve_fit_roles(row, manifest=manifest, p02_tables=p02_tables)
        fit_metadata = dataset.subset(roles.fit_uids)[1]
        permutation = permute_master_labels(
            fit_metadata, seed=int(control_parameters["permutation_seed"])
        )
        mapping_hashes.add(permutation.mapping_sha256)
        label_hashes.add(permutation.label_sha256)
        fixed_points.add(permutation.fixed_points)
        effective_seed: int | str = (
            int(row.seed) if model_id in STOCHASTIC_MODELS else "deterministic"
        )
        outcome = run_final_fit(
            dataset=dataset,
            fit_id=str(row.fit_id),
            model_id=model_id,
            candidate_id=candidate_id,
            parameters=parameters,
            seed=effective_seed,
            fit_uids=roles.fit_uids,
            expected_fit_uid_sha256=str(row.fit_uid_sha256),
            fit_labels=permutation.labels,
            expected_fit_label_sha256=permutation.label_sha256,
        )
        status = outcome.status_record()
        status.update(
            {
                "outer_run_id": row.outer_run_id,
                "control_candidate_id": control.control_candidate_id,
                "permutation_mapping_sha256": permutation.mapping_sha256,
                "permutation_fixed_points": permutation.fixed_points,
                "inference_seconds": None,
                "prediction_rows": 0,
            }
        )
        if outcome.estimator is not None:
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
            prediction["seed"] = effective_seed
            seed_predictions.append(prediction)
        statuses.append(status)
    expected_completed = 3 if model_id in STOCHASTIC_MODELS else 1
    if sum(row["status"] == "complete" for row in statuses) != expected_completed:
        return FinalOuterResult(pd.DataFrame(statuses), pd.DataFrame())
    if len(mapping_hashes) != 1 or len(label_hashes) != 1 or len(fixed_points) != 1:
        raise RuntimeError("Technical seeds used different master-label permutations.")
    vocabulary = p03_contract["class_vocabulary"][permutation_fit_rows.station.iloc[0]]
    aggregate = aggregate_seed_prediction_frames(
        seed_predictions,
        model_id=model_id,
        aggregate_fit_id=f"aggregate:{permutation_fit_rows.outer_run_id.iloc[0]}",
        class_vocabulary=vocabulary,
        calibration=None,
    )
    first = permutation_fit_rows.iloc[0]
    for field in (
        "experiment_id",
        "task_id",
        "outer_run_id",
        "source_outer_run_id",
        "domain",
        "station",
        "held_instrument",
        "outer_repeat",
        "outer_fold",
        "control_type",
        "control_candidate_id",
        "control_replicate",
    ):
        aggregate[field] = first[field]
    aggregate["model_id"] = "C-PERMUTED-SELECTED"
    aggregate["base_model_id"] = model_id
    aggregate["base_candidate_id"] = candidate_id
    aggregate["base_hyperparameter_sha256"] = hyperparameter_sha256
    aggregate["real_selection_state_sha256"] = real_selection.selection_state_sha256
    aggregate["permutation_mapping_sha256"] = next(iter(mapping_hashes))
    aggregate["permuted_fit_label_sha256"] = next(iter(label_hashes))
    aggregate["permutation_fixed_points"] = next(iter(fixed_points))
    aggregate["probability_status"] = "omitted_by_permutation_control_protocol"
    aggregate["probabilities"] = None
    aggregate["procedure_id"] = (
        f"C-PERMUTED-SELECTED:{str(control.control_candidate_id)}"
    )
    return FinalOuterResult(pd.DataFrame(statuses), aggregate)


def execute_prior_control_outer_refit(
    *,
    dataset: P03Dataset,
    prior_fit_row: pd.Series,
    control_registry: pd.DataFrame,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
    p03_contract: dict[str, Any],
    scientific_fitting_authorized: bool,
) -> FinalOuterResult:
    """Fit one frozen source-station prior on the exact C09 control role."""

    if not scientific_fitting_authorized:
        raise PermissionError("P03 scientific fitting is disabled by the frozen budget gate.")
    if p03_contract["negative_controls"]["status"] != "resolved_p03_negative_controls_v1":
        raise PermissionError("P03 negative-control scope remains unresolved.")
    row = prior_fit_row
    if str(row.stage) != "prior_control_final" or str(row.model_id) != "C-PRIOR":
        raise ValueError("Prior control received a non-prior final fit row.")
    registered = control_registry[
        control_registry.control_candidate_id.astype(str).eq(
            str(row.control_candidate_id)
        )
        & control_registry.control_type.astype(str).eq("station_or_target_prior")
    ]
    if len(registered) != 1:
        raise ValueError("Prior control configuration is absent or non-unique.")
    control = registered.iloc[0]
    parameters = json.loads(str(control.parameters_json))
    if sha256_value(parameters) != str(control.configuration_sha256):
        raise ValueError("Prior control parameters fail their frozen hash.")
    if str(row.hyperparameter_sha256) != str(control.configuration_sha256):
        raise ValueError("Prior fit row differs from its frozen control parameters.")
    roles = resolve_fit_roles(row, manifest=manifest, p02_tables=p02_tables)
    outcome = run_final_fit(
        dataset=dataset,
        fit_id=str(row.fit_id),
        model_id="C-PRIOR",
        candidate_id=str(row.control_candidate_id),
        parameters=parameters,
        seed="deterministic",
        fit_uids=roles.fit_uids,
        expected_fit_uid_sha256=str(row.fit_uid_sha256),
    )
    status = outcome.status_record()
    status.update(
        {
            "outer_run_id": row.outer_run_id,
            "control_candidate_id": row.control_candidate_id,
            "inference_seconds": None,
            "prediction_rows": 0,
        }
    )
    if outcome.estimator is None:
        return FinalOuterResult(pd.DataFrame([status]), pd.DataFrame())
    inference_start = time.perf_counter()
    predictions = run_final_prediction(
        dataset=dataset,
        estimator=outcome.estimator,
        fit_id=str(row.fit_id),
        test_uids=roles.test_uids,
        forbidden_fit_uids=roles.fit_uids,
        class_vocabulary=p03_contract["class_vocabulary"][row.station],
        calibration=None,
    )
    status["inference_seconds"] = time.perf_counter() - inference_start
    status["prediction_rows"] = len(predictions)
    for field in (
        "experiment_id",
        "task_id",
        "outer_run_id",
        "source_outer_run_id",
        "domain",
        "station",
        "held_instrument",
        "outer_repeat",
        "outer_fold",
        "control_type",
        "control_candidate_id",
    ):
        predictions[field] = row[field]
    predictions["model_id"] = "C-PRIOR"
    predictions["candidate_id"] = str(row.control_candidate_id)
    predictions["hyperparameter_sha256"] = str(control.configuration_sha256)
    predictions["procedure_id"] = f"C-PRIOR:{str(row.control_candidate_id)}"
    return FinalOuterResult(pd.DataFrame([status]), predictions)
