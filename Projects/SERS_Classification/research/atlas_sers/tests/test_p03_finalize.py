from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.classical import TemperatureCalibration
from atlas_sers.evaluation.p03_finalize import (
    execute_selected_calibration_refits,
    execute_selected_outer_refit,
    fit_cached_selected_calibration,
    materialize_conditional_seed_rows,
)
from atlas_sers.evaluation.p03_runtime import P03Dataset
from atlas_sers.governance.canonical import sha256_value


def test_conditional_seed_materialization_is_mutually_exclusive() -> None:
    rows = pd.DataFrame(
        [
            {
                "accounting": "conditional_fit",
                "condition": "selected_model_is_deterministic",
                "seed": "deterministic",
            },
            *[
                {
                    "accounting": "conditional_fit",
                    "condition": "selected_model_is_stochastic",
                    "seed": seed,
                }
                for seed in (1, 2, 3)
            ],
        ]
    )
    active, excluded = materialize_conditional_seed_rows(
        rows, selected_model_id="C-PCA-LDA"
    )
    assert active.seed.tolist() == ["deterministic"]
    assert len(excluded) == 3
    active, excluded = materialize_conditional_seed_rows(
        rows, selected_model_id="C-RANDOM-FOREST"
    )
    assert active.seed.tolist() == [1, 2, 3]
    assert excluded.seed.tolist() == ["deterministic"]


def _outer_inputs() -> tuple[
    P03Dataset,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    rng = np.random.default_rng(20260809)
    records: list[dict[str, object]] = []
    matrix: list[np.ndarray] = []
    splits: list[dict[str, object]] = []
    for fold in range(4):
        for label, center in (("a", -1.0), ("b", 1.0)):
            uid = f"{label}-{fold}"
            master = f"{label}-m{fold}"
            records.append(
                {
                    "observation_uid": uid,
                    "master_sample_id": master,
                    "target_analyte": label,
                    "instrument": "unit-1",
                    "station": "cwa",
                }
            )
            matrix.append(rng.normal(center, 0.05, 8))
            splits.append(
                {
                    "outer_repeat": 1,
                    "station": "cwa",
                    "master_sample_id": master,
                    "outer_fold": fold,
                }
            )
    manifest = pd.DataFrame(records)
    dataset = P03Dataset.from_frozen_representation(
        intensity=np.vstack(matrix),
        representation_uids=manifest.observation_uid.to_numpy(),
        metadata=manifest,
    )
    fit = manifest[~manifest.master_sample_id.isin({"a-m0", "b-m0"})]
    test = manifest[manifest.master_sample_id.isin({"a-m0", "b-m0"})]
    parameters = {"metric": "cosine"}
    hp_hash = sha256_value(parameters)
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-1",
                "model_id": "C-SPECTRAL-MATCH",
                "parameters_json": json.dumps(parameters, separators=(",", ":")),
                "hyperparameter_sha256": hp_hash,
            }
        ]
    )
    final_rows = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-C01-T1",
                "task_id": "T1-CWA",
                "outer_run_id": "outer-1",
                "domain": "cwa:within",
                "station": "cwa",
                "held_instrument": "not_applicable",
                "outer_repeat": 1,
                "outer_fold": 0,
                "selection_mode": "inner_master_cv",
                "stage": "final_selected_refit",
                "selection_unit_id": "outer_train",
                "model_id": "C-SPECTRAL-MATCH",
                "candidate_id": "selected_after_inner",
                "hyperparameter_sha256": "0" * 64,
                "seed": "deterministic",
                "fit_rows": len(fit),
                "fit_masters": fit.master_sample_id.nunique(),
                "fit_uid_sha256": sha256_value(sorted(fit.observation_uid)),
                "validation_rows": 0,
                "validation_masters": 0,
                "validation_uid_sha256": sha256_value([]),
                "test_rows": len(test),
                "test_masters": test.master_sample_id.nunique(),
                "test_uid_sha256": sha256_value(sorted(test.observation_uid)),
                "accounting": "new_fit",
                "condition": "selected_candidate_supports_fit",
                "fit_id": "final-1",
            }
        ]
    )
    selection = pd.Series(
        {
            "status": "complete",
            "outer_run_id": "outer-1",
            "selected_candidate_id": "candidate-1",
            "selected_hyperparameter_sha256": hp_hash,
            "selection_state_sha256": "b" * 64,
        }
    )
    return (
        dataset,
        manifest,
        {"master_split_registry.csv": pd.DataFrame(splits)},
        final_rows,
        selection,
        candidates,
    )


def test_selected_refit_calibrates_only_after_fit_freeze() -> None:
    dataset, manifest, tables, rows, selection, candidates = _outer_inputs()
    calibration = TemperatureCalibration(
        temperature=1.0,
        class_vocabulary=("a", "b"),
        observations=2,
        masters=2,
        fit_observation_uid_sha256="c" * 64,
        fit_master_uid_sha256="d" * 64,
        optimizer_success=True,
        optimizer_objective=0.2,
    )
    result = execute_selected_outer_refit(
        dataset=dataset,
        final_fit_rows=rows,
        selection=selection,
        candidate_registry=candidates,
        manifest=manifest,
        p02_tables=tables,
        p03_contract={"class_vocabulary": {"cwa": ["a", "b"]}},
        calibration=calibration,
        calibration_observation_uids=["a-1", "b-1"],
        scientific_fitting_authorized=True,
    )
    assert result.fit_status.status.tolist() == ["complete"]
    assert result.fit_status.prediction_rows.tolist() == [2]
    assert result.fit_status.inference_seconds.gt(0).all()
    assert len(result.final_predictions) == 2
    assert result.final_predictions.probability_status.eq(
        "cross_fitted_temperature"
    ).all()
    assert result.final_predictions.outer_fold.eq(0).all()


def test_selected_refit_authorization_gate_precedes_work() -> None:
    dataset, manifest, tables, rows, selection, candidates = _outer_inputs()
    calibration = TemperatureCalibration(
        1.0, ("a", "b"), 2, 2, "c" * 64, "d" * 64, True, 0.2
    )
    with pytest.raises(PermissionError, match="disabled"):
        execute_selected_outer_refit(
            dataset=dataset,
            final_fit_rows=rows,
            selection=selection,
            candidate_registry=candidates,
            manifest=manifest,
            p02_tables=tables,
            p03_contract={"class_vocabulary": {"cwa": ["a", "b"]}},
            calibration=calibration,
            calibration_observation_uids=[],
            scientific_fitting_authorized=False,
        )


def test_cached_calibration_requires_every_selected_crossfit_id() -> None:
    parameters = {"metric": "cosine"}
    hp_hash = sha256_value(parameters)
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-1",
                "model_id": "C-SPECTRAL-MATCH",
                "parameters_json": json.dumps(parameters, separators=(",", ":")),
                "hyperparameter_sha256": hp_hash,
            }
        ]
    )
    selection = pd.Series(
        {
            "status": "complete",
            "selected_candidate_id": "candidate-1",
            "selected_hyperparameter_sha256": hp_hash,
        }
    )
    fit_manifest = pd.DataFrame(
        [
            {
                "fit_id": fit_id,
                "candidate_id": "candidate-1",
                "task_id": "T1-CWA",
                "station": "cwa",
                "selection_unit_id": unit,
            }
            for fit_id, unit in (("fit-1", "fold-1"), ("fit-2", "fold-2"))
        ]
    )
    prediction_rows: list[dict[str, object]] = []
    for fit_id, unit, offset in (("fit-1", "fold-1", 0), ("fit-2", "fold-2", 2)):
        for label, scores, local in (("a", [2.0, -2.0], 0), ("b", [-2.0, 2.0], 1)):
            prediction_rows.append(
                {
                    "fit_id": fit_id,
                    "candidate_id": "candidate-1",
                    "selection_unit_id": unit,
                    "seed": "deterministic",
                    "observation_uid": f"row-{offset + local}",
                    "master_sample_id": f"master-{offset + local}",
                    "instrument": "unit-1",
                    "station": "cwa",
                    "true_label": label,
                    "predicted_label": label,
                    "class_vocabulary": json.dumps(["a", "b"], separators=(",", ":")),
                    "scores": json.dumps(scores, separators=(",", ":")),
                    "probabilities": None,
                    "probability_status": "uncalibrated",
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    result = fit_cached_selected_calibration(
        selection_fit_manifest=fit_manifest,
        selection_predictions=predictions,
        selection=selection,
        candidate_registry=candidates,
    )
    assert result.selection_unit_count == 2
    assert result.calibration.observations == 4
    with pytest.raises(ValueError, match="do not cover"):
        fit_cached_selected_calibration(
            selection_fit_manifest=fit_manifest,
            selection_predictions=predictions[predictions.fit_id != "fit-2"],
            selection=selection,
            candidate_registry=candidates,
        )


def test_calibration_seed_branches_are_materialized_per_crossfit_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {"metric": "cosine"}
    hp_hash = sha256_value(parameters)
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-1",
                "model_id": "C-SPECTRAL-MATCH",
                "parameters_json": json.dumps(parameters, separators=(",", ":")),
                "hyperparameter_sha256": hp_hash,
            }
        ]
    )
    selection = pd.Series(
        {
            "status": "complete",
            "outer_run_id": "outer-1",
            "selected_candidate_id": "candidate-1",
            "selected_hyperparameter_sha256": hp_hash,
        }
    )
    rows: list[dict[str, object]] = []
    for unit in range(3):
        for seed, condition in (
            ("deterministic", "selected_model_is_deterministic"),
            (1, "selected_model_is_stochastic"),
            (2, "selected_model_is_stochastic"),
            (3, "selected_model_is_stochastic"),
        ):
            rows.append(
                {
                    "fit_id": f"fit-{unit}-{seed}",
                    "outer_run_id": "outer-1",
                    "task_id": "T3-ZS",
                    "station": "cwa",
                    "stage": "calibration_crossfit",
                    "selection_unit_id": f"calibration_master_cv:{unit}",
                    "accounting": "conditional_fit",
                    "condition": condition,
                    "seed": seed,
                    "fit_uid_sha256": "a" * 64,
                    "validation_uid_sha256": "b" * 64,
                }
            )
    fit_rows = pd.DataFrame(rows)

    def fake_roles(*_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(fit_uids=["fit"], validation_uids=["validation"])

    def fake_fit(**kwargs: object) -> SimpleNamespace:
        fit_id = str(kwargs["fit_id"])
        unit = int(fit_id.split("-")[1])
        predictions = pd.DataFrame(
            [
                {
                    "fit_id": fit_id,
                    "observation_uid": f"o-{unit}-{label}",
                    "master_sample_id": f"m-{unit}-{label}",
                    "instrument": "unit-1",
                    "station": "cwa",
                    "true_label": label,
                    "predicted_label": label,
                    "class_vocabulary": json.dumps(["a", "b"], separators=(",", ":")),
                    "scores": json.dumps(scores, separators=(",", ":")),
                    "probabilities": None,
                    "probability_status": "uncalibrated",
                }
                for label, scores in (("a", [2.0, -2.0]), ("b", [-2.0, 2.0]))
            ]
        )
        return SimpleNamespace(
            status="complete",
            status_record=lambda: {"fit_id": fit_id, "status": "complete"},
            validation_predictions=predictions,
        )

    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_finalize.resolve_fit_roles", fake_roles
    )
    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_finalize.run_candidate_fit", fake_fit
    )
    result = execute_selected_calibration_refits(
        dataset=SimpleNamespace(),  # type: ignore[arg-type]
        calibration_fit_rows=fit_rows,
        selection=selection,
        candidate_registry=candidates,
        manifest=pd.DataFrame(),
        p02_tables={},
        p03_contract={"class_vocabulary": {"cwa": ["a", "b"]}},
        scientific_fitting_authorized=True,
    )
    assert result.calibration_result is not None
    assert result.calibration_result.selection_unit_count == 3
    assert result.fit_status.status.value_counts().to_dict() == {
        "excluded_by_protocol": 9,
        "complete": 3,
    }
