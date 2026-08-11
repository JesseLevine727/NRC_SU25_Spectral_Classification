from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p03_plan import (
    STOCHASTIC_MODELS,
    _fit_record,
    assign_selection_shards,
    build_candidate_registry,
    build_control_registry,
    build_coral_candidate_registry,
    summarize_compute,
)

PROJECT = Path(__file__).resolve().parents[1]
CONTRACTS = PROJECT / "plan" / "contracts"


def _contracts() -> tuple[dict[str, object], dict[str, object]]:
    hyperparameters = json.loads((CONTRACTS / "hyperparameter_registry.json").read_text())
    p03 = json.loads((CONTRACTS / "p03_governance_contract.json").read_text())
    return hyperparameters, p03


def _rows(prefix: str, masters: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_uid": [f"{prefix}-{index}" for index in range(len(masters))],
            "master_sample_id": masters,
        }
    )


def test_p03_candidate_registry_expands_exact_frozen_suite() -> None:
    hyperparameters, contract = _contracts()
    candidates = build_candidate_registry(hyperparameters, contract)
    assert len(candidates) == 126
    assert candidates.candidate_id.is_unique
    assert candidates.declared_candidate_order.tolist() == list(range(126))
    counts = candidates.groupby("model_id").size().to_dict()
    assert counts == {
        "C-PRIOR": 2,
        "C-SPECTRAL-MATCH": 3,
        "C-NEAREST-CENTROID": 8,
        "C-PCA-LDA": 10,
        "C-PLS-DA": 5,
        "C-LOGREG-EN": 30,
        "C-RBF-SVM": 36,
        "C-RANDOM-FOREST": 16,
        "C-EXTRA-TREES": 16,
    }
    assert set(candidates.loc[candidates.stochastic, "model_id"]) == STOCHASTIC_MODELS
    assert candidates.loc[candidates.stochastic, "seed_count"].eq(3).all()
    assert candidates.loc[~candidates.stochastic, "seed_count"].eq(1).all()
    coral = build_coral_candidate_registry(candidates, contract)
    assert len(coral) == 46
    assert coral.candidate_id.is_unique
    assert set(coral.base_model_id) == {"C-PCA-LDA", "C-RBF-SVM"}
    assert coral.seed_count.eq(1).all()
    assert coral.method_status.eq(
        "resolved_source_to_source_covariance_augmentation_v1"
    ).all()
    controls = build_control_registry(candidates, contract)
    assert len(controls) == 52
    assert controls.control_candidate_id.is_unique
    assert controls.control_type.value_counts().to_dict() == {
        "acquisition_metadata_only": 30,
        "master_label_permutation": 20,
        "station_or_target_prior": 2,
    }
    assert controls.method_status.eq(
        "resolved_p03_negative_controls_v1"
    ).all()


def test_selection_shards_never_mix_execution_kind_or_stage() -> None:
    rows = pd.DataFrame(
        [
            {"fit_id": f"standard-{index}", "stage": "inner_selection"}
            for index in range(3)
        ]
        + [
            {"fit_id": f"t2-{index}", "stage": "training_station_inner_selection"}
            for index in range(3)
        ]
        + [
            {"fit_id": f"coral-{index}", "stage": "inner_source_coral_selection"}
            for index in range(3)
        ]
        + [
            {"fit_id": f"metadata-{index}", "stage": "metadata_inner_selection"}
            for index in range(3)
        ]
    )
    assigned = assign_selection_shards(rows, target=2)
    for _, shard in assigned.groupby("selection_shard_id"):
        assert shard.selection_kind.nunique() == 1
        assert shard.stage.nunique() == 1
        assert len(shard) <= 2


def test_fit_record_does_not_reuse_parent_role_cache_for_empty_slice() -> None:
    fit = _rows("fit", ["m1", "m2"])
    validation = _rows("validation", ["m3"])
    test = _rows("test", ["m4"])
    outer = {
        "experiment_id": "EXP-C03-T1",
        "task_id": "T1-CWA",
        "outer_run_id": "P03OUTER-test",
        "domain": "cwa:within",
        "station": "cwa",
        "held_instrument": "not_applicable",
        "outer_repeat": 0,
        "outer_fold": 0,
        "selection_mode": "inner_master_cv",
    }
    first = _fit_record(
        outer=outer,
        stage="inner_selection",
        model_id="C-PCA-LDA",
        candidate_id="C-PCA-LDA-000",
        hyperparameter_sha256="a" * 64,
        seed="deterministic",
        unit_id="inner:0",
        fit=fit,
        validation=validation,
        test=test,
    )
    second = _fit_record(
        outer=outer,
        stage="final_selected_refit",
        model_id="C-PCA-LDA",
        candidate_id="selected_after_inner",
        hyperparameter_sha256="b" * 64,
        seed="deterministic",
        unit_id="outer_train",
        fit=fit,
        validation=fit.iloc[0:0],
        test=test,
    )
    assert first["fit_validation_master_disjoint"]
    assert second["validation_rows"] == second["validation_masters"] == 0
    assert second["fit_validation_master_disjoint"]


def test_compute_summary_fails_closed_above_registered_budget() -> None:
    _, contract = _contracts()
    row = {
        "fit_id": "P03FIT-test",
        "experiment_id": "EXP-C06-T1",
        "task_id": "T1-CWA",
        "model_id": "C-RBF-SVM",
        "outer_run_id": "P03OUTER-test",
        "test_uid_sha256": "c" * 64,
        "test_rows": 1,
        "validation_rows": 1,
        "accounting": "new_fit",
        "stage": "inner_selection",
        "condition": "always",
    }
    ceiling = int(contract["planning"]["registered_fit_budget_high"])
    fits = pd.DataFrame([row] * (ceiling + 1))
    tables: dict[str, pd.DataFrame | dict[str, object]] = {"fit_manifest.csv": fits}
    result = summarize_compute(tables, contract)
    budget = result["budget_gate.json"]
    assert isinstance(budget, dict)
    assert budget["status"] == "fail"
    assert budget["planned_new_fit_count"] == ceiling + 1
    assert not budget["scientific_fitting_authorized"]
    assert "literal_grid_exceeds_registered_compute_budget" in budget["blocking_reasons"]


def test_p03_contract_keeps_primary_preprocessing_and_fitting_gate_frozen() -> None:
    _, contract = _contracts()
    assert contract["primary_population"] == "primary_598"
    assert contract["primary_policy"] == "PP-U-MIN"
    assert contract["primary_representation"] == "R_MIN_400_1800"
    assert contract["planning"]["model_fitting_authorized"] is True
    assert contract["planning"]["registered_fit_budget_high"] == 250000
    assert contract["planning"]["pseudo_domain_cells"] == 128
    assert contract["planning"]["master_cv_fallback_cells"] == 132
    assert contract["coral"]["held_target_covariance_allowed"] is False
    assert contract["coral"]["status"] == (
        "resolved_source_to_source_covariance_augmentation_v1"
    )
    assert contract["negative_controls"]["status"] == (
        "resolved_p03_negative_controls_v1"
    )
    assert contract["authorization"]["decision_id"] == "P03-AUTH-20260810"
