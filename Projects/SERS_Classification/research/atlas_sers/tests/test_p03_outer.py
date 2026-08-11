from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from atlas_sers.evaluation.classical import TemperatureCalibration
from atlas_sers.evaluation.p03_finalize import FinalOuterResult
from atlas_sers.evaluation.p03_outer import execute_selected_procedure_outer


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fit_id": "selection-fit",
                "outer_run_id": "outer-1",
                "stage": "inner_selection",
                "accounting": "new_fit",
                "model_id": "C-PCA-LDA",
                "candidate_id": "candidate-1",
                "seed": "deterministic",
                "fit_uid_sha256": "a" * 64,
            },
            {
                "fit_id": "calibration-cache",
                "outer_run_id": "outer-1",
                "stage": "calibration_crossfit",
                "accounting": "cache_reuse",
                "model_id": "C-SELECTED",
                "candidate_id": "selected_after_inner",
                "seed": "cache",
                "fit_uid_sha256": "b" * 64,
            },
            {
                "fit_id": "final-fit",
                "outer_run_id": "outer-1",
                "stage": "final_selected_refit",
                "accounting": "new_fit",
                "model_id": "C-SELECTED",
                "candidate_id": "selected_after_inner",
                "seed": "deterministic",
                "fit_uid_sha256": "c" * 64,
            },
        ]
    )


def test_outer_dependency_failure_terminally_accounts_every_nonselection_row() -> None:
    rows = _rows()
    result = execute_selected_procedure_outer(
        dataset=SimpleNamespace(),  # type: ignore[arg-type]
        outer_fit_rows=rows,
        selection=pd.Series(
            {
                "status": "selection_failure",
                "outer_run_id": "outer-1",
            }
        ),
        candidate_registry=pd.DataFrame(),
        selection_fit_manifest=pd.DataFrame(),
        cached_selection_predictions=pd.DataFrame(),
        manifest=pd.DataFrame(),
        p02_tables={},
        p03_contract={},
        scientific_fitting_authorized=True,
    )
    assert set(result.fit_status.fit_id) == {"calibration-cache", "final-fit"}
    assert result.fit_status.status.eq("excluded_by_protocol").all()
    assert result.final_predictions.empty


def test_cached_calibration_and_final_refit_cover_all_dependency_rows(
    monkeypatch,
) -> None:
    rows = _rows()
    calibration = TemperatureCalibration(
        1.0, ("a", "b"), 2, 2, "d" * 64, "e" * 64, True, 0.1
    )
    crossfit = pd.DataFrame(
        {
            "observation_uid": ["o1", "o2"],
            "master_sample_id": ["m1", "m2"],
        }
    )
    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_outer.fit_cached_selected_calibration",
        lambda **_: SimpleNamespace(
            calibration=calibration,
            cross_fitted_predictions=crossfit,
            selection_unit_count=1,
            evidence_fit_id_sha256="f" * 64,
        ),
    )
    final_predictions = pd.DataFrame({"observation_uid": ["test-1"]})
    monkeypatch.setattr(
        "atlas_sers.evaluation.p03_outer.execute_selected_outer_refit",
        lambda **_: FinalOuterResult(
            pd.DataFrame(
                [
                    {
                        "fit_id": "final-fit",
                        "status": "complete",
                    }
                ]
            ),
            final_predictions,
        ),
    )
    result = execute_selected_procedure_outer(
        dataset=SimpleNamespace(),  # type: ignore[arg-type]
        outer_fit_rows=rows,
        selection=pd.Series(
            {
                "status": "complete",
                "outer_run_id": "outer-1",
            }
        ),
        candidate_registry=pd.DataFrame(),
        selection_fit_manifest=pd.DataFrame(),
        cached_selection_predictions=pd.DataFrame({"fit_id": ["selection-fit"]}),
        manifest=pd.DataFrame(),
        p02_tables={},
        p03_contract={},
        scientific_fitting_authorized=True,
    )
    assert set(result.fit_status.fit_id) == {"calibration-cache", "final-fit"}
    assert result.fit_status.status.eq("complete").all()
    assert result.calibration is not None
    assert result.calibration["state_sha256"] == calibration.state_sha256
    assert result.final_predictions.equals(final_predictions)
