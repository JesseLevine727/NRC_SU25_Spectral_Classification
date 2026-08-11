from __future__ import annotations

import pandas as pd
import pytest

from atlas_sers.evaluation.p03_selection import (
    derive_fixed_family_selections,
    select_outer_candidates,
)


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "simple",
                "model_id": "m-simple",
                "complexity_rank": 0,
                "declared_candidate_order": 0,
                "seed_count": 1,
                "hyperparameter_sha256": "a" * 64,
            },
            {
                "candidate_id": "complex",
                "model_id": "m-complex",
                "complexity_rank": 1,
                "declared_candidate_order": 1,
                "seed_count": 1,
                "hyperparameter_sha256": "b" * 64,
            },
        ]
    )


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for candidate in ("simple", "complex"):
        for unit in ("u1", "u2"):
            rows.append(
                {
                    "fit_id": f"fit-{index}",
                    "experiment_id": "EXP-C09-T3",
                    "task_id": "T3-ZS",
                    "outer_run_id": "outer-1",
                    "domain": "cwa:held",
                    "station": "cwa",
                    "held_instrument": "held",
                    "outer_repeat": 1,
                    "outer_fold": 0,
                    "selection_mode": "pseudo_domain",
                    "selection_unit_id": unit,
                    "candidate_id": candidate,
                    "model_id": "m-simple" if candidate == "simple" else "m-complex",
                    "hyperparameter_sha256": "a" * 64
                    if candidate == "simple"
                    else "b" * 64,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest[
        ["fit_id", "outer_run_id", "candidate_id", "selection_unit_id"]
    ].copy()
    result["seed"] = "deterministic"
    result["status"] = "complete"
    result["balanced_accuracy"] = result.candidate_id.map(
        {"simple": 0.8, "complex": 0.8}
    )
    result["macro_f1"] = 0.75
    return result


def test_outer_selection_is_lexicographic_hashed_and_complete() -> None:
    manifest = _manifest()
    result = select_outer_candidates(
        selection_fit_manifest=manifest,
        selection_unit_metrics=_metrics(manifest),
        candidate_registry=_registry(),
    )
    selection = result.selections.iloc[0]
    assert selection.status == "complete"
    assert selection.selected_candidate_id == "simple"
    assert len(selection.selection_state_sha256) == 64
    assert selection.expected_fit_count == selection.terminal_fit_count == 4
    assert result.traces.selected.sum() == 1


def test_outer_selection_fails_closed_on_fit_coverage_drift() -> None:
    manifest = _manifest()
    metrics = _metrics(manifest).iloc[:-1]
    with pytest.raises(ValueError, match="coverage differs"):
        select_outer_candidates(
            selection_fit_manifest=manifest,
            selection_unit_metrics=metrics,
            candidate_registry=_registry(),
        )


def test_outer_selection_records_terminal_selection_failure() -> None:
    manifest = _manifest()
    metrics = _metrics(manifest)
    metrics["status"] = "rank_failure"
    metrics[["balanced_accuracy", "macro_f1"]] = None
    result = select_outer_candidates(
        selection_fit_manifest=manifest,
        selection_unit_metrics=metrics,
        candidate_registry=_registry(),
    )
    selection = result.selections.iloc[0]
    assert selection.status == "selection_failure"
    assert pd.isna(selection.selected_candidate_id)
    assert "no_candidate" in selection.reason_code


def test_fixed_family_selection_maps_c09_evidence_to_c10_run() -> None:
    manifest = _manifest()
    expected_runs = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-C09-T3",
                "outer_run_id": "outer-1",
                "model_id": "C-SELECTED",
                "domain": "cwa:held",
                "station": "cwa",
                "held_instrument": "held",
                "outer_repeat": 1,
                "outer_fold": 0,
            },
            {
                "experiment_id": "EXP-C10-T3",
                "outer_run_id": "outer-c10",
                "model_id": "m-simple",
                "domain": "cwa:held",
                "station": "cwa",
                "held_instrument": "held",
                "outer_repeat": 1,
                "outer_fold": 0,
            },
        ]
    )
    result = derive_fixed_family_selections(
        c09_selection_fit_manifest=manifest,
        c09_selection_unit_metrics=_metrics(manifest),
        candidate_registry=_registry(),
        expected_run_registry=expected_runs,
        fixed_models=("m-simple",),
    )
    assert result.selections.outer_run_id.tolist() == ["outer-c10"]
    assert result.selections.source_c09_outer_run_id.tolist() == ["outer-1"]
    assert result.selections.selected_candidate_id.tolist() == ["simple"]
