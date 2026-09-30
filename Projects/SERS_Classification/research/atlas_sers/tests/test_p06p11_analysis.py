"""CPU synthetic tests for the pure P06/P11 analysis orchestration (T017)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p06p11_analysis as analysis
from atlas_sers.evaluation import p06p11_diagnostics as diagnostics
from atlas_sers.evaluation import p06p11_hierarchy as hierarchy
from atlas_sers.evaluation import p06p11_inference as inference
from atlas_sers.evaluation import p06p11_predictions as predictions
from tests.test_p05_comparison import _fixture, _run

DRAWS = 8
CONTRASTS = 34
CHECK_FLOOR = 2 + 2 * CONTRASTS + 2 * CONTRASTS


def _prepare():
    fixture = _fixture()
    panel = predictions.prepare_panel(
        p05_ensemble=fixture["p05_ensemble"],
        p04_ensemble=fixture["p04_ensemble"],
        p03_predictions=fixture["p03_predictions"],
        contexts=fixture["contexts"],
    )
    paired_metrics = _run(fixture)["paired_metrics"]
    return panel, paired_metrics


def _domain_count(design):
    for attribute in ("domains", "domain_ids", "domain_order"):
        value = getattr(design, attribute, None)
        if value is not None:
            return len(tuple(value))
    raise AssertionError("design is missing its domain identity")


def test_interval_and_feasibility_structure(monkeypatch):
    panel, paired = _prepare()
    calls = []
    original = inference.positive_weights

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "positive_weights", wrapper)
    out = analysis.analyze_panel(panel, paired, draws=DRAWS)
    assert len(calls) == 1
    intervals = out["tables"]["intervals"]
    feasibility = out["tables"]["feasibility"]
    assert len(intervals) == 4 * 34
    assert len(feasibility) == 34
    assert set(intervals["method"]) == {
        "crossed_weight",
        "master_weight",
        "instrument_weight",
        "hierarchical",
    }
    assert (
        intervals["bca_reason_code"]
        .isin([analysis.BCA_WEIGHT_REASON, analysis.BCA_HIERARCHY_REASON])
        .all()
    )


def test_arrays_are_deterministic_and_global_weights_are_reused():
    panel, paired = _prepare()
    first = analysis.analyze_panel(panel, paired, draws=DRAWS)
    second = analysis.analyze_panel(panel, paired, draws=DRAWS)
    assert set(first["arrays"]) == set(second["arrays"])
    for key, array in first["arrays"].items():
        np.testing.assert_array_equal(array, second["arrays"][key])
    assert "global.master_weights" in first["arrays"]
    assert "global.instrument_weights" in first["arrays"]


def test_global_weight_lookup_matches_pair_specific_order():
    panel, paired = _prepare()
    out = analysis.analyze_panel(panel, paired, draws=DRAWS)
    registry = out["registry"]
    entry = registry["contrasts"]["c000_M01"]
    rows = predictions.pair_units(
        panel,
        model_id=entry["model_id"],
        reference_model_id=entry["reference_model_id"],
        aggregation_id=entry["aggregation_id"],
    )
    design = inference.compile_pair(rows)
    positions_m = {name: i for i, name in enumerate(registry["masters"])}
    positions_i = {name: i for i, name in enumerate(registry["instruments"])}
    master_weight = out["arrays"][registry["master_weight_array"]]
    instrument_weight = out["arrays"][registry["instrument_weight_array"]]
    local_m = master_weight[:, [positions_m[name] for name in entry["masters"]]]
    local_i = instrument_weight[:, [positions_i[name] for name in entry["instruments"]]]
    overall, _ = inference.score_weights(design, local_m, local_i)
    np.testing.assert_allclose(overall, out["arrays"][entry["method_scores"]["crossed_weight"]])


def test_inputs_are_not_mutated():
    panel, paired = _prepare()
    panel_before = {key: value.copy(deep=True) for key, value in panel.items()}
    paired_before = paired.copy(deep=True)
    analysis.analyze_panel(panel, paired, draws=DRAWS)
    for key, value in panel_before.items():
        pd.testing.assert_frame_equal(value, panel[key])
    pd.testing.assert_frame_equal(paired_before, paired)


def test_check_callable_runs_around_each_unit_of_work():
    panel, paired = _prepare()
    calls = []
    analysis.analyze_panel(panel, paired, draws=DRAWS, check=lambda: calls.append(1))
    assert len(calls) >= CHECK_FLOOR


def test_check_failure_propagates_and_stops_the_run():
    panel, paired = _prepare()
    state = {"calls": 0}

    def check():
        state["calls"] += 1
        if state["calls"] == 3:
            raise RuntimeError("stop after first contrast")

    with pytest.raises(RuntimeError):
        analysis.analyze_panel(panel, paired, draws=DRAWS, check=check)


def test_frozen_point_mismatch_rejected_before_weight_generation(monkeypatch):
    panel, paired = _prepare()

    def explode(*args, **kwargs):
        raise AssertionError("positive_weights must not run before the point checks")

    monkeypatch.setattr(inference, "positive_weights", explode)
    broken = paired.copy()
    first = broken.index[0]
    broken.loc[first, "model_balanced_accuracy"] = 0.5
    broken.loc[first, "reference_balanced_accuracy"] = 0.0
    broken.loc[first, "delta_balanced_accuracy"] = 0.5
    with pytest.raises(analysis.P06P11AnalysisError) as excinfo:
        analysis.analyze_panel(panel, broken, draws=DRAWS)
    assert excinfo.value.reason_code == "frozen_point_estimate_mismatch"


def test_frozen_point_mismatch_uses_absolute_not_relative_tolerance(monkeypatch):
    panel, paired = _prepare()
    original_diagnostics = diagnostics.domain_diagnostics

    def shifted_diagnostics(paired_metrics):
        tables = {
            name: original_diagnostics(paired_metrics)[name].copy()
            for name in ("domain_metrics", "summary", "leave_one_out", "sign_flip")
        }
        tables["domain_metrics"]["delta"] = 0.5000000001
        tables["summary"]["mean_delta"] = 0.5000000001
        return tables

    def fixed_score_weights(design, factor_master, factor_instrument):
        rows = np.asarray(factor_master, dtype=float).shape[0]
        domains = _domain_count(design)
        return np.full((rows,), 0.5), np.full((1, domains), 0.5)

    def forbidden_positive_weights(*args, **kwargs):
        raise AssertionError("positive_weights must not run for a frozen mismatch")

    monkeypatch.setattr(diagnostics, "domain_diagnostics", shifted_diagnostics)
    monkeypatch.setattr(inference, "score_weights", fixed_score_weights)
    monkeypatch.setattr(inference, "positive_weights", forbidden_positive_weights)
    with pytest.raises(analysis.P06P11AnalysisError) as excinfo:
        analysis.analyze_panel(panel, paired, draws=DRAWS)
    assert excinfo.value.reason_code == "frozen_point_estimate_mismatch"


def test_hierarchy_undefined_draws_are_reported_without_conditional_bounds(monkeypatch):
    panel, paired = _prepare()
    real = hierarchy.hierarchical_draws

    def fake(design, *, draws=10000, seed=2026092903):
        result = dict(real(design, draws=draws, seed=seed))
        scores = np.asarray(result["draws"], dtype=float).copy()
        empty = np.asarray(result["empty_cells"]).copy()
        scores[0] = np.nan
        empty[0] = 1
        result["draws"] = scores
        result["empty_cells"] = empty
        return result

    monkeypatch.setattr(hierarchy, "hierarchical_draws", fake)
    out = analysis.analyze_panel(panel, paired, draws=DRAWS)
    feasibility = out["tables"]["feasibility"]
    assert (feasibility["undefined_draws"] > 0).all()
    hierarchical = out["tables"]["intervals"]
    hierarchical = hierarchical[hierarchical["method"].eq("hierarchical")]
    assert hierarchical["lower"].isna().all()
    assert hierarchical["upper"].isna().all()
    assert hierarchical["reason_code"].eq("hierarchical_fixed_support_undefined").all()


def test_paired_metrics_failure_propagates(monkeypatch):
    panel, paired = _prepare()

    def explode(*args, **kwargs):
        raise RuntimeError("missing whole common support")

    monkeypatch.setattr(predictions, "pair_units", explode)
    with pytest.raises(RuntimeError):
        analysis.analyze_panel(panel, paired, draws=DRAWS)
