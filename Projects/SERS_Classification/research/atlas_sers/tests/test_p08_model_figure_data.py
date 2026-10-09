"""Supervisor regressions using the actual accepted numerical API on invented rows."""

import copy
import hashlib
import json

import numpy as np
import pytest

from atlas_sers.evaluation import p08_universal_analysis as analysis
from atlas_sers.evaluation import p08_universal_units as units
from atlas_sers.visualization import p08_model_figure_data as figures
from tests.test_p08_universal_analysis import (
    GLOBAL_INSTRUMENTS,
    GLOBAL_MASTERS,
    _predictions_frame,
    _registered_frame,
)


@pytest.fixture(scope="module")
def computed():
    registered = _registered_frame()
    panels = units.build_units(_predictions_frame(registered), registered)
    old = analysis.DRAW_COUNT
    analysis.DRAW_COUNT = 4
    try:
        return analysis.analyze_panel(
            panels,
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
            domain_families={"d1": "unknown"},
        )
    finally:
        analysis.DRAW_COUNT = old


@pytest.fixture()
def small(monkeypatch, computed):
    monkeypatch.setattr(figures, "EXPECTED_DOMAINS", 1)
    return copy.deepcopy(computed)


def test_actual_api_shape_hash_and_no_mutation(small):
    prior = copy.deepcopy(small)
    result = figures.prepare_model_figures(small)
    assert result["manifest"]["counts"] == {
        "domains": 1,
        "registry_entries": 52,
        "f02_pairs": 40,
        "f03_effects": 40,
        "f03_domains": 40,
        "f04_interactions": 64,
        "f04_domains": 48,
        "future_qc_unavailable": 8,
        "available_interactions": 24,
    }
    payload = json.dumps(
        result["semantic"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == result["semantic_sha256"]
    assert (
        figures.prepare_model_figures(small)["semantic_sha256"]
        == result["semantic_sha256"]
    )
    assert small["registry"] == prior["registry"]
    assert small["contrast_summary"].equals(prior["contrast_summary"])
    np.testing.assert_array_equal(
        small["weights"]["master_weights"], prior["weights"]["master_weights"]
    )
    assert result["manifest"]["reviewed"] is False


def test_orientation_interaction_values_and_missingness(small):
    sem = figures.prepare_model_figures(small)["semantic"]
    for row in sem["f02_pairs"]:
        assert row["effect"] == pytest.approx(
            row["y_balanced_accuracy"] - row["x_balanced_accuracy"]
        )
    for row in sem["f04_domains"]:
        a, b, c, d = row["procedure_balanced_accuracies"]
        assert row["point_effect"] == pytest.approx(a - b - c + d)
        assert row["physical_masters"] == 8
    absent = [row for row in sem["f04_interactions"] if not row["available"]]
    assert len(absent) == 16
    for row in absent:
        assert row["reason"] == "outside_universal_execution_scope"
        assert row["family_size"] == 32
        assert (
            row["point_effect"]
            is row["domain_raw_p"]
            is row["domain_adjusted_p"]
            is None
        )
    for row in sem["f03_effects"]:
        assert row["family_size"] == 20
        assert isinstance(row["domain_raw_p"], float)


def test_public_whitelist_and_alias_retention(small):
    sem = figures.prepare_model_figures(small)["semantic"]
    forbidden = {
        "master_sample_id",
        "observation_uid",
        "unit_id",
        "context_id",
        "master_weights",
        "instrument_weights",
        "cell_keys",
        "global_masters",
    }

    def walk(node):
        if isinstance(node, dict):
            assert not forbidden.intersection(node)
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(sem)
    text = json.dumps(sem)
    assert all(json.dumps(master) not in text for master in GLOBAL_MASTERS)
    assert {row["model_id"] for row in sem["f02_pairs"]} == set(analysis.MODELS)
    for model in ("D0-M", "P05-SELECTED"):
        assert len([row for row in sem["f02_pairs"] if row["model_id"] == model]) == 8


@pytest.mark.parametrize(
    "defect", ["registry", "boundary", "p_payload", "p_nan", "family", "four_cells"]
)
def test_refuses_unfaithful_contrasts(small, defect):
    entry = next(
        e
        for e in small["registry"]
        if e["family_id"] == "policy_model_interaction" and e["available"]
    )
    result = small["contrasts"]["equal_context"][entry["contrast_id"]]
    if defect == "registry":
        small["registry"] = small["registry"][:-1]
    elif defect == "boundary":
        small["boundary"]["no_G4_decision"] = False
    elif defect == "p_payload":
        result["sensitivity_adjustment"]["domain_raw_p"] = {"m1": 0.5}
    elif defect == "p_nan":
        result["sensitivity_adjustment"]["domain_raw_p"] = float("nan")
    elif defect == "family":
        result["sensitivity_adjustment"]["family_size"] = 24
    elif defect == "four_cells":
        result["point"]["procedure_domain_effects"][0]["d1"] += 0.1
        result["point"]["procedure_domain_effects"][1]["d1"] += 0.1
    with pytest.raises(ValueError):
        figures.prepare_model_figures(small)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "nan", "support", "bool_count"]
)
def test_refuses_metric_support_defects(small, defect):
    table = small["metrics"]["equal_context"]["PP-U-MIN"]["M01"]
    frame = table["domain_metrics"]
    if defect == "missing":
        table["domain_metrics"] = frame.iloc[:-1].copy()
    elif defect == "duplicate":
        table["domain_metrics"] = frame.iloc[[0, 0, 1, 2, 3, 4]].copy()
    elif defect == "nan":
        frame.loc[0, "balanced_accuracy"] = float("nan")
    elif defect == "support":
        frame.loc[0, "physical_masters"] += 1
    elif defect == "bool_count":
        frame["physical_masters"] = frame["physical_masters"].astype(object)
        frame.loc[0, "physical_masters"] = True
    with pytest.raises(ValueError):
        figures.prepare_model_figures(small)
